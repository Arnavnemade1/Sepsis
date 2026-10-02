"""The paging OS: alert intake, the page queue, clinicians, sentinel and policy.

Routes only; the engine is backend/aegis/paging. Identity matters here more than
anywhere: answering a page is done AS a clinician, so the handler checks that the
caller is the clinician the page is ringing for (or an operator acting for them,
which is recorded as such). Capability checks go through the kernel gate.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.schemas import (
    PagingAlertIn,
    PagingApproveIn,
    PagingCancelIn,
    PagingRespondIn,
    PagingSettingsIn,
    PagingStatusIn,
)
from backend.aegis.server import (
    GATE,
    PAGING,
    _PermissionDenied,
    _cred,
    _paging_clients,
)

router = APIRouter()


class _SessionRejected(Exception):
    """EAUTH — the caller presented a session the kernel no longer accepts."""


def _need(cap: str, what: str):
    """The caller's credential, after the gate's verdict on ``cap``. Enforcing mode
    refuses here; audit mode journals a would-deny and lets it through.

    One thing audit mode does NOT let through here: a presented session the kernel
    rejected (forged, expired, or signed before a restart). Elsewhere that caller is
    served as the guest it now is; on the paging OS every write carries identity, and
    serving it as guest would record a charge nurse's alert, or a clinician's answer,
    as guest's. The client renews the session and retries."""
    cred = _cred()
    if cred.auth_error:
        raise _SessionRejected(f"presented session rejected: {cred.auth_error}")
    GATE.require(cap, cred, what=what)
    return cred


def _deny(exc: Exception) -> Dict[str, Any]:
    errno = "EAUTH" if isinstance(exc, _SessionRejected) else "EPERM"
    return {"status": "error", "errno": errno, "detail": str(exc)}


@router.get("/aegis/paging/hospital")
def paging_hospital() -> Dict[str, Any]:
    """The building: floors, units, rooms. Static; fetch once."""
    return PAGING.layout()


@router.get("/aegis/paging/snapshot")
def paging_snapshot() -> Dict[str, Any]:
    return PAGING.snapshot()


@router.post("/aegis/paging/alerts")
async def paging_raise(payload: PagingAlertIn) -> Dict[str, Any]:
    """Raise an alert in plain language. Runs the whole pipeline — triage, routing,
    guardrails, policy, Z3 certificate — and returns the dispatch decision."""
    try:
        cred = _need("page:raise", "raise a page")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    source = payload.source if payload.source in ("nurse", "voice") else "nurse"
    # off the event loop: triage may call the model, and certify holds the Z3 lock
    return await asyncio.to_thread(PAGING.raise_alert, payload.text, payload.room, source, cred.username)


@router.get("/aegis/paging/pages/{page_id}")
def paging_page(page_id: str) -> Dict[str, Any]:
    p = PAGING.page_detail(page_id)
    return p if p is not None else {"status": "error", "detail": "no such page"}


@router.post("/aegis/paging/pages/{page_id}/respond")
async def paging_respond(page_id: str, payload: PagingRespondIn) -> Dict[str, Any]:
    try:
        cred = _need("page:respond", "answer a page")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    who = payload.clinician or cred.username
    if who != cred.username and not cred.has("page:dispatch"):
        return {"status": "error", "errno": "EPERM",
                "detail": f"{cred.username} cannot answer a page for {who}"}
    actor = cred.username if who == cred.username else f"{cred.username} for {who}"
    return await asyncio.to_thread(PAGING.respond, page_id, who, payload.action, payload.reason, actor)


@router.post("/aegis/paging/pages/{page_id}/resolve")
async def paging_resolve(page_id: str) -> Dict[str, Any]:
    try:
        cred = _need("page:respond", "resolve a page")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    page = PAGING.page_detail(page_id) or {}
    if page.get("assignee") != cred.username and not cred.has("page:dispatch"):
        return {"status": "error", "errno": "EPERM", "detail": "only the accepting clinician or an operator can resolve"}
    return await asyncio.to_thread(PAGING.resolve, page_id, cred.username)


@router.post("/aegis/paging/pages/{page_id}/escalate")
async def paging_escalate(page_id: str) -> Dict[str, Any]:
    try:
        cred = _need("page:dispatch", "escalate a page")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return await asyncio.to_thread(PAGING.escalate_now, page_id, cred.username)


@router.post("/aegis/paging/pages/{page_id}/approve")
async def paging_approve(page_id: str, payload: PagingApproveIn) -> Dict[str, Any]:
    try:
        cred = _need("page:dispatch", "approve a page")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return await asyncio.to_thread(PAGING.approve, page_id, cred.username, payload.clinician)


@router.post("/aegis/paging/pages/{page_id}/cancel")
async def paging_cancel(page_id: str, payload: PagingCancelIn) -> Dict[str, Any]:
    try:
        cred = _need("page:dispatch", "cancel a page")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return await asyncio.to_thread(PAGING.cancel, page_id, cred.username, payload.reason)


@router.get("/aegis/paging/clinicians")
def paging_clinicians() -> Dict[str, Any]:
    return {"clinicians": PAGING.snapshot()["clinicians"]}


@router.post("/aegis/paging/clinicians/{clinician_id}/status")
async def paging_status(clinician_id: str, payload: PagingStatusIn) -> Dict[str, Any]:
    try:
        cred = _need("page:respond", "set a status")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    if clinician_id != cred.username and not cred.has("page:dispatch"):
        return {"status": "error", "errno": "EPERM", "detail": "you can only set your own status"}
    return await asyncio.to_thread(PAGING.set_status, clinician_id, payload.status, cred.username)


@router.post("/aegis/paging/recs/{rec_id}/ack")
def paging_ack(rec_id: str) -> Dict[str, Any]:
    try:
        cred = _need("page:dispatch", "acknowledge a recommendation")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return PAGING.acknowledge_rec(rec_id, cred.username)


@router.get("/aegis/paging/rooms/{room}")
def paging_room(room: str) -> Dict[str, Any]:
    r = PAGING.room_detail(room.upper())
    return r if r is not None else {"status": "error", "detail": "no such room"}


@router.get("/aegis/rooms")
def rooms_board() -> Dict[str, Any]:
    """Every room as a unit: occupant, record flags, latest vitals, NEWS2, active findings, episode."""
    return PAGING.rooms_board()


@router.get("/aegis/rooms/{room}")
def room_unit(room: str) -> Dict[str, Any]:
    """One room as a unit: the record, the vitals series, labs, every detector's findings, the
    pages, the care team, the devices and the merged timeline."""
    u = PAGING.room_unit(room.upper())
    return u if u is not None else {"status": "error", "detail": "no such room"}


@router.get("/aegis/paging/settings")
def paging_settings() -> Dict[str, Any]:
    snap = PAGING.snapshot()
    return {"settings": snap["settings"], "invariants": snap["invariants"]}


@router.post("/aegis/paging/settings")
def paging_settings_update(payload: PagingSettingsIn) -> Dict[str, Any]:
    try:
        cred = _need("page:configure", "change the paging policy")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return PAGING.update_settings(payload.patch, cred.username)


@router.websocket("/aegis/paging/stream")
async def paging_stream(ws: WebSocket) -> None:
    """A snapshot on connect, then one batch of events per second. A clinician's
    device passes ?clinician=<id>: while it is connected that clinician answers
    their own pages (the demo's simulated responders stand down for them)."""
    await ws.accept()
    clinician = ws.query_params.get("clinician")
    PAGING.presence_join(clinician)
    _paging_clients.append(ws)
    try:
        await ws.send_json({"type": "snapshot", "data": await asyncio.to_thread(PAGING.snapshot)})
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _paging_clients:
            _paging_clients.remove(ws)
        PAGING.presence_leave(clinician)
