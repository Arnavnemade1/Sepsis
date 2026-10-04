"""The integration plane: the hospital's feeds, their mappings, and the evidence behind them.

Routes only; the engine is backend/aegis/interop. Reading is a floor-level act
(``interop:read``); deciding what a local code means, or putting a quarantined feed
back, changes what every detector believes and is the chief's (``interop:configure``);
sending HL7 is the interface engine's (``interop:ingest``). Every decision is journalled
by the engine with who made it.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import Response

from backend.api.schemas import (
    InteropAdoptIn,
    InteropDrillIn,
    InteropMessageIn,
    InteropRejectIn,
    InteropSettingsIn,
)
from backend.aegis.server import GATE, INTEROP, INTEROP_SIM, _PermissionDenied, _cred

router = APIRouter()
HL7_MEDIA = "x-application/hl7-v2+er7"
_AUDIT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                      "artifacts", "interop", "cinc_audit.json")


class _SessionRejected(Exception):
    """EAUTH: the caller presented a session the kernel no longer accepts."""


def _need(cap: str, what: str):
    """As on the paging routes: a rejected session never acts as guest here, because a
    mapping decision or an inbound message carries the identity that made it."""
    cred = _cred()
    if cred.auth_error:
        raise _SessionRejected(f"presented session rejected: {cred.auth_error}")
    GATE.require(cap, cred, what=what)
    return cred


def _deny(exc: Exception) -> Dict[str, Any]:
    errno = "EAUTH" if isinstance(exc, _SessionRejected) else "EPERM"
    return {"status": "error", "errno": errno, "detail": str(exc)}


def _read(what: str) -> Optional[Dict[str, Any]]:
    try:
        _need("interop:read", what)
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return None


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #
@router.get("/aegis/interop")
def interop_overview() -> Dict[str, Any]:
    """Interfaces, mapping counts, the review queue, quarantines, the model proposer."""
    return _read("read the integration plane") or INTEROP.overview()


@router.get("/aegis/interop/interfaces/{iid}")
def interop_interface(iid: str) -> Dict[str, Any]:
    denied = _read("read an interface")
    if denied:
        return denied
    i = INTEROP.interface(iid)
    return i if i is not None else {"status": "error", "detail": "no such interface"}


@router.get("/aegis/interop/messages")
def interop_messages(interface: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    denied = _read("read interface messages")
    if denied:
        return denied
    return {"messages": INTEROP.message_list(interface, max(1, min(400, limit)))}


@router.get("/aegis/interop/messages/{mid}")
def interop_message(mid: str) -> Dict[str, Any]:
    denied = _read("read an interface message")
    if denied:
        return denied
    m = INTEROP.message(mid)
    return m if m is not None else {"status": "error", "detail": "no such message"}


@router.get("/aegis/interop/mappings")
def interop_mappings(interface: Optional[str] = None) -> Dict[str, Any]:
    denied = _read("read mappings")
    if denied:
        return denied
    return {"mappings": INTEROP.mapping_list(interface)}


@router.get("/aegis/interop/mappings/{mid}")
def interop_mapping(mid: str) -> Dict[str, Any]:
    denied = _read("read a mapping")
    if denied:
        return denied
    m = INTEROP.mapping(mid)
    return m if m is not None else {"status": "error", "detail": "no such mapping"}


@router.get("/aegis/interop/census")
def interop_census() -> Dict[str, Any]:
    denied = _read("read the interface census")
    if denied:
        return denied
    return {"patients": INTEROP.census()}


@router.get("/aegis/interop/vocabulary")
def interop_vocabulary() -> Dict[str, Any]:
    """The canonical concepts (LOINC, unit, window, reference), the identities and the
    fixed thresholds every mapping is judged by."""
    return INTEROP.vocabulary()


@router.get("/aegis/interop/audit")
def interop_audit() -> Dict[str, Any]:
    """What the a-priori checks found in the PhysioNet/CinC 2019 data itself."""
    try:
        with open(_AUDIT) as fh:
            data = json.load(fh)
        return {"meta": data.get("meta"), "findings": data.get("findings"),
                "invariants": {"training": data["training"]["invariants"], "holdout": data["holdout"]["invariants"]}}
    except (OSError, ValueError, KeyError):
        return {"status": "error", "detail": "no audit artifact (run python -m backend.ml.scripts.interop_reference)"}


_BENCH = os.path.join(os.path.dirname(_AUDIT), "interop_bench.json")


@router.get("/aegis/interop/bench")
def interop_bench() -> Dict[str, Any]:
    """The registered benchmark's verdicts and per-fault rows (docs/INTEROP.md)."""
    try:
        with open(_BENCH) as fh:
            b = json.load(fh)
    except (OSError, ValueError):
        return {"available": False}
    rows = []
    for fid, v in (b.get("faults") or {}).items():
        arms = v.get("arms") or {}
        has_fix = any((v.get("fixes") or {}).values())
        rows.append({"id": fid, "what": v.get("what"), "silent_dauroc": v["silent"]["vs_clean"]["delta"],
                     "silent_dfar": v["silent"]["far"] - b["clean"]["far"], "detected": v.get("detected"),
                     "faulty_to_quarantine": v.get("faulty_to_quarantine"),
                     "delivered_faulty": sum((v.get("delivered_faulty") or {}).values()),
                     "quarantine_dauroc": arms.get("gate_quarantine", {}).get("vs_clean", {}).get("delta"),
                     "fix_dauroc": arms.get("gate_fix", {}).get("vs_clean", {}).get("delta") if has_fix else None})
    post = None
    try:
        with open(os.path.join(os.path.dirname(_AUDIT), "interop_bench_posthoc.json")) as fh:
            ph = json.load(fh)
        pv = ph.get("verdicts") or {}
        post = {"note": ph.get("note"),
                "h4_per_100k": {k: v.get("per_100k") for k, v in (ph.get("h4") or {}).items()},
                "h5": {k: (pv.get("H5_onboarding") or {}).get(k) for k in
                       ("correct_auto_adopted", "unsafe_wrong_unit", "unsafe_wrong_concept")},
                "delivered_faulty": {fid: sum((v.get("delivered_faulty") or {}).values())
                                     for fid, v in (ph.get("faults") or {}).items()},
                "fix_dauroc": {fid: (v.get("arms") or {}).get("gate_fix", {}).get("vs_clean", {}).get("delta")
                               for fid, v in (ph.get("faults") or {}).items()},
                "detected": (pv.get("H2_detection") or {}).get("detected")}
    except (OSError, ValueError):
        post = None
    return {"available": True, "clean": b.get("clean"), "verdicts": b.get("verdicts"), "faults": rows,
            "posthoc": post,
            "h4": {k: {kk: vv for kk, vv in v.items() if kk in ("observations", "false", "per_100k", "refused_at_onboarding")}
                   for k, v in (b.get("h4") or {}).items()},
            "proposers": {k: v for k, v in (b.get("proposers") or {}).items() if k != "rows"}}


@router.get("/aegis/interop/sim")
def interop_sim() -> Dict[str, Any]:
    """The simulated hospital on the other side of the interfaces, and its fault drills."""
    return INTEROP_SIM.status()


# --------------------------------------------------------------------------- #
# inbound HL7
# --------------------------------------------------------------------------- #
@router.post("/aegis/interop/hl7")
async def interop_hl7(request: Request):
    """One HL7 v2 message in, its acknowledgement out. Send ER7 text as the body
    (``x-application/hl7-v2+er7`` or ``text/plain``) and the ACK comes back the same way;
    send JSON ``{"message": ...}`` and the ACK comes back with the engine's record of
    what it did with every OBX."""
    try:
        cred = _need("interop:ingest", "send HL7 into the hospital")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    raw = await request.body()
    ctype = (request.headers.get("content-type") or "").lower()
    as_json = "json" in ctype
    if as_json:
        try:
            text = InteropMessageIn(**json.loads(raw.decode("utf-8"))).message
        except Exception:
            return {"status": "error", "detail": "body must be {\"message\": \"MSH|...\"}"}
    else:
        text = raw.decode("utf-8", errors="replace")
    ack, rec = await asyncio.to_thread(INTEROP.ingest_hl7, text, "http", cred.username)
    if as_json:
        return {"status": "ok", "ack": ack, "record": {k: v for k, v in rec.items() if k != "raw"}}
    return Response(content=ack, media_type=HL7_MEDIA)


# --------------------------------------------------------------------------- #
# people deciding
# --------------------------------------------------------------------------- #
@router.post("/aegis/interop/mappings/{mid}/adopt")
async def interop_adopt(mid: str, payload: InteropAdoptIn) -> Dict[str, Any]:
    try:
        cred = _need("interop:configure", "adopt a mapping")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return await asyncio.to_thread(INTEROP.adopt, mid, cred.username, payload.concept, payload.conversion)


@router.post("/aegis/interop/mappings/{mid}/reject")
async def interop_reject(mid: str, payload: InteropRejectIn) -> Dict[str, Any]:
    try:
        cred = _need("interop:configure", "reject a mapping")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return await asyncio.to_thread(INTEROP.reject, mid, cred.username, payload.reason)


@router.post("/aegis/interop/mappings/{mid}/restore")
async def interop_restore(mid: str) -> Dict[str, Any]:
    try:
        cred = _need("interop:configure", "restore a quarantined mapping")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return await asyncio.to_thread(INTEROP.restore, mid, cred.username)


@router.post("/aegis/interop/mappings/{mid}/reevaluate")
async def interop_reevaluate(mid: str) -> Dict[str, Any]:
    """Run the evidence again now (it also runs on its own as values arrive)."""
    try:
        _need("interop:read", "re-run a mapping's evidence")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return await asyncio.to_thread(INTEROP.reevaluate, mid)


@router.post("/aegis/interop/settings")
def interop_settings(payload: InteropSettingsIn) -> Dict[str, Any]:
    try:
        cred = _need("interop:configure", "change the integration policy")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    return {"status": "ok", "settings": INTEROP.update_settings(payload.patch, cred.username)}


@router.post("/aegis/interop/sim/drill")
def interop_drill(payload: InteropDrillIn) -> Dict[str, Any]:
    """Break the SIMULATED hospital's interface on purpose and watch the evidence catch it.
    The integration plane's administrator (``interop:configure``, the chief, who then
    decides the fix) or a clinician who may drill devices (``drill:device``) runs one."""
    try:
        cap = "interop:configure" if _cred().has("interop:configure") else "drill:device"
        cred = _need(cap, "run an interface fault drill")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _deny(exc)
    out = INTEROP_SIM.drill(payload.kind, payload.on)
    try:
        from backend.aegis.kernel.journal import KJOURNAL, NOTICE
        KJOURNAL.append("interop", "drill", severity=NOTICE, uid=cred.username, kind=payload.kind, on=payload.on)
    except Exception:
        pass
    return out
