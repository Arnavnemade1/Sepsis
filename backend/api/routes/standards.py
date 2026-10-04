"""The standards surface: FHIR R4, CDS Hooks and outbound webhooks.

Routes only; the logic is backend/aegis/interop/{fhir,cdshooks,webhooks}.py. The FHIR
CapabilityStatement and the CDS Hooks discovery document are public, as their specs
expect; reading patient-level resources and calling a CDS service need
``interop:read`` (an EHR signs in as the interface service account); writing
Observations needs ``interop:ingest``; managing webhook subscriptions needs
``interop:configure``.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.api.schemas import WebhookIn
from backend.aegis.interop import fhir as F
from backend.aegis.server import CDS, FHIR, GATE, INTEROP, WEBHOOKS, _PermissionDenied, _cred

router = APIRouter()
FHIR_JSON = "application/fhir+json"


class _SessionRejected(Exception):
    pass


def _need(cap: str, what: str):
    cred = _cred()
    if cred.auth_error:
        raise _SessionRejected(f"presented session rejected: {cred.auth_error}")
    GATE.require(cap, cred, what=what)
    return cred


def _fhir(body: Dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse(body, status_code=status, media_type=FHIR_JSON)


def _fhir_deny(exc: Exception) -> JSONResponse:
    code = 401 if isinstance(exc, _SessionRejected) else 403
    return _fhir(F.outcome("error", "security", str(exc)), code)


def _room(ref: Optional[str]) -> Optional[str]:
    """A room from a FHIR patient reference: Patient/room-512, room-512, or an MRN."""
    if not ref:
        return None
    pid = ref.split("/")[-1]
    r = F.room_of(pid)
    if r:
        return r
    for row in FHIR.board().get("rooms") or []:
        if row.get("mrn") == pid:
            return row["room"]
    return None


def _read(what: str) -> Optional[JSONResponse]:
    try:
        _need("interop:read", what)
    except (_PermissionDenied, _SessionRejected) as exc:
        return _fhir_deny(exc)
    return None


# --------------------------------------------------------------------------- #
# FHIR R4
# --------------------------------------------------------------------------- #
@router.get("/aegis/fhir/R4/metadata")
def fhir_metadata():
    return _fhir(FHIR.capability())


@router.get("/aegis/fhir/R4/Patient")
def fhir_patients(identifier: Optional[str] = None, _id: Optional[str] = None):
    denied = _read("search FHIR patients")
    if denied:
        return denied
    if _id:
        r = _room(_id)
        p = FHIR.patient(r) if r else None
        return _fhir(F.bundle([p] if p else [], base=FHIR.base_url))
    return _fhir(F.bundle(FHIR.patients(identifier), base=FHIR.base_url))


@router.get("/aegis/fhir/R4/Patient/{pid}")
def fhir_patient(pid: str):
    denied = _read("read a FHIR patient")
    if denied:
        return denied
    r = _room(pid)
    p = FHIR.patient(r) if r else None
    return _fhir(p) if p else _fhir(F.outcome("error", "not-found", f"no patient {pid}"), 404)


@router.get("/aegis/fhir/R4/Encounter")
def fhir_encounters(patient: Optional[str] = None):
    denied = _read("search FHIR encounters")
    if denied:
        return denied
    rooms = [_room(patient)] if patient else [r["room"] for r in FHIR.board().get("rooms") or [] if r.get("kind")]
    return _fhir(F.bundle([e for e in (FHIR.encounter(r) for r in rooms if r) if e], base=FHIR.base_url))


@router.get("/aegis/fhir/R4/Observation")
def fhir_observations(patient: Optional[str] = None, subject: Optional[str] = None, category: Optional[str] = None,
                      code: Optional[str] = None):
    denied = _read("search FHIR observations")
    if denied:
        return denied
    r = _room(patient or subject)
    if r is None:
        return _fhir(F.outcome("error", "required", "search Observation by patient (or subject)"), 400)
    return _fhir(F.bundle(FHIR.observations(r, category, code), base=FHIR.base_url))


@router.get("/aegis/fhir/R4/RiskAssessment")
def fhir_risk(patient: Optional[str] = None, subject: Optional[str] = None):
    denied = _read("search FHIR risk assessments")
    if denied:
        return denied
    r = _room(patient or subject)
    if r is None:
        return _fhir(F.outcome("error", "required", "search RiskAssessment by patient (or subject)"), 400)
    return _fhir(F.bundle(FHIR.risk(r), base=FHIR.base_url))


@router.get("/aegis/fhir/R4/Flag")
def fhir_flags(patient: Optional[str] = None, subject: Optional[str] = None):
    denied = _read("search FHIR flags")
    if denied:
        return denied
    r = _room(patient or subject)
    if r is None:
        return _fhir(F.outcome("error", "required", "search Flag by patient (or subject)"), 400)
    return _fhir(F.bundle(FHIR.flags(r), base=FHIR.base_url))


@router.get("/aegis/fhir/R4/Communication")
def fhir_communications(patient: Optional[str] = None, subject: Optional[str] = None, status: Optional[str] = None):
    denied = _read("search FHIR communications")
    if denied:
        return denied
    r = _room(patient or subject) if (patient or subject) else None
    return _fhir(F.bundle(FHIR.communications(r, status), base=FHIR.base_url))


async def _ingest_observations(obs: list, who: str):
    er7, problems = F.FHIRFacade.to_oru(obs)
    acks = []
    if er7:
        for msg in er7.split("\x00"):
            ack, rec = await asyncio.to_thread(INTEROP.ingest_hl7, msg, "fhir", who)
            acks.append({"ack": rec["ack"], "summary": rec["summary"], "error": rec["error"],
                         "outcomes": rec["outcomes"]})
    return acks, problems


@router.post("/aegis/fhir/R4/Observation")
async def fhir_create_observation(request: Request):
    try:
        cred = _need("interop:ingest", "write a FHIR observation")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _fhir_deny(exc)
    try:
        body = await request.json()
    except Exception:
        return _fhir(F.outcome("error", "structure", "body must be a FHIR Observation"), 400)
    acks, problems = await _ingest_observations([body], cred.username)
    ok = acks and all(a["ack"] == "AA" for a in acks) and not problems
    issues = problems + [f"{a['ack']}: {a['summary']}" for a in acks]
    return _fhir({"resourceType": "OperationOutcome",
                  "issue": [{"severity": "information" if ok else "error", "code": "informational" if ok else
                             "processing", "diagnostics": i} for i in issues] or
                           [{"severity": "error", "code": "invalid", "diagnostics": "nothing to ingest"}]},
                 201 if ok else 422)


@router.post("/aegis/fhir/R4")
async def fhir_transaction(request: Request):
    """A transaction or batch Bundle of Observations, normalised into the integration
    engine's ORU path: every code is mapped on evidence, exactly as an HL7 feed's is."""
    try:
        cred = _need("interop:ingest", "write a FHIR bundle")
    except (_PermissionDenied, _SessionRejected) as exc:
        return _fhir_deny(exc)
    try:
        body = await request.json()
    except Exception:
        return _fhir(F.outcome("error", "structure", "body must be a FHIR Bundle"), 400)
    if body.get("resourceType") != "Bundle" or body.get("type") not in ("transaction", "batch"):
        return _fhir(F.outcome("error", "invalid", "expected a transaction or batch Bundle"), 400)
    obs = [(e or {}).get("resource") for e in body.get("entry") or []]
    acks, problems = await _ingest_observations(obs, cred.username)
    kind = "transaction-response" if body["type"] == "transaction" else "batch-response"
    entries = [{"response": {"status": "201 Created" if a["ack"] == "AA" else "422 Unprocessable Entity",
                             "outcome": F.outcome("information" if a["ack"] == "AA" else "error",
                                                  "informational" if a["ack"] == "AA" else "processing",
                                                  a["summary"])}} for a in acks]
    entries += [{"response": {"status": "400 Bad Request", "outcome": F.outcome("error", "invalid", p)}}
                for p in problems]
    return _fhir({"resourceType": "Bundle", "type": kind, "entry": entries})


# --------------------------------------------------------------------------- #
# CDS Hooks 2.0
# --------------------------------------------------------------------------- #
@router.get("/aegis/cds-services")
def cds_discovery() -> Dict[str, Any]:
    return CDS.discovery()


@router.post("/aegis/cds-services/{service_id}")
async def cds_call(service_id: str, request: Request):
    try:
        _need("interop:read", "call a CDS service")
    except (_PermissionDenied, _SessionRejected) as exc:
        return JSONResponse({"cards": [], "error": str(exc)}, status_code=401 if isinstance(exc, _SessionRejected)
                            else 403)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"cards": [], "error": "body must be a CDS Hooks request"}, status_code=400)
    return await asyncio.to_thread(CDS.call, service_id, body)


@router.post("/aegis/cds-services/{service_id}/feedback")
async def cds_feedback(service_id: str, request: Request) -> Dict[str, Any]:
    try:
        cred = _need("interop:read", "send CDS feedback")
    except (_PermissionDenied, _SessionRejected) as exc:
        return {"status": "error", "detail": str(exc)}
    try:
        body = await request.json()
    except Exception:
        body = {}
    return CDS.feedback(service_id, body, cred.username)


# --------------------------------------------------------------------------- #
# outbound webhooks
# --------------------------------------------------------------------------- #
@router.get("/aegis/interop/webhooks")
def webhooks_status() -> Dict[str, Any]:
    try:
        _need("interop:read", "read webhook subscriptions")
    except (_PermissionDenied, _SessionRejected) as exc:
        return {"status": "error", "detail": str(exc)}
    return {**WEBHOOKS.status(), "cds": CDS.calls[-20:][::-1]}


@router.post("/aegis/interop/webhooks")
def webhooks_subscribe(payload: WebhookIn) -> Dict[str, Any]:
    try:
        cred = _need("interop:configure", "subscribe a webhook")
    except (_PermissionDenied, _SessionRejected) as exc:
        return {"status": "error", "detail": str(exc)}
    return WEBHOOKS.subscribe(payload.url, payload.events, cred.username, payload.description)


@router.post("/aegis/interop/webhooks/{sid}/delete")
def webhooks_delete(sid: str) -> Dict[str, Any]:
    try:
        _need("interop:configure", "delete a webhook")
    except (_PermissionDenied, _SessionRejected) as exc:
        return {"status": "error", "detail": str(exc)}
    return WEBHOOKS.unsubscribe(sid)


@router.post("/aegis/interop/webhooks/{sid}/test")
def webhooks_test(sid: str) -> Dict[str, Any]:
    try:
        _need("interop:configure", "send a test webhook")
    except (_PermissionDenied, _SessionRejected) as exc:
        return {"status": "error", "detail": str(exc)}
    return WEBHOOKS.test(sid)
