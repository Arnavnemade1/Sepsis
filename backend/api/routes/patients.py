"""The patient roster, per-bed snapshots and dossiers, medical history, the cohort ward and physiologic identity.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.schemas import (
    ChaosIn,
    HistoryIn,
    IdentitySwapIn,
    LinkIn,
    SynthIn,
    TrackIn,
)
from backend.aegis.server import (
    DATA_MODE,
    EXEC,
    _bed_snapshot,
    _cluster_clients,
    _cluster_payload,
    _identity_service,
    _patient_ctx,
    _patient_roster,
    _roster_entry,
    bundle_engine,
    cluster,
    patient_terminal,
    session,
)

router = APIRouter()


@router.get("/aegis/patient/snapshot")
def patient_snapshot(bed: Optional[str] = None) -> Dict[str, Any]:
    """Per-patient desktop snapshot: live session, or a cohort bed's recorded state."""
    if not bed or bed == session.patient_id:
        return session.snapshot()
    s = _bed_snapshot(bed)
    return s or {"status": "error", "detail": f"no such patient '{bed}'"}


@router.get("/aegis/patients")
def patients() -> Dict[str, Any]:
    """The patient picker roster (live monitored patient + real cohort beds)."""
    return {"patients": _patient_roster(), "default": session.patient_id}


@router.get("/aegis/identity")
def identity_status() -> Dict[str, Any]:
    """Enrolled-cohort status + the measured same-vs-different separability."""
    svc = _identity_service()
    return {"status": svc.status(), "validation": svc.validation()}


@router.post("/aegis/identity/swap")
def identity_swap(p: IdentitySwapIn) -> Dict[str, Any]:
    """Demo: connect ``source``'s stream to ``target``'s monitor → identity fault →
    Z3-locked dispensing. With no args, picks two enrolled patients."""
    svc = _identity_service()
    pids = svc.patients()
    if not pids:
        return {"status": "error", "detail": "no enrolled patients"}
    target = p.target or pids[0]
    source = p.source or next((x for x in pids if x != target), pids[0])
    if target not in pids or source not in pids:
        return {"status": "error", "detail": "unknown patient id"}
    return svc.swap(target, source)


@router.post("/aegis/identity/self_check")
def identity_self_check(p: IdentitySwapIn) -> Dict[str, Any]:
    """Verify a patient against their own later stream (the no-fault baseline case)."""
    svc = _identity_service()
    pids = svc.patients()
    target = p.target or (pids[0] if pids else None)
    if target not in pids:
        return {"status": "error", "detail": "unknown patient id"}
    return svc.self_check(target)


@router.get("/aegis/cluster/snapshot")
def cluster_snapshot() -> Dict[str, Any]:
    """Full Tactical-Sphere snapshot: topology (A), memfabric (B), lane map (C),
    devbio latency (D), scheduler + L4 cage counters + interrupt events."""
    return _cluster_payload()


@router.get("/aegis/cohort")
def cohort_roster() -> Dict[str, Any]:
    """The monitored patient roster — real de-identified CinC-2019 ICU stays."""
    if not hasattr(cluster, "roster"):
        return {"source": "synthetic", "mode": DATA_MODE, "n": len(cluster.node_ids),
                "patients": []}
    return {"source": "PhysioNet/CinC-2019 Sepsis Challenge", "mode": DATA_MODE,
            "n": len(cluster.node_ids), "patients": cluster.roster()}


@router.get("/aegis/cohort/validation")
def cohort_validation() -> Dict[str, Any]:
    """Real sepsis-detection performance across the cohort vs ground-truth labels."""
    if not hasattr(cluster, "validation"):
        return {"status": "unavailable", "mode": DATA_MODE}
    return cluster.validation(alarm=0.70)


@router.get("/aegis/cohort/{bed_id}/bundle")
def cohort_bundle(bed_id: str) -> Dict[str, Any]:
    """On-demand certified Hour-1 bundle for ANY bed (the dossier's protocol)."""
    if bed_id not in getattr(cluster, "profiles", {}):
        return {"status": "error", "detail": f"unknown bed '{bed_id}'"}
    return bundle_engine.generate(
        cluster.profiles[bed_id], getattr(cluster, "histories", {}).get(bed_id, {}))


@router.get("/aegis/cohort/{bed_id}/chart")
def cohort_chart(bed_id: str) -> Dict[str, Any]:
    """One patient's real vitals/labs trajectory + ground-truth sepsis label."""
    if not hasattr(cluster, "chart"):
        return {"status": "unavailable", "mode": DATA_MODE}
    ch = cluster.chart(bed_id)
    if ch is None:
        return {"status": "error", "detail": f"unknown bed '{bed_id}'"}
    return ch


@router.get("/aegis/patient/{bed}/tracking")
def patient_tracking(bed: str) -> Dict[str, Any]:
    """This patient's tracking profile (foci + watch-rules) + any watch-rules that
    fired against the current vitals (edge-triggered here, the single evaluator) +
    the command vocabulary. The dossier polls this."""
    from backend.aegis.patient_terminal import METRICS as _M, ORGANS as _O
    snap = _bed_snapshot(bed) or {}
    fired = patient_terminal.evaluate(bed, snap.get("vitals", {}))
    return {"profile": patient_terminal.profile(bed).to_dict(), "fired": fired,
            "vocab": {"metrics": {k: v["label"] for k, v in _M.items()}, "organs": _O}}


@router.post("/aegis/patient/{bed}/track")
def patient_track(bed: str, payload: TrackIn) -> Dict[str, Any]:
    """Run one per-patient tracking command (structured verb OR natural language).
    Observation-only: the result can add a focus or a watch, never an order."""
    return patient_terminal.exec(bed, payload.command, context=_patient_ctx(bed))


@router.get("/aegis/records/sources")
def records_sources() -> Dict[str, Any]:
    """Registered records-source connectors (manual paste/upload + the MCP-shaped
    demo hospital stub). Each carries a `live` flag so the UI is honest about stubs."""
    return {"sources": patient_terminal.sources()}


@router.get("/aegis/patient/{bed}/history")
def patient_history_get(bed: str) -> Dict[str, Any]:
    """The imported medical history for this patient (if any) + what it specialised."""
    return {"history": patient_terminal.get_history(bed),
            "sources": patient_terminal.sources()}


@router.post("/aegis/patient/{bed}/history")
def patient_history_import(bed: str, payload: HistoryIn) -> Dict[str, Any]:
    """Import a pasted/uploaded medical history → parse → seed foci + watch-rules so
    the monitor specialises toward this patient. Observation-only."""
    if not (payload.text or "").strip():
        return {"ok": False, "reply": "empty history"}
    return patient_terminal.import_history(bed, payload.text, source=payload.source or "manual")


@router.post("/aegis/patient/{bed}/history/link")
def patient_history_link(bed: str, payload: LinkIn) -> Dict[str, Any]:
    """Pull a history from a registered records source (stub / live FHIR EHR) and
    import it. ``patient_ref`` is the source's patient identifier (FHIR id / MRN)."""
    return patient_terminal.link_source(bed, payload.source, patient=_roster_entry(bed),
                                        ref=payload.patient_ref)


@router.get("/aegis/patient/{bed}/synthesis")
def patient_synthesis_get(bed: str) -> Dict[str, Any]:
    """The clinician synthesis / instruction log for this patient (annotations shown
    in the clinical story)."""
    return {"annotations": patient_terminal.get_annotations(bed)}


@router.post("/aegis/patient/{bed}/synthesis")
def patient_synthesis_add(bed: str, payload: SynthIn) -> Dict[str, Any]:
    """Record a doctor's detailed synthesis/instruction. Stored as a timestamped
    annotation; monitoring intent (safe vocab) is applied; treatment intent is
    flagged and redirected to the certified order gate — never auto-executed."""
    return patient_terminal.add_synthesis(bed, payload.text, context=_patient_ctx(bed))


@router.post("/aegis/cluster/chaos")
async def cluster_chaos(payload: ChaosIn) -> Dict[str, Any]:
    """The judges' Chaos Controller: inject/clear a crisis or a noise surge on a
    node and watch the cluster reallocate compute in real time."""
    if payload.node not in cluster.node_ids:
        return {"status": "error", "detail": f"unknown node '{payload.node}'"}
    act = payload.action.lower()
    if act == "crisis":
        EXEC.call(cluster.inject_crisis, payload.node, intensity=payload.intensity)
    elif act == "clear":
        EXEC.call(cluster.clear_crisis, payload.node)
    elif act == "noise":
        EXEC.call(cluster.inject_noise_surge, payload.node)
    else:
        return {"status": "error", "detail": f"unknown action '{payload.action}'"}
    return {"status": "ok", "node": payload.node, "action": act}


@router.websocket("/aegis/cluster/stream")
async def cluster_stream(ws: WebSocket) -> None:
    await ws.accept()
    _cluster_clients.append(ws)
    await ws.send_json(_cluster_payload())
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _cluster_clients:
            _cluster_clients.remove(ws)

