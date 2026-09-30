"""The live feed: the snapshot WebSocket, telemetry ingest, the simulator drive, what-if and the decision surface.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import backend.aegis.server as S
from backend.api.schemas import (
    TelemetryIn,
)
from backend.aegis.server import (
    _apply_treatment_from,
    _broadcast,
    _clients,
    _persist_snapshot,
    session,
    simulator,
)

router = APIRouter()


@router.get("/aegis/timeline")
def risk_timeline(bed: Optional[str] = None, horizon: int = 8) -> Dict[str, Any]:
    """The risk timeline, drawn from the model that actually decides.

    Replaces the fused-risk chart, which plotted a fusion score against a hardcoded
    0.70 while the shipped detector alarms at 0.03 on its calibrated probability.
    Every element here resolves to a shipped artifact and ``provenance`` names which.
    """
    from backend.aegis import risk_timeline as _rt
    snap = S.patient_snapshot(bed) if bed else snapshot()
    hist = (snap or {}).get("history") or {}
    probs = list(hist.get("probability") or [])
    if not probs:
        cur = ((snap or {}).get("risk") or {}).get("ml_probability")
        probs = [float(cur)] if cur is not None else []
    onset = None
    det = (snap or {}).get("detection") or {}
    if det.get("labelled_onset_hour") is not None:
        onset = float(det["labelled_onset_hour"])
    tl = _rt.build(probs, horizon=horizon, onset_hour=onset)
    out = tl.as_dict()
    out["bed"] = bed
    return out


@router.get("/aegis/decision")
def clinical_decision(bed: Optional[str] = None) -> Dict[str, Any]:
    """What the system DOES with this patient's score, and why.

    The decision layer is where the remaining clinical value is: the same detector read
    three ways produces false-alarm rates of 14%, 27% and 94%, a wider spread than any
    modelling change this project has found. Returns one auditable disposition — the
    action, who it routes to, the ordered reasons, the measured cost of the policy, and
    the limits that qualify it.
    """
    from backend.aegis import disposition as _disp
    snap = S.patient_snapshot(bed) if bed else snapshot()
    risk = (snap or {}).get("risk") or {}
    cert = risk.get("certainty") or {}
    # A hard override is qSOFA >= 2 or a shock signal — core.py's own definition, and
    # it stamps ``certainty.override`` when one fires. It is NOT "any triggered rule":
    # ``triggered_rules`` carries descriptive SIRS criteria such as
    # "SIRS: Tachycardia (98.0 bpm)", and reading the first of those as an escalation
    # made almost every mildly abnormal patient ALARM and turned the whole abstention
    # path into dead code. Derived here rather than re-declared, with a fallback for
    # payloads that predate the stamp.
    override = cert.get("override")
    if not override:
        det_blk = (snap or {}).get("detection") or {}
        if int(risk.get("qsofa_score") or 0) >= 2:
            override = "qSOFA>=2"
        elif det_blk.get("shock_signal"):
            override = "shock signal"
    d = _disp.decide(
        risk_level=str(risk.get("level") or "LOW"),
        ml_probability=risk.get("ml_probability"),
        certainty=risk.get("certainty"),
        override=override,
    )
    out = d.as_dict()
    out["bed"] = bed
    out["risk_level"] = risk.get("level")
    out["ml_probability"] = risk.get("ml_probability")
    return out


@router.get("/aegis/snapshot")
def snapshot() -> Dict[str, Any]:
    return session.snapshot()


@router.get("/aegis/system")
def system() -> Dict[str, Any]:
    """The OS layer map + status (HAL, kernel, scheduler, reasoning, ...)."""
    return session.os.status()


@router.post("/aegis/whatif")
def whatif() -> Dict[str, Any]:
    """Run the predictive what-if treatment exploration on demand."""
    return session.os.whatif(session.latest)


@router.post("/aegis/ingest")
async def ingest(payload: TelemetryIn) -> Dict[str, Any]:
    """Push a REAL telemetry reading. This takes over the live stream from the
    simulator demo, is broadcast to all WS clients, and is persisted (throttled)
    to Supabase off the realtime path."""
    # session.tick runs a Z3-guarded assessment. This handler is async, so calling
    # it inline put a blocking solve on the event loop — which z3lock.py:123-126
    # explicitly forbids, and which stalls every other client for its duration.
    snap = await asyncio.to_thread(
        session.tick, payload.telemetry, source=payload.source,
        camera_vitals=payload.camera_vitals,
        signal_quality=payload.signal_quality,
        evidence_capabilities=payload.evidence_capabilities)
    S._last_real_ingest = time.time()
    session.os.register_source(payload.source)
    _apply_treatment_from(snap)
    await _broadcast(snap)
    asyncio.create_task(asyncio.to_thread(_persist_snapshot, snap, payload.telemetry, payload.source))
    return snap


@router.post("/aegis/simulate/{stage}")
def simulate(stage: str) -> Dict[str, str]:
    stage = stage.upper()
    if stage not in {"STABLE", "DETERIORATING", "SEPSIS", "RECOVERING"}:
        return {"status": "error", "detail": f"unknown stage '{stage}'"}
    simulator.set_stage(stage)
    if stage == "STABLE":
        S._active_treatment = {"fluid_rate": 0.0, "vasopressor_dose": 0.0, "antibiotic_code": 0}
    return {"status": "ok", "stage": stage}


@router.websocket("/aegis/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    _clients.append(ws)
    await ws.send_json(session.snapshot())     # send current state immediately
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
        if ws in _clients:
            _clients.remove(ws)

