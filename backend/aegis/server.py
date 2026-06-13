"""AegisOS API server — the integration surface for the (future) frontend.

Exposes the unified runtime over a stable WebSocket + REST contract:

  WS   /aegis/stream         push one snapshot per tick (the live feed)
  GET  /aegis/snapshot       current full snapshot (schema in aegis/stream.py)
  POST /aegis/ingest         push a real telemetry reading (wearable/camera)
  POST /aegis/simulate/{stage}  drive the built-in physiology simulator
  GET  /aegis/health         liveness + session/source state

A built-in simulator drive loop produces a live stream out of the box (so the
frontend can be developed against real-shaped data), and verified protocols are
fed back into the simulator to close the treatment loop. Real telemetry pushed
via /aegis/ingest takes precedence. No frontend is bundled — this is the backend
contract only.

Run:  python -m backend.aegis.server      (or: uvicorn backend.aegis.server:app)
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import os as _os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.aegis.stream import AegisSession
from backend.aegis.kernel import ALERT, BioCluster, NeuroSymbolicRuntime
from backend.core.simulator import PatientTelemetrySimulator

app = FastAPI(title="AegisOS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Single live session (one monitored patient per process for now) -------- #
session = AegisSession(
    patient_id="aegis-demo-patient",
    patient_name="Jane Doe",
    age=67, gender="Female",
    clinical_history={"chf": True, "renal_failure": False,
                      "penicillin_allergy": True, "wbc": 7.5, "gcs": 15},
    use_llm=False,
)
simulator = PatientTelemetrySimulator()
DRIVE_FROM_SIMULATOR = True
TICK_SECONDS = 1.5

_clients: List[WebSocket] = []
_active_treatment: Dict[str, float] = {"fluid_rate": 0.0, "vasopressor_dose": 0.0, "antibiotic_code": 0}

# --- The simulated ICU wing: a live BioCluster driving the Tactical-Sphere
#     cockpit. This is the kernel's multi-tenant face — N patient nodes on one
#     DevBio bus + zero-copy memory fabric, governed by the cognitive scheduler,
#     with a Chaos Controller for live crisis injection. Every headline alert is
#     routed through the Layer-4 egress cage so the dashboard shows real Z3
#     certification counts. -------------------------------------------------- #
CLUSTER_NODE_NAMES = {
    "node_0": "Okafor, A.", "node_1": "Bauer, M.", "node_2": "Reyes, L.",
    "node_3": "Nguyen, T.", "node_4": "Castellano, R.", "node_5": "Yusuf, D.",
}
cluster = BioCluster(n_nodes=6, lane_budget=96,
                     weights=[1.0, 1.2, 1.5, 1.0, 1.3, 1.1], seed=7,
                     names=[CLUSTER_NODE_NAMES[f"node_{i}"] for i in range(6)])
cluster_safety = NeuroSymbolicRuntime()
CLUSTER_TICK_SECONDS = 0.6
_cluster_clients: List[WebSocket] = []
cluster.run(12)                                   # warm the rings before serving

# --- Real-data persistence (Supabase). The simulator demo is NEVER written; only
#     real ingested readings are, and they are throttled + tagged so the database
#     and egress stay well under the free-tier limits. -------------------------- #
DB_PATIENT_ID = "a0e1d2c3-0000-4000-8000-000000000001"   # stable UUID for the monitored patient
_db = None
_last_real_ingest = 0.0
LIVE_HOLD_SECONDS = 6.0           # after a real reading the camera owns the stream this long
TELEMETRY_EVERY = 3.0            # max one telemetry row / 3 s  (history fidelity; tiny footprint)
ALERT_EVERY = 15.0              # max one alert row / 15 s while elevated
PROTOCOL_EVERY = 30.0
RETAIN_EVERY = 3600.0          # opportunistic retention call (belt-and-suspenders w/ pg_cron)
_persist_last = {"telemetry": 0.0, "alert": 0.0, "protocol": 0.0}
_retain_last = 0.0


def _live_mode() -> bool:
    """True while real telemetry is actively flowing — the simulator stands down."""
    return (time.time() - _last_real_ingest) < LIVE_HOLD_SECONDS


def _persist_snapshot(snap: Dict[str, Any], telemetry: Dict[str, float], source: str) -> None:
    """Persist a REAL reading to Supabase, throttled. Runs in a worker thread so it
    never adds latency to the live stream. All writes are best-effort."""
    if _db is None:
        return
    now = time.time()
    try:
        if telemetry.get("heart_rate") and now - _persist_last["telemetry"] >= TELEMETRY_EVERY:
            _db.save_telemetry(DB_PATIENT_ID, telemetry, source=source)
            _persist_last["telemetry"] = now
    except Exception as exc:
        print(f"[aegis] telemetry persist failed: {exc}")
    try:
        risk = snap.get("risk") or {}
        if risk.get("level") in ("MEDIUM", "HIGH") and now - _persist_last["alert"] >= ALERT_EVERY:
            det = snap.get("detection") or {}
            sim = det.get("sepsis_similarity", det.get("similarity", 0.0)) or 0.0
            _db.save_alert(str(uuid4()), DB_PATIENT_ID, risk["level"], float(sim),
                           int(risk.get("sirs_score") or 0), int(risk.get("qsofa_score") or 0),
                           list(snap.get("triggered_rules") or []), source=source)
            _persist_last["alert"] = now
    except Exception as exc:
        print(f"[aegis] alert persist failed: {exc}")
    try:
        proto = snap.get("protocol")
        if proto and now - _persist_last["protocol"] >= PROTOCOL_EVERY:
            p = proto.get("protocol") or {}
            ver = snap.get("verification") or {}
            pid = str(uuid4())
            _db.save_protocol(pid, DB_PATIENT_ID, None,
                              float(p.get("fluid_rate") or 0), float(p.get("vasopressor_dose") or 0),
                              str(p.get("antibiotic_selection") or p.get("antibiotic_code") or "n/a"),
                              float(proto.get("fitness") or 0),
                              "VERIFIED" if ver.get("safe") else "REJECTED")
            _db.save_verification_proof(str(uuid4()), pid,
                              str(ver.get("message") or "safety_constraints"),
                              str(ver.get("z3_status") or "unknown"),
                              {"violated_rules": ver.get("violated_rules", [])})
            _persist_last["protocol"] = now
    except Exception as exc:
        print(f"[aegis] protocol persist failed: {exc}")
    global _retain_last
    if now - _retain_last >= RETAIN_EVERY:
        _retain_last = now
        try:
            if getattr(_db, "use_supabase", False) and _db.supabase_client:
                _db.supabase_client.rpc("aegis_retain").execute()
        except Exception:
            pass


class TelemetryIn(BaseModel):
    telemetry: Dict[str, float]
    source: str = "wearable"
    camera_vitals: Optional[Dict[str, Any]] = None
    signal_quality: float = 1.0


async def _broadcast(snapshot: Dict[str, Any]) -> None:
    dead = []
    for ws in _clients:
        try:
            await ws.send_json(snapshot)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _clients:
            _clients.remove(ws)


def _apply_treatment_from(snapshot: Dict[str, Any]) -> None:
    """Close the loop: feed a verified+applied protocol back to the simulator."""
    global _active_treatment
    proto = snapshot.get("protocol")
    if proto and proto.get("applied"):
        _active_treatment = dict(proto["protocol"])


async def _drive_loop() -> None:
    print("[aegis] simulator drive loop started (demo mode; yields to real data).")
    while True:
        try:
            # The simulator only drives the stream as a DEMO — the moment real
            # telemetry arrives it stands down so the camera owns the live feed.
            if DRIVE_FROM_SIMULATOR and not _live_mode():
                vitals = simulator.next_tick(_active_treatment)
                snap = session.tick(vitals, source="simulator")
                _apply_treatment_from(snap)
                await _broadcast(snap)
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] drive loop error: {exc}")
        await asyncio.sleep(TICK_SECONDS)


@app.on_event("startup")
async def _startup() -> None:
    global _db
    try:
        from backend.core.database import DatabaseClient
        _db = DatabaseClient()
        _db.save_patient(DB_PATIENT_ID, session.patient_name, session.age or 0,
                         session.gender or "Unknown", session.core.clinical_history or {})
        session.os.register_source("supabase-persistence")
        print("[aegis] real-data persistence layer ready.")
    except Exception as exc:  # pragma: no cover
        print(f"[aegis] persistence unavailable ({exc}); running without it.")
    asyncio.create_task(_drive_loop())
    asyncio.create_task(_cluster_loop())


# --- REST ------------------------------------------------------------------- #
@app.get("/aegis/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "source": session.source,
            "mode": "live" if _live_mode() else "demo",
            "persistence": bool(_db is not None and getattr(_db, "use_supabase", False)),
            "model_loaded": session.core.model is not None,
            "schema_version": session.snapshot()["schema_version"]}


@app.get("/aegis/snapshot")
def snapshot() -> Dict[str, Any]:
    return session.snapshot()


@app.get("/aegis/system")
def system() -> Dict[str, Any]:
    """The OS layer map + status (HAL, kernel, scheduler, reasoning, ...)."""
    return session.os.status()


@app.post("/aegis/whatif")
def whatif() -> Dict[str, Any]:
    """Run the predictive what-if treatment exploration on demand."""
    return session.os.whatif(session.latest)


@app.post("/aegis/ingest")
async def ingest(payload: TelemetryIn) -> Dict[str, Any]:
    """Push a REAL telemetry reading. This takes over the live stream from the
    simulator demo, is broadcast to all WS clients, and is persisted (throttled)
    to Supabase off the realtime path."""
    global _last_real_ingest
    snap = session.tick(payload.telemetry, source=payload.source,
                        camera_vitals=payload.camera_vitals,
                        signal_quality=payload.signal_quality)
    _last_real_ingest = time.time()
    session.os.register_source(payload.source)
    _apply_treatment_from(snap)
    await _broadcast(snap)
    asyncio.create_task(asyncio.to_thread(_persist_snapshot, snap, payload.telemetry, payload.source))
    return snap


@app.post("/aegis/simulate/{stage}")
def simulate(stage: str) -> Dict[str, str]:
    global _active_treatment
    stage = stage.upper()
    if stage not in {"STABLE", "DETERIORATING", "SEPSIS", "RECOVERING"}:
        return {"status": "error", "detail": f"unknown stage '{stage}'"}
    simulator.set_stage(stage)
    if stage == "STABLE":
        _active_treatment = {"fluid_rate": 0.0, "vasopressor_dose": 0.0, "antibiotic_code": 0}
    return {"status": "ok", "stage": stage}


# --- Cluster cockpit (Tactical Sphere) -------------------------------------- #
def _evidence_count(v: Dict[str, float]) -> int:
    """How many vital axes are deranged — feeds the L4 alert evidence term."""
    n = 0
    if v.get("hr", 0) > 100: n += 1
    if v.get("resp_rate", 0) > 22: n += 1
    t = v.get("temp_c")
    if t is not None and (t > 38.0 or t < 36.0): n += 1
    if v.get("map", 999) < 65: n += 1
    if v.get("spo2", 100) < 92: n += 1
    return n


def _certify_cluster_headline(decision) -> None:
    """Route the cluster's highest-risk node alert through the Z3 egress cage."""
    if not cluster.sched.nodes:
        return
    top = max(cluster.sched.nodes.values(), key=lambda nd: nd.p)
    p = float(top.p)
    severity = 3 if p >= 0.70 else 2 if p >= 0.40 else 1 if p >= 0.20 else 0
    cluster_safety.dispatch(ALERT, {"severity": severity}, {
        "probability": p, "calibrated": True,
        "evidence_count": _evidence_count(top.vitals),
    })


def _cluster_payload() -> Dict[str, Any]:
    snap = cluster.snapshot()
    for node in snap["topology"]:                 # attach human-readable names
        node["name"] = CLUSTER_NODE_NAMES.get(node["node_id"], node["node_id"])
    snap["devbio"] = cluster.bus.status()         # Panel D — µs conditioning latency
    snap["safety"] = cluster_safety.status()      # L4 egress-cage counters
    snap["ts"] = time.time()
    return snap


async def _broadcast_cluster(payload: Dict[str, Any]) -> None:
    dead = []
    for ws in _cluster_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _cluster_clients:
            _cluster_clients.remove(ws)


async def _cluster_loop() -> None:
    print("[aegis] ICU-wing cluster loop started (Tactical Sphere telemetry).")
    while True:
        try:
            decision = cluster.tick()
            _certify_cluster_headline(decision)
            await _broadcast_cluster(_cluster_payload())
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] cluster loop error: {exc}")
        await asyncio.sleep(CLUSTER_TICK_SECONDS)


class ChaosIn(BaseModel):
    node: str
    action: str = "crisis"            # crisis | clear | noise
    intensity: float = 1.0


@app.get("/aegis/cluster/snapshot")
def cluster_snapshot() -> Dict[str, Any]:
    """Full Tactical-Sphere snapshot: topology (A), memfabric (B), lane map (C),
    devbio latency (D), scheduler + L4 cage counters + interrupt events."""
    return _cluster_payload()


@app.post("/aegis/cluster/chaos")
async def cluster_chaos(payload: ChaosIn) -> Dict[str, Any]:
    """The judges' Chaos Controller: inject/clear a crisis or a noise surge on a
    node and watch the cluster reallocate compute in real time."""
    if payload.node not in cluster.node_ids:
        return {"status": "error", "detail": f"unknown node '{payload.node}'"}
    act = payload.action.lower()
    if act == "crisis":
        cluster.inject_crisis(payload.node, intensity=payload.intensity)
    elif act == "clear":
        cluster.clear_crisis(payload.node)
    elif act == "noise":
        cluster.inject_noise_surge(payload.node)
    else:
        return {"status": "error", "detail": f"unknown action '{payload.action}'"}
    return {"status": "ok", "node": payload.node, "action": act}


@app.websocket("/aegis/cluster/stream")
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


# --- WebSocket -------------------------------------------------------------- #
@app.websocket("/aegis/stream")
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


# --- AegisOS shell (the OS frontend) ---------------------------------------- #
# Mounted LAST so the API routes above take precedence; "/" serves the shell.
_WEBUI = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "webui")


class _NoCacheStatic(StaticFiles):
    """Serve the shell with no-store so live edits (incl. ES modules) always
    load fresh — browsers otherwise cache modules across reloads/restarts."""

    def is_not_modified(self, *a, **k) -> bool:  # defeat 304 revalidation
        return False

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp


if _os.path.isdir(_WEBUI):
    app.mount("/", _NoCacheStatic(directory=_WEBUI, html=True), name="aegisos-shell")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("backend.aegis.server:app", host="127.0.0.1", port=8100, reload=False)
