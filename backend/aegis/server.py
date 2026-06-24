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
from backend.aegis.kernel import ALERT, BioCluster, NeuroSymbolicRuntime, SepsisBundleEngine
from backend.aegis.cohort import CohortCluster
from backend.aegis.devices import DeviceRegistry
from backend.aegis.syscalls import SyscallTable
from backend.aegis.kernel.runtime import KernelDeviceRuntime
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

# --- The live ICU wing. Preferred: a CohortCluster of REAL de-identified ICU
#     patients from the PhysioNet/CinC-2019 corpus (each bed replays its actual
#     hourly vitals + labs through /dev/bio -> memory fabric -> fusion -> scheduler
#     -> L4 cage; septic patients deteriorate per the real record and cross
#     threshold near their true onset). The corpus is gitignored (PhysioNet
#     redistribution terms), so when it is absent — e.g. on a fresh deploy — we
#     fall back to the synthetic wing, which exercises the identical kernel. ---- #
try:
    cluster = CohortCluster(n=12, lane_budget=160, seed=7, ebpf_optical=3)
    if not cluster.node_ids:
        raise RuntimeError("CinC-2019 corpus not found under data/cinc2019")
    DATA_MODE = "real-cinc2019"
    print(f"[aegis] ICU wing: {len(cluster.node_ids)} REAL CinC-2019 patients.")
except Exception as exc:  # pragma: no cover
    print(f"[aegis] real cohort unavailable ({exc}); using synthetic wing.")
    cluster = BioCluster(
        n_nodes=6, lane_budget=96, weights=[1.0, 1.2, 1.5, 1.0, 1.3, 1.1], seed=7,
        names=["Okafor, A.", "Bauer, M.", "Reyes, L.", "Nguyen, T.",
               "Castellano, R.", "Yusuf, D."])
    DATA_MODE = "synthetic"
cluster_safety = NeuroSymbolicRuntime()
bundle_engine = SepsisBundleEngine()
device_registry = DeviceRegistry()                # simulated sensor fleet (Phase A)
# The write-side trap gate: every treat/escalate/crisis/clear from the console is
# routed through this audited, Z3-gated interface before it touches the kernel.
syscalls = SyscallTable(cluster, bundle_engine, cluster_safety)
# The input-safety verifier service: holds the live Aegis-eBPF static verifier +
# the driver catalogue used by `verify`. The verified-driver DATA PLANE itself is
# the cohort's own beds (those with `ebpf_optical` enabled) — real patients whose
# optical sensor is driven by proven-safe bytecode, supervised for fault isolation.
kernel_devices = KernelDeviceRuntime()
# Live PPG-morphology vascular-tone monitor (real BIDMC waveform replay). Standalone
# pathway — NOT fused into the cohort resonance gate (different patients / no
# synchronized vitals); see backend/physio/ppg_monitor.py.
try:
    from backend.physio.ppg_monitor import PPGMorphologyMonitor
    ppg_monitor = PPGMorphologyMonitor()
except Exception as _exc:                          # pragma: no cover
    print(f"[aegis] PPG morphology monitor unavailable: {_exc}")
    ppg_monitor = None

# --- aegisd: the live application/service plane ----------------------------- #
# The scattered diagnostics now run as managed, supervised SERVICES on the kernel:
# registered with capability manifests, started in dependency order, ticked each
# cycle inside the app sandbox, fault-isolated, and introspectable via /proc.
import inspect as _inspect
from backend.aegis.apps.examples import tachy_app as _tachy_mod
from backend.aegis.apps.examples.tachy_app import TachyApp
from backend.aegis.apps.sdk import AppManifest as _AppManifest
from backend.aegis.apps.radar_app import RadarApp
from backend.aegis.apps.resonance_app import ResonanceApp
from backend.aegis.apps.triage_app import TriageApp
from backend.aegis.initd import ServiceManager
from backend.aegis.kernel.ebpf import Verifier as _EbpfVerifier
from backend.aegis.kernel.ebpfdrv import DEFAULT_HELPERS as _HELPERS
from backend.aegis.pkg import AppPackage, PackageRegistry
from backend.aegis.procfs import ProcFS

apps_cage = NeuroSymbolicRuntime()                 # egress cage for app outputs
procfs = ProcFS()
aegisd = ServiceManager(apps_cage, procfs)
# the package registry: every install is gated (manifest → policy → code
# verification → integrity) before aegisd ever schedules it.
pkg_registry = PackageRegistry(aegisd, verifier=_EbpfVerifier(_HELPERS))
# the installable catalogue (an example third-party app); factory rebuilds a fresh,
# signed package on demand.
AVAILABLE_PKGS = {
    "tachy": lambda: AppPackage(TachyApp.manifest, _inspect.getsource(_tachy_mod),
                                kind="python", app=TachyApp()).sign(),
    # a deliberately-hostile package so the install gate's refusal is demonstrable
    # live: its source imports a network module → the static audit rejects it.
    "rogue": lambda: AppPackage(
        _AppManifest(name="rogue", reads=("hr",), emits=(ALERT,),
                     summary="(demo) tries to open a socket — the registry must refuse"),
        "import socket\nclass Rogue:\n    def on_tick(self, ctx):\n        return {}\n",
        kind="python", app=TachyApp()).sign(),
}
# kernel state, exposed read-only through the one introspection namespace
procfs.register("/proc/sched", lambda: cluster.sched.kstat() if hasattr(cluster, "sched") else {})
procfs.register("/proc/mem", lambda: cluster.fabric.address_matrix()["arena"] if hasattr(cluster, "fabric") else {})
procfs.register("/proc/audit", lambda: syscalls.status())
procfs.register("/proc/cage", lambda: apps_cage.status())
try:
    if hasattr(cluster, "resonance"):              # real cohort only (fitted gate)
        aegisd.register(ResonanceApp(cluster.resonance))
        aegisd.register(RadarApp())
        aegisd.register(TriageApp())
        aegisd.start_all()
    # the service plane runs on one monitored patient (a septic bed → coherent
    # stateful radar window); the per-bed resonance still drives the cohort UI.
    SERVICE_BED = next((nid for nid in getattr(cluster, "node_ids", [])
                        if getattr(cluster._by.get(nid), "is_septic", False)),
                       (cluster.node_ids[0] if getattr(cluster, "node_ids", None) else None))
except Exception as _exc:                          # pragma: no cover
    print(f"[aegis] aegisd service plane unavailable: {_exc}")
    SERVICE_BED = None


def _kernel_plane(full: bool = False) -> Dict[str, Any]:
    """Unify the verified device plane for the UI/console: the real eBPF-driven
    cohort beds (data plane) + the live verifier counters + the supervisor log."""
    devices = cluster.driver_plane_status() if hasattr(cluster, "driver_plane_status") else []
    supv = cluster.driver_sup.status() if hasattr(cluster, "driver_sup") else {"events": [], "slots": {}}
    base = kernel_devices.status(full=full)
    out: Dict[str, Any] = {"devices": devices, "verifier": base["verifier"], "supervisor": supv}
    if full:
        out["programs"] = base.get("programs", [])
    return out


def _resolve_ebpf_bed(q: str) -> Optional[str]:
    """Map a console token (bed id, 1-based tag, or name) to an eBPF-driven bed."""
    beds = getattr(cluster, "ebpf_beds", {})
    q = (q or "").strip().lower()
    if q in beds:
        return q
    import re
    m = re.match(r"^(?:bed[ _-]?)?0*(\d+)$", q)
    if m and f"bed_{int(m.group(1)) - 1}" in beds:
        return f"bed_{int(m.group(1)) - 1}"
    for bid in beds:
        if q and q in cluster.names.get(bid, "").lower():
            return bid
    return None
_bundle = {"bed": None, "ts": 0.0, "data": None}
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


def _focus_bundle() -> Optional[Dict[str, Any]]:
    """Compile + certify the Hour-1 bundle for the critical focus bed (the preempt
    owner, else the highest-risk bed once it breaches). Throttled so the Z3 prover
    runs at most ~every 2.5 s per bed rather than every broadcast tick."""
    owner = cluster.sched.last.owner if cluster.sched.last else None
    focus = None
    if owner and owner in cluster.profiles and cluster.profiles[owner].fused_risk >= 0.70:
        focus = owner
    else:
        crit = [(p.fused_risk, nid) for nid, p in cluster.profiles.items() if p.fused_risk >= 0.70]
        if crit:
            focus = max(crit)[1]
    if not focus:
        _bundle.update(bed=None, data=None)
        return None
    now = time.time()
    if _bundle["bed"] != focus or now - _bundle["ts"] > 2.5:
        _bundle["data"] = bundle_engine.generate(
            cluster.profiles[focus], getattr(cluster, "histories", {}).get(focus, {}))
        _bundle["bed"] = focus
        _bundle["ts"] = now
    return _bundle["data"]


def _cluster_payload() -> Dict[str, Any]:
    snap = cluster.snapshot()
    for node in snap["topology"]:                 # attach human-readable names
        node["name"] = cluster.names.get(node["node_id"], node["node_id"])
    snap["devbio"] = cluster.bus.status()         # Panel D — µs conditioning latency
    snap["safety"] = cluster_safety.status()      # L4 egress-cage counters
    snap["syscalls"] = syscalls.status()          # write-side trap gate: audit + live interventions
    snap["kernel_devices"] = _kernel_plane(full=False)  # verified eBPF device plane (real beds)
    if SERVICE_BED is not None:                    # managed app/service plane (aegisd)
        s = aegisd.status()
        s["bed"] = SERVICE_BED
        s["bed_name"] = cluster.names.get(SERVICE_BED, SERVICE_BED) if hasattr(cluster, "names") else SERVICE_BED
        snap["services"] = s
        snap["packages"] = {"installed": list(pkg_registry.installed.keys()),
                            "available": [p["name"] for p in _pkg_catalogue()]}
    if ppg_monitor is not None:                   # compact live vascular-tone summary
        st = ppg_monitor.state()
        snap["ppg"] = {k: st.get(k) for k in
                       ("available", "record", "tone", "notch_rate", "ba_ratio",
                        "reflection_index", "hr", "alpha")}
    snap["bundle"] = _focus_bundle()              # Module C — certified Hour-1 bundle
    snap["data_mode"] = DATA_MODE
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
            decision = cluster.tick()              # ticks the cohort incl. its eBPF beds
            if SERVICE_BED is not None:            # tick the managed app/service plane
                aegisd.tick(cluster.frame_for(SERVICE_BED))
            if ppg_monitor is not None:
                ppg_monitor.tick(CLUSTER_TICK_SECONDS)   # advance the live PPG waveform
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


@app.get("/aegis/cohort")
def cohort_roster() -> Dict[str, Any]:
    """The monitored patient roster — real de-identified CinC-2019 ICU stays."""
    if not hasattr(cluster, "roster"):
        return {"source": "synthetic", "mode": DATA_MODE, "n": len(cluster.node_ids),
                "patients": []}
    return {"source": "PhysioNet/CinC-2019 Sepsis Challenge", "mode": DATA_MODE,
            "n": len(cluster.node_ids), "patients": cluster.roster()}


@app.get("/aegis/cohort/validation")
def cohort_validation() -> Dict[str, Any]:
    """Real sepsis-detection performance across the cohort vs ground-truth labels."""
    if not hasattr(cluster, "validation"):
        return {"status": "unavailable", "mode": DATA_MODE}
    return cluster.validation(alarm=0.70)


@app.get("/aegis/cohort/{bed_id}/bundle")
def cohort_bundle(bed_id: str) -> Dict[str, Any]:
    """On-demand certified Hour-1 bundle for ANY bed (the dossier's protocol)."""
    if bed_id not in getattr(cluster, "profiles", {}):
        return {"status": "error", "detail": f"unknown bed '{bed_id}'"}
    return bundle_engine.generate(
        cluster.profiles[bed_id], getattr(cluster, "histories", {}).get(bed_id, {}))


@app.get("/aegis/cohort/{bed_id}/chart")
def cohort_chart(bed_id: str) -> Dict[str, Any]:
    """One patient's real vitals/labs trajectory + ground-truth sepsis label."""
    if not hasattr(cluster, "chart"):
        return {"status": "unavailable", "mode": DATA_MODE}
    ch = cluster.chart(bed_id)
    if ch is None:
        return {"status": "error", "detail": f"unknown bed '{bed_id}'"}
    return ch


class DeviceAssignIn(BaseModel):
    bed_id: str
    label: str = ""


class DeviceActionIn(BaseModel):
    action: str                           # calibrate | reboot | ota | wake | decommission


@app.get("/aegis/devices")
def devices_fleet() -> Dict[str, Any]:
    """The simulated sensor fleet — every device, health-rolled-up for the hub."""
    return device_registry.fleet()


@app.get("/aegis/devices/discover")
def devices_discover() -> Dict[str, Any]:
    """Nearby unpaired devices (simulated BLE scan) available to pair."""
    return device_registry.discover()


@app.get("/aegis/devices/audit")
def devices_audit() -> Dict[str, Any]:
    """Pairing / assignment audit trail (device↔patient binding is PHI)."""
    return {"events": device_registry.audit_log()}


@app.get("/aegis/devices/{device_id}")
def device_detail(device_id: str) -> Dict[str, Any]:
    d = device_registry.get(device_id)
    return d.to_dict() if d else {"status": "error", "detail": "unknown device"}


@app.post("/aegis/devices/{device_id}/pair")
def device_pair(device_id: str) -> Dict[str, Any]:
    return device_registry.pair(device_id)


@app.post("/aegis/devices/{device_id}/assign")
def device_assign(device_id: str, payload: DeviceAssignIn) -> Dict[str, Any]:
    return device_registry.assign(device_id, payload.bed_id, payload.label)


@app.post("/aegis/devices/{device_id}/unassign")
def device_unassign(device_id: str) -> Dict[str, Any]:
    return device_registry.unassign(device_id)


@app.post("/aegis/devices/{device_id}/action")
def device_action(device_id: str, payload: DeviceActionIn) -> Dict[str, Any]:
    return device_registry.action(device_id, payload.action)


class SyscallIn(BaseModel):
    syscall: str                                  # treat | escalate | crisis | clear
    bed: str
    args: Dict[str, Any] = {}
    caller: str = "console"


@app.post("/aegis/syscall")
async def aegis_syscall(payload: SyscallIn) -> Dict[str, Any]:
    """The write-side trap gate. The console invokes treat/escalate/crisis/clear
    here; the mutation is applied to the running kernel ONLY if the Z3 egress cage
    certifies it. Every call — allowed or denied — is appended to the audit log.
    A denied write returns EPERM and changes nothing; the certificate explains why."""
    res = syscalls.invoke(payload.syscall, payload.bed, payload.args, caller=payload.caller)
    # push the post-write state to all cockpits immediately so the grid reacts.
    if res.allowed:
        await _broadcast_cluster(_cluster_payload())
    return res.to_dict()


@app.get("/aegis/syscall/audit")
def aegis_syscall_audit(limit: int = 64) -> Dict[str, Any]:
    """The append-only syscall audit log — the system of record for every
    operator write and the prover's verdict on it."""
    return {"status": syscalls.status(), "log": syscalls.audit_log(limit)}


# --- Aegis-eBPF verified device plane (input-side safety) ------------------- #
class VerifyIn(BaseModel):
    name: Optional[str] = None             # a driver from the catalogue, or…
    source: Optional[str] = None           # …raw eBPF assembly to verify
    win_len: int = 2


class KernelFaultIn(BaseModel):
    slot: str                              # device key ("dev_0"), bay index, or label
    reason: str = "sensor disconnected (injected)"


@app.get("/aegis/kernel/devices")
def kernel_device_status() -> Dict[str, Any]:
    """The verified device plane: the real eBPF-driven cohort beds + their
    supervisor state, the live verifier counters, and the driver catalogue."""
    return _kernel_plane(full=True)


@app.post("/aegis/kernel/verify")
def kernel_verify(payload: VerifyIn) -> Dict[str, Any]:
    """Submit a driver to the LIVE static verifier. A catalogue ``name`` or raw
    ``source`` is statically analyzed; the verdict (ACCEPT + proof, or REJECT +
    the exact offending instruction and violated invariant) is computed now."""
    if payload.name:
        return kernel_devices.verify_named(payload.name)
    if payload.source:
        return kernel_devices.verify_source(payload.source, payload.win_len)
    return {"ok": False, "stage": "input", "error": "provide 'name' or 'source'"}


@app.post("/aegis/kernel/fault")
async def kernel_fault(payload: KernelFaultIn) -> Dict[str, Any]:
    """Kill a REAL patient bed's optical sensor → watch the supervisor quarantine
    the verified driver, fail over to its backup, and keep the patient monitored
    while every other bed keeps ticking."""
    bid = _resolve_ebpf_bed(payload.slot)
    if bid is None:
        return {"status": "error",
                "detail": f"{payload.slot!r} is not an eBPF-driven bed "
                          f"(verified beds: {', '.join(getattr(cluster, 'ebpf_beds', {}))})"}
    res = cluster.inject_driver_fault(bid, payload.reason)
    res = {"status": "ok", "bed": bid, "name": cluster.names.get(bid, bid), **(res or {})}
    await _broadcast_cluster(_cluster_payload())
    return res


@app.get("/aegis/services")
def services_status() -> Dict[str, Any]:
    """The managed app/service plane: every service's capabilities, health, and
    the supervisor's event log."""
    s = aegisd.status()
    s["bed"] = SERVICE_BED
    return s


@app.post("/aegis/services/{name}/{action}")
async def service_control(name: str, action: str) -> Dict[str, Any]:
    """systemctl-style control: start | stop | restart a service."""
    act = action.lower()
    ok = (aegisd.stop(name) if act == "stop" else
          aegisd.restart(name) if act in ("restart", "start") else None)
    if ok is None:
        return {"status": "error", "detail": f"unknown action {action!r}"}
    if not ok:
        return {"status": "error", "detail": f"unknown service {name!r}"}
    await _broadcast_cluster(_cluster_payload())
    return {"status": "ok", "service": name, "action": act,
            "state": aegisd.services[name].state}


class PkgIn(BaseModel):
    name: str


def _pkg_catalogue() -> List[Dict[str, Any]]:
    out = []
    for n, factory in AVAILABLE_PKGS.items():
        if n in pkg_registry.installed:
            continue
        try:
            out.append({"name": n, "manifest": factory().manifest.to_dict()})
        except Exception:
            pass
    return out


@app.get("/aegis/pkg")
def pkg_list() -> Dict[str, Any]:
    """Installed packages + the installable catalogue + the install audit log."""
    lst = pkg_registry.list()
    lst["available"] = _pkg_catalogue()
    return lst


@app.post("/aegis/pkg/install")
async def pkg_install(payload: PkgIn) -> Dict[str, Any]:
    """Install a catalogue package through the verify-then-register gate. On
    success it becomes a live, supervised service."""
    factory = AVAILABLE_PKGS.get(payload.name)
    if factory is None:
        return {"ok": False, "stage": "lookup",
                "reason": f"no such package {payload.name!r}",
                "available": list(AVAILABLE_PKGS)}
    res = pkg_registry.install(factory()).to_dict()
    await _broadcast_cluster(_cluster_payload())
    return res


@app.post("/aegis/pkg/uninstall")
async def pkg_uninstall(payload: PkgIn) -> Dict[str, Any]:
    ok = pkg_registry.uninstall(payload.name)
    await _broadcast_cluster(_cluster_payload())
    return {"ok": ok, "name": payload.name}


@app.get("/aegis/proc")
def proc_list() -> Dict[str, Any]:
    """The /proc introspection namespace (every live path + value)."""
    return {"paths": procfs.list(), "tree": procfs.tree()}


@app.get("/aegis/proc/{path:path}")
def proc_read(path: str) -> Dict[str, Any]:
    """Read one /proc path (e.g. services, triage, sched, mem, audit, cage)."""
    full = "/proc/" + path.lstrip("/")
    try:
        return {"path": full, "value": procfs.read(full)}
    except KeyError:
        return {"path": full, "error": "no such proc path",
                "available": procfs.list()}


class VasodilationIn(BaseModel):
    alpha: float                           # 0 = native PPG, 1 = dicrotic notch fully eroded


@app.get("/aegis/ppg/morphology")
def ppg_morphology() -> Dict[str, Any]:
    """The live PPG-morphology vascular-tone monitor: current waveform window +
    dicrotic-notch / b-a features extracted in real time from a real BIDMC PPG."""
    if ppg_monitor is None:
        return {"available": False}
    return ppg_monitor.state()


@app.post("/aegis/ppg/vasodilation")
async def ppg_vasodilation(payload: VasodilationIn) -> Dict[str, Any]:
    """Dial the vasodilation model and watch the dicrotic notch flatten + the
    vascular-tone metrics collapse in real time (the distributive-shock signature)."""
    if ppg_monitor is None:
        return {"available": False}
    ppg_monitor.set_vasodilation(payload.alpha)
    return ppg_monitor.state()


@app.post("/aegis/kernel/recover")
async def kernel_recover(payload: KernelFaultIn) -> Dict[str, Any]:
    """Recover a quarantined bed: re-verify and re-arm its primary optical driver."""
    bid = _resolve_ebpf_bed(payload.slot)
    if bid is None:
        return {"status": "error", "detail": f"{payload.slot!r} is not an eBPF-driven bed"}
    res = cluster.recover_driver(bid)
    res = {"status": "ok", "bed": bid, "name": cluster.names.get(bid, bid), **(res or {})}
    await _broadcast_cluster(_cluster_payload())
    return res


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
