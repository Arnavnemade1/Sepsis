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
from types import SimpleNamespace
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
from backend.aegis.kernel.hdkernel import (
    run_workload as _run_kworkload, KWORKLOADS as _KWORKLOADS, POLICIES as _KPOLICIES)
from backend.llm.openrouter import OpenRouterClient as _ORClient
from backend.llm.assistant import AegisAssistant as _AegisAssistant
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
# Stream cadence — kept modest to bound WebSocket egress (Render free-tier
# bandwidth). Each tick pushes a full snapshot per open client; clients also
# pause the socket on hidden tabs (see webui/os.js, tactical.js).
TICK_SECONDS = 3.0

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
# On the throttled free-tier CPU a Z3 solve every scheduler tick starves the box;
# verify every ~30th tick instead (deterministic policy, last proof holds between).
import backend.aegis.kernel.scheduler as _sched_mod
_sched_mod.VERIFY_EVERY = 30
cluster_safety = NeuroSymbolicRuntime()
bundle_engine = SepsisBundleEngine()
device_registry = DeviceRegistry()                # simulated sensor fleet (Phase A)
# The write-side trap gate: every treat/escalate/crisis/clear from the console is
# routed through this audited, Z3-gated interface before it touches the kernel.
syscalls = SyscallTable(cluster, bundle_engine, cluster_safety)
# The Autonomous ICU Orchestrator (MEGA-CONCEPT 5): composes the live wing under one
# Non-Maleficence invariant, sharing the SAME Z3 cage + bundle engine as the syscall
# gate so there is a single egress authority. Every governed action is appended to a
# tamper-evident attestation ledger.
from backend.aegis.orchestrator import Orchestrator
orchestrator = Orchestrator(cluster, cage=cluster_safety, bundle=bundle_engine,
                            histories=getattr(cluster, "histories", {}))


def _warm_orchestrator() -> None:
    """Pay the one-time synthesis warmup cost OFF the request path.

    The synthesis algorithm itself is <1s, but the FIRST governed scenario after a
    cold boot pays a large one-time PROCESS-level cost (lazy Z3 native init, the
    proof-export import + first SMT solve, Ed25519 keygen, first-touch memory
    residency on the 512MB free tier) that then caches — measured ~30s→502 on the
    first hit, ~0.15s after. We run one full scenario on a THROWAWAY 2-bed cluster
    in a daemon thread right after boot: it exercises the identical fusion →
    scheduler → BFT-consensus → Z3-cage → proof-export → independent-reverify code
    paths, warming every process-level singleton, WITHOUT injecting a crisis or a
    ledger entry into the live wing. First user click then synthesises fast every
    time (including after each free-tier spin-down)."""
    import time as _t
    try:
        _t.sleep(2.0)                                  # let the rest of boot settle
        from backend.aegis.kernel.scheduler import BioCluster as _BioCluster
        from backend.aegis.orchestrator import Orchestrator as _Orch
        from backend.aegis.z3lock import Z3_LOCK
        t0 = _t.perf_counter()
        warm_cluster = _BioCluster(n_nodes=2, lane_budget=32, seed=1)
        warm_orch = _Orch(warm_cluster, cage=NeuroSymbolicRuntime(),
                          bundle=SepsisBundleEngine())
        # Hold the Z3 lock for the whole warmup: its scenario builds solvers on this
        # daemon thread, which must NEVER run concurrently with the main thread's
        # import-time cluster.run() or the live cluster loop (shared global context).
        with Z3_LOCK:
            warm_cluster.run(4)
            res = warm_orch.run_scenario(warm_cluster.node_ids[0], warmup=0, crisis_ticks=6)
            pf = res.get("proof_certificate")          # warm the re-verify path too
            if pf is not None:
                from backend.aegis.proofexport import verify_certificate as _vc
                _vc(pf)
        print(f"[aegis] orchestrator warmup complete in {_t.perf_counter()-t0:.2f}s "
              f"(first user synthesis will be fast; live wing untouched)")
    except Exception as exc:  # pragma: no cover — warmup must never break boot
        print(f"[aegis] orchestrator warmup skipped: {exc}")


import threading as _threading
_threading.Thread(target=_warm_orchestrator, name="orch-warmup", daemon=True).start()


def _keep_alive() -> None:
    """Free-tier keep-warm: ping our OWN public health URL every ~10 min so the
    Render instance never hits the 15-min idle spin-down (each self-request is
    inbound traffic that resets the timer). Only active when a public URL is known —
    ``RENDER_EXTERNAL_URL`` (set automatically on Render) or a manual ``KEEPALIVE_URL``
    — so it is a no-op in local dev. Best-effort; failures are swallowed."""
    import time as _t
    url = (_os.environ.get("RENDER_EXTERNAL_URL") or _os.environ.get("KEEPALIVE_URL") or "").strip()
    if not url:
        return
    ping = url.rstrip("/") + "/aegis/health"
    try:
        import httpx
    except Exception:  # pragma: no cover
        return
    _t.sleep(90)                                 # let boot + warmup settle first
    print(f"[aegis] keep-alive active → {ping} every 10 min (free-tier anti-sleep)")
    while True:
        try:
            httpx.get(ping, timeout=10.0)
        except Exception:
            pass
        _t.sleep(600)                            # 10 min < the 15-min spin-down window


_threading.Thread(target=_keep_alive, name="keep-alive", daemon=True).start()
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
# L3c live process plane (HDKernel): run a named workload to completion and expose
# its final process table through the one /proc namespace. Seed with "mixed" so
# `proc kthreads` / `proc ksched` return real data before any explicit run.
_hdk_state: Dict[str, Any] = {"last": None}


def _hdk_last() -> Dict[str, Any]:
    """Lazy-seed the process table on first read (keeps the 8192-D HDC encoder +
    workload OUT of the import path so the app starts fast + light for the deploy)."""
    if _hdk_state["last"] is None:
        _hdk_state["last"] = _run_kworkload("mixed", "HDSched")
    return _hdk_state["last"]


procfs.register("/proc/kthreads", lambda: _hdk_last()["ps"])           # the process table
procfs.register("/proc/ksched", lambda: _hdk_last()["top"])            # scheduler/top summary
# The resilient LLM layer (shared, budgeted, multi-model). On-demand only — the
# realtime loop never calls it; operators drive it via `ask`/`brief`. Telemetry
# is read-only through /proc/llm.
_llm_client = _ORClient()
_llm = _AegisAssistant(_llm_client)
procfs.register("/proc/llm", lambda: _llm_client.status())
# The ML control plane: train + specialize real detectors from the OS terminal.
from backend.aegis.mlctl import ModelControlPlane as _ModelControlPlane
mlctl = _ModelControlPlane()
procfs.register("/proc/models", lambda: mlctl.status())
# Per-patient tracking terminals: each bed gets its own shell where the attending
# customises what AegisOS watches (foci + doctor watch-rules), in plain English via
# the shared resilient LLM. Observation-only; the LLM output is sanitised to a fixed
# vocabulary and can never place an order.
from backend.aegis.patient_terminal import PatientTerminal as _PatientTerminal
patient_terminal = _PatientTerminal(llm_client=_llm_client)


def _bed_susceptibility(bed: str) -> float:
    """Comorbidity risk-prior uplift for a bed (0 when no history imported). Injected
    into the cohort so grid tiles + dossier reflect the same personalization the
    per-patient snapshot does."""
    h = patient_terminal.get_history(bed)
    return float(((h or {}).get("susceptibility") or {}).get("logit_uplift", 0.0) or 0.0)


try:
    cluster.susceptibility_provider = _bed_susceptibility
except Exception:
    pass


def _llm_state() -> Dict[str, Any]:
    """Assemble a compact, grounded snapshot for the assistant (read-only)."""
    st: Dict[str, Any] = {}
    try:
        st["topology"] = cluster.sched.topology()
    except Exception:
        pass
    try:
        st["ksched"] = cluster.sched.kstat()
    except Exception:
        pass
    try:
        st["hdkernel"] = _hdk_last()["top"]
    except Exception:
        pass
    return st
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
CLUSTER_TICK_SECONDS = 2.0   # was 0.6 — full cluster payload per tick; throttled to bound egress
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
                # session.tick can run Z3 (protocol synth + consensus on HIGH). Run it
                # OFF the event loop so a request thread holding the Z3 lock can never
                # freeze this loop (and its Z3 serialises cleanly with the threadpool).
                snap = await asyncio.to_thread(session.tick, vitals, source="simulator")
                _apply_treatment_from(snap)
                await _broadcast(snap)
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] drive loop error: {exc}")
        await asyncio.sleep(TICK_SECONDS)


@app.on_event("startup")
async def _startup() -> None:
    # Become ready IMMEDIATELY: start the realtime loops and return so uvicorn
    # signals healthy and the deploy's health check passes right away.
    asyncio.create_task(_drive_loop())
    asyncio.create_task(_cluster_loop())

    # The persistence init makes BLOCKING Supabase calls — never run them on the
    # event loop during startup: a slow/cold DB would stall the startup event,
    # uvicorn would never become ready, and the deploy's health check would time
    # out ("failed to deploy"). Do it on a daemon thread; the app runs fine without it.
    def _init_persistence() -> None:
        global _db
        try:
            from backend.core.database import DatabaseClient
            db = DatabaseClient()
            db.save_patient(DB_PATIENT_ID, session.patient_name, session.age or 0,
                            session.gender or "Unknown", session.core.clinical_history or {})
            _db = db
            session.os.register_source("supabase-persistence")
            print("[aegis] real-data persistence layer ready.")
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] persistence unavailable ({exc}); running without it.")
    _threading.Thread(target=_init_persistence, name="persist-init", daemon=True).start()


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


# --- Temporal Causal Graph (the patient's clinical story) ------------------- #
class ReplayIn(BaseModel):
    outcome_id: Optional[str] = None       # replay at the earliest actionable node upstream
    node_id: Optional[str] = None          # …or replay at a specific branch point
    treatment: Dict[str, float] = {"fluid_rate": 5.0, "vasopressor_dose": 0.0,
                                    "antibiotic_code": 3}
    horizon: Optional[int] = None
    bed: Optional[str] = None              # which patient (live session or cohort bed)


# Per-patient causal graphs: the live session patient + any cohort bed (built on
# demand from its real recorded history, then cached).
_BED_CAUSAL: Dict[str, Any] = {}


def _patient_roster() -> List[Dict[str, Any]]:
    """Unified patient picker list: the live monitored patient + the cohort beds."""
    out = [{"id": session.patient_id, "name": session.patient_name or "Monitored patient",
            "kind": "live", "septic": None,
            "history": session.core.clinical_history or {}}]
    hist = getattr(cluster, "histories", {})
    for r in (cluster.roster() if hasattr(cluster, "roster") else []):
        h = hist.get(r["bed_id"], {})
        out.append({"id": r["bed_id"], "name": r.get("name", r["bed_id"]),
                    "kind": "cohort", "septic": r.get("will_be_septic"),
                    "icu_hour": r.get("icu_hour"), "los": r.get("los_total"),
                    "history": {"chf": bool(h.get("chf")), "renal_failure": bool(h.get("renal_failure")),
                                "hepatic": bool(h.get("hepatic") or h.get("hepatic_failure"))}})
    return out


_organ_trackers: Dict[str, Any] = {}     # bed -> OrganFailureTracker (per-bed organ trajectory)


def _bed_snapshot(bed: str) -> Optional[Dict[str, Any]]:
    """A snapshot-shaped payload for a cohort bed, built from real per-bed state so
    the desktop panels (risk/vitals/twin/agents) reflect the selected patient. Cohort
    beds are replayed records — there is no live waveform; HR/SpO₂ drive the trace."""
    rp = getattr(cluster, "_by", {}).get(bed)
    if rp is None:
        return None
    prof = getattr(cluster, "profiles", {}).get(bed)
    raw = cluster._sample_vitals(bed) if hasattr(cluster, "_sample_vitals") else {}
    hist = getattr(cluster, "histories", {}).get(bed, {})
    mp = float(raw.get("map", 85.0))
    v = {"heart_rate": float(raw.get("hr", 82.0)), "spo2": float(raw.get("spo2", 97.0)),
         "skin_temp": float(raw.get("temp_c", 37.0)),
         "respiratory_rate": float(raw.get("resp_rate", 16.0)), "map": mp,
         "sweat_lactate": float(raw.get("lactate", 1.2)), "hrv": 55.0,
         "systolic_bp": round(mp * 1.4, 1), "diastolic_bp": round(mp * 0.8, 1)}
    fused = float(getattr(prof, "fused_risk", 0.0)) if prof else 0.0
    # Personalisation: an imported/linked chart shifts the alerting PRIOR for a
    # high-risk host (immunosuppression, CKD, …). Population risk is preserved; the
    # band is taken from the personalised risk so the OS alerts these patients sooner.
    from backend.aegis.patient_records import personalize_risk as _personalize
    _hobj = patient_terminal.get_history(bed)
    _susc = (_hobj or {}).get("susceptibility") or {}
    _uplift = float(_susc.get("logit_uplift", 0.0) or 0.0)
    personalized = _personalize(fused, _uplift) if _uplift else round(fused, 4)
    level = "HIGH" if personalized >= 0.6 else "MEDIUM" if personalized >= 0.35 else "LOW"
    wbc = float(hist.get("wbc", 7.5)); gcs = int(hist.get("gcs", 15))
    sirs = (int(v["skin_temp"] > 38 or v["skin_temp"] < 36) + int(v["heart_rate"] > 90)
            + int(v["respiratory_rate"] > 20) + int(wbc > 12 or wbc < 4))
    qsofa = int(v["respiratory_rate"] >= 22) + int(v["systolic_bp"] <= 100) + int(gcs < 15)
    det = {"ml_probability": round(fused, 4), "sepsis_probability": round(fused, 4),
           "organ_dysfunction_score": 0, "hdc_probability": 0.0,
           "confidence": 0.6, "sensor_quality": 1.0}
    from backend.aegis.agents import ReasoningPlane, ReasoningContext
    from backend.aegis.body_state import build_body_state, OrganFailureTracker
    reasoning = ReasoningPlane().deliberate(
        ReasoningContext(v, det, {}, {}, hist)).to_dict()
    _bs = build_body_state(v, det, {}, level, time.time(), clinical_history=hist)
    _organ_trackers.setdefault(bed, OrganFailureTracker()).update(_bs)  # per-bed trajectory
    body = _bs.to_dict()
    body["tracking"] = patient_terminal.profile(bed).to_dict()   # foci + watches (display)
    ing = _causal_for(bed)
    return {
        "schema_version": "1.4", "timestamp": time.time(), "source": "recorded · " + bed,
        "patient": {"id": bed, "name": rp.name, "age": rp.age, "gender": rp.gender,
                    "clinical_history": hist},
        "risk": {"level": level, "ml_probability": round(personalized, 4), "ns_risk_level": level,
                 "population_risk": round(fused, 4), "personalized_risk": round(personalized, 4),
                 "susceptibility": _susc or None,
                 "sirs_score": sirs, "qsofa_score": qsofa, "signal_quality": 1.0},
        "vitals": v, "detection": det, "triggered_rules": [],
        "body_state": body, "reasoning": reasoning, "whatif": None, "consensus": None,
        "history": {"t": [], "channels": {}, "risk": [], "probability": [], "region_severity": {}},
        "protocol": None, "verification": None, "narrative": None, "narrative_source": None,
        "causal_graph": ing.summary() if ing else None,
        "system": session.os.status(),
        "recorded": True,
    }


@app.get("/aegis/patient/snapshot")
def patient_snapshot(bed: Optional[str] = None) -> Dict[str, Any]:
    """Per-patient desktop snapshot: live session, or a cohort bed's recorded state."""
    if not bed or bed == session.patient_id:
        return session.snapshot()
    s = _bed_snapshot(bed)
    return s or {"status": "error", "detail": f"no such patient '{bed}'"}


def _causal_for(bed: Optional[str]):
    """Resolve the causal-graph ingestor for a patient id (session or cohort bed)."""
    if not bed or bed == session.patient_id:
        return session.causal
    if bed in _BED_CAUSAL:
        return _BED_CAUSAL[bed]
    rp = getattr(cluster, "_by", {}).get(bed)
    if rp is None:
        return None
    from backend.aegis.causal_bed import build_bed_causal
    _BED_CAUSAL[bed] = build_bed_causal(rp.record, bed)
    return _BED_CAUSAL[bed]


def _replay_for(bed: Optional[str]):
    """A CounterfactualReplay over the chosen patient's graph, scored by the model."""
    from backend.aegis.counterfactual import CounterfactualReplay
    ing = _causal_for(bed)
    sched = session.os.scheduler
    hist = (session.core.clinical_history if (not bed or bed == session.patient_id)
            else getattr(cluster, "histories", {}).get(bed, {}))
    return CounterfactualReplay(ing.graph, risk_fn=lambda v: sched._risk_of(v, hist))


@app.get("/aegis/patients")
def patients() -> Dict[str, Any]:
    """The patient picker roster (live monitored patient + real cohort beds)."""
    return {"patients": _patient_roster(), "default": session.patient_id}


@app.get("/aegis/causal")
def causal_graph(bed: Optional[str] = None) -> Dict[str, Any]:
    """The tamper-evident causal graph for a patient (live session or any cohort bed,
    built on demand from its real recorded history). Each edge carries its
    evidence_type (z3_proven | model_predicted | observed_association)."""
    ing = _causal_for(bed)
    if ing is None:
        return {"status": "error", "detail": f"no such patient '{bed}'"}
    return {"patient": bed or session.patient_id,
            "summary": ing.summary(), "graph": ing.graph.to_dict()}


@app.get("/aegis/causal/incident/{node_id}")
def causal_incident(node_id: str, bed: Optional[str] = None) -> Dict[str, Any]:
    """Auto-generated incident report: traverse backward from an outcome node to
    the earliest point an intervention could have changed the trajectory."""
    ing = _causal_for(bed)
    if ing is None:
        return {"status": "error", "detail": f"no such patient '{bed}'"}
    try:
        return ing.graph.incident_report(node_id)
    except KeyError:
        return {"status": "error", "detail": f"no such node '{node_id}'"}


@app.get("/aegis/causal/patterns")
def causal_patterns(min_support: int = 5, min_lift: float = 1.2,
                    window_h: float = 4.0) -> Dict[str, Any]:
    """Epidemiological pattern mining across the persisted cohort of causal graphs.
    Returns observed associations with support, Wilson CI, baseline + lift — NOT
    causal guidelines (confounded by indication; disclosed in the payload)."""
    from backend.aegis.causal_mine import PatternMiner
    store = getattr(session.causal, "store", None)
    graphs = []
    if store is not None:
        try:
            graphs = store.load_all()
        except Exception as e:
            print(f"[patterns] cohort load failed: {e}")
    # Always include the live graph (it may not be persisted in this config).
    live = session.causal.graph
    if all(g.bed != live.bed for g in graphs):
        graphs.append(live)
    res = PatternMiner(graphs, outcome_window_h=window_h).mine(
        min_support=min_support, min_lift=min_lift)
    return res


# --- Physiologic Identity: continuous proof-of-patient (MEGA-UPGRADE L5) --------- #
_IDENTITY: Dict[str, Any] = {}


def _identity_service():
    """Lazily enroll the SAME cohort beds shown in the patient selector, keyed by bed
    id, so identity targets line up with the chosen patient (cached). Falls back to a
    broader CinC sample if the live wing isn't the real cohort."""
    if "svc" not in _IDENTITY:
        from backend.aegis.identity import IdentityService
        recs = []
        by = getattr(cluster, "_by", {})
        for bed in getattr(cluster, "node_ids", []):
            rp = by.get(bed)
            if rp is None:
                continue
            rec = rp.record
            recs.append(SimpleNamespace(patient_id=bed, frame=rec.frame,
                                        n_hours=rec.n_hours, is_septic=rec.is_septic,
                                        labels=rec.labels))
        if len(recs) < 4:
            from backend.ml.data.cinc2019 import load_patients
            recs = load_patients(sets=["training_setA"], limit=120)
        _IDENTITY["svc"] = IdentityService(recs)
    return _IDENTITY["svc"]


@app.get("/aegis/identity")
def identity_status() -> Dict[str, Any]:
    """Enrolled-cohort status + the measured same-vs-different separability."""
    svc = _identity_service()
    return {"status": svc.status(), "validation": svc.validation()}


class IdentitySwapIn(BaseModel):
    target: Optional[str] = None         # whose monitor
    source: Optional[str] = None         # whose stream is (mis)connected


@app.post("/aegis/identity/swap")
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


@app.post("/aegis/identity/self_check")
def identity_self_check(p: IdentitySwapIn) -> Dict[str, Any]:
    """Verify a patient against their own later stream (the no-fault baseline case)."""
    svc = _identity_service()
    pids = svc.patients()
    target = p.target or (pids[0] if pids else None)
    if target not in pids:
        return {"status": "error", "detail": "unknown patient id"}
    return svc.self_check(target)


# --- Living Pharmacopoeia: drug-interaction graph → dynamic Z3 (MEGA-UPGRADE L4) -- #
from backend.aegis.pharmacopoeia import Pharmacopoeia
pharmacopoeia = Pharmacopoeia()                   # loads the fused DDInter+curated KB


class PharmaScreenIn(BaseModel):
    drugs: List[str] = ["warfarin", "fluconazole", "ciprofloxacin", "ondansetron"]
    context: Dict[str, Any] = {"renal_failure": False}


@app.get("/aegis/pharmacopoeia")
def pharmacopoeia_stats() -> Dict[str, Any]:
    """The fused knowledge base: DDInter pairwise graph + curated CYP/QT/renal layer."""
    return pharmacopoeia.stats()


@app.post("/aegis/pharmacopoeia/screen")
def pharmacopoeia_screen(p: PharmaScreenIn) -> Dict[str, Any]:
    """Traverse the interaction graph for a regimen, compose a Z3 constraint per
    discovered hazard, and return the proof tree (globally safe or the violations)."""
    return pharmacopoeia.verify(p.drugs, p.context)


# --- Clinical Safety DSL: hospital-authored rules compiled to Z3 (MEGA-UPGRADE L2) - #
from backend.aegis.clinical_dsl import (compile_rules, verify_protocol as dsl_verify,
                                        check_consistency, DEFAULT_POLICY)


class DslVerifyIn(BaseModel):
    policy: str = DEFAULT_POLICY
    protocol: Dict[str, Any] = {"antibiotic_code": 1, "antibiotic_time_hr": 0.5,
                                "fluid_rate": 4.0, "vasopressor_dose": 0.1}
    context: Dict[str, Any] = {"risk_level": "HIGH", "chf": True,
                               "penicillin_allergy": True, "renal_failure": False}


@app.get("/aegis/dsl/default")
def dsl_default() -> Dict[str, Any]:
    """The example hospital policy + a sample protocol/context to verify against."""
    return {"policy": DEFAULT_POLICY,
            "example_protocol": {"antibiotic_code": 1, "antibiotic_time_hr": 0.5,
                                 "fluid_rate": 4.0, "vasopressor_dose": 0.1},
            "example_context": {"risk_level": "HIGH", "chf": True,
                                "penicillin_allergy": True, "renal_failure": False}}


@app.post("/aegis/dsl/verify")
def dsl_verify_endpoint(p: DslVerifyIn) -> Dict[str, Any]:
    """Compile a hospital's safety rules to Z3 and prove a protocol satisfies them.
    Returns a per-rule proof certificate (with the compiled Z3 s-expression) plus a
    consistency check (is the rule set jointly satisfiable for this patient?)."""
    try:
        rules = compile_rules(p.policy)
    except SyntaxError as e:
        return {"status": "error", "detail": f"parse/compile error: {e}"}
    cert = dsl_verify(rules, p.protocol, p.context)
    consistency = check_consistency(rules, p.context)
    return {"status": "ok", "certificate": cert.to_dict(), "consistency": consistency,
            "n_rules": len(rules)}


# --- Clinical RBAC: privilege-ring access control over the OS (MEGA-UPGRADE L3) -- #
from backend.aegis.rbac import (RBAC, RBACGate, GRANTS, RING, CAP_MIN_RING,
                                prove_policy_sound)
rbac = RBAC()
rbac_gate = RBACGate(rbac, syscalls.invoke)      # gates the live write-syscall trap


class RbacLoginIn(BaseModel):
    user: str = "clinician"
    role: str


class RbacAuthIn(BaseModel):
    token: str
    capability: str


@app.post("/aegis/rbac/login")
def rbac_login(p: RbacLoginIn) -> Dict[str, Any]:
    """Mint a signed, expiring session token for a role + return that role's shell
    capability contract (what the role's UI may render and do)."""
    try:
        s = rbac.login(p.user, p.role)
    except ValueError as e:
        return {"status": "error", "detail": str(e)}
    return {"session": s.to_dict(), "token": s.token, "shell": rbac.shell_capabilities(p.role)}


@app.get("/aegis/rbac/policy")
def rbac_policy() -> Dict[str, Any]:
    """The full RBAC matrix + the Z3 proof that the policy is sound (no grant above
    its capability's ring)."""
    return {"rings": RING, "cap_min_ring": CAP_MIN_RING,
            "grants": {r: sorted(c) for r, c in GRANTS.items()},
            "soundness": prove_policy_sound()}


@app.post("/aegis/rbac/authorize")
def rbac_authorize(p: RbacAuthIn) -> Dict[str, Any]:
    """Check whether a session may perform a capability (audited, no mutation)."""
    ok, reason, s = rbac.authorize(p.token, p.capability)
    return {"ok": ok, "reason": reason, "capability": p.capability,
            "role": s.role if s else None, "audit_head": rbac.audit.head()[:12]}


@app.get("/aegis/rbac/audit")
def rbac_audit(limit: int = 24) -> Dict[str, Any]:
    return {"entries": [e.to_dict() for e in rbac.audit.entries[-limit:]],
            "count": len(rbac.audit.entries), "head": rbac.audit.head()[:12],
            "integrity_ok": rbac.audit.verify()}


_FED_CACHE: Dict[str, Any] = {}


@app.get("/aegis/fedlearn")
def fedlearn_result(limit: int = 400, refresh: bool = False) -> Dict[str, Any]:
    """Federated-learning experiment on the real CinC two-hospital split: does
    cross-site AUROC recover without sharing data, and does robust aggregation
    survive a poisoned hospital? Cached (first call computes ~5-15s)."""
    key = f"l{limit}"
    if refresh or key not in _FED_CACHE:
        from backend.aegis.fedlearn import run_experiment, run_poisoning
        _FED_CACHE[key] = {
            "experiment": run_experiment(limit=limit, n_nodes=4, rounds=10, epochs=50),
            "poisoning": run_poisoning(limit=limit, n_nodes=5, rounds=10, epochs=50),
            "caveats": [
                "Federated model is logistic regression (averageable weights); the "
                "production GBM is not FL-compatible.",
                "Only signed weight-deltas cross the wire — no patient records. "
                "Data-minimization, not differential privacy.",
                "Byzantine tolerance = signed updates + robust aggregation "
                "(Krum/median), not a proof that a gradient is benign.",
            ],
        }
    return _FED_CACHE[key]


@app.get("/aegis/orchestrator")
def orchestrator_status() -> Dict[str, Any]:
    """The orchestration hierarchy (sensor/bed/wing/global) under the one
    Non-Maleficence invariant, plus the attestation-ledger head + integrity."""
    return orchestrator.status()


@app.get("/aegis/orchestrator/attest")
def orchestrator_attest() -> Dict[str, Any]:
    """Continuous attestation: prove the invariant currently holds across the wing
    (ledger intact + no critical bed dropped below its priority floor)."""
    return orchestrator.attest()


class GovernIn(BaseModel):
    layer: str = "bed"
    action: str                                    # treat | escalate | pin | actuator
    payload: Dict[str, Any] = {}
    target: Optional[str] = None                   # bed id


@app.post("/aegis/orchestrator/govern")
def orchestrator_govern(payload: GovernIn) -> Dict[str, Any]:
    """Route a single action through the non-maleficence invariant. Returns its
    certificate (allowed → applied; blocked → nothing mutates). Every call is
    appended to the attestation ledger."""
    cert = orchestrator.govern(payload.layer, payload.action,
                               {k: v for k, v in payload.payload.items()},
                               target=payload.target)
    return cert.to_dict()


@app.post("/aegis/orchestrator/scenario")
def orchestrator_scenario(bed: Optional[str] = None) -> Dict[str, Any]:
    """The demo cascade: induce a sepsis crisis in a wing bed and walk the governed
    response sensor→bed→wing→global→governance, ending in a signed, Z3-certified,
    immutably-logged order. A deliberate drill on the live wing (clear via syscalls)."""
    beds = getattr(cluster, "node_ids", [])
    if not beds:
        return {"status": "error", "detail": "no wing beds available"}
    if bed is None:
        # prefer a known-septic bed, else the first.
        bed = next((b for b in beds if getattr(cluster, "_by", {}).get(b)
                    and getattr(cluster._by[b], "is_septic", False)), beds[0])
    if bed not in beds:
        return {"status": "error", "detail": f"no such bed '{bed}'"}
    # The live wing is already warm (the stream loop ticks it); keep the drill short.
    # Fewer ticks: enough to drive the crisis + trip PREEMPT, but light on a
    # throttled CPU so the synthesis returns well inside the gateway timeout.
    return orchestrator.run_scenario(bed, warmup=0, crisis_ticks=12)


# --- Portable, third-party-checkable proof certificates (proof-export layer) ----- #
from backend.aegis.proofexport import verify_certificate as _verify_cert, export_bundle

_PROOF_CA = {"ca": None, "tried": False}


def _proof_ca():
    """Lazy shared Ed25519 signing authority for server-issued certificates.
    Returns None (→ unsigned but still math-checkable cert) if crypto is unavailable."""
    if not _PROOF_CA["tried"]:
        _PROOF_CA["tried"] = True
        try:
            from backend.aegis.proofexport import CertificateAuthority
            _PROOF_CA["ca"] = CertificateAuthority()
        except Exception as exc:
            print(f"[aegis] server proof signing disabled ({exc})")
    return _PROOF_CA["ca"]


class ProofVerifyIn(BaseModel):
    certificate: Dict[str, Any]
    require_signature: bool = True


@app.get("/aegis/orchestrator/proof")
def orchestrator_proof_latest() -> Dict[str, Any]:
    """The most recent portable proof certificate the orchestrator emitted for a
    treatment order, plus a fresh independent re-check of it."""
    pc = orchestrator.last_proof
    if pc is None:
        return {"status": "empty",
                "detail": "no treatment order has been governed yet (run a scenario)"}
    return {"status": "ok", "certificate": pc,
            "audit": _verify_cert(pc).to_dict()}


@app.get("/aegis/orchestrator/proof/{seq}")
def orchestrator_proof_by_seq(seq: int) -> Dict[str, Any]:
    """The portable proof certificate bound to attestation-ledger entry ``seq``."""
    pc = orchestrator.proof_for(seq)
    if pc is None:
        return {"status": "not_found", "detail": f"no proof bound to ledger seq {seq}"}
    return {"status": "ok", "certificate": pc, "audit": _verify_cert(pc).to_dict()}


class CoreProofIn(BaseModel):
    clinical_history: Dict[str, Any] = {"chf": True, "renal_failure": False,
                                        "penicillin_allergy": True}
    vitals: Dict[str, Any] = {"map": 62.0}
    risk_level: str = "HIGH"
    protocol: Dict[str, Any] = {"fluid_rate": 4.0, "vasopressor_dose": 0.2,
                                "antibiotic_code": 3}


@app.post("/aegis/proof/core")
def proof_core_certify(p: CoreProofIn) -> Dict[str, Any]:
    """Export a portable, signed proof certificate from the CORE Z3 cage — the exact
    prover the live runtime uses on every HIGH-risk decision — one re-checkable
    SMT-LIB2 obligation per clinical invariant, then independently re-audit it."""
    from backend.aegis.proofexport import certify_core_protocol
    cert = certify_core_protocol(p.clinical_history, p.vitals, p.risk_level,
                                 p.protocol, ca=_proof_ca())
    cd = cert.to_dict()
    return {"status": "ok", "certificate": cd, "audit": _verify_cert(cd).to_dict()}


@app.get("/aegis/proof/core/latest")
def proof_core_latest() -> Dict[str, Any]:
    """The portable proof certificate for the live patient's most recent HIGH-risk
    protocol decision, plus a fresh independent re-check. Emitted by AegisCore."""
    pc = getattr(session.core, "last_proof", None)
    if pc is None:
        return {"status": "empty",
                "detail": "no HIGH-risk protocol has been certified for this patient yet"}
    return {"status": "ok", "certificate": pc, "audit": _verify_cert(pc).to_dict()}


@app.get("/aegis/proof/core/latest/bundle")
def proof_core_latest_bundle():
    """Download a self-contained .zip audit bundle for the live patient's most recent
    certified order — verifiable by a third party with only z3 + cryptography."""
    import io
    import shutil
    import tempfile
    import zipfile
    from fastapi import Response
    from backend.aegis.proofexport import ProofCertificate

    pc = getattr(session.core, "last_proof", None)
    if pc is None:
        return {"status": "empty", "detail": "no certified order yet"}
    tmp = tempfile.mkdtemp(prefix="aegis-core-bundle-")
    try:
        export_bundle(ProofCertificate.from_dict(pc), tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in _os.walk(tmp):
                for fn in files:
                    full = _os.path.join(root, fn)
                    zf.write(full, _os.path.relpath(full, tmp))
        buf.seek(0)
        return Response(
            content=buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition":
                     'attachment; filename="aegis-core-proof-bundle.zip"'})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/aegis/proof/verify")
def proof_verify(p: ProofVerifyIn) -> Dict[str, Any]:
    """Independently audit ANY proof certificate the caller supplies — re-runs every
    SMT-LIB2 obligation in a fresh solver, recomputes the content digest, and checks
    the Ed25519 signature. This is the third-party audit path: a signed-but-false
    certificate fails here because the math is re-run, not trusted."""
    return _verify_cert(p.certificate, require_signature=p.require_signature).to_dict()


@app.get("/aegis/orchestrator/proof/{seq}/bundle")
def orchestrator_proof_bundle(seq: int):
    """Download a self-contained audit bundle (.zip): certificate.json, one .smt2 per
    obligation, the public key, AUDIT.md, and a standalone verifier that needs only
    z3 + cryptography. A third party verifies the order without any AegisOS code."""
    import io
    import shutil
    import tempfile
    import zipfile
    from fastapi import Response
    from backend.aegis.proofexport import ProofCertificate

    pc = orchestrator.proof_for(seq)
    if pc is None:
        return {"status": "not_found", "detail": f"no proof bound to ledger seq {seq}"}
    tmp = tempfile.mkdtemp(prefix=f"aegis-bundle-{seq}-")
    try:
        export_bundle(ProofCertificate.from_dict(pc), tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in _os.walk(tmp):
                for fn in files:
                    full = _os.path.join(root, fn)
                    zf.write(full, _os.path.relpath(full, tmp))
        buf.seek(0)
        return Response(
            content=buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition":
                     f'attachment; filename="aegis-proof-bundle-seq{seq}.zip"'})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/aegis/consensus")
def consensus_round(deterioration: Optional[float] = None,
                    faulty: Optional[str] = None,
                    remote: Optional[str] = None,
                    bed: Optional[str] = None) -> Dict[str, Any]:
    """Run one Byzantine-fault-tolerant specialist consensus round over a patient's
    state (live session, or a cohort bed via ``bed``). ``faulty`` marks a replica id
    as Byzantine; ``remote`` marks a remote consultant; leave ``deterioration``
    unset to derive it from the patient's risk band."""
    from backend.aegis.consensus import ClinicalBFT, default_panel
    if bed and bed != session.patient_id and getattr(cluster, "_by", {}).get(bed):
        prof = getattr(cluster, "profiles", {}).get(bed)
        raw = cluster._sample_vitals(bed) if hasattr(cluster, "_sample_vitals") else {}
        fused = float(getattr(prof, "fused_risk", 0.0)) if prof else 0.0
        vit = {"map": float(raw.get("map", 80.0))}
        risk = "HIGH" if fused >= 0.6 else "MEDIUM" if fused >= 0.35 else "LOW"
        hist = getattr(cluster, "histories", {}).get(bed, {})
    else:
        vit = session.core.twin.latest_vitals or {}
        risk = session.latest.risk_level if session.latest else "MEDIUM"
        hist = session.core.clinical_history or {}
    if deterioration is None:
        deterioration = {"HIGH": 0.7, "MEDIUM": 0.35, "LOW": 0.1}.get(risk, 0.3)
        if float(vit.get("map", 90)) < 60:        # very low MAP ⇒ crashing faster
            deterioration = max(deterioration, 0.85)
    state = {"map": float(vit.get("map", 75.0)), "risk_level": risk,
             "clinical_history": hist,
             "infection_suspected": risk in ("HIGH", "MEDIUM")}
    panel = default_panel(remote_id=remote)
    if faulty:
        for a in panel:
            if a.agent_id == faulty:
                a.byzantine = True
    bft = ClinicalBFT(panel)
    out = bft.run_round(state, deterioration=float(deterioration))
    res = out.to_dict()
    res["state"] = {"map": state["map"], "risk_level": risk,
                    "clinical_history": state["clinical_history"]}
    res["panel"] = [{"id": a.agent_id, "specialty": a.specialty,
                     "byzantine": a.byzantine, "remote": a.remote} for a in panel]
    return res


@app.post("/aegis/causal/replay")
def causal_replay(payload: ReplayIn) -> Dict[str, Any]:
    """Counterfactual replay: 'what would have happened under an alternative action?'
    Treatment effect is measured model-internally (alternative vs action-taken),
    scored by the live model; recorded reality is returned for context."""
    eng = _replay_for(payload.bed)
    if eng is None:
        return {"status": "error", "detail": f"no such patient '{payload.bed}'"}
    try:
        if payload.outcome_id:
            return eng.replay_earliest(payload.outcome_id, payload.treatment,
                                       horizon=payload.horizon)
        if payload.node_id:
            return eng.replay(payload.node_id, payload.treatment, horizon=payload.horizon)
        return {"status": "error", "detail": "provide outcome_id or node_id"}
    except (KeyError, ValueError) as e:
        return {"status": "error", "detail": str(e)}


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


class HDKernelRunIn(BaseModel):
    workload: str = "mixed"                 # mixed | pipeline | fanout | stress
    policy: str = "HDSched"                 # FIFO | RR | CFS | EDF | HDSched


@app.get("/aegis/hdkernel")
def hdkernel_state() -> Dict[str, Any]:
    """The live process plane (Layer 3c): last run's summary/ps/top + the menu."""
    return {**_hdk_last(),
            "workloads": list(_KWORKLOADS), "policies": list(_KPOLICIES)}


class MLTrainIn(BaseModel):
    name: str
    feature_set: str = "wearable"          # wearable | lab
    horizon: Optional[int] = None          # early-warning hours before onset (specialise)
    limit: int = 1000
    folds: int = 3


class MLUseIn(BaseModel):
    name: str


class TrackIn(BaseModel):
    command: str


def _patient_ctx(bed: str) -> Dict[str, Any]:
    """A compact, read-only snapshot for grounding the tracking LLM ('her kidneys')."""
    try:
        snap = _bed_snapshot(bed) or {}
        bs = snap.get("body_state", {})
        return {"bed": bed, "risk": snap.get("risk", {}).get("level"),
                "vitals": snap.get("vitals", {}),
                "organs": [{"name": o.get("name"), "status": o.get("status")}
                           for o in bs.get("organ_systems", []) if o.get("monitored")]}
    except Exception:
        return {"bed": bed}


@app.get("/aegis/patient/{bed}/tracking")
def patient_tracking(bed: str) -> Dict[str, Any]:
    """This patient's tracking profile (foci + watch-rules) + any watch-rules that
    fired against the current vitals (edge-triggered here, the single evaluator) +
    the command vocabulary. The dossier polls this."""
    from backend.aegis.patient_terminal import METRICS as _M, ORGANS as _O
    snap = _bed_snapshot(bed) or {}
    fired = patient_terminal.evaluate(bed, snap.get("vitals", {}))
    return {"profile": patient_terminal.profile(bed).to_dict(), "fired": fired,
            "vocab": {"metrics": {k: v["label"] for k, v in _M.items()}, "organs": _O}}


@app.post("/aegis/patient/{bed}/track")
def patient_track(bed: str, payload: TrackIn) -> Dict[str, Any]:
    """Run one per-patient tracking command (structured verb OR natural language).
    Observation-only: the result can add a focus or a watch, never an order."""
    return patient_terminal.exec(bed, payload.command, context=_patient_ctx(bed))


# --- Medical-history intake + records sources (specialise the monitor) ------ #
class HistoryIn(BaseModel):
    text: str = ""
    source: str = "manual"


class LinkIn(BaseModel):
    source: str = "mock_hospital"
    patient_ref: Optional[str] = None        # source-specific id (e.g. FHIR patient id / MRN)


class SynthIn(BaseModel):
    text: str = ""


def _roster_entry(bed: str) -> Dict[str, Any]:
    for p in _patient_roster():
        if p["id"] == bed:
            return p
    return {"id": bed, "septic": None}


@app.get("/aegis/records/sources")
def records_sources() -> Dict[str, Any]:
    """Registered records-source connectors (manual paste/upload + the MCP-shaped
    demo hospital stub). Each carries a `live` flag so the UI is honest about stubs."""
    return {"sources": patient_terminal.sources()}


@app.get("/aegis/patient/{bed}/history")
def patient_history_get(bed: str) -> Dict[str, Any]:
    """The imported medical history for this patient (if any) + what it specialised."""
    return {"history": patient_terminal.get_history(bed),
            "sources": patient_terminal.sources()}


@app.post("/aegis/patient/{bed}/history")
def patient_history_import(bed: str, payload: HistoryIn) -> Dict[str, Any]:
    """Import a pasted/uploaded medical history → parse → seed foci + watch-rules so
    the monitor specialises toward this patient. Observation-only."""
    if not (payload.text or "").strip():
        return {"ok": False, "reply": "empty history"}
    return patient_terminal.import_history(bed, payload.text, source=payload.source or "manual")


@app.post("/aegis/patient/{bed}/history/link")
def patient_history_link(bed: str, payload: LinkIn) -> Dict[str, Any]:
    """Pull a history from a registered records source (stub / live FHIR EHR) and
    import it. ``patient_ref`` is the source's patient identifier (FHIR id / MRN)."""
    return patient_terminal.link_source(bed, payload.source, patient=_roster_entry(bed),
                                        ref=payload.patient_ref)


@app.get("/aegis/patient/{bed}/synthesis")
def patient_synthesis_get(bed: str) -> Dict[str, Any]:
    """The clinician synthesis / instruction log for this patient (annotations shown
    in the clinical story)."""
    return {"annotations": patient_terminal.get_annotations(bed)}


@app.post("/aegis/patient/{bed}/synthesis")
def patient_synthesis_add(bed: str, payload: SynthIn) -> Dict[str, Any]:
    """Record a doctor's detailed synthesis/instruction. Stored as a timestamped
    annotation; monitoring intent (safe vocab) is applied; treatment intent is
    flagged and redirected to the certified order gate — never auto-executed."""
    return patient_terminal.add_synthesis(bed, payload.text, context=_patient_ctx(bed))


@app.get("/aegis/ml")
def ml_status() -> Dict[str, Any]:
    """The ML control plane: registered models + running training jobs."""
    return mlctl.status()


@app.post("/aegis/ml/train")
def ml_train(payload: MLTrainIn) -> Dict[str, Any]:
    """Launch a REAL training/specialisation job (patient-grouped CV on CinC-2019).
    Returns the job id immediately; poll /aegis/ml/job/{id} for progress + metrics."""
    job = mlctl.train(payload.name, feature_set=payload.feature_set,
                      horizon=payload.horizon, limit=payload.limit, folds=payload.folds)
    return {"launched": True, "job": job.to_dict()}


@app.get("/aegis/ml/job/{job_id}")
def ml_job(job_id: str) -> Dict[str, Any]:
    j = mlctl.job(job_id)
    return j if j is not None else {"error": "no such job", "id": job_id}


@app.post("/aegis/ml/use")
def ml_use(payload: MLUseIn) -> Dict[str, Any]:
    return mlctl.use(payload.name)


@app.post("/aegis/hdkernel/run")
def hdkernel_run(payload: HDKernelRunIn) -> Dict[str, Any]:
    """Run a named workload under a chosen scheduling policy on the live kernel and
    return its process table — a genuine 'launch a program on the OS and inspect
    it'. Swapping ``policy`` visibly changes turnaround / context switches."""
    wl = payload.workload if payload.workload in _KWORKLOADS else "mixed"
    pol = payload.policy if payload.policy in _KPOLICIES else "HDSched"
    _hdk_state["last"] = _run_kworkload(wl, pol)
    return {**_hdk_state["last"],
            "workloads": list(_KWORKLOADS), "policies": list(_KPOLICIES)}


class LLMAskIn(BaseModel):
    question: str


@app.get("/aegis/llm")
def llm_status() -> Dict[str, Any]:
    """Telemetry for the resilient LLM layer: chain, budget, cooldowns, cache."""
    return _llm_client.status()


@app.post("/aegis/llm/ask")
def llm_ask(payload: LLMAskIn) -> Dict[str, Any]:
    """Grounded natural-language Q&A over the LIVE system state (cohort + kernel).
    Read-only: the assistant answers only from the current state and refuses to
    issue orders — actions go through the certified syscall gate, not prose."""
    return {**_llm.ask(payload.question, _llm_state()),
            "budget_remaining": _llm_client.budget_remaining()}


@app.post("/aegis/llm/brief")
def llm_brief() -> Dict[str, Any]:
    """A shift-handoff briefing of the current cohort (grounded, budgeted)."""
    return {**_llm.brief(_llm_state()),
            "budget_remaining": _llm_client.budget_remaining()}


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


class _SmartStatic(StaticFiles):
    """Cache the heavy immutable vendor assets (GLBs, three.js — ~35 MB, the
    bulk of egress) for a year, but keep the app source revalidating so live
    edits load fresh. This is what stops Render re-serving 35 MB per reload.

    Note: in the split deploy the static frontend lives on Cloudflare Pages
    (unlimited bandwidth) and this mount is only a local-dev / fallback path —
    see webui/_headers for the Cloudflare-side cache policy."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        p = path.replace("\\", "/")
        if p.startswith("vendor/"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            resp.headers["Cache-Control"] = "no-cache"   # revalidate (cheap 304s), always fresh source
        return resp


if _os.path.isdir(_WEBUI):
    app.mount("/", _SmartStatic(directory=_WEBUI, html=True), name="aegisos-shell")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("backend.aegis.server:app", host="127.0.0.1", port=8100, reload=False)
