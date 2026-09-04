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
import json
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
# Resolve server-side credentials BEFORE the LLM layer is constructed: the
# clients below read os.getenv at import time, so a key recovered from the
# Supabase runtime_config fallback has to be in the environment by now or it
# arrives too late to matter. The env var still wins when it is set.
from backend.core import runtime_config as _rtcfg
_rtcfg.prime("GEMINI_API_KEY")
_rtcfg.prime("OPENROUTER_API_KEY")

from backend.llm.provider import LLMProvider as _LLMProvider
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


def _warm_pk() -> None:
    """Same cold-start problem, different subsystem: the first treatment
    projection pays for importing SciPy, the first ``expm`` call reaching BLAS,
    and building the reference projections that every adverse-event probability is
    anchored to. That is seconds on the free tier and then microseconds forever.
    Doing it here means the first clinician to open the treatment twin does not
    pay for it. Uses only the reference patient — the live wing is untouched."""
    import time as _t
    try:
        _t.sleep(3.0)                                  # after the orchestrator warmup
        t0 = _t.perf_counter()
        from backend.aegis.adverse import (AdverseEventEngine, REFERENCE_PATIENT,
                                           reference_projection)
        from backend.aegis.pharmacokinetics import DRUGS, project_protocol
        for key in DRUGS:
            reference_projection(key)
        reference_projection("crystalloid")
        warm = project_protocol({"fluid_rate": 10.0, "vasopressor_dose": 0.10,
                                 "antibiotic_code": 1}, REFERENCE_PATIENT)
        AdverseEventEngine().project(warm["arms"], REFERENCE_PATIENT, {})
        print(f"[aegis] treatment-projection warmup complete in "
              f"{_t.perf_counter()-t0:.2f}s ({len(DRUGS)+1} reference projections "
              f"cached; live wing untouched)")
    except Exception as exc:  # pragma: no cover — warmup must never break boot
        print(f"[aegis] treatment-projection warmup skipped: {exc}")


import threading as _threading
_threading.Thread(target=_warm_orchestrator, name="orch-warmup", daemon=True).start()
_threading.Thread(target=_warm_pk, name="pk-warmup", daemon=True).start()


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
from backend.aegis.z3lock import PROVER as _PROVER, ProverUnavailable
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
# The resilient LLM layer (shared, budgeted, multi-model, multi-VENDOR: Gemini
# primary → Nemotron backup). On-demand only — the realtime loop never calls it;
# operators drive it via `ask`/`brief` and the consult call. Telemetry is
# read-only through /proc/llm.
_llm_client = _LLMProvider()
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
    # Explicit source contract for AERIS. Omitted means no additional source is
    # assumed available; the advisory router returns WAIT rather than inventing one.
    evidence_capabilities: Optional[Dict[str, bool]] = None


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
        except ProverUnavailable as exc:
            # The safety prover is known-corrupted. Ticking on would keep asking a
            # broken prover to certify treatments, which is the one thing this loop
            # must not do — so it stops driving and leaves the wing read-only. The
            # escalation itself (fail-closed, /aegis/health, process exit) belongs
            # to z3lock; here we just stop generating work for a prover we do not
            # trust. Restarting the process is the only way back.
            print(f"[aegis] drive loop HALTED — {exc}", flush=True)
            return
        except Exception as exc:  # pragma: no cover
            # An ordinary transient failure: log and keep ticking. Z3 CONTEXT
            # corruption is not this — it arrives as ProverUnavailable above, after
            # z3lock has already escalated it.
            print(f"[aegis] drive loop error: {exc}")
        await asyncio.sleep(TICK_SECONDS)


@app.on_event("startup")
async def _startup() -> None:
    # Become ready IMMEDIATELY: start the realtime loops and return so uvicorn
    # signals healthy and the deploy's health check passes right away.
    global _MAIN_LOOP
    # The agent plane runs from threadpool workers (sync REST) as well as from
    # the consult socket. A cockpit broadcast has to be scheduled back onto THIS
    # loop — the one that owns the client WebSockets — or we would be writing to
    # a socket from a foreign loop.
    _MAIN_LOOP = asyncio.get_event_loop()
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
    # The prover's state is part of health, not a log line. If the shared Z3
    # context is corrupted the OS can still serve reads, but it can no longer
    # certify anything — and a monitor that only ever sees "ok" would never learn
    # that the safety gate has stopped issuing verdicts.
    prover = _PROVER.status()
    return {"status": "ok" if prover["healthy"] else "degraded",
            "source": session.source,
            "mode": "live" if _live_mode() else "demo",
            "persistence": bool(_db is not None and getattr(_db, "use_supabase", False)),
            "model_loaded": session.core.model is not None,
            "prover": {"healthy": prover["healthy"],
                       "fail_closed": prover["fail_closed"],
                       "failures": prover["failures"],
                       "last_error": prover["last_error"]},
            "schema_version": session.snapshot()["schema_version"]}


@app.get("/aegis/health/system")
def health_system() -> Dict[str, Any]:
    """How well the detector actually performs, and where it falls short.

    Reads the cross-hospital evaluation artefact rather than measuring anything
    live: the demo cohort is twelve stays, which cannot support an AUROC. Strengths
    and gaps are derived from those numbers, so the page cannot drift into being a
    brochure."""
    from backend.aegis.system_health import report
    out = report()
    # The live half: subsystem state the artefact cannot know about.
    prover = _PROVER.status()
    out["subsystems"] = [
        {"name": "Detector", "ok": session.core.model is not None,
         "note": ("stacked ensemble loaded" if session.core.model is not None
                  else "no model artefact — running on rules only")},
        {"name": "Safety prover", "ok": prover["healthy"],
         "note": ("Z3 context healthy" if prover["healthy"]
                  else "CORRUPTED — failing closed, nothing is being certified")},
        {"name": "Data source", "ok": True,
         "note": f"{session.source} · {'live' if _live_mode() else 'demo'}"},
        {"name": "Persistence", "ok": bool(_db is not None and getattr(_db, "use_supabase", False)),
         "note": ("Supabase connected" if (_db is not None and getattr(_db, "use_supabase", False))
                  else "local only — nothing is being retained")},
    ]
    try:
        out["cohort"] = cluster.validation(alarm=0.70) if hasattr(cluster, "validation") else None
    except Exception:
        out["cohort"] = None
    return out


def _read_json(path: str):
    """Read a small artifact, or None. A missing artifact is a state, not an error."""
    import json as _j, os as _o
    if not _o.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return _j.load(fh)
    except Exception:
        return None


@app.get("/aegis/lab")
def lab_state() -> Dict[str, Any]:
    """The Lab: agents propose, a locked gate adjudicates, refusals are the record.

    Read-only. Binding a real adjudicator fits models on held-out patients and takes
    minutes, so it is a job (`make lab-run`), not a request handler — a cheap gate is
    one somebody runs a thousand times until something passes.
    """
    from backend.aegis import lab as _lab
    from backend.ml import config as _cfg
    import os as _os, json as _json
    path = _os.path.join(_cfg.ARTIFACTS_DIR, "lab_ledger.json")
    ledger = None
    if _os.path.exists(path):
        try:
            with open(path) as fh:
                ledger = _json.load(fh)
            # Sanitise on the way OUT too. An older ledger on disk may predate the
            # json_safe fix, and FastAPI's encoder would happily re-emit its NaN tokens
            # to a browser that cannot parse them.
            ledger = _lab.Lab.json_safe(ledger)
        except Exception:
            ledger = None
    # The PROGRAMME, not just the last campaign. A ledger that is overwritten every run
    # cannot show whether a finding came from one campaign or from the tenth attempt, and
    # that is the number a reader most needs. The registry carries every campaign ever
    # run, hash-chained, and reports cumulative alpha as the expected false-finding count.
    programme, campaigns, chain, record = None, [], None, []
    try:
        from backend.aegis.lab_registry import LabRegistry as _Reg
        reg = _Reg.load()
        programme = reg.programme()
        campaigns = reg.campaigns[-12:]
        # The permanent record, newest first. The console showed only the last
        # campaign's ledger, which is the one thing a research record must not do —
        # a reader cannot tell a finding from the tenth attempt at one.
        record = [{"campaign": e.campaign, "agent": e.agent, "kind": e.kind,
                   "title": e.title, "status": e.status, "accepted": e.accepted,
                   "observed_delta": e.observed_delta, "mde": e.mde,
                   "alpha_spent": e.alpha_spent, "protocol": e.protocol,
                   "change_key": e.change_key}
                  for e in reversed(reg.entries[-80:])]
        ok, bad = reg.verify_chain()
        chain = {"ok": ok, "broken_at": bad, "n_entries": len(reg.entries),
                 "note": ("every recorded verdict re-hashes to its stored value"
                          if ok else
                          f"BROKEN at seq {bad} — a verdict has been edited or removed "
                          f"since it was recorded")}
    except Exception:
        pass

    return {
        "proposers": [{"name": p.name, "prior": p.prior} for p in _lab.default_proposers()],
        "roles": {
            "Proposer": "generates a change from a stated prior; never judges it",
            "Critic": "argues against it from this project's recorded nulls and the "
                      "Lab's own refusals; cannot veto",
            "SafetyOfficer": "absolute veto over anything that weakens a safety property",
            "Gate": "held-out patients, paired bootstrap; the only thing that can accept",
            "Replication": "a SECOND pool, disjoint from the gate by construction; the "
                           "only thing that can turn a candidate into a finding",
        },
        "programme": programme,
        "campaigns": campaigns,
        "record": record,
        "chain": chain,
        "partition": _read_json(_os.path.join(_cfg.ARTIFACTS_DIR, "lab_partition.json")),
        "min_effect": _lab.MIN_EFFECT,
        "claim_types": {
            "HELD_OUT": "measured on patients no agent could see or fit on",
            "MECHANISM": "PBPK twin only — a simulator result, never an efficacy claim",
            "REFUSED": "did not clear the gate, or was vetoed",
        },
        "ledger": ledger,
        "note": "A low acceptance rate is the expected outcome. selfimprove.py accepted "
                "0 of 16 under this same protocol; the gate refusing is the gate working.",
    }


@app.get("/aegis/timeline")
def risk_timeline(bed: Optional[str] = None, horizon: int = 8) -> Dict[str, Any]:
    """The risk timeline, drawn from the model that actually decides.

    Replaces the fused-risk chart, which plotted a fusion score against a hardcoded
    0.70 while the shipped detector alarms at 0.03 on its calibrated probability.
    Every element here resolves to a shipped artifact and ``provenance`` names which.
    """
    from backend.aegis import risk_timeline as _rt
    snap = patient_snapshot(bed) if bed else snapshot()
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


@app.get("/aegis/decision")
def clinical_decision(bed: Optional[str] = None) -> Dict[str, Any]:
    """What the system DOES with this patient's score, and why.

    The decision layer is where the remaining clinical value is: the same detector read
    three ways produces false-alarm rates of 14%, 27% and 94%, a wider spread than any
    modelling change this project has found. Returns one auditable disposition — the
    action, who it routes to, the ordered reasons, the measured cost of the policy, and
    the limits that qualify it.
    """
    from backend.aegis import disposition as _disp
    snap = patient_snapshot(bed) if bed else snapshot()
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


@app.get("/aegis/audit")
def model_audit() -> Dict[str, Any]:
    """What the shipped detector's performance is actually made of.

    Serves ``artifacts/model_audit.json`` rather than measuring live, for two reasons.
    The suspicion-echo probe fits two models, which does not belong in a request handler.
    And a number a judge reads should be the number that was committed and can be
    reproduced from the repository, not one a warm cache happened to produce.

    Regenerate with ``python -m backend.ml.scripts.run_audit``.
    """
    import json as _json
    import os as _os
    from backend.ml import config as _cfg
    path = _os.path.join(_cfg.ARTIFACTS_DIR, "model_audit.json")
    try:
        with open(path) as fh:
            payload = _json.load(fh)
    except (OSError, ValueError):
        # Absent is a normal state, not an error: a machine without the artefact should
        # say the audit has not been run rather than invent one.
        return {"available": False,
                "reason": "artifacts/model_audit.json not present — run "
                          "python -m backend.ml.scripts.run_audit"}
    payload["available"] = True
    return payload


@app.get("/aegis/health/prover")
def health_prover() -> Dict[str, Any]:
    """Full detail on the shared Z3 context, including the recent failure log.

    Separate from /aegis/health because the failure log is the thing you want when
    diagnosing, and noise in a liveness probe is the thing you do not."""
    st = _PROVER.status()
    st["note"] = (
        "A corrupted Z3 context cannot be repaired in-process. While unhealthy the "
        "safety cage fails CLOSED — no protocol is certified — and the simulator "
        "drive loop halts rather than keep asking a prover it does not trust. "
        "Recovery is a process restart.")
    return st


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


# Per-bed advisory-synthesis cache: the GA + Z3 solve need not re-run on every 4s
# desktop poll while the bed's band is unchanged (keyed on bed + risk band).
_bed_advisory_cache: Dict[str, Dict[str, Any]] = {}


def _bed_advisory(bed: str, v: Dict[str, float], det: Dict[str, Any],
                  level: str, hist: Dict[str, Any]) -> Dict[str, Any]:
    """Projected treatment (protocol + Z3 proof + narrative) and what-if trajectories
    for an ELEVATED cohort bed, so the desktop's protocol/what-if cards are live for a
    selected septic patient — not a permanent 'awaiting'. At MEDIUM it is advisory (a
    proven candidate, never auto-applied); only a certified HIGH protocol is eligible."""
    empty = {"protocol": None, "verification": None, "whatif": None, "narrative": None}
    if level not in ("MEDIUM", "HIGH"):
        return empty
    cached = _bed_advisory_cache.get(bed)
    if cached and cached.get("band") == level:
        return cached["out"]
    try:
        from backend.core.evolution import EvolutionaryOptimizer
        from backend.core.verifier import ProtocolVerifier
        advisory = level != "HIGH"
        protocol, fitness = EvolutionaryOptimizer().optimize(v, hist, level)
        is_safe, msg, proof = ProtocolVerifier().verify_protocol(hist, v, level, protocol)
        whatif = session.os.scheduler.project(v, det, level, hist)
        try:
            from backend.llm.explainer import AlertExplainer
            narr = AlertExplainer().narrate(level, det, v, triggered_rules=[], use_llm=False)
        except Exception:
            narr = {"narrative": None, "source": None}
        out = {
            "protocol": {"protocol": protocol, "fitness": round(float(fitness), 4),
                         "applied": bool(is_safe) and not advisory, "advisory": advisory},
            "verification": {"safe": bool(is_safe), "message": msg,
                             "z3_status": proof.get("z3_status"),
                             "violated_rules": proof.get("violated_rules", [])},
            "whatif": whatif,
            "narrative": narr.get("narrative"), "narrative_source": narr.get("source"),
        }
    except Exception as exc:  # pragma: no cover — never break the bed snapshot
        print(f"[aegis] bed advisory synth failed for {bed}: {exc}")
        return empty
    _bed_advisory_cache[bed] = {"band": level, "out": out}
    return out


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
    # Elevated cohort beds get a projected treatment + what-if so their protocol/what-if
    # cards are live (advisory at MEDIUM); stable beds stay null (nothing to synthesize).
    adv = _bed_advisory(bed, v, det, level, hist)
    return {
        "schema_version": "1.4", "timestamp": time.time(), "source": "recorded · " + bed,
        "patient": {"id": bed, "name": rp.name, "age": rp.age, "gender": rp.gender,
                    "clinical_history": hist},
        "risk": {"level": level, "ml_probability": round(personalized, 4), "ns_risk_level": level,
                 "population_risk": round(fused, 4), "personalized_risk": round(personalized, 4),
                 "susceptibility": _susc or None,
                 "sirs_score": sirs, "qsofa_score": qsofa, "signal_quality": 1.0,
                 # The conformal bands describe the CALIBRATED DETECTOR's output
                 # distribution and nothing else. A cohort bed's risk is a fused
                 # cohort score read against entirely different bands (0.35/0.6, not
                 # the detector's 0.0525), so no conformal claim covers it. Said out
                 # loud rather than omitted: a missing field reads as "resolved",
                 # and quietly applying the bands to a differently-scaled score is
                 # the threshold/fit-coupling error this repo already paid for once.
                 "certainty": {"unavailable": True, "state": "no_claim",
                               "reason": "cohort bed risk is a fused cohort score, not "
                                         "the calibrated detector probability the "
                                         "conformal bands were derived from"}},
        "vitals": v, "detection": det, "triggered_rules": [],
        "body_state": body, "reasoning": reasoning, "whatif": adv["whatif"], "consensus": None,
        "history": {"t": [], "channels": {}, "risk": [], "probability": [], "region_severity": {}},
        "protocol": adv["protocol"], "verification": adv["verification"],
        "narrative": adv["narrative"], "narrative_source": adv.get("narrative_source"),
        "causal_graph": ing.summary() if ing else None,
        # The imported chart, not just the risk uplift derived from it. It was
        # already loaded here to compute susceptibility, but only the number
        # travelled onward — so the consult could tell you a patient's alerting
        # prior had been raised and not that they are dialysis-dependent with a
        # penicillin allergy. The record itself is the clinically useful half.
        "record": _hobj or None,
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
                        signal_quality=payload.signal_quality,
                        evidence_capabilities=payload.evidence_capabilities)
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


# --- Treatment projection: the pharmacokinetic twin ------------------------ #
# Where an ordered drug actually goes, and what that exposure risks. The heavy
# work is a matrix-exponential PBPK solve plus a Monte Carlo, so results are
# cached on the exact inputs that determine them — the protocol and the patient
# covariates. A bed whose vitals have not moved re-serves its projection for free.
_PK_CACHE: Dict[str, Any] = {}
_PK_CACHE_MAX = 64


def _pk_patient(bed_id: str):
    """Derive PK covariates from a live bed: its profile plus imported history."""
    from backend.aegis.pharmacokinetics import patient_from_profile
    prof = getattr(cluster, "profiles", {}).get(bed_id)
    hist = getattr(cluster, "histories", {}).get(bed_id, {}) or {}
    return patient_from_profile(prof, hist), hist


def _pk_protocol(bed_id: str) -> Dict[str, Any]:
    """The order to project: what was actually ADMINISTERED if this bed has been
    treated, otherwise the protocol the bundle engine would certify for it. A
    projection of an order nobody placed would be a different question."""
    iv = syscalls.interventions.get(bed_id)
    if iv is not None and iv.treated and iv.protocol:
        return dict(iv.protocol)
    prof = getattr(cluster, "profiles", {}).get(bed_id)
    hist = getattr(cluster, "histories", {}).get(bed_id, {}) or {}
    return dict(bundle_engine._derive_protocol(prof, hist))


def _pk_key(protocol: Dict[str, Any], patient, adverse: bool) -> str:
    p = patient.to_dict()
    return (f"{protocol.get('antibiotic_code', 0)}|{protocol.get('fluid_rate', 0):.2f}"
            f"|{protocol.get('vasopressor_dose', 0):.3f}|{p['weight_kg']:.1f}"
            f"|{p['egfr']:.0f}|{p['map_mmhg']:.0f}|{p['hepatic_factor']:.2f}"
            f"|{p['cardiac_output_factor']:.2f}|{int(adverse)}")


def _pk_store(key: str, value: Any) -> Any:
    if len(_PK_CACHE) >= _PK_CACHE_MAX:
        _PK_CACHE.pop(next(iter(_PK_CACHE)), None)
    _PK_CACHE[key] = value
    return value


@app.get("/aegis/pk/drugs")
def pk_drugs() -> Dict[str, Any]:
    """The drug library behind the treatment twin, with every PK parameter's
    citation and the adverse-event catalogue's evidence tiers. This is the
    provenance surface: everything the projection is built from, inspectable."""
    from backend.aegis.pharmacokinetics import (DRUGS, ANTIBIOTIC_CODE_TO_DRUG,
                                                FLUID_SOURCE, ORGAN_MESH,
                                                PHYSIOLOGY_SOURCE)
    from backend.aegis.adverse import CATALOGUE, VARIABILITY_SOURCE
    return {
        "drugs": [{
            "key": s.key, "label": s.label, "class": s.drug_class,
            "vd_l_per_kg": s.vd_l_per_kg, "cl_l_per_h_per_kg": s.cl_l_per_h_per_kg,
            "t_half_h": s.t_half_h, "f_renal": s.f_renal,
            "protein_binding": s.protein_binding, "regimen": s.regimen.label,
            "horizon_h": s.horizon_h, "unit": s.conc_unit,
            "mic_mg_l": s.mic_mg_l, "target": s.target_note, "citation": s.citation,
        } for s in DRUGS.values()],
        "fluid": {"key": "crystalloid", "label": "Crystalloid (balanced)",
                  "class": "fluid", "citation": FLUID_SOURCE},
        "antibiotic_codes": ANTIBIOTIC_CODE_TO_DRUG,
        "organ_mesh": ORGAN_MESH,
        "adverse_events": [e.to_dict() for e in CATALOGUE],
        "provenance": {"physiology": PHYSIOLOGY_SOURCE,
                       "variability": VARIABILITY_SOURCE},
        "honesty": ("Literature-parameterised population PK and population "
                    "adverse-event rates. Not therapeutic drug monitoring, and "
                    "not a prediction for an individual patient."),
    }


class PKProjectIn(BaseModel):
    bed: Optional[str] = None                    # derive patient + order from a bed
    protocol: Optional[Dict[str, Any]] = None    # or state the order explicitly
    patient: Optional[Dict[str, Any]] = None     # or the covariates explicitly
    adverse: bool = True                         # also run the side-effect projection
    n_points: int = 121


def _pk_resolve(payload: "PKProjectIn"):
    """Turn a request into (protocol, patient, history) or an error string."""
    from backend.aegis.pharmacokinetics import PatientState
    if payload.bed:
        if payload.bed not in getattr(cluster, "profiles", {}):
            return None, None, None, f"unknown bed '{payload.bed}'"
        patient, hist = _pk_patient(payload.bed)
        protocol = payload.protocol or _pk_protocol(payload.bed)
    else:
        patient = PatientState(**(payload.patient or {}))
        protocol = payload.protocol or {}
        hist = {}
    if payload.patient and payload.bed:
        # An explicit covariate override on a real bed: honour it, but only the
        # fields actually supplied — the rest stay derived from the live patient.
        for k, v in payload.patient.items():
            if hasattr(patient, k) and v is not None:
                setattr(patient, k, float(v))
    return protocol, patient, hist, None


@app.post("/aegis/pk/project")
def pk_project(payload: PKProjectIn) -> Dict[str, Any]:
    """Project a certified order through the body: per-organ drug concentration
    over each arm's own timeframe, plus (by default) the side-effect projection.

    This is what the treatment twin renders. It never modifies anything — a
    projection is a read, and the Z3 cage remains the only thing that can gate an
    order."""
    from backend.aegis.pharmacokinetics import project_protocol
    protocol, patient, hist, err = _pk_resolve(payload)
    if err:
        return {"status": "error", "detail": err}

    key = _pk_key(protocol, patient, payload.adverse)
    if key in _PK_CACHE:
        return _PK_CACHE[key]

    out = project_protocol(protocol, patient, n_points=max(41, min(401, payload.n_points)))
    out["status"] = "ok"
    out["bed"] = payload.bed
    if payload.adverse and out["arms"]:
        from backend.aegis.adverse import AdverseEventEngine
        out["adverse"] = AdverseEventEngine().project(out["arms"], patient, hist)
    return _pk_store(key, out)


@app.get("/aegis/pk/bed/{bed_id}")
def pk_bed(bed_id: str, adverse: bool = True, n_points: int = 121) -> Dict[str, Any]:
    """The one-call form the dossier uses: project this bed's order as it stands.

    If the bed has been treated, this projects what was actually administered; if
    not, it projects the protocol the bundle engine would certify — and says which,
    so the UI never implies a drug is running when none was ordered."""
    if bed_id not in getattr(cluster, "profiles", {}):
        return {"status": "error", "detail": f"unknown bed '{bed_id}'"}
    iv = syscalls.interventions.get(bed_id)
    administered = bool(iv is not None and iv.treated and iv.protocol)
    out = pk_project(PKProjectIn(bed=bed_id, adverse=adverse, n_points=n_points))
    if out.get("status") == "ok":
        # Copied, not mutated in place: the cached payload is shared, and the
        # administered flag depends on syscall state that can change under it.
        out = dict(out)
        out["administered"] = administered
        out["source"] = "administered order" if administered else "certified protocol (not yet ordered)"
        prof = cluster.profiles[bed_id]
        out["bed_name"] = getattr(prof, "name", bed_id)
    return out


def _pk_context(bed: Optional[str]):
    """(base protocol, patient, history, risk level, display name) for a bed, or
    for the live single-patient session when ``bed`` is omitted."""
    from backend.aegis.pharmacokinetics import PatientState
    if bed:
        if bed not in getattr(cluster, "profiles", {}):
            return None
        patient, hist = _pk_patient(bed)
        prof = cluster.profiles[bed]
        risk = float(getattr(prof, "fused_risk", 0.0) or 0.0)
        level = "HIGH" if risk >= 0.70 else "MEDIUM" if risk >= 0.40 else "LOW"
        return (_pk_protocol(bed), patient, hist, level,
                getattr(prof, "name", bed))
    snap = session.snapshot()
    vitals = snap.get("vitals") or {}
    hist = (snap.get("patient") or {}).get("clinical_history") or {}
    proto = dict((snap.get("protocol") or {}).get("protocol") or {})
    mapv = float(vitals.get("map", 80.0) or 80.0)
    hr = float(vitals.get("heart_rate", 90.0) or 90.0)
    level = (snap.get("risk") or {}).get("level", "LOW")
    if not proto:
        prof = SimpleNamespace(
            enterprise={"map": mapv,
                        "lactate": float(vitals.get("sweat_lactate",
                                                    vitals.get("lactate", 1.0)) or 1.0)},
            fused_risk=float((snap.get("risk") or {}).get("ml_probability", 0.0) or 0.0),
            name="patient", drivers=[], primary_system="")
        proto = dict(bundle_engine._derive_protocol(prof, hist))
    patient = PatientState(
        weight_kg=float(hist.get("weight_kg", 78.0) or 78.0),
        egfr=25.0 if hist.get("renal_failure") else 90.0,
        hepatic_factor=0.6 if hist.get("cirrhosis") else 1.0, map_mmhg=mapv,
        cardiac_output_factor=float(min(1.6, max(0.7, 0.85 + (hr - 80.0) / 200.0))))
    return (proto, patient, hist, level,
            (snap.get("patient") or {}).get("name", "Monitored patient"))


@app.get("/aegis/pk/options")
def pk_options(bed: Optional[str] = None, n_points: int = 121) -> Dict[str, Any]:
    """Several candidate orders, each projected through the body.

    Same candidate names the what-if card already shows, but each one carries a
    real PBPK projection, its adverse-event profile, a verdict from the same Z3
    cage the write path uses, and upsides/downsides generated from those numbers.
    This is what makes the treatment twin a comparison rather than a single
    scenario."""
    from backend.aegis.treatment_options import evaluate_options
    ctx = _pk_context(bed)
    if ctx is None:
        return {"status": "error", "detail": f"unknown bed '{bed}'"}
    proto, patient, hist, level, name = ctx
    key = "opt|" + (bed or "session") + "|" + _pk_key(proto, patient, True) + f"|{level}"
    if key in _PK_CACHE:
        return _PK_CACHE[key]
    out = evaluate_options(proto, patient, hist, risk_level=level,
                           n_points=max(41, min(401, n_points)))
    out["status"] = "ok"
    out["bed"] = bed
    out["bed_name"] = name
    iv = syscalls.interventions.get(bed) if bed else None
    out["administered"] = bool(iv is not None and iv.treated and iv.protocol)
    return _pk_store(key, out)


@app.get("/aegis/pk/session")
def pk_session(adverse: bool = True, n_points: int = 121) -> Dict[str, Any]:
    """The same projection for the LIVE single-patient session the clinical console
    watches, rather than for a cohort bed.

    The clinical console has one patient and no bed id, so it cannot use
    ``/aegis/pk/bed/{id}``. Same contract, same honesty: if a protocol has actually
    been dispatched into the session it projects that, otherwise it projects what
    the current assessment would certify and says so."""
    from backend.aegis.pharmacokinetics import PatientState, project_protocol
    snap = session.snapshot()
    vitals = snap.get("vitals") or {}
    hist = (snap.get("patient") or {}).get("clinical_history") or {}
    proto_block = snap.get("protocol") or {}
    proto = dict(proto_block.get("protocol") or {})
    administered = bool(proto_block.get("applied"))
    if not proto:
        # Nothing certified yet — derive what would be ordered for this patient, so
        # the twin can still answer "where would this go" before anyone commits.
        prof = SimpleNamespace(
            enterprise={"map": float(vitals.get("map", 80.0) or 80.0),
                        "lactate": float(vitals.get("sweat_lactate",
                                                    vitals.get("lactate", 1.0)) or 1.0)},
            fused_risk=float((snap.get("risk") or {}).get("ml_probability", 0.0) or 0.0),
            name=(snap.get("patient") or {}).get("name", "patient"),
            drivers=[], primary_system="")
        proto = dict(bundle_engine._derive_protocol(prof, hist))
        administered = False

    mapv = float(vitals.get("map", 80.0) or 80.0)
    hr = float(vitals.get("heart_rate", 90.0) or 90.0)
    egfr = 25.0 if hist.get("renal_failure") else 90.0
    patient = PatientState(
        weight_kg=float(hist.get("weight_kg", 78.0) or 78.0), egfr=egfr,
        hepatic_factor=0.6 if hist.get("cirrhosis") else 1.0, map_mmhg=mapv,
        cardiac_output_factor=float(min(1.6, max(0.7, 0.85 + (hr - 80.0) / 200.0))))

    key = "session|" + _pk_key(proto, patient, adverse)
    if key in _PK_CACHE:
        out = dict(_PK_CACHE[key])
    else:
        out = project_protocol(proto, patient,
                               n_points=max(41, min(401, n_points)))
        out["status"] = "ok"
        if adverse and out["arms"]:
            from backend.aegis.adverse import AdverseEventEngine
            out["adverse"] = AdverseEventEngine().project(out["arms"], patient, hist)
        _pk_store(key, out)
        out = dict(out)
    out["administered"] = administered
    out["source"] = ("administered order" if administered
                     else "certified protocol (not yet ordered)")
    out["bed_name"] = (snap.get("patient") or {}).get("name", "Monitored patient")
    return out


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
    out = res.to_dict()
    if res.allowed and res.syscall == "treat":
        # A certified order now has somewhere to go: the treatment twin can show
        # where this drug travels and what it risks. The trap gate deliberately
        # does NOT compute it here — a PBPK solve plus a Monte Carlo does not
        # belong on the write path, and a projection is a read. The console opens
        # the twin against this href.
        out["projection"] = {"available": True,
                             "href": f"/aegis/pk/bed/{payload.bed}",
                             "label": "treatment projection"}
    return out


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


# ====================================================================== #
# AI DOCTOR CONSULT — call a colleague about a case.
#
# Read-only, like every other prose surface in AegisOS: the doctor discusses
# state that already exists and has no path to the syscall gate. What is new is
# that the conversation is two-way and realtime, which is only safe because the
# grounding (backend/llm/case.py) travels with the model as a locked system
# instruction and carries the detector's real external-validation numbers.
#
# The audio socket relays through THIS process on purpose. Ephemeral Gemini auth
# tokens are rejected by the Live socket on this account, so a browser-direct
# design would have to ship the API key to a static page. It does not.
# ====================================================================== #
from backend.llm.doctor import AIDoctor as _AIDoctor
from backend.llm.live import LiveConsult as _LiveConsult, LiveUnavailable as _LiveUnavailable

_doctor = _AIDoctor(_llm_client)


# The banding that actually produces the tier shown on screen (server.py:719).
# Passed to the dossier so the doctor states THIS scale and is barred from
# comparing it against the detector's out-of-fold operating point, which is a
# different quantity entirely.
_TIER_BANDS = {"high": 0.6, "medium": 0.35}


def _doctor_dossier(bed: Optional[str]) -> Dict[str, Any]:
    """Grounded dossier for a bed (or the live session when no bed is named)."""
    snap = patient_snapshot(bed) if bed else patient_snapshot(None)
    try:
        roster = _patient_roster()
    except Exception:
        roster = None
    return _doctor.dossier(snap or {}, cohort=roster, tier_bands=_TIER_BANDS)


class DoctorConsultIn(BaseModel):
    question: str
    bed: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    agentic: bool = True                  # may drive the OS to answer


@app.get("/aegis/doctor/status")
def doctor_status() -> Dict[str, Any]:
    """Can we place a call, by which transport, and if not — why not?"""
    voice_ok, voice_reason = _LiveConsult.availability()
    return {
        "voice": voice_ok,                        # realtime audio socket
        "voice_reason": voice_reason,
        "key_source": _rtcfg.source("GEMINI_API_KEY"),   # environment | supabase | unset
        "text": bool(_llm_client.available),      # typed consult / fallback
        "llm": _llm_client.status(),
    }


@app.get("/aegis/doctor/brief")
def doctor_brief(bed: Optional[str] = None, opening: bool = True) -> Dict[str, Any]:
    """The case dossier the doctor is grounded in, plus its opening statement.

    Returned before the call connects so the UI can show what the doctor can see —
    a consult whose grounding is invisible is impossible to audit."""
    dossier = _doctor_dossier(bed)
    out: Dict[str, Any] = {
        "bed": dossier.get("bed"), "headline": dossier.get("headline"),
        "facts": dossier.get("facts"), "unmeasured": dossier.get("unmeasured"),
        "provenance": dossier.get("provenance"), "dossier_text": dossier.get("text"),
        "voice": _LiveConsult.available(),
        "voice_reason": _LiveConsult.availability()[1],
    }
    if opening:
        out["opening"] = _doctor.opening(dossier)
    return out


from backend.aegis.hyperscan import (Hyperscan as _Hyperscan, STAGES as _HYPERSCAN_STAGES,
                                     REVIEW_NOTICE as _REVIEW)

_hyperscan_engine: Optional[_Hyperscan] = None


def _run_hyperscan(bed: Optional[str],
                   emit: Optional[Any] = None) -> Dict[str, Any]:
    # Built on first use: this block sits above the agent plane in the module, so
    # binding the tool registry at import time would depend on statement order.
    global _hyperscan_engine
    if _hyperscan_engine is None:
        _hyperscan_engine = _Hyperscan(_tools.invoke, _llm_client)
    dossier = _doctor_dossier(bed)
    collected = _hyperscan_engine.collect(bed, emit)
    return _hyperscan_engine.synthesize(bed, collected, dossier, emit)


from backend.llm import speech as _speech


class SpeakIn(BaseModel):
    text: str
    voice: Optional[str] = None


@app.get("/aegis/speech")
def speech_status() -> Dict[str, Any]:
    """Can the assistant read something out, and in whose voice?"""
    return _speech.status()


@app.post("/aegis/speech")
def speak(payload: SpeakIn) -> Dict[str, Any]:
    """Render text to speech in the assistant's voice.

    Same voice and same 24 kHz mono PCM as the consult call, so the assistant
    sounds like one clinician wherever you reach it. Returns
    {available:false} rather than an error when speech is unavailable — the
    caller always has the text and must not lose it because audio failed."""
    out = _speech.synthesize(payload.text, payload.voice)
    if not out:
        return {"available": False, "reason": (
            "no GEMINI_API_KEY on the server" if not _speech.available()
            else "speech budget exhausted or the renderer failed")}
    return {"available": True, **out}


@app.get("/aegis/hyperscan")
def hyperscan(bed: Optional[str] = None) -> Dict[str, Any]:
    """Full-subsystem sweep + situation report for one patient (blocking)."""
    return _run_hyperscan(bed)


@app.post("/aegis/hyperscan/stream")
def hyperscan_stream(payload: DoctorConsultIn):
    """The sweep, narrated stage by stage.

    Streamed because the point is watchable: each subsystem lands as its own
    event so the body on screen can flag the finding while the rest is still
    being collected. A nine-stage sweep behind a spinner would be the same work
    and none of the legibility."""
    from fastapi.responses import StreamingResponse
    import queue as _queue

    q: "_queue.Queue" = _queue.Queue()

    def work() -> None:
        try:
            q.put({"type": "begin", "stages": len(_HYPERSCAN_STAGES) + 1})
            rep = _run_hyperscan(payload.bed, q.put)
            # Report first, THEN the voice. Rendering speech takes ~12 s; holding
            # the finished report back for it would trade a fast read for a slow
            # listen. This way the report is on screen immediately and the audio
            # arrives while it is being read.
            q.put({"type": "report", "report": rep})
            spoken = rep.get("spoken")
            if spoken:
                q.put({"type": "speech_pending"})
                audio = _speech.synthesize(spoken)
                q.put({"type": "speech", "text": spoken,
                       **(audio or {"available": False})})
        except Exception as exc:                    # pragma: no cover
            q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            q.put(None)

    _threading.Thread(target=work, name="hyperscan", daemon=True).start()

    def events():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/aegis/doctor/consult/stream")
def doctor_consult_stream(payload: DoctorConsultIn):
    """The typed consult, narrated as it works.

    An agentic answer that chains four tools takes about nine seconds. Returned
    as one blob that is nine seconds of nothing, which reads as a hang — the user
    reported it as "takes a long time" long before the model was actually slow.
    Same work, streamed: each tool is announced before it runs, so the wait is
    legible and the cockpit can follow along live instead of catching up at the
    end."""
    from fastapi.responses import StreamingResponse
    import queue as _queue

    q: "_queue.Queue" = _queue.Queue()

    def work() -> None:
        try:
            dossier = _doctor_dossier(payload.bed)
            q.put({"type": "grounded", "bed": dossier.get("bed"),
                   "headline": dossier.get("headline"), "facts": dossier.get("facts")})
            res = _doctor.act(payload.question, dossier, _tools.declarations(),
                              _tools.invoke, history=payload.history,
                              budgets={t.name: t.budget for t in _tools._tools.values()
                                       if t.budget},
                              on_event=q.put)
            q.put({"type": "answer", **res})
        except Exception as exc:                       # pragma: no cover
            q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            q.put(None)

    _threading.Thread(target=work, name="consult-stream", daemon=True).start()

    def events():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/aegis/doctor/consult")
def doctor_consult(payload: DoctorConsultIn) -> Dict[str, Any]:
    """One typed turn of the consult — the transport when there is no microphone,
    and the fallback when the realtime socket is unavailable.

    Agentic by default: the doctor may drive the OS to answer, and any cockpit
    commands it issues come back in ``actions`` for the browser to run, so the
    typed path has the same reach as the voice path."""
    dossier = _doctor_dossier(payload.bed)
    if payload.agentic:
        res = _doctor.act(payload.question, dossier, _tools.declarations(),
                          _tools.invoke, history=payload.history,
                          budgets={t.name: t.budget for t in _tools._tools.values()
                                   if t.budget})
    else:
        res = _doctor.consult(payload.question, dossier, history=payload.history)
    return {**res, "bed": dossier.get("bed"), "headline": dossier.get("headline"),
            "budget_remaining": _llm_client.budget_remaining()}


# ---------------------------------------------------------------------- #
# The agent plane: bind every tool in the catalogue to a real handler.
#
# Reads go straight to the endpoint functions. Writes go to the SAME Z3-gated,
# audited trap gate the console uses — the agent may invoke any syscall in the
# table, and the prover decides which ones take effect. UI tools have no server
# handler; they are forwarded to the cockpit over the consult socket.
# ---------------------------------------------------------------------- #
from backend.aegis.agent import registry as _tools, summarize as _tool_summarize

# The read primitive is confined to this app's own GET routes. It is genuinely
# unrestricted over AegisOS state — and it is a *read*: it cannot leave the
# process and cannot mutate anything.
_READ_ROUTES: Dict[str, Any] = {}
# The write primitive covers the OS's own POST surface. Clinical egress still
# lands in the Z3 trap gate, because the gate lives INSIDE /aegis/syscall — it is
# a property of the endpoint, not a restriction bolted on at this layer.
_WRITE_ROUTES: Dict[str, Any] = {}
_MAIN_LOOP: Optional[Any] = None


def _index_read_routes() -> None:
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", "")
        if not path.startswith("/aegis") or not getattr(route, "endpoint", None):
            continue
        if "GET" in methods:
            _READ_ROUTES[path] = route
        if "POST" in methods:
            _WRITE_ROUTES[path] = route


def _agent_await(coro: Any, timeout: float = 30.0) -> Any:
    """Run a coroutine endpoint from the agent plane, wherever we are called from.

    Sync REST handlers run on a threadpool worker with no loop, so ``asyncio.run``
    is right there. The consult relay is already ON the main loop, where
    ``asyncio.run`` would raise — so we hand the coroutine to that loop and wait
    on the future from this thread."""
    running = None
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        pass
    if running is None:
        if _MAIN_LOOP is not None and _MAIN_LOOP.is_running():
            fut = asyncio.run_coroutine_threadsafe(coro, _MAIN_LOOP)
            return fut.result(timeout)
        return asyncio.run(coro)
    # Called from inside the loop that would have to run it: we cannot block on
    # ourselves. Schedule it and report that it is in flight.
    running.create_task(coro)
    return {"status": "scheduled", "note": "dispatched on the server's event loop"}


def _agent_os_read(path: str, params: str = "") -> Dict[str, Any]:
    """Generic GET over the OS's own state surface, by path.

    Path params are resolved with Starlette's own matcher rather than by string
    surgery — a hand-rolled prefix match happily read '/aegis/cohort/bed_0/bundle'
    as bed_id='bed_0/bundle'. Query values are coerced from the parameter's
    DEFAULT, not its annotation: this module runs under
    `from __future__ import annotations`, so every annotation is a string and
    type-based coercion silently never fires."""
    from urllib.parse import parse_qs
    import inspect

    raw = (path or "").split("?")[0].rstrip("/") or "/aegis/health"
    if not raw.startswith("/aegis"):
        return {"error": f"{raw!r} is not an AegisOS state path (must start /aegis)"}
    if "?" in (path or ""):
        params = params or path.split("?", 1)[1]
    if not _READ_ROUTES:
        _index_read_routes()

    query: Dict[str, Any] = {k: v[0] for k, v in parse_qs(params or "").items()}
    scope = {"type": "http", "method": "GET", "path": raw, "headers": [],
             "root_path": "", "query_string": b""}
    fn = None
    path_params: Dict[str, Any] = {}
    for route in _READ_ROUTES.values():
        try:
            match, child = route.matches(scope)
        except Exception:
            continue
        if match.value == 2:                       # Match.FULL
            fn = route.endpoint
            path_params = dict(child.get("path_params") or {})
            break
    if fn is None:
        return {"error": f"no readable surface at {raw!r}",
                "hint": "call list_surfaces to see what exists"}

    supplied = dict(query)
    supplied.update(path_params)                   # path wins over query
    call: Dict[str, Any] = {}
    for name, prm in inspect.signature(fn).parameters.items():
        if name in supplied:
            val = supplied[name]
            default = prm.default
            try:
                if isinstance(default, bool):
                    val = str(val).lower() in ("1", "true", "yes")
                elif isinstance(default, int):
                    val = int(val)
                elif isinstance(default, float):
                    val = float(val)
            except (TypeError, ValueError):
                pass
            call[name] = val
        elif prm.default is inspect.Parameter.empty:
            return {"error": f"{raw!r} needs a '{name}' argument",
                    "hint": f"pass it in params, e.g. params='{name}=…'"}
    out = fn(**call)
    if inspect.iscoroutine(out):
        out.close()
        return {"error": f"{raw!r} is async and not reachable from the agent plane"}
    return out


def _body_model(field: Any) -> Any:
    """The Pydantic model behind a FastAPI body param, across both major versions.

    Pydantic v1 exposed ``ModelField.type_``; under v2 (what this project pins)
    FastAPI's shim carries the real annotation on ``field_info.annotation``, and
    reaching for ``.type_`` raises."""
    ann = getattr(getattr(field, "field_info", None), "annotation", None)
    return ann if ann is not None else getattr(field, "type_", None)


def _model_fields(model: Any) -> List[str]:
    return list(getattr(model, "model_fields", None) or getattr(model, "__fields__", {}) or {})


def _agent_os_write(path: str, body: Optional[Dict[str, Any]] = None,
                    params: str = "") -> Dict[str, Any]:
    """Generic POST over the OS's own actuation surface — the universal DO.

    Reaches every mutating endpoint AegisOS exposes: the ML control plane, the
    package manager, chaos injection and recovery, counterfactual replay, the
    eBPF driver verifier, device pairing, record import, proof certification.

    Clinical egress is NOT special-cased here and does not need to be. The Z3
    trap gate lives inside ``/aegis/syscall`` itself, so a write that would reach
    a patient still has to come back with a certificate whether it arrives from
    the console, from a named tool, or from this primitive. There is one gate and
    every road leads to it."""
    from urllib.parse import parse_qs
    import inspect

    raw = (path or "").split("?")[0].rstrip("/")
    if not raw.startswith("/aegis"):
        return {"error": f"{raw!r} is not an AegisOS path (must start /aegis)"}
    if not _WRITE_ROUTES:
        _index_read_routes()
    if raw in ("/aegis/agent/invoke", "/aegis/doctor/consult"):
        # Re-entering the agent through itself is a loop, not a capability.
        return {"error": f"{raw!r} would re-enter the agent plane; call the tool directly"}

    scope = {"type": "http", "method": "POST", "path": raw, "headers": [],
             "root_path": "", "query_string": b""}
    route = None
    path_params: Dict[str, Any] = {}
    for r in _WRITE_ROUTES.values():
        try:
            match, child = r.matches(scope)
        except Exception:
            continue
        if match.value == 2:
            route = r
            path_params = dict(child.get("path_params") or {})
            break
    if route is None:
        return {"error": f"no actuation surface at {raw!r}",
                "hint": "call list_actions to see what can be driven"}

    fn = route.endpoint
    supplied: Dict[str, Any] = {k: v[0] for k, v in parse_qs(params or "").items()}
    supplied.update(path_params)
    # FastAPI has already resolved the body model for us — important, because this
    # module's `from __future__ import annotations` makes raw annotations strings.
    body_models = {p.name: _body_model(p)
                   for p in (getattr(route, "dependant", None).body_params
                             if getattr(route, "dependant", None) else [])}
    call: Dict[str, Any] = {}
    for name, prm in inspect.signature(fn).parameters.items():
        if name in body_models:
            try:
                call[name] = body_models[name](**(body or {}))
            except Exception as exc:
                return {"error": f"invalid body for {raw!r}: {exc}",
                        "expects": _model_fields(body_models[name])}
        elif name in supplied:
            val, default = supplied[name], prm.default
            try:
                if isinstance(default, bool):
                    val = str(val).lower() in ("1", "true", "yes")
                elif isinstance(default, int):
                    val = int(val)
                elif isinstance(default, float):
                    val = float(val)
            except (TypeError, ValueError):
                pass
            call[name] = val
        elif prm.default is inspect.Parameter.empty:
            return {"error": f"{raw!r} needs a '{name}' argument"}
    out = fn(**call)
    if inspect.iscoroutine(out):
        out = _agent_await(out)
    return out if isinstance(out, dict) else {"result": out}


def _agent_list_actions() -> Dict[str, Any]:
    if not _WRITE_ROUTES:
        _index_read_routes()
    out = []
    for path in sorted(_WRITE_ROUTES):
        if path in ("/aegis/agent/invoke", "/aegis/doctor/consult"):
            continue
        route = _WRITE_ROUTES[path]
        doc = (route.endpoint.__doc__ or "").strip().split("\n")[0]
        fields: List[str] = []
        dep = getattr(route, "dependant", None)
        for p in (dep.body_params if dep else []):
            fields += _model_fields(_body_model(p))
        out.append({"path": path, "what": doc[:140], "body": fields})
    return {"count": len(out), "actions": out,
            "note": "drive any of these with os_write. Clinical egress "
                    "(/aegis/syscall) still passes the Z3 gate and may be refused."}


def _agent_list_surfaces() -> Dict[str, Any]:
    if not _READ_ROUTES:
        _index_read_routes()
    out = []
    for path in sorted(_READ_ROUTES):
        doc = (_READ_ROUTES[path].endpoint.__doc__ or "").strip().split("\n")[0]
        out.append({"path": path, "what": doc[:150]})
    return {"count": len(out), "surfaces": out}


def _agent_syscall(syscall: str, bed: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every agent write lands here — the same trap gate the console uses."""
    res = syscalls.invoke(syscall, bed, args or {}, caller="dr-aegis")
    return res.to_dict()


def _agent_patient_history(bed: Optional[str] = None) -> Dict[str, Any]:
    """The prior record for a bed, plus the chart flags the monitor already had.

    Two different things live under the word "history" here and conflating them
    would be a grounding bug: the CHART FLAGS come with the monitored record and
    are always present; the IMPORTED RECORD is a chart someone brought in and is
    usually absent. Returning both, labelled, means the doctor can say which one
    it is reading from."""
    target = bed or (cluster.node_ids[0] if getattr(cluster, "node_ids", None) else None)
    record = patient_terminal.get_history(target) if target else None
    flags = (getattr(cluster, "histories", {}) or {}).get(target) or {}
    out: Dict[str, Any] = {
        "bed_handle": target,
        "chart_flags": flags or None,
        "imported_record": record or None,
        "has_prior_record": bool(record),
    }
    if not record:
        out["note"] = ("No prior chart is on file for this patient. Say so plainly; "
                       "it can be imported (import_history) or linked from a source "
                       "(list_record_sources). The chart_flags above still apply.")
    else:
        out["note"] = ("imported_record is REPORTED HISTORY from before this "
                       "admission — not measured today, and not independently "
                       "verified. Its susceptibility uplift shifts when the system "
                       "alerts, not the patient's measured risk.")
    return out


def _agent_import_history(bed: str, text: str = "", source: Optional[str] = None,
                          patient_ref: Optional[str] = None) -> Dict[str, Any]:
    """Import a chart as free text, or link one from a record source."""
    if source and not text:
        return patient_history_link(bed, LinkIn(source=source, patient_ref=patient_ref))
    return patient_history_import(bed, HistoryIn(text=text, source=source or "manual"))


def _agent_device_action(device_id: str, action: str, bed: Optional[str] = None,
                         command: Optional[str] = None) -> Dict[str, Any]:
    """Pair / assign / unassign a device, or send it a command."""
    act = (action or "").lower()
    if act == "pair":
        return device_pair(device_id)
    if act == "assign":
        if not bed:
            return {"status": "error", "detail": "assign needs a bed"}
        return device_assign(device_id, DeviceAssignIn(bed_id=bed))
    if act == "unassign":
        return device_unassign(device_id)
    if act == "command":
        if not command:
            return {"status": "error", "detail": "command needs a command name"}
        return device_action(device_id, DeviceActionIn(action=command))
    return {"status": "error", "detail": f"unknown device action {action!r}"}


def _agent_service_control(name: str, action: str) -> Dict[str, Any]:
    """systemctl-style control, synchronous (the async endpoint also broadcasts;
    the agent plane cannot await, so it drives the supervisor directly)."""
    act = (action or "").lower()
    ok = (aegisd.stop(name) if act == "stop" else
          aegisd.restart(name) if act in ("restart", "start") else None)
    if ok is None:
        return {"status": "error", "detail": f"unknown action {action!r}"}
    if not ok:
        return {"status": "error", "detail": f"unknown service {name!r}"}
    return {"status": "ok", "service": name, "action": act,
            "state": aegisd.services[name].state}


def _agent_list_patients() -> Dict[str, Any]:
    """The roster WITH each bed's current risk and tier, sorted sickest first.

    /aegis/patients alone carries no risk at all, so "who is the sickest patient"
    was unanswerable from it — the agent picked a bed and then described a LOW
    tier as the sickest, because nothing in its input could have told it
    otherwise. The live per-bed risk lives on the cluster topology; joining the
    two here makes the ranking a fact the agent reads instead of a guess."""
    roster = patients().get("patients", [])
    live: Dict[str, Dict[str, Any]] = {}
    try:
        for node in (cluster_snapshot().get("topology") or []):
            live[str(node.get("node_id"))] = node
    except Exception:
        pass
    out = []
    for p in roster:
        pid = str(p.get("id"))
        node = live.get(pid) or {}
        risk = node.get("p")
        out.append({"name": p.get("name"),          # name FIRST — this is what gets said
                    "bed_handle": pid, "kind": p.get("kind"),
                    "risk": round(float(risk), 4) if risk is not None else None,
                    "tier": node.get("tier"), "trend": node.get("slope"),
                    "septic_course": p.get("septic"), "icu_hour": p.get("icu_hour")})
    out.sort(key=lambda r: (r["risk"] is None, -(r["risk"] or 0.0)))
    return {"n": len(out), "sorted_by": "risk, highest first", "patients": out,
            "note": "risk is the live fused per-bed risk; tier bands at "
                    f"HIGH>={_TIER_BANDS['high']}, MEDIUM>={_TIER_BANDS['medium']}. "
                    "'bed_handle' is an internal identifier for passing to other "
                    "tools — refer to patients by NAME when you speak.",
            "patients_by_risk": [p["name"] for p in out[:5]]}


def _agent_treatment_options(bed: Optional[str] = None) -> Dict[str, Any]:
    """Candidate treatment paths, trimmed to what a person can be told out loud.

    The raw payload carries full PK concentration curves per drug per arm —
    hundreds of KB the model cannot use and would have to be truncated past. What
    a clinician actually wants spoken is: which path is recommended, what each
    one trades, and which the cage refused and why. So this returns the argument,
    not the arrays."""
    raw = pk_options(bed=bed) if bed else pk_options()

    def _short(items: Any, n: int = 3) -> List[str]:
        """The top few reasons, as sentences. Each already traces to a number in
        its own option (that is what verify_treatment_options proves), so the
        first few are the argument — the rest is corroboration."""
        out = []
        for it in (items or [])[:n]:
            if isinstance(it, dict):
                txt = it.get("text") or it.get("reason") or it.get("label") or ""
            else:
                txt = str(it)
            txt = str(txt).strip()
            if txt:
                out.append(txt[:220])
        return out

    opts = []
    for o in (raw.get("options") or []):
        ver = o.get("verification") or {}
        opts.append({
            "key": o.get("key"), "label": o.get("label"), "note": o.get("note"),
            "antibiotic": o.get("antibiotic"), "protocol": o.get("protocol"),
            "allowed": bool(o.get("safe")),
            "verdict": ver.get("message"), "z3_status": ver.get("z3_status"),
            "violations": ver.get("violations") or [],
            "upsides": _short(o.get("upsides")), "downsides": _short(o.get("downsides")),
            "serious_adverse_risk": o.get("p_any_serious"),
            "n_serious": o.get("n_serious"),
        })
    refused = [o["label"] for o in opts if not o["allowed"]]
    return {
        "patient": raw.get("bed_name"), "risk_level": raw.get("risk_level"),
        "recommended": raw.get("recommended"),
        "administered": raw.get("administered"),
        "n_options": len(opts), "n_refused": len(refused), "refused": refused,
        "options": opts,
        "honesty": raw.get("honesty"),
        "note": "These are projections over candidate orders, not orders. Options "
                "marked allowed=false were REFUSED by the Z3 safety cage — quote "
                "the verdict verbatim if you mention them.",
    }


def _agent_open_panel(panel: str, bed: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The companion read behind `open_panel`: what that surface actually shows.

    Without this the agent opened a panel and then described contents it had
    never received. Now "pull it up" returns the panel's data in the same call,
    which is what the request meant and leaves nothing to invent. Panels with no
    readable payload (the camera is a live device) return None, and the registry
    then tells the model plainly that it has not seen anything."""
    p = (panel or "").lower()
    try:
        if p == "story":
            return causal_graph(bed)
        if p == "consensus":
            return consensus_round(None, bed)
        if p == "orchestrator":
            return {"status": orchestrator_status(), "attestation": orchestrator_attest()}
        if p == "federation":
            return fedlearn_result()
        if p == "pharma":
            return pharmacopoeia_stats()
        if p == "identity":
            return identity_status()
        if p == "rbac":
            return rbac_policy()
        if p == "dsl":
            return dsl_default()
        if p == "twin":
            return pk_bed(bed) if bed else pk_session()
        if p == "tactical":
            return cohort_roster()
        if p == "consult":
            return doctor_brief(bed, opening=False)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return None                          # camera: a live device, nothing to read


def _agent_create_scenario(**kw: Any) -> Dict[str, Any]:
    """Adapt the model's loose arguments onto ScenarioIn.

    The catalogue advertises numbers, and Gemini sends floats for all of them, so
    the hour-shaped fields are coerced to int here rather than trusting the wire —
    ScenarioSpec slices arrays with them, and a float index raises deep inside the
    synthesizer where the message would tell the model nothing useful."""
    def _i(v, default):
        return default if v is None else int(round(float(v)))
    try:
        payload = ScenarioIn(
            name=str(kw.get("name") or "unnamed"),
            arc=str(kw.get("arc") or "septic_shock"),
            los_hours=_i(kw.get("los_hours"), 48),
            onset_hour=None if kw.get("onset_hour") in (None, "", "none")
                       else _i(kw.get("onset_hour"), 30),
            ramp_hours=_i(kw.get("ramp_hours"), 12),
            severity=float(kw.get("severity") or 1.0),
            age=float(kw.get("age") or 68.0),
            lab_interval_hours=_i(kw.get("lab_interval_hours"), 8),
            seed=_i(kw.get("seed"), 7),
            notes=str(kw.get("notes") or ""),
            expectations=list(kw.get("expectations") or []),
        )
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"bad scenario arguments: {exc}"}
    return _agent_await(scenarios_create(payload))


def _bind_agent_tools() -> None:
    b = _tools.bind
    b("open_panel", _agent_open_panel)
    # Switching beds must re-ground the agent, not just re-scope the screen. The
    # dossier is locked into the session at connect time and never changes, so an
    # agent that moved the cockpit to bed_3 kept narrating bed_0's numbers under
    # bed_3's name. The companion read hands it the new patient's dossier in the
    # same call that makes the switch.
    b("select_patient", lambda bed: doctor_brief(bed, opening=False))
    b("show_treatment_options", _agent_treatment_options)
    b("hyperscan", lambda bed=None: _run_hyperscan(bed))
    b("os_read", _agent_os_read)
    b("os_write", _agent_os_write)
    b("list_surfaces", _agent_list_surfaces)
    b("list_actions", _agent_list_actions)
    b("list_patients", _agent_list_patients)
    b("get_patient", lambda bed=None: patient_snapshot(bed))
    b("get_case_dossier", lambda bed=None: doctor_brief(bed, opening=False))
    b("get_patient_history", _agent_patient_history)
    b("list_record_sources", lambda: records_sources())
    b("get_vitals_trend", lambda bed: cohort_chart(bed))
    b("get_causal_graph", lambda bed=None: causal_graph(bed))
    b("trace_incident", lambda node_id, bed=None: causal_incident(node_id, bed))
    b("run_consensus", lambda bed=None: consensus_round(None, bed))
    b("run_whatif", lambda bed=None: (_bed_snapshot(bed) or {}).get("whatif") if bed else whatif())
    b("project_treatment", lambda bed, adverse=True: pk_bed(bed, adverse=adverse))
    b("screen_drugs", lambda drugs, bed=None: pharmacopoeia_screen(PharmaScreenIn(
        drugs=list(drugs),
        # renal state comes from the bed's own chart, so the hazard set is this
        # patient's rather than a generic screen
        context={"renal_failure": bool(((getattr(cluster, "histories", {}) or {})
                                        .get(bed) or {}).get("renal_failure", False))})))
    # -- scenario lab: the agent poses a case, the OS is graded on it.
    #    Confined to the sandbox by construction — these handlers only ever touch
    #    the in-memory ScenarioLab, never the cohort.
    b("scenario_vocabulary", lambda: scenarios_vocabulary())
    b("list_scenarios", lambda: scenarios_list())
    b("get_scenario", lambda bed: scenarios_get(bed))
    b("create_scenario", lambda **kw: _agent_create_scenario(**kw))
    b("run_lab_scenario", lambda bed: _agent_await(scenarios_run(bed)))
    b("run_scenario_suite", lambda: _agent_await(scenarios_suite()))
    b("reset_scenarios", lambda: _agent_await(scenarios_reset()))
    b("get_validation", lambda: cohort_validation())
    b("get_orchestrator", lambda: {"status": orchestrator_status(),
                                   "attestation": orchestrator_attest()})
    b("get_kernel", lambda: _kernel_plane(full=False))
    b("get_audit", lambda limit=24: aegis_syscall_audit(int(limit)))
    b("get_activity", lambda limit=20: {"activity": _tools.activity(int(limit))})
    b("syscall", _agent_syscall)
    b("set_tracking", lambda bed, command: patient_terminal.exec(
        bed, command, context=_patient_ctx(bed)))
    b("service_control", _agent_service_control)
    b("simulate_stage", lambda stage: simulate(stage))
    # -- named actuation: thin wrappers so the model reaches for the right thing.
    #    Every signature below mirrors the endpoint's REAL payload model; do not
    #    invent a friendlier shape here, or the tool will fail on first use.
    b("counterfactual_replay", lambda bed=None, outcome_id=None, node_id=None,
      treatment=None, horizon=None: causal_replay(ReplayIn(**{
          k: v for k, v in {"bed": bed, "outcome_id": outcome_id, "node_id": node_id,
                            "treatment": treatment, "horizon": horizon}.items()
          if v is not None})))
    b("govern", lambda action, target=None, layer="bed", payload=None:
      orchestrator_govern(GovernIn(action=action, target=target, layer=layer,
                                   payload=payload or {})))
    b("run_scenario", lambda bed=None: orchestrator_scenario(bed))
    b("certify_proof", lambda risk_level=None, protocol=None, vitals=None,
      clinical_history=None: proof_core_certify(CoreProofIn(**{
          k: v for k, v in {"risk_level": risk_level, "protocol": protocol,
                            "vitals": vitals, "clinical_history": clinical_history}.items()
          if v is not None})))
    b("verify_proof", lambda certificate, require_signature=True: proof_verify(
        ProofVerifyIn(certificate=certificate, require_signature=require_signature)))
    b("verify_driver", lambda name=None, source=None, win_len=2: kernel_verify(
        VerifyIn(**{k: v for k, v in
                    {"name": name, "source": source, "win_len": win_len}.items()
                    if v is not None})))
    b("verify_rules", lambda policy=None, protocol=None, context=None:
      dsl_verify_endpoint(DslVerifyIn(**{
          k: v for k, v in {"policy": policy, "protocol": protocol,
                            "context": context}.items() if v is not None})))
    b("check_identity", lambda target=None, source=None: (
        identity_swap(IdentitySwapIn(target=target, source=source)) if source
        else identity_self_check(IdentitySwapIn(target=target))))
    b("train_model", lambda name, feature_set="wearable", horizon=None,
      limit=1000, folds=3: ml_train(MLTrainIn(**{
          k: v for k, v in {"name": name, "feature_set": feature_set,
                            "horizon": horizon, "limit": int(limit),
                            "folds": int(folds)}.items() if v is not None})))
    b("run_kernel_workload", lambda workload="mixed", policy="HDSched": hdkernel_run(
        HDKernelRunIn(workload=workload, policy=policy)))
    b("inject_fault", lambda slot, reason="injected by the consult agent":
      _agent_await(kernel_fault(KernelFaultIn(slot=slot, reason=reason))))
    b("recover_device", lambda slot, reason="recovered by the consult agent":
      _agent_await(kernel_recover(KernelFaultIn(slot=slot, reason=reason))))
    b("run_chaos", lambda node, action="crisis", intensity=1.0: _agent_await(
        cluster_chaos(ChaosIn(node=node, action=action, intensity=float(intensity)))))
    b("manage_package", lambda action, name: _agent_await(
        pkg_install(PkgIn(name=name)) if action == "install"
        else pkg_uninstall(PkgIn(name=name))))
    b("import_history", _agent_import_history)
    b("device_action", _agent_device_action)


def _gate_verdict(out: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the prover's decision out of a gated write so the cockpit can show a
    refusal as a refusal, in the certificate's own words."""
    res = out.get("result")
    if not isinstance(res, dict) or "z3_status" not in res:
        return None
    return {"allowed": res.get("allowed"), "errno": res.get("errno_name", res.get("errno")),
            "z3_status": res.get("z3_status"), "message": res.get("message"),
            "violations": res.get("violations") or []}


class AgentInvokeIn(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


@app.get("/aegis/agent/tools")
def agent_tools() -> Dict[str, Any]:
    """The agent's capability catalogue — what Dr. Aegis can reach, read and do,
    and which of those go through the certified gate."""
    cat = _tools.catalogue()
    return {"tools": cat, "bound": len(_tools.bound()), "unbound": _tools.unbound(),
            "counts": {k: sum(1 for t in cat if t["kind"] == k)
                       for k in ("read", "ui", "write")}}


@app.get("/aegis/agent/activity")
def agent_activity(limit: int = 50) -> Dict[str, Any]:
    """What the agent has actually done — every tool call and its outcome."""
    return {"activity": _tools.activity(limit)}


@app.post("/aegis/agent/invoke")
def agent_invoke(payload: AgentInvokeIn) -> Dict[str, Any]:
    """Invoke one tool directly (the text consult and the console use this)."""
    return _tools.invoke(payload.tool, payload.args)


# Uplink ceiling for one call. A stuck browser that streams microphone audio
# forever would otherwise burn the quota silently; this caps a single consult at
# a length no real bedside conversation exceeds.
_CALL_MAX_AUDIO_S = 15 * 60.0

# What counts as a MATERIAL mid-call move, per vital: (field, label, unit, delta).
# The thresholds are deliberately coarse — roughly "a nurse would mention it" —
# because a note on every flicker is noise, and an assistant that interrupts
# constantly gets tuned out exactly like an alarm that cries wolf.
_WATCHED_VITALS = (
    ("systolic_bp", "SBP", "mmHg", 10.0),
    ("map", "MAP", "mmHg", 8.0),
    ("heart_rate", "HR", "bpm", 12.0),
    ("respiratory_rate", "RR", "/min", 4.0),
    ("spo2", "SpO2", "%", 3.0),
    ("skin_temp", "temp", "°C", 0.5),
)


@app.websocket("/aegis/doctor/live")
async def doctor_live(ws: WebSocket) -> None:
    """Relay: browser audio/video  <->  Gemini Live, grounded in the bed's dossier.

    Browser frames (JSON): audio{data} · video{data,mime} · text{text} ·
    end_audio · bye. Server frames: ready · transcript · audio · turn_end ·
    interrupted · case_update · error · closed."""
    await ws.accept()
    bed = ws.query_params.get("bed") or None
    session_live: Optional[_LiveConsult] = None
    try:
        dossier = _doctor_dossier(bed)
    except Exception as exc:
        await ws.send_json({"type": "error", "fatal": True,
                            "message": f"could not assemble the case: {exc}"})
        await ws.close()
        return

    try:
        session_live = _LiveConsult(_doctor.system_instruction(dossier),
                                    tools=_tools.declarations())
        await session_live.connect()
    except _LiveUnavailable as exc:
        # Not an error the clinician can act on — tell the UI to fall back to the
        # typed consult, which is grounded in the identical dossier.
        await ws.send_json({"type": "error", "fatal": True, "fallback": "text",
                            "message": str(exc)})
        await ws.close()
        return

    await ws.send_json({
        "type": "ready", "bed": dossier.get("bed"), "headline": dossier.get("headline"),
        "facts": dossier.get("facts"), "unmeasured": dossier.get("unmeasured"),
        "provenance": dossier.get("provenance"),
        "input_rate": 16000, "output_rate": 24000, "voice": session_live.voice,
        "model": session_live.model, "tools": len(session_live.tools),
    })

    stop = asyncio.Event()
    budget = {"audio_s": 0.0}
    # Which bed the conversation is actually on. The agent can move it mid-call.
    scope: Dict[str, Any] = {
        "bed": bed,
        "tier": (dossier.get("facts") or {}).get("tier"),
        "vitals": dict((dossier.get("facts") or {}).get("vitals") or {}),
        "dossier": dossier,
        # A rolling tail of what was said, so a session that expires can be
        # rebuilt with the conversation intact rather than starting over.
        "tail": [],
        "session": 1,
    }

    def remember(role: str, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        tail = scope["tail"]
        if tail and tail[-1]["role"] == role:
            tail[-1]["text"] = (tail[-1]["text"] + " " + t)[-1200:]
        else:
            tail.append({"role": role, "text": t[:1200]})
        del tail[:-10]

    async def run_tools(calls: List[Dict[str, Any]]) -> None:
        """Execute what the agent asked for and answer every call it made.

        UI tools are not run here — they are forwarded to the cockpit, which
        drives the real shell. The model is told the command was dispatched so it
        can narrate ("pulling that up now") without waiting on the browser."""
        responses: List[Dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        for call in calls:
            name = call.get("name") or ""
            # Off the event loop: tools are synchronous and some are slow (a
            # kernel workload, a training launch), and the ones that wrap async
            # endpoints have to be able to schedule back onto THIS loop and wait.
            # Running them inline would block the audio relay and deadlock those.
            out = await loop.run_in_executor(
                None, _tools.invoke, name, call.get("args") or {})
            ui = out.get("ui")
            if ui:
                await ws.send_json({"type": "ui", "action": ui.get("action"),
                                    "args": ui.get("args") or {}})
                # the tier watcher has to follow the agent, or it keeps reporting
                # changes on the bed the call opened on rather than the one being
                # discussed now
                if name == "select_patient" and (ui.get("args") or {}).get("bed"):
                    scope["bed"] = ui["args"]["bed"]
                    scope["tier"] = None
            # mirror every call into the cockpit's activity feed: an agent whose
            # actions are invisible is an agent nobody can check
            await ws.send_json({
                "type": "tool", "tool": name, "args": call.get("args") or {},
                "ok": bool(out.get("ok")), "kind": (_tools.get(name).kind
                                                    if _tools.get(name) else "unknown"),
                "error": out.get("error"),
                "verdict": _gate_verdict(out)})
            spec = _tools.get(name)
            responses.append({"id": call.get("id"), "name": name,
                              "response": _tool_summarize(
                                  out, (spec.budget if spec and spec.budget else 3500))})
        if responses:
            await session_live.send_tool_response(responses)

    async def pump_up() -> None:
        """Browser -> Gemini. Forwards chunk-for-chunk; never batches (pacing is
        what the model's turn detection runs on)."""
        from backend.llm.live import pcm_duration_s
        try:
            while not stop.is_set():
                msg = await ws.receive_json()
                kind = msg.get("type")
                if kind == "audio" and msg.get("data"):
                    budget["audio_s"] += pcm_duration_s(msg["data"])
                    if budget["audio_s"] > _CALL_MAX_AUDIO_S:
                        await ws.send_json({"type": "error", "fatal": True,
                                            "message": "call length limit reached"})
                        break
                    await session_live.send_audio(msg["data"])
                elif kind == "video" and msg.get("data"):
                    await session_live.send_video(msg["data"],
                                                  msg.get("mime", "image/jpeg"))
                elif kind == "text" and msg.get("text"):
                    await session_live.send_text(str(msg["text"])[:2000])
                elif kind == "end_audio":
                    await session_live.end_audio()
                elif kind == "bye":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:                          # pragma: no cover
            print(f"[doctor] uplink ended: {type(exc).__name__}: {exc}")
        finally:
            stop.set()

    async def reconnect() -> bool:
        """Rebuild the realtime session, carrying the conversation across.

        Gemini Live sessions expire (~15 min) and announce it with `goAway`. The
        old behaviour was a banner and then silence mid-sentence, which in front
        of anyone reads as broken. Instead: open a fresh session on the CURRENT
        bed's dossier, replay the last few turns as context, and keep going. The
        clinician sees a short reconnecting blip, not the end of the consult."""
        nonlocal session_live
        old = session_live
        try:
            fresh = _doctor_dossier(scope["bed"])
            scope["dossier"] = fresh
            tail = scope["tail"]
            context = ""
            if tail:
                lines = "\n".join(
                    f"{'You' if t['role'] == 'doctor' else 'The clinician'}: {t['text']}"
                    for t in tail)
                context = (
                    "\n\n=== CONVERSATION SO FAR ===\nThis consult is already in "
                    "progress; the realtime session was renewed underneath it. Do "
                    "NOT greet them again or re-introduce yourself. Pick up exactly "
                    "where this left off:\n" + lines + "\n=== END ===")
            new = _LiveConsult(_doctor.system_instruction(fresh) + context,
                               tools=_tools.declarations())
            await new.connect()
            session_live = new
            scope["session"] += 1
        except Exception as exc:
            print(f"[doctor] reconnect failed: {type(exc).__name__}: {exc}")
            return False
        finally:
            try:
                await old.close()
            except Exception:
                pass
        return True

    async def pump_down() -> None:
        """Gemini -> browser, with tool calls executed and expiry healed in between."""
        try:
            while not stop.is_set():
                renewed = False
                async for ev in session_live.events():
                    if stop.is_set():
                        break
                    kind = ev.get("type")
                    if kind == "tool_call":
                        await run_tools(ev.get("calls") or [])
                        continue                   # the raw call is not the UI's business
                    if kind == "tool_cancel":
                        continue
                    if kind == "transcript":
                        remember("doctor" if ev.get("role") == "doctor" else "clinician",
                                 ev.get("text", ""))
                    if kind == "going_away":
                        # Renew BEFORE it dies, so the seam falls in a gap rather
                        # than in the middle of a sentence.
                        await ws.send_json({"type": "renewing"})
                        renewed = await reconnect()
                        if renewed:
                            await ws.send_json({"type": "renewed",
                                                "session": scope["session"]})
                        break
                    await ws.send_json(ev)
                    if kind == "closed":
                        break
                if not renewed:
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:                          # pragma: no cover
            print(f"[doctor] downlink ended: {type(exc).__name__}: {exc}")
        finally:
            stop.set()

    async def watch_case() -> None:
        """A call outlives the state it opened on, and the dossier does not move.

        The system instruction is fixed at connect, so ten minutes into a consult
        the doctor is still holding the vitals it was handed at the start. Tier
        changes were caught; a systolic sliding from 104 to 84 was not, and a
        system whose whole argument is honesty about what it knows should not
        state a twenty-minute-old number in a live conversation.

        So watch the numbers too, and only speak up when a move is material —
        a note on every tick would be noise the doctor learns to ignore."""
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=20.0)
                return
            except asyncio.TimeoutError:
                pass
            try:
                fresh = _doctor_dossier(scope["bed"])
                facts = fresh.get("facts") or {}
                changes: List[str] = []

                tier, last = facts.get("tier"), scope["tier"]
                if tier and last and tier != last:
                    changes.append(f"tier {last} -> {tier}")
                scope["tier"] = tier

                now_v = facts.get("vitals") or {}
                was_v = scope["vitals"] or {}
                for key, label, unit, delta in _WATCHED_VITALS:
                    a, b = was_v.get(key), now_v.get(key)
                    if a is None or b is None:
                        continue
                    if abs(float(b) - float(a)) >= delta:
                        changes.append(
                            f"{label} {float(a):.0f} -> {float(b):.0f} {unit}".strip())
                        was_v[key] = b
                scope["vitals"] = {**was_v, **{k: v for k, v in now_v.items()
                                               if k not in was_v}}

                if changes:
                    await session_live.note(
                        f"{fresh.get('headline')} — " + "; ".join(changes)
                        + ". These supersede the numbers in your dossier; use "
                          "these, and say they have moved since you last looked.")
                    await ws.send_json({"type": "case_update", "facts": facts,
                                        "headline": fresh.get("headline"),
                                        "changes": changes, "to_tier": tier})
            except Exception:
                return

    try:
        await asyncio.gather(pump_up(), pump_down(), watch_case(),
                             return_exceptions=True)
    finally:
        stop.set()
        if session_live is not None:
            await session_live.close()
        try:
            await ws.close()
        except Exception:
            pass


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


# --------------------------------------------------------------------------- #
# Scenario Lab — the AI's sandbox for posing clinical situations
#
# These are ordinary /aegis routes, so the agent's generic os_read / os_write
# primitives reach them without any special-casing; the named tools in agent.py
# exist for discoverability, not access. Everything here is SYNTHETIC and stays in
# memory: no scenario touches the live cohort, Supabase, or any reported metric.
# --------------------------------------------------------------------------- #
class ScenarioIn(BaseModel):
    name: str
    arc: str = "septic_shock"
    los_hours: int = 48
    onset_hour: Optional[int] = 30
    ramp_hours: int = 12
    severity: float = 1.0
    age: float = 68.0
    gender: str = "M"
    weight: float = 78.0
    lab_interval_hours: int = 8
    vitals_dropout: float = 0.04
    #: [[hour, kind], ...] with kind in fluids | antibiotics | vasopressors.
    interventions: List[List[Any]] = []
    seed: int = 7
    notes: str = ""
    #: [{"kind": ..., "value": ..., "because": ...}, ...]
    expectations: List[Dict[str, Any]] = []


@app.get("/aegis/scenarios/vocabulary")
def scenarios_vocabulary() -> Dict[str, Any]:
    """Everything needed to author a scenario: the deterioration arcs, the gradeable
    expectations, the interventions, and the limits. Read this before creating one."""
    from backend.aegis import scenarios as sc
    return {
        "arcs": sc.ARCS,
        "expectations": sc.EXPECTATION_KINDS,
        "interventions": {"fluids": "raises MAP/SBP, lowers HR",
                          "antibiotics": "4h lag, then bends the whole arc back",
                          "vasopressors": "raises MAP/SBP, raises HR"},
        "limits": {"max_scenarios": sc.ScenarioLab.MAX_SCENARIOS,
                   "los_hours": [8, 336], "label_lead_hours": sc.LABEL_LEAD},
        "synthetic": True,
        "caveat": "Scenarios grade BEHAVIOUR (did the alarm fire, in what order). "
                  "They can never measure accuracy — the patients are written by "
                  "hand, so any rate computed on them measures the generator.",
    }


@app.get("/aegis/scenarios")
def scenarios_list() -> Dict[str, Any]:
    """Every scenario in the sandbox, plus the pass/fail report for those that ran."""
    from backend.aegis.scenarios import LAB
    return {"scenarios": LAB.list(), "report": LAB.report()}


@app.get("/aegis/scenarios/{bed}")
def scenarios_get(bed: str) -> Dict[str, Any]:
    """One scenario's full result, including the hour-by-hour risk trace."""
    from dataclasses import asdict
    from backend.aegis.scenarios import LAB
    res = LAB.result(bed)
    if res is None:
        return {"bed": bed, "ran": False,
                "note": "not run yet — POST /aegis/scenarios/{bed}/run"}
    return asdict(res)


@app.post("/aegis/scenarios/create")
async def scenarios_create(payload: ScenarioIn) -> Dict[str, Any]:
    """Author a scenario. Returns its sandbox bed handle (always 'sim_'-prefixed)."""
    from backend.aegis.scenarios import LAB, Expectation, ScenarioSpec
    try:
        spec = ScenarioSpec(
            name=payload.name, arc=payload.arc, los_hours=payload.los_hours,
            onset_hour=payload.onset_hour, ramp_hours=payload.ramp_hours,
            severity=payload.severity, age=payload.age, gender=payload.gender,
            weight=payload.weight, lab_interval_hours=payload.lab_interval_hours,
            vitals_dropout=payload.vitals_dropout,
            interventions=[(int(h), str(k)) for h, k in payload.interventions],
            seed=payload.seed, notes=payload.notes)
        exps = [Expectation(kind=e.get("kind", ""), value=float(e.get("value", 0)),
                            because=str(e.get("because", "")))
                for e in payload.expectations]
        bed = LAB.create(spec, exps)
    except (ValueError, TypeError) as exc:
        # Returned rather than raised: the agent gets a correctable message instead
        # of a stack trace, and a rejected spec is a normal outcome here.
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "bed": bed, "fingerprint": spec.fingerprint(),
            "synthetic": True}


@app.post("/aegis/scenarios/{bed}/run")
async def scenarios_run(bed: str) -> Dict[str, Any]:
    """Run one scenario against the SHIPPED detector and grade its expectations."""
    from dataclasses import asdict
    from backend.aegis.scenarios import LAB
    try:
        return asdict(LAB.run(bed))
    except KeyError as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/aegis/scenarios/suite")
async def scenarios_suite() -> Dict[str, Any]:
    """Install and run the standing suite: deteriorations the OS must catch and
    confounders it must ignore. Replaces whatever is in the sandbox."""
    from backend.aegis.scenarios import LAB
    LAB.load_suite()
    LAB.run_all()
    return LAB.report()


@app.post("/aegis/scenarios/run_all")
async def scenarios_run_all() -> Dict[str, Any]:
    """Run every scenario currently in the sandbox."""
    from backend.aegis.scenarios import LAB
    LAB.run_all()
    return LAB.report()


@app.post("/aegis/scenarios/reset")
async def scenarios_reset() -> Dict[str, Any]:
    """Empty the sandbox."""
    from backend.aegis.scenarios import LAB
    return {"ok": True, "cleared": LAB.reset()}


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


# Bind the agent's tools LAST: the handlers close over endpoint functions defined
# throughout this module, and the read primitive indexes the finished route table.
_bind_agent_tools()
_index_read_routes()
print(f"[aegis] agent plane: {len(_tools.bound())} tools bound"
      + (f", UNBOUND: {_tools.unbound()}" if _tools.unbound() else ""))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("backend.aegis.server:app", host="127.0.0.1", port=8100, reload=False)
