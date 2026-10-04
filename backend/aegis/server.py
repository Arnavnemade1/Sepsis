"""AegisOS API server — the composition root of the REST + WebSocket surface.

This module owns the running system: the kernel bring-up, the live session, the
ICU cohort cluster, the scheduler, the gate, users, the filesystem and the
background drive loops. It holds no routes of its own. Every endpoint lives in a
domain router under ``backend/api/routes/`` (live, health, patients, causal,
pharmacy, governance, system, kernel, devices, research, ai), and request bodies
live in ``backend/api/schemas.py``. The routers are included near the bottom of
this file, after every piece of state they import exists and before the route
table is frozen for gate coverage.

Run:  uvicorn backend.aegis.server:app      (or: python -m backend.aegis.server)
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

from backend.api.schemas import *  # noqa: F401,F403 — request bodies

from backend.aegis.stream import AegisSession
from backend.aegis.kernel import ALERT, BioCluster, NeuroSymbolicRuntime, SepsisBundleEngine
from backend.aegis.cohort import CohortCluster
from backend.aegis.ward_risk import INDEX_HIGH as _INDEX_HIGH, INDEX_MEDIUM as _INDEX_MEDIUM
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
# The kernel gate: one reference monitor for every /aegis call (kernel/gate.py).
# Added BEFORE CORS so CORS stays outermost — a 403 from the gate must still carry
# the Access-Control headers, or the browser reports it as a CORS failure and the
# page never sees the EPERM. It passes everything through until the kernel
# configures it later in this module.
from backend.aegis.kernel.gate import GateMiddleware as _GateMiddleware
app.add_middleware(_GateMiddleware)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Kernel bring-up: the journal, the process table, the boot record ------- #
# These come first so everything below can be recorded AS IT HAPPENS. The journal
# is the one ordered record (replacing nineteen disconnected logs), PROCS is the
# one process table, and BOOT times the phases this module runs at import — which
# is what the boot screen will show instead of eight hardcoded lines on timers.
from backend.aegis.kernel.boot import BOOT
from backend.aegis.kernel.journal import BOOT_ID, KJOURNAL, JournalStore
from backend.aegis.kernel.proc import (PROCS, BLOCKED, DRIVER, KTHREAD, MONITOR,
                                       RUNNING, SERVICE, SLEEPING,
                                       SIGCONT, SIGHUP, SIGSTOP, SIGTERM, SIGUSR1,
                                       NoSuchProcess, SignalFailed, SignalUnsupported)

try:
    # SQLite DDL is local and fast; create_client opens no connection. Persistence
    # failing must never stop the kernel — it degrades to the in-memory ring.
    KJOURNAL.store = JournalStore().start()
    _jhead = KJOURNAL.rehydrate()
    print("[aegis] journal: %s (%s)" % (
        ("resumed at seq %d" % _jhead["seq"]) if _jhead else "new chain",
        "supabase" if KJOURNAL.store.use_supabase else "sqlite"))
except Exception as _exc:                          # pragma: no cover
    print(f"[aegis] journal persistence unavailable ({_exc}); in-memory only")
KJOURNAL.append("kern", "boot.start", pid=0, boot_id=BOOT_ID, version=app.version)
if KJOURNAL.store is not None:
    # journald is a real thread; it is listed, and it honours no signal (stopping
    # the record while the kernel runs is not an operation this kernel offers).
    PROCS.spawn("journald", KTHREAD, uid="kernel", state=SLEEPING,
                sink="batched SQLite / Supabase writer", thread="journald")

BOOT.checkpoint("session", "load the calibrated detector and the live patient twin")


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
BOOT.note(detector_loaded=session.core.model is not None, patient=session.patient_id)
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
BOOT.checkpoint("wing", "mount the ICU wing on /dev/bio")
try:
    # The ward is scored by the same detector the live session serves, in the same
    # object: one copy of the stack in the process (backend/aegis/ward_risk.py).
    from backend.aegis.ward_risk import ward_detector as _ward_detector
    cluster = CohortCluster(n=12, lane_budget=160, seed=7, ebpf_optical=3,
                            ward=_ward_detector(stack_model=session.core.model))
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
BOOT.note(beds=len(getattr(cluster, "node_ids", [])), mode=DATA_MODE,
          ebpf_drivers=len(getattr(cluster, "ebpf_beds", {})))
BOOT.checkpoint("cage", "egress cages, syscall table, orchestrator")
# Every scheduling decision is proven. This was throttled to one tick in 30 when a
# verify built ~4 Z3 terms per bed through the Python API (3.6 ms at 12 beds, 107 ms
# at 400); as one SMT-LIB2 script it is ~0.25 ms at 12 beds, so proving every tick
# costs about what proving every 30th used to — and "the last proof holds" is gone.
import backend.aegis.kernel.scheduler as _sched_mod
_sched_mod.VERIFY_EVERY = 1
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


BOOT.checkpoint("kthreads", "warm-up and keep-alive kernel threads")
import threading as _threading


def _kthread(name: str, fn, delay: float = 0.0, measure: bool = True, **detail):
    """Run ``fn`` as a listed kernel thread.

    The startup delay is spent SLEEPING, outside the measured slice — the warmups
    used to sleep inside their own bodies, which would have booked two and three
    seconds of idle time as work. ``measure=False`` is for loops that mostly sleep
    (keep-alive): wall time around a sleeping loop is not a cost."""
    def run() -> None:
        import time as _t
        pid = PROCS.spawn(name, KTHREAD, uid="kernel",
                          state=SLEEPING if delay else RUNNING, thread=name, **detail).pid
        if delay:
            _t.sleep(delay)
        code = 0
        try:
            if measure:
                with PROCS.slice(pid):
                    fn()
            else:
                PROCS.set_state(pid, SLEEPING)
                fn()
        except BaseException:
            code = 1
            raise
        finally:
            PROCS.exit(pid, code=code)
    return run


_threading.Thread(target=_kthread("orch-warmup", _warm_orchestrator, delay=2.0,
                                  purpose="pay the first synthesis cost off the request path"),
                  name="orch-warmup", daemon=True).start()
_threading.Thread(target=_kthread("pk-warmup", _warm_pk, delay=3.0,
                                  purpose="cache the reference drug projections"),
                  name="pk-warmup", daemon=True).start()


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


_threading.Thread(target=_kthread("keep-alive", _keep_alive, measure=False,
                                  purpose="free-tier anti-sleep self-ping"),
                  name="keep-alive", daemon=True).start()
BOOT.checkpoint("devices", "eBPF verifier service, PPG morphology monitor")
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

BOOT.checkpoint("aegisd", "service manager, /proc namespace, package registry")
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


# The REAL process table and scheduler view. These used to return the HDKernel
# demo workload, which the read itself created — a /proc that fabricated its own
# contents. The demo still runs on demand via POST /aegis/hdkernel/run.
procfs.register("/proc/kthreads", lambda: PROCS.ps())
procfs.register("/proc/ksched", lambda: PROCS.top())
procfs.register("/proc/journal", lambda: KJOURNAL.status())
procfs.register("/proc/boot", lambda: BOOT.summary())
procfs.register("/proc/executor", lambda: EXEC.status())
# /proc/<pid> for every live process, resolved on read (no stale providers).
procfs.register_dynamic(r"/proc/(\d+)", lambda m: PROCS.proc(int(m.group(1))),
                        lambda: ["/proc/%d" % p["pid"] for p in PROCS.ps()])
BOOT.checkpoint("llm", "language-model provider, model control plane, patient terminals")
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

# --- The paging OS: nurse + detector alerts, triaged, routed, certified, escalated --- #
from backend.aegis.paging import PagingOS as _PagingOS
PAGING = _PagingOS(llm=_llm_client)
_paging_clients: List[WebSocket] = []
# A sepsis page is the start of the Hour-1 bundle; the sentinel watches whether a
# certified antibiotic order (the treat syscall) followed. None = not a ward bed.
PAGING.bundle_started = lambda bed: (bool(getattr(syscalls.interventions.get(bed), "treated", False))
                                     if bed else None)
# The brief a clinician reads is grounded in the same dossier the consult uses.
PAGING.dossier_text = lambda bed: (_doctor_dossier(bed) or {}).get("text")
# Every room is watched as a unit (backend/aegis/rooms). The ICU rooms read their real replayed
# stay up to the current hour, an imported external history if a clinician linked one, the live
# session's history, and the pharmacopoeia's interaction screen. Resolved at call time.
PAGING.units.icu_patient = lambda bed: getattr(cluster, "_by", {}).get(bed)
PAGING.units.imported_history = lambda bed: patient_terminal.get_history(bed)
PAGING.units.live_history = lambda: session.core.clinical_history
PAGING.units.screen = lambda drugs, ctx: pharmacopoeia.screen(drugs, ctx)

# --- The integration plane: the hospital's HL7 feeds into the rooms (backend/aegis/interop) --- #
# Each sender's local codes are mapped onto AegisOS concepts. A mapping is proposed (LOINC,
# IEEE 11073, lexical, the language model) and adopted only on evidence from the data; an
# adopted feed keeps being checked, and one that drifts is quarantined before a wrong value
# reaches a detector. The demo hospital's systems are simulated and speak only HL7 to it.
BOOT.checkpoint("interop", "integration plane: HL7 feeds, mappings earned on evidence")
from backend.aegis.interop.engine import Integration as _Integration
from backend.aegis.interop.sender import SimulatedHospital as _SimulatedHospital
from backend.aegis.paging.hospital import ROOMS as _HOSPITAL_ROOMS
INTEROP = _Integration(rooms=PAGING, llm=_llm_client, hospital_rooms=list(_HOSPITAL_ROOMS))
PAGING.units.feed_status = INTEROP.room_feed
PAGING.sentinel.external = INTEROP.sentinel_recs
INTEROP_SIM = _SimulatedHospital(INTEROP, lambda: [r for r in _HOSPITAL_ROOMS if PAGING.interface_room_free(r)])
INTEROP_SIM.register()
procfs.register("/proc/interop", lambda: {k: v for k, v in INTEROP.overview().items()
                                          if k in ("interfaces", "mappings", "counters", "census", "model",
                                                   "settings", "reference")})
BOOT.note(interfaces=len(INTEROP.interfaces), vocabulary=len(INTEROP.vocabulary()["concepts"]))

# --- The standards surface: FHIR R4 reads, CDS Hooks, outbound webhooks ------- #
# An EHR reads what AegisOS sees in FHIR, gets its findings as CDS Hooks cards at the
# moment a chart is opened or an order signed, and the hospital's messaging receives
# its events as signed webhooks. All three read the same live state as the console.
from backend.aegis.interop.cdshooks import CDSHooks as _CDSHooks
from backend.aegis.interop.fhir import FHIRFacade as _FHIRFacade, room_of as _fhir_room_of
from backend.aegis.interop.webhooks import Webhooks as _Webhooks
FHIR = _FHIRFacade(room_unit=PAGING.room_unit, board=PAGING.rooms_board,
                   pages=lambda: list(PAGING.snapshot()["pages"]))


def _cds_room(pid: str) -> Optional[str]:
    r = _fhir_room_of(pid)
    if r:
        return r
    for row in PAGING.rooms_board().get("rooms") or []:
        if row.get("mrn") == pid:
            return row["room"]
    return None


def _cds_journal(event: str, **kw: Any) -> None:
    from backend.aegis.kernel.journal import INFO as _INFO
    KJOURNAL.append("interop", event, severity=_INFO, uid=str(kw.pop("by", "cds")), **kw)


CDS = _CDSHooks(room_for=_cds_room, room_unit=PAGING.room_unit, feed=INTEROP.room_feed,
                screen=lambda drugs, ctx: pharmacopoeia.screen(drugs, ctx) if "pharmacopoeia" in globals() else [],
                app_url=_os.getenv("AEGIS_APP_URL", "https://aegis-clinical-beryl.vercel.app"), journal=_cds_journal)
WEBHOOKS = _Webhooks()
INTEROP.listeners.append(WEBHOOKS.from_interop)
procfs.register("/proc/webhooks", lambda: {k: v for k, v in WEBHOOKS.status().items()
                                           if k in ("subscriptions", "queued", "counters")})


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
BOOT.checkpoint("services", "start aegisd services in dependency order")
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
# The wing has exactly ONE writer thread from here on. Everything that mutates
# `cluster` (the loop's tick, chaos injection, driver fault/recover, the
# orchestrator's scenario) goes through it — see kernel/executor.py for why and
# for the executor -> Z3 lock order. Started before the warm-up so even the warm
# ticks run as the writer.
BOOT.checkpoint("executor", "start the single writer and warm the rings")
from backend.aegis.kernel.executor import EXEC

EXEC.start()
if hasattr(cluster, "fabric"):
    EXEC.own(cluster.fabric.arena)                # the live wing, and only it
EXEC.call(cluster.run, 12)                        # warm the rings before serving

# --- What is actually running, in the one process table --------------------- #
# Before this, `ps` returned an HDKernel demo workload that the read itself
# created, while the twelve beds, the supervised drivers, the services and the
# loops — the things genuinely executing — appeared nowhere.
_KPROC: Dict[str, int] = {}


def _bed_slice(nid: str):
    """Per-bed accounting hook handed to the cluster (see cohort.tick)."""
    pid = _KPROC.get("bed:" + nid)
    return PROCS.slice(pid) if pid is not None else _nullctx()


def _nullctx():
    # Local import: this module's aliasing convention (os as _os, threading as
    # _threading) makes a module-level contextlib import easy to get wrong, and a
    # NameError here would break boot rather than one bed's accounting.
    from contextlib import nullcontext
    return nullcontext()


def _driver_signal(bid: str):
    def handle(sig: str):
        if sig == SIGHUP:                            # re-verify the bytecode and re-arm
            return EXEC.call(cluster.recover_driver, bid)
        if sig == SIGUSR1:                           # the operator's fault injection
            return EXEC.call(cluster.inject_driver_fault, bid, "SIGUSR1 from operator")
        raise ValueError(sig)
    return handle


def _service_signal(name: str):
    def handle(sig: str):
        if sig == SIGTERM:
            return aegisd.stop(name)
        if sig == SIGHUP:
            return aegisd.restart(name)
        raise ValueError(sig)
    return handle


def _register_processes() -> None:
    # Each bed's monitoring pipeline: sample -> fuse -> learn -> schedule ->
    # resonance. Its CPU time is measured around that body, not estimated. It
    # honours no signal: a patient's monitoring is not paused from `kill`.
    for nid in getattr(cluster, "node_ids", []):
        p = PROCS.spawn(cluster.names.get(nid, nid) if hasattr(cluster, "names") else nid,
                        MONITOR, uid="kernel", bed=nid,
                        sched_class="LANE (cognitive lane governor)")
        _KPROC["bed:" + nid] = p.pid
    cluster.proc_slice = _bed_slice

    # Supervised verified drivers: SIGHUP re-verifies and re-arms, SIGUSR1 is the
    # operator fault injection the console already exposes.
    for bid in getattr(cluster, "ebpf_beds", {}):
        p = PROCS.spawn("%s optical (eBPF)" % cluster.names.get(bid, bid), DRIVER,
                        uid="kernel", bed=bid, on_signal=_driver_signal(bid),
                        signals=(SIGHUP, SIGUSR1),
                        verified="Aegis-eBPF static verifier")
        _KPROC["drv:" + bid] = p.pid

    # The single writer thread for the wing. Listed, and protected: it honours no
    # signal, because stopping it stops every bed's monitoring at once.
    PROCS.spawn("kexec", KTHREAD, uid="kernel", state=SLEEPING, thread="aegis-kexec",
                role="single writer for every mutation of the ICU wing")
    _reconcile_kernel()                              # services + drivers, first pass


def _reconcile_kernel() -> None:
    """Keep the process table true to what the subsystems themselves report.

    A service can be stopped, restarted, faulted into quarantine or uninstalled
    through aegisd's routes, the package manager or the agent plane — not only
    through a signal. Trusting every one of those paths to remember the process
    table is how a table goes stale, so it is reconciled every tick instead."""
    services = getattr(aegisd, "services", {})
    for name, svc in list(services.items()):
        live = PROCS.find(kind=SERVICE, svc=name)
        if svc.state == "RUNNING" and live is None:
            PROCS.spawn(name, SERVICE, uid="kernel", svc=name, bed=SERVICE_BED,
                        on_signal=_service_signal(name), signals=(SIGTERM, SIGHUP))
        elif svc.state != "RUNNING" and live is not None:
            PROCS.exit(live.pid, code=0 if svc.state == "STOPPED" else 1)
    for p in PROCS.ps(kind=SERVICE):                  # uninstalled packages
        if p["state"] not in ("ZOMBIE", "EXITED") and p["detail"].get("svc") not in services:
            PROCS.exit(p["pid"], code=0)
    if hasattr(cluster, "driver_plane_status"):
        for d in cluster.driver_plane_status():
            pid = _KPROC.get("drv:" + d["key"])
            if pid is not None:
                PROCS.set_state(pid, BLOCKED if d.get("state") == "DOWN" else RUNNING,
                                driver_state=d.get("state"), active=d.get("active"),
                                faults=d.get("faults"), failovers=d.get("failovers"))
    PROCS.reap()


BOOT.checkpoint("proc", "register the running subsystems in the process table")
try:
    _register_processes()
    print("[aegis] process table: %d processes registered" % len(PROCS.procs))
except Exception as _exc:                          # pragma: no cover
    print(f"[aegis] process registration failed: {_exc}")
BOOT.note(**{k: v for k, v in PROCS.status()["by_kind"].items() if v})

# The session's AegisOS object reports THIS kernel — not idle copies of its own.
session.os.attach_kernel(cluster=cluster, procs=PROCS, journal=KJOURNAL,
                         boot=BOOT, executor=EXEC)
BOOT.checkpoint("routes", "persistence hooks and the REST / WebSocket surface")

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


async def _fanout(clients: List[Any], payload: Dict[str, Any]) -> None:
    """Send one payload to every socket in ``clients``, encoded ONCE (the bytes
    ``send_json`` would have produced per socket), dropping sockets that fail."""
    if not clients:
        return
    msg = _osstream_encode(payload)
    dead = []
    for ws in list(clients):
        try:
            await _osstream_send(ws, msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in clients:
            clients.remove(ws)


async def _broadcast(snapshot: Dict[str, Any]) -> None:
    OS_HUB.publish("snapshot", snapshot)
    await _fanout(_clients, snapshot)
    # simulator ticks and real ingest both pass through here; the pager grades the
    # live patient (room 402) off the loop, since opening a case certifies with Z3
    asyncio.get_running_loop().run_in_executor(None, PAGING.observe_live, snapshot)


def _apply_treatment_from(snapshot: Dict[str, Any]) -> None:
    """Close the loop: feed a verified+applied protocol back to the simulator."""
    global _active_treatment
    proto = snapshot.get("protocol")
    if proto and proto.get("applied"):
        _active_treatment = dict(proto["protocol"])


_DRIVE_PAUSED = {"paused": False}


def _drive_signal(sig: str) -> str:
    # SIGSTOP/SIGCONT really pause the DEMO simulator: the loop checks this flag
    # each tick. (The ward loop has no equivalent — see _cluster_loop.)
    if sig == SIGSTOP:
        _DRIVE_PAUSED["paused"] = True
        return "demo simulator paused"
    if sig == SIGCONT:
        _DRIVE_PAUSED["paused"] = False
        return "demo simulator resumed"
    raise ValueError(sig)


async def _drive_loop() -> None:
    print("[aegis] simulator drive loop started (demo mode; yields to real data).")
    _pid = PROCS.spawn("drive-loop", KTHREAD, uid="kernel",
                       cadence_s=TICK_SECONDS, drives="live session snapshot (demo)",
                       on_signal=_drive_signal, signals=(SIGSTOP, SIGCONT)).pid
    while True:
        try:
            # The simulator only drives the stream as a DEMO — the moment real
            # telemetry arrives it stands down so the camera owns the live feed.
            if DRIVE_FROM_SIMULATOR and not _live_mode() and not _DRIVE_PAUSED["paused"]:
              with PROCS.slice(_pid):
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


async def _paging_loop() -> None:
    """Once a second: the queue's clock (timeouts, escalation, holds, review
    fail-safes, simulated staff, demo traffic, the sentinel), then one batch of
    events to every paging socket."""
    _pid = PROCS.spawn("paging", KTHREAD, uid="kernel", cadence_s=1.0,
                       drives="page queue: timeouts, escalation, sentinel").pid
    def tick() -> None:
        with PROCS.slice(_pid):
            PAGING.tick()
    while True:
        try:
            await asyncio.to_thread(tick)
            events = PAGING.drain()
            if events:
                WEBHOOKS.from_paging(events)
                await _fanout(_paging_clients, {"type": "batch", "events": events})
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] paging loop error: {exc}")
        await asyncio.sleep(1.0)


async def _interop_loop() -> None:
    """Once a second: the simulated hospital's feeds, interface health, and any codes
    waiting for the language-model proposer (which runs on its own thread)."""
    _pid = PROCS.spawn("interop", KTHREAD, uid="kernel", cadence_s=1.0,
                       drives="integration plane: HL7 feeds, live evidence, model proposer").pid
    def tick() -> None:
        with PROCS.slice(_pid):
            INTEROP_SIM.tick()
            INTEROP.tick()
    while True:
        try:
            await asyncio.to_thread(tick)
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] interop loop error: {exc}")
        await asyncio.sleep(1.0)


async def _webhook_loop() -> None:
    """Deliver due webhooks once a second, off the event loop. A slow receiver delays
    only this loop, never a feed or a page."""
    _pid = PROCS.spawn("webhooks", KTHREAD, uid="kernel", cadence_s=1.0,
                       drives="signed outbound events: retries, dead letters").pid
    def pump() -> None:
        with PROCS.slice(_pid):
            WEBHOOKS.pump()
    while True:
        try:
            await asyncio.to_thread(pump)
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] webhook loop error: {exc}")
        await asyncio.sleep(1.0)


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
    asyncio.create_task(_paging_loop())
    asyncio.create_task(_interop_loop())
    asyncio.create_task(_webhook_loop())
    asyncio.create_task(OS_HUB.run())
    if _os.getenv("AEGIS_MLLP_PORT"):
        # on-premises: the hospital's interface engine connects over MLLP (off on a hosted deploy)
        from backend.aegis.interop import mllp as _mllp
        asyncio.create_task(_mllp.start(int(_os.getenv("AEGIS_MLLP_PORT")), INTEROP.ingest_hl7,
                                        principal="svc.hl7"))

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
    _threading.Thread(target=_kthread("persist-init", _init_persistence,
                                      purpose="connect the telemetry database"),
                      name="persist-init", daemon=True).start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    # A clean stop leaves the record complete: the journal's queue is flushed
    # rather than lost with the process.
    KJOURNAL.append("kern", "shutdown", boot_id=BOOT_ID,
                    uptime_s=round(PROCS.status()["uptime_s"], 1))
    if KJOURNAL.store is not None:
        await asyncio.to_thread(KJOURNAL.store.stop)


# --- REST ------------------------------------------------------------------- #


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


# --- Temporal Causal Graph (the patient's clinical story) ------------------- #


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
    # An imported/linked chart shifts the alerting PRIOR for a high-risk host
    # (immunosuppression, CKD, …). The effective risk carries that shift and is the
    # number the ward board shows for this bed.
    personalized = round(_effective_risk(bed), 4) if prof else 0.0
    # The ward is scored by the shipped detector (ward_risk): its calibrated
    # probability is what every field named "probability" carries, for this bed and
    # the live session alike, and the tier is read off the index, whose lines ARE the
    # detector's operating point. Without a detector, the old fusion semantics.
    prob = cluster.effective_probability(bed) if hasattr(cluster, "effective_probability") else None
    prob_pop = getattr(prof, "probability", None) if prof else None
    _hobj = patient_terminal.get_history(bed)
    _susc = (_hobj or {}).get("susceptibility") or {}
    level = ("HIGH" if personalized >= _INDEX_HIGH else "MEDIUM" if personalized >= _INDEX_MEDIUM
             else "LOW") if prob is not None else (
             "HIGH" if personalized >= 0.6 else "MEDIUM" if personalized >= 0.35 else "LOW")
    wbc = float(hist.get("wbc", 7.5)); gcs = int(hist.get("gcs", 15))
    sirs = (int(v["skin_temp"] > 38 or v["skin_temp"] < 36) + int(v["heart_rate"] > 90)
            + int(v["respiratory_rate"] > 20) + int(wbc > 12 or wbc < 4))
    qsofa = int(v["respiratory_rate"] >= 22) + int(v["systolic_bp"] <= 100) + int(gcs < 15)
    p_det = round(prob, 5) if prob is not None else round(fused, 4)
    det = {"ml_probability": p_det, "sepsis_probability": p_det,
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
        "risk": ({"level": level, "ml_probability": round(prob, 5), "index": personalized,
                  "ns_risk_level": level, "source": "detector", "model": _ward_model_code(),
                  "population_risk": round(prob_pop, 5) if prob_pop is not None else None,
                  "personalized_risk": round(prob, 5),
                  "susceptibility": _susc or None,
                  "sirs_score": sirs, "qsofa_score": qsofa, "signal_quality": 1.0,
                  # The bed is now scored by the very fit the conformal bands were cut
                  # on, so the claim is available - unless an imported chart has
                  # shifted the score, which the bands do not describe.
                  "certainty": cluster.certainty(bed) or {
                      "unavailable": True, "state": "no_claim",
                      "reason": ("an imported chart shifted this bed's prior; the conformal "
                                 "bands describe the detector's own output"
                                 if cluster._history_uplift(bed) else
                                 "no conformal bands for the served detector")}}
                 if prob is not None else
                 {"level": level, "ml_probability": round(personalized, 4), "ns_risk_level": level,
                  "population_risk": round(fused, 4), "personalized_risk": round(personalized, 4),
                  "susceptibility": _susc or None,
                  "sirs_score": sirs, "qsofa_score": qsofa, "signal_quality": 1.0,
                  # A fused cohort score is read against different bands than the
                  # detector's, so no conformal claim covers it. Said out loud rather
                  # than omitted: a missing field reads as "resolved".
                  "certainty": {"unavailable": True, "state": "no_claim",
                                "reason": "cohort bed risk is a fused cohort score, not "
                                          "the calibrated detector probability the "
                                          "conformal bands were derived from"}}),
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


# --- Living Pharmacopoeia: drug-interaction graph → dynamic Z3 (MEGA-UPGRADE L4) -- #
BOOT.checkpoint("pharmacopoeia", "load the fused drug-interaction knowledge base")
from backend.aegis.pharmacopoeia import Pharmacopoeia
pharmacopoeia = Pharmacopoeia()                   # loads the fused DDInter+curated KB
BOOT.checkpoint("routes-2", "clinical REST surface")


# --- Clinical Safety DSL: hospital-authored rules compiled to Z3 (MEGA-UPGRADE L2) - #
from backend.aegis.clinical_dsl import (compile_rules, verify_protocol as dsl_verify,
                                        check_consistency, DEFAULT_POLICY)


# --- Clinical RBAC: privilege-ring access control over the OS (MEGA-UPGRADE L3) -- #
BOOT.checkpoint("rbac", "role and capability policy")
from backend.aegis.rbac import (RBAC, GRANTS, RING, CAP_MIN_RING, SYSCALL_CAP,
                                prove_policy_sound)
rbac = RBAC()

# --- Identity and the gate --------------------------------------------------- #
# The RBAC policy used to be a demonstration beside an unguarded API (its gate
# object was constructed here and never called). It is now what the kernel gate
# enforces on every /aegis call: accounts own their roles, sessions are tracked,
# every route has a syscall-table entry, and Dr. Aegis is a user acting for a
# clinician. Ships in AEGIS_GATE_MODE=audit: refusals are counted and journalled
# but served, until the desktop sends session tokens.
BOOT.checkpoint("gate", "accounts, sessions, the syscall table and the gate")
from backend.aegis.kernel import gate as _gate_mod
from backend.aegis.kernel.abi import ABI as _ABI, live_routes as _live_routes
from backend.aegis.kernel.users import (GUEST as _GUEST, AuthError as _AuthError,
                                        PermissionDenied as _PermissionDenied,
                                        CURRENT as _CURRENT_CRED, UserDB as _UserDB,
                                        current_credential as _cred)
USERS = _UserDB(rbac)
ABI_TABLE = _ABI.load()
GATE = _gate_mod.configure(ABI_TABLE, USERS, KJOURNAL)
BOOT.note(mode=GATE.mode, syscalls=len(ABI_TABLE.entries), accounts=len(USERS.accounts),
          demo_login=USERS.demo_login)
GATE_COVERAGE: Optional[Dict[str, Any]] = None   # set once the route table is final
print(f"[aegis] gate: {GATE.mode} mode, {len(ABI_TABLE.entries)} syscalls, "
      f"{len(USERS.accounts)} accounts (demo sign-in {'on' if USERS.demo_login else 'off'})")


def _prove_policy() -> None:
    from backend.aegis.kernel.policy import prove_kernel_policy
    res = prove_kernel_policy(ABI_TABLE)
    GATE.policy_proof = res
    KJOURNAL.append("policy", "proof", severity=6 if res.get("sound") else 2,
                    sound=res.get("sound"), ms=res.get("ms"),
                    failed=[p["property"] for p in res.get("properties", []) if not p["proven"]])
    print(f"[aegis] kernel policy {'PROVEN' if res.get('sound') else 'HAS HOLES'} "
          f"({len(res.get('properties', []))} properties, {res.get('ms')} ms)")


_threading.Thread(target=_kthread("policy-prover", _prove_policy, delay=4.0,
                                  purpose="prove the access policy over the syscall table"),
                  name="policy-prover", daemon=True).start()


# --- The filesystem: the kernel's objects as one namespace ------------------ #
BOOT.checkpoint("vfs", "mount /proc /dev /patients /proofs /var/log /etc /models /apps "
                       "/artifacts /home /tmp")
import json as _json
from backend.aegis.kernel.vfs import (VFS as _VFSClass, VirtualFS as _VirtualFS,
                                      Leaf as _Leaf, HomeFS as _HomeFS, TmpFS as _TmpFS,
                                      FileStore as _FileStore, VFSError as _VFSError)
from backend.aegis.kernel.policyfile import (PolicyStore as _PolicyStore,
                                             PolicyRejected as _PolicyRejected)
from backend.aegis.kernel.osstream import (OSHub as _OSHub, fingerprint as _vfs_fingerprint,
                                          encode as _osstream_encode, send as _osstream_send)

FILES = _FileStore()
# The active /etc/policy.dsl. Starts from the default rules; a persisted policy is
# loaded on a kernel thread after boot (a network read at import time is exactly
# what once stalled a deploy), and every certified bundle is proven against it.
POLICY = _PolicyStore(store=None)
POLICY.store = FILES
bundle_engine.policy = POLICY
KVFS = _VFSClass()


def _load_policy() -> None:
    text = POLICY._load_persisted()
    if text and text != POLICY.text:
        try:
            from backend.aegis.kernel.users import KERNEL as _K
            POLICY.write(_K, text)
            print(f"[aegis] /etc/policy.dsl restored (version {POLICY.version})")
        except _PolicyRejected as exc:
            KJOURNAL.append("policy", "policy.restore_refused", severity=3, reason=str(exc))


_threading.Thread(target=_kthread("policy-load", _load_policy, delay=2.0,
                                  purpose="restore /etc/policy.dsl from the file store"),
                  name="policy-load", daemon=True).start()


def _ring_sample(fd: int) -> Dict[str, Any]:
    """The latest conditioned sample in a live ring, read with the seqlock check the
    fabric was designed around: re-read the slot's sequence stamp after copying and
    retry if the writer overtook the read."""
    ring = cluster.fabric.rings.get(fd)
    if ring is None:
        return {"error": "no such ring"}
    for _ in range(3):
        w = ring.write_index
        if w == 0:
            return {"ring": ring.name, "writes": 0}
        slot = (w - 1) % ring.capacity
        seq = int(ring.seq[slot])
        vals = {f: float(ring.values[slot, j]) for f, j in ring._fidx.items()}
        t_us = float(ring.t_us[slot])
        if int(ring.seq[slot]) == seq and ring.write_index == w:
            return {"ring": ring.name, "seq": seq, "t_us": t_us, "writes": w,
                    "overwrites": ring.overwrites,
                    "values": {k: (None if v != v else round(v, 4)) for k, v in vals.items()}}
    return {"ring": ring.name, "torn": True,
            "note": "the writer overtook this read three times; try again"}


def _log_lines(facility: Optional[str] = None, limit: int = 400) -> str:
    rows = KJOURNAL.tail(limit=limit, facility=facility)
    return "\n".join(
        "%s %6d %s.%s %s: %s %s" % (time.strftime("%H:%M:%S", time.localtime(r["ts"])), r["seq"],
                                   r["facility"], r["severity_name"], r["uid"], r["event"],
                                   _json.dumps(r["detail"], default=str)[:200])
        for r in rows) or "(empty)"


def _tree_proc() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in procfs.list("/proc/"):
        name = p[len("/proc/"):]
        if name and "/" not in name:
            out[name] = _Leaf(lambda c, _p=p: procfs.read(_p))
    return out


def _tree_dev() -> Dict[str, Any]:
    bio: Dict[str, Any] = {}
    for bed, kinds in (getattr(cluster, "_fds", {}) or {}).items():
        bio[bed] = {f"{k}_feed": _Leaf(lambda c, _fd=fd: _ring_sample(_fd), kind="dev",
                                       note="latest conditioned sample from the live ring")
                    for k, fd in kinds.items()}
    drivers = {d["key"]: _Leaf(lambda c, _k=d["key"]: next(
                   (x for x in cluster.driver_plane_status() if x["key"] == _k), {}), kind="dev")
               for d in (cluster.driver_plane_status() if hasattr(cluster, "driver_plane_status") else [])}
    fleet = device_registry.fleet()
    devices = {d["id"]: _Leaf(lambda c, _d=d["id"]: (device_registry.get(_d).to_dict()
                                                     if device_registry.get(_d) else {}),
                              kind="dev", note="SIMULATED device fleet")
               for d in fleet.get("devices", [])}
    return {"bio": bio, "drivers": drivers, "devices": devices,
            "null": _Leaf(lambda c: "", write=lambda c, d, a: None, cap_write="view:public",
                          kind="dev"),
            "random": _Leaf(lambda c: _os.urandom(16).hex(), kind="dev")}


def _roster_entry(bed: str) -> Dict[str, Any]:
    return next((p for p in cohort_roster().get("patients", []) if p.get("bed_id") == bed
                 or p.get("id") == bed), {"bed": bed})


def _tree_patients() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for bed in getattr(cluster, "node_ids", []):
        out[bed] = {
            "record": _Leaf(lambda c, b=bed: _roster_entry(b)),
            "vitals": _Leaf(lambda c, b=bed: cluster.profiles[b].to_dict()
                            if b in cluster.profiles else {}),
            "chart": _Leaf(lambda c, b=bed: cohort_chart(b)),
            "bundle": _Leaf(lambda c, b=bed: cohort_bundle(b),
                            note="certified by the Z3 cage AND /etc/policy.dsl"),
            "causal": _Leaf(lambda c, b=bed: causal_graph(b)),
            "history": _Leaf(read=lambda c, b=bed: patient_terminal.get_history(b)
                             or "no imported record",
                             write=lambda c, d, a, b=bed: patient_terminal.import_history(
                                 b, d, source="vfs:" + c.username),
                             cap_write="record:import",
                             note="write free text to import a prior record"),
            "tracking": _Leaf(read=lambda c, b=bed: patient_terminal.profile(b).to_dict(),
                              write=lambda c, d, a, b=bed: [
                                  patient_terminal.exec(b, ln, context=_patient_ctx(b))
                                  for ln in d.splitlines() if ln.strip()],
                              cap_write="annotate",
                              note="one monitoring command per line: focus lungs · watch lactate > 3"),
            "notes": _Leaf(read=lambda c, b=bed: patient_terminal.get_annotations(b),
                           write=lambda c, d, a, b=bed: patient_terminal.add_synthesis(
                               b, d, context=_patient_ctx(b)),
                           cap_write="annotate", append_only=True,
                           note="append-only clinician notes (>>)"),
        }
    return out


def _tree_proofs() -> Dict[str, Any]:
    out: Dict[str, Any] = {f"{seq}.json": _Leaf(lambda c, _s=seq: orchestrator.proof_for(_s),
                                                note="Ed25519-signed SMT-LIB2 certificate")
                           for seq in sorted(getattr(orchestrator, "proofs", {}))}
    out["core-latest.json"] = _Leaf(lambda c: proof_core_latest())
    return out


_LOG_FACILITIES = ("kern", "boot", "proc", "auth", "syscall", "agent", "egress", "clinical",
                   "policy", "vfs", "sched", "driver", "svc", "pkg", "net", "irq", "cron")


def _tree_log() -> Dict[str, Any]:
    out: Dict[str, Any] = {"journal": _Leaf(lambda c: _log_lines(), cap_read="view:audit",
                                            note="the whole hash-chained journal (tail)")}
    for f in _LOG_FACILITIES:
        public = f in ("kern", "boot", "proc", "sched", "driver", "svc", "pkg", "policy")
        out[f] = _Leaf(lambda c, _f=f: _log_lines(_f),
                       cap_read="view:public" if public else "view:audit")
    return out


def _os_release() -> str:
    import platform as _pf
    return "\n".join([
        'NAME="AegisOS"', 'VERSION="1.0.0-dev"', "ID=aegisos",
        'PRETTY_NAME="AegisOS 1.0 (Clinical Glass)"',
        f'KERNEL="python {_pf.python_version()} · {_pf.system()} {_pf.machine()}"',
        f"BOOT_ID={BOOT_ID}", f"DATA_MODE={DATA_MODE}", f"GATE_MODE={GATE.mode}",
        'NOTE="A research operating system in the browser. Not a medical device."', ""])


def _policy_write(cred, data: str, append: bool) -> Any:
    try:
        return POLICY.write(cred, (POLICY.text + data) if append else data)
    except _PolicyRejected as exc:
        raise _VFSError("EINVAL", f"{exc} {_json.dumps(exc.detail, default=str)}")


def _tree_etc() -> Dict[str, Any]:
    return {
        "os-release": _Leaf(lambda c: _os_release()),
        "hostname": _Leaf(lambda c: "aegis-icu-wing"),
        "motd": _Leaf(lambda c: "AegisOS — one kernel, one gate, one journal. "
                                "Every order is proven before it leaves.\n"),
        "policy.dsl": _Leaf(lambda c: POLICY.text, write=_policy_write, cap_write="policy:write",
                            note="the hospital safety policy every certified bundle is proven "
                                 "against; a contradictory policy is refused"),
        "policy.status": _Leaf(lambda c: POLICY.status()),
        "rbac.json": _Leaf(lambda c: {"rings": RING, "cap_min_ring": CAP_MIN_RING,
                                      "grants": {r: sorted(g) for r, g in GRANTS.items()}}),
        "abi.json": _Leaf(lambda c: {**ABI_TABLE.summary(),
                                     "syscalls": [e.to_json() for e in ABI_TABLE.entries]}),
        "policy-proof.json": _Leaf(lambda c: GATE.policy_proof or {"status": "pending"}),
        "accounts": _Leaf(lambda c: USERS.public_accounts()),
    }


def _tree_models() -> Dict[str, Any]:
    out: Dict[str, Any] = {f"{m['name']}.json": _Leaf(lambda c, _m=m: _m)
                           for m in mlctl.list_models()}
    manifest = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.abspath(__file__)))), "artifacts", "detector_manifest.json")
    if _os.path.isfile(manifest):
        out["production.json"] = _Leaf(lambda c: _json.load(open(manifest)),
                                       note="the certified production detector")
    return out


def _tree_apps() -> Dict[str, Any]:
    listing = pkg_registry.list()
    out = {f"{n}.json": _Leaf(lambda c, _n=n: next(
        (p for p in pkg_registry.list().get("catalogue", []) if p.get("name") == _n), {"name": _n}))
        for n in listing.get("installed", [])}
    out["catalogue.json"] = _Leaf(lambda c: pkg_registry.list())
    return out


_ARTIFACTS_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))), "artifacts")


def _tree_artifacts() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        names = sorted(n for n in _os.listdir(_ARTIFACTS_DIR) if n.endswith(".json"))
    except OSError:
        names = []
    for n in names:
        p = _os.path.join(_ARTIFACTS_DIR, n)
        if _os.path.getsize(p) <= 2 * 1024 * 1024:           # an allowlist, not the disk
            out[n] = _Leaf(lambda c, _p=p: _json.load(open(_p)),
                           note="committed validation result")
    return out


def _home_users() -> List[str]:
    return sorted({a["username"] for a in USERS.public_accounts() if a["username"] != "guest"})


KVFS.mount("/proc", _VirtualFS(_tree_proc, "/proc", note="kernel providers + /proc/<pid>"))
KVFS.mount("/dev", _VirtualFS(_tree_dev, "/dev", note="live rings, drivers, devices"))
KVFS.mount("/patients", _VirtualFS(_tree_patients, "/patients", note="one directory per bed"))
KVFS.mount("/proofs", _VirtualFS(_tree_proofs, "/proofs", note="signed certificates"))
KVFS.mount("/var/log", _VirtualFS(_tree_log, "/var/log", note="views of the one journal"))
KVFS.mount("/etc", _VirtualFS(_tree_etc, "/etc", note="system configuration"))
KVFS.mount("/models", _VirtualFS(_tree_models, "/models", note="trained detectors"))
KVFS.mount("/apps", _VirtualFS(_tree_apps, "/apps", note="installed packages"))
KVFS.mount("/artifacts", _VirtualFS(_tree_artifacts, "/artifacts", note="committed results"))
KVFS.mount("/home", _HomeFS(FILES, _home_users))
KVFS.mount("/tmp", _TmpFS())

OS_HUB = _OSHub()
OS_HUB.ps = lambda: PROCS.ps()
OS_HUB.boot = lambda: BOOT.summary()
OS_HUB.vfs_fingerprint = lambda cred, path: _vfs_fingerprint(KVFS, cred, path)
BOOT.note(mounts=len(KVFS.mounts), policy_rules=len(POLICY.rules))
BOOT.checkpoint("routes-3", "access-control and consult REST surface")


_FED_CACHE: Dict[str, Any] = {}


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


def _effective_risk(bed: str) -> float:
    """The risk the system acts on for a bed (see CohortCluster.effective_risk).
    Falls back to the raw fused risk on a cluster that does not personalize."""
    if hasattr(cluster, "effective_risk"):
        return cluster.effective_risk(bed)
    return float(cluster.profiles[bed].fused_risk)


def _effective_profile(bed: str):
    if hasattr(cluster, "effective_profile"):
        return cluster.effective_profile(bed)
    return cluster.profiles[bed]


def _focus_bundle() -> Optional[Dict[str, Any]]:
    """Compile + certify the Hour-1 bundle for the critical focus bed (the preempt
    owner, else the highest-risk bed once it breaches). Throttled so the Z3 prover
    runs at most ~every 2.5 s per bed rather than every broadcast tick."""
    owner = cluster.sched.last.owner if cluster.sched.last else None
    risk = {nid: _effective_risk(nid) for nid in cluster.profiles}
    focus = None
    if owner and risk.get(owner, 0.0) >= 0.70:
        focus = owner
    else:
        crit = [(r, nid) for nid, r in risk.items() if r >= 0.70]
        if crit:
            focus = max(crit)[1]
    if not focus:
        _bundle.update(bed=None, data=None)
        return None
    now = time.time()
    if _bundle["bed"] != focus or now - _bundle["ts"] > 2.5:
        _bundle["data"] = bundle_engine.generate(
            _effective_profile(focus), getattr(cluster, "histories", {}).get(focus, {}))
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
    OS_HUB.publish("cluster", payload)
    await _fanout(_cluster_clients, payload)


async def _cluster_loop() -> None:
    print("[aegis] ICU-wing cluster loop started (Tactical Sphere telemetry).")
    # The loop's OWN cost is the part not already charged to a bed: the service
    # plane tick, the headline certificate, the payload build and the broadcast.
    # Per-bed work is accounted to each bed's monitor process inside cluster.tick,
    # so wrapping the whole body here would count it twice.
    # Protected: honours no signal. Pausing this loop pauses every patient's
    # monitoring on the ward at once, which is not an administrative action.
    _pid = PROCS.spawn("cluster-loop", KTHREAD, uid="kernel",
                       cadence_s=CLUSTER_TICK_SECONDS,
                       drives="ward telemetry + service plane",
                       protected="the ward monitoring loop honours no signal").pid
    def side_work(decision) -> Dict[str, Any]:
        # Services certify their outputs, the headline alert is certified and the
        # focus bundle proven: all of it Z3, which must never run on the event loop
        # (a request thread holding the prover for a long scenario would freeze
        # every socket). One iteration at a time, so these stay single-threaded.
        with PROCS.slice(_pid):
            if SERVICE_BED is not None:            # tick the managed app/service plane
                aegisd.tick(cluster.frame_for(SERVICE_BED))
            if ppg_monitor is not None:
                ppg_monitor.tick(CLUSTER_TICK_SECONDS)  # advance the live PPG waveform
            _certify_cluster_headline(decision)
            _reconcile_kernel()
            payload = _cluster_payload()
            try:
                hist = {r["id"]: r.get("history") or {} for r in _patient_roster() if r.get("kind") == "cohort"}
                PAGING.observe_ward(payload.get("beds") or [], payload.get("topology"), hist)
            except Exception as exc:  # pragma: no cover
                print(f"[aegis] paging observe error: {exc}")
            return payload

    while True:
        try:
            # try_call from the loop thread used to BLOCK the loop while the writer
            # ran the tick; from a worker thread it blocks only that worker.
            decision = await asyncio.to_thread(EXEC.try_call, cluster.tick)
            if decision is EXEC.BUSY:
                # A long write already owns the wing (a scenario run). Drop this
                # tick instead of queueing it: at a 2 s cadence a queued tick would
                # run late and stale, and a burst of them would land at once. The
                # drop is counted in EXEC.status()["skipped_ticks"].
                await asyncio.sleep(CLUSTER_TICK_SECONDS)
                continue
            payload = await asyncio.to_thread(side_work, decision)
            await _broadcast_cluster(payload)
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] cluster loop error: {exc}")
        await asyncio.sleep(CLUSTER_TICK_SECONDS)


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


def _pk_context(bed: Optional[str]):
    """(base protocol, patient, history, risk level, display name) for a bed, or
    for the live single-patient session when ``bed`` is omitted."""
    from backend.aegis.pharmacokinetics import PatientState
    if bed:
        if bed not in getattr(cluster, "profiles", {}):
            return None
        patient, hist = _pk_patient(bed)
        prof = cluster.profiles[bed]
        risk = _effective_risk(bed)
        level = "HIGH" if risk >= _INDEX_HIGH else "MEDIUM" if risk >= _INDEX_MEDIUM else "LOW"
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


# --- Aegis-eBPF verified device plane (input-side safety) ------------------- #


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


# --- Medical-history intake + records sources (specialise the monitor) ------ #


def _roster_entry(bed: str) -> Dict[str, Any]:
    for p in _patient_roster():
        if p["id"] == bed:
            return p
    return {"id": bed, "septic": None}


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
BOOT.checkpoint("doctor", "AI consult relay and the rest of the surface")
from backend.llm.doctor import AIDoctor as _AIDoctor
from backend.llm.live import LiveConsult as _LiveConsult, LiveUnavailable as _LiveUnavailable

_doctor = _AIDoctor(_llm_client)


# The banding that produced the tier shown on screen, in the units of the probability
# the dossier quotes. Every bed's "probability" is now the shipped detector's
# calibrated output and every tier is read at its operating point (the ward index's
# 0.35 / 0.70 lines are that point by construction), so the bands ARE the operating
# point. Passed to the dossier so the doctor states the scale the tier came from.
_OP = session.core.operating_point
_TIER_BANDS = ({"high": float(_OP.high), "medium": float(_OP.medium)} if _OP is not None
               else {"high": 0.6, "medium": 0.35})


def _doctor_dossier(bed: Optional[str]) -> Dict[str, Any]:
    """Grounded dossier for a bed (or the live session when no bed is named)."""
    snap = patient_snapshot(bed) if bed else patient_snapshot(None)
    try:
        roster = _patient_roster()
    except Exception:
        roster = None
    return _doctor.dossier(snap or {}, cohort=roster, tier_bands=_bands_for(bed))


def _ward_model_code() -> Optional[str]:
    """Which named model (models_registry) scores the ward: A40, or V7 as fallback."""
    from backend.aegis.models_registry import code_for
    w = getattr(cluster, "ward", None)
    return code_for(getattr(getattr(w, "model", None), "detector_id", None)
                    or getattr(getattr(w, "op", None), "detector", None))


def _bands_for(bed: Optional[str]) -> Dict[str, float]:
    """The bands the bed's own detector reads its probability against: a ward bed is
    scored by the ward's detector (the all-variable ICU model when it is serving), the
    live session by the stack."""
    op = getattr(getattr(cluster, "ward", None), "op", None) if bed else None
    if op is not None and getattr(cluster.ward, "ready", False):
        return {"high": float(op.high), "medium": float(op.medium)}
    return _TIER_BANDS


from backend.aegis.hyperscan import (Hyperscan as _Hyperscan, STAGES as _HYPERSCAN_STAGES,
                                     REVIEW_NOTICE as _REVIEW)

_hyperscan_engine: Optional[_Hyperscan] = None


def _run_hyperscan(bed: Optional[str], emit: Optional[Any] = None,
                   on_behalf_of: Any = None) -> Dict[str, Any]:
    # One engine per sweep, bound to whoever asked: the sweep invokes tools, and
    # each must cross the gate as Dr. Aegis acting for THAT person. (A shared
    # engine held one invoke function for every caller.)
    engine = _Hyperscan(_invoke_for(on_behalf_of), _llm_client)
    dossier = _doctor_dossier(bed)
    collected = engine.collect(bed, emit)
    return engine.synthesize(bed, collected, dossier, emit)


from backend.llm import speech as _speech


# ---------------------------------------------------------------------- #
# The agent plane: bind every tool in the catalogue to a real handler.
#
# Reads go straight to the endpoint functions. Writes go to the SAME Z3-gated,
# audited trap gate the console uses — the agent may invoke any syscall in the
# table, and the prover decides which ones take effect. UI tools have no server
# handler; they are forwarded to the cockpit over the consult socket.
# ---------------------------------------------------------------------- #
from backend.aegis.agent import (registry as _tools, summarize as _tool_summarize,
                                 TOOL_CAPS as _TOOL_CAPS)


def _invoke_for(on_behalf_of: Any):
    """A tool invoker bound to the person Dr. Aegis is acting for."""
    return lambda name, args=None: _tools.invoke(name, args, on_behalf_of=on_behalf_of)


def _agent_authorize(tool: Any, args: Dict[str, Any], on_behalf_of: Any) -> Optional[Dict[str, Any]]:
    """The agent's calls cross the SAME gate as a person's: as dr-aegis, holding the
    clinician's capabilities minus the few no model may hold. os_read/os_write are
    decided by the syscall-table entry of the path they target — the generic
    primitive used to reach every endpoint without any identity at all."""
    human = on_behalf_of if on_behalf_of is not None else _GUEST
    agent = USERS.agent_credential(human)
    try:
        if tool.name in ("os_read", "os_write"):
            raw = "/" + str(args.get("path") or "").split("?")[0].strip().lstrip("/")
            GATE.enter(agent, "GET" if tool.name == "os_read" else "POST", raw.rstrip("/"))
        elif tool.name == "syscall":
            GATE.enter(agent, "POST", "/aegis/syscall")
            cap = SYSCALL_CAP.get(str(args.get("syscall") or "").lower().strip())
            if cap is not None:
                GATE.require(cap, agent, what=f"agent syscall {args.get('syscall')}")
        else:
            GATE.require(_TOOL_CAPS[tool.name], agent, what=f"agent tool {tool.name}")
    except _PermissionDenied as exc:
        return {"ok": False, "errno": "EPERM",
                "error": f"refused by the kernel gate: {exc}",
                "capability": exc.capability, "acting_for": human.username,
                "note": "Say plainly that this was refused and why; do not retry it "
                        "another way."}
    return None


import contextlib as _contextlib


@_contextlib.contextmanager
def _agent_principal(on_behalf_of: Any):
    """While a tool runs, the credential in context is Dr. Aegis's — so an endpoint
    it calls audits the agent (acting for the clinician), not the clinician."""
    token = _CURRENT_CRED.set(USERS.agent_credential(
        on_behalf_of if on_behalf_of is not None else _GUEST))
    try:
        yield
    finally:
        _CURRENT_CRED.reset(token)


_tools.authorizer = _agent_authorize
_tools.principal = _agent_principal

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
    res = syscalls.invoke(syscall, bed, args or {}, caller=_cred().username)
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
            "note": "risk is each bed's sepsis risk index (0-1; it pages at 0.70, MEDIUM from "
                    "0.35, the scoring detector's own operating point). "
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


# --------------------------------------------------------------------------- #
# Scenario Lab — the AI's sandbox for posing clinical situations
#
# These are ordinary /aegis routes, so the agent's generic os_read / os_write
# primitives reach them without any special-casing; the named tools in agent.py
# exist for discoverability, not access. Everything here is SYNTHETIC and stays in
# memory: no scenario touches the live cohort, Supabase, or any reported metric.
# --------------------------------------------------------------------------- #


async def _lab_job(name: str, fn, *args, **detail):
    """Run Scenario Lab work as a JOB, OFF the event loop (it runs the detector)."""
    with PROCS.job(name, uid="operator", **detail):
        return await asyncio.to_thread(fn, *args)


# --- identity: sign-in, sessions, break-glass --------------------------------- #


def _session_payload(token: Optional[str], cred) -> Dict[str, Any]:
    return {"status": "ok", "token": token, "credential": cred.to_dict(),
            "gate": GATE.mode}


# --- the filesystem syscalls ------------------------------------------------- #


def _fs_error(exc: Exception) -> Dict[str, Any]:
    errno = getattr(exc, "errno", "EIO")
    return {"status": "error", "errno": errno if isinstance(errno, str) else "EIO",
            "detail": getattr(exc, "message", str(exc))}


# --- WebSocket -------------------------------------------------------------- #



# --- The REST surface: one router per domain (backend/api/routes/) ------------ #
# Included here, after every piece of state the handlers import exists, and
# before the route table is frozen (gate coverage) and the static shell mount.
from backend.api.routes import ROUTERS as _ROUTERS  # noqa: E402
for _r in _ROUTERS:
    app.include_router(_r.router)
    # re-export handlers so helpers (and callers holding `server`) still reach them
    globals().update({k: v for k, v in vars(_r).items()
                      if callable(v) and getattr(v, '__module__', '') == _r.__name__})


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


BOOT.checkpoint("shell", "mount the web shell")
if _os.path.isdir(_WEBUI):
    app.mount("/", _SmartStatic(directory=_WEBUI, html=True), name="aegisos-shell")


# Bind the agent's tools LAST: the handlers close over endpoint functions defined
# throughout this module, and the read primitive indexes the finished route table.
BOOT.checkpoint("agent", "bind the agent plane to the finished route table")
_bind_agent_tools()
_index_read_routes()
print(f"[aegis] agent plane: {len(_tools.bound())} tools bound"
      + (f", UNBOUND: {_tools.unbound()}" if _tools.unbound() else ""))
BOOT.note(tools=len(_tools.bound()), routes=len(app.routes))
# The syscall table must describe EVERY route the kernel serves. A route without an
# entry is a door the gate cannot decide about; say so loudly at boot (verify_abi
# fails the build on it).
GATE_COVERAGE = ABI_TABLE.coverage(_live_routes(app))
if not GATE_COVERAGE["complete"]:
    print(f"[aegis] GATE COVERAGE INCOMPLETE — missing {GATE_COVERAGE['missing']} "
          f"stale {GATE_COVERAGE['stale']}; run python -m backend.aegis.kernel.abi --sync")
    KJOURNAL.append("policy", "abi.incomplete", severity=2,
                    missing=[f"{m} {p}" for m, p in GATE_COVERAGE["missing"]],
                    stale=[f"{m} {p}" for m, p in GATE_COVERAGE["stale"]])

# Import-time construction is finished: THIS is when the kernel is up. (It used to
# be declared up before the pharmacopoeia, RBAC and the agent plane were built, so
# the reported cold start was shorter than the real one.)
print("[aegis] kernel up in %.1fs (%d boot phases, boot %s)"
      % (BOOT.ready(), len(BOOT.phases), BOOT_ID))

# Everything built above lives as long as the process: the ward's precedent bank (tens
# of thousands of case records), the pharmacopoeia graph, the route table. Left in the
# cyclic collector's view, every gen-2 pass re-walks all of it — measured at up to
# 35 ms stalls mid-tick at 800 beds. Frozen, a pass scans only what the running
# system allocates (worst pause 2 ms). Refcounting still frees anything released.
import gc as _gc
_gc.collect()
_gc.freeze()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("backend.aegis.server:app", host="127.0.0.1", port=8100, reload=False)
