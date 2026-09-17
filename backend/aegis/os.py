"""AegisOS — the operating system that boots and owns the layers.

This is the top-level object that makes "AEGIS-OS" a real OS structure rather than
a loose collection of modules. It boots the named layers, owns them, exposes a
single `assess()` entry point, and provides `status()`/`layers()` introspection
so any client (tests, CLI, server, future dashboard) can see the live OS state.

Layers (mapped to the architecture spec):
  1. HAL                  — device/telemetry sources (wearable, camera, EHR)
  2. Hyperdimensional Kernel — HDCEngine: bind/bundle/similarity primitives
  3. Digital Twin + Scheduler — PatientDigitalTwin + WhatIfScheduler (predictive)
  4. Neurosymbolic Reasoning  — multi-agent SIRS/qSOFA/organ consensus
  5. Evolutionary Synthesis   — GA protocol optimizer
  6. Formal Verification      — Z3 SMT safety prover
  7. Persistence              — Supabase/SQLite (optional)
  8. Interface                — versioned streaming snapshot contract

It composes (does not duplicate) AegisCore: `os.core` is the runtime; AegisOS adds
the OS framing + the predictive what-if scheduler on top.

THE KERNEL IS ATTACHED, NOT OWNED A SECOND TIME. This object used to construct its
own DevBio bus, MemFabric arena, HDKernel and egress cage. Nothing ever mounted a
device on them, so every snapshot's ``system.kernel`` reported an empty bus, an
empty arena and a zero-process table while the real ones — the ICU wing's — ran
elsewhere. The server now calls :meth:`attach_kernel` with the live wing, the one
process table, the journal, the boot record and the executor, and
:meth:`kernel_status` reports those. Standalone (tests, notebooks) it says it has no
kernel attached rather than presenting empty structures as a running kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.aegis.core import AegisCore, AegisAssessment
from backend.aegis.scheduler import WhatIfScheduler

AEGIS_OS_VERSION = "1.0.0-dev"


@dataclass
class LayerStatus:
    index: int
    name: str
    role: str
    ready: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class AegisOS:
    name = "AEGIS-OS"
    version = AEGIS_OS_VERSION

    def __init__(
        self,
        patient_id: str = "aegis-patient",
        clinical_history: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        feature_set: str = "wearable",
        use_llm: bool = False,
        enable_protocols: bool = True,
        whatif_horizon: int = 24,
        enable_persistence: bool = False,
    ):
        # Layer 3-6 runtime is AegisCore; AegisOS owns it + the scheduler.
        self.core = AegisCore(
            patient_id=patient_id, clinical_history=clinical_history,
            model_path=model_path, feature_set=feature_set,
            use_llm=use_llm, enable_protocols=enable_protocols,
        )
        self.kernel = self.core.hdc                       # Layer 2
        self.scheduler = WhatIfScheduler(                 # Layer 3 (predictive)
            self.core.hdc, self.core.neuro, horizon=whatif_horizon)
        self._sources: List[str] = []                     # Layer 1 (legacy source tags)
        # The live kernel, attached by the server (see attach_kernel). None when
        # this OS object runs standalone.
        self._kernel: Dict[str, Any] = {}
        self._db = None
        if enable_persistence:
            try:
                from backend.core.database import DatabaseClient
                self._db = DatabaseClient()
            except Exception as exc:  # pragma: no cover
                print(f"[aegis-os] persistence unavailable: {exc}")
        self._booted = False

    # ------------------------------------------------------------------ #
    def boot(self) -> List[LayerStatus]:
        """Bring the OS up and report each layer's status (the boot sequence)."""
        layers = self.layers()
        self._booted = True
        return layers

    def register_source(self, name: str) -> None:
        """HAL: register a telemetry source tag (wearable, camera, EHR…)."""
        if name not in self._sources:
            self._sources.append(name)

    # -- the live kernel ------------------------------------------------- #
    def attach_kernel(self, cluster: Any = None, procs: Any = None, journal: Any = None,
                      boot: Any = None, executor: Any = None) -> "AegisOS":
        """Bind the running kernel: the ICU wing (its /dev/bio bus, memory fabric
        and cognitive scheduler), the process table, the journal, the boot record
        and the single-writer executor."""
        self._kernel = {k: v for k, v in (("cluster", cluster), ("procs", procs),
                                          ("journal", journal), ("boot", boot),
                                          ("executor", executor)) if v is not None}
        return self

    @property
    def kernel_attached(self) -> bool:
        return "cluster" in self._kernel

    def ps(self) -> List[Dict[str, Any]]:
        """The process table — the kernel's, or empty when none is attached."""
        procs = self._kernel.get("procs")
        return procs.ps() if procs is not None else []

    def ktop(self) -> Dict[str, Any]:
        procs = self._kernel.get("procs")
        return procs.top() if procs is not None else {"processes": 0, "attached": False}

    def kproc(self, pid: int) -> Optional[Dict[str, Any]]:
        procs = self._kernel.get("procs")
        if procs is None:
            return None
        try:
            return procs.proc(pid)
        except KeyError:
            return None

    def layers(self) -> List[LayerStatus]:
        c = self.core
        cluster = self._kernel.get("cluster")
        bus = cluster.bus.status() if cluster is not None and hasattr(cluster, "bus") else None
        if bus and bus.get("mounted"):
            lat = bus.get("latency") or {}
            hal_detail = (f"/dev/bio: {bus['mounted']} dev / {bus['open_fds']} fd "
                          f"on {bus['nodes']} beds; cond p50={lat.get('p50_us', 0):.1f}µs")
        else:
            hal_detail = ("standalone (no kernel attached); legacy sources="
                          f"{self._sources or ['(none)']}")
        return [
            LayerStatus(1, "Stochastic Hardware Abstraction Layer",
                        "/dev/bio device-driver bus + inline Kalman", True, hal_detail),
            LayerStatus(2, "Hyperdimensional Kernel", "bind/bundle/similarity",
                        self.kernel is not None, f"D={self.kernel.D}"),
            LayerStatus(3, "Digital Twin + What-if Scheduler", "live twin + predictive sim",
                        True, f"horizon={self.scheduler.horizon} steps"),
            LayerStatus(4, "Neurosymbolic Reasoning Plane", "multi-agent debate + consensus",
                        getattr(c, "reasoning", None) is not None,
                        "Cardiac · Inflammatory · Infection · Metabolic + Safety Oracle"),
            LayerStatus(5, "Evolutionary Synthesis", "GA protocol optimizer",
                        True, "lazy-initialized on first HIGH risk"),
            LayerStatus(6, "Formal Verification", "Z3 SMT safety prover",
                        True, "proves every applied protocol"),
            LayerStatus(7, "Persistence", "Supabase / SQLite",
                        self._db is not None, "enabled" if self._db else "disabled"),
            LayerStatus(8, "Interface", "versioned streaming snapshot",
                        True, "WS + REST contract (aegis/stream.py)"),
        ]

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "booted": self._booted,
            "model_loaded": self.core.model is not None,
            "patient_id": self.core.patient_id,
            "layers": [l.to_dict() for l in self.layers()],
            "kernel": self.kernel_status(),
        }

    def kernel_status(self) -> Dict[str, Any]:
        """The running kernel, reported from the live objects — or an explicit
        ``attached: False`` rather than empty structures dressed as a kernel.

        Kept light: this rides every session snapshot. The arena's full address
        matrix is NOT included (it is per-ring and large); its summary is."""
        if not self._kernel:
            return {"attached": False}
        out: Dict[str, Any] = {"attached": True}
        cluster = self._kernel.get("cluster")
        if cluster is not None:
            if hasattr(cluster, "bus"):
                out["devbio"] = cluster.bus.status()
            if hasattr(cluster, "fabric"):
                try:
                    out["memfabric"] = cluster.fabric.address_matrix().get("arena")
                except Exception:
                    out["memfabric"] = None
            if hasattr(cluster, "sched"):
                out["sched"] = cluster.sched.kstat()
        for name in ("procs", "journal", "executor"):
            obj = self._kernel.get(name)
            if obj is not None:
                out[name] = obj.status()
        boot = self._kernel.get("boot")
        if boot is not None:
            summ = boot.summary()
            out["boot"] = {k: summ[k] for k in ("booted", "boot_seconds", "phases",
                                                "failed", "slowest")}
        return out

    # ------------------------------------------------------------------ #
    def assess(
        self,
        telemetry: Dict[str, float],
        source: str = "wearable",
        whatif: str = "auto",     # "auto" (on HIGH) | "always" | "never"
        **kwargs: Any,
    ) -> AegisAssessment:
        """Single OS entry point: ingest telemetry -> full assessment, with the
        predictive what-if attached per the ``whatif`` policy."""
        self.register_source(source)
        a = self.core.ingest(telemetry, source=source, **kwargs)
        # What-if projection is a read-only pharmacodynamic simulation, so it engages
        # whenever risk is ELEVATED (MEDIUM or HIGH) — a MEDIUM patient benefits from
        # seeing where each treatment path leads, not just a crashing one.
        elevated = a.risk_level in ("MEDIUM", "HIGH")
        if whatif == "always" or (whatif == "auto" and elevated):
            a.whatif = self.whatif(a)
        # The BFT specialist panel auto-convenes only on HIGH risk: it is a heavier
        # quorum decision for a treatment that actually touches the patient (or the
        # pre-certified emergency order if they crash faster than it converges).
        # Best-effort: never break the tick.
        if whatif != "never" and a.risk_level == "HIGH":
            try:
                a.consensus = self.consensus(a)
            except Exception as exc:   # pragma: no cover
                print(f"[aegis-os] consensus skipped: {exc}")
        return a

    def whatif(self, assessment: Optional[AegisAssessment] = None) -> Dict[str, Any]:
        """Run the predictive treatment exploration on the current twin state."""
        det = assessment.detection if assessment else {}
        risk = assessment.risk_level if assessment else "MEDIUM"
        return self.scheduler.project(
            self.core.twin.latest_vitals, det, risk, self.core.clinical_history)

    def consensus(self, assessment: Optional[AegisAssessment] = None) -> Dict[str, Any]:
        """Convene the BFT specialist panel on the current state. Deterioration is
        derived from the risk band + perfusion so a crashing patient gets a tighter
        debate deadline (and, if it can't converge, the pre-certified emergency order)."""
        from backend.aegis.consensus import ClinicalBFT, default_panel
        vit = self.core.twin.latest_vitals or {}
        risk = assessment.risk_level if assessment else "MEDIUM"
        det = {"HIGH": 0.7, "MEDIUM": 0.35, "LOW": 0.1}.get(risk, 0.3)
        if float(vit.get("map", 90)) < 60:
            det = max(det, 0.85)
        state = {"map": float(vit.get("map", 75.0)), "risk_level": risk,
                 "clinical_history": self.core.clinical_history or {},
                 "infection_suspected": risk in ("HIGH", "MEDIUM")}
        out = ClinicalBFT(default_panel()).run_round(state, deterioration=det)
        return out.to_dict()
