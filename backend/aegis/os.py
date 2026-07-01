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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.aegis.core import AegisCore, AegisAssessment
from backend.aegis.scheduler import WhatIfScheduler
from backend.aegis.kernel.devbio import BioDevice, DevBio
from backend.aegis.kernel.memfabric import FabricPump, MemFabric, ReadCursor
from backend.aegis.kernel.neurosym import NeuroSymbolicRuntime, SafetyCertificate

AEGIS_OS_VERSION = "0.8.0"


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
        self.devbio = DevBio()                            # Layer 1 (/dev/bio bus)
        self.fabric = MemFabric()                         # Layer 2 (memory fabric)
        self.safety = NeuroSymbolicRuntime()              # Layer 4 (egress cage)
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

    def mount_sensor(self, node: str, name: str, device: BioDevice,
                     ring_capacity: int = 1024) -> int:
        """Mount a sensor on the /dev/bio bus (L1) and carve its ring in the
        shared memory fabric (L2). Returns the file descriptor.

        This is the honest ingress path: a driver only appears once something is
        streaming through it, and its conditioned samples flow into a zero-copy
        ring rather than ad-hoc copies.
        """
        path = self.devbio.mount(node, name, device)
        fd = self.devbio.open(path)
        self.fabric.map_descriptor(fd, path, device.channel_keys(), ring_capacity)
        return fd

    def subscribe(self, fd: int, replay: bool = False) -> ReadCursor:
        """L2: get an independent zero-copy read cursor over a sensor's ring."""
        return self.fabric.subscribe(fd, replay=replay)

    def pump(self, fds: Optional[List[int]] = None) -> FabricPump:
        """Build the L1->L2 pump for the given fds (default: all open fds)."""
        if fds is None:
            fds = self.devbio.open_fds()
        return FabricPump(self.devbio, self.fabric, fds)

    def safe_dispatch(self, egress: str, payload: Dict[str, Any],
                      context: Optional[Dict[str, Any]] = None
                      ) -> tuple[bool, SafetyCertificate]:
        """L4: route an automated output through the formal egress cage. Nothing
        reaches an external channel without a Z3 certificate (default-deny)."""
        return self.safety.dispatch(egress, payload, context or {})

    def layers(self) -> List[LayerStatus]:
        c = self.core
        bus = self.devbio.status()
        if bus["mounted"]:
            lat = bus["latency"]
            hal_detail = (f"/dev/bio: {bus['mounted']} dev / {bus['open_fds']} fd "
                          f"on {bus['nodes']}; cond p50={lat['p50_us']:.1f}µs")
        else:
            hal_detail = f"/dev/bio ready; legacy sources={self._sources or ['(none)']}"
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
        """The bare-metal substrate (backend/aegis/kernel): L1 /dev/bio bus +
        L2 zero-copy memory fabric. Reported separately from the clinical layer
        list so the Performance Diagnostics dashboard has one source of truth."""
        return {
            "devbio": self.devbio.status(),          # L1
            "memfabric": self.fabric.address_matrix(),  # L2
            "neurosym": self.safety.status(),        # L4 egress cage
        }

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
        if whatif == "always" or (whatif == "auto" and a.risk_level == "HIGH"):
            a.whatif = self.whatif(a)
            # On HIGH risk the specialist panel auto-convenes: a treatment that
            # touches the patient should carry a multi-signed, Z3-gated quorum (or
            # the pre-certified emergency order if the patient is crashing faster
            # than the panel can converge). Best-effort: never break the tick.
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
