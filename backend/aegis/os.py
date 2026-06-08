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
        self._sources: List[str] = []                     # Layer 1 (HAL registry)
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
        """HAL: register a telemetry device/driver (wearable, camera, EHR…)."""
        if name not in self._sources:
            self._sources.append(name)

    def layers(self) -> List[LayerStatus]:
        c = self.core
        return [
            LayerStatus(1, "Hardware Abstraction Layer", "device drivers / ingestion",
                        True, f"sources={self._sources or ['(none registered)']}"),
            LayerStatus(2, "Hyperdimensional Kernel", "bind/bundle/similarity",
                        self.kernel is not None, f"D={self.kernel.D}"),
            LayerStatus(3, "Digital Twin + What-if Scheduler", "live twin + predictive sim",
                        True, f"horizon={self.scheduler.horizon} steps"),
            LayerStatus(4, "Neurosymbolic Reasoning Plane", "multi-agent SIRS/qSOFA/organ",
                        c.neuro is not None, "6 weighted agents + consensus"),
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
        return a

    def whatif(self, assessment: Optional[AegisAssessment] = None) -> Dict[str, Any]:
        """Run the predictive treatment exploration on the current twin state."""
        det = assessment.detection if assessment else {}
        risk = assessment.risk_level if assessment else "MEDIUM"
        return self.scheduler.project(
            self.core.twin.latest_vitals, det, risk, self.core.clinical_history)
