"""AegisOS streaming layer — the single contract the frontend consumes.

`AegisSession` wraps the dashboard-independent `AegisCore`, keeps rolling
signal-history buffers, and emits ONE versioned JSON snapshot per tick that
contains everything a UI needs to render the live digital-twin: current vitals,
per-channel waveform history, the organ-system / anatomical-region body model,
the risk + probability timeline, the active verified protocol, and the narrative.

Keeping this schema in one place means the (future) frontend integrates against a
stable, documented payload — never against internal objects.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
import time

from backend.aegis.core import AegisAssessment
from backend.aegis.os import AegisOS

SNAPSHOT_SCHEMA_VERSION = "1.2"   # + multi-agent reasoning (agents debate/consensus)


class SignalHistory:
    """Rolling per-channel time series + risk/severity timelines for plotting."""

    def __init__(self, maxlen: int = 600):
        self.maxlen = maxlen
        self.t: Deque[float] = deque(maxlen=maxlen)
        self.channels: Dict[str, Deque[Optional[float]]] = {}
        self.risk: Deque[str] = deque(maxlen=maxlen)
        self.probability: Deque[Optional[float]] = deque(maxlen=maxlen)
        self.region_severity: Dict[str, Deque[float]] = {}

    def record(self, ts: float, body_state: Dict[str, Any]) -> None:
        self.t.append(round(ts, 3))
        for ch in body_state.get("channels", []):
            buf = self.channels.setdefault(ch["key"], deque([None] * len(self.t), maxlen=self.maxlen))
            buf.append(ch.get("value"))
        self.risk.append(body_state.get("overall_risk", "LOW"))
        self.probability.append(body_state.get("sepsis_probability"))
        for region, info in body_state.get("regions", {}).items():
            buf = self.region_severity.setdefault(region, deque([0.0] * len(self.t), maxlen=self.maxlen))
            buf.append(info.get("severity", 0.0))

    def snapshot(self, max_points: int = 300) -> Dict[str, Any]:
        n = len(self.t)
        k = min(max_points, n)
        t = list(self.t)[-k:]
        return {
            "t": t,
            "channels": {key: list(buf)[-k:] for key, buf in self.channels.items()},
            "risk": list(self.risk)[-k:],
            "probability": list(self.probability)[-k:],
            "region_severity": {r: list(b)[-k:] for r, b in self.region_severity.items()},
        }


@dataclass
class AegisSession:
    """A live monitoring session: drives AegisCore and serializes its state."""

    patient_id: str = "aegis-patient"
    patient_name: str = "Unknown"
    age: Optional[int] = None
    gender: Optional[str] = None
    clinical_history: Optional[Dict[str, Any]] = None
    model_path: Optional[str] = None
    use_llm: bool = False
    history_len: int = 600

    os: AegisOS = field(init=False)
    history: SignalHistory = field(init=False)
    latest: Optional[AegisAssessment] = field(init=False, default=None)
    source: str = field(init=False, default="idle")

    def __post_init__(self) -> None:
        self.os = AegisOS(
            patient_id=self.patient_id,
            clinical_history=self.clinical_history,
            model_path=self.model_path,
            use_llm=self.use_llm,
        )
        self.os.boot()
        self.history = SignalHistory(maxlen=self.history_len)

    @property
    def core(self):
        """Back-compat accessor for the runtime layer."""
        return self.os.core

    # ------------------------------------------------------------------ #
    def tick(
        self,
        telemetry: Dict[str, float],
        source: str = "wearable",
        camera_vitals: Optional[Dict[str, Any]] = None,
        signal_quality: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        ts = timestamp if timestamp is not None else time.time()
        a = self.os.assess(
            telemetry, source=source, whatif="auto", timestamp=ts,
            signal_quality=signal_quality, camera_vitals=camera_vitals,
        )
        self.latest = a
        self.source = source
        if a.body_state:
            self.history.record(ts, a.body_state)
        return self.snapshot()

    # ------------------------------------------------------------------ #
    def snapshot(self, history_points: int = 300) -> Dict[str, Any]:
        """THE frontend contract. Stable, versioned, fully JSON-serializable."""
        a = self.latest
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "timestamp": a.timestamp if a else time.time(),
            "source": self.source,
            "patient": {
                "id": self.patient_id, "name": self.patient_name,
                "age": self.age, "gender": self.gender,
                "clinical_history": self.core.clinical_history,
            },
            "risk": {
                "level": a.risk_level if a else "LOW",
                "ml_probability": a.ml_probability if a else None,
                "ns_risk_level": a.ns_risk_level if a else "LOW",
                "sirs_score": a.sirs_score if a else 0,
                "qsofa_score": a.qsofa_score if a else 0,
                "signal_quality": a.signal_quality if a else 0.0,
            },
            "vitals": self.core.twin.latest_vitals,
            "detection": a.detection if a else {},
            "triggered_rules": a.triggered_rules if a else [],
            "body_state": a.body_state if a else None,
            "reasoning": a.reasoning if a else None,
            "whatif": a.whatif if a else None,
            "history": self.history.snapshot(history_points),
            "protocol": a.protocol if a else None,
            "verification": a.verification if a else None,
            "narrative": a.narrative if a else None,
            "narrative_source": a.narrative_source if a else None,
            "system": self.os.status(),
        }
