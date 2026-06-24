"""resonance.app — the cross-modal resonance gate, as a capability-confined app.

Wraps (does not reimplement) :class:`backend.aegis.resonance.CrossModalResonance`.
Its manifest declares exactly the sensor channels the gate consumes and that it
may emit a single ALERT egress — so the kernel can prove this app never reads a
waveform, never writes a treatment protocol, never touches an actuator. It is the
proof that a real, validated engine drops cleanly onto the app platform.
"""
from __future__ import annotations

from typing import Any, Dict

from backend.aegis.apps.sdk import AegisApp, AppContext, AppManifest
from backend.aegis.kernel.neurosym import ALERT
from backend.aegis.resonance import MODALITIES, CrossModalResonance

# the union of every channel the three modality pathways consume (levels + trends)
_READS = tuple(sorted({ch for chans in MODALITIES.values() for ch in chans}))


class ResonanceApp(AegisApp):
    manifest = AppManifest(
        name="resonance",
        version="1.0.0",
        author="aegis-core",
        summary="Cross-modal HDC agreement gate — artifact-robust sepsis evidence.",
        reads=_READS,
        emits=(ALERT,),
        lane_class="ml_tracking",
        lane_budget=2,
    )

    def __init__(self, gate: CrossModalResonance):
        self.gate = gate                       # a fitted CrossModalResonance

    def on_tick(self, ctx: AppContext) -> Dict[str, Any]:
        state = ctx.read_all()                 # only the declared channels
        r = self.gate.assess(state)
        risk = float(r["gated_risk"])
        severity = 3 if risk >= 0.70 else 2 if risk >= 0.40 else 1 if risk >= 0.20 else 0
        # corroborating pathways count as evidence; an artifact-rejected reading does not
        evidence = sum(1 for z in r["per_modality"].values() if z > 0.4)
        cert = ctx.emit(ALERT, {"severity": severity}, {
            "probability": risk, "calibrated": True, "evidence_count": evidence})
        return {
            "gated_risk": risk, "naive_risk": r["naive_risk"],
            "resonance": r["resonance"], "isolated": r["isolated"],
            "uncorroborated_pathway": r["uncorroborated_pathway"],
            "per_modality": r["per_modality"], "severity": severity,
            "alert_certified": cert.ok, "explain": r["explain"],
        }
