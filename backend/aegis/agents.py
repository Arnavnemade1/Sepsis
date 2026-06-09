"""Neurosymbolic multi-agent reasoning plane (AEGIS-OS layer 4).

Specialised clinical agents each examine the patient through one lens, state an
opinion with a confidence and an explicit rationale, then participate in a single
transparent *debate* round (concordant systems reinforce each other, the way a
clinician trusts a diagnosis more when independent signs agree) before the plane
reaches a confidence-weighted **consensus**. A non-scoring **Safety Oracle**
guards the result and can escalate the tier on hard danger criteria.

Every number an agent emits is traceable to a named clinical signal — this plane
is *explainable by construction*, which is exactly what a critical-care system
must be. Agents reuse the same physiology the rest of AEGIS-OS uses; they add
structure, transparency, and a debate, not a second source of truth.

Agents (per spec): Cardiac · Inflammatory/Immune · Infection/Pathogen ·
Metabolic/Organ · Safety Oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Opinion + context
# --------------------------------------------------------------------------- #
@dataclass
class AgentOpinion:
    agent: str
    concern: float                 # 0..1 — how concerning this lens finds the state
    confidence: float              # 0..1 — how much evidence backed the opinion
    rationale: str
    evidence: List[str] = field(default_factory=list)
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent, "concern": round(self.concern, 3),
            "confidence": round(self.confidence, 3), "rationale": self.rationale,
            "evidence": self.evidence, "recommendation": self.recommendation,
        }


@dataclass
class ReasoningContext:
    """Everything the agents are allowed to look at (read-only)."""
    vitals: Dict[str, float]
    detection: Dict[str, Any]
    twin_features: Dict[str, Any]
    trends: Dict[str, Any]
    clinical_history: Dict[str, Any]

    def v(self, key: str, default: float) -> float:
        val = self.vitals.get(key)
        return float(val) if val is not None else default


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
class ClinicalAgent:
    name: str = "agent"

    def assess(self, ctx: ReasoningContext) -> AgentOpinion:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _band(value: float, lo: float, hi: float) -> float:
        """Linear 0..1 ramp between lo (0) and hi (1), clipped."""
        if hi == lo:
            return 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


class CardiacAgent(ClinicalAgent):
    """Heart rate, perfusion pressure, and autonomic (HRV) tone."""
    name = "Cardiac"

    def assess(self, ctx: ReasoningContext) -> AgentOpinion:
        hr = ctx.v("heart_rate", 75.0)
        sbp = ctx.v("systolic_bp", 120.0)
        map_ = ctx.v("map", 90.0)
        hrv = ctx.v("hrv", 80.0)
        shock_index = hr / max(sbp, 1.0)

        ev, parts = [], []
        tachy = self._band(hr, 90.0, 150.0)
        if hr > 90:
            ev.append(f"tachycardia {hr:.0f} bpm"); parts.append(tachy)
        si = self._band(shock_index, 0.7, 1.3)
        if shock_index > 0.9:
            ev.append(f"shock index {shock_index:.2f}"); parts.append(si)
        hypo = self._band(70.0 - map_, 5.0, 25.0)
        if map_ < 70:
            ev.append(f"MAP {map_:.0f} mmHg"); parts.append(hypo)
        auto = self._band(40.0 - hrv, 5.0, 30.0)
        if hrv < 35:
            ev.append(f"suppressed HRV {hrv:.0f} ms"); parts.append(auto)

        concern = float(np.clip(max([0.0] + parts) * 0.6 + np.mean(parts or [0.0]) * 0.4, 0, 1))
        confidence = 0.55 + 0.45 * min(1.0, len(parts) / 3.0)
        rec = "assess volume status / perfusion" if concern > 0.5 else None
        return AgentOpinion(self.name, concern, confidence,
                            "cardiovascular & autonomic stress" if ev else "cardiovascular stable",
                            ev, rec)


class InflammatoryAgent(ClinicalAgent):
    """Host inflammatory response: temperature dysregulation, WBC, SIRS burden."""
    name = "Inflammatory"

    def assess(self, ctx: ReasoningContext) -> AgentOpinion:
        temp = ctx.v("skin_temp", 36.8)
        wbc = float(ctx.clinical_history.get("wbc", 7.5))
        sirs = float(ctx.detection.get("agents", {}).get("infection", {}).get("score", 0.0))
        sirs_count = self._sirs(ctx)

        ev, parts = [], []
        if temp > 38.0:
            parts.append(self._band(temp, 38.0, 40.5)); ev.append(f"fever {temp:.1f}°C")
        elif temp < 36.0:
            parts.append(self._band(36.0 - temp, 0.3, 2.0)); ev.append(f"hypothermia {temp:.1f}°C")
        if wbc > 12 or wbc < 4:
            parts.append(self._band(abs(wbc - 8), 4, 12)); ev.append(f"WBC {wbc:.1f} k/µL")
        if sirs_count >= 2:
            parts.append(self._band(sirs_count, 2, 4)); ev.append(f"SIRS {sirs_count}/4")

        concern = float(np.clip(np.mean(parts) if parts else 0.0, 0, 1))
        confidence = 0.5 + 0.5 * min(1.0, len(parts) / 2.0)
        rec = "source control work-up; consider cultures" if concern > 0.5 else None
        return AgentOpinion(self.name, concern, confidence,
                            "systemic inflammatory response" if ev else "no inflammatory signal",
                            ev, rec)

    @staticmethod
    def _sirs(ctx: ReasoningContext) -> int:
        hr = ctx.v("heart_rate", 75); temp = ctx.v("skin_temp", 36.8)
        rr = ctx.v("respiratory_rate", 14); wbc = float(ctx.clinical_history.get("wbc", 7.5))
        return int((temp > 38 or temp < 36)) + int(hr > 90) + int(rr > 20) + int(wbc > 12 or wbc < 4)


class InfectionAgent(ClinicalAgent):
    """Pathogen likelihood: concordance of inflammatory + perfusion + HDC pattern."""
    name = "Infection"

    def assess(self, ctx: ReasoningContext) -> AgentOpinion:
        hdc_p = float(ctx.detection.get("hdc_probability", 0.0))
        ml_p = ctx.detection.get("ml_probability")
        lactate = ctx.v("sweat_lactate", 1.0)
        sirs = InflammatoryAgent._sirs(ctx)

        ev, parts = [], []
        if sirs >= 2:
            parts.append(self._band(sirs, 2, 4)); ev.append(f"SIRS {sirs}/4")
        if lactate >= 2.0:
            parts.append(self._band(lactate, 2.0, 5.0)); ev.append(f"lactate {lactate:.1f} mM")
        if hdc_p > 0.3:
            parts.append(hdc_p); ev.append(f"HDC sepsis pattern P={hdc_p:.2f}")
        if ml_p is not None:
            parts.append(float(ml_p)); ev.append(f"model P={float(ml_p):.2f}")

        concern = float(np.clip(np.mean(parts) if parts else 0.0, 0, 1))
        confidence = 0.5 + 0.5 * min(1.0, len(parts) / 3.0)
        rec = "empiric antibiotics within 1 h if sepsis suspected" if concern > 0.55 else None
        return AgentOpinion(self.name, concern, confidence,
                            "infection / sepsis pattern" if ev else "no convergent infection pattern",
                            ev, rec)


class MetabolicAgent(ClinicalAgent):
    """Organ function & tissue perfusion: lactate, oxygenation, MAP, organ score."""
    name = "Metabolic"

    def assess(self, ctx: ReasoningContext) -> AgentOpinion:
        lactate = ctx.v("sweat_lactate", 1.0)
        spo2 = ctx.v("spo2", 98.0)
        organ = float(ctx.detection.get("organ_dysfunction_score", 0))
        perf = float(ctx.twin_features.get("perfusion_index", 1.0))

        ev, parts = [], []
        if lactate >= 2.0:
            parts.append(self._band(lactate, 2.0, 6.0)); ev.append(f"lactate {lactate:.1f} mM")
        if spo2 < 94:
            parts.append(self._band(94 - spo2, 2, 10)); ev.append(f"SpO₂ {spo2:.0f}%")
        if organ >= 1:
            parts.append(self._band(organ, 1, 6)); ev.append(f"organ-dysfunction score {organ:.0f}")
        if perf < 0.8:
            parts.append(self._band(0.8 - perf, 0.1, 0.6)); ev.append(f"perfusion index {perf:.2f}")

        concern = float(np.clip(max([0.0] + parts) * 0.6 + np.mean(parts or [0.0]) * 0.4, 0, 1))
        confidence = 0.55 + 0.45 * min(1.0, len(parts) / 3.0)
        rec = "lactate clearance; reassess end-organ perfusion" if concern > 0.5 else None
        return AgentOpinion(self.name, concern, confidence,
                            "metabolic / end-organ stress" if ev else "metabolic stable",
                            ev, rec)


class SafetyOracleAgent(ClinicalAgent):
    """Guardian — does not score risk; vetoes/escalates on hard danger criteria
    and flags low signal quality (so a confident wrong answer can't slip through)."""
    name = "SafetyOracle"

    def verdict(self, ctx: ReasoningContext, consensus: float) -> Dict[str, Any]:
        sbp = ctx.v("systolic_bp", 120.0); map_ = ctx.v("map", 90.0)
        spo2 = ctx.v("spo2", 98.0); lactate = ctx.v("sweat_lactate", 1.0)
        rr = ctx.v("respiratory_rate", 14.0)
        gcs = int(ctx.clinical_history.get("gcs", 15))
        quality = float(ctx.detection.get("sensor_quality", 1.0))

        qsofa = int(rr >= 22) + int(sbp <= 100) + int(gcs < 15)
        shock = (map_ < 65.0 and lactate >= 2.0)
        critical = (spo2 < 90.0) or shock or qsofa >= 2

        flags = []
        if qsofa >= 2:
            flags.append(f"qSOFA {qsofa}/3")
        if shock:
            flags.append("septic-shock physiology (MAP<65 + lactate≥2)")
        if spo2 < 90:
            flags.append(f"severe hypoxemia SpO₂ {spo2:.0f}%")
        force = "HIGH" if critical else ("MEDIUM" if consensus >= 0.45 else None)
        return {
            "agent": self.name,
            "escalate_to": force,
            "critical": bool(critical),
            "qsofa": qsofa,
            "flags": flags,
            "low_signal_quality": quality < 0.5,
            "rationale": ("hard danger criteria met — escalate" if critical
                          else "no hard danger criteria"),
        }


# --------------------------------------------------------------------------- #
# The plane
# --------------------------------------------------------------------------- #
@dataclass
class ConsensusResult:
    opinions: List[AgentOpinion]
    consensus: float               # 0..1 weighted concern
    agreement: float               # 0..1 (1 = agents concur)
    dissent: List[str]             # agents that diverged from consensus
    safety: Dict[str, Any]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opinions": [o.to_dict() for o in self.opinions],
            "consensus": round(self.consensus, 3),
            "agreement": round(self.agreement, 3),
            "dissent": self.dissent,
            "safety": self.safety,
            "summary": self.summary,
        }


class ReasoningPlane:
    def __init__(self) -> None:
        self.clinical: List[ClinicalAgent] = [
            CardiacAgent(), InflammatoryAgent(), InfectionAgent(), MetabolicAgent()
        ]
        self.oracle = SafetyOracleAgent()

    def deliberate(self, ctx: ReasoningContext) -> ConsensusResult:
        # 1) Independent opinions.
        opinions = [a.assess(ctx) for a in self.clinical]

        # 2) One debate round: when ≥2 agents are concordantly concerned, raise
        #    each concerned agent's confidence (independent corroboration is
        #    stronger evidence). Bounded and transparent — no hidden coupling.
        concerned = [o for o in opinions if o.concern >= 0.5 and o.confidence >= 0.5]
        if len(concerned) >= 2:
            for o in concerned:
                o.confidence = float(min(1.0, o.confidence + 0.12 * (len(concerned) - 1)))
                o.evidence.append(f"corroborated by {len(concerned) - 1} other system(s)")

        # 3) Confidence-weighted consensus + agreement.
        w = np.array([o.confidence for o in opinions])
        c = np.array([o.concern for o in opinions])
        consensus = float(np.average(c, weights=w)) if w.sum() > 0 else float(c.mean())
        agreement = float(np.clip(1.0 - 2.0 * np.std(c), 0.0, 1.0))
        dissent = [o.agent for o in opinions if abs(o.concern - consensus) > 0.3]

        # 4) Safety Oracle verdict.
        safety = self.oracle.verdict(ctx, consensus)

        top = max(opinions, key=lambda o: o.concern * o.confidence)
        summary = (f"Consensus concern {consensus:.2f} (agreement {agreement:.2f}); "
                   f"led by {top.agent}"
                   + (f"; Safety Oracle escalates to {safety['escalate_to']}"
                      if safety["escalate_to"] else "")
                   + (f"; dissent: {', '.join(dissent)}" if dissent else ""))
        return ConsensusResult(opinions, consensus, agreement, dissent, safety, summary)
