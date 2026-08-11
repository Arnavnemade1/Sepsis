"""Clinical-narrative explainer.

Turns the detector's structured output into a short, clinician-readable summary:
what is driving the risk, which signals are involved, and suggested next checks.
Uses the LLM only on meaningful events (e.g. a HIGH-risk transition), and always
has a deterministic template fallback so behaviour is reliable offline.

IMPORTANT: this is decision-SUPPORT phrasing, not medical advice. The narrative
restates the model's own evidence; it does not introduce new diagnoses.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from backend.llm.provider import LLMProvider

_SYSTEM = (
    "You are a concise clinical decision-support assistant for an EXPERIMENTAL "
    "sepsis early-warning research system. You receive a structured risk "
    "assessment and must explain it in 3-5 short sentences for a clinician. "
    "Rules: (1) only restate and interpret the evidence given; never invent "
    "vitals or diagnoses. (2) Name the top contributing signals. (3) Suggest "
    "appropriate confirmatory steps (e.g. lactate, blood cultures, manual vitals). "
    "(4) End with a one-line reminder that this is a research tool, not a "
    "diagnosis. Be calm and specific."
)


class AlertExplainer:
    def __init__(self, client: Optional[LLMProvider] = None):
        self.client = client or LLMProvider()

    def narrate(
        self,
        risk_level: str,
        detection: Dict[str, Any],
        vitals: Dict[str, float],
        camera_vitals: Optional[Dict[str, Any]] = None,
        triggered_rules: Optional[list] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """Return {'narrative': str, 'source': 'llm'|'template'}."""
        prompt = self._build_prompt(risk_level, detection, vitals, camera_vitals, triggered_rules)
        if use_llm and self.client.available:
            text = self.client.chat(_SYSTEM, prompt)
            if text:
                return {"narrative": text, "source": "llm"}
        return {"narrative": self._template(risk_level, detection, vitals, triggered_rules),
                "source": "template"}

    # ------------------------------------------------------------------ #
    def _build_prompt(self, risk_level, detection, vitals, camera_vitals, rules) -> str:
        lines = [f"Risk level: {risk_level}",
                 f"Model sepsis probability: {detection.get('sepsis_probability')}",
                 f"Confidence: {detection.get('confidence')}",
                 f"SIRS/qSOFA contributions and scores: {detection.get('raw_score')}"]
        key_vitals = {k: vitals.get(k) for k in
                      ("heart_rate", "hrv", "respiratory_rate", "spo2", "skin_temp",
                       "systolic_bp", "map", "sweat_lactate") if k in vitals}
        lines.append(f"Vitals: {key_vitals}")
        if camera_vitals:
            lines.append(f"Camera-derived (experimental) signals: {camera_vitals}")
        if rules:
            lines.append("Triggered rules: " + "; ".join(rules[:8]))
        lines.append("\nExplain this assessment for the clinician.")
        return "\n".join(lines)

    def _template(self, risk_level, detection, vitals, rules) -> str:
        prob = detection.get("sepsis_probability")
        top = "; ".join(rules[:4]) if rules else "no specific rules triggered"
        return (
            f"{risk_level} sepsis risk (model probability {prob}). "
            f"Primary contributors: {top}. "
            f"Recommended next steps: confirm with manual vitals, consider serum "
            f"lactate and blood cultures, and reassess shortly. "
            f"This is an experimental research tool, not a diagnosis."
        )
