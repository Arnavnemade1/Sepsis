"""Optical sepsis assessment: camera vitals -> risk -> narrative.

Ties the contactless biosensor to the existing detection stack:

    camera vitals (HR, HRV, RR + experimental proxies)
        -> PatientDigitalTwin (HDC state + personal baseline + trends)
        -> NeurosymbolicEngine (transparent SIRS/qSOFA + organ/trend fusion)
        -> AlertExplainer (clinician-readable narrative, LLM-optional)

Why the neurosymbolic engine and not the CinC-trained GBM/HDC model here?
Because the camera's most informative sepsis signal is **HRV collapse**, and the
CinC 2019 dataset has no HRV channel — so a CinC-trained model literally cannot
use the camera's key evidence. Until we obtain a labelled dataset that includes
continuous HRV, camera-based detection is validated only at the *signal* level
(HR/RR vs ground truth, already done) and reasoned about with the transparent
clinical engine. This is an explicit, honest boundary, not an oversight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.core.hdc import HDCEngine
from backend.core.twin import PatientDigitalTwin
from backend.core.neurosymbolic import NeurosymbolicEngine
from backend.llm.explainer import AlertExplainer
from backend.vision.pipeline import CameraVitals


@dataclass
class OpticalAssessment:
    risk_level: str
    sirs_score: int
    qsofa_score: int
    detection: Dict[str, Any]
    triggered_rules: list
    camera_quality: float
    channels_used: Dict[str, float]
    narrative: Optional[str] = None
    narrative_source: Optional[str] = None


class OpticalSepsisAssessor:
    def __init__(
        self,
        patient_id: str = "optical-subject",
        clinical_history: Optional[Dict[str, Any]] = None,
        hdc_engine: Optional[HDCEngine] = None,
        explain_on: str = "HIGH",
    ):
        self.hdc = hdc_engine or HDCEngine()
        self.twin = PatientDigitalTwin(patient_id, self.hdc)
        self.neuro = NeurosymbolicEngine()
        self.explainer = AlertExplainer()
        self.clinical_history = clinical_history or {"wbc": 7.5, "gcs": 15}
        self.explain_on = explain_on

    def assess(self, cam: CameraVitals, explain: Optional[bool] = None) -> OpticalAssessment:
        """Assess one camera reading. ``explain`` overrides the auto LLM trigger."""
        channels = cam.to_detection_vitals()
        # Feed available camera channels into the twin (missing ones keep defaults).
        twin_update = self.twin.update_telemetry(channels)
        twin_state = self.twin.get_current_twin_state()

        risk, sirs, qsofa, rules, detection = self.neuro.evaluate_patient(
            self.twin.latest_vitals,
            twin_update["sepsis_similarity"],
            self.clinical_history,
            twin_update["healthy_similarity"],
            twin_state["trends"],
            twin_state["features"],
        )

        # The detector's own confidence is capped by camera signal quality — we
        # cannot be more certain than the measurement that fed it.
        detection = dict(detection)
        detection["confidence"] = round(
            float(detection.get("confidence", 0.0)) * max(cam.overall_quality, 0.05), 4
        )

        assessment = OpticalAssessment(
            risk_level=risk,
            sirs_score=sirs,
            qsofa_score=qsofa,
            detection=detection,
            triggered_rules=rules,
            camera_quality=cam.overall_quality,
            channels_used={k: cam.confidence.get(k, 0.0) for k in channels},
        )

        do_explain = (risk == self.explain_on) if explain is None else explain
        if do_explain:
            cam_summary = {
                "perfusion_proxy": cam.perfusion_index,
                "stress_vector": cam.stress_vector,
                "hrv_ms": cam.hrv,
                "overall_quality": cam.overall_quality,
            }
            narr = self.explainer.narrate(
                risk, detection, self.twin.latest_vitals,
                camera_vitals=cam_summary, triggered_rules=rules,
            )
            assessment.narrative = narr["narrative"]
            assessment.narrative_source = narr["source"]
        return assessment
