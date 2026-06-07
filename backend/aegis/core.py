"""AegisCore — the unified critical-care monitoring runtime.

One object that ingests streaming telemetry (from a wearable or the camera) and
produces a complete, coherent assessment: a calibrated sepsis probability from
the real data-grounded model, a transparent clinical cross-check, a Z3-verified
treatment protocol when risk is high, and a clinician-readable narrative.

It degrades gracefully: with no trained model file it still runs on the
transparent neurosymbolic engine, so the system is never dead-on-arrival.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from backend.core.hdc import HDCEngine
from backend.core.twin import PatientDigitalTwin
from backend.core.neurosymbolic import NeurosymbolicEngine
from backend.ml import config
from backend.ml.data.features import FeatureConfig
from backend.aegis.streaming_features import StreamingFeatureBuilder


@dataclass
class AegisAssessment:
    timestamp: float
    source: str
    risk_level: str                       # LOW | MEDIUM | HIGH
    ml_probability: Optional[float]       # calibrated sepsis probability
    ns_risk_level: str                    # neurosymbolic (interpretable) risk
    sirs_score: int
    qsofa_score: int
    triggered_rules: List[str]
    detection: Dict[str, Any]
    twin_features: Dict[str, Any]
    signal_quality: float
    protocol: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    narrative: Optional[str] = None
    narrative_source: Optional[str] = None
    body_state: Optional[Dict[str, Any]] = None   # organ/region/channel model for visualization

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class AegisCore:
    def __init__(
        self,
        patient_id: str = "aegis-patient",
        clinical_history: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        feature_set: str = "wearable",
        high_threshold: float = 0.12,     # calibrated-probability operating points
        medium_threshold: float = 0.04,   # (derive from validation; see ml/README)
        use_llm: bool = True,
        enable_protocols: bool = True,
    ):
        self.patient_id = patient_id
        self.clinical_history = clinical_history or {
            "chf": False, "renal_failure": False, "penicillin_allergy": False,
            "wbc": 7.5, "gcs": 15,
        }
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.enable_protocols = enable_protocols

        self.hdc = HDCEngine()
        self.twin = PatientDigitalTwin(patient_id, self.hdc)
        self.neuro = NeurosymbolicEngine()
        self.feature_cfg = (FeatureConfig.lab_augmented() if feature_set == "lab"
                            else FeatureConfig.wearable_only())
        self.features = StreamingFeatureBuilder(self.feature_cfg)

        self.model = self._load_model(model_path, feature_set)
        self.model_feature_names = getattr(self.model, "feature_names", None) if self.model else None

        # Lazy heavy/optional collaborators.
        self._optimizer = None
        self._verifier = None
        self._explainer = None
        self.use_llm = use_llm

    # ------------------------------------------------------------------ #
    def ingest(
        self,
        telemetry: Dict[str, float],
        source: str = "wearable",
        timestamp: Optional[float] = None,
        signal_quality: float = 1.0,
        explain: Optional[bool] = None,
        camera_vitals: Optional[Dict[str, Any]] = None,
    ) -> AegisAssessment:
        ts = timestamp if timestamp is not None else time.time()

        twin_update = self.twin.update_telemetry(telemetry)
        twin_state = self.twin.get_current_twin_state()
        self.features.add(telemetry)

        # 1) Calibrated ML probability from the real data-grounded model.
        ml_prob = self._ml_probability()

        # 2) Transparent neurosymbolic cross-check (also handles missing channels).
        ns_risk, sirs, qsofa, rules, detection = self.neuro.evaluate_patient(
            self.twin.latest_vitals,
            twin_update["sepsis_similarity"],
            self.clinical_history,
            twin_update["healthy_similarity"],
            twin_state["trends"],
            twin_state["features"],
        )
        detection = dict(detection)
        if ml_prob is not None:
            detection["ml_probability"] = round(ml_prob, 4)
        # Confidence cannot exceed the quality of the signal that produced it.
        detection["confidence"] = round(float(detection.get("confidence", 0.0)) * max(signal_quality, 0.05), 4)

        # 3) Unified risk: calibrated ML probability OR hard clinical overrides.
        risk = self._fuse_risk(ml_prob, ns_risk, sirs, qsofa, detection)

        assessment = AegisAssessment(
            timestamp=ts, source=source, risk_level=risk,
            ml_probability=None if ml_prob is None else round(ml_prob, 4),
            ns_risk_level=ns_risk, sirs_score=sirs, qsofa_score=qsofa,
            triggered_rules=rules, detection=detection,
            twin_features=twin_state["features"], signal_quality=round(signal_quality, 3),
        )

        # 4) Body-state model (organ systems / regions / channels) for the
        #    anatomical visualization layer — built every tick, cheap.
        from backend.aegis.body_state import build_body_state
        assessment.body_state = build_body_state(
            self.twin.latest_vitals, detection, twin_state["features"], risk,
            ts, camera_vitals=camera_vitals, clinical_history=self.clinical_history,
        ).to_dict()

        # 5) On HIGH risk: synthesize + formally verify a treatment protocol.
        if risk == "HIGH" and self.enable_protocols:
            self._attach_protocol(assessment)

        # 6) Narrative (LLM-optional), default only on HIGH transitions.
        if (risk == "HIGH") if explain is None else explain:
            self._attach_narrative(assessment)

        return assessment

    # ------------------------------------------------------------------ #
    def _ml_probability(self) -> Optional[float]:
        if self.model is None or not self.features.ready():
            return None
        try:
            feats = self.features.latest_features(self.model_feature_names)
            if self.model_feature_names:           # GBMDetector expects an array
                return float(self.model.predict_proba(feats.to_numpy())[0])
            return float(self.model.predict_proba(feats)[0])   # HDCDetector takes a frame
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] ML inference failed, using neurosymbolic only: {exc}")
            return None

    def _fuse_risk(self, ml_prob, ns_risk, sirs, qsofa, detection) -> str:
        """Fuse the calibrated ML probability with hard clinical safety rules.

        The calibrated, data-grounded model (AUROC ~0.77) drives the risk tiers
        when available; the transparent heuristic (measured near-chance on real
        data) is used only for hard safety OVERRIDES that must escalate
        regardless of the model — and as the sole fallback when no model is
        loaded. This stops the weak heuristic from inflating healthy patients.
        """
        shock = bool(detection.get("shock_signal"))
        # Hard clinical overrides — escalate regardless of the model.
        if (qsofa >= 2) or shock:
            return "HIGH"
        if ml_prob is not None:
            if ml_prob >= self.high_threshold:
                return "HIGH"
            if ml_prob >= self.medium_threshold or sirs >= 2:
                return "MEDIUM"
            return "LOW"
        # Degraded mode (no trained model): defer to the neurosymbolic engine.
        if ns_risk in {"MEDIUM", "HIGH"} or sirs >= 2:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------ #
    def _attach_protocol(self, a: AegisAssessment) -> None:
        if self._optimizer is None:
            from backend.core.evolution import EvolutionaryOptimizer
            from backend.core.verifier import ProtocolVerifier
            self._optimizer = EvolutionaryOptimizer()
            self._verifier = ProtocolVerifier()
        vitals = self.twin.latest_vitals
        protocol, fitness = self._optimizer.optimize(vitals, self.clinical_history, a.risk_level)
        is_safe, msg, proof = self._verifier.verify_protocol(
            self.clinical_history, vitals, a.risk_level, protocol
        )
        a.protocol = {"protocol": protocol, "fitness": round(float(fitness), 4),
                      "applied": bool(is_safe)}
        a.verification = {"safe": bool(is_safe), "message": msg,
                          "z3_status": proof.get("z3_status"),
                          "violated_rules": proof.get("violated_rules", [])}

    def _attach_narrative(self, a: AegisAssessment) -> None:
        if self._explainer is None:
            from backend.llm.explainer import AlertExplainer
            self._explainer = AlertExplainer()
        narr = self._explainer.narrate(
            a.risk_level, a.detection, self.twin.latest_vitals,
            triggered_rules=a.triggered_rules, use_llm=self.use_llm,
        )
        a.narrative = narr["narrative"]
        a.narrative_source = narr["source"]

    # ------------------------------------------------------------------ #
    def _load_model(self, model_path: Optional[str], feature_set: str):
        if model_path is None:
            model_path = os.path.join(config.ARTIFACTS_DIR, f"gbm_{feature_set}.joblib")
        if not os.path.exists(model_path):
            print(f"[aegis] no trained model at {model_path}; "
                  f"running on neurosymbolic engine only.")
            return None
        try:
            import joblib
            model = joblib.load(model_path)
            print(f"[aegis] loaded detector: {type(model).__name__} ({model_path})")
            return model
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] failed to load model ({exc}); neurosymbolic only.")
            return None
