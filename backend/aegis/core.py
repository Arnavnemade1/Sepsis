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
    whatif: Optional[Dict[str, Any]] = None        # predictive treatment projections (scheduler)
    reasoning: Optional[Dict[str, Any]] = None     # multi-agent debate + consensus
    consensus: Optional[Dict[str, Any]] = None     # BFT specialist consensus (auto on HIGH)

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
        from backend.aegis.agents import ReasoningPlane
        self.reasoning = ReasoningPlane()
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
        # Throttle for the advisory synthesis path (protocol + narrative). The GA +
        # Z3 solve is <1s but need not re-run every 3s tick while the risk band and
        # vitals are unchanged; we reuse the last result until the band changes or it
        # goes stale. Bounds CPU on constrained (free-tier) hosts.
        self._proto_cache: Optional[Dict[str, Any]] = None
        self._proto_stamp: float = 0.0
        self._proto_band: Optional[str] = None
        self._proto_ttl: float = 15.0
        from backend.aegis.body_state import OrganFailureTracker
        self._organ_tracker = OrganFailureTracker()   # per-patient organ trajectory

        # Portable proof export: the cage ENFORCES the protocol; this emits the
        # third-party-checkable, Ed25519-signed evidence. Full cert kept here and
        # fetched on demand (REST); only a compact summary rides the live stream.
        self.enable_proof_export = enable_protocols
        self._proof_ca = None
        self.last_proof: Optional[Dict[str, Any]] = None

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

        # 3) Multi-agent reasoning plane: explicit Cardiac/Inflammatory/Infection/
        #    Metabolic agents debate; Safety Oracle guards (layer 4).
        from backend.aegis.agents import ReasoningContext
        consensus = self.reasoning.deliberate(ReasoningContext(
            self.twin.latest_vitals, detection, twin_state["features"],
            twin_state["trends"], self.clinical_history))

        # 4) Unified risk: calibrated ML probability OR hard clinical overrides,
        #    with the Safety Oracle as an escalation backstop (never downgrades).
        risk = self._fuse_risk(ml_prob, ns_risk, sirs, qsofa, detection)
        risk = self._escalate(risk, consensus.safety.get("escalate_to"))

        assessment = AegisAssessment(
            timestamp=ts, source=source, risk_level=risk,
            ml_probability=None if ml_prob is None else round(ml_prob, 4),
            ns_risk_level=ns_risk, sirs_score=sirs, qsofa_score=qsofa,
            triggered_rules=rules, detection=detection,
            twin_features=twin_state["features"], signal_quality=round(signal_quality, 3),
            reasoning=consensus.to_dict(),
        )

        # 4) Body-state model (organ systems / regions / channels) for the
        #    anatomical visualization layer — built every tick, cheap.
        from backend.aegis.body_state import build_body_state
        _bs = build_body_state(
            self.twin.latest_vitals, detection, twin_state["features"], risk,
            ts, camera_vitals=camera_vitals, clinical_history=self.clinical_history,
        )
        self._organ_tracker.update(_bs)          # annotate trend / ETA / MODS-onset
        assessment.body_state = _bs.to_dict()

        # 5) When risk is ELEVATED (MEDIUM or HIGH): synthesize + formally verify a
        #    treatment protocol. At MEDIUM it is advisory (a projected candidate,
        #    labelled as such and never auto-applied); the actual apply path stays
        #    HIGH-gated downstream. This keeps the protocol/proof card live whenever
        #    there is genuine clinical concern instead of only in a full crisis.
        elevated = risk in ("MEDIUM", "HIGH")
        if elevated and self.enable_protocols:
            self._attach_protocol(assessment)

        # 6) Narrative (LLM-optional): on any elevated risk by default.
        if (elevated) if explain is None else explain:
            self._attach_narrative(assessment)

        return assessment

    # ------------------------------------------------------------------ #
    @staticmethod
    def _escalate(risk: str, target: Optional[str]) -> str:
        """Raise the risk tier toward ``target`` (never lowers it)."""
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if target is None:
            return risk
        return risk if order.get(risk, 0) >= order.get(target, 0) else target

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
        # Advisory = elevated-but-not-crisis. The candidate is shown and proven, but
        # only a HIGH-risk protocol is ever eligible to be auto-applied to the twin.
        advisory = a.risk_level != "HIGH"
        # Reuse the cached solve while the band is unchanged and it is fresh — the GA
        # + Z3 need not re-run every tick when nothing material has moved.
        if (self._proto_cache is not None and self._proto_band == a.risk_level
                and (a.timestamp - self._proto_stamp) < self._proto_ttl):
            a.protocol = dict(self._proto_cache["protocol"], advisory=advisory)
            a.verification = self._proto_cache["verification"]
            return
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
        # Only a certified HIGH-risk protocol is eligible for auto-application.
        a.protocol = {"protocol": protocol, "fitness": round(float(fitness), 4),
                      "applied": bool(is_safe) and not advisory, "advisory": advisory}
        a.verification = {"safe": bool(is_safe), "message": msg,
                          "z3_status": proof.get("z3_status"),
                          "violated_rules": proof.get("violated_rules", [])}

        # Portable, third-party-checkable proof for THIS decision (the core cage path).
        if self.enable_proof_export:
            cert = self._export_core_proof(protocol, vitals, a.risk_level)
            if cert is not None:
                self.last_proof = cert.to_dict()
                sig = self.last_proof.get("signature", {})
                v = self.last_proof.get("verdict", {})
                a.verification["portable_proof"] = {     # compact summary for the stream
                    "available": True,
                    "n_obligations": v.get("n_obligations"),
                    "n_satisfied": v.get("n_satisfied"),
                    "all_satisfied": v.get("all_satisfied"),
                    "key_id": sig.get("key_id"),
                    "content_sha256": self.last_proof.get("content_sha256", "")[:16],
                    "policy_id": "aegis-core-cage",
                }

        # Cache this solve so subsequent same-band ticks reuse it (see throttle above).
        self._proto_cache = {"protocol": dict(a.protocol), "verification": a.verification}
        self._proto_stamp = a.timestamp
        self._proto_band = a.risk_level

    def _get_proof_ca(self):
        """Lazy Ed25519 authority; disables export (never breaks monitoring) on error."""
        if not self.enable_proof_export:
            return None
        if self._proof_ca is None:
            try:
                from backend.aegis.proofexport import CertificateAuthority
                self._proof_ca = CertificateAuthority()
            except Exception as exc:  # pragma: no cover
                print(f"[aegis] proof-export disabled ({exc})")
                self.enable_proof_export = False
                return None
        return self._proof_ca

    def _export_core_proof(self, protocol, vitals, risk_level):
        ca = self._get_proof_ca()
        if ca is None:
            return None
        try:
            from backend.aegis.proofexport import certify_core_protocol
            return certify_core_protocol(
                self.clinical_history, vitals, risk_level, protocol,
                ca=ca, context_salt=self.patient_id)
        except Exception as exc:  # pragma: no cover
            print(f"[aegis] core proof export failed ({exc})")
            return None

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
