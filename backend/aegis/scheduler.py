"""What-if Twin Scheduler — predictive treatment exploration.

Projects the digital twin forward under several candidate treatment strategies,
scores each projected trajectory (risk path + perfusion), and Z3-verifies safety,
so the system can show *what would happen* under each option and recommend the
best safe one. This is the "runs predictive simulations in the background"
capability of the digital-twin layer.

Each candidate's projected sepsis risk is scored with the transparent
neurosymbolic engine on every projected step (fast, deterministic). The forward
physiology is illustrative (see physiology.py), so this answers "which direction
does each option push the patient", not "exactly what will happen".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from backend.core.hdc import HDCEngine
from backend.core.neurosymbolic import NeurosymbolicEngine
from backend.aegis.physiology import forward_project, severity_from_detection


@dataclass
class WhatIfScheduler:
    hdc: HDCEngine
    neuro: NeurosymbolicEngine
    optimizer: Optional[Any] = None       # EvolutionaryOptimizer (lazy)
    verifier: Optional[Any] = None        # ProtocolVerifier (lazy)
    horizon: int = 24

    def _ensure_synthesis(self):
        if self.optimizer is None or self.verifier is None:
            from backend.core.evolution import EvolutionaryOptimizer
            from backend.core.verifier import ProtocolVerifier
            self.optimizer = self.optimizer or EvolutionaryOptimizer()
            self.verifier = self.verifier or ProtocolVerifier()

    # ------------------------------------------------------------------ #
    def candidates(self, vitals, clinical_history, risk_level) -> Dict[str, Dict[str, float]]:
        """Build the candidate treatment set. Antibiotic 3 (Cefepime) is the
        default broad-spectrum choice with no penicillin-allergy conflict."""
        self._ensure_synthesis()
        recommended, _ = self.optimizer.optimize(vitals, clinical_history, risk_level)
        return {
            "no_intervention": {"fluid_rate": 0.0, "vasopressor_dose": 0.0, "antibiotic_code": 0},
            "antibiotics_only": {"fluid_rate": 0.0, "vasopressor_dose": 0.0, "antibiotic_code": 3},
            "standard_bundle": {"fluid_rate": 5.0, "vasopressor_dose": 0.0, "antibiotic_code": 3},
            "vasopressor_forward": {"fluid_rate": 3.0, "vasopressor_dose": 0.1, "antibiotic_code": 3},
            "recommended": recommended,
        }

    def project(self, vitals, detection, risk_level, clinical_history) -> Dict[str, Any]:
        """Run the full what-if exploration and rank candidates."""
        self._ensure_synthesis()
        severity = severity_from_detection(detection, risk_level)
        cands = self.candidates(vitals, clinical_history, risk_level)

        results: List[Dict[str, Any]] = []
        for name, treatment in cands.items():
            traj = forward_project(vitals, treatment, severity, self.horizon)
            risks = [self._risk_of(v, clinical_history) for v in traj]
            maps = [v["map"] for v in traj]
            is_safe, msg, proof = self.verifier.verify_protocol(
                clinical_history, vitals, risk_level, treatment)
            mean_risk = float(np.mean(risks))
            final_risk = float(risks[-1])
            min_map = float(np.min(maps))
            # Score: reward low projected risk + adequate perfusion; unsafe = -inf.
            score = (0.6 * (1.0 - mean_risk) + 0.4 * float(np.clip(min_map / 65.0, 0, 1))) \
                if is_safe else float("-inf")
            results.append({
                "name": name,
                "treatment": treatment,
                "safe": bool(is_safe),
                "verification": {"safe": bool(is_safe), "message": msg,
                                 "z3_status": proof.get("z3_status")},
                "summary": {
                    "mean_risk": round(mean_risk, 4),
                    "final_risk": round(final_risk, 4),
                    "min_map": round(min_map, 1),
                    "final_map": round(maps[-1], 1),
                    "score": round(score, 4) if score != float("-inf") else None,
                },
                "trajectory": [
                    {"step": i + 1, "map": v["map"], "heart_rate": v["heart_rate"],
                     "sweat_lactate": v["sweat_lactate"], "spo2": v["spo2"],
                     "projected_risk": round(risks[i], 4)}
                    for i, v in enumerate(traj)
                ],
            })

        safe = [r for r in results if r["safe"]]
        best = max(safe, key=lambda r: r["summary"]["score"]) if safe else None
        return {
            "horizon_steps": self.horizon,
            "severity": round(severity, 3),
            "candidates": results,
            "recommended": best["name"] if best else None,
            "note": "Illustrative pharmacodynamic projection for decision "
                    "exploration — not a validated outcome forecast.",
        }

    # ------------------------------------------------------------------ #
    def _risk_of(self, vitals: Dict[str, float], clinical_history: Dict[str, Any]) -> float:
        """Projected sepsis probability for a hypothetical vitals state."""
        state = self.hdc.encode_patient_state(vitals)
        h_sim, s_sim = self.hdc.classify_state(state)
        _, _, _, _, det = self.neuro.evaluate_patient(
            vitals, s_sim, clinical_history, h_sim, {}, {})
        return float(det.get("sepsis_probability", 0.0))
