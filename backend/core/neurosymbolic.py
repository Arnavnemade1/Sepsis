import math
from typing import Dict, List, Any, Tuple, Optional

class NeurosymbolicEngine:
    def __init__(self, hd_margin_threshold: float = 0.04):
        self.hd_margin_threshold = hd_margin_threshold
        self.sensor_ranges = {
            "heart_rate": (30.0, 220.0),
            "hrv": (0.0, 220.0),
            "spo2": (50.0, 100.0),
            "skin_temp": (30.0, 43.5),
            "systolic_bp": (50.0, 250.0),
            "diastolic_bp": (25.0, 160.0),
            "map": (30.0, 160.0),
            "sweat_lactate": (0.0, 20.0),
            "respiratory_rate": (4.0, 60.0),
        }

    def evaluate_patient(
        self, 
        vitals: Dict[str, float], 
        sepsis_similarity: float,
        clinical_history: Dict[str, Any],
        healthy_similarity: Optional[float] = None,
        trends: Optional[Dict[str, Any]] = None,
        twin_features: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, int, int, List[str], Dict[str, Any]]:
        """
        Main reasoning entry point.
        Combines neural/pattern matching (HDC similarity) with symbolic rules.
        
        Returns:
            (risk_level, sirs_score, qsofa_score, triggered_rules, detection_details)
        """
        triggered_rules = []
        trends = trends or {}
        twin_features = twin_features or {}
        sensor_quality = self._sensor_quality(vitals)
        
        # 1. Evaluate SIRS (Systemic Inflammatory Response Syndrome) - Criteria (Max 4)
        sirs_score = 0
        
        # Temp > 38°C or < 36°C
        temp = vitals.get("skin_temp", 37.0)
        if temp > 38.0 or temp < 36.0:
            sirs_score += 1
            triggered_rules.append(f"SIRS: Abnormal temperature ({temp}°C)")
            
        # Heart Rate > 90 bpm
        hr = vitals.get("heart_rate", 75.0)
        if hr > 90.0:
            sirs_score += 1
            triggered_rules.append(f"SIRS: Tachycardia ({hr} bpm)")
            
        # Respiratory Rate > 20 bpm (derived or defaults)
        # Note: If not explicitly measured, we can estimate respiratory rate from HR + motion.
        # Let's say we have an estimated or reported respiratory rate (normal 12-20)
        rr = vitals.get("respiratory_rate", 16.0)
        if rr > 20.0:
            sirs_score += 1
            triggered_rules.append(f"SIRS: Tachypnea ({rr} breaths/min)")
            
        # White Blood Cell count (from history/labs, defaults to normal if missing)
        wbc = clinical_history.get("wbc", 7.5) # normal is 4.0 - 11.0 k/uL
        if wbc > 12.0 or wbc < 4.0:
            sirs_score += 1
            triggered_rules.append(f"SIRS: Abnormal WBC count ({wbc} k/uL)")

        # 2. Evaluate qSOFA (quick Sequential Organ Failure Assessment) - Criteria (Max 3)
        qsofa_score = 0
        
        # Respiratory rate >= 22 breaths/min
        if rr >= 22.0:
            qsofa_score += 1
            triggered_rules.append(f"qSOFA: High respiratory rate ({rr} breaths/min)")
            
        # Systolic blood pressure <= 100 mmHg
        sbp = vitals.get("systolic_bp", 120.0)
        if sbp <= 100.0:
            qsofa_score += 1
            triggered_rules.append(f"qSOFA: Hypotension (SBP {sbp} mmHg)")
            
        # Altered Coma/Mental state (from history or sensors, GCS < 15)
        gcs = clinical_history.get("gcs", 15)
        if gcs < 15:
            qsofa_score += 1
            triggered_rules.append(f"qSOFA: Altered mental status (GCS {gcs}/15)")

        # 3. Organ dysfunction and wearable-derived early warning evidence.
        organ_dysfunction_score = 0
        map_val = vitals.get("map")
        if map_val is None:
            map_val = (sbp + (2.0 * vitals.get("diastolic_bp", 80.0))) / 3.0
        spo2 = vitals.get("spo2", 98.0)
        lactate = vitals.get("sweat_lactate", vitals.get("lactate", 1.0))
        hrv = vitals.get("hrv", 80.0)

        if map_val < 65.0:
            organ_dysfunction_score += 2
            triggered_rules.append(f"Organ dysfunction: MAP below perfusion target ({map_val:.1f} mmHg)")
        elif sbp <= 100.0:
            organ_dysfunction_score += 1

        if lactate >= 4.0:
            organ_dysfunction_score += 2
            triggered_rules.append(f"Organ dysfunction: Severe lactate elevation ({lactate:.1f} mM)")
        elif lactate >= 2.0:
            organ_dysfunction_score += 1
            triggered_rules.append(f"Organ stress: Elevated lactate ({lactate:.1f} mM)")

        if spo2 < 90.0:
            organ_dysfunction_score += 2
            triggered_rules.append(f"Organ dysfunction: Severe hypoxemia (SpO2 {spo2:.1f}%)")
        elif spo2 < 92.0:
            organ_dysfunction_score += 1
            triggered_rules.append(f"Organ stress: Hypoxemia (SpO2 {spo2:.1f}%)")

        if hrv < 20.0:
            organ_dysfunction_score += 1
            triggered_rules.append(f"Autonomic stress: Critically suppressed HRV ({hrv:.1f} ms)")

        trend_score = 0
        if trends.get("hrv") in {"DECLINING", "STEEP_DECLINE"}:
            trend_score += 1 if trends["hrv"] == "DECLINING" else 2
            triggered_rules.append(f"Trend: HRV is {trends['hrv'].replace('_', ' ').lower()}")
        if trends.get("temp") in {"RISING", "FEVER", "HYPOTHERMIA"}:
            trend_score += 1
            triggered_rules.append(f"Trend: Temperature state is {trends['temp'].lower()}")

        baseline_z = twin_features.get("baseline_z_scores", {})
        instability_score = float(twin_features.get("instability_score", 0.0))
        perfusion_index = float(twin_features.get("perfusion_index", 1.0))
        baseline_score = 0
        if instability_score >= 8.0:
            baseline_score = 2
            triggered_rules.append(f"Digital twin: Severe personalized baseline deviation ({instability_score:.2f})")
        elif instability_score >= 4.5:
            baseline_score = 1
            triggered_rules.append(f"Digital twin: Elevated personalized baseline deviation ({instability_score:.2f})")

        # 4. Incorporate HDC neural/similarity evidence as a relative margin. Absolute
        # sepsis similarity is intentionally not used alone because normal physiology can
        # have non-zero similarity to multiple archetypes in high-dimensional space.
        healthy_similarity = 0.0 if healthy_similarity is None else healthy_similarity
        hdc_margin = sepsis_similarity - healthy_similarity
        hdc_probability = self._sigmoid((hdc_margin - self.hd_margin_threshold) * 9.0)
        hdc_sepsis_detected = hdc_margin > self.hd_margin_threshold
        if hdc_sepsis_detected:
            triggered_rules.append(
                f"HDC: Sepsis archetype margin positive ({hdc_margin:+.4f}, P={hdc_probability:.2f})"
            )

        agent_findings = self._agent_findings(
            sirs_score=sirs_score,
            qsofa_score=qsofa_score,
            organ_dysfunction_score=organ_dysfunction_score,
            trend_score=trend_score,
            baseline_score=baseline_score,
            hdc_probability=hdc_probability,
            hdc_margin=hdc_margin,
            perfusion_index=perfusion_index,
            baseline_z=baseline_z,
            vitals=vitals,
        )
        consensus_score = sum(agent["score"] * agent["weight"] for agent in agent_findings.values())
        consensus_weight = sum(agent["weight"] for agent in agent_findings.values())
        normalized_consensus = consensus_score / max(consensus_weight, 1e-6)

        # 5. Calibrated neurosymbolic fusion. This is not a replacement for a
        # prospectively validated clinical model; it is a deterministic risk ensemble
        # that makes every contribution visible.
        raw_score = (
            1.15 * sirs_score
            + 1.85 * qsofa_score
            + 1.15 * organ_dysfunction_score
            + 0.55 * min(trend_score, 3)
            + 0.70 * baseline_score
            + 2.10 * hdc_probability
            + 1.15 * normalized_consensus
            - 3.15
        )
        sepsis_probability = self._sigmoid(raw_score)
        confidence = max(0.05, min(0.99, sensor_quality * (0.72 + 0.28 * min(1.0, len(vitals) / 9.0))))

        septic_shock_signal = map_val < 65.0 and lactate >= 2.0 and sirs_score >= 2
        probable_sepsis_signal = sirs_score >= 2 and organ_dysfunction_score >= 2

        if (
            qsofa_score >= 2
            or septic_shock_signal
            or (probable_sepsis_signal and sepsis_probability >= 0.55)
            or (sirs_score >= 2 and hdc_sepsis_detected and organ_dysfunction_score >= 1 and sepsis_probability >= 0.70)
        ):
            risk_level = "HIGH"
        elif (
            sirs_score >= 2
            or organ_dysfunction_score >= 1
            or baseline_score >= 1
            or hdc_sepsis_detected
            or sepsis_probability >= 0.38
        ):
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        detection_details = {
            "sepsis_probability": round(sepsis_probability, 4),
            "confidence": round(confidence, 4),
            "sensor_quality": round(sensor_quality, 4),
            "hdc_margin": round(hdc_margin, 4),
            "hdc_probability": round(hdc_probability, 4),
            "organ_dysfunction_score": organ_dysfunction_score,
            "trend_score": trend_score,
            "baseline_score": baseline_score,
            "instability_score": round(instability_score, 4),
            "perfusion_index": round(perfusion_index, 4),
            "consensus_score": round(normalized_consensus, 4),
            "agents": agent_findings,
            "raw_score": round(raw_score, 4),
            "shock_signal": septic_shock_signal,
            "probable_sepsis_signal": probable_sepsis_signal,
        }

        return risk_level, sirs_score, qsofa_score, triggered_rules, detection_details

    def _sensor_quality(self, vitals: Dict[str, float]) -> float:
        expected = ["heart_rate", "hrv", "spo2", "skin_temp", "systolic_bp", "diastolic_bp", "sweat_lactate", "respiratory_rate"]
        present = 0
        plausible = 0

        for name in expected:
            value = vitals.get(name)
            if value is None:
                continue
            present += 1
            low, high = self.sensor_ranges.get(name, (-math.inf, math.inf))
            if low <= float(value) <= high:
                plausible += 1

        if not expected:
            return 0.0
        completeness = present / len(expected)
        plausibility = plausible / max(1, present)
        return max(0.0, min(1.0, 0.65 * completeness + 0.35 * plausibility))

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _agent_findings(
        self,
        sirs_score: int,
        qsofa_score: int,
        organ_dysfunction_score: int,
        trend_score: int,
        baseline_score: int,
        hdc_probability: float,
        hdc_margin: float,
        perfusion_index: float,
        baseline_z: Dict[str, float],
        vitals: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        infection_score = min(1.0, (sirs_score / 4.0) + (0.2 if vitals.get("sweat_lactate", 0.0) >= 2.0 else 0.0))
        perfusion_score = min(1.0, (organ_dysfunction_score / 5.0) + max(0.0, 1.0 - perfusion_index))
        respiratory_score = min(1.0, (qsofa_score / 3.0) + (0.25 if vitals.get("spo2", 100.0) < 92.0 else 0.0))
        autonomic_score = min(1.0, (trend_score / 4.0) + (0.25 if vitals.get("hrv", 80.0) < 25.0 else 0.0))
        twin_score = min(1.0, baseline_score / 2.0)
        hdc_score = max(0.0, min(1.0, hdc_probability))

        return {
            "infection": {
                "score": round(infection_score, 4),
                "weight": 1.05,
                "rationale": "SIRS burden plus lactate-associated inflammatory stress",
            },
            "perfusion": {
                "score": round(perfusion_score, 4),
                "weight": 1.25,
                "rationale": "MAP, lactate, oxygenation, and organ dysfunction evidence",
            },
            "respiratory": {
                "score": round(respiratory_score, 4),
                "weight": 0.95,
                "rationale": "qSOFA respiratory load and hypoxemia evidence",
            },
            "autonomic": {
                "score": round(autonomic_score, 4),
                "weight": 0.75,
                "rationale": "HRV suppression and short-window physiologic trends",
            },
            "digital_twin": {
                "score": round(twin_score, 4),
                "weight": 0.90,
                "rationale": f"Personal baseline z-score drift: {baseline_z}",
            },
            "hdc": {
                "score": round(hdc_score, 4),
                "weight": 1.10,
                "rationale": f"Hyperdimensional sepsis margin {hdc_margin:+.4f}",
            },
        }
