from typing import Dict, List, Any, Tuple

class NeurosymbolicEngine:
    def __init__(self, hd_sepsis_threshold: float = 0.48):
        self.hd_sepsis_threshold = hd_sepsis_threshold

    def evaluate_patient(
        self, 
        vitals: Dict[str, float], 
        sepsis_similarity: float,
        clinical_history: Dict[str, Any]
    ) -> Tuple[str, int, int, List[str]]:
        """
        Main reasoning entry point.
        Combines neural/pattern matching (HDC similarity) with symbolic rules.
        
        Returns:
            (risk_level, sirs_score, qsofa_score, triggered_rules)
        """
        triggered_rules = []
        
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

        # 3. Incorporate HDC Neural/Similarity Engine
        hdc_sepsis_detected = sepsis_similarity > self.hd_sepsis_threshold
        if hdc_sepsis_detected:
            triggered_rules.append(f"HDC: Sepsis hypervector similarity threshold exceeded ({sepsis_similarity:.4f})")

        # 4. Neurosymbolic Rule Fusion: Decide risk level
        # Risk classification hierarchy:
        # HIGH Risk:
        #   - qSOFA >= 2 (strongly indicates organ dysfunction)
        #   - OR (SIRS >= 2 AND HDC similarity indicates high sepsis likelihood)
        #   - OR (Severe hypotension with sepsis similarity > 0.4)
        # MEDIUM Risk:
        #   - SIRS >= 2
        #   - OR HDC similarity is elevated (e.g. > 0.4) without meeting full SIRS/qSOFA
        # LOW Risk:
        #   - Baseline state, no criteria met.
        
        if qsofa_score >= 2 or (sirs_score >= 2 and hdc_sepsis_detected) or (sbp < 90.0 and sepsis_similarity > 0.40):
            risk_level = "HIGH"
        elif sirs_score >= 2 or hdc_sepsis_detected or sepsis_similarity > 0.35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        return risk_level, sirs_score, qsofa_score, triggered_rules
