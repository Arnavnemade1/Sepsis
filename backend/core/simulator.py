import random
from typing import Dict, Any

class PatientTelemetrySimulator:
    def __init__(self):
        self.stage = "STABLE"  # STABLE, DETERIORATING, SEPSIS, RECOVERING
        self.tick_count = 0
        
        # Base physiological state
        self.vitals = {
            "heart_rate": 72.0,
            "hrv": 85.0,
            "spo2": 98.5,
            "skin_temp": 36.6,
            "motion": 12.0,
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "sweat_lactate": 0.8,
            "respiratory_rate": 14.0
        }

    def set_stage(self, stage: str):
        if stage in ["STABLE", "DETERIORATING", "SEPSIS", "RECOVERING"]:
            self.stage = stage
            self.tick_count = 0
            print(f"[SIMULATOR] Transitioned patient state to: {stage}")

    def next_tick(self, active_treatment: Dict[str, Any]) -> Dict[str, float]:
        """
        Advances the patient physiology by one time step, incorporating the
        effects of active treatments (fluid rate, vasopressors, antibiotics).
        """
        self.tick_count += 1
        
        # Extract treatment values
        fluid_rate = float(active_treatment.get("fluid_rate", 0.0))
        vaso_dose = float(active_treatment.get("vasopressor_dose", 0.0))
        ab_code = int(active_treatment.get("antibiotic_code", 0))

        # Core logic depending on the stage
        if self.stage == "STABLE":
            # Stable baseline with small noise
            self.vitals["heart_rate"] = 70.0 + random.uniform(-3, 3)
            self.vitals["hrv"] = 80.0 + random.uniform(-5, 5)
            self.vitals["spo2"] = 98.0 + random.uniform(-0.5, 0.5)
            self.vitals["skin_temp"] = 36.6 + random.uniform(-0.2, 0.2)
            self.vitals["motion"] = 15.0 + random.uniform(-5, 5)
            self.vitals["systolic_bp"] = 120.0 + random.uniform(-5, 5)
            self.vitals["diastolic_bp"] = 80.0 + random.uniform(-4, 4)
            self.vitals["sweat_lactate"] = 0.8 + random.uniform(-0.1, 0.1)
            self.vitals["respiratory_rate"] = 14.0 + random.uniform(-1, 1)

            # Auto deteriorate after 20 ticks for testing
            if self.tick_count > 30:
                self.set_stage("DETERIORATING")

        elif self.stage == "DETERIORATING":
            # Rising HR, falling HRV, rising temp (infection), sliding blood pressure
            self.vitals["heart_rate"] += random.uniform(0.5, 1.5)
            self.vitals["hrv"] -= random.uniform(1.0, 3.0)
            self.vitals["skin_temp"] += random.uniform(0.05, 0.15)
            self.vitals["systolic_bp"] -= random.uniform(0.5, 1.5)
            self.vitals["diastolic_bp"] -= random.uniform(0.3, 1.0)
            self.vitals["respiratory_rate"] += random.uniform(0.2, 0.6)
            self.vitals["sweat_lactate"] += random.uniform(0.05, 0.15)
            self.vitals["motion"] *= 0.9  # patient lies down

            # Clamp boundaries
            self.vitals["heart_rate"] = min(110.0, self.vitals["heart_rate"])
            self.vitals["hrv"] = max(30.0, self.vitals["hrv"])
            self.vitals["skin_temp"] = min(38.8, self.vitals["skin_temp"])
            self.vitals["systolic_bp"] = max(95.0, self.vitals["systolic_bp"])
            self.vitals["respiratory_rate"] = min(22.0, self.vitals["respiratory_rate"])

            # Progress to severe sepsis if no intervention occurs
            if self.vitals["heart_rate"] > 95.0 and self.vitals["skin_temp"] > 38.2:
                self.set_stage("SEPSIS")

        elif self.stage == "SEPSIS":
            # Severe deterioration - shock phase
            self.vitals["heart_rate"] += random.uniform(0.5, 2.0)
            self.vitals["hrv"] -= random.uniform(0.5, 2.0)
            self.vitals["spo2"] -= random.uniform(0.1, 0.5)
            self.vitals["systolic_bp"] -= random.uniform(1.0, 3.0)
            self.vitals["diastolic_bp"] -= random.uniform(0.8, 2.0)
            self.vitals["respiratory_rate"] += random.uniform(0.3, 0.8)
            self.vitals["sweat_lactate"] += random.uniform(0.1, 0.3)
            self.vitals["motion"] = max(1.0, self.vitals["motion"] - 1.0)

            # Cap sepsis extremes
            self.vitals["heart_rate"] = min(140.0, self.vitals["heart_rate"])
            self.vitals["hrv"] = max(10.0, self.vitals["hrv"])
            self.vitals["spo2"] = max(86.0, self.vitals["spo2"])
            self.vitals["systolic_bp"] = max(70.0, self.vitals["systolic_bp"])
            self.vitals["diastolic_bp"] = max(40.0, self.vitals["diastolic_bp"])
            self.vitals["respiratory_rate"] = min(30.0, self.vitals["respiratory_rate"])
            self.vitals["sweat_lactate"] = min(8.0, self.vitals["sweat_lactate"])

            # If a verified treatment is applied (specifically fluids + antibiotics), transition to RECOVERING
            if ab_code > 0 and (fluid_rate > 2.0 or vaso_dose > 0.05):
                self.set_stage("RECOVERING")

        elif self.stage == "RECOVERING":
            # Physiological indicators trend back to normal
            # Fluids boost blood pressure
            bp_boost = (fluid_rate * 0.8) + (vaso_dose * 120.0)
            self.vitals["systolic_bp"] = min(120.0, self.vitals["systolic_bp"] + 1.5 + bp_boost * 0.1)
            self.vitals["diastolic_bp"] = min(80.0, self.vitals["diastolic_bp"] + 1.0 + bp_boost * 0.05)
            
            # Antibiotics clear infection, reducing HR and temp
            self.vitals["heart_rate"] = max(72.0, self.vitals["heart_rate"] - 1.0)
            self.vitals["hrv"] = min(80.0, self.vitals["hrv"] + 1.5)
            self.vitals["skin_temp"] = max(36.6, self.vitals["skin_temp"] - 0.05)
            self.vitals["spo2"] = min(99.0, self.vitals["spo2"] + 0.2)
            self.vitals["respiratory_rate"] = max(14.0, self.vitals["respiratory_rate"] - 0.4)
            self.vitals["sweat_lactate"] = max(0.8, self.vitals["sweat_lactate"] - 0.1)
            self.vitals["motion"] = min(15.0, self.vitals["motion"] + 0.5)

            # Once recovered, transition back to stable
            if self.vitals["systolic_bp"] > 110.0 and self.vitals["heart_rate"] < 85.0 and self.vitals["sweat_lactate"] < 1.5:
                self.set_stage("STABLE")
                self.tick_count = 0

        # Calculate MAP (Mean Arterial Pressure)
        map_val = (self.vitals["systolic_bp"] + (2.0 * self.vitals["diastolic_bp"])) / 3.0
        self.vitals["map"] = round(map_val, 2)
        
        # Round all vitals for clean representation
        return {k: round(v, 2) for k, v in self.vitals.items()}
