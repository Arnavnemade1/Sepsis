import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from backend.core.hdc import HDCEngine

class PatientDigitalTwin:
    def __init__(self, patient_id: str, hdc_engine: HDCEngine, history_limit: int = 120):
        self.patient_id = patient_id
        self.hdc = hdc_engine
        self.history_limit = history_limit
        self.baseline_window = 30
        
        # Telemetry history buffer
        self.telemetry_history: List[Dict[str, Any]] = []
        
        # Current state representation
        self.latest_vitals: Dict[str, float] = {
            "heart_rate": 75.0,
            "hrv": 80.0,
            "spo2": 98.0,
            "skin_temp": 36.6,
            "motion": 10.0,
            "sweat_lactate": 1.0,
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "map": 93.3,
            "respiratory_rate": 14.0
        }
        
        self.current_state_vector: np.ndarray = self.hdc.encode_patient_state(self.latest_vitals)
        self.temporal_state_vector: np.ndarray = self.current_state_vector
        self.healthy_similarity: float = 1.0
        self.sepsis_similarity: float = 0.0
        self.personal_baseline: Dict[str, Dict[str, float]] = self._default_baseline()

    def update_telemetry(self, telemetry_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Updates the digital twin with new sensor readings.
        Encodes the updated state in hyperdimensional space.
        """
        # Apply sanity checking / sensor bounds
        for key, value in telemetry_data.items():
            if key in self.latest_vitals and value is not None:
                self.latest_vitals[key] = float(value)
                
        # Timestamp the reading
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "vitals": self.latest_vitals.copy()
        }
        
        # Maintain history buffer
        self.telemetry_history.append(record)
        if len(self.telemetry_history) > self.history_limit:
            self.telemetry_history.pop(0)
            
        # Re-encode patient state vector using the HDC engine
        self.current_state_vector = self.hdc.encode_patient_state(self.latest_vitals)
        self.temporal_state_vector = self.hdc.encode_temporal_window(
            [r["vitals"] for r in self.telemetry_history[-8:]]
        )
        
        # Perform HDC similarity classification
        point_healthy, point_sepsis = self.hdc.classify_state(self.current_state_vector)
        temporal_healthy, temporal_sepsis = self.hdc.classify_temporal_state(self.temporal_state_vector)
        self.healthy_similarity = round((0.65 * point_healthy) + (0.35 * temporal_healthy), 4)
        self.sepsis_similarity = round((0.65 * point_sepsis) + (0.35 * temporal_sepsis), 4)
        self._update_personal_baseline()
        
        # Calculate physiological features
        hrv_trend = self._calculate_hrv_trend()
        temp_trend = self._calculate_temp_trend()
        features = self._extract_features()
        
        return {
            "patient_id": self.patient_id,
            "vitals": self.latest_vitals,
            "healthy_similarity": self.healthy_similarity,
            "sepsis_similarity": self.sepsis_similarity,
            "hrv_trend": hrv_trend,
            "temp_trend": temp_trend,
            "features": features,
            "timestamp": record["timestamp"]
        }

    def _calculate_hrv_trend(self) -> str:
        """Analyzes HRV trends over the last 10 readings. Drops are clinical indicators of sepsis stress."""
        if len(self.telemetry_history) < 10:
            return "STABLE"
        
        recent_hrvs = [r["vitals"]["hrv"] for r in self.telemetry_history[-10:]]
        slope = np.polyfit(range(len(recent_hrvs)), recent_hrvs, 1)[0]
        
        if slope < -2.0:
            return "STEEP_DECLINE"
        elif slope < -0.5:
            return "DECLINING"
        else:
            return "STABLE"

    def _calculate_temp_trend(self) -> str:
        """Analyzes temperature trends (spiking vs. hypothermia vs. stable)."""
        if len(self.telemetry_history) < 10:
            return "STABLE"
            
        recent_temps = [r["vitals"]["skin_temp"] for r in self.telemetry_history[-10:]]
        current_temp = recent_temps[-1]
        
        if current_temp > 38.3:
            return "FEVER"
        elif current_temp < 36.0:
            return "HYPOTHERMIA"
        
        # Check slope for rising fever
        slope = np.polyfit(range(len(recent_temps)), recent_temps, 1)[0]
        if slope > 0.1:
            return "RISING"
        elif slope < -0.1:
            return "FALLING"
        
        return "STABLE"

    def get_current_twin_state(self) -> Dict[str, Any]:
        """Returns the serialized latest state of the digital twin."""
        return {
            "patient_id": self.patient_id,
            "latest_vitals": self.latest_vitals,
            "hdc_similarity": {
                "healthy": self.healthy_similarity,
                "sepsis": self.sepsis_similarity
            },
            "history_count": len(self.telemetry_history),
            "personal_baseline": self.personal_baseline,
            "features": self._extract_features(),
            "trends": {
                "hrv": self._calculate_hrv_trend(),
                "temp": self._calculate_temp_trend(),
                "heart_rate_slope": self._calculate_numeric_slope("heart_rate"),
                "map_slope": self._calculate_numeric_slope("map"),
                "lactate_slope": self._calculate_numeric_slope("sweat_lactate")
            }
        }

    def _calculate_numeric_slope(self, vital_name: str, window: int = 10) -> float:
        if len(self.telemetry_history) < 3:
            return 0.0

        recent = [
            r["vitals"].get(vital_name)
            for r in self.telemetry_history[-window:]
            if r["vitals"].get(vital_name) is not None
        ]
        if len(recent) < 3:
            return 0.0

        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        return round(float(slope), 4)

    def _default_baseline(self) -> Dict[str, Dict[str, float]]:
        return {
            "heart_rate": {"mean": 75.0, "std": 8.0},
            "hrv": {"mean": 80.0, "std": 12.0},
            "spo2": {"mean": 98.0, "std": 1.0},
            "skin_temp": {"mean": 36.6, "std": 0.3},
            "map": {"mean": 93.3, "std": 6.0},
            "sweat_lactate": {"mean": 1.0, "std": 0.4},
            "respiratory_rate": {"mean": 14.0, "std": 2.0},
        }

    def _update_personal_baseline(self) -> None:
        if len(self.telemetry_history) < 8:
            return

        recent = [r["vitals"] for r in self.telemetry_history[-self.baseline_window:]]
        if not self._looks_baseline_eligible(recent[-1]):
            return

        for vital_name in self.personal_baseline:
            values = [float(r[vital_name]) for r in recent if vital_name in r and r[vital_name] is not None]
            if len(values) < 8:
                continue
            mean = float(np.mean(values))
            std = max(float(np.std(values)), self.personal_baseline[vital_name]["std"] * 0.35, 0.05)
            old = self.personal_baseline[vital_name]
            self.personal_baseline[vital_name] = {
                "mean": round((0.92 * old["mean"]) + (0.08 * mean), 4),
                "std": round((0.92 * old["std"]) + (0.08 * std), 4),
            }

    def _looks_baseline_eligible(self, vitals: Dict[str, float]) -> bool:
        return (
            55.0 <= vitals.get("heart_rate", 999.0) <= 95.0
            and vitals.get("skin_temp", 99.0) < 37.8
            and vitals.get("systolic_bp", 0.0) > 105.0
            and vitals.get("spo2", 0.0) >= 95.0
            and vitals.get("sweat_lactate", 99.0) < 2.0
            and vitals.get("respiratory_rate", 99.0) < 20.0
        )

    def _extract_features(self) -> Dict[str, Any]:
        z_scores = {}
        for vital_name, stats in self.personal_baseline.items():
            value = self.latest_vitals.get(vital_name)
            if value is None:
                continue
            z_scores[vital_name] = round((float(value) - stats["mean"]) / max(stats["std"], 0.05), 4)

        hr_z = max(0.0, z_scores.get("heart_rate", 0.0))
        hrv_z = max(0.0, -z_scores.get("hrv", 0.0))
        temp_z = abs(z_scores.get("skin_temp", 0.0))
        map_z = max(0.0, -z_scores.get("map", 0.0))
        lactate_z = max(0.0, z_scores.get("sweat_lactate", 0.0))
        rr_z = max(0.0, z_scores.get("respiratory_rate", 0.0))
        instability_score = min(10.0, hr_z + hrv_z + temp_z + map_z + lactate_z + rr_z)

        return {
            "baseline_z_scores": z_scores,
            "instability_score": round(float(instability_score), 4),
            "trajectory_velocity": self._trajectory_velocity(),
            "perfusion_index": self._perfusion_index(),
        }

    def _trajectory_velocity(self, window: int = 6) -> float:
        if len(self.telemetry_history) < 2:
            return 0.0
        states = [self.hdc.encode_patient_state(r["vitals"]) for r in self.telemetry_history[-window:]]
        if len(states) < 2:
            return 0.0
        distances = [1.0 - self.hdc.similarity(states[i - 1], states[i]) for i in range(1, len(states))]
        return round(float(np.mean(distances)), 4)

    def _perfusion_index(self) -> float:
        map_val = self.latest_vitals.get("map", 93.3)
        lactate = self.latest_vitals.get("sweat_lactate", 1.0)
        spo2 = self.latest_vitals.get("spo2", 98.0)
        penalty = 0.0
        penalty += max(0.0, 65.0 - map_val) / 20.0
        penalty += max(0.0, lactate - 2.0) / 4.0
        penalty += max(0.0, 92.0 - spo2) / 8.0
        return round(max(0.0, min(1.0, 1.0 - penalty)), 4)
