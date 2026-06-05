import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from backend.core.hdc import HDCEngine

class PatientDigitalTwin:
    def __init__(self, patient_id: str, hdc_engine: HDCEngine, history_limit: int = 120):
        self.patient_id = patient_id
        self.hdc = hdc_engine
        self.history_limit = history_limit
        
        # Telemetry history buffer
        self.telemetry_history: List[Dict[str, Any]] = []
        
        # Current state representation
        self.latest_vitals: Dict[str, float] = {
            "heart_rate": 75.0,
            "hrv": 80.0,
            "spo2": 98.0,
            "skin_temp": 36.6,
            "motion": 10.0,
            "sweat_lactate": 1.0
        }
        
        self.current_state_vector: np.ndarray = self.hdc.encode_patient_state(self.latest_vitals)
        self.healthy_similarity: float = 1.0
        self.sepsis_similarity: float = 0.0

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
        
        # Perform HDC similarity classification
        self.healthy_similarity, self.sepsis_similarity = self.hdc.classify_state(self.current_state_vector)
        
        # Calculate physiological features
        hrv_trend = self._calculate_hrv_trend()
        temp_trend = self._calculate_temp_trend()
        
        return {
            "patient_id": self.patient_id,
            "vitals": self.latest_vitals,
            "healthy_similarity": self.healthy_similarity,
            "sepsis_similarity": self.sepsis_similarity,
            "hrv_trend": hrv_trend,
            "temp_trend": temp_trend,
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
            "trends": {
                "hrv": self._calculate_hrv_trend(),
                "temp": self._calculate_temp_trend()
            }
        }
