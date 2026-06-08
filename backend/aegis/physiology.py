"""Forward physiology projection — the engine behind the what-if twin scheduler.

Given the patient's current vitals, an illness "severity" drive, and a candidate
treatment, this projects the vitals forward a number of steps using transparent,
monotonic pharmacodynamics:

  * untreated sepsis  -> continued deterioration (rising HR/lactate/temp, falling
                         MAP/SpO2/HRV)
  * antibiotics       -> gradual clearance of the infection drive (with a lag)
  * IV fluids         -> MAP/BP support, mild lactate clearance
  * vasopressors      -> proportional MAP support

⚠️ This is an ILLUSTRATIVE mechanistic model for decision *exploration*, not a
validated outcome predictor. It exists so a clinician can compare the projected
*direction* of candidate treatments — never as a guaranteed forecast.
"""
from __future__ import annotations

from typing import Dict, List


def _toward(current: float, target: float, rate: float) -> float:
    return current + (target - current) * rate


def forward_project(
    vitals: Dict[str, float],
    treatment: Dict[str, float],
    severity: float,
    steps: int = 16,
) -> List[Dict[str, float]]:
    """Project ``steps`` future vitals dicts under ``treatment``.

    ``severity`` in [0,1] is the current illness drive (e.g. from sepsis
    probability / organ dysfunction). Returns a trajectory INCLUDING the
    projected steps (not the starting point).
    """
    v = dict(vitals)
    infection = max(0.0, min(1.0, float(severity)))     # internal illness load
    fluid = float(treatment.get("fluid_rate", 0.0))
    vaso = float(treatment.get("vasopressor_dose", 0.0))
    antibiotic = int(treatment.get("antibiotic_code", 0)) > 0

    traj: List[Dict[str, float]] = []
    for _ in range(steps):
        # Infection load dynamics: antibiotics clear it (with lag); else it
        # smolders/worsens in proportion to how septic the patient already is.
        if antibiotic:
            infection *= 0.90
        else:
            infection = min(1.0, infection * (1.0 + 0.03 * infection))

        # Targets driven by the (decaying) infection load.
        hr_t = 70.0 + 60.0 * infection
        temp_t = 36.7 + 2.6 * infection
        lact_t = max(0.6, 1.0 + 6.0 * infection - 0.12 * fluid)
        spo2_t = 99.0 - 11.0 * infection
        hrv_t = 80.0 - 66.0 * infection
        rr_t = 14.0 + 16.0 * infection
        # MAP: sepsis lowers it; fluids + vasopressors raise it.
        map_t = (92.0 - 38.0 * infection) + 0.8 * fluid + 130.0 * vaso
        map_t = max(40.0, min(110.0, map_t))

        v["heart_rate"] = round(_toward(v.get("heart_rate", 75), hr_t, 0.12), 2)
        v["skin_temp"] = round(_toward(v.get("skin_temp", 36.8), temp_t, 0.10), 2)
        v["sweat_lactate"] = round(max(0.4, _toward(v.get("sweat_lactate", 1.0), lact_t, 0.12)), 2)
        v["spo2"] = round(min(100.0, _toward(v.get("spo2", 98), spo2_t, 0.12)), 2)
        v["hrv"] = round(max(5.0, _toward(v.get("hrv", 80), hrv_t, 0.10)), 2)
        v["respiratory_rate"] = round(_toward(v.get("respiratory_rate", 14), rr_t, 0.12), 2)
        v["map"] = round(_toward(v.get("map", 90), map_t, 0.16), 2)
        # Keep SBP/DBP consistent with MAP so downstream qSOFA logic is coherent.
        v["systolic_bp"] = round(v["map"] * 1.35, 2)
        v["diastolic_bp"] = round(v["map"] * 0.82, 2)
        traj.append(dict(v))

    return traj


def severity_from_detection(detection: Dict[str, float], risk_level: str) -> float:
    """Map a detection result to a 0..1 illness drive for the forward model."""
    prob = detection.get("ml_probability")
    if prob is None:
        prob = detection.get("sepsis_probability", 0.0)
    organ = float(detection.get("organ_dysfunction_score", 0)) / 6.0
    base = max(float(prob), organ)
    if risk_level == "HIGH":
        base = max(base, 0.7)
    elif risk_level == "MEDIUM":
        base = max(base, 0.4)
    return float(max(0.0, min(1.0, base)))
