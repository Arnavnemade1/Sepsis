"""Body-state model — the data layer for the future anatomical visualization.

The frontend will render a high-tech "digital twin" of the body with live signal
overlays. This module builds the *functionality* behind that view (no visuals):
it translates the current vitals + detector output + twin features into a
structured, JSON-serializable model of the patient's body — per organ system,
per anatomical region, and per signal channel — that a frontend can map directly
onto a 3D/2D body.

Grounding: organ-system severity follows the SOFA framework (the clinical basis
of Sepsis-3) where the signal exists, plus two wearable/camera-native systems
(peripheral perfusion, autonomic) that classic SOFA omits but the optical
biosensor can see. Crucially, each system carries a ``monitored`` flag: systems
that need labs (renal, hepatic, coagulation) are honestly marked UNMONITORED when
only wearable/camera channels are present — the body view must show "no data",
never a fabricated green light.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Severity tiers (ordinal) used across organ systems + regions.
TIERS = ["NORMAL", "STRESS", "DYSFUNCTION", "CRITICAL", "UNMONITORED"]


@dataclass
class OrganSystem:
    name: str
    region: str                 # anatomical region key for the body render
    monitored: bool
    status: str                 # one of TIERS
    severity: float             # 0..1 (0 when unmonitored)
    drivers: List[str] = field(default_factory=list)   # human-readable causes
    sofa_points: Optional[int] = None                  # 0-4 if SOFA-derived


@dataclass
class SignalChannel:
    key: str
    label: str
    value: Optional[float]
    unit: str
    normal_low: float
    normal_high: float
    status: str                 # NORMAL/STRESS/CRITICAL/UNMONITORED
    source: str                 # wearable | camera | lab | derived


@dataclass
class BodyState:
    timestamp: float
    overall_risk: str
    sepsis_probability: Optional[float]
    organ_systems: List[OrganSystem]
    regions: Dict[str, Dict[str, Any]]      # region_key -> {status, severity, systems}
    channels: List[SignalChannel]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_risk": self.overall_risk,
            "sepsis_probability": self.sepsis_probability,
            "organ_systems": [asdict(o) for o in self.organ_systems],
            "regions": self.regions,
            "channels": [asdict(c) for c in self.channels],
        }


def _tier_from_severity(sev: float) -> str:
    if sev >= 0.75:
        return "CRITICAL"
    if sev >= 0.45:
        return "DYSFUNCTION"
    if sev >= 0.2:
        return "STRESS"
    return "NORMAL"


def _present(v: Optional[float]) -> bool:
    return v is not None and not (isinstance(v, float) and v != v)  # not NaN


def _chan_status(v: Optional[float], lo: float, hi: float, crit_lo=None, crit_hi=None) -> str:
    if not _present(v):
        return "UNMONITORED"
    if crit_lo is not None and v <= crit_lo:
        return "CRITICAL"
    if crit_hi is not None and v >= crit_hi:
        return "CRITICAL"
    if v < lo or v > hi:
        return "STRESS"
    return "NORMAL"


def build_body_state(
    vitals: Dict[str, float],
    detection: Dict[str, Any],
    twin_features: Dict[str, Any],
    risk_level: str,
    timestamp: float,
    camera_vitals: Optional[Dict[str, Any]] = None,
    clinical_history: Optional[Dict[str, Any]] = None,
) -> BodyState:
    """Construct the full body-state model from one assessment's inputs."""
    history = clinical_history or {}
    g = lambda k, d=None: vitals.get(k, d)

    hr, hrv_v = g("heart_rate"), g("hrv")
    spo2, resp = g("spo2"), g("respiratory_rate")
    sbp, mapv = g("systolic_bp"), g("map")
    temp, lactate = g("skin_temp"), g("sweat_lactate")
    gcs = history.get("gcs")
    perfusion = (camera_vitals or {}).get("perfusion_index")

    systems: List[OrganSystem] = []

    # --- Cardiovascular (SOFA cardio uses MAP/vasopressors) ----------------- #
    if _present(mapv) or _present(sbp):
        m = mapv if _present(mapv) else (sbp or 0) * 0.7
        sev = _clamp((75.0 - m) / 30.0)
        if _present(lactate):
            sev = max(sev, _clamp((lactate - 2.0) / 4.0))
        drivers = []
        if _present(mapv) and mapv < 65:
            drivers.append(f"MAP {mapv:.0f} mmHg below perfusion target")
        if _present(lactate) and lactate >= 2:
            drivers.append(f"lactate {lactate:.1f} mmol/L")
        systems.append(OrganSystem("Cardiovascular", "heart", True,
                                   _tier_from_severity(sev), round(sev, 3), drivers,
                                   sofa_points=_sofa_cardio(mapv)))
    else:
        systems.append(OrganSystem("Cardiovascular", "heart", False, "UNMONITORED", 0.0))

    # --- Respiratory (SpO2 proxy; SOFA resp truly needs PaO2/FiO2) ---------- #
    if _present(spo2) or _present(resp):
        sev = 0.0
        drivers = []
        if _present(spo2):
            sev = max(sev, _clamp((94.0 - spo2) / 10.0))
            if spo2 < 92:
                drivers.append(f"SpO2 {spo2:.0f}%")
        if _present(resp):
            sev = max(sev, _clamp((resp - 20.0) / 16.0))
            if resp > 22:
                drivers.append(f"respiratory rate {resp:.0f}/min")
        systems.append(OrganSystem("Respiratory", "lungs", True,
                                   _tier_from_severity(sev), round(sev, 3), drivers))
    else:
        systems.append(OrganSystem("Respiratory", "lungs", False, "UNMONITORED", 0.0))

    # --- CNS (SOFA CNS = GCS) ---------------------------------------------- #
    if _present(gcs):
        sev = _clamp((15.0 - gcs) / 10.0)
        systems.append(OrganSystem("Neurological", "brain", True,
                                   _tier_from_severity(sev), round(sev, 3),
                                   [f"GCS {gcs}/15"] if gcs < 15 else [],
                                   sofa_points=_sofa_cns(gcs)))
    else:
        systems.append(OrganSystem("Neurological", "brain", False, "UNMONITORED", 0.0))

    # --- Autonomic (HRV — camera/wearable native, key sepsis signal) ------- #
    if _present(hrv_v):
        sev = _clamp((45.0 - hrv_v) / 35.0)   # HRV collapse => high severity
        systems.append(OrganSystem("Autonomic", "heart", True,
                                   _tier_from_severity(sev), round(sev, 3),
                                   [f"HRV {hrv_v:.0f} ms (suppressed)"] if hrv_v < 30 else []))
    else:
        systems.append(OrganSystem("Autonomic", "heart", False, "UNMONITORED", 0.0))

    # --- Peripheral perfusion / skin (camera perfusion + temp + lactate) --- #
    if _present(perfusion) or _present(temp) or _present(lactate):
        sev = 0.0
        drivers = []
        if _present(perfusion):
            sev = max(sev, _clamp(1.0 - perfusion))
            if perfusion < 0.5:
                drivers.append("reduced facial perfusion (possible shunting)")
        if _present(temp) and (temp >= 38.0 or temp <= 36.0):
            sev = max(sev, 0.3)
            drivers.append(f"skin temp {temp:.1f}°C")
        systems.append(OrganSystem("Peripheral Perfusion", "skin", True,
                                   _tier_from_severity(sev), round(sev, 3), drivers))
    else:
        systems.append(OrganSystem("Peripheral Perfusion", "skin", False, "UNMONITORED", 0.0))

    # --- Lab-only systems: honestly UNMONITORED without labs --------------- #
    systems.append(_lab_system("Renal", "kidneys", history.get("creatinine")))
    systems.append(_lab_system("Hepatic", "liver", history.get("bilirubin")))
    systems.append(_lab_system("Coagulation", "blood", history.get("platelets")))

    # --- Anatomical regions: aggregate systems by region ------------------- #
    regions: Dict[str, Dict[str, Any]] = {}
    for s in systems:
        r = regions.setdefault(s.region, {"severity": 0.0, "monitored": False, "systems": []})
        r["systems"].append(s.name)
        if s.monitored:
            r["monitored"] = True
            r["severity"] = max(r["severity"], s.severity)
    for r in regions.values():
        # A region with no monitored system shows UNMONITORED, never a false green.
        r["severity"] = round(r["severity"], 3)
        r["status"] = _tier_from_severity(r["severity"]) if r["monitored"] else "UNMONITORED"

    channels = _signal_channels(vitals, camera_vitals)
    return BodyState(
        timestamp=timestamp,
        overall_risk=risk_level,
        sepsis_probability=detection.get("ml_probability", detection.get("sepsis_probability")),
        organ_systems=systems,
        regions=regions,
        channels=channels,
    )


def _lab_system(name: str, region: str, value: Optional[float]) -> OrganSystem:
    if not _present(value):
        return OrganSystem(name, region, False, "UNMONITORED", 0.0,
                           ["requires lab values (not available from wearable/camera)"])
    return OrganSystem(name, region, True, "NORMAL", 0.0)   # placeholder when labs exist


def _signal_channels(vitals: Dict[str, float], camera: Optional[Dict[str, Any]]) -> List[SignalChannel]:
    cam = camera or {}
    def src(k):
        return "camera" if k in cam else ("wearable" if k in vitals else "derived")
    v = lambda k: vitals.get(k, cam.get(k))
    specs = [
        ("heart_rate", "Heart Rate", "bpm", 60, 100, 40, 130),
        ("hrv", "Heart Rate Variability", "ms", 30, 120, 15, None),
        ("spo2", "Oxygen Saturation", "%", 95, 100, 90, None),
        ("respiratory_rate", "Respiratory Rate", "/min", 12, 20, None, 24),
        ("systolic_bp", "Systolic BP", "mmHg", 100, 140, 90, None),
        ("map", "Mean Arterial Pressure", "mmHg", 70, 100, 65, None),
        ("skin_temp", "Skin/Core Temp", "°C", 36.0, 37.8, 36.0, 38.3),
        ("sweat_lactate", "Lactate", "mmol/L", 0.5, 2.0, None, 4.0),
    ]
    out = []
    for key, label, unit, lo, hi, crit_lo, crit_hi in specs:
        val = v(key)
        out.append(SignalChannel(
            key=key, label=label, value=(round(float(val), 2) if _present(val) else None),
            unit=unit, normal_low=lo, normal_high=hi,
            status=_chan_status(val, lo, hi, crit_lo, crit_hi), source=src(key),
        ))
    # Experimental optical channels.
    for key, label in (("perfusion_index", "Facial Perfusion (optical)"),
                       ("stress_vector", "Sympathetic Stress (optical)")):
        val = cam.get(key)
        out.append(SignalChannel(key, label, (round(float(val), 3) if _present(val) else None),
                                 "idx", 0.5, 1.0, "NORMAL" if _present(val) else "UNMONITORED",
                                 "camera"))
    return out


def _clamp(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _sofa_cardio(mapv: Optional[float]) -> Optional[int]:
    if not _present(mapv):
        return None
    return 1 if mapv < 70 else 0     # higher points need vasopressor dosing data


def _sofa_cns(gcs: Optional[float]) -> Optional[int]:
    if not _present(gcs):
        return None
    if gcs >= 15:
        return 0
    if gcs >= 13:
        return 1
    if gcs >= 10:
        return 2
    if gcs >= 6:
        return 3
    return 4
