"""Body-state model + multi-organ dysfunction engine — the data layer behind the
anatomical digital twin.

This module translates the current vitals + detector output + twin features into a
structured, JSON-serializable model of the patient's body — per organ system, per
anatomical region, per signal channel — that the 3D twin (``webui/twin3d.js``)
maps directly onto real anatomy.

Clinical grounding:
  * Per-organ severity follows the **SOFA** framework (the organ-dysfunction basis
    of Sepsis-3). Each of the six SOFA organs is scored 0–4 by its own validated
    thresholds (respiration, coagulation, liver, cardiovascular, CNS, renal); the
    points sum to a **total SOFA**. A rise of ≥2 with suspected infection is the
    Sepsis-3 organ-dysfunction criterion.
  * **qSOFA** (RR≥22, SBP≤100, GCS<15) is computed as the fast bedside screen.
  * **MODS** (Multiple Organ Dysfunction Syndrome) is derived: how many systems are
    failing at once, which one dominates, and a cascade-risk score — because sepsis
    mortality scales with the *number* of failing organs, not any single one.
  * Two wearable/camera-native systems SOFA omits (peripheral perfusion, autonomic)
    are scored continuously from the optical biosensor.

Honesty rules (unchanged, load-bearing): every system carries a ``monitored`` flag;
systems that need labs (renal, hepatic, coagulation) are marked **UNMONITORED**
when only wearable/camera channels are present — the body view shows "no data",
never a fabricated healthy light. SpO2-derived respiratory SOFA is capped at 2 and
labelled a proxy (true SOFA 3–4 needs PaO2/FiO2 + ventilation).

The stateful :class:`OrganFailureTracker` adds the *trajectory* layer — per-organ
trend, a short-horizon projection, an ETA-to-critical, and MODS-onset edge
detection — so the system catches an organ *heading* for failure, not just one
already failed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional, Tuple

# Severity tiers (ordinal) used across organ systems + regions.
TIERS = ["NORMAL", "STRESS", "DYSFUNCTION", "CRITICAL", "UNMONITORED"]
DYSFUNCTION_SEV = 0.45          # severity at/above which an organ counts as "failing"


@dataclass
class OrganSystem:
    name: str
    region: str                 # anatomical region key for the body render
    monitored: bool
    status: str                 # one of TIERS
    severity: float             # 0..1 (0 when unmonitored)
    drivers: List[str] = field(default_factory=list)   # human-readable causes
    sofa_points: Optional[int] = None                  # 0-4 if SOFA-derived
    sofa_basis: Optional[str] = None                   # what the score was computed from
    # -- trajectory (populated by OrganFailureTracker; None when stateless) --
    trend: Optional[str] = None                        # RISING | FALLING | STABLE
    slope_per_min: Optional[float] = None              # d(severity)/dt
    projected_severity: Optional[float] = None         # severity at +horizon
    eta_critical_min: Optional[float] = None           # minutes to CRITICAL if RISING


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
    sofa: Dict[str, Any] = field(default_factory=dict)     # per-organ points + total
    qsofa: Dict[str, Any] = field(default_factory=dict)    # 0-3 + components
    mods: Dict[str, Any] = field(default_factory=dict)     # multi-organ dysfunction
    sepsis3: Dict[str, Any] = field(default_factory=dict)  # organ-dysfunction criterion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_risk": self.overall_risk,
            "sepsis_probability": self.sepsis_probability,
            "organ_systems": [asdict(o) for o in self.organ_systems],
            "regions": self.regions,
            "channels": [asdict(c) for c in self.channels],
            "sofa": self.sofa,
            "qsofa": self.qsofa,
            "mods": self.mods,
            "sepsis3": self.sepsis3,
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tier_from_severity(sev: float) -> str:
    if sev >= 0.75:
        return "CRITICAL"
    if sev >= DYSFUNCTION_SEV:
        return "DYSFUNCTION"
    if sev >= 0.2:
        return "STRESS"
    return "NORMAL"


def _present(v: Optional[float]) -> bool:
    return v is not None and not (isinstance(v, float) and v != v)  # not NaN


def _clamp(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


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


# --------------------------------------------------------------------------- #
# SOFA sub-scores — each returns (points 0-4, basis text) or None if unmonitored
# --------------------------------------------------------------------------- #
def sofa_respiration(spo2: Optional[float]) -> Optional[Tuple[int, str]]:
    """SpO2-derived proxy for the PaO2/FiO2 SOFA sub-score, CAPPED at 2 (points
    3–4 require mechanical ventilation, which a wearable cannot know)."""
    if not _present(spo2):
        return None
    if spo2 >= 95:
        pts = 0
    elif spo2 >= 92:
        pts = 1
    else:
        pts = 2
    return pts, f"SpO2 {spo2:.0f}% (PaO2/FiO2 proxy, capped at 2)"


def sofa_coagulation(platelets: Optional[float]) -> Optional[Tuple[int, str]]:
    """Platelets ×10³/µL."""
    if not _present(platelets):
        return None
    p = platelets
    pts = 0 if p >= 150 else 1 if p >= 100 else 2 if p >= 50 else 3 if p >= 20 else 4
    return pts, f"platelets {p:.0f}×10³/µL"


def sofa_liver(bilirubin: Optional[float]) -> Optional[Tuple[int, str]]:
    """Total bilirubin mg/dL."""
    if not _present(bilirubin):
        return None
    b = bilirubin
    pts = 0 if b < 1.2 else 1 if b < 2.0 else 2 if b < 6.0 else 3 if b < 12.0 else 4
    return pts, f"bilirubin {b:.1f} mg/dL"


def sofa_cardiovascular(mapv: Optional[float],
                        vaso: Optional[float] = None) -> Optional[Tuple[int, str]]:
    """MAP (mmHg) + any vasopressor requirement (µg/kg/min, norepi-equivalent)."""
    if not _present(mapv) and not _present(vaso):
        return None
    if _present(vaso) and vaso > 0:
        pts = 4 if vaso > 0.1 else 3       # high- vs low-dose pressor
        return pts, f"vasopressor {vaso:.2f} µg/kg/min"
    pts = 1 if (_present(mapv) and mapv < 70) else 0
    return pts, f"MAP {mapv:.0f} mmHg" if _present(mapv) else "no pressor"


def sofa_cns(gcs: Optional[float]) -> Optional[Tuple[int, str]]:
    if not _present(gcs):
        return None
    g = gcs
    pts = 0 if g >= 15 else 1 if g >= 13 else 2 if g >= 10 else 3 if g >= 6 else 4
    return pts, f"GCS {int(g)}/15"


def sofa_renal(creatinine: Optional[float]) -> Optional[Tuple[int, str]]:
    """Creatinine mg/dL."""
    if not _present(creatinine):
        return None
    c = creatinine
    pts = 0 if c < 1.2 else 1 if c < 2.0 else 2 if c < 3.5 else 3 if c < 5.0 else 4
    return pts, f"creatinine {c:.1f} mg/dL"


def _sev_from_sofa(pts: int, cap: int = 4) -> float:
    """Map SOFA points to the continuous 0–1 severity the twin renders."""
    return _clamp(pts / 4.0)


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #
def build_body_state(
    vitals: Dict[str, float],
    detection: Dict[str, Any],
    twin_features: Dict[str, Any],
    risk_level: str,
    timestamp: float,
    camera_vitals: Optional[Dict[str, Any]] = None,
    clinical_history: Optional[Dict[str, Any]] = None,
    infection_suspected: bool = True,
) -> BodyState:
    """Construct the full body-state model + SOFA/qSOFA/MODS aggregates."""
    history = clinical_history or {}
    g = lambda k, d=None: vitals.get(k, d)

    hr, hrv_v = g("heart_rate"), g("hrv")
    spo2, resp = g("spo2"), g("respiratory_rate")
    sbp, mapv = g("systolic_bp"), g("map")
    temp, lactate = g("skin_temp"), g("sweat_lactate")
    vaso = g("vasopressor_dose", history.get("vasopressor_dose"))
    gcs = history.get("gcs")
    creatinine = history.get("creatinine")
    bilirubin = history.get("bilirubin")
    platelets = history.get("platelets")
    wbc = g("wbc", history.get("wbc"))
    perfusion = (camera_vitals or {}).get("perfusion_index")

    systems: List[OrganSystem] = []

    def sofa_system(name: str, region: str, sofa: Optional[Tuple[int, str]],
                    extra_sev: float = 0.0, extra_drivers: Optional[List[str]] = None) -> None:
        if sofa is None and extra_sev <= 0.0 and not _present(extra_sev):
            systems.append(OrganSystem(name, region, False, "UNMONITORED", 0.0,
                                       ["requires lab/vital not currently streamed"]))
            return
        pts = sofa[0] if sofa else None
        sev = max(_sev_from_sofa(pts) if pts is not None else 0.0, extra_sev)
        drivers = list(extra_drivers or [])
        if sofa and pts and pts > 0:
            drivers.insert(0, f"SOFA {pts} — {sofa[1]}")
        systems.append(OrganSystem(
            name, region, True, _tier_from_severity(sev), round(sev, 3), drivers,
            sofa_points=pts, sofa_basis=(sofa[1] if sofa else None)))

    # --- Respiratory (SpO2 proxy + tachypnea) ------------------------------ #
    resp_sofa = sofa_respiration(spo2)
    resp_extra, resp_drivers = 0.0, []
    if _present(resp):
        resp_extra = _clamp((resp - 20.0) / 16.0)
        if resp > 22:
            resp_drivers.append(f"respiratory rate {resp:.0f}/min")
    if resp_sofa is not None or _present(resp):
        sofa_system("Respiratory", "lungs", resp_sofa, resp_extra, resp_drivers)
    else:
        systems.append(OrganSystem("Respiratory", "lungs", False, "UNMONITORED", 0.0))

    # --- Cardiovascular (MAP / vasopressor — pure SOFA) -------------------- #
    # (lactate is scored under its own Metabolic axis below, not double-counted.)
    cardio_sofa = sofa_cardiovascular(mapv, vaso)
    if cardio_sofa is None and _present(sbp):
        cardio_sofa = sofa_cardiovascular(sbp * 0.7, None)   # SBP-derived MAP fallback
    if cardio_sofa is not None:
        sofa_system("Cardiovascular", "heart", cardio_sofa)
    else:
        systems.append(OrganSystem("Cardiovascular", "heart", False, "UNMONITORED", 0.0))

    # --- CNS (GCS) --------------------------------------------------------- #
    cns_sofa = sofa_cns(gcs)
    if cns_sofa is not None:
        sofa_system("Neurological", "brain", cns_sofa)
    else:
        systems.append(OrganSystem("Neurological", "brain", False, "UNMONITORED", 0.0))

    # --- Renal / Hepatic / Coagulation (lab-driven SOFA) ------------------- #
    for name, region, sofa in (
        ("Renal", "kidneys", sofa_renal(creatinine)),
        ("Hepatic", "liver", sofa_liver(bilirubin)),
        ("Coagulation", "blood", sofa_coagulation(platelets)),
    ):
        if sofa is not None:
            sofa_system(name, region, sofa)
        else:
            systems.append(OrganSystem(name, region, False, "UNMONITORED", 0.0,
                                       ["requires lab values (not from wearable/camera)"]))

    # --- Autonomic (HRV — camera/wearable native, key sepsis signal) ------- #
    if _present(hrv_v):
        sev = _clamp((45.0 - hrv_v) / 35.0)
        systems.append(OrganSystem("Autonomic", "heart", True,
                                   _tier_from_severity(sev), round(sev, 3),
                                   [f"HRV {hrv_v:.0f} ms (suppressed)"] if hrv_v < 30 else []))
    else:
        systems.append(OrganSystem("Autonomic", "heart", False, "UNMONITORED", 0.0))

    # --- Peripheral perfusion / skin (camera perfusion + temp) ------------- #
    if _present(perfusion) or _present(temp):
        sev, drivers = 0.0, []
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

    # --- Metabolic (lactate — perfusion / acid-base axis; continuous) ------ #
    if _present(lactate):
        sev = _clamp((lactate - 2.0) / 4.0)
        systems.append(OrganSystem("Metabolic", "metabolic", True,
                                   _tier_from_severity(sev), round(sev, 3),
                                   [f"lactate {lactate:.1f} mmol/L (hyperlactataemia)"] if lactate >= 2 else []))
    else:
        systems.append(OrganSystem("Metabolic", "metabolic", False, "UNMONITORED", 0.0))

    # --- Immune / inflammatory (WBC + temperature — SIRS host response) ---- #
    febrile = _present(temp) and (temp >= 38.0 or temp <= 36.0)
    if _present(wbc) or febrile:
        sev, drivers = 0.0, []
        if _present(wbc):
            sev = max(sev, _clamp((wbc - 12.0) / 12.0), _clamp((4.0 - wbc) / 3.0))
            if wbc > 12:
                drivers.append(f"leukocytosis (WBC {wbc:.1f}×10³)")
            elif wbc < 4:
                drivers.append(f"leukopenia (WBC {wbc:.1f}×10³ — worse in sepsis)")
        if febrile:
            sev = max(sev, 0.35)
            drivers.append(f"{'fever' if temp >= 38 else 'hypothermia'} {temp:.1f}°C")
        systems.append(OrganSystem("Immune", "immune", True,
                                   _tier_from_severity(sev), round(sev, 3), drivers))
    else:
        systems.append(OrganSystem("Immune", "immune", False, "UNMONITORED", 0.0))

    # --- anatomical regions: aggregate systems by region ------------------- #
    regions: Dict[str, Dict[str, Any]] = {}
    for s in systems:
        r = regions.setdefault(s.region, {"severity": 0.0, "monitored": False, "systems": []})
        r["systems"].append(s.name)
        if s.monitored:
            r["monitored"] = True
            r["severity"] = max(r["severity"], s.severity)
    for r in regions.values():
        r["severity"] = round(r["severity"], 3)
        r["status"] = _tier_from_severity(r["severity"]) if r["monitored"] else "UNMONITORED"

    # --- SOFA / qSOFA / MODS aggregates ------------------------------------ #
    sofa_agg = _aggregate_sofa(systems)
    qsofa_agg = _qsofa(resp, sbp, gcs)
    mods_agg = _mods(systems)
    sepsis3 = {
        "organ_dysfunction": sofa_agg["total"] >= 2,
        "criterion": "acute rise in total SOFA ≥ 2 with suspected infection",
        "infection_suspected": bool(infection_suspected),
        "flag": bool(sofa_agg["total"] >= 2 and infection_suspected),
        "note": "Sepsis-3 needs a BASELINE SOFA; total≥2 shown as current dysfunction, "
                "not a confirmed acute delta.",
    }

    channels = _signal_channels(vitals, camera_vitals)
    return BodyState(
        timestamp=timestamp,
        overall_risk=risk_level,
        sepsis_probability=detection.get("ml_probability", detection.get("sepsis_probability")),
        organ_systems=systems,
        regions=regions,
        channels=channels,
        sofa=sofa_agg,
        qsofa=qsofa_agg,
        mods=mods_agg,
        sepsis3=sepsis3,
    )


def _aggregate_sofa(systems: List[OrganSystem]) -> Dict[str, Any]:
    per: Dict[str, int] = {}
    total = 0
    monitored_organs = 0
    for s in systems:
        if s.sofa_points is not None:
            per[s.name] = s.sofa_points
            total += s.sofa_points
            monitored_organs += 1
    worst = max((s for s in systems if s.sofa_points is not None),
                key=lambda s: s.sofa_points, default=None)
    return {
        "total": total,
        "per_organ": per,
        "organs_scored": monitored_organs,
        "organs_possible": 6,
        "worst_organ": worst.name if worst else None,
        "mortality_band": _sofa_mortality_band(total),
    }


def _sofa_mortality_band(total: int) -> str:
    """Coarse published SOFA→ICU-mortality banding (Vincent et al.); indicative."""
    if total <= 1:
        return "<10%"
    if total <= 6:
        return "~10–20%"
    if total <= 9:
        return "~15–30%"
    if total <= 12:
        return "~40–50%"
    return ">50%"


def _qsofa(resp: Optional[float], sbp: Optional[float], gcs: Optional[float]) -> Dict[str, Any]:
    comps = {
        "resp_ge_22": bool(_present(resp) and resp >= 22),
        "sbp_le_100": bool(_present(sbp) and sbp <= 100),
        "gcs_lt_15": bool(_present(gcs) and gcs < 15),
    }
    score = sum(comps.values())
    return {"score": score, "components": comps, "high_risk": score >= 2,
            "monitored": _present(resp) or _present(sbp) or _present(gcs)}


def _mods(systems: List[OrganSystem]) -> Dict[str, Any]:
    """Multiple Organ Dysfunction Syndrome: how many systems are failing at once."""
    failing = [s for s in systems if s.monitored and s.severity >= DYSFUNCTION_SEV]
    monitored = [s for s in systems if s.monitored]
    count = len(failing)
    total_sev = sum(s.severity for s in failing)
    # cascade risk rises with the NUMBER of failing organs (super-additive) + their depth
    cascade = _clamp(0.28 * count + 0.12 * total_sev)
    dominant = max(failing, key=lambda s: s.severity, default=None)
    return {
        "count": count,
        "systems": [s.name for s in failing],
        "monitored_systems": len(monitored),
        "dominant": dominant.name if dominant else None,
        "cascade_risk": round(cascade, 3),
        "is_mods": count >= 2,      # clinical MODS = dysfunction in ≥2 systems
    }


def _lab_system(name: str, region: str, value: Optional[float]) -> OrganSystem:
    """Kept for backward compatibility (unused by build_body_state now)."""
    if not _present(value):
        return OrganSystem(name, region, False, "UNMONITORED", 0.0,
                           ["requires lab values (not available from wearable/camera)"])
    return OrganSystem(name, region, True, "NORMAL", 0.0)


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
    for key, label in (("perfusion_index", "Facial Perfusion (optical)"),
                       ("stress_vector", "Sympathetic Stress (optical)")):
        val = cam.get(key)
        out.append(SignalChannel(key, label, (round(float(val), 3) if _present(val) else None),
                                 "idx", 0.5, 1.0, "NORMAL" if _present(val) else "UNMONITORED",
                                 "camera"))
    return out


# --------------------------------------------------------------------------- #
# OrganFailureTracker — the trajectory / early-failure layer (stateful)
# --------------------------------------------------------------------------- #
class OrganFailureTracker:
    """Per-patient trajectory tracker over successive body states.

    Feeds each organ's severity into a rolling window and, per update, annotates
    every monitored organ with a trend (RISING/FALLING/STABLE), a slope
    (Δseverity per minute), a short-horizon projection, and — when RISING toward
    failure — an ETA to CRITICAL. It also edge-detects **MODS onset** (the tick at
    which the failing-organ count first reaches ≥2). This is the difference between
    reporting an organ that *has* failed and flagging one that is *about to*.
    """

    CRITICAL_SEV = 0.75

    def __init__(self, window: int = 12, horizon_min: float = 15.0,
                 rising_eps: float = 0.01):
        self.window = window
        self.horizon_min = horizon_min
        self.rising_eps = rising_eps        # min |slope/min| to call a trend
        self._hist: Dict[str, Deque[Tuple[float, float]]] = {}   # organ -> (t_min, sev)
        self._prev_mods = 0
        self.mods_onset_ts: Optional[float] = None
        self.updates = 0

    def _slope(self, series: Deque[Tuple[float, float]]) -> float:
        """Least-squares d(sev)/d(min) over the window."""
        if len(series) < 2:
            return 0.0
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 1e-9:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den

    def update(self, body: BodyState) -> BodyState:
        """Annotate ``body``'s organ systems in place with trajectory fields and
        record MODS-onset. Returns the same BodyState for chaining."""
        self.updates += 1
        t_min = body.timestamp / 60.0
        for s in body.organ_systems:
            if not s.monitored:
                continue
            series = self._hist.setdefault(s.name, deque(maxlen=self.window))
            series.append((t_min, s.severity))
            slope = self._slope(series)
            s.slope_per_min = round(slope, 5)
            if slope > self.rising_eps:
                s.trend = "RISING"
            elif slope < -self.rising_eps:
                s.trend = "FALLING"
            else:
                s.trend = "STABLE"
            s.projected_severity = round(_clamp(s.severity + slope * self.horizon_min), 3)
            if s.trend == "RISING" and s.severity < self.CRITICAL_SEV and slope > 1e-6:
                eta = (self.CRITICAL_SEV - s.severity) / slope
                s.eta_critical_min = round(eta, 1) if 0 < eta <= 24 * 60 else None
            else:
                s.eta_critical_min = None

        # MODS-onset edge detection + trajectory summary folded into body.mods.
        count = body.mods.get("count", 0)
        onset = count >= 2 and self._prev_mods < 2
        if onset:
            self.mods_onset_ts = body.timestamp
        self._prev_mods = count
        rising = [s.name for s in body.organ_systems if s.trend == "RISING"]
        soonest = min((s.eta_critical_min for s in body.organ_systems
                       if s.eta_critical_min is not None), default=None)
        body.mods.update({
            "onset_now": bool(onset),
            "onset_ts": self.mods_onset_ts,
            "worsening": rising,
            "worsening_count": len(rising),
            "soonest_eta_critical_min": soonest,
        })
        return body
