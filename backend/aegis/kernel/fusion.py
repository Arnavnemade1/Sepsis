"""The Core Data Fusion Engine — the Sepsis Ontology.

Sepsis does not announce itself on a single channel. It is a *trajectory* that
shows up first as subtle autonomic changes (heart-rate variability collapse,
microvascular vasomotion loss) seen by the high-frequency optical edge, and only
later as the slow-moving chemistry (rising lactate, deranged white-cell count)
that the enterprise lab tier reports. AegisOS binds both tiers, per bed, into one
continuously-updated risk profile.

Two ingress tiers, both landing in the Layer-2 memory fabric:

  * **Edge tier (high frequency).** Optical rPPG → heart rate + perfusion, already
    conditioned by the inline Kalman in `/dev/bio`. From a rolling window the
    engine derives **HRV** (RMSSD of the beat-to-beat interval) and **vasomotion**
    (perfusion oscillation amplitude) — autonomic markers no single sample shows.
  * **Enterprise tier (low frequency).** MIMIC-IV/eICU-shaped structured labs and
    vitals — temperature, respiratory rate, SpO2, mean arterial pressure, white
    blood cell count, serum lactate. From their windows the engine derives **ABP
    variability** and **lactate clearance** (is lactate falling or climbing?).

The fusion (`_fuse`) is a transparent, qSOFA/SIRS-grounded weighted model — not a
black box — so every contribution can be named back to the clinician. It emits a
fused risk in [0,1], four organ-system component scores, and a ranked list of the
specific derangements driving the score (which the neurosymbolic terminal later
turns into the Hour-1 bundle note).

Honesty: with no live database credentials the enterprise streams are synthetic,
MIMIC-IV/eICU-*shaped* — the seam where a credentialed adapter drops in. The
fusion math, HRV/vasomotion/clearance derivations and the ontology are real.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

# Clinical thresholds (SIRS / qSOFA / Sepsis-3 flavored).
HR_TACHY = 90.0
RR_TACHYPNEA = 20.0
TEMP_FEVER, TEMP_HYPO = 38.0, 36.0
MAP_HYPOTENSION = 65.0
SPO2_HYPOXIA = 92.0
WBC_HIGH, WBC_LOW = 12.0, 4.0
LACTATE_ELEVATED, LACTATE_SEVERE = 2.0, 4.0
HRV_SUPPRESSED = 18.0           # ms; below this is autonomic suppression


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class BedProfile:
    """The fused, inspectable risk profile for one bed."""
    bed_id: str
    name: str
    fused_risk: float
    edge: Dict[str, float]              # hr, hrv_ms, perfusion, vasomotion
    enterprise: Dict[str, float]        # temp_c, resp_rate, spo2, map, abp_var, wbc, lactate, lactate_clearance
    components: Dict[str, float]        # cardiac, inflammatory, perfusion, metabolic
    drivers: List[str]                  # ranked human-readable derangements
    primary_system: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bed_id": self.bed_id, "name": self.name,
            "fused_risk": round(self.fused_risk, 4),
            "edge": {k: round(v, 3) for k, v in self.edge.items()},
            "enterprise": {k: round(v, 3) for k, v in self.enterprise.items()},
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "drivers": self.drivers, "primary_system": self.primary_system,
        }


class FusionEngine:
    """Stateful per-bed fusion: maintains rolling windows of the edge + enterprise
    signals and binds them into a :class:`BedProfile` every tick."""

    def __init__(self, window: int = 48):
        self.window = window
        self._w: Dict[str, Dict[str, Deque[float]]] = {}

    def _win(self, bed: str, key: str) -> Deque[float]:
        b = self._w.setdefault(bed, {})
        if key not in b:
            b[key] = deque(maxlen=self.window)
        return b[key]

    # -- derived edge-tier features ------------------------------------- #
    @staticmethod
    def hrv_rmssd(hr_series: Deque[float]) -> float:
        """RMSSD (ms) from beat-to-beat intervals derived from instantaneous HR."""
        if len(hr_series) < 3:
            return 35.0
        rr = [60000.0 / max(hr, 1.0) for hr in hr_series]      # ms intervals
        diffs = [rr[i + 1] - rr[i] for i in range(len(rr) - 1)]
        ms2 = sum(d * d for d in diffs) / len(diffs)
        return math.sqrt(ms2)

    @staticmethod
    def vasomotion(perf_series: Deque[float]) -> float:
        """Perfusion oscillation amplitude (a.u.) — microvascular tone."""
        if len(perf_series) < 3:
            return 0.0
        m = sum(perf_series) / len(perf_series)
        var = sum((p - m) ** 2 for p in perf_series) / len(perf_series)
        return math.sqrt(var)

    @staticmethod
    def variability(series: Deque[float]) -> float:
        if len(series) < 3:
            return 0.0
        m = sum(series) / len(series)
        return math.sqrt(sum((x - m) ** 2 for x in series) / len(series))

    @staticmethod
    def lactate_clearance(series: Deque[float]) -> float:
        """% change of lactate across the window. Positive = clearing (good),
        negative = accumulating (poor clearance = worsening perfusion)."""
        if len(series) < 2:
            return 0.0
        first, last = series[0], series[-1]
        if first <= 1e-6:
            return 0.0
        return (first - last) / first * 100.0

    # -- the bind ------------------------------------------------------- #
    def update(self, bed_id: str, name: str,
               edge: Dict[str, float], enterprise: Dict[str, float]) -> BedProfile:
        hr = float(edge.get("hr", 82.0))
        perf = float(edge.get("perfusion", 1.4))
        self._win(bed_id, "hr").append(hr)
        self._win(bed_id, "perf").append(perf)
        self._win(bed_id, "map").append(float(enterprise.get("map", 88.0)))
        self._win(bed_id, "lactate").append(float(enterprise.get("lactate", 1.0)))

        hrv = self.hrv_rmssd(self._win(bed_id, "hr"))
        vaso = self.vasomotion(self._win(bed_id, "perf"))
        abp_var = self.variability(self._win(bed_id, "map"))
        clearance = self.lactate_clearance(self._win(bed_id, "lactate"))

        e = {"hr": hr, "hrv_ms": hrv, "perfusion": perf, "vasomotion": vaso}
        ent = {
            "temp_c": float(enterprise.get("temp_c", 37.0)),
            "resp_rate": float(enterprise.get("resp_rate", 16.0)),
            "spo2": float(enterprise.get("spo2", 98.0)),
            "map": float(enterprise.get("map", 88.0)),
            "abp_var": abp_var,
            "wbc": float(enterprise.get("wbc", 8.0)),
            "lactate": float(enterprise.get("lactate", 1.0)),
            "lactate_clearance": clearance,
        }
        risk, comps, drivers, primary = self._fuse(e, ent)
        return BedProfile(bed_id, name, risk, e, ent, comps, drivers, primary)

    def _fuse(self, e, ent):
        drivers: List[Dict[str, Any]] = []
        def flag(sys, weight, text):
            drivers.append({"sys": sys, "w": weight, "text": text})

        # --- cardiac (circulatory rhythm) ---
        # NOTE: HRV (RMSSD) and vasomotion are derived from the Kalman-conditioned
        # optical stream, which smooths beat-to-beat variability — so they are
        # surfaced as autonomic *markers* in the dossier but are NOT scored here.
        # Risk rests on the validated qSOFA/SIRS + Sepsis-3 lactate terms.
        cardiac = 0.0
        if e["hr"] > HR_TACHY:
            t = (e["hr"] - HR_TACHY) / 40.0; cardiac += t
            flag("cardiac", t, f"Tachycardia {e['hr']:.0f} bpm")
        if ent["abp_var"] > 8.0:
            t = (ent["abp_var"] - 8.0) / 8.0; cardiac += 0.5 * t
            flag("cardiac", 0.5 * t, f"Labile arterial pressure (σ {ent['abp_var']:.0f})")

        # --- inflammatory ---
        inflam = 0.0
        if ent["temp_c"] > TEMP_FEVER:
            t = (ent["temp_c"] - TEMP_FEVER) / 2.5; inflam += t
            flag("inflammatory", t, f"Fever {ent['temp_c']:.1f}°C")
        elif ent["temp_c"] < TEMP_HYPO:
            t = (TEMP_HYPO - ent["temp_c"]) / 2.5; inflam += t
            flag("inflammatory", t, f"Hypothermia {ent['temp_c']:.1f}°C")
        if ent["wbc"] > WBC_HIGH:
            t = (ent["wbc"] - WBC_HIGH) / 8.0; inflam += t
            flag("inflammatory", t, f"Leukocytosis WBC {ent['wbc']:.1f}")
        elif ent["wbc"] < WBC_LOW:
            t = (WBC_LOW - ent["wbc"]) / 3.0; inflam += t
            flag("inflammatory", t, f"Leukopenia WBC {ent['wbc']:.1f}")

        # --- perfusion (circulatory) ---
        perf = 0.0
        if ent["map"] < MAP_HYPOTENSION:
            t = (MAP_HYPOTENSION - ent["map"]) / 20.0; perf += 1.2 * t
            flag("perfusion", 1.2 * t, f"Hypotension MAP {ent['map']:.0f} mmHg")
        if ent["spo2"] < SPO2_HYPOXIA:
            t = (SPO2_HYPOXIA - ent["spo2"]) / 8.0; perf += t
            flag("perfusion", t, f"Hypoxia SpO2 {ent['spo2']:.0f}%")

        # --- metabolic ---
        metab = 0.0
        if ent["lactate"] > LACTATE_ELEVATED:
            t = (ent["lactate"] - LACTATE_ELEVATED) / 3.0
            sev = "severe" if ent["lactate"] >= LACTATE_SEVERE else "elevated"
            metab += 1.5 * t
            flag("metabolic", 1.5 * t, f"Lactate {ent['lactate']:.1f} mmol/L ({sev})")
        # poor clearance only matters once lactate is actually elevated — on a
        # normal (~1 mmol/L) lactate the clearance % is just measurement noise.
        if ent["lactate"] > LACTATE_ELEVATED and ent["lactate_clearance"] < -5.0:
            t = min(1.0, -ent["lactate_clearance"] / 30.0); metab += 0.6 * t
            flag("metabolic", 0.6 * t, f"Poor lactate clearance ({ent['lactate_clearance']:.0f}%)")
        if ent["resp_rate"] > RR_TACHYPNEA:
            t = (ent["resp_rate"] - RR_TACHYPNEA) / 12.0; metab += t
            flag("metabolic", t, f"Tachypnea {ent['resp_rate']:.0f}/min")

        s = cardiac + inflam + perf + metab
        risk = _logistic(1.5 * s - 1.3)
        comps = {"cardiac": cardiac, "inflammatory": inflam,
                 "perfusion": perf, "metabolic": metab}
        primary = max(comps, key=comps.get) if s > 0.05 else "none"
        ranked = [d["text"] for d in sorted(drivers, key=lambda d: d["w"], reverse=True)]
        return risk, comps, ranked, primary
