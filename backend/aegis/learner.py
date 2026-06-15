"""Per-patient behavioural learning — personalized deviation detection.

Population thresholds (qSOFA/SIRS) ask "is this value abnormal for *people*?".
But sepsis often shows first as a patient drifting away from *their own* stable
baseline while still inside the population-normal band — a runner whose resting
HR climbs from 52 to 78 is deteriorating, yet 78 trips no population alarm. And a
patient whose baseline is chronically deranged (COPD, CHF) trips population
alarms constantly, drowning the real signal in false positives.

:class:`PatientBaselineLearner` learns each patient's individual behaviour online
and scores **personalized deviation** — how far the patient is from their *own*
established baseline, in their own units (robust z-scores). The baseline is built
from a short warm-up, then adapted slowly and **gated**: when the patient is
clearly deviating the baseline freezes, so a slow deterioration can't quietly
drag the reference along with it (the boiling-frog failure).

:func:`personalized_risk` blends this with the population fusion risk so the
personalization helps most exactly where the population model is blind — a
low-population-risk patient who is nonetheless abnormal *for themselves*.

Honesty: this is a transparent online baseline + deviation model, validated
against the ground-truth labels (see verify_learner.py / the Validation view);
its real effect on lead time is measured and reported, not assumed.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

SCORED = ["hr", "temp_c", "resp_rate", "spo2", "map", "wbc", "lactate"]
# direction (per channel) that increases sepsis risk; temp is handled two-sided
_WORSE = {"hr": 1, "resp_rate": 1, "wbc": 1, "lactate": 1, "map": -1, "spo2": -1}
# measurement-noise floors on the personal std so a flat channel can't blow up z
_FLOOR = {"hr": 3.0, "temp_c": 0.2, "resp_rate": 1.5, "spo2": 1.0,
          "map": 3.0, "wbc": 0.8, "lactate": 0.3}
WARMUP = 6
MATURE_AT = 12


def _clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v


class PatientBaselineLearner:
    def __init__(self):
        self.mean: Dict[str, float] = {}
        self.var: Dict[str, float] = {}
        self.n = 0
        self.dev_ewma = 0.0          # smoothed deviation — sustained, not transient

    @property
    def maturity(self) -> float:
        return min(1.0, self.n / MATURE_AT)

    def _std(self, ch: str) -> float:
        return max(math.sqrt(max(self.var.get(ch, 0.0), 0.0)), _FLOOR[ch])

    def _z(self, ch: str, v: float) -> float:
        if ch not in self.mean:
            return 0.0
        zr = (v - self.mean[ch]) / self._std(ch)
        if ch == "temp_c":
            return abs(zr)                       # any temp shift from baseline
        return max(0.0, _WORSE[ch] * zr)         # only the worsening direction

    def per_channel(self, vals: Dict[str, float]) -> Dict[str, float]:
        return {ch: round(self._z(ch, float(vals[ch])), 2)
                for ch in SCORED if ch in vals}

    def _raw_deviation(self, vals: Dict[str, float]) -> float:
        """Instantaneous deviation = mean of the patient's top-3 channel z-scores."""
        z = sorted((self._z(ch, float(vals[ch])) for ch in SCORED if ch in vals),
                   reverse=True)
        return sum(z[:3]) / 3.0 if z else 0.0

    @property
    def deviation(self) -> float:
        """Sustained (smoothed) deviation — what the risk acts on. A transient
        one-hour spike is damped; only a deviation that persists builds up."""
        return round(self.dev_ewma, 3)

    def observe(self, vals: Dict[str, float]) -> None:
        self.n += 1
        raw = self._raw_deviation(vals) if self.n > WARMUP else 0.0
        if self.n > WARMUP:
            self.dev_ewma = 0.32 * raw + 0.68 * self.dev_ewma     # require persistence
        a = (1.0 / self.n) if self.n <= WARMUP else (0.06 if raw < 1.2 else 0.008)
        for ch in SCORED:
            if ch not in vals:
                continue
            v = float(vals[ch])
            if ch not in self.mean:
                self.mean[ch] = v
                self.var[ch] = 0.0
                continue
            d = v - self.mean[ch]
            self.mean[ch] += a * d
            self.var[ch] = (1.0 - a) * (self.var[ch] + a * d * d)

    def snapshot(self, vals: Dict[str, float]) -> Dict[str, Any]:
        pc = self.per_channel(vals)
        top = sorted(pc.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "deviation": self.deviation,
            "maturity": round(self.maturity, 2),
            "per_channel": pc,
            "top": [{"ch": k, "z": v, "baseline": round(self.mean.get(k, 0.0), 1)}
                    for k, v in top[:3] if v > 0.4],
        }


def personalized_risk(pop: float, deviation: float, maturity: float) -> float:
    """Modulate the population fusion risk by how far the patient is from *their
    own* baseline. Two-sided, which is the whole point:

      * at baseline (deviation ≈ 0) the population risk is DAMPENED — a patient
        whose vitals are chronically deranged but unchanged for them no longer
        trips a constant alarm (fewer false positives);
      * deviating from baseline LIFTS risk toward and above the population value —
        a patient drifting from their own norm is caught early, even while still
        inside the population-normal band.

    Personalization is gated by baseline maturity: until the baseline is learned
    the system simply trusts the population model.
    """
    # deviation floor of 0.4 ignores ordinary baseline fluctuation; only a clear,
    # sustained departure from the patient's own norm drives the personalization.
    dfac = min(max(0.0, deviation - 0.4) / 2.2, 1.0) * maturity   # 0 = at baseline
    pers = pop * (0.62 + 0.38 * dfac) + (1.0 - pop) * 0.30 * dfac
    blended = maturity * pers + (1.0 - maturity) * pop  # immature -> trust population
    return round(_clamp(blended, 0.0, 1.0), 4)


class CohortLearner:
    """One PatientBaselineLearner per bed."""

    def __init__(self):
        self._by: Dict[str, PatientBaselineLearner] = {}

    def get(self, bed_id: str) -> PatientBaselineLearner:
        if bed_id not in self._by:
            self._by[bed_id] = PatientBaselineLearner()
        return self._by[bed_id]
