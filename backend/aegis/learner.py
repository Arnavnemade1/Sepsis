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

# Population reference points used ONLY to put what the model learned into words —
# whether this patient's *own* learned baseline runs high/low/normal vs people, and
# how tightly each channel is regulated. (The risk model itself never uses these;
# they are for the human-readable description of learned tendencies.)
_CENTER = {"hr": 75, "temp_c": 37.0, "resp_rate": 16, "spo2": 97, "map": 85,
           "wbc": 8.0, "lactate": 1.0}
_HI = {"hr": 90, "temp_c": 37.5, "resp_rate": 21, "spo2": 101, "map": 102,
       "wbc": 11.5, "lactate": 2.2}
_LO = {"hr": 58, "temp_c": 36.3, "resp_rate": 9, "spo2": 94, "map": 72,
       "wbc": 3.8, "lactate": -1.0}
_LABILE = {"hr": 11, "temp_c": 0.55, "resp_rate": 4.0, "spo2": 2.5, "map": 9.0,
           "wbc": 2.5, "lactate": 0.8}
_UNIT = {"hr": "", "temp_c": "°C", "resp_rate": "/min", "spo2": "%", "map": "",
         "wbc": "", "lactate": ""}
_NAME = {"hr": "heart rate", "temp_c": "temperature", "resp_rate": "respiratory rate",
         "spo2": "SpO₂", "map": "MAP", "wbc": "WBC", "lactate": "lactate"}
_PHRASE = {
    ("hr", "hi"): "runs tachycardic", ("hr", "lo"): "runs bradycardic",
    ("map", "lo"): "runs hypotensive", ("map", "hi"): "runs hypertensive",
    ("spo2", "lo"): "chronically low SpO₂ (possible chronic lung disease)",
    ("resp_rate", "hi"): "baseline tachypnea",
    ("temp_c", "lo"): "runs cool", ("temp_c", "hi"): "runs warm",
    ("wbc", "hi"): "chronic leukocytosis", ("wbc", "lo"): "baseline leukopenia",
    ("lactate", "hi"): "elevated baseline lactate",
}


def _clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v


def _fmt(v: float) -> str:
    return str(int(round(v))) if abs(v) >= 10 else f"{v:.1f}"


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

    def profile(self) -> List[Dict[str, Any]]:
        """What the model has LEARNED about this patient — each channel's personal
        baseline, its variability, and whether that baseline runs high/low/normal
        and tightly- or loosely-regulated *for this individual* vs the population."""
        out: List[Dict[str, Any]] = []
        if self.n <= WARMUP:
            return out
        for ch in SCORED:
            if ch not in self.mean:
                continue
            base, std = self.mean[ch], self._std(ch)
            level = "high" if base >= _HI[ch] else "low" if base <= _LO[ch] else "normal"
            labile = std >= _LABILE[ch]
            tight = std <= _FLOOR[ch] * 1.4
            note = (_PHRASE.get((ch, "hi"), "") if level == "high"
                    else _PHRASE.get((ch, "lo"), "") if level == "low" else "")
            out.append({
                "ch": ch, "name": _NAME[ch], "baseline": round(base, 2),
                "std": round(std, 2), "unit": _UNIT[ch], "level": level,
                "labile": labile, "tight": tight, "note": note,
            })
        return out

    def narrative(self, vals: Dict[str, float]) -> Dict[str, Any]:
        """A plain-English account of the learned baseline + the patient's current
        departures from it — the dossier reads this verbatim."""
        if self.n <= WARMUP + 2:
            return {"status": "learning", "n_hours": self.n, "learned": [],
                    "drifting": [], "summary": f"Still learning this patient's "
                    f"baseline ({self.n}h observed)."}
        prof = self.profile()
        learned: List[str] = []
        for p in prof:                                  # notable tendencies first
            if not p["note"]:
                continue
            phr = p["note"] + (f" with a labile {p['name']}" if p["labile"] else "")
            learned.append(f"{phr[:1].upper()}{phr[1:]} — baseline "
                           f"{_fmt(p['baseline'])}{p['unit']}.")
        for p in prof:                                  # one "tightly regulated" note
            if p["tight"] and p["level"] == "normal" and p["ch"] in ("map", "hr"):
                learned.append(f"Tightly-regulated {p['name']} "
                               f"(baseline {_fmt(p['baseline'])}{p['unit']}, ±{_fmt(p['std'])}).")
                break
        if not learned:
            learned.append("Baseline within population norms — a physiologically "
                           "unremarkable patient so far.")
        drifting: List[str] = []
        for ch, z in sorted(self.per_channel(vals).items(), key=lambda kv: -kv[1]):
            if z >= 0.8 and ch in self.mean:
                dirn = ("below" if _WORSE.get(ch, 1) == -1
                        else "off" if ch == "temp_c" else "above")
                drifting.append(f"{_NAME[ch]} {z:.1f}σ {dirn} own baseline "
                                f"(now {_fmt(float(vals[ch]))} vs "
                                f"{_fmt(self.mean[ch])}{_UNIT[ch]})")
        summary = "; ".join(learned[:2])
        return {"status": "mature", "n_hours": self.n,
                "learned": learned[:3], "drifting": drifting[:3], "summary": summary}

    def snapshot(self, vals: Dict[str, float]) -> Dict[str, Any]:
        pc = self.per_channel(vals)
        top = sorted(pc.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "deviation": self.deviation,
            "maturity": round(self.maturity, 2),
            "per_channel": pc,
            "top": [{"ch": k, "z": v, "baseline": round(self.mean.get(k, 0.0), 1)}
                    for k, v in top[:3] if v > 0.4],
            "learned": self.narrative(vals),
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
