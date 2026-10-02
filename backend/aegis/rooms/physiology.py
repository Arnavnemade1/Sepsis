"""Vitals for the synthetic rooms, and the ways a synthetic patient can deteriorate.

The 12 ICU rooms replay real stays and room 402 is live; every other occupied room
is a synthetic patient, and until now those rooms produced no vitals at all, so
nothing could watch them. This module gives each one a monitor feed:

* the patient's **own baseline** (from the record's earlier encounters), shifted by
  what today's illness does (a pneumonia runs a low fever and needs 2 L of oxygen),
* slow physiological drift (an Ornstein–Uhlenbeck process per channel) plus small
  measurement noise,
* and, now and then, a **deterioration** the record makes plausible: sepsis in an
  infected or post-operative patient, opioid respiratory depression on a PCA,
  rapid AF in a patient with AF, hypoglycaemia on insulin, a bleed on an
  anticoagulant, fluid overload in heart failure, hyperkalaemia in CKD, a COPD
  exacerbation, neutropenic fever. A deterioration ramps in over a few minutes, holds
  until someone has the page in hand, then recovers.

Everything here is synthetic and labelled so; the detectors that read it are the
same ones that read the real ICU replay.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

SAMPLE_S = 10.0                 # one set of vitals every 10 s of wall time
RECOVER_S = (150.0, 240.0)
SELF_LIMIT_S = 12 * 60.0        # a deterioration nobody paged recovers on its own after this

# channel -> (OU sigma per sqrt(s), measurement noise sd)
_NOISE = {"hr": (0.55, 1.0), "sbp": (0.7, 1.6), "rr": (0.16, 0.5), "spo2": (0.10, 0.35), "temp_c": (0.008, 0.03)}
_TAU = 140.0


@dataclass
class Scenario:
    kind: str
    label: str
    t0: float
    ramp: float
    targets: Dict[str, Tuple[str, float]]       # channel -> ("abs" | "add" | "mul", value)
    mental: Optional[Tuple[float, str]] = None  # (progress at which it changes, AVPU letter / "C")
    labs: List[Tuple[float, str, Tuple[str, float]]] = field(default_factory=list)  # (progress, lab, target)
    o2: Optional[float] = None                  # oxygen the nurse starts when sats fall
    recover_t0: Optional[float] = None
    recover_s: float = 200.0
    recover_from: float = 1.0                   # how far it had got when recovery began
    page: Optional[str] = None
    drawn: Dict[str, bool] = field(default_factory=dict)
    observed: bool = False                      # set by a nurse's report, not the simulator

    def progress(self, now: float) -> float:
        if self.recover_t0 is not None:
            r = min(1.0, max(0.0, (now - self.recover_t0) / self.recover_s))
            return self.recover_from * (1.0 - _ease(r))
        return _ease(min(1.0, max(0.0, (now - self.t0) / self.ramp)))

    def done(self, now: float) -> bool:
        return self.recover_t0 is not None and now - self.recover_t0 >= self.recover_s

    def to_dict(self, now: float) -> Dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "started_at": self.t0,
                "phase": "recovering" if self.recover_t0 is not None else
                         "developing" if now - self.t0 < self.ramp else "established",
                "progress": round(self.progress(now), 2), "page": self.page, "observed": self.observed}


def _ease(x: float) -> float:
    return x * x * (3 - 2 * x)


# kind -> (label, fits(flags, orders), builder(rng) -> Scenario kwargs)
def _sepsis(rng: random.Random) -> Dict[str, Any]:
    return {"targets": {"temp_c": ("abs", rng.uniform(38.7, 39.4)), "hr": ("add", rng.uniform(34, 44)),
                        "rr": ("add", rng.uniform(8, 11)), "sbp": ("mul", rng.uniform(0.66, 0.74)),
                        "spo2": ("add", -rng.uniform(3, 5))},
            "mental": (0.8, "C"), "o2": 2.0,
            "labs": [(0.55, "lactate", ("abs", rng.uniform(2.9, 4.4))), (0.55, "wbc", ("abs", rng.uniform(16, 21)))]}


def _opioid(rng: random.Random) -> Dict[str, Any]:
    return {"targets": {"rr": ("abs", rng.uniform(6.5, 8.0)), "spo2": ("add", -rng.uniform(7, 10)),
                        "hr": ("add", -rng.uniform(6, 10)), "sbp": ("mul", 0.92)},
            "mental": (0.55, "V")}


def _af(rng: random.Random) -> Dict[str, Any]:
    return {"targets": {"hr": ("abs", rng.uniform(138, 152)), "sbp": ("mul", rng.uniform(0.80, 0.86)),
                        "rr": ("add", 3)}}


def _hypo(rng: random.Random) -> Dict[str, Any]:
    return {"targets": {"hr": ("add", rng.uniform(16, 24)), "sbp": ("add", 8)}, "mental": (0.6, "C"),
            "labs": [(0.35, "glucose", ("abs", rng.uniform(62, 68))), (0.75, "glucose", ("abs", rng.uniform(41, 49)))]}


def _bleed(rng: random.Random) -> Dict[str, Any]:
    return {"targets": {"hr": ("add", rng.uniform(30, 38)), "sbp": ("mul", rng.uniform(0.68, 0.75)),
                        "rr": ("add", 4)},
            "labs": [(0.65, "hemoglobin", ("add", -rng.uniform(2.3, 3.0)))]}


def _hf(rng: random.Random) -> Dict[str, Any]:
    return {"targets": {"rr": ("add", rng.uniform(10, 13)), "spo2": ("add", -rng.uniform(6, 8)),
                        "hr": ("add", rng.uniform(16, 22)), "sbp": ("add", 14)}, "o2": 4.0}


def _hyperk(rng: random.Random) -> Dict[str, Any]:
    return {"targets": {"hr": ("add", -rng.uniform(16, 22))},
            "labs": [(0.3, "potassium", ("abs", rng.uniform(6.1, 6.6)))]}


def _copd(rng: random.Random) -> Dict[str, Any]:
    return {"targets": {"rr": ("add", rng.uniform(9, 12)), "spo2": ("abs", rng.uniform(83.5, 85.5)),
                        "hr": ("add", rng.uniform(12, 18))}, "o2": 2.0}


def _neutropenic(rng: random.Random) -> Dict[str, Any]:
    return {"targets": {"temp_c": ("abs", rng.uniform(38.5, 38.9)), "hr": ("add", rng.uniform(18, 24)),
                        "rr": ("add", 3)}}


SCENARIOS: Dict[str, Tuple[str, Callable[[set], bool], Callable[[random.Random], Dict[str, Any]]]] = {
    "sepsis": ("Evolving sepsis", lambda f: bool(f & {"infection", "post_op"}) and "neutropenia" not in f, _sepsis),
    "neutropenic_fever": ("Neutropenic fever", lambda f: "neutropenia" in f, _neutropenic),
    "opioid": ("Opioid-induced respiratory depression", lambda f: "opioid" in f, _opioid),
    "af_rvr": ("Atrial fibrillation with rapid ventricular response", lambda f: "af" in f, _af),
    "hypoglycemia": ("Hypoglycaemia", lambda f: "insulin" in f, _hypo),
    "bleed": ("Bleeding on anticoagulation", lambda f: "anticoagulant" in f, _bleed),
    "hf_overload": ("Fluid overload", lambda f: "hf" in f, _hf),
    "hyperkalemia": ("Hyperkalaemia", lambda f: "ckd" in f, _hyperk),
    "copd_exac": ("COPD exacerbation", lambda f: "copd" in f, _copd),
}


# how often each course is chosen when a room fits several (sepsis is what this OS is for)
WEIGHTS = {"sepsis": 3.0, "opioid": 2.0, "copd_exac": 1.5, "hf_overload": 1.5, "af_rvr": 1.5, "hypoglycemia": 1.5,
           "neutropenic_fever": 1.5, "hyperkalemia": 1.0, "bleed": 1.0}


def fitting(flags: set) -> List[str]:
    return [k for k, (_l, fits, _b) in SCENARIOS.items() if fits(flags)]


class Physiology:
    """One synthetic patient's monitor feed."""

    def __init__(self, record: Dict[str, Any], seed: int, now: float) -> None:
        self.rng = random.Random(seed)
        b = record["baseline"]
        off = record.get("admission_state") or {}
        self.flags = set(record.get("flags") or [])
        self.base = {k: float(b[k]) + float(off.get(k, 0.0)) for k in ("hr", "sbp", "rr", "spo2", "temp_c")}
        self.base["dbp"] = float(b["dbp"])
        if "copd" in self.flags:          # at the prescribed target on oxygen
            self.base["spo2"] = min(self.base["spo2"] + (1.0 if off.get("o2") else 0.0), 92.0)
        self.base_o2 = float(off.get("o2", 0.0))
        self.chronic_confusion = (b.get("mental") == "confused at baseline")
        self.ou = {k: 0.0 for k in _NOISE}
        self.scenario: Optional[Scenario] = None
        self.cooldown_until = 0.0
        self.last_t = now
        self.labs: Dict[str, List[Dict[str, float]]] = {}
        self.glucose_base = None
        g = (record.get("labs") or {}).get("glucose")
        if g:
            self.glucose_base = float(g[-1]["v"])
            self.next_poc = now + self.rng.uniform(30, 300)
        else:
            self.next_poc = None
        self.lab_base = {k: float(v[-1]["v"]) for k, v in (record.get("labs") or {}).items() if v}
        self.new_labs: List[Tuple[str, float]] = []      # resulted by the latest sample

    # -- deterioration ---------------------------------------------------------------
    def start(self, kind: str, now: float, page: Optional[str] = None) -> Scenario:
        label, _fits, build = SCENARIOS[kind]
        kw = build(self.rng)
        self.scenario = Scenario(kind=kind, label=label, t0=now, ramp=self.rng.uniform(170, 280), page=page, **kw)
        return self.scenario

    def observe(self, vitals: Dict[str, float], now: float, label: str = "Change reported by the nurse") -> Scenario:
        """A nurse reported values: the room's feed moves to them (the report is a measurement)."""
        tg: Dict[str, Tuple[str, float]] = {}
        for k in ("hr", "rr", "spo2", "temp_c", "sbp"):
            if vitals.get(k) is not None:
                tg[k] = ("abs", float(vitals[k]))
        labs = []
        if vitals.get("glucose") is not None:
            labs.append((0.0, "glucose", ("abs", float(vitals["glucose"]))))
        if vitals.get("lactate") is not None:
            labs.append((0.0, "lactate", ("abs", float(vitals["lactate"]))))
        self.scenario = Scenario(kind="observed", label=label, t0=now - 1, ramp=20.0, targets=tg, labs=labs,
                                 mental=(0.0, "C") if vitals.get("confused") else None, observed=True)
        return self.scenario

    def recover(self, now: float) -> None:
        s = self.scenario
        if s and s.recover_t0 is None:
            s.recover_from = s.progress(now)      # recover from where it is, never jump to the peak first
            s.recover_t0 = now
            s.recover_s = self.rng.uniform(*RECOVER_S)

    # -- the feed ----------------------------------------------------------------------
    def _target(self, k: str, base: float) -> float:
        s = self.scenario
        if not s or k not in s.targets:
            return base
        mode, val = s.targets[k]
        full = val if mode == "abs" else base + val if mode == "add" else base * val
        return base + (full - base) * s.progress(self.last_t)

    def sample(self, now: float) -> Dict[str, Any]:
        dt = max(0.0, now - self.last_t)
        self.last_t = now
        for k, (sig, _m) in _NOISE.items():
            self.ou[k] += -self.ou[k] * dt / _TAU + sig * math.sqrt(dt) * self.rng.gauss(0, 1)
        s = self.scenario
        if s and s.done(now):
            self.scenario, s = None, None
            self.cooldown_until = now + 20 * 60
        v: Dict[str, Any] = {}
        for k in ("hr", "sbp", "rr", "spo2", "temp_c"):
            x = self._target(k, self.base[k]) + self.ou[k] + self.rng.gauss(0, _NOISE[k][1])
            v[k] = x
        dbp_ratio = self.base["dbp"] / self.base["sbp"]
        v["dbp"] = v["sbp"] * dbp_ratio + self.rng.gauss(0, 1.2)
        v["spo2"] = min(100.0, max(70.0, v["spo2"]))
        v["rr"] = max(4.0, v["rr"])
        v["temp_c"] = min(41.5, max(34.0, v["temp_c"]))
        out = {k: (round(x, 1) if k == "temp_c" else int(round(x))) for k, x in v.items()}
        out["map"] = int(round((out["sbp"] + 2 * out["dbp"]) / 3))
        o2 = self.base_o2
        if s and s.o2 and s.progress(now) > 0.45:
            o2 = max(o2, s.o2)
        out["o2_lpm"] = o2
        mental = "A"
        if s and s.mental and s.progress(now) >= s.mental[0]:
            mental = s.mental[1]
        out["avpu"] = mental
        out["chronic_confusion"] = self.chronic_confusion
        self.new_labs = self._draw_labs(now)
        return out

    def _draw_labs(self, now: float) -> List[Tuple[str, float]]:
        drawn: List[Tuple[str, float]] = []
        s = self.scenario
        if s:
            p = s.progress(now)
            for at, lab, (mode, val) in s.labs:
                key = f"{lab}@{at}"
                if s.recover_t0 is None and p >= at and not s.drawn.get(key):
                    s.drawn[key] = True
                    base = self.lab_base.get(lab, {"lactate": 1.3, "wbc": 9.0, "glucose": 140.0,
                                                   "hemoglobin": 12.5, "potassium": 4.4}.get(lab, 0.0))
                    x = val if mode == "abs" else base + val if mode == "add" else base * val
                    drawn.append((lab, round(x, 1)))
            if s.recover_t0 is not None and not s.drawn.get("repeat") and s.progress(now) < 0.35 and s.labs:
                s.drawn["repeat"] = True
                for _at, lab, _t in s.labs:
                    base = self.lab_base.get(lab)
                    if base is not None:
                        drawn.append((lab, round(base * self.rng.uniform(0.97, 1.05), 1)))
                    elif lab == "lactate":
                        drawn.append((lab, round(self.rng.uniform(1.4, 1.9), 1)))
        if self.next_poc is not None and now >= self.next_poc:
            # point-of-care glucose for a diabetic: every ~30 min, more often while symptomatic
            hypo = s is not None and s.kind == "hypoglycemia" and s.recover_t0 is None
            self.next_poc = now + (self.rng.uniform(60, 120) if hypo else self.rng.uniform(1500, 2100))
            if not hypo:
                drawn.append(("glucose", round((self.glucose_base or 150) * self.rng.uniform(0.9, 1.1))))
        for lab, x in drawn:
            self.labs.setdefault(lab, []).append({"t": now, "v": x})
            del self.labs[lab][:-24]
        return drawn
