"""Scenario Lab — synthetic clinical cases the AI can author, run and be graded on.

Dr. Aegis can already read every surface of this OS and drive its actuators. What it
could not do is **pose a situation**: build a patient who deteriorates in a particular
way, run the system against them, and see whether the OS did the right thing. This
module is that missing half, plus the sandbox it runs in.

WHAT THIS IS FOR — AND THE LINE IT MUST NOT CROSS
------------------------------------------------
A synthetic patient is written by hand, by us. Anything measured on one measures our
own generator, not the world. This project has already been burned by exactly that:
the original detector looked flawless on a simulator that had been tuned alongside it,
and scored ~chance (AUROC 0.64) the first time it met real patients. The lesson was not
"simulators are bad" — it was that a number from a simulator is circular.

So the split is enforced rather than remembered:

  * Scenarios test **behaviour** — did the alarm fire before onset, did the tier
    escalate, did the safety gate refuse an unfounded order, did the panel populate.
    These are properties of the OS, and a hand-written patient is a legitimate way to
    probe them, the same way a unit test uses a hand-written input.
  * Scenarios NEVER test **accuracy**. :class:`ScenarioResult` has no AUROC, no AUPRC,
    no utility, and no place to put one. Synthetic records carry ``set_name =
    "synthetic"`` forever, and :func:`backend.ml.data.features.build_dataset` refuses
    them unless a caller explicitly opts in — so a synthetic case cannot reach the
    evaluation harness by accident, only by intent that is visible in the diff.

Every accuracy claim in this project stays where it was: real PhysioNet/CinC 2019
patients, patient-grouped, cross-hospital. See docs/FRONTIER_REPORT.md.

THE SANDBOX
-----------
Scenario patients live in a :class:`ScenarioLab` held in memory. They are never added
to the live cohort, never persisted to Supabase, and their bed handles are prefixed
``sim_`` so a handle cannot be confused with a monitored bed at a glance. Deleting the
lab deletes them; nothing survives a restart.

DETERMINISM
-----------
A spec plus a seed is a complete description: the same spec always yields the same
trajectory, hour for hour. A scenario that fails is therefore reproducible from the
spec alone, which is the only way a failing case is worth reporting.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.ml import config as mlconfig
from backend.ml.data.cinc2019 import PatientRecord

#: The provenance marker. Anything carrying this is hand-written, not observed.
#: It is checked by value in several places on purpose — a constant that is only
#: ever compared against itself is a constant nobody can accidentally route around.
SYNTHETIC_SET = "synthetic"

#: Hours the challenge label is pulled forward of clinical onset. Mirrored from the
#: ML config so a synthetic label means exactly what a real one means; if these ever
#: disagree, every downstream lead-time number quietly shifts by the difference.
LABEL_LEAD = mlconfig.SEPSIS_LABEL_LEAD_HOURS


def is_synthetic(record: PatientRecord) -> bool:
    """True if this record was authored rather than observed."""
    return getattr(record, "set_name", "") == SYNTHETIC_SET


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #
#: Named deterioration arcs. Each is a function of ``u`` — progress through the
#: deterioration, 0 before it starts and 1 at its worst — returning per-channel
#: offsets from the patient's own baseline. Keeping them as pure functions of one
#: scalar means an arc can be reasoned about and plotted without running a patient.
ARCS: Dict[str, str] = {
    "septic_shock": "Classic warm septic shock: tachycardia, fever, tachypnoea, "
                    "falling MAP, rising lactate and WBC.",
    "cold_sepsis": "Hypothermic presentation — the one clinicians miss, because the "
                   "temperature moves the wrong way.",
    "occult": "Deteriorating physiology with an almost-normal temperature and a "
              "late, small lactate rise. Deliberately hard.",
    "stable": "No deterioration. The control arm — the OS should stay quiet.",
    "recovering": "Sick on admission, improving throughout. Should not alarm late.",
    "post_op_fever": "A benign inflammatory fever with stable perfusion — a "
                     "confounder that should NOT be treated as sepsis.",
}


@dataclass
class ScenarioSpec:
    """A clinical situation, small enough to write by hand or by model."""

    name: str
    arc: str = "septic_shock"
    los_hours: int = 48
    #: Clinical onset hour. ``None`` for arcs that never become septic.
    onset_hour: Optional[int] = 30
    #: How steeply the deterioration develops, in hours from first drift to worst.
    ramp_hours: int = 12
    #: 0.4 subtle, 1.0 textbook, 1.6 fulminant.
    severity: float = 1.0
    age: float = 68.0
    gender: str = "M"
    weight: float = 78.0
    #: Hours between lab draws. Real ICU labs are sparse and their sparsity is
    #: itself predictive, so this is a first-class knob rather than a detail.
    lab_interval_hours: int = 8
    #: Probability any given vital sign is missing in a given hour.
    vitals_dropout: float = 0.04
    #: Interventions as (hour, kind). "fluids" lifts MAP; "antibiotics" bends the
    #: whole arc back toward baseline after a delay.
    interventions: List[Tuple[int, str]] = field(default_factory=list)
    seed: int = 7
    notes: str = ""

    def fingerprint(self) -> str:
        """Stable short hash of the spec — the scenario's identity."""
        blob = repr(sorted(asdict(self).items())).encode()
        return hashlib.sha256(blob).hexdigest()[:10]

    def validate(self) -> None:
        if self.arc not in ARCS:
            raise ValueError(f"unknown arc {self.arc!r}; have {sorted(ARCS)}")
        if self.los_hours < 8:
            raise ValueError("a stay shorter than 8 hours cannot exercise the windows")
        if self.los_hours > 336:
            raise ValueError("CinC stays top out at 336 hours; keep synthetic ones inside that")
        if self.onset_hour is not None:
            if not (0 < self.onset_hour < self.los_hours):
                raise ValueError("onset_hour must fall inside the stay")
            if self.onset_hour <= LABEL_LEAD:
                raise ValueError(
                    f"onset_hour must exceed the {LABEL_LEAD}h label lead, or the case "
                    "starts already labelled septic and there is nothing to predict")
        if not 0.0 <= self.vitals_dropout < 0.9:
            raise ValueError("vitals_dropout must be a probability below 0.9")


# --------------------------------------------------------------------------- #
# Physiology
# --------------------------------------------------------------------------- #
#: Per-arc channel deltas at full deterioration (u = 1), before severity scaling.
#: Written as plain numbers rather than a model because they ARE the assumption —
#: a reader should be able to disagree with a specific figure.
_ARC_DELTAS: Dict[str, Dict[str, float]] = {
    "septic_shock":  {"HR": 38, "Temp": 1.9, "Resp": 12, "MAP": -26, "O2Sat": -5,
                      "Lactate": 3.6, "WBC": 8.0},
    "cold_sepsis":   {"HR": 30, "Temp": -1.6, "Resp": 10, "MAP": -24, "O2Sat": -6,
                      "Lactate": 3.9, "WBC": -2.5},
    "occult":        {"HR": 20, "Temp": 0.4, "Resp": 7, "MAP": -14, "O2Sat": -3,
                      "Lactate": 1.5, "WBC": 3.0},
    "stable":        {},
    "recovering":    {"HR": -14, "Temp": -0.8, "Resp": -4, "MAP": 10, "O2Sat": 2,
                      "Lactate": -1.4, "WBC": -3.0},
    # Fever and a mild tachycardia, but perfusion holds and lactate does not move.
    # If the OS calls this sepsis, it is reading the thermometer, not the patient.
    "post_op_fever": {"HR": 16, "Temp": 1.6, "Resp": 4, "MAP": -3, "O2Sat": -1,
                      "Lactate": 0.2, "WBC": 4.5},
}

_BASELINE = {"HR": 78.0, "Temp": 36.9, "Resp": 16.0, "MAP": 86.0, "O2Sat": 97.0,
             "Lactate": 1.1, "WBC": 8.0, "SBP": 124.0, "DBP": 70.0}

#: Physiological bounds. A generator that emits a MAP of -4 produces features no
#: real patient could, and the detector's response to it means nothing.
_BOUNDS = {"HR": (30, 200), "Temp": (32.0, 42.0), "Resp": (4, 60), "MAP": (25, 140),
           "O2Sat": (60, 100), "Lactate": (0.3, 20.0), "WBC": (0.5, 60.0),
           "SBP": (50, 220), "DBP": (25, 130)}


def _progress(spec: ScenarioSpec, t: np.ndarray) -> np.ndarray:
    """Deterioration progress ``u(t)`` in [0, 1] — a smooth ramp, not a step.

    Physiology does not switch on at an hour boundary, and a step would let a model
    look prescient for noticing the discontinuity rather than the deterioration.
    """
    if spec.onset_hour is None:
        if spec.arc == "recovering":
            # Improvement runs from admission, so progress is time itself.
            return np.clip(t / max(spec.los_hours - 1, 1), 0.0, 1.0)
        return np.zeros_like(t, dtype=float)
    start = spec.onset_hour - spec.ramp_hours
    u = (t - start) / max(spec.ramp_hours, 1)
    # Logistic shoulder so the approach is gradual and the peak plateaus.
    return 1.0 / (1.0 + np.exp(-6.0 * (np.clip(u, -0.4, 1.6) - 0.5)))


def _intervention_effect(spec: ScenarioSpec, t: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-channel correction from treatments, ramping in over a few hours."""
    eff = {k: np.zeros_like(t, dtype=float) for k in _BASELINE}
    for hour, kind in spec.interventions:
        onset = np.clip((t - hour) / 3.0, 0.0, 1.0)     # 3h to full effect
        if kind == "fluids":
            eff["MAP"] += 12.0 * onset
            eff["SBP"] += 15.0 * onset
            eff["HR"] -= 8.0 * onset
        elif kind == "antibiotics":
            # Antibiotics do not undo the hour they are given; they bend the curve.
            # A 4h lag before anything measurable, then a broad partial reversal.
            lag = np.clip((t - hour - 4.0) / 8.0, 0.0, 1.0)
            eff["HR"] -= 14.0 * lag
            eff["Temp"] -= 0.9 * lag
            eff["Lactate"] -= 1.6 * lag
            eff["MAP"] += 9.0 * lag
            eff["Resp"] -= 4.0 * lag
        elif kind == "vasopressors":
            eff["MAP"] += 18.0 * onset
            eff["SBP"] += 20.0 * onset
            eff["HR"] += 6.0 * onset
        else:
            raise ValueError(f"unknown intervention {kind!r}")
    return eff


def synthesize(spec: ScenarioSpec) -> PatientRecord:
    """Turn a spec into a :class:`PatientRecord` the real stack can consume.

    The frame carries the full CinC column set so every downstream consumer — the
    feature builder, the detector, the cohort viewer, the causal ingest — sees the
    shape it expects. ``set_name`` is the synthetic marker and is never overridable.
    """
    spec.validate()
    rng = np.random.default_rng(spec.seed)
    n = int(spec.los_hours)
    t = np.arange(n, dtype=float)

    u = _progress(spec, t)
    deltas = _ARC_DELTAS.get(spec.arc, {})
    eff = _intervention_effect(spec, t)

    # Age shifts the starting point a little; an 85-year-old is not a 40-year-old.
    age_shift = (spec.age - 65.0) / 20.0
    chan: Dict[str, np.ndarray] = {}
    for name, base in _BASELINE.items():
        series = np.full(n, base, dtype=float)
        if name == "HR":
            series -= 3.0 * age_shift
        elif name == "MAP":
            series += 2.0 * age_shift
        series = series + deltas.get(name, 0.0) * spec.severity * u + eff[name]
        # Correlated drift plus per-hour noise: real vitals wander, they do not
        # resample independently each hour, and a detector that keys off
        # hour-to-hour independence would be learning an artifact of the generator.
        wander = np.cumsum(rng.normal(0.0, 1.0, n)) * 0.15
        scale = {"Temp": 0.12, "Lactate": 0.10, "O2Sat": 0.5, "WBC": 0.35}.get(name, 1.6)
        series = series + wander * scale + rng.normal(0.0, scale * 0.5, n)
        lo, hi = _BOUNDS[name]
        chan[name] = np.clip(series, lo, hi)

    # Blood pressure is not three free channels: keep MAP ≈ (SBP + 2·DBP)/3 so the
    # shock index and the MAP-derived rules see a coherent patient.
    chan["SBP"] = np.clip(chan["MAP"] * 1.45 + rng.normal(0, 3.0, n), *_BOUNDS["SBP"])
    chan["DBP"] = np.clip((3.0 * chan["MAP"] - chan["SBP"]) / 2.0, *_BOUNDS["DBP"])

    frame = pd.DataFrame(index=range(n), columns=mlconfig.CINC_COLUMNS, dtype=float)
    for name, series in chan.items():
        frame[name] = series

    # --- measurement pattern: labs sparse, vitals occasionally dropped -------- #
    lab_cols = ["Lactate", "WBC"]
    drawn = np.zeros(n, dtype=bool)
    drawn[:: max(spec.lab_interval_hours, 1)] = True
    for col in lab_cols:
        frame.loc[~drawn, col] = np.nan
    for col in ("HR", "Temp", "Resp", "MAP", "O2Sat", "SBP", "DBP"):
        drop = rng.random(n) < spec.vitals_dropout
        frame.loc[drop, col] = np.nan

    frame["Age"] = spec.age
    frame["Gender"] = 1.0 if spec.gender.upper().startswith("M") else 0.0
    frame["HospAdmTime"] = -2.0
    frame["ICULOS"] = np.arange(1, n + 1, dtype=float)

    # --- label, on the challenge's own convention ---------------------------- #
    labels = np.zeros(n, dtype=np.int8)
    if spec.onset_hour is not None:
        labels[max(0, spec.onset_hour - LABEL_LEAD):] = 1
    frame[mlconfig.LABEL_COLUMN] = labels.astype(float)

    return PatientRecord(
        patient_id=f"SYN-{spec.fingerprint().upper()}",
        set_name=SYNTHETIC_SET,
        frame=frame,
        labels=labels,
    )


# --------------------------------------------------------------------------- #
# The detector under test
# --------------------------------------------------------------------------- #
_MODEL: Any = None
_MODEL_PATH: Optional[str] = None


def _detector():
    """The PRODUCTION detector artifact, loaded once.

    The lab deliberately exercises the shipped model rather than a fresh fit: the
    question a scenario asks is "what would the deployed system have done", and a
    retrained stand-in answers a different question.
    """
    global _MODEL, _MODEL_PATH
    if _MODEL is None:
        import os
        import joblib
        _MODEL_PATH = os.path.join(mlconfig.ARTIFACTS_DIR, "gbm_wearable.joblib")
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"no production detector at {_MODEL_PATH}; scenarios grade the shipped "
                "model, so there is nothing to grade against")
        _MODEL = joblib.load(_MODEL_PATH)
    return _MODEL


def _operating_point() -> Dict[str, float]:
    """The SHIPPED alarm thresholds, not fresh ones.

    A scenario that graded itself against a threshold derived for the occasion would
    be marking its own homework; these are the numbers the live system alarms on.
    """
    from backend.ml import operating_point as opmod
    op = opmod.load()
    if op is None:
        return {"high": 0.12, "medium": 0.04, "derived": 0.0}
    return {"high": float(op.high), "medium": float(op.medium), "derived": 1.0}


def risk_trace(record: PatientRecord) -> np.ndarray:
    """Per-hour P(sepsis) across the whole stay, causally.

    The feature builder is the same one production uses and is causal by
    construction, so hour ``t``'s score depends only on hours ``≤ t``.
    """
    from backend.ml.data.features import FeatureConfig, build_patient_features
    feats = build_patient_features(record, FeatureConfig.wearable_only())
    model = _detector()
    names = getattr(model, "feature_names", None)
    if names:                                   # array-taking detector (GBMDetector)
        return np.asarray(model.predict_proba(feats.to_numpy()), dtype=float)
    return np.asarray(model.predict_proba(feats), dtype=float)


# --------------------------------------------------------------------------- #
# Expectations — behaviour, never accuracy
# --------------------------------------------------------------------------- #
@dataclass
class Expectation:
    """One graded assertion about how the OS should behave on a scenario.

    Kept declarative so a scenario reads as a statement of intent rather than a
    script, and so the AI can author one without writing code.
    """

    kind: str
    #: Interpretation depends on kind; see EXPECTATION_KINDS.
    value: float = 0.0
    #: Free text explaining WHY this is the right behaviour. Printed on failure,
    #: because a red check with no rationale gets deleted rather than investigated.
    because: str = ""


EXPECTATION_KINDS: Dict[str, str] = {
    "alarm_lead_at_least": "HIGH alarm must fire at least `value` hours before "
                           "clinical onset.",
    "alarms_at_all": "A HIGH alarm must occur at some point in the stay.",
    "never_alarms": "No HIGH alarm at any hour — the control arms and the "
                    "confounders depend on this.",
    "no_alarm_before_hour": "No HIGH alarm before hour `value`; guards against a "
                            "detector that simply alarms on everyone at admission.",
    "peak_risk_at_least": "Peak risk must reach `value`.",
    "peak_risk_at_most": "Peak risk must stay under `value`.",
    "risk_rises_into_onset": "Mean risk in the 6h before onset must exceed the mean "
                             "over the first 6h of the stay by `value`.",
}


@dataclass
class Graded:
    kind: str
    passed: bool
    expected: float
    observed: float
    because: str
    detail: str = ""


@dataclass
class ScenarioResult:
    """The outcome of one scenario run.

    NOTE what is absent: there is no AUROC, AUPRC or utility field, and adding one
    would be a category error. A synthetic cohort can tell you the alarm fired in the
    right order; it cannot tell you how often it is right, because we wrote it.
    """

    scenario: str
    fingerprint: str
    bed: str
    arc: str
    onset_hour: Optional[int]
    hours: int
    peak_risk: float
    first_alarm_hour: Optional[int]
    lead_time_hours: Optional[float]
    alarm_hours: int
    thresholds: Dict[str, float]
    graded: List[Graded] = field(default_factory=list)
    risk: List[float] = field(default_factory=list)
    synthetic: bool = True

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.graded)

    def summary(self) -> str:
        head = (f"{self.scenario} [{self.arc}] — "
                f"{'PASS' if self.passed else 'FAIL'} "
                f"({sum(g.passed for g in self.graded)}/{len(self.graded)})")
        lines = [head,
                 f"  peak risk {self.peak_risk:.3f} · alarm hours {self.alarm_hours}"
                 + (f" · first alarm h{self.first_alarm_hour}" if self.first_alarm_hour is not None else "")
                 + (f" · lead {self.lead_time_hours:.0f}h" if self.lead_time_hours is not None else "")]
        for g in self.graded:
            mark = "ok  " if g.passed else "FAIL"
            lines.append(f"  [{mark}] {g.kind}: expected {g.expected:g}, "
                         f"observed {g.observed:g}"
                         + (f" — {g.because}" if not g.passed and g.because else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _grade(spec: ScenarioSpec, risk: np.ndarray, exps: List[Expectation],
           thr: Dict[str, float]) -> Tuple[List[Graded], Dict[str, Any]]:
    high = thr["high"]
    alarms = np.flatnonzero(risk >= high)
    first = int(alarms[0]) if alarms.size else None
    lead = (float(spec.onset_hour - first)
            if first is not None and spec.onset_hour is not None else None)
    facts = {"first_alarm_hour": first, "lead_time_hours": lead,
             "alarm_hours": int(alarms.size), "peak_risk": float(risk.max())}

    out: List[Graded] = []
    for e in exps:
        if e.kind not in EXPECTATION_KINDS:
            raise ValueError(f"unknown expectation {e.kind!r}; have {sorted(EXPECTATION_KINDS)}")
        obs, ok, detail = 0.0, False, ""
        if e.kind == "alarm_lead_at_least":
            obs = lead if lead is not None else -1.0
            ok = lead is not None and lead >= e.value
            if lead is None:
                detail = "no alarm fired at or before onset"
        elif e.kind == "alarms_at_all":
            obs, ok = float(alarms.size), alarms.size > 0
        elif e.kind == "never_alarms":
            obs, ok = float(alarms.size), alarms.size == 0
            if not ok:
                detail = f"first spurious alarm at hour {first}"
        elif e.kind == "no_alarm_before_hour":
            obs = float(first) if first is not None else float(spec.los_hours)
            ok = first is None or first >= e.value
        elif e.kind == "peak_risk_at_least":
            obs, ok = float(risk.max()), float(risk.max()) >= e.value
        elif e.kind == "peak_risk_at_most":
            obs, ok = float(risk.max()), float(risk.max()) <= e.value
        elif e.kind == "risk_rises_into_onset":
            if spec.onset_hour is None:
                raise ValueError("risk_rises_into_onset needs a scenario with an onset")
            pre = float(risk[: min(6, len(risk))].mean())
            near = float(risk[max(0, spec.onset_hour - 6): spec.onset_hour].mean())
            obs, ok = round(near - pre, 4), (near - pre) >= e.value
            detail = f"baseline {pre:.4f} -> pre-onset {near:.4f}"
        out.append(Graded(e.kind, ok, e.value, round(obs, 4), e.because, detail))
    return out, facts


def run(spec: ScenarioSpec, expectations: Optional[List[Expectation]] = None,
        bed: str = "sim_0") -> ScenarioResult:
    """Synthesize, score with the shipped detector, and grade the expectations."""
    record = synthesize(spec)
    risk = risk_trace(record)
    thr = _operating_point()
    graded, facts = _grade(spec, risk, expectations or [], thr)
    return ScenarioResult(
        scenario=spec.name, fingerprint=spec.fingerprint(), bed=bed, arc=spec.arc,
        onset_hour=spec.onset_hour, hours=int(spec.los_hours),
        peak_risk=round(facts["peak_risk"], 4),
        first_alarm_hour=facts["first_alarm_hour"],
        lead_time_hours=facts["lead_time_hours"],
        alarm_hours=facts["alarm_hours"], thresholds=thr, graded=graded,
        risk=[round(float(v), 4) for v in risk],
    )


# --------------------------------------------------------------------------- #
# The standing suite
# --------------------------------------------------------------------------- #
def _spec(**kw) -> Tuple[ScenarioSpec, List[Expectation]]:
    exps = kw.pop("expect")
    return ScenarioSpec(**kw), exps


#: The canonical cases. Half are deteriorations the OS must catch; half are
#: confounders it must ignore. A suite of only sick patients grades a detector that
#: alarms on everybody as perfect, so the negatives are the load-bearing half.
def suite() -> List[Tuple[ScenarioSpec, List[Expectation]]]:
    return [
        _spec(name="textbook-shock", arc="septic_shock", los_hours=48, onset_hour=30,
              severity=1.0, seed=11,
              notes="The case everything else is measured against.",
              expect=[Expectation("alarms_at_all", 1,
                                  "A textbook septic shock the system misses is not a monitor."),
                      Expectation("alarm_lead_at_least", 2,
                                  "Warning after the fact has no clinical value."),
                      Expectation("risk_rises_into_onset", 0.01,
                                  "Risk must track the deterioration, not jump at onset.")]),
        # KNOWN FAILING, deliberately. This case is unambiguously shocked — qSOFA 2,
        # MAP 60, SBP 88, lactate 5.3 — and differs from textbook-shock essentially
        # only in the SIGN of the temperature. The shipped detector peaks at ~0.033
        # against a 0.055 alarm line and stays silent. Do not repair this by relaxing
        # the expectation: the check failing is the check working. It is a HYPOTHESIS
        # about fever-leaning, raised on a patient we wrote, and it is settled on real
        # hypothermic septic patients in CinC — a run nobody has done yet.
        _spec(name="cold-sepsis", arc="cold_sepsis", los_hours=54, onset_hour=34,
              severity=1.1, seed=23,
              notes="Hypothermic presentation — the temperature moves the WRONG way. "
                    "KNOWN FAILING: raises a fever-leaning hypothesis, unconfirmed on "
                    "real data.",
              expect=[Expectation("alarms_at_all", 1,
                                  "Missing hypothermic sepsis means keying on fever alone.")]),
        _spec(name="occult-slow", arc="occult", los_hours=72, onset_hour=48,
              severity=0.9, ramp_hours=20, lab_interval_hours=12, seed=31,
              notes="Subtle, slow, sparse labs. Expected to be hard.",
              expect=[Expectation("alarms_at_all", 1,
                                  "The subtle case is the one worth catching early.")]),
        _spec(name="stable-control", arc="stable", los_hours=60, onset_hour=None,
              seed=5, notes="Never deteriorates. The silence control.",
              expect=[Expectation("never_alarms", 0,
                                  "Alarming on a stable patient is the failure mode "
                                  "that gets a monitor unplugged.")]),
        _spec(name="post-op-fever", arc="post_op_fever", los_hours=48, onset_hour=None,
              seed=17, notes="Fever + tachycardia, perfusion intact, lactate flat.",
              expect=[Expectation("never_alarms", 0,
                                  "Inflammation is not infection; a thermometer is "
                                  "not a sepsis detector.")]),
        _spec(name="recovering", arc="recovering", los_hours=48, onset_hour=None,
              seed=41, notes="Sick on arrival, improving. Should quieten, not escalate.",
              expect=[Expectation("peak_risk_at_most", 0.5,
                                  "A patient getting better must not drive risk up.")]),
        # OBSERVATION-ONLY, on purpose. The obvious assertion — "treatment should
        # lower peak risk" — is one this pair does not support: at an identical seed
        # the treated arm currently peaks HIGHER (0.132 vs 0.101), plausibly because
        # the intervention puts a sharp inflection into the slope/delta features. That
        # may be an artifact of how this generator applies treatment rather than
        # anything about the detector, and asserting a direction nobody has
        # established on real data would bake a guess into the suite. So the pair is
        # RECORDED and left ungraded until the direction is known.
        _spec(name="treated-early", arc="septic_shock", los_hours=56, onset_hour=34,
              severity=1.2, seed=53, interventions=[(24, "antibiotics"), (24, "fluids")],
              notes="Same shock as untreated-late, same seed, treated 10h pre-onset. "
                    "OBSERVATION-ONLY: the treated arm peaks higher; direction "
                    "unestablished, so nothing is asserted.",
              expect=[]),
        _spec(name="untreated-late", arc="septic_shock", los_hours=56, onset_hour=34,
              severity=1.2, seed=53,
              notes="The control for treated-early: identical seed, no treatment.",
              expect=[Expectation("alarms_at_all", 1,
                                  "The untreated arm of the pair must alarm.")]),
    ]


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #
class ScenarioLab:
    """In-memory sandbox. Nothing here is persisted, monitored, or clinical.

    Bed handles are prefixed ``sim_`` so a synthetic handle never reads like a real
    one, and the lab holds its own records rather than registering them with the live
    cohort — a scenario cannot appear on the ward board by accident.
    """

    PREFIX = "sim_"
    MAX_SCENARIOS = 32           # a sandbox, not a database

    def __init__(self) -> None:
        self._specs: Dict[str, ScenarioSpec] = {}
        self._expect: Dict[str, List[Expectation]] = {}
        self._results: Dict[str, ScenarioResult] = {}
        self._seq = 0
        self.created_at = time.time()

    # -- authoring ------------------------------------------------------ #
    def create(self, spec: ScenarioSpec,
               expectations: Optional[List[Expectation]] = None) -> str:
        spec.validate()
        if len(self._specs) >= self.MAX_SCENARIOS:
            raise ValueError(
                f"scenario lab holds at most {self.MAX_SCENARIOS}; reset it or delete one")
        bed = f"{self.PREFIX}{self._seq}"
        self._seq += 1
        self._specs[bed] = spec
        self._expect[bed] = list(expectations or [])
        return bed

    def delete(self, bed: str) -> bool:
        found = bed in self._specs
        self._specs.pop(bed, None)
        self._expect.pop(bed, None)
        self._results.pop(bed, None)
        return found

    def reset(self) -> int:
        n = len(self._specs)
        self._specs.clear(); self._expect.clear(); self._results.clear()
        self._seq = 0
        return n

    # -- running -------------------------------------------------------- #
    def run(self, bed: str) -> ScenarioResult:
        if bed not in self._specs:
            raise KeyError(f"no scenario at {bed!r}")
        res = run(self._specs[bed], self._expect[bed], bed=bed)
        self._results[bed] = res
        return res

    def run_all(self) -> List[ScenarioResult]:
        return [self.run(b) for b in list(self._specs)]

    def load_suite(self) -> List[str]:
        """Install the standing suite into an empty lab."""
        self.reset()
        return [self.create(s, e) for s, e in suite()]

    # -- reading -------------------------------------------------------- #
    def list(self) -> List[Dict[str, Any]]:
        out = []
        for bed, spec in self._specs.items():
            res = self._results.get(bed)
            out.append({"bed": bed, "name": spec.name, "arc": spec.arc,
                        "onset_hour": spec.onset_hour, "hours": spec.los_hours,
                        "fingerprint": spec.fingerprint(),
                        "expectations": len(self._expect[bed]),
                        "ran": res is not None,
                        "passed": None if res is None else res.passed,
                        "synthetic": True})
        return out

    def result(self, bed: str) -> Optional[ScenarioResult]:
        return self._results.get(bed)

    def report(self) -> Dict[str, Any]:
        ran = [r for r in self._results.values()]
        return {
            "scenarios": len(self._specs),
            "ran": len(ran),
            "passed": sum(1 for r in ran if r.passed),
            "failed": sum(1 for r in ran if not r.passed),
            "thresholds": _operating_point(),
            "synthetic": True,
            "caveat": "Behavioural checks on hand-written patients. These say what the "
                      "OS DID, never how accurate it is — accuracy lives in "
                      "docs/FRONTIER_REPORT.md, on real patients.",
            "results": [{"bed": r.bed, "scenario": r.scenario, "arc": r.arc,
                         "passed": r.passed, "peak_risk": r.peak_risk,
                         "first_alarm_hour": r.first_alarm_hour,
                         "lead_time_hours": r.lead_time_hours,
                         "failures": [g.kind for g in r.graded if not g.passed]}
                        for r in ran],
        }


#: Process-wide lab. One sandbox, like one cohort.
LAB = ScenarioLab()
