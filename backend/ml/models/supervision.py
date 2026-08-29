"""Supervision density — how much of a septic stay should be labelled positive.

THE FINDING
-----------
Sepsis "early warning" on CinC 2019 is, at the accuracy anyone currently reaches, a
**patient-classification problem wearing the costume of an event-timing problem**.

The challenge label marks an hour positive when ``t ≥ t_sepsis − 6h``, which on this
cohort is 2.1% of hours even though 12.8% of the patients become septic. Training on a
label that says positive for *more* of a septic patient's stay improves detection at
every alarm budget tested — and **placing those extra positive hours before onset does
not help; it slightly hurts**. The same number of hours scattered at random through the
stay does as well or better:

    training target        prevalence   AUROC    det@5%   det@10%   det@15%
    challenge label          0.021      0.7411    0.274    0.353     0.411
    96h window before onset  0.105      0.7505    0.358    0.437     0.496
    same count, RANDOM hours 0.105      0.7538    0.384    0.461     0.505
    patient label (all hours) 0.128     0.7534    0.406    0.468     0.508

Window vs. random placement, measured where the control can actually discriminate
(at W = 24 the window claims 34% of the available non-labelled septic hours, so a
random draw genuinely differs; at W = 96 it claims 79% and the control is near-vacuous):

    budget   window − sham, seed 1        window − sham, seed 2
      5%     −0.063  CI [−0.099, −0.032]  −0.062  CI [−0.088, −0.028]
     10%     −0.057  CI [−0.093, −0.024]  −0.031  CI [−0.063, +0.000]
     15%     −0.035  CI [−0.077, +0.000]  −0.007  CI [−0.043, +0.028]

Every point estimate is negative and the tight-budget intervals sit entirely below
zero. **The window is not merely inert — concentrating the extra supervision just
before onset is worse than scattering it.** Which fits: the hours immediately before
onset are the ones most similar to hours the model already sees labelled, so they add
the least new information about *who this patient is*. Hours drawn from across the
stay teach the model to recognise the patient as septic from a wider range of
physiologic states.

What the model gains is not a better sense of *when*; it is more, and more varied,
supervision about *which patients*.

So the recommended target is the simplest one, the one with no temporal content at all
and the widest possible spread: **the patient label**. It carries det@5% 0.274 → 0.406 (+48% relative) and det@10%
0.353 → 0.468 (+33%), and it has no window parameter to tune or to transfer.

HOW THIS CONCLUSION WAS REACHED, INCLUDING THE PART THAT WAS WRONG
------------------------------------------------------------------
This module was first written as a "supervision horizon" — the hypothesis being that a
wider *pre-onset window* lets the model learn the physiology of the approach to sepsis.
That hypothesis produced the right numbers for the wrong reason, and it was the
negative control in ``verify_supervision.py`` that caught it: a prevalence-matched
random relabelling scored **as well or better** than the carefully-placed window. The
window was inert. The documentation is rewritten here rather than quietly adjusted,
because the difference between "wider early-warning window" and "more patient-level
supervision" is the difference between a claim about physiology and a claim about
labelling, and only the second one is supported.

WHY IT IS TRUE, AND WHAT ELSE IT EXPLAINS
-----------------------------------------
Decomposing the pooled AUROC into the two questions it conflates makes the mechanism
visible. On this cohort the production ensemble scores 0.79 at *between-patient*
discrimination (is this the kind of patient who becomes septic?) and its within-stay
timing cannot be separated from simply knowing how long the patient has been in the
unit — inside a septic stay the label is a perfect monotone function of the clock, so
a model that recovers elapsed time gets within-patient "timing skill" for free and for
no clinical reason (see :mod:`backend.ml.models.timing`).

Widening supervision trades exactly along that seam: between-patient discrimination
rises (0.788 → 0.799) and within-stay ordering falls (0.629 → 0.579). Detection at a
fixed alarm budget is a *patient-level* statistic — it compares each patient's peak
score against a threshold set by the non-septic patients' peaks — so it inherits the
gain and is untouched by the loss. Pooled AUROC barely moves because the two effects
cancel, which is why this lever stayed invisible to a project that had been reading
AUROC.

That also retro-explains a long run of null results in this repository. Multi-scale
temporal features, MiniRocket, Forest-VSA, a dilated TCN, a Mamba-2 state space model,
critical-slowing-down, quantum HDC and cross-channel coordination all failed to beat a
flat-feature booster. If most of the accessible signal is patient identity rather than
temporal shape, that is not eight unlucky experiments — it is one fact, measured eight
times.

WHAT THIS IS NOT
----------------
It is not an AUROC gain: not significant at any supervision density (W=96: +0.0067,
95% CI [−0.0066, +0.0202]), and it is stated as n.s. here rather than omitted. The
claim is about detection at a fixed alarm budget and about false-alarm rate.

It also genuinely costs within-stay resolution, so it is the right target for *which
patient to watch* and the wrong one for *which hour to escalate*. Whether the second
question is answerable at all from hourly vitals is, on the evidence in
:mod:`backend.ml.models.timing`, unresolved and possibly no.

Cross-hospital (set A ↔ set B, both directions, 2000 patient bootstraps) the gain
holds: B→A significant at all four budgets (+0.118/+0.103/+0.093/+0.073), A→B
significant at 5% and 10% (+0.019/+0.028) and positive but n.s. at 15% and 20%. The
asymmetry tracks prevalence — set B is the sparser site (1.41% vs 2.17% of hours) and
gains more from denser supervision — which is a hypothesis consistent with the numbers
rather than a tested claim.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.ml.models.boosters import DEFAULTS, balanced_target
from backend.ml.models.calibration import ProbabilityCalibrator
from backend.ml.models.hdc_detector import _grouped_holdout

#: Supervision densities the sweep covers, as hours added *before* the labelled onset.
#: Note the reference point: ``onset_hour`` in this project's feature matrix is the
#: LABELLED onset, which the challenge has already pulled 6 hours ahead of clinical
#: onset. ``W`` is hours before *that*, so the window in hours before true clinical
#: onset is ``W + 6``. Getting this wrong shifts the entire axis by six hours and
#: silently compares against a baseline that is not the incumbent — which is exactly
#: what the first version of this sweep did.
DENSITIES: Tuple[float, ...] = (0.0, 6.0, 12.0, 24.0, 48.0, 96.0, np.inf)


def supervision_target(y: np.ndarray, hour_index: np.ndarray, onset_hour: np.ndarray,
                       width_hours: float) -> np.ndarray:
    """The training label at supervision width ``width_hours``.

    ``W = 0`` reproduces the challenge label exactly; ``W = inf`` is the patient label,
    which marks every hour of a septic stay positive and is the recommended target.
    Intermediate widths exist so the sweep can be reproduced, not because the window
    itself was found to matter — it was not.
    """
    hour_index = np.asarray(hour_index)
    onset_hour = np.asarray(onset_hour)
    septic = onset_hour >= 0
    if np.isinf(width_hours):
        return septic.astype(np.int8)
    rel = hour_index - onset_hour.astype(float)
    return (septic & (rel >= -width_hours)).astype(np.int8)


def sham_target(y: np.ndarray, hour_index: np.ndarray, onset_hour: np.ndarray,
                width_hours: float, seed: int = 1) -> np.ndarray:
    """A prevalence-matched control with the temporal placement destroyed.

    Adds exactly as many positive hours as ``supervision_target(width_hours)`` does,
    but scatters them at random through septic stays instead of placing them before
    onset. Same patients, same class balance, no window. This is the control that
    decided the mechanism, so it lives in the module rather than in a scratch script.
    """
    y = np.asarray(y).astype(np.int8)
    real = supervision_target(y, hour_index, onset_hour, width_hours)
    out = y.copy()
    pool = np.flatnonzero((np.asarray(onset_hour) >= 0) & (y == 0))
    n_extra = int(real.sum() - y.sum())
    if n_extra > 0 and len(pool):
        rng = np.random.default_rng(seed)
        out[rng.choice(pool, size=min(n_extra, len(pool)), replace=False)] = 1
    return out


@dataclass
class SupervisionBooster:
    """A booster fitted at a chosen supervision density.

    Deliberately a thin wrapper over the same libraries and defaults as
    :class:`~backend.ml.models.boosters.UtilityAlignedBooster`, so a comparison between
    the two is a comparison of *targets* and not of tuning. Default is the patient
    label, which the evidence above says is as good as any window and has no parameter.
    """
    kind: str = "hgb"
    width_hours: float = float("inf")
    seed: int = 42
    calib_fraction: float = 0.15
    params: Dict[str, object] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    model_: Optional[object] = None
    calibrator_: Optional[ProbabilityCalibrator] = None

    def _make(self):
        p = {**DEFAULTS[self.kind], **(self.params or {})}
        if self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(random_state=self.seed, **p)
        if self.kind == "lgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(random_state=self.seed, n_jobs=-1, verbose=-1, **p)
        if self.kind == "xgb":
            import xgboost as xgb
            return xgb.XGBClassifier(random_state=self.seed, n_jobs=-1,
                                     tree_method="hist", eval_metric="logloss", **p)
        raise ValueError(f"unknown booster kind {self.kind!r}")

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
            hour_index: Optional[np.ndarray] = None,
            onset_hour: Optional[np.ndarray] = None) -> "SupervisionBooster":
        if hour_index is None or onset_hour is None:
            raise ValueError("the supervision target needs hour_index and onset_hour")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        groups = groups if groups is not None else np.arange(len(y))

        z = supervision_target(y, hour_index, onset_hour, self.width_hours)
        _, w = balanced_target(z)

        fit_idx, cal_idx = _grouped_holdout(y, groups, self.calib_fraction, self.seed)
        self.model_ = self._make()
        self.model_.fit(X[fit_idx], z[fit_idx], sample_weight=w[fit_idx])
        raw = self.model_.predict_proba(X[cal_idx])[:, 1]
        # Calibrated against the CHALLENGE label, never the denser one. The denser
        # target is a training device; a served probability still has to mean
        # P(label = 1) or the alarm threshold and the conformal bands are nonsense.
        # Isotonic is monotone, so this step cannot move a ranking metric — the gains
        # reported above are not a calibration artifact.
        self.calibrator_ = ProbabilityCalibrator().fit(raw, y[cal_idx])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.model_.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
        return self.calibrator_.transform(raw) if self.calibrator_ is not None else raw

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SupervisionBooster":
        import joblib
        return joblib.load(path)
