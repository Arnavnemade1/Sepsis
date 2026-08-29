"""WHO × WHEN — factorising sepsis detection into the two questions it conflates.

THE MEASUREMENT THAT MOTIVATES THIS MODULE
------------------------------------------
The pooled AUROC every sepsis paper reports — this project's included — silently
averages two different questions:

    BETWEEN : is this patient the kind of patient who becomes septic?
    WITHIN  : inside a septic patient's own stay, which hours are the pre-onset ones?

Decomposed on the dev cohort, the incumbent scores **0.77 between-patient and 0.50
within-patient**. The second number is chance. The detector is very good at picking
which patients to worry about and, having picked them, has no idea *when*. Every
lead-time figure this project has published is produced by thresholding a score that
is, inside the stay it is supposedly warning about, flat noise — so the alarm lands at
an effectively arbitrary hour, and a low threshold makes that arbitrary hour early.

That is worth stating plainly because it inverts the usual reading: a long median lead
time is not, by itself, evidence of early warning. It is equally consistent with a
timing-blind detector alarming at random.

WHY THE POOLED MODEL IS BLIND, AND WHY THAT IS FIXABLE
------------------------------------------------------
This is a textbook confounded regression. 94% of the rows belong to patients who never
become septic, so almost the entire gradient is spent on between-patient contrast, and
the model settles on features that identify *chronically sick patients*. Within-stay
dynamics are a small perturbation on top of a large patient effect, and they are
fitted last and worst.

Worse, the two directions can genuinely disagree. Time-in-ICU is the clearest case:
between patients, a long stay marks someone sicker; within a stay, later hours are the
ones nearer onset. A single pooled coefficient must serve both and ends up serving
neither — Simpson's paradox, in a feature the model already holds.

The classical remedy is not a better feature set: it is a **conditional likelihood
stratified by the confounder**. Conditional logistic regression and the Cox partial
likelihood both work this way — the patient's own baseline hazard cancels out of the
likelihood entirely, so only within-stay contrasts contribute. For a gradient-boosted
model the equivalent object is a pairwise ranking objective whose query groups are
patients: a pair contributes to the loss **only if both hours come from the same
stay**. The patient effect cannot be learned because it is differenced away, and the
model has nothing left to fit but the dynamics.

Two consequences make this more promising than it first looks:

  * the timing task is **balanced**. Restricted to septic stays, roughly a third of
    hours carry the label, against 2% pooled. The extreme imbalance that shapes every
    other decision in this project simply is not present in this sub-problem.
  * the timing task is **unexploited**. A within-patient AUROC of 0.50 means there is
    no existing skill to preserve, so any real signal here is additive rather than
    traded against something already working.

THE FACTORISATION
-----------------
For a patient-hour, the event "this hour is inside the alarm window" decomposes exactly:

    P(y_t = 1)  =  P(patient becomes septic)  ×  P(hour t is in the window | septic)
                   \_______ WHO ________/       \_________ WHEN __________/

The incumbent is a good estimator of the first term and a chance-level estimator of
the second. This module supplies the second and lets a meta-learner combine them, so
the two questions are estimated by two models that see the contrasts appropriate to
each, rather than by one model that sees both at once and resolves them badly.

HONESTY BOUNDARY
----------------
Whether this raises the pooled metric is an empirical question and is measured on the
project's fixed folds, not asserted. It is entirely possible that within-stay timing
is close to unpredictable from hourly vitals, in which case the ranker will score near
0.50 as well and the honest conclusion is that the lead times in this project's
reports are not recoverable — which is a finding about the deployability of every
early-warning claim on this dataset, and worth having either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def within_patient_auroc(score: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         keep: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """Mean per-patient AUROC over stays that contain both labels.

    Each stay is its own stratum, so between-patient variance is removed by
    construction rather than merely diluted — a pooled AUROC restricted to septic
    patients still lets an easy patient's hours outrank a hard patient's and reads
    higher than the timing skill actually present.
    """
    from backend.ml.eval import metrics
    keep = np.ones(len(y), dtype=bool) if keep is None else keep
    aucs: List[float] = []
    order = np.argsort(groups[keep], kind="stable")
    g = groups[keep][order]
    yy = y[keep][order]
    ss = score[keep][order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(g)]):
        seg_y = yy[a:b]
        if seg_y.min() == seg_y.max():
            continue
        aucs.append(metrics.auroc(seg_y, ss[a:b]))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


@dataclass
class WithinPatientTimingModel:
    """Ranks the hours inside one stay. Never learns which stay is which.

    Fitted with a pairwise ranking objective whose query group is the patient, so the
    loss contains only same-patient comparisons. The output is therefore meaningful as
    a *within-stay ordering* and carries no calibrated between-patient meaning at all —
    which is the point, and the reason it composes with the incumbent instead of
    competing with it.
    """
    kind: str = "xgb"
    params: Dict = field(default_factory=dict)
    feature_names: Optional[List[str]] = None
    model_: object = None
    n_queries_: int = 0

    _DEFAULTS = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
                 "subsample": 0.85, "colsample_bytree": 0.7,
                 "min_child_weight": 10, "reg_lambda": 2.0}

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "WithinPatientTimingModel":
        # Only stays containing BOTH labels carry any within-patient information; a
        # never-septic stay is all-zero and contributes no comparable pair. Dropping
        # them is not a convenience, it is what makes the objective conditional.
        informative = _mixed_label_patients(y, groups)
        mask = np.isin(groups, informative)
        Xf, yf, gf = X[mask], y[mask], groups[mask]

        order = np.argsort(gf, kind="stable")
        Xf, yf, gf = Xf[order], yf[order], gf[order]
        starts = np.flatnonzero(np.r_[True, gf[1:] != gf[:-1]])
        sizes = np.diff(np.r_[starts, len(gf)])
        self.n_queries_ = len(sizes)

        p = {**self._DEFAULTS, **self.params}
        if self.kind == "xgb":
            from xgboost import XGBRanker
            m = XGBRanker(objective="rank:pairwise", tree_method="hist",
                          random_state=0, **p)
            m.fit(Xf, yf, group=sizes)
        elif self.kind == "lgbm":
            from lightgbm import LGBMRanker
            m = LGBMRanker(objective="lambdarank", random_state=0, verbose=-1,
                           n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                           max_depth=p["max_depth"], subsample=p["subsample"],
                           colsample_bytree=p["colsample_bytree"],
                           min_child_samples=p["min_child_weight"])
            m.fit(Xf, yf, group=sizes)
        else:
            raise ValueError(f"unknown ranker kind {self.kind!r}")
        self.model_ = m
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("timing model is not fitted")
        return np.asarray(self.model_.predict(X), dtype=float)


def _mixed_label_patients(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Patients whose stay contains both a 0 hour and a 1 hour."""
    order = np.argsort(groups, kind="stable")
    g, yy = groups[order], y[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    keep = [g[a] for a, b in zip(starts, ends) if yy[a:b].min() != yy[a:b].max()]
    return np.array(keep, dtype=groups.dtype)


def causal_within_percentile(score: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Each hour's rank among that patient's hours **so far**, in [0, 1].

    The timing model's raw output has no cross-patient meaning, so it cannot be handed
    to a pooled meta-learner as-is. This converts it to a within-stay quantity that
    can be — and does so using only the prefix, so hour *t* is never positioned using
    hours the ward has not lived through yet. A retrospective percentile over the whole
    stay would be a far stronger feature and completely undeployable.
    """
    order = np.lexsort((np.arange(len(groups)), groups))
    inv = np.argsort(order, kind="stable")
    g, s = groups[order], score[order]
    out = np.empty(len(s), dtype=float)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(g)]):
        seg = s[a:b]
        # rank of each element among those at or before it: an O(n log n) scan would
        # be nicer, but stays are short (median ~38h) and this stays readable.
        out[a:b] = [(seg[: i + 1] <= seg[i]).mean() for i in range(len(seg))]
    return out[inv]
