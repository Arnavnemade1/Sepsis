"""Boosted-tree detectors trained against the objective we are actually scored on.

Every tree model in this project so far has been fitted to the challenge label with
balanced class weights: each of a septic patient's labelled hours counts the same,
each non-septic hour counts ``1/47`` as much, and the hours the utility function
values most highly are not distinguished from the ones it values least.

:class:`UtilityAlignedBooster` fits ``(z, w)`` from :mod:`backend.ml.eval.utility`
instead — the label and per-row stake the official utility implies. Measured on this
cohort that moves ~19% of the positive-side utility mass onto rows the challenge
label calls negative (the ``[-12h, -6h)`` credit ramp), and replaces one flat
positive weight with the real 0.05 → 2.0 spread across the alarm window.

``kind`` selects the booster so the objective change can be separated from the
library change: ``hgb`` is the incumbent sklearn implementation, ``lgbm`` and ``xgb``
are the stronger modern ones. Any (kind × objective) pair is a valid ablation cell.

Probabilities are always isotonic-calibrated **against the challenge label** on a
patient-grouped holdout, so the output keeps meaning ``P(label = 1)`` no matter what
the model was fitted on, ECE stays comparable across cells, and — isotonic being
monotone — the ranking metrics are untouched by the calibration step.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.ml.eval.utility import utility_target
from backend.ml.models.calibration import ProbabilityCalibrator
from backend.ml.models.hdc_detector import _grouped_holdout

#: Per-booster defaults. Deliberately conservative and identical in spirit across
#: libraries so a cell-to-cell difference is the library, not a tuning accident.
DEFAULTS: Dict[str, Dict[str, object]] = {
    "hgb":  {"learning_rate": 0.06, "max_iter": 400, "max_leaf_nodes": 63,
             "l2_regularization": 1.0},
    "lgbm": {"learning_rate": 0.05, "n_estimators": 600, "num_leaves": 63,
             "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
             "colsample_bytree": 0.7, "reg_lambda": 1.0, "max_depth": -1},
    "xgb":  {"learning_rate": 0.05, "n_estimators": 600, "max_depth": 7,
             "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_lambda": 1.0},
}


def balanced_target(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The incumbent objective: the challenge label, inverse-prevalence weighted."""
    y = np.asarray(y).astype(int)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return y, np.where(y == 1, neg / pos, 1.0).astype(float)


@dataclass
class UtilityAlignedBooster:
    kind: str = "lgbm"
    objective: str = "utility"          # "utility" | "balanced"
    seed: int = 42
    calib_fraction: float = 0.15
    params: Dict[str, object] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    model_: Optional[object] = None
    calibrator_: Optional[ProbabilityCalibrator] = None

    # ------------------------------------------------------------------ #
    def _p(self) -> Dict[str, object]:
        merged = dict(DEFAULTS[self.kind])
        merged.update(self.params or {})
        return merged

    def _make(self):
        p = self._p()
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

    def _target(self, y, groups, hour_index, onset_hour):
        """The four cells of the objective ablation.

        ``utility`` changes two things at once — which rows count as positive, and
        how much each row counts — so it is decomposed here. ``utility-z`` isolates
        the relabelling (the early credit ramp becomes positive) and ``utility-w``
        isolates the stake-weighting (the label is untouched). Running all four says
        which half of the alignment is doing the work, instead of leaving a
        two-part change credited as one.
        """
        y = np.asarray(y).astype(int)
        if self.objective == "balanced":
            return balanced_target(y)
        if self.objective not in ("utility", "utility-z", "utility-w"):
            raise ValueError(f"unknown objective {self.objective!r}")
        if hour_index is None or onset_hour is None:
            raise ValueError(f"the {self.objective} objective needs hour_index and onset_hour")
        z, w = utility_target(y, groups, hour_index, onset_hour)
        if self.objective == "utility-z":            # relabel only
            return balanced_target(z.astype(int))
        # Rows with zero stake would vanish entirely; keep them at a floor so the
        # model still sees they exist but they cannot drive the fit.
        w = np.where(w <= 0, 1e-4, w)
        if self.objective == "utility-w":            # reweight only
            return y, w
        return z.astype(int), w

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None,
            hour_index: Optional[np.ndarray] = None,
            onset_hour: Optional[np.ndarray] = None) -> "UtilityAlignedBooster":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        groups = groups if groups is not None else np.arange(len(y))
        z, w = self._target(y, groups, hour_index, onset_hour)

        fit_idx, cal_idx = _grouped_holdout(y, groups, self.calib_fraction, self.seed)
        self.model_ = self._make()
        self.model_.fit(X[fit_idx], z[fit_idx], sample_weight=w[fit_idx])
        raw = self.model_.predict_proba(X[cal_idx])[:, 1]
        # Calibrate against the CHALLENGE label, not the training target: the output
        # must keep meaning P(label=1) whatever the fit optimised.
        self.calibrator_ = ProbabilityCalibrator().fit(raw, y[cal_idx])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.model_.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
        return self.calibrator_.transform(raw) if self.calibrator_ is not None else raw

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "UtilityAlignedBooster":
        import joblib
        return joblib.load(path)
