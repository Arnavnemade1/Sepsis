"""Stacked-ensemble sepsis detector — the gap-closing production detector.

The rigorous finding of this project is that no single hyperdimensional model beats
gradient-boosted trees on this tabular task (see backend/ml/README.md). The honest
way to *exceed* the single-GBM benchmark is not a cleverer HDC, but **stacked
generalisation**: train several learners with genuinely different inductive biases,
then let a meta-learner combine their out-of-fold predictions. Where their errors are
uncorrelated, the ensemble beats every member.

Base learners (diverse on purpose, all size-bounded so the artifact stays small):
  * ``hgb``  — histogram gradient boosting (the strong tabular baseline)
  * ``hgb2`` — a shallower, subsampled booster with a different bias (diversity)
  * ``et``   — extremely randomised trees, depth/leaf-bounded (bagging, not boosting)
  * ``lr``   — L2 logistic regression (linear signal)
  * ``hdc``  — the data-grounded HDC detector (case-based similarity; optional) — so
               the project's hyperdimensional engine remains a first-class contributor
               and its meta-weight is reported honestly.

An unbounded ExtraTrees on the full data serialises to ~300 MB; the depth/leaf bounds
here keep the whole ensemble a few MB at negligible accuracy cost — small enough to
ship in the repo like the single-GBM artifact it replaces.

Leakage discipline: the meta-features are **patient-grouped out-of-fold** predictions
(a patient's rows never train a base learner whose prediction on that patient feeds
the meta-learner), and a final patient-grouped holdout calibrates the output so the
probabilities — and every clinician threshold — stay trustworthy (matches how the GBM
and HDC are calibrated). Takes a feature DataFrame, exposing no ``feature_names``, so
:class:`~backend.aegis.core.AegisCore` routes it a DataFrame at serve time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backend.ml.eval.splits import stratified_group_folds
from backend.ml.models.calibration import ProbabilityCalibrator
from backend.ml.models.hdc_detector import _grouped_holdout


def _class_weights(y: np.ndarray) -> np.ndarray:
    p = max(int((y == 1).sum()), 1)
    n = max(int((y == 0).sum()), 1)
    return np.where(y == 1, n / p, 1.0)


#: Default base-learner hyperparameters. ``StackedEnsembleDetector.params`` overrides
#: these per learner; the self-improvement search (:mod:`backend.ml.selfimprove`)
#: mutates them. Kept as data so a searched config is a plain JSON-serialisable dict.
DEFAULT_PARAMS: Dict[str, Dict[str, float]] = {
    "hgb":  {"learning_rate": 0.06, "max_iter": 400, "max_leaf_nodes": 63,
             "l2_regularization": 1.0},
    "hgb2": {"learning_rate": 0.12, "max_iter": 300, "max_leaf_nodes": 15,
             "max_depth": 4, "l2_regularization": 0.1, "max_features": 0.6},
    "et":   {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 50},
    "lr":   {"C": 0.5},
    "meta": {"C": 1.0},
}

DEFAULT_ROSTER: List[str] = ["hgb", "hgb2", "et", "lr"]


@dataclass
class StackedEnsembleDetector:
    channels: List[str] = field(default_factory=list)
    use_hdc: bool = True
    inner_folds: int = 3
    calib_fraction: float = 0.15
    seed: int = 42
    #: Which base learners to stack (subset of DEFAULT_ROSTER). Searchable.
    roster: List[str] = field(default_factory=lambda: list(DEFAULT_ROSTER))
    #: Per-learner hyperparameter overrides, e.g. {"hgb": {"max_iter": 250}}.
    #: Only the named keys override; everything else falls back to DEFAULT_PARAMS.
    params: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Attribute access is routed through these helpers rather than read
    # directly off ``self`` because detectors are shipped as pickles: an artifact
    # trained before ``roster``/``params`` existed unpickles WITHOUT those
    # attributes (dataclass defaults do not run on unpickle), and must keep
    # serving. getattr-with-default keeps old artifacts loadable.
    def _roster(self) -> List[str]:
        return list(getattr(self, "roster", None) or DEFAULT_ROSTER)

    def _p(self, name: str) -> Dict[str, float]:
        merged = dict(DEFAULT_PARAMS.get(name, {}))
        merged.update((getattr(self, "params", None) or {}).get(name, {}))
        return merged

    # learned state
    feature_names_: List[str] = field(default_factory=list)
    base_order_: List[str] = field(default_factory=list)
    base_models_: Dict[str, object] = field(default_factory=dict)
    scaler_: Optional[StandardScaler] = None
    meta_: Optional[LogisticRegression] = None
    calibrator_: Optional[ProbabilityCalibrator] = None
    hdc_: Optional[object] = None

    # ------------------------------------------------------------------ #
    def _make(self, name: str):
        s = self.seed
        p = self._p(name)
        if name == "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=float(p["learning_rate"]), max_iter=int(p["max_iter"]),
                max_leaf_nodes=int(p["max_leaf_nodes"]),
                l2_regularization=float(p["l2_regularization"]),
                class_weight="balanced", random_state=s)
        if name == "hgb2":
            # a deliberately different-biased booster (shallower, faster, row/col
            # subsampled) for ensemble diversity — histogram-based so it stays tiny.
            return HistGradientBoostingClassifier(
                learning_rate=float(p["learning_rate"]), max_iter=int(p["max_iter"]),
                max_leaf_nodes=int(p["max_leaf_nodes"]), max_depth=int(p["max_depth"]),
                l2_regularization=float(p["l2_regularization"]),
                max_features=float(p["max_features"]),
                class_weight="balanced", random_state=s + 7)
        if name == "et":
            # bagged randomised trees — the key complement to boosting. Depth/leaf
            # bounds keep the serialised model compact (unbounded ET on 150k rows is
            # ~300 MB; these bounds cost negligible accuracy and shrink it ~30x).
            return ExtraTreesClassifier(
                n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
                min_samples_leaf=int(p["min_samples_leaf"]),
                max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=s)
        if name == "lr":
            return LogisticRegression(
                class_weight="balanced", max_iter=2000, C=float(p["C"]))
        raise ValueError(name)

    def _fit_base(self, name, X, y, Xpred, scaler):
        m = self._make(name)
        if name == "lr":
            m.fit(scaler.transform(X), y)
            return m.predict_proba(scaler.transform(Xpred))[:, 1], m
        m.fit(X, y)
        return m.predict_proba(Xpred)[:, 1], m

    def _fit_hdc(self, ftr, ytr, gtr, fpred):
        from backend.ml.models.hdc_detector import HDCDetector
        h = HDCDetector(channels=list(self.channels)).fit(ftr, ytr, groups=gtr)
        return h.predict_proba(fpred), h

    # ------------------------------------------------------------------ #
    def fit(self, feats: pd.DataFrame, y: np.ndarray,
            groups: Optional[np.ndarray] = None) -> "StackedEnsembleDetector":
        y = np.asarray(y).astype(int)
        self.feature_names_ = list(feats.columns)
        X = np.nan_to_num(feats.to_numpy())
        base = self._roster()
        self.base_order_ = base + (["hdc"] if self.use_hdc else [])
        groups = groups if groups is not None else np.arange(len(y))

        # --- out-of-fold meta-features (patient-grouped, leakage-safe) --- #
        oof = {nm: np.zeros(len(y)) for nm in self.base_order_}
        for itr, ite in stratified_group_folds(groups, y, self.inner_folds, self.seed):
            sc = StandardScaler().fit(X[itr])
            for nm in base:
                oof[nm][ite] = self._fit_base(nm, X[itr], y[itr], X[ite], sc)[0]
            if self.use_hdc:
                oof["hdc"][ite] = self._fit_hdc(
                    feats.iloc[itr], y[itr], groups[itr], feats.iloc[ite])[0]
        OOF = np.column_stack([oof[nm] for nm in self.base_order_])

        # --- meta-learner on a fit slice; calibrate on a held-out slice -- #
        fit_idx, cal_idx = _grouped_holdout(y, groups, self.calib_fraction, self.seed)
        self.meta_ = LogisticRegression(
            class_weight="balanced", max_iter=2000, C=float(self._p("meta")["C"]))
        self.meta_.fit(OOF[fit_idx], y[fit_idx])
        self.calibrator_ = ProbabilityCalibrator().fit(
            self.meta_.predict_proba(OOF[cal_idx])[:, 1], y[cal_idx])

        # --- refit every base learner on ALL data for inference ---------- #
        self.scaler_ = StandardScaler().fit(X)
        self.base_models_ = {}
        for nm in base:
            self.base_models_[nm] = self._fit_base(nm, X, y, X[:1], self.scaler_)[1]
        if self.use_hdc:
            from backend.ml.models.hdc_detector import HDCDetector
            self.hdc_ = HDCDetector(channels=list(self.channels)).fit(feats, y, groups=groups)
        return self

    # ------------------------------------------------------------------ #
    def _base_matrix(self, feats: pd.DataFrame) -> np.ndarray:
        X = np.nan_to_num(feats.reindex(columns=self.feature_names_, fill_value=0.0).to_numpy())
        cols = []
        for nm in self.base_order_:
            if nm == "hdc":
                cols.append(self.hdc_.predict_proba(feats))
            elif nm == "lr":
                cols.append(self.base_models_[nm].predict_proba(self.scaler_.transform(X))[:, 1])
            else:
                cols.append(self.base_models_[nm].predict_proba(X)[:, 1])
        return np.column_stack(cols)

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        """Calibrated P(sepsis) per row. Accepts a feature DataFrame."""
        if not isinstance(feats, pd.DataFrame):
            raise TypeError("StackedEnsembleDetector.predict_proba expects a DataFrame")
        raw = self.meta_.predict_proba(self._base_matrix(feats))[:, 1]
        return self.calibrator_.transform(raw) if self.calibrator_ is not None else raw

    def meta_weights(self) -> Dict[str, float]:
        """The meta-learner's per-base-learner coefficient (honest contribution)."""
        return dict(zip(self.base_order_, [float(c) for c in self.meta_.coef_[0]]))

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "StackedEnsembleDetector":
        import joblib
        return joblib.load(path)
