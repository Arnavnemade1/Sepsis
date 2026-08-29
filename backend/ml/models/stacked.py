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

**Objective as a diversity axis.** A roster entry may name its own objective with an
``@`` suffix — ``"xgb@utility"`` — and only that member is fitted differently. This
matters because of a measured fact: applying the utility objective to *every* member
changes almost nothing (the resulting scores rank-correlate 0.965 with the incumbent
stack, so it is redundant with what stacking already does), whereas a member from a
different library rank-correlates only ~0.68 and is worth adding even though it is
weaker alone. Stacking pays for disagreement, not for strength, so the objective is
most useful applied selectively rather than uniformly.

**Objective.** By default every learner here is fitted to the challenge label with
inverse-prevalence class weights. Setting ``objective="patient"`` fits the patient-level
label — positive for every hour of a septic stay — which is the shipped production
target: see ``docs/SUPERVISION_DENSITY.md`` for why a denser label raises detection at a
fixed alarm budget while leaving AUROC flat. Setting ``objective="utility"`` fits the target the
official scoring function implies instead (:mod:`backend.ml.eval.utility`): the hours
in the ``[-12h, -6h)`` pre-onset credit ramp become positives — the label calls them
negatives, yet alarming there is worth up to a full utility point — and every row is
weighted by the utility actually riding on it rather than by one flat class weight.
Measured on this cohort that is worth more than any architectural change tried here.

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
from backend.ml.eval.utility import utility_target
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
    # n_jobs is a resource knob, not a model choice: -1 saturates every core, which
    # is right on a training box and hostile on the laptop someone is using.
    "et":   {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 50, "n_jobs": -1},
    "lr":   {"C": 0.5},
    "lgbm": {"learning_rate": 0.05, "n_estimators": 400, "num_leaves": 63,
             "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
             "colsample_bytree": 0.7, "reg_lambda": 1.0, "n_jobs": -1},
    "xgb":  {"learning_rate": 0.05, "n_estimators": 400, "max_depth": 7,
             "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_lambda": 1.0, "n_jobs": -1},
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
    #: "balanced" (challenge label, inverse-prevalence weights — the historic
    #: behaviour and the default so old artifacts and callers are unaffected) or
    #: "utility" (the target the official scoring function implies). Needs
    #: ``hour_index``/``onset_hour`` at fit time.
    objective: str = "balanced"

    # ------------------------------------------------------------------ #
    # Attribute access is routed through these helpers rather than read
    # directly off ``self`` because detectors are shipped as pickles: an artifact
    # trained before ``roster``/``params`` existed unpickles WITHOUT those
    # attributes (dataclass defaults do not run on unpickle), and must keep
    # serving. getattr-with-default keeps old artifacts loadable.
    def _roster(self) -> List[str]:
        return list(getattr(self, "roster", None) or DEFAULT_ROSTER)

    def _objective(self) -> str:
        return str(getattr(self, "objective", None) or "balanced")

    def _member(self, name: str):
        """``"xgb@utility"`` -> ``("xgb", "utility")``; a bare name inherits the stack's."""
        kind, _, obj = name.partition("@")
        return kind, (obj or self._objective())

    def _p(self, name: str) -> Dict[str, float]:
        kind = self._member(name)[0]
        merged = dict(DEFAULT_PARAMS.get(kind, {}))
        overrides = getattr(self, "params", None) or {}
        merged.update(overrides.get(kind, {}))
        merged.update(overrides.get(name, {}))     # exact-name override wins
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
    def _make(self, name: str, weighted: bool = False):
        name = self._member(name)[0]
        # ``weighted`` means the caller supplies per-row sample weights that already
        # carry the class balance; stacking sklearn's own "balanced" reweighting on
        # top would double-count it, so it is switched off in that case.
        s = self.seed
        p = self._p(name)
        cw = None if weighted else "balanced"
        if name == "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=float(p["learning_rate"]), max_iter=int(p["max_iter"]),
                max_leaf_nodes=int(p["max_leaf_nodes"]),
                l2_regularization=float(p["l2_regularization"]),
                class_weight=cw, random_state=s)
        if name == "hgb2":
            # a deliberately different-biased booster (shallower, faster, row/col
            # subsampled) for ensemble diversity — histogram-based so it stays tiny.
            return HistGradientBoostingClassifier(
                learning_rate=float(p["learning_rate"]), max_iter=int(p["max_iter"]),
                max_leaf_nodes=int(p["max_leaf_nodes"]), max_depth=int(p["max_depth"]),
                l2_regularization=float(p["l2_regularization"]),
                max_features=float(p["max_features"]),
                class_weight=cw, random_state=s + 7)
        if name == "et":
            # bagged randomised trees — the key complement to boosting. Depth/leaf
            # bounds keep the serialised model compact (unbounded ET on 150k rows is
            # ~300 MB; these bounds cost negligible accuracy and shrink it ~30x).
            return ExtraTreesClassifier(
                n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
                min_samples_leaf=int(p["min_samples_leaf"]),
                max_features="sqrt", class_weight=cw,
                n_jobs=int(p.get("n_jobs", -1)), random_state=s)
        if name == "lr":
            return LogisticRegression(class_weight=cw, max_iter=2000, C=float(p["C"]))
        if name == "xgb":
            # The strongest single learner measured in the frontier sweep, and — more
            # to the point for a stack — the one whose errors look least like the
            # others': under the utility objective it alarms on a third fewer
            # non-septic patients than the sklearn boosters do. Optional (a 7.5 MB
            # install), so the sklearn-only roster stays the dependency-free default.
            import xgboost as xgb
            return xgb.XGBClassifier(
                learning_rate=float(p["learning_rate"]), n_estimators=int(p["n_estimators"]),
                max_depth=int(p["max_depth"]), min_child_weight=float(p["min_child_weight"]),
                subsample=float(p["subsample"]), colsample_bytree=float(p["colsample_bytree"]),
                reg_lambda=float(p["reg_lambda"]),
                random_state=s + 11, n_jobs=int(p.get("n_jobs", -1)),
                tree_method="hist", eval_metric="logloss")
        if name == "lgbm":
            # Optional: a stronger booster library, but a native dependency the
            # free-tier deployment does not otherwise carry. Only reachable if the
            # caller puts it in the roster.
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                learning_rate=float(p["learning_rate"]), n_estimators=int(p["n_estimators"]),
                num_leaves=int(p["num_leaves"]), min_child_samples=int(p["min_child_samples"]),
                subsample=float(p["subsample"]), subsample_freq=int(p["subsample_freq"]),
                colsample_bytree=float(p["colsample_bytree"]),
                reg_lambda=float(p["reg_lambda"]), class_weight=cw,
                random_state=s + 3, n_jobs=int(p.get("n_jobs", -1)), verbose=-1)
        raise ValueError(name)

    def _fit_base(self, name, X, y, Xpred, scaler, w=None):
        kind = self._member(name)[0]
        m = self._make(kind, weighted=w is not None)
        if kind == "xgb" and w is None:
            # XGBoost has no ``class_weight``; without this it would be the one member
            # of a "balanced" roster that silently trains unbalanced.
            pos = max(int((np.asarray(y) == 1).sum()), 1)
            neg = max(int((np.asarray(y) == 0).sum()), 1)
            m.set_params(scale_pos_weight=neg / pos)
        if kind == "lr":
            m.fit(scaler.transform(X), y, sample_weight=w)
            return m.predict_proba(scaler.transform(Xpred))[:, 1], m
        m.fit(X, y, sample_weight=w)
        return m.predict_proba(Xpred)[:, 1], m

    def _targets(self, objective, y, groups, hour_index, onset_hour):
        """(target, sample_weight or None) for one objective."""
        if objective == "balanced":
            return np.asarray(y).astype(int), None
        if hour_index is None or onset_hour is None:
            raise ValueError(f'objective="{objective}" needs hour_index and onset_hour '
                             'at fit time')
        if objective == "patient":
            # Every hour of a septic stay counts as positive. The challenge label marks
            # 2.1% of hours positive though 12.8% of PATIENTS become septic, and the
            # measured consequence of that gap is in docs/SUPERVISION_DENSITY.md:
            # denser supervision raises detection at a fixed alarm budget at every
            # budget tested, by trading within-stay ordering for between-patient
            # discrimination. Returning ``None`` for the weight is deliberate — every
            # base learner then applies its own "balanced" class weighting, which is
            # how this target was measured.
            from backend.ml.models.supervision import supervision_target
            z = supervision_target(y, hour_index, onset_hour, float("inf"))
            return z.astype(int), None
        z, w = utility_target(y, groups, hour_index, onset_hour)
        # A zero-stake row would drop out of the fit entirely; a floor keeps it
        # present without letting it steer anything.
        return z.astype(int), np.maximum(w, 1e-4)

    def _target_book(self, y, groups, hour_index, onset_hour):
        """Every objective the roster (and the meta-learner) asks for, computed once."""
        wanted = {self._member(n)[1] for n in self._roster()} | {self._objective()}
        return {o: self._targets(o, y, groups, hour_index, onset_hour) for o in wanted}

    def _fit_hdc(self, ftr, ytr, gtr, fpred):
        from backend.ml.models.hdc_detector import HDCDetector
        h = HDCDetector(channels=list(self.channels)).fit(ftr, ytr, groups=gtr)
        return h.predict_proba(fpred), h

    # ------------------------------------------------------------------ #
    def fit(self, feats: pd.DataFrame, y: np.ndarray,
            groups: Optional[np.ndarray] = None,
            hour_index: Optional[np.ndarray] = None,
            onset_hour: Optional[np.ndarray] = None) -> "StackedEnsembleDetector":
        y = np.asarray(y).astype(int)
        self.feature_names_ = list(feats.columns)
        X = np.nan_to_num(feats.to_numpy())
        base = self._roster()
        self.base_order_ = base + (["hdc"] if self.use_hdc else [])
        groups = groups if groups is not None else np.arange(len(y))
        # Each roster member may ask for its own objective; compute each one once.
        book = self._target_book(y, groups, hour_index, onset_hour)
        target, weight = book[self._objective()]

        # --- out-of-fold meta-features (patient-grouped, leakage-safe) --- #
        oof = {nm: np.zeros(len(y)) for nm in self.base_order_}
        for itr, ite in stratified_group_folds(groups, y, self.inner_folds, self.seed):
            sc = StandardScaler().fit(X[itr])
            for nm in base:
                t_nm, w_nm = book[self._member(nm)[1]]
                oof[nm][ite] = self._fit_base(
                    nm, X[itr], t_nm[itr], X[ite], sc,
                    None if w_nm is None else w_nm[itr])[0]
            if self.use_hdc:
                # The HDC detector learns class archetypes, so it takes the target
                # but has no sample-weight channel to give it.
                oof["hdc"][ite] = self._fit_hdc(
                    feats.iloc[itr], target[itr], groups[itr], feats.iloc[ite])[0]
        OOF = np.column_stack([oof[nm] for nm in self.base_order_])

        # --- meta-learner on a fit slice; calibrate on a held-out slice -- #
        # The meta-learner combines under the same objective as its members, but
        # calibration is always fitted against the CHALLENGE label so the served
        # probability keeps meaning P(label = 1) whatever the stack optimised.
        fit_idx, cal_idx = _grouped_holdout(y, groups, self.calib_fraction, self.seed)
        self.meta_ = LogisticRegression(
            class_weight=None if weight is not None else "balanced",
            max_iter=2000, C=float(self._p("meta")["C"]))
        self.meta_.fit(OOF[fit_idx], target[fit_idx],
                       sample_weight=None if weight is None else weight[fit_idx])
        self.calibrator_ = ProbabilityCalibrator().fit(
            self.meta_.predict_proba(OOF[cal_idx])[:, 1], y[cal_idx])

        # --- refit every base learner on ALL data for inference ---------- #
        self.scaler_ = StandardScaler().fit(X)
        self.base_models_ = {}
        for nm in base:
            t_nm, w_nm = book[self._member(nm)[1]]
            self.base_models_[nm] = self._fit_base(
                nm, X, t_nm, X[:1], self.scaler_, w_nm)[1]
        if self.use_hdc:
            from backend.ml.models.hdc_detector import HDCDetector
            self.hdc_ = HDCDetector(channels=list(self.channels)).fit(
                feats, target, groups=groups)
        return self

    # ------------------------------------------------------------------ #
    def _base_matrix(self, feats: pd.DataFrame) -> np.ndarray:
        X = np.nan_to_num(feats.reindex(columns=self.feature_names_, fill_value=0.0).to_numpy())
        cols = []
        for nm in self.base_order_:
            if nm == "hdc":
                cols.append(self.hdc_.predict_proba(feats))
            elif self._member(nm)[0] == "lr":
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
