"""HDC-augmented gradient-boosted sepsis detector — an EXPERIMENTAL, NOT-YET-VALIDATED
alternative to the shipped GBM. Read the negative-result history before using this.

The honest finding (patient-grouped CV, CinC 2019, wearable-only channels — see
``backend/ml/README.md`` for the exact numbers and how to reproduce them):

  * A single 800-patient train/test split showed a dramatic-looking jump from feeding
    HDC archetype-similarity signals into the GBM as extra columns (+79% AUPRC).
  * That did **not** replicate. A fast-config 3-fold CV showed no benefit (a tie, then
    a loss). A CV run using the exact retrained-archetype config that produced the
    800-patient result gave: GBM 0.7235±0.004 AUROC vs GBM+HDC 0.7263±0.011 — a gap
    smaller than the fold-to-fold noise. **Statistically inconclusive, not a
    confirmed win.** The single-split result was noise from a small, high-variance
    test slice at ~2% prevalence — exactly the failure mode patient-grouped CV exists
    to catch.

Conclusion: this class is kept as a documented, reproducible experiment — not wired
into production. ``backend/aegis/core.py`` loads the plain calibrated GBM
(``artifacts/gbm_wearable.joblib``), which has no evidence of being beaten. Do not
promote ``HDCAugmentedGBM`` to production without a larger-scale CV (more patients,
more folds) that confirms a gain outside the noise band. This file exists so the null
result is reproducible, not to imply the experiment succeeded.

Probability-level averaging of HDC and GBM was also tested and did not help (HDC
alone is too weak to average in usefully) — stated, not hidden.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from backend.ml.models.hdc_detector import HDCDetector
from backend.ml.models.gbm_baseline import GBMDetector


@dataclass
class HDCAugmentedGBM:
    """Gradient booster over [physiological features ⊕ HDC archetype signals].

    Takes a feature DataFrame (like :class:`HDCDetector`) so it can compute the HDC
    signals from named columns. Deliberately exposes no ``feature_names`` attribute,
    so :class:`~backend.aegis.core.AegisCore` routes a DataFrame to ``predict_proba``."""

    channels: List[str]
    seed: int = 42
    hdc_kwargs: dict = field(default_factory=dict)
    # learned state
    raw_feature_names_: List[str] = field(default_factory=list)
    hdc_signal_names_: List[str] = field(default_factory=list)
    hdc_: Optional[HDCDetector] = None
    gbm_: Optional[GBMDetector] = None

    # ------------------------------------------------------------------ #
    def _hdc_matrix(self, feats: pd.DataFrame) -> np.ndarray:
        sig = self.hdc_.hdc_signals(feats)
        return np.column_stack([sig[k] for k in self.hdc_signal_names_])

    def _augment(self, feats: pd.DataFrame) -> np.ndarray:
        raw = feats.reindex(columns=self.raw_feature_names_, fill_value=0.0).to_numpy()
        H = self._hdc_matrix(feats)
        X = np.column_stack([raw, H])
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------ #
    def fit(self, feats: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray] = None
            ) -> "HDCAugmentedGBM":
        self.raw_feature_names_ = list(feats.columns)
        self.hdc_ = HDCDetector(channels=list(self.channels), **self.hdc_kwargs).fit(
            feats, y, groups=groups)
        self.hdc_signal_names_ = list(self.hdc_.hdc_signals(feats.iloc[:1]).keys())
        X = self._augment(feats)
        self.gbm_ = GBMDetector(
            feature_names=self.raw_feature_names_ + self.hdc_signal_names_,
            seed=self.seed,
        ).fit(X, y, groups=groups)
        return self

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        """Calibrated P(sepsis) per row. Accepts a feature DataFrame."""
        if not isinstance(feats, pd.DataFrame):
            raise TypeError("HDCAugmentedGBM.predict_proba expects a feature DataFrame")
        return self.gbm_.predict_proba(self._augment(feats))

    # ------------------------------------------------------------------ #
    def hdc_signals(self, feats: pd.DataFrame):
        """Expose the underlying HDC archetype signals (case-based evidence)."""
        return self.hdc_.hdc_signals(feats)

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HDCAugmentedGBM":
        import joblib
        return joblib.load(path)
