"""Gradient-boosted-trees benchmark.

Gradient boosting on tabular features is the de-facto state of the art for the
CinC 2019 task, so it is the right yardstick: if the data-grounded HDC detector
lands in the same neighbourhood, the HDC approach is scientifically credible
rather than just novel. We use sklearn's HistGradientBoostingClassifier so there
is no native build dependency, it handles NaNs natively, and it is fast.

The classifier is wrapped in isotonic probability calibration (out-of-fold) so
its probabilities — and therefore every threshold the clinician sets — are
trustworthy, matched to how the HDC detector is also calibrated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class GBMDetector:
    feature_names: List[str] = field(default_factory=list)
    seed: int = 42
    calibrate: bool = True
    model_: Optional[object] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBMDetector":
        y = np.asarray(y).astype(int)
        # Balance the rare positive class via sample weights.
        pos = max(int((y == 1).sum()), 1)
        neg = max(int((y == 0).sum()), 1)
        w = np.where(y == 1, neg / pos, 1.0)

        base = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=400,
            max_leaf_nodes=63,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.seed,
        )
        if self.calibrate:
            # Isotonic calibration via internal CV; sample_weight flows through.
            self.model_ = CalibratedClassifierCV(base, method="isotonic", cv=3)
            self.model_.fit(X, y, sample_weight=w)
        else:
            base.fit(X, y, sample_weight=w)
            self.model_ = base
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "GBMDetector":
        import joblib
        return joblib.load(path)
