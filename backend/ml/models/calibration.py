"""Post-hoc probability calibration.

The detectors are trained with class balancing so they *rank* the rare positive
class well. But balancing makes their raw probabilities reflect a 50/50 world,
not the true ~1.8% hourly prevalence — so "0.6" doesn't mean "60% chance". That
is exactly the miscalibration (high ECE) we measured.

The fix here is standard and honest: hold out a patient-grouped slice of the
training data at its TRUE prevalence and, *without any class weights*, fit a
monotonic map from the model's raw score to the observed frequency. Isotonic is
the default (flexible, monotonic); sigmoid (Platt) is available for small data.

Calibrating on a separate split (never the data the base model trained on) keeps
the calibration honest and avoids the optimism of in-sample fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    method: str = "isotonic"          # "isotonic" | "sigmoid"
    iso_: Optional[IsotonicRegression] = None
    platt_: Optional[LogisticRegression] = None

    def fit(self, raw_prob: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        raw_prob = np.clip(np.asarray(raw_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
        y = np.asarray(y).astype(int)
        if self.method == "sigmoid" or len(np.unique(y)) < 2 or len(y) < 50:
            # Platt scaling on the logit of the raw probability.
            logit = np.log(raw_prob / (1 - raw_prob)).reshape(-1, 1)
            self.platt_ = LogisticRegression(max_iter=2000).fit(logit, y)
            self.method = "sigmoid"
        else:
            self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.iso_.fit(raw_prob, y)
        return self

    def transform(self, raw_prob: np.ndarray) -> np.ndarray:
        raw_prob = np.clip(np.asarray(raw_prob, dtype=np.float64), 1e-6, 1 - 1e-6)
        if self.iso_ is not None:
            return self.iso_.transform(raw_prob)
        if self.platt_ is not None:
            logit = np.log(raw_prob / (1 - raw_prob)).reshape(-1, 1)
            return self.platt_.predict_proba(logit)[:, 1]
        return raw_prob
