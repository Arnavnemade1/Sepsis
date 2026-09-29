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

Isotonic has a cost that took a long time to see: its map is a staircase. The shipped
stack's calibrator has 22 steps, so it emits a few hundred distinct probabilities, and
every hour on one step is tied with every other — the ranking the model made inside
the step is thrown away (AUROC 0.7476 raw vs 0.7425 calibrated on 6,852 unseen
patients), and one step held 46% of non-septic patients' peak scores, leaving no
operating point between a 5% and a 51% false-alarm rate. ``centred_map`` keeps the
calibration and drops the staircase: centred isotonic regression (Oron & Flournoy,
2017) interpolates through each step's centre instead of holding the step flat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    method: str = "isotonic"          # "isotonic" | "sigmoid" | "centred"
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
        if self.iso_ is not None and self.method == "centred":
            return centred_transform(self.iso_, raw_prob)
        if self.iso_ is not None:
            return self.iso_.transform(raw_prob)
        if self.platt_ is not None:
            logit = np.log(raw_prob / (1 - raw_prob)).reshape(-1, 1)
            return self.platt_.predict_proba(logit)[:, 1]
        return raw_prob


#: Slope of the tie-break beyond the first and last step centres, where the centred
#: map is flat. It moves no probability by more than 1e-7 and separates raw scores down
#: to ~3e-10 apart (below that the output's float spacing rounds them together); at
#: 1e-9 scores 1.4e-8 apart near the top step still tied.
TIE_SLOPE = 1e-7


def centred_map(iso: IsotonicRegression) -> Tuple[np.ndarray, np.ndarray]:
    """(centres, values) of a fitted isotonic map's constant steps.

    Read from the fitted thresholds alone — sklearn keeps each step's first and last
    raw score — so it applies to any fitted calibrator without its training data. The
    centre is the midpoint of the step's raw-score span, not the mean of the data on it
    (which the thresholds do not keep)."""
    x, v = iso.X_thresholds_, iso.y_thresholds_
    starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    ends = np.r_[starts[1:], len(v)] - 1
    return (x[starts] + x[ends]) / 2.0, v[starts]


def centred_transform(iso: IsotonicRegression, raw_prob: np.ndarray) -> np.ndarray:
    """Centred isotonic regression: linear through the step centres, so strictly
    increasing between the first and last centre and equal to the isotonic value at
    every centre; beyond them a ``TIE_SLOPE`` tie-break keeps the order of raw scores
    to float resolution."""
    c, v = centred_map(iso)
    raw_prob = np.asarray(raw_prob, dtype=np.float64)
    return np.clip(np.interp(raw_prob, c, v) + TIE_SLOPE * raw_prob, 0.0, 1.0)
