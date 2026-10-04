"""The all-variable detector: one LightGBM over all 40 CinC 2019 variables.

GENESIS (docs/GENESIS.md) measured this model on a locked test set: AUROC 0.862 against
0.805 for the 7-vital stacked detector, the difference coming from reading the labs and
blood gases the wearable line never saw, and from regularising the booster properly. It
was never served because the locked run kept scores, not a model. This is the served
form, fitted by ``backend.ml.scripts.train_icu_detector``.

It reads the GENESIS features (``backend.ml.genesis.features``): 510 strictly causal
columns per hour, missing kept missing. ``record_features`` builds them for ONE record by
packing it as a one-patient ``RawStore``, so serving runs the exact code the model was
trained on; ``verify_ward_risk`` checks the per-record matrix against the cohort matrix.

Its probability is the booster's margin through the Platt map fitted on the patient-
held-out early-stopping slice (``genesis.learners.fit_lgb``), the same map the locked
evaluation used. Its alarm lines are its OWN operating point
(``artifacts/operating_point_icu.json``), derived on this fit and stamped with its id;
the wearable detector's lines do not transfer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AllVariableDetector:
    model_str: str                       # LightGBM model text
    best_iter: int
    calib: tuple                         # Platt (a, b) on the raw margin
    feature_names: List[str]
    spec: Dict[str, Any]                 # FeatureSpec fields
    detector_id: str
    fit_digest_: str
    train_patients: int
    meta: Dict[str, Any] = field(default_factory=dict)
    _booster: Any = field(default=None, repr=False, compare=False)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_booster"] = None                 # rebuilt from the text on first use
        return d

    # ------------------------------------------------------------------ #
    @property
    def booster(self):
        if self._booster is None:
            import lightgbm as lgb
            self._booster = lgb.Booster(model_str=self.model_str)
        return self._booster

    def feature_spec(self):
        from backend.ml.genesis.features import FeatureSpec
        return FeatureSpec(**self.spec)

    def predict_matrix(self, F: np.ndarray) -> np.ndarray:
        """Calibrated P(sepsis label) for rows of a GENESIS feature matrix."""
        m = self.booster.predict(np.ascontiguousarray(F, dtype=np.float32),
                                 num_iteration=self.best_iter, raw_score=True, num_threads=1)
        z = self.calib[0] * m + self.calib[1]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def record_features(self, frame, patient_id: str = "record") -> np.ndarray:
        """The GENESIS features of one hourly CinC frame, every hour, in column order."""
        from backend.ml.genesis import features as feat
        from backend.ml.genesis.rawstore import DYNAMIC, STATIC, RawStore
        n = len(frame)
        X = np.column_stack([frame[c].to_numpy(dtype=np.float32) if c in frame
                             else np.full(n, np.nan, np.float32) for c in DYNAMIC + STATIC])
        raw = RawStore(X=np.ascontiguousarray(X, dtype=np.float32), y=np.zeros(n, np.int8),
                       lens=np.array([n], dtype=np.int64), pid=np.array([patient_id]),
                       site=np.zeros(1, np.int8), columns=list(DYNAMIC + STATIC))
        F, names = feat.build(raw, self.feature_spec())
        if names != self.feature_names:
            raise ValueError("feature columns differ from the ones the model was trained on")
        return F

    def predict_record(self, record) -> np.ndarray:
        """Calibrated P(sepsis label) for every hour of a ``PatientRecord``."""
        return self.predict_matrix(self.record_features(record.frame, record.patient_id))
