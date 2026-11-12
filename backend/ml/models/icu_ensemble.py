"""Research ICU ensemble: complementary boosters over causal observation dynamics.

This is deliberately separate from the promoted detector. The experiment in
``scripts.improve_icu`` selects its blend on development patients and measures it
on other patients; neither inference nor feature construction reads outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.special import expit, logit

from backend.ml.genesis.features import _Seg, _clean, build
from backend.ml.genesis.rawstore import DYNAMIC, STATIC, RawStore

FEATURE_VERSION = "observed-dynamics-v1"


def trajectory_block(raw: RawStore) -> Tuple[np.ndarray, List[str]]:
    """Changes relative to *observed* history, excluding the current observation.

    Forward-filled values are useful levels but are not independent measurements.
    Here slopes and variability use only actual samples. Innovations compare the
    latest known value to the preceding 6/24 hours, with missing history left NaN.
    All sums restart at patient boundaries; no fitted population statistics exist.
    """
    s = _Seg(raw)
    columns, names = [], []
    for ch in DYNAMIC:
        v = _clean(raw, ch)
        obs = np.isfinite(v)
        last = s.last_obs(obs)
        lv = np.where(last >= 0, v[np.maximum(last, 0)], np.nan)
        value = np.where(obs, v, 0.0)
        t = s.hour.astype(float)
        terms = (obs, value, value * value, obs * t, obs * t * t, value * t)
        # Prefix at i-1: the current measurement cannot dilute its own surprise.
        prefixes = [np.nan_to_num(s.shift(s.prefix(a), 1), nan=0.0) for a in terms]
        for window in (6, 24):
            n, sy, syy, st, stt, sty = [s.window_sum(p, window) for p in prefixes]
            mean = sy / np.maximum(n, 1)
            std = np.sqrt(np.maximum(syy / np.maximum(n, 1) - mean * mean, 0))
            den = n * stt - st * st
            slope = np.divide(n * sty - st * sy, den,
                              out=np.full(s.N, np.nan), where=(n >= 2) & (den > 1e-9))
            innovation = np.where(n >= 2, (lv - mean) / (std + 0.1 + 0.01 * np.abs(mean)), np.nan)
            for suffix, a in (("obs_slope", slope), ("obs_std", np.where(n >= 2, std, np.nan)),
                              ("innovation", np.clip(innovation, -50, 50))):
                names.append(f"{ch}__{suffix}{window}")
                columns.append(a.astype(np.float32))
    return np.column_stack(columns), names


def blend(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    p, w = np.asarray(probabilities), np.asarray(weights, dtype=float)
    if p.ndim != 2 or w.shape != (p.shape[1],):
        raise ValueError("one blend weight is required per member")
    if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError("blend weights must be finite, nonnegative, and sum to one")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("members must return finite probabilities in [0, 1]")
    return p @ w


@dataclass
class ICUEnsembleDetector:
    """Native model payloads plus a frozen convex blend and monotone calibration.

    ``columns`` is a prefix length: base members read the original 510 columns,
    trajectory members also read the appended observation-dynamics block.
    """
    members: List[Dict[str, Any]]
    weights: List[float]
    feature_names: List[str]
    calib: Tuple[float, float] = (1.0, 0.0)
    detector_id: str = "research:icu-ensemble"
    meta: Dict[str, Any] = field(default_factory=dict)
    _models: Dict[int, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        blend(np.zeros((1, len(self.members))), np.asarray(self.weights))
        if not np.isfinite(self.calib).all() or self.calib[0] <= 0:
            raise ValueError("calibration must preserve ranking")
        for member in self.members:
            if member["kind"] not in ("lgbm", "xgb"):
                raise ValueError("unknown ensemble member")
            if not 0 < member["columns"] <= len(self.feature_names):
                raise ValueError("invalid member feature schema")

    def __getstate__(self):
        return {**self.__dict__, "_models": {}}

    def member_predict(self, F: np.ndarray, index: int, threads: int = 1) -> np.ndarray:
        member = self.members[index]
        if index not in self._models:
            if member["kind"] == "lgbm":
                import lightgbm as lgb
                self._models[index] = lgb.Booster(model_str=member["model"])
            else:
                import xgboost as xgb
                self._models[index] = xgb.Booster(model_file=bytearray(member["model"]))
        model = self._models[index]
        X = np.ascontiguousarray(F[:, :member["columns"]], dtype=np.float32)
        if member["kind"] == "lgbm":
            margin = model.predict(X, raw_score=True, num_iteration=member["trees"], num_threads=threads)
        else:
            model.set_param({"nthread": threads})
            margin = model.inplace_predict(X, predict_type="margin", iteration_range=(0, member["trees"]))
        a, b = member["calib"]
        return expit(a * margin + b)

    def predict_matrix(self, F: np.ndarray, batch: int = 16384, threads: int = 1) -> np.ndarray:
        if len(F.shape) != 2 or F.shape[1] != len(self.feature_names):
            raise ValueError("feature columns differ from the trained schema")
        if batch <= 0:
            raise ValueError("batch must be positive")
        out = np.empty(F.shape[0])
        for start in range(0, len(out), batch):
            X = F[start:start + batch]
            # Zero-weight members need neither loading nor evaluation.
            p = np.zeros((len(X), len(self.members)))
            for j, weight in enumerate(self.weights):
                if weight:
                    p[:, j] = self.member_predict(X, j, threads)
            raw = blend(p, np.asarray(self.weights))
            out[start:start + batch] = expit(self.calib[0] * logit(np.clip(raw, 1e-12, 1-1e-12)) + self.calib[1])
        return out

    def record_features(self, frame, patient_id: str = "record") -> np.ndarray:
        if not len(frame):
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        X = np.column_stack([frame[c].to_numpy(dtype=np.float32) if c in frame
                             else np.full(len(frame), np.nan, np.float32) for c in DYNAMIC + STATIC])
        raw = RawStore(X, np.zeros(len(frame), np.int8), np.array([len(frame)]),
                       np.array([patient_id]), np.zeros(1, np.int8), DYNAMIC + STATIC)
        base, names = build(raw)
        dynamic, extra = trajectory_block(raw)
        if names + extra != self.feature_names:
            raise ValueError("feature implementation differs from the trained schema")
        return np.hstack((base, dynamic))

    def predict_record(self, record) -> np.ndarray:
        return self.predict_matrix(self.record_features(record.frame, record.patient_id))
