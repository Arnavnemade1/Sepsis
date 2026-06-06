"""Causal feature engineering for hourly sepsis prediction.

Every feature at hour ``t`` is computed using only information available at or
before ``t``. This is the single most important property of the whole project:
a feature that peeks at the future inflates every metric and makes the detector
useless in deployment. The functions here are written so that leakage is
structurally impossible (forward-fill, expanding/rolling-past windows only).

Design mirrors the project's "digital twin" idea:
  * raw forward-filled values            (what the wearable last saw)
  * measurement flags + recency          (sparsity is itself predictive)
  * short-window trends                  (slopes / deltas — deterioration)
  * personal-baseline z-scores           (expanding per-patient normalization)
  * interpretable clinical composites    (shock index, SIRS/qSOFA counts)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.data.cinc2019 import PatientRecord

# Physiologically normal fallbacks for hours before a channel is ever measured.
# Used only to keep the feature matrix dense; the companion ``__measured`` /
# ``__age`` columns let the model know the value is imputed, not observed.
NORMAL_DEFAULTS: Dict[str, float] = {
    "HR": 75.0, "O2Sat": 98.0, "Temp": 36.8, "SBP": 120.0, "MAP": 90.0,
    "DBP": 75.0, "Resp": 16.0, "Lactate": 1.2, "WBC": 7.5, "Creatinine": 1.0,
    "Platelets": 250.0, "BUN": 15.0, "HCO3": 24.0, "pH": 7.4, "PaCO2": 40.0,
    "Bilirubin_total": 0.8, "Glucose": 110.0,
}


@dataclass
class FeatureConfig:
    """Controls which channels feed the model and the trend-window length."""
    channels: List[str]
    window: int = 8                 # hours for rolling trend features
    include_demographics: bool = True
    include_clinical_composites: bool = True

    @staticmethod
    def wearable_only(window: int = 8) -> "FeatureConfig":
        return FeatureConfig(channels=list(config.WEARABLE_CHANNELS), window=window)

    @staticmethod
    def lab_augmented(window: int = 8) -> "FeatureConfig":
        return FeatureConfig(
            channels=list(config.WEARABLE_CHANNELS) + list(config.LAB_CHANNELS),
            window=window,
        )


def build_patient_features(rec: PatientRecord, cfg: FeatureConfig) -> pd.DataFrame:
    """Return an (n_hours x n_features) causal feature frame for one patient."""
    raw = rec.frame
    n = len(raw)
    out: Dict[str, np.ndarray] = {}
    w = cfg.window

    for ch in cfg.channels:
        series = pd.to_numeric(raw.get(ch), errors="coerce") if ch in raw else pd.Series([np.nan] * n)
        series = series.reset_index(drop=True)
        measured = series.notna().astype(np.float32).to_numpy()

        # Forward fill (causal) then backfill-free default for the warm-up gap.
        ffilled = series.ffill()
        default = NORMAL_DEFAULTS.get(ch, float(np.nanmedian(series)) if measured.any() else 0.0)
        ffilled = ffilled.fillna(default)

        # Hours since the channel was last actually measured (recency of truth).
        age = _measurement_age(measured)

        # Short-window causal trend descriptors.
        roll = ffilled.rolling(window=w, min_periods=1)
        slope = _causal_slope(ffilled.to_numpy(), w)

        # Expanding personal baseline (the "digital twin" normalization).
        exp_mean = ffilled.expanding(min_periods=1).mean()
        exp_std = ffilled.expanding(min_periods=2).std().fillna(0.0)
        bz = (ffilled - exp_mean) / exp_std.replace(0.0, np.nan)
        bz = bz.fillna(0.0)

        out[ch] = ffilled.to_numpy(dtype=np.float32)
        out[f"{ch}__measured"] = measured
        out[f"{ch}__age"] = age
        out[f"{ch}__d1"] = ffilled.diff().fillna(0.0).to_numpy(dtype=np.float32)
        out[f"{ch}__rmean"] = roll.mean().to_numpy(dtype=np.float32)
        out[f"{ch}__rstd"] = roll.std().fillna(0.0).to_numpy(dtype=np.float32)
        out[f"{ch}__rmin"] = roll.min().to_numpy(dtype=np.float32)
        out[f"{ch}__rmax"] = roll.max().to_numpy(dtype=np.float32)
        out[f"{ch}__slope"] = slope
        out[f"{ch}__bz"] = bz.to_numpy(dtype=np.float32)

    # How many channels were observed this hour — a strong, honest signal.
    if cfg.channels:
        measured_cols = np.stack([out[f"{ch}__measured"] for ch in cfg.channels], axis=1)
        out["n_measured_this_hour"] = measured_cols.sum(axis=1).astype(np.float32)

    if cfg.include_demographics:
        out["Age"] = pd.to_numeric(raw.get("Age"), errors="coerce").fillna(65.0).to_numpy(dtype=np.float32)
        out["Gender"] = pd.to_numeric(raw.get("Gender"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        out["ICULOS"] = pd.to_numeric(raw.get("ICULOS"), errors="coerce").ffill().fillna(0.0).to_numpy(dtype=np.float32)
        out["HospAdmTime"] = pd.to_numeric(raw.get("HospAdmTime"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    if cfg.include_clinical_composites:
        for name, arr in _clinical_composites(out).items():
            out[name] = arr

    return pd.DataFrame(out)


def _measurement_age(measured: np.ndarray, cap: float = 48.0) -> np.ndarray:
    """Hours since the last actual measurement, capped. 0 on a measured hour."""
    age = np.empty(len(measured), dtype=np.float32)
    current = cap
    for i, m in enumerate(measured):
        current = 0.0 if m else min(cap, current + 1.0)
        age[i] = current
    return age


def _causal_slope(values: np.ndarray, w: int) -> np.ndarray:
    """Per-hour slope estimated from the last ``w`` points (units per hour).

    Uses a closed-form least-squares line over the trailing window — fully
    causal and vectorization-friendly without an O(n*w) python loop per row.
    """
    n = len(values)
    slope = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        lo = max(0, i - w + 1)
        seg = values[lo:i + 1]
        m = len(seg)
        if m < 2:
            continue
        x = np.arange(m, dtype=np.float64)
        x -= x.mean()
        denom = float((x * x).sum())
        if denom == 0.0:
            continue
        slope[i] = float((x * (seg - seg.mean())).sum() / denom)
    return slope


def _clinical_composites(out: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Interpretable bedside scores computed from the ffilled vitals.

    These keep continuity with the existing NeurosymbolicEngine (SIRS/qSOFA) and
    give the tree/HDC models a head start on well-established risk signals.
    """
    def g(ch: str, default: float) -> np.ndarray:
        return out.get(ch, np.full_like(next(iter(out.values())), default))

    hr = g("HR", 75.0)
    sbp = g("SBP", 120.0)
    temp = g("Temp", 36.8)
    resp = g("Resp", 16.0)
    o2 = g("O2Sat", 98.0)
    map_ = g("MAP", 90.0)

    shock_index = hr / np.clip(sbp, 1.0, None)

    sirs = (
        ((temp > 38.0) | (temp < 36.0)).astype(np.float32)
        + (hr > 90.0).astype(np.float32)
        + (resp > 20.0).astype(np.float32)
    )
    qsofa = (
        (resp >= 22.0).astype(np.float32)
        + (sbp <= 100.0).astype(np.float32)
    )
    hypoxia = (o2 < 92.0).astype(np.float32)
    hypotension = (map_ < 65.0).astype(np.float32)

    return {
        "comp_shock_index": shock_index.astype(np.float32),
        "comp_sirs": sirs,
        "comp_qsofa": qsofa,
        "comp_hypoxia": hypoxia,
        "comp_hypotension": hypotension,
    }


@dataclass
class FeatureMatrix:
    """Stacked design matrix across patients with grouping for safe CV."""
    X: np.ndarray                 # (n_rows, n_features) float32
    y: np.ndarray                 # (n_rows,) int8 label
    groups: np.ndarray           # (n_rows,) patient_id per row (no leakage)
    feature_names: List[str]
    hour_index: np.ndarray       # ICU hour (0-based) of each row
    onset_hour: np.ndarray       # per-row labelled onset hour (-1 if never)


def build_dataset(records: List[PatientRecord], cfg: FeatureConfig) -> FeatureMatrix:
    """Assemble a leakage-free stacked dataset from many patients."""
    frames: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    hours: List[np.ndarray] = []
    onsets: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None

    for rec in records:
        feats = build_patient_features(rec, cfg)
        if feature_names is None:
            feature_names = list(feats.columns)
        frames.append(feats.to_numpy(dtype=np.float32))
        ys.append(rec.labels.astype(np.int8))
        groups.append(np.array([rec.patient_id] * rec.n_hours))
        hours.append(np.arange(rec.n_hours))
        onset = rec.label_onset_hour
        onsets.append(np.full(rec.n_hours, -1 if onset is None else onset))

    if not frames:
        raise ValueError("No records to build a dataset from.")

    return FeatureMatrix(
        X=np.concatenate(frames, axis=0),
        y=np.concatenate(ys, axis=0),
        groups=np.concatenate(groups, axis=0),
        feature_names=feature_names or [],
        hour_index=np.concatenate(hours, axis=0),
        onset_hour=np.concatenate(onsets, axis=0),
    )
