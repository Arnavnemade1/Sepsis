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

from dataclasses import dataclass, field
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
    """Controls which channels feed the model and the temporal context.

    The base single ``window`` features (``HR__slope`` etc.) are always produced
    so existing model/serving references stay valid. Setting ``extra_windows`` /
    ``lags`` / ``burden`` ADDS multi-scale temporal context on top — the Phase-7
    accuracy lever — without removing or renaming anything.
    """
    channels: List[str]
    window: int = 8                 # base window for rolling trend features
    extra_windows: List[int] = field(default_factory=list)   # additional scales
    lags: List[int] = field(default_factory=list)            # value - value[t-L]
    burden: bool = False            # cumulative abnormality counters (causal)
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

    @staticmethod
    def wearable_temporal() -> "FeatureConfig":
        """Multi-scale temporal feature set (Phase-7): short + long windows,
        multi-horizon lags, and cumulative deterioration burden."""
        return FeatureConfig(
            channels=list(config.WEARABLE_CHANNELS),
            window=8, extra_windows=[4, 16, 32], lags=[3, 6, 12], burden=True,
        )

    @staticmethod
    def lab_temporal() -> "FeatureConfig":
        return FeatureConfig(
            channels=list(config.WEARABLE_CHANNELS) + list(config.LAB_CHANNELS),
            window=8, extra_windows=[4, 16, 32], lags=[3, 6, 12], burden=True,
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

        # --- Phase-7 multi-scale temporal context (additive) --------------- #
        for ew in cfg.extra_windows:
            r = ffilled.rolling(window=ew, min_periods=1)
            out[f"{ch}__rmean{ew}"] = r.mean().to_numpy(dtype=np.float32)
            out[f"{ch}__rstd{ew}"] = r.std().fillna(0.0).to_numpy(dtype=np.float32)
            out[f"{ch}__slope{ew}"] = _causal_slope(ffilled.to_numpy(), ew)
        for lag in cfg.lags:
            out[f"{ch}__lag{lag}"] = (ffilled - ffilled.shift(lag)).fillna(0.0).to_numpy(dtype=np.float32)
        # Expanding (whole-stay-so-far) extremes — cumulative worst seen.
        out[f"{ch}__emin"] = ffilled.expanding(min_periods=1).min().to_numpy(dtype=np.float32)
        out[f"{ch}__emax"] = ffilled.expanding(min_periods=1).max().to_numpy(dtype=np.float32)

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

    if cfg.burden:
        for name, arr in _burden_counters(out).items():
            out[name] = arr

    return pd.DataFrame(out)


def _burden_counters(out: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Cumulative (causal) counts of abnormal hours — deterioration *dose*.

    A single abnormal reading is weak; sustained abnormality is the real sepsis
    trajectory. These running totals give the model cumulative burden, computed
    only from past+present hours (np.cumsum is strictly causal).
    """
    def g(ch, d):
        return out.get(ch, np.full_like(next(iter(out.values())), d))
    hr, sbp, temp = g("HR", 75.0), g("SBP", 120.0), g("Temp", 36.8)
    resp, o2, map_ = g("Resp", 16.0), g("O2Sat", 98.0), g("MAP", 90.0)
    flags = {
        "burden_tachy": (hr > 90.0),
        "burden_hypotension": (sbp <= 100.0),
        "burden_tachypnea": (resp >= 22.0),
        "burden_febrile": ((temp > 38.0) | (temp < 36.0)),
        "burden_hypoxia": (o2 < 92.0),
        "burden_map_low": (map_ < 65.0),
    }
    res = {k: np.cumsum(v.astype(np.float32)) for k, v in flags.items()}
    res["burden_total"] = np.sum([res[k] for k in flags], axis=0).astype(np.float32)
    return res


def _measurement_age(measured: np.ndarray, cap: float = 48.0) -> np.ndarray:
    """Hours since the last actual measurement, capped. 0 on a measured hour."""
    age = np.empty(len(measured), dtype=np.float32)
    current = cap
    for i, m in enumerate(measured):
        current = 0.0 if m else min(cap, current + 1.0)
        age[i] = current
    return age


def _causal_slope(values: np.ndarray, w: int) -> np.ndarray:
    """Per-hour least-squares slope over the trailing ``w`` points (units/hour).

    Fully causal and **vectorized**: for full windows the LS slope is a fixed
    FIR kernel applied by correlation (slope_t = Σ_j k_j · y_{t-w+1+j}, with
    k_j = (j - x̄) / Σ(x-x̄)²), which is mathematically identical to fitting a
    line but runs in one numpy pass instead of an O(n·w) Python loop. Only the
    short warm-up (first w-1 points) uses a tiny expanding-window loop.
    """
    n = len(values)
    slope = np.zeros(n, dtype=np.float32)
    if n < 2:
        return slope
    vals = values.astype(np.float64)
    w = min(w, n)

    if w >= 2:
        x = np.arange(w, dtype=np.float64)
        xc = x - x.mean()
        denom = float((xc * xc).sum())
        if denom > 0:
            # Correlate: out[k] = Σ_j vals[k+j]·xc[j] for window ending at k+w-1.
            conv = np.correlate(vals, xc, mode="valid") / denom
            slope[w - 1:] = conv

    # Warm-up: expanding-window LS slope for the first w-1 points (few iters).
    for i in range(1, w - 1):
        seg = vals[: i + 1]
        xs = np.arange(i + 1, dtype=np.float64)
        xs -= xs.mean()
        d = float((xs * xs).sum())
        if d > 0:
            slope[i] = float((xs * (seg - seg.mean())).sum() / d)
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


#: Records carrying this ``set_name`` were authored, not observed
#: (:mod:`backend.aegis.scenarios`). Duplicated here rather than imported so this
#: guard has no dependency on the agent layer and cannot be disabled by breaking it.
SYNTHETIC_SET = "synthetic"


def build_dataset(records: List[PatientRecord], cfg: FeatureConfig,
                  allow_synthetic: bool = False) -> FeatureMatrix:
    """Assemble a leakage-free stacked dataset from many patients.

    Refuses hand-written patients unless a caller says otherwise, in one line, in
    the diff. Every evaluation in this project funnels through here, so this is the
    single place a synthetic trajectory could turn into a reported accuracy number —
    and that is exactly the mistake the project already made once, when a detector
    tuned against its own simulator looked flawless and then scored near chance on
    real patients. A scenario is a legitimate way to ask what the system DID; it can
    never be evidence of how often the system is RIGHT, because we wrote it.
    """
    if not allow_synthetic:
        fake = [r.patient_id for r in records
                if getattr(r, "set_name", "") == SYNTHETIC_SET]
        if fake:
            raise ValueError(
                f"refusing to build an evaluation dataset from {len(fake)} synthetic "
                f"record(s) (e.g. {fake[0]}). Numbers from authored patients measure "
                "the generator, not the world. Pass allow_synthetic=True only for "
                "pipeline tests whose output is never reported as accuracy.")
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


def build_dataset_cached(records_fn, cfg: FeatureConfig, cache_key: str,
                         refresh: bool = False) -> FeatureMatrix:
    """Build a dataset with an on-disk cache (feature build is the slow step).

    ``records_fn`` is a zero-arg callable returning the records — only invoked on
    a cache miss, so a hit avoids even loading the raw data. The cache key must
    uniquely capture (feature_set, sets, limit) — the caller's responsibility.
    """
    import os
    cache_dir = os.path.join(config.ARTIFACTS_DIR, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{cache_key}.npz")
    if os.path.exists(path) and not refresh:
        d = np.load(path, allow_pickle=True)
        return FeatureMatrix(
            X=d["X"], y=d["y"], groups=d["groups"],
            feature_names=list(d["feature_names"]),
            hour_index=d["hour_index"], onset_hour=d["onset_hour"],
        )
    ds = build_dataset(records_fn(), cfg)
    np.savez_compressed(
        path, X=ds.X, y=ds.y, groups=ds.groups,
        feature_names=np.array(ds.feature_names, dtype=object),
        hour_index=ds.hour_index, onset_hour=ds.onset_hour,
    )
    return ds
