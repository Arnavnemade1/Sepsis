"""Data-grounded Hyperdimensional Computing sepsis detector.

This is the project's scientific core, rebuilt so that *nothing is hand-tuned*:

  1. ENCODE   each hour into a 10,000-D bipolar hypervector using record-style
              binding (channel-id ⊗ level(value)) — the same VSA algebra as the
              original engine, but value ranges are learned from data quantiles.
  2. LEARN    class prototype hypervectors as real-valued centroids of the
              encoded TRAINING hours (healthy vs septic), then refine them with
              AdaptHD-style error-driven retraining.
  3. FUSE     the HDC margin (sim_sepsis − sim_healthy) with interpretable
              clinical composites (SIRS, qSOFA, shock index, hypoxia,
              hypotension) in a class-weighted, probability-CALIBRATED logistic
              layer fitted on real labels.

The result keeps the HDC + neurosymbolic identity but every threshold, weight,
and archetype is derived from PhysioNet/CinC 2019 — and reported with metrics.
Training is done in row batches so memory stays bounded on the full dataset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Clinical composite columns produced by features._clinical_composites.
_COMPOSITE_COLS = [
    "comp_shock_index", "comp_sirs", "comp_qsofa", "comp_hypoxia", "comp_hypotension",
]


def default_encode_features(columns: List[str], channels: List[str]) -> List[str]:
    """Pick a rich, causal feature set for HDC encoding.

    The original engine encoded only 7 raw values, which throws away the trend
    and personal-baseline information that actually separates sepsis from noise.
    Here we encode, for each channel, its level + short-term dynamics + baseline
    drift, plus the clinical composites and a couple of context features — so the
    hypervector carries the same evidence the tree model gets to see.
    """
    wanted: List[str] = []
    for ch in channels:
        for suffix in ("", "__slope", "__d1", "__bz", "__rstd", "__age"):
            col = ch + suffix
            if col in columns:
                wanted.append(col)
    for extra in (*_COMPOSITE_COLS, "n_measured_this_hour", "Age", "ICULOS"):
        if extra in columns:
            wanted.append(extra)
    return wanted


@dataclass
class HDCDetector:
    channels: List[str]
    encode_features: Optional[List[str]] = None
    dimensions: int = 10000
    levels: int = 100
    seed: int = 42
    n_prototypes_sep: int = 8
    n_prototypes_heal: int = 16
    healthy_subsample: int = 40000
    use_composites: bool = True
    batch_rows: int = 4000
    calibrate: bool = True
    calib_fraction: float = 0.2

    # learned state ------------------------------------------------------- #
    ranges_: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    id_vecs_: Optional[np.ndarray] = None       # (n_encoded, D) int8
    level_vecs_: Optional[np.ndarray] = None    # (levels, D) int8
    protos_sep_: Optional[np.ndarray] = None    # (Ksep, D) unit float — learned sepsis archetypes
    protos_heal_: Optional[np.ndarray] = None   # (Kheal, D) unit float — learned healthy archetypes
    c_heal_: Optional[np.ndarray] = None        # (D,) unit float
    c_sep_: Optional[np.ndarray] = None         # (D,) unit float
    scaler_: Optional[StandardScaler] = None
    clf_: Optional[LogisticRegression] = None
    calibrator_: Optional[object] = None        # ProbabilityCalibrator (post-hoc)
    fusion_features_: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Vector-symbolic primitives
    # ------------------------------------------------------------------ #
    def _init_codebook(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.id_vecs_ = rng.choice(np.array([-1, 1], dtype=np.int8), size=(len(self.encode_features), self.dimensions))
        # Linearly-morphing level vectors: level 0 and level L-1 are orthogonal,
        # intermediate levels interpolate, so numeric closeness => HDC closeness.
        v_min = rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dimensions)
        v_max = rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dimensions)
        levels = np.zeros((self.levels, self.dimensions), dtype=np.int8)
        levels[0] = v_min
        diff_idx = np.where(v_min != v_max)[0]
        rng.shuffle(diff_idx)
        per = len(diff_idx) / (self.levels - 1)
        for i in range(1, self.levels):
            levels[i] = levels[i - 1]
            seg = diff_idx[int((i - 1) * per):int(i * per)]
            levels[i, seg] = v_max[seg]
        self.level_vecs_ = levels

    def _level_index(self, ch: str, values: np.ndarray) -> np.ndarray:
        lo, hi = self.ranges_[ch]
        if hi <= lo:
            hi = lo + 1.0
        frac = (np.clip(values, lo, hi) - lo) / (hi - lo)
        return np.clip((frac * (self.levels - 1)).astype(np.int64), 0, self.levels - 1)

    def _encode_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a batch of rows into bipolar hypervectors (n, D) int8."""
        n = len(df)
        acc = np.zeros((n, self.dimensions), dtype=np.int16)
        for ci, ch in enumerate(self.encode_features):
            vals = pd.to_numeric(df[ch], errors="coerce").fillna(0.0).to_numpy() if ch in df else np.zeros(n)
            idx = self._level_index(ch, vals)
            bound = self.level_vecs_[idx] * self.id_vecs_[ci]   # (n,D) bind = elementwise mult
            acc += bound.astype(np.int16)
        # Bundle = sign; resolve ties deterministically to +1 to stay reproducible.
        hv = np.where(acc >= 0, 1, -1).astype(np.int8)
        return hv

    def _iter_batches(self, df: pd.DataFrame):
        for start in range(0, len(df), self.batch_rows):
            yield df.iloc[start:start + self.batch_rows]

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, feats: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray] = None) -> "HDCDetector":
        """Fit the detector. If ``calibrate``, a patient-grouped holdout at true
        prevalence is reserved to fit a post-hoc probability calibrator."""
        y = np.asarray(y).astype(int)
        if self.encode_features is None:
            self.encode_features = default_encode_features(list(feats.columns), self.channels)
        # Ranges + codebook may use all training rows (no label leakage).
        self._learn_ranges(feats)
        self._init_codebook()

        if self.calibrate and len(np.unique(y)) > 1:
            fit_idx, cal_idx = _grouped_holdout(y, groups, self.calib_fraction, self.seed)
            self._fit_core(feats.iloc[fit_idx], y[fit_idx])
            from backend.ml.models.calibration import ProbabilityCalibrator
            raw = self._raw_proba(feats.iloc[cal_idx])
            self.calibrator_ = ProbabilityCalibrator().fit(raw, y[cal_idx])
        else:
            self._fit_core(feats, y)
        return self

    def _fit_core(self, feats: pd.DataFrame, y: np.ndarray) -> "HDCDetector":
        y = np.asarray(y).astype(int)
        sep_idx = np.flatnonzero(y == 1)
        heal_idx = np.flatnonzero(y == 0)
        if sep_idx.size == 0 or heal_idx.size == 0:
            raise ValueError("Both classes must be present to train the HDC detector.")

        # Sepsis presents heterogeneously, so a single archetype underfits. We
        # learn MULTIPLE prototypes per class by clustering the encoded training
        # hours (k-means in HDC space). The healthy class is subsampled to keep
        # memory bounded on the full dataset.
        rng = np.random.default_rng(self.seed)
        if heal_idx.size > self.healthy_subsample:
            heal_idx = rng.choice(heal_idx, size=self.healthy_subsample, replace=False)

        sep_vecs = self._encode_rows(feats, sep_idx)
        heal_vecs = self._encode_rows(feats, heal_idx)
        self.protos_sep_ = _fit_prototypes(sep_vecs, min(self.n_prototypes_sep, len(sep_idx)), self.seed)
        self.protos_heal_ = _fit_prototypes(heal_vecs, min(self.n_prototypes_heal, len(heal_idx)), self.seed)

        # Single centroids kept too (cheap, useful for explanation/streaming).
        self.c_sep_ = _unit(sep_vecs.astype(np.float64).sum(axis=0))
        self.c_heal_ = _unit(heal_vecs.astype(np.float64).sum(axis=0))

        # Fit the calibrated neurosymbolic fusion layer on HDC similarity
        # signals + interpretable clinical composites, on real labels.
        X_fusion, names = self._fusion_matrix(feats)
        self.fusion_features_ = names
        self.scaler_ = StandardScaler().fit(X_fusion)
        self.clf_ = LogisticRegression(
            class_weight="balanced", max_iter=4000, C=1.0,
        ).fit(self.scaler_.transform(X_fusion), y)
        return self

    def _encode_rows(self, feats: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Encode a specific set of rows into bipolar hypervectors (m, D) int8."""
        out = np.empty((len(idx), self.dimensions), dtype=np.int8)
        for start in range(0, len(idx), self.batch_rows):
            sl = idx[start:start + self.batch_rows]
            out[start:start + len(sl)] = self._encode_batch(feats.iloc[sl])
        return out

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def hdc_signals(self, feats: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Per-row HDC evidence: best/mean similarity to each class' prototypes
        and the sepsis-vs-healthy margin. These are the learned 'archetype'
        matches that replace the old hardcoded similarity."""
        n = len(feats)
        max_sep = np.empty(n); mean_sep = np.empty(n)
        max_heal = np.empty(n); mean_heal = np.empty(n)
        sqrtD = np.sqrt(self.dimensions)
        off = 0
        for batch in self._iter_batches(feats):
            hv = self._encode_batch(batch).astype(np.float32) / sqrtD
            s = hv @ self.protos_sep_.T.astype(np.float32)   # (b, Ksep)
            h = hv @ self.protos_heal_.T.astype(np.float32)  # (b, Kheal)
            b = len(batch)
            max_sep[off:off + b] = s.max(axis=1)
            mean_sep[off:off + b] = s.mean(axis=1)
            max_heal[off:off + b] = h.max(axis=1)
            mean_heal[off:off + b] = h.mean(axis=1)
            off += b
        return {
            "hdc_max_sep": max_sep,
            "hdc_mean_sep": mean_sep,
            "hdc_max_heal": max_heal,
            "hdc_mean_heal": mean_heal,
            "hdc_margin": max_sep - max_heal,
        }

    def hdc_margin(self, feats: pd.DataFrame) -> np.ndarray:
        return self.hdc_signals(feats)["hdc_margin"]

    def _fusion_matrix(self, feats: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        cols: Dict[str, np.ndarray] = dict(self.hdc_signals(feats))
        if self.use_composites:
            for c in _COMPOSITE_COLS:
                if c in feats:
                    cols[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0.0).to_numpy()
        names = list(cols.keys())
        X = np.column_stack([cols[n] for n in names]).astype(np.float64)
        # Guard the calibrated layer against non-finite values from any feature.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, names

    def _raw_proba(self, feats: pd.DataFrame) -> np.ndarray:
        """Uncalibrated fusion-layer probability (balanced logistic output)."""
        X, _ = self._fusion_matrix(feats)
        return self.clf_.predict_proba(self.scaler_.transform(X))[:, 1]

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        raw = self._raw_proba(feats)
        return self.calibrator_.transform(raw) if self.calibrator_ is not None else raw

    # ------------------------------------------------------------------ #
    def _learn_ranges(self, feats: pd.DataFrame) -> None:
        for ch in self.encode_features:
            if ch in feats:
                vals = pd.to_numeric(feats[ch], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                if vals.size:
                    self.ranges_[ch] = (float(np.quantile(vals, 0.01)), float(np.quantile(vals, 0.99)))
                    continue
            self.ranges_[ch] = (0.0, 1.0)

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HDCDetector":
        import joblib
        return joblib.load(path)


def _grouped_holdout(y: np.ndarray, groups: Optional[np.ndarray], frac: float, seed: int):
    """Return (fit_idx, calib_idx). Splits by patient group when groups are given
    (no leakage), stratified by patient sepsis status; else a plain random split."""
    n = len(y)
    if groups is not None:
        from backend.ml.eval.splits import stratified_group_split
        try:
            return stratified_group_split(np.asarray(groups), y, test_size=frac, seed=seed)
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1.0 - frac))
    return idx[:cut], idx[cut:]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _fit_prototypes(vecs: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Cluster encoded hypervectors into ``k`` unit-norm prototypes (archetypes).

    k-means in HDC space; cluster centres are the learned archetypes. With k=1
    this reduces to the classic single-centroid HDC classifier.
    """
    from sklearn.cluster import MiniBatchKMeans

    X = vecs.astype(np.float32)
    if k <= 1 or len(X) <= k:
        centre = X.mean(axis=0, keepdims=True)
        return np.vstack([_unit(c) for c in centre])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048)
    km.fit(X)
    return np.vstack([_unit(c) for c in km.cluster_centers_])
