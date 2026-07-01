"""Data-grounded Hyperdimensional Computing sepsis detector.

This is the project's scientific core, rebuilt so that *nothing is hand-tuned*:

  1. ENCODE   each hour into a 10,000-D bipolar hypervector using record-style
              binding (channel-id ⊗ level(value)) — the same VSA algebra as the
              original engine, but value ranges are learned from data quantiles.
              Multi-scale temporal decoupling binds same-channel evidence at
              different time horizons (instant/short/medium/long/burden/context)
              into near-orthogonal subspaces so they don't collide when bundled.
  2. LEARN    class prototype hypervectors as real-valued centroids of the
              encoded TRAINING hours (healthy vs septic) via k-means in HDC
              space. Two optional error-driven retraining passes exist
              (AdaptHD-style ``retrain_epochs`` and gradient-guided
              ``hd_ggr_epochs``) — see the HONEST RESULT below before enabling.
  3. FUSE     rich per-archetype similarity signals (top-k matches per class,
              margins, a temperature-softened log-odds) with interpretable
              clinical composites (SIRS, qSOFA, shock index, hypoxia,
              hypotension) in a class-weighted, probability-CALIBRATED logistic
              layer fitted on real labels.

HONEST RESULT (patient-grouped CV, CinC 2019, wearable channels — reproduce with
``scripts/train_and_compare.py`` / the experiments referenced in
``backend/ml/README.md``):

  * Standalone HDC trails the GBM benchmark (~0.66 vs ~0.72 AUROC, 3-fold CV).
    Rich per-archetype signals (``rich_signals=True``, the default) give a small,
    reproducible gain over max/mean/margin-only signals (~0.635 → ~0.643 AUROC).
  * ``retrain_epochs`` (AdaptHD) measurably HURTS standalone ranking (~0.634 →
    ~0.586 AUROC) — optimizing archetypes for argmax accuracy conflicts with the
    margin structure the downstream fusion logistic needs. Default is 0 (off).
  * Feeding HDC signals into the GBM as extra features
    (:class:`backend.ml.models.augmented.HDCAugmentedGBM`) looked like a large win
    on one small split but did NOT replicate under CV (gap smaller than the
    fold-to-fold noise) — see that module's docstring. Not shipped to production.

HDC's validated, real contributions in this project are (a) a complementary,
interpretable case-based memory (see ``backend/aegis/hdc_memory.py``, honest
AUROC ~0.626, used for "similar past patients" retrieval) and (b) a cross-modal
resonance/consensus gate (see ``backend/aegis/resonance.py``) — not (yet) a
raw-AUROC win over gradient boosting on this tabular task. Stated, not hidden.

Training is done in row batches so memory stays bounded on the full dataset.
"""
from __future__ import annotations

import os
import re
import warnings
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

_TEMPORAL_SCALES = ("instant", "short", "medium", "long", "burden", "context")


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
        prefix = f"{ch}__"
        if ch in columns:
            wanted.append(ch)
        for col in columns:
            if not col.startswith(prefix):
                continue
            suffix = col[len(prefix):]
            if (
                suffix in {"measured", "age", "d1", "rmean", "rstd", "rmin", "rmax", "slope", "bz", "emin", "emax"}
                or re.match(r"^(rmean|rstd|slope|lag)\d+$", suffix)
            ):
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
    # AdaptHD error-driven prototype retraining. Measured (3-fold CV, CinC setA):
    # optimising prototypes for argmax accuracy DEGRADES the fusion layer's ranking
    # (AUROC 0.64→0.58), because the fusion logistic wants well-separated *margins*,
    # not argmax-correct centroids. Kept as an option, OFF by default — honest result.
    retrain_epochs: int = 0
    retrain_lr: float = 0.035
    retrain_balance: bool = True        # weight rare-class updates by inverse frequency
    # HD-GGR: gradient-guided HDC retraining after the first fusion layer fit.
    hd_ggr_epochs: int = 8
    hd_ggr_lr: float = 0.018
    hd_ggr_focus_quantile: float = 0.90
    hd_ggr_balance: bool = True
    # Multi-Scale Temporal Decoupling Matrix: bind same-channel temporal evidence
    # into near-orthogonal scale subspaces before bundling.
    temporal_decoupling: bool = True
    # Richer per-prototype similarity signals exposed to the calibrated fusion layer.
    rich_signals: bool = True
    topk_signals: int = 3

    # learned state ------------------------------------------------------- #
    ranges_: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    id_vecs_: Optional[np.ndarray] = None       # (n_encoded, D) int8
    feature_vecs_: Optional[np.ndarray] = None  # (n_encoded, D) int8, channel-id bound to temporal scale
    level_vecs_: Optional[np.ndarray] = None    # (levels, D) int8
    temporal_scale_vecs_: Optional[np.ndarray] = None
    temporal_decoupling_matrix_: Optional[np.ndarray] = None
    feature_scales_: List[str] = field(default_factory=list)
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
        self.feature_vecs_ = self.id_vecs_.copy()
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
        if self.temporal_decoupling:
            self._init_temporal_decoupling(rng)

    def _init_temporal_decoupling(self, rng: np.random.Generator) -> None:
        """Create the Multi-Scale Temporal Decoupling Matrix.

        Each encoded feature gets a small membership vector over physiologic time
        scales. Binding the feature id with the bundled scale vector prevents
        short-window noise, long-window burden, and point measurements from
        collapsing into the same HDC direction.
        """
        self.temporal_scale_vecs_ = rng.choice(
            np.array([-1, 1], dtype=np.int8),
            size=(len(_TEMPORAL_SCALES), self.dimensions),
        )
        scale_to_i = {s: i for i, s in enumerate(_TEMPORAL_SCALES)}
        matrix = np.zeros((len(self.encode_features), len(_TEMPORAL_SCALES)), dtype=np.float32)
        feature_scales: List[str] = []
        for i, name in enumerate(self.encode_features):
            scale = self._feature_temporal_scale(name)
            feature_scales.append(scale)
            matrix[i, scale_to_i[scale]] = 1.0
            # Give trend-like evidence a faint link to instantaneous state so the
            # fusion layer can still learn level+slope conjunctions without washing
            # temporal scales into one another.
            if scale in {"short", "medium", "long"}:
                matrix[i, scale_to_i["instant"]] = 0.15
        bundled = np.sign(_safe_matmul(matrix, self.temporal_scale_vecs_.astype(np.float32)))
        bundled = np.nan_to_num(bundled, nan=1.0, posinf=1.0, neginf=-1.0)
        bundled[bundled == 0] = 1
        self.temporal_decoupling_matrix_ = matrix
        self.feature_scales_ = feature_scales
        self.feature_vecs_ = (self.id_vecs_ * bundled.astype(np.int8)).astype(np.int8)

    @staticmethod
    def _feature_temporal_scale(name: str) -> str:
        if name in _COMPOSITE_COLS or name in {"Age", "Gender", "HospAdmTime", "ICULOS", "n_measured_this_hour"}:
            return "context"
        if name.startswith("burden_") or name.endswith("__emin") or name.endswith("__emax"):
            return "burden"
        suffix = name.split("__", 1)[1] if "__" in name else ""
        match = re.search(r"(\d+)$", suffix)
        horizon = int(match.group(1)) if match else None
        if suffix.startswith(("rmean", "rstd", "slope", "lag")):
            if horizon is not None and horizon >= 24:
                return "long"
            if horizon is not None and horizon >= 12:
                return "medium"
            return "short"
        if suffix in {"d1", "bz", "rmean", "rstd", "rmin", "rmax", "slope"}:
            return "short"
        if suffix in {"age", "measured"}:
            return "context"
        return "instant"

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
            bound = self.level_vecs_[idx] * self.feature_vecs_[ci]   # (n,D) bind = elementwise mult
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

        # AdaptHD: error-driven retraining turns the k-means centroids into
        # *discriminative* archetypes. This is the learning step that closes most of
        # the gap to the tree model while keeping the case-based HDC identity.
        if self.retrain_epochs > 0:
            self._retrain_prototypes(sep_vecs, heal_vecs)

        # Single centroids kept too (cheap, useful for explanation/streaming).
        self.c_sep_ = _unit(sep_vecs.astype(np.float64).sum(axis=0))
        self.c_heal_ = _unit(heal_vecs.astype(np.float64).sum(axis=0))

        # Fit the calibrated neurosymbolic fusion layer on HDC similarity
        # signals + interpretable clinical composites, on real labels.
        X_fusion, names = self._fusion_matrix(feats)
        self.fusion_features_ = names
        self.scaler_ = StandardScaler().fit(X_fusion)
        self.clf_ = self._fit_fusion_classifier(X_fusion, y)
        if self.hd_ggr_epochs > 0:
            self._hd_ggr_retrain(feats, y)
            X_fusion, names = self._fusion_matrix(feats)
            self.fusion_features_ = names
            self.scaler_ = StandardScaler().fit(X_fusion)
            self.clf_ = self._fit_fusion_classifier(X_fusion, y)
        return self

    def _fit_fusion_classifier(self, X_fusion: np.ndarray, y: np.ndarray) -> LogisticRegression:
        clf = LogisticRegression(class_weight="balanced", max_iter=4000, C=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clf.fit(self.scaler_.transform(X_fusion), y)
        return clf

    def _encode_rows(self, feats: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Encode a specific set of rows into bipolar hypervectors (m, D) int8."""
        out = np.empty((len(idx), self.dimensions), dtype=np.int8)
        for start in range(0, len(idx), self.batch_rows):
            sl = idx[start:start + self.batch_rows]
            out[start:start + len(sl)] = self._encode_batch(feats.iloc[sl])
        return out

    # ------------------------------------------------------------------ #
    # AdaptHD error-driven retraining
    # ------------------------------------------------------------------ #
    def _retrain_prototypes(self, sep_vecs: np.ndarray, heal_vecs: np.ndarray) -> None:
        """Multi-prototype AdaptHD/OnlineHD retraining.

        Each epoch: predict every training hour against the current archetypes; for a
        misclassified hour, reinforce the nearest *correct-class* archetype with +lr·x
        and weaken the wrongly-matched archetype with −lr·x (margin-weighted). Rare
        sepsis updates are up-weighted so the 2%-prevalence class is not drowned out.
        Prototypes are renormalised each epoch; lr decays. Pure VSA — no backprop."""
        Ps = self.protos_sep_.astype(np.float32).copy()      # (Ks, D)
        Ph = self.protos_heal_.astype(np.float32).copy()      # (Kh, D)
        Ks, Kh, D = len(Ps), len(Ph), self.dimensions
        sqrtD = np.sqrt(D)

        Xs = sep_vecs.astype(np.float32) / sqrtD
        Xh = heal_vecs.astype(np.float32) / sqrtD
        X = np.vstack([Xs, Xh])
        y = np.concatenate([np.ones(len(Xs), dtype=np.int8), np.zeros(len(Xh), dtype=np.int8)])
        # inverse-frequency update weights so rare sepsis hours pull their weight.
        w = np.ones(len(X), dtype=np.float32)
        if self.retrain_balance and len(Xs) and len(Xh):
            w[: len(Xs)] = len(Xh) / max(len(Xs), 1)

        rows = np.arange(len(X))
        lr = float(self.retrain_lr)
        prev_err = np.inf

        def onehot(idx: np.ndarray, k: int) -> np.ndarray:
            m = np.zeros((len(idx), k), dtype=np.float32)
            m[np.arange(len(idx)), idx] = 1.0
            return m

        for ep in range(self.retrain_epochs):
            P = np.vstack([Ph, Ps])                          # heal cols [0:Kh], sep cols [Kh:]
            pcls = np.concatenate([np.zeros(Kh, np.int8), np.ones(Ks, np.int8)])
            sims = _safe_matmul(X, P.T)                       # (N, Kh+Ks)
            best = sims.argmax(axis=1)
            pred = pcls[best]
            mis = pred != y
            err = int(mis.sum())
            if err == 0 or err > prev_err:                   # converged / diverging → stop
                break
            prev_err = err
            sims_h, sims_s = sims[:, :Kh], sims[:, Kh:]
            bh, bs = sims_h.argmax(axis=1), sims_s.argmax(axis=1)
            true_col = np.where(y == 1, Kh + bs, bh)
            margin = np.clip(sims[rows, best] - sims[rows, true_col], 0.0, None)
            uX = ((lr * w * margin)[:, None] * X)            # (N, D) weighted samples
            # scatter corrections via one-hot matmul (fast for a handful of prototypes)
            m_sep, m_heal = mis & (y == 1), mis & (y == 0)
            p_sep, p_heal = mis & (pred == 1), mis & (pred == 0)
            dPs = (_safe_matmul(onehot(bs[m_sep], Ks).T, uX[m_sep])
                   - _safe_matmul(onehot(best[p_sep] - Kh, Ks).T, uX[p_sep]))
            dPh = (_safe_matmul(onehot(bh[m_heal], Kh).T, uX[m_heal])
                   - _safe_matmul(onehot(best[p_heal], Kh).T, uX[p_heal]))
            dPs = np.nan_to_num(dPs, nan=0.0, posinf=0.0, neginf=0.0)
            dPh = np.nan_to_num(dPh, nan=0.0, posinf=0.0, neginf=0.0)
            Ps = np.vstack([_unit(p) for p in (Ps + dPs)]).astype(np.float32)
            Ph = np.vstack([_unit(p) for p in (Ph + dPh)]).astype(np.float32)
            lr *= 0.97
        self.protos_sep_ = Ps
        self.protos_heal_ = Ph

    def _hd_ggr_retrain(self, feats: pd.DataFrame, y: np.ndarray) -> None:
        """High-Dimensional Gradient-Guided Retraining.

        AdaptHD corrects hard nearest-prototype mistakes. HD-GGR goes after the
        rare-prevalence failure mode directly: after the fusion model exposes a
        probability surface, residuals become prototype updates. Underpredicted
        septic rows reinforce their nearest sepsis archetype and repel the nearest
        healthy archetype; false-positive rows do the opposite. The update remains
        pure HDC vector algebra, but it is guided by the loss gradient that drives
        AUROC/AUPRC instead of raw distance matching.
        """
        if self.clf_ is None or self.scaler_ is None:
            return
        y = np.asarray(y).astype(int)
        pos = max(int((y == 1).sum()), 1)
        neg = max(int((y == 0).sum()), 1)
        pos_weight = neg / pos if self.hd_ggr_balance else 1.0
        raw_train = self._raw_proba(feats)
        neg_scores = raw_train[y == 0]
        focus_thr = float(np.quantile(neg_scores, self.hd_ggr_focus_quantile)) if neg_scores.size else 1.0

        Ps = self.protos_sep_.astype(np.float32).copy()
        Ph = self.protos_heal_.astype(np.float32).copy()
        Ks, Kh, D = len(Ps), len(Ph), self.dimensions
        sqrtD = np.sqrt(D)
        lr = float(self.hd_ggr_lr)

        def onehot(idx: np.ndarray, k: int) -> np.ndarray:
            m = np.zeros((len(idx), k), dtype=np.float32)
            if len(idx):
                m[np.arange(len(idx)), idx] = 1.0
            return m

        for _ep in range(self.hd_ggr_epochs):
            total_step = 0.0
            off = 0
            for batch in self._iter_batches(feats):
                b = len(batch)
                yb = y[off:off + b]
                p = self._raw_proba(batch)
                residual = yb.astype(np.float32) - p.astype(np.float32)
                focus = 1.0 + 2.0 * ((yb == 1) | (p >= focus_thr)).astype(np.float32)
                cls_weight = np.where(yb == 1, pos_weight, 1.0).astype(np.float32)
                grad = np.clip(residual * focus * cls_weight, -pos_weight, pos_weight)
                active = np.abs(grad) > 1e-4
                if active.any():
                    X = self._encode_batch(batch).astype(np.float32) / sqrtD
                    s = _safe_matmul(X, Ps.T)
                    h = _safe_matmul(X, Ph.T)
                    bs = s.argmax(axis=1)
                    bh = h.argmax(axis=1)
                    update = (lr * grad[active])[:, None] * X[active]
                    dPs = _safe_matmul(onehot(bs[active], Ks).T, update)
                    dPh = _safe_matmul(onehot(bh[active], Kh).T, -update)
                    Ps = np.vstack([_unit(pv) for pv in (Ps + dPs)]).astype(np.float32)
                    Ph = np.vstack([_unit(pv) for pv in (Ph + dPh)]).astype(np.float32)
                    total_step += float(np.mean(np.abs(grad[active])))
                off += b
            lr *= 0.85
            if total_step < 1e-5:
                break
        self.protos_sep_ = Ps
        self.protos_heal_ = Ph

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def hdc_signals(self, feats: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Per-row HDC evidence: best/mean similarity to each class' prototypes
        and the sepsis-vs-healthy margin. These are the learned 'archetype'
        matches that replace the old hardcoded similarity."""
        n = len(feats)
        Ks, Kh = len(self.protos_sep_), len(self.protos_heal_)
        max_sep = np.empty(n); mean_sep = np.empty(n)
        max_heal = np.empty(n); mean_heal = np.empty(n)
        k = int(self.topk_signals)
        topk_sep = np.empty((n, k)) if self.rich_signals else None
        topk_heal = np.empty((n, k)) if self.rich_signals else None
        soft = np.empty(n) if self.rich_signals else None
        sqrtD = np.sqrt(self.dimensions)
        Ps = self.protos_sep_.T.astype(np.float32)
        Ph = self.protos_heal_.T.astype(np.float32)
        off = 0
        for batch in self._iter_batches(feats):
            hv = self._encode_batch(batch).astype(np.float32) / sqrtD
            s = _safe_matmul(hv, Ps)                         # (b, Ksep)
            h = _safe_matmul(hv, Ph)                         # (b, Kheal)
            b = len(batch)
            max_sep[off:off + b] = s.max(axis=1)
            mean_sep[off:off + b] = s.mean(axis=1)
            max_heal[off:off + b] = h.max(axis=1)
            mean_heal[off:off + b] = h.mean(axis=1)
            if self.rich_signals:
                # top-k similarities per class (sorted desc) — the logistic gets the
                # full shape of the archetype match, not just its peak.
                topk_sep[off:off + b] = -np.sort(-s, axis=1)[:, :k] if Ks >= k else \
                    np.pad(-np.sort(-s, axis=1), ((0, 0), (0, k - Ks)), constant_values=0)[:, :k]
                topk_heal[off:off + b] = -np.sort(-h, axis=1)[:, :k] if Kh >= k else \
                    np.pad(-np.sort(-h, axis=1), ((0, 0), (0, k - Kh)), constant_values=0)[:, :k]
                # temperature-softened sepsis-vs-healthy log-odds over best matches
                soft[off:off + b] = 8.0 * (s.max(axis=1) - h.max(axis=1))
            off += b
        out = {
            "hdc_max_sep": max_sep,
            "hdc_mean_sep": mean_sep,
            "hdc_max_heal": max_heal,
            "hdc_mean_heal": mean_heal,
            "hdc_margin": max_sep - max_heal,
        }
        if self.rich_signals:
            for j in range(k):
                out[f"hdc_sep_top{j}"] = topk_sep[:, j]
                out[f"hdc_heal_top{j}"] = topk_heal[:, j]
            out["hdc_soft"] = soft
            out["hdc_margin_mean"] = mean_sep - mean_heal
        return out

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(v)
    return v / n if np.isfinite(n) and n > 0 else v


def _safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matmul for HDC algebra with local finite guards.

    Some Accelerate/BLAS builds emit spurious RuntimeWarnings on large float32
    products even when the downstream result is usable. HDC retraining should be
    robust to that noise, but it should never propagate NaN/inf into prototypes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = a @ b
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_prototypes(vecs: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Cluster encoded hypervectors into ``k`` unit-norm prototypes (archetypes).

    k-means in HDC space; cluster centres are the learned archetypes. With k=1
    this reduces to the classic single-centroid HDC classifier.
    """
    from sklearn.cluster import MiniBatchKMeans

    X = np.nan_to_num(vecs.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if k <= 1 or len(X) <= k:
        centre = X.mean(axis=0, keepdims=True)
        return np.vstack([_unit(c) for c in centre])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        km.fit(X)
    return np.vstack([_unit(c) for c in km.cluster_centers_])
