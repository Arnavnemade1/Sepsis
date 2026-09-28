"""Boosters over the full-cohort matrix without copying it.

A fold's training set is ~1.24M rows x 510 features = 2.5 GB as float32. Fancy-indexing
that out of the cohort matrix doubles the resident set on a 16 GB laptop, so LightGBM
is fed through :class:`RowView` (an ``lgb.Sequence``): it reads the rows it needs in
batches and bins them into its own compact uint8 store. Peak memory stays near the
cohort matrix plus the binned copy.

Early stopping uses a patient-grouped validation slice carved out of the TRAINING
patients only. The outer test fold never influences the number of trees.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # the booster used throughout this line
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

#: Threads for every fit. The user's laptop is shared; saturating all cores freezes it.
THREADS = int(os.environ.get("GENESIS_THREADS", "6"))

LGB_BASE: Dict[str, object] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
}


class RowView(lgb.Sequence if lgb is not None else object):
    """Rows ``rows`` (and optionally columns ``cols``) of ``F``, served in batches.

    Batches are served C-contiguous: ``F[r][:, cols]`` is not, and LightGBM's row push
    flattens its input, so every 16k-row batch used to be copied a second time (3.8 s of
    a 14 s ingestion at 360k x 800). A float16/float32 source is pushed as float32 —
    LightGBM widens it to double exactly, so the bins are those of a float64 push at
    half the bytes. Single rows (LightGBM's bin sampler) stay float64, which it requires.
    """

    def __init__(self, F: np.ndarray, rows: np.ndarray, cols: Optional[np.ndarray] = None,
                 batch_size: int = 16384):
        self.F, self.rows, self.cols = F, np.asarray(rows), cols
        self.batch_size = batch_size
        dt = np.dtype(getattr(F, "dtype", np.float64))
        self.push_dtype = np.float32 if (dt.kind == "f" and dt.itemsize <= 4) else np.float64

    def __len__(self) -> int:
        return len(self.rows)

    def take(self, r: np.ndarray, dtype=None) -> np.ndarray:
        """Global rows ``r`` (restricted to ``cols``), C-contiguous, as ``dtype`` (default:
        the source's own)."""
        block = self.F[r]
        if self.cols is not None:
            block = np.take(block, self.cols, axis=1)
        return np.ascontiguousarray(block, dtype=dtype)

    def __getitem__(self, idx):
        r = self.rows[idx]
        if np.ndim(r) == 0:                 # LightGBM sampling one row for bin boundaries
            row = self.F[r]
            return np.asarray(row if self.cols is None else row[self.cols], dtype=np.float64)
        return self.take(r, self.push_dtype)


class SampledDataset(lgb.Dataset if lgb is not None else object):
    """``lgb.Dataset`` whose bin-boundary sample is read in ONE vectorised gather.

    LightGBM samples up to ``bin_construct_sample_cnt`` (200,000) rows of a Sequence with
    one ``seq[i]`` call each, collects them in a Python list and stacks it: 200k calls and
    two float64 copies of the sample (~3 GB at 1,000 columns) before a single bin exists.
    Here the SAME indices (LightGBM's own sampler) are gathered at once in the source
    dtype and widened a column at a time, with the SAME filter, so the bin boundaries —
    and the model — are identical (``verify_genesis`` trains both and compares the
    model text). Any other input, or a LightGBM without these internals, takes the
    stock path."""

    def _Dataset__sample(self, seqs, total_nrow):          # overrides Dataset.__sample
        stock = lgb.Dataset._Dataset__sample
        if len(seqs) != 1 or not isinstance(seqs[0], RowView):
            return stock(self, seqs, total_nrow)
        try:
            indices = self._create_sample_indices(total_nrow)
            zero = lgb.basic.ZERO_THRESHOLD
        except AttributeError:                                # pragma: no cover
            return stock(self, seqs, total_nrow)
        seq = seqs[0]
        cols_major = np.ascontiguousarray(seq.take(seq.rows[indices]).T)   # widen contiguous columns
        filtered, filtered_idx = [], []
        sampled_row_range = np.arange(len(indices), dtype=np.int32)
        for j in range(cols_major.shape[0]):
            col = cols_major[j].astype(np.float64)
            keep = (np.abs(col) > zero) | np.isnan(col)
            filtered.append(col[keep])
            filtered_idx.append(sampled_row_range[keep])
        return filtered, filtered_idx


class HStack:
    """Row-indexable horizontal concatenation of matrices, without materialising it.

    Lets a learner read ``[features | stage-1 margins]`` for any row subset while the
    two blocks stay separate in memory (the 3 GB feature map is never duplicated)."""

    def __init__(self, *blocks: np.ndarray):
        n = {b.shape[0] for b in blocks}
        if len(n) != 1:
            raise ValueError("blocks disagree on row count")
        self.blocks = blocks
        self.shape = (blocks[0].shape[0], sum(b.shape[1] for b in blocks))
        self.dtype = np.dtype(np.float32)               # what __getitem__ produces

    def __getitem__(self, r):
        return np.hstack([np.asarray(b[r], dtype=np.float32) for b in self.blocks])


class _SharedDatasets:
    """One constructed (train, valid) LightGBM pair, handed to the next fit that would bin
    EXACTLY the same rows, columns, weights and early-stopping split under the same dataset
    parameters — only the labels differ, and they are set on the handle. Binning is a
    function of the feature values alone, so the reused pair is the pair that fit would
    have built (``verify_genesis`` trains heads both ways and compares the model text).
    A fit is ~18% ingestion, and a tier's direct / who / soft / distill heads all bin the
    same fold. Off unless inside :func:`shared_datasets`; one slot, freed before a miss
    builds, so it never holds two datasets at once."""

    def __init__(self) -> None:
        self.on, self.key, self.pair = False, None, None
        self.hits = self.misses = 0


_SHARED = _SharedDatasets()
#: training parameters that do not touch the binned data
_NOT_DATASET = frozenset({"objective", "metric"})


@contextlib.contextmanager
def shared_datasets():
    _SHARED.on, _SHARED.key, _SHARED.pair = True, None, None
    try:
        yield _SHARED
    finally:
        _SHARED.on, _SHARED.key, _SHARED.pair = False, None, None


def _digest(*arrays) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        if a is None:
            h.update(b"-")
            continue
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def grouped_holdout(seg: np.ndarray, frac: float, seed: int,
                    strat: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Split row positions into (fit, valid) by patient. ``strat`` (per row) keeps the
    septic share equal on both sides."""
    rng = np.random.default_rng(seed)
    pts = np.unique(seg)
    if strat is None:
        chosen = rng.choice(pts, size=max(1, int(round(frac * len(pts)))), replace=False)
    else:
        lab = np.zeros(seg.max() + 1, dtype=np.int8)
        lab[seg] = strat
        chosen = []
        for v in (0, 1):
            grp = pts[lab[pts] == v]
            if len(grp):
                chosen.append(rng.choice(grp, size=max(1, int(round(frac * len(grp)))), replace=False))
        chosen = np.concatenate(chosen)
    is_val = np.zeros(seg.max() + 1, dtype=bool)
    is_val[chosen] = True
    v = is_val[seg]
    return np.flatnonzero(~v), np.flatnonzero(v)


def platt(margin: np.ndarray, target: np.ndarray, weight: Optional[np.ndarray] = None,
          iters: int = 100) -> Tuple[float, float]:
    """1-D logistic recalibration ``sigmoid(a * margin + b)``; returns ``(a, b)``.

    Accepts soft targets in [0, 1] and row weights (sklearn's LogisticRegression takes
    neither soft labels nor an unregularised fit, and this is two parameters).

    Damped Newton with a backtracking line search on the negative log-likelihood, so
    every accepted step DECREASES the loss. The first version took raw Newton steps and
    diverged to a = -1.3e9 on boosters with ~500 trees, whose margins are extreme enough
    to make the problem nearly separable — flipping two folds of a learning curve
    upside down. The map is only ever allowed to be increasing: a result that is not a
    finite, positive slope falls back to the identity, because a calibrator that can
    reverse a ranking must never be trusted with one."""
    m = np.clip(np.asarray(margin, dtype=np.float64), -30.0, 30.0)
    t = np.asarray(target, dtype=np.float64)
    w = np.ones_like(m) if weight is None else np.asarray(weight, dtype=np.float64)
    sw = float(w.sum())
    if sw <= 0 or len(m) < 10 or float(np.ptp(m)) < 1e-9:
        return 1.0, 0.0

    def nll(a, b):
        z = a * m + b
        return float((w * (np.logaddexp(0.0, z) - t * z)).sum()) / sw

    a, b = 1.0, 0.0
    f = nll(a, b)
    for _ in range(iters):
        z = a * m + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        g = w * (p - t)
        h = w * p * (1 - p) + 1e-12
        ga, gb = float((g * m).sum()) / sw, float(g.sum()) / sw
        haa, hab, hbb = (float((h * m * m).sum()) / sw, float((h * m).sum()) / sw,
                         float(h.sum()) / sw)
        det = haa * hbb - hab * hab
        if det > 1e-12:
            da = (hbb * ga - hab * gb) / det
            db = (haa * gb - hab * ga) / det
        else:                                    # degenerate curvature: gradient step
            da, db = ga, gb
        step, improved = 1.0, False
        while step > 1e-6:
            na, nb = a - step * da, b - step * db
            nf = nll(na, nb)
            if np.isfinite(nf) and nf <= f - 1e-12:
                improved = True
                break
            step *= 0.5
        if not improved:
            break
        a, b, f = na, nb, nf
        if abs(step * da) < 1e-9 and abs(step * db) < 1e-9:
            break
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 1e-6:
        return 1.0, 0.0
    return float(a), float(b)


@dataclass
class LGBFit:
    booster: object
    best_iter: int
    cols: Optional[np.ndarray] = None
    #: Platt map fitted on the patient-held-out early-stopping slice. Every fold model
    #: stops at a different tree count, so raw margins are NOT on a common scale across
    #: folds (fold means of one head ranged -4.66..-5.23); a stage-2 learner trained on
    #: four folds' margins would misread the fifth. Calibrated margins share one scale.
    calib: Tuple[float, float] = (1.0, 0.0)

    def predict(self, F: np.ndarray, rows: np.ndarray, raw: bool = False,
                batch: int = 200_000, calibrated: bool = True) -> np.ndarray:
        """Probabilities, or margins with ``raw=True``. Both pass through the Platt map
        unless ``calibrated=False``."""
        out = np.empty(len(rows))
        for a in range(0, len(rows), batch):
            r = rows[a:a + batch]
            X = F[r] if self.cols is None else np.take(F[r], self.cols, axis=1)
            # C-contiguous, or LightGBM's predict flattens it with a copy of its own
            out[a:a + batch] = self.booster.predict(
                np.ascontiguousarray(X, dtype=np.float32), num_iteration=self.best_iter,
                raw_score=True, num_threads=THREADS)
        if calibrated:
            out = self.calib[0] * out + self.calib[1]
        return out if raw else 1.0 / (1.0 + np.exp(-np.clip(out, -35, 35)))


def fit_lgb(F: np.ndarray, rows: np.ndarray, target: np.ndarray, seg: np.ndarray,
            weight: Optional[np.ndarray] = None, params: Optional[Dict] = None,
            rounds: int = 3000, valid_frac: float = 0.1, early_stop: int = 150,
            cols: Optional[np.ndarray] = None, seed: int = 0, strat: Optional[np.ndarray] = None,
            fixed_rounds: Optional[int] = None, verbose: bool = False,
            feval_metric: str = "auc") -> LGBFit:
    """Fit on ``rows`` of ``F`` to ``target`` (aligned to ``rows``; binary or soft in
    [0, 1] for the cross-entropy objective). ``seg`` gives each row's patient so the
    early-stopping slice is patient-grouped."""
    p = dict(LGB_BASE)
    p.update(params or {})
    p["num_threads"] = THREADS
    p["seed"] = seed
    p["metric"] = feval_metric
    target = np.asarray(target, dtype=np.float64)
    w = None if weight is None else np.asarray(weight, dtype=np.float64)
    if fixed_rounds is not None:
        ds = SampledDataset(RowView(F, rows, cols), label=target, weight=w, free_raw_data=True,
                            params={"max_bin": p["max_bin"], "verbose": -1})
        b = lgb.train(p, ds, num_boost_round=fixed_rounds)
        return LGBFit(b, fixed_rounds, cols)
    fit_pos, val_pos = grouped_holdout(seg, valid_frac, seed, strat)
    key = None
    if _SHARED.on:
        dparams = tuple(sorted((k, repr(v)) for k, v in p.items() if k not in _NOT_DATASET))
        key = (id(F), getattr(F, "shape", None), dparams,
               _digest(np.asarray(rows), None if cols is None else np.asarray(cols), w, fit_pos, val_pos))
    if key is not None and _SHARED.key == key:
        ds, dv = _SHARED.pair
        ds.set_label(target[fit_pos])
        dv.set_label(target[val_pos])
        _SHARED.hits += 1
    else:
        if key is not None:
            _SHARED.key, _SHARED.pair = None, None       # free the old pair before building
            _SHARED.misses += 1
        ds = SampledDataset(RowView(F, rows[fit_pos], cols), label=target[fit_pos],
                            weight=None if w is None else w[fit_pos], free_raw_data=True,
                            params={"max_bin": p["max_bin"], "verbose": -1})
        dv = lgb.Dataset(RowView(F, rows[val_pos], cols), label=target[val_pos],
                         weight=None if w is None else w[val_pos], reference=ds, free_raw_data=True)
    cb = [lgb.early_stopping(early_stop, verbose=False)]
    if verbose:
        cb.append(lgb.log_evaluation(100))
    b = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[dv], callbacks=cb)
    if key is not None:
        _SHARED.key, _SHARED.pair = key, (ds, dv)
    fit = LGBFit(b, int(b.best_iteration or rounds), cols)
    vm = fit.predict(F, rows[val_pos], raw=True, calibrated=False)
    fit.calib = platt(vm, target[val_pos], None if w is None else w[val_pos])
    return fit
