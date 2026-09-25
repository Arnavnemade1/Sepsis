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
    """Rows ``rows`` (and optionally columns ``cols``) of ``F``, served in batches."""

    def __init__(self, F: np.ndarray, rows: np.ndarray, cols: Optional[np.ndarray] = None,
                 batch_size: int = 16384):
        self.F, self.rows, self.cols = F, np.asarray(rows), cols
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        block = self.F[r]
        if self.cols is not None:
            # LightGBM asks for single rows (an int) while sampling bin boundaries
            block = block[self.cols] if np.ndim(r) == 0 else block[:, self.cols]
        # LightGBM's sequence path samples bin boundaries in float64 only
        return np.asarray(block, dtype=np.float64)


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

    def __getitem__(self, r):
        return np.hstack([np.asarray(b[r], dtype=np.float32) for b in self.blocks])


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
            X = F[r] if self.cols is None else F[r][:, self.cols]
            out[a:a + batch] = self.booster.predict(
                np.asarray(X, dtype=np.float32), num_iteration=self.best_iter,
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
        ds = lgb.Dataset(RowView(F, rows, cols), label=target, weight=w, free_raw_data=True,
                         params={"max_bin": p["max_bin"], "verbose": -1})
        b = lgb.train(p, ds, num_boost_round=fixed_rounds)
        return LGBFit(b, fixed_rounds, cols)
    fit_pos, val_pos = grouped_holdout(seg, valid_frac, seed, strat)
    ds = lgb.Dataset(RowView(F, rows[fit_pos], cols), label=target[fit_pos],
                     weight=None if w is None else w[fit_pos], free_raw_data=True,
                     params={"max_bin": p["max_bin"], "verbose": -1})
    dv = lgb.Dataset(RowView(F, rows[val_pos], cols), label=target[val_pos],
                     weight=None if w is None else w[val_pos], reference=ds, free_raw_data=True)
    cb = [lgb.early_stopping(early_stop, verbose=False)]
    if verbose:
        cb.append(lgb.log_evaluation(100))
    b = lgb.train(p, ds, num_boost_round=rounds, valid_sets=[dv], callbacks=cb)
    fit = LGBFit(b, int(b.best_iteration or rounds), cols)
    vm = fit.predict(F, rows[val_pos], raw=True, calibrated=False)
    fit.calib = platt(vm, target[val_pos], None if w is None else w[val_pos])
    return fit
