"""GENESIS stacking: label-genesis heads, then a temporal meta-learner.

Stage 1 — HEADS. One booster per target in :mod:`backend.ml.genesis.targets`, each
trained on causal features. A head's output is a margin (logit) per hour.

Stage 2 — TEMPORAL META-LEARNER. For every head, causal summaries of its margin over
the stay so far (now, running max, running mean, 6-hour mean, 6-hour change, deviation
from the patient's own running mean) plus the clock. A small booster learns the
challenge label from those. The summaries are what let the second stage answer the
question the decomposition says is weakest — *is this septic patient near onset yet?*
— because "risk is higher than this patient's own history" is a within-patient
statement that no single-hour first stage can make.

Cross-fitting. Every stage-1 prediction a stage-2 learner trains on is out-of-fold:
for dev fold ``k`` the head is fitted on the other dev folds and predicts fold ``k``.
Stage-2 comparisons on the dev folds share those OOF vectors across folds (standard
stacking; any optimism is common to every variant compared). The headline number never
comes from here — it comes from the locked set, where stage 1 is refitted on all dev
patients and stage 2 on the dev OOF, then both are applied once.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backend.ml.genesis import learners as L
from backend.ml.genesis.harness import RESULTS_DIR, Cohort
from backend.ml.genesis.targets import Target

HEAD_DIR = os.path.join(RESULTS_DIR, "heads")


@dataclass
class Head:
    name: str
    target: Callable[[Cohort], Target]
    params: Dict[str, object] = field(default_factory=dict)
    #: Keep this fraction of target-0 rows (weight-corrected by 1/rate). Speeds up
    #: heads whose question is dominated by easy negatives.
    neg_rate: float = 1.0
    cols: Optional[np.ndarray] = None
    seed: int = 0
    #: Matrix the head reads instead of the cohort's features (e.g. an ``HStack`` of the
    #: features and an extra block). Must be row-aligned with the cohort.
    data: Optional[object] = None


def _train_rows(c: Cohort, head: Head, rows: np.ndarray, tgt: np.ndarray,
                w: np.ndarray, rng: np.random.Generator):
    use = rows[w[rows] > 0]
    ww = w[use].copy()
    if head.neg_rate < 1.0:
        neg = tgt[use] == 0
        keep = ~neg | (rng.random(len(use)) < head.neg_rate)
        ww = np.where(neg, ww / head.neg_rate, ww)
        use, ww = use[keep], ww[keep]
    return use, ww


def fit_head(c: Cohort, head: Head, rows: np.ndarray, tgt: np.ndarray, w: np.ndarray,
             seed: int = 0) -> L.LGBFit:
    rng = np.random.default_rng(seed + head.seed)
    use, ww = _train_rows(c, head, rows, tgt, w, rng)
    soft = np.any((tgt[use] > 0) & (tgt[use] < 1))
    params = dict(head.params)
    if soft:
        params.setdefault("objective", "cross_entropy")
    return L.fit_lgb(c.F if head.data is None else head.data, use, tgt[use], c.seg[use],
                     weight=None if np.all(ww == 1) else ww, params=params,
                     cols=head.cols, seed=seed + head.seed, strat=c.plabel[c.seg[use]])


def head_oof(c: Cohort, head: Head, refresh: bool = False, verbose: bool = True) -> np.ndarray:
    """Out-of-fold margins of ``head`` on every dev row (NaN elsewhere). Cached."""
    os.makedirs(HEAD_DIR, exist_ok=True)
    path = os.path.join(HEAD_DIR, f"{head.name}_oof.npy")
    if os.path.exists(path) and not refresh:
        return np.load(path).astype(np.float64)
    tgt, w = head.target(c)
    oof = np.full(len(c.y), np.nan)
    meta = {"name": head.name, "params": head.params, "neg_rate": head.neg_rate, "folds": []}
    for k in range(c.n_folds):
        t0 = time.time()
        tr, te = c.fold_rows(k)
        fit = fit_head(c, head, tr, tgt, w, seed=k)
        oof[te] = fit.predict(c.F if head.data is None else head.data, te, raw=True)
        meta["folds"].append({"k": k, "trees": fit.best_iter, "sec": round(time.time() - t0, 1)})
        if verbose:
            print(f"    head {head.name} fold {k}: trees {fit.best_iter} [{time.time()-t0:.0f}s]",
                  flush=True)
    np.save(path, oof.astype(np.float32))
    with open(os.path.join(HEAD_DIR, f"{head.name}.json"), "w") as fh:
        json.dump(meta, fh, indent=1, default=str)
    return oof


# --------------------------------------------------------------------------- #
# Causal temporal summaries of a per-row score
# --------------------------------------------------------------------------- #
def temporal_summaries(c: Cohort, z: np.ndarray, prefix: str,
                       windows: Sequence[int] = (6,)) -> Dict[str, np.ndarray]:
    """Summaries of ``z`` over each row's stay so far (strictly causal)."""
    idx = np.arange(len(z), dtype=np.int64)
    rs = c.raw.row_start
    zz = np.nan_to_num(z, nan=0.0)
    # stay-local running sum (see features._Seg.prefix for why not one cohort-wide cumsum)
    P = pd.Series(zz).groupby(c.seg, sort=False).cumsum().to_numpy()
    n_all = idx - rs + 1
    out: Dict[str, np.ndarray] = {prefix: z}
    out[f"{prefix}__cmax"] = pd.Series(zz).groupby(c.seg, sort=False).cummax().to_numpy()
    run_mean = P / n_all
    out[f"{prefix}__cmean"] = run_mean
    out[f"{prefix}__dev"] = zz - run_mean
    for w in windows:
        lo = np.maximum(idx - w + 1, rs)
        before = np.where(lo > rs, P[np.maximum(lo - 1, 0)], 0.0)
        out[f"{prefix}__mean{w}"] = (P - before) / (idx + 1 - lo)
        src = idx - w
        lag = np.where(src >= rs, zz[np.maximum(src, 0)], np.nan)
        out[f"{prefix}__d{w}"] = zz - lag
    return out


def meta_matrix(c: Cohort, margins: Dict[str, np.ndarray], context: Sequence[str] = ("hour", "ICULOS"),
                temporal: bool = True) -> (np.ndarray, List[str]):
    cols: Dict[str, np.ndarray] = {}
    for name, z in margins.items():
        if temporal:
            cols.update(temporal_summaries(c, z, name))
        else:
            cols[name] = z
    for ctx in context:
        cols[ctx] = np.asarray(c.col(ctx), dtype=np.float64)
    names = list(cols)
    M = np.empty((len(c.y), len(names)), dtype=np.float32)
    for j, n in enumerate(names):
        M[:, j] = cols[n]
    return M, names


META_PARAMS: Dict[str, object] = {
    "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 1000,
    "feature_fraction": 0.8, "bagging_fraction": 0.7, "lambda_l2": 10.0,
}


def meta_candidate(M: np.ndarray, params: Optional[Dict] = None, with_features: bool = False):
    """Stage-2 candidate over a precomputed meta matrix (rows aligned to the cohort).

    ``with_features`` also hands stage 2 the full causal feature matrix, so it can
    condition the heads on raw state (read through :class:`learners.HStack`, never
    copied)."""
    p = dict(META_PARAMS)
    p.update(params or {})

    def cand(c: Cohort, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        data = L.HStack(c.F, M) if with_features else M
        fit = L.fit_lgb(data, tr, c.y[tr], c.seg[tr], params=p, strat=c.plabel[c.seg[tr]])
        return fit.predict(data, te)
    return cand


def lr_meta_candidate(M: np.ndarray, C: float = 1.0, neg_rate: float = 0.25, seed: int = 0):
    """Stage 2 as a logistic regression over (standardised) stage-1 summaries: a handful
    of weights, so it cannot overfit the fold structure the way a booster can."""
    from sklearn.linear_model import LogisticRegression

    def cand(c: Cohort, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(seed)
        y = c.y[tr]
        keep = (y == 1) | (rng.random(len(tr)) < neg_rate)
        rows = tr[keep]
        X = np.nan_to_num(np.asarray(M[rows], dtype=np.float64))
        mu, sd = X.mean(0), X.std(0) + 1e-9
        w = np.where(c.y[rows] == 1, 1.0, 1.0 / neg_rate)
        lr = LogisticRegression(C=C, max_iter=500)
        lr.fit((X - mu) / sd, c.y[rows], sample_weight=w)
        Xt = np.nan_to_num(np.asarray(M[te], dtype=np.float64))
        return lr.decision_function((Xt - mu) / sd)
    return cand


def fixed_candidate(score: np.ndarray):
    """A 'candidate' that just reads a precomputed OOF vector (for parameter-free
    combinations like the who x when product)."""
    def cand(c: Cohort, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        return score[te]
    return cand


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)
