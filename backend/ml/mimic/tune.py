"""A budgeted hyperparameter search for the boosters — on development subjects only.

On CinC 2019 tuning was the largest single gain of the whole line: the same 40 variables,
default LightGBM 0.805 -> tuned 0.862 on the locked test set (docs/GENESIS.md §4). MIMIC's
features differ (more channels, sparser labs, the label-structure block), so the full run
re-tunes rather than assuming CinC's optimum transfers.

The search is a random sample of a compact space, always including the incumbent (the
scale's configured parameters) as trial 0, so it can only move away from it when something
measurably better exists. Each trial trains on dev subjects outside validation fold v and
scores the design-weighted AUROC on fold v's early-warning rows; the lock is never read.
Selecting among ~20 configurations on dev data makes the dev estimate of the winner a
little optimistic — the locked set, spent once through ``trust``, is what reports it.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from backend.ml.genesis import learners as L
from backend.ml.genesis.harness import Ranked

SPACE: Dict[str, Sequence] = {
    "num_leaves": (15, 31, 63, 127),
    "min_data_in_leaf": (50, 200, 800, 2000),
    "feature_fraction": (0.15, 0.3, 0.5, 0.7),
    "bagging_fraction": (0.6, 0.8, 1.0),
    "lambda_l2": (0.0, 1.0, 10.0, 50.0),
    "learning_rate": (0.03, 0.05),
}


def sample(rng, n: int, base: Dict) -> List[Dict]:
    out, seen = [dict(base)], {tuple(sorted(base.items()))}
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        cfg = dict(base) | {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def tune(led, tier: str = "P", n_trials: int = 16, val_folds: Sequence[int] = (0,),
         max_train_rows: int = 600_000, early_stop: int = 100, base: Optional[Dict] = None,
         seed: int = 0, verbose: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    s = led.store
    f = led.folds[s.seg]
    septic = np.isfinite(led.onset[s.seg])
    cols = led.columns(tier)
    led.assert_tier(cols, tier)
    y = led.y.astype(float)
    splits = []
    for v in val_folds:
        tr_stays = np.flatnonzero((led.folds >= 0) & (led.folds != v))
        rng.shuffle(tr_stays)
        keep, total = [], 0
        for k in tr_stays:                                  # whole stays, up to the row budget
            if total >= max_train_rows:
                break
            keep.append(k)
            total += int(s.lens[k])
        on = np.zeros(s.n_patients, dtype=bool)
        on[keep] = True
        tr = np.flatnonzero(on[s.seg] & led.train_mask)
        te = np.flatnonzero((f == v) & led.eval_mask)
        if len(tr) and len(te) and 0 < y[te].sum() < len(te):
            splits.append((tr, te))
    trials = []
    for i, cfg in enumerate(sample(rng, n_trials, base or {})):
        t0 = time.time()
        aucs, iters = [], []
        for tr, te in splits:
            fit = L.fit_lgb(led.F, tr, y[tr], s.seg[tr], cols=cols, params=cfg, weight=led.design_w[tr],
                            strat=septic[tr].astype(int), early_stop=early_stop, seed=seed)
            p = fit.predict(led.F, te, raw=True)
            aucs.append(Ranked(p).auc(led.y[te], led.design_w[te]))
            iters.append(fit.best_iter)
        trials.append({"params": cfg, "auroc": float(np.mean(aucs)) if aucs else float("nan"),
                       "trees": iters, "seconds": time.time() - t0})
        if verbose:
            print(f"    tune {i:>2}: AUROC {trials[-1]['auroc']:.4f}  {cfg}  [{trials[-1]['seconds']:.0f}s]", flush=True)
    ok = [t for t in trials if np.isfinite(t["auroc"])]
    best = max(ok, key=lambda t: t["auroc"]) if ok else trials[0]
    return {"best": best["params"], "best_auroc": best["auroc"], "incumbent_auroc": trials[0]["auroc"],
            "gain": best["auroc"] - trials[0]["auroc"], "trials": trials, "tier": tier,
            "val_folds": list(val_folds), "note": "dev subjects only; the winner's dev AUROC is optimistic by selection"}
