"""The one-shot locked evaluation.

Everything before this module was chosen on the dev folds. Here each detector is fitted
ONCE on all 32,268 dev patients and scored ONCE on the 8,067 locked patients that no
design decision has seen. Every call to :meth:`Cohort.lock_rows` is written to
``artifacts/genesis/lock_ledger.jsonl``, so the number of looks is on the record.

Detectors compared on the identical locked patients:

``shipped``    the production wearable detector's exact architecture
               (``StackedEnsembleDetector``, patient objective,
               hgb+hgb2+et+lr+xgb@utility) on its own 94 wearable features — the
               "0.77" line, refitted on the dev patients so it gets the same data
               GENESIS gets;
``wearable``   one LightGBM on the GENESIS features of the seven vitals only;
``direct``     one LightGBM on all 510 GENESIS features (the all-variable baseline);
``genesis``    the full architecture: label-genesis heads + stage-2 combiner.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from backend.ml.genesis import harness as H
from backend.ml.genesis import learners as L
from backend.ml.genesis import stack as S

OUT = os.path.join(H.RESULTS_DIR, "locked")


def head_locked(c: H.Cohort, head: S.Head, lock: np.ndarray, seed: int = 99) -> np.ndarray:
    """Fit ``head`` on every dev row, return calibrated margins on the locked rows."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"head_{head.name}.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float64)
    tgt, w = head.target(c)
    t0 = time.time()
    fit = S.fit_head(c, head, c.dev_rows, tgt, w, seed=seed)
    z = fit.predict(c.F if head.data is None else head.data, lock, raw=True)
    np.save(path, z.astype(np.float32))
    print(f"    locked head {head.name}: trees {fit.best_iter} [{time.time()-t0:.0f}s]", flush=True)
    return z


def full_vector(c: H.Cohort, dev_oof: np.ndarray, lock: np.ndarray, lock_vals: np.ndarray) -> np.ndarray:
    """Dev rows carry their OOF margin, locked rows their locked-fit margin."""
    v = np.array(dev_oof, dtype=np.float64, copy=True)
    v[lock] = lock_vals
    return v


def shipped_stack_scores(c: H.Cohort, lock: np.ndarray) -> Dict[str, np.ndarray]:
    """The production detector's architecture, refitted on the dev patients."""
    path = os.path.join(OUT, "shipped_stack.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    from threadpoolctl import threadpool_limits
    from backend.ml.eval.bench import _load_cached, concat
    from backend.ml.models.stacked import StackedEnsembleDetector
    fm = concat([_load_cached("wearable_training_setA_limitall"),
                 _load_cached("wearable_training_setB_limitall")])
    if not (np.array_equal(fm.groups, c.raw.pid[c.seg]) and np.array_equal(fm.hour_index, c.hour)):
        raise RuntimeError("wearable cache rows do not align with the GENESIS raw store")
    if not np.array_equal(fm.y, c.y):
        raise RuntimeError("wearable cache labels disagree with the raw store")
    dev = c.dev_rows
    det = StackedEnsembleDetector(roster=["hgb", "hgb2", "et", "lr", "xgb@utility"],
                                  objective="patient", use_hdc=False, inner_folds=3,
                                  params={"et": {"n_jobs": L.THREADS}, "lgbm": {"n_jobs": L.THREADS},
                                          "xgb": {"n_jobs": L.THREADS}})
    t0 = time.time()
    with threadpool_limits(L.THREADS):
        det.fit(pd.DataFrame(fm.X[dev], columns=fm.feature_names), fm.y[dev],
                groups=fm.groups[dev], hour_index=fm.hour_index[dev], onset_hour=fm.onset_hour[dev])
        feats = pd.DataFrame(fm.X[lock], columns=fm.feature_names)
        served = det.predict_proba(feats)
        raw_meta = det.meta_.predict_proba(det._base_matrix(feats))[:, 1]
    print(f"    shipped stack refit on dev + scored locked [{time.time()-t0:.0f}s]", flush=True)
    np.savez(path, served=served, meta=raw_meta)
    return {"served": served, "meta": raw_meta}


def score_all(c: H.Cohort, lock: np.ndarray, preds: Dict[str, np.ndarray],
              reference: str, n_boot: int = 2000) -> Dict[str, Dict]:
    """Score every detector on the locked rows; paired bootstrap vs ``reference``."""
    out: Dict[str, Dict] = {}
    full = {k: np.full(len(c.y), np.nan) for k in preds}
    for k, v in preds.items():
        full[k][lock] = v
    for k, v in preds.items():
        s = H.score(c, v, lock)
        cmp = None if k == reference else H.compare(c, full[k], full[reference], n_boot=n_boot, rows=lock)
        out[k] = {"score": s, f"vs_{reference}": cmp}
    return out
