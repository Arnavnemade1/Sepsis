"""Does the shipped detector get better with the data we already have? — the registered run.

    python -m backend.ml.scripts.data_lever

The shipped artifact was fitted on 5,996 CinC patients. This fits the SAME architecture on
6,000 versus all training patients of each fold of the full cohort (40,335 patients, 3
patient-grouped folds) and scores both on the identical held-out patients.
Registration: ``artifacts/data_lever_prereg.json`` (committed before the run).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from backend.ml import config
from backend.ml.eval.bench import Benchmark, BenchResult
from backend.ml.genesis.harness import Ranked, hour_matched_auroc

PREREG = os.path.join(config.ARTIFACTS_DIR, "data_lever_prereg.json")
OOF = os.path.join(config.ARTIFACTS_DIR, "frontier", "data_lever_full_oof.npz")
OUT = os.path.join(config.ARTIFACTS_DIR, "data_lever.json")
SMALL = 6000


def fit_predict(ds, tr, te):
    from backend.ml.models.stacked import StackedEnsembleDetector
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    m = StackedEnsembleDetector(channels=list(config.WEARABLE_CHANNELS), use_hdc=False, objective="patient",
                                roster=["hgb", "hgb2", "et", "lr", "xgb@utility"])
    m.fit(feats.iloc[tr], ds.y[tr], groups=ds.groups[tr], hour_index=ds.hour_index[tr],
          onset_hour=ds.onset_hour[tr])
    return m.predict_proba(feats.iloc[te])


def main() -> int:
    reg_hash = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()[:16]
    bench = Benchmark.load("full", n_folds=3, seed=42)
    ds = bench.ds
    print(bench.describe(), flush=True)
    oof = {"small": np.full(len(ds.y), np.nan), "all": np.full(len(ds.y), np.nan)}
    rng = np.random.default_rng(0)
    for k, (tr, te) in enumerate(bench.folds):
        pts = np.unique(ds.groups[tr])
        keep = rng.choice(pts, size=min(SMALL, len(pts)), replace=False)
        tr_small = tr[np.isin(ds.groups[tr], keep)]
        for arm, rows in (("small", tr_small), ("all", tr)):
            t0 = time.time()
            oof[arm][te] = fit_predict(ds, rows, te)
            print(f"  fold {k} {arm:<5} {len(np.unique(ds.groups[rows])):>6,} patients  "
                  f"AUROC {Ranked(oof[arm][te]).auc(ds.y[te]):.4f}  [{time.time() - t0:.0f}s]", flush=True)
        np.savez(OOF, **oof)
    r = {a: Ranked(v) for a, v in oof.items()}
    pts, inv = np.unique(ds.groups, return_inverse=True)
    br = np.random.default_rng(0)
    d = np.empty(1000)
    for i in range(1000):
        w = br.multinomial(len(pts), np.full(len(pts), 1.0 / len(pts)))[inv].astype(np.float64)
        d[i] = r["all"].auc(ds.y, w) - r["small"].auc(ds.y, w)
    lo, hi = np.percentile(d, [2.5, 97.5])
    det = bench.compare_alarm_budget(BenchResult("all", [], oof["all"], 0.0),
                                     BenchResult("small", [], oof["small"], 0.0), target_far=0.15, n_boot=1000)
    rep = {"prereg_sha256_16": reg_hash, "patients": int(len(pts)),
           "auroc": {a: r[a].auc(ds.y) for a in oof},
           "hour_matched_auroc": {a: float(hour_matched_auroc(ds.y, oof[a], ds.hour_index)) for a in oof},
           "auprc": {a: float(average_precision_score(ds.y, oof[a])) for a in oof},
           "delta_auroc": r["all"].auc(ds.y) - r["small"].auc(ds.y), "ci95": [float(lo), float(hi)],
           "success": bool(lo > 0),
           "per_fold_delta": [Ranked(oof["all"][te]).auc(ds.y[te]) - Ranked(oof["small"][te]).auc(ds.y[te])
                              for _, te in bench.folds],
           "delta_detect_at_far15": det}
    json.dump(rep, open(OUT, "w"), indent=1, default=float)
    print(json.dumps({k: rep[k] for k in ("auroc", "hour_matched_auroc", "auprc", "delta_auroc", "ci95",
                                          "success", "per_fold_delta")}, indent=1, default=float))
    print("detect@FA15:", {k: det[k] for k in ("delta_detection", "ci_lo", "ci_hi", "significant")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
