"""The definitive GB-QHDC ablation + Dimensional-Capacity sweep, on real CinC 2019.

Runs the exact scientific comparison an ISEF panel needs, leakage-free (patient-
grouped CV), on identical splits:

  A. ABLATION MATRIX
       1. GBM              — classical gradient-boosted trees (the benchmark)
       2. HDC              — classical hyperdimensional detector (collision-capped)
       3. GB-QHDC          — the unified quantum-hyperdimensional booster (this work)

  B. DIMENSIONAL-CAPACITY SWEEP
       GB-QHDC at N = 4,6,8,10 qubits (D = 2^N virtual dimensions), plotting
       information/AUROC vs simulated quantum dimensionality vs wall-clock — showing
       where added quantum dimensions stop paying off under classical simulation.

    python -m backend.ml.scripts.qhdc_ablation --limit 1500 --folds 2 --qubits 4,6,8

Honesty: GB-QHDC is an EXACT statevector simulation (no shot noise unless --shots).
The claim this script supports is whatever the numbers below show — nothing more.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import FeatureConfig, build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_folds
from backend.ml.models.gbm_baseline import GBMDetector
from backend.ml.models.hdc_detector import HDCDetector
from backend.ml.quantum.gb_qhdc import GBQHDC


def _cv_eval(fit_predict, ds, feats, folds, seed):
    keys = ("auroc", "auprc", "normalized_utility")
    acc = {k: [] for k in keys}
    t0 = time.time()
    for tr, te in stratified_group_folds(ds.groups, ds.y, folds, seed):
        p = fit_predict(feats.iloc[tr], ds.y[tr], ds.groups[tr], feats.iloc[te])
        r = metrics.evaluate(ds.y[te], p, ds.groups[te], ds.hour_index[te], ds.onset_hour[te])
        for k in keys:
            acc[k].append(getattr(r, k))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--qubits", type=str, default="4,6,8")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--spsa", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = FeatureConfig.wearable_only()
    ds = build_dataset_cached(lambda: load_patients(limit=args.limit), cfg,
                              f"wearable_limit{args.limit}")
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    print(f"Dataset: {len(ds.y)} rows, prevalence {ds.y.mean():.4f}, "
          f"{len(set(ds.groups.tolist()))} patients, {len(ds.feature_names)} features\n")

    def fit_gbm(ftr, ytr, gtr, fte):
        return GBMDetector(feature_names=ds.feature_names).fit(
            ftr.to_numpy(), ytr, groups=gtr).predict_proba(fte.to_numpy())

    def fit_hdc(ftr, ytr, gtr, fte):
        return HDCDetector(channels=list(cfg.channels)).fit(
            ftr, ytr, groups=gtr).predict_proba(fte)

    def make_qhdc(nq):
        def f(ftr, ytr, gtr, fte):
            return GBQHDC(n_qubits=nq, n_rounds=args.rounds, spsa_steps=args.spsa,
                          seed=args.seed).fit(ftr.to_numpy(), ytr).predict_proba(fte.to_numpy())
        return f

    results = {}
    print("=" * 66)
    print("A. ABLATION MATRIX  (mean ± std over folds)")
    print("=" * 66)
    print(f"{'model':<22}{'AUROC':>16}{'AUPRC':>12}{'utility':>12}{'sec':>8}")
    qubit_list = [int(q) for q in args.qubits.split(",")]
    rows = [("GBM (trees)", fit_gbm), ("HDC (classical)", fit_hdc),
            (f"GB-QHDC (N={qubit_list[-1]})", make_qhdc(qubit_list[-1]))]
    for name, fn in rows:
        (m, t) = _cv_eval(fn, ds, feats, args.folds, args.seed)
        results[name] = {k: v for k, v in m.items()}
        print(f"{name:<22}{m['auroc'][0]:>8.4f}±{m['auroc'][1]:.3f}"
              f"{m['auprc'][0]:>12.4f}{m['normalized_utility'][0]:>12.4f}{t:>8.0f}")

    print("\n" + "=" * 66)
    print("B. DIMENSIONAL-CAPACITY SWEEP  (GB-QHDC vs qubit count)")
    print("=" * 66)
    print(f"{'qubits':>7}{'dims(2^N)':>11}{'AUROC':>16}{'AUPRC':>12}{'sec':>8}")
    sweep = {}
    for nq in qubit_list:
        (m, t) = _cv_eval(make_qhdc(nq), ds, feats, args.folds, args.seed)
        sweep[nq] = {"auroc": m["auroc"], "auprc": m["auprc"], "dim": 1 << nq, "sec": t}
        print(f"{nq:>7}{1 << nq:>11}{m['auroc'][0]:>8.4f}±{m['auroc'][1]:.3f}"
              f"{m['auprc'][0]:>12.4f}{t:>8.0f}")

    out = {"ablation": results, "sweep": sweep,
           "config": {"limit": args.limit, "folds": args.folds, "rounds": args.rounds,
                      "spsa_steps": args.spsa}}
    import os
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(config.ARTIFACTS_DIR, "qhdc_ablation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
