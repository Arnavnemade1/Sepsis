"""Phase-7 accuracy evaluation: compare feature sets with the production GBM.

GBM-only (the deployed detector) patient-grouped K-fold CV, using the cached
feature matrices so reruns are instant. Reports mean ± std for the metrics that
matter at this prevalence.

    python -m backend.ml.scripts.phase7_eval --limit 6000 --folds 4 \
        --feature-sets wearable wearable_temporal
"""
from __future__ import annotations

import argparse

import numpy as np

from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_folds
from backend.ml.models.gbm_baseline import GBMDetector
from backend.ml.scripts.train_and_compare import _feature_config


def _cv(feature_set: str, limit, folds: int, seed: int):
    cfg = _feature_config(feature_set)
    key = f"{feature_set}_limit{limit or 'all'}"
    ds = build_dataset_cached(lambda: load_patients(limit=limit), cfg, key)
    keys = ("auroc", "auprc", "ece", "normalized_utility")
    acc = {k: [] for k in keys}
    for tr, te in stratified_group_folds(ds.groups, ds.y, folds, seed):
        m = GBMDetector(feature_names=ds.feature_names)
        m.fit(ds.X[tr], ds.y[tr], groups=ds.groups[tr])
        rep = metrics.evaluate(ds.y[te], m.predict_proba(ds.X[te]),
                               ds.groups[te], ds.hour_index[te], ds.onset_hour[te])
        for k in keys:
            acc[k].append(getattr(rep, k))
    return ds, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature-sets", nargs="+", default=["wearable", "wearable_temporal"])
    args = ap.parse_args()

    rows = {}
    for fs in args.feature_sets:
        ds, acc = _cv(fs, args.limit, args.folds, args.seed)
        rows[fs] = (ds, acc)
        print(f"[{fs}] {len(ds.feature_names)} features, {len(ds.y)} rows")

    print(f"\n===== Phase-7 GBM {args.folds}-fold CV (limit={args.limit}) =====")
    hdr = f"{'feature_set':<20}" + "".join(f"{k:>16}" for k in ("auroc", "auprc", "ece", "utility"))
    print(hdr)
    for fs, (_, acc) in rows.items():
        cells = []
        for k in ("auroc", "auprc", "ece", "normalized_utility"):
            a = np.array(acc[k]); cells.append(f"{a.mean():.4f}±{a.std():.3f}")
        print(f"{fs:<20}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
