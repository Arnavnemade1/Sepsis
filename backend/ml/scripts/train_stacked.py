"""Train + evaluate the stacked-ensemble detector, and (optionally) promote it to
the production artifact if it beats the plain GBM on a held-out patient-grouped test.

    python -m backend.ml.scripts.train_stacked --feature-set wearable [--hdc] \
        [--limit N] [--promote]

Under patient-grouped CV the stack beats the single GBM benchmark (see
backend/ml/README.md). This script re-checks that on a held-out slice, then — only
with --promote AND a real gain — refits on all data and replaces
artifacts/gbm_<set>.joblib (backing up the previous detector). AegisCore loads that
path transparently (the stack takes a DataFrame, like the HDC detector).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import FeatureConfig, build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_split
from backend.ml.models.gbm_baseline import GBMDetector
from backend.ml.models.stacked import StackedEnsembleDetector


def _feature_config(name: str) -> FeatureConfig:
    return {"wearable": FeatureConfig.wearable_only,
            "lab": FeatureConfig.lab_augmented}[name]()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["wearable", "lab"], default="wearable")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hdc", action="store_true", help="include the HDC detector as a base learner")
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--promote", action="store_true",
                    help="replace artifacts/gbm_<set>.joblib if the stack beats the GBM")
    args = ap.parse_args()

    cfg = _feature_config(args.feature_set)
    ds = build_dataset_cached(lambda: load_patients(limit=args.limit), cfg,
                              f"{args.feature_set}_limit{args.limit or 'all'}")
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    print(f"Dataset: {len(ds.y)} rows, prevalence {ds.y.mean():.4f}, "
          f"{len(set(ds.groups.tolist()))} patients")

    tr, te = stratified_group_split(ds.groups, ds.y, args.holdout, args.seed)
    yte, gte, hte, ote = ds.y[te], ds.groups[te], ds.hour_index[te], ds.onset_hour[te]

    gbm = GBMDetector(feature_names=ds.feature_names).fit(
        feats.iloc[tr].to_numpy(), ds.y[tr], groups=ds.groups[tr])
    rep_gbm = metrics.evaluate(yte, gbm.predict_proba(feats.iloc[te].to_numpy()), gte, hte, ote)
    print("\n=== GBM (benchmark) — held-out test ===\n" + rep_gbm.pretty())

    stack = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=args.hdc).fit(
        feats.iloc[tr], ds.y[tr], groups=ds.groups[tr])
    rep_st = metrics.evaluate(yte, stack.predict_proba(feats.iloc[te]), gte, hte, ote)
    print("\n=== Stacked ensemble — held-out test ===\n" + rep_st.pretty())
    print(f"\nmeta weights: { {k: round(v, 2) for k, v in stack.meta_weights().items()} }")
    delta = rep_st.auroc - rep_gbm.auroc
    print(f"AUROC delta (stack - GBM): {delta:+.4f}")

    if args.promote:
        prod = os.path.join(config.ARTIFACTS_DIR, f"gbm_{args.feature_set}.joblib")
        if delta <= 0.005:
            print(f"\nREFUSED to promote: gain {delta:+.4f} within noise (<=0.005).")
            return
        print("\nRefitting the stack on ALL data for the production artifact...")
        full = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=args.hdc).fit(
            feats, ds.y, groups=ds.groups)
        if os.path.exists(prod):
            os.replace(prod, prod + ".gbm.bak")
            print(f"Backed up previous detector -> {prod}.gbm.bak")
        full.save(prod)
        sz = os.path.getsize(prod) / 1e6
        print(f"PROMOTED stacked ensemble -> {prod} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
