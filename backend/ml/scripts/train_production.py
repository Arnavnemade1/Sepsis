"""Train the EXPERIMENTAL HDC-augmented GBM and evaluate it against a held-out slice.

This does **not** overwrite the shipped production detector by default. Read
``backend/ml/models/augmented.py`` first: patient-grouped CV found the HDC-augmented
GBM's improvement over the plain GBM is smaller than the fold-to-fold noise
(0.7235±0.004 vs 0.7263±0.011 AUROC, 2-fold CV) — a single small split had made it
look like a big win, but that did not replicate. This script exists to reproduce
that evaluation, not to promote the model.

    python -m backend.ml.scripts.train_production --feature-set wearable [--limit N]

Saves the fitted model to ``artifacts/experimental_hdc_augmented_<feature-set>.joblib``
— a clearly separate path from ``artifacts/gbm_<feature-set>.joblib`` (the one
AegisCore actually loads). Pass ``--promote-to-production`` only if a NEW, larger-scale
CV run has confirmed a real gain outside the noise band; it is refused otherwise.
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
from backend.ml.models.augmented import HDCAugmentedGBM
from backend.ml.models.gbm_baseline import GBMDetector


def _feature_config(name: str) -> FeatureConfig:
    return {"wearable": FeatureConfig.wearable_only,
            "lab": FeatureConfig.lab_augmented}[name]()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["wearable", "lab"], default="wearable")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--holdout", type=float, default=0.15,
                    help="patient-grouped holdout for a final honest test report")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--promote-to-production", action="store_true",
                    help="DANGEROUS: overwrite artifacts/gbm_<set>.joblib. Refused "
                         "unless this run's held-out AUROC beats the plain GBM by "
                         "more than 1 std of the CV noise (~0.011).")
    args = ap.parse_args()

    cfg = _feature_config(args.feature_set)
    cache_key = f"{args.feature_set}_limit{args.limit or 'all'}"
    print(f"Building features (set='{args.feature_set}')...")
    ds = build_dataset_cached(lambda: load_patients(limit=args.limit), cfg, cache_key)
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    print(f"Dataset: {len(ds.y)} rows, prevalence {ds.y.mean():.4f}, "
          f"{len(set(ds.groups.tolist()))} patients, {len(ds.feature_names)} features")

    tr, te = stratified_group_split(ds.groups, ds.y, args.holdout, args.seed)
    yte, gte, hte, ote = ds.y[te], ds.groups[te], ds.hour_index[te], ds.onset_hour[te]

    print(f"\nTraining plain GBM (production baseline) on {len(tr)} rows...")
    gbm = GBMDetector(feature_names=ds.feature_names).fit(
        feats.iloc[tr].to_numpy(), ds.y[tr], groups=ds.groups[tr])
    rep_gbm = metrics.evaluate(yte, gbm.predict_proba(feats.iloc[te].to_numpy()), gte, hte, ote)
    print("=== GBM (baseline) — held-out test ===")
    print(rep_gbm.pretty())

    print(f"\nTraining EXPERIMENTAL HDC-augmented GBM on {len(tr)} rows...")
    aug = HDCAugmentedGBM(channels=list(cfg.channels), seed=args.seed)
    aug.fit(feats.iloc[tr], ds.y[tr], groups=ds.groups[tr])
    rep_aug = metrics.evaluate(yte, aug.predict_proba(feats.iloc[te]), gte, hte, ote)
    print("=== HDC-augmented GBM (experimental) — held-out test ===")
    print(rep_aug.pretty())

    delta = rep_aug.auroc - rep_gbm.auroc
    print(f"\nAUROC delta (augmented - plain): {delta:+.4f}  "
          f"(CV noise band observed: ~±0.011)")

    exp_out = os.path.join(config.ARTIFACTS_DIR, f"experimental_hdc_augmented_{args.feature_set}.joblib")
    print(f"\nRefitting the experimental model on the full dataset -> {exp_out}")
    aug_full = HDCAugmentedGBM(channels=list(cfg.channels), seed=args.seed)
    aug_full.fit(feats, ds.y, groups=ds.groups)
    aug_full.save(exp_out)
    print(f"Saved (experimental path, NOT the production artifact) -> {exp_out}")

    if args.promote_to_production:
        prod_out = os.path.join(config.ARTIFACTS_DIR, f"gbm_{args.feature_set}.joblib")
        if delta <= 0.011:
            print(f"\nREFUSED: held-out AUROC gain ({delta:+.4f}) does not clear the "
                  f"observed CV noise band (0.011). Not promoting to production. "
                  f"Rerun with more data/folds if you believe the gain is real.")
            return
        os.replace(prod_out, prod_out + ".bak") if os.path.exists(prod_out) else None
        aug_full.save(prod_out)
        print(f"PROMOTED: {exp_out} -> {prod_out} (previous backed up to .bak)")


if __name__ == "__main__":
    main()
