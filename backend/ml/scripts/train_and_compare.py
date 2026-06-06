"""Train the data-grounded HDC detector and the GBM benchmark on identical
patient-level splits and report the full clinical metric suite for each.

    python -m backend.ml.scripts.train_and_compare --limit 2400 --feature-set wearable

Everything is leakage-free: one patient's hours never cross the split boundary,
ranges/centroids/calibration are learned on TRAIN only, metrics are on TEST.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.data.cinc2019 import load_patients, dataset_summary
from backend.ml.data.features import FeatureConfig, build_dataset
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_split, stratified_group_folds
from backend.ml.models.hdc_detector import HDCDetector
from backend.ml.models.gbm_baseline import GBMDetector


def _fit_eval_fold(cfg, feats, ds, train_idx, test_idx):
    """Train both models on one fold and return their EvalReports."""
    f_tr, f_te = feats.iloc[train_idx], feats.iloc[test_idx]
    y_tr, y_te = ds.y[train_idx], ds.y[test_idx]
    g_te, h_te, o_te = ds.groups[test_idx], ds.hour_index[test_idx], ds.onset_hour[test_idx]

    hdc = HDCDetector(channels=list(cfg.channels)).fit(f_tr, y_tr)
    rep_hdc = metrics.evaluate(y_te, hdc.predict_proba(f_te), g_te, h_te, o_te)

    gbm = GBMDetector(feature_names=ds.feature_names).fit(f_tr.to_numpy(), y_tr)
    rep_gbm = metrics.evaluate(y_te, gbm.predict_proba(f_te.to_numpy()), g_te, h_te, o_te)
    return rep_hdc, rep_gbm


def _run_cv(cfg, feats, ds, folds, seed):
    keys = ("auroc", "auprc", "brier", "ece", "normalized_utility")
    acc = {"hdc": {k: [] for k in keys}, "gbm": {k: [] for k in keys}}
    for i, (tr, te) in enumerate(stratified_group_folds(ds.groups, ds.y, folds, seed), 1):
        print(f"\n--- Fold {i}/{folds} (test rows={len(te)}, prev={ds.y[te].mean():.4f}) ---")
        rep_hdc, rep_gbm = _fit_eval_fold(cfg, feats, ds, tr, te)
        for k in keys:
            acc["hdc"][k].append(getattr(rep_hdc, k))
            acc["gbm"][k].append(getattr(rep_gbm, k))
        print(f"  HDC  AUROC={rep_hdc.auroc:.3f} AUPRC={rep_hdc.auprc:.3f} util={rep_hdc.normalized_utility:.3f}")
        print(f"  GBM  AUROC={rep_gbm.auroc:.3f} AUPRC={rep_gbm.auprc:.3f} util={rep_gbm.normalized_utility:.3f}")

    print(f"\n=========== {folds}-FOLD CROSS-VALIDATION (mean ± std) ===========")
    print(f"{'metric':<20}{'HDC(grounded)':>22}{'GBM(benchmark)':>22}")
    for k in keys:
        h, g = np.array(acc["hdc"][k]), np.array(acc["gbm"][k])
        print(f"{k:<20}{h.mean():>13.4f} ± {h.std():.3f}{g.mean():>13.4f} ± {g.std():.3f}")
    print("================================================================")
    return acc


def _feature_config(name: str) -> FeatureConfig:
    return FeatureConfig.lab_augmented() if name == "lab" else FeatureConfig.wearable_only()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--feature-set", choices=["wearable", "lab"], default="wearable")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=1,
                    help="if >1, run patient-grouped cross-validation (mean ± std)")
    ap.add_argument("--save", action="store_true", help="persist fitted models")
    args = ap.parse_args()

    print("Loading patients...")
    records = load_patients(limit=args.limit)
    if not records:
        raise SystemExit("No data. Run: python -m backend.ml.data.cinc2019 --download")
    print("Dataset:", dataset_summary(records))

    cfg = _feature_config(args.feature_set)
    print(f"\nBuilding causal features (set='{args.feature_set}', channels={len(cfg.channels)})...")
    ds = build_dataset(records, cfg)
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)

    if args.folds > 1:
        acc = _run_cv(cfg, feats, ds, args.folds, args.seed)
        if args.save:
            os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
            with open(os.path.join(config.ARTIFACTS_DIR, f"cv_{args.feature_set}.json"), "w") as fh:
                json.dump({m: {k: [float(x) for x in v] for k, v in d.items()}
                           for m, d in acc.items()}, fh, indent=2)
        return

    train_idx, test_idx = stratified_group_split(ds.groups, ds.y, args.test_size, args.seed)
    f_tr, f_te = feats.iloc[train_idx], feats.iloc[test_idx]
    y_tr, y_te = ds.y[train_idx], ds.y[test_idx]
    g_te, h_te, o_te = ds.groups[test_idx], ds.hour_index[test_idx], ds.onset_hour[test_idx]
    print(f"Train rows={len(train_idx)} (prev {y_tr.mean():.4f}) | "
          f"Test rows={len(test_idx)} (prev {y_te.mean():.4f})")

    results = {}

    # --- Data-grounded HDC detector --------------------------------------- #
    print("\n[1/2] Training data-grounded HDC detector...")
    hdc = HDCDetector(channels=list(cfg.channels))
    hdc.fit(f_tr, y_tr)
    p_hdc = hdc.predict_proba(f_te)
    rep_hdc = metrics.evaluate(y_te, p_hdc, g_te, h_te, o_te)
    print("\n=== Data-grounded HDC detector (TEST) ===")
    print(rep_hdc.pretty())
    results["hdc"] = rep_hdc.__dict__

    # --- Gradient-boosting benchmark -------------------------------------- #
    print("\n[2/2] Training gradient-boosting benchmark...")
    gbm = GBMDetector(feature_names=ds.feature_names)
    gbm.fit(f_tr.to_numpy(), y_tr)
    p_gbm = gbm.predict_proba(f_te.to_numpy())
    rep_gbm = metrics.evaluate(y_te, p_gbm, g_te, h_te, o_te)
    print("\n=== Gradient-boosting benchmark (TEST) ===")
    print(rep_gbm.pretty())
    results["gbm"] = rep_gbm.__dict__

    # --- Summary ---------------------------------------------------------- #
    print("\n========================= HEAD-TO-HEAD =========================")
    print(f"{'metric':<22}{'HDC(grounded)':>16}{'GBM(benchmark)':>16}")
    for k in ("auroc", "auprc", "brier", "ece", "normalized_utility"):
        print(f"{k:<22}{getattr(rep_hdc, k):>16.4f}{getattr(rep_gbm, k):>16.4f}")
    print("===============================================================")

    if args.save:
        os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
        hdc.save(os.path.join(config.ARTIFACTS_DIR, f"hdc_{args.feature_set}.joblib"))
        gbm.save(os.path.join(config.ARTIFACTS_DIR, f"gbm_{args.feature_set}.joblib"))
        with open(os.path.join(config.ARTIFACTS_DIR, f"compare_{args.feature_set}.json"), "w") as fh:
            json.dump({k: {m: float(v) for m, v in r.items() if isinstance(v, (int, float))}
                       for k, r in results.items()}, fh, indent=2)
        print(f"Saved models + report to {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
