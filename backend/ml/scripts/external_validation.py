"""Cross-hospital external validation — the strongest generalization test.

CinC 2019 set A is Beth Israel Deaconess (BIDMC) and set B is Emory. Training on
one hospital and testing on the OTHER measures whether the detector generalizes
across institutions, populations, and monitoring practices — not just across
patients within one hospital. This is the difference between "works on our data"
and "works".

    python -m backend.ml.scripts.external_validation --feature-set wearable_temporal

Reports both directions (A->B and B->A) with the full calibrated metric suite.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from backend.ml import config
from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.models.gbm_baseline import GBMDetector
from backend.ml.scripts.train_and_compare import _feature_config


def _dataset(feature_set: str, set_name: str, limit):
    cfg = _feature_config(feature_set)
    key = f"{feature_set}_{set_name}_limit{limit or 'all'}"
    return build_dataset_cached(
        lambda: load_patients(sets=[set_name], limit=limit), cfg, key)


def _train_test(train_ds, test_ds):
    model = GBMDetector(feature_names=train_ds.feature_names)
    model.fit(train_ds.X, train_ds.y, groups=train_ds.groups)
    p = model.predict_proba(test_ds.X)
    return metrics.evaluate(test_ds.y, p, test_ds.groups, test_ds.hour_index, test_ds.onset_hour)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="wearable_temporal")
    ap.add_argument("--limit-per-set", type=int, default=None)
    args = ap.parse_args()

    print(f"Building per-hospital datasets (feature_set='{args.feature_set}')...")
    A = _dataset(args.feature_set, "training_setA", args.limit_per_set)
    B = _dataset(args.feature_set, "training_setB", args.limit_per_set)
    print(f"  set A (BIDMC): {len(A.y)} rows, prev {A.y.mean():.4f}")
    print(f"  set B (Emory): {len(B.y)} rows, prev {B.y.mean():.4f}")

    print("\n=== Train BIDMC (A) -> Test Emory (B) ===")
    ab = _train_test(A, B)
    print(ab.pretty())
    print("\n=== Train Emory (B) -> Test BIDMC (A) ===")
    ba = _train_test(B, A)
    print(ba.pretty())

    out = {
        "feature_set": args.feature_set,
        "A_to_B": {k: float(getattr(ab, k)) for k in ("auroc", "auprc", "brier", "ece", "normalized_utility")},
        "B_to_A": {k: float(getattr(ba, k)) for k in ("auroc", "auprc", "brier", "ece", "normalized_utility")},
    }
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(os.path.join(config.ARTIFACTS_DIR, f"external_validation_{args.feature_set}.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    mean_auroc = 0.5 * (ab.auroc + ba.auroc)
    print(f"\nCross-hospital mean AUROC: {mean_auroc:.4f}")
    print("(Generalization across institutions — the honest deployment estimate.)")


if __name__ == "__main__":
    main()
