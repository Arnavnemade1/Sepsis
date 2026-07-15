"""Run the recursive self-improvement loop over the stacked detector.

    python -m backend.ml.scripts.self_improve --limit 2000 --rounds 4 --population 4
    python -m backend.ml.scripts.self_improve --limit 4000 --promote

The loop mutates its own champion each round, adapts its own step size, and scores
every candidate under paired patient-grouped CV; the winner is then confronted once
with a locked set of patients that the search never touched (see
backend/ml/selfimprove.py for why the protocol is built this way).

``--promote`` is honoured ONLY if that locked gate confirms a real gain — a gain whose
95% patient-bootstrap CI excludes zero. Anything less leaves the shipped detector
alone. Expect that outcome: this project's own evidence says detection is data-limited,
so a no-gain run is a legitimate, fully-recorded result, not a failure of the script.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import FeatureConfig, build_dataset_cached
from backend.ml.selfimprove import RecursiveSelfImprover, save_ledger
from backend.ml.models.stacked import StackedEnsembleDetector


def _feature_config(name: str) -> FeatureConfig:
    return {"wearable": FeatureConfig.wearable_only,
            "lab": FeatureConfig.lab_augmented}[name]()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["wearable", "lab"], default="wearable")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--gate-fraction", type=float, default=0.2)
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hdc", action="store_true",
                    help="include the HDC detector as a base learner (slow)")
    ap.add_argument("--ledger", default=os.path.join(config.ARTIFACTS_DIR,
                                                     "self_improve_ledger.json"))
    ap.add_argument("--promote", action="store_true",
                    help="replace the production artifact IF the locked gate confirms a gain")
    args = ap.parse_args()

    cfg = _feature_config(args.feature_set)
    ds = build_dataset_cached(lambda: load_patients(limit=args.limit), cfg,
                              f"{args.feature_set}_limit{args.limit or 'all'}")
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    print(f"Dataset: {len(ds.y)} rows, prevalence {ds.y.mean():.4f}, "
          f"{len(set(ds.groups.tolist()))} patients")

    imp = RecursiveSelfImprover(
        feats, ds.y, ds.groups, hour_index=ds.hour_index, onset_hour=ds.onset_hour,
        channels=list(cfg.channels), rounds=args.rounds, population=args.population,
        folds=args.folds, gate_fraction=args.gate_fraction, min_gain=args.min_gain,
        alpha=args.alpha, use_hdc=args.hdc, seed=args.seed)
    res = imp.run()

    save_ledger(res, args.ledger)
    print(f"\nLedger ({len(res.trials)} trials, accepted + rejected) -> {args.ledger}")

    print("\n" + "=" * 72)
    print("VERDICT: " + res.verdict)
    print("=" * 72)

    if args.promote:
        if not res.promoted:
            print("\nNOT promoting: the locked gate did not confirm a gain. "
                  "The shipped detector is unchanged.")
            return
        prod = os.path.join(config.ARTIFACTS_DIR, f"gbm_{args.feature_set}.joblib")
        print("\nGate confirmed. Refitting the champion on ALL data for production...")
        full = StackedEnsembleDetector(
            channels=list(cfg.channels), use_hdc=args.hdc,
            roster=list(res.champion_config["roster"]),
            params=dict(res.champion_config["params"]),
        ).fit(feats, ds.y, groups=ds.groups)
        if os.path.exists(prod):
            os.replace(prod, prod + ".selfimprove.bak")
            print(f"Backed up previous detector -> {prod}.selfimprove.bak")
        full.save(prod)
        print(f"PROMOTED -> {prod} ({os.path.getsize(prod)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
