"""Derive the alarm operating point from real data and gate it on held-out patients.

    python -m backend.ml.scripts.derive_operating_point --limit 4000
    python -m backend.ml.scripts.derive_operating_point --limit 4000 --objective utility
    python -m backend.ml.scripts.derive_operating_point --limit 4000 --save

Builds patient-grouped out-of-fold predictions over the training patients, derives the
threshold from those (see backend/ml/operating_point.py for why OOF is non-negotiable),
then reports what it does on patients whose data never touched the derivation — beside
the shipped hand-set constants, so the change is always quoted against what it replaces.

``--save`` writes artifacts/operating_point.json, which AegisCore loads at boot. It
refuses to save a point that loses to the incumbent on held-out utility.

``--detector-objective`` must match the promoted artifact. A threshold is a reading of
one particular score distribution: derive it against a differently-fitted detector and
the number is meaningless even though nothing errors.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import FeatureConfig, build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_folds, stratified_group_split
from backend.ml.models.stacked import StackedEnsembleDetector
from backend.ml import operating_point as opmod

SHIPPED_HIGH, SHIPPED_MEDIUM = 0.12, 0.04     # the constants this script exists to replace


def _feature_config(name: str) -> FeatureConfig:
    return {"wearable": FeatureConfig.wearable_only, "lab": FeatureConfig.lab_augmented}[name]()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["wearable", "lab"], default="wearable")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--objective", choices=["constrained", "utility"], default="constrained",
                    help="how the THRESHOLD is chosen from the out-of-fold scores")
    ap.add_argument("--detector-objective", choices=["balanced", "utility"], default="utility",
                    help="which objective the DETECTOR itself is fitted with; must match "
                         "the promoted artifact or the derived threshold belongs to a "
                         "different score distribution than the one it will be applied to")
    ap.add_argument("--max-far", type=float, default=0.15,
                    help="cap on the fraction of non-septic patients ever alarmed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true", help="write artifacts/operating_point.json")
    args = ap.parse_args()

    cfg = _feature_config(args.feature_set)
    ds = build_dataset_cached(lambda: load_patients(limit=args.limit), cfg,
                              f"{args.feature_set}_limit{args.limit or 'all'}")
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    print(f"Dataset: {len(ds.y)} rows / {len(set(ds.groups.tolist()))} patients / "
          f"prevalence {ds.y.mean():.4f}")

    tr, te = stratified_group_split(ds.groups, ds.y, args.holdout, args.seed)
    ytr, gtr, htr, otr = ds.y[tr], ds.groups[tr], ds.hour_index[tr], ds.onset_hour[tr]
    yte, gte, hte, ote = ds.y[te], ds.groups[te], ds.hour_index[te], ds.onset_hour[te]
    ftr = feats.iloc[tr].reset_index(drop=True)

    print(f"\nBuilding {args.folds}-fold out-of-fold predictions on train patients...")
    oof = np.zeros(len(ytr))
    for k, (a, b) in enumerate(stratified_group_folds(gtr, ytr, args.folds, args.seed)):
        mk = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=False,
                                     objective=args.detector_objective).fit(
            ftr.iloc[a], ytr[a], groups=gtr[a],
            hour_index=htr[a], onset_hour=otr[a])
        oof[b] = mk.predict_proba(ftr.iloc[b])
        print(f"  fold {k+1}/{args.folds}")

    op = opmod.derive(ytr, oof, gtr, htr, otr,
                      objective=args.objective, max_false_alarm_rate=args.max_far)

    print("\n=== The model's actual output distribution (why the guess failed) ===")
    print(f"  p95 {op.prob_p95:.4f} | p99 {op.prob_p99:.4f} | max {op.prob_max:.4f}")
    print(f"  fraction of hours above the SHIPPED {SHIPPED_HIGH}: {np.mean(oof >= SHIPPED_HIGH):.4f}")
    print(f"  -> the shipped alarm sat at the {100*np.mean(oof < SHIPPED_HIGH):.1f}th percentile "
          f"of the detector's own output")

    print("\n=== Threshold sweep (out-of-fold, train patients) ===")
    print(opmod.sweep_table(ytr, oof, gtr, htr, otr,
                           [0.01, 0.02, 0.025, 0.05, 0.075, 0.10, 0.12, 0.15, 0.20, 0.30]))

    print(f"\nDerived ({op.objective}"
          + (f", FAR<={op.max_false_alarm_rate}" if op.max_false_alarm_rate else "")
          + f"): HIGH={op.high:.4f}  MEDIUM={op.medium:.4f}")

    print("\nFitting the deployed model on all train patients...")
    m = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=False,
                                objective=args.detector_objective).fit(
        ftr, ytr, groups=gtr, hour_index=htr, onset_hour=otr)
    p_te = m.predict_proba(feats.iloc[te])

    print("\n=== GATE — held-out patients (never used to derive the threshold) ===")
    def row(name, thr):
        u = metrics.normalized_utility(yte, (p_te >= thr).astype(int), gte, hte)
        s = metrics.early_warning(yte, p_te, gte, ote, hte, thr)
        print(f"{name:24s} thr={thr:.4f}  utility={u:+.4f}  detect={s['detection_rate']:.3f}  "
              f"FAR={s['false_alarm_rate']:.3f}  lead={s['median_lead_time_hours']:.1f}h")
        return u
    u_ship = row("shipped (hand-set)", SHIPPED_HIGH)
    u_new = row(f"derived ({op.objective})", op.high)
    print(f"\nAUROC is UNCHANGED at {metrics.auroc(yte, p_te):.4f} — this is the same "
          f"detector, read at a different point.")
    print(f"Held-out utility delta: {u_new - u_ship:+.4f}")

    if args.save:
        if not (u_new > u_ship):
            print(f"\nREFUSED to save: derived point does not beat the shipped constants on "
                  f"held-out utility ({u_new:+.4f} vs {u_ship:+.4f}).")
            return
        opmod.save(op)
        print(f"\nSAVED -> {opmod.DEFAULT_PATH}\nAegisCore will load it at boot.")


if __name__ == "__main__":
    main()
