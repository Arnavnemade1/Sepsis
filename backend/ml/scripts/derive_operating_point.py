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

The detector configuration is read from ``artifacts/detector_manifest.json`` — the
same file the promotion gate writes — rather than defaulted, because a threshold is a
reading of one particular score distribution. Derive it against a differently-fitted
detector and every number in the artifact is meaningless while nothing errors.

That was a live hazard in THIS script specifically: it used to default to
``[hgb, hgb2, et, lr]`` under the ``utility`` objective, while the promoted detector
is ``[hgb, hgb2, et, lr, xgb@utility]`` under ``balanced``. Anyone running it with
``--save`` would have installed a threshold measured on a model nobody deploys. (The
shipped artifact did not come from here — ``promote_frontier.py`` writes it, and does
use the promoted configuration — so this was a loaded gun rather than a fired one.)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import FeatureConfig, build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_folds, stratified_group_split
from backend.ml.models.stacked import StackedEnsembleDetector
from backend.ml import config
from backend.ml import operating_point as opmod

SHIPPED_HIGH, SHIPPED_MEDIUM = 0.12, 0.04     # the constants this script exists to replace

MANIFEST = os.path.join(config.ARTIFACTS_DIR, "detector_manifest.json")


def _feature_config(name: str) -> FeatureConfig:
    return {"wearable": FeatureConfig.wearable_only, "lab": FeatureConfig.lab_augmented}[name]()


def promoted_config():
    """The detector configuration the promotion gate actually shipped.

    Read rather than assumed. The alternative — a default roster in this script — is
    what let the threshold and the detector drift apart silently.
    """
    with open(MANIFEST) as fh:
        man = json.load(fh)
    if not man.get("promoted"):
        raise SystemExit(f"{MANIFEST} records promoted=false; there is no shipped "
                         f"detector to derive a threshold for")
    cfg = man["configs"]["candidate"]
    return list(cfg["roster"]), str(cfg["objective"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["wearable", "lab"], default="wearable")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--holdout", type=float, default=0.25,
                    help="patient fraction reserved for the GATE, which verifies the "
                         "chosen point and is never used to choose it")
    ap.add_argument("--calibrate", type=float, default=0.30,
                    help="fraction of the remaining patients used to CHOOSE the "
                         "threshold, scored by a model fit without them")
    ap.add_argument("--oof-diagnostic", action="store_true",
                    help="additionally compute cross-validated scores and print how far "
                         "their distribution sits from the single-fit one (costs N extra fits)")
    ap.add_argument("--objective", choices=["constrained", "utility"], default="constrained",
                    help="how the THRESHOLD is chosen from the out-of-fold scores")
    ap.add_argument("--detector-objective", choices=["balanced", "utility"], default=None,
                    help="objective the DETECTOR is fitted with; defaults to the promoted "
                         "config in artifacts/detector_manifest.json")
    ap.add_argument("--roster", default=None,
                    help="comma-separated stack members; defaults to the promoted config. "
                         "Overriding either this or --detector-objective derives a "
                         "threshold for a detector that is not the one deployed")
    ap.add_argument("--max-far", type=float, default=0.15,
                    help="cap on the fraction of non-septic patients ever alarmed")
    ap.add_argument("--far-boot", type=int, default=400,
                    help="patient-bootstrap draws used to enforce the cap on its UPPER "
                         "confidence bound; 0 enforces it on the point estimate, which "
                         "is what let a capped point breach the cap on held-out data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true", help="write artifacts/operating_point.json")
    args = ap.parse_args()

    roster, objective = promoted_config()
    if args.roster:
        roster = [m.strip() for m in args.roster.split(",") if m.strip()]
    if args.detector_objective:
        objective = args.detector_objective
    detector_id = f"{objective}:{'+'.join(roster)}"
    promoted_id = "{}:{}".format(*(lambda r, o: (o, "+".join(r)))(*promoted_config()))
    print(f"Detector: {detector_id}"
          + ("" if detector_id == promoted_id
             else f"   ** OVERRIDDEN — promoted is {promoted_id} **"))

    cfg = _feature_config(args.feature_set)
    ds = build_dataset_cached(lambda: load_patients(limit=args.limit), cfg,
                              f"{args.feature_set}_limit{args.limit or 'all'}")
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    print(f"Dataset: {len(ds.y)} rows / {len(set(ds.groups.tolist()))} patients / "
          f"prevalence {ds.y.mean():.4f}")

    # THREE-way split, not two. The threshold used to be chosen from cross-validated
    # out-of-fold scores and then applied to a model fit on all the data — two
    # different fitting regimes producing two different score distributions. The gap
    # is not subtle: the OOF scores put the false-alarm rate at 0.014 at threshold
    # 0.12 where the deployed fit delivered 0.122, nearly nine times higher, while the
    # bootstrap correctly said the sampling error was +/-0.01. An absolute probability
    # threshold simply does not transfer between fits of this ensemble — its output
    # scale moves with the training set (p95 0.0606 at 3k patients, 0.0884 at 8k).
    #
    # So the threshold is now chosen on CALIBRATE: patients scored by a model fit
    # without them, which is the regime the deployed model is actually in. GATE is
    # then a second untouched set that verifies the choice.
    fit_idx, gate_idx = stratified_group_split(ds.groups, ds.y, args.holdout, args.seed)
    sub = stratified_group_split(ds.groups[fit_idx], ds.y[fit_idx],
                                 args.calibrate, args.seed + 1)
    train_idx, cal_idx = fit_idx[sub[0]], fit_idx[sub[1]]

    def part(idx):
        return (ds.y[idx], ds.groups[idx], ds.hour_index[idx], ds.onset_hour[idx],
                feats.iloc[idx].reset_index(drop=True))
    ytr, gtr, htr, otr, ftr = part(train_idx)
    yca, gca, hca, oca, fca = part(cal_idx)
    yte, gte, hte, ote, fte = part(gate_idx)
    npts = lambda g: len(set(g.tolist()))
    # Asserted, not assumed. Every number below is meaningless if one patient's hours
    # appear in two of these three sets, and a silent overlap looks exactly like a
    # good result.
    sets = [set(gtr.tolist()), set(gca.tolist()), set(gte.tolist())]
    for a, b, nm in ((0, 1, "train/calibrate"), (0, 2, "train/gate"), (1, 2, "calibrate/gate")):
        shared = sets[a] & sets[b]
        if shared:
            raise SystemExit(f"{nm} share {len(shared)} patients — the split leaks")
    print(f"Split: train {npts(gtr)} / calibrate {npts(gca)} / gate {npts(gte)} patients "
          f"(disjoint by patient)")

    print("\nFitting the detector on TRAIN...")
    m = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=False,
                                objective=objective, roster=list(roster)).fit(
        ftr, ytr, groups=gtr, hour_index=htr, onset_hour=otr)
    p_cal = m.predict_proba(fca)

    op = opmod.derive(yca, p_cal, gca, hca, oca,
                      objective=args.objective, max_false_alarm_rate=args.max_far,
                      far_margin=args.far_boot > 0, far_n_boot=max(args.far_boot, 1),
                      seed=args.seed, detector=detector_id)

    print("\n=== The detector's output distribution on CALIBRATE ===")
    print(f"  p95 {op.prob_p95:.4f} | p99 {op.prob_p99:.4f} | max {op.prob_max:.4f}")
    print(f"  -> the old hand-set {SHIPPED_HIGH} sat at the "
          f"{100*np.mean(p_cal < SHIPPED_HIGH):.1f}th percentile of this output")

    print("\n=== Threshold sweep (calibrate patients) ===")
    print(opmod.sweep_table(yca, p_cal, gca, hca, oca,
                           [0.01, 0.02, 0.025, 0.05, 0.075, 0.10, 0.12, 0.15, 0.20, 0.30]))

    print(f"\nDerived ({op.objective}"
          + (f", FAR<={op.max_false_alarm_rate}" if op.max_false_alarm_rate else "")
          + f"): HIGH={op.high:.4f}  MEDIUM={op.medium:.4f}")
    if not np.isnan(op.oof_false_alarm_upper):
        print(f"  calibrate FAR {op.oof_false_alarm_rate:.3f}, enforced against its "
              f"upper bound {op.oof_false_alarm_upper:.3f}")

    if args.oof_diagnostic:
        print(f"\n=== DIAGNOSTIC: the skew this split exists to remove ===")
        oof = np.zeros(len(ytr))
        for k, (a, b) in enumerate(stratified_group_folds(gtr, ytr, args.folds, args.seed)):
            mk = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=False,
                                         objective=objective, roster=list(roster)).fit(
                ftr.iloc[a], ytr[a], groups=gtr[a],
                hour_index=htr[a], onset_hour=otr[a])
            oof[b] = mk.predict_proba(ftr.iloc[b])
            print(f"  fold {k+1}/{args.folds}")
        print(f"  cross-validated p95 {np.quantile(oof, 0.95):.4f} vs "
              f"single-fit p95 {op.prob_p95:.4f} on calibrate")
        print(opmod.sweep_table(ytr, oof, gtr, htr, otr, [0.025, 0.05, 0.075, 0.12]))

    p_te = m.predict_proba(fte)

    print("\n=== GATE — held-out patients (never used to derive the threshold) ===")
    cap = args.max_far if args.objective == "constrained" else None

    def row(name, thr):
        u = metrics.normalized_utility(yte, (p_te >= thr).astype(int), gte, hte)
        s = metrics.early_warning(yte, p_te, gte, ote, hte, thr)
        far = s["false_alarm_rate"]
        flag = ""
        if cap is not None:
            flag = "  OVER CAP" if far > cap else "  within cap"
        print(f"{name:24s} thr={thr:.4f}  utility={u:+.4f}  detect={s['detection_rate']:.3f}  "
              f"FAR={far:.3f}  lead={s['median_lead_time_hours']:.1f}h{flag}")
        return u, far
    u_ship, _ = row("hand-set constant", SHIPPED_HIGH)
    incumbent = opmod.load()
    u_inc, far_inc_holdout = row("incumbent (on disk)", incumbent.high) if incumbent else (None, None)
    u_new, far_new = row(f"derived ({op.objective})", op.high)
    print(f"\nAUROC is UNCHANGED at {metrics.auroc(yte, p_te):.4f} — this is the same "
          f"detector, read at a different point.")
    print(f"Held-out utility vs hand-set constant: {u_new - u_ship:+.4f}"
          + (f"   vs incumbent: {u_new - u_inc:+.4f}" if u_inc is not None else ""))
    if incumbent is not None:
        print(f"\nThe incumbent's recorded evidence says FAR={incumbent.oof_false_alarm_rate:.3f} "
              f"on detector '{incumbent.detector}'; on THIS detector at the same threshold "
              f"held-out FAR is {far_inc_holdout:.3f}.")

    if args.save:
        # This point was read off THIS script's own fits. Once the manifest names the
        # shipped fit, installing it would pair a threshold with a model it was never
        # read off, which is the threshold/fit coupling the promoter exists to respect;
        # the promoter derives the point on the shipped fit and ships both together.
        with open(MANIFEST) as fh:
            shipped_fit = json.load(fh).get("fit_digest")
        if shipped_fit:
            print(f"\nREFUSED to save: the shipped detector is fit {shipped_fit} and this "
                  f"point comes from this script's own fits. Thresholds do not transfer "
                  f"between fits; re-run promote_frontier to derive one on the shipped fit.")
            return
        # THE refusal that was missing. The false-alarm cap was enforced only on the
        # scores used to CHOOSE the threshold, and never re-checked on patients held
        # back from that choice — so a point could be saved, and its cap advertised,
        # while the cap failed on unseen patients. It did: the derivation sample said
        # 0.105 where the gate said 0.170. A constraint verified only where it was
        # imposed is not a constraint.
        if cap is not None and far_new > cap:
            print(f"\nREFUSED to save: the derived point violates its own false-alarm cap on "
                  f"held-out patients (FAR {far_new:.3f} > {cap}). The cap held on the "
                  f"derivation sample and did not generalise; widen the margin "
                  f"(--far-boot) or raise --max-far deliberately.")
            return
        # Beating the 0.12 hand-set constant only says the derivation still beats the
        # number it originally replaced; it says nothing about regressing against what
        # is actually in production.
        if not (u_new > u_ship):
            print(f"\nREFUSED to save: derived point does not beat the hand-set constant on "
                  f"held-out utility ({u_new:+.4f} vs {u_ship:+.4f}).")
            return
        # ... but only compare utility against an incumbent that is itself FEASIBLE.
        # An incumbent that breaches the alarm cap scores well precisely because it is
        # spending a budget it does not have, so requiring a lawful point to out-score
        # it would lock the violation in permanently. Replacing an infeasible incumbent
        # with a feasible point is the fix, not a regression.
        if u_inc is not None:
            inc_feasible = cap is None or far_inc_holdout <= cap
            if not inc_feasible:
                print(f"\nNOTE: the incumbent is INFEASIBLE on held-out data "
                      f"(FAR {far_inc_holdout:.3f} > {cap}), so the utility comparison "
                      f"against it is waived — it out-scores a lawful point only by "
                      f"exceeding the alarm budget.")
            elif u_new < u_inc - 1e-9:
                print(f"\nREFUSED to save: derived point LOSES to the incumbent already on "
                      f"disk on held-out utility ({u_new:+.4f} vs {u_inc:+.4f}).")
                return
        opmod.save(op)
        print(f"\nSAVED -> {opmod.DEFAULT_PATH}\nAegisCore will load it at boot.")


if __name__ == "__main__":
    main()
