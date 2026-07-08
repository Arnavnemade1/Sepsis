"""Rigorous evaluation package — the honest, judge-facing results.

This is the evidence a reviewer actually interrogates. It runs the STRONGEST
generalisation test we can on CinC-2019 — train on one hospital, test on the OTHER
(set A = Beth Israel / BIDMC, set B = Emory) in both directions — and reports the
full clinical picture, NOT just AUROC:

  * ranking (AUROC / AUPRC) with PATIENT-LEVEL bootstrap 95% CIs,
  * probability calibration (Brier, ECE, reliability curve),
  * early warning (detection rate, median lead time, false-alarm rate),
  * the official CinC-2019 normalized utility,
  * decision-curve analysis (net clinical benefit vs treat-all / treat-none),
  * alarm burden (false alerts per 100 non-septic patient-hours),

for a model ladder (an ablation): a single calibrated GBM vs the production
leakage-safe stacked ensemble. Everything is written to
``artifacts/rigorous_eval.json`` for the results report.

    python -m backend.ml.scripts.rigorous_eval --feature-set wearable [--boot 500]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from backend.ml import config
from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.models.gbm_baseline import GBMDetector
from backend.ml.models.stacked import StackedEnsembleDetector
from backend.ml.scripts.train_and_compare import _feature_config


def _dataset(feature_set: str, set_name: str, limit):
    cfg = _feature_config(feature_set)
    key = f"{feature_set}_{set_name}_limit{limit or 'all'}"
    return build_dataset_cached(
        lambda: load_patients(sets=[set_name], limit=limit), cfg, key)


def _as_df(X, names):
    return X if hasattr(X, "columns") else pd.DataFrame(np.asarray(X), columns=names)


def _thin(x, y, n=160):
    """Subsample a monotone curve to ~n points to keep the artifact small."""
    x = np.asarray(x); y = np.asarray(y)
    if len(x) <= n:
        return x.tolist(), y.tolist()
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx].tolist(), y[idx].tolist()


def _fit_predict(kind, train, test):
    names = train.feature_names
    if kind == "gbm":
        m = GBMDetector(feature_names=names)
        m.fit(train.X, train.y, groups=train.groups)
        return m.predict_proba(test.X)
    # stacked ensemble (production detector: hgb+hgb2+et+lr, leakage-safe) — DataFrame in
    m = StackedEnsembleDetector(use_hdc=False)
    m.fit(_as_df(train.X, names), train.y, groups=train.groups)
    return m.predict_proba(_as_df(test.X, names))


def _score(y, p, groups, hour_index, onset_hour, boot):
    rep = metrics.evaluate(y, p, groups, hour_index, onset_hour)
    mp, of, cnt = metrics.reliability_curve(y, p, n_bins=10)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    fpr, tpr = _thin(fpr, tpr)
    rec2, prec2 = _thin(rec, prec)
    lo_au, hi_au = metrics.bootstrap_ci(y, p, groups, "auroc", n_boot=boot)
    lo_ap, hi_ap = metrics.bootstrap_ci(y, p, groups, "auprc", n_boot=boot)
    return {
        "auroc": rep.auroc, "auroc_ci": [lo_au, hi_au],
        "auprc": rep.auprc, "auprc_ci": [lo_ap, hi_ap],
        "brier": rep.brier, "ece": rep.ece, "prevalence": rep.prevalence,
        "normalized_utility": rep.normalized_utility,
        "operating_points": rep.operating_points,
        "early_warning": rep.early_warning,
        "reliability": {"mean_pred": mp.tolist(), "obs_freq": of.tolist(), "count": cnt.tolist()},
        "roc": {"fpr": fpr, "tpr": tpr},
        "pr": {"recall": rec2, "precision": prec2},
        "decision_curve": metrics.decision_curve(y, p),
        "alarm_burden": metrics.alarm_burden(y, p, groups),
    }


def _direction(train, test, boot, models):
    out = {}
    for kind in models:
        t0 = time.time()
        p = _fit_predict(kind, train, test)
        out[kind] = _score(test.y, p, test.groups, test.hour_index, test.onset_hour, boot)
        print(f"    {kind:8s} AUROC {out[kind]['auroc']:.3f} "
              f"[{out[kind]['auroc_ci'][0]:.3f}-{out[kind]['auroc_ci'][1]:.3f}] "
              f"· util {out[kind]['normalized_utility']:.3f} · {time.time()-t0:.0f}s")
    return out


def _n_patients(ds):
    return int(len(set(ds.groups.tolist())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="wearable")
    ap.add_argument("--limit-per-set", type=int, default=None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--models", default="gbm,stacked")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"[rigorous_eval] feature_set={args.feature_set} boot={args.boot} models={models}")
    A = _dataset(args.feature_set, "training_setA", args.limit_per_set)
    B = _dataset(args.feature_set, "training_setB", args.limit_per_set)
    print(f"  set A (BIDMC): {len(A.y)} rows / {_n_patients(A)} patients, prev {A.y.mean():.4f}")
    print(f"  set B (Emory): {len(B.y)} rows / {_n_patients(B)} patients, prev {B.y.mean():.4f}")

    print("\n=== A (BIDMC) -> B (Emory) ===")
    ab = _direction(A, B, args.boot, models)
    print("=== B (Emory) -> A (BIDMC) ===")
    ba = _direction(B, A, args.boot, models)

    out = {
        "meta": {
            "feature_set": args.feature_set, "boot": args.boot, "models": models,
            "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "setA": {"rows": int(len(A.y)), "patients": _n_patients(A), "prevalence": float(A.y.mean())},
            "setB": {"rows": int(len(B.y)), "patients": _n_patients(B), "prevalence": float(B.y.mean())},
            "protocol": "cross-hospital external validation (train one site, test the other)",
        },
        "directions": {
            "A_to_B": {"train": "BIDMC (set A)", "test": "Emory (set B)", "models": ab},
            "B_to_A": {"train": "Emory (set B)", "test": "BIDMC (set A)", "models": ba},
        },
    }
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(config.ARTIFACTS_DIR, "rigorous_eval.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    # headline
    hb = [ab, ba]
    for kind in models:
        ms = np.mean([d[kind]["auroc"] for d in hb])
        print(f"\n  cross-hospital mean AUROC · {kind}: {ms:.4f}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
