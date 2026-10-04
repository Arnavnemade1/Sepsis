"""Fit, judge and (if it passes) ship the all-variable detector for the ICU beds.

Protocol: ``artifacts/icu_detector_prereg.json``, registered before this ran.

  * fit ONCE: the GENESIS direct head (tuned LightGBM, challenge label, 510 features over
    all 40 variables) on exactly the shipped detector's 24,206 training patients;
  * judge ONCE on its 8,062 holdout against the DEPLOYED stack, every hour, with a paired
    patient bootstrap;
  * alarms: the candidate's operating point is derived on holdout half A, and both
    detectors are judged on half B, each at its own threshold;
  * ship iff every registered gate passes: ``artifacts/gbm_icu.joblib`` +
    ``artifacts/operating_point_icu.json``. The result is written either way
    (``artifacts/icu_detector.json``).

    python -m backend.ml.scripts.train_icu_detector            # judge only
    python -m backend.ml.scripts.train_icu_detector --ship     # write the artifact if it passes
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np
import pandas as pd

from backend.ml import config, fitid
from backend.ml import operating_point as opmod
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_split
from backend.ml.genesis import harness as H
from backend.ml.genesis import stack as S
from backend.ml.genesis import targets as T
from backend.ml.genesis.features import FeatureSpec
from backend.ml.models.allvar import AllVariableDetector
from backend.ml.scripts.promote_frontier import MANIFEST, _budget_delta, promotion_split

OUT = os.path.join(config.ARTIFACTS_DIR, "icu_detector.json")
MODEL = os.path.join(config.ARTIFACTS_DIR, "gbm_icu.joblib")
OP = os.path.join(config.ARTIFACTS_DIR, "operating_point_icu.json")
PARAMS = {"learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 800,
          "feature_fraction": 0.3, "bagging_fraction": 0.7, "lambda_l2": 10.0}
SEED = 99


def _boot_auc(y, a, b, groups, n_boot, seed=0):
    """Paired patient bootstrap of AUROC(a) - AUROC(b) via weighted O(n) AUROC."""
    Ra, Rb = H.Ranked(a), H.Ranked(b)
    _, inv = np.unique(groups, return_inverse=True)
    P = inv.max() + 1
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for j in range(n_boot):
        w = np.bincount(rng.integers(0, P, P), minlength=P)[inv].astype(np.float64)
        d[j] = Ra.auc(y, w) - Rb.auc(y, w)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(Ra.auc(y) - Rb.auc(y)), "lo": float(lo), "hi": float(hi), "draws": n_boot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()

    with open(MANIFEST) as fh:
        man = json.load(fh)
    ds, tr, te = promotion_split(man)
    digest = man["fit_digest"]
    train_ids = np.unique(ds.groups[tr])
    hold_ids = np.unique(ds.groups[te])
    print(f"promotion split: {len(train_ids):,} train / {len(hold_ids):,} holdout patients (fit {digest})")

    c = H.Cohort.load()
    pid_row = c.raw.pid[c.seg]
    train_rows = np.flatnonzero(np.isin(pid_row, train_ids))
    hold_rows = np.flatnonzero(np.isin(pid_row, hold_ids))
    assert len(np.unique(pid_row[train_rows])) == len(train_ids), "training patients missing from the raw store"
    assert len(hold_rows) == len(te), (len(hold_rows), len(te))
    print(f"GENESIS matrix {c.F.shape}; train rows {len(train_rows):,}, holdout rows {len(hold_rows):,}")

    # ---- the one fit --------------------------------------------------- #
    head = S.Head("icu_direct", T.challenge, params=dict(PARAMS))
    tgt, w = head.target(c)
    fit = S.fit_head(c, head, train_rows, tgt, w, seed=SEED)
    print(f"fitted: {fit.best_iter} trees, Platt {fit.calib}  [{time.time()-t0:.0f}s]", flush=True)
    p_rows = fit.predict(c.F, hold_rows)

    # ---- align to the promotion holdout's row order -------------------- #
    key_c = pd.MultiIndex.from_arrays([pid_row[hold_rows], c.hour[hold_rows]])
    key_d = pd.MultiIndex.from_arrays([ds.groups[te], ds.hour_index[te]])
    pos = pd.Series(np.arange(len(hold_rows)), index=key_c).reindex(key_d).to_numpy()
    assert not np.isnan(pos).any(), "holdout rows do not align between the two stores"
    p_icu = p_rows[pos.astype(int)]
    y, g, h, o = ds.y[te].astype(int), ds.groups[te], ds.hour_index[te], ds.onset_hour[te]
    assert np.array_equal(c.y[hold_rows][pos.astype(int)], y), "labels disagree between the stores"

    from backend.ml.models.stacked import StackedEnsembleDetector
    stack = StackedEnsembleDetector.load(os.path.join(config.ARTIFACTS_DIR, "gbm_wearable.joblib"))
    with np.errstate(all="ignore"):
        p_stack = stack.predict_proba(pd.DataFrame(ds.X[te], columns=ds.feature_names))

    # ---- discrimination ------------------------------------------------- #
    res = {}
    for name, p in (("candidate", p_icu), ("incumbent", p_stack)):
        res[name] = {"auroc": metrics.auroc(y, p), "hour_matched": H.hour_matched_auroc(y, p, h),
                     "auprc": metrics.auprc(y, p), "ece": metrics.expected_calibration_error(y, p)}
        print(f"  {name:9} AUROC {res[name]['auroc']:.4f}  hour-matched {res[name]['hour_matched']:.4f}  "
              f"AUPRC {res[name]['auprc']:.4f}  ECE {res[name]['ece']:.4f}")
    delta = _boot_auc(y, p_icu, p_stack, g, args.boot)
    print(f"  delta AUROC {delta['delta']:+.4f} [{delta['lo']:+.4f}, {delta['hi']:+.4f}]")

    # ---- alarms: derived on half A, judged on half B --------------------- #
    ha, hb = stratified_group_split(g, y, 0.5, int(man["seed"]) + 1)
    det_id = f"icu:lgbm-all40~platt#{digest}"
    op = None
    try:
        op = opmod.derive(y[ha], p_icu[ha], g[ha], h[ha], o[ha], objective="constrained",
                          max_false_alarm_rate=0.15, seed=int(man["seed"]), detector=det_id)
    except opmod.Infeasible as e:
        print(f"  no operating point under the 15% cap: {e}")
    alarms = {}
    if op is not None:
        for name, p, thr in (("candidate", p_icu, op.high), ("incumbent", p_stack, man["operating_point"]["high"])):
            ew = metrics.early_warning(y[hb], p[hb], g[hb], o[hb], h[hb], thr)
            alarms[name] = {"threshold": thr, "detection": ew["detection_rate"], "false_alarm": ew["false_alarm_rate"],
                            "median_lead_h": ew["median_lead_time_hours"],
                            "utility": metrics.normalized_utility(y[hb], (p[hb] >= thr).astype(int), g[hb], h[hb])}
            print(f"  {name:9} at its own point {thr:.4f}: detect {alarms[name]['detection']:.3f}  "
                  f"FAR {alarms[name]['false_alarm']:.3f}  utility {alarms[name]['utility']:+.4f}")
    bud = _budget_delta(y[hb], p_icu[hb], p_stack[hb], g[hb], o[hb], h[hb], 0.15, seed=int(man["seed"]))
    print(f"  detection inside a 15% budget (half B): delta {bud['delta']:+.4f} [{bud['lo']:+.4f}, {bud['hi']:+.4f}]")

    gates = {"auroc>=0.83": res["candidate"]["auroc"] >= 0.83, "delta_lo>0": delta["lo"] > 0,
             "budget15_not_worse": bud["hi"] >= 0, "operating_point_exists": op is not None}
    passed = all(gates.values())
    print(f"GATES: {gates} -> {'PASS' if passed else 'REFUSED'}")

    # ---- the served object must reproduce the fit, row for row ---------- #
    det = AllVariableDetector(model_str=fit.booster.model_to_string(), best_iter=int(fit.best_iter),
                              calib=tuple(float(x) for x in fit.calib), feature_names=list(c.names),
                              spec=dataclasses.asdict(FeatureSpec()), detector_id=det_id, fit_digest_=digest,
                              train_patients=int(len(train_ids)),
                              meta={"params": PARAMS, "seed": SEED, "trees": int(fit.best_iter)})
    sample = hold_rows[:20000]
    served = det.predict_matrix(np.asarray(c.F[sample]))
    gap = float(np.max(np.abs(served - fit.predict(c.F, sample))))
    print(f"served object vs fit on {len(sample):,} rows: max |diff| {gap:.2e}")
    assert gap < 1e-9, gap

    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "prereg": "artifacts/icu_detector_prereg.json",
           "detector": det_id, "incumbent": man["detector_id"], "train_patients": int(len(train_ids)),
           "judged_patients": int(len(hold_ids)), "rows": int(len(te)), "trees": int(fit.best_iter),
           "scores": res, "delta": delta, "alarms_half_b": alarms, "budget15_half_b": bud,
           "operating_point": None if op is None else {"high": op.high, "medium": op.medium},
           "gates": gates, "passed": passed, "shipped": bool(passed and args.ship)}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}")
    if passed and args.ship:
        import joblib
        joblib.dump(det, MODEL, compress=3)
        opmod.save(op, OP)
        print(f"SHIPPED {MODEL} ({os.path.getsize(MODEL)/1e6:.1f} MB) and {OP}")


if __name__ == "__main__":
    main()
