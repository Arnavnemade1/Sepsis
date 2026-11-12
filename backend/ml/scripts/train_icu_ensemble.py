"""Fit, judge and stage the frozen ICU ensemble for the ICU replay beds.

Protocol: ``artifacts/icu_ensemble_prereg.json``, registered before this ran.

  * fit ONCE: the three members the October study kept (``improve_icu``), with its frozen
    weights, on exactly the shipped fits' 24,206 training patients;
  * judge ONCE on the 8,062 holdout against the DEPLOYED all-variable detector
    (``artifacts/gbm_icu.joblib``, scored not refitted), paired patient bootstrap;
  * alarms: the candidate's operating point is derived on holdout half A, and both
    detectors are judged on half B, each at its own threshold;
  * the result is written either way (``artifacts/icu_ensemble.json``). The fitted
    detector and its operating point are staged beside it; ``--promote`` copies a staged
    PASS over the served artifact and refuses anything else.

    make icu-ensemble            # fit + judge + stage
    make icu-ensemble-promote    # serve the staged detector, only if it passed
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from backend.ml import config
from backend.ml import operating_point as opmod
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_split
from backend.ml.genesis import harness as H
from backend.ml.genesis.learners import HStack, THREADS, grouped_holdout, platt
from backend.ml.models.icu_ensemble import FEATURE_VERSION, ICUEnsembleDetector, blend, trajectory_block
from backend.ml.scripts.improve_icu import CANDIDATES, SEED, fit_member, predict_member
from backend.ml.scripts.promote_frontier import MANIFEST, _budget_delta, promotion_split
from backend.ml.scripts.train_icu_detector import MODEL, OP, _boot_auc

A = config.ARTIFACTS_DIR
OUT = os.path.join(A, "icu_ensemble.json")
STAGED_MODEL = os.path.join(A, "icu_ensemble_staged.joblib")
STAGED_OP = os.path.join(A, "icu_ensemble_staged_op.json")
MEMBERS = ("trajectory", "randomized", "depthwise")
WEIGHTS = (0.5, 0.25, 0.25)          # artifacts/icu_architecture/selection.json, frozen
ROUNDS = 1600


def _squash(raw, calib):
    return expit(calib[0] * logit(np.clip(raw, 1e-12, 1 - 1e-12)) + calib[1])


def promote() -> None:
    """Serve the staged detector. Only a staged PASS whose identity matches is copied."""
    import joblib
    with open(OUT) as fh:
        res = json.load(fh)
    if not res.get("passed"):
        raise SystemExit("the registered test did not pass: nothing is promoted")
    det, op = joblib.load(STAGED_MODEL), opmod.load(STAGED_OP)
    if not (det.detector_id == op.detector == res["detector"]):
        raise SystemExit("the staged model, its operating point and the result disagree on identity")
    shutil.copyfile(STAGED_MODEL, MODEL)
    shutil.copyfile(STAGED_OP, OP)
    res["shipped"] = True
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"SHIPPED {MODEL} ({os.path.getsize(MODEL)/1e6:.1f} MB) and {OP}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    if args.promote:
        return promote()
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists: the holdout is scored once for this candidate")
    t0 = time.time()

    with open(MANIFEST) as fh:
        man = json.load(fh)
    ds, tr, te = promotion_split(man)
    digest = man["fit_digest"]
    train_ids, hold_ids = np.unique(ds.groups[tr]), np.unique(ds.groups[te])
    print(f"promotion split: {len(train_ids):,} train / {len(hold_ids):,} holdout patients (fit {digest})")

    c = H.Cohort.load()
    pid_row = c.raw.pid[c.seg]
    train_rows = np.flatnonzero(np.isin(pid_row, train_ids))
    hold_rows = np.flatnonzero(np.isin(pid_row, hold_ids))
    assert len(np.unique(pid_row[train_rows])) == len(train_ids), "training patients missing from the raw store"
    assert len(hold_rows) == len(te), (len(hold_rows), len(te))
    assert not np.intersect1d(c.seg[train_rows], c.seg[hold_rows]).size

    dynamic, extra = trajectory_block(c.raw)
    F, names = HStack(c.F, dynamic), c.names + extra
    print(f"features: {len(c.names)} base + {len(extra)} observed dynamics; "
          f"train rows {len(train_rows):,}, holdout rows {len(hold_rows):,}", flush=True)

    # ---- the one fit of each member ------------------------------------- #
    by_name = {cand["name"]: cand for cand in CANDIDATES}
    members, training = [], []
    for name in MEMBERS:
        t = time.time()
        members.append(fit_member(by_name[name], c, F, train_rows, ROUNDS))
        training.append({"name": name, "trees": int(members[-1]["trees"]), "seconds": time.time() - t})
        print(json.dumps(training[-1]), flush=True)
    w = np.asarray(WEIGHTS)

    # ---- blend calibration on the members' shared early-stopping slice --- #
    _, stop_pos = grouped_holdout(c.seg[train_rows], 0.1, SEED, c.plabel[c.seg[train_rows]])
    stop_rows = train_rows[stop_pos]
    raw_stop = blend(np.column_stack([predict_member(m, F, stop_rows, names) for m in members]), w)
    calib = tuple(float(x) for x in platt(logit(np.clip(raw_stop, 1e-12, 1 - 1e-12)), c.y[stop_rows]))

    det_id = f"icu:ens-all40-dyn~platt#{digest}"
    det = ICUEnsembleDetector(members, list(WEIGHTS), names, calib, detector_id=det_id,
                              meta={"fit_digest": digest, "train_patients": int(len(train_ids)),
                                    "feature_version": FEATURE_VERSION, "seed": SEED,
                                    "prereg": "artifacts/icu_ensemble_prereg.json"})

    # ---- the one look at the holdout ------------------------------------ #
    import joblib
    member_hold = np.column_stack([predict_member(m, F, hold_rows, names) for m in members])
    p_rows = _squash(blend(member_hold, w), calib)
    incumbent = joblib.load(MODEL)
    inc_rows = np.concatenate([incumbent.predict_matrix(np.asarray(c.F[hold_rows[a:a + 16384]]))
                               for a in range(0, len(hold_rows), 16384)])

    key_c = pd.MultiIndex.from_arrays([pid_row[hold_rows], c.hour[hold_rows]])
    key_d = pd.MultiIndex.from_arrays([ds.groups[te], ds.hour_index[te]])
    pos = pd.Series(np.arange(len(hold_rows)), index=key_c).reindex(key_d).to_numpy()
    assert not np.isnan(pos).any(), "holdout rows do not align between the two stores"
    pos = pos.astype(int)
    p_new, p_inc = p_rows[pos], inc_rows[pos]
    y, g, h, o = ds.y[te].astype(int), ds.groups[te], ds.hour_index[te], ds.onset_hour[te]
    assert np.array_equal(c.y[hold_rows][pos], y), "labels disagree between the stores"

    res = {}
    for name, p in (("candidate", p_new), ("incumbent", p_inc)):
        res[name] = {"auroc": metrics.auroc(y, p), "hour_matched": H.hour_matched_auroc(y, p, h),
                     "auprc": metrics.auprc(y, p), "ece": metrics.expected_calibration_error(y, p)}
        print(f"  {name:9} AUROC {res[name]['auroc']:.5f}  hour-matched {res[name]['hour_matched']:.5f}  "
              f"AUPRC {res[name]['auprc']:.5f}  ECE {res[name]['ece']:.5f}")
    member_scores = {n: metrics.auroc(y, member_hold[pos, j]) for j, n in enumerate(MEMBERS)}
    delta = _boot_auc(y, p_new, p_inc, g, args.boot)
    print(f"  delta AUROC {delta['delta']:+.5f} [{delta['lo']:+.5f}, {delta['hi']:+.5f}]")

    # ---- alarms: derived on half A, judged on half B --------------------- #
    ha, hb = stratified_group_split(g, y, 0.5, int(man["seed"]) + 1)
    op = None
    try:
        op = opmod.derive(y[ha], p_new[ha], g[ha], h[ha], o[ha], objective="constrained",
                          max_false_alarm_rate=0.15, seed=int(man["seed"]), detector=det_id)
    except opmod.Infeasible as e:
        print(f"  no operating point under the 15% cap: {e}")
    inc_op = opmod.load(OP)
    assert inc_op.detector == incumbent.detector_id, "the served operating point is not the incumbent's"
    alarms = {}
    if op is not None:
        for name, p, thr in (("candidate", p_new, op.high), ("incumbent", p_inc, inc_op.high)):
            ew = metrics.early_warning(y[hb], p[hb], g[hb], o[hb], h[hb], thr)
            alarms[name] = {"threshold": thr, "detection": ew["detection_rate"], "false_alarm": ew["false_alarm_rate"],
                            "median_lead_h": ew["median_lead_time_hours"],
                            "utility": metrics.normalized_utility(y[hb], (p[hb] >= thr).astype(int), g[hb], h[hb])}
            print(f"  {name:9} at its own point {thr:.4f}: detect {alarms[name]['detection']:.3f}  "
                  f"FAR {alarms[name]['false_alarm']:.3f}  utility {alarms[name]['utility']:+.4f}")
    bud = _budget_delta(y[hb], p_new[hb], p_inc[hb], g[hb], o[hb], h[hb], 0.15, seed=int(man["seed"]))
    print(f"  detection inside a 15% budget (half B): delta {bud['delta']:+.4f} [{bud['lo']:+.4f}, {bud['hi']:+.4f}]")

    gates = {"delta_lo>0": delta["lo"] > 0,
             "hour_matched_not_worse": res["candidate"]["hour_matched"] >= res["incumbent"]["hour_matched"],
             "auprc_not_worse": res["candidate"]["auprc"] >= res["incumbent"]["auprc"],
             "budget15_not_worse": bud["hi"] >= 0, "operating_point_exists": op is not None}
    passed = all(gates.values())
    print(f"GATES: {gates} -> {'PASS' if passed else 'REFUSED'}")

    # ---- the staged object must reproduce the evaluated scores ----------- #
    joblib.dump(det, STAGED_MODEL, compress=3)
    if op is not None:
        opmod.save(op, STAGED_OP)
    sample = hold_rows[:20000]
    gap = float(np.max(np.abs(joblib.load(STAGED_MODEL).predict_matrix(F[sample]) - p_rows[:20000])))
    print(f"staged object vs evaluated scores on {len(sample):,} rows: max |diff| {gap:.2e}")
    assert gap < 1e-7, gap

    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "prereg": "artifacts/icu_ensemble_prereg.json",
           "detector": det_id, "incumbent": incumbent.detector_id, "train_patients": int(len(train_ids)),
           "judged_patients": int(len(hold_ids)), "rows": int(len(te)), "weights": dict(zip(MEMBERS, WEIGHTS)),
           "training": training, "calibration": calib, "scores": res, "member_auroc": member_scores,
           "delta": delta, "alarms_half_b": alarms, "budget15_half_b": bud,
           "operating_point": None if op is None else {"high": op.high, "medium": op.medium},
           "gates": gates, "passed": passed, "shipped": False, "reload_max_error": gap,
           "threads": THREADS, "seconds": time.time() - t0}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
