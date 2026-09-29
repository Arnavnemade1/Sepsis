"""Does calibrating without a staircase raise the detector's AUROC? — the registered run.

    python -m backend.ml.scripts.calibration_resolution --jobs 6

The shipped stack ends in isotonic regression, whose map is a staircase: every hour on
one step ties with every other, and the ranking the model made inside the step is
discarded. Centred isotonic regression keeps the calibration and drops the steps. The
design, arms, endpoints and bar are fixed in ``artifacts/calibration_resolution_prereg.json``,
committed before this ran; its hash is written into the result.

One stack fit per fold serves every arm: the arms share the meta-learner's raw score
and differ only in the final map, so any difference is the calibrator's alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from backend.ml import config
from backend.ml.eval import metrics
from backend.ml.eval.bench import Benchmark, BenchResult
from backend.ml.genesis.harness import Ranked, hour_matched_auroc
from backend.ml.models import calibration as C
from backend.ml.models import stacked as S

PREREG = os.path.join(config.ARTIFACTS_DIR, "calibration_resolution_prereg.json")
OOF = os.path.join(config.ARTIFACTS_DIR, "frontier", "calres_dev_oof.npz")
OUT = os.path.join(config.ARTIFACTS_DIR, "calibration_resolution.json")
ARMS = ("isotonic", "centred", "sigmoid", "raw")
N_BOOT = 1000


def fit_fold(bench: Benchmark, tr: np.ndarray, te: np.ndarray, jobs: int) -> Dict[str, np.ndarray]:
    from backend.ml.scripts.promote_frontier import _cpu_cap
    ds = bench.ds
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    seen: Dict[str, np.ndarray] = {}

    class Recorder(C.ProbabilityCalibrator):          # keeps the calibration slice
        def fit(self, raw_prob, y):
            seen["raw"], seen["y"] = np.asarray(raw_prob, float), np.asarray(y)
            return super().fit(raw_prob, y)

    params = {k: {"n_jobs": jobs} for k in ("et", "lgbm", "xgb")} if jobs else {}
    m = S.StackedEnsembleDetector(channels=list(config.WEARABLE_CHANNELS), use_hdc=False,
                                  params=params, objective="patient",
                                  roster=["hgb", "hgb2", "et", "lr", "xgb@utility"])
    S.ProbabilityCalibrator = Recorder
    try:
        with _cpu_cap(jobs):
            m.fit(feats.iloc[tr], ds.y[tr], groups=ds.groups[tr], hour_index=ds.hour_index[tr],
                  onset_hour=ds.onset_hour[tr])
            raw = m.meta_.predict_proba(m._base_matrix(feats.iloc[te]))[:, 1]
    finally:
        S.ProbabilityCalibrator = C.ProbabilityCalibrator
    platt = C.ProbabilityCalibrator(method="sigmoid").fit(seen["raw"], seen["y"])
    return {"isotonic": m.calibrator_.transform(raw),
            "centred": C.centred_transform(m.calibrator_.iso_, np.clip(raw, 1e-6, 1 - 1e-6)),
            "sigmoid": platt.transform(raw), "raw": raw}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    reg = json.load(open(PREREG))
    reg_hash = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()[:16]
    bench = Benchmark.load("dev", n_folds=3, seed=42)
    ds = bench.ds
    print(bench.describe(), flush=True)
    if a.reuse and os.path.exists(OOF):
        oof = dict(np.load(OOF))
    else:
        oof = {k: np.full(len(ds.y), np.nan) for k in ARMS}
        for k, (tr, te) in enumerate(bench.folds):
            t0 = time.time()
            for arm, v in fit_fold(bench, tr, te, a.jobs).items():
                oof[arm][te] = v
            print(f"  fold {k}: isotonic {Ranked(oof['isotonic'][te]).auc(ds.y[te]):.4f}  "
                  f"centred {Ranked(oof['centred'][te]).auc(ds.y[te]):.4f}  [{time.time() - t0:.0f}s]",
                  flush=True)
        os.makedirs(os.path.dirname(OOF), exist_ok=True)
        np.savez(OOF, **oof)

    y, g, h = ds.y, ds.groups, ds.hour_index
    ranked = {k: Ranked(v) for k, v in oof.items()}
    pts, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(0)
    boot = {k: np.empty(N_BOOT) for k in ARMS}
    for i in range(N_BOOT):
        w = rng.multinomial(len(pts), np.full(len(pts), 1.0 / len(pts)))[inv].astype(np.float64)
        for k in ARMS:
            boot[k][i] = ranked[k].auc(y, w)
    arms = {}
    for k in ARMS:
        arms[k] = {"auroc": ranked[k].auc(y), "hour_matched_auroc": float(hour_matched_auroc(y, oof[k], h)),
                   "auprc": float(average_precision_score(y, oof[k])),
                   "ece": float(metrics.expected_calibration_error(y, oof[k])),
                   "distinct_scores": int(len(np.unique(oof[k]))),
                   "per_fold_auroc": [Ranked(oof[k][te]).auc(y[te]) for _, te in bench.folds],
                   "per_fold_ece": [float(metrics.expected_calibration_error(y[te], oof[k][te]))
                                    for _, te in bench.folds]}
    comp = {}
    for k in ("centred", "sigmoid"):
        d = boot[k] - boot["isotonic"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        budgets = {}
        for far in (0.05, 0.10, 0.15, 0.20):
            cb = bench.compare_alarm_budget(BenchResult(k, [], oof[k], 0.0),
                                            BenchResult("isotonic", [], oof["isotonic"], 0.0),
                                            target_far=far, n_boot=N_BOOT)
            budgets[f"{int(far * 100)}%"] = {x: cb[x] for x in ("delta_detection", "ci_lo", "ci_hi")}
        comp[k] = {"delta_auroc": arms[k]["auroc"] - arms["isotonic"]["auroc"], "ci95": [float(lo), float(hi)],
                   "per_fold_delta": [p - q for p, q in zip(arms[k]["per_fold_auroc"], arms["isotonic"]["per_fold_auroc"])],
                   "delta_ece": arms[k]["ece"] - arms["isotonic"]["ece"],
                   "delta_hour_matched": arms[k]["hour_matched_auroc"] - arms["isotonic"]["hour_matched_auroc"],
                   "detection_at_budget": budgets}
    c = comp["centred"]
    verdict = {"auroc_gain": bool(c["ci95"][0] > 0),
               "calibration_kept": bool(c["delta_ece"] <= reg["safety"]["max_ece_increase"]),
               "no_budget_regression": not any(v["ci_hi"] < 0 for v in c["detection_at_budget"].values())}
    verdict["success"] = all(verdict.values())
    rep = {"preregistration": reg, "prereg_sha256_16": reg_hash, "arms": arms, "comparisons": comp,
           "verdict": verdict, "n_boot": N_BOOT}
    json.dump(rep, open(OUT, "w"), indent=1, default=float)
    for k in ARMS:
        r = arms[k]
        print(f"  {k:<9} AUROC {r['auroc']:.4f}  hour-matched {r['hour_matched_auroc']:.4f}  "
              f"AUPRC {r['auprc']:.4f}  ECE {r['ece']:.4f}  distinct {r['distinct_scores']:,}")
    for k, r in comp.items():
        print(f"  {k} - isotonic: AUROC {r['delta_auroc']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  "
              f"per-fold {[round(x, 4) for x in r['per_fold_delta']]}  ECE {r['delta_ece']:+.4f}  "
              + "  ".join(f"det@{b} {v['delta_detection']:+.3f}" for b, v in r["detection_at_budget"].items()))
    print("VERDICT:", "SUCCESS" if verdict["success"] else "NULL", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
