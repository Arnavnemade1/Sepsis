"""Bounded architecture experiment with separate selection and confirmation patients.

Run: GENESIS_THREADS=4 .venv/bin/python -m backend.ml.scripts.improve_icu
Writes a research model and complete result table. Never updates a deployed model
or reads the historical GENESIS lock. The confirmation patients are development
patients used by older studies, so this is not a new external/locked validation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split

from backend.ml import config
from backend.ml.eval import metrics
from backend.ml.genesis import harness as H
from backend.ml.genesis.learners import HStack, THREADS, fit_lgb, grouped_holdout, platt
from backend.ml.jsonio import json_safe
from backend.ml.models.icu_ensemble import FEATURE_VERSION, ICUEnsembleDetector, blend, trajectory_block

BASE = {"learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 800,
        "feature_fraction": 0.3, "bagging_fraction": 0.7, "lambda_l2": 10.0}
CANDIDATES = [
    {"name": "baseline", "kind": "lgbm", "dynamic": False, "params": BASE},
    {"name": "trajectory", "kind": "lgbm", "dynamic": True, "params": BASE},
    {"name": "randomized", "kind": "lgbm", "dynamic": True,
     "params": {**BASE, "extra_trees": True, "num_leaves": 63, "feature_fraction": 0.6,
                "min_data_in_leaf": 400, "learning_rate": 0.04}},
    {"name": "depthwise", "kind": "xgb", "dynamic": False,
     "params": {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
                "eta": 0.03, "max_depth": 5, "min_child_weight": 40, "lambda": 20,
                "subsample": 0.8, "colsample_bytree": 0.6, "max_bin": 128}},
]
SEED = 20261024


def write_json(path, obj):
    Path(path).write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False) + "\n")


def split_roles(c):
    """Fixed dev folds: 2/3/4 train, 0 select/calibrate, 1 confirm; -1 excluded."""
    train = np.flatnonzero(c.folds >= 2)
    tune = np.flatnonzero(c.folds == 0)
    selection, calibration = train_test_split(
        tune, test_size=0.35, random_state=SEED,
        stratify=c.plabel[tune] * 2 + c.raw.site[tune])
    roles = {"train": train, "selection": selection, "calibration": calibration,
             "confirmation": np.flatnonzero(c.folds == 1)}
    seen = set()
    for role, patients in roles.items():
        if not len(patients) or np.any(c.folds[patients] < 0) or seen.intersection(patients):
            raise ValueError(f"invalid patient partition: {role}")
        seen.update(patients.tolist())
    return {role: c.rows_of(patients) for role, patients in roles.items()}


def schema_hash(c, rows):
    pts = np.unique(c.seg[rows])
    identities = [f"{c.raw.site[p]}:{c.raw.pid[p]}" for p in pts]
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()


def weight_grid(n, denominator=4):
    return [np.array(w, dtype=float) / denominator
            for w in itertools.product(range(denominator + 1), repeat=n) if sum(w) == denominator]


def choose_blend(y, hour, predictions):
    """Select among a finite preregistered grid; reject clock-control regressions."""
    baseline_matched = H.hour_matched_auroc(y, predictions[:, 0], hour)
    table = []
    best, best_auc = np.eye(predictions.shape[1])[0], -np.inf
    # Baseline first, so a tie never adds complexity.
    candidates = [best] + weight_grid(predictions.shape[1])
    for w in candidates:
        p = blend(predictions, w)
        auc = H.auroc(y, p)
        matched = H.hour_matched_auroc(y, p, hour)
        admissible = bool(matched >= baseline_matched - 0.002)
        table.append({"weights": w.tolist(), "auroc": auc, "hour_matched": matched,
                      "admissible": admissible})
        if admissible and auc > best_auc + 1e-8:
            best, best_auc = w, auc
    return best, table


def predict_member(member, F, rows, names):
    detector = ICUEnsembleDetector([member], [1.0], names)
    out = np.empty(len(rows))
    for a in range(0, len(rows), 16384):
        out[a:a+16384] = detector.member_predict(F[rows[a:a+16384]], 0, THREADS)
    return out


def fit_member(candidate, c, F, train, rounds):
    matrix = F if candidate["dynamic"] else c.F
    if candidate["kind"] == "lgbm":
        fit = fit_lgb(matrix, train, c.y[train], c.seg[train], params=candidate["params"],
                      rounds=rounds, early_stop=100, seed=SEED, strat=c.plabel[c.seg[train]])
        return {"kind": "lgbm", "name": candidate["name"], "columns": matrix.shape[1],
                "model": fit.booster.model_to_string(), "trees": fit.best_iter, "calib": fit.calib}
    import xgboost as xgb
    fit_pos, stop_pos = grouped_holdout(c.seg[train], 0.1, SEED, c.plabel[c.seg[train]])
    # Quantile matrices keep the CPU histogram learner within laptop memory.
    ds = xgb.QuantileDMatrix(np.asarray(matrix[train[fit_pos]], dtype=np.float32),
                            label=c.y[train[fit_pos]], max_bin=128, nthread=THREADS)
    dv = xgb.QuantileDMatrix(np.asarray(matrix[train[stop_pos]], dtype=np.float32),
                            label=c.y[train[stop_pos]], ref=ds, max_bin=128, nthread=THREADS)
    model = xgb.train({**candidate["params"], "seed": SEED, "nthread": THREADS}, ds,
                      num_boost_round=rounds, evals=[(dv, "stop")], early_stopping_rounds=100,
                      verbose_eval=False)
    trees = model.best_iteration + 1
    margin = model.predict(dv, output_margin=True, iteration_range=(0, trees))
    calib = platt(margin, c.y[train[stop_pos]])
    return {"kind": "xgb", "name": candidate["name"], "columns": matrix.shape[1],
            "model": bytes(model.save_raw(raw_format="ubj")), "trees": trees, "calib": calib}


def summarize(c, rows, p, threshold):
    y, groups, hour = c.y[rows], c.seg[rows], c.hour[rows]
    out = H.score(c, p, rows, with_utility=False)
    # The frontier above is descriptive. These alarms use a calibration-only threshold.
    alarm = metrics.early_warning(y, p, groups, c.onset_row[rows], hour, threshold)
    out["fixed_alarm"] = {"threshold": float(threshold), "detection": alarm["detection_rate"],
                          "false_alarm": alarm["false_alarm_rate"]}
    out["brier"] = float(np.mean((p-y)**2))
    out["ece"] = metrics.expected_calibration_error(y, p)
    out["by_hospital"] = {}
    for site in np.unique(c.raw.site[c.seg[rows]]):
        mask = c.raw.site[c.seg[rows]] == site
        out["by_hospital"][str(int(site))] = {"auroc": H.auroc(y[mask], p[mask]),
            "auprc": metrics.auprc(y[mask], p[mask])}
    return out


def alarm_threshold(c, rows, p):
    # Complete never-septic calibration stays, never confirmation outcomes.
    pts, inv = np.unique(c.seg[rows], return_inverse=True)
    peak = np.full(len(pts), -np.inf)
    np.maximum.at(peak, inv, p)
    return metrics.budget_threshold(peak[c.plabel[pts] == 0], 0.15)


def bootstrap(y, candidate, baseline, groups, draws):
    a, b = H.Ranked(candidate), H.Ranked(baseline)
    _, inv = np.unique(groups, return_inverse=True)
    n = inv.max()+1
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(draws):
        w = np.bincount(rng.integers(0, n, n), minlength=n)[inv]
        aa, bb = a.auc(y, w), b.auc(y, w)
        values.append([aa, aa-bb])
    values = np.asarray(values)
    return {"draws": draws, "unit": "patient", "auroc_ci": np.percentile(values[:, 0], [2.5, 97.5]).tolist(),
            "delta": a.auc(y)-b.auc(y), "delta_ci": np.percentile(values[:, 1], [2.5, 97.5]).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(config.ARTIFACTS_DIR) / "icu_architecture")
    ap.add_argument("--rounds", type=int, default=1600)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    if args.rounds < 1 or args.boot < 100:
        ap.error("rounds must be positive and bootstrap draws must be at least 100")
    # Never silently overwrite a study or make a second confirmation look like a first.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    c = H.Cohort.load()
    roles = split_roles(c)
    plan = {"seed": SEED, "feature_version": FEATURE_VERSION, "candidates": CANDIDATES,
            "rounds": args.rounds, "early_stop": 100, "target": "original hourly SepsisLabel",
            "blend_grid_step": 0.25, "selection": "max AUROC, hour-matched regression <=0.002",
            "target_auroc": 0.9, "bootstrap_draws": args.boot,
            "gates": "delta CI lower >0, hour-matched >= baseline, AUPRC >= baseline, fixed detection >= baseline, fixed FAR <=0.15",
            "historical_locked_set": "excluded from all roles, never scored",
            "confirmation_caveat": "Previously used development patients; not a new external or pristine lock.",
            "roles": {k: {"patients": len(np.unique(c.seg[r])), "rows": len(r),
                            "patient_sha256": schema_hash(c, r)} for k, r in roles.items()}}
    source = [Path(__file__), Path(__file__).parents[1]/"models"/"icu_ensemble.py"]
    plan["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    write_json(args.output / "protocol.json", plan)
    print(json.dumps(plan["roles"], indent=2), flush=True)
    dynamic, extra_names = trajectory_block(c.raw)
    F = HStack(c.F, dynamic)
    names = c.names + extra_names
    print(f"Features: {len(c.names)} base + {len(extra_names)} observed dynamics", flush=True)
    members, selection, calibration, training = [], [], [], []
    for candidate in CANDIDATES:
        t = time.time()
        print(f"Fitting {candidate['name']}...", flush=True)
        member = fit_member(candidate, c, F, roles["train"], args.rounds)
        members.append(member)
        selection.append(predict_member(member, F, roles["selection"], names))
        calibration.append(predict_member(member, F, roles["calibration"], names))
        info = {"name": candidate["name"], "trees": member["trees"], "seconds": time.time()-t,
                "selection_auroc": H.auroc(c.y[roles["selection"]], selection[-1])}
        training.append(info)
        joblib.dump(members, args.output / "members.joblib", compress=3)
        write_json(args.output / "training.json", training)
        print(json.dumps(info), flush=True)
        gc.collect()
    ps, pc = np.column_stack(selection), np.column_stack(calibration)
    sr, cr = roles["selection"], roles["calibration"]
    weights, selection_table = choose_blend(c.y[sr], c.hour[sr], ps)
    raw_cal = blend(pc, weights)
    calibrator = platt(logit(np.clip(raw_cal, 1e-12, 1-1e-12)), c.y[cr])
    calibrated = expit(calibrator[0] * logit(np.clip(raw_cal, 1e-12, 1-1e-12)) + calibrator[1])
    threshold, base_threshold = alarm_threshold(c, cr, calibrated), alarm_threshold(c, cr, pc[:, 0])
    frozen = {"weights": weights.tolist(), "calibration": calibrator, "threshold": threshold,
              "baseline_threshold": base_threshold, "selection_table": selection_table}
    write_json(args.output / "selection.json", frozen)
    detector = ICUEnsembleDetector(members, weights.tolist(), names, calibrator,
        detector_id="research:icu-ensemble#" + plan["roles"]["train"]["patient_sha256"][:12],
        meta={"protocol": plan, "threshold": threshold})
    joblib.dump(detector, args.output / "candidate.joblib", compress=3)
    # No confirmation predictions have been computed before the design is frozen.
    rows = roles["confirmation"]
    print(f"Frozen weights {weights.tolist()}; evaluating {len(rows):,} confirmation hours", flush=True)
    confirm = np.column_stack([predict_member(m, F, rows, names) for m in members])
    p = expit(calibrator[0] * logit(np.clip(blend(confirm, weights), 1e-12, 1-1e-12)) + calibrator[1])
    base = confirm[:, 0]
    result = {"protocol": plan, "training": training, "weights": weights.tolist(),
              "candidate": summarize(c, rows, p, threshold),
              "baseline": summarize(c, rows, base, base_threshold),
              "members": {m["name"]: {"auroc": H.auroc(c.y[rows], confirm[:, j]),
                          "auprc": metrics.auprc(c.y[rows], confirm[:, j])} for j, m in enumerate(members)},
              "paired_bootstrap": bootstrap(c.y[rows], p, base, c.seg[rows], args.boot)}
    ca, ba = result["candidate"], result["baseline"]
    result["gates"] = {"auroc_gain": result["paired_bootstrap"]["delta_ci"][0] > 0,
        "hour_matched_not_worse": ca["auroc_hourmatched"] >= ba["auroc_hourmatched"],
        "auprc_not_worse": ca["auprc"] >= ba["auprc"],
        "fixed_detection_not_worse": ca["fixed_alarm"]["detection"] >= ba["fixed_alarm"]["detection"],
        "fixed_false_alarm_within_budget": ca["fixed_alarm"]["false_alarm"] <= 0.15}
    result["target_reached"] = bool(ca["auroc"] >= 0.9)
    result["research_gate_passed"] = all(result["gates"].values())
    result["production_promoted"] = False
    # Native serialization must reproduce the exact evaluated predictions.
    loaded = joblib.load(args.output / "candidate.joblib")
    served = loaded.predict_matrix(F[rows[:1024]], threads=THREADS)
    result["reload_max_error"] = float(np.max(np.abs(served-p[:1024])))
    if result["reload_max_error"] > 1e-7:
        raise RuntimeError("serialized detector does not reproduce the evaluated predictions")
    result["seconds"] = time.time()-started
    np.savez_compressed(args.output / "predictions.npz", rows=rows, candidate=p, baseline=base, members=confirm)
    write_json(args.output / "results.json", result)
    print(json.dumps(json_safe(result), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
