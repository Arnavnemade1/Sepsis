"""Train and score our model under YAIB's protocol; compare it with YAIB's baselines.

THE METRIC, AS YAIB COMPUTES IT
-------------------------------
For each of the cv_repetitions x cv_folds runs, a model is fitted on that run's train
stays (validation stays only for early stopping), and ``roc_auc_score`` /
``average_precision_score`` are computed over every test-stay hour whose label is finite
(``MLWrapper.test_step`` -> ``log_metrics``; ``get_data_and_labels`` drops NaN labels).
The reported number is the mean and standard deviation of those per-run values. That
is what ``run`` returns under ``"yaib_metric"``.

PAIRED COMPARISON
-----------------
YAIB saves each run's test predictions as ``pred_indicators.csv``
(stay id, hour, label, p0, p1). ``compare`` joins ours to theirs on (run, stay, hour),
refuses if the rows or labels differ, and bootstraps the AUROC difference over test
STAYS within each repetition (the inner folds of a repetition share one test set, so
their difference is averaged before resampling). A claim of "better than the baseline"
needs that interval to exclude zero; the mean of 25 numbers with overlapping error bars
is not one.

OUTPUT
------
Aggregates only: per-run AUROC / AUPRC, their mean and std, the split fingerprints, the
comparison. Never a row.
"""
from __future__ import annotations

import glob
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from backend.ml.yaib import features as FE
from backend.ml.yaib import splits as SP
from backend.ml.yaib.cohort import GROUP, Cohort


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 200
    feature_fraction: float = 0.5
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    max_rounds: int = 3000
    early_stopping: int = 100
    threads: int = 2
    seed: int = 0


def _fit_predict(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray,
                 cfg: TrainConfig) -> Dict[str, Any]:
    import lightgbm as lgb
    pos = max(float(y[tr].sum()), 1.0)
    neg = max(float(len(tr) - y[tr].sum()), 1.0)
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        # never ask for more rows per leaf than a small cohort can give
        "min_child_samples": int(min(cfg.min_child_samples, max(5, len(tr) // 50))),
        "feature_fraction": cfg.feature_fraction, "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": 1, "lambda_l2": cfg.lambda_l2,
        # YAIB trains with weight="balanced"; the same class balance here
        "scale_pos_weight": neg / pos,
        "num_threads": cfg.threads, "seed": cfg.seed, "deterministic": True, "verbose": -1,
    }
    dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
    dva = lgb.Dataset(X[va], y[va], reference=dtr, free_raw_data=True)
    booster = lgb.train(params, dtr, cfg.max_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])
    return {"p": booster.predict(X[te], num_iteration=booster.best_iteration),
            "rounds": int(booster.best_iteration)}


def yaib_summary(name: str, v: np.ndarray) -> Dict[str, Any]:
    """Summarise per-run values as ``run_utils.aggregate_results`` does: the mean, the
    field YAIB labels "std" (population SD / sqrt(n): a standard error, not an SD), and a
    t interval. The plain SD over runs is kept too, under its own name."""
    from scipy import stats
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return {f"{name}_mean": float(v.mean()) if n else math.nan}
    m, sd = float(v.mean()), float(v.std())
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=stats.sem(v))
    return {f"{name}_mean": m, f"{name}_yaib_std_field": sd / math.sqrt(n), f"{name}_sd_over_runs": sd,
            f"{name}_ci95": [float(lo), float(hi)]}


def _metric(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auroc": math.nan, "auprc": math.nan}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


def run(cohort: Cohort, cfg: TrainConfig = TrainConfig(), cv_repetitions: int = 5, cv_folds: int = 5,
        seed: int = SP.YAIB_CLI_SEED, runs: Optional[Sequence[tuple]] = None,
        predictions_dir: Optional[str] = None) -> Dict[str, Any]:
    """Every (repetition, fold) of YAIB's protocol. ``runs`` limits it to some pairs.
    ``predictions_dir`` writes each run's test predictions (rows: keep them in the lake)."""
    t0 = time.time()
    X, names = FE.build(cohort)
    y_all = cohort.label
    stay = cohort.dyn[GROUP].to_numpy()
    hour = cohort.dyn["hour"].to_numpy()
    labels = cohort.stay_labels()
    per_run: List[Dict[str, Any]] = []
    for sp in SP.all_splits(labels, cv_repetitions, cv_folds, seed):
        if runs is not None and (sp.repetition, sp.fold) not in runs:
            continue
        lab = np.isfinite(y_all)
        tr = np.flatnonzero(np.isin(stay, sp.train) & lab)
        va = np.flatnonzero(np.isin(stay, sp.val) & lab)
        te = np.flatnonzero(np.isin(stay, sp.test) & lab)
        y = np.nan_to_num(y_all, nan=0.0)
        fit = _fit_predict(X, y, tr, va, te, cfg)
        m = _metric(y[te], fit["p"])
        per_run.append({"repetition": sp.repetition, "fold": sp.fold, "split": sp.fingerprint(),
                        "rounds": fit["rounds"], "test_rows": int(len(te)),
                        "test_stays": int(len(sp.test)), **m})
        if predictions_dir:
            os.makedirs(predictions_dir, exist_ok=True)
            pd.DataFrame({"stay_id": stay[te], "hour": hour[te], "label": y[te], "p": fit["p"]}).to_parquet(
                os.path.join(predictions_dir, f"ours_r{sp.repetition}_f{sp.fold}.parquet"), index=False)
    au = np.array([r["auroc"] for r in per_run], float)
    pr = np.array([r["auprc"] for r in per_run], float)
    return {
        "schema": "yaib_bench/1",
        "cohort": {"name": cohort.name, "stays": int(len(labels)), "septic_stays": int(labels.sum()),
                   "rows": int(len(stay)), "labelled_rows": int(np.isfinite(y_all).sum())},
        "protocol": {"cv_repetitions": cv_repetitions, "cv_folds": cv_folds, "seed": seed,
                     "split_source": "YAIB make_single_split_polars, reproduced (backend/ml/yaib/splits.py)"},
        "model": {"learner": "LightGBM", **asdict(cfg), "n_features": len(names)},
        "runs": per_run,
        "yaib_metric": {**yaib_summary("auroc", au), **yaib_summary("auprc", pr),
                        "n_runs": int(np.isfinite(au).sum())},
        "seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# paired comparison with YAIB's own saved predictions
# --------------------------------------------------------------------------- #
def read_yaib_predictions(path: str) -> pd.DataFrame:
    """One run's ``pred_indicators.csv``: stay id, hour, label, p(class 0), p(class 1)."""
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    if a.shape[1] != 5:
        raise ValueError(f"{path}: expected 5 columns (id, hour, label, p0, p1), got {a.shape[1]}")
    return pd.DataFrame({"stay_id": a[:, 0].astype(np.int64), "hour": a[:, 1], "label": a[:, 2], "p": a[:, 4]})


#: YAIB's polars loader writes the time column as it is held ("already in hours?" in its
#: source), so the saved unit depends on the cohort's dtype. Seconds through nanoseconds
#: are tried; a unit is accepted only if it makes the two row sets identical.
_UNITS = (1.0, 3600.0, 3.6e6, 3.6e9, 3.6e12)


def _hours_like(theirs: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    want = set(zip(ours["stay_id"].to_numpy(), np.round(ours["hour"].to_numpy(), 6)))
    for u in _UNITS:
        h = np.round(theirs["hour"].to_numpy() / u, 6)
        if set(zip(theirs["stay_id"].to_numpy(), h)) == want:
            return theirs.assign(hour=h)
    return theirs


def _join(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Pair our rows with the baseline's.

    YAIB's saved labels and predictions are in (stay, hour) order: its loader sorts the
    feature rows, and the labels come out in the same order. The stay/hour ids written
    beside them are NOT in that order (the polars loader takes them before the sort;
    seen on YAIB af9a5b3, eICU demo: 324 of 7,970 rows carried another hour's label when
    joined on the written ids). So rows are paired by sorted position, and that is
    accepted only if the two id sets are identical AND every label agrees position by
    position. Anything else is refused.
    """
    ours = ours.assign(hour=np.round(ours["hour"].to_numpy(), 6))
    theirs = _hours_like(theirs, ours)
    a = ours.sort_values(["stay_id", "hour"], kind="mergesort").reset_index(drop=True)
    ka = pd.MultiIndex.from_arrays([a["stay_id"], a["hour"]])
    kb = pd.MultiIndex.from_arrays([theirs["stay_id"], theirs["hour"]])
    if len(a) != len(theirs) or not ka.sort_values().equals(kb.sort_values()):
        only_a, only_b = len(ka.difference(kb)), len(kb.difference(ka))
        raise ValueError(f"rows differ: {only_a} only ours, {only_b} only the baseline's "
                         f"({len(a)} vs {len(theirs)} rows)")
    if not np.array_equal(a["label"].to_numpy(), theirs["label"].to_numpy()):
        raise ValueError("labels differ position by position: the baseline's rows are not in "
                         "(stay, hour) order, so they cannot be paired")
    return pd.DataFrame({"stay_id": a["stay_id"], "hour": a["hour"], "label_ours": a["label"],
                         "p_ours": a["p"], "p_base": theirs["p"].to_numpy()})


def compare(ours: Dict[tuple, pd.DataFrame], theirs: Dict[tuple, pd.DataFrame], n_boot: int = 1000,
            seed: int = 0) -> Dict[str, Any]:
    """Paired AUROC difference (ours - baseline), bootstrapping test stays per repetition."""
    keys = sorted(set(ours) & set(theirs))
    if not keys or set(ours) != set(theirs):
        raise ValueError(f"runs differ: ours {sorted(ours)}, baseline {sorted(theirs)}")
    by_rep: Dict[int, List[pd.DataFrame]] = {}
    for k in keys:
        by_rep.setdefault(k[0], []).append(_join(ours[k], theirs[k]))
    rng = np.random.default_rng(seed)
    point, draws = [], np.zeros(n_boot)
    for r, frames in sorted(by_rep.items()):
        stays = np.unique(frames[0]["stay_id"].to_numpy())
        codes = [np.searchsorted(stays, f["stay_id"].to_numpy()) for f in frames]
        point.append(np.mean([roc_auc_score(f["label_ours"], f["p_ours"]) - roc_auc_score(f["label_ours"], f["p_base"])
                              for f in frames]))
        for b in range(n_boot):
            m = np.bincount(rng.integers(0, len(stays), len(stays)), minlength=len(stays)).astype(float)
            d = []
            for f, c in zip(frames, codes):
                w = m[c]
                yy = f["label_ours"].to_numpy()
                if w.sum() == 0 or len(np.unique(yy[w > 0])) < 2:
                    continue
                d.append(roc_auc_score(yy, f["p_ours"], sample_weight=w)
                         - roc_auc_score(yy, f["p_base"], sample_weight=w))
            draws[b] += np.mean(d) if d else np.nan
    draws /= len(by_rep)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    delta = float(np.mean(point))
    return {"auroc_delta": delta, "ci_low": float(lo), "ci_high": float(hi), "repetitions": len(by_rep),
            "runs": len(keys), "better": bool(lo > 0), "worse": bool(hi < 0)}


def load_runs(pattern_ours: str, yaib_root: str) -> tuple:
    """Our ``ours_r{r}_f{f}.parquet`` files and YAIB's ``repetition_{r}/fold_{f}/pred_indicators.csv``."""
    ours, theirs = {}, {}
    for f in glob.glob(pattern_ours):
        b = os.path.basename(f)[len("ours_"):-len(".parquet")]
        r, k = b.split("_")
        ours[(int(r[1:]), int(k[1:]))] = pd.read_parquet(f)
    for f in glob.glob(os.path.join(yaib_root, "**", "pred_indicators.csv"), recursive=True):
        parts = os.path.normpath(f).split(os.sep)
        rep = next((p for p in parts if p.startswith("repetition_")), None)
        fold = next((p for p in parts if p.startswith("fold_")), None)
        if rep and fold:
            theirs[(int(rep.split("_")[1]), int(fold.split("_")[1]))] = read_yaib_predictions(f)
    return ours, theirs


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    from backend.ml.jsonio import json_safe
    from backend.ml.yaib.cohort import load

    ap = argparse.ArgumentParser(description="Our model under YAIB's protocol (aggregates only).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="5x5 runs on a YAIB cohort directory")
    r.add_argument("cohort")
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=SP.YAIB_CLI_SEED, help="the seed the YAIB baselines ran with")
    r.add_argument("--repetitions", type=int, default=5)
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--predictions", help="directory for per-run test predictions (inside the lake)")
    c = sub.add_parser("compare", help="paired AUROC vs YAIB's saved baseline predictions")
    c.add_argument("--ours", required=True, help="directory of ours_r*_f*.parquet")
    c.add_argument("--yaib", required=True, help="YAIB log directory of one baseline model")
    c.add_argument("--out", required=True)
    c.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        res = run(load(a.cohort), TrainConfig(threads=a.threads), a.repetitions, a.folds, a.seed,
                  predictions_dir=a.predictions)
        m = res["yaib_metric"]
        print(f"AUROC {m['auroc_mean']:.4f} ± {m['auroc_yaib_std_field']:.4f}  AUPRC {m['auprc_mean']:.4f} "
              f"± {m['auprc_yaib_std_field']:.4f}  over {m['n_runs']} runs (± as YAIB reports it)")
    else:
        ours, theirs = load_runs(os.path.join(a.ours, "ours_r*_f*.parquet"), a.yaib)
        res = compare(ours, theirs, a.n_boot)
        print(f"AUROC ours - baseline {res['auroc_delta']:+.4f} [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]")
    with open(a.out, "w") as fh:
        json.dump(json_safe(res), fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
