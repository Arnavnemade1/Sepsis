"""Screen YAIB-model candidates on an OPEN cohort before any credentialed run.

The eICU benchmark's 25 runs are scored and must not be re-chosen on. A new candidate
therefore has to earn a place in a registered eICU sweep somewhere else first. This is
that somewhere else: PhysioNet/CinC 2019 (40,336 ICU stays from two hospitals, open
access, CC BY 4.0), converted to YAIB's cohort format, split by YAIB's own protocol, and
fitted through exactly the streamed LightGBM path the worker uses (``bench._fit_predict``).

    python -m backend.ml.scripts.yaib_proxy_screen convert --cinc data/cinc2019 --out <cohort dir>
    python -m backend.ml.scripts.yaib_proxy_screen screen <cohort dir> --runs 0:0,0:1,0:2 \\
        --candidates rich,temporal --work <private dir> --out screen.json

Because the proxy is not the benchmark, its test folds may be read: candidates are
compared on them, paired (bootstrap over test stays, as ``bench.compare``), with the
AUROC of every run AND its hour-matched AUROC. On CinC the ICU clock alone scores about
0.64, because never-septic records stop near 60 h (docs/GENESIS.md); hour-matched AUROC
compares only rows at the same hour, so the clock is worth exactly 0.5 there, and a
candidate that only learned stay length more precisely gains nothing under it.

What carries over to eICU is the ranking of candidates, not the numbers: CinC has 31 of
YAIB's 48 variables (no urine output, PaO2, sodium, albumin, INR, CRP or differential).
Outputs are aggregates only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.ml.yaib import bench as B
from backend.ml.yaib import extension as EX
from backend.ml.yaib import features as FE
from backend.ml.yaib import splits as SP
from backend.ml.yaib.cohort import DYNAMIC, GROUP, STATIC, load

#: CinC 2019 column -> YAIB variable. CinC's FiO2 is a fraction, which features.py accepts.
#: TroponinI stands in for YAIB's troponin T (both "troponin measured, and how high").
CINC_TO_YAIB = {
    "HR": "hr", "O2Sat": "o2sat", "Temp": "temp", "SBP": "sbp", "MAP": "map", "DBP": "dbp",
    "Resp": "resp", "BaseExcess": "be", "HCO3": "bicar", "FiO2": "fio2", "pH": "ph",
    "PaCO2": "pco2", "AST": "ast", "BUN": "bun", "Alkalinephos": "alp", "Calcium": "ca",
    "Chloride": "cl", "Creatinine": "crea", "Bilirubin_direct": "bili_dir", "Glucose": "glu",
    "Lactate": "lact", "Magnesium": "mg", "Phosphate": "phos", "Potassium": "k",
    "Bilirubin_total": "bili", "TroponinI": "tnt", "Hgb": "hgb", "PTT": "ptt", "WBC": "wbc",
    "Fibrinogen": "fgn", "Platelets": "plt",
}

#: the configuration the registered eICU sweep chose (artifacts/yaib/eicu_sweep_selection.json)
DEEP_UNBALANCED = B.TrainConfig(learning_rate=0.03, num_leaves=127, min_child_samples=400,
                                feature_fraction=0.3, bagging_fraction=0.7, lambda_l2=10.0,
                                early_stopping=150, balanced=False)

#: name -> extension families read beside rich, and booster seeds averaged (``bench``'s bag).
#: Every candidate runs the configuration round 1 chose on eICU (DEEP_UNBALANCED).
CANDIDATES: Dict[str, Dict[str, Any]] = {
    "rich": {"families": (), "cfg": {}},
    "temporal": {"families": ("obs", "dmean", "roll", "process"), "cfg": {}},
    "only-obs": {"families": ("obs",), "cfg": {}},
    "only-obs_common": {"families": ("obs_common",), "cfg": {}},
    "only-dlast": {"families": ("dlast",), "cfg": {}},
    "only-dmean": {"families": ("dmean",), "cfg": {}},
    "only-roll": {"families": ("roll",), "cfg": {}},
    "only-process": {"families": ("process",), "cfg": {}},
    "rich_bag3": {"families": (), "cfg": {}, "bag": 3},
    "obs_common_bag3": {"families": ("obs_common",), "cfg": {}, "bag": 3},
}


# --------------------------------------------------------------------------- #
# CinC 2019 -> YAIB cohort format
# --------------------------------------------------------------------------- #
def convert(cinc_root: str, out: str) -> Dict[str, Any]:
    """``dyn`` / ``sta`` / ``outc`` parquet, as YAIB's cohort generation writes them.

    stay id: the record number (setA p000001 -> 1, setB p100001 -> 100001); time: hours
    since ICU admission (``ICULOS - 1``) as a timedelta; label: ``SepsisLabel``, which
    like YAIB's turns positive 6 h before onset. YAIB variables CinC lacks stay missing."""
    files = sorted(glob.glob(os.path.join(cinc_root, "training_set*", "p*.psv")))
    if not files:
        raise FileNotFoundError(f"no p*.psv under {cinc_root}/training_set*")
    dyn, sta, outc = [], [], []
    for path in files:
        f = pd.read_csv(path, sep="|")
        sid = int(os.path.basename(path)[1:-4])
        hours = f["ICULOS"].to_numpy(dtype=float) - 1.0
        frame = pd.DataFrame({GROUP: np.full(len(f), sid, dtype=np.int32),
                              "time": pd.to_timedelta(hours, unit="h").astype("timedelta64[s]")})
        for v in DYNAMIC:
            frame[v] = np.nan
        for c, v in CINC_TO_YAIB.items():
            frame[v] = f[c].to_numpy(dtype=float)
        dyn.append(frame)
        outc.append(pd.DataFrame({GROUP: frame[GROUP], "time": frame["time"],
                                  "label": f["SepsisLabel"].to_numpy().astype(bool)}))
        sta.append((sid, float(f["Age"].iloc[0]), "Male" if f["Gender"].iloc[0] == 1 else "Female"))
    os.makedirs(out, exist_ok=True)
    d = pd.concat(dyn, ignore_index=True)
    o = pd.concat(outc, ignore_index=True)
    s = pd.DataFrame(sta, columns=[GROUP, "age", "sex"])
    s["height"], s["weight"] = np.nan, np.nan
    s[GROUP] = s[GROUP].astype(np.int32)
    d.to_parquet(os.path.join(out, "dyn.parquet"), index=False)
    o.to_parquet(os.path.join(out, "outc.parquet"), index=False)
    s[[GROUP] + STATIC].to_parquet(os.path.join(out, "sta.parquet"), index=False)
    return {"stays": len(files), "rows": len(d), "septic_stays": int(o.groupby(GROUP)["label"].max().sum()),
            "positive_hours": int(o["label"].sum())}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
class GroupedAUC:
    """Weighted AUROC over (positive, negative) pairs that share a group, for fixed scores
    and any row weights: sort once, then each call is O(n). Ties count one half, as in
    ``roc_auc_score``. With one group it is the AUROC; grouped by stay hour it is the
    hour-matched AUROC (the pair-weighted mean of per-hour AUROCs), under which a model
    that knows only the hour scores exactly 0.5."""

    def __init__(self, y: np.ndarray, p: np.ndarray, group: Optional[np.ndarray] = None):
        group = np.zeros(len(y)) if group is None else np.asarray(group)
        self.order = np.lexsort((p, group))
        g, ps = group[self.order], np.asarray(p)[self.order]
        self.y = np.asarray(y, dtype=float)[self.order]
        g_start = np.r_[True, g[1:] != g[:-1]]
        t_start = g_start | np.r_[True, ps[1:] != ps[:-1]]
        self.g_id = np.cumsum(g_start) - 1
        self.t_id = np.cumsum(t_start) - 1
        self.t_first = np.flatnonzero(t_start)
        self.t_group_first = np.flatnonzero(g_start)[self.g_id[self.t_first]]

    def __call__(self, w: Optional[np.ndarray] = None) -> float:
        w = np.ones(len(self.y)) if w is None else np.asarray(w, dtype=float)[self.order]
        wp, wn = w * self.y, w * (1.0 - self.y)
        cn = np.r_[0.0, np.cumsum(wn)]
        below = cn[self.t_first] - cn[self.t_group_first]      # negatives under each tie block, same group
        num = float(np.dot(np.bincount(self.t_id, wp), below + 0.5 * np.bincount(self.t_id, wn)))
        den = float(np.dot(np.bincount(self.g_id, wp), np.bincount(self.g_id, wn)))
        return num / den if den > 0 else float("nan")


def hour_matched(y: np.ndarray, p: np.ndarray, hour: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    return GroupedAUC(y, p, hour)(w)


def paired(frames: Dict[str, Dict[tuple, pd.DataFrame]], a: str, b: str, n_boot: int, seed: int = 0) -> Dict[str, Any]:
    """Paired deltas (a - b), AUROC and hour-matched AUROC, bootstrapping test stays within
    each repetition (its folds share one test set), as ``bench.compare`` does."""
    keys = sorted(set(frames[a]) & set(frames[b]))
    by_rep: Dict[int, List[tuple]] = {}
    for k in keys:
        fa, fb = frames[a][k], frames[b][k]
        if not np.array_equal(fa["stay_id"].to_numpy(), fb["stay_id"].to_numpy()):
            raise ValueError("candidates were scored on different rows")
        by_rep.setdefault(k[0], []).append((fa, fb))
    rng = np.random.default_rng(seed)
    point = {"auroc": [], "hour_matched": []}
    draws = {"auroc": np.zeros(n_boot), "hour_matched": np.zeros(n_boot)}
    for rep, pairs in sorted(by_rep.items()):
        stays = np.unique(pairs[0][0]["stay_id"].to_numpy())
        codes = [np.searchsorted(stays, fa["stay_id"].to_numpy()) for fa, _ in pairs]
        aucs = []
        for fa, fb in pairs:
            y, h = fa["label"].to_numpy(), fa["hour"].to_numpy()
            aucs.append({"auroc": (GroupedAUC(y, fa["p"].to_numpy()), GroupedAUC(y, fb["p"].to_numpy())),
                         "hour_matched": (GroupedAUC(y, fa["p"].to_numpy(), h), GroupedAUC(y, fb["p"].to_numpy(), h))})

        def delta(weights):
            out = {"auroc": [], "hour_matched": []}
            for (fa, _), c, auc in zip(pairs, codes, aucs):
                w = None if weights is None else weights[c]
                for m, (ga, gb) in auc.items():
                    d = ga(w) - gb(w)
                    if np.isfinite(d):
                        out[m].append(d)
            return {m: float(np.mean(v)) if v else np.nan for m, v in out.items()}

        d0 = delta(None)
        for m in point:
            point[m].append(d0[m])
        for i in range(n_boot):
            m_ = np.bincount(rng.integers(0, len(stays), len(stays)), minlength=len(stays)).astype(float)
            di = delta(m_)
            for m in draws:
                draws[m][i] += di[m]
    out: Dict[str, Any] = {"runs": len(keys), "repetitions": len(by_rep)}
    for m in point:
        dr = draws[m] / len(by_rep)
        lo, hi = np.nanpercentile(dr, [2.5, 97.5])
        out[m] = {"delta": float(np.mean(point[m])), "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


_FRAMES: Dict[str, Dict[tuple, pd.DataFrame]] = {}


def _paired_job(job):
    return paired(_FRAMES, job[0], job[1], job[2])


def _compare_all(frames, jobs, n_boot: int, workers: int) -> List[Dict[str, Any]]:
    """Every paired comparison, in forked workers that inherit the frames (each comparison
    seeds its own bootstrap, so the result equals running them one after another)."""
    global _FRAMES
    _FRAMES = frames
    if workers <= 1 or len(jobs) <= 1:
        return [_paired_job((a, b, n_boot)) for a, b in jobs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        return pool.map(_paired_job, [(a, b, n_boot) for a, b in jobs])


# --------------------------------------------------------------------------- #
# screening
# --------------------------------------------------------------------------- #
def _fit(X, y, tr, va, te, cfg: B.TrainConfig, bag: int) -> Dict[str, Any]:
    """One candidate on one run, through ``bench``'s own fit (``bag`` > 1 is its bag:
    member 0 is the single model, the rest differ in their sampling seeds), so what is
    screened here is what the worker would run."""
    fit = B._fit_predict(X, y, tr, va, te, replace(cfg, bag=bag), stream=True)
    return {"p": fit["p"], "val_auroc": fit["val_auroc"], "rounds": [fit["rounds"]]}


def screen(cohort_dir: str, runs: List[tuple], names: List[str], work: str, threads: int,
           n_boot: int, reference: str = "rich", pairs: Optional[List[tuple]] = None) -> Dict[str, Any]:
    os.umask(0o077)
    Path(work).mkdir(parents=True, exist_ok=True, mode=0o700)
    t0 = time.time()
    cohort = load(cohort_dir)
    base, base_names = FE.build(cohort, feature_set="rich")
    ext, ext_names = EX.build(cohort, extension="temporal")
    built = time.time() - t0
    y_all = cohort.label
    lab = np.isfinite(y_all)
    y = np.nan_to_num(y_all, nan=0.0)
    stay = cohort.dyn[GROUP].to_numpy()
    hour = cohort.dyn["hour"].to_numpy()
    labels = cohort.stay_labels()
    cfg0 = replace(DEEP_UNBALANCED, threads=threads)
    frames: Dict[str, Dict[tuple, pd.DataFrame]] = {n: {} for n in names}
    table: Dict[str, Any] = {}
    for name in names:
        cand = CANDIDATES[name]
        keep = [i for i, c in enumerate(ext_names) if EX.in_families(c, cand["families"])]
        X = EX.Columns([base, ext], keep=[None, np.array(keep, dtype=np.int64)]) if keep else base
        cfg = replace(cfg0, **cand["cfg"])
        rows = []
        for (r, f) in runs:
            sp = SP.split(labels, r, f)
            tr = np.flatnonzero(np.isin(stay, sp.train) & lab)
            va = np.flatnonzero(np.isin(stay, sp.val) & lab)
            te = np.flatnonzero(np.isin(stay, sp.test) & lab)
            cache = Path(work) / f"{name}-r{r}-f{f}.npz"
            if cache.exists():
                saved = np.load(cache)
                fit = {"p": saved["p"], "val_auroc": float(saved["val_auroc"]), "rounds": saved["rounds"].tolist()}
            else:
                t1 = time.time()
                fit = _fit(X, y, tr, va, te, cfg, cand.get("bag", 1))
                fit["seconds"] = time.time() - t1
                np.savez(cache, p=fit["p"], val_auroc=fit["val_auroc"], rounds=np.array(fit["rounds"]))
            frames[name][(r, f)] = pd.DataFrame({"stay_id": stay[te], "hour": hour[te], "label": y[te], "p": fit["p"]})
            rows.append({"run": f"{r}:{f}", "val_auroc": fit["val_auroc"], "rounds": fit["rounds"],
                         "test_auroc": float(roc_auc_score(y[te], fit["p"])),
                         "test_hour_matched": hour_matched(y[te], fit["p"], hour[te]),
                         "seconds": round(fit.get("seconds", float("nan")), 1)})
            print(json.dumps({"candidate": name, **rows[-1]}), flush=True)
        table[name] = {"families": list(cand["families"]), "n_features": X.shape[1], "bag": cand.get("bag", 1),
                       "config": asdict(cfg), "runs": rows,
                       "mean_val_auroc": float(np.mean([r_["val_auroc"] for r_ in rows])),
                       "mean_test_auroc": float(np.mean([r_["test_auroc"] for r_ in rows])),
                       "mean_test_hour_matched": float(np.mean([r_["test_hour_matched"] for r_ in rows]))}
    jobs = [(n, reference) for n in names if n != reference] + list(pairs or [])
    comparisons = dict(zip([f"{a} - {b}" for a, b in jobs], _compare_all(frames, jobs, n_boot, threads)))
    return {"schema": "yaib_proxy_screen/1",
            "cohort": {"name": cohort.name, "stays": int(len(labels)), "septic_stays": int(labels.sum()),
                       "rows": int(len(stay)), "labelled_rows": int(lab.sum())},
            "protocol": {"split": "YAIB make_single_split_polars (splits.py), seed 1234", "runs": [f"{r}:{f}" for r, f in runs],
                         "early_stopping": "validation stays", "compared_on": "test stays of the proxy, paired bootstrap over stays",
                         "n_boot": n_boot},
            "seconds_to_build_features": round(built, 1), "candidates": table, "comparisons": comparisons,
            "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--cinc", required=True)
    c.add_argument("--out", required=True)
    s = sub.add_parser("screen")
    s.add_argument("cohort")
    s.add_argument("--runs", default="0:0,0:1,0:2")
    s.add_argument("--candidates", default="rich,temporal")
    s.add_argument("--work", required=True, help="private cache of per-run test probabilities")
    s.add_argument("--out", required=True)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--n-boot", type=int, default=200)
    s.add_argument("--pairs", default="", help="extra paired comparisons, e.g. only-obs_common:only-obs")
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        print(json.dumps(convert(a.cinc, a.out)))
        return 0
    runs = [tuple(int(x) for x in r.split(":")) for r in a.runs.split(",")]
    names = a.candidates.split(",")
    unknown = set(names) - set(CANDIDATES)
    if unknown:
        raise SystemExit(f"unknown candidates {sorted(unknown)}")
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if p]
    if any(x not in names for p in pairs for x in p):
        raise SystemExit("--pairs must name screened candidates")
    res = screen(a.cohort, runs, names, a.work, a.threads, a.n_boot, pairs=pairs)
    from backend.ml.jsonio import json_safe
    Path(a.out).write_text(json.dumps(json_safe(res), indent=1))
    for k, v in res["comparisons"].items():
        print(f"{k}: AUROC {v['auroc']['delta']:+.4f} {v['auroc']['ci95']}, "
              f"hour-matched {v['hour_matched']['delta']:+.4f} {v['hour_matched']['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
