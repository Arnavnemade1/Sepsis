"""Planted-failure checks on our model under YAIB's protocol (backend/ml/yaib).

A benchmark number is only comparable if it was made on the benchmark's cohort, splits
and metric. Each check below plants a way that could silently stop being true:

  [1] cohort     a missing column, a duplicated stay-hour, an outcome row with no
                 feature row are refused; labels land on their own hour
  [2] splits     each repetition's test sets partition the stays; train / val / test
                 are disjoint; the ids equal YAIB's own function (pinned fingerprints
                 computed with YAIB af9a5b3, and a live check when YAIB is installed);
                 too few septic stays is refused, as YAIB refuses it
  [3] features   a planted future spike moves no earlier hour and no other stay; the
                 labels are never read
  [4] metric     per-run AUROC is sklearn's on the labelled test hours; the summary's
                 "std" field is YAIB's (population SD / sqrt(n))
  [5] compare    the baseline's ids written out of order (as YAIB writes them) pair
                 by position; a label disagreement, a missing row and a different run
                 set are refused; a real difference is found and a null is not
  [6] end to end the 5 x 5 protocol runs on a synthetic cohort and finds a planted signal

    python -m backend.ml.scripts.verify_yaib
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pandas as pd

from backend.ml.yaib import bench as B
from backend.ml.yaib import cohort as C
from backend.ml.yaib import features as FE
from backend.ml.yaib import splits as SP

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


def refused(fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
    except (ValueError, KeyError):
        return True
    return False


# YAIB's own split (icu_benchmarks.data.split_process_data.make_single_split_polars at
# af9a5b3, seed 1234), on ``synthetic(n=200, seed=7)``: fingerprints of (r0 f0), (r2 f3),
# (r4 f4). Computed in an environment with YAIB installed, where SP.conformance passed for
# all 25 (repetition, fold) pairs.
PINNED = {(0, 0): "752afea6ba895e75", (2, 3): "33c48f06ace5bb05", (4, 4): "0e940a5a9fce9bd5"}


def synthetic(n: int = 200, seed: int = 7, frac: float = 0.25):
    """A YAIB-format cohort: hourly heart rate and lactate rise before onset in septic
    stays; the label is 1 from 6 h before onset on (YAIB's sepsis label is a horizon)."""
    rng = np.random.default_rng(seed)
    dyn, outc, sta = [], [], []
    for i in range(n):
        sid = 1000 + 7 * i
        T = int(rng.integers(20, 60))
        septic = rng.random() < frac
        onset = int(rng.integers(10, T)) if septic else None
        h = np.arange(T)
        hr = 80 + rng.normal(0, 8, T) + (np.clip(h - onset + 8, 0, 8) * 4 if septic else 0)
        lact = np.where(rng.random(T) < 0.2, 1.2 + rng.normal(0, .3, T) + (np.clip(h - onset + 6, 0, 6) * .4
                                                                            if septic else 0), np.nan)
        row = {v: np.full(T, np.nan) for v in C.DYNAMIC}
        row.update(hr=hr, lact=lact, sbp=120 + rng.normal(0, 10, T), map=80 + rng.normal(0, 6, T))
        dyn.append(pd.DataFrame({"stay_id": sid, "time": pd.to_timedelta(h, unit="h"), **row}))
        lab = (h >= onset - 6) if septic else np.zeros(T, bool)
        outc.append(pd.DataFrame({"stay_id": sid, "time": pd.to_timedelta(h, unit="h"), "label": lab}))
        sta.append({"stay_id": sid, "age": float(rng.integers(20, 90)), "sex": rng.choice(["Male", "Female"]),
                    "height": 170.0, "weight": 70.0})
    return pd.concat(dyn, ignore_index=True), pd.DataFrame(sta), pd.concat(outc, ignore_index=True)


def write(d: str, dyn, sta, outc) -> str:
    os.makedirs(d, exist_ok=True)
    dyn.sample(frac=1.0, random_state=0).to_parquet(os.path.join(d, "dyn.parquet"), index=False)  # unsorted on disk
    sta.to_parquet(os.path.join(d, "sta.parquet"), index=False)
    outc.to_parquet(os.path.join(d, "outc.parquet"), index=False)
    return d


# --------------------------------------------------------------------------- #
def cohort_checks(tmp: str) -> C.Cohort:
    print("[1] cohort")
    dyn, sta, outc = synthetic()
    c = C.load(write(os.path.join(tmp, "ok"), dyn, sta, outc))
    truth = outc.assign(hour=outc.time / pd.Timedelta(hours=1)).set_index(["stay_id", "hour"])["label"]
    got = pd.Series(c.label, index=pd.MultiIndex.from_arrays([c.dyn.stay_id, c.dyn.hour]))
    check("labels land on their own hour, though dyn is shuffled on disk",
          np.array_equal(got.sort_index().to_numpy(), truth.sort_index().astype(float).to_numpy()))
    check("rows come out sorted by stay and hour",
          bool((c.dyn[["stay_id", "hour"]].apply(tuple, axis=1).is_monotonic_increasing)))
    check("sex is read as 1 / 0", set(c.sta.sex.dropna().unique()) <= {0.0, 1.0})
    check("a missing dynamic column is refused",
          refused(C.load, write(os.path.join(tmp, "nocol"), dyn.drop(columns="hr"), sta, outc)))
    check("a duplicated stay-hour is refused",
          refused(C.load, write(os.path.join(tmp, "dup"), pd.concat([dyn, dyn.iloc[:1]]), sta, outc)))
    extra = pd.concat([outc, pd.DataFrame({"stay_id": [1000], "time": [pd.Timedelta(hours=999)], "label": [True]})])
    check("an outcome row with no feature row is refused",
          refused(C.load, write(os.path.join(tmp, "extra"), dyn, sta, extra)))
    return c


def split_checks(c: C.Cohort) -> None:
    print("[2] splits")
    lab = c.stay_labels()
    stays = set(lab.index)
    outer = [set(SP.split(lab, r, 0).test) for r in range(5)]
    check("the 5 outer test sets partition the stays",
          set().union(*outer) == stays and sum(len(t) for t in outer) == len(stays))
    s = SP.split(lab, 2, 3)
    tr, va, te = set(s.train), set(s.val), set(s.test)
    check("train, val and test are disjoint and cover every stay",
          not (tr & va or tr & te or va & te) and tr | va | te == stays)
    check("the inner folds of one repetition share its test set",
          all(set(SP.split(lab, 2, f).test) == te for f in range(5)))
    check("every test set holds septic stays (stratified)", all(lab[list(t)].sum() > 0 for t in outer))
    check("the split is deterministic", SP.split(lab, 1, 1).fingerprint() == SP.split(lab, 1, 1).fingerprint())
    check("a different seed gives a different split",
          SP.split(lab, 0, 0, seed=1).fingerprint() != SP.split(lab, 0, 0).fingerprint())
    got = {k: SP.split(lab, *k).fingerprint() for k in PINNED}
    check("ids equal YAIB's own split (pinned, YAIB af9a5b3)", got == PINNED, f"{got}")
    live = SP.conformance(lab)
    if live is None:
        print("  [SKIP] live conformance: icu_benchmarks is not installed here")
    else:
        check("ids equal YAIB's own split (live, all 25 runs)", all(live.values()), f"{sum(live.values())}/25")
    few = lab.copy()
    few[:] = 0.0
    few.iloc[:3] = 1.0
    check("three septic stays for five folds is refused", refused(SP.split, few, 0, 0))


def feature_checks(c: C.Cohort) -> None:
    print("[3] features")
    X, names = FE.build(c)
    check("one row per cohort hour", X.shape[0] == len(c.dyn), f"{X.shape}")
    sid = c.dyn.stay_id.to_numpy()
    first = sid[0]
    rows = np.flatnonzero(sid == first)
    nxt = np.flatnonzero(sid == np.unique(sid)[1])
    k = rows[len(rows) // 2]
    d2 = c.dyn.copy()
    d2.loc[k, "hr"] = 400.0
    d2.loc[k, "lact"] = 25.0
    X2, _ = FE.build(C.Cohort(d2, c.sta, c.label))
    earlier = rows[rows < k]
    check("a spike at hour h moves no earlier hour of that stay",
          np.array_equal(X[earlier], X2[earlier], equal_nan=True))
    check("... and it does move hour h (the check can fail)", not np.array_equal(X[k], X2[k], equal_nan=True))
    check("... and no hour of the next stay", np.array_equal(X[nxt], X2[nxt], equal_nan=True))
    X3, _ = FE.build(C.Cohort(c.dyn, c.sta, 1 - np.nan_to_num(c.label)))
    check("flipping every label changes no feature", np.array_equal(X, X3, equal_nan=True))
    hr_since = X[:, names.index("hr__since")]
    check("hours-since-measured is 0 where measured", bool(np.all(hr_since[np.isfinite(c.dyn.hr.to_numpy())] == 0)))
    check("columns come out in the declared order", names == FE.feature_names() and X.shape[1] == len(names))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "X.npy")
        Xm, _ = FE.build(c, out=path)
        check("a memmapped build is byte-identical to the in-memory one",
              isinstance(Xm, np.memmap) and X.dtype == Xm.dtype and X.tobytes() == np.asarray(Xm).tobytes())
        check("... and is readable back from disk", np.array_equal(np.load(path, mmap_mode="r"), X, equal_nan=True))
        del Xm


def memory_checks(c: C.Cohort) -> None:
    print("[3b] memory plan")
    big = FE.memory_plan(4_000_000, available_bytes=int(7.6e9), on_disk=False)
    disk = FE.memory_plan(4_000_000, available_bytes=int(7.6e9), on_disk=True)
    check("a memmapped matrix lowers the planned heap by exactly the matrix",
          math.isclose(big["peak_heap_gb"] - disk["peak_heap_gb"], big["matrix_gb"]))
    check("4M stay-hours in RAM do not fit a 7.6 GB worker", big["fits"] is False, f"{big['peak_heap_gb']:.1f} GB")
    real = FE._mem_available
    built = []
    real_build = FE.build
    FE._mem_available = lambda: 1_000_000          # a 1 MB host
    FE.build = lambda *a, **k: built.append(1) or real_build(*a, **k)
    try:
        B.run(c, B.TrainConfig(), 5, 5, runs=[(0, 0)])
        refused_ = False
    except MemoryError:
        refused_ = True
    finally:
        FE._mem_available, FE.build = real, real_build
    check("an over-budget run is refused before anything is built", refused_ and not built)


def metric_checks() -> None:
    print("[4] metric")
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500).astype(float)
    p = rng.random(500) + y * 0.3
    check("per-run AUROC is sklearn's", B._metric(y, p)["auroc"] == roc_auc_score(y, p))
    v = np.array([0.6, 0.7, 0.65, 0.8, 0.55])
    s = B.yaib_summary("auroc", v)
    check("the 'std' field is YAIB's population SD / sqrt(n)",
          math.isclose(s["auroc_yaib_std_field"], float(np.std(v)) / math.sqrt(5)))
    check("the t interval holds the mean", s["auroc_ci95"][0] < s["auroc_mean"] < s["auroc_ci95"][1])


def _runs(n_runs=(2, 2), n_stays=120, seed=0, signal=1.0):
    """ours and the baseline's per-run frames over the same rows; ours sees ``signal``."""
    rng = np.random.default_rng(seed)
    ours, base = {}, {}
    for r in range(n_runs[0]):
        # the inner folds of a repetition share one test set, and so its labels
        stays = np.arange(n_stays) + 1000 * r
        sid = np.repeat(stays, 10)
        hr = np.tile(np.arange(10.0), n_stays)
        y = (rng.random(len(sid)) < 0.15).astype(float)
        for f in range(n_runs[1]):
            ours[(r, f)] = pd.DataFrame({"stay_id": sid, "hour": hr, "label": y,
                                         "p": rng.random(len(sid)) + signal * y})
            base[(r, f)] = pd.DataFrame({"stay_id": sid, "hour": hr, "label": y, "p": rng.random(len(sid))})
    return ours, base


def compare_checks() -> None:
    print("[5] compare")
    ours, base = _runs()
    # YAIB writes the ids in another order and the hour in milliseconds; labels/preds stay sorted
    scrambled = {}
    for k, b in base.items():
        ids = b[["stay_id", "hour"]].sample(frac=1.0, random_state=1).to_numpy()
        scrambled[k] = pd.DataFrame({"stay_id": ids[:, 0].astype(int), "hour": ids[:, 1] * 3.6e6,
                                     "label": b["label"].to_numpy(), "p": b["p"].to_numpy()})
    res = B.compare(ours, scrambled, n_boot=200)
    check("out-of-order ids in ms pair by position and find a real difference",
          res["better"] and res["auroc_delta"] > 0.2, f"{res['auroc_delta']:+.3f} [{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]")
    # one null seed can exclude zero by chance (that is what 95% means), so the claim
    # checked is the rate: across null seeds the interval should miss zero about 5% of
    # the time, and a liberal test (say 20%+) would hand out false "better" verdicts
    hits = 0
    n_null = 40
    for sd in range(n_null):
        o, b = _runs(n_stays=80, seed=100 + sd, signal=0.0)
        r0 = B.compare(o, b, n_boot=120, seed=sd)
        hits += r0["better"] or r0["worse"]
    check("no signal: the interval excludes zero at about the nominal rate", hits / n_null <= 0.125,
          f"{hits}/{n_null} null comparisons rejected")
    bad = dict(base)
    k0 = next(iter(bad))
    flipped = bad[k0].copy()
    flipped.loc[flipped.index[5], "label"] = 1 - flipped.loc[flipped.index[5], "label"]
    bad[k0] = flipped
    check("a single label disagreement is refused", refused(B.compare, ours, bad, n_boot=10))
    short = dict(base)
    short[k0] = base[k0].iloc[1:]
    check("a missing row is refused", refused(B.compare, ours, short, n_boot=10))
    fewer = {k: v for k, v in base.items() if k != k0}
    check("a different set of runs is refused", refused(B.compare, ours, fewer, n_boot=10))


def end_to_end(c: C.Cohort, tmp: str) -> None:
    print("[6] end to end")
    pdir = os.path.join(tmp, "preds")
    res = B.run(c, B.TrainConfig(max_rounds=200, early_stopping=20), predictions_dir=pdir)
    m = res["yaib_metric"]
    check("25 runs, one per (repetition, fold)", m["n_runs"] == 25 and len(res["runs"]) == 25)
    check("the planted signal is found", m["auroc_mean"] > 0.75, f"AUROC {m['auroc_mean']:.3f}")
    check("each run records its split fingerprint",
          all(len(r["split"]) == 16 for r in res["runs"]) and len({r["split"] for r in res["runs"]}) == 25)
    check("predictions written per run", len(os.listdir(pdir)) == 25)
    blob = str(res)
    check("the result carries no row", "stay_id" not in blob and len(blob) < 20_000, f"{len(blob)} chars")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        c = cohort_checks(tmp)
        split_checks(c)
        feature_checks(c)
        memory_checks(c)
        metric_checks()
        compare_checks()
        end_to_end(c, tmp)
    print("-" * 72)
    print(f"{sum(results)}/{len(results)} YAIB-protocol checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
