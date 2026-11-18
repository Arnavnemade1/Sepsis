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
  [3c] extension rewriting every stay's future moves no earlier hour, bit for bit; each
                 stay built alone gives the same rows (no cohort-wide running sum); hand-
                 checked values; the side-by-side view; the base matrix keeps its store
                 identity with an extension, and the extension gets its own
  [3d] bag       a bag of one is the single model; a bag of three averages boosters,
                 is judged on its averaged validation scores, and streams exactly
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
        back = np.load(path, mmap_mode="r")
        check("... stored row-major, with its column scratch file removed",
              back.flags.c_contiguous and os.listdir(tmp) == ["X.npy"])
        del Xm, back
        saved = FE.STREAM_ROWS
        FE.STREAM_ROWS = 97                       # many row blocks, and a ragged last one
        try:
            Xs, _ = FE.build(c, out=os.path.join(tmp, "Xs.npy"))
            check("... identical when copied in many small row blocks", X.tobytes() == np.asarray(Xs).tobytes())
            del Xs
        finally:
            FE.STREAM_ROWS = saved


def rich_checks(c: C.Cohort) -> None:
    print("[3a] rich feature set")
    Xb, nb = FE.build(c)
    X, names = FE.build(c, feature_set="rich")
    check("rich keeps every base column unchanged, in front",
          names[:len(nb)] == nb and np.array_equal(X[:, :len(nb)], Xb, equal_nan=True), f"{X.shape[1]} columns")
    check("columns come out in the declared order", names == FE.feature_names("rich"))
    check("an unknown feature set is refused", refused(FE.feature_names, "everything"))
    sid = c.dyn.stay_id.to_numpy()
    rows = np.flatnonzero(sid == sid[0])
    nxt = np.flatnonzero(sid == np.unique(sid)[1])
    k = rows[len(rows) // 2]
    d2 = c.dyn.copy()
    for v in ("hr", "lact", "plt", "crea", "po2", "urine"):
        d2.loc[k, v] = 400.0 if v != "plt" else 5.0
    X2, _ = FE.build(C.Cohort(d2, c.sta, c.label), feature_set="rich")
    earlier = rows[rows < k]
    check("rich: a spike at hour h moves no earlier hour", np.array_equal(X[earlier], X2[earlier], equal_nan=True))
    moved = [n for j, n in enumerate(names) if j >= len(nb) and not np.array_equal(X[k, j], X2[k, j], equal_nan=True)]
    check("... and it moves the windowed and organ columns at h (the check can fail)",
          {"hr__max6", "sofa_proxy", "urine6"} <= set(moved), f"{len(moved)} rich columns moved")
    check("... and no hour of the next stay", np.array_equal(X[nxt], X2[nxt], equal_nan=True))
    X3, _ = FE.build(C.Cohort(c.dyn, c.sta, 1 - np.nan_to_num(c.label)), feature_set="rich")
    check("rich: flipping every label changes no feature", np.array_equal(X, X3, equal_nan=True))
    # hand-checked values on a two-stay cohort
    dyn = pd.DataFrame({"stay_id": [1, 1, 1, 1, 2, 2], "hour": [0.0, 1, 2, 3, 0, 1]})
    for v in C.DYNAMIC:
        dyn[v] = np.nan
    dyn["hr"] = [80, np.nan, 120, 90, 60, np.nan]
    dyn["plt"] = [np.nan, 40, np.nan, np.nan, 200, np.nan]
    dyn["urine"] = [100, np.nan, 50, 25, 10, 10]
    sta = pd.DataFrame({"stay_id": [1, 2], "age": [60.0, 70], "sex": [1.0, 0], "height": [170.0, 160], "weight": [70.0, 60]})
    t = C.Cohort(dyn, sta, np.zeros(6))
    Xt, nt = FE.build(t, feature_set="rich")
    col = lambda n: Xt[:, nt.index(n)]
    check("max6 is the max of measured values so far in the window",
          np.allclose(col("hr__max6")[:4], [80, 80, 120, 120]) and np.allclose(col("hr__max6")[4], 60))
    check("dev_first is carried value minus the stay's first value",
          np.allclose(col("hr__dev_first")[:4], [0, 0, 40, 10]) and col("hr__dev_first")[4] == 0)
    check("platelets 40 score coagulation 3; unmeasured scores 0 (SOFA's convention)",
          np.allclose(col("sofa_coag"), [0, 3, 3, 3, 0, 0]))
    check("urine6 sums within the stay only", np.allclose(col("urine6"), [100, 100, 150, 175, 10, 20]))


def memory_checks(c: C.Cohort) -> None:
    print("[3b] memory plan and streaming")
    big = FE.memory_plan(4_000_000, available_bytes=int(7.6e9), on_disk=False)
    disk = FE.memory_plan(4_000_000, available_bytes=int(7.6e9), on_disk=True)
    check("streaming from a memmap plans below the matrix itself (no matrix, no row copies on the heap)",
          disk["peak_heap_gb"] < disk["matrix_gb"] < big["peak_heap_gb"],
          f"{disk['peak_heap_gb']:.1f} GB vs a {disk['matrix_gb']:.1f} GB matrix")
    check("4M stay-hours in RAM do not fit a 7.6 GB worker", big["fits"] is False, f"{big['peak_heap_gb']:.1f} GB")
    eicu = FE.memory_plan(6_000_000, available_bytes=int(7.6e9), on_disk=True, feature_set="rich")
    check("full eICU (~6M stay-hours), rich, streamed, fits it", eicu["fits"] is True, f"{eicu['peak_heap_gb']:.1f} GB")
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
    with tempfile.TemporaryDirectory() as tmp:
        cfg = B.TrainConfig(max_rounds=60, early_stopping=10)
        mem = B.run(c, cfg, runs=[(0, 0), (1, 2)])
        old = FE.STREAM_ROWS
        FE.STREAM_ROWS = 97                          # many uneven batches
        try:
            own = C.Cohort(c.dyn.copy(), c.sta, c.label)          # the run releases its variables
            st = B.run(own, cfg, runs=[(0, 0), (1, 2)], features_path=os.path.join(tmp, "X.npy"))
        finally:
            FE.STREAM_ROWS = old
        check("a streamed fit scores exactly as the in-memory fit",
              [r["auroc"] for r in st["runs"]] == [r["auroc"] for r in mem["runs"]]
              and st["model"]["streamed_from_memmap"] and not mem["model"]["streamed_from_memmap"])
        import lightgbm as lgb
        sequence = B._row_sequence(np.zeros((4, 2), dtype=np.float32), np.arange(4))
        check("streamed input is a real LightGBM Sequence, not a dense fallback",
              isinstance(sequence, lgb.Sequence) and sequence[0].dtype == np.float64)
        try:
            np.asarray(sequence)
            refused_conversion = False
        except TypeError:
            refused_conversion = True
        check("whole-matrix conversion of a row adapter is refused", refused_conversion)
        check("a streamed run releases the cohort's variables once the memmap is written",
              list(own.dyn.columns) == [C.GROUP, "hour"] and len(own.dyn) == len(c.dyn))
        check("each run reports a validation AUROC for selection",
              all(0.0 <= r["val_auroc"] <= 1.0 for r in st["runs"]) and "val_auroc_mean" in st["validation_metric"])


def extension_checks(c: C.Cohort, tmp: str) -> None:
    print("[3c] extension: columns beside the base matrix")
    from backend.ml.yaib import checkpoint as CP
    from backend.ml.yaib import extension as EX
    E, names = EX.build(c)
    check("columns come out in the declared order", names == EX.extension_names("temporal") and E.shape == (len(c.dyn), len(names)),
          f"{E.shape[1]} columns")
    check("an unknown extension is refused", refused(EX.extension_names, "everything"))
    check("no extension column shares a name with a rich column",
          not set(names) & set(FE.feature_names("rich")))
    sid = c.dyn.stay_id.to_numpy()
    rng = np.random.default_rng(3)
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    ends = np.r_[starts[1:], len(sid)]
    future = np.zeros(len(sid), bool)
    for a, b in zip(starts, ends):
        future[rng.integers(a, b) + 1:b] = True
    d2 = c.dyn.copy()
    vals = d2[C.DYNAMIC].to_numpy()
    d2[C.DYNAMIC] = np.where(future[:, None], np.where(rng.random(vals.shape) < 0.5, np.nan, vals * 3 + 50), vals)
    E2, _ = EX.build(C.Cohort(d2, c.sta, c.label))
    check("rewriting every stay's future moves no earlier hour (bit for bit)",
          np.array_equal(E[~future], E2[~future], equal_nan=True))
    check("... and it does move later hours (the check can fail)",
          not np.array_equal(E[future], E2[future], equal_nan=True))
    alone = np.concatenate([EX.build(C.Cohort(c.dyn.iloc[a:b].reset_index(drop=True), c.sta, c.label[a:b]))[0]
                            for a, b in zip(starts[:6], ends[:6])])
    check("each stay built on its own gives the same rows (no cohort-wide running sum)",
          np.array_equal(alone, E[:ends[5]], equal_nan=True))
    E3, _ = EX.build(C.Cohort(c.dyn, c.sta, 1 - np.nan_to_num(c.label)))
    check("flipping every label changes no column", np.array_equal(E, E3, equal_nan=True))
    # hand-checked values on a two-stay cohort
    dyn = pd.DataFrame({"stay_id": [1, 1, 1, 1, 2, 2], "hour": [0.0, 1, 2, 3, 0, 1]})
    for v in C.DYNAMIC:
        dyn[v] = np.nan
    dyn["hr"] = [80, np.nan, 100, 90, 70, 75]
    dyn["lact"] = [1.0, np.nan, np.nan, 4.0, 2.0, np.nan]
    dyn["crea"] = [np.nan, 1.0, np.nan, np.nan, np.nan, 1.5]
    dyn["ph"] = [np.nan, np.nan, 7.3, np.nan, np.nan, np.nan]
    sta = pd.DataFrame({"stay_id": [1, 2], "age": [60.0, 70], "sex": [1.0, 0], "height": np.nan, "weight": np.nan})
    H, hn = EX.build(C.Cohort(dyn, sta, np.zeros(6)))

    def col(n):
        return H[:, hn.index(n)].astype(float)
    same = lambda a, b: np.allclose(a, b, equal_nan=True)  # noqa: E731
    check("dlast / dtlast: change and gap between the last two measurements, within the stay",
          same(col("lact__dlast"), [np.nan, np.nan, np.nan, 3, np.nan, np.nan])
          and same(col("lact__dtlast"), [np.nan, np.nan, np.nan, 3, np.nan, np.nan]))
    check("dmean: carried value minus the running mean of measurements",
          same(col("hr__dmean"), [0, 0, 10, 0, 0, 2.5]))
    check("slope6 on the carried value, per hour; std6 of it",
          same(col("hr__slope6")[:2], [np.nan, 0]) and same(col("hr__slope6")[4:], [np.nan, 5])
          and same(col("hr__std6")[4:], [0, 2.5]))
    check("labs drawn this hour, in 24 h, new lab types and types so far (ph, lact, crea are labs; hr is not)",
          same(col("proc__labs_now"), [1, 1, 1, 1, 1, 1]) and same(col("proc__labs24"), [1, 2, 3, 4, 1, 2])
          and same(col("proc__new_now"), [1, 1, 1, 0, 1, 1]) and same(col("proc__lab_types"), [1, 2, 3, 3, 1, 2]))
    check("hours since a blood gas, and gases in 24 h",
          same(col("proc__gas_since"), [FE.NEVER, FE.NEVER, 0, 1, FE.NEVER, FE.NEVER])
          and same(col("proc__gas24"), [0, 0, 1, 1, 0, 0]))
    # the side-by-side view
    A = np.arange(20, dtype=np.float32).reshape(5, 4)
    Bm = -np.arange(10, dtype=np.float32).reshape(5, 2)
    view = EX.Columns([A, Bm], keep=[np.array([0, 2]), None])
    rows = np.array([4, 0, 2])
    check("Columns gathers rows from every part, keeping the chosen columns",
          view.shape == (5, 4) and np.array_equal(view[rows], np.hstack([A[rows][:, [0, 2]], Bm[rows]]))
          and np.array_equal(view[3], np.r_[A[3, [0, 2]], Bm[3]]))
    try:
        np.asarray(view)
        whole = True
    except TypeError:
        whole = False
    check("... and refuses whole-matrix conversion", not whole)
    same_plan = all(EX.memory_plan(n, "rich", None, available_bytes=int(7.6e9), on_disk=d)
                    == FE.memory_plan(n, available_bytes=int(7.6e9), on_disk=d, feature_set="rich")
                    for n in (10_000, 6_000_000) for d in (True, False))
    check("the plan without an extension is features.memory_plan exactly", same_plan)
    O, on = EX.build(c, extension="obs")
    check("the obs extension is exactly temporal's obs columns for the common variables",
          on == [n for n in names if EX.in_families(n, ("obs_common",))]
          and np.array_equal(O, E[:, [names.index(n) for n in on]], equal_nan=True), f"{len(on)} columns")
    real_limit = FE._cgroup_limit
    FE._cgroup_limit = lambda: 7 * 1024 ** 3          # the worker's container cap
    try:
        fit_obs = EX.memory_plan(6_416_789, "rich", "obs", available_bytes=int(7.6e9), cohort_loaded=True)
        fit_all = EX.memory_plan(6_416_789, "rich", "temporal", available_bytes=int(7.6e9), cohort_loaded=True)
    finally:
        FE._cgroup_limit = real_limit
    check("full eICU under a 7 GiB cap: rich + obs is planned to fit, rich + temporal is not",
          fit_obs["fits"] is True and fit_all["fits"] is False,
          f"{fit_obs['peak_heap_gb']:.2f} vs {fit_all['peak_heap_gb']:.2f} GB")
    grown = EX.memory_plan(6_000_000, "rich", "temporal", available_bytes=int(7.6e9))
    check("... and an extension raises the planned fit peak", grown["peak_heap_gb"] >
          FE.memory_plan(6_000_000, available_bytes=int(7.6e9), feature_set="rich")["peak_heap_gb"],
          f"{grown['peak_heap_gb']:.1f} GB for {grown['features']} columns")
    # through bench: in memory, streamed, and from checkpoints and the shared store
    cfg = B.TrainConfig(max_rounds=60, early_stopping=10)
    mem = B.run(c, cfg, runs=[(0, 0), (1, 2)], feature_set="rich", extension="temporal")
    st = B.run(C.Cohort(c.dyn.copy(), c.sta, c.label), cfg, runs=[(0, 0), (1, 2)], feature_set="rich",
               extension="temporal", features_path=os.path.join(tmp, "Xe.npy"))
    check("bench hands LightGBM base + extension columns",
          mem["model"]["n_features"] == len(FE.feature_names("rich")) + len(names) and mem["model"]["extension"] == "temporal")
    check("a streamed fit with an extension scores exactly as in memory",
          [r["auroc"] for r in st["runs"]] == [r["auroc"] for r in mem["runs"]])
    store = os.path.join(tmp, "store")
    first = B.run(C.Cohort(c.dyn.copy(), c.sta, c.label), cfg, runs=[(0, 0)], feature_set="rich",
                  resume_dir=os.path.join(tmp, "e1"), feature_store=store, extension="temporal")
    plain = B.run(C.Cohort(c.dyn.copy(), c.sta, c.label), cfg, runs=[(0, 0)], feature_set="rich",
                  resume_dir=os.path.join(tmp, "e0"), feature_store=store)
    spec0 = CP.experiment_spec(c, {}, {}, "rich")
    spec1 = CP.experiment_spec(c, {}, {}, "rich", extension="temporal")
    check("the base matrix keeps one identity with or without an extension",
          CP.feature_spec(spec0) == CP.feature_spec(spec1) and len(os.listdir(store)) == 2,
          f"{len(os.listdir(store))} matrices in the store")
    check("... the extension has its own", CP.identity_of(CP.extension_spec(spec1)) != CP.identity_of(CP.feature_spec(spec1)))
    check("checkpointed with an extension scores as in memory",
          first["runs"][0]["val_auroc"] == mem["runs"][0]["val_auroc"]
          and plain["runs"][0]["val_auroc"] != first["runs"][0]["val_auroc"])


def bag_checks(c: C.Cohort, tmp: str) -> None:
    print("[3d] bag: boosters averaged per run")
    X, _ = FE.build(c, feature_set="rich")
    y = np.nan_to_num(c.label)
    n = len(y)
    tr, va, te = np.arange(n * 6 // 10), np.arange(n * 6 // 10, n * 8 // 10), np.arange(n * 8 // 10, n)
    cfg = B.TrainConfig(max_rounds=60, early_stopping=10)
    one = B._fit_predict(X, y, tr, va, te, cfg)
    three = B._fit_predict(X, y, tr, va, te, B.TrainConfig(max_rounds=60, early_stopping=10, bag=3))
    check("a bag of one is the single model (the default)", B.TrainConfig().bag == 1)
    check("a bag of three averages different boosters into probabilities",
          not np.array_equal(one["p"], three["p"]) and bool(((three["p"] >= 0) & (three["p"] <= 1)).all())
          and isinstance(three["rounds"], int) and three["rounds"] >= 1)
    check("... and is judged on its averaged validation scores", 0.0 <= three["val_auroc"] <= 1.0
          and three["val_auroc"] != one["val_auroc"])
    st = B._fit_predict(X, y, tr, va, te, B.TrainConfig(max_rounds=60, early_stopping=10, bag=3), stream=True)
    check("a streamed bag scores exactly as in memory", np.array_equal(st["p"], three["p"]) and st["val_auroc"] == three["val_auroc"])
    check("a bag below one is refused", refused(B._fit_predict, X, y, tr, va, te, B.TrainConfig(bag=0)))


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
        rich_checks(c)
        memory_checks(c)
        extension_checks(c, tmp)
        bag_checks(c, tmp)
        metric_checks()
        compare_checks()
        end_to_end(c, tmp)
    print("-" * 72)
    print(f"{sum(results)}/{len(results)} YAIB-protocol checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
