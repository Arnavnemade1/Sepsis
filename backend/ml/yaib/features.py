"""Our hourly features from a YAIB cohort's own variables, causal by construction.

Every feature of hour ``h`` is a function of the stay's rows at hours <= ``h`` only:
forward fills, cumulative sums and lagged differences, all within the stay. Nothing reads
the label, and nothing crosses from one stay into the next (``verify_yaib`` plants a
future spike and a neighbouring stay and checks no earlier hour moves).

What YAIB's own ML baselines see (``generate_features=True``): the forward-filled value,
and the min, max, mean and count of each variable over the whole history so far. This
set keeps those and adds what a clinician reads off a chart:

    since      hours since the variable was last measured (missingness is informative)
    mean24     mean of the measured values over the last 24 hours
    count24    how many times it was measured in the last 24 hours (care intensity)
    delta6     the change in the carried-forward value over 6 hours (trend)
    composite  shock index, MAP < 65, lactate >= 2 / >= 4, SIRS and qSOFA-style counts,
               computed from the carried-forward values of the same hour

Values are kept in the cohort's units; trees do not need them scaled.

The ``rich`` set appends, after every ``base`` column (the base columns are unchanged):

    min6/max6  extremes of the measured values over the last 6 hours
    delta1/24  change in the carried-forward value over 1 and 24 hours
    dev_first  carried-forward value minus the stay's first measurement (a tree cannot
               subtract two columns; "worse than on arrival" has to be handed to it)
    organ      a SOFA proxy from what the cohort carries (P/F ratio, platelets,
               bilirubin, MAP, creatinine; no GCS, no vasopressor doses), its 24-hour
               change and its rise above the stay's best so far, since Sepsis-3 itself
               is an acute SOFA rise; urine output over 6 and 24 hours; BUN/creatinine

All of them still read hours <= h of the same stay only.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.ml.yaib.cohort import DYNAMIC, GROUP, STATIC, Cohort

NEVER = 1e4                     # "hours since" for a variable never measured yet


def _segments(stay: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(np.r_[True, stay[1:] != stay[:-1]])
    return starts, np.r_[starts[1:], len(stay)]


def _seg_index(stay: np.ndarray) -> np.ndarray:
    """Row index of each row's stay start."""
    starts, ends = _segments(stay)
    return np.repeat(starts, ends - starts)


def _ffill(v: np.ndarray, seg_start: np.ndarray) -> np.ndarray:
    """Last observed value at or before each row, within the stay (NaN before the first)."""
    n = len(v)
    idx = np.where(np.isfinite(v), np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    ok = idx >= seg_start                # the carried index must belong to this stay
    return np.where(ok, v[np.maximum(idx, 0)], np.nan)


def _cum_within(x: np.ndarray, seg_start: np.ndarray) -> np.ndarray:
    """Cumulative sum restarting at each stay."""
    c = np.cumsum(x, dtype=float)
    base = np.where(seg_start > 0, c[np.maximum(seg_start - 1, 0)], 0.0)
    return c - base


def _lag_within(c: np.ndarray, k: int, seg_start: np.ndarray, fill: float) -> np.ndarray:
    """``c`` at row - k when that row is in the same stay, else ``fill``."""
    n = len(c)
    j = np.arange(n) - k
    ok = j >= seg_start
    return np.where(ok, c[np.maximum(j, 0)], fill)


PER_VARIABLE = ("", "__since", "__count", "__min", "__max", "__mean", "__mean24", "__count24", "__delta6")
COMPOSITE = ("shock_index", "map_lt65", "lact_ge2", "lact_ge4", "sirs_count", "qsofa_vitals")
RICH_PER_VARIABLE = ("__min6", "__max6", "__delta1", "__delta24", "__dev_first")
RICH_COMPOSITE = ("pf_ratio", "sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_renal",
                  "sofa_proxy", "sofa_proxy_delta24", "sofa_proxy_rise", "urine6", "urine24", "bun_crea")
FEATURE_SETS = ("base", "rich")


def feature_names(feature_set: str = "base") -> List[str]:
    """Column names in build order, known before any data is read (to preallocate)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, not {feature_set!r}")
    base = (["hour"] + [f"{v}{s}" for v in DYNAMIC for s in PER_VARIABLE] + list(COMPOSITE)
            + [f"static_{v}" for v in STATIC])
    if feature_set == "base":
        return base
    return base + [f"{v}{s}" for v in DYNAMIC for s in RICH_PER_VARIABLE] + list(RICH_COMPOSITE)


def _window(v: np.ndarray, k: int, seg_start: np.ndarray, op) -> np.ndarray:
    """``op`` (np.fmin / np.fmax) over the measured values of hours h-k+1..h, within the stay."""
    acc = v.copy()
    for lag in range(1, k):
        acc = op(acc, _lag_within(v, lag, seg_start, np.nan))
    return acc


def _first(v: np.ndarray, seg_start: np.ndarray) -> np.ndarray:
    """The stay's first measured value, from the hour it is measured on (NaN before)."""
    n = len(v)
    idx = np.where(np.isfinite(v), np.arange(n), n)
    # first measured index at or after each stay start, carried to the stay's rows
    nxt = np.minimum.accumulate(idx[::-1])[::-1]
    first_idx = nxt[seg_start]
    ok = (first_idx < n) & (np.arange(n) >= first_idx)
    return np.where(ok, v[np.minimum(first_idx, n - 1)], np.nan)


def _band(x: np.ndarray, cuts, higher_is_worse: bool) -> np.ndarray:
    """SOFA-style points 0-4 from cut points; 0 where unmeasured (SOFA's convention)."""
    pts = np.zeros(len(x))
    ok = np.isfinite(x)
    for i, c in enumerate(cuts, start=1):
        hit = ok & ((x >= c) if higher_is_worse else (x < c))
        pts = np.where(hit, float(i), pts)
    return pts


def build(cohort: Cohort, out: Optional[str] = None, feature_set: str = "base") -> Tuple[np.ndarray, List[str]]:
    """(rows x features) float32 matrix aligned with ``cohort.dyn`` and the column names.

    Columns are written straight into one preallocated array, so peak memory is the
    matrix plus a few working columns, not twice the matrix. With ``out`` (a path; keep
    it inside the lake for credentialed cohorts) the matrix is a disk-backed memmap and
    the page cache, not the heap, holds it."""
    d = cohort.dyn
    stay = d[GROUP].to_numpy()
    hour = d["hour"].to_numpy(dtype=float)
    if len(stay) > 1 and np.any((stay[1:] == stay[:-1]) & (np.diff(hour) <= 0)):
        raise ValueError("rows must be sorted by stay and strictly increasing hour")
    s0 = _seg_index(stay)
    expected = feature_names(feature_set)
    shape = (len(stay), len(expected))
    X = (np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=shape) if out
         else np.empty(shape, dtype=np.float32))
    names: List[str] = []

    def add(name: str, a: np.ndarray) -> None:
        j = len(names)
        if j >= len(expected) or expected[j] != name:
            raise AssertionError(f"feature {j} built as {name!r}, declared as {expected[j:j + 1]}")
        X[:, j] = a
        names.append(name)

    add("hour", hour)
    last = {}
    for v in DYNAMIC:
        x = d[v].to_numpy(dtype=float)
        seen = np.isfinite(x)
        ff = _ffill(x, s0)
        last[v] = ff
        t_obs = _ffill(np.where(seen, hour, np.nan), s0)
        cnt = _cum_within(seen.astype(float), s0)
        tot = _cum_within(np.where(seen, x, 0.0), s0)
        # expanding min / max of measured values: accumulate on the carried-forward series
        lo = pd.Series(ff).groupby(stay, sort=False).cummin().to_numpy()
        hi = pd.Series(ff).groupby(stay, sort=False).cummax().to_numpy()
        c24 = cnt - _lag_within(cnt, 24, s0, 0.0)
        t24 = tot - _lag_within(tot, 24, s0, 0.0)
        add(f"{v}", ff)
        add(f"{v}__since", np.where(np.isfinite(t_obs), hour - t_obs, NEVER))
        add(f"{v}__count", cnt)
        add(f"{v}__min", lo)
        add(f"{v}__max", hi)
        add(f"{v}__mean", np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan))
        add(f"{v}__mean24", np.where(c24 > 0, t24 / np.maximum(c24, 1), np.nan))
        add(f"{v}__count24", c24)
        add(f"{v}__delta6", ff - _lag_within(ff, 6, s0, np.nan))

    hr, sbp, mp, lac = last["hr"], last["sbp"], last["map"], last["lact"]
    temp, resp, wbc = last["temp"], last["resp"], last["wbc"]
    with np.errstate(invalid="ignore", divide="ignore"):
        add("shock_index", hr / sbp)
        add("map_lt65", np.where(np.isfinite(mp), (mp < 65).astype(float), np.nan))
        add("lact_ge2", np.where(np.isfinite(lac), (lac >= 2).astype(float), np.nan))
        add("lact_ge4", np.where(np.isfinite(lac), (lac >= 4).astype(float), np.nan))
        sirs = ((temp > 38) | (temp < 36)).astype(float) + (hr > 90) + (resp > 20) + ((wbc > 12) | (wbc < 4))
        add("sirs_count", sirs)
        add("qsofa_vitals", (resp >= 22).astype(float) + (sbp <= 100))

    st = cohort.sta.set_index(GROUP).reindex(stay)
    for v in STATIC:
        add(f"static_{v}", st[v].to_numpy(dtype=float))

    if feature_set == "rich":
        for v in DYNAMIC:
            x = d[v].to_numpy(dtype=float)
            ff = last[v]
            with np.errstate(invalid="ignore"):
                add(f"{v}__min6", _window(x, 6, s0, np.fmin))
                add(f"{v}__max6", _window(x, 6, s0, np.fmax))
            add(f"{v}__delta1", ff - _lag_within(ff, 1, s0, np.nan))
            add(f"{v}__delta24", ff - _lag_within(ff, 24, s0, np.nan))
            add(f"{v}__dev_first", ff - _first(x, s0))
        fio2 = last["fio2"]
        with np.errstate(invalid="ignore", divide="ignore"):
            pf = last["po2"] / np.where(fio2 > 1.0, fio2 / 100.0, fio2)      # fio2 in % or as a fraction
            pf = np.where(np.isfinite(pf), pf, np.nan)
            resp = _band(pf, (400, 300, 200, 100), higher_is_worse=False)
            coag = _band(last["plt"], (150, 100, 50, 20), higher_is_worse=False)
            liver = _band(last["bili"], (1.2, 2.0, 6.0, 12.0), higher_is_worse=True)
            cardio = np.where(np.isfinite(mp) & (mp < 70), 1.0, 0.0)
            renal = _band(last["crea"], (1.2, 2.0, 3.5, 5.0), higher_is_worse=True)
            sofa = resp + coag + liver + cardio + renal
            best = pd.Series(sofa).groupby(stay, sort=False).cummin().to_numpy()
            urine = np.nan_to_num(d["urine"].to_numpy(dtype=float), nan=0.0)
            cu = _cum_within(urine, s0)
            add("pf_ratio", pf)
            add("sofa_resp", resp)
            add("sofa_coag", coag)
            add("sofa_liver", liver)
            add("sofa_cardio", cardio)
            add("sofa_renal", renal)
            add("sofa_proxy", sofa)
            add("sofa_proxy_delta24", sofa - _lag_within(sofa, 24, s0, 0.0))
            add("sofa_proxy_rise", sofa - best)
            add("urine6", cu - _lag_within(cu, 6, s0, 0.0))
            add("urine24", cu - _lag_within(cu, 24, s0, 0.0))
            add("bun_crea", last["bun"] / last["crea"])
    if names != expected:
        raise AssertionError("built fewer features than declared")
    if out:
        X.flush()
    return X, names


BYTES_PER_ROW = 4 * len(feature_names())
#: rows handed to LightGBM per batch when streaming from a memmap, and predicted per batch
STREAM_ROWS = 250_000


def memory_plan(rows: int, train_fraction: float = 0.64, available_bytes: Optional[int] = None,
                on_disk: bool = True, feature_set: str = "base") -> dict:
    """Peak resident bytes of one ``bench.run`` fit, from the row count alone.

    In memory: the matrix + the training and validation rows' copies handed to LightGBM +
    LightGBM's binned datasets (about one byte per cell). Memmapped (``on_disk``): the
    matrix stays in the page cache and LightGBM is fed and scored in batches of
    ``STREAM_ROWS``, so neither the matrix nor a row copy is on the heap; only the binned
    datasets and one batch are. ``fits`` is judged against ``available_bytes`` (default:
    this host's MemAvailable)."""
    cols = len(feature_names(feature_set))
    per_row = 4 * cols
    matrix = rows * per_row
    train = int(rows * train_fraction) * per_row
    valid = int(rows * train_fraction / 4) * per_row
    binned = int(rows * train_fraction * 1.25) * cols
    frame = rows * (len(DYNAMIC) + 4) * 8          # the loaded cohort itself (float64 columns)
    batch = min(rows, STREAM_ROWS) * per_row
    if on_disk:
        # bench.run drops the cohort frame once the memmap is written, so the build phase
        # (frame + a dozen float64 working columns) and the fit phase (binned + one batch +
        # ids and labels) peak at different times
        build_peak = frame + 12 * rows * 8
        fit_peak = binned + batch + 3 * rows * 8
        heap = max(build_peak, fit_peak)
    else:
        heap = frame + binned + matrix + train + valid
    if available_bytes is None:
        available_bytes = _mem_available()
    return {"rows": rows, "features": cols, "matrix_gb": matrix / 1e9, "peak_heap_gb": heap / 1e9,
            "available_gb": None if available_bytes is None else available_bytes / 1e9,
            "fits": None if available_bytes is None else heap < 0.85 * available_bytes,
            "matrix_on_disk": on_disk}


def _mem_available() -> Optional[int]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
