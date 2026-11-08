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


def feature_names() -> List[str]:
    """Column names in build order, known before any data is read (to preallocate)."""
    return (["hour"] + [f"{v}{s}" for v in DYNAMIC for s in PER_VARIABLE] + list(COMPOSITE)
            + [f"static_{v}" for v in STATIC])


def build(cohort: Cohort, out: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
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
    expected = feature_names()
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
    if names != expected:
        raise AssertionError("built fewer features than declared")
    if out:
        X.flush()
    return X, names


BYTES_PER_ROW = 4 * len(feature_names())


def memory_plan(rows: int, train_fraction: float = 0.64, available_bytes: Optional[int] = None,
                on_disk: bool = True) -> dict:
    """Peak resident bytes of one ``bench.run`` fit, from the row count alone.

    Matrix (heap, or page cache when memmapped) + the training rows' copy handed to
    LightGBM + LightGBM's binned dataset (about one byte per cell) + the validation copy.
    ``fits`` is judged against ``available_bytes`` (default: this host's MemAvailable)."""
    matrix = rows * BYTES_PER_ROW
    train = int(rows * train_fraction) * BYTES_PER_ROW
    valid = int(rows * train_fraction / 4) * BYTES_PER_ROW
    binned = int(rows * train_fraction * 1.25) * len(feature_names())
    frame = rows * (len(DYNAMIC) + 4) * 8          # the loaded cohort itself (float64 columns)
    heap = frame + train + valid + binned + (0 if on_disk else matrix)
    if available_bytes is None:
        available_bytes = _mem_available()
    return {"rows": rows, "matrix_gb": matrix / 1e9, "peak_heap_gb": heap / 1e9,
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
