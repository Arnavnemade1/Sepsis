"""Columns that sit beside a ``features.py`` matrix, in a matrix of their own.

``features.py`` is deliberately left byte-for-byte unchanged by this module. The feature
store keys a matrix on the hash of ``features.py`` (``checkpoint.FEATURE_SOURCES``), and
the full-eICU ``rich`` matrix is 17.8 GB on a worker with about 16 GB free: one edited
line there would ask for a rebuild that does not fit. An extension is therefore built,
stored and validated as its own (much smaller) matrix, and ``bench`` hands LightGBM the
two side by side, row by row, without concatenating them on disk or on the heap.

The ``temporal`` extension adds the families of GENESIS's CinC 2019 representation
(``backend/ml/genesis/features.py``) that the ``rich`` set does not have, translated to
YAIB's 48 dynamic variables:

    obs      per variable: the change between its last two MEASUREMENTS (``dlast``) and
             the hours between them (``dtlast``). ``rich``'s deltas are hour-lagged
             differences of the carried-forward value, which read 0 whenever the
             variable was not re-measured inside the lag.
    dmean    per variable: the carried-forward value minus the stay's running mean of
             its measurements ("worse than this patient's usual"); a tree cannot
             subtract ``rich``'s ``__mean`` column from the value itself.
    roll     per vital sign: standard deviation and least-squares slope of the
             carried-forward value over the last 6 and 24 hours (variability and trend).
    process  care intensity, from when things were measured rather than their values:
             lab values drawn this hour / in 6 h / in 24 h, hours since the last draw,
             draw rate over the stay, lab types ordered for the first time (the work-up
             widening), distinct lab types so far, blood gases in 24 h and hours since
             the last, vital-sign charting this hour and over 6 h.

Every column of hour ``h`` reads rows ``<= h`` of the same stay only. Running sums
RESTART at each stay (a pandas group cumsum), never one cohort-wide cumulative sum whose
floating-point rounding would couple a stay to every earlier stay's values, futures
included (the defect GENESIS's audit caught at 1e-12). Rolling windows are explicit
backward shifts that return NaN across a stay boundary. ``verify_yaib`` rewrites every
stay's future and checks no earlier column moves.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.ml.yaib.cohort import DYNAMIC, GROUP, Cohort
from backend.ml.yaib.features import NEVER, STREAM_ROWS, _ffill, _lag_within, _seg_index

VITALS: Tuple[str, ...] = ("hr", "o2sat", "temp", "sbp", "dbp", "map", "resp")
#: drawn samples: everything charted that is neither a vital sign, a setting nor an output
LABS: Tuple[str, ...] = tuple(v for v in DYNAMIC if v not in VITALS and v not in ("urine", "fio2"))
#: an arterial blood gas is an escalation of care
GAS: Tuple[str, ...] = ("ph", "pco2", "po2", "be")
WINDOWS: Tuple[int, ...] = (6, 24)
#: variables that fewer than 5% of the stays of YAIB's open eICU demo (896 stays) measure
#: twice; ``dlast`` / ``dtlast`` need two measurements, so for these they are almost always
#: missing. Chosen from measurement counts alone (no label), on open data.
RARE: Tuple[str, ...] = ("cai", "crp", "bili_dir", "fgn", "bnd", "methb", "tnt", "ckmb")
COMMON: Tuple[str, ...] = tuple(v for v in DYNAMIC if v not in RARE)

PROCESS = ("proc__labs_now", "proc__labs6", "proc__labs24", "proc__draw_since", "proc__draws24",
           "proc__draws_all", "proc__draw_rate", "proc__new_now", "proc__new24", "proc__lab_types",
           "proc__gas24", "proc__gas_since", "proc__vitals_now", "proc__vitals6")
FAMILIES: Dict[str, List[str]] = {
    "obs": [f"{v}__{s}" for v in DYNAMIC for s in ("dlast", "dtlast")],
    "obs_common": [f"{v}__{s}" for v in COMMON for s in ("dlast", "dtlast")],
    "dmean": [f"{v}__dmean" for v in DYNAMIC],
    "roll": [f"{v}__{s}{w}" for v in VITALS for w in WINDOWS for s in ("std", "slope")],
    "process": list(PROCESS),
    # a subset of obs, named only so a screen can ask whether dtlast adds anything
    "dlast": [f"{v}__dlast" for v in DYNAMIC],
}
#: ``obs`` is the obs family over the 40 common variables: 80 columns, which is what fits
#: beside ``rich`` under the worker's 7 GiB container cap on full eICU (``memory_plan``)
EXTENSIONS: Dict[str, Tuple[str, ...]] = {"temporal": ("obs", "dmean", "roll", "process"),
                                          "obs": ("obs_common",)}


def extension_names(extension: str) -> List[str]:
    """Column names in build order, known before any data is read."""
    if extension not in EXTENSIONS:
        raise ValueError(f"extension must be one of {tuple(EXTENSIONS)}, not {extension!r}")
    return [name for family in EXTENSIONS[extension] for name in FAMILIES[family]]


def in_families(name: str, families: Sequence[str]) -> bool:
    """Whether a column belongs to any of ``families`` (a column can be in two: ``obs``
    and ``obs_common`` share the common variables' columns)."""
    return any(name in FAMILIES[f] for f in families)


class _Stay:
    """Row bookkeeping for one cohort: stay starts, row positions, hours."""

    def __init__(self, stay: np.ndarray, hour: np.ndarray):
        self.stay, self.hour = stay, hour
        self.n = len(stay)
        self.idx = np.arange(self.n)
        self.s0 = _seg_index(stay)

    def prefix(self, x: np.ndarray) -> np.ndarray:
        """Stay-local running sum, restarting at every stay."""
        return pd.Series(np.asarray(x, dtype=np.float64)).groupby(self.stay, sort=False).cumsum().to_numpy()

    def window(self, P: np.ndarray, w: Optional[int]) -> np.ndarray:
        """Sum over the last ``w`` rows of the stay ending at each row (``None``: whole stay so far)."""
        if w is None:
            return P
        lo = np.maximum(self.idx - w + 1, self.s0)
        return P - np.where(lo > self.s0, P[np.maximum(lo - 1, 0)], 0.0)

    def last(self, seen: np.ndarray) -> np.ndarray:
        """Row of the latest ``seen`` row at or before each row, within the stay (-1: none)."""
        last = np.maximum.accumulate(np.where(seen, self.idx, -1))
        last[last < self.s0] = -1
        return last

    def since(self, seen: np.ndarray) -> np.ndarray:
        last = self.last(seen)
        return np.where(last >= 0, self.hour - self.hour[np.maximum(last, 0)], NEVER)


def build(cohort: Cohort, out: Optional[str] = None, extension: str = "temporal") -> Tuple[np.ndarray, List[str]]:
    """(rows x columns) float32 matrix aligned with ``cohort.dyn``, and its column names.

    Same storage contract as ``features.build``: preallocated, and with ``out`` written
    column-major to a scratch file first and then copied row-major to ``out`` in blocks."""
    d = cohort.dyn
    stay = d[GROUP].to_numpy()
    hour = d["hour"].to_numpy(dtype=float)
    if len(stay) > 1 and np.any((stay[1:] == stay[:-1]) & (np.diff(hour) <= 0)):
        raise ValueError("rows must be sorted by stay and strictly increasing hour")
    S = _Stay(stay, hour)
    expected = extension_names(extension)
    shape = (S.n, len(expected))
    scratch = f"{out}.columns.npy" if out else None
    X = (np.lib.format.open_memmap(scratch, mode="w+", dtype=np.float32, shape=shape, fortran_order=True)
         if out else np.empty(shape, dtype=np.float32))
    names: List[str] = []

    def add(name: str, a: np.ndarray) -> None:
        j = len(names)
        if j >= len(expected) or expected[j] != name:
            raise AssertionError(f"extension column {j} built as {name!r}, declared as {expected[j:j + 1]}")
        X[:, j] = a
        names.append(name)

    def values(v: str) -> np.ndarray:
        return d[v].to_numpy(dtype=float)

    families = EXTENSIONS[extension]
    obs_vars = DYNAMIC if "obs" in families else COMMON if "obs_common" in families else ()
    if obs_vars:
        # the latest strictly-earlier measurement of each row: shift the running "last seen" by one
        for v in obs_vars:
            x = values(v)
            seen = np.isfinite(x)
            last = S.last(seen)
            prior = np.r_[-1, np.maximum.accumulate(np.where(seen, S.idx, -1))[:-1]]
            prev = np.where(last >= 0, prior[np.maximum(last, 0)], -1)
            prev[prev < S.s0] = -1
            ok = prev >= 0
            add(f"{v}__dlast", np.where(ok, x[np.maximum(last, 0)] - x[np.maximum(prev, 0)], np.nan))
            add(f"{v}__dtlast", np.where(ok, hour[np.maximum(last, 0)] - hour[np.maximum(prev, 0)], np.nan))
    if "dmean" in families:
        for v in DYNAMIC:
            x = values(v)
            seen = np.isfinite(x)
            cnt = S.prefix(seen)
            tot = S.prefix(np.where(seen, x, 0.0))
            ff = _ffill(x, S.s0)
            with np.errstate(invalid="ignore", divide="ignore"):
                add(f"{v}__dmean", np.where(cnt > 0, ff - tot / np.maximum(cnt, 1), np.nan))
    if "roll" in families:
        for v in VITALS:
            ff = _ffill(values(v), S.s0)
            for w in WINDOWS:
                n = np.zeros(S.n); sx = np.zeros(S.n); sy = np.zeros(S.n)
                sxx = np.zeros(S.n); sxy = np.zeros(S.n); syy = np.zeros(S.n)
                for lag in range(w):
                    y = _lag_within(ff, lag, S.s0, np.nan) if lag else ff
                    ok = np.isfinite(y)
                    yz = np.where(ok, y, 0.0)
                    t = -float(lag)
                    n += ok; sx += ok * t; sy += yz; sxx += ok * t * t; sxy += yz * t; syy += yz * yz
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = sy / np.maximum(n, 1)
                    var = np.where(n > 0, syy / np.maximum(n, 1) - mean * mean, np.nan)
                    den = n * sxx - sx * sx
                    slope = np.where(den > 1e-9, (n * sxy - sx * sy) / np.where(den > 1e-9, den, 1.0), np.nan)
                add(f"{v}__std{w}", np.sqrt(np.clip(var, 0.0, None)))
                add(f"{v}__slope{w}", slope)
    if "process" in families:
        lab_now = np.zeros(S.n)
        new = np.zeros(S.n)
        types = np.zeros(S.n)
        for v in LABS:
            seen = np.isfinite(values(v))
            lab_now += seen
            so_far = S.prefix(seen)
            new += seen & (so_far == 1)
            types += so_far > 0
        Cl = S.prefix(lab_now)
        draw = lab_now > 0
        Cd = S.prefix(draw)
        Cn = S.prefix(new)
        gas = np.zeros(S.n, dtype=bool)
        for v in GAS:
            gas |= np.isfinite(values(v))
        vit = np.zeros(S.n)
        for v in VITALS:
            vit += np.isfinite(values(v))
        elapsed = S.idx - S.s0 + 1.0
        add("proc__labs_now", lab_now)
        add("proc__labs6", S.window(Cl, 6))
        add("proc__labs24", S.window(Cl, 24))
        add("proc__draw_since", S.since(draw))
        add("proc__draws24", S.window(Cd, 24))
        add("proc__draws_all", Cd)
        add("proc__draw_rate", Cd / elapsed)
        add("proc__new_now", new)
        add("proc__new24", S.window(Cn, 24))
        add("proc__lab_types", types)
        add("proc__gas24", S.window(S.prefix(gas), 24))
        add("proc__gas_since", S.since(gas))
        add("proc__vitals_now", vit)
        add("proc__vitals6", S.window(S.prefix(vit), 6))
    if names != expected:
        raise AssertionError("built fewer extension columns than declared")
    if out:
        X.flush()
        rows = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=shape)
        for r in range(0, shape[0], STREAM_ROWS):
            rows[r:r + STREAM_ROWS] = X[r:r + STREAM_ROWS]
        rows.flush()
        del X
        os.remove(scratch)
        X = rows
    return X, names


def memory_plan(rows: int, feature_set: str = "rich", extension: Optional[str] = None,
                train_fraction: float = 0.64, available_bytes: Optional[int] = None, on_disk: bool = True,
                cohort_loaded: bool = False) -> dict:
    """``features.memory_plan`` for a base feature set plus an extension.

    The same terms, counted over every column LightGBM is handed (binned data, sampled
    column buffers, histograms and batches all scale with it); ``verify_yaib`` checks that
    without an extension this equals ``features.memory_plan`` exactly. Building the
    extension happens while the cohort frame is loaded, with fewer working columns than
    the base build, so it does not raise the build-phase peak."""
    from backend.ml.yaib.features import DATASET_BATCH_ROWS, _cgroup_limit, _mem_available, feature_names
    from backend.ml.yaib.cohort import DYNAMIC as _DYN
    cols = len(feature_names(feature_set)) + (len(extension_names(extension)) if extension else 0)
    per_row = 4 * cols
    matrix = rows * per_row
    train = int(rows * train_fraction) * per_row
    valid = int(rows * train_fraction / 4) * per_row
    binned = int(rows * train_fraction * 1.25) * cols
    frame = rows * (len(_DYN) + 4) * 8
    batch = min(rows, STREAM_ROWS, DATASET_BATCH_ROWS) * per_row
    sampled = min(int(rows * train_fraction), 200000)
    sampled_columns = sampled * cols * (8 + 4)
    sampling_peak = sampled * cols * (8 + 8 + 4)
    gradients = int(rows * train_fraction) * 16
    histograms = cols * 255 * 16 * 63
    if on_disk:
        build_peak = frame + (len(_DYN) + 12) * rows * 8
        fit_peak = max(sampling_peak, binned + sampled_columns + batch + 3 * rows * 8 + gradients + histograms)
        heap = max(build_peak, fit_peak)
    else:
        heap = frame + binned + matrix + train + valid
    if available_bytes is None:
        available_bytes = _mem_available()
    additional = max(0, heap - frame) if cohort_loaded else heap
    limit = _cgroup_limit()
    fits = None if available_bytes is None else additional < 0.85 * available_bytes
    if limit is not None and heap >= 0.85 * limit:
        fits = False
    return {"rows": rows, "features": cols, "matrix_gb": matrix / 1e9, "peak_heap_gb": heap / 1e9,
            "available_gb": None if available_bytes is None else available_bytes / 1e9,
            "fits": fits, "additional_heap_gb": additional / 1e9, "cohort_loaded": cohort_loaded,
            "cgroup_limit_gb": None if limit is None else limit / 1e9,
            "matrix_on_disk": on_disk, "dataset_batch_rows": min(STREAM_ROWS, DATASET_BATCH_ROWS),
            "sampled_columns_gb": sampled_columns / 1e9, "sampling_peak_gb": sampling_peak / 1e9}


class Columns:
    """Row-aligned matrices read side by side, optionally keeping only some columns.

    ``view[rows]`` gathers those rows from every part and concatenates them, so a
    LightGBM ``Sequence`` batch or a prediction batch is the only copy ever made; the
    parts themselves (memmaps of 18 GB and a few GB on full eICU) are never joined."""

    def __init__(self, parts: Sequence[np.ndarray], keep: Optional[Sequence[Optional[np.ndarray]]] = None):
        if not parts:
            raise ValueError("no parts")
        rows = parts[0].shape[0]
        if any(p.shape[0] != rows or p.ndim != 2 for p in parts):
            raise ValueError("parts must be 2-D with the same rows")
        self.parts = list(parts)
        self.keep = [None if k is None else np.asarray(k, dtype=np.int64) for k in (keep or [None] * len(parts))]
        if len(self.keep) != len(self.parts):
            raise ValueError("one keep entry per part")
        widths = [p.shape[1] if k is None else len(k) for p, k in zip(self.parts, self.keep)]
        self.shape = (rows, int(sum(widths)))
        self.ndim, self.dtype = 2, np.dtype(np.float32)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows):
        pieces = []
        for p, k in zip(self.parts, self.keep):
            block = np.asarray(p[rows], dtype=np.float32)
            pieces.append(block if k is None else block[..., k])
        return np.concatenate(pieces, axis=-1)

    def __array__(self, *args, **kwargs):
        raise TypeError("Columns is read in row batches, never converted whole")
