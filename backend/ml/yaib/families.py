"""Optional feature families: column blocks appended to a YAIB feature set, causal by construction.

A family is built once into its own matrix (a block) and read beside the feature set's
matrix row by row, so adding one never rebuilds, or changes the identity of, the rich or
trend matrix: ``features.py`` and ``cohort.py`` decide that identity, and this module is
neither. Every column of hour ``h`` reads hours <= ``h`` of the same stay only, nothing
reads the label, and no sum runs across stays (``verify_yaib_v4`` plants a future spike
and a neighbouring stay and checks no earlier hour moves).

    baseline  for the often-charted variables (``features.TREND_VARIABLES``):
              mean48, min48, max48, mean72   the last 2-3 days of measured values
              z_own                          the carried-forward value against the stay's own
                                             history so far, (value - mean) / SD: abnormal
                                             for THIS patient, the CinC lane's own-history idea
    ewm       exponentially weighted means of the carried-forward value per charted hour, at
              half-lives 3, 12 and 48 h, and the 3 h minus 48 h crossover (a smooth trend)
    ratios    bedside combinations a tree cannot form from two columns: MAP per heart beat,
              pulse pressure (and as a share of systolic), SpO2/FiO2, anion gap,
              neutrophil/lymphocyte ratio, the 48 h rise of creatinine, BUN and bilirubin
              (KDIGO-style), platelets against the stay's highest so far, lactate against its
              24 h peak, urine over 6 h per kg, |temperature - 37| and heart rate corrected
              for temperature
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.ml.yaib import features as FE
from backend.ml.yaib.cohort import GROUP, Cohort

FAMILIES = ("baseline", "ewm", "ratios")
BASELINE_PER_VARIABLE = ("__mean48", "__min48", "__max48", "__mean72", "__z_own")
EWM_HALF_LIVES = (3, 12, 48)
EWM_PER_VARIABLE = ("__ewm3", "__ewm12", "__ewm48", "__ewm3_48")
RATIOS = ("map_per_hr", "pulse_pressure", "pulse_pressure_frac", "sf_ratio", "anion_gap", "nlr",
          "crea_rise48", "bun_rise48", "bili_rise48", "plt_frac_max", "lact_vs_peak24", "urine6_per_kg",
          "temp_abs_dev", "hr_temp_corrected")


def family_names(family: str) -> List[str]:
    """A family's column names in build order, known before any data is read."""
    if family == "baseline":
        return [f"{v}{s}" for v in FE.TREND_VARIABLES for s in BASELINE_PER_VARIABLE]
    if family == "ewm":
        return [f"{v}{s}" for v in FE.TREND_VARIABLES for s in EWM_PER_VARIABLE]
    if family == "ratios":
        return list(RATIOS)
    raise ValueError(f"family must be one of {FAMILIES}, not {family!r}")


def canonical(families: Optional[Sequence[str]]) -> List[str]:
    """Families in their one column order; an unknown or repeated family is refused."""
    given = list(families or [])
    unknown = [f for f in given if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families {unknown}; choose from {FAMILIES}")
    if len(set(given)) != len(given):
        raise ValueError("a family is named twice")
    return [f for f in FAMILIES if f in given]


def _window_mean(x: np.ndarray, k: int, s0: np.ndarray) -> np.ndarray:
    """Mean of the measured values of hours h-k+1..h within the stay (NaN when none).
    Exact sums over the k lags, centred on the stay's first measurement, so nothing from
    another stay enters the rounding."""
    base = FE._first(x, s0)
    n = np.zeros(len(x))
    s1 = np.zeros(len(x))
    for lag in range(k):
        xl = FE._lag_within(x, lag, s0, np.nan) - base
        ok = np.isfinite(xl)
        n += ok
        s1 += np.where(ok, xl, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s1 / n + base, np.nan)


def _own_z(x: np.ndarray, ff: np.ndarray, stay: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """(carried-forward value - mean of the stay's measured values so far) / their SD.
    The running sums restart at each stay (a grouped cumulative sum), centred on the stay's
    first measurement; NaN under two measurements or with no spread."""
    base = FE._first(x, s0)
    seen = np.isfinite(x)
    xc = np.where(seen, x - base, 0.0)
    g = pd.DataFrame({"n": seen.astype(float), "s1": xc, "s2": xc * xc}).groupby(stay, sort=False).cumsum()
    n, s1, s2 = (g[c].to_numpy() for c in ("n", "s1", "s2"))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
        z = (ff - base - mean) / sd
    return np.where((n >= 2) & (sd > 0) & np.isfinite(z), z, np.nan)


def _ewm(ff: np.ndarray, s0: np.ndarray, half_life: float) -> np.ndarray:
    """Exponentially weighted mean of a carried-forward series, restarting at each stay,
    one step per row (per charted hour). It starts at the stay's first value; NaN before.
    The recursion runs over the rows' positions within their stays, all stays at once, so
    no state passes from one stay to the next."""
    alpha = 1.0 - 0.5 ** (1.0 / half_life)
    n = len(ff)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    pos = np.arange(n) - s0
    order = np.argsort(pos, kind="stable")
    counts = np.bincount(pos)
    start = 0
    for p, c in enumerate(counts):
        r = order[start:start + c]
        start += c
        x = ff[r]
        if p == 0:
            out[r] = x
            continue
        prev = out[r - 1]
        out[r] = np.where(np.isnan(prev), x, np.where(np.isnan(x), prev, alpha * x + (1.0 - alpha) * prev))
    return out


def _window_sum(x: np.ndarray, k: int, s0: np.ndarray) -> np.ndarray:
    """Sum of hours h-k+1..h within the stay, missing as 0 (urine output), exact over lags."""
    acc = np.zeros(len(x))
    for lag in range(k):
        acc += np.nan_to_num(FE._lag_within(x, lag, s0, 0.0), nan=0.0)
    return acc


def _columns(cohort: Cohort, family: str) -> List[Tuple[str, Callable[[], np.ndarray]]]:
    d = cohort.dyn
    stay = d[GROUP].to_numpy()
    s0 = FE._seg_index(stay)
    raw: Dict[str, np.ndarray] = {}
    carried: Dict[str, np.ndarray] = {}

    def x(v):
        if v not in raw:
            raw[v] = d[v].to_numpy(dtype=float)
        return raw[v]

    def ff(v):
        if v not in carried:
            carried[v] = FE._ffill(x(v), s0)
        return carried[v]

    cols: List[Tuple[str, Callable[[], np.ndarray]]] = []
    if family == "baseline":
        for v in FE.TREND_VARIABLES:
            with np.errstate(invalid="ignore"):
                cols += [
                    (f"{v}__mean48", lambda v=v: _window_mean(x(v), 48, s0)),
                    (f"{v}__min48", lambda v=v: FE._window(x(v), 48, s0, np.fmin)),
                    (f"{v}__max48", lambda v=v: FE._window(x(v), 48, s0, np.fmax)),
                    (f"{v}__mean72", lambda v=v: _window_mean(x(v), 72, s0)),
                    (f"{v}__z_own", lambda v=v: _own_z(x(v), ff(v), stay, s0)),
                ]
    elif family == "ewm":
        for v in FE.TREND_VARIABLES:
            e = {}

            def ewm(h, v=v, e=e):
                if h not in e:
                    e[h] = _ewm(ff(v), s0, h)
                return e[h]

            cols += [(f"{v}__ewm{h}", lambda h=h, ewm=ewm: ewm(h)) for h in EWM_HALF_LIVES]
            cols.append((f"{v}__ewm3_48", lambda ewm=ewm: ewm(3) - ewm(48)))
    elif family == "ratios":
        def rise48(v):
            with np.errstate(invalid="ignore"):
                return ff(v) - FE._window(x(v), 48, s0, np.fmin)

        def fio2_fraction():
            f = ff("fio2")
            return np.where(f > 1.0, f / 100.0, f)

        weight = cohort.sta.set_index(GROUP).reindex(stay)["weight"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols += [
                ("map_per_hr", lambda: ff("map") / ff("hr")),
                ("pulse_pressure", lambda: ff("sbp") - ff("dbp")),
                ("pulse_pressure_frac", lambda: (ff("sbp") - ff("dbp")) / ff("sbp")),
                ("sf_ratio", lambda: ff("o2sat") / fio2_fraction()),
                ("anion_gap", lambda: ff("na") - ff("cl") - ff("bicar")),
                ("nlr", lambda: ff("neut") / ff("lymph")),
                ("crea_rise48", lambda: rise48("crea")),
                ("bun_rise48", lambda: rise48("bun")),
                ("bili_rise48", lambda: rise48("bili")),
                ("plt_frac_max", lambda: ff("plt") / pd.Series(ff("plt")).groupby(stay, sort=False).cummax().to_numpy()),
                ("lact_vs_peak24", lambda: ff("lact") - FE._window(x("lact"), 24, s0, np.fmax)),
                ("urine6_per_kg", lambda: _window_sum(x("urine"), 6, s0) / weight),
                ("temp_abs_dev", lambda: np.abs(ff("temp") - 37.0)),
                ("hr_temp_corrected", lambda: ff("hr") - 10.0 * (ff("temp") - 37.0)),
            ]
    else:
        family_names(family)          # raises with the list of families
    return cols


def build_family(cohort: Cohort, family: str, out: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
    """(rows x columns) float32 block aligned with ``cohort.dyn``, and its column names.
    With ``out`` (inside the lake for credentialed cohorts) it is written column-major to a
    scratch file and copied row-major into ``out`` in row blocks, as ``features.build``
    does, so the result can be read a row batch at a time beside the feature set's matrix."""
    d = cohort.dyn
    stay = d[GROUP].to_numpy()
    hour = d["hour"].to_numpy(dtype=float)
    if len(stay) > 1 and np.any((stay[1:] == stay[:-1]) & (np.diff(hour) <= 0)):
        raise ValueError("rows must be sorted by stay and strictly increasing hour")
    expected = family_names(family)
    shape = (len(stay), len(expected))
    scratch = f"{out}.columns.npy" if out else None
    X = (np.lib.format.open_memmap(scratch, mode="w+", dtype=np.float32, shape=shape, fortran_order=True)
         if out else np.empty(shape, dtype=np.float32))
    names: List[str] = []
    for name, make in _columns(cohort, family):
        j = len(names)
        if j >= len(expected) or expected[j] != name:
            raise AssertionError(f"family column {j} built as {name!r}, declared as {expected[j:j + 1]}")
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.asarray(make(), dtype=float)
        X[:, j] = np.where(np.isfinite(a), a, np.nan)
        names.append(name)
    if names != expected:
        raise AssertionError("built fewer family columns than declared")
    if out:
        X.flush()
        rows = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=shape)
        for r in range(0, shape[0], FE.STREAM_ROWS):
            rows[r:r + FE.STREAM_ROWS] = X[r:r + FE.STREAM_ROWS]
        rows.flush()
        del X
        os.remove(scratch)
        X = rows
    return X, names


class Columns:
    """Row-aligned matrices read as one: blocks side by side, a row batch at a time.

    ``X[rows]`` (an index array, a slice or one row) returns those rows of every block,
    concatenated along columns, which is all the streamed fit and prediction read. The
    blocks stay memmapped; nothing is copied whole."""

    def __init__(self, blocks: Sequence[np.ndarray]):
        rows = {b.shape[0] for b in blocks}
        if len(rows) != 1:
            raise ValueError("blocks must have the same rows")
        self.blocks = list(blocks)
        self.shape = (rows.pop(), sum(b.shape[1] for b in blocks))
        self.dtype = np.dtype(np.float32)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            raise TypeError("index Columns by rows only")
        return np.concatenate([np.asarray(b[key]) for b in self.blocks], axis=-1)
