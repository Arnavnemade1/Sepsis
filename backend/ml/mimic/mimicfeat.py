"""The label's own structure, as causal features — each at the tier allowed to see it.

Sepsis-3 is a formula: suspicion of infection AND an organ-dysfunction rise of >= 2 SOFA
points above a MEASURED baseline (``sepsis3``). The GENESIS feature engine sees the hourly
24-h-worst SOFA totals and their deltas, but not the formula's ingredients — which organ
moved, against which measured baseline, and whether a culture and an antibiotic have
already paired into "suspected infection". A model has to re-learn those from raw values.
This block computes them, strictly from the past:

    SOFA_P__{organ}_pts / _rise   tier P: per-organ points from physiology alone (P/F,
                                   MAP) and the measured rise against the preceding 48 h
    SOFA_P__rise_total, __since_rise2
    SOFA__{organ}_pts / _rise ...  tier T: the same with vasopressors and ventilation
    Antibiotic__si_known           tier S: a culture and an antibiotic have paired inside
    Antibiotic__since_si            the Sepsis-3 windows (culture -> antibiotic <= 72 h, or
                                   antibiotic -> culture <= 24 h), known once BOTH exist
    UrineOutput__mlkgh{6,12,24}    tier P: urine output per kg per hour (KDIGO oliguria)
    Neutrophils__nlr               tier P: neutrophil / lymphocyte ratio
    Norepinephrine__nee(_max24)    tier T: norepinephrine-equivalent vasopressor dose
    {X}__qv{w}, {X}__levy_{Y}_{w}  tier P: the level-2 PATH SIGNATURE of the physiology over the
                                   last w hours (w = 6, 24) — each stream's quadratic
                                   variation (its lead-lag area) and each pair's Lévy area,
                                   the signed area that says which stream moved FIRST (heart
                                   rate before blood pressure, or after). The CinC 2019 winner
                                   (Morrill et al.) fed order-3 lead-lag signatures to LightGBM;
                                   the engine here had no cross-stream temporal term at all.

Tier comes from the prefix (``ledger.channel_of``): the firewall and the strict tier treat
these columns exactly like the channel they are named after. Reading the formula is what
the definitional oracle (``crossexam``) is there to measure; these features let a model do
it well, and the harness still reports how much of the AUROC is only that.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from backend.ml.mimic import sepsis3 as S3

ORGANS = ["resp", "coag", "liver", "cardio", "cns", "renal"]
#: the streams whose signature is taken, and the windows
LEVY_CH = ["HR", "MAP_min", "Resp", "Temp", "O2Sat", "SOFA_P", "Lactate"]
LEVY_WINDOWS = (6, 24)
#: norepinephrine equivalents (Goradia et al., J Crit Care 2021): mcg/kg/min, vasopressin U/min x 2.5
NEE = {"Norepinephrine": 1.0, "Epinephrine": 1.0, "Dopamine": 0.01, "Phenylephrine": 0.1}
VASOPRESSIN_U_PER_H = 2.5 / 60.0


def _first_true_since(flag: np.ndarray, seg: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Hours since ``flag`` was FIRST true in the stay (NaN before it ever was)."""
    big = np.iinfo(np.int64).max
    first = pd.Series(np.where(flag, idx, big)).groupby(seg, sort=False).cummin().to_numpy()
    return np.where(first <= idx, idx - first, np.nan).astype(np.float64)


def _window_sum(v: np.ndarray, rs: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """(sum of finite values, count of finite values) over the stay's last ``w`` rows."""
    n = len(v)
    idx = np.arange(n)
    s = np.zeros(n)
    c = np.zeros(n)
    for lag in range(w):
        src = idx - lag
        ok = src >= rs
        vv = np.full(n, np.nan)
        vv[ok] = v[src[ok]]
        f = np.isfinite(vv)
        s += np.where(f, vv, 0.0)
        c += f
    return s, c


def _locf(v: np.ndarray, rs: np.ndarray) -> np.ndarray:
    idx = np.arange(len(v))
    last = np.maximum.accumulate(np.where(np.isfinite(v), idx, -1))
    return np.where((last >= rs) & (last >= 0), v[np.maximum(last, 0)], np.nan)


def suspicion_known(cult: np.ndarray, abx: np.ndarray, seg: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """The hour (per row, its stay's) at which a culture / antibiotic pair first completes
    inside the Sepsis-3 windows — NaN if never. Completion = the LATER of the two events, the
    first moment both are known."""
    out = np.full(len(seg), np.nan)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], len(seg)]
    for a, b in zip(starts, ends):
        c_h = hour[a:b][cult[a:b]]
        a_h = hour[a:b][abx[a:b]]
        if not len(c_h) or not len(a_h):
            continue
        C, A = c_h[:, None].astype(float), a_h[None, :].astype(float)
        paired = ((A >= C) & (A <= C + 72)) | ((C >= A) & (C <= A + 24))
        if paired.any():
            out[a:b] = np.min(np.maximum(C, A)[paired])
    return out


def _stay_cumsum(x: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """A running sum that RESTARTS at every stay (one cohort-wide cumsum would couple patients
    through floating-point rounding — the GENESIS bug the admission audit exists to catch)."""
    return pd.Series(np.asarray(x, dtype=np.float64)).groupby(seg, sort=False).cumsum().to_numpy()


def signature_block(store, channels=None, windows=LEVY_WINDOWS) -> Tuple[List[np.ndarray], List[str]]:
    """Level-2 path signature terms over sliding windows, EXACT for the piecewise-linear
    path of each stream's carried-forward values (increments are 0 where either end is
    missing). With X^ the path and d its increments, over the window (a, t]:

        quadratic variation  QV = sum d^2
        Levy area  A_ij = 1/2 sum_r [(X^_i(r-1) - X^_i(a)) d_j(r) - (X^_j(r-1) - X^_j(a)) d_i(r)]

    both from stay-local prefix sums, so every term is O(N) and strictly causal."""
    channels = [c for c in (channels or LEVY_CH) if c in store.columns]
    rs, seg = store.row_start, store.seg
    N = store.n_rows
    idx = np.arange(N)
    X, D = {}, {}
    for c in channels:
        lv = _locf(store.col(c).astype(np.float64), rs)
        prev = np.full(N, np.nan)
        ok = idx - 1 >= rs
        prev[ok] = lv[idx[ok] - 1]
        d = np.where(np.isfinite(lv) & np.isfinite(prev), lv - prev, 0.0)
        D[c], X[c] = d, _stay_cumsum(d, seg)
    Q = {c: _stay_cumsum(D[c] ** 2, seg) for c in channels}
    S = {(i, j): _stay_cumsum((X[i] - D[i]) * D[j], seg) for i in channels for j in channels if i != j}

    def at(P, w):                                  # P at the window start a = t - w (0 before the stay)
        a = idx - w
        return np.where(a >= rs, P[np.maximum(a, 0)], 0.0)
    cols, names = [], []
    for w in windows:
        Xa = {c: at(X[c], w) for c in channels}
        for c in channels:
            cols.append(Q[c] - at(Q[c], w))
            names.append(f"{c}__qv{w}")
        for n_i, i in enumerate(channels):
            for j in channels[n_i + 1:]:
                sij = (S[(i, j)] - at(S[(i, j)], w)) - Xa[i] * (X[j] - Xa[j])
                sji = (S[(j, i)] - at(S[(j, i)], w)) - Xa[j] * (X[i] - Xa[i])
                cols.append(0.5 * (sij - sji))
                names.append(f"{i}__levy_{j}_{w}")
    return cols, names


def block(store) -> Tuple[np.ndarray, List[str]]:
    """``(F, names)``: the block for every row of ``store`` (causal, stay-local)."""
    rs, hour, seg = store.row_start, store.hour, store.seg
    N = store.n_rows
    idx = np.arange(N)
    col = lambda nm: store.col(nm).astype(np.float64)
    cols, names = [], []
    for prefix, phys in (("SOFA_P", True), ("SOFA", False)):
        parts = S3.component_points_now(col, rs, hour, physiology_only=phys)
        rise = S3.measured_rise_parts(parts, rs, 48)
        for o, organ in enumerate(ORGANS):
            cols += [parts[:, o], rise[:, o]]
            names += [f"{prefix}__{organ}_pts", f"{prefix}__{organ}_rise"]
        total = rise.sum(axis=1)
        cols += [total, _first_true_since(total >= 2, seg, idx)]
        names += [f"{prefix}__rise_total", f"{prefix}__since_rise2"]

    known = suspicion_known(np.isfinite(col("Culture")), np.isfinite(col("Antibiotic")), seg, hour)
    cols += [(np.isfinite(known) & (hour >= known)).astype(np.float64),
             np.where(hour >= known, hour - known, np.nan)]
    names += ["Antibiotic__si_known", "Antibiotic__since_si"]

    uo, w = col("UrineOutput"), col("Weight")
    for win in (6, 12, 24):
        tot, cnt = _window_sum(uo, rs, win)
        hrs = np.minimum(hour + 1, win)
        ok = (cnt > 0) & (w > 0)
        cols.append(np.where(ok, tot / np.where(ok, w, 1.0) / hrs, np.nan))
        names.append(f"UrineOutput__mlkgh{win}")

    neu, lym = _locf(col("Neutrophils"), rs), _locf(col("Lymphocytes"), rs)
    cols.append(np.where(lym > 0, neu / np.where(lym > 0, lym, 1.0), np.nan))
    names.append("Neutrophils__nlr")

    nee = sum(k * np.nan_to_num(col(ch)) for ch, k in NEE.items()) + \
        VASOPRESSIN_U_PER_H * np.nan_to_num(col("Vasopressin"))
    mx = np.zeros(N)
    for lag in range(24):
        src = idx - lag
        ok = src >= rs
        v = np.zeros(N)
        v[ok] = nee[src[ok]]
        mx = np.maximum(mx, v)
    cols += [nee, mx]
    names += ["Norepinephrine__nee", "Norepinephrine__nee_max24"]
    sc, sn = signature_block(store)
    cols += sc
    names += sn
    return np.stack(cols, axis=1).astype(np.float32), names
