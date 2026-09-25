"""Sepsis-3 from scratch: hourly SOFA (all six organ systems), suspicion of infection, onset.

CinC 2019 could score half of SOFA; MIMIC-IV records all of it:

    respiration     PaO2 / FiO2 (3-4 points only while ventilated)
    coagulation     platelets
    liver           bilirubin
    cardiovascular  MAP < 70, then vasopressor doses (dopamine, dobutamine, epi, norepi)
    CNS             Glasgow Coma Scale (intubated = unassessable = 15, per mimic-code)
    renal           creatinine, or 24-h urine output < 500 / < 200 mL

Each component is the WORST value over the preceding 24 h (so hourly SOFA at hour t uses
only hours <= t — the same score can serve as a causal feature). Missing -> 0 points.

Suspicion of infection (Seymour et al. 2016, mimic-code): an antibiotic and a body-fluid
culture close in time — culture first and antibiotic within 72 h, or antibiotic first and
culture within 24 h; the suspicion time is the earlier of the pair.

Onset. For a suspicion time s, the first hour in [s - 48 h, s + 24 h] at which SOFA has
risen by >= 2; onset = min(s, that hour) (the CinC 2019 anchor). What "risen" means is the
subtle part, and three rules are kept:

``measured`` (PRIMARY) — each organ component may only rise relative to an earlier
    MEASUREMENT of the same component (carried forward <= 24 h). Built because the other
    two fire on data arrival: on the MIMIC demo, mean 24-h-worst SOFA climbs 1.8 -> 3.2 ->
    5.3 over hours 0 / 3 / 24 simply because labs arrive after admission and a 24-h worst
    can only ratchet up — so a window that starts at ICU admission "detects" a 2-point
    rise within hours for most patients, and even drags later genuine onsets back to the
    first hours of the stay. Under ``measured`` a first-ever measurement cannot create a
    rise; organ failure present on arrival is sepsis-on-admission, which the early-warning
    cohort excludes anyway.
``rise``     — CinC 2019 as written: total 24-h-worst SOFA minus its running minimum in the window.
``absolute`` — SOFA >= 2 (the Seymour / mimic-code baseline-0 convention).
Anchors: min(s, t_SOFA) (primary), s, or t_SOFA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SOFA_PARTS = ("sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns", "sofa_renal")


def _points(x: np.ndarray, cuts: Sequence[float], worse_high: bool) -> np.ndarray:
    """Ascending points 1..len(cuts) as x crosses each cut; NaN stays NaN."""
    p = np.where(np.isfinite(x), 0.0, np.nan)
    for k, c in enumerate(cuts, start=1):
        hit = (x >= c) if worse_high else (x < c)
        p = np.where(np.isfinite(x) & hit, float(k), p)
    return p


def _rolling_max(v: np.ndarray, row_start: np.ndarray, w: int) -> np.ndarray:
    """max over rows [i-w+1, i] of the same stay, ignoring NaN (NaN if none)."""
    n = len(v)
    idx = np.arange(n)
    out = np.full(n, -np.inf)
    for lag in range(w):
        src = idx - lag
        ok = src >= row_start
        vv = np.full(n, -np.inf)
        vv[ok] = np.nan_to_num(v[src[ok]], nan=-np.inf)
        out = np.maximum(out, vv)
    return np.where(np.isfinite(out), out, np.nan)


def _rolling_sum(v: np.ndarray, row_start: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """(sum of finite values, number of finite values) over rows [i-w+1, i] of the stay."""
    n = len(v)
    idx = np.arange(n)
    s = np.zeros(n)
    c = np.zeros(n)
    for lag in range(w):
        src = idx - lag
        ok = src >= row_start
        vv = np.zeros(n)
        vv[ok] = v[src[ok]]
        f = np.isfinite(vv)
        s += np.where(f, vv, 0.0)
        c += f & ok
    return s, c


def _ffill_limit(v: np.ndarray, row_start: np.ndarray, limit: int) -> np.ndarray:
    idx = np.arange(len(v))
    last = np.maximum.accumulate(np.where(np.isfinite(v), idx, -1))
    ok = (last >= row_start) & (idx - last <= limit) & (last >= 0)
    return np.where(ok, v[np.maximum(last, 0)], np.nan)


def gcs_total(eye: np.ndarray, verbal: np.ndarray, motor: np.ndarray,
              row_start: np.ndarray, carry: int = 6) -> np.ndarray:
    """Total GCS at hours where any component was charted; components carried forward up
    to ``carry`` hours, missing ones imputed as normal (4/5/6); intubated (verbal 0) = 15."""
    e = _ffill_limit(eye, row_start, carry)
    v = _ffill_limit(verbal, row_start, carry)
    m = _ffill_limit(motor, row_start, carry)
    tot = np.where(np.isnan(e), 4, e) + np.where(np.isnan(v), 5, v) + np.where(np.isnan(m), 6, m)
    tot = np.where(v == 0, 15.0, tot)
    charted = np.isfinite(eye) | np.isfinite(verbal) | np.isfinite(motor)
    return np.where(charted, tot, np.nan)


def hourly_sofa(col, row_start: np.ndarray, hour: np.ndarray) -> Dict[str, np.ndarray]:
    """Hourly SOFA components (24-h worst, causal) + totals.

    ``col(name)`` returns an hourly channel (NaN = not measured that hour; vasopressors and
    ventilation are dense with 0 = off). ``sofa_total_P`` is the physiology-tier variant:
    the cardiovascular score from MAP alone (no vasopressor doses, which are treatment)."""
    out: Dict[str, np.ndarray] = {}
    vent = np.nan_to_num(col("VentInvasive")) > 0
    pao2 = col("PaO2")
    fio2 = _ffill_limit(col("FiO2"), row_start, 4)
    fio2 = np.where(np.isfinite(fio2), fio2, np.where(vent, np.nan, 0.21))
    pf = pao2 / fio2
    resp = _points(pf, [400, 300], worse_high=False)
    resp = np.where(np.isfinite(pf) & vent & (pf < 200), 3.0, resp)
    resp = np.where(np.isfinite(pf) & vent & (pf < 100), 4.0, resp)
    out["sofa_resp"] = resp
    out["sofa_coag"] = _points(col("Platelets"), [150, 100, 50, 20], worse_high=False)
    out["sofa_liver"] = _points(col("Bilirubin_total"), [1.2, 2.0, 6.0, 12.0], worse_high=True)
    mapv = col("MAP")
    dop, dob = np.nan_to_num(col("Dopamine")), np.nan_to_num(col("Dobutamine"))
    epi, nor = np.nan_to_num(col("Epinephrine")), np.nan_to_num(col("Norepinephrine"))
    cv_map = np.where(np.isfinite(mapv), np.where(mapv < 70, 1.0, 0.0), np.nan)
    cv = cv_map.copy()
    cv = np.where((dop > 0) | (dob > 0), 2.0, cv)
    cv = np.where((dop > 5) | (epi > 0) | (nor > 0), 3.0, cv)
    cv = np.where((dop > 15) | (epi > 0.1) | (nor > 0.1), 4.0, cv)
    out["sofa_cardio"] = cv
    out["sofa_cns"] = _gcs_points(col("GCS"))
    cr = _points(col("Creatinine"), [1.2, 2.0, 3.5, 5.0], worse_high=True)
    uo_sum, uo_n = _rolling_sum(col("UrineOutput"), row_start, 24)
    uo_ok = (uo_n > 0) & (hour >= 23)
    uo_pts = np.where(uo_ok & (uo_sum < 200), 4.0, np.where(uo_ok & (uo_sum < 500), 3.0, np.nan))
    out["sofa_renal"] = np.fmax(cr, uo_pts)
    tot = np.zeros(len(hour))
    for k in SOFA_PARTS:
        out[k] = np.nan_to_num(_rolling_max(out[k], row_start, 24), nan=0.0)
        tot += out[k]
    out["sofa_total"] = tot
    cv_p = np.nan_to_num(_rolling_max(cv_map, row_start, 24), nan=0.0)
    out["sofa_total_P"] = tot - out["sofa_cardio"] + cv_p
    return out


def component_points_now(col, row_start: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """(N, 6) organ points from each hour's OWN measurements, carried forward up to 24 h;
    NaN until a component has been measured. The basis of the ``measured`` onset rule."""
    vent = np.nan_to_num(col("VentInvasive")) > 0
    fio2 = _ffill_limit(col("FiO2"), row_start, 4)
    fio2 = np.where(np.isfinite(fio2), fio2, np.where(vent, np.nan, 0.21))
    pf = col("PaO2") / fio2
    resp = _points(pf, [400, 300], worse_high=False)
    resp = np.where(np.isfinite(pf) & vent & (pf < 200), 3.0, resp)
    resp = np.where(np.isfinite(pf) & vent & (pf < 100), 4.0, resp)
    coag = _points(col("Platelets"), [150, 100, 50, 20], worse_high=False)
    liver = _points(col("Bilirubin_total"), [1.2, 2.0, 6.0, 12.0], worse_high=True)
    mapv = col("MAP")
    dop, dob = np.nan_to_num(col("Dopamine")), np.nan_to_num(col("Dobutamine"))
    epi, nor = np.nan_to_num(col("Epinephrine")), np.nan_to_num(col("Norepinephrine"))
    on_vaso = (dop > 0) | (dob > 0) | (epi > 0) | (nor > 0)
    cv = np.where(np.isfinite(mapv), np.where(mapv < 70, 1.0, 0.0), np.nan)
    cv = np.where((dop > 0) | (dob > 0), 2.0, cv)
    cv = np.where((dop > 5) | (epi > 0) | (nor > 0), 3.0, cv)
    cv = np.where((dop > 15) | (epi > 0.1) | (nor > 0.1), 4.0, cv)
    cv = np.where(on_vaso | np.isfinite(mapv), cv, np.nan)
    cns = _gcs_points(col("GCS"))
    cr = _points(col("Creatinine"), [1.2, 2.0, 3.5, 5.0], worse_high=True)
    uo_sum, uo_n = _rolling_sum(col("UrineOutput"), row_start, 24)
    uo_ok = (uo_n > 0) & (hour >= 23)
    uo_pts = np.where(uo_ok & (uo_sum < 200), 4.0, np.where(uo_ok & (uo_sum < 500), 3.0, np.nan))
    renal = np.fmax(_ffill_limit(cr, row_start, 24), uo_pts)
    parts = [resp, coag, liver, cv, cns, renal]
    return np.stack([_ffill_limit(p, row_start, 24) for p in parts], axis=1)


def _gcs_points(g: np.ndarray) -> np.ndarray:
    p = np.where(np.isfinite(g), 0.0, np.nan)
    p = np.where(np.isfinite(g) & (g < 15), 1.0, p)
    p = np.where(np.isfinite(g) & (g < 13), 2.0, p)
    p = np.where(np.isfinite(g) & (g < 10), 3.0, p)
    p = np.where(np.isfinite(g) & (g < 6), 4.0, p)
    return p


# --------------------------------------------------------------------------- #
def suspicion_times(abx: np.ndarray, cult: np.ndarray) -> np.ndarray:
    """Sorted suspicion-of-infection times (hours) from one stay's antibiotic and culture
    times: culture then antibiotic within 72 h -> the culture time; antibiotic then culture
    within 24 h -> the antibiotic time."""
    abx = np.sort(np.asarray(abx, dtype=float))
    cult = np.sort(np.asarray(cult, dtype=float))
    out = []
    for a in abx:
        before = cult[(cult <= a) & (cult >= a - 72.0)]
        after = cult[(cult > a) & (cult <= a + 24.0)]
        if len(before):
            out.append(before.min())
        if len(after):
            out.append(a)
    return np.unique(np.array(out, dtype=float))


@dataclass
class Onset:
    t_si: float = np.nan        # the suspicion time that produced the onset
    t_sofa: float = np.nan      # the qualifying SOFA hour
    t_sepsis: float = np.nan    # onset per the chosen anchor (NaN = never septic)
    first_si: float = np.nan    # first suspicion of the stay, qualifying or not


def onset_for_stay(sofa: np.ndarray, si: np.ndarray, rule: str = "measured",
                   anchor: str = "min", parts: Optional[np.ndarray] = None) -> Onset:
    """``sofa``: the stay's hourly 24-h-worst SOFA; ``parts``: its (T, 6) carried-forward
    component points (needed by the ``measured`` rule). Index = hour since ICU admission."""
    o = Onset(first_si=float(si[0]) if len(si) else np.nan)
    T = len(sofa)
    for s in si:
        lo = int(max(0, np.floor(s - 48.0)))
        hi = int(min(T - 1, np.floor(s + 24.0)))
        if hi < lo:
            continue
        if rule == "measured":
            win = parts[lo:hi + 1]
            base = np.fmin.accumulate(win, axis=0)          # NaN-skipping running minimum
            rise = np.nansum(np.where(np.isfinite(win) & np.isfinite(base), win - base, 0.0), axis=1)
            hit = np.flatnonzero(rise >= 2)
        elif rule == "rise":
            win = sofa[lo:hi + 1]
            hit = np.flatnonzero(win - np.minimum.accumulate(win) >= 2)
        else:
            win = sofa[lo:hi + 1]
            hit = np.flatnonzero(win >= 2)
        if len(hit):
            h = float(lo + hit[0])
            o.t_si, o.t_sofa = float(s), h
            o.t_sepsis = {"min": min(s, h), "si": float(s), "sofa": h}[anchor]
            return o
    return o
