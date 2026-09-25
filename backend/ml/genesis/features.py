"""Vectorized, strictly causal features over all 40 CinC 2019 variables.

Every feature at row ``i`` is a function of rows ``row_start[i] .. i`` of the same
patient and nothing else. The operations are chosen so this is structural rather
than a matter of care:

* last-observation indices come from ``np.maximum.accumulate`` over a row index
  masked to observed rows — an accumulate only ever looks backwards;
* window sums come from a running sum that RESTARTS at every stay (never one
  cohort-wide cumulative sum, whose rounding would couple patients);
* rolling windows are built from explicit backward shifts that return NaN when the
  shift would cross into the previous patient.

``verify_genesis.py`` rewrites every patient's future and asserts that no earlier
feature moves, so the claim is tested rather than asserted.

Missing is kept missing (NaN) — never imputed to a "normal" value. A lab that was never
drawn is information (nobody suspected anything), and the boosters used downstream
learn a split direction for missing values natively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.ml.genesis.rawstore import DYNAMIC, VITALS, RawStore

#: Channels that make up a blood-gas draw — an arterial sample is an escalation of care.
BLOOD_GAS = ["BaseExcess", "HCO3", "pH", "PaCO2", "SaO2"]
LABS = [c for c in DYNAMIC if c not in VITALS]


@dataclass
class FeatureSpec:
    channels: List[str] = field(default_factory=lambda: list(DYNAMIC))
    rolling: List[str] = field(default_factory=lambda: list(VITALS))
    windows: Tuple[int, ...] = (6, 24)
    detail: bool = True        # previous value, deltas, first value, expanding extremes
    process: bool = True       # measurement-process (clinician behaviour) features
    composites: bool = True    # SOFA-proxy, SIRS, qSOFA, shock index ...
    static: bool = True
    site: bool = False         # hospital indicator — off by default, reported separately
    #: Dataset overrides (None = CinC 2019). ``labs``: channels counted as lab draws by the
    #: measurement-process features; ``static_cols``: per-stay columns passed through as-is.
    labs: Optional[List[str]] = None
    static_cols: Optional[List[str]] = None

    @staticmethod
    def wearable() -> "FeatureSpec":
        """The same seven vitals the production wearable detector sees."""
        return FeatureSpec(channels=list(VITALS), process=False, composites=False)

    def key(self) -> str:
        parts = [f"c{len(self.channels)}", f"r{len(self.rolling)}",
                 "w" + "-".join(map(str, self.windows)),
                 "d" if self.detail else "", "p" if self.process else "",
                 "x" if self.composites else "", "s" if self.static else "",
                 "h" if self.site else ""]
        return "_".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Segment primitives
# --------------------------------------------------------------------------- #
class _Seg:
    """Precomputed per-row segment geometry shared by every primitive."""

    def __init__(self, raw: RawStore):
        self.N = raw.n_rows
        self.idx = np.arange(self.N, dtype=np.int64)
        self.row_start = raw.row_start
        self.hour = self.idx - self.row_start
        self.seg = raw.seg
        self.starts = raw.starts

    def last_obs(self, obs: np.ndarray) -> np.ndarray:
        """Index of the most recent observed row at or before i in the same stay (-1)."""
        last = np.maximum.accumulate(np.where(obs, self.idx, -1))
        last[last < self.row_start] = -1
        return last

    def prefix(self, x: np.ndarray) -> np.ndarray:
        """Stay-local running sum: ``P[i]`` = sum of ``x`` over rows ``row_start[i] .. i``.

        The sum restarts at every stay. Differencing ONE cohort-wide cumulative sum is
        exact in real arithmetic but not in floating point: the rounding of a patient's
        window would then depend on every earlier patient's values — including their
        futures — which ``verify_genesis`` catches at the 1e-12 level. Restarting per stay
        makes a row's value a function of its own stay's past and nothing else."""
        return pd.Series(np.asarray(x, dtype=np.float64)).groupby(
            self.seg, sort=False).cumsum().to_numpy()

    def window_sum(self, P: np.ndarray, w: Optional[int]) -> np.ndarray:
        """Sum over the last ``w`` rows of the stay ending at i (``None`` = whole stay)."""
        lo = self.row_start if w is None else np.maximum(self.idx - w + 1, self.row_start)
        before = np.where(lo > self.row_start, P[np.maximum(lo - 1, 0)], 0.0)
        return P - before

    def shift(self, v: np.ndarray, lag: int) -> np.ndarray:
        """v[i - lag] if that row belongs to the same stay, else NaN."""
        out = np.full(self.N, np.nan, dtype=np.float64)
        if lag == 0:
            out[:] = v
            return out
        src = self.idx - lag
        ok = src >= self.row_start
        out[ok] = v[src[ok]]
        return out

    def cumextreme(self, v: np.ndarray, how: str) -> np.ndarray:
        s = pd.Series(v)
        g = s.groupby(self.seg, sort=False)
        return (g.cummax() if how == "max" else g.cummin()).to_numpy(dtype=np.float64)


class _Cols:
    """Column sink that fills a Fortran-ordered float32 matrix in place.

    Over-allocates columns; untouched trailing columns of an F-ordered array are never
    written, so their pages are never committed and cost no physical memory."""

    def __init__(self, n_rows: int, cap: int):
        self.F = np.empty((n_rows, cap), dtype=np.float32, order="F")
        self.names: List[str] = []

    def add(self, name: str, v: np.ndarray) -> None:
        j = len(self.names)
        if j >= self.F.shape[1]:
            raise RuntimeError("feature capacity exceeded; raise cap")
        self.F[:, j] = v
        self.names.append(name)

    def done(self) -> Tuple[np.ndarray, List[str]]:
        return self.F[:, : len(self.names)], self.names


# --------------------------------------------------------------------------- #
def _clean(raw: RawStore, ch: str) -> np.ndarray:
    v = raw.col(ch).astype(np.float64)
    if ch == "FiO2":
        # recorded as a fraction; a handful of rows are percentages or garbage
        v = np.where((v > 1.0) & (v <= 100.0), v / 100.0, v)
        v = np.where((v < 0.21) | (v > 1.0), np.nan, v)
    return v


def build(raw: RawStore, spec: Optional[FeatureSpec] = None,
          cap: int = 900) -> Tuple[np.ndarray, List[str]]:
    """Return ``(F, names)``: an (N, D) float32 causal feature matrix, rows aligned to ``raw``."""
    spec = spec or FeatureSpec()
    S = _Seg(raw)
    out = _Cols(S.N, cap)
    last_val: Dict[str, np.ndarray] = {}
    obs_of: Dict[str, np.ndarray] = {}

    for ch in spec.channels:
        v = _clean(raw, ch)
        obs = np.isfinite(v)
        obs_of[ch] = obs
        last = S.last_obs(obs)
        has = last >= 0
        lv = np.where(has, v[np.maximum(last, 0)], np.nan)
        last_val[ch] = lv
        C = S.prefix(obs)
        n_all = S.window_sum(C, None)
        out.add(ch, lv)
        out.add(f"{ch}__age", np.where(has, S.idx - last, np.nan))
        out.add(f"{ch}__n", n_all)
        out.add(f"{ch}__n24", S.window_sum(C, 24))
        if spec.detail:
            # second-most-recent observation: the last observation strictly before `last`
            before = np.r_[-1, np.maximum.accumulate(np.where(obs, S.idx, -1))[:-1]]
            sl = np.where(has, before[np.maximum(last, 0)], -1)
            sl[sl < S.row_start] = -1
            has2 = sl >= 0
            pv = np.where(has2, v[np.maximum(sl, 0)], np.nan)
            out.add(f"{ch}__prev", pv)
            out.add(f"{ch}__dlast", lv - pv)
            out.add(f"{ch}__dtlast", np.where(has2, last - sl, np.nan))
            # first observed value of the stay, visible only once it has happened
            big = np.iinfo(np.int64).max
            first_idx = np.minimum.reduceat(np.where(obs, S.idx, big), S.starts)[S.seg]
            seen = has & (first_idx <= S.idx)
            fv = np.where(seen, v[np.minimum(first_idx, S.N - 1)], np.nan)
            out.add(f"{ch}__first", fv)
            out.add(f"{ch}__dfirst", lv - fv)
            out.add(f"{ch}__emax", S.cumextreme(lv, "max"))
            out.add(f"{ch}__emin", S.cumextreme(lv, "min"))
            sm = S.window_sum(S.prefix(np.where(obs, v, 0.0)), None)
            emean = np.where(n_all > 0, sm / np.maximum(n_all, 1), np.nan)
            out.add(f"{ch}__dmean", lv - emean)

    for ch in spec.rolling:
        if ch not in last_val:
            continue
        lv = last_val[ch]
        for w in spec.windows:
            n = np.zeros(S.N); sx = np.zeros(S.N); sy = np.zeros(S.N)
            sxx = np.zeros(S.N); sxy = np.zeros(S.N); syy = np.zeros(S.N)
            mx = np.full(S.N, -np.inf); mn = np.full(S.N, np.inf)
            for lag in range(w):
                y = S.shift(lv, lag)
                ok = np.isfinite(y)
                yz = np.where(ok, y, 0.0)
                x = -float(lag)
                n += ok; sx += ok * x; sy += yz; sxx += ok * x * x
                sxy += yz * x; syy += yz * yz
                mx = np.where(ok, np.maximum(mx, yz), mx)
                mn = np.where(ok, np.minimum(mn, yz), mn)
            has = n > 0
            mean = np.where(has, sy / np.maximum(n, 1), np.nan)
            var = np.where(has, syy / np.maximum(n, 1) - np.nan_to_num(mean) ** 2, np.nan)
            den = n * sxx - sx * sx
            slope = np.where(den > 1e-9, (n * sxy - sx * sy) / np.where(den > 1e-9, den, 1.0), np.nan)
            out.add(f"{ch}__mean{w}", mean)
            out.add(f"{ch}__min{w}", np.where(has, mn, np.nan))
            out.add(f"{ch}__max{w}", np.where(has, mx, np.nan))
            out.add(f"{ch}__std{w}", np.sqrt(np.clip(var, 0.0, None)))
            out.add(f"{ch}__slope{w}", slope)

    if spec.process:
        labs = [c for c in (spec.labs if spec.labs is not None else LABS) if c in obs_of]
        if labs:
            lab_now = np.sum([obs_of[c] for c in labs], axis=0).astype(np.float64)
            Cl = S.prefix(lab_now)
            out.add("proc__labs_now", lab_now)
            out.add("proc__labs6", S.window_sum(Cl, 6))
            out.add("proc__labs24", S.window_sum(Cl, 24))
            draw = lab_now > 0
            ld = S.last_obs(draw)
            out.add("proc__draw_age", np.where(ld >= 0, S.idx - ld, np.nan))
            Cd = S.prefix(draw)
            out.add("proc__draws24", S.window_sum(Cd, 24))
            out.add("proc__draws_all", S.window_sum(Cd, None))
            # lab types ordered for the FIRST time this stay — the work-up widening
            new = np.zeros(S.N)
            for c in labs:
                Co = S.prefix(obs_of[c])
                first_time = obs_of[c] & (S.window_sum(Co, None) == 1)
                new += first_time
            Cn = S.prefix(new)
            out.add("proc__new_now", new)
            out.add("proc__new24", S.window_sum(Cn, 24))
            out.add("proc__lab_types", np.sum([S.window_sum(S.prefix(obs_of[c]), None) > 0
                                               for c in labs], axis=0).astype(np.float64))
        gas = [c for c in BLOOD_GAS if c in obs_of]
        if gas:
            g_now = np.any([obs_of[c] for c in gas], axis=0)
            Cg = S.prefix(g_now)
            out.add("proc__gas24", S.window_sum(Cg, 24))
            lg = S.last_obs(g_now)
            out.add("proc__gas_age", np.where(lg >= 0, S.idx - lg, np.nan))
        vit = [c for c in VITALS if c in obs_of]
        if vit:
            v_now = np.sum([obs_of[c] for c in vit], axis=0).astype(np.float64)
            out.add("proc__vitals_now", v_now)
            out.add("proc__vitals6", S.window_sum(S.prefix(v_now), 6))

    if spec.composites:
        for name, arr in _composites(last_val, S).items():
            out.add(name, arr)

    if spec.static and spec.static_cols is not None:
        for name in spec.static_cols:
            out.add(name, raw.col(name))
        out.add("hour", S.hour.astype(np.float64))
    elif spec.static:
        age = raw.col("Age").astype(np.float64)
        out.add("Age", age)
        out.add("Gender", raw.col("Gender"))
        u1, u2 = raw.col("Unit1"), raw.col("Unit2")
        out.add("unit_sicu", np.where(np.isfinite(u2), u2, np.where(np.isfinite(u1), 1 - u1, np.nan)))
        out.add("HospAdmTime", raw.col("HospAdmTime"))
        icu = raw.col("ICULOS").astype(np.float64)
        out.add("ICULOS", icu)
        out.add("hour", S.hour.astype(np.float64))
        out.add("icu_start", icu[S.row_start])
    if spec.site:
        out.add("site", raw.site[S.seg].astype(np.float64))
    return out.done()


#: Labs whose off-cycle re-draw is a clinical decision rather than routine: a second
#: lactate within hours is the sepsis bundle's "re-measure if elevated"; a repeat CBC or
#: chemistry the same day is a STAT order, not the 4 a.m. round.
INTENT_LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total", "pH", "HCO3"]


def intent_block(raw: RawStore) -> Tuple[np.ndarray, List[str]]:
    """Clinician-intent features: when the team acted, relative to its own routine.

    The trajectories around onset show the sharpest change at clinical onset is not
    physiology but behaviour — lab orders per hour jump 1.8 -> 3.0 and lactate draws
    5% -> 9% in the hours when cultures and antibiotics (unrecorded) are ordered. The
    base features count orders; these features read the orders as DECISIONS: a repeat
    draw inside a day, an ordering rate above the stay's own baseline, the clock since
    the first abnormal value of each sepsis marker. Strictly causal (same primitives)."""
    S = _Seg(raw)
    cols: Dict[str, np.ndarray] = {}
    lab_obs = {c: np.isfinite(raw.col(c)) for c in LABS}
    labs_now = np.sum([lab_obs[c] for c in LABS], axis=0).astype(np.float64)
    Cl = S.prefix(labs_now)
    all_rate = S.window_sum(Cl, None) / (S.hour + 1.0)
    cols["intent__labs3"] = S.window_sum(Cl, 3)
    cols["intent__rate_ratio6"] = (S.window_sum(Cl, 6) / np.minimum(S.hour + 1.0, 6.0)) / \
        np.maximum(all_rate, 0.05)
    new = np.zeros(S.N)
    for c in LABS:
        new += lab_obs[c] & (S.window_sum(S.prefix(lab_obs[c]), None) == 1)
    cols["intent__new3"] = S.window_sum(S.prefix(new), 3)
    stat_total = np.zeros(S.N)
    for c in INTENT_LABS:
        obs = lab_obs[c]
        last = S.last_obs(obs)
        before = np.r_[-1, np.maximum.accumulate(np.where(obs, S.idx, -1))[:-1]]
        prev = np.where(obs, before, -1)
        prev[prev < S.row_start] = -1
        gap = np.where(obs & (prev >= 0), S.idx - prev, np.inf)
        stat = obs & (gap <= 12)                      # re-drawn within half a day
        Cs = S.prefix(stat)
        cols[f"intent__{c}_stat24"] = S.window_sum(Cs, 24)
        stat_total += stat
    cols["intent__stat24"] = S.window_sum(S.prefix(stat_total), 24)
    # the bundle's signature: a lactate repeated within 6 h
    lac = lab_obs["Lactate"]
    before = np.r_[-1, np.maximum.accumulate(np.where(lac, S.idx, -1))[:-1]]
    prev = np.where(lac, before, -1)
    prev[prev < S.row_start] = -1
    rep = lac & (prev >= 0) & ((S.idx - prev) <= 6)
    lr = S.last_obs(rep)
    cols["intent__lactate_repeat_age"] = np.where(lr >= 0, S.idx - lr, np.nan)
    # hours since the FIRST abnormal value of each sepsis marker (onset anchors)
    marks = {"lactate_gt2": ("Lactate", lambda v: v > 2.0),
             "temp_gt383": ("Temp", lambda v: v >= 38.3),
             "wbc_gt12": ("WBC", lambda v: (v > 12.0) | (v < 4.0)),
             "map_lt65": ("MAP", lambda v: v < 65.0),
             "resp_gt22": ("Resp", lambda v: v >= 22.0)}
    big = np.iinfo(np.int64).max
    for name, (ch, fn) in marks.items():
        v = raw.col(ch).astype(np.float64)
        ev = np.isfinite(v) & fn(np.nan_to_num(v, nan=0.0 if ch != "MAP" else 999.0))
        first = np.minimum.reduceat(np.where(ev, S.idx, big), S.starts)[S.seg]
        cols[f"intent__since_{name}"] = np.where(first <= S.idx, S.idx - first, np.nan)
        Ce = S.prefix(ev)
        cols[f"intent__{name}_n24"] = S.window_sum(Ce, 24)
    names = list(cols)
    return np.stack([cols[n].astype(np.float32) for n in names], axis=1), names


def _sofa_points(x: np.ndarray, cuts: Sequence[float], higher_is_worse: bool) -> np.ndarray:
    pts = np.zeros(len(x))
    for k, c in enumerate(cuts, start=1):
        hit = (x >= c) if higher_is_worse else (x < c)
        pts = np.where(np.isfinite(x) & hit, k, pts)
    return pts


def _composites(lv: Dict[str, np.ndarray], S: _Seg) -> Dict[str, np.ndarray]:
    nan = np.full(S.N, np.nan)
    g = lambda c: lv.get(c, nan)
    res: Dict[str, np.ndarray] = {}
    # Partial SOFA — the organ-dysfunction half of the Sepsis-3 label, as far as the
    # record allows (no GCS, urine output, vasopressor dose or PaO2 are available).
    coag = _sofa_points(g("Platelets"), [150, 100, 50, 20], higher_is_worse=False)
    liver = _sofa_points(g("Bilirubin_total"), [1.2, 2.0, 6.0, 12.0], higher_is_worse=True)
    renal = _sofa_points(g("Creatinine"), [1.2, 2.0, 3.5, 5.0], higher_is_worse=True)
    cardio = np.where(np.isfinite(g("MAP")) & (g("MAP") < 70), 1.0, 0.0)
    fio2 = np.where(np.isfinite(g("FiO2")), g("FiO2"), 0.21)
    sat = np.where(np.isfinite(g("SaO2")), g("SaO2"), g("O2Sat"))
    sf = sat / fio2
    # Rice et al. 2007, S/F = 64 + 0.84 P/F: P/F 400/300/200/100 -> S/F 400/316/232/148
    resp = _sofa_points(sf, [400, 316, 232, 148], higher_is_worse=False)
    sofa = coag + liver + renal + cardio + resp
    res["sofa_proxy"] = sofa
    res["sofa_resp"] = resp
    res["sf_ratio"] = sf
    C = S.prefix(sofa)
    for w in (24, 48):
        mn = np.full(S.N, np.inf)
        for lag in range(w):
            y = S.shift(sofa, lag)
            mn = np.where(np.isfinite(y), np.minimum(mn, y), mn)
        res[f"sofa_rise{w}"] = sofa - mn
    res["sofa_rise_all"] = sofa - S.cumextreme(sofa, "min")
    res["sofa_mean24"] = S.window_sum(C, 24) / np.minimum(S.hour + 1, 24)
    hr, sbp, temp, resp_r = g("HR"), g("SBP"), g("Temp"), g("Resp")
    wbc, paco2 = g("WBC"), g("PaCO2")
    sirs = ((np.nan_to_num(temp, nan=37.0) > 38.0) | (np.nan_to_num(temp, nan=37.0) < 36.0)).astype(float)
    sirs += (np.nan_to_num(hr, nan=70.0) > 90.0)
    sirs += (np.nan_to_num(resp_r, nan=16.0) > 20.0) | (np.nan_to_num(paco2, nan=40.0) < 32.0)
    sirs += (np.nan_to_num(wbc, nan=8.0) > 12.0) | (np.nan_to_num(wbc, nan=8.0) < 4.0)
    res["sirs"] = sirs
    res["qsofa_proxy"] = (np.nan_to_num(resp_r, nan=16.0) >= 22.0).astype(float) + \
        (np.nan_to_num(sbp, nan=120.0) <= 100.0)
    res["shock_index"] = hr / np.where(sbp > 1, sbp, np.nan)
    res["pulse_pressure"] = sbp - g("DBP")
    res["bun_cr"] = g("BUN") / np.where(g("Creatinine") > 0, g("Creatinine"), np.nan)
    return res


if __name__ == "__main__":  # pragma: no cover
    import time
    from backend.ml.genesis import rawstore
    raw = rawstore.load()
    t0 = time.time()
    F, names = build(raw)
    print(F.shape, f"{time.time()-t0:.1f}s", f"{F.nbytes/1e9:.2f} GB")
