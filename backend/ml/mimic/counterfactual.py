"""Counterfactual patients — every fitted model re-scored on EDITED inputs.

CARE-PROCESS TRANSPLANT (CPT). An ICU record mixes two signals: the patient's physiology
(what was measured) and the care process (when, and how often, clinicians chose to
measure it). A lactate drawn at all is a decision, and decisions track suspicion — in
EHR data the timing of tests can predict outcomes better than their results (Agniel,
Kohane & Weber, BMJ 2018). A "physiology" model can therefore be reading the doctors.
Missingness-aware models make this worse: they are built to exploit it.

The transplant separates the two by editing patients, not models. For each evaluated stay
the grid is rebuilt four ways:

    factual           its own schedule, its own values
    physiology-only   the SCHEDULE of a random donor stay (label-independent), filled from
                      the patient's OWN physiology: the patient's real value where they were
                      measured that hour; elsewhere a STATE-MATCHED SUCCESSOR — a value drawn
                      from what real next measurements looked like, across the cohort, after
                      a measurement in the same state as the patient's latest (a random real
                      value before their first)
    process-only      its own schedule, every value replaced by a neutral one
    neither           donor schedule, neutral values — static context and clock only

and the SAME fitted fold models (each stay scored by the fold that never saw it) re-score
each. Derived channels (total GCS, SOFA) are recomputed from the edited grid, so no stale
score can smuggle the old schedule through. The four AUROCs give a two-factor Shapley
decomposition of the model's discrimination above baseline:

    phi_process    = 1/2 [(A_proc - A_base) + (A_fact - A_phys)]
    phi_physiology = 1/2 [(A_phys - A_base) + (A_fact - A_proc)]
    phi_process + phi_physiology = A_fact - A_base

and the headline, ``process_reliance = A_fact - A_phys``: the AUROC the model loses when
care is no longer informative about the label — what it would lose at a hospital whose
clinicians order differently. ``learnable`` asks the complementary question by RETRAINING
inside each edited world: how much can any model learn from physiology alone (a model
trained there is process-invariant by construction), and how much from the process alone.

Why successors and not the last value carried forward: a carried-forward value only
changes when the patient is re-measured, so "how often does this value change" re-encodes
the patient's own schedule inside the physiology-only world. A frozen model rarely reads
that code; a model RETRAINED there learns it (a planted schedule-only world reached AUROC
0.73 "from values" that way before this fix; ``verify_mimic`` §15 keeps it near chance).
A successor draw preserves the cohort's marginal distribution and its one-step dynamics
(a persistently high creatinine stays high; an i.i.d. channel stays i.i.d.), so a change
no longer marks a real measurement. Urine output is volume, not state, so it moves as a
RATE (the patient's causal 24-h rate times the donor's charting interval). Donors are drawn
uniformly from the dev stays, independent of the recipient's label — never matched on stay
length, which itself carries the label (the length tripwire).

ADMISSION AUDIT. Before any number about a model is believed, the model is attacked on
edited patients: (1) every stay's future is rewritten after a random cut; (2) every stay is
truncated at its cut; (3) every OTHER stay is scrambled. No score at or before the cut may
move in (1)-(2), and no score of an untouched stay may move in (3). A model that fails is
not admitted to the ledger. The first catches look-ahead features, the second stay-length
or stay-level statistics, the third any coupling across patients (test-set normalisation,
batch statistics — or the floating-point coupling a cohort-wide cumulative sum caused in
GENESIS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.ml.genesis.harness import Ranked
from backend.ml.mimic import extract as E
from backend.ml.mimic.ledger import FoldModels, Ledger, build_features, stratified_auc

#: state channels whose measurement SCHEDULE is a care decision (derived GCS / SOFA are
#: recomputed; treatment and order channels are not tier P and are left alone)
STATE_CH = E.VITAL_CH + E.HEMO_CH + ["FiO2", "GCS_eye", "GCS_verbal", "GCS_motor"] + E.LAB_CH
RATE_CH = "UrineOutput"
DERIVED_CH = ("GCS", "SOFA_P", "SOFA")
CONDITIONS = {"physiology_only": (False, True), "process_only": (True, False), "neither": (False, False)}


# --------------------------------------------------------------------------- #
# The transplant
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Deterministic, patient-local randomness
# --------------------------------------------------------------------------- #
def _splitmix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def hash_uniform(*keys) -> np.ndarray:
    """U[0, 1) that is a pure function of its integer keys (splitmix64). Every counterfactual
    draw is keyed by (seed, stay, hour, channel): a draw never depends on how many draws
    other patients needed, or on this patient's future — so a model that uses the transplant
    at inference is still causal and patient-local (the admission audit checks it)."""
    arrs = np.broadcast_arrays(*[np.asarray(k).astype(np.int64).astype(np.uint64) for k in keys])
    h = np.zeros(arrs[0].shape, dtype=np.uint64)
    for a in arrs:
        h = _splitmix(h ^ a)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))


# --------------------------------------------------------------------------- #
# References and donors
# --------------------------------------------------------------------------- #
@dataclass
class Reference:
    """What an uninformative or a state-matched reading looks like, from a set of stays."""
    neutral: Dict[str, float]                       # per channel: the median reading
    uo_rate: float                                  # the median hourly urine rate
    #: per channel: quantile-bin edges, the NEXT measured value after a measurement in each
    #: bin (successive measurements within a stay), and the marginal pool
    successors: Dict[str, Tuple[np.ndarray, List[np.ndarray], np.ndarray]]


def reference_from(store: E.MimicStore, stays: np.ndarray, n_bins: int = 20, max_pairs: int = 200_000,
                   seed: int = 0) -> Reference:
    rng = np.random.default_rng(seed)
    keep = np.zeros(store.n_patients, dtype=bool)
    keep[np.asarray(stays, dtype=np.int64)] = True
    rowmask = keep[store.seg]
    neutral, succ = {}, {}
    for c in STATE_CH:
        v = store.col(c).astype(np.float64)
        idx = np.flatnonzero(np.isfinite(v) & rowmask)
        vals, st = v[idx], store.seg[idx]
        neutral[c] = float(np.median(vals)) if len(vals) else 0.0
        same = st[1:] == st[:-1]
        prev, nxt = vals[:-1][same], vals[1:][same]
        if len(prev) > max_pairs:
            pick = rng.choice(len(prev), max_pairs, replace=False)
            prev, nxt = prev[pick], nxt[pick]
        marginal = vals if len(vals) <= max_pairs else rng.choice(vals, max_pairs, replace=False)
        edges = np.unique(np.quantile(marginal, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(marginal) \
            else np.zeros(0)
        b = np.searchsorted(edges, prev, side="right")
        succ[c] = (edges, [nxt[b == k] for k in range(len(edges) + 1)], marginal)
    uo = store.col(RATE_CH)
    rates = []
    for k in np.flatnonzero(keep):
        u = uo[store.starts[k]:store.starts[k] + store.lens[k]]
        f = np.flatnonzero(np.isfinite(u))
        if len(f) >= 2:
            rates.append(np.nansum(u) / (f[-1] - f[0] + 1))
    return Reference(neutral, float(np.median(rates)) if rates else 50.0, succ)


def reference(led: Ledger, n_bins: int = 20, max_pairs: int = 200_000, seed: int = 0,
              exclude_fold: Optional[int] = None) -> Reference:
    """The reference over the dev stays — without fold ``exclude_fold``, so a patient's
    counterfactual values are never drawn from their own (future) measurements."""
    ok = led.folds >= 0
    if exclude_fold is not None:
        ok &= led.folds != exclude_fold
    return reference_from(led.store, np.flatnonzero(ok), n_bins, max_pairs, seed)


@dataclass
class DonorBank:
    """Measurement schedules to transplant (state channels, then urine), kept apart from the
    store they came from so a fitted model can carry its training cohort's schedules."""
    masks: List[np.ndarray]

    @staticmethod
    def from_store(store: E.MimicStore, stays: np.ndarray) -> "DonorBank":
        j = {c: i for i, c in enumerate(store.columns)}
        cols = np.array([j[c] for c in STATE_CH] + [j[RATE_CH]])
        return DonorBank([np.isfinite(store.X[store.starts[k]:store.starts[k] + store.lens[k]][:, cols])
                          for k in np.asarray(stays, dtype=np.int64)])

    def __len__(self) -> int:
        return len(self.masks)

    def mask(self, d: int, L: int) -> np.ndarray:
        """Donor ``d``'s pattern over ``L`` hours, aligned on ICU admission; a shorter donor
        is extended by repeating its last day (the prefix never depends on ``L``)."""
        M = self.masks[int(d)]
        Ld = len(M)
        if Ld >= L:
            return M[:L]
        period = M[max(0, Ld - 24):]
        reps = int(np.ceil((L - Ld) / len(period)))
        return np.concatenate([M, np.tile(period, (reps, 1))])[:L]


def _interval(sched: np.ndarray) -> np.ndarray:
    """Hours since the previous charted hour (hours since admission for the first), <= 24."""
    h = np.arange(len(sched))
    last = np.maximum.accumulate(np.where(sched, h, -1))
    prev = np.r_[-1, last[:-1]]
    return np.minimum(np.where(prev >= 0, h - prev, h + 1), 24).astype(np.float64)


def _uo_rate(own: np.ndarray, neutral_rate: float) -> np.ndarray:
    """The patient's causal hourly urine rate: volume charted in the last 24 h over the
    hours covered; carried forward through gaps; neutral before the first chart."""
    L = len(own)
    h = np.arange(L)
    obs = np.isfinite(own)
    cs = np.cumsum(np.where(obs, own, 0.0))
    cn = np.cumsum(obs)
    lo = h - 24
    s24 = cs - np.where(lo >= 0, cs[np.maximum(lo, 0)], 0.0)
    n24 = cn - np.where(lo >= 0, cn[np.maximum(lo, 0)], 0)
    rate = np.where(n24 > 0, s24 / np.minimum(h + 1, 24), np.nan)
    last = np.maximum.accumulate(np.where(np.isfinite(rate), h, -1))
    return np.where(last >= 0, rate[np.maximum(last, 0)], neutral_rate)


def _successor_fill(own: np.ndarray, last: np.ndarray, ref: Reference, c: str, u: np.ndarray) -> np.ndarray:
    """A state-matched successor for each position (``last`` = the row of the patient's latest
    own measurement, -1 if none yet), picked by the uniforms ``u``."""
    edges, pools, marginal = ref.successors[c]
    fallback = marginal if len(marginal) else np.array([ref.neutral[c]])

    def pick(pool, uu):
        return pool[np.minimum((uu * len(pool)).astype(np.int64), len(pool) - 1)]
    out = np.empty(len(last))
    none = last < 0
    out[none] = pick(fallback, u[none])
    if (~none).any():
        b = np.searchsorted(edges, own[last[~none]], side="right")
        uu = u[~none]
        vals = np.empty(len(b))
        for k in np.unique(b):
            m = b == k
            vals[m] = pick(pools[k] if len(pools[k]) else fallback, uu[m])
        out[~none] = vals
    return out


def transplant(store: E.MimicStore, stays: np.ndarray, donors: np.ndarray, bank: DonorBank,
               own_schedule: bool, own_values: bool, ref: Reference, seed: int = 0) -> np.ndarray:
    """The edited grid (rows of ``subset(store, stays)``; derived channels are recomputed by
    ``Ledger.view`` / ``transplanted_store``). ``donors`` index ``bank``."""
    j = {c: i for i, c in enumerate(store.columns)}
    sc = np.array([j[c] for c in STATE_CH])
    uo = j[RATE_CH]
    stays = np.asarray(stays, dtype=np.int64)
    lens = store.lens[stays].astype(np.int64)
    if not len(stays):
        return np.zeros((0, store.X.shape[1]), np.float32)
    rows = np.concatenate([np.arange(store.starts[k], store.starts[k] + store.lens[k]) for k in stays])
    sub_start = np.r_[0, np.cumsum(lens)[:-1]]
    rs = np.repeat(sub_start, lens)
    X = store.X[rows].astype(np.float64)
    own = X[:, sc]
    obs = np.isfinite(own)
    full = None if own_schedule else np.concatenate([bank.mask(d, int(L)) for d, L in zip(donors, lens)])
    sched = obs if own_schedule else full[:, :-1]
    if not own_values:
        X[:, sc] = np.where(sched, np.array([ref.neutral[c] for c in STATE_CH])[None, :], np.nan)
    elif not own_schedule:
        last = np.maximum.accumulate(np.where(obs, np.arange(len(rows))[:, None], -1), axis=0)
        last[last < rs[:, None]] = -1                      # never across a stay boundary
        vals = np.where(obs, own, np.nan)                  # measured this hour: the real value
        need = sched & ~obs
        pid = np.repeat(store.pid[stays].astype(np.int64), lens)
        hour = np.arange(len(rows)) - rs
        for ci, c in enumerate(STATE_CH):
            r = np.flatnonzero(need[:, ci])
            if len(r):
                u = hash_uniform(seed, pid[r], hour[r], ci)
                vals[r, ci] = _successor_fill(own[:, ci], last[r, ci], ref, c, u)
        X[:, sc] = np.where(sched, vals, np.nan)
    if not (own_schedule and own_values):
        for k, (a, L) in enumerate(zip(sub_start, lens)):
            u = X[a:a + L, uo].copy()
            us = np.isfinite(u) if own_schedule else full[a:a + L, -1]
            rate = _uo_rate(u, ref.uo_rate) if own_values else ref.uo_rate
            X[a:a + L, uo] = np.where(us, rate * _interval(us), np.nan)
    return X.astype(np.float32)


def transplanted_store(store: E.MimicStore, stays: np.ndarray, donors: np.ndarray, bank: DonorBank,
                       ref: Reference, seed: int = 0, own_schedule: bool = False,
                       own_values: bool = True) -> E.MimicStore:
    """``stays`` rebuilt on donor schedules, derived channels recomputed — the counterfactual
    patients a temporal model trains and infers on."""
    return E.rederive(E.subset(store, stays),
                      transplant(store, stays, donors, bank, own_schedule, own_values, ref, seed))


def _transplant_by_fold(led: Ledger, stays: np.ndarray, donors: np.ndarray, bank: DonorBank,
                        own_schedule: bool, own_values: bool, seed: int, refs: Dict[int, Reference]) -> np.ndarray:
    """``transplant`` with each stay's reference built WITHOUT its own fold."""
    s = led.store
    lens = s.lens[stays].astype(np.int64)
    offs = np.r_[0, np.cumsum(lens)]
    X = np.empty((int(offs[-1]), s.X.shape[1]), dtype=np.float32)
    for k in np.unique(led.folds[stays]):
        pick = np.flatnonzero(led.folds[stays] == k)
        Xk = transplant(s, stays[pick], donors[pick], bank, own_schedule, own_values, refs[int(k)], seed)
        pos = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in pick])
        X[pos] = Xk
    return X


def _draw_donors(led: Ledger, stays: np.ndarray, rng) -> Tuple[np.ndarray, DonorBank]:
    """Uniform donors from the dev stays (never the patient itself), as bank indices."""
    pool = np.flatnonzero(led.folds >= 0)
    bank = DonorBank.from_store(led.store, pool)
    idx = rng.integers(0, len(pool), len(stays))
    clash = pool[idx] == stays
    while clash.any():
        idx[clash] = rng.integers(0, len(pool), int(clash.sum()))
        clash = pool[idx] == stays
    return idx, bank


def _dev_eval_stays(led: Ledger) -> np.ndarray:
    s = led.store
    has = np.zeros(s.n_patients, dtype=bool)
    has[np.unique(s.seg[led.eval_mask])] = True
    return np.flatnonzero((led.folds >= 0) & has)


def process_physiology(led: Ledger, models: Sequence[FoldModels], n_stays: Optional[int] = None,
                       seed: int = 0, n_boot: int = 200) -> Dict[str, Dict]:
    """The Care-Process Transplant for every model in ``models`` (shared edited cohorts)."""
    rng = np.random.default_rng(seed)
    s = led.store
    pool = _dev_eval_stays(led)
    stays = np.sort(rng.choice(pool, n_stays, replace=False)) if n_stays and n_stays < len(pool) else pool
    donors, bank = _draw_donors(led, stays, rng)
    refs = {int(k): reference(led, seed=seed, exclude_fold=int(k)) for k in np.unique(led.folds[stays])}
    factual = led.view(stays)
    preds = {fm.name: {"factual": fm.oof[factual.parent_rows]} for fm in models}
    for c, (cond, (own_sched, own_vals)) in enumerate(CONDITIONS.items()):
        v = led.view(stays, _transplant_by_fold(led, stays, donors, bank, own_sched, own_vals,
                                                seed + 1 + c, refs))
        allrows = np.arange(v.store.n_rows)
        for fm in models:
            preds[fm.name][cond] = fm.predict(v, allrows)
        del v
    out = {}
    subj = factual.store.subject[factual.store.seg]
    for fm in models:
        P = preds[fm.name]
        ok = factual.eval_mask & np.all([np.isfinite(P[c]) for c in P], axis=0)
        rows = np.flatnonzero(ok)
        y, w, hr = factual.y[rows], factual.design_w[rows], factual.store.hour[rows]
        if not len(rows) or y.min() == y.max():
            out[fm.name] = {"skipped": "no evaluable rows"}
            continue
        R = {c: Ranked(P[c][rows]) for c in P}

        def terms(ww):
            A = {c: R[c].auc(y, ww) for c in R}
            return A, {"phi_process": 0.5 * ((A["process_only"] - A["neither"]) + (A["factual"] - A["physiology_only"])),
                       "phi_physiology": 0.5 * ((A["physiology_only"] - A["neither"]) + (A["factual"] - A["process_only"])),
                       "process_reliance": A["factual"] - A["physiology_only"]}
        A, T = terms(w)
        codes, inv = np.unique(subj[rows], return_inverse=True)
        draws = {k: [] for k in T}
        for _ in range(n_boot):
            m = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[inv]
            _, t = terms(w * m)
            for k in t:
                draws[k].append(t[k])
        out[fm.name] = {
            "auroc": A, "auroc_hourmatched": {c: stratified_auc(y, P[c][rows], hr, w) for c in P},
            **{k: {"value": T[k], "ci": tuple(np.nanpercentile(draws[k], [2.5, 97.5]))} for k in T},
            "n_stays": int(len(np.unique(factual.store.seg[rows]))), "n_rows": int(len(rows))}
    return out


def learnable(led: Ledger, cand, n_stays: Optional[int] = None, seed: int = 0, n_boot: int = 200) -> Dict:
    """The transplant's second question. ``process_physiology`` asks how much a FITTED model
    leans on the care process; this asks how much ANY model can learn from each part of
    the record. The candidate is RETRAINED inside each edited world — same stays, same
    subject folds, same training size — so the three AUROCs differ only in what the world
    lets a model see:

        factual           schedule and values (the ordinary number, at this sample size)
        physiology_only   values on a label-independent schedule: the physiology ceiling
        process_only      the schedule with neutral values: the care-process ceiling

    ``learnable_process_information`` = factual - physiology_only (with a paired subject
    bootstrap): what a model gains from being allowed to read the clinicians. A model
    trained in the physiology-only world is process-invariant by construction."""
    rng = np.random.default_rng(seed)
    s = led.store
    pool = np.flatnonzero(led.folds >= 0)       # every dev stay trains; early-warning rows score
    stays = np.sort(rng.choice(pool, n_stays, replace=False)) if n_stays and n_stays < len(pool) else pool
    donors, bank = _draw_donors(led, stays, rng)
    refs = {int(k): reference(led, seed=seed, exclude_fold=int(k)) for k in np.unique(led.folds[stays])}
    oof, ref = {}, None
    for c, cond in enumerate(("factual", "physiology_only", "process_only")):
        v = led.view(stays) if cond == "factual" else \
            led.view(stays, _transplant_by_fold(led, stays, donors, bank, *CONDITIONS[cond], seed + 1 + c, refs))
        oof[cond] = v.fit_oof(cand, verbose=False).oof
        if ref is None:
            ref = v
        del v
    ok = ref.eval_mask & (ref.folds[ref.store.seg] >= 0) & np.all([np.isfinite(o) for o in oof.values()], axis=0)
    rows = np.flatnonzero(ok)
    y, w = ref.y[rows], ref.design_w[rows]
    if not len(rows) or y.min() == y.max():
        return {"skipped": "no evaluable rows"}
    R = {c: Ranked(o[rows]) for c, o in oof.items()}
    A = {c: R[c].auc(y, w) for c in R}
    codes, inv = np.unique(ref.store.subject[ref.store.seg[rows]], return_inverse=True)
    d_proc, d_phys = [], []
    for _ in range(n_boot):
        m = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[inv]
        a = {c: R[c].auc(y, w * m) for c in R}
        d_proc.append(a["factual"] - a["physiology_only"])
        d_phys.append(a["factual"] - a["process_only"])
    return {"auroc_retrained": A,
            "learnable_process_information": {"value": A["factual"] - A["physiology_only"],
                                              "ci": tuple(np.nanpercentile(d_proc, [2.5, 97.5]))},
            "learnable_physiology_beyond_process": {"value": A["factual"] - A["process_only"],
                                                    "ci": tuple(np.nanpercentile(d_phys, [2.5, 97.5]))},
            "n_stays": int(len(stays)), "n_rows": int(len(rows)), "candidate": cand.name}


# --------------------------------------------------------------------------- #
# The admission audit
# --------------------------------------------------------------------------- #
def _scramble(X: np.ndarray, rows: np.ndarray, store: E.MimicStore, rng) -> None:
    """Rewrite every base (non-derived) dynamic channel on ``rows``: values drawn from the
    channel's own distribution, half of them missing."""
    for c in E.DYNAMIC:
        if c in DERIVED_CH:
            continue
        j = store.columns.index(c)
        pool = store.X[:, j]
        pool = pool[np.isfinite(pool)]
        vals = rng.choice(pool, len(rows)) if len(pool) else rng.normal(size=len(rows))
        X[rows, j] = np.where(rng.random(len(rows)) < 0.5, np.nan, vals)


def _truncated(led: Ledger, base: Ledger, cuts: np.ndarray) -> Ledger:
    s = base.store
    keep = np.concatenate([np.arange(s.starts[k], s.starts[k] + cuts[k] + 1) for k in range(s.n_patients)])
    lens = (cuts + 1).astype(np.int64)
    st = E.MimicStore(X=s.X[keep], y=s.y[keep], lens=lens, pid=s.pid, site=s.site, columns=list(s.columns),
                      tiers=dict(s.tiers), static_columns=list(s.static_columns), meta=s.meta,
                      onsets=s.onsets, sofa={})
    st = E.rederive(st, st.X)
    F, names = build_features(st, dtype=led.F.dtype)
    return Ledger(store=st, F=F, names=names, ftier=led.ftier, folds=base.folds, rule=led.rule,
                  anchor=led.anchor, design_w=np.ones(st.n_rows), covert=led.covert, parent_rows=keep)


def audit(led: Ledger, models: Sequence[FoldModels], n_stays: int = 8, seed: int = 0,
          tol: float = 1e-5) -> Dict[str, Dict]:
    """Attack every model with edited patients; ``admissible`` only if nothing moved."""
    rng = np.random.default_rng(seed)
    s = led.store
    folds = [k for k in range(led.n_folds) if all(k in fm.predictors for fm in models)]
    k = max(folds, key=lambda f: int(((led.folds == f) & (s.lens >= 6)).sum()))
    pool = np.flatnonzero((led.folds == k) & (s.lens >= 6))
    stays = np.sort(rng.choice(pool, min(n_stays, len(pool)), replace=False))
    base = led.view(stays)
    bs = base.store
    rows = np.arange(bs.n_rows)
    cuts = np.array([int(rng.integers(1, bs.lens[i] - 1)) for i in range(bs.n_patients)])
    before = bs.hour <= cuts[bs.seg]
    p0 = {fm.name: fm.predict(base, rows) for fm in models}

    X1 = bs.X.copy()
    _scramble(X1, np.flatnonzero(~before), s, rng)
    fut = led.view(stays, X1)
    trunc = _truncated(led, base, cuts)
    half = bs.n_patients // 2
    target = bs.seg < half
    X3 = bs.X.copy()
    _scramble(X3, np.flatnonzero(~target), s, rng)
    iso = led.view(stays, X3)

    out = {}
    for fm in models:
        a = p0[fm.name]
        dev = {
            "reproducible": float(np.nanmax(np.abs(a - fm.oof[base.parent_rows]))),
            "future_rewritten": float(np.nanmax(np.abs(fm.predict(fut, rows)[before] - a[before]))),
            "stay_truncated": float(np.nanmax(np.abs(fm.predict(trunc, np.arange(trunc.store.n_rows)) - a[before]))),
            "neighbours_scrambled": float(np.nanmax(np.abs(fm.predict(iso, rows)[target] - a[target]))),
        }
        passed = {kk: bool(v <= tol) for kk, v in dev.items()}
        out[fm.name] = {"max_abs_change": dev, "passed": passed, "admissible": all(passed.values()),
                        "stays": int(len(stays)), "fold": int(k)}
    return out
