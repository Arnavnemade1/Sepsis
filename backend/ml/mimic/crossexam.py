"""The cross-examination — the questions a skeptical intensivist asks of any AUROC.

A pooled hourly AUROC answers "does the score rank label-1 hours above label-0 hours?".
Every question below is a way that number can be high for a reason nobody would deploy:

 1. Is it predicting, or re-deriving the label?          ``definitional``
    Sepsis-3 is a FORMULA over inputs the model reads (SOFA components, and at tier S the
    culture / antibiotic pair). At and after onset a model can score well by recomputing
    it. Every model is split into PREDICTION (label-1 hours before the onset hour) and
    DETECTION (onset hour to onset + 3 h), and set against the definitional oracle — the
    label's own formula evaluated on what that tier can see. The margin over the oracle
    is what the model knows that the definition does not.
 2. Is it detecting sepsis, or just sickness?            ``contrasts``
    Sepsis-3 is infection AND organ dysfunction. A severity score separates septic hours
    from well patients and fails against the two halves on their own. Septic hours are set
    against event-aligned negatives: infection-only stays around their suspicion time,
    organ-dysfunction-only stays around their measured SOFA rise, and stays with neither.
    ``specificity_index`` = the weaker of the two halves.
 3. Did it know anything the clinicians hadn't acted on? ``clinician_conditional``
    A "clinician model" is fitted on everything clinicians' ACTIONS reveal (orders,
    treatments, and when / how often each variable was measured — never a value). The
    model is then scored only against hours the clinician model found equally suspicious
    (a stratified AUROC over its deciles). 0.5 = nothing beyond what the team's actions
    already showed. The pre-suspicion AUROC (Kamran et al., NEJM AI 2024) is the special
    case "no order yet"; this is its continuous generalisation.
 4. How early — really?                                  ``horizon``
    "Lead time" measured on alarms is stay geometry (docs: lead-time null). Here: for each
    horizon h, the patients who will become septic h hours after THIS hour, against
    never-septic patients at the SAME ICU hour (clock-matched), stays with onset on
    arrival excluded.
 5. Would it survive another onset definition?           ``ambiguity``
    Rules disagree about WHEN sepsis began (docs: MIMIC §2). Every hour gets a soft label
    — the share of the 9 definitions (3 rules x 3 anchors) calling it positive — and the
    AUROC is recomputed on soft labels and on the unambiguous hours only.
 6. For whom does it fail?                               ``subgroups``
    Design-weighted AUROC per level of every audit-only attribute (sex, age band, race,
    insurance, admission type, care unit, era), each against its complement with a
    subject-bootstrap interval, Holm-corrected. The attributes never enter a model.
 7. What would a ward pay to run it?                     ``alert_frontier``
    Alerts with a snooze, at fixed alert budgets per patient-day: the share of septic
    episodes alerted in [onset - 12 h, onset + 3 h], before onset, and the alert PPV.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Optional, Sequence

import numpy as np

from backend.ml.genesis.harness import Ranked
from backend.ml.mimic import extract as E
from backend.ml.mimic import sepsis3 as S3
from backend.ml.mimic.ledger import ONSET_MIN, Ledger, pairs_auc, stratified_auc

DEFINITIONS = [(r, a) for r in E.RULES for a in ("min", "si", "sofa")]
HORIZONS = (0, 1, 2, 3, 4, 6, 8, 12, 24)
BUDGETS = (0.25, 0.5, 1.0, 2.0)
NAN = float("nan")
_Z = NormalDist()


def _rows(led: Ledger, p: np.ndarray) -> np.ndarray:
    return np.flatnonzero(led.eval_mask & (led.folds[led.store.seg] >= 0) & np.isfinite(p))


def _auc(s, y, w, pos=None) -> float:
    """AUROC of ``pos`` rows (default y == 1) against the y == 0 rows."""
    pos = (y == 1) if pos is None else pos
    neg = y == 0
    if not pos.any() or not neg.any():
        return NAN
    return pairs_auc(s, w * pos, w * neg)


def onset_row(led: Ledger) -> np.ndarray:
    """(N,) the hour of the row's stay's onset (floor), NaN if never septic."""
    return np.floor(led.onset)[led.store.seg]


# --------------------------------------------------------------------------- #
# 1. definitional oracle
# --------------------------------------------------------------------------- #
def oracle(led: Ledger, tier: str) -> np.ndarray:
    """The label's own formula as a score, from what ``tier`` can see: the measured SOFA
    rise (physiology components only at tier P), plus suspicion-known at tier S."""
    key = ("oracle", tier)
    if key not in led.cache:
        s = led.store
        col = lambda nm: s.col(nm).astype(np.float64)
        parts = S3.component_points_now(col, s.row_start, s.hour, physiology_only=tier == "P")
        r = S3.measured_rise(parts, s.row_start, 48)
        if tier == "S":
            fsi = s.onsets.first_si.to_numpy()[s.seg]
            r = r + 2.0 * (np.isfinite(fsi) & (fsi <= s.hour + 1))
        led.cache[key] = r
    return led.cache[key]


def definitional(led: Ledger, p: np.ndarray, tier: str) -> Dict:
    rows = _rows(led, p)
    y, w, hr = led.y[rows], led.design_w[rows], led.store.hour[rows]
    o = onset_row(led)[rows]
    pred = (y == 1) & (hr < o)
    det = (y == 1) & (hr >= o)
    orc = oracle(led, tier)[rows]
    m = {"all": _auc(p[rows], y, w), "prediction": _auc(p[rows], y, w, pred), "detection": _auc(p[rows], y, w, det)}
    q = {"all": _auc(orc, y, w), "prediction": _auc(orc, y, w, pred), "detection": _auc(orc, y, w, det)}
    return {"tier": tier, "model": m, "oracle": q,
            "margin": {k: m[k] - q[k] for k in m},
            "n_pos": {"prediction": int(pred.sum()), "detection": int(det.sum())}}


# --------------------------------------------------------------------------- #
# 2. sepsis vs sickness
# --------------------------------------------------------------------------- #
def first_rise(led: Ledger) -> np.ndarray:
    """(P,) first hour each stay's measured SOFA rise reaches 2 (the organ half of the
    label, with or without infection), NaN if never."""
    if "first_rise" not in led.cache:
        s = led.store
        col = lambda nm: s.col(nm).astype(np.float64)
        hit = S3.measured_rise(S3.component_points_now(col, s.row_start, s.hour), s.row_start, 48) >= 2
        big = np.iinfo(np.int64).max
        first = np.minimum.reduceat(np.where(hit, s.hour, big), s.starts) if s.n_rows else np.zeros(0)
        led.cache["first_rise"] = np.where(first == big, np.nan, first).astype(float)
    return led.cache["first_rise"]


def contrast_groups(led: Ledger) -> Dict[str, np.ndarray]:
    s = led.store
    fsi = s.onsets.first_si.to_numpy()
    fr = first_rise(led)
    dev = led.folds >= 0
    septic = np.isfinite(led.onset)
    return {"septic": dev & septic & (led.onset >= ONSET_MIN),
            "infection_only": dev & ~septic & np.isfinite(fsi),
            "organ_only": dev & ~septic & ~np.isfinite(fsi) & np.isfinite(fr),
            "neither": dev & ~septic & ~np.isfinite(fsi) & ~np.isfinite(fr)}


def contrasts(led: Ledger, p: np.ndarray, before: int = 6, after: int = 3) -> Dict:
    s = led.store
    g = contrast_groups(led)
    seg, hr = s.seg, s.hour
    ok = np.isfinite(p) & led.train_mask
    pos = np.flatnonzero(ok & led.eval_mask & (led.y == 1) & g["septic"][seg])
    event = {"infection_only": s.onsets.first_si.to_numpy(), "organ_only": first_rise(led)}
    out = {"event_aligned": {}, "all_hours": {}, "n_stays": {k: int(v.sum()) for k, v in g.items()}}
    for name in ("infection_only", "organ_only", "neither"):
        in_g = ok & g[name][seg]
        if name in event:
            t = np.floor(event[name])[seg]
            aligned = in_g & (hr >= t - before) & (hr <= t + after)
        else:
            aligned = in_g
        for key, neg in (("event_aligned", np.flatnonzero(aligned)), ("all_hours", np.flatnonzero(in_g))):
            if not len(pos) or not len(neg):
                out[key][name] = NAN
                continue
            r = np.r_[pos, neg]
            yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            out[key][name] = pairs_auc(p[r], led.design_w[r] * yy, led.design_w[r] * (1 - yy))
    ea = out["event_aligned"]
    halves = [ea[k] for k in ("infection_only", "organ_only") if np.isfinite(ea[k])]
    out["specificity_index"] = min(halves) if len(halves) == 2 else NAN
    return out


# --------------------------------------------------------------------------- #
# 3. beyond the clinicians
# --------------------------------------------------------------------------- #
def clinician_conditional(led: Ledger, p: np.ndarray, clinician: np.ndarray, n_bins: int = 10) -> Dict:
    rows = _rows(led, p)
    rows = rows[np.isfinite(clinician[rows])]
    y, w, c = led.y[rows], led.design_w[rows], clinician[rows]
    if not len(rows) or y.min() == y.max():
        return {"conditional_auroc": NAN}
    edges = np.unique(np.quantile(c, np.linspace(0, 1, n_bins + 1)[1:-1]))
    strata = np.searchsorted(edges, c, side="right")
    return {"clinician_auroc": Ranked(c).auc(y, w), "model_auroc": Ranked(p[rows]).auc(y, w),
            "conditional_auroc": stratified_auc(y, p[rows], strata, w),
            "n_strata": int(len(np.unique(strata)))}


# --------------------------------------------------------------------------- #
# 4. how early
# --------------------------------------------------------------------------- #
def horizon(led: Ledger, p: np.ndarray, horizons: Sequence[int] = HORIZONS) -> Dict:
    s = led.store
    dev = led.folds >= 0
    sep = np.flatnonzero(dev & np.isfinite(led.onset) & (led.onset >= ONSET_MIN))
    never = dev & ~np.isfinite(led.onset)
    neg = np.flatnonzero(never[s.seg] & np.isfinite(p))
    out = {"horizon_h": list(horizons), "auroc_clock_matched": [], "n_pos": []}
    for h in horizons:
        r = np.floor(led.onset[sep]).astype(int) - h
        okk = (r >= 0) & (r < s.lens[sep])
        pos = s.starts[sep[okk]] + r[okk]
        pos = pos[np.isfinite(p[pos])]
        nn = neg[np.isin(s.hour[neg], s.hour[pos])]
        if not len(pos) or not len(nn):
            out["auroc_clock_matched"].append(NAN)
            out["n_pos"].append(int(len(pos)))
            continue
        rr = np.r_[pos, nn]
        yy = np.r_[np.ones(len(pos)), np.zeros(len(nn))].astype(np.int8)
        out["auroc_clock_matched"].append(stratified_auc(yy, p[rr], s.hour[rr], led.design_w[rr]))
        out["n_pos"].append(int(len(pos)))
    return out


# --------------------------------------------------------------------------- #
# 5. onset ambiguity
# --------------------------------------------------------------------------- #
def soft_labels(led: Ledger) -> np.ndarray:
    if "soft" not in led.cache:
        led.cache["soft"] = np.mean([led.store.label_for(r, a) for r, a in DEFINITIONS], axis=0)
    return led.cache["soft"]


def soft_auc(s: np.ndarray, q: np.ndarray, w: np.ndarray) -> float:
    """AUROC under soft labels: every pair (i, j), i != j, weighted w_i q_i w_j (1 - q_j)."""
    wp, wn = w * q, w * (1 - q)
    A = pairs_auc(s, wp, wn)
    self_w = float((w * w * q * (1 - q)).sum())
    den = float(wp.sum() * wn.sum())
    if not np.isfinite(A) or den - self_w <= 0:
        return NAN
    return (A * den - 0.5 * self_w) / (den - self_w)


def ambiguity(led: Ledger, p: np.ndarray) -> Dict:
    rows = _rows(led, p)
    y, w, q = led.y[rows], led.design_w[rows], soft_labels(led)[rows]
    sc = p[rows]
    unamb = (q == 0) | (q == 1)
    return {"auroc_primary": _auc(sc, y, w), "auroc_soft": soft_auc(sc, q, w),
            "auroc_unambiguous": _auc(sc[unamb], y[unamb], w[unamb]),
            "ambiguous_share_pos": float((w * ((y == 1) & (q < 1))).sum() / max((w * (y == 1)).sum(), 1e-12)),
            "ambiguous_share_neg": float((w * ((y == 0) & (q > 0))).sum() / max((w * (y == 0)).sum(), 1e-12)),
            "n_definitions": len(DEFINITIONS)}


# --------------------------------------------------------------------------- #
# 6. subgroups
# --------------------------------------------------------------------------- #
def subgroups(led: Ledger, p: np.ndarray, n_boot: int = 200, seed: int = 0, alpha: float = 0.05,
              min_pos_stays: int = 3) -> Dict:
    s = led.store
    rows = _rows(led, p)
    y, w = led.y[rows], led.design_w[rows]
    R = Ranked(p[rows])
    seg = s.seg[rows]
    overall = R.auc(y, w)
    tests = []
    groups: Dict[str, Dict] = {}
    for attr in E.AUDIT_ATTRS:
        lv = s.meta[attr].astype(str).to_numpy()[seg]
        groups[attr] = {}
        for level in np.unique(lv):
            g = lv == level
            n_pos_stays = int(len(np.unique(seg[g & (y == 1)])))
            info = {"n_stays": int(len(np.unique(seg[g]))), "n_pos_stays": n_pos_stays}
            if n_pos_stays < min_pos_stays or not (g & (y == 0)).any() or not (~g & (y == 1)).any():
                groups[attr][level] = info | {"auroc": NAN}
                continue
            info["auroc"] = R.auc(y, w * g)
            info["gap"] = info["auroc"] - R.auc(y, w * ~g)
            groups[attr][level] = info
            tests.append((attr, level, g))
    subj = s.subject[seg]
    codes, inv = np.unique(subj, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, len(tests)), np.nan)
    for b in range(n_boot):
        m = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[inv]
        ww = w * m
        for t, (_, _, g) in enumerate(tests):
            draws[b, t] = R.auc(y, ww * g) - R.auc(y, ww * ~g)
    pvals = []
    for t, (attr, level, _) in enumerate(tests):
        d = draws[:, t]
        d = d[np.isfinite(d)]
        info = groups[attr][level]
        info["ci"] = tuple(np.percentile(d, [2.5, 97.5])) if len(d) else (NAN, NAN)
        # Wald test on the bootstrap SE: a percentile p-value cannot go below 1/n_boot,
        # which a Holm correction over a dozen groups would never reject
        se = float(np.std(d, ddof=1)) if len(d) > 1 else NAN
        info["p"] = float(2 * (1 - _Z.cdf(abs(info["gap"]) / se))) if se > 0 else 1.0
        pvals.append(info["p"])
    order = np.argsort(pvals)
    reject = np.zeros(len(tests), dtype=bool)
    for i, t in enumerate(order):                                   # Holm step-down
        if pvals[t] <= alpha / (len(tests) - i):
            reject[t] = True
        else:
            break
    for t, (attr, level, _) in enumerate(tests):
        groups[attr][level]["flag_below"] = bool(reject[t] and groups[attr][level]["gap"] < 0)
    worst = {}
    for attr, lv in groups.items():
        scored = {k: v["auroc"] for k, v in lv.items() if np.isfinite(v.get("auroc", NAN))}
        if scored:
            k = min(scored, key=scored.get)
            worst[attr] = {"level": k, "auroc": scored[k]}
    return {"overall": overall, "groups": groups, "worst": worst, "n_tests": len(tests),
            "flagged": [f"{a}={l}" for t, (a, l, _) in enumerate(tests) if groups[a][l]["flag_below"]]}


# --------------------------------------------------------------------------- #
# 7. alert burden
# --------------------------------------------------------------------------- #
def _fire(score: np.ndarray, seg: np.ndarray, hr: np.ndarray, theta: float, snooze: int) -> np.ndarray:
    fired = []
    last_s, last_h = -1, -10 ** 9
    for i in np.flatnonzero(score >= theta):
        if seg[i] != last_s or hr[i] >= last_h + snooze:
            fired.append(i)
            last_s, last_h = seg[i], hr[i]
    return np.array(fired, dtype=np.int64)


def alert_frontier(led: Ledger, p: np.ndarray, budgets: Sequence[float] = BUDGETS, snooze: int = 6,
                   early: int = 12, late: int = 3) -> Dict:
    s = led.store
    rows = _rows(led, p)
    seg, hr, w, sc = s.seg[rows], s.hour[rows], led.design_w[rows], p[rows]
    o = onset_row(led)[rows]
    window = np.isfinite(o) & (hr >= o - early) & (hr <= o + late)
    pt_days = float(w.sum()) / 24.0
    septic = np.unique(seg[np.isfinite(o)])

    def evaluate(theta):
        f = _fire(sc, seg, hr, theta, snooze)
        if not len(f):
            return {"alerts_per_pt_day": 0.0}
        wa, tru = w[f], window[f]
        caught = np.unique(seg[f[tru]])
        pre = np.unique(seg[f[tru & (hr[f] < o[f])]])
        return {"threshold": float(theta), "alerts_per_pt_day": float(wa.sum()) / pt_days,
                "false_alerts_per_pt_day": float((wa * ~tru).sum()) / pt_days,
                "ppv": float((wa * tru).sum() / wa.sum()),
                "episodes_caught": float(len(caught) / max(len(septic), 1)),
                "episodes_caught_before_onset": float(len(pre) / max(len(septic), 1))}
    out = {"snooze_h": snooze, "window_h": [-early, late], "septic_episodes": int(len(septic)),
           "patient_days": pt_days, "budgets": {}}
    if not len(rows):
        return out
    for b in budgets:
        lo, hi = 0.0, 1.0                                  # quantile of the score
        best = None
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            r = evaluate(float(np.quantile(sc, mid)))
            if r["alerts_per_pt_day"] > b:
                lo = mid
            else:
                hi, best = mid, r
        out["budgets"][str(b)] = best or evaluate(float(np.quantile(sc, hi)))
    return out


# --------------------------------------------------------------------------- #
def examine(led: Ledger, p: np.ndarray, tier: str, clinician: Optional[np.ndarray] = None,
            n_boot: int = 200) -> Dict:
    out = {"definitional": definitional(led, p, tier), "contrasts": contrasts(led, p),
           "horizon": horizon(led, p), "ambiguity": ambiguity(led, p),
           "subgroups": subgroups(led, p, n_boot=n_boot), "alert_frontier": alert_frontier(led, p)}
    if clinician is not None:
        out["clinician_conditional"] = clinician_conditional(led, p, clinician)
    return out
