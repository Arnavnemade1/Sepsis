"""Clinical evaluation metrics for hourly sepsis prediction.

Because sepsis is rare per-hour (~2% prevalence in CinC 2019), plain accuracy is
meaningless: a model predicting "never septic" scores ~98%. We therefore report:

  * AUROC / AUPRC                – ranking quality (AUPRC is the honest one here)
  * Brier score + ECE           – probability calibration
  * sensitivity @ fixed spec     – the operating point a clinician actually sets
  * patient-level early warning  – median hours of lead time before onset,
                                   and false-alarm rate among non-septic patients
  * normalized utility           – the OFFICIAL CinC 2019 score (see below)

The utility functions ``compute_prediction_utility`` and the normalization are
adapted directly from the official PhysioNet/CinC 2019 evaluation script
(physionetchallenges/evaluation-2019, MIT-licensed) to guarantee our number is
comparable to the published leaderboard. Default constants are the official ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from backend.ml.eval.runs import first_rows, patient_runs


# --------------------------------------------------------------------------- #
# Threshold-free ranking + calibration
# --------------------------------------------------------------------------- #
def auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, np.clip(p, 0.0, 1.0)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average gap between predicted confidence and observed frequency."""
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Return (mean_predicted, observed_freq, count) per bin for plotting."""
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mp, of, cnt = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        mp.append(float(p[mask].mean()))
        of.append(float(y[mask].mean()))
        cnt.append(int(mask.sum()))
    return np.array(mp), np.array(of), np.array(cnt)


# --------------------------------------------------------------------------- #
# Operating-point metrics
# --------------------------------------------------------------------------- #
def sensitivity_at_specificity(y: np.ndarray, p: np.ndarray, target_spec: float) -> Tuple[float, float]:
    """Max sensitivity achievable while holding specificity >= target.
    Returns (sensitivity, threshold)."""
    neg = p[y == 0]
    if neg.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, target_spec))            # spec target via neg quantile
    pred = p >= thr
    tp = float(((pred == 1) & (y == 1)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1.0)
    return sens, thr


def specificity_at_sensitivity(y: np.ndarray, p: np.ndarray, target_sens: float) -> Tuple[float, float]:
    pos = p[y == 1]
    if pos.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(pos, 1.0 - target_sens))
    pred = p >= thr
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    spec = tn / max(tn + fp, 1.0)
    return spec, thr


# --------------------------------------------------------------------------- #
# Patient-level early-warning metrics
# --------------------------------------------------------------------------- #
def early_warning(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    onset_hour: np.ndarray,
    hour_index: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Per-patient: did we alarm before the labelled onset, and how early?

    * lead_time_hours: for septic patients we alarmed on, hours between first
      alarm and labelled onset (positive = warned early).
    * detection_rate: fraction of septic patients alarmed on at/before onset.
    * false_alarm_rate: fraction of non-septic patients ever alarmed.
    """
    # One sort instead of a scan per patient (backend/ml/eval/runs.py). Only the
    # EARLIEST alarm at or before onset and WHETHER a non-septic stay ever alarmed
    # matter, so both are per-patient reductions and hour order within a stay is moot.
    order, starts, ends = patient_runs(groups)
    onset = np.asarray(onset_hour)[first_rows(order, starts)].astype(np.int64)
    hs = np.asarray(hour_index)[order]
    alarm = np.asarray(p)[order] >= threshold
    septic = onset >= 0
    seg = np.repeat(np.arange(len(starts)), ends - starts)
    pre = alarm & septic[seg] & (hs <= onset[seg])
    first_alarm = (np.minimum.reduceat(np.where(pre, hs, np.inf), starts)
                   if len(starts) else np.zeros(0))
    hit = septic & (first_alarm < np.inf)
    lead_times = (onset[hit] - first_alarm[hit]).astype(float)
    detected, septic_total = int(hit.sum()), int(septic.sum())
    nonseptic_total = int((~septic).sum())
    false_alarms = (int((~septic & np.logical_or.reduceat(alarm, starts)).sum())
                    if len(starts) else 0)

    return {
        "detection_rate": detected / max(septic_total, 1),
        "median_lead_time_hours": float(np.median(lead_times)) if lead_times.size else 0.0,
        "mean_lead_time_hours": float(np.mean(lead_times)) if lead_times.size else 0.0,
        "false_alarm_rate": false_alarms / max(nonseptic_total, 1),
        "septic_patients": septic_total,
        "nonseptic_patients": nonseptic_total,
    }


def _pre_onset_segments(y, groups, onset_hour, hour_index):
    """Per septic patient, the hours at or before onset, flattened with boundaries.

    Computed once. The null below evaluates hundreds of candidate thresholds and a naive
    implementation re-walked all 8,000 patients for each one, which turned a diagnostic
    into a ten-minute job.
    """
    groups = np.asarray(groups)
    order = np.lexsort((np.asarray(hour_index), groups))
    g = groups[order]
    hs = np.asarray(hour_index)[order]
    os_ = np.asarray(onset_hour)[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    bounds = np.r_[starts, len(g)]
    hours, onsets, lens = [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        onset = int(os_[lo])
        if onset < 0:
            continue
        h_pre = hs[lo:hi][hs[lo:hi] <= onset]
        if h_pre.size:
            hours.append(h_pre.astype(float))
            onsets.append(float(onset))
            lens.append(h_pre.size)
    if not hours:
        return np.array([]), np.array([]), np.array([], dtype=int)
    return (np.concatenate(hours), np.asarray(onsets, dtype=float),
            np.asarray(lens, dtype=int))


def _null_lead_at_detection(flat_hours, onsets, lens, target_det, rng):
    """One draw: random scores, threshold tuned to ``target_det``, median lead returned.

    Fully vectorised over patients. Segment boundaries let the first crossing be found
    with one ``np.maximum.accumulate`` pass instead of a Python loop per patient.
    """
    n_pt = len(lens)
    if n_pt == 0:
        return float("nan")
    ends = np.cumsum(lens)
    begins = ends - lens
    scores = rng.random(flat_hours.size)

    lo, hi, best = 0.0, 1.0, None
    for _ in range(24):
        thr = 0.5 * (lo + hi)
        fired = scores >= thr
        # index of the first True inside each segment, or -1 when the segment never fires
        idx = np.full(n_pt, -1, dtype=np.int64)
        any_fire = np.zeros(n_pt, dtype=bool)
        for k in range(n_pt):          # cheap: n_pt is ~600, the inner work is numpy
            seg = fired[begins[k]:ends[k]]
            if seg.any():
                idx[k] = begins[k] + int(np.argmax(seg))
                any_fire[k] = True
        det = float(any_fire.mean())
        if det >= target_det:
            best = (idx.copy(), any_fire.copy())
            lo = thr                    # afford to be stricter
        else:
            hi = thr
    if best is None:
        return float("nan")
    idx, any_fire = best
    leads = onsets[any_fire] - flat_hours[idx[any_fire]]
    return float(np.median(leads)) if leads.size else float("nan")


def lead_time_vs_null(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    onset_hour: np.ndarray,
    hour_index: np.ndarray,
    threshold: float,
    n_draws: int = 25,
    seed: int = 0,
) -> Dict[str, float]:
    """Median lead time, and the median lead a RANDOM score achieves at the same
    detection rate. The difference is the only part that evidences early warning.

    WHY A BARE LEAD TIME MUST NOT BE REPORTED
    -----------------------------------------
    ``early_warning`` above returns ``median_lead_time_hours`` as
    ``onset - first_alarm_hour``. That number was measured against this null and found to
    carry no information about skill (``docs/LEAD_TIME_NULL.md``). At matched detection a
    pure random score produced 35.2 / 39.0 / 38.0 / 39.0 hours against the shipped
    detector's 40.0 / 40.0 / 38.5 / 38.5, and random *beat* the real model at 70%
    detection. A detector alarms at the FIRST crossing of its threshold, which for any low
    threshold lands near the start of a stay, so a long lead is manufactured by stay
    geometry and alarm rate rather than earned.

    Every lead time this project published before that audit — 26, 29, 34, 37, 39, 40h —
    sits inside this null. This function exists so no future one is quoted bare.

    THE MATCHING IS ON DETECTION RATE, NOT ALARM RATE
    -------------------------------------------------
    This is the part that is easy to get wrong, and getting it wrong reverses the
    conclusion. A detector firing on more stays reaches further back on average for
    reasons unrelated to skill, so the null is held at the SAME detection rate as the
    observed detector. The first version of this analysis matched on alarm rate, compared
    511 fired stays against 367, and would have reported a win.

    ``excess_lead_hours`` is the reportable quantity. It can be negative, and on this
    project's own detector it has been.
    """
    obs = early_warning(y, p, groups, onset_hour, hour_index, threshold)
    observed = float(obs["median_lead_time_hours"])
    target_det = float(obs["detection_rate"])
    blank = {"observed_lead_hours": observed, "null_lead_hours": float("nan"),
             "null_lo": float("nan"), "null_hi": float("nan"),
             "excess_lead_hours": float("nan"), "detection_rate": target_det,
             "n_draws": 0}
    if not np.isfinite(target_det) or target_det <= 0:
        return blank

    flat_hours, onsets, lens = _pre_onset_segments(y, groups, onset_hour, hour_index)
    if lens.size == 0:
        return blank
    rng = np.random.default_rng(seed)
    nulls = [_null_lead_at_detection(flat_hours, onsets, lens, target_det, rng)
             for _ in range(n_draws)]
    arr = np.asarray([x for x in nulls if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return blank
    lo, hi = np.percentile(arr, [2.5, 97.5])
    med = float(np.median(arr))
    return {
        "observed_lead_hours": observed,
        "null_lead_hours": med,
        "null_lo": float(lo), "null_hi": float(hi),
        "excess_lead_hours": float(observed - med),
        "detection_rate": target_det,
        "n_draws": int(arr.size),
    }


def budget_threshold(nonseptic_peaks: np.ndarray, far: float) -> float:
    """The smallest threshold whose alarmed fraction of non-septic patients is at most
    ``far``, alarming at ``score >= threshold``.

    Four copies of this used to take the k-th sorted non-septic peak as the threshold.
    That alarms every patient TIED at that peak, and one patient too many even without
    ties. Calibrated stacks emit a few hundred distinct scores, so ties are the rule:
    on the deployed artifact one isotonic step held 46% of non-septic peaks, and its
    "5% budget" alarmed 50.8% of non-septic patients. When the budget falls inside a
    tie, the honest threshold sits just above it and alarms fewer patients than the
    budget allows: a detector with a coarse score cannot use that budget."""
    pk = np.sort(np.asarray(nonseptic_peaks, dtype=float))
    n = len(pk)
    if n == 0:
        return float("nan")
    j = n - int(np.floor(far * n + 1e-9))        # peaks that must fall below threshold
    if j <= 0:
        return float(pk[0])
    if j >= n:
        return float(np.nextafter(pk[-1], np.inf))
    return float(pk[j] if pk[j - 1] < pk[j] else np.nextafter(pk[j], np.inf))


def detection_at_false_alarm_rate(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    onset_hour: np.ndarray,
    hour_index: np.ndarray,
    target_far: float = 0.15,
) -> Dict[str, float]:
    """How many septic patients are caught when the alarm budget is fixed.

    Comparing detectors at a shared *score* threshold — or at matched specificity —
    is misleading when their scores are shaped differently: a model whose output is
    smooth over time crosses any threshold on fewer patients, which looks like worse
    sensitivity when it is really a different point on the same curve. A ward has an
    alarm budget, not a threshold, so this fixes the thing the ward actually fixes —
    the fraction of non-septic patients ever alarmed — and reports what each detector
    buys inside it.

    The threshold comes from the *non-septic* patients' peak scores alone, so it is
    set without consulting a single septic outcome.
    """
    groups = np.asarray(groups)
    order = np.lexsort((hour_index, groups))
    g, ps, hs, os_ = groups[order], np.asarray(p)[order], np.asarray(hour_index)[order], np.asarray(onset_hour)[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    bounds = np.r_[starts, len(g)]

    nonseptic_peak, septic = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        onset = int(os_[a])
        if onset < 0:
            nonseptic_peak.append(ps[a:b].max())
        else:
            pre = ps[a:b][hs[a:b] <= onset]
            if pre.size:
                septic.append((onset, hs[a:b][hs[a:b] <= onset], pre))
    if not nonseptic_peak or not septic:
        return {"threshold": float("nan"), "detection_rate": float("nan"),
                "median_lead_time_hours": float("nan"), "realised_far": float("nan")}

    peaks = np.sort(np.asarray(nonseptic_peak))
    thr = budget_threshold(peaks, target_far)
    realised = float((peaks >= thr).mean())

    detected, leads = 0, []
    for onset, hours, pre in septic:
        hit = hours[pre >= thr]
        if hit.size:
            detected += 1
            leads.append(float(onset - hit.min()))
    return {
        "threshold": thr,
        "detection_rate": detected / len(septic),
        "median_lead_time_hours": float(np.median(leads)) if leads else 0.0,
        "realised_far": realised,
    }


# --------------------------------------------------------------------------- #
# Official CinC 2019 normalized utility
# (adapted from physionetchallenges/evaluation-2019, MIT license)
# --------------------------------------------------------------------------- #
def compute_prediction_utility(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: float = -12,
    dt_optimal: float = -6,
    dt_late: float = 3.0,
    max_u_tp: float = 1,
    min_u_fn: float = -2,
    u_fp: float = -0.05,
    u_tn: float = 0,
) -> float:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if np.any(labels):
        is_septic = True
        t_sepsis = int(np.argmax(labels)) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float("inf")
    n = len(labels)

    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:                       # TP
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:                 # FP
                u[t] = u_fp
            elif is_septic and not predictions[t]:                 # FN
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:             # TN
                u[t] = u_tn
    return float(np.sum(u))


def normalized_utility(
    y: np.ndarray,
    pred_binary: np.ndarray,
    groups: np.ndarray,
    hour_index: np.ndarray,
) -> float:
    """Official normalized utility across the cohort:
        (observed - inaction) / (best_possible - inaction)
    where inaction = predict all-zeros and best = predict the true labels."""
    obs = inact = best = 0.0
    # Patients in id order, each stay in hour order, from one sort. The scan this
    # replaced visited patients in set() order, which for string ids follows the
    # per-process hash seed, so its float sum was never reproducible across runs.
    order, starts, ends = patient_runs(groups, hour_index)
    ys = np.asarray(y)[order].astype(int)
    ps = np.asarray(pred_binary)[order].astype(int)
    for a, b in zip(starts, ends):
        yl, pl = ys[a:b], ps[a:b]
        obs += compute_prediction_utility(yl, pl)
        inact += compute_prediction_utility(yl, np.zeros_like(yl))
        best += compute_prediction_utility(yl, yl)
    denom = best - inact
    return float((obs - inact) / denom) if denom != 0 else float("nan")


# --------------------------------------------------------------------------- #
# Decision-curve analysis (net clinical benefit) + alarm burden
# --------------------------------------------------------------------------- #
def net_benefit(y: np.ndarray, p: np.ndarray, pt: float) -> float:
    """Net benefit at threshold probability ``pt`` (Vickers & Elkin 2006):
        NB = TP/N - FP/N * (pt / (1 - pt))
    i.e. true positives credited, false positives penalised by the odds a clinician
    would accept to act. Lets us compare against treat-all / treat-none policies."""
    n = len(y)
    if n == 0 or pt >= 1.0:
        return float("nan")
    pred = p >= pt
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    return float(tp / n - (fp / n) * (pt / (1.0 - pt)))


def decision_curve(y: np.ndarray, p: np.ndarray,
                   thresholds: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
    """Net benefit of the model vs treat-all and treat-none across threshold
    probabilities. Per-hour sepsis prevalence is ~2%, so the clinically relevant
    threshold-probability range is low (≈0.5%–20%)."""
    if thresholds is None:
        thresholds = np.linspace(0.005, 0.20, 40)
    prev = float(y.mean())
    model = [net_benefit(y, p, float(pt)) for pt in thresholds]
    treat_all = [float(prev - (1.0 - prev) * (pt / (1.0 - pt))) for pt in thresholds]
    return {"thresholds": [float(t) for t in thresholds], "model": model,
            "treat_all": treat_all, "treat_none": [0.0] * len(thresholds)}


def alarm_burden(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                 thresholds: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
    """Across alarm thresholds: detection rate (sensitivity over septic patients),
    false-alarm rate (fraction of non-septic patients ever alarmed), and alarm
    burden as false alerts per 100 non-septic patient-hours — the alarm-fatigue axis
    a clinician cares about as much as sensitivity."""
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.6, 30)
    # A patient alarms at t iff its peak score reaches t: one reduction per patient
    # (fmax ignores NaN rows, as ``(p >= t).any()`` did), then one comparison per t.
    order, starts, _ = patient_runs(groups)
    pos = np.maximum.reduceat(np.asarray(y)[order], starts) != 0
    peak = np.fmax.reduceat(np.asarray(p, dtype=float)[order], starts)
    npos, nneg = int(pos.sum()), int((~pos).sum())
    neg_hours = int((y == 0).sum())
    det, far, burden = [], [], []
    for t in thresholds:
        any_alarm = peak >= t
        alarmed_pos = int((pos & any_alarm).sum())
        alarmed_neg = int((~pos & any_alarm).sum())
        det.append(alarmed_pos / max(npos, 1))
        far.append(alarmed_neg / max(nneg, 1))
        fp_hours = int(((p >= t) & (y == 0)).sum())
        burden.append(100.0 * fp_hours / max(neg_hours, 1))
    return {"thresholds": [float(t) for t in thresholds], "detection_rate": det,
            "false_alarm_rate": far, "false_alerts_per_100h": burden}


def _rows_by_patient(groups: np.ndarray, uniq: np.ndarray) -> Dict[object, np.ndarray]:
    """``{g: np.where(groups == g)[0]}`` for every ``g`` in ``uniq`` (sorted), from one
    stable sort: the same arrays, ascending, without a scan per patient."""
    order, starts, ends = patient_runs(groups)
    if not np.array_equal(np.asarray(groups)[order[starts]], uniq):
        raise ValueError("patient ids do not sort the way sorted() orders them")
    return {g: order[a:b] for g, a, b in zip(uniq, starts, ends)}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                 stat: str = "auroc", n_boot: int = 500, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Patient-level (cluster) bootstrap 95% CI. Resampling PATIENTS (not rows)
    respects the within-patient correlation, so the interval is honest."""
    fn = {"auroc": auroc, "auprc": auprc, "brier": brier}[stat]
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    idx_by = _rows_by_patient(groups, uniq)
    vals: List[float] = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[g] for g in samp])
        yb, pb = y[rows], p[rows]
        if len(np.unique(yb)) > 1:
            v = fn(yb, pb)
            if not np.isnan(v):
                vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------------- #
# One-shot report
# --------------------------------------------------------------------------- #
@dataclass
class EvalReport:
    auroc: float
    auprc: float
    brier: float
    ece: float
    prevalence: float
    operating_points: Dict[str, Dict[str, float]] = field(default_factory=dict)
    early_warning: Dict[str, float] = field(default_factory=dict)
    normalized_utility: float = float("nan")

    def pretty(self) -> str:
        lines = [
            f"  AUROC                 : {self.auroc:.4f}",
            f"  AUPRC                 : {self.auprc:.4f}   (prevalence {self.prevalence:.4f})",
            f"  Brier / ECE           : {self.brier:.4f} / {self.ece:.4f}",
            f"  Normalized utility    : {self.normalized_utility:.4f}   (official CinC 2019)",
        ]
        for name, op in self.operating_points.items():
            lines.append(
                f"  {name:<21}: sens={op['sensitivity']:.3f} spec={op['specificity']:.3f} thr={op['threshold']:.3f}"
            )
        if self.early_warning:
            ew = self.early_warning
            lines.append(
                f"  Early warning         : detect={ew['detection_rate']:.3f} "
                f"median_lead={ew['median_lead_time_hours']:.1f}h "
                f"false_alarm={ew['false_alarm_rate']:.3f}"
            )
        return "\n".join(lines)


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    groups: Optional[np.ndarray] = None,
    hour_index: Optional[np.ndarray] = None,
    onset_hour: Optional[np.ndarray] = None,
    spec_targets: Tuple[float, ...] = (0.85, 0.90, 0.95),
) -> EvalReport:
    """Compute the full report. Group/hour/onset arrays unlock the patient-level
    and official-utility metrics; omit them for a quick row-level summary."""
    rep = EvalReport(
        auroc=auroc(y, p),
        auprc=auprc(y, p),
        brier=brier(y, p),
        ece=expected_calibration_error(y, p),
        prevalence=float(y.mean()),
    )
    for spec in spec_targets:
        sens, thr = sensitivity_at_specificity(y, p, spec)
        # realised specificity at that threshold
        pred = p >= thr
        tn = float(((pred == 0) & (y == 0)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        rep.operating_points[f"sens@spec>={spec:.2f}"] = {
            "sensitivity": sens,
            "specificity": tn / max(tn + fp, 1.0),
            "threshold": thr,
        }

    if groups is not None and hour_index is not None:
        # Use the spec>=0.90 threshold as the default alarm operating point.
        op = rep.operating_points.get("sens@spec>=0.90")
        thr = op["threshold"] if op else 0.5
        pred_bin = (p >= thr).astype(int)
        rep.normalized_utility = normalized_utility(y, pred_bin, groups, hour_index)
        if onset_hour is not None:
            rep.early_warning = early_warning(y, p, groups, onset_hour, hour_index, thr)
    return rep
