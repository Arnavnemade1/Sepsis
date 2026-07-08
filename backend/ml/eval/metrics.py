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
    lead_times: List[float] = []
    detected = 0
    septic_total = 0
    false_alarms = 0
    nonseptic_total = 0

    for g in set(groups.tolist()):
        mask = groups == g
        pg = p[mask]
        hg = hour_index[mask]
        order = np.argsort(hg)
        pg, hg = pg[order], hg[order]
        onset = int(onset_hour[mask][0])
        alarm_hours = hg[pg >= threshold]

        if onset >= 0:  # septic patient
            septic_total += 1
            pre = alarm_hours[alarm_hours <= onset]
            if pre.size:
                detected += 1
                lead_times.append(float(onset - pre.min()))
        else:           # non-septic patient
            nonseptic_total += 1
            if alarm_hours.size:
                false_alarms += 1

    return {
        "detection_rate": detected / max(septic_total, 1),
        "median_lead_time_hours": float(np.median(lead_times)) if lead_times else 0.0,
        "mean_lead_time_hours": float(np.mean(lead_times)) if lead_times else 0.0,
        "false_alarm_rate": false_alarms / max(nonseptic_total, 1),
        "septic_patients": septic_total,
        "nonseptic_patients": nonseptic_total,
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
    for g in set(groups.tolist()):
        mask = groups == g
        order = np.argsort(hour_index[mask])
        yl = y[mask][order].astype(int)
        pl = pred_binary[mask][order].astype(int)
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
    uniq = list(set(groups.tolist()))
    pos_g = {g: int(y[groups == g].max()) for g in uniq}
    npos = sum(1 for g in uniq if pos_g[g])
    nneg = sum(1 for g in uniq if not pos_g[g])
    neg_hours = int((y == 0).sum())
    det, far, burden = [], [], []
    for t in thresholds:
        alarmed_pos = alarmed_neg = 0
        for g in uniq:
            m = groups == g
            any_alarm = bool((p[m] >= t).any())
            if pos_g[g] and any_alarm:
                alarmed_pos += 1
            elif not pos_g[g] and any_alarm:
                alarmed_neg += 1
        det.append(alarmed_pos / max(npos, 1))
        far.append(alarmed_neg / max(nneg, 1))
        fp_hours = int(((p >= t) & (y == 0)).sum())
        burden.append(100.0 * fp_hours / max(neg_hours, 1))
    return {"thresholds": [float(t) for t in thresholds], "detection_rate": det,
            "false_alarm_rate": far, "false_alerts_per_100h": burden}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                 stat: str = "auroc", n_boot: int = 500, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Patient-level (cluster) bootstrap 95% CI. Resampling PATIENTS (not rows)
    respects the within-patient correlation, so the interval is honest."""
    fn = {"auroc": auroc, "auprc": auprc, "brier": brier}[stat]
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
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
