"""Derive the alarm operating point from data instead of guessing it.

WHY THIS EXISTS
---------------
Everything else in this subsystem is measured. The alarm thresholds were not: they were
literal constants in :class:`~backend.aegis.core.AegisCore` —

    high_threshold: float = 0.12,     # calibrated-probability operating points
    medium_threshold: float = 0.04,   # (derive from validation; see ml/README)

— carrying a comment instructing someone to derive them, which never happened. That is
not a cosmetic gap. A detector's *ranking* and its *operating point* are separate
things, and this project has spent a lot of effort proving the ranking is at its data
ceiling (docs/PHASE7_REPORT.md, backend/ml/selfimprove.py) while leaving the point at
which that ranking is read as an unmeasured guess.

WHAT THE GUESS COST (real CinC, 4000 patients, threshold derived on train patients and
scored on held-out patients — measurements in backend/ml/README.md):

    shipped 0.12          utility +0.2946   detection 35.2%   FAR  9.3%
    OOF utility-optimal   utility +0.3670   detection 52.7%   FAR 34.8%
    OOF with FAR <= 15%   utility +0.3357   detection 45.1%   FAR 19.2%

Same model. Same predictions. Identical AUROC (0.7334). The entire difference is where
the line was drawn.

THE CLIFF (the part that matters for safety)
--------------------------------------------
Sepsis is ~2% of hours, and the detector is well calibrated (ECE ~0.005), so it
*correctly* almost never emits a high probability. Its out-of-fold distribution:

    p95 = 0.086    p99 = 0.140    only 1.7% of hours exceed 0.12

The shipped 0.12 therefore sat ABOVE THE 98TH PERCENTILE of the model's own output —
not conservative, nearly out of range. Just above it the system falls off a cliff:
at 0.16 detection collapses from 35% to 3.3% and utility goes to zero. A small
calibration drift would silently turn the alarm off. Deriving the point from the
model's actual output distribution removes that failure mode by construction.

METHOD (the discipline is the point)
------------------------------------
* Thresholds are derived from **patient-grouped out-of-fold** predictions, never from
  in-sample ones. A model's training predictions are far sharper than anything it emits
  in deployment (train utility 0.918 vs held-out 0.347 on the same fit), so a threshold
  tuned on them is tuned to a distribution that does not exist at serve time.
* The objective is explicit. ``utility`` maximises the official CinC 2019 score, which
  weights a missed case at -2 and a false alarm at only -0.05 and therefore pushes the
  alarm low and loud. ``constrained`` maximises the same score subject to a cap on the
  fraction of non-septic patients ever alarmed. Which of those a hospital wants is a
  clinical values judgement about alarm fatigue, NOT a modelling question — so it is a
  parameter, not a constant buried in a class signature.
* MEDIUM is derived as a genuine pre-alarm below HIGH, not an independent knob.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np

from backend.ml.eval import metrics


@dataclass
class OperatingPoint:
    """A derived alarm configuration plus the evidence that produced it."""
    high: float
    medium: float
    objective: str
    max_false_alarm_rate: Optional[float]
    # evidence, recorded so the number is never again "just a constant"
    oof_utility: float
    oof_detection_rate: float
    oof_false_alarm_rate: float
    oof_median_lead_hours: float
    prob_p95: float
    prob_p99: float
    prob_max: float
    frac_above_high: float
    n_patients: int
    #: The detector configuration this evidence was measured on, as
    #: ``"objective:member+member+..."``. Defaulted rather than required so an
    #: artifact written before this field existed still loads — but an artifact
    #: that says ``"unknown"`` is exactly the failure this field exists to expose:
    #: a threshold whose recorded p95, utility and false-alarm rate belong to some
    #: other model than the one it will govern.
    detector: str = "unknown"
    #: The bound the false-alarm cap was actually enforced against. When the cap is
    #: enforced with margin this is the upper end of the patient-bootstrap CI, which
    #: is >= ``oof_false_alarm_rate``; the gap between the two is the margin.
    oof_false_alarm_upper: float = float("nan")

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


def _utility(y, p, groups, hour_index, thr) -> float:
    return metrics.normalized_utility(y, (p >= thr).astype(int), groups, hour_index)


def _alarm_stats(y, p, groups, hour_index, onset_hour, thr) -> Dict[str, float]:
    return metrics.early_warning(y, p, groups, onset_hour, hour_index, thr)


def _nonseptic_peaks(p: np.ndarray, groups: np.ndarray, onset_hour: np.ndarray) -> np.ndarray:
    """Each never-septic patient's highest score anywhere in the stay.

    The false-alarm rate is exactly ``mean(peak >= thr)`` over these patients — one
    number per patient rather than one per hour — which turns a bootstrap of the cap
    from a re-scan of 300k rows into a mean over a few thousand floats.
    """
    order = np.lexsort((np.arange(len(groups)), groups))
    g, ps, on = groups[order], np.asarray(p)[order], np.asarray(onset_hour)[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    bounds = np.r_[starts, len(g)]
    return np.array([ps[a:b].max() for a, b in zip(bounds[:-1], bounds[1:])
                     if on[a] < 0], dtype=float)


def far_upper_bound(peaks: np.ndarray, grid: np.ndarray, n_boot: int = 400,
                    seed: int = 0, alpha: float = 0.05) -> np.ndarray:
    """Upper end of the patient-bootstrap CI of the false-alarm rate, per threshold.

    A cap enforced on a point estimate is a cap that holds on the patients used to
    choose it and nowhere else. That is not a hypothetical failure mode: the point
    estimate said 0.077 at the chosen threshold while the same threshold delivered
    0.170 on held-out patients from the same cohort. Enforcing the cap on the upper
    confidence bound instead asks the question a safety limit actually poses — is the
    TRUE rate below the cap? — and buys the margin that makes the answer hold on
    patients nobody has seen.
    """
    if peaks.size == 0:
        return np.zeros(len(grid))
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        s = peaks[rng.integers(0, peaks.size, size=peaks.size)]
        draws[b] = (s[:, None] >= grid[None, :]).mean(axis=0)
    return np.percentile(draws, 100 * (1 - alpha / 2), axis=0)


def derive(
    y: np.ndarray,
    p_oof: np.ndarray,
    groups: np.ndarray,
    hour_index: np.ndarray,
    onset_hour: np.ndarray,
    *,
    objective: str = "constrained",
    max_false_alarm_rate: Optional[float] = 0.15,
    grid: Optional[Sequence[float]] = None,
    medium_fraction: float = 0.5,
    require_in_range: bool = True,
    far_margin: bool = True,
    far_n_boot: int = 400,
    seed: int = 0,
    detector: str = "unknown",
) -> OperatingPoint:
    """Derive HIGH/MEDIUM thresholds from out-of-fold predictions.

    ``p_oof`` **must** be out-of-fold — predictions from models that never saw the
    patient being scored. Passing in-sample predictions here silently produces a
    threshold tuned to a distribution the model will never emit again; that is the
    single easiest way to make this whole exercise worse than the guess it replaces.

    ``objective``:
      * ``"utility"``     — maximise the official CinC 2019 normalized utility.
      * ``"constrained"`` — maximise it subject to ``max_false_alarm_rate`` (default),
        because the utility function prices a false alarm at -0.05 and a missed case at
        -2, so left alone it will happily alarm on a third of healthy patients. Alarm
        fatigue is a real harm that the CinC score does not model.

    ``require_in_range`` keeps the chosen threshold at or below the p95 of the
    detector's own output. This module exists because a hand-set 0.12 had wandered
    out into the sparse tail where a calibrated 2%-prevalence model almost never
    emits, and the guard against that lived only in the integrity audit — i.e. it
    fired on the artifact *after* it was written and shipped. An invariant checked
    downstream of the thing that violates it is not an invariant, so it is enforced
    here, at the point of choosing.

    ``detector`` names the configuration whose out-of-fold scores are being read, and
    is stored on the result. A threshold is only meaningful against the score
    distribution it was derived from; recording which one that was is what makes a
    later mismatch detectable instead of invisible.
    """
    y = np.asarray(y).astype(int)
    p_oof = np.asarray(p_oof, dtype=float)
    if objective not in ("utility", "constrained"):
        raise ValueError(f"unknown objective {objective!r}")
    if grid is None:
        # Search where a calibrated 2%-prevalence model actually lives. A linear sweep
        # to 0.5 wastes almost every point above the p99 (0.14) — and it is precisely
        # that dead zone the shipped 0.12 had wandered into.
        grid = np.unique(np.round(np.concatenate([
            np.linspace(0.005, 0.15, 59), np.linspace(0.16, 0.40, 13)]), 4))
    grid = np.asarray(grid, dtype=float)

    p95 = float(np.quantile(p_oof, 0.95))
    util = np.array([_utility(y, p_oof, groups, hour_index, t) for t in grid])
    far = np.array([_alarm_stats(y, p_oof, groups, hour_index, onset_hour, t)["false_alarm_rate"]
                    for t in grid])

    # Every constraint is assembled before the pick, so an infeasible combination is
    # reported as such instead of quietly resolving to whichever bound happened to be
    # checked last.
    feasible = np.ones(len(grid), dtype=bool)
    unmet = []
    if require_in_range:
        in_range = grid <= p95
        if not in_range.any():
            raise ValueError(
                f"no threshold on the grid sits at or below the detector's p95 "
                f"({p95:.4f}) — the grid does not reach the range where this model's "
                f"output actually lives")
        feasible &= in_range
        unmet.append(f"<= p95 {p95:.4f}")
    far_hi = np.full(len(grid), np.nan)
    if objective == "constrained":
        cap = max_false_alarm_rate if max_false_alarm_rate is not None else 1.0
        if far_margin:
            far_hi = far_upper_bound(_nonseptic_peaks(p_oof, groups, onset_hour),
                                     grid, n_boot=far_n_boot, seed=seed)
            under_cap = far_hi <= cap
        else:
            under_cap = far <= cap
        if not under_cap.any():
            # Never silently fall back to an infeasible point: a caller who asked for a
            # cap and got a threshold that violates it would be actively misled.
            raise ValueError(
                f"no threshold on the grid meets false_alarm_rate <= {cap}"
                + (" (upper bound)" if far_margin else "")
                + f"; the lowest achievable is "
                + f"{np.nanmin(far_hi if far_margin else far):.3f}")
        feasible &= under_cap
        unmet.append(f"FAR <= {cap}")
    if not feasible.any():
        raise Infeasible(
            "no threshold satisfies every constraint at once (" + " and ".join(unmet) +
            f"); in-range thresholds reach FAR "
            f"{np.nanmin((far_hi if far_margin else far)[grid <= p95]):.3f} at best",
            grid, far, far_hi if far_margin else far, util, grid <= p95, p95)
    pick = int(np.nanargmax(np.where(feasible, util, -np.inf)))

    high = float(grid[pick])
    stats = _alarm_stats(y, p_oof, groups, hour_index, onset_hour, high)
    # MEDIUM is a pre-alarm: a fraction of HIGH, so the two can never cross or invert
    # as HIGH moves. It was previously an independent constant (0.04 vs 0.12), which
    # meant re-tuning one silently changed the relationship between them.
    medium = round(high * medium_fraction, 4)

    return OperatingPoint(
        high=high, medium=medium, objective=objective,
        max_false_alarm_rate=max_false_alarm_rate if objective == "constrained" else None,
        oof_utility=float(util[pick]),
        oof_detection_rate=float(stats["detection_rate"]),
        oof_false_alarm_rate=float(stats["false_alarm_rate"]),
        oof_median_lead_hours=float(stats["median_lead_time_hours"]),
        prob_p95=p95,
        prob_p99=float(np.quantile(p_oof, 0.99)),
        prob_max=float(p_oof.max()),
        frac_above_high=float(np.mean(p_oof >= high)),
        oof_false_alarm_upper=float(far_hi[pick]),
        n_patients=int(len(set(groups.tolist()))),
        detector=detector,
    )


class Infeasible(ValueError):
    """No threshold satisfies every constraint at once.

    Carries the frontier rather than just a message, because the caller's next
    question is always the same one: what WOULD it take? The alarm budget is a
    clinical values judgement, so the answer to an infeasible request is a table the
    operator can decide from, not a stack trace.
    """

    def __init__(self, message: str, grid: np.ndarray, far: np.ndarray,
                 far_hi: np.ndarray, util: np.ndarray, in_range: np.ndarray, p95: float):
        super().__init__(message)
        self.grid, self.far, self.far_hi = grid, far, far_hi
        self.util, self.in_range, self.p95 = util, in_range, p95

    def frontier(self, caps: Sequence[float] = (0.15, 0.175, 0.20, 0.25)) -> str:
        """What each alarm budget would buy, and whether it needs an out-of-range point."""
        lines = [f"{'cap':>7}  {'thr':>7}  {'FAR':>6}  {'FAR hi':>7}  {'utility':>8}  in-range?"]
        for cap in caps:
            ok = self.far_hi <= cap
            for label, mask in (("in-range", ok & self.in_range), ("any", ok)):
                if not mask.any():
                    continue
                i = int(np.nanargmax(np.where(mask, self.util, -np.inf)))
                lines.append(f"{cap:7.3f}  {self.grid[i]:7.4f}  {self.far[i]:6.3f}  "
                             f"{self.far_hi[i]:7.3f}  {self.util[i]:+8.4f}  "
                             f"{'yes' if self.in_range[i] else 'NO (above p95 ' + format(self.p95, '.4f') + ')'}")
                break
        return "\n".join(lines)


def sweep_table(y, p, groups, hour_index, onset_hour,
                thresholds: Sequence[float]) -> str:
    """Human-readable threshold sweep — the evidence behind a chosen point."""
    lines = [f"{'thr':>7}  {'utility':>8}  {'detect':>7}  {'FAR':>6}  {'lead(h)':>7}"]
    for t in thresholds:
        s = _alarm_stats(y, p, groups, hour_index, onset_hour, t)
        lines.append(f"{t:7.4f}  {_utility(y,p,groups,hour_index,t):+8.4f}  "
                     f"{s['detection_rate']:7.3f}  {s['false_alarm_rate']:6.3f}  "
                     f"{s['median_lead_time_hours']:7.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "artifacts", "operating_point.json")


def save(op: OperatingPoint, path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(op.to_json(), fh, indent=2)


def load(path: str = DEFAULT_PATH) -> Optional[OperatingPoint]:
    """Load the derived operating point, or None if it has not been derived yet.

    Returning None (rather than raising or inventing a default) lets the serving path
    fall back to its documented constants and say so, instead of failing closed on a
    machine where the artifact was never generated.
    """
    try:
        with open(path) as fh:
            d = json.load(fh)
        return OperatingPoint(**d)
    except (OSError, TypeError, ValueError):
        return None
