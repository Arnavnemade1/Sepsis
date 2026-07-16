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

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


def _utility(y, p, groups, hour_index, thr) -> float:
    return metrics.normalized_utility(y, (p >= thr).astype(int), groups, hour_index)


def _alarm_stats(y, p, groups, hour_index, onset_hour, thr) -> Dict[str, float]:
    return metrics.early_warning(y, p, groups, onset_hour, hour_index, thr)


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

    util = np.array([_utility(y, p_oof, groups, hour_index, t) for t in grid])
    far = np.array([_alarm_stats(y, p_oof, groups, hour_index, onset_hour, t)["false_alarm_rate"]
                    for t in grid])

    if objective == "utility":
        pick = int(np.nanargmax(util))
    else:
        cap = max_false_alarm_rate if max_false_alarm_rate is not None else 1.0
        feasible = far <= cap
        if not feasible.any():
            # Never silently fall back to an infeasible point: a caller who asked for a
            # cap and got a threshold that violates it would be actively misled.
            raise ValueError(
                f"no threshold on the grid meets false_alarm_rate <= {cap}; "
                f"the lowest achievable is {np.nanmin(far):.3f}")
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
        prob_p95=float(np.quantile(p_oof, 0.95)),
        prob_p99=float(np.quantile(p_oof, 0.99)),
        prob_max=float(p_oof.max()),
        frac_above_high=float(np.mean(p_oof >= high)),
        n_patients=int(len(set(groups.tolist()))),
    )


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
