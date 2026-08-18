"""The official CinC utility, vectorised — and turned into a training target.

Two jobs, deliberately in one place because they are the same arithmetic:

1. **Scoring, fast.** :func:`utility_terms` precomputes, for every row, the utility
   of raising the alarm at that hour (``u1``) and of staying silent (``u0``). Any
   alarm policy then scores in one vectorised pass instead of the reference
   implementation's per-patient Python loop, which makes threshold sweeps and
   bootstraps affordable.

2. **Training, aligned.** The same two numbers expose a mismatch that every model
   in this project has been trained into. The challenge label flips to 1 at
   ``t_sepsis - 6`` and then stays 1 for the rest of the stay, but the utility
   function does not agree with that label at either end:

   * hours in ``[t_sepsis-12, t_sepsis-6)`` are labelled **0**, yet alarming there
     is worth up to **+1** — the single most valuable prediction in the whole
     scoring scheme is one the label calls a false positive;
   * hours after ``t_sepsis+3`` are labelled **1**, yet alarming there is worth
     exactly **0** — and for a septic patient that is most of the stay, so the bulk
     of the positive class carries no score at all.

   :func:`utility_target` returns the target the scoring function actually implies:
   label ``z = 1`` where alarming beats silence, and weight ``w = |u1 - u0|``, the
   number of utility points riding on that row. Rows worth nothing get weight 0 and
   drop out. A weighted classifier's Bayes rule under ``(z, w)`` is
   ``E[u1 - u0 | x] > 0`` — exactly the utility-optimal alarm — so this is a
   re-derivation of the objective, not a heuristic re-weighting.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Official challenge constants (physionetchallenges/evaluation-2019).
DT_EARLY, DT_OPTIMAL, DT_LATE = -12.0, -6.0, 3.0
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0


def utility_terms(
    y: np.ndarray,
    groups: np.ndarray,
    hour_index: np.ndarray,
    onset_hour: np.ndarray,
    label_lead: int = 6,
) -> Dict[str, np.ndarray]:
    """Per-row utility of alarming (``u1``) vs staying silent (``u0``).

    ``onset_hour`` is the *labelled* onset (where the challenge label flips), so
    the clinical onset used by the scoring function is ``onset_hour + label_lead``.
    Rows of never-septic patients carry ``onset_hour = -1``.
    """
    t = np.asarray(hour_index, dtype=np.float64)
    onset = np.asarray(onset_hour, dtype=np.float64)
    septic = onset >= 0
    t_sepsis = np.where(septic, onset + label_lead, np.inf)
    dt = t - t_sepsis                      # hours relative to clinical onset

    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
    b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
    b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
    b3 = -m3 * DT_OPTIMAL

    u1 = np.zeros(len(t))
    u0 = np.zeros(len(t))

    # Never-septic patients: a constant small penalty for every alarm, forever.
    ns = ~septic
    u1[ns] = U_FP
    u0[ns] = U_TN

    # Septic patients, only while the prediction can still score (t <= onset+3).
    live = septic & (dt <= DT_LATE)
    early = live & (dt <= DT_OPTIMAL)      # the pre-onset credit ramp
    late = live & (dt > DT_OPTIMAL)        # the window where silence is punished
    u1[early] = np.maximum(m1 * dt[early] + b1, U_FP)
    u1[late] = m2 * dt[late] + b2
    u0[early] = 0.0
    u0[late] = m3 * dt[late] + b3
    # dt > DT_LATE stays (0, 0): after that point nothing a model says can score.

    return {"u1": u1, "u0": u0, "gain": u1 - u0, "septic": septic, "dt": dt}


def utility_target(
    y: np.ndarray,
    groups: np.ndarray,
    hour_index: np.ndarray,
    onset_hour: np.ndarray,
    label_lead: int = 6,
    min_weight: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(z, w)``: the label and sample weight the utility function implies.

    ``z`` is 1 where alarming is worth more than silence and ``w`` is how much is
    at stake. Rows where nothing is at stake get weight 0 rather than being
    dropped, so the arrays stay row-aligned with the feature matrix.
    """
    g = utility_terms(y, groups, hour_index, onset_hour, label_lead)["gain"]
    z = (g > 0).astype(np.int8)
    w = np.abs(g)
    w[w < min_weight] = 0.0
    return z, w


def normalized_utility_fast(
    y: np.ndarray,
    pred_binary: np.ndarray,
    terms: Dict[str, np.ndarray],
) -> float:
    """``(observed - inaction) / (best - inaction)`` in three vectorised sums.

    Matches :func:`backend.ml.eval.metrics.normalized_utility` exactly; ``best`` is
    the utility of predicting the true labels (the official normaliser), not the
    utility of the oracle alarm policy.
    """
    u1, u0 = terms["u1"], terms["u0"]
    pred = np.asarray(pred_binary).astype(bool)
    lab = np.asarray(y).astype(bool)
    observed = float(np.where(pred, u1, u0).sum())
    inaction = float(u0.sum())
    best = float(np.where(lab, u1, u0).sum())
    denom = best - inaction
    return float((observed - inaction) / denom) if denom != 0 else float("nan")


def best_threshold_utility(
    y: np.ndarray,
    p: np.ndarray,
    terms: Dict[str, np.ndarray],
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """``(best_utility, threshold)`` over a threshold sweep.

    This is an **oracle** number — the threshold is chosen on the same rows it is
    scored on — so it is only ever comparable between candidates evaluated the same
    way, never quotable as a deployed result. It exists because the fixed
    ``spec>=0.90`` alarm policy suits some score distributions better than others,
    and a model must not look worse merely because a shared policy fits it badly.
    """
    if grid is None:
        qs = np.linspace(0.80, 0.999, 60)
        grid = np.unique(np.quantile(p, qs))
    best_u, best_t = -np.inf, float("nan")
    for t in grid:
        u = normalized_utility_fast(y, p >= t, terms)
        if u > best_u:
            best_u, best_t = u, float(t)
    return float(best_u), best_t
