"""Training targets that mirror how the Sepsis-3 label is generated.

The CinC label is not a measurement; it is manufactured. For a patient who becomes
septic, clinicians first suspect infection (cultures + antibiotics — never recorded
in this dataset), organ-dysfunction scores rise by >= 2 points (partly recorded: the
labs that feed SOFA are here), onset ``T`` is stamped as the earlier of the two, and
the label is switched on for every hour ``t >= T - 6``. The record is then cut three
hours after ``T``.

A single classifier on that label has to learn all of it at once from 2% positive
hours. The targets below split the job along the label's own seams, so each learner
gets a dense, well-posed question:

``patient``   — WHO: every hour of a stay that ever turns septic. The supervision-
                density finding, used as one factor instead of the whole answer.
``timing``    — WHEN: the challenge label, scored only among septic stays. A balanced
                (1:5) contrast between the approach to onset and the same patients'
                earlier hours; no never-septic patient enters it.
``horizon``   — onset within ``h`` hours: the discrete hazard curve. ``h = 6`` is the
                challenge label itself; wider ``h`` interpolate toward ``patient``.
``future_*``  — label-genesis auxiliaries. Where the future of a stay is recorded,
                the next hours tell us whether clinicians escalated the work-up (a
                lactate or blood gas was drawn, new tests were ordered) and whether an
                organ score rose. Those are the observable halves of the two processes
                that generate the label, and every hour with a recorded future carries
                one — dense supervision about the mechanism rather than the outcome.

Every function returns ``(target, weight)`` aligned to all rows; weight 0 marks rows a
learner must not train on. Targets may look into the future — that is what a target
is — but the learners only ever see causal features, and ``verify_genesis.py`` asserts
that no target leaks into any feature column.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from backend.ml.genesis.harness import Cohort

Target = Tuple[np.ndarray, np.ndarray]

#: Hours between the labelled onset (where SepsisLabel flips) and clinical onset T.
LABEL_LEAD = 6


def challenge(c: Cohort) -> Target:
    return c.y.astype(np.float64), np.ones(len(c.y))


def patient(c: Cohort) -> Target:
    return c.plabel[c.seg].astype(np.float64), np.ones(len(c.y))


def timing(c: Cohort) -> Target:
    septic = c.plabel[c.seg] == 1
    return c.y.astype(np.float64), septic.astype(np.float64)


def horizon(c: Cohort, h: float) -> Target:
    """1 where clinical onset T is at most ``h`` hours ahead (or already passed)."""
    septic = c.onset_row >= 0
    t_onset = c.onset_row + LABEL_LEAD
    tgt = septic & ((t_onset - c.hour) <= h)
    return tgt.astype(np.float64), np.ones(len(c.y))


def _stay_end(c: Cohort) -> np.ndarray:
    """Last row index of each row's stay."""
    ends = c.raw.starts + c.raw.lens - 1
    return ends[c.seg]


def future_any(c: Cohort, event: np.ndarray, k: int) -> Target:
    """1 if ``event`` happens in rows (t, t+k] of the same stay.

    Rows whose next ``k`` hours are not fully recorded get weight 0 — an unrecorded
    future is not a negative. That exclusion is itself informative about the label (a
    septic record is cut 3 h after onset), which is exactly why it is a WEIGHT and not a
    feature: the learner trains on fewer rows, it never sees where a record ends."""
    ev = np.asarray(event, dtype=np.float64)
    C = np.r_[0.0, np.cumsum(ev)]
    idx = np.arange(len(ev))
    end = _stay_end(c)
    hi = np.minimum(idx + k, end)
    tgt = (C[hi + 1] - C[idx + 1]) > 0
    w = (idx + k <= end).astype(np.float64)
    return tgt.astype(np.float64), w


def future_rise(c: Cohort, x: np.ndarray, k: int, delta: float) -> Target:
    """1 if ``x`` rises by at least ``delta`` above its current value within (t, t+k]."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    idx = np.arange(n)
    end = _stay_end(c)
    best = np.full(n, -np.inf)
    for lag in range(1, k + 1):
        j = idx + lag
        ok = j <= end
        v = np.full(n, -np.inf)
        v[ok] = x[j[ok]]
        best = np.maximum(best, np.nan_to_num(v, nan=-np.inf))
    cur = np.nan_to_num(x, nan=np.inf)
    tgt = (best - cur) >= delta
    w = (idx + k <= end).astype(np.float64)
    return tgt.astype(np.float64), w
