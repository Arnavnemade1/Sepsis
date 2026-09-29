"""Rows grouped by patient in one sort — the replacement for ``groups == g`` per patient.

Scanning every row once per patient is O(patients x rows). It was harmless on the
8,000-patient development set and is not on the full cohort: one early-warning call
cost 4.5 s on an 8,062-patient holdout, and the operating-point derivation makes
~150 of them. One sort is O(rows log rows) and the rest is slicing.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def patient_runs(groups, hour_index: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(order, starts, ends)``: ``order[starts[i]:ends[i]]`` are patient ``i``'s rows.

    Patients come in sorted id order, the order ``sorted(set(groups))`` gives. Within
    a patient, rows keep their stored order, or run in hour order when ``hour_index``
    is given; both sorts are stable, so tied hours keep their stored order too."""
    groups = np.asarray(groups)
    order = (np.argsort(groups, kind="stable") if hour_index is None
             else np.lexsort((np.asarray(hour_index), groups)))
    if len(order) == 0:
        empty = np.zeros(0, dtype=np.intp)
        return order, empty, empty
    g = groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return order, starts, np.r_[starts[1:], len(g)]


def first_rows(order: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Each patient's first row as stored — what ``x[groups == g][0]`` read."""
    return np.minimum.reduceat(order, starts) if len(starts) else starts
