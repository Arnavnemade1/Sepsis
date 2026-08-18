"""Causal temporal filters over a row-wise detector's own score.

The sequence models in :mod:`backend.ml.models.sequence` buy their biggest practical
win not in AUROC but in **alarm burden**: on the same cohort they raise roughly a
third fewer false alarms than the boosters and warn ~13 hours earlier. A plausible
explanation is cheap — a model that reads the whole prefix produces a score that
moves smoothly hour to hour, so a patient's trajectory crosses the alarm threshold
once instead of flickering across it — and if that is the whole story, a row-wise
booster plus a causal filter on its output should recover most of the benefit with
no new runtime dependency at all.

This module exists to test that claim rather than assume it. Each filter is strictly
causal (hour ``t`` sees only hours ``≤ t`` of the same patient) and stateless across
patients, so it is exactly implementable in the live streaming path:

``ema``
    Exponential moving average — the smoothing hypothesis in its simplest form.
``cummax``
    Running maximum: once a patient has looked bad, they stay flagged. The opposite
    prior to smoothing — no forgetting at all.
``hazard``
    Accumulated log-odds with decay: evidence adds up over time and fades if the
    patient recovers, which is the closest row-wise analogue to what the selective
    state-space layer learns.
``blend``
    A fixed 50/50 mix of the raw score and its EMA, so a sudden deterioration is not
    smoothed away while sustained risk still accumulates.

The filters have no fitted parameters beyond their declared constants, so wrapping a
detector cannot leak anything from the evaluation fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _per_patient_order(groups: np.ndarray, hour_index: np.ndarray):
    """Row order that walks each patient forward in time, and the inverse map."""
    order = np.lexsort((hour_index, groups))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    g = groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return order, inv, starts


def causal_filter(p: np.ndarray, groups: np.ndarray, hour_index: np.ndarray,
                  mode: str = "ema", halflife: float = 6.0,
                  decay: float = 0.9, weight: float = 0.5) -> np.ndarray:
    """Apply a causal filter to a per-hour score, patient by patient."""
    p = np.asarray(p, dtype=float)
    order, inv, starts = _per_patient_order(groups, hour_index)
    ps = p[order]
    out = np.empty_like(ps)
    alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1e-6))
    bounds = np.r_[starts, len(ps)]

    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = ps[a:b]
        if mode == "cummax":
            out[a:b] = np.maximum.accumulate(seg)
            continue
        if mode == "hazard":
            # log-odds accumulate and decay: s_t = decay*s_{t-1} + logit(p_t)
            lo = np.log(np.clip(seg, 1e-6, 1 - 1e-6) / (1 - np.clip(seg, 1e-6, 1 - 1e-6)))
            s = np.empty_like(lo)
            acc = 0.0
            for i, v in enumerate(lo):
                acc = decay * acc + v
                s[i] = acc
            out[a:b] = 1.0 / (1.0 + np.exp(-s * (1.0 - decay)))
            continue
        # ema (and the raw half of blend)
        s = np.empty_like(seg)
        acc = seg[0]
        for i, v in enumerate(seg):
            acc = alpha * v + (1 - alpha) * acc
            s[i] = acc
        out[a:b] = s if mode == "ema" else weight * seg + (1 - weight) * s
    return out[inv]


@dataclass
class TemporallyFilteredDetector:
    """Wraps a fitted row-wise detector with a causal filter on its output.

    The wrapped detector is untouched; only the score it emits is filtered, so the
    calibration it carries still applies up to the filter's monotone-ish transform.
    ``predict_proba`` therefore needs the patient grouping and hour index that the
    underlying detector does not.
    """
    inner: object
    mode: str = "ema"
    halflife: float = 6.0
    decay: float = 0.9
    weight: float = 0.5

    def predict_proba(self, X, groups: np.ndarray, hour_index: np.ndarray) -> np.ndarray:
        raw = self.inner.predict_proba(X)
        return causal_filter(raw, groups, hour_index, self.mode,
                             self.halflife, self.decay, self.weight)
