"""Critical Slowing Down (CSD) early-warning engine — the resilience radar.

Dynamical-systems theory: as a system approaches a critical transition (a tipping
point), it loses RESILIENCE — its recovery from small perturbations slows, so the
signal's lag-1 AUTOCORRELATION rises and its VARIANCE widens. Tracking those two
generic early-warning signals across several vitals gives a continuous "stability
score" that can move before any single vital crosses a clinical threshold.

This module computes those indicators CAUSALLY (past-only windows, no leakage),
per channel, and fuses them into one resilience-loss score in [0,1]. It is used
two ways: incrementally per-tick (`ResilienceMonitor`) for the live radar, and in
batch (`ews_series`) for the honest validation in verify_resilience.py.

HONESTY: the validation MEASURES whether this score actually precedes sepsis on
CinC-2019 before any clinical claim is made. ICU data is hourly + sparsely
sampled, which blunts CSD; the engine detrends with first differences to focus on
real fluctuation dynamics, and we report the lead time / AUROC it truly earns.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# Continuously-monitored vitals carry fluctuation structure; the hourly labs
# (WBC/lactate) are too sparse/slow for autocorrelation, so CSD uses the vitals.
CSD_CHANNELS = ["hr", "resp_rate", "spo2", "map", "temp_c"]
WIN = 12          # rolling window (hours) for the indicators
BASE_W = 10       # early window used to freeze each patient's own baseline
MIN_N = 6         # min samples in a window before an indicator is defined


def _ar1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a (detrended) series."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    den = float(np.dot(x, x))
    if len(x) < 4 or den < 1e-12:
        return 0.0
    return float(np.dot(x[1:], x[:-1]) / den)


def _indicators(window: np.ndarray):
    """(ar1, variance) of the detrended window, or None.

    Canonical CSD: subtract a smooth (moving-average) trend, then take the
    residual's lag-1 autocorrelation + variance — rising AR1 = slowing recovery,
    rising variance = wider fluctuations. (AR1 must be on the residual, NOT on
    first differences, which would measure a different quantity.)"""
    w = np.asarray(window, dtype=float)
    if len(w) < MIN_N:
        return None
    r = np.diff(w)                       # first-difference detrend → fluctuations
    if r.size < 3 or np.std(r) < 1e-9:
        return 0.0, 0.0
    return _ar1(r), float(np.var(r))


def ews_series(chans: Dict[str, List[float]], n: int) -> Dict[str, np.ndarray]:
    """Causal early-warning series for one patient.

    chans: {channel -> forward-filled hourly values (len n)}.
    Returns per-hour arrays: composite EWS in [0,1] plus the two raw indicator
    sub-scores (variance ratio, autocorrelation rise) for honest per-indicator
    evaluation. Everything at hour t uses only data ≤ t.
    """
    ar1_z = np.zeros(n)
    var_z = np.zeros(n)
    comp = np.zeros(n)
    # per-channel frozen baselines (median indicator over the early stable window)
    base: Dict[str, Dict[str, float]] = {}
    # rolling per-channel indicator history to set the baseline
    early: Dict[str, Dict[str, List[float]]] = {ch: {"ar1": [], "var": []} for ch in CSD_CHANNELS}
    for t in range(n):
        ch_ar1, ch_var, used = 0.0, 0.0, 0
        for ch in CSD_CHANNELS:
            s = chans.get(ch)
            if s is None:
                continue
            ind = _indicators(np.asarray(s[max(0, t - WIN + 1):t + 1], dtype=float))
            if ind is None:
                continue
            a, v = ind
            e = early[ch]
            if t <= WIN + BASE_W:                       # still collecting baseline
                e["ar1"].append(a); e["var"].append(v)
            if ch not in base and t >= WIN + BASE_W:    # freeze baseline once
                base[ch] = {"ar1": float(np.median(e["ar1"])) if e["ar1"] else 0.0,
                            "var": float(np.median(e["var"])) if e["var"] else 1e-6}
            b = base.get(ch)
            if not b:
                continue
            var_ratio = v / (b["var"] + 1e-9)
            ar1_rise = a - b["ar1"]
            ch_ar1 += max(0.0, ar1_rise)
            ch_var += max(0.0, var_ratio - 1.0)
            used += 1
        if used:
            ar1_z[t] = ch_ar1 / used
            var_z[t] = ch_var / used
            raw = 0.35 * ar1_z[t] + 0.95 * var_z[t]
            comp[t] = 1.0 - np.exp(-max(0.0, raw) / 1.2)   # squash to [0,1)
    return {"ews": comp, "ar1": ar1_z, "var": var_z}


class ResilienceMonitor:
    """Live, incremental CSD for one bed — feed hourly vitals, get the score."""

    def __init__(self) -> None:
        self.hist: Dict[str, List[float]] = {ch: [] for ch in CSD_CHANNELS}
        self.t = 0
        self.trace: List[float] = []
        self._base: Dict[str, Dict[str, float]] = {}
        self._early: Dict[str, Dict[str, List[float]]] = {ch: {"ar1": [], "var": []} for ch in CSD_CHANNELS}
        self.ews = 0.0
        self.ar1 = 0.0
        self.var = 0.0
        self.slope = 0.0

    def observe(self, vit: Dict[str, float]) -> float:
        self.t += 1
        for ch in CSD_CHANNELS:
            v = vit.get(ch)
            if v is not None and v == v:
                self.hist[ch].append(float(v))
        ch_ar1, ch_var, used = 0.0, 0.0, 0
        for ch in CSD_CHANNELS:
            s = self.hist[ch]
            ind = _indicators(np.asarray(s[-WIN:], dtype=float))
            if ind is None:
                continue
            a, v = ind
            e = self._early[ch]
            if len(s) <= WIN + BASE_W:
                e["ar1"].append(a); e["var"].append(v)
            if ch not in self._base and len(s) >= WIN + BASE_W:
                self._base[ch] = {"ar1": float(np.median(e["ar1"])) if e["ar1"] else 0.0,
                                  "var": float(np.median(e["var"])) if e["var"] else 1e-6}
            b = self._base.get(ch)
            if not b:
                continue
            ch_ar1 += max(0.0, a - b["ar1"])
            ch_var += max(0.0, v / (b["var"] + 1e-9) - 1.0)
            used += 1
        if used:
            self.ar1 = ch_ar1 / used
            self.var = ch_var / used
            raw = 0.35 * self.ar1 + 0.95 * self.var
            self.ews = float(1.0 - np.exp(-max(0.0, raw) / 1.2))
        self.trace.append(self.ews)
        if len(self.trace) >= 4:
            self.slope = float(np.polyfit(range(4), self.trace[-4:], 1)[0])
        return self.ews

    def snapshot(self) -> Dict[str, object]:
        warming = self.t < WIN + BASE_W
        return {
            "ews": round(self.ews, 3),
            "ar1": round(self.ar1, 3),
            "var": round(self.var, 3),
            "slope": round(self.slope, 4),
            "trace": [round(x, 3) for x in self.trace[-48:]],
            "warming": warming,
            "rising": self.slope > 0.01,
        }
