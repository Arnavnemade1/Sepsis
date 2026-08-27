"""Causal MiniRocket — random convolutional kernels over the patient's own history.

WHAT THIS IS
------------
MiniRocket (Dempster et al., 2021) transforms a time series with a fixed, deterministic
bank of length-9 dilated kernels whose weights are drawn from {-1, +2} with exactly
three +2s — all C(9,3) = 84 of them — and summarises each convolution by PPV, the
proportion of positive values. No weights are learned, so it is a transform rather
than a model: fast, deterministic, CPU-only.

Phase 7 found that *hand-authored* temporal aggregates (rolling means, slopes,
multi-scale windows, 94 -> 185 features) bought nothing. This is a different bet: not
more summaries chosen by a human, but a fixed random basis over trajectory SHAPE.
Whether that bet pays is an empirical question this module exists to let the bench
answer; nothing here asserts that it does.

TWO THINGS THE TEXTBOOK VERSION GETS WRONG FOR THIS PROBLEM
-----------------------------------------------------------
**1. It is not causal, and that is not a detail.** Standard MiniRocket convolves the
whole series and takes PPV over all of it, producing ONE feature vector per series.
Both halves leak: the convolution is centred (it reads hours after t), and the PPV
pools the entire stay. Used as-is for hour-by-hour prediction it would let hour 20 see
hour 60, and the model would post a magnificent lead time that is pure hindsight. This
implementation is strictly causal by construction — the kernel's LAST tap sits on hour
t, the left edge is padded with the series' own first value, and PPV is a trailing
window ending at t. ``verify_minirocket.py`` plants a future spike and asserts no
feature at any earlier hour moves.

**2. It does not fit in memory here.** The standard bank emits 9,996 features. On this
cohort that is 310,061 x 9,996 float32 = **11.6 GB** for the transform alone, before any
model sees it. So the bank is subsampled deterministically and the output is float32
and preallocated: the defaults below emit 336 features (~0.39 GB), and
``estimate_bytes`` will tell you the cost of any other configuration BEFORE you pay it.

WHAT THE FEATURES MEAN
----------------------
For channel ``c``, kernel ``k``, dilation ``d``, bias ``b``, window ``W``:

    y[t]      = sum_j w_k[j] * x_c[t - (8-j)*d]        (causal dilated convolution)
    feature   = mean over the trailing W hours of  1[ y > b ]

Dilation sets the timescale the kernel looks at (d=1 is hour-to-hour, d=12 is
half-a-day), the bias sets which part of the response distribution counts as an event,
and the window sets how long an event is remembered. Biases are quantiles of the
TRAINING fold's own convolution outputs, so they are fitted parameters and must never
be computed on the rows being scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KERNEL_LEN = 9
N_TWOS = 3


def kernel_bank(n_kernels: int = 8, seed: int = 42) -> np.ndarray:
    """A deterministic subset of MiniRocket's 84 length-9 {-1,+2} kernels.

    The full bank is C(9,3) = 84 sign patterns. Taking an evenly-spaced subset rather
    than a random sample keeps the selection reproducible without a seed mattering,
    and keeps the chosen patterns spread across the enumeration instead of clustered
    at one end of it.
    """
    all_pos = list(combinations(range(KERNEL_LEN), N_TWOS))
    if n_kernels >= len(all_pos):
        picks = all_pos
    else:
        idx = np.linspace(0, len(all_pos) - 1, n_kernels).round().astype(int)
        picks = [all_pos[i] for i in idx]
    W = np.full((len(picks), KERNEL_LEN), -1.0, dtype=np.float32)
    for r, pos in enumerate(picks):
        W[r, list(pos)] = 2.0
    return W


def causal_conv(x: np.ndarray, w: np.ndarray, d: int) -> np.ndarray:
    """Dilated convolution whose LAST tap lands on t — never reads past t.

    ``y[t] = sum_j w[j] * x[t - (K-1-j)*d]``. Positions before the series start are
    padded with ``x[0]``, which is the only padding available that does not invent a
    value and does not reach forward. (Zero-padding would inject a fake 0 mmHg into
    the first hours of every stay; edge-padding repeats a real observation.)
    """
    T = x.shape[0]
    y = np.zeros(T, dtype=np.float32)
    for j in range(KERNEL_LEN):
        lag = (KERNEL_LEN - 1 - j) * d
        if lag == 0:
            y += w[j] * x
        elif lag < T:
            y[:lag] += w[j] * x[0]
            y[lag:] += w[j] * x[:T - lag]
        else:
            y += w[j] * x[0]
    return y


def trailing_ppv(ind: np.ndarray, window: int) -> np.ndarray:
    """Causal proportion-of-positive-values over the trailing ``window`` hours.

    Expanding until the window is full, so hour 3 of a stay averages 4 values rather
    than borrowing 20 that do not exist. Computed by cumulative sum: the alternative,
    a rolling apply, is ~100x slower on 310k rows and this runs inside every fold.
    """
    T = ind.shape[0]
    c = np.cumsum(ind, dtype=np.float32)
    idx = np.arange(T)
    lo = np.maximum(0, idx - window + 1)
    prior = np.where(lo > 0, c[np.maximum(lo - 1, 0)], 0.0)
    return (c - prior) / (idx - lo + 1).astype(np.float32)


@dataclass
class CausalMiniRocket:
    """Fit biases on training rows; transform any rows into causal PPV features."""
    channels: List[str] = field(default_factory=list)
    n_kernels: int = 8
    dilations: Tuple[int, ...] = (1, 4, 12)
    quantiles: Tuple[float, ...] = (0.25, 0.75)
    window: int = 24
    seed: int = 42
    kernels_: Optional[np.ndarray] = None
    biases_: Optional[np.ndarray] = None          # (n_channels, n_kernels, n_dil, n_q)
    feature_names_: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def n_features(self) -> int:
        return len(self.channels) * self.n_kernels * len(self.dilations) * len(self.quantiles)

    def estimate_bytes(self, n_rows: int) -> int:
        """Cost of the output matrix, so a configuration is priced before it is run."""
        return n_rows * self.n_features() * 4

    def _names(self) -> List[str]:
        return [f"mr__{c}__k{k}_d{d}_q{int(q * 100)}"
                for c in self.channels for k in range(self.n_kernels)
                for d in self.dilations for q in self.quantiles]

    # ------------------------------------------------------------------ #
    def _series(self, X: np.ndarray, col: int, groups: np.ndarray,
                order: np.ndarray, bounds: np.ndarray):
        """Yield each patient's contiguous, hour-ordered channel series."""
        v = X[:, col].astype(np.float32)
        for a, b in zip(bounds[:-1], bounds[1:]):
            sl = order[a:b]
            yield sl, v[sl]

    @staticmethod
    def _patient_bounds(groups: np.ndarray, hour_index: np.ndarray):
        """Row order grouped by patient and sorted by hour, plus group boundaries.

        Sorting by hour inside the patient is load-bearing: a causal convolution over
        rows that are not in time order is not causal, it is arbitrary.
        """
        order = np.lexsort((hour_index, groups))
        g = groups[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        return order, np.r_[starts, len(g)]

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, feature_names: Sequence[str], groups: np.ndarray,
            hour_index: np.ndarray, max_patients: int = 400) -> "CausalMiniRocket":
        """Learn per-(channel,kernel,dilation) bias quantiles from these rows only.

        ``max_patients`` subsamples the stays used to estimate the quantiles. A
        quantile is a distributional summary; it converges long before 8,000 patients,
        and estimating it on all of them would triple the cost of the transform for a
        change in the third decimal place.
        """
        names = list(feature_names)
        self.kernels_ = kernel_bank(self.n_kernels, self.seed)
        order, bounds = self._patient_bounds(groups, hour_index)
        n_pat = len(bounds) - 1
        rng = np.random.default_rng(self.seed)
        pick = np.sort(rng.choice(n_pat, size=min(max_patients, n_pat), replace=False))

        self.biases_ = np.zeros((len(self.channels), self.n_kernels,
                                 len(self.dilations), len(self.quantiles)), np.float32)
        for ci, ch in enumerate(self.channels):
            col = names.index(ch)
            v = X[:, col].astype(np.float32)
            for ki in range(self.n_kernels):
                for di, d in enumerate(self.dilations):
                    pool = []
                    for p in pick:
                        sl = order[bounds[p]:bounds[p + 1]]
                        pool.append(causal_conv(v[sl], self.kernels_[ki], d))
                    cat = np.concatenate(pool) if pool else np.zeros(1, np.float32)
                    self.biases_[ci, ki, di] = np.quantile(cat, self.quantiles)
        self.feature_names_ = self._names()
        return self

    def transform(self, X: np.ndarray, feature_names: Sequence[str], groups: np.ndarray,
                  hour_index: np.ndarray) -> np.ndarray:
        """(n_rows, n_features) float32, in the caller's original row order."""
        if self.biases_ is None:
            raise RuntimeError("CausalMiniRocket.fit() must be called first")
        names = list(feature_names)
        order, bounds = self._patient_bounds(groups, hour_index)
        out = np.zeros((X.shape[0], self.n_features()), dtype=np.float32)
        nq = len(self.quantiles)
        cols = [names.index(c) for c in self.channels]
        slices = [order[a:z] for a, z in zip(bounds[:-1], bounds[1:])]
        # Patient-outermost, and the convolution computed ONCE per
        # (patient, channel, kernel, dilation) and reused across that triple's
        # bias quantiles. The obvious loop order puts the quantile above the
        # convolution and recomputes it per bias — the same arithmetic, `nq` times,
        # inside every fold.
        for sl in slices:
            for ci, col in enumerate(cols):
                v = X[sl, col].astype(np.float32)
                f = ci * self.n_kernels * len(self.dilations) * nq
                for ki in range(self.n_kernels):
                    for di, d in enumerate(self.dilations):
                        y = causal_conv(v, self.kernels_[ki], d)
                        for qi in range(nq):
                            out[sl, f] = trailing_ppv(
                                (y > self.biases_[ci, ki, di, qi]).astype(np.float32),
                                self.window)
                            f += 1
        return out
