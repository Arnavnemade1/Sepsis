"""Adversarial checks on the causal MiniRocket transform.

The failure this file exists to exclude is not a crash. It is a transform that quietly
reads forward in time, scores a magnificent lead time, and is wrong in the one
direction that would matter at a bedside. Standard MiniRocket does exactly that by
construction — a centred convolution plus whole-series PPV — so "we used MiniRocket"
is not evidence of anything until causality is demonstrated on this implementation.

Each check plants the specific violation and asserts it is caught. Where a check could
pass vacuously it also asserts the SAME probe moves something it is supposed to move.

    python -m backend.ml.scripts.verify_minirocket
"""
from __future__ import annotations

import numpy as np

from backend.ml.models.minirocket import (
    CausalMiniRocket, kernel_bank, causal_conv, trailing_ppv, KERNEL_LEN)

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


CH = ["HR", "MAP"]
NAMES = ["HR", "MAP", "junk"]


def cohort(n_patients=6, hours=60, seed=0):
    rng = np.random.default_rng(seed)
    n = n_patients * hours
    X = rng.normal(80, 8, size=(n, 3)).astype(np.float32)
    groups = np.repeat(np.arange(n_patients), hours)
    hour = np.tile(np.arange(hours), n_patients)
    return X, groups, hour


def fitted(X, groups, hour, **kw):
    mr = CausalMiniRocket(channels=CH, n_kernels=4, dilations=(1, 4),
                          quantiles=(0.5,), window=12, **kw)
    return mr.fit(X, NAMES, groups, hour, max_patients=6)


# --------------------------------------------------------------------------- #
def check_causality():
    """1. A spike at hour T cannot move any feature at an hour < T."""
    X, g, h = cohort()
    mr = fitted(X, g, h)
    base = mr.transform(X, NAMES, g, h)

    spike_t = 40                      # an hour in the middle of patient 0's stay
    X2 = X.copy()
    X2[spike_t, 0] += 500.0           # a violent, unmissable HR excursion
    after = mr.transform(X2, NAMES, g, h)

    p0 = np.flatnonzero(g == 0)
    earlier = p0[h[p0] < spike_t]
    later_or_same = p0[h[p0] >= spike_t]
    past_moved = np.abs(after[earlier] - base[earlier]).max()
    future_moved = np.abs(after[later_or_same] - base[later_or_same]).max()

    check("a future spike moves NO earlier hour", past_moved == 0.0,
          f"max |delta| over {len(earlier)} earlier hours = {past_moved:.2e}")
    # Without this the check above passes for a transform that outputs constants.
    check("...and the same spike DOES move the hours at/after it", future_moved > 0.0,
          f"max |delta| = {future_moved:.4f}")


def check_patient_isolation():
    """2. One patient's data cannot reach another's features."""
    X, g, h = cohort()
    mr = fitted(X, g, h)
    base = mr.transform(X, NAMES, g, h)
    X2 = X.copy()
    X2[g == 3] += 300.0                       # rewrite patient 3 entirely
    after = mr.transform(X2, NAMES, g, h)
    others = np.flatnonzero(g != 3)
    moved = np.abs(after[others] - base[others]).max()
    own = np.abs(after[g == 3] - base[g == 3]).max()
    check("rewriting one patient leaves every other patient untouched", moved == 0.0,
          f"max |delta| elsewhere = {moved:.2e}")
    check("...and that patient's own features did change", own > 0.0, f"{own:.4f}")


def check_row_order_invariance():
    """3. Features follow HOUR order, not the order rows happen to arrive in."""
    X, g, h = cohort()
    mr = fitted(X, g, h)
    base = mr.transform(X, NAMES, g, h)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(g))
    shuf = mr.transform(X[perm], NAMES, g[perm], h[perm])
    check("a shuffled row order yields identical per-row features",
          np.allclose(base[perm], shuf, atol=1e-6),
          f"max |delta| = {np.abs(base[perm] - shuf).max():.2e}")


def check_zero_sum_kernels():
    """4. The kernels are zero-sum, so they encode SHAPE and ignore level."""
    W = kernel_bank(84)
    sums = W.sum(axis=1)
    check("every kernel sums to zero (three +2, six -1)", np.all(sums == 0),
          f"{len(W)} kernels, max |sum| = {np.abs(sums).max():.0f}")
    const = np.full(50, 97.0, dtype=np.float32)
    y = causal_conv(const, W[0], 3)
    check("...so a constant series convolves to exactly zero", np.all(y == 0),
          f"max |y| = {np.abs(y).max():.2e}")


def check_bias_is_fitted():
    """5. Biases are parameters learned from the fitting rows, not constants."""
    X, g, h = cohort(seed=0)
    X2, g2, h2 = cohort(seed=9)
    X2 *= 2.5
    a, b = fitted(X, g, h), fitted(X2, g2, h2)
    check("different fitting data gives different biases",
          not np.allclose(a.biases_, b.biases_),
          f"max |delta| = {np.abs(a.biases_ - b.biases_).max():.3f}")
    # A bias fitted on the SCORED rows is the leak this separation exists to prevent.
    base = a.transform(X, NAMES, g, h)
    cross = b.transform(X, NAMES, g, h)
    check("...and the fitted bias actually changes the output",
          not np.allclose(base, cross), f"max |delta| = {np.abs(base - cross).max():.4f}")


def check_ppv_semantics():
    """6. PPV is a proportion: in [0,1], expanding before the window fills."""
    ind = np.array([1, 0, 1, 1, 0, 1], dtype=np.float32)
    p = trailing_ppv(ind, window=3)
    check("hour 0 averages exactly one value", p[0] == 1.0, f"p[0]={p[0]}")
    check("hour 1 averages two", abs(p[1] - 0.5) < 1e-6, f"p[1]={p[1]}")
    check("the window then stops growing", abs(p[3] - (0 + 1 + 1) / 3) < 1e-6, f"p[3]={p[3]:.4f}")
    X, g, h = cohort()
    F = fitted(X, g, h).transform(X, NAMES, g, h)
    check("every feature is a proportion in [0,1]", F.min() >= 0.0 and F.max() <= 1.0,
          f"range [{F.min():.3f}, {F.max():.3f}]")


def check_memory_estimate():
    """7. estimate_bytes is honest — it is what the guard rail is read from."""
    X, g, h = cohort()
    mr = fitted(X, g, h)
    F = mr.transform(X, NAMES, g, h)
    check("estimate_bytes matches the real output size",
          mr.estimate_bytes(len(X)) == F.nbytes,
          f"{mr.estimate_bytes(len(X)):,} == {F.nbytes:,} bytes")
    full = CausalMiniRocket(channels=["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"])
    gb = full.estimate_bytes(310_061) / 1024 ** 3
    check("the shipped default stays under 1 GB on the full cohort", gb < 1.0,
          f"{full.n_features()} features -> {gb:.2f} GB")


def check_dilation_reach():
    """8. Dilation d reads exactly (K-1)*d hours back — no further, no less."""
    T = 80
    x = np.zeros(T, dtype=np.float32)
    d = 5
    reach = (KERNEL_LEN - 1) * d
    W = kernel_bank(84)
    # An impulse at hour 0 must still be visible at exactly its reach, and gone after.
    x[0] = 1.0
    # Use a kernel whose FIRST tap is a +2 so the oldest position carries weight.
    k = next(w for w in W if w[0] == 2.0)
    y = causal_conv(x, k, d)
    check("an impulse is visible exactly (K-1)*d hours later", y[reach] != 0.0,
          f"reach = {reach}h, y[{reach}] = {y[reach]:.2f}")
    check("...and has left the window one hour after that", y[reach + 1] == 0.0,
          f"y[{reach + 1}] = {y[reach + 1]:.2f}")


# --------------------------------------------------------------------------- #
def main() -> None:
    print("Causal MiniRocket — adversarial checks\n")
    for fn in (check_causality, check_patient_isolation, check_row_order_invariance,
               check_zero_sum_kernels, check_bias_is_fitted, check_ppv_semantics,
               check_memory_estimate, check_dilation_reach):
        fn()
    ok, n = sum(results), len(results)
    print(f"\n{ok}/{n} checks passed")
    raise SystemExit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
