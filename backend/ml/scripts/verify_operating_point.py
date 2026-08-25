"""Adversarial checks on the alarm operating point and the constraints that pick it.

Three bugs lived in this derivation at once, and all three were the same shape: an
invariant enforced somewhere other than where it could be violated.

* The **in-range rule** (a threshold must sit where the model's output density
  actually lives) was checked only by the integrity audit — i.e. after the artifact
  had been written and shipped.
* The **detector identity** was never recorded, so a threshold derived from a
  ``[hgb,hgb2,et,lr]/utility`` stack was published for a
  ``[hgb,hgb2,et,lr,xgb@utility]/balanced`` one and nothing could tell.
* The **false-alarm cap** was imposed on the very scores used to choose the
  threshold and never re-checked on held-out patients, so a point could be saved
  advertising ``FAR <= 0.15`` while delivering 0.170 to patients nobody had seen.

These checks plant each failure and assert the constraint now catches it — and, for
the cap, measure that enforcing it with a bootstrap margin actually generalises
better rather than merely sounding more careful.

    python -m backend.ml.scripts.verify_operating_point
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from backend.ml import operating_point as opmod

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


def cohort(n=1500, seed=0, septic_frac=0.12, tail_frac=0.0):
    """Patient-hours with a calibrated-looking score column.

    ``tail_frac`` sprinkles a few very high scores onto non-septic patients, which
    is what creates the sparse upper tail a threshold can wander into.
    """
    rng = np.random.default_rng(seed)
    y, g, h, o, p = [], [], [], [], []
    for i in range(n):
        T = int(rng.integers(30, 70))
        septic = rng.random() < septic_frac
        onset = int(T * 0.7) if septic else -1
        s = np.clip(rng.uniform(0.005, 0.03) + rng.normal(0, 0.012, T), 1e-4, 1)
        if septic:
            s = np.clip(s + np.clip((np.arange(T) - onset + 24) / 24, 0, 1) * 0.10, 1e-4, 1)
        elif tail_frac and rng.random() < tail_frac:
            s[rng.integers(0, T)] = rng.uniform(0.30, 0.60)
        lab = np.zeros(T, dtype=np.int8)
        if septic:
            lab[onset:] = 1
        y.append(lab); g.append(np.full(T, i)); h.append(np.arange(T, dtype=float))
        o.append(np.full(T, onset, dtype=float)); p.append(s)
    return (np.concatenate(y), np.concatenate(p), np.concatenate(g),
            np.concatenate(h), np.concatenate(o))


print("=" * 78)
print("OPERATING-POINT VERIFICATION")
print("=" * 78)

# ------------------------------------------------------------------- in range
print("\n[1] the threshold must sit where the model's output actually lives")
# The constraint only proves anything on a cohort whose unconstrained optimum is
# ACTUALLY out in the tail. With an ordinary cohort both arms pick the same point and
# the check passes while testing nothing — so the cohort here is deliberately one
# where alarming is rarely worth it (very low prevalence, no ramp), which pushes the
# utility optimum to the top of the grid.
y, p, g, h, o = cohort(n=1200, seed=1, septic_frac=0.01, tail_frac=0.10)
grid = np.linspace(0.01, 0.50, 60)
op = opmod.derive(y, p, g, h, o, objective="utility", grid=grid, detector="test")
loose = opmod.derive(y, p, g, h, o, objective="utility", grid=grid,
                     require_in_range=False, detector="test")
check("left alone, the utility optimum wanders above p95",
      loose.high > op.prob_p95,
      f"unconstrained picks {loose.high:.4f} vs p95 {op.prob_p95:.4f}")
check("the in-range rule pulls it back to where the output density lives",
      op.high <= op.prob_p95 < loose.high,
      f"constrained {op.high:.4f} <= p95 {op.prob_p95:.4f} < unconstrained {loose.high:.4f}")
check("MEDIUM is a pre-alarm strictly below HIGH", 0 < op.medium < op.high,
      f"{op.medium:.4f} < {op.high:.4f}")
check("the detector whose scores these are is recorded",
      op.detector == "test", f"detector={op.detector!r}")

# -------------------------------------------------------------- infeasibility
print("\n[2] an impossible request is refused, not quietly resolved")
raised = ""
try:
    opmod.derive(y, p, g, h, o, objective="constrained",
                 max_false_alarm_rate=1e-6, far_n_boot=60, detector="test")
except ValueError as e:
    raised = str(e)
check("an unreachable false-alarm cap raises", "false_alarm_rate <=" in raised,
      raised[:70])
raised2 = ""
try:
    opmod.derive(y, p, g, h, o, objective="constrained", max_false_alarm_rate=0.15,
                 grid=np.array([0.9, 0.95]), far_n_boot=60, detector="test")
except ValueError as e:
    raised2 = str(e)
check("a grid that cannot satisfy the in-range rule raises",
      "p95" in raised2, raised2[:70])

# ------------------------------------------------------------------ the cap
print("\n[3] the false-alarm cap is enforced with margin, and the margin is real")
y, p, g, h, o = cohort(n=1500, seed=2)
peaks = opmod._nonseptic_peaks(p, g, o)
grid = np.linspace(0.01, 0.20, 40)
point = np.array([(peaks >= t).mean() for t in grid])
upper = opmod.far_upper_bound(peaks, grid, n_boot=300, seed=0)
check("the enforced bound is never below the point estimate",
      bool((upper >= point - 1e-12).all()))
_, pb, gb, _, ob = cohort(n=6000, seed=3)
peaks_big = opmod._nonseptic_peaks(pb, gb, ob)
upper_big = opmod.far_upper_bound(peaks_big, grid, n_boot=300, seed=0)
point_big = np.array([(peaks_big >= t).mean() for t in grid])
m_small = float(np.mean(upper - point))
m_big = float(np.mean(upper_big - point_big))
check("the margin shrinks as the cohort grows (it is uncertainty, not padding)",
      m_big < m_small, f"mean margin {m_small:.4f} at n=1500 -> {m_big:.4f} at n=6000")

# ------------------------------------------------- what the margin does and does not
# The bootstrap margin was added after a capped point breached its cap on held-out
# patients. It turned out to fix only half of that: a margin answers SAMPLING noise
# in the false-alarm estimate, and is powerless against a score distribution that
# MOVES between the fit used to choose the threshold and the fit that gets deployed.
# Both halves are asserted here, because the second one is the reason the derivation
# had to be restructured rather than merely padded.
print("\n[4] the margin answers sampling noise, at the boundary where it matters")
CAP = 0.15

def _thr_at_cap(peaks, grid, cap, margin, seed):
    """Lowest threshold the cap admits — i.e. the most sensitive lawful alarm.

    This is deliberately the BOUNDARY case. A constrained optimum that lands well
    inside the cap has slack, and slack hides the whole question: sampling error only
    breaches a cap when the chosen point is already pressed against it, which is
    exactly where a utility objective puts it when missing cases is costly.
    """
    far = np.array([(peaks >= t).mean() for t in grid])
    if margin:
        far = opmod.far_upper_bound(peaks, grid, n_boot=200, seed=seed)
    ok = np.flatnonzero(far <= cap)
    return float(grid[ok[0]]) if ok.size else float("nan")

def boundary_trial(seed, n=260, shift=1.0):
    y, p, g, h, o = cohort(n=n, seed=seed, septic_frac=0.10)
    pts = np.array(sorted(set(g.tolist())))
    rng = np.random.default_rng(seed); rng.shuffle(pts)
    a = set(pts[: len(pts) // 2].tolist())
    ma = np.array([x in a for x in g])
    grid = np.linspace(0.005, 0.25, 120)
    derive_peaks = opmod._nonseptic_peaks(p[ma], g[ma], o[ma])
    hold_peaks = opmod._nonseptic_peaks(np.clip(p[~ma] * shift, 1e-4, 1.0),
                                        g[~ma], o[~ma])
    out = {}
    for name, margin in (("point", False), ("margin", True)):
        t = _thr_at_cap(derive_peaks, grid, CAP, margin, seed)
        out[name] = float((hold_peaks >= t).mean()) if not np.isnan(t) else 0.0
    return out

bt = [boundary_trial(s) for s in range(30)]
b_point = sum(1 for t in bt if t["point"] > CAP)
b_margin = sum(1 for t in bt if t["margin"] > CAP)
check("a cap set on the point estimate breaches on held-out patients",
      b_point > 0, f"{b_point}/{len(bt)} trials over cap")
check("setting it on the upper bound breaches strictly less often",
      b_margin < b_point, f"margin {b_margin}/{len(bt)} vs point {b_point}/{len(bt)}")

print("\n[5] ... and is powerless against a shift in the score distribution")
# The real failure was not noise. Holding the threshold and the patients fixed, this
# ensemble's false-alarm rate measured 0.085 / 0.178 / 0.209 / 0.165 across fits on
# 750 / 1500 / 2250 / 3000 patients. No confidence bound computed on one of those
# distributions describes another, which is why the threshold must be derived from
# scores produced the way the deployed model produces them.
shifted = [boundary_trial(s, shift=1.6) for s in range(30)]
s_margin = sum(1 for t in shifted if t["margin"] > CAP)
check("a shifted deployment distribution breaches the cap even WITH the margin",
      s_margin > 0, f"margin {s_margin}/{len(shifted)} trials over cap")
check("the margin cannot rescue a shift the way it rescues noise",
      s_margin > b_margin,
      f"shifted {s_margin}/{len(shifted)} vs unshifted {b_margin}/{len(bt)} — "
      f"only deriving in the deployment regime fixes this")

# ------------------------------------------------------------------ round trip
print("\n[6] the artifact survives a round trip")
op = opmod.derive(y, p, g, h, o, objective="constrained", max_false_alarm_rate=0.30,
                  far_n_boot=100, detector="balanced:a+b")
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "op.json")
    opmod.save(op, path)
    back = opmod.load(path)
    check("saved and reloaded unchanged", back is not None and back.high == op.high
          and back.detector == op.detector)
    # an artifact written before these fields existed must still load, or the serving
    # path silently falls back to hand-set guesses.
    d = json.load(open(path))
    for k in ("detector", "oof_false_alarm_upper"):
        d.pop(k, None)
    json.dump(d, open(path, "w"))
    old = opmod.load(path)
    check("an older artifact without the new fields still loads",
          old is not None and old.detector == "unknown",
          "otherwise AegisCore silently reverts to the hand-set constants")

print("\n" + "=" * 78)
ok = sum(results)
print(f"{ok}/{len(results)} checks passed")
print("=" * 78)
raise SystemExit(0 if ok == len(results) else 1)
