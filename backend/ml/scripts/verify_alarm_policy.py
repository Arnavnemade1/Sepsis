"""Adversarial checks on the alarm-policy search and its promotion gate.

The point of a gate is that it says no. These checks plant policies that SHOULD
be refused — one that is identical to the champion, one that buys detection by
alarming on everybody, one that wins only by luck on a handful of patients — and
assert the gate refuses each. Then they plant a genuinely better policy on
separable synthetic data and assert it is promoted, so the checks cannot be
passed by a gate that simply refuses everything.
"""
from __future__ import annotations

import numpy as np

from backend.ml.alarm_policy import (Policy, PatientSeries, evaluate, gate,
                                     search, to_series)

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


def synth(n_septic=140, n_plain=700, seed=0, separable=True):
    """Septic stays ramp; non-septic wander. Separable=False makes them identical,
    which is the case where nothing should ever be promotable."""
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n_septic):
        T = int(rng.integers(40, 90)); onset = int(T * 0.7)
        base = rng.uniform(0.01, 0.03)
        s = base + rng.normal(0, 0.01, T).clip(-0.02, 0.05)
        if separable:
            ramp = np.clip((np.arange(T) - onset + 30) / 30, 0, 1) * 0.16
            s = s + ramp
        pts.append(PatientSeries(np.clip(s, 0, 1), np.arange(T, dtype=float), onset))
    for i in range(n_plain):
        T = int(rng.integers(40, 90))
        s = rng.uniform(0.01, 0.03) + rng.normal(0, 0.012, T).clip(-0.02, 0.09)
        pts.append(PatientSeries(np.clip(s, 0, 1), np.arange(T, dtype=float), -1))
    return pts


print("=" * 72)
print("ALARM-POLICY VERIFICATION")
print("=" * 72)

# ---------------------------------------------------------------- structure
print("\n[1] policy mechanics")
p = PatientSeries(np.array([0.0, 0.2, 0.0, 0.2, 0.2, 0.2]), np.arange(6.0), 4)
check("plain fires on the first crossing",
      evaluate(Policy("plain", {"threshold": 0.1}), [p]).detection_rate == 1.0)
check("persistent(k=3) waits for a real run",
      Policy("persistent", {"threshold": 0.1, "k": 3}).build()(p.scores, p.hours) == 5.0)
check("persistent(k=3) ignores the isolated spike",
      Policy("persistent", {"threshold": 0.1, "k": 3}).build()(
          np.array([0.2, 0.0, 0.2, 0.0]), np.arange(4.0)) is None)
q = PatientSeries(np.array([0.05] * 30), np.arange(30.0), -1)
check("per_patient does not fire on a flat stay",
      evaluate(Policy("per_patient", {"warmup": 6, "z": 3.0, "floor": 0.0}), [q])
      .false_alarm_rate == 0.0)
check("a non-septic stay that never crosses is not a false alarm",
      evaluate(Policy("plain", {"threshold": 0.9}), [q]).false_alarm_rate == 0.0)

# ---------------------------------------------------------------- lead time
print("\n[2] detection requires firing BEFORE onset")
late = PatientSeries(np.array([0.0] * 5 + [0.9] * 3), np.arange(8.0), 4)
check("an alarm after onset does not count as a detection",
      evaluate(Policy("plain", {"threshold": 0.5}), [late]).detection_rate == 0.0)
early = PatientSeries(np.array([0.0, 0.9] + [0.0] * 6), np.arange(8.0), 4)
r = evaluate(Policy("plain", {"threshold": 0.5}), [early])
check("lead time is measured from the FIRST qualifying alarm",
      r.detection_rate == 1.0 and r.median_lead_hours == 3.0, f"lead {r.median_lead_hours}h")

# ---------------------------------------------------------------- the gate
print("\n[3] the gate refuses what it should")
locked = synth(seed=7)
champ = Policy("plain", {"threshold": 0.08})

v = gate(champ, champ, locked, n_boot=600, seed=1)
check("identical challenger is refused", not v.promoted, v.reason)

alarm_all = Policy("plain", {"threshold": 0.0})
v2 = gate(champ, alarm_all, locked, n_boot=600, seed=1)
check("a policy that alarms on everyone is refused", not v2.promoted,
      f"FAR {v2.champion_far:.1%} -> {v2.challenger_far:.1%}")

tiny = synth(n_septic=6, n_plain=40, seed=3)
v3 = gate(champ, Policy("plain", {"threshold": 0.079}), tiny, n_boot=600, seed=1)
check("a near-tie on 6 septic patients is refused", not v3.promoted, v3.reason)

flat = synth(seed=11, separable=False)
best_f, res_f, _ = search(flat, max_false_alarm_rate=0.10)
v4 = gate(champ, best_f, flat, n_boot=600, seed=1) if best_f else None
check("nothing is promoted when the score carries no signal",
      v4 is None or not v4.promoted, v4.reason if v4 else "no candidate met the ceiling")

# ---------------------------------------------------------------- and accepts
print("\n[4] the gate is not merely a refuser")
weak = Policy("plain", {"threshold": 0.5})       # far too strict: misses almost everything
strong = Policy("plain", {"threshold": 0.08})
v5 = gate(weak, strong, locked, n_boot=600, seed=1)
check("a genuinely better policy IS promoted", v5.promoted,
      f"detection {v5.champion_detection:.1%} -> {v5.challenger_detection:.1%}, "
      f"CI {v5.ci_low:+.3f}..{v5.ci_high:+.3f}")
check("promotion never raises the false-alarm rate",
      v5.challenger_far <= v5.champion_far + 1e-12,
      f"{v5.champion_far:.1%} -> {v5.challenger_far:.1%}")

# ---------------------------------------------------------------- search
print("\n[5] the search respects its ceiling")
best, res, field = search(locked, max_false_alarm_rate=0.08)
check("winner is inside the false-alarm ceiling", res.false_alarm_rate <= 0.08,
      f"{best.describe()} FAR {res.false_alarm_rate:.1%} det {res.detection_rate:.1%}")
check("every family was actually evaluated",
      {p.family for p, _ in field} == {"plain", "persistent", "integrator", "per_patient"},
      f"{len(field)} candidates")
over = [r.false_alarm_rate for p, r in field if r.false_alarm_rate > 0.08]
check("candidates over the ceiling exist and were excluded", len(over) > 0,
      f"{len(over)} of {len(field)} rejected on FAR")

print("\n" + "=" * 72)
print(f"ALARM-POLICY VERIFICATION: {sum(results)}/{len(results)} checks passed")
print("=" * 72)
raise SystemExit(0 if all(results) else 1)
