"""Adversarial checks on the alarm-policy search and its promotion gate.

The point of a gate is that it says no. These checks plant policies that SHOULD
be refused — one that is identical to the champion, one that buys detection by
alarming on everybody, one that wins only by luck on a handful of patients — and
assert the gate refuses each. Then they plant a genuinely better policy on
separable synthetic data and assert it is promoted, so the checks cannot be
passed by a gate that simply refuses everything.

The episode-economics half does the same for the OPERATIONAL layer. Its central
claim is structural rather than statistical — a notification schedule cannot move
the first alarm — so that is asserted directly (detection and false-alarm deltas
must be exactly zero) instead of being measured and hoped for. The rest plants:
a replay that disagrees with the live controller, an escalation someone tried to
suppress, a NaN that would break the website's strict JSON parser, and an
operating point derived on the wrong detector.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from backend.aegis.alert_episodes import AlertEpisodeController
from backend.ml.alarm_policy import (EpisodePolicy, PatientSeries, Policy, Verdict,
                                     episode_grid, evaluate, evaluate_episodes, gate,
                                     gate_burden, replay_episodes, save, search,
                                     tier_of, to_series)
from backend.ml.operating_point import (OperatingPoint, detector_key,
                                        require_shipped_detector)

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


# ------------------------------------------------- episode economics: conformance
print("\n[6] the replay IS the runtime controller, not a copy of its rules")
HI, MED = 0.04, 0.02
scores = np.array([0.00, 0.03, 0.05, 0.05, 0.03, 0.00, 0.05])
ep_pt = PatientSeries(scores, np.arange(len(scores), dtype=float), -1)
replayed = [e["state"] for e in replay_episodes(ep_pt, HI, MED, EpisodePolicy())]
ctl = AlertEpisodeController(high_renotify_seconds=4 * 3600.0)
direct = [ctl.update(tier_of(float(s), HI, MED), float(h) * 3600.0).state
          for s, h in zip(scores, np.arange(len(scores), dtype=float))]
check("replay state sequence equals the live controller's", replayed == direct,
      " -> ".join(replayed))

ctl2 = AlertEpisodeController()
ctl2.update("HIGH", 100.0)
before = ctl2._episode_id
stale = ctl2.update("HIGH", 50.0)
check("an out-of-order timestamp is CLOCK_ERROR and mutates nothing",
      stale.state == "CLOCK_ERROR" and not stale.notify and ctl2._episode_id == before,
      stale.state)

# ------------------------------------------------- escalation is not suppressible
print("\n[7] no schedule can suppress an escalation")
esc = PatientSeries(np.array([0.03, 0.05]), np.array([0.0, 1.0]), -1)
esc_ok = []
for pol in episode_grid():
    ev = replay_episodes(esc, HI, MED, pol)
    esc_ok.append(any(e["state"] == "ESCALATED" and e["notify"] for e in ev))
check("MEDIUM -> HIGH still notifies under every searched schedule", all(esc_ok),
      f"{sum(esc_ok)}/{len(esc_ok)} schedules")

# ------------------------------------------------- the structural safety guard
print("\n[8] a schedule cannot move the first alarm (asserted, not measured)")
locked_ep = synth(seed=7)
deltas = []
for pol in episode_grid():
    bv = gate_burden(EpisodePolicy(), pol, locked_ep, 0.08, 0.04, n_boot=120, seed=1)
    deltas.append((bv.detection_delta, bv.false_alarm_delta))
check("first-alarm detection and false-alarm deltas are exactly zero",
      all(d == 0.0 and f == 0.0 for d, f in deltas),
      f"{len(deltas)} schedules, max |delta| "
      f"{max(max(abs(d), abs(f)) for d, f in deltas):.1e}")

# ------------------------------------------------- the burden gate discriminates
print("\n[9] the burden gate refuses a tie and promotes a real reduction")
bv_same = gate_burden(EpisodePolicy(3600.0), EpisodePolicy(3600.0), locked_ep,
                      0.08, 0.04, n_boot=400, seed=1)
check("an identical schedule is refused", not bv_same.promoted, bv_same.reason)

# Non-septic stays only: there is no septic episode to go quiet on, so the guard
# below cannot fire and a real burden reduction is the only thing left to measure.
sustained = [PatientSeries(np.full(48, 0.9), np.arange(48.0), -1) for _ in range(60)]
bv_win = gate_burden(EpisodePolicy(3600.0), EpisodePolicy(None), sustained,
                     0.08, 0.04, n_boot=400, seed=1)
check("a genuinely lower-burden schedule IS promoted", bv_win.promoted,
      f"burden {bv_win.champion_burden:.2f} -> {bv_win.challenger_burden:.2f}, "
      f"CI {bv_win.ci_low:+.3f}..{bv_win.ci_high:+.3f}")

# ------------------------------------------- burden must not be bought with silence
print("\n[9b] a schedule cannot win by going quiet on a deteriorating patient")
# Septic stays that sit at HIGH for two days after onset, plus non-septic stays that
# carry the burden. "Never re-notify" minimises interruptions BY CONSTRUCTION here,
# which is exactly the tautology the guard exists to refuse.
mixed = ([PatientSeries(np.full(48, 0.9), np.arange(48.0), 2) for _ in range(40)]
         + [PatientSeries(np.full(48, 0.9), np.arange(48.0), -1) for _ in range(40)])
bv_quiet = gate_burden(EpisodePolicy(3600.0), EpisodePolicy(None), mixed,
                       0.08, 0.04, n_boot=400, seed=1)
check("burden falls, and the gate refuses it anyway",
      (not bv_quiet.promoted) and bv_quiet.challenger_burden < bv_quiet.champion_burden,
      bv_quiet.reason)
check("the refusal names the unannounced-HIGH cost",
      bv_quiet.challenger_silent_high > bv_quiet.champion_silent_high,
      f"silent HIGH h/episode {bv_quiet.champion_silent_high:.1f} -> "
      f"{bv_quiet.challenger_silent_high:.1f}")

# ------------------------------------------------- strict JSON
print("\n[10] a NaN metric serialises as null, not bare NaN")
quiet = [PatientSeries(np.zeros(24), np.arange(24.0), -1) for _ in range(5)]
quiet_res = evaluate_episodes(EpisodePolicy(), quiet, 0.08, 0.04)
check("an empty episode set really does produce NaN",
      math.isnan(quiet_res.median_time_to_clear_h), "median time-to-clear")
dummy = Verdict(False, "test", "a", "b", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "alarm_policy.json"
    save(dummy, Policy("plain", {"threshold": 0.03}), out,
         episode_metrics={"shipped": {
             "median_time_to_clear_h": quiet_res.median_time_to_clear_h,
             "escalation_recall": quiet_res.escalation_recall}})
    raw = out.read_text()
parsed = json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(
    ValueError("bare " + c)))
check("no bare NaN in the artifact and it parses strictly", "NaN" not in raw,
      "strict parse ok")
check("the NaN became null",
      parsed["episode_metrics"]["shipped"]["median_time_to_clear_h"] is None)
check("lead time is recorded as not reported",
      parsed["lead_time"]["reported"] is False, parsed["lead_time"]["reason"][:40])

# ------------------------------------------------- stale-artifact refusal
print("\n[11] a study refuses to measure at a threshold from another fit")


def _raises(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


def _op(detector: str) -> OperatingPoint:
    return OperatingPoint(high=0.03, medium=0.015, objective="constrained",
                          max_false_alarm_rate=0.2, oof_utility=0.0,
                          oof_detection_rate=0.0, oof_false_alarm_rate=0.0,
                          oof_median_lead_hours=0.0, prob_p95=0.05, prob_p99=0.1,
                          prob_max=0.2, frac_above_high=0.05, n_patients=100,
                          detector=detector)


check("a mismatched detector is fatal",
      _raises(lambda: require_shipped_detector(_op("balanced:hgb+lr"),
                                               ["hgb", "hgb2", "et", "lr", "xgb@utility"],
                                               "patient")))
check("a missing operating point is fatal",
      _raises(lambda: require_shipped_detector(None, ["hgb"], "patient")))
check("the matching pair is accepted",
      require_shipped_detector(_op("patient:hgb+lr"), ["hgb", "lr"], "patient")
      == detector_key(["hgb", "lr"], "patient"))

print("\n" + "=" * 72)
print(f"ALARM-POLICY VERIFICATION: {sum(results)}/{len(results)} checks passed")
print("=" * 72)
raise SystemExit(0 if all(results) else 1)
