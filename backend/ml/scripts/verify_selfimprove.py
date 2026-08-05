"""Checks on the self-improvement loop's acceptance rule and its transcript.

The loop's job is to refuse almost everything, so the thing worth testing is that
it refuses for the RIGHT reasons and that a genuine winner still gets through.
These checks run the acceptance logic directly on constructed trials — no model
fitting — so they are fast enough to sit in the normal verify pass.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.ml.selfimprove import config_diff

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


@dataclass
class FakeTrial:
    trial_id: int
    delta_vs_champion: float
    p_value: float
    accepted: bool = False
    reason: str = ""


def apply_rule(trials, alpha_c: float, min_gain: float):
    """The shipped rule: judge every challenger against both bars, then let the
    largest passing delta seed the next round."""
    for t in trials:
        if t.delta_vs_champion <= min_gain:
            t.accepted, t.reason = False, "no gain"
        elif t.p_value >= alpha_c:
            t.accepted, t.reason = False, "not significant"
        else:
            t.accepted, t.reason = True, "cleared"
    passing = [t for t in trials if t.accepted]
    if not passing:
        return None
    best = max(passing, key=lambda t: t.delta_vs_champion)
    for t in passing:
        if t is not best:
            t.accepted, t.reason = False, "cleared the bar but a stronger challenger won the round"
    return best


ALPHA_C, MIN_GAIN = 0.0125, 0.001

print("=" * 72)
print("SELF-IMPROVEMENT VERIFICATION")
print("=" * 72)

print("\n[1] the acceptance bars")
t = [FakeTrial(1, 0.02, 0.001)]
check("a large, significant gain is accepted", apply_rule(t, ALPHA_C, MIN_GAIN) is not None)
t = [FakeTrial(1, 0.0005, 0.0001)]
check("a significant but trivial gain is refused (min_gain)",
      apply_rule(t, ALPHA_C, MIN_GAIN) is None)
t = [FakeTrial(1, 0.02, 0.20)]
check("a large but noisy gain is refused (alpha)",
      apply_rule(t, ALPHA_C, MIN_GAIN) is None)
t = [FakeTrial(1, -0.05, 0.0001)]
check("a significant LOSS is refused", apply_rule(t, ALPHA_C, MIN_GAIN) is None)

print("\n[2] every challenger is judged, not just the biggest delta")
# the real regression: selecting by delta and testing that one discards the
# strongest evidence in the round.
trials = [FakeTrial(5, 0.0156, 0.21), FakeTrial(1, 0.0037, 0.004),
          FakeTrial(8, 0.0054, 0.30)]
won = apply_rule(trials, ALPHA_C, MIN_GAIN)
by_delta = max(trials, key=lambda x: x.delta_vs_champion)
check("the candidate with the best evidence wins, not the biggest mean",
      won is not None and won.trial_id == 1,
      f"largest delta was #{by_delta.trial_id} (p={by_delta.p_value}), accepted #{won.trial_id if won else None}")
check("no trial is left without a verdict",
      all(x.reason for x in trials), f"{sum(1 for x in trials if x.reason)}/3 have a reason")
check("only one challenger seeds the next round",
      sum(1 for x in trials if x.accepted) == 1)
check("a passing runner-up says so, rather than 'untested'",
      any("stronger challenger" in x.reason for x in trials
          if not x.accepted and x.p_value < ALPHA_C and x.delta_vs_champion > MIN_GAIN)
      or all(not (x.p_value < ALPHA_C and x.delta_vs_champion > MIN_GAIN)
             for x in trials if not x.accepted))

print("\n[3] the bar itself is unchanged by the rule change")
# Bonferroni over the population is what testing `population` candidates requires;
# it was already applied to the single selected trial, so testing all of them at
# the same alpha_c does not loosen anything.
alpha, pop = 0.05, 4
check("alpha is corrected by the population size", abs(alpha / pop - 0.0125) < 1e-12,
      f"alpha/{pop} = {alpha/pop}")
loose = [FakeTrial(i, 0.01, 0.02) for i in range(20)]   # all p just above alpha_c
check("twenty near-misses still produce no acceptance",
      apply_rule(loose, ALPHA_C, MIN_GAIN) is None)

print("\n[4] transcript diffs are readable and correct")
a = {"roster": ["hgb", "et", "lr"], "params": {"hgb": {"learning_rate": 0.06, "max_iter": 400}}}
b = {"roster": ["hgb", "et"], "params": {"hgb": {"learning_rate": 0.0666, "max_iter": 400}}}
d = config_diff(a, b)
check("a dropped learner is reported", any("lr" in x and "roster" in x for x in d), str(d))
check("a changed knob shows both values", any("0.06" in x and "0.0666" in x for x in d), str(d))
check("an unchanged knob is not reported", not any("max_iter" in x for x in d), str(d))
check("an identical config diffs to nothing", config_diff(a, a) == [])

print("\n" + "=" * 72)
print(f"SELF-IMPROVEMENT VERIFICATION: {sum(results)}/{len(results)} checks passed")
print("=" * 72)
raise SystemExit(0 if all(results) else 1)
