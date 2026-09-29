# Patient supervision on a sequence architecture — refused, and what it taught

**Date:** 2026-08-30 · **Verdict:** bar not met, nothing promoted.

## What was tried and why it looked promising

Eleven nulls in this project share one shape: a new *feature* over a fixed window fed to
a generic classifier. The one thing that has worked is a change to the *label* —
supervision density. So this run added no new signal and instead combined three things
that each had prior evidence:

1. **The TCN** was the only frontier candidate to beat everything on false-alarm rate
   (0.299 vs the stack's 0.333), and `FRONTIER_REPORT` concluded that advantage "is
   learned in the representation, not recoverable by post-processing". It had never been
   added to the stack.
2. **The patient label**, which shipped as the production objective (+20–48% detection).
3. **Stacking pays for disagreement**, not strength.

A sequence model reading the whole prefix is the architecture that *should* exploit a
patient-level label best — it can attribute the label across the stay itself rather than
being handed the attribution by a hand-set 6-hour window.

## Result — 3 locked folds, 8,000 patients

| candidate | AUROC | detect@FA15 | FA@spec90 | util |
|---|---|---|---|---|
| **stack-patient** (shipped) | **0.7689 ± 0.0011** | **0.610** | 0.258 | **+0.4013** |
| tcn-patient | 0.7256 ± 0.0032 | 0.571 | 0.312 | +0.2759 |
| tcn-utility | 0.7267 ± 0.0038 | 0.387 | **0.247** | +0.3294 |

Rank correlation with the stack: **0.619** (tcn-patient), 0.666 (tcn-utility). The member
*was* decorrelated — comparable to the 0.679 that made `xgb@utility` pay. Disagreement
was not the problem; the TCN simply is not strong enough on this cohort.

## The result that was significant, and why it is not claimed

> **Measurement note (2026-09-29).** The detection losses below were measured with a budget
> threshold rule that alarmed every patient tied at the budget's peak (`DATA_LEVER.md` §4),
> which flatters a tied baseline like the calibrated stack. They are unverified. The
> refusal still stands on its second ground, which does not depend on that rule: the gain
> exists only at the boundary blend weight.

`stack + tcn-patient` at blend weight 0.2 gives **ΔAUROC +0.0038, CI [+0.0006, +0.0070]**.
It is refused on two grounds.

**It loses the metric that matters.** The same blend is **Δdetect@FA15 −0.032**, and at
w = 0.3 it is **−0.047, CI [−0.069, −0.013]** — significantly negative. The trade is a
0.004 AUROC crumb for a 3–7 point loss in detection inside the alarm budget.

**It exists only at the boundary.** ΔAUROC decays monotonically with blend weight:
+0.0038 → +0.0026 → +0.0007 → −0.0030 at w = 0.2/0.3/0.4/0.5. As w → 0 the blend becomes
the baseline and ΔAUROC → 0 by construction, so a gain surviving only at the smallest
weight tested is the boundary, not a finding. This is the same signature `FRONTIER_REPORT`
caught when a `detect@FA15` gain appeared at one alarm budget and vanished at both
neighbours, and it was correctly called metric shopping then.

## Two findings worth keeping

**Supervision density generalises to sequence architectures — strongly.** The patient
label moved the TCN's detect@FA15 from **0.387 to 0.571 (+0.184)**, a far larger effect
than it had on the flat-feature boosters. Independent confirmation that the one lever
that works is a label change, and that it works harder the more temporal context the
architecture has.

**The TCN's low-alarm advantage and the patient label's detection advantage are in
tension.** The TCN's celebrated false-alarm rate (0.247 under `utility`) *degrades* to
0.312 under `patient`, while its detection rises. You cannot have both from this
architecture: patient supervision buys detection by spending the alarm budget the
utility objective was saving. That trade-off had not been measured before and it
explains why this combination was never going to be free.

## What this does not rule out

The TCN was tested only as an unfitted rank blend, deliberately — no fitted parameter
means no leakage, and no torch dependency enters production until the benchmark earns it.
A *fitted* meta-learner might weight it better than a fixed rank average. Given the
detection loss above, that is not obviously worth the dependency, and it is recorded as
untested rather than as excluded.
