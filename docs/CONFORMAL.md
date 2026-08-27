# Conformal abstention — the third state, and what it cost to find out it does not travel

**Question.** The OS reads one scalar and sorts it into LOW / MEDIUM / HIGH. There is
no way for the detector to say *I cannot separate the two classes for this hour*, so an
ambiguous hour is rounded down to LOW and disappears. What does the alarm look like when
that state exists, and can it be given a guarantee that holds?

**Answer, in one line.** It can be given one that holds **within a hospital** — at 94%
requested coverage, delivered 93.5% on locked-out patients, with the detector declining
to distinguish on **72% of hours**. It does **not** hold across hospitals: calibrated on
CinC set A and served to set B, septic-hour coverage collapses from 0.929 to **0.739**
while the layer *halves* its abstention rate — it becomes confident exactly where it
should not be.

---

## Why this and not another alarm policy

Two findings in this repo are the same failure seen from different sides:

* `artifacts/alarm_policy.json` records a **refusal** — four firing-rule families
  (plain, persistent, integrator, per-patient) and none beats a plain threshold. The
  false-alarm rate is not fixable by a better rule.
* `docs/SUBGROUP_AUDIT.md` measured that the alarm burden **is not flat** (80+ absorb
  +6.3pp more false alarms than under-45s) and deliberately refused to launder that
  into per-group thresholds.

Both are one scalar being asked to carry a decision it has no vocabulary to qualify.
Split conformal prediction supplies the vocabulary: a per-hour prediction set over
{0,1}, where `{0,1}` is an explicit refusal to decide.

## Method

| | |
|---|---|
| detector | `stack+xgb@utility`, read from `artifacts/detector_manifest.json` |
| data | CinC 2019 dev spec — 310,061 patient-hours, 8,000 patients, prevalence 0.0209 |
| scores | patient-grouped out-of-fold (`artifacts/subgroup_oof_dev_stack+xgb@utility.npz`) |
| splits | **50 / 25 / 25 by patient** — calibrate / select / locked |
| taxonomy | class × data-quality stratum; `n_measured_this_hour` × `max({ch}__age)` |
| code | `backend/ml/conformal.py`, `backend/ml/scripts/derive_conformal.py` |

Three splits, because there are three jobs and merging any two launders a choice into
the evidence for it. Quantiles are **fit** on *calibrate*. α is **chosen** on *select*.
Every number published below comes from *locked*, which nothing was tuned against —
the same discipline `search_alarm_policy.py` uses.

**Class-conditional quantiles are mandatory, not a refinement.** At 2% prevalence with
`prob_max` 0.167, a pooled quantile is set almost entirely by negatives, label `1` never
enters any set, and the construction degenerates to "always {0}" — with *empty* sets for
the highest-scoring hours. `verify_conformal.py` plants exactly this and asserts it.

**The taxonomy is data quality, never the patient.** Strata come from how many channels
were actually observed and how stale they are. Age and sex are excluded from the
taxonomy and only *reported* on, preserving the subgroup audit's commitment that the bar
is never re-tuned according to who the patient is.

## α is a request, not a result

On exchangeable data the requested and delivered levels coincide. ICU hours are not
exchangeable — a stay is one correlated block — so they do not. Measured on *select*:

| α | nominal | cov⁺ | cov⁻ | macro⁺ | abstain | miss/req | holds? |
|---|---|---|---|---|---|---|---|
| 0.030 | 0.970 | 0.9341 | 0.9739 | 0.9343 | 0.766 | 2.20 | NO |
| 0.040 | 0.960 | 0.9231 | 0.9636 | 0.9234 | 0.739 | 1.92 | NO |
| 0.050 | 0.950 | 0.9231 | 0.9602 | 0.9234 | 0.735 | 1.54 | NO |
| **0.060** | **0.940** | **0.9231** | **0.9544** | **0.9234** | **0.729** | **1.28** | **yes** |
| 0.080 | 0.920 | 0.9036 | 0.9346 | 0.9045 | 0.661 | 1.21 | yes |
| 0.100 | 0.900 | 0.8786 | 0.9146 | 0.8797 | 0.562 | 1.21 | yes |

The overrun shrinks monotonically as α grows — the signature of a tail quantile
estimated from 3,317 correlated septic hours belonging to 348 patients. The gate is the
**miscoverage ratio**, not an absolute tolerance: an absolute 0.02 waves through five
times the requested error at α=0.005 and rejects a 1.2× overrun at α=0.10, which inverts
the real quality ordering. The chosen level is the **smallest** α that holds — the
strongest claim the data supports, not the easiest one to keep.

**Confirmed on the locked patients** (α = 0.06, 76,011 hours / 1,995 patients):

| | micro | macro |
|---|---|---|
| septic hours | 0.9350 | 0.9363 |
| non-septic | 0.9631 | 0.9771 |

Missed 0.0650 against a requested 0.06 — **1.08×**. Abstention rate **0.723**.

`macro` weights each patient once; `micro` pools hours. Both are always reported,
because a guarantee that survives only when long stays vote repeatedly is not one to
install.

## What abstention does to the alarm: nothing good, and that is the finding

At the shipped threshold 0.0525, on the locked patients:

| policy | utility | detect | FA | lead (h) |
|---|---|---|---|---|
| shipped (`p ≥ thr`) | **+0.3704** | 0.581 | 0.231 | 27.0 |
| UNCERTAIN silenced | +0.2587 | 0.419 | 0.099 | 30.0 |
| UNCERTAIN escalated | −0.1455 | 0.975 | 0.949 | 27.5 |

Neither reading beats the shipped policy. **Conformal abstention is not a better firing
rule and this repo does not claim it is** — it joins the four families
`search_alarm_policy.py` already refused. Its value is informational, and the
information is this:

> Of the 7,348 alarming hours, **4,184 (57%) are ones the detector cannot distinguish
> from noise** at 94% coverage. Of the 68,663 silent hours, **50,781 (74%) are not
> "ruled out" — they are unresolved.**

That is the honest reading of AUROC 0.78 at 2% prevalence, stated per hour instead of
per cohort.

## Reported, never enforced

Coverage per protected group on the locked patients (α = 0.06):

| group | hours | cov⁺ | cov⁻ | abstain |
|---|---|---|---|---|
| female | 32,191 | 0.8929 | 0.9605 | 0.696 |
| male | 43,820 | 0.9646 | 0.9650 | 0.743 |
| age <45 | 10,054 | 0.9966 | 0.9504 | 0.681 |
| age 45-64 | 27,803 | 0.8913 | 0.9686 | 0.697 |
| age 65-79 | 26,452 | 0.9320 | 0.9692 | 0.754 |
| age 80+ | 11,702 | 0.9530 | 0.9471 | 0.752 |

The septic-coverage spread across sex (0.893 vs 0.965) and age (0.891 vs 0.997) is
**a flag, not a claim**: unlike `docs/SUBGROUP_AUDIT.md` these cells carry no patient
bootstrap and no multiplicity correction. Running them through that treatment is the
obvious next question and has not been done.

## Cross-hospital: CORRECTED — the guarantee mostly does travel

> **This section previously reported that septic-hour coverage collapsed 0.929 -> 0.739
> across hospitals, and concluded "the guarantee does not travel". That was wrong, and
> the cause was a bug in this repo's own external harness.**
>
> `derive_conformal.py --external` fitted the detector **twice**: once on A-train to
> score the A-calibration rows, and again on *all* of A to score B. The bands were
> therefore calibrated on one model's score distribution and applied to a **different
> model's** scores. The apparent collapse was that mismatch, not a property of the
> transfer. This is the same threshold/fit-coupling failure documented elsewhere in
> this repo, at the level of conformal bands.

With a single consistent fit (4,000 patients per set, alpha = 0.06, nominal 0.94):

| | septic (micro / macro) | non-septic (micro / macro) |
|---|---|---|
| within set A | 0.9526 / 0.9527 | 0.9672 / 0.9875 |
| **across to set B** | **0.9250 / 0.9253** | 0.9826 / 0.9955 |

**Septic coverage falls 2.8 points, not 19.** At nominal 0.94 the transferred bands
miss 0.075 against a requested 0.06 — a **1.25x** overrun, inside the 1.5x gate this
layer already applies within-site.

Independently confirmed by `scripts/derive_transfer.py`, which calibrates on a source
hospital and measures on a locked target half. Transferred bands land within **0.002**
of bands calibrated on the target's own labels:

| detector | direction | plain (transferred) | oracle (target labels) |
|---|---|---|---|
| `xgb@utility` | A->B | 0.9260 | 0.9408 |
| `xgb@utility` | B->A | 0.9481 | 0.9722 |
| `stack+xgb@utility` | A->B | 0.9577 | 0.9577 |
| `stack+xgb@utility` | B->A | 0.8677 | 0.8695 |

The B->A row is the instructive one: the oracle reaches only 0.8695 itself, so that
shortfall is the **within-site** finite-sample and exchangeability limit, not a
transfer failure. There was nothing there for adaptation to recover.

**Weighted conformal (Tibshirani/Barber/Candes) is implemented and does not help here.**
`conformal_quantile_weighted` reweights the calibration scores by an estimated density
ratio, and on synthetic covariate shift it works dramatically — coverage 0.657 -> 0.889
against a nominal 0.90 in the under-covering direction. On the real hospital pair it is
neutral-to-harmful in every cell (worst: 0.8677 -> 0.7965), because once the fit is
consistent there is almost no gap left for it to close. It stays in the codebase behind
an optional `weights=` argument; uniform weights reproduce the unweighted bands exactly,
which `verify_conformal.py` asserts.

## Where the layer has no jurisdiction

The bands describe the **calibrated detector's** output distribution and nothing else.
Two serving paths therefore carry no conformal claim, and both say so rather than
omitting the field — a missing `certainty` reads as "resolved", which is the one thing
it must never imply:

* **Cohort beds** (`_bed_snapshot` in `backend/aegis/server.py`) report a fused cohort
  risk read against 0.35 / 0.6 bands, not the detector's 0.0525. Applying bands
  calibrated on one score distribution to a differently-scaled one is the exact
  threshold/fit-coupling error this repo has already paid for once. These emit
  `certainty: {unavailable: true, …}` and the UI shows **NO CLAIM**.
* **Hard clinical overrides** (qSOFA ≥ 2, shock) are rules, not model predictions. The
  set is still shown, but `abstain` is forced false and the UI shows **RULE OVERRIDE**.
  An abstention badge on a rule-driven escalation would suggest the system is unsure
  about a decision it did not make with the model at all.

## Honesty boundary

* Conformal does not reduce false alarms for free. It re-labels some UNCERTAIN. Both
  readings are reported above and neither wins.
* The finite-sample guarantee assumes exchangeability. ICU hours violate it twice
  (within-stay correlation, cross-site shift). The nominal 1−α is a hypothesis; the
  measured coverage is the evidence.
* **Nothing here satisfies any FDA SaMD requirement.** Conformal prediction is a
  statistical construction, not a regulatory artifact.

## Reproduce

```bash
make verify-conformal
```

```bash
make conformal
```

```bash
make conformal-external
```

Three of the 20 checks in `verify_conformal.py` pass by demonstrating a **loss** of
coverage — row-level calibration flattering itself, shift breaking the guarantee, and the
pooled taxonomy going degenerate. Every guard has been mutation-tested: break it, and the
matching check fails. Two checks were found vacuous that way and rewritten.
