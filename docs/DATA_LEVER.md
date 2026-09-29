# More patients: a registered win, two refusals, and the four bugs shipping it exposed

**Dates:** 2026-09-28 to 29 · **Registrations, each committed before its run:** `ac684ff`
(data lever), `08949bf` (refit), `b2de39c` (refit re-test), `4c5a6ac` (calibration),
`20ec206` (refit with the centred calibrator: **promoted**) ·
**Results:** `artifacts/data_lever.json`, `artifacts/promotion_attempt.json` (latest
attempt; each attempt's record is in the commit that reported it)

## 1 · The lever: the same detector on four times the patients

The shipped detector was trained on 5,996 of CinC 2019's 40,335 patients. The registered
study fitted the same configuration (`patient:hgb+hgb2+et+lr+xgb@utility`) on 6,000 versus
all training patients of each fold of the full cohort, 3 patient-grouped folds, and scored
both arms on identical held-out patients.

| | 6,000 per fold | all (~26,900 per fold) | |
|---|---|---|---|
| pooled AUROC | 0.7693 | **0.7989** | **+0.0296 [+0.0256, +0.0334]** |
| per-fold delta | | | +0.035 / +0.032 / +0.027 |
| hour-matched AUROC | 0.693 | 0.735 | |
| AUPRC | 0.0773 | 0.0868 | |

**Registered verdict: success.** The registered secondary, detection at a 15% alarm budget,
read −0.001 (n.s.). That number was measured with the broken budget rule described in §4.
Re-measured with the corrected rule from the same saved scores (exploratory, not
registered): **+0.047 at 5% and +0.036 at 10% (both significant)**, +0.007 at 15% and
+0.003 at 20% (n.s.).

More patients is the only lever in this project that has moved ranking on hourly vitals,
where nine method families did not.

## 2 · Which patients to train on

All of CinC except the 8,067 patients GENESIS locked before any architecture work: 32,268
patients (`bench.genesis_dev()`), so the one untouched test set stays untouched. The
promotion split holds out 25%: 24,206 training patients, 8,062 held out.

## 3 · Refusal one: the configuration promoter, and the incumbent it had frozen

The promoter compares two *configurations* refitted on the same patients. Its incumbent was
hard-coded to the balanced four-member stack, and it stayed hard-coded after the
patient-objective stack replaced that in September. Against it, the 24k refit of the
shipped configuration lost 0.0051 held-out AUROC (zero tolerance), while winning
utility at each detector's own threshold (+0.3955 vs +0.3539). **Refused.**

Two things came out of it. At 24,000 patients the balanced objective ranks hours slightly
better (0.8043 vs 0.7992) while the patient objective still catches more patients. And
the incumbent is now read from the manifest; a refit of the *same* configuration is judged
by `--incumbent deployed` against the artifact actually serving.

## 4 · What shipping exposed

None of these showed up while the study was only measuring. Each surfaced as soon as a
model had to be installed.

1. **Identity by configuration, not by fit.** The conformal bands and the alarm threshold
   were stamped `patient:hgb+hgb2+et+lr+xgb@utility`. The refit carries the same string,
   so bands cut on the old model's score distribution would have been accepted for the
   new one. Keys now carry a training-set digest (`...#7fbbdd0ad322`); serving flags a
   model file whose fit differs from its threshold's and turns abstention off. `8068951`.
2. **An audit that would have lied.** `verify_integrity` check 6 rebuilt the
   8,000-patient promotion split by hand. After a 32k-patient refit it would have scored
   the model on its own training patients and still printed "never trained on". It now
   rebuilds the split from the manifest and proves it by the digest.
3. **A budget rule that broke its own specification.** Four modules chose the alarm
   threshold for a false-alarm budget as the k-th sorted non-septic peak, and alarmed at
   `>=`. That alarms every patient tied at that peak, plus one patient too many even
   without ties. Calibrated stacks emit only a few hundred distinct scores, so ties are the
   rule. On the deployed artifact one isotonic step held **46%** of non-septic patients'
   peaks, so its "5% budget" alarmed **50.8%** of them and was credited with 68.8%
   detection. `metrics.budget_threshold` now returns the smallest threshold whose alarmed
   fraction is at most the budget (0 violations in 5,000 random tied cases; the old rule is
   a planted defect in `verify_fastpaths`). `688dedf`.
4. **Quadratic metrics.** Every per-patient metric scanned all rows once per patient.
   On 8,000 patients the alarm-burden sweep took 286 s, and it now takes 0.01 s. Results are identical,
   except that utility's float sum used to follow Python's per-process hash seed. `688dedf`.

What the corrected budget rule changes elsewhere:

- **Supervision density** survives: significant at 5/5 budgets, sham control intact
  (`verify_supervision` 15/15). The exact sweep numbers in `SUPERVISION_DENSITY.md` were
  measured with the old rule. On the dev benchmark the patient-trained stack still beats
  the balanced one at every budget with honest thresholds (+0.05 to +0.09).
- **The TCN refusal** (`SEQUENCE_PATIENT_NULL.md`) had two grounds. Its detection loss was
  measured with the old rule and is unverified. Its other ground, a gain that exists only
  at the boundary blend weight, does not depend on the rule and still stands.
- **The HDC member null** stands on AUROC; its "costs detection" note used the old rule.

## 5 · Refusals two and three: the registered refit gate

The refit (fit `7fbbdd0ad322`) against the deployed artifact, on the **6,852** holdout
patients neither model trained on (495 septic):

| gate | deployed | refit | |
|---|---|---|---|
| AUROC | 0.7425 | **0.7987** | +0.0562, pass |
| utility at each own threshold | +0.3505 | **+0.4006** | pass |
| detection @ 5% budget | **0.413** | 0.324 | −0.089 [−0.126, −0.049] |
| detection @ 10% | 0.413 | 0.457 | +0.045 [+0.020, +0.089] |
| detection @ 15% | 0.413 | 0.474 | +0.061 [+0.028, +0.093] |
| detection @ 20% | 0.413 | 0.474 | +0.061 [+0.028, +0.158] |

The first run (`08949bf`) was refused through the broken budget rule, which credited the
deployed model with 0.688 at every budget. The re-test (`b2de39c`) corrected only that
rule and was **refused again**, this time on a real regression at 5%. Both verdicts
stand. The refit ranks far better and loses the tightest budget. There will be no further
attempt at this candidate.

## 6 · The staircase

The regression comes from how coarse both detectors are. The deployed model's detection is flat from
5% to 20%, and the refit's from 15% to 20%. The cause is the final isotonic calibration: a
22-step map, so every hour on a step ties with every other. On the deployed fit's unseen
patients:

| | raw ensemble score | isotonic (shipped) |
|---|---|---|
| distinct scores | 266,193 | 219 |
| AUROC | **0.7476** | 0.7425 |
| hour-matched AUROC | **0.6445** | 0.6383 |
| ECE | 0.39 | **0.006** |
| detection @ 5 / 10 / 15 / 20% | 0.410 / 0.465 / 0.495 / 0.537 | 0.386 / 0.410 / 0.410 / 0.410 |

The step discards 0.005 AUROC and leaves the shipped model with no operating point
between a 5% and a 51% false-alarm rate. Its threshold, 0.03, sits 0.00004 above the step
holding 46% of non-septic peaks. Whether a calibrator without steps keeps the calibration
and recovers the ranking was registered as its own question (`4c5a6ac`). It did: pooled
AUROC +0.0021 [+0.0006, +0.0036], ECE unchanged, and +0.037 detection at 5%
([CALIBRATION_RESOLUTION](CALIBRATION_RESOLUTION.md)).

## 7 · What shipped

The refit read through the centred calibrator (`20ec206`) composes the two changes that
each passed their own registered test. Registered before it ran, with its prediction: the
5% regression came from the staircase, so a map without steps should remove it. It did.
The same 6,852 patients, the same deployed incumbent, the same gates:

| gate | deployed | shipped | |
|---|---|---|---|
| AUROC | 0.7425 | **0.7996** | +0.0571, pass |
| utility at each own threshold | +0.3505 | **+0.4027** | pass |
| detection @ 5% budget | 0.413 | 0.421 | +0.008 [−0.020, +0.049] |
| detection @ 10% | 0.413 | **0.474** | +0.061 [+0.028, +0.101] |
| detection @ 15% | 0.413 | **0.551** | +0.138 [+0.089, +0.178] |
| detection @ 20% | 0.413 | **0.575** | +0.162 [+0.117, +0.211] |

**Promoted.** The artifact is the fit on the 24,206 training patients (12.9 MB), with its
threshold derived on half the judged patients: HIGH 0.0525, MEDIUM 0.0262. At that threshold,
on the other half, it flags 46.2% of septic patients and 8.5% of non-septic ones. Its key,
`patient:hgb+hgb2+et+lr+xgb@utility~centred#7fbbdd0ad322`, is on the model, the threshold,
the manifest and the conformal bands. The bands were re-derived on the shipped fit's own
8,062 holdout patients. The strongest level on the grid now holds: α = 0.03, locked
coverage 0.973 septic and 0.967 non-septic, where only 0.08 held before. Serving checks all of
it at load. Pointed at the old model file, it reports the mismatch and turns abstention off.

A caveat that matters: these 6,852 patients judged this refit three times. Each component's
significance comes from its own registered cross-validated study, and the gates here are
regression guards. The number to quote for the detector is the cross-validated one; the
held-out 0.800 is its confirmation on unseen patients, not an independent discovery.

Still to re-run on the new detector: the subgroup audit, the alarm-policy search, the
conformal cross-hospital study, and the external validation.

## Lessons

- A threshold rule that never checks its realised rate is a guess. The fix is the budget
  rule's own docstring, finally implemented.
- Identity has to include the data. Configuration plus seed says what a model *is*, not
  which score distribution it produces.
- Every refusal here found a real defect before it could ship: the frozen incumbent, the
  configuration-only key, the hand-rebuilt audit split, the tied threshold. The gate did
  its job twice by saying no, and the third registration shipped only after the
  mechanism it predicted was confirmed.
