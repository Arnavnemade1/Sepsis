# HDC case retrieval as a stack member — a registered null

**Date:** 2026-09-28 · **Registration:** `artifacts/hdc_member_prereg.json`, committed in
`f815ffe` before the run · **Result:** `artifacts/hdc_member.json` · **Reproduce:**
`python -m backend.ml.scripts.hdc_member --exploratory`

## The question

Every earlier attempt made hyperdimensional computing the *classifier*: archetype
prototypes, then HDC features fed to a booster. All lost to gradient boosting
(`backend/ml/README.md`). But stacking pays for disagreement, not strength: a weaker
XGBoost that rank-correlated 0.68 with the ensemble added AUROC where re-aiming the strong
members added nothing (`docs/FRONTIER_REPORT.md`). So this run gave HDC the role it plays
live in the case memory, an instance-based retriever and the least tree-like model in the
project, and asked whether its disagreement is worth anything to the detector.

## The member

`backend/ml/models/hdc_knn.py`. Each patient-hour becomes a 4,096-D hypervector with
32 data-driven levels per feature. Its score is the similarity-weighted vote of its 64
nearest hours in a bank drawn only from the training folds' patients. There are four
pre-declared variants: the classic flat encoding versus per-channel segments with integer
bundles, each voting on the hour label or the patient label.

## Registered result: null

On the fixed dev benchmark (8,000 CinC patients, 3 folds), the member was added to each
baseline through a combiner fitted only on the other folds, with a paired patient
bootstrap and Bonferroni correction over the four variants.

| | flat-hour | flat-patient | segmented-hour | segmented-patient |
|---|---|---|---|---|
| **+ best-AUROC stack** (pooled 0.7709) | −0.0001 | +0.0002 | −0.0021 | −0.0012 |
| **+ shipped detector** (pooled 0.7605) | +0.0021 | +0.0006 | +0.0004 | −0.0003 |

No interval excludes zero. Detection at the 15% false-alarm budget fell in all eight
combinations, by 1.5 to 5 points. The oracle control (label + noise) came out
significant (+0.071) and the noise control did not, so the test could have succeeded.

**Correction to the registration:** it quoted the primary baseline as 0.7811. That is the
benchmark's per-fold mean AUROC; the pooled out-of-fold AUROC this endpoint measures is
0.7709. The comparison is unaffected.

## What the member is on its own

The strongest HDC result in the project, and well decorrelated:

| variant | standalone AUROC | Spearman vs best stack |
|---|---|---|
| flat-hour | 0.7199 | 0.575 |
| flat-patient | **0.7260** | 0.598 |
| segmented-hour | 0.7095 | 0.569 |
| segmented-patient | 0.7177 | 0.596 |

Earlier HDC detectors reached 0.64–0.68. Instance retrieval with a patient-label vote puts
HDC within 0.02 of a single tuned booster on the same pooled measure (xgb-utility: 0.7456
pooled, 0.750 per-fold mean), and it disagrees with the ensemble more than the XGBoost member
that did pay off. It still adds nothing. Being
different is necessary for a stack member to help, but not sufficient.

## A flaw in the test, caught by its own control (exploratory)

The registered noise control was not significant by the registered rule, but it was
significantly **negative** (−0.0021 against the best stack, −0.0085 against the shipped
detector). The combiner re-scales each fold differently, and pooled AUROC punishes that
even when every fold's ranking is untouched. So the registered test carried a penalty
against any member.

Unregistered and labelled as such, each combination was then compared with the baseline
passed through the same combiner on its own, and with per-fold deltas, which cross-fold
scale cannot move:

- **Best stack:** nothing. Pooled +0.000 to +0.002, not significant. Per-fold deltas are
  negative in two of three folds (about −0.005).
- **Shipped detector:** within-fold hour AUROC rises by +0.005 to +0.013, clearing the
  corrected interval, but detection at the 15% budget falls in every variant. The shipped
  detector was trained on the patient label precisely to trade hour-level ranking for
  patients caught inside the alarm budget. The member pulls it back toward the hour label
  rather than adding information, and the noise control was also slightly positive here
  (+0.0016). Net against the raw shipped detector: +0.002, not significant.

**Lesson for every future stacking test in this repo:** compare against the baseline passed
through the same combiner, and report per-fold deltas beside the pooled one. A noise
member is the control that shows whether the combiner is neutral.

## What it means

This is the third HDC null and the ninth method family to come back empty on hourly CinC
vitals: the ceiling is the information in the data, not the model. HDC's validated value
stays where it was: agreement between physically independent pathways (the resonance gate,
91% of artifact false alarms removed) and interpretable precedent (the case memory).

**Worth keeping:** the retriever is a better precedent scorer than the live case memory's.
Its standalone AUROC is 0.726, against a published 0.626 that was measured through a
since-fixed tie-break bug. Adopting it for the case memory needs that re-measurement first.

**Where HDC could still add:** MIMIC-IV, where the bedside exam, vitals, labs and the care
process are genuinely independent pathways. That test must be registered before the data is
read.
