# ICU architecture experiment — October 24, 2026

This study targets **0.90 hourly AUROC** on the original CinC 2019 task. It does
not substitute whole-stay classification for early prediction. The existing
historical locked result remains 0.862; see [GENESIS](GENESIS.md).

## Measured result

The frozen ensemble improved hourly AUROC from **0.84977 to 0.85176** on
**6,454 confirmation patients / 252,704 hours**. The 0.90 target was **not reached**.
The paired patient-bootstrap improvement was **+0.00199**, with a 95% interval of
**[+0.00008, +0.00413]** (1,000 draws). This is a small gain, with the interval's
lower bound close to zero. The candidate's AUROC interval was [0.83737, 0.86491].

| Confirmation metric | Same-split baseline | Frozen ensemble |
|---|---:|---:|
| Hourly AUROC | 0.84977 | **0.85176** |
| Hour-matched AUROC | 0.79299 | **0.79513** |
| AUPRC | 0.12170 | **0.12556** |
| Detection at calibration-derived threshold | 53.19% | **55.11%** |
| Never-septic patients with an alarm | 13.85% | 14.09% |
| Brier score | 0.01666 | **0.01639** |
| Expected calibration error | 0.00267 | **0.00111** |

All five preregistered research gates passed. The modest increase in detection
comes with a small increase in false alarms, still below the 15% cap. The detection
change is a point estimate; this study does not establish a statistically significant
detection improvement. The ensemble has its own final probability calibration,
whereas the baseline uses its original internal-holdout calibration.

Selection retained **50% trajectory LightGBM + 25% randomized LightGBM + 25%
XGBoost**, with zero weight on the baseline member. The individual confirmation
AUROCs were 0.85105, 0.84764, and 0.84876 respectively. Those weaker members were
retained because their combined predictions improved the selection score; their
weights were frozen before confirmation. The full candidate field is preserved in
[results.json](../artifacts/icu_architecture/results.json) and every selection-grid
entry in [selection.json](../artifacts/icu_architecture/selection.json).

The run took **26.8 minutes** on four CPU threads; XGBoost accounted for 21.5 minutes.
Reloaded native model predictions matched the evaluated ensemble exactly (maximum
absolute error 0). The saved `candidate.joblib` is a **research artifact**. It has
not been promoted into the ward. These development confirmation results should not
be compared directly to the older 0.862 locked result, which used different training
and evaluation patients. Independent validation and a production-compatible fit
remain necessary before deployment; 0.9 remains an unmet target.

## Architecture

The candidate is an ensemble with four possible members:

1. The existing tuned LightGBM architecture over the 510 GENESIS features.
2. The same learner with 204 additional causal observation-dynamics features.
3. A randomized-threshold LightGBM with those additional features.
4. A depth-wise XGBoost over the original features.

The additional features measure slopes, variability and deviations from recent
history over 6 and 24 hours. They use actual observations, not repeated imputed
values, and exclude the current observation from its own reference distribution.
Missing history remains missing. Every sum restarts at the patient boundary.
No new packages are needed beyond the pinned project dependencies.

## Evaluation contract

The existing development folds are allocated once:

| Role | Patients | Purpose |
|---|---:|---|
| Training | 19,360 | Fit members; hold out 10% internally for early stopping |
| Selection | 4,195 | Choose a convex probability blend on a fixed quarter-step grid |
| Calibration | 2,259 | Fit a monotone probability map and a 15% false-alarm threshold |
| Confirmation | 6,454 | Evaluate the frozen candidate, once |
| Historical locked set | 8,067 | Excluded from this study |

The confirmation patients have been used as development patients in earlier
research. This is a patient-held-out **development confirmation**, not a pristine
test set or cross-hospital transport validation. Both hospitals occur in training;
the per-hospital metrics are subgroup results, not external validation.

The protocol records candidate parameters, source hashes, seeds, patient partition
fingerprints, and selection rules before training. Selection must preserve
hour-matched AUROC within 0.002 of the baseline; confirmation reports this metric
without a tolerance. The research gate requires a positive paired patient-bootstrap
AUROC delta, no decrease in hour-matched AUROC or AUPRC, no decrease in detection at
the calibration-derived thresholds, and a confirmation false-alarm rate at most 15%.
No threshold is selected using confirmation labels. The separately reported
`detect_far5`/`detect_far15` values describe a retrospective frontier and are not
deployment operating points. Detection means an alarm at or before the first
positive challenge label, which is already shifted six hours before clinical onset.

The original challenge labels and all scored hours are retained. The official
[PhysioNet task](https://physionet.org/content/challenge-2019/1.0.0/) requires a
prediction each hour using only current and preceding measurements. AUROC, AUPRC,
hour-matched AUROC, calibration, and fixed-threshold alarms are reported together.

## Reproduce and inspect

```bash
make verify-icu-ensemble
make improve-icu
# Existing experiment directories are never overwritten. A deliberate rerun needs:
make improve-icu ARGS='--output artifacts/icu_architecture_replication'
```

The full CinC cache must be present locally. The command uses four CPU threads and
memory-mapped base features. Results go to `artifacts/icu_architecture/`:

- `protocol.json`: frozen design, patient fingerprints and code hashes.
- `training.json`: tree counts, timings, and each member's selection score.
- `selection.json`: every candidate blend, selected weights, calibration and thresholds.
- `results.json`: all confirmation outcomes, bootstrap intervals and gate decisions.
- `environment.json` and `verification.json`: runtime versions and checks from this run.
- `candidate.joblib`: standalone research detector using native model payloads.
- `members.joblib` and `predictions.npz`: reproducibility artifacts, excluded from Git.

The detector offers `predict_record` and bounded-batch `predict_matrix`; zero-weight
members incur no inference cost. Loading native model payloads is lazy. Prediction
after serialization must reproduce the evaluated result to within 1e-7.

This command cannot promote a model. The existing production detector, its fit-bound
alarm thresholds, and the unseen-patient requirements of the replay rooms remain
governed by their existing promotion contract. A research win would still need a
compatible production fit, independently derived thresholds, and transport testing.

## Verification

`verify_icu_ensemble` checks future rewriting and truncation, patient reordering and
isolation, outcome/site independence, hand-computed observed slopes, missing-history
handling, record/cohort feature parity, partition disjointness and lock preservation,
blend constraints, rejecting a harmful member, monotone calibration, native model
reloads, batch invariance, and skipping zero-weight members. It needs no clinical data.

## Production fit — November 11, 2026: refused

The frozen ensemble was fitted on the shipped fits' 24,206 training patients and judged
once on the 8,062-patient holdout against the served single LightGBM
(`make icu-ensemble`; protocol [icu_ensemble_prereg.json](../artifacts/icu_ensemble_prereg.json),
result [icu_ensemble.json](../artifacts/icu_ensemble.json)). It did **not** reproduce the
October gain. The served detector stays.

| Holdout metric (8,062 patients, 312,901 hours) | Served single LightGBM | Frozen ensemble |
|---|---:|---:|
| Hourly AUROC | **0.84633** | 0.84272 |
| Hour-matched AUROC | **0.79862** | 0.79467 |
| AUPRC | 0.13769 | **0.14264** |
| Detection / false alarms at own point (half B) | 51.1% / 14.3% | 50.8% / 12.5% |

Paired patient-bootstrap delta AUROC: **−0.00362 [−0.00682, −0.00048]** (1,000 draws).
Two of five registered gates failed (AUROC delta, hour-matched AUROC), so nothing was
promoted. Detection inside a fixed 15% alarm budget was +0.0215 [+0.0000, +0.0400], and
AUPRC was higher; neither was a promotion criterion on its own.

One unregistered observation, not a tested explanation: the members stopped early at
213, 148 and 118 trees, against 484, 212 and 367 in October and 539 for the served model
(which uses a patience of 150 rounds, not 100). Per the registration, this candidate is
not re-tested on these patients.

The run also rebuilt the local feature caches, which had been cleared; the rebuilt split
reproduced the shipped fit digest `7fbbdd0ad322` before anything was fitted.
