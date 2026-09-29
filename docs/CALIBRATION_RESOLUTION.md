# Calibrating without a staircase: a registered gain

**Date:** 2026-09-29 · **Registration:** `artifacts/calibration_resolution_prereg.json`,
committed in `4c5a6ac` before the run · **Result:** `artifacts/calibration_resolution.json`
· **Reproduce:** `python -m backend.ml.scripts.calibration_resolution --jobs 6`

## The question

The stack's last step is isotonic regression, which turns its class-balanced score into a
calibrated probability. Isotonic's map is a staircase: 22 steps on the deployed stack.
Every hour on a step ties with every other, so the ranking the model made inside the step
is thrown away. On the deployed fit's 6,852 unseen patients the raw score ranked at AUROC
0.7476 and the calibrated one at 0.7425. The shipped model was also left with no operating
point between a 5% and a 51% false-alarm rate, which is how both refit gates tripped
(`DATA_LEVER.md`).

**Centred isotonic regression** (Oron & Flournoy, 2017) keeps the calibration and drops the
steps. It draws a line through the centre of each step instead of holding the step flat.
That makes it strictly increasing and equal to the isotonic value at every centre.
`calibration.centred_transform` reads the steps from the fitted thresholds alone, so it
applies to any already-fitted calibrator.

## Design

This used the fixed dev benchmark: 8,000 CinC patients, 3 folds, and the shipped configuration. Each fold
has one stack fit, shared by every arm, so the arms differ only in the final map:

- **isotonic** is the baseline;
- **centred** is the candidate;
- **sigmoid** (Platt, on the same calibration slice) is secondary;
- **raw** is for reference.

Success needed all three conditions:

- the pooled AUROC interval (paired patient bootstrap, 1,000 draws) lies above zero;
- ECE rises by at most 0.002;
- no alarm budget regresses.

## Result: success

| | isotonic | **centred** | sigmoid | raw |
|---|---|---|---|---|
| pooled AUROC | 0.7605 | **0.7626** | 0.7608 | 0.7734 |
| hour-matched AUROC | 0.6742 | **0.6764** | 0.6721 | 0.6971 |
| AUPRC | 0.0834 | 0.0870 | 0.0891 | 0.0906 |
| ECE | 0.0016 | **0.0014** | 0.0016 | 0.3739 |
| distinct scores | 296 | 310,048 | 310,048 | 310,048 |

- **AUROC, centred − isotonic: +0.0021 [+0.0006, +0.0036].** Per fold: +0.0028 / +0.0083 /
  +0.0023.
- **Calibration kept.** ECE fell slightly.
- **Detection at the alarm budget:** +0.037 [+0.022, +0.087] at 5% and +0.041 [+0.019,
  +0.057] at 20%, both significant; +0.004 at 10% and +0.010 at 15%, n.s. Nothing regressed.

## Reading it

- **Within a fold every strictly increasing map ranks exactly like the raw score.** So
  centred and sigmoid have identical per-fold deltas. Their difference is in the pooled
  number: each fold's calibrator puts scores on a slightly different scale. Centred holds
  that scale where isotonic did; sigmoid does not. That is why only centred clears the
  pooled bar.
- **The raw score's pooled 0.7734 is not available to ship.** It has no calibrator to
  disagree across folds, and it is uncalibrated (ECE 0.37). A probability of 0.4 from it
  means nothing.
- **The gain is recovered information, not new signal.** It is small (+0.002 pooled,
  +0.002 to +0.008 within fold). Its main practical effect is on the alarm ladder: the
  shipped model can place its threshold anywhere instead of on one of 22 steps.

## What it licenses

The registration ships nothing. It licenses registering a promotion that uses the centred
calibrator. That promotion is judged against the deployed artifact by the same gates, and
its threshold and conformal bands are re-derived, because a new calibrator is a new score
distribution.
