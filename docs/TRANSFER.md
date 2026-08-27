# Label-free deployment transfer — a decisive negative, and a correction

**Question.** Cross-hospital, `external_validation_wearable.json` records AUROC falling
0.7811 → 0.6797 (−13%) while normalized utility falls +0.3849 → +0.1759 (−54%). The
decision appeared to degrade four times faster than the ranking. Can a new hospital
recover that with **no labels of its own** — by estimating the shift from unlabeled
data and re-deriving its own operating point?

**Answer, in one line.** No — and the premise was partly wrong. Label-free prior
correction makes things **significantly worse** in both directions, and the gap a
re-derived threshold could recover *even with labels* is only **+0.007 to +0.034**
utility. The apparent "decision collapse" was substantially **threshold/fit coupling,
not hospital shift**.

---

## Method

| | |
|---|---|
| data | CinC 2019 setA / setB, 4,000 patients per site (~155k hours each) |
| detector | `xgb@utility`, fitted on the source hospital only |
| code | `backend/ml/shift.py`, `scripts/derive_transfer.py`, `scripts/verify_shift.py` |
| artifact | `artifacts/transfer.json` |

Three patient-disjoint roles. The detector is fitted on **source**. The **target/adapt**
half is the only target data any estimator may see, and it is seen *without labels*.
The **target/locked** half is tuned against by nothing, and every number below comes
from it. Both directions are run, because a method that works one way only has fitted
one hospital's idiosyncrasy.

**Pre-registered before the run:** primary = locked-half utility, adapted vs unadapted,
paired patient bootstrap, success = CI excluding zero. Reported regardless of outcome.

## The diagnosis: this is not label shift

| direction | domain AUC | true prior ratio | EM estimate | BBSE |
|---|---|---|---|---|
| A→B | **0.893** | 0.56× | 0.11× | failed (negative class weight) |
| B→A | **0.901** | 1.99× | **29.12×** | 0.124 |

A domain classifier separates the two hospitals' *covariates* at AUC ≈ 0.90. That is
enormous covariate shift, and it means the label-shift assumption EM requires is
violated — so EM's estimates are not merely noisy but wildly wrong (29× against a true
1.99×). `verify_shift.py` had already established the mechanism on synthetic data:
under covariate shift the source model is **already calibrated** on the target
(P(Y|X) never moved), and correcting its prior double-counts a shift the model handles
through X, driving mean calibration from 1.01 to 11.4.

The 1.54× prevalence difference between the hospitals is real. It is not the mechanism.

## The result

Locked-half utility, target hospital, threshold in parentheses:

| policy | A→B | B→A |
|---|---|---|
| unadapted (source threshold) | **+0.2369** (0.0275) | **+0.1684** (0.0450) |
| prior-corrected (EM, label-free) | −0.0001 | −0.3496 |
| rate-matched (assumption-free) | −0.0993 | +0.1684 |
| oracle (needs target labels) | +0.2436 (0.0575) | +0.2025 (0.0325) |

| direction | prior-corrected − unadapted | rate-matched − unadapted |
|---|---|---|
| A→B | **−0.2391** [−0.3317, −0.1259] **significant** | −0.3425 [−0.5014, −0.2293] **significant** |
| B→A | **−0.5342** [−0.7438, −0.3256] **significant** | +0.0000 [0, 0] n.s. |

Both label-free adaptations are significantly **harmful** or inert. The pre-registered
criterion is met in the negative direction, and that is the finding.

## The correction to our own understanding

The reachable gap — what a hospital *with* labels gains by re-tuning the threshold — is
**+0.0068 (A→B)** and **+0.0341 (B→A)**. The source-derived threshold transfers almost
optimally: 0.0275 vs an oracle 0.0575 on A→B, and both land within 0.007 utility of each
other. The utility surface is flat across that range.

So the dramatic "utility falls 54% while AUROC falls 13%" was **not** mostly a transfer
failure. `external_validation_wearable.json` scored the target with the *shipped*
constant 0.0525, which was derived against a **different fit**. Deriving the threshold
on the same fit that serves it — the discipline `derive_operating_point.py` already
enforces — recovers most of that apparent collapse before any adaptation is attempted.
This is the threshold/fit-coupling failure this repo documented once at model scale,
re-appearing at deployment scale.

**What is left is real and is not a decision problem.** After consistent thresholding,
A→B utility is +0.237 against a within-site +0.385. The residual lives in the ranking
under covariate and concept shift, where no label-free method reaches — consistent with
the proven impossibility boundary that no label-free procedure is uniformly valid under
joint shift.

## Honesty boundary

- Nothing here is installed. `artifacts/operating_point.json` and
  `artifacts/conformal.json` are untouched; a shipped detector is not modified on a
  hypothesis, least of all a refuted one.
- `--limit 4000` patients per site, one fit per direction. The direction and
  significance of the effect are unambiguous; the exact magnitudes are not tight.
- BBSE returns **NaN on failure** rather than a clipped 0.0000, because on real A→B
  data it produced a negative class weight and a clipped value would have read as a
  confident "prevalence zero".

## Reproduce

```bash
make verify-shift
```
20 checks; three of them pass by demonstrating a correction *failing* where it should.
All 6 guards mutation-audited.

```bash
make transfer
```
