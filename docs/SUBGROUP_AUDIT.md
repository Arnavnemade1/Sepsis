# Subgroup audit — who the detector serves worse

**Question.** Every accuracy number this project has published is a cohort average,
and a cohort average is exactly the shape of number that hides a disparity. `Age`
and `Gender` have been in the feature matrix since the beginning
(`backend/ml/data/features.py`). This is the first pass that asks what the *shipped*
detector does to different people at the *shipped* threshold.

**Answer, in one line.** Discrimination is statistically flat across age and sex —
the AUROC gaps do not survive correction. What is **not** flat is the **alarm
burden**: at the one threshold the OS actually alarms on, patients 80+ absorb
**+6.3 percentage points** more false alarms than under-45s, and men absorb **+4.9
points** more than women — both surviving Bonferroni correction — with no detection
benefit that survives correction to pay for it.

---

## Method

| | |
|---|---|
| detector | `stack+xgb@utility` — the shipped production detector, fit fresh on the fixed folds |
| data | CinC 2019, 310,061 patient-hours, 8,000 patients, hour-prevalence 0.0209 |
| folds | 3, patient-grouped, seed 42 (`backend/ml/eval/bench.py`) |
| threshold | **0.0525**, from `artifacts/operating_point.json` |
| bootstrap | 400 draws, resampling **patients** |
| scores | grouped-CV out-of-fold — see the regime caveat below |
| multiplicity | Bonferroni over 16 pre-specified tests, α = 0.003125 |

Three commitments, because a fairness audit is easy to write so that it cannot
embarrass the model:

**1. The threshold is global and deployed, never re-tuned per subgroup.** Re-tuning
inside each group measures a system that does not exist and launders the disparity
into the threshold — the entire finding here is that *one shared threshold lands
differently on different people*. `verify_subgroup.py` plants a group with deflated
scores and asserts its detection rate falls at the shared threshold, and separately
asserts a per-group threshold *would* have differed, so the choice is not cosmetic.

**2. Patients are the unit everywhere.** Subgroups are defined per patient (age and
sex are verified constant within a stay — a patient whose age changes mid-stay is
rejected, not silently split across two cells), and every interval resamples
patients. The verification duplicates every patient's rows 4× and asserts the
interval barely moves, then shows a row bootstrap on the same data claiming an
interval less than half as wide.

**3. Gaps get intervals and a correction; small cells get neither.** Twenty-four
gap tests will produce a nominal "finding" by chance, so the significance verdict is
taken at α/n. Cells under 100 patients or 10 septic patients are printed — hiding
them would conceal who we cannot speak about — but are barred from claims.

---

## Results

At threshold 0.0525. `cal` is mean predicted risk ÷ observed rate: above 1.0 the
model over-states risk for that group, which at a shared threshold becomes more
alarms. `FA` is the fraction of never-septic patients alarmed at least once.

| subgroup | pts | septic | AUROC [95% CI] | cal | detect | FA [95% CI] | lead | utility |
|---|---:|---:|---|---:|---:|---|---:|---:|
| **cohort** | 8000 | 679 | 0.771 [0.753, 0.788] | 1.07 | 0.580 | 0.244 [0.235, 0.254] | 33.5 h | 0.375 |
| female | 3377 | 270 | 0.776 [0.749, 0.799] | 1.06 | 0.589 | **0.216** [0.199, 0.231] | 37.0 h | 0.385 |
| male | 4623 | 409 | 0.766 [0.745, 0.788] | 1.08 | 0.575 | **0.265** [0.252, 0.278] | 32.0 h | 0.369 |
| age <45 | 1165 | 100 | 0.792 [0.753, 0.829] | 0.99 | 0.620 | **0.216** [0.190, 0.238] | 37.0 h | 0.416 |
| age 45–64 | 2837 | 214 | 0.772 [0.742, 0.800] | 1.08 | 0.584 | 0.226 [0.208, 0.241] | 36.0 h | 0.390 |
| age 65–79 | 2815 | 267 | 0.767 [0.741, 0.792] | 1.07 | 0.581 | 0.261 [0.243, 0.276] | 32.0 h | 0.369 |
| age 80+ | 1183 | 98 | 0.751 [0.706, 0.796] | 1.13 | 0.531 | **0.279** [0.253, 0.307] | 32.0 h | 0.318 |
| age <45 · female | 489 | 44 | 0.784 [0.721, 0.847] | 0.88 | 0.568 | 0.182 [0.146, 0.214] | 35.0 h | 0.417 |
| age <45 · male | 676 | 56 | 0.799 [0.746, 0.842] | 1.09 | 0.661 | 0.240 [0.204, 0.270] | 39.0 h | 0.415 |
| age 45–64 · female | 1084 | 71 | 0.777 [0.718, 0.837] | 1.25 | 0.577 | 0.218 [0.192, 0.243] | 43.0 h | 0.370 |
| age 45–64 · male | 1753 | 143 | 0.769 [0.730, 0.812] | 1.00 | 0.587 | 0.230 [0.212, 0.254] | 26.5 h | 0.400 |
| age 65–79 · female | 1204 | 110 | 0.770 [0.730, 0.817] | 0.99 | 0.582 | 0.224 [0.201, 0.246] | 30.5 h | 0.395 |
| age 65–79 · male | 1611 | 157 | 0.763 [0.732, 0.794] | 1.13 | 0.580 | 0.288 [0.267, 0.309] | 33.0 h | 0.350 |
| age 80+ · female | 600 | 45 | 0.782 [0.712, 0.848] | 1.12 | 0.644 | 0.223 [0.191, 0.258] | 37.0 h | 0.353 |
| **age 80+ · male** | 583 | 53 | **0.719** [0.653, 0.781] | 1.13 | **0.434** | **0.338** [0.301, 0.381] | 27.0 h | 0.287 |

### What survives correction

| gap | Δ | 95% CI |
|---|---:|---|
| false-alarm rate, **female − male** | **−0.0493** | [−0.0706, −0.0302] |
| false-alarm rate, **age 80+ − age <45** | **+0.0633** | [+0.0300, +0.1003] |

Nothing else does. Every AUROC, AUPRC, utility, detection-rate and lead-time gap has
an interval that includes zero after correction.

---

## Reading this honestly

**The finding is about burden, not about discrimination.** The model does not rank
older patients or men detectably worse — those intervals are wide because the cells
are small, and a wide interval is not evidence of parity. What is measurable is that
the same threshold produces a materially heavier alarm load on the old and on men.
Alarm burden is the currency of alarm fatigue, so this is a clinical finding even
though it is not an accuracy finding.

**The mechanism is visible in the calibration column.** `cal` rises with age —
0.99 → 1.08 → 1.07 → 1.13 — meaning the model increasingly over-states risk relative
to what those patients actually experience. At a fixed threshold that over-statement
converts directly into alarms. Detection does not rise to match it (0.620 at <45 vs
0.531 at 80+, interval straddling zero), so the extra alarms are not buying extra
catches: **the oldest patients pay the alarm cost without a demonstrated detection
benefit.** Utility, which prices both sides, is the lowest of any age band at 80+
(0.318 vs 0.416).

**Men over 80 are the worst-served cell — and that claim is exploratory.** AUROC
0.719 against a cohort 0.771, detection 0.434 against 0.580, false alarms 0.338
against 0.244. This cell was chosen *by looking at the results*, and nothing here
corrects for that choice, so it is flagged exploratory in the artifact and is
excluded from the Bonferroni family. It is a hypothesis about where to look next,
not an established disparity. The honest next step is to test it on a site whose
data has not been examined — which is what the MIMIC-IV / eICU adapters are for.

**Superseded detector (2026-09-29).** The shipped detector has since been refitted on
24,206 patients and calibrated with centred isotonic regression, at a new threshold
(`DATA_LEVER.md`). Everything below describes the previous detector's score set; the
audit has not been re-run on the refit.

**A regime caveat on the absolute numbers.** These scores come from grouped
cross-validation over 8,000 patients, while the shipped detector was a single fit on
5,996. This ensemble's output scale is not stable across fits — holding the threshold
and the patients fixed, its false-alarm rate measures 0.085 / 0.178 / 0.209 / 0.165
for fits on 750 / 1500 / 2250 / 3000 patients — so the *absolute* FA levels in the
table above are a property of this score set, not a prediction of what the deployed
fit will produce. The **gaps** are what this audit claims, and a gap is a comparison
between subgroups inside one score set at one threshold, which a global change of
scale largely cancels out of. Read the columns as ranking who bears the burden, not
as forecasting the ward's alarm count.

**What this audit cannot say.** CinC 2019 carries age and sex and nothing else, so
race, ethnicity, insurance status and language — the axes a policy reviewer will ask
about first — are simply absent. The de-identification also caps age at 89, so the
"80+" band is compressed. This audit is a floor on what should be reported, not a
complete fairness assessment, and the missing axes are a property of the dataset
rather than a choice made here.

---

## Reproducing

```bash
make subgroup-audit      # fits the shipped detector on the fixed folds, writes artifacts/subgroup_audit.json
make verify-subgroup     # 15 adversarial checks on the audit's own machinery
make verify-integrity    # includes the guard that re-checks these published claims and rejects a forged one
```

The published claims are re-checked by `verify_integrity.py` check [9], which reads
`artifacts/subgroup_audit.json` and asserts every claim's arithmetic — a significant
gap must have an adjusted interval excluding zero, α must equal 0.05/family, no
post-hoc cell may be claimed, no underpowered cell may carry a claim — then forges a
straddling gap into a significant one and asserts the same guard rejects it.
