# Detection ML Subsystem — Data-Grounded Sepsis Detection

This package is the part of HyperSepsisTwin-Ω that makes the detection **real**.
The original detector (`backend/core/neurosymbolic.py`) is a hand-tuned heuristic
whose weights, thresholds, and HDC archetypes were invented. That is fine for a
live demo but it cannot be called "accurate" because it was never measured
against ground truth. This subsystem fixes that: it grounds, trains, and
validates detection on **real labelled clinical data** (PhysioNet/CinC 2019) with
leakage-free, patient-level evaluation.

> **Honesty principle:** no number in this project is reported without the metric
> and dataset that justify it. Every threshold/weight/archetype is either learned
> from data or accompanied by the evaluation that earned it.

---

## The dataset: PhysioNet/CinC 2019

~40,000 ICU patients, one hourly row per patient, with a `SepsisLabel` that is
set to 1 starting **6 hours before** Sepsis-3 onset (the challenge rewards *early*
prediction). Sepsis is rare: ≈7% of patients, ≈1.8% of hours. That imbalance is
why we report AUPRC, sensitivity@specificity, early-warning lead time, and the
official utility score — **not** accuracy (a "never septic" model scores ~98%).

```bash
# Download everything (resumable, ~ tens of minutes):
python -m backend.ml.data.cinc2019 --download
# Or a quick subset to iterate:
python -m backend.ml.data.cinc2019 --download --limit-per-set 1200
python -m backend.ml.data.cinc2019 --summary
```

Data lands in `data/cinc2019/` (git-ignored — redistribute via PhysioNet only).

---

## Package layout

| Module | Responsibility |
|---|---|
| `config.py` | Paths, channel taxonomy (wearable vs lab), label semantics |
| `data/cinc2019.py` | Parallel downloader + PSV parser → `PatientRecord` |
| `data/features.py` | **Causal** feature engineering (no future leakage) |
| `eval/splits.py` | Patient-level, prevalence-stratified splits |
| `eval/metrics.py` | AUROC/AUPRC, calibration, early warning, **official CinC utility** |
| `models/heuristic_adapter.py` | Runs the *existing* detector on real data (baseline) |
| `models/hdc_detector.py` | **Data-grounded HDC detector** (learned archetypes + calibrated fusion) |
| `models/gbm_baseline.py` | Gradient-boosting benchmark (SOTA yardstick) |
| `scripts/` | Runnable entry points |

---

## Reproduce the results

```bash
# 1. Honest baseline: how good is the SHIPPED heuristic on real patients?
python -m backend.ml.scripts.evaluate_current_detector --limit 2400

# 2. Train + compare the grounded HDC detector vs the GBM benchmark:
python -m backend.ml.scripts.train_and_compare --limit 2400 --feature-set wearable --save
#    --feature-set lab   adds sparsely-sampled labs (lactate, WBC, ...)
```

---

## What we learned (and why it matters)

Measured on real CinC 2019 patients (leakage-free patient-level test split):

- The **original hand-tuned heuristic** scores barely above chance
  (AUROC ≈ 0.64, normalized utility ≈ 0.09, essentially uncalibrated). The
  "perfect" demo was an illusion of co-tuning the simulator and detector.
- The **data-grounded HDC detector** (learned multi-prototype archetypes +
  calibrated neurosymbolic fusion) is far better and notably **well-calibrated**,
  while keeping the project's HDC identity.
- The **gradient-boosting benchmark** sets the discrimination ceiling and is the
  yardstick the HDC model is held to.

See `artifacts/compare_*.json` for the exact, current numbers from your run.

---

## Honest limitations (deliberately not hidden)

- CinC has **no HRV or motion** channel, so the wearable-only feature set is a
  subset of what a real wristband streams; results here are conservative for HRV.
- CinC `Temp` is **core** temperature; a wrist sensor reads **skin** temp.
- CinC `Lactate` is **blood** lactate, standing in for the project's
  (still-aspirational) continuous sweat-lactate sensor.
- A retrospective ICU dataset is **not** a prospective wearable cohort. These
  metrics validate the *algorithm*, not a deployed medical device.
