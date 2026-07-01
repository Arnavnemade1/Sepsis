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
| `quantum/qhdc.py` | **Quantum HDC primitives** (exact statevector sim: LCU bundling, phase binding, Hadamard/SWAP overlap) — Cumbo et al. 2026 |
| `quantum/gb_qhdc.py` | **GB-QHDC** — gradient boosting with a QHDC weak learner (SPSA, data re-uploading) |
| `scripts/qhdc_ablation.py` | Ablation matrix (GBM vs HDC vs GB-QHDC) + dimensional-capacity sweep |
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

Measured on real CinC 2019 patients (leakage-free, **patient-grouped CV**, wearable
channels — a single train/test split is too high-variance to trust at ~2%
prevalence; every number below survived multi-fold CV):

- The **original hand-tuned heuristic** scores barely above chance
  (AUROC ≈ 0.64, normalized utility ≈ 0.09, essentially uncalibrated). The
  "perfect" demo was an illusion of co-tuning the simulator and detector.
- The **data-grounded HDC detector** (learned multi-prototype archetypes +
  calibrated neurosymbolic fusion) is real and well-calibrated, but **trails the
  GBM benchmark** on this tabular task (AUROC ≈0.66 vs ≈0.72, 3-fold CV, 3000
  patients). "Rich" per-archetype similarity signals give it a small, real gain
  (≈0.635→≈0.643 standalone AUROC) — kept as the default.
- The **gradient-boosting benchmark** is the current production detector
  (`artifacts/gbm_wearable.joblib`) and the honest discrimination ceiling on
  wearable-only features: AUROC ≈0.72–0.76, AUPRC ≈0.07–0.08, at spec≥0.90
  detects ~52–76% of septic patients with a median lead time of **~20 hours**
  before Sepsis-3 onset — at the cost of a **~48% false-alarm rate** (a
  non-septic patient gets ≥1 alarm sometime during their stay). That false-alarm
  rate is the real bottleneck for a doctor-paging product, not raw AUROC.
- **AdaptHD-style archetype retraining** (`retrain_epochs`) was implemented (the
  docstring had promised it, unimplemented) and **measurably hurts** standalone
  HDC ranking (AUROC 0.634→0.586) — optimizing prototypes for argmax accuracy
  fights the margin structure the fusion layer needs. Off by default; the honest
  negative result is documented in code, not hidden.
- **HDC-as-feature-extractor for the GBM**
  (`backend.ml.models.augmented.HDCAugmentedGBM`) looked like a large win on one
  800-patient split (+79% AUPRC) — but that **did not replicate** under CV
  (GBM 0.7235±0.004 vs GBM+HDC 0.7263±0.011 AUROC — a gap smaller than the
  fold-to-fold noise). Kept as a documented, reproducible **negative result**,
  not wired into production. See that module's docstring for the full story.
- **Alarm-persistence debounce** (requiring N consecutive/windowed above-threshold
  hours before paging, the standard ICU-alarm-fatigue mitigation) was tested and
  trades away detection rate faster than it saves false alarms on this dataset —
  because `SEPSIS_LABEL_LEAD_HOURS=6` gives a narrow pre-onset alerting window.
  Not a free lunch here; not adopted as a default.
- **Lab-augmented features** (adding sparse labs: lactate, WBC, etc. — realistic
  for a clinician-facing app) give a modest AUROC bump (≈0.76→≈0.78, single
  split) but need CV to confirm before being treated as a reliable ceiling raise.

**Bottom line:** on this dataset and feature set, wearable-only sepsis detection
is close to a data ceiling around AUROC 0.72–0.78. The rigorous, CV-checked
finding is that HDC has **not** (yet) been made to beat gradient boosting on raw
discrimination here — every promising-looking shortcut either didn't replicate or
had a real, principled reason it hurt. HDC's genuine, validated value in this
project is as a complementary **interpretable case-based memory**
(`backend/aegis/hdc_memory.py`) and a **cross-modal consensus/resonance gate**
(`backend/aegis/resonance.py`), not as a raw-accuracy replacement for the tree
model. Stated, not hidden — see `backend/ml/models/hdc_detector.py` and
`backend/ml/models/augmented.py` docstrings for the full experimental trail.

See `artifacts/compare_*.json` for the exact, current numbers from your run.

---

## Quantum HDC — GB-QHDC (`backend/ml/quantum/`)

A rigorously-built, honestly-benchmarked exploration of whether a quantum
hyperdimensional representation can beat the classical ceiling. It implements the
QHDC primitives of **Cumbo, Blankenberg et al., npj Unconventional Computing (2026),
[arXiv:2511.12664]** as an exact, dependency-free numpy statevector simulator
(`qhdc.py`: LCU bundling, phase-oracle binding, Hadamard/SWAP overlap, NISQ shots),
and a novel **Gradient-Boosted QHDC** detector (`gb_qhdc.py`): a classical boosting
envelope whose weak learner is a QHDC circuit (residual-weighted LCU template +
Hadamard-test scoring, SPSA-tuned variational encoder with data re-uploading).

What is genuinely demonstrated (all verified, `verify_qhdc.py` 16/16):
- The QHDC algebra is exact (binding invertible & similarity-preserving; overlaps =
  ⟨a|b⟩; random hypervectors quasi-orthogonal).
- **The honest superposition thesis:** classical sign-bundling quantises each
  coordinate to 1 bit and cannot represent *weighted* superpositions; quantum LCU
  bundling preserves `Σ αₖ|ψₖ⟩` — weighted-bundle fidelity 0.88 (quantum) vs 0.71
  (classical) at k=32, and the gap *widens* with more superposed vectors. (Note: for
  *un*weighted orthogonal retrieval the two are equivalent — an earlier over-claim we
  caught and corrected.)

What is honestly NOT achieved (the CV-checked result):
- GB-QHDC does **not** beat — or match — the classical detectors on this tabular
  task. A naive quantum encoding produces a near-constant kernel (encoded states
  concentrate, median pairwise overlap ~0.81) so the weak learner cannot discriminate
  (train AUROC ~0.51). A quantile-spread encoding decorrelates the states
  (overlap ~0.25) and lets it learn (train AUROC ~0.62), but it generalises below
  classical HDC with an overfitting gap, and the statevector cost grows as 2^N. The
  ablation matrix + dimensional-capacity sweep are in `scripts/qhdc_ablation.py`
  (results → `artifacts/qhdc_ablation.json`).
- This matches the QML-for-tabular literature: no quantum accuracy advantage on
  structured clinical features at this scale. The paper's headline "500× faster" is a
  *throughput* claim on quantum-native ops, **not** an accuracy claim — we make none.

The contribution is the **novel, verified, citable framework and its rigorous honest
ablation**, not a benchmark-beating number. The production detector remains the
classical calibrated GBM.

---

## Honest limitations (deliberately not hidden)

- CinC has **no HRV or motion** channel, so the wearable-only feature set is a
  subset of what a real wristband streams; results here are conservative for HRV.
- CinC `Temp` is **core** temperature; a wrist sensor reads **skin** temp.
- CinC `Lactate` is **blood** lactate, standing in for the project's
  (still-aspirational) continuous sweat-lactate sensor.
- A retrospective ICU dataset is **not** a prospective wearable cohort. These
  metrics validate the *algorithm*, not a deployed medical device.
