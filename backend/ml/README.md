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
| `models/stacked.py` | **Stacked ensemble** (hgb+hgb2+et+lr [+hdc]) — the gap-closing production detector that beats the single GBM |
| `selfimprove.py` | **Recursive self-improvement** — a self-adaptive search that mutates its own champion, gated on locked patients it never searched |
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

- **Stacked ensemble beats the single-GBM benchmark** (`models/stacked.py`). No
  single hyperdimensional model closed the gap, but **stacked generalisation** does:
  a leakage-safe (patient-grouped out-of-fold) meta-learner over diverse learners —
  histogram boosting, a second differently-biased booster, depth-bounded extremely
  randomised trees, logistic regression (and optionally HDC) — beats the single GBM
  under 3-fold CV: **AUROC 0.768 vs 0.741 (+0.026), utility 0.40 vs 0.34**, better
  calibrated (ECE 0.005), *lower* variance, and a lower false-alarm rate. The
  extremely-randomised-trees member (a bagged, not boosted, learner) carries most of
  the complementary signal. Depth/leaf bounds keep the artifact ~8 MB. **This is now
  the production detector** (`artifacts/gbm_wearable.joblib`); train/promote with
  `scripts/train_stacked.py`.

**Bottom line:** wearable-only sepsis detection sits near a data ceiling around
AUROC 0.74–0.80, and the honest way we *raised* our number to the top of that band
was ensemble diversity, not a cleverer single model. No single HDC variant beat
gradient boosting on raw discrimination — every shortcut either didn't replicate or
had a principled reason it hurt (kept as documented negatives). HDC's genuine,
validated value here is as a complementary **interpretable case-based memory**
(`backend/aegis/hdc_memory.py`), a **cross-modal consensus/resonance gate**
(`backend/aegis/resonance.py`), and an optional stack member — not as a standalone
raw-accuracy replacement for trees. Stated, not hidden — see the
`hdc_detector.py` / `augmented.py` / `stacked.py` docstrings for the full trail.

See `artifacts/compare_*.json` for the exact, current numbers from your run.

---

## Recursive self-improvement (`selfimprove.py`)

```bash
make self-improve                 # search only; changes nothing
python -m backend.ml.scripts.self_improve --limit 2000 --rounds 4 --population 4 --promote
```

A self-adaptive **(1+λ) evolution strategy** over the stacked detector's configuration.
It is "recursive" in two senses: each round mutates the *champion the previous round
produced* (the search walks toward its own best result), and the mutation step size is
adapted from the loop's own hit-rate — shrink after a miss, widen after a win — so the
search tunes the search. It is **not** self-rewriting code, and it cannot invent signal
the data does not contain.

The interesting engineering here is not the search — it is the **refusal machinery**,
because a self-improving loop's characteristic failure is improving its own scoreboard:

- **A locked gate set.** Patients are split off *before* the search starts and are never
  scored, fitted, or selected on. The champion meets them once, at the end. Verified
  patient-disjoint from the search set.
- **Paired folds + Bonferroni.** Every candidate is scored on the champion's *identical*
  patient-grouped folds, so the comparison is paired. Since each round takes the best of
  λ challengers and *then* tests it, the winner's p-value is max-selection-biased; the
  threshold is corrected by the population size. A candidate must clear a minimum effect
  size **and** the corrected test.
- **A bootstrap CI at the gate.** Promotion needs the gate gain's 95% patient-level
  bootstrap CI to exclude zero — a CI straddling zero means the "gain" is patient
  sampling.
- **A full ledger** (`artifacts/self_improve_ledger.json`) recording every trial,
  accepted or rejected, with its per-fold scores and the reason. The rejects are the
  scientific content.

**Measured result (2,000 real CinC patients, 16 candidates over 4 rounds):** the loop
**accepted nothing** and left the shipped detector untouched. The incumbent scored
0.7314 search AUROC; no challenger cleared the bar. The instructive rejections:

| Trial | ΔAUROC | p | Outcome |
|---|---|---|---|
| 5 | **+0.0156** | 0.211 | rejected — big-looking gain, indistinguishable from fold noise |
| 8 | +0.0054 | 0.166 | rejected — clears the effect-size floor, fails significance |
| 1 | +0.0037 | 0.071 | rejected — below the effect-size floor |

Trial 5 is the whole point. A tuning loop that compared means would have banked
+0.0156 AUROC (+2.1% relative) and reported a better model; across three folds that
"gain" was noise. Step scale self-contracted 0.350 → 0.245 → 0.171 → 0.120 as rounds
missed.

This is a **negative result, and it is the expected one**: it independently reproduces
this subsystem's central finding — detection here is **data-limited, not model-limited**
(see *Honest limitations* below and `docs/PHASE7_REPORT.md`). Hyperparameters were never
the constraint, so a hyperparameter search should find nothing, and does. The value
shipped is the falsifiable apparatus, not a leaderboard number.

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
- **The detector does not "detect everything", and no amount of tuning will make it.**
  Within-site AUROC sits ~0.77 and cross-hospital ~0.69; at a usable threshold the
  early-warning false-alarm rate (~40%) is the real product bottleneck, not
  discrimination. The recursive self-improvement search (`selfimprove.py`) was run
  precisely to test whether tuning could move this and **found no admissible gain in 16
  candidates** — independent evidence that the ceiling is the *data* (domain shift,
  label noise, missing channels), not the model. Raising it requires better/more
  diverse data or new signal (HRV, waveforms, multi-site), not better hyperparameters.
