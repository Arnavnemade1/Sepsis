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

> **Verifying the honesty principle itself.** `make verify-integrity`
> (`verify_integrity.py`) is an *adversarial* audit: rather than checking the pipeline
> produces the right answer, it feeds the machinery a **known lie** and asserts the guard
> fires. Every claim this project rests on is a *negative* one ("no leakage", "no
> admissible gain", "the operating point is derived, not fitted to the test") — and
> negative claims are only falsifiable, never provable by a green functional test. So the
> audit plants the pathology: it shuffles the labels and demands AUROC collapse to chance
> (leakage tripwire); it shows in-sample AUROC (0.999) vs OOF (0.560) to prove the OOF
> discipline is load-bearing, not cosmetic; it feeds the statistical gate a zero-mean
> noise delta and a real gain and demands it reject one and accept the other; it hands the
> promotion CI two identical predictors and demands the interval straddle zero; it proves
> the search/gate patient sets are disjoint; and it guards the shipped alarm against ever
> again sitting in the sparse output tail (`high ≤ p95`) where the old 0.12 lived. Each
> check prints what it planted and what it expected. Complements — does not replace —
> `verify_full_system.py`.

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
| `models/stacked.py` | **Stacked ensemble** (hgb+hgb2+et+lr [+hdc, +lgbm]) — the production detector; now fittable against the utility-aligned objective |
| `eval/utility.py` | **The official utility, vectorised — and turned into a training target.** The scoring function's own decision rule, which the challenge label does not match |
| `eval/bench.py` | **Fixed-fold benchmark harness** — identical patient-grouped folds + paired patient-level bootstrap for every candidate |
| `models/boosters.py` | **Utility-aligned boosters** (sklearn HGB / LightGBM / XGBoost × objective) — the ablation behind the base-model upgrade |
| `models/sequence.py` | **Causal deep sequence detectors** — dilated TCN, Mamba-2-style selective state space, causal transformer; per-patient trajectories, not flat rows |
| `models/temporal.py` | **Causal decision-layer filters** (EMA / running max / decaying log-odds) over a row-wise detector's own score |
| `scripts/frontier*.py` | Run the benchmark, rebuild its table from saved out-of-fold predictions, confirm cross-hospital |
| `selfimprove.py` | **Recursive self-improvement** — a self-adaptive search that mutates its own champion, gated on locked patients it never searched |
| `operating_point.py` | **Derived alarm thresholds** — the operating point, computed from out-of-fold predictions against the official utility, instead of guessed |
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

## The accuracy frontier — the objective was wrong, but the ensemble already knew

Full write-up: [`docs/FRONTIER_REPORT.md`](../../docs/FRONTIER_REPORT.md).

```bash
make frontier SPEC=dev          # every registered technology, identical folds
make verify-frontier            # adversarial audit of the claims below
python -m backend.ml.scripts.promote_frontier --limit 8000 --jobs 6 --promote
```

Detection had been stuck (within-site ≈0.73–0.77, cross-hospital ≈0.69) and Phase 7
concluded the ceiling was data rather than capacity. This pass went looking outside the
family of things already tried — LightGBM/XGBoost, deep causal sequence models, a
tabular foundation model — and the interesting part is where it *didn't* pay off.

- **The challenge label is not the scoring function.** The label flips to 1 at
  `t_sepsis − 6h`, but the official utility pays up to **+1** for alarming in
  `[−12h, −6h)`, where the label says 0 — on this cohort 3,419 patient-hours carrying
  ~19% of all positive-side utility that every model had been trained to call negative.
  `eval/utility.py` derives the target the utility actually implies (`z = sign(u₁−u₀)`,
  `w = |u₁−u₀|`); the weighted Bayes rule under `(z,w)` *is* the utility-optimal alarm.
  Verified to reproduce the reference utility to 3e-16.
- **It is a large win for a single booster**: hgb 0.7283→**0.7467**, lgbm
  0.7408→**0.7533**, xgb 0.7213→**0.7590** (4-fold grouped CV, 8k patients). Worth more
  than the library choice. Calibration untouched (ECE ≤0.003).
- **…and worth nothing inside the stack.** `stack-utility` vs the shipped `stack`:
  ΔAUROC −0.0011, CI [−0.0043, +0.0020], n.s. The reason is a rank correlation of
  **0.965** between the two — re-aiming every member is *redundant with what stacking
  already does*. (`detect@FA15` looked like +8.1pp, but it is significant only at a 15%
  alarm budget and vanishes at 10% and 20%; quoting it would have been metric shopping.)
- **What the ensemble actually lacks is a disagreeing member.** `xgb-utility` is weaker
  alone (0.7503) but rank-correlates only **0.679** with the stack, and an *unfitted*
  50/50/50 rank average including it beats the incumbent: AUROC +0.0053, CI [+0.0015,
  +0.0095] (significant), AUPRC 0.0834→0.0912 (**+9.4%**, the metric that matters at 2%
  prevalence). Stacking is paid for disagreement, not for strength.
- So `StackedEnsembleDetector` now takes a **per-member objective** (`"xgb@utility"` in
  the roster) — the objective goes on the diverse member, not on all of them.
- **Sequence models change the alarm economics, not the AUROC.** `models/sequence.py`
  adds a dilated TCN, a Mamba-2-style selective state space and a causal transformer.
  None beats the boosters on AUROC and the fanciest is the worst (SSM 0.7000). But the
  TCN cuts **false alarms 0.498→0.299** and stretches lead **26.6h→39.5h**, paying in
  detection at matched specificity (0.73→0.50) — hence the new `detect@FA15` metric.
  Kept off the serving path: torch is 339 MB against xgboost's 7.5 MB.
  NEGATIVE: causal filters (`models/temporal.py`) do **not** recover that alarm win, so
  it is learned in the representation rather than output smoothness.
- **A leak in the incumbent baseline.** `GBMDetector` uses
  `early_stopping=True, validation_fraction=0.1`, and sklearn draws that split at random
  *row* level — so a patient sits on both sides and the stopping round is chosen against
  partly-memorised data. Controls (`hgb-*-noes`) registered; not yet run.
- **Honest gaps** (see the report): the causal transformer, the `utility-z`/`utility-w`
  decomposition and the early-stopping controls are unrun; TabPFN is blocked behind a
  licence acceptance requiring a vendor account; cross-hospital confirmation of the new
  candidates is the confirmation that still matters most.

Everything here is measured by `eval/bench.py`, which owns the folds so no candidate can
pick its own split, and bootstraps differences by resampling **patients** — hours inside
one stay are far too correlated for a row bootstrap, which would report an interval
several times too narrow and turn fold luck into a finding.

---

## The operating point (`operating_point.py`) — where the real headroom was

```bash
make operating-point                        # derive + gate, change nothing
python -m backend.ml.scripts.derive_operating_point --limit 4000 --save
```

Every other number in this subsystem is measured. The alarm thresholds were not — they
were constants in `AegisCore.__init__` (`high=0.12`, `medium=0.04`) carrying a comment
telling someone to derive them from validation, which never happened.

A detector's **ranking** and its **operating point** are different things. This project
spent enormous effort proving the ranking is at its data ceiling and none at all on the
point where that ranking is read. That asymmetry was expensive.

**What the guess cost** (real CinC, 4,000 patients; threshold derived from patient-grouped
out-of-fold predictions on train patients, then scored on held-out patients):

| operating point | utility | detection | false-alarm | lead |
|---|---|---|---|---|
| shipped `0.12` (hand-set) | +0.2946 | 35.2% | 9.3% | 50 h |
| **derived, FAR ≤ 15% (`0.0725`, now default)** | **+0.3356** | **45.1%** | 19.2% | **52 h** |
| derived, utility-optimal (`0.025`) | +0.3670 | 52.7% | 34.8% | 40 h |

**AUROC is identical (0.7334) across all three rows.** Same model, same predictions —
the entire difference is where the line was drawn. The shipped default is now the
FAR-capped point: it beats the old constant on utility, detection *and* lead time, and
loses only on alarm burden.

**The cliff — the part that actually mattered.** Sepsis is ~2% of hours and the detector
is well calibrated (ECE ~0.005), so it *correctly* almost never emits a high probability.
Its out-of-fold distribution is `p95 = 0.086`, `p99 = 0.140`. The shipped `0.12` therefore
sat at the **98.3rd percentile of the detector's own output** — not conservative, nearly
out of range — and just above it the system falls off a cliff:

| thr | OOF utility | detection |
|---|---|---|
| 0.025 | +0.3751 | 65.3% |
| 0.12 (shipped) | +0.0850 | 13.4% |
| 0.16 | +0.0309 | 6.3% |

A small calibration drift would have silently turned the alarm off. Deriving the point
from the model's real output distribution removes that failure mode by construction.

**Method notes (both are load-bearing):**
- Thresholds come from **out-of-fold** predictions only. A model's training predictions
  are far sharper than anything it emits in deployment (train utility 0.918 vs held-out
  0.347 on the same fit); a threshold tuned on those is tuned to a distribution that does
  not exist at serve time. An early version of this analysis made exactly that mistake.
- The objective is **explicit and configurable**. The CinC utility prices a missed case
  at −2 and a false alarm at −0.05, so left alone it will alarm on a third of healthy
  patients. Whether that trade is acceptable is a clinical judgement about alarm fatigue,
  not a modelling question — so it is a parameter, not a hidden constant.
- **Honest limitation:** the FAR cap constrains the *derivation* (10.6% out-of-fold), not
  the deployed rate — held-out FAR came in at 19.2%. The cap is a design input, not a
  guarantee.

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
  Within-site AUROC sits ~0.77 and cross-hospital ~0.69; the alarm burden, not
  discrimination, is the real product bottleneck. The recursive self-improvement search
  (`selfimprove.py`) was run precisely to test whether tuning could move this and
  **found no admissible gain in 16 candidates** — independent evidence that the ceiling
  is the *data* (domain shift, label noise, missing channels), not the model. Raising
  AUROC requires better/more diverse data or new signal (HRV, waveforms, multi-site),
  not better hyperparameters.
- **Corollary worth internalising:** when AUROC is at its ceiling, the remaining wins are
  in *decisions*, not *scores*. Deriving the operating point bought +14% utility and +10
  points of detection **at AUROC 0.7334 — completely unchanged**. It is worth auditing
  every other unmeasured constant in the serving path on the same suspicion.
