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
| `models/minirocket.py` | **Causal MiniRocket** — fixed random convolutional kernels over trajectory *shape*; strictly causal, memory-bounded (tested, no gain — see below) |
| `models/forest_vsa.py` | **Forest-VSA** — tree-leaf topology as a representation, with its own lossless ceiling for comparison (tested, no gain — see below) |
| `conformal.py` | **Mondrian conformal abstention** — the alarm's third state; see `docs/CONFORMAL.md` |
| `falsify.py` | **Falsification engine** — mutates the *scientific invariant* and asserts a NAMED check fails; the survivor list is the deliverable. `docs/FALSIFICATION.md` |
| `mutations.py` | The mutation catalogue — one entry per failure mode this repo has actually suffered |
| `fitid.py` | **Fit fingerprints** — a quantity is valid only against the fit it came from; refuses cross-fit reads at the point of use |
| `shift.py` | Label/covariate/concept shift diagnosis (label-free); see `docs/TRANSFER.md` |
| `scripts/frontier*.py` | Run the benchmark, rebuild its table from saved out-of-fold predictions, confirm cross-hospital |
| `scripts/verify_minirocket.py` | 17 adversarial checks on the MiniRocket transform — causality, patient isolation, memory honesty |
| `scripts/falsify_audit.py` | Run the falsification audit; `--gate` fails only on regression, `--slow` includes the expensive suites |
| `scripts/verify_falsify.py` | The engine held to its own rule (a crash is not a kill; the wrong check is not a kill) |
| `scripts/verify_fitid.py` | Fit fingerprints, incl. the regression test that this session's three false findings are now impossible |
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
| `mimic/` | **MIMIC-IV line.** Sepsis-3 from scratch with all six SOFA organ systems (a `measured`-baseline onset rule that does not fire on data arrival), the Ledger harness (tier firewall P/T/S, pre-suspicion AUROC, symmetric censoring, design-weighted evaluation, label sensitivity, temporal validation, information ledger) and the Sepsis-3 Process Twin. Around it: the **Care-Process Transplant** (is a model reading the patient or the doctors?), the **admission audit** (causality / truncation / isolation attacks on every model), the **information-flow audit** (covert channels into treatment state; the strict tier), the **cross-examination** (definitional oracle, sepsis-vs-sickness contrasts, clinician-conditional AUROC, clock-matched horizons, soft-label AUROC, subgroups, alert frontier) and the **trust layer** (hash-chained code-bound ledger, power-gated pre-registration, Ladder lock). Verified on the open demo (71 checks, mutation-tested); full run via BigQuery once credentialed. `docs/MIMIC.md`, `make verify-mimic` |
| `genesis/` | **GENESIS — the all-variable line.** Everything above is the *wearable* detector (7 vitals). `genesis/` asks how far hourly early warning goes when the model may read all 40 CinC variables: cohort-wide raw store, 510 vectorised causal features, a full-cohort harness with a **locked 20% test set**, the pooled = between/within decomposition, an **hour-matched AUROC** that removes the ICU-clock artifact, factorised who × when × direct heads, and the Genesis Network (causal GRU with a Sepsis-3 AND-gate head and a physiological world model). Results, nulls and the measured ceiling: `docs/GENESIS.md`. Audit: `make verify-genesis` |

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

## Architectures proposed, tested, and not adopted

A 2024–2026 "Forest-VSA" proposal argued the ceiling was structural: boosters and HDC
trained in isolation and blended only as scalars, HDC's isotropic linear quantisation,
and no fast multi-scale temporal kernel. **Its three diagnoses of the code were all
correct** — the meta-learner really is `LogisticRegression` (`models/stacked.py`), HDC
quantisation really is linear (`models/hdc_detector.py`), and leaf-index features had
never been tried. Two of its four remedies were then ruled out on structure and two
were built and benchmarked. None is adopted.

**Everything below is `--spec dev`, 3 patient-grouped folds, 310,061 hours / 8,000
patients — the same protocol as `artifacts/frontier_summary.json`, so these rows are
directly comparable to that table.** Reference points from it: production
`stack+xgb@utility` **0.7811**, `stack` 0.7765, `xgb-utility` **0.7503**.

| candidate | AUROC | AUPRC | utility | detect@FA15 | cost |
|---|---|---|---|---|---|
| `xgb@u-donor` (= `xgb-utility`, the paired baseline) | **0.7503 ± 0.0098** | 0.0774 | **+0.3381** | **0.497** | 22 s |
| `mr-xgb@utility` (base features + 336 causal MiniRocket) | 0.7495 ± 0.0088 | 0.0728 | +0.3246 | 0.468 | 162 s |

> Paired patient-bootstrap vs the baseline: **+0.0012 AUROC, 95% CI [−0.0090, +0.0129],
> n.s.** AUPRC, utility and detection all move *down*; the cost moves up 7.4×.

### 1. Causal MiniRocket — built, tested, no gain (`models/minirocket.py`)

A fixed bank of length-9 dilated kernels with weights from {−1, +2} (exactly three
+2s), summarised by PPV. No weights are learned. Phase 7 had already shown that
*hand-authored* temporal aggregates bought nothing; this was the different bet — a
random basis over trajectory **shape** rather than more human-chosen summaries. It is
the one proposal that was genuinely untried, and it was worth running. It does not help
on this data.

Two things the textbook version gets wrong here, both fixed in this implementation:

- **It is not causal.** Standard MiniRocket centres the convolution and pools PPV over
  the whole series, so hour 20 reads hour 60. Used as-is for hour-by-hour prediction it
  would post a superb lead time made entirely of hindsight. Here the kernel's last tap
  sits on hour *t*, the left edge is padded with the series' own first value, and PPV is
  a trailing window. `verify_minirocket.py` plants a future spike and asserts **no**
  earlier hour moves — and asserts the same spike *does* move hours at and after it, so
  the check cannot pass vacuously.
- **It does not fit in memory.** The standard bank emits 9,996 features: on this cohort
  that is 310,061 × 9,996 float32 = **11.6 GB for the transform alone**. The bank is
  subsampled deterministically instead; the shipped default emits 336 features
  (**0.39 GB**, 31 s for the full cohort), and `estimate_bytes()` prices any other
  configuration before you run it.

A useful property that fell out: the kernels are **zero-sum** (3×(+2) + 6×(−1) = 0), so
they respond to shape and are blind to level — a constant series convolves to exactly
zero. That is asserted, not assumed.

`make verify-minirocket` → 17 checks. All 17 were mutation-audited: a centred
convolution, whole-series PPV pooling, a missing per-patient split, a missing hour sort
and a lying `estimate_bytes` each make the matching check fail (5/5 non-vacuous). **The
negative result is therefore evidence about the data, not a symptom of a broken
transform.**

### 2. Forest-VSA — built, tested, dominated (`models/forest_vsa.py`)

The proposal: bind tree-id ⊗ leaf-id into a 10,000-D hypervector and retrieve
associatively. The module implements it **and its own ceiling**, because the
hypervector is a random projection of the exact leaf one-hot and cannot carry
information the one-hot does not. Measuring only the lossy form would leave the result
uninterpretable — which is how a representation with no information gain ships as an
architecture.

On the first fold the *lossless* one-hot read of the topology scored **0.7146 against
the donor booster's 0.7384 on the identical fold** — i.e. the leaf indices, read
perfectly, are already worse than the model that produced them. That is expected in
hindsight: leaf *indices* discard the fitted leaf *values*, which is where a booster
keeps most of what it learned. If the ceiling loses to the donor, the lossy projection
underneath it cannot win.

### 3. Quantile-warped VSA encoding — cannot move the shipped number

Ruled out on structure, without a run:

- **GBMs are invariant to monotone transforms of their features**, so warping the input
  distribution does nothing to the trees either way.
- `HistGradientBoosting` **already bins by quantiles internally** — the proposal
  describes what the incumbent does by construction.
- It could therefore only affect the HDC channel, and the shipped roster is
  `['hgb','hgb2','et','lr','xgb@utility']` with `use_hdc=False`. **HDC is not in the
  production detector.**

### 4. Energy-based meta-learner — a rename of an existing configuration

`σ(w₁·p_GBM + w₂·exp(−E) + b)` is a logistic regression over two features, which is
structurally what the current meta-learner already *is* once an HDC member joins the
roster. That configuration exists behind `use_hdc=True` and was not selected on merit.
The conformal half of the proposal is built and documented separately in
`docs/CONFORMAL.md`.

### On the proposal's cost figures

The quoted "≈1.4 ms per bed tick" is not the measured number: `artifacts/scale.json`
records **1.57 ms** at one bed and **1.69 ms** at 350. The "≈1.8 ms" for the proposed
system is an estimate of something never built. For scale, HDC already consumes
**43% of the tick budget** (257 ms of 593 ms at 350 beds) and the HDC-based resonance
gate another 28% — so hyperdimensional work is ~71% of the runtime while contributing
nothing to the shipped detector.

### What this run actually adds

A third independent line of evidence for the same conclusion. Phase 7 ruled out richer
hand-authored temporal features; `selfimprove.py` searched 16 champion mutations and
admitted none; this pass ruled out a random convolutional temporal basis **and** tree
topology as a representation. Different mechanisms, same answer: **the ceiling is the
data, not the architecture.** The remaining honest levers are new signal (waveforms,
HRV, multi-site) and better decisions — see the operating-point result below.

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
