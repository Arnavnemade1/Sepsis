# GENESIS — how far hourly sepsis early warning can go on CinC 2019

**The brief.** The production detector sits at AUROC ≈ 0.77 (wearable features, within
site). The goal set for February is a *world-class* detector, taken to mean AUROC ≥ 0.90,
and the hypothesis was that the gradient-boosting infrastructure caps out around
0.80–0.85, so a genuinely new architecture — and possibly a new harness — was needed.

**What came back, in order of how much it matters:**

1. **On a locked test set, hourly AUROC 0.862 — up from 0.805 for the shipped
   architecture on the same patients (+0.057 [+0.046, +0.068]).** 8,067 patients were set
   aside before any design work and scored once. The gain comes from two things the old
   line never did: reading all 40 CinC variables (the whole detection line was built for a
   7-vital wearable) and regularising the booster properly.
2. **The new architecture did not survive the lock.** GENESIS — the Sepsis-3 label
   factorised into *who* × *when* × *direct* experts, plus a causal GRU with a Sepsis-3
   AND-gate head and a physiological world model — beat a tuned single model on the dev
   folds (+0.0034 [+0.0016, +0.0052]) and then did **not** beat it on the locked patients
   (−0.0012 [−0.0046, +0.0021]). Recorded as a failed replication; the simpler model is the
   recommended detector.
3. **About 0.05 of every CinC 2019 AUROC is a dataset artifact.** Never-septic records
   stop at ~60 h (99th percentile 59 h at hospital A) while septic records run to 336 h,
   so the ICU clock *alone* scores 0.636. Every published CinC result that uses `ICULOS`
   carries this. Every number here is paired with an **hour-matched AUROC**, under which
   the clock is worth exactly 0.500: the locked 0.862 is 0.816 hour-matched.
4. **0.90 is not reachable on this dataset with this label, and the reason is
   measurable.** The learning curve's asymptote — the AUROC this representation would
   approach with *unlimited* patients from these two hospitals — is **0.887 ± 0.014**
   (hour-matched **0.847 ± 0.011**). The half of Sepsis-3 that sets the onset clock
   (suspected infection = cultures + antibiotics) is not recorded in CinC 2019 at all (§9).

---

## 1. What "0.90" means in the literature

Three different tasks travel under the same name:

| task | typical AUROC | what it actually measures |
|---|---|---|
| **P19** (Raindrop, ViTST, …) | 0.870 / 0.892 / 0.900 | one label per *record*, scored from the whole record — which for a septic patient includes the hours after onset |
| row-split papers | 0.95–0.98 | hours of the same patient on both sides of the split — memorisation |
| **hourly early warning, patient-grouped** | ≈ 0.80–0.86 | the challenge label, every hour, unseen patients — this project's metric |

On the P19 definition a plain patient-level booster over GENESIS features scores
**0.964** (5-fold dev CV) — far above the published 0.900 — and we do **not** claim it:
most of it is record length (septic records are either very short, onset at admission,
or very long; never-septic records are 24–56 h), which the whole-record task leaks.

## 2. The data, measured

**All variables** (single LightGBM, identical folds, patient-grouped 5-fold CV over the
32,268 dev patients):

| features | hourly AUROC | hour-matched | between | within-septic | P19 |
|---|---|---|---|---|---|
| 7 wearable vitals + context | 0.8090 | 0.7471 | 0.8382 | 0.5275 | 0.8186 |
| all 40 variables (510 causal features) | 0.8385 | 0.7884 | 0.8689 | 0.5451 | 0.8552 |

**The decomposition.** Pooled hourly AUROC is *exactly* a weighted mean of two AUROCs,
because its negatives come in two kinds:

    pooled = 0.905 × between + 0.095 × within_septic

`between` — a septic patient near onset vs. an hour of a patient who never becomes
septic. `within_septic` — near onset vs. the not-yet-labelled hours of septic patients.
A model that knew *perfectly* which patients will become septic, and nothing about
timing, would score 0.953. The within-septic term is ≈ 0.55 for every model tested: septic
patients look sick for their whole stay (their hours 72+ h before onset already score at
the 83rd percentile of never-septic hours), so ranking the approach to onset above
their own earlier hours is barely possible.

**Why timing is so weak.** Mean trajectories of septic stays around clinical onset `T`:

| hours from T | HR | Temp | SOFA-proxy | labs / h | lactate drawn | new test types |
|---|---|---|---|---|---|---|
| −48 | 87.4 | 37.01 | 3.65 | 1.94 | 5.8% | 0.28 |
| −12 | 89.1 | 37.09 | 3.88 | 1.63 | 4.3% | 0.03 |
| −6 | 89.9 | 37.17 | 3.92 | 1.54 | 3.7% | 0.05 |
| −1 | 91.1 | 37.27 | 4.13 | 1.82 | 5.3% | 0.05 |
| **0** | 91.1 | 37.29 | 4.24 | **2.51** | **8.0%** | **0.10** |
| **+1** | 91.1 | 37.30 | 4.49 | **3.00** | **9.2%** | **0.15** |
| never-septic, 24–48 h | 84.3 | 36.88 | 2.55 | 1.08 | 1.3% | 0.05 |

Physiology drifts slowly; the sharp change arrives *at* onset and it is **clinician
behaviour** — the work-up that the (unrecorded) cultures and antibiotics belong to. Of the
ten positive hours of a septic stay, the first six look like the hours before them.

**The clock artifact.**

| rows scored | AUROC of the all-variable model |
|---|---|
| every hour (standard) | 0.8367 |
| first 60 h of every record only | 0.8032 |
| **hour-matched** (pairs compared only at the same ICU hour) | **0.7876** |
| the ICU clock alone, every hour | 0.6356 |
| the ICU clock alone, hour-matched | 0.5000 |

## 3. The architecture

**Representation.** `genesis/features.py`: 510 strictly causal features over all 34
time-varying channels and 6 static ones — last value, measurement age and counts, previous
value and delta, first value, expanding extremes, deviation from the stay's own mean,
6 h / 24 h rolling statistics and slopes for vitals, measurement-process features (labs
per hour, new test types ordered, blood gases), a partial SOFA and its rise, SIRS, qSOFA,
shock index. Missing stays missing. Vectorised over the whole cohort (6.6 s for 1.55M
rows); every column matches a slow pandas reference, and `verify_genesis.py` rewrites and
truncates every patient's future and asserts no earlier feature moves.

**Heads — the label, factorised along its own seams** (`genesis/targets.py`):

* **who** — every hour of a stay that ever turns septic (the patient label);
* **when** — the challenge label *inside septic stays only*, a balanced 1:5 contrast
  between the approach to onset and the same patients' earlier hours;
* **direct** — the challenge label.

`P(label) = P(septic stay) × P(near onset | septic stay)` exactly, so who × when is the
Bayes decomposition of the label; the direct head is the third expert. Each head is a
LightGBM whose margin is Platt-recalibrated on a patient-held-out slice of its *own
training* patients — without that, the five fold models emit margins on different scales
(fold means −4.66 … −5.23) and any learned combiner misreads the fold it was not fitted on.

**Combiner.** A product of experts, `log σ(who) + log σ(when) + log σ(direct)`, with no
fitted parameters. Every learned alternative was measured and lost to it (§4).

**The Genesis Network** (`genesis/network.py`) — the non-tree component. A causal GRU
over hourly inputs whose output layer has the shape of the Sepsis-3 definition: two latent
factors, suspicion and organ dysfunction, combined as an AND gate
`σ(a_susp) · σ(a_organ)`, each factor tied to its own observable traces (lab-ordering
bursts, lactate and blood-gas draws for suspicion; SOFA-proxy rise, hypotension, fever
for organ dysfunction). It also learns a **physiological world model**: at every hour,
the next value of all 34 channels and whether each will be measured in the next 6 h —
68 self-supervised targets per hour, the densest supervision in the dataset. Results in §4.

## 4. Results on the dev folds (32,268 patients, 5-fold, paired patient bootstrap)

**The fair baseline is a *tuned* single model.** The first all-variable LightGBM
early-stopped at ~160 trees — it overfits patients fast. Slower, more regularised boosting
(learning rate 0.02, 31 leaves, ≥800 rows per leaf, 30% feature sampling, L2 10) is worth
**+0.0065 [+0.0047, +0.0084]** on its own, and that correction matters for how big the
architecture's gain is: against the untuned model the factorised product looked like
+0.0067; against the tuned one it is +0.002. The table is against the tuned model.

| detector | AUROC | Δ vs tuned single [95% CI] | hour-matched | between | within |
|---|---|---|---|---|---|
| single LightGBM, 7 wearable vitals | 0.8090 | — | 0.7471 | 0.8382 | 0.5275 |
| single LightGBM, all variables, default | 0.8385 | −0.0065 | 0.7884 | 0.8689 | 0.5451 |
| **single LightGBM, all variables, tuned** | **0.8450** | — | 0.7964 | 0.8755 | 0.5508 |
| Genesis Network alone (3 seeds) | 0.8265 | −0.0185 | 0.7707 | 0.8573 | 0.5296 |
| tuned single + network | 0.8454 | +0.0004 [−0.0016, +0.0023] | 0.7971 | 0.8769 | 0.5419 |
| who × when × tuned direct | 0.8469 | +0.0019 [+0.0004, +0.0034] | 0.8005 | 0.8786 | 0.5411 |
| **GENESIS: who × when × tuned direct × network** | **0.8484** | **+0.0034 [+0.0016, +0.0052]** | **0.8018** | **0.8805** | 0.5384 |

The network helps only *beside* the factorised heads: added to the tuned single model it is
worth nothing, added to the product it is worth +0.0015. Its Sepsis-3 AND-gate head and
its plain direct head are statistically identical as experts (+0.0025 each over the
untuned product); the gate's value is interpretability, not accuracy.

**Measured and not adopted** (each against the same folds):

| idea | result |
|---|---|
| learned stage 2 — logistic regression / booster over head margins | +0.0039 … +0.0056 over the untuned direct head, always below the parameter-free product |
| temporal meta-learner (running max, 6 h change, deviation from own history of each head's risk) | +0.0000 over the same combiner without it |
| horizon heads (onset within 12 h / 24 h) | +0.0007, n.s. |
| label-genesis auxiliaries (future lactate / gas / new test / lab burst / fever / hypotension / SOFA rise) as heads | +0.0000 in any combiner |
| clinician-intent features (repeat lactate within 6 h, off-cycle STAT re-draws, ordering rate vs own baseline, hours since first abnormal marker) | +0.0019 alone (CI touches 0), +0.0002 inside the product |
| a hospital indicator | +0.0000 even as an in-sample upper bound (best per-site offset) |
| the network's own who × when × direct factorisation as an expert | +0.0000 |
| world-model + AND-gate network, single seed, uncalibrated | −0.0008 as an expert — the calibrated 3-seed version is what passed |

**A bug worth recording.** Platt recalibration was first written as raw Newton steps. On
boosters with ~500 trees the margins are extreme enough that the problem is nearly
separable, and Newton diverged to a slope of −1.3 × 10⁹ — flipping two folds of the
learning curve upside down (its 1/8-data point read 0.547 instead of 0.787). The solver
now line-searches every step and may only return an increasing map;
`verify_genesis.py` replays the failing case. Every saved head was re-audited: none had
diverged, so no other result moved.

**And one the audit caught.** Window sums were first taken as differences of ONE
cohort-wide cumulative sum. Exact in real arithmetic; in floating point, a patient's
window then carries the rounding of every earlier patient's values — including their
futures. The stage-2 causality check failed at the 1e-12 level. Running sums now restart
at every stay; the rebuilt 1.55M × 510 feature matrix is bit-identical in float32 to the
one every result was computed from, so no number moved.

## 5. The locked evaluation

The design was frozen in `artifacts/genesis/genesis_spec.json` (with the rule for adding
any expert pre-registered in `lock_ledger.jsonl` before the network was run), every model
was refitted once on all 32,268 dev patients, and each was scored once on the locked
patients. The shipped detector's exact architecture (`StackedEnsembleDetector`, patient
objective, hgb+hgb2+et+lr+xgb@utility, its own 94 wearable features) was refitted on the
same dev patients so it is compared on equal data. Generated by
`python -m backend.ml.genesis.report` from `artifacts/genesis/locked_results.json`:

Locked set: 8,067 patients, 309,253 hours, fingerprint `77e793853667b28e`, design `genesis-1.0` frozen 2026-09-24T19:54:27.

| detector | AUROC | hour-matched | between | within | AUPRC | util* | det@FA15 |
|---|---|---|---|---|---|---|---|
| shipped wearable stack (the ~0.77 detector), refit on dev, served output | 0.8051 | 0.7359 | 0.8386 | 0.4975 | 0.0968 | +0.424 | 0.522 |
| shipped wearable stack, pre-isotonic meta score | 0.8058 | 0.7371 | 0.8395 | 0.4962 | 0.1038 | +0.424 | 0.500 |
| single LightGBM, 7 wearable vitals | 0.8127 | 0.7401 | 0.8425 | 0.5392 | 0.1109 | +0.435 | 0.498 |
| single LightGBM, all variables, tuned | 0.8623 | 0.8164 | 0.8932 | 0.5792 | 0.1490 | +0.522 | 0.531 |
| Genesis Network alone (3 seeds) | 0.8415 | 0.7857 | 0.8736 | 0.5477 | 0.1300 | +0.478 | 0.476 |
| who × when × tuned direct (trees only) | 0.8595 | 0.8148 | 0.8916 | 0.5649 | 0.1503 | +0.527 | 0.546 |
| **GENESIS** (trees × network) | 0.8611 | 0.8160 | 0.8938 | 0.5607 | 0.1515 | +0.520 | 0.541 |

Paired patient bootstrap on the locked patients (2,000 draws):

* shipped wearable stack (the ~0.77 detector), refit on dev, served output vs shipped stack: ΔAUROC **-0.0007** [-0.0011, -0.0002]; Δbetween -0.0009, Δwithin +0.0013
* single LightGBM, 7 wearable vitals vs shipped stack: ΔAUROC **+0.0069** [-0.0029, +0.0164]; Δbetween +0.0029, Δwithin +0.0430
* single LightGBM, all variables, tuned vs shipped stack: ΔAUROC **+0.0565** [+0.0456, +0.0682]; Δbetween +0.0536, Δwithin +0.0830
* Genesis Network alone (3 seeds) vs shipped stack: ΔAUROC **+0.0357** [+0.0231, +0.0493]; Δbetween +0.0340, Δwithin +0.0515
* who × when × tuned direct (trees only) vs shipped stack: ΔAUROC **+0.0537** [+0.0424, +0.0652]; Δbetween +0.0520, Δwithin +0.0687
* **GENESIS** (trees × network) vs shipped stack: ΔAUROC **+0.0553** [+0.0443, +0.0674]; Δbetween +0.0543, Δwithin +0.0645
* **GENESIS** (trees × network) vs direct lgbm: ΔAUROC **-0.0012** [-0.0046, +0.0021]; Δbetween +0.0007, Δwithin -0.0186
* **GENESIS** (trees × network) vs genesis trees: ΔAUROC **+0.0016** [-0.0006, +0.0040]; Δbetween +0.0023, Δwithin -0.0042

P19-style patient task on the locked patients: AUROC 0.9731 (8067 patients, 586 septic); without the explicit record-length features 0.9625.

Every locked AUROC is higher than its dev-CV counterpart (0.862 vs 0.845 for the tuned
single model): the locked models trained on 32k patients instead of 26k, which the
learning curve prices at roughly +0.005, and the rest is the sampling variation of one
8,067-patient draw. The comparisons that matter are the paired ones on the same patients.

## 6. Self-supervised pretraining — a pre-registered null

An outside proposal ranked self-supervised pretraining as the most promising untested
idea, and it was the one with a mechanism: the learning curve (§8) says the model is partly
sample-limited, and pretraining is a sample-efficiency lever. It was tested on the laptop,
once, with the rules written into `lock_ledger.jsonl` before the run (`experiments/e12_ssl.py`,
40 CPU-minutes, no paid compute).

**Design.** The Genesis Network's encoder is first trained with no sepsis label on the
fold's training patients: hide each channel's values for blocks of 1–6 hours (13.6% of
values) and reconstruct the measured ones, plus the world-model forecasts. Hiding a channel
hides every composite computed from it — shock index = HR / SBP would otherwise hand a
hidden heart rate straight back — and `verify_genesis.py` checks that guard, including
that the check fires when the guard is switched off. Fine-tuning is then identical to the
from-scratch network (3 seeds, same schedule, same calibration); the from-scratch arm is
the existing run, not recomputed.

**The pretext was learned** in every fold: held-out reconstruction error 0.14–0.16 against
0.21–0.22 for carrying the last visible value forward and 0.32 for predicting the median.
So a null here is a statement about the idea, not about a broken run.

| dev CV, 32,268 patients | pretrained | from scratch | Δ AUROC [95% CI] |
|---|---|---|---|
| network alone | 0.8274 | 0.8265 | +0.0009 [−0.0019, +0.0036] |
| **GENESIS product (primary endpoint)** | **0.8492** | **0.8484** | **+0.0008 [−0.00002, +0.0017]** |

**Verdict: NULL.** The primary endpoint needed a lower bound above zero and missed by
0.00002; by the pre-registered rules the Transformer stage and a second locked look were
not run. Had it passed, +0.0008 would not have mattered for the 0.90 question anyway.

**What it did move (exploratory — the decomposition was not a named endpoint).** Pretraining
raised the *within-septic* AUROC — is this septic patient near onset yet? — by **+0.0119
[+0.0077, +0.0165]** for the network alone (+0.0043 [+0.0029, +0.0058] inside the product)
while leaving between-patient discrimination flat (−0.0002 [−0.0029, +0.0025]). That is the
decomposition's prediction playing out: learning physiological dynamics helps *when*, the
component every model is weakest at, but *when* is 9.5% of pooled AUROC, so the headline
cannot follow. It is the first lever in this project that moved within-stay timing, and the
natural place to use it is the question it answers — which hour to escalate — rather than
the pooled number.

## 7. The model has used up its inputs

Two experiments after pretraining asked whether *any* reorganisation of these inputs has
headroom left.

**Sufficiency (`experiments/e14_sufficiency.py`).** If a score equals P(label | features),
the features carry nothing further once the score is known. A second booster was given
the 510 features *plus* the tuned model's out-of-fold score on a random sample (all
positive hours + 10% of negative hours) and asked to beat the score alone. It could not:
**−0.0033 [−0.0047, −0.0020]**. Adding the features back on top of the score only adds
variance. Stacking, cascades, new learners and new architectures all draw on the same
information, and that information is spent.

A trap worth recording: the first version sampled cases *by the score* — low-scored septic
hours against high-scored never-septic hours — and a fresh model separated them with AUROC
0.886. That is Berkson's paradox, not signal: inside a set selected by the score, whatever
the model reads as risky is automatically associated with the other class, so the fresh
model scores well by learning the main model backwards. A sufficiency test must never
select on the score it tests.

**Physiological-twin features (`experiments/e13_twin_features.py`, pre-registered).** The
self-supervised twin (§6) forecasts every channel from the stay so far; its *surprise* —
the observed value minus the forecast made an hour earlier — is a residual a tree cannot
compute for itself. Handed to the tuned booster as 69 features (surprise for all 34
channels, its 6 h / 24 h averages, and 32 PCA dimensions of the twin's state), they carried
8–9% of the booster's split gain in every fold and moved AUROC by **+0.0007 [−0.0007,
+0.0022]** — NULL on the primary endpoint (+0.0008 [+0.0001, +0.0015] inside the product,
significant and immaterial). Used, but redundant: the surprise is still a function of the
same 40 variables, and the sufficiency result above says those are spent.

## 8. The ceiling, measured

Learning curve of the direct model (training patients subsampled per fold; test folds
unchanged; fit `AUROC(n) = a − b·n^−α`):

| training patients | AUROC | hour-matched |
|---|---|---|
| 3,227 | 0.7866 | 0.7176 |
| 6,454 | 0.8071 | 0.7481 |
| 12,907 | 0.8249 | 0.7701 |
| 25,814 | 0.8373 | 0.7885 |
| 2× data (projected) | 0.848 | 0.802 |
| 10× data (projected) | 0.865 | 0.822 |
| **asymptote** | **0.887 ± 0.014** | **0.847 ± 0.011** |

GENESIS's architectural gain (+0.0067) is worth roughly a doubling of the data. Even
unlimited patients from these two hospitals do not reach 0.90 for this representation.

## 9. What would actually reach 0.90

§7 settles the question the brief asked: no framework over these 40 variables has
meaningful headroom, because the tuned score is already sufficient for them. Only new
information can move the number, and CinC 2019 is missing two kinds of it:

* **Physiology that SOFA is defined on.** Three of SOFA's six organ systems cannot be scored
  from CinC: the central nervous system (Glasgow Coma Scale), the renal urine-output
  criterion, and the cardiovascular vasopressor criterion. The SOFA-proxy here is half a
  score. These are measurements of the patient, not of the clinicians.
* **The suspicion record.** The Sepsis-3 onset clock is the *earlier* of suspected
  infection and the SOFA rise, and suspected infection is defined by blood cultures and
  antibiotics — neither is in CinC 2019. The trajectories in §2 show the model sees that
  half only indirectly, through the lab orders that accompany it.

MIMIC-IV (and eICU-CRD) record both — credentialed PhysioNet access, typically a few weeks,
queryable in Google BigQuery so nothing large is stored locally.

That lever has a price worth stating before anyone pulls it. A model that reads antibiotic
and culture orders partly detects that *clinicians already suspect sepsis* — the central
published criticism of the Epic Sepsis Model. So the honest options are:

1. **Physiology-only inputs, now complete** — add GCS, urine output and vasopressors,
   report the hour-matched AUROC beside the standard one. This is the honest lever: on
   CinC this line is at 0.816 hour-matched (locked) against an asymptote of 0.847 for an
   input set missing half of SOFA.
2. **Add the suspicion record** (MIMIC-IV) and report *both* numbers — with and without
   order data — so the gain from reading clinicians' decisions is visible, not hidden.

A new architecture on CinC is not on the list: every architecture in this pass (factorised
experts, learned and temporal stage-2 combiners, a GRU with a Sepsis-3 AND-gate and a world
model, self-supervised pretraining) and ten before it land within ±0.01 of a well-tuned
booster on the same inputs, and the learning curve says the inputs, not the learner, set
the ceiling.

## 10. Reproduce

    .venv/bin/python -m backend.ml.genesis.rawstore          # pack 40,335 patients
    .venv/bin/python verify_genesis.py                        # the audit
    .venv/bin/python -m backend.ml.genesis.run_locked         # the one locked evaluation

The locked set (8,067 patients, fingerprint `77e793853667b28e`) was drawn before any
architecture work; every access is appended to `artifacts/genesis/lock_ledger.jsonl`,
including one disclosed look at a plain baseline made before the lock existed.
