# MIMIC-IV — the GENESIS findings, carried to where Sepsis-3 is fully recorded

**Why this exists.** On CinC 2019 the tuned booster's score turned out to be *sufficient*
for its 40 variables (docs/GENESIS.md §7): adding every feature back on top of the score
made it worse, so no framework over those inputs had headroom left. Only new information
could move the number, and CinC is missing two kinds of it — three of SOFA's six organ
systems (Glasgow Coma Scale, urine output, vasopressor doses) and the suspected-infection
record (cultures, antibiotics) that sets the Sepsis-3 clock. MIMIC-IV records all of it.

**Status.** Built end to end and verified on the open MIMIC-IV demo (100 patients, ODbL,
16 MB, `data/mimic_demo/`, gitignored). `verify_mimic.py`: 103/103 (~5 min), and every harness,
temporal-model, v3 and research-lever check has been mutation-tested (the mechanism broken on
purpose, the check seen to go red). The full
run waits on credentialed PhysioNet access; the demo's numbers are noise by design (99
subjects) and every demo report says so.

| module (`backend/ml/mimic/`) | job |
|---|---|
| `concepts.py` | the variable catalog: 135 itemids, every one checked against MIMIC's own `d_items` / `d_labitems`; plausibility windows; input tier per variable; the bedside exam's text codings |
| `history.py` | what was known on arrival: admission source, admitting service, comorbidities from prior admissions only |
| `mimicfeat.py` | the label's structure as causal features, each at the tier allowed to see it |
| `sources.py` | `CsvSource` (demo / full CSVs, the reference semantics) and `BigQuerySource` (the same hourly binning as SQL, run server-side) |
| `lake.py` | `LakeSource`: the same binning over a local Parquet lake, one bucket at a time ([PHYSIONET.md](PHYSIONET.md)) |
| `contracts.py` | is this the data the catalog thinks it is: repointed items, unit changes, orphans, a partial cohort |
| `sqlcheck.py` | the BigQuery SQL, transpiled and run on the demo without BigQuery; must match the reference exactly |
| `checkpoint.py` | every fit kept as it finishes, keyed by what it read, so a crash costs only the fit it hit |
| `extract.py` | events → the hourly grid, a `RawStore` the GENESIS feature engine runs on unchanged; subsetting and design sampling |
| `sepsis3.py` | hourly SOFA with all six organ systems, suspicion of infection, onset under three rules |
| `ledger.py` | the Ledger harness core: splits, tiers, masks, design weights; the `Candidate` interface and ledger *views* |
| `counterfactual.py` | the Care-Process Transplant and the admission audit — every fitted model re-scored on edited patients |
| `infoflow.py` | the information-flow audit — covert channels into treatment state; the strict tier |
| `crossexam.py` | the cross-examination — seven questions asked of every headline model |
| `trust.py` | the hash chain, power-gated pre-registration, confirmation, the Ladder lock |
| `tune.py` | budgeted hyperparameter search on dev subjects only |
| `plan.py` | will the full run fit this laptop — rows, disk, memory, hours — before anything is built |
| `parity.py` | BigQuery and the lake must each reproduce the CSV binning on the demo's 100 patients |
| `twin.py` | the temporal model: Sepsis-3 Process Twin v2 (two input streams, a physiology twin trained on counterfactual patients, discrete-time hazards, a definition-composed organ outlook) |
| `run.py` | the whole pipeline, one command, one JSON report |

## 1. Data: nothing large ever lands on this laptop

There are two routes now, both orchestrated by `backend/ml/physionet` ([PHYSIONET.md](PHYSIONET.md)):
the lake streams only the rows the catalog reads into local Parquet, and BigQuery bins on Google's
side. Both must reproduce the CSV path; on the demo they do, byte for byte.

`BigQuerySource` renders every extraction as SQL that bins events to hours on Google's
servers — last value / count per stay-hour for vitals, GCS and labs; hourly urine sums;
vasopressor rates expanded over the hours each infusion covers; culture and antibiotic
times. What comes down is hourly aggregates (hundreds of MB), not the tens of GB of raw
`chartevents`. The same binning in pandas (`CsvSource`) is the reference the SQL mirrors —
same itemids, conversions, windows and hour arithmetic — and `verify_mimic` checks the SQL
names exactly its concepts' itemids and keeps every conversion.

Traps found on the demo and handled before they could touch a label:

* **Intubated GCS.** "No Response-ETT" carries `valuenum` 1, identical to a truly
  unresponsive patient, and is the most common non-normal verbal score (1,348 rows). Summed
  naively, every ventilated patient reads as comatose and SOFA inflates. Recoded to 0 and
  total GCS set to 15 (unassessable), the MIT-LCP convention.
* **Units.** FiO2 is charted in percent (with invalid zeros); one "°C" temperature reads 99.0.
* **Antibiotics.** Eye drops, ointments, irrigants and enemas are filtered out by route and name.
* **Same-timestamp ties.** Readings can share a timestamp, and about 0.9% of chart cells have more
  than one candidate for "the hour's last reading". The rule is now fixed in both paths: lowest
  itemid (the catalog lists the invasive source first), then highest value. It was chosen before
  any AUROC was looked at, and it is a real definitional choice: on the demo, the opposite rule
  moves the primary from 0.790 to 0.733, and the old file-order accident gave 0.755. All three lie
  inside each other's intervals at 75 stays, and the movement says nothing about which is better.
  **Pre-register this rule**, and on the full run compare it with the opposite rule on the dev
  folds before the lock is read.

## 2. The label: Sepsis-3 from scratch, and a rule that does not fire on data arrival

SOFA is computed hourly with all six organ systems, each the worst value of the preceding
24 h (causal — the same score doubles as an input). Suspicion of infection is an antibiotic
and a culture close in time (culture first and antibiotic within 72 h, or antibiotic first
and culture within 24 h). Onset follows the CinC anchor: min(suspicion, SOFA-rise hour).

What counts as a SOFA *rise* turned out to matter more than anything else in the label. On
the demo, mean 24-h-worst SOFA climbs 1.8 → 3.2 → 5.3 over hours 0 / 3 / 24 — not because
patients deteriorate, but because labs arrive after admission (only 49% of stays have one
at hour 0) and a 24-h worst can only ratchet up. CinC's rule (a 2-point rise over the
window's running minimum) therefore fires on data arrival, and — worse — drags later,
genuine onsets back to the first hours of the stay. The primary rule here, ``measured``,
lets an organ component rise only relative to an earlier *measurement* of the same
component. Onsets within 4 h of admission, per rule: absolute 93%, CinC 79%, measured 69%
on the demo; the remainder is real sepsis-on-arrival (suspicion in the ED), which the
early-warning cohort excludes.

## 3. The Ledger harness — one mechanism per CinC lesson

| CinC lesson (docs/GENESIS.md) | Ledger mechanism |
|---|---|
| record length leaked the label | **symmetric censoring by construction** (one length rule for every stay) + tripwires: clock-only AUROC, onsets-at-admission share per rule |
| ~0.05 of any AUROC was the ICU clock | **hour-matched AUROC** on every result |
| pooled AUROC hides who vs when | **between / within-septic decomposition** on every result |
| a model reading orders detects suspicion (Epic) | **tier firewall** (P physiology · T treatment · S suspicion; a model asking above its tier raises) + **pre-suspicion AUROC** (only hours before the first order) |
| the score was sufficient for its inputs | **information ledger**: each variable family's AUROC when removed |
| the onset definition was a choice | **label sensitivity**: every model scored under 3 rules × 3 anchors |
| one dataset, one era | **temporal validation**: train on earlier `anchor_year_group`s, test on later |
| a 16 GB laptop | **design weights**: never-septic stays subsampled, every metric re-weighted to the full cohort; features built in chunks into a float16 map on disk (bit-identical to a whole build). A value past float16's 65,504 saturates there instead of becoming inf: on full MIMIC-IV that's ferritin, CK, NT-proBNP and MAP's 24 h variation, 7,559 of 2.9 billion cells |
| lucky dev folds and tuned-to-the-test numbers | **subject-grouped** folds and a **locked 20% of subjects**, every access chained |

A model enters the harness as a `Candidate`: a fit function the harness calls fold by fold,
which returns a predictor it can apply to any ledger-shaped **view** — the real patients or
edited ones (`Ledger.view(stays, X)` rebuilds features from an edited grid, recomputes the
derived GCS and SOFA channels, and inherits labels, masks, folds and weights). That single
interface is what lets every counterfactual test in §4 run against every model, trees and
Process Twin alike, and it is also how a candidate reaches the lock (§4.5).

## 4. What the harness does to a model

CinC taught the *mechanisms* above. Each layer below attacks a way a number can be high
for a reason nobody would deploy. Each one is verified by planting that exact failure
(`verify_mimic` §14-19) and mutation-tested: the mechanism is broken on purpose and the
check is seen to go red.

| question | layer | a failing model looks like | planted failure it must catch |
|---|---|---|---|
| Is the number even legitimate? | **admission audit** | a score before hour *t* moves when the future, the stay length or another patient changes | look-ahead lactate, a stay-length feature, test-set normalisation — each refused by the attack aimed at it |
| Does "physiology" carry treatment? | **information-flow audit** | a single physiology channel recovers ventilation / vasopressors | a vasopressor copy hidden in heart rate is flagged; noise is not |
| Is it reading the doctors or the patient? | **Care-Process Transplant** | AUROC collapses when the measurement schedule stops carrying the label | schedule-only world → attributed to process; value-only world → to physiology, exactly |
| Predicting, or re-deriving the label? | **definitional oracle** | no margin over the label's own formula at detection hours | a monotone copy of the formula earns exactly zero margin |
| Sepsis, or just sickness? | **contrast cohorts** | fails against infection-only or organ-dysfunction-only stays | a suspicion reader fails vs infection-only, a severity score vs organ-only |
| Anything the clinicians hadn't acted on? | **clinician-conditional AUROC** | ~0.5 among hours the team's actions made look equally suspicious | a copy of the clinician score scores exactly 0.5 |
| How early, really? | **clock-matched horizons** | no skill *h* hours ahead against patients at the same ICU hour | the ICU clock scores exactly 0.5 at every horizon |
| Would it survive another onset definition? | **soft-label AUROC** | good only on hours the 9 definitions disagree about | matches its brute-force definition; exact when definitions agree |
| For whom does it fail? | **subgroup audit** | one sex / age / race / insurance / unit / era group far below its complement | a group scored as noise is flagged after Holm correction |
| What would a ward pay? | **alert frontier** | few episodes caught at a tolerable alert rate | budgets respected; a perfect score catches every episode with no false alert |
| Is the locked number trustworthy? | **trust layer** | edited ledger, drifted code, underpowered or repeatedly-peeked lock | tampering, code drift, an unanswerable registration and the boosting attack are each caught |

### 4.1 The admission audit (`counterfactual.audit`)

Before any number about a model is believed, the model is attacked on edited patients:
every stay's future is rewritten after a random cut; every stay is truncated at its cut;
every *other* stay is scrambled. No score at or before the cut may move under the first
two, and no score of an untouched patient under the third. A model that fails is refused
by the ledger. The three attacks are complementary, and the verifier shows each one
catching a failure the others miss. For example, a stay-length feature survives the
future rewrite, because the length is unchanged, but not the truncation. The third attack
catches the class of bug GENESIS actually shipped: a cohort-wide cumulative sum whose
floating-point rounding coupled patients. On the demo all 17 models are admitted:
booster scores are bit-identical under every attack, and the Process Twin moves by at
most 1e-7 (float noise from a different sequence length, within tolerance).

### 4.2 The information-flow audit (`infoflow`)

The tier firewall used to trust declarations: a column computed from a physiology channel
is tier P. But declared information flows are not actual flows. This is the confinement
problem (Lampson 1973) and information-flow control (Denning 1976), imported into
clinical ML. For each physiology channel, a small booster reads only that channel's
columns and predicts each treatment state (ventilation, any vasopressor) out of fold over
subjects. A channel reaching AUROC ≥ 0.90 is a **covert channel**. It is flagged, and the
**strict tier** (`Ledger.columns("P", strict=True)`) drops every column computed from it,
including composites built on it. Suspicion (orders begun) is reported but never flagged,
because anticipating clinicians is what early warning is for.

What it found in our own pipeline, on the demo:

* **Verbal GCS → ventilation, AUROC 0.962.** The intubated recode ("No Response-ETT" → 0,
  §1) makes a verbal score of 0 a record of the breathing tube: 1,262 of 1,283 such hours
  are ventilated.
* **FiO2 → ventilation, AUROC 0.949.** FiO2 is a ventilator *setting* a clinician chose.
* **SOFA_P no longer leaks.** The first version of the physiology SOFA kept Sepsis-3's
  "3–4 respiratory points only while ventilated", so the "physiology" score carried the
  ventilator. It now scores respiration from P/F alone, and the audit no longer flags it.

Dropping the two covert channels moves demo tier-P AUROC 0.716 → 0.664 (noise at this n; the
full run answers it).

### 4.3 The Care-Process Transplant (`counterfactual.process_physiology`, `.learnable`)

An ICU record mixes the patient's physiology (*what* was measured) with the care process
(*when*, and how often, clinicians chose to measure it). In EHR data the timing of tests
can predict outcomes better than their results (Agniel, Kohane & Weber, *BMJ* 2018), and
missingness-aware models are built to exploit exactly that. The transplant separates the
two by editing patients rather than models. Every evaluated stay is rebuilt four ways:

| world | schedule | values |
|---|---|---|
| factual | its own | its own |
| physiology-only | a random donor's (label-independent) | its own: the real value where it was measured that hour, otherwise a **state-matched successor** |
| process-only | its own | neutral (the cohort median) |
| neither | the donor's | neutral |

The same fitted fold models then re-score each world, each stay scored by the fold that
never saw it, and derived channels are recomputed. This gives a two-factor Shapley split of
the model's discrimination above baseline, `φ_process + φ_physiology = A_factual − A_neither`,
and the headline **process reliance** = `A_factual − A_physiology-only`. That is the AUROC
the model would lose at a hospital whose clinicians order differently. `learnable` asks
the complementary question by *retraining* inside each world, on the same stays, folds and
training size: how much can any model learn from physiology alone? A model trained in
the physiology-only world is process-invariant by construction.

Three design decisions, each forced by a failure:

* **Donors are drawn uniformly, never matched on stay length.** Length carries the label
  (demo tripwire: AUROC 0.76), so a length-matched donor would hand septic patients sicker
  schedules.
* **State-matched successors, not carried-forward values.** A carried-forward value only
  changes when the patient is re-measured, so "how often does this value change"
  re-encodes the patient's own schedule inside the physiology-only world. Frozen models
  rarely read that code, but a retrained one did: a planted schedule-only world reached
  AUROC 0.731 "from values" through it. Each unmeasured donor hour now gets a value drawn
  from what real next measurements looked like, across the cohort, after a measurement in
  the same state as the patient's latest. That keeps the marginal distribution and the
  one-step dynamics, so the planted world falls to 0.542.
* **Urine moves as a rate**: the patient's causal 24-h rate times the donor's charting
  interval. Repeating the last chart would triple a patient's 24-h volume.
* **Successor pools exclude the patient's own fold, and every draw is keyed** by hashing
  (stay, hour, channel). A patient's counterfactual values are never drawn from their own
  future measurements, and the transplant is reproducible and patient-local. That locality
  is what lets a model use the transplant at inference and still pass the admission
  audit (§5).

On the demo (noise, 75 stays), the pattern the full run must test: the best tier-P model
among those transplanted (`genesis_twin_P`, AUROC 0.729) keeps **0.751 on its own schedule with
every value neutralised**, and falls to **0.594 on its own values under a random schedule**:
process reliance 0.136 (95% CI −0.060–0.340). The intervals clear of zero belong to the direct
booster (0.120, 0.030–0.217) and the strict-P booster (0.089, 0.025–0.179).

### 4.4 The cross-examination (`crossexam`)

1. **Definitional oracle.** Sepsis-3 is a formula over inputs the model reads. Every model
   is split into *prediction* (label-1 hours before the onset hour) and *detection* (onset
   hour to onset + 3 h), and set against the label's own formula computed from what that
   tier can see. At tier P that is the measured SOFA rise from physiology components; at
   tier S, plus suspicion. On the demo the tier-S model's detection margin is **−0.055**:
   at detection hours it knows nothing the definition doesn't.
2. **Contrast cohorts.** Septic hours are set against event-aligned negatives:
   infection-only stays around their suspicion time, organ-dysfunction-only stays around
   their measured SOFA rise, and stays with neither. The *specificity index* is the weaker
   of the two halves. On the demo the Process Twin, whose gate factors are trained on
   exactly these two halves, has the highest specificity index (0.673 and 0.739 for its full
   and physiology scores, vs 0.369 for the direct booster). That is a hypothesis for the full run, not a result.
3. **Clinician-conditional AUROC.** A clinician model is fitted on everything clinicians'
   actions reveal: orders, treatments, and when and how often each variable was measured,
   never a value. Every model is then scored only against hours that model found equally
   suspicious (stratified over its deciles). The pre-suspicion AUROC (Kamran et al.,
   *NEJM AI* 2024) is the special case "no order yet".
4. **Clock-matched horizons.** For each horizon *h*, the patients who will become septic *h*
   hours later are ranked against never-septic patients at the same ICU hour. This is the
   replacement for alarm lead time, which measured stay geometry (docs: the lead-time null).
5. **Soft-label AUROC.** Each hour's label is the share of the 9 onset definitions calling
   it positive. On the demo, 88% of positive hours are disputed by at least one definition.
6. **Subgroups** (sex, age band, race, insurance, admission type, care unit, era): each
   level against its complement, subject-bootstrap SE, Wald test, Holm-corrected. The
   attributes live in `meta` only and can never reach a model.
7. **Alert frontier.** At 0.25 / 0.5 / 1 / 2 alerts per patient-day with a 6-h snooze: the
   share of septic episodes alerted in [onset − 12 h, onset + 3 h], before onset, and the
   alert PPV.

`run.py` condenses all of this into one **verdict row per headline model**.

### 4.5 The trust layer (`trust`)

* **Hash chain.** Pre-registrations, lock accesses, confirmations, deviations and Ladder
  submissions are entries in one append-only JSONL chain (`artifacts/mimic/lock_ledger.jsonl`).
  Each entry commits to the previous one's SHA-256 and to a fingerprint of the harness code:
  every module that computes a number, plus the git commit. An edited, deleted, inserted or
  reordered entry is caught at the first bad entry, and a broken chain refuses to grow.
  Committing the chain to git anchors it in time, the idea Irving & Holden (*F1000Research*
  2016) proposed for trial protocols, applied inside a harness.
* **Power-gated pre-registration.** The harness estimates, on the dev folds, how precisely
  the lock can answer the claim: a subject-bootstrap SE scaled to the locked subjects. It
  refuses a comparative claim whose minimum important difference is below the lock's
  minimum detectable effect, and an absolute claim it would confirm with < 80% probability.
  On the demo lock (20 subjects) the MDE is ~0.18, so the demo cannot pre-register anything.
* **Confirmation** is spent once, released in full with its one-sided bound. If the harness
  code changed since registration, confirmation is refused unless the deviation, with its
  reason, is chained.
* **The Ladder lock** (Blum & Hardt, ICML 2015). Exploratory lock submissions are scored,
  but a number is released only when it beats the best released one by more than a step η
  (the lock's own SE), rounded to η, from a fixed budget of 8. The raw score is never
  written. In the verifier's boosting attack, an attacker submitting pure noise climbs a
  naive holdout to +0.107 AUROC over the truth, and the Ladder to +0.014.

## 5. The temporal model — Sepsis-3 Process Twin v2 (`twin.py`)

v1 made Sepsis-3's two halves *observed* targets. On CinC, the Genesis Network's
`sigmoid(a_susp) · sigmoid(a_organ)` gate was a prior over two latent factors; MIMIC-IV
records both halves. `a_susp` learns "suspicion has begun, or begins within 6 h" on every
stay. `a_organ` learns "SOFA has risen ≥ 2 above a measured baseline in [t − 48 h, t + 6 h]".
Their product is trained on the label, with dense supervision around it: the GENESIS
who / when / direct heads, suspicion within 24 h, per-organ rises within 12 h, and a world
model. v2 keeps all of that and rebuilds the rest around what the harness found:

| v2 | why | verified by (`verify_mimic` §7, §20) |
|---|---|---|
| **raw sequence, two streams**: per channel its latest value (physiology) and whether / how long ago it was measured (process), fused through a learned gate | a recurrent model should learn its own temporal summaries, and the raw grid is what lets it train on counterfactual patients built on the fly | rewrite the future: no past output of any head moves |
| **a step budget, not an epoch count** | v1's "8 epochs" on the demo were 24 optimiser steps (3 batches per epoch) | planted value world: a signal the booster finds at 0.85 went from 0.58 (v1) to 0.74 |
| **a physiology twin**, trained only on transplanted patients (fresh random donor schedules, state-matched successor values, every batch), scored averaged over 4 donors | is this alert the patient, or the team? | keeps a heart-rate signal: **0.740**, vs **0.605** for a record-trained twin scored the same way; in a schedule-only world, **0.523** hour-matched while the full score reads **0.815** |
| **per-alert process attribution**: full-score logit minus physiology-score logit, both calibrated | how much of *this* alert is the care process | passes the admission audit; the same model with a sequential random generator is refused by all four attacks |
| **discrete-time hazards** over 8 bins out to 24 h, censoring-aware | a risk curve, F(t, H), for the clock-matched horizon lens; every at-risk hour supervises, and a discharge never counts as survival it didn't observe | targets: event / survival / censoring / not-at-risk on planted stays; the per-bin loss equals the survival likelihood (brute force, exact); F monotone in H |
| **definition-composed organ outlook**: per organ, the measured SOFA rise 6 h ahead (0 / 1 / ≥ 2); P(total ≥ 2) composed exactly | every alert in the definition's own language | equals brute-force enumeration of all 729 organ outcomes |
| **temporal capacity** | a sequence model should read what fixed windows cannot | a 30-h lag world: **0.925** vs the booster's **0.781** (hour-matched) |

Every v2 check was mutation-tested (8 deliberate breakages, 8 caught). Examples:

* a twin without memory fails the lag world;
* a "physiology" twin that secretly trains on records fails the physiology check;
* hazard targets that count a discharge as survival fail the survival-semantics check;
* an organ composition missing its one-organ term fails the enumeration.

**Two regimes, not one network doing both — measured.** The first version trained one
network on recorded and transplanted patients at once. On a planted world where the care
process carries real signal, that cost the full score AUROC **0.903 → 0.763** (d = 128).
A learned "this record is transplanted" flag did not win it back (0.763 vs 0.752), so the flag
was removed. The twin is now fitted twice per fold:
* on records as they were (`regime="record"`), which gives the full score, hazard, outlook
  and explanation;
* on transplanted records only (`regime="transplanted"`), which gives the physiology score.

**What one alert looks like** (`twin.explain`; the report carries the riskiest pre-onset
hour of the dev folds). For example: Sepsis-3 probability; its physiology-only counterpart
and the process attribution between them; suspicion within 6 h; each organ's chance of a
measured SOFA rise within 6 h; the composed organ half; onset within 6 h. On the demo the
example is an ICU-hour-3 alert 9 h before onset: 0.997 overall and 0.868 from physiology
alone, with cardiovascular and respiration the organs expected to move. At 99 subjects that
is an illustration of the readout, not a finding.

**Accuracy is not measurable on the demo.** Every twin configuration scores 0.55–0.63 on
the 99-subject demo, and early stopping on ~8 validation patients mostly keeps the first
checkpoint. The same code learns every planted world built on those same demo schedules
and labels, so this is sample size, not a broken real-data path; boosters also win on tiny
tabular data. The twin is deliberately **not** tuned on the demo, because that is exactly
the overfitting the harness exists to catch. Full MIMIC decides: twin vs trees (the
`genesis_twin_P` product), the physiology twin's process reliance against the booster's,
and the hazard curve against the clock-matched horizons, all under the §4 layers.

**Cost on full MIMIC** (≈ 20k training stays per fold after design sampling, one fold at a
time): the record twin at d = 128 should run ≈ 35 ms per step (extrapolated from ≈ 17 ms at
d = 64 on the demo), and the physiology twin ≈ 2× with its per-batch transplant. With 2 + 1 seeds and early stopping, expect ≈ 3 h for the twins
out of a full run of several hours. Run it overnight.

## 6. Getting the number as high as it honestly goes

On CinC the ceiling was the data: the tuned booster's score was sufficient for its 40
variables. So the full run is built to extract every legitimate bit of information MIMIC-IV
holds, fit it as well as a laptop allows, and report the result in every setting a paper
might use — with what each one means printed beside it.

**More information (store v4, 155 channels, 1,244 features; §7 has the evidence for each).**
* **Labs:** inflammation and the differential (CRP, neutrophils, lymphocytes, immature
  granulocytes, eosinophils, absolute counts, anion gap, ferritin, LDH, CK, lipase, ammonia,
  NT-proBNP), plus urinalysis coded from text (leukocytes, nitrite: a urinary source).
* **Chart:** CVP, and the ventilator settings at tier T.
* **Treatment (tier T):** crystalloids and blood products by volume, sedation, dialysis.
* **Suspicion (tier S):** IV antibiotics as administered at the bedside.
* **Known on arrival:** admission source, admitting service, and comorbidities from
  admissions *discharged before* this one (Charlson groups in the Quan 2005 ICD-9/10 codings,
  prior sepsis, readmission recency).
* **The bedside exam** (tier P): capillary refill, mottling, skin temperature and moisture,
  peripheral pulses, lung sounds by lobe, cough, urine appearance, abdominal exam, delirium
  screen, RASS, pupils. The worst finding of each hour is kept.
* **Within-hour extremes** (tier P): each hour's lowest MAP, SBP and SpO2, and highest HR,
  respiratory rate and temperature. SOFA now reads the hour's lowest MAP.
* **Infection-source detail** (tier S): culture specimen groups, and anti-MRSA / broad
  anti-pseudomonal antibiotic spectrum.
* All 135 item IDs are verified against MIMIC's own dictionaries. D-dimer is left out on
  purpose: its item IDs are charted in two unit systems.

**The label's structure as features** (`mimicfeat.py`, each at the tier allowed to see it):
* the per-organ SOFA points and measured rise that Sepsis-3's organ half is computed from,
  from physiology alone at tier P and with vasopressors and ventilation at tier T;
* the level-2 path signature of seven physiology streams over 6 and 24 h: each stream's
  quadratic variation, and each pair's Lévy area (which stream moved first);
* whether a culture and an antibiotic have already paired into suspected infection (tier S);
* urine per kg per hour, the neutrophil–lymphocyte ratio, norepinephrine-equivalent dose.

The definitional oracle (§4.4) still reports how much of any AUROC is only reading the
formula.

**Fitting it well.**
* A budgeted hyperparameter search on dev subjects only, with the incumbent as trial 0.
  Tuning was CinC's largest single gain.
* A soft-label booster trained on all 9 onset definitions at once.
* A **distilled physiology student**. Its training target is half the label and half an
  orders-aware teacher's score. Every teacher that wrote its targets was held out of the fold
  the student is scored on (§7 has why that matters).
* A **prior-centred stacker** per tier. It starts at the GENESIS product and learns only
  shrunken corrections, with the strength chosen per fold by nested cross-validation.
  Two free stackers were measured first and lost to the product on the demo (0.654, 0.628
  vs 0.739).
* The Process Twin v2. It reads the dynamics: the arrival context is the trees' job. With
  context included, the planted-world physiology twin fell 0.757 → 0.609.

**The headline** (`report["headline"]`). The best model of each tier is chosen by the
primary dev AUROC and reported under every setting:
* early-warning hourly (the pre-registered primary);
* clock-matched;
* every stay, sepsis on arrival included;
* patient level (a stay's highest score);
* patient level, every stay.

The single highest number is printed as `IDEAL CIRCUMSTANCES`, always with its tier and
setting spelled out and with the primary beside it. On the demo that line reads 0.869:
the distilled physiology-only student, hourly, over every stay, beside a primary of 0.790.
That is noise at 99 subjects, but it shows exactly which knobs carry a number toward 0.9.

**What to expect on full MIMIC-IV** (expectations, not results):
* **Physiology-only, early-warning, hourly** (the honest number): somewhere around
  0.80–0.87. CinC's tuned booster reached 0.862 on its lock, with a learning-curve ceiling
  of 0.887, and MIMIC adds the three organ systems CinC lacked.
* **All information, every stay, hourly:** plausibly around 0.9. It reads the orders that
  make up half of the label and includes sepsis present on arrival; the harness will say
  how much of it is the formula and the clinicians.

Which of these to put on a poster is a pre-registration decision, made before the lock is
read (§8).

**Fitting a laptop.**
* `plan.py` estimates rows, disk, memory and hours from the extracted store before any
  feature is built. It picks the richest plan that fits, or refuses and names what doesn't
  fit.
* `trim_septic` drops septic stays' hours after their latest onset across all 9
  definitions (+3 h). No model trains on those hours and no metric scores them, and because
  everything is causal no kept feature changes (verified bit for bit).
* Full-MIMIC-sized, the plan comes to ≈ 2.4 M rows, 6.1 GB of features on disk, 7.4 GB peak
  RAM and about 9.3 h: 73 booster fits at ~5 min each (7.0 h, tuning included) and the twin
  (1.6 h). The per-fit constant is measured with the full preset itself — learning rate
  0.02, on 995k real CinC rows x 510 columns, 102.6 s and 452 trees. The first estimate
  (~7 h) was calibrated on an untuned model that stopped at ~160 trees; the RAM figure now
  also counts LightGBM's bin-boundary sample, which the first one left out.
* **Ingestion is squeezed, exactly.** Every fit bins its data first: one vectorised read of
  LightGBM's own 200k-row bin sample instead of 200,000 single-row calls, C-contiguous float32
  batches instead of float64 copies it had to flatten again. 13.5 s -> 6.6 s at 360k x 800,
  and the model text is byte-identical (`verify_genesis` §10). Then a tier's direct, who,
  soft and distilled heads are fitted fold by fold and share ONE binned dataset per fold —
  only their labels differ — so 25 of the 73 fits skip ingestion entirely. The whole demo
  report is identical to the one produced before either change, to the last digit.
* BigQuery results are placed into the grid page by page, never held whole.

**Not fitting a laptop (the `cloud` scale).** A dedicated machine ([deploy/physionet](../deploy/physionet/README.md))
starts the planner from every never-septic stay (keep rate 1.0) instead of 0.25. Both scales tune 8 trials: on the first full run 32 trials moved the dev AUROC from 0.8131 to 0.8163, every setting landed within 0.004 of the others, and the first 8 already held 70% of that gain.
When time runs short, it gives up negatives (1.0 → 0.5 → 0.25) before it drops a stage. The laptop scale walks
the other way, and `verify_mimic` checks both orders. Booster and twin threads follow `GENESIS_THREADS`, which
is one per physical core there and 6 on the laptop. These are dev-fold choices made before the lock, like the
rest of this section.

**Parity first (`parity.py`).** The demo is 100 real MIMIC-IV patients. The parity check fetches
every binned table for exactly those stays, through BigQuery or the lake, and compares it cell by
cell with the CSV path. Neither is trusted except where it reproduces the reference. `sqlcheck.py`
runs the same comparison on the SQL offline, and requires exact agreement: the 0.99 bar had been
hiding a 0.93% disagreement in how same-timestamp ties were broken.

## 7. The research map — every lever, with its evidence

What moves a sepsis AUROC, from this project's own measurements on CinC (docs/GENESIS.md,
same folds, locked test set) and the literature. Each lever has a status. The CinC column is
decisive where it exists: the tuned score there was sufficient for its inputs, so only new
information moves a number, and most of the model-side ideas below were measured at zero.

| lever | evidence | status here |
|---|---|---|
| **new information** | CinC: 7 vitals → all 40 variables, +0.03; the only lever that moved the ceiling | store v3 / v4 (below) |
| **the bedside exam** — perfusion (capillary refill, mottling, cool clammy skin, weak pulses), infection source (lung sounds, productive cough, cloudy urine, acute abdomen), encephalopathy (delirium screen, RASS, pupils) | clinical: mottling and capillary refill predict septic-shock mortality (mottling score > 2: 6-fold, capillary refill > 4 s: 2-fold day-28 mortality in a [prehospital septic-shock cohort](https://www.sciencedirect.com/science/article/abs/pii/S0735675718305679)); charted by nurses as TEXT, so almost no ML model reads it | **new here**: 12 ordinal channels from 27 items, worst reading of the hour; 1,000–2,800 assessed hours per finding in 99 demo patients |
| **within-hour extremes** | demo: 12.5% of MAP-hours hold several readings; 205 of 2,169 hypotensive hours (9.5%) dip and recover before the hour's last reading; mimic-code's SOFA uses the hour's lowest MAP | **new here**: 6 extreme channels; SOFA (label and inputs) reads `MAP_min` |
| **infection-source detail** | suspicion is a class of orders: blood vs urine vs respiratory cultures, anti-MRSA vs anti-pseudomonal cover; routine MRSA screens are not a workup | **new here**: tier-S specimen and spectrum channels (oral vancomycin, a C. difficile drug, is not counted as anti-MRSA) |
| **path signatures** | CinC 2019 winner ([Morrill et al.](https://physionet.org/files/challenge-2019/1.0.0/papers/CinC2019-014.pdf)): order-3 lead-lag signatures + LightGBM, +0.012 utility over hand-crafted features | **implemented** exactly at level 2 (quadratic variation; Lévy areas, i.e. which stream moved first) via stay-local prefix sums. Planted lag world: windowed trees 0.800 → 0.813 with signatures (the twin reads 0.899) |
| **privileged-tier distillation** | learning using privileged information as generalised distillation (Lopez-Paz et al., 2016; [Karlsson et al.](https://arxiv.org/pdf/2110.14993) for time-series privileged information) | **new here**. A physiology-only student learns from an orders-aware teacher's scores, through training targets only. Ordinary cross-fitting LEAKED: +0.048 of apparent demo gain vanished with nested teachers (0.859 → 0.811, label-only 0.770). Pairwise-held-out teachers (10, not 20) remove it |
| **definition choice** | Sepsis-3 onset variants move AUROC 0–6%, more than model choice (1–5%) ([Sci Rep 2024](https://www.nature.com/articles/s41598-024-51989-6)) | pre-registered primary rule; soft-label booster; 9-definition sensitivity on every model |
| **clinical notes** (radiology reports) | multimodal notes + measurements: +2.9% AUROC on MIMIC-III ([Improving Early Sepsis Prediction with Multi Modal Learning](https://arxiv.org/pdf/2107.11094)) | **not built**: needs the separately credentialed MIMIC-IV-Note module, and there is no open demo to verify an extractor against. The strongest remaining lever |
| **external data** (eICU, 200+ hospitals) | multi-centre training and transfer ([YAIB](https://arxiv.org/pdf/2306.05109)) | not built: a second credentialed dataset with another schema |
| hyperparameter tuning | CinC: +0.0065, the largest model-side gain | built; dev only, 2 validation folds at full scale. On the 99-subject demo it overfits one fold: noise |
| factorised product (who × when × direct) | CinC: +0.0019, +0.0034 with the network | built |
| learned stage-2 stacking | CinC: always below the product; demo: free stackers lost to it too | prior-centred stacker (starts AT the product) |
| deep sequence models | YAIB: GRU the strongest deep baseline for sepsis; CinC: the network was worth +0.0015 inside the product | Process Twin v2, in the stack |
| temporal meta-learning (score smoothing, running max) | CinC: +0.0000 | not repeated |
| horizon / auxiliary "future lab" heads | CinC: +0.0007 / +0.0000 as tree experts | only as dense supervision inside the twin |
| clinician-intent features | CinC: +0.0019 alone, +0.0002 inside the product | superseded by the process features and the clinician-action model |
| self-supervised pretraining | CinC: pre-registered null | not repeated |
| hospital indicator | CinC: +0.0000 | not used |
| case-control / onset-aligned evaluation | where 0.96–0.99 "AUROCs" in recent papers come from | refused by construction (symmetric censoring, clock-matched horizons) |

## 8. Running it on full MIMIC-IV

Credentialing and each dataset's Data Use Agreement come first. After that the path is one
command, documented in [PHYSIONET.md](PHYSIONET.md):

1. `make physionet-login`, then `make physionet-doctor`, `make physionet-access` and `make physionet-plan`.
2. `make physionet-up` (or `ROUTE=bigquery ARGS="--project <project>"`). It verifies, ingests, checks
   the contract and parity, plans, runs every stage on the dev folds only, and scans the report before
   releasing it. Resumable. The lock is never read.
3. Read the verdicts and the headline, and pick the claims.
4. **Before any locked number exists**, pre-register through `trust.preregister`. The harness
   refuses a claim the lock cannot answer. Registration fixes the primary rule (`measured`),
   the cohort (onset ≥ 4 h), the metric and setting, the comparator and the bar, bound to the
   code fingerprint. Commit the chain.
5. `trust.lock_scores` + `trust.confirm` for each registered claim, spent once. Anything
   exploratory goes through `trust.Ladder` (8 submissions).

Steps 3 to 5 are a person's, and the pipeline has no shortcut to them.

## 9. Reproduce the demo

    python -m backend.ml.mimic.extract          # build the demo store (under a second)
    python -m backend.ml.mimic.run --source demo  # every stage, ~3.5 min
    python verify_mimic.py                        # 103 adversarial checks, ~5 min
