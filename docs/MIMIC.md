# MIMIC-IV — the GENESIS findings, carried to where Sepsis-3 is fully recorded

**Why this exists.** On CinC 2019 the tuned booster's score turned out to be *sufficient*
for its 40 variables (docs/GENESIS.md §7): adding every feature back on top of the score
made it worse, so no framework over those inputs had headroom left. Only new information
could move the number, and CinC is missing two kinds of it — three of SOFA's six organ
systems (Glasgow Coma Scale, urine output, vasopressor doses) and the suspected-infection
record (cultures, antibiotics) that sets the Sepsis-3 clock. MIMIC-IV records all of it.

**Status.** Built end to end and verified on the open MIMIC-IV demo (100 patients, ODbL,
16 MB, `data/mimic_demo/`, gitignored). `verify_mimic.py`: 71/71, and every harness check has
been mutation-tested (the mechanism broken on purpose, the check seen to go red). The full
run waits on credentialed PhysioNet access; the demo's numbers are noise by design (99
subjects) and every demo report says so.

| module (`backend/ml/mimic/`) | job |
|---|---|
| `concepts.py` | the variable catalog: 75 itemids, every one checked against MIMIC's own `d_items` / `d_labitems`; plausibility windows; input tier per variable |
| `sources.py` | `CsvSource` (demo / full CSVs, the reference semantics) and `BigQuerySource` (the same hourly binning as SQL, run server-side) |
| `extract.py` | events → the hourly grid, a `RawStore` the GENESIS feature engine runs on unchanged; subsetting and design sampling |
| `sepsis3.py` | hourly SOFA with all six organ systems, suspicion of infection, onset under three rules |
| `ledger.py` | the Ledger harness core: splits, tiers, masks, design weights; the `Candidate` interface and ledger *views* |
| `counterfactual.py` | the Care-Process Transplant and the admission audit — every fitted model re-scored on edited patients |
| `infoflow.py` | the information-flow audit — covert channels into treatment state; the strict tier |
| `crossexam.py` | the cross-examination — seven questions asked of every headline model |
| `trust.py` | the hash chain, power-gated pre-registration, confirmation, the Ladder lock |
| `twin.py` | the Sepsis-3 Process Twin |
| `run.py` | the whole pipeline, one command, one JSON report |

## 1. Data: nothing large ever lands on this laptop

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
| a 16 GB laptop | **design weights**: never-septic stays subsampled, every metric re-weighted to the full cohort; features built in chunks into a float16 map on disk (bit-identical to a whole build) |
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
floating-point rounding coupled patients. On the demo all 11 models are admitted:
booster scores are bit-identical under every attack, and the Process Twin moves by at
most 2e-7 (float noise from a different sequence length, within tolerance).

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

Dropping the two covert channels moves demo tier-P AUROC 0.690 → 0.627 (noise at this n; the
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

On the demo (noise, 75 stays), the pattern the full run must test: the best tier-P model
(`genesis_twin_P`, AUROC 0.768) keeps **0.738 on its own schedule with every value
neutralised**, and falls to **0.599 on its own values under a random schedule**. That is
process reliance 0.169 (95% CI 0.027–0.309), the only interval clear of zero. The strict-P
booster's reliance is 0.010.

### 4.4 The cross-examination (`crossexam`)

1. **Definitional oracle.** Sepsis-3 is a formula over inputs the model reads. Every model
   is split into *prediction* (label-1 hours before the onset hour) and *detection* (onset
   hour to onset + 3 h), and set against the label's own formula computed from what that
   tier can see. At tier P that is the measured SOFA rise from physiology components; at
   tier S, plus suspicion. On the demo the tier-S model's detection margin is **−0.026**:
   at detection hours it knows nothing the definition doesn't.
2. **Contrast cohorts.** Septic hours are set against event-aligned negatives:
   infection-only stays around their suspicion time, organ-dysfunction-only stays around
   their measured SOFA rise, and stays with neither. The *specificity index* is the weaker
   of the two halves. On the demo the Process Twin, whose gate factors are trained on
   exactly these two halves, has the highest specificity index (0.704 vs 0.414 for the
   direct booster). That is a hypothesis for the full run, not a result.
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

## 5. The Sepsis-3 Process Twin

On CinC, the Genesis Network's `sigmoid(a_susp) · sigmoid(a_organ)` gate was a structural
prior over two unobserved halves. In MIMIC both halves are observed for every patient, so
each factor gets its own ground truth: `a_susp` learns "suspicion has begun, or begins within
6 h" (for every stay — suspicion without organ failure is common and is what the factor must
separate), `a_organ` learns "SOFA has risen ≥ 2 above a measured baseline in [t − 48 h,
t + 6 h]", and their product is trained on the label. Around it: per-organ rises within
12 h (6 heads), suspicion within 24 h, the GENESIS who / when / direct heads, and a world
model of every input — dense supervision on every hour. Inputs are **tier P**: the
suspicion record is a target, never an input, so the model learns to anticipate clinicians'
suspicion from physiology rather than reading their orders.

## 6. Running it on full MIMIC-IV

1. PhysioNet: complete CITI "Data or Specimens Only Research", request credentialed access,
   sign the MIMIC-IV data use agreement.
2. Link a Google Cloud project on the MIMIC-IV page (BigQuery access); the BigQuery sandbox
   is free (1 TB of queries / month; this extraction scans well under that).
3. `pip install google-cloud-bigquery db-dtypes` and `gcloud auth application-default login`.
4. `python -m backend.ml.mimic.run --source bigquery --project <project> --scale full`: every
   stage on the dev folds only (the lock is never read). Read the verdicts, pick the claims.
5. **Before any locked number exists**, pre-register through `trust.preregister`. The
   harness refuses a claim the lock cannot answer. Registration fixes the primary rule
   (`measured`), the cohort (onset ≥ 4 h), the metric, the comparator and the bar, bound
   to the code fingerprint. Commit the chain.
6. `trust.lock_scores` + `trust.confirm` for each registered claim, spent once. Anything
   exploratory goes through `trust.Ladder` (8 submissions).

## 7. Reproduce the demo

    python -m backend.ml.mimic.extract          # build the demo store (under a second)
    python -m backend.ml.mimic.run --source demo  # every stage, ~45 s
    python verify_mimic.py                        # 71 adversarial checks
