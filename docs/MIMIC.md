# MIMIC-IV — the GENESIS findings, carried to where Sepsis-3 is fully recorded

**Why this exists.** On CinC 2019 the tuned booster's score turned out to be *sufficient*
for its 40 variables (docs/GENESIS.md §7): adding every feature back on top of the score
made it worse, so no framework over those inputs had headroom left. Only new information
could move the number, and CinC is missing two kinds of it — three of SOFA's six organ
systems (Glasgow Coma Scale, urine output, vasopressor doses) and the suspected-infection
record (cultures, antibiotics) that sets the Sepsis-3 clock. MIMIC-IV records all of it.

**Status.** Built end to end and verified on the open MIMIC-IV demo (100 patients, ODbL,
16 MB, `data/mimic_demo/`, gitignored). `verify_mimic.py`: 33/33. The full run waits on
credentialed PhysioNet access; the demo's numbers are noise by design (99 subjects) and every
demo report says so.

| module (`backend/ml/mimic/`) | job |
|---|---|
| `concepts.py` | the variable catalog: 75 itemids, every one checked against MIMIC's own `d_items` / `d_labitems`; plausibility windows; input tier per variable |
| `sources.py` | `CsvSource` (demo / full CSVs, the reference semantics) and `BigQuerySource` (the same hourly binning as SQL, run server-side) |
| `extract.py` | events → the hourly grid, a `RawStore` the GENESIS feature engine runs on unchanged; subsetting and design sampling |
| `sepsis3.py` | hourly SOFA with all six organ systems, suspicion of infection, onset under three rules |
| `ledger.py` | the Ledger harness |
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

## 3. The Ledger harness — one innovation per CinC lesson

| CinC lesson (docs/GENESIS.md) | Ledger mechanism |
|---|---|
| record length leaked the label | **symmetric censoring by construction** (one length rule for every stay) + tripwires: clock-only AUROC, onsets-at-admission share per rule |
| ~0.05 of any AUROC was the ICU clock | **hour-matched AUROC** on every result |
| pooled AUROC hides who vs when | **between / within-septic decomposition** on every result |
| a model reading orders detects suspicion (Epic) | **tier firewall** (P physiology · T treatment · S suspicion; a model asking above its tier raises) + **pre-suspicion AUROC** (only hours before the first order) |
| the score was sufficient for its inputs | **information ledger**: each variable family's AUROC when removed; sufficiency test |
| the onset definition was a choice | **label sensitivity**: every model scored under 3 rules × 3 anchors |
| one dataset, one era | **temporal validation**: train on earlier `anchor_year_group`s, test on later |
| a 16 GB laptop | **design weights**: never-septic stays subsampled, every metric re-weighted to the full cohort; features built in chunks into a float16 map on disk (bit-identical to a whole build) |
| lucky dev folds and tuned-to-the-test numbers | **subject-grouped** folds and a **locked 20% of subjects**, every access ledgered |

## 4. The Sepsis-3 Process Twin

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

## 5. Running it on full MIMIC-IV

1. PhysioNet: complete CITI "Data or Specimens Only Research", request credentialed access,
   sign the MIMIC-IV data use agreement.
2. Link a Google Cloud project on the MIMIC-IV page (BigQuery access); the BigQuery sandbox
   is free (1 TB of queries / month; this extraction scans well under that).
3. `pip install google-cloud-bigquery db-dtypes` and `gcloud auth application-default login`.
4. **Before looking at any full-data number:** write the pre-registration into
   `artifacts/mimic/lock_ledger.jsonl` — primary rule `measured`, primary cohort (onset ≥ 4 h),
   primary metric (tier-P AUROC and its hour-matched twin), and the bar each component must
   clear.
5. `python -m backend.ml.mimic.run --source bigquery --project <project> --scale full`

## 6. Reproduce the demo

    python -m backend.ml.mimic.extract          # build the demo store (under a second)
    python -m backend.ml.mimic.run --source demo  # every stage, ~25 s
    python verify_mimic.py                        # 33 adversarial checks
