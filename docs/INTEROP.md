# Verified interoperability: interface mappings earned on evidence

AegisOS could watch a hospital only if the hospital could feed it. A hospital's
admissions, monitors and laboratories speak HL7 v2, each in its own dialect: local
codes (`CREA`, `B-Natr`, `147842^MDC_ECG_HEART_RATE`), local units (µmol/L, mmol/L,
g/L, °F), local habits. Turning those feeds into the variables a detector was trained
on is called interface mapping. In practice an integration analyst builds a table by
hand over weeks, and the hospital trusts it from then on.

That trust is the hazard. A mapping is a claim about data ("this code is creatinine,
in µmol/L"), and nothing checks it after go-live. Laboratories swap analysers and
units change silently. Gateways get reconfigured and fields swap. A detector at AUROC
0.85 reading creatinine 88 times too high is not a detector at 0.85.

The integration plane (`backend/aegis/interop`) treats every mapping like every other
claim in AegisOS: it is checked against the data before anything acts on it, and it
keeps being checked.

## What is built

| Piece | What it does |
|:--|:--|
| `hl7.py` | HL7 v2 codec: ER7 parse and build, separators taken from MSH-2, escapes, CR/LF/CRLF, ACK/NAK carrying the table-0357 error code and its location, MLLP de-framing |
| `mllp.py` | The TCP listener an on-premises interface engine connects to (`AEGIS_MLLP_PORT`). The hosted deployment uses `POST /aegis/interop/hl7` instead, which runs the same code |
| `engine.py` | The inbound interface engine. ADT builds the census in the hospital's rooms. ORU results pass through per-code mappings. ORU^R40 device alarms become pages (IHE PCD-04 subset), one per room while one is open. A resent control id is answered with the same ACK and not processed again |
| `vocab.py` | 39 canonical concepts, each with its LOINC code, canonical unit, the analyte-specific conversions (glucose ×18.016, creatinine ÷88.42, valence-aware mEq/L), a physiological window fixed a priori, and a reference distribution |
| `evidence.py` | The checks (below) |
| `proposer.py` | Four proposers: LOINC, IEEE 11073, lexical, and a language model. **They only propose.** |
| `sender.py` | The demo hospital's systems, simulated: Epic ADT, a Philips IntelliBridge gateway (°F, MDC codes), a laboratory with SI units, and a Nordic-coded reference laboratory. They speak only HL7 to the engine |

## How a mapping is earned

1. A new local code collects values; nothing is delivered yet.
2. At 12 values the proposers suggest what it is. LOINC and IEEE 11073 codes name their
   concept outright; the lexical rules match mnemonics and display text; codes no
   rule recognises go to the language model with the vocabulary in front of it.
3. **The evidence engine decides.** For each proposal it reads the values in the unit
   the feed wrote and in every unit the analyte is reported in:
   - *plausibility*: values outside the physiological window (a creatinine of 132 mg/dL
     is impossible);
   - *location*: the median against the reference median. Refusing needs the whole
     99.9% interval outside the population band; adopting needs the median inside it,
     with plausibility and tail mass passing. The asymmetry is deliberate: a small
     sample can never refuse a correct mapping;
   - *tail mass*: catches a mixture that barely moves the median;
   - *identities across fields* (SBP > MAP > DBP; Hct/Hgb between 2.3 and 3.8;
     Henderson–Hasselbalch; SpO₂ against SaO₂; direct ≤ total bilirubin; BUN/creatinine;
     ANC ≤ WBC; the anion gap; Van Slyke). These catch swaps whose marginal ranges
     overlap, from a handful of samples.
4. Policy `evidence` adopts a proposal automatically **only when the data agrees in the
   unit the feed wrote**. An inferred unit, an ambiguous match, or a refusal waits for
   a person (`interop:configure`), with the evidence beside the decision. A person may
   adopt against the evidence, and the record says so.
5. Results learned before adoption are backfilled into the room; vitals are not
   (a monitor wants current vitals).

## How an adopted mapping is watched

Every 4 values, over the last 12 and the last 40:

- values a body cannot produce (never delivered, even before any quarantine);
- a jump of unit-error size against **the feed's own baseline at adoption**: twice the
  population band, and at least a doubling on a log scale;
- the identities over recent co-observed messages (a withheld partner value is left
  out, so it cannot convict its neighbour);
- for cuff pressures, 8 identical consecutive readings (a frozen gateway).

A failure quarantines the mapping. New values are withheld and the room says so.
Values delivered since the change are withdrawn: per value for a unit fault, from the
onset of the violations for a swap. The evidence names the fix when exactly one
reading fits. The Sentinel puts it on the operator's console.

**Why the jump test is feed-relative.** The first design compared live values with the
global reference. A ward whose patients all turn septic shifts its heart-rate median
from 84 to 125. That is outside the population band, so the monitor would cut the
heart-rate feed exactly when it matters. Judged against its own baseline with a
unit-error tolerance, the same ward keeps its feed. A switch to mg/dL (×9 for lactate)
is still caught. Both halves are checked as a pair in `verify_interop`.

## What the checks found in the benchmark data itself

The thresholds are fixed from physiology and device specifications, never fitted to
data, so a dataset that is itself wrong cannot widen them. Pointed at PhysioNet/CinC
2019 (the shipped detector's 24,206 training patients, replicated on its 8,062-patient
holdout), the same checks find two real interface faults (`artifacts/interop/cinc_audit.json`):

| Column | Hospital | Finding | Training | Holdout |
|:--|:--|:--|--:|--:|
| `Calcium` | B (Emory) | values below the total-calcium window, median 1.17: **ionized calcium in mmol/L recorded under total calcium (mg/dL)**. 3,969 of 18,449 patients; 99% of them also have mg/dL values | 19.1% | 18.1% |
| `SaO2` | A (Beth Israel) | SpO₂ and SaO₂ in the same hour differ by more than 15 points (pulse oximeters are specified to Arms ≤ 4%): **venous saturations recorded as arterial** | 18.0% of pairs | 18.1% of pairs |

Every other identity violates at its clean rate at both hospitals (BP order 1.7%/0.2%,
Hct/Hgb 0.07%/0.5%, Henderson–Hasselbalch 0.7%/0.4%). The feature builder of the
ICU detector already carries a hand-written fix for a third fault in the same data
(FiO₂ recorded as a percentage). The ICU detector was trained on these columns as
they are.

## What single-variable evidence cannot see

Evidence verifies scale and gross identity, not meaning. Computed from the reference
alone, 28 directed concept confusions fall inside the target's population band. They
include creatinine↔bilirubin (both near 1 mg/dL), AST↔ALP, DBP↔MAP, SpO₂↔SaO₂,
EtCO₂↔PaCO₂, total↔direct bilirubin, and BUN→lactate once the unit conversion is
applied. Their defences are the proposer (the names) and the identities (direct ≤
total bilirubin, SBP > MAP > DBP, SpO₂ ≈ SaO₂). Ambiguity is named, not resolved by
guessing: bilirubin in µmol/L fits both ÷17.1 (µmol/L) and ÷10 (mg/L, a real unit in
some countries), so it waits for a person.

## Prior work, and what is new here

Distribution features for automated LOINC mapping are not new (Parr et al., JAMIA
2018, on VA data). Neither is language-model mapping (Tu et al., MLHC 2022; LabBridge,
JMIR 2026), nor plausibility checking (Kahn et al.'s harmonised data-quality framework;
the OHDSI Data Quality Dashboard). What this adds:

1. **An adoption gate.** No proposer, human or model, adopts on its own say-so; the data
   must agree in the written unit, and everything else waits for a person.
2. **Physiological identities as the oracle for joint mappings.** They catch field
   swaps that marginal distributions cannot.
3. **The same evidence running live, against the feed's own baseline.** A quarantine
   happens before a wrong value reaches a detector, and the feed is not cut when
   patients get sicker.
4. **A measurement**, registered below before it ran: what silent interface faults
   cost the shipped detector on patients it never saw, and how much of that loss the
   gate recovers.

## Pre-registration (registered before the run)

Protocol: `artifacts/interop/interop_prereg.json`, committed before
`backend/ml/scripts/interop_bench.py` was run.

- **Data.** The shipped detectors' promotion holdout: 8,062 patients, fingerprint
  `bbc2a255253ae549`, never trained on. The GENESIS locked test set is not touched.
  Gate thresholds are as committed in `082a142`; nothing is tuned on these patients.
- **Detector.** `icu:lgbm-all40~platt#7fbbdd0ad322`, at its deployed threshold 0.045.
- **Stream.** The holdout replayed through the deployed `Integration` engine as HL7
  ORU^R01, one message per patient-hour, patients interleaved by hour since admission.
  Mappings are learned on the stream itself. For a fault, rows 0–1,999 are clean and
  the fault affects every later observation.
- **Faults (19).** Unit errors for creatinine, glucose, urea/BUN, lactate, bilirubin,
  haemoglobin, platelets, white cells, temperature, FiO₂, calcium, magnesium, phosphate
  and troponin; a potassium decimal slip; and field swaps SBP↔MAP, Hgb↔Hct,
  PaCO₂↔HCO₃ and HR↔respiratory rate.

| | Bar |
|:--|:--|
| **H1 harm** | ≥ 3 faults with ΔAUROC < −0.01 (paired bootstrap CI excluding 0), or a false-alarm rate moved by more than 5 points at the deployed threshold |
| **H2 detection** | ≥ 15 of 19 quarantined; median faulty observations to quarantine ≤ 40. **Predicted blind spot:** urea in mmol/L read as BUN mg/dL (×0.36, inside BUN's window, its jump tolerance and the BUN/creatinine identity) |
| **H3 recovery** | For each harmful detected fault: gate→quarantine within 0.005 AUROC of "variable missing"; gate→fix within 0.002 of clean where a fix was suggested |
| **H4 false quarantines** | ≤ 1 per 100,000 observations on the clean holdout, on each hospital onboarded alone, and on a ward fed only septic patients from 24 h before onset. Hospital B calcium and hospital A SaO₂ are reported separately (audited as contaminated) |
| **H5 onboarding** | Correct mappings (canonical or alternative unit, written correctly) ≥ 80% auto-adopted. Wrong unit label ≤ 1% unsafely adopted. Wrong concept ≤ 1% unsafely adopted over the non-blind pairs; the 28 blind pairs are reported separately |

Results are appended below once, as they come out, including the bars that fail.
