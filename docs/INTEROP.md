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

## Results: the registered run (2026-10-04)

`backend/ml/scripts/interop_bench.py` as committed in `9c75f1f`, against the gate as
committed in `082a142`, on the registered holdout (fingerprint `bbc2a255253ae549`). One
run, 43 minutes; `artifacts/interop/interop_bench.json` and `interop_bench.log`.

Clean detector on these patients: AUROC 0.8463 (hour-matched 0.7986). At its deployed
threshold it alarms on 14.1% of never-septic patients and 68.0% of septic patients at
least once.

| Fault (from row 2,000 on) | Silent ΔAUROC [95% CI] | Silent Δ false alarms | Caught after (faulty values) | Faulty values delivered | Gate→quarantine | Gate→fix | Variable missing |
|:--|--:|--:|--:|--:|--:|--:|--:|
| temperature °F read as °C | **−0.0463** [−0.0564, −0.0367] | **+35.0 pt** | 4 | 0 | −0.0123 | **−0.0000** | −0.0123 |
| white cells /µL read as 10³/µL | **−0.0122** [−0.0166, −0.0075] | **+13.1 pt** | 4 | 0 | −0.0017 | **+0.0001** | −0.0017 |
| SBP ↔ MAP swapped | −0.0092 [−0.0133, −0.0056] | −3.0 pt | 5 | 11 | −0.0172 | (no fix) | −0.0173 |
| HR ↔ respiratory rate swapped | −0.0066 [−0.0107, −0.0025] | +4.4 pt | 5 | 4 | −0.0073 | (no fix) | −0.0064 |
| creatinine µmol/L read as mg/dL | −0.0056 [−0.0096, −0.0014] | **+8.2 pt** | 4 | 0 | −0.0000 | +0.0000 | −0.0000 |
| platelets /µL read as 10³/µL | −0.0026 [−0.0062, +0.0009] | +4.3 pt | 4 | 0 | −0.0009 | +0.0000 | −0.0009 |
| PaCO₂ ↔ HCO₃ swapped | −0.0025 [−0.0042, −0.0005] | +0.4 pt | 4 | **6,821** | −0.0006 | (no fix) | −0.0020 |
| potassium decimal slip (×10) | −0.0009 [−0.0020, +0.0001] | −0.4 pt | 4 | 0 | −0.0012 | −0.0000 | −0.0012 |
| urea mmol/L read as BUN mg/dL | −0.0005 [−0.0016, +0.0005] | −0.8 pt | **4,378** | **4,374** | −0.0004 | (no fix) | −0.0001 |
| troponin ng/L read as ng/mL | −0.0004 [−0.0011, +0.0003] | +0.0 pt | never adopted | 0 | −0.0000 | — | −0.0000 |
| calcium mmol/L read as mg/dL | −0.0003 [−0.0010, +0.0003] | −0.7 pt | 4 | 0 | −0.0010 | +0.0000 | −0.0010 |
| glucose mmol/L read as mg/dL | −0.0001 [−0.0017, +0.0015] | +2.7 pt | 4 | 0 | −0.0007 | +0.0000 | −0.0007 |
| lactate mg/dL read as mmol/L | −0.0001 [−0.0011, +0.0008] | −1.2 pt | 12 | 10 | −0.0026 | −0.0000 | −0.0026 |
| FiO₂ % read as fraction | −0.0001 [−0.0004, +0.0002] | +0.1 pt | never adopted | 0 | −0.0141 | — | −0.0141 |
| Hgb ↔ Hct swapped | −0.0001 [−0.0014, +0.0011] | −0.2 pt | 4 | 10 | −0.0003 | (no fix) | −0.0002 |
| haemoglobin g/L read as g/dL | +0.0001 [−0.0008, +0.0009] | −0.3 pt | 4 | 0 | −0.0000 | −0.0000 | −0.0000 |
| phosphate mmol/L read as mg/dL | +0.0001 [−0.0005, +0.0006] | −0.5 pt | 33 | 33 | +0.0002 | +0.0000 | +0.0002 |
| bilirubin µmol/L read as mg/dL | +0.0007 [−0.0005, +0.0020] | −0.1 pt | 12 | 10 | −0.0005 | +0.0000 | −0.0005 |
| magnesium mmol/L read as mg/dL | −0.0000 [−0.0008, +0.0008] | +0.5 pt | 43 | 43 | +0.0003 | +0.0000 | +0.0003 |

### Verdicts against the registered bars

| | Bar | Result | |
|:--|:--|:--|:--|
| **H1 harm** | ≥ 3 harmful faults | temperature, white cells, creatinine | **met** |
| **H2 detection** | ≥ 15 of 19 quarantined; median ≤ 40 | 17 of 19; median 4 faulty values | **met** |
| **H3 recovery** | quarantine within 0.005 of missing; fix within 0.002 of clean | all three harmful faults: quarantine = missing exactly; fix +0.00004, +0.00006, −0.00002 | **met** |
| **H4 false quarantines** | ≤ 1 per 100k observations on each stream | clean 1.23 · hospital A 2.71 · hospital B 0.20 · septic ward **0.00** | **missed** |
| **H5 onboarding** | ≥ 80% correct auto-adopted; ≤ 1% unsafe | 93.8% adopted (121/129); wrong-unit 3.4% unsafe (2/58); wrong-concept 6.9% (162/2,359) | **missed** |

Descriptive (no bar): on 43 local code names in three dialects, the lexical rules named
31 correctly (all 24 US mnemonics) and abstained on the rest, with no wrong concepts.
The language model (`gemini-3.5-flash`) named **43 of 43** correctly, including every
Nordic `P-Kreatinin`-style name and vendor string, and declined the four analytes
outside the vocabulary. The model does the semantic work, and the gate checks the
scale.

### What the numbers say

1. **Silent interface faults are clinically material for this detector, mostly as
   alarm burden.** A Fahrenheit temperature triples the share of never-septic patients
   who are alarmed; a white count in cells/µL nearly doubles it. The ranking loss is
   smaller than the alarm loss: a tree model ranks patients on many variables at once,
   but its absolute threshold is crossed far more often.
2. **The gate catches unit errors before they deliver anything.** In 12 of the 14 unit
   faults the first impossible value is withheld and the feed is quarantined after 4
   values, with none delivered. The suggested fix restores the detector to clean
   (within 0.0001).
3. **Withholding has a price of its own.** For lactate, withholding costs 0.0026 AUROC
   against 0.0001 for the silent fault, because the detector reads *when* lactate is
   measured (a proxy for clinical suspicion) as well as its value, and withholding
   removes both. For swaps the gate suggests no fix, and withholding SBP and MAP
   (−0.017) costs more than the swap (−0.009). Quarantine-then-fix is the design point;
   quarantine-and-forget is not free.
4. **Three honest failures of detection.** (a) Urea in mmol/L read as BUN mg/dL is a
   ×0.36 shift that stays plausible: caught only after 4,374 delivered values, as the
   pre-registration predicted it might not be caught at all. (b) The PaCO₂↔HCO₃ swap was
   caught by the Van Slyke identity, which quarantined base excess, HCO₃ and pH but not
   PaCO₂. With HCO₃ withheld, Henderson–Hasselbalch could no longer be evaluated, and
   PaCO₂, now carrying bicarbonate values, delivered 6,821 wrong values. A partial
   quarantine can blind the identity that would have caught the rest. (c) FiO₂ and
   troponin were never adopted, so there was nothing to quarantine: CinC's FiO₂ column
   mixes percentages (which the ICU detector's own feature builder silently repairs),
   and its troponin populations differ ×20 between the two hospitals. Refusing FiO₂
   costs the detector 0.014 AUROC.
5. **H4 missed.** Of the false quarantines the artifact lists, most fall in the last 0.4%
   of the stream. The registered stream orders patients by hour since admission, so its
   tail is a few very-long-stay patients; a 12-message window then comes from one or two
   patients and stops being independent. The rest are genuine: two bilirubin "jumps"
   against a baseline that 12 early values made unrepresentative, and BP-order bursts at
   hospital A, whose clean violation rate (1.7%) sits close to the identity's 2%
   allowance. The stream of patients getting sicker had **zero** false quarantines in
   116,474 observations, which is the property the feed-relative design was built for.
6. **H5 missed, for two reasons, one of them a defect.** Magnesium written in mEq/L
   (×1.22, inside magnesium's band) is adopted: a small-factor unit confusion
   single-variable evidence cannot see. The 162 wrong-concept adoptions are almost all
   one defect, found by this benchmark's own smoke run before the registered run and
   deliberately not fixed before it: a written unit that cannot be the proposed concept
   (heart-rate values written "/min", proposed as mean arterial pressure) was treated
   as no unit at all, so numerically fitting values were adopted. The 28 blind pairs
   were adopted 87% of the time, as expected.
7. **Pulse oximetry was never auto-adopted** at hospital A, B or mixed. At onboarding,
   the SpO₂/SaO₂ identity fails against hospital A's venous-contaminated SaO₂ (and
   hospital B's SaO₂ is itself refused), so SpO₂ waits for a person. This is safe, and
   operationally costly.

Post-hoc work, labelled exploratory, follows below. It is measured on the same patients
and is not a confirmation.
