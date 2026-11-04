# Clinical text: the infrastructure for reading MIMIC-IV-Note

**Status:** infrastructure only. It's built and verified on invented notes (`make verify-notes`, 61 checks, 14 of them mutations). No real note has been read, because this account has no MIMIC-IV-Note access yet. Nothing here is wired into the modelling run. There is no result, and no claim.

## Why text, and what it can't do

On CinC and in the MIMIC-IV harness, every model family hit the same ceiling: the trees were already reading everything the numbers carry. Text is the one source that carries information the trees can't see. But MIMIC-IV-Note 2.2 has only two kinds of note:

| Note | Written | Used as |
|---|---|---|
| Radiology reports | when the read is stored, during the stay | hourly features, at the hour of `storetime` |
| Discharge summaries | at discharge | history only, from admissions that ended before the stay |

It has no nursing or physician progress notes, which is where "concern for sepsis" is usually written. So expect a modest gain, not a jump. A plausible guess is +0.01 to +0.03 in the suspicion tier. That is a guess and should be read as one.

## Design

```
radiology report (at storetime) ─┐
                                 ├─► reader ─► finding scores ─► (stay_id, hr, name, value)  tier S
prior discharge summaries ───────┘                         └──► (stay_id, name, value)      static, tier S
```

**Timing** (`notes/timing.py`) holds every availability rule:
- A read lands at `floor(storetime − intime)`, never at `charttime`, because the film was taken before anyone read it.
- A report with no `storetime` is dropped, not guessed.
- A read stored up to 24 h before ICU admission, such as the ED film, is known at hour 0. Nothing past the stay's end is used.
- A discharge summary counts only if its admission ended, and the summary was written, before the stay began. So the current admission's summary can never be read.

**Readers** (`notes/readers.py`) score the same findings on the same scale: 1 asserted, 0.5 hedged, 0 negated or absent.
- **Findings:** pneumonia, effusion, empyema, abscess, free air, colitis, cholecystitis, urinary source and soft tissue. Edema and atelectasis are also scored, as alternative explanations for the same picture.
- **Indication:** the clinician's question ("fever, eval for pneumonia") is scored apart, as `ind_infection` and `ind_respiratory`. A question is suspicion, not a finding.
- **`KeywordReader`:** a lexicon with sentence-scoped negation before or after the term, and hedging, in the spirit of NegEx.
- **`EncoderReader`:** a local text encoder compares each sentence with a positive and a negated prototype of each finding. Its `floor` and `scale` are untuned defaults, to be fitted on dev folds only.

The encoder is kept only if it beats the keywords on the same dev folds. Each reader carries a fingerprint, so cached scores and checkpoints are never mixed across readers or versions.

**Features** (`notes/features.py`): each hour gets each finding's highest score among the reads that arrived in it, plus `Note_rad_n`, the number of reads. Carrying a value forward is the grid builder's job, as for a lab. History gets each finding's highest score over prior summaries, plus `NoteHist_n`.

**Tier S, deliberately.** Ordering a film is already a clinician's suspicion, and a note can describe the orders the Sepsis-3 label is built from. So note features never enter the physiology-only primary. The existing audits (information flow, the definitional oracle, the cross-examination) say how much of any gain is the label read in text form.

## Privacy, as code

- Text never leaves the machine. PhysioNet forbids sending notes to third-party model APIs. `local_encoder` loads a model from disk with `local_files_only`: a missing model is an error, never a download, and nothing is sent anywhere.
- Readers return numbers. `features` refuses any non-numeric column, and any feature name it didn't define, in case text is smuggled in as a name.
- `CachedReader` keys scores by `sha256(text)` under a directory in the lake, and refuses one inside the repository. Neither file names nor files hold text.
- The note tables ingest through the same hash-verified lake as MIMIC-IV (`notes.sources.table_specs`), bucketed by subject.

## What `verify_notes` checks

1. **Readers:** assertion, negation before and after the term, never across sentences, hedging, and the indication kept apart from the findings.
2. **Encoder:** prototype scoring on a deterministic stand-in embedder. One check sets the floor to 0, so only the negated prototype can hold a negation at 0. Without it, the floor alone was passing that test (a vacuous check, caught by its mutation).
3. **Timing:** every trap planted in `notes/synthetic.py`.
4. **Features:** the rows those traps produce, and **prefix invariance**: an hour's rows are identical whether or not any later note exists.
5. **Privacy:** as above.
6. **Sources:** the release layout round-trips, and the lake ingests the note tables with every report's text intact.
7. **Mutations:** 14 guards are broken one at a time, and a check is seen to go red each time.

## To use it on real notes

1. Sign the MIMIC-IV-Note agreement on PhysioNet. `make physionet-access` shows when it's open.
2. Ingest `note/radiology` and `note/discharge` through the lake, using `table_specs()`.
3. Score with `KeywordReader` first. For the encoder: put a clinical encoder's weights on the machine, install `backend/ml/notes/requirements.txt`, and use `EncoderReader(local_encoder(path))`. Both run through `CachedReader` into the lake.
4. Join the rows into the grid as tier S channels. This is the separate, audited step not built yet: it needs `extract.DYNAMIC`, the ledger's tiers and the information-flow audit to learn the new channels.
5. Compare on dev folds: the tier-S booster with and without notes, keyword versus encoder. Pre-register before the lock.
