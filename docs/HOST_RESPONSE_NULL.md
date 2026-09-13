# Host-response expression does not clear its pre-registered transfer bar — a null

> **Scored once, 2026-09-12**, against pre-registration `b79eacd703210ad8`. The rules,
> the code and the discovery record were committed (`a1ef44b`) *before* the locked
> cohort was loaded, so version control rather than a claim shows the bar predates the
> result. Nothing is published to the site.

**Date:** 2026-09-12 · **Cohorts:** GSE74224 (GPL5175, discovery) and GSE40012
(GPL6947, locked) — public GEO, no credentialing · **Verdict:** **BAR MISSED** under
the rules in force; **CONFOUNDED** under the frozen rules. `published: false`.

## What was claimed in advance

`backend/hostresponse/` asked one question: does a blood expression signature separate
**infectious** sepsis from **sterile** systemic inflammation, and does that separation
survive a change of cohort *and* array platform? That is Link A only. Whether it would
reduce this product's false-alarm rate (Link B) was never testable — no bed carries
transcriptomics — and is not addressed here.

The bar in force (amendment 4) required **all** of: mean per-direction
leave-one-cohort-out AUROC ≥ 0.75; 95% patient-cluster lower bound > 0.70; **each
direction ≥ 0.70 on its own**; and a label-permutation null centred in [0.42, 0.58]
over at least 20 draws.

Power was stated before the run, from cluster counts alone: the lower bound clears 0.70
in 6/20 simulated replicates at a true transfer AUROC of 0.80, 18/20 at 0.85 and 20/20
at 0.90. So a miss was pre-committed to read *"transfer at or above ~0.85 not
detected"*, never *"no transfer"*.

## What happened

145 patients (259 samples) on a 13,785 gene-symbol intersection. The sterile arm is 44
patients; in the locked cohort it is 13.

| quantity | result | condition | |
|---|---|---|---|
| held-out GSE40012 (trained on GSE74224) | **0.773** | each direction ≥ 0.70 | pass |
| held-out GSE74224 (trained on GSE40012) | **0.682** | each direction ≥ 0.70 | **miss** |
| mean of the two directions | **0.727** | ≥ 0.75 | **miss** |
| 95% patient-cluster interval | **0.636 – 0.810** | lower bound > 0.70 | **miss** |
| label-permutation null, 20 draws | 0.495 (sd 0.050) | mean in [0.42, 0.58] | pass |
| cohort identity | **1.000** | diagnostic; frozen rule voided at ≥ 0.90 | — |
| easiness calibrator: infectious vs *healthy*, in-cohort | 1.000 | never an endpoint | — |

## Reading it honestly

1. **It is not chance.** The permutation null sits at 0.495 and the interval excludes
   0.5 by a wide margin, so some cross-platform separation between infectious and
   sterile patients is present. That is not the question that was pre-registered, and
   it does not rescue it.

2. **The interval is informative in both directions.** Its upper bound, 0.810, sits
   *below* the ~0.85 this study was powered to detect. The data do not merely fail to
   show strong transfer; a true transfer AUROC of 0.85 lies outside the interval. The
   most defensible summary is modest transfer, somewhere around 0.64–0.81.

3. **Transfer is asymmetric** — 0.773 one way, 0.682 the other — and the per-direction
   floor exists precisely to catch that. One specific explanation fits, and this design
   cannot test it: the two sterile arms are *different phenotypes*. GSE74224's are
   post-surgical; GSE40012's are non-infective medical SIRS. A model trained on medical
   SIRS transferring worse to post-surgical patients is what you would expect if
   "sterile inflammation" is not one thing. That is a hypothesis, not a finding.

4. **The easiness gap, quantified.** Infection against *healthy* volunteers separates
   perfectly (1.000) inside a single cohort. Infection against *sterile inflammation*,
   across cohorts, reaches 0.727. That gap is the whole reason healthy controls were
   refused at the gate: a cohort with healthy controls would have produced a
   spectacular number answering a question nobody asked.

## The confound, and the amendment that relabelled this result

Cohort identity came in at **1.000**: a model can tell which platform a sample came
from perfectly. Under the frozen rule that alone would have reported this study
**CONFOUNDED**.

Amendment 4 retired that rule *before* identity was computed, and its committed text
predicted this outcome. The argument: the primary is a mean of per-direction AUROCs,
each computed inside one held-out cohort, where cohort identity is constant and cannot
rank patients. A cohort offset cannot move that number — verified, not assumed: a +3.0
offset left it identical while a pooled AUROC on the same scores fell from 1.000 to
0.750. The frozen rule could therefore only fire on platform differences. It was
replaced by something stricter, the per-direction floor, and that stricter rule is what
this result actually failed.

**The amendment changed the label of the failure, not the outcome.** Both rule sets
leave the result unpublished. Had a mid-study judgment call turned an unpublishable
result into a publishable one, it would deserve far more suspicion than one that turned
CONFOUNDED into BAR MISSED. The artifact prints both verdicts side by side, so that
decision is visible in the result itself.

## What this does NOT establish

- **Not** that host-response expression cannot separate infection from sterile
  inflammation — the interval excludes chance.
- **Not** a comparison with SeptiCyte RAPID or IMX-BVN/TriVerity, which report
  0.83–0.91 in their own validation cohorts. Different cohorts, platforms and case
  definitions; beating or missing those numbers here would not be like-for-like.
- **Nothing** about this product's false-alarm rate.
- **Not** a closed question: two cohorts, one technology, 44 sterile patients.

## What would change the answer

- **A third admissible cohort, ideally RNA-seq.** GSE63042 has exactly the right design
  (106 septic / 23 SIRS) but GEO holds no usable expression table for it — only an
  Excel sheet of relative RPM. Re-admitting it would be a *new* pre-registration.
- **A larger sterile arm.** The locked cohort's is 13 patients; that, not the feature
  space, bounds every interval here.
- **Cohorts whose sterile arms share a phenotype**, which would test reading 3
  directly.

## Traps found on the way

Each of these would have produced a confident wrong answer, and each is now a guard in
`make verify-hostresponse` (28/28):

- **A sample is not a patient.** GSE40012 is 190 samples from 58 people at days
  1/2/3/5. Resampling samples would have narrowed the headline interval; in the
  suite's planted case, row-level resampling gives a width of 0.150 against 0.344 for
  patient clustering.
- **Two thin nulls raised false alarms.** A single permutation read 0.626; five
  permutations that reused one fold split read 0.592. With 20 draws and fresh splits
  the null centres at 0.476–0.495.
- **`GBMDetector` is degenerate at p ≫ n.** At 13,785 features and 52 training rows it
  emits one constant prediction, which is why an early mechanics check read exactly
  0.500.
- **Label fields named by guesswork.** A fixed list of key names picked "patient id",
  "tissue" and "gender", and would have excluded three labelled cohorts;
  `POST_SURGICAL` fell through on an underscore. Cohorts now name their own label field
  by how well it partitions.
- **A false fact about GEO, nearly recorded.** A wrong FTP shard reported GPL570 as
  having no annotation; `GPL570.annot.gz` exists and is 8.5 MB.
- **A 4.5 GB detour.** GPL5175's only platform-level annotation is 4.5 GB; the
  series-level SOFT carries the same table for 25.6 MB.

## Provenance

Pre-registration, amendments 1–4 and the canonical digest chain:
`backend/hostresponse/preregistration.json`. Rules committed before the locked run:
`a1ef44b`. Result: `artifacts/hostresponse.json`. Discovery record:
`artifacts/hostresponse/discovery_record.json`.
