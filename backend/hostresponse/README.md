# Host-response gene expression — infection vs sterile inflammation

Can a blood expression signature separate **infectious sepsis** from **sterile**
(non-infectious) systemic inflammation, across cohorts and platforms?

That axis is the one the production detector is structurally blind to. It reads
hourly vitals and labs, and trauma, major surgery and pancreatitis derange those
same channels that infection does. `backend/ml/alarm_policy.py` has already
established the consequence: no post-processing of that score recovers a
separation the input channels do not carry. This subsystem tests a **different
measurement**, not a better reading of the same one.

## The two links, and which one this can test

| Link | Claim | Status |
|---|---|---|
| **A** | An expression signature separates infection from sterile inflammation, and survives a change of cohort and platform | **Testable here**, on public GEO cohorts |
| **B** | That separation reduces *this product's* false-alarm floor | **Not testable, and not claimed.** No bed in the CinC cohort carries transcriptomics, so there is no per-patient input to fuse |

Stating B explicitly is the point. A module that quietly implied it from an A
result would be making exactly the leap this repository refuses.

## Cohorts — decided by evidence, not by title

The gate reads each candidate's own `!Sample_characteristics_ch1` cells, records
the verbatim strings, and refuses anything whose control arm cannot be positively
identified as sterile. Healthy-volunteer controls are never admissible: "is this
person ill?" is a much easier question than the one being asked.

| Accession | Platform | Infectious / Sterile | Role |
|---|---|---|---|
| GSE74224 | GPL5175 (array) | 74 / 31 post-surgical | discovery |
| GSE40012 | GPL6947 (array) | 114 / 40 non-infective SIRS | locked |
| GSE28750 | GPL570 | 10 / 11 | excluded — only 10 infectious (minimum 20) |
| GSE65682 | GPL13667 | — | excluded — no key identifies an infectious vs sterile arm |
| GSE63042 | GPL9115 (RNA-seq) | 106 / 23 | **withdrawn**, see amendment 1 |

GSE65682 is the instructive one: its GEO summary *implies* a non-infectious
comparator, and its per-sample characteristics say otherwise (its arms are
CAP/HAP, ICU-acquired infection, abdominal sepsis vs GI control). That is why
admissibility is decided from characteristics rather than from a summary.

**The unit is the patient, not the sample.** GSE40012's 154 infectious/sterile
samples come from 40 people sampled repeatedly (27 infectious, 13 sterile), so the
study is 145 patients, not 259 samples — and its sterile arm, 44 patients, binds
every interval it reports.

GSE63042 was pre-registered and then withdrawn: GEO holds no usable expression
table for it — metadata-only series matrix, empty family SOFT, and a single
`.xlsx.gz` of relative RPM that would need both a new dependency and a fragile
patient-id→GSM join on a *locked* cohort. The cost is recorded: the study loses
its only RNA-seq cohort, so platform transfer is array-to-array only.

## Feature space

Cohorts share no identifier, so probes are mapped to gene symbols:

* **GPL6947** — curated `GPL6947.annot.gz`, 7.0 MB, explicit `Gene symbol` column.
* **GPL5175** — no curated annot exists. Its platform SOFT is **4.5 GB**; the
  *series*-level `GSE74224_family.soft.gz` is **25.6 MB** and embeds the same
  table. The 25.6 MB route is used.

Measured: 15,812/15,989 and 29,240/48,754 probes annotate, giving a **13,785
gene-symbol intersection**. The feature space is not the limiting factor.

## Pipeline

1. **Rank within sample**, then z-score the ranks. Nothing is computed across
   samples or pooled across cohorts, so no batch statistic can enter a feature —
   and it is invariant to any monotone per-sample transform, which is why
   GSE74224's log-scale values (2–14) and GSE40012's linear intensities
   (42–34,356) never have to be reconciled.
2. **Collapse probes to symbols on the ranks.** The order matters: a mean is not
   invariant to a monotone per-sample transform, a rank is.
3. **Intersect** to the shared symbols.
4. **Leave one cohort out**, reporting *every* direction — a signature that
   transfers one way and not the other has fitted one cohort's idiosyncrasy.
5. **Learner: L2 logistic regression, no scaling, C = 0.01**, chosen on the
   discovery cohort by a rule stated before looking. `GBMDetector` emits a single
   constant prediction at 13,785 features × 52 rows; a scaler would re-introduce a
   cross-cohort statistic by fitting gene means on one cohort and applying them to
   the other.

## What would make a good number meaningless

Every guard is computed in the same run as the primary, never on request:

* **Per-direction interval** — each held-out cohort's AUROC is computed inside that
  cohort and the directions are averaged, so a platform offset cannot enter. A +3.0
  offset leaves it unchanged while a pooled AUROC on the same data falls 0.834 → 0.714.
* **Label permutation** within cohort, **≥ 20 draws**, each with a fresh fold split —
  the *mean* of that null must land in [0.42, 0.58]. Two thin nulls raised false
  alarms here before this was fixed.
* **Cohort identity** — the same pipeline predicting *which cohort* a sample came
  from. Reported. The frozen rule voided the study at ≥ 0.90; amendment 4 retired
  that, because under a per-direction endpoint a cohort signal cannot rank patients
  and the rule could only fire on platform differences.
* **Easiness calibrator** — infection vs *healthy*, reported only to show how much
  easier the trivial task is. Never quotable as the result.

## The bar, and the three ways this ends

`preregistration.json` is frozen and hashed; `run_study.py` refuses the locked
phase if the digest changed after the discovery phase ran. Four amendments are
recorded inside it, each with the canonical digest it superseded.

The bar **in force** (amendment 4) requires all of: mean per-direction AUROC ≥ 0.75,
95% patient-cluster lower bound > 0.70, **each direction ≥ 0.70 on its own**, and a
permutation null centred in [0.42, 0.58]. The artifact also reports the verdict
under the **frozen** rules, so the effect of every amendment is visible in the
result, not only in the document's history.

**Power, stated before any outcome:** simulated from cluster counts alone, the
lower bound clears 0.70 in 6/20 replicates at a true transfer AUROC of 0.80, 18/20
at 0.85 and 20/20 at 0.90. This study can detect transfer at or above ~0.85 and not
below it, so a miss reads "not detected at ≥ 0.85" — never "no transfer".

The bar sits deliberately below the 0.83–0.91 that SeptiCyte RAPID and
IMX-BVN/TriVerity report in their own validation cohorts, because those are
different cohorts, platforms and case definitions — beating or missing them here
would not be a like-for-like comparison.

Three terminal states are all legitimate and all get written down: **gate
failure**, **confounded**, and **bar missed** → `docs/HOST_RESPONSE_NULL.md`, with
`published: false` and no site block.

## Running it

```bash
make geo-manifest        # fetch candidates, classify control arms, apply the gate
make hostresponse PHASE=discovery   # iterate freely; records the pre-registration digest
make hostresponse PHASE=locked      # ONCE; refuses if the digest moved since discovery
make verify-hostresponse # the adversarial suite
```

`--inspect` on the manifest builder dumps every characteristics key and its
distinct values, so any classification can be read against the source.

## Status

**Locked result, 2026-09-12: BAR MISSED** under the rules in force, **CONFOUNDED**
under the frozen rules; `published: false`.

Per direction 0.773 (held-out GSE40012) and 0.682 (held-out GSE74224); mean 0.727,
95% patient-cluster interval 0.636–0.810; permutation null 0.495; cohort identity
1.000. The interval excludes chance, so some transfer is present — but it misses the
pre-registered bar, and its upper bound sits below the ~0.85 the study was powered
to detect. Amendment 4 changed the failure's label, not its outcome: neither rule set
publishes. The rules and code were committed (`a1ef44b`) before the locked cohort
was loaded.

Full write-up: [`docs/HOST_RESPONSE_NULL.md`](../../docs/HOST_RESPONSE_NULL.md).
