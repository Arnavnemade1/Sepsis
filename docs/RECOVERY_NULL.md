# Autonomic recovery (τ) does not predict circulatory decompensation — a null

**Date:** 2026-08-29 · **Cohort:** open-access MIMIC-III Waveform DB, no credentialing
**Verdict:** the pre-registered claim failed. Reported in full.

## What was claimed in advance

`backend/physio/recovery.py` proposed measuring the *speed* of autonomic recovery — the
time constant τ of return from spontaneous RR perturbations — rather than the *size* of
variability, on the argument that τ is amplitude-invariant and therefore survives the
confounds (beta-blockade, sedation, ageing) that stop adult HeRO analogues replicating.

The bar was stated before the run: **`recovery` or `all` must beat `hrv` with a paired
patient-bootstrap CI excluding zero**, on within-patient data only.

## What happened

Within-patient case-control, 29 patients contributing both a decompensation event and a
matched stable control from the same admission. Features from ECG only; outcome from the
arterial line — a separate transducer, so non-circular by construction.

| lead | hr | hrv | recovery (τ) | all |
|---|---|---|---|---|
| 15 min | 0.493 | 0.538 | 0.376 | 0.467 |
| 30 min | 0.481 | 0.467 | 0.462 | 0.445 |
| 60 min | 0.607 | 0.580 | 0.482 | 0.514 |

Does τ add over HRV magnitude? **−0.071 / −0.022 / −0.066**, no CI excluding zero at any
lead. Over heart rate alone? **−0.026 / −0.036 / −0.093**, likewise. The bar was not met.

Label permutation within patient collapses every model (0.40 / 0.45 / 0.42), so the null
is a null and not a broken pipeline.

## Reading it honestly

1. **τ failed the test it set itself.** No hedging available: the point estimates are
   negative.
2. **Every other family failed too, including plain heart rate.** When the trivial
   baseline also sits at chance, the usual reading is not "the new feature is bad" but
   "this target is not predictable from this signal". Predicting a MAP fall 15–60 minutes
   ahead *within the same patient* from beat timing alone may simply be very hard — which
   agrees with everything else this project has measured about within-patient timing
   (`docs/SUPERVISION_DENSITY.md`).
3. **It is underpowered and rules out only a large effect.** 29 patients, ~50 event rows
   per lead, CI half-widths ±0.10–0.15.

## The confound the workaround cannot remove

Many ICU hypotensive episodes are **iatrogenic** — a sedation bolus, a position change,
dialysis initiation, an anaesthetic. Those are not predictable from prior autonomic state
by construction, and they dilute any real signal. Excluding them needs the medication and
procedure chart, which is exactly what the open-access waveform database does not carry.
This is a structural limit of using a proxy outcome without a chart, not a bug.

## What this does NOT establish

It does not falsify τ for **sepsis**. The outcome here is circulatory decompensation, a
mechanistically-aligned proxy chosen because it was reachable without credentialing.
Septic shock is a decompensation, but not every decompensation is sepsis, and the
iatrogenic fraction above is likely large.

## Where that leaves the question

Two honest options, and they differ in kind:

* **Scale the cohort.** Getting CI half-widths near ±0.05 needs roughly 6× the patients
  (~175 dual-label). That is on the order of 12–20 hours of streaming. It would settle
  whether a *small* effect exists — but it cannot remove the iatrogenic confound.
* **Get the credentialed data.** MIMIC-IV Waveform / HiRID carry real Sepsis-3 labels and
  the chart needed to exclude intervention-driven events. That answers the question
  actually being asked instead of a proxy for it. `sepsis_hrv.py` and this module both
  emit through the same `RRLabeledSource` contract, so that run needs no new code.

The infrastructure built for this null is reusable either way: the artifact-gated outcome
detector (16/16), the τ estimator with its audit (14/14), the numerics-first retrieval,
and the within-patient design that makes patient identity unable to explain a result.
