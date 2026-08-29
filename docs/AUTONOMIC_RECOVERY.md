# A new detection algorithm, and the honest state of it

**Why this exists.** The shipped detector is not an early-warning system. Decomposing its
AUROC showed 0.79 on *which patient becomes septic* and performance on *when* that is
not identifiable from a clock. The supervision-density work then improved it by giving
it **more patient-identity supervision**, and a placement control proved the timing
information contributed nothing. Every lead-time figure this project has published comes
from a timing-blind detector alarming at a semi-arbitrary point in a stay it has already
flagged.

Eight method families have now failed on hourly vitals. That is not eight unlucky
experiments; it is one fact measured eight times. The sepsis cascade runs

```
pathogen → cytokine surge → endothelial activation → MICROCIRCULATORY failure
        → capillary leak → macrocirculatory compensation → decompensation
```

and heart rate, blood pressure and respiratory rate only move at the **compensation**
stage. By the time an hourly vital sign changes, the cascade is advanced. The signal
arrives late because we read it late, so a ninth model on the same inputs produces a
ninth null.

## The precedent

Exactly one sepsis early-warning system has ever reduced mortality in a randomised
trial: **HeRO** (Moorman et al., ~3,000 VLBW infants, >20% mortality reduction, 22%
relative risk reduction for sepsis). It reads **beat-to-beat** heart-rate
characteristics and predicts deterioration ~24h out. Its model is close to a logistic
regression — *the power is in the signal, not the algorithm*, and it is precisely the
signal hourly sampling destroys by construction.

`backend/physio/sepsis_hrv.py` already implements that feature family. What it cannot do
is survive an adult ICU: adult HRV **magnitude** is dominated by beta-blockade, sedation
depth, mechanical ventilation, atrial fibrillation, pacing and age-related autonomic
decline. Measuring SDNN in an adult ICU is substantially measuring the drug chart, which
is the standing explanation for why adult HeRO analogues have not replicated.

## What is new here

`backend/physio/recovery.py` measures the **speed** of the autonomic response rather
than its size.

An ICU record is full of spontaneous perturbations — transient decelerations,
respiratory excursions, brief pressure dips. Each is a free stress test the patient runs
on themselves. A healthy control loop is knocked off its operating point and returns
quickly; a failing one returns slowly, or not at all. For every detected perturbation
the module fits a **recovery time constant** τ (time to decay to 1/e of peak deviation)
and profiles the distribution: median, spread, the asymmetry between accelerations and
decelerations, and the fraction that never resolve.

**Why this survives what kills adult HRV.** τ is amplitude-invariant. Scale every RR
deviation by α — roughly what beta-blockade or deep sedation does — and SDNN scales by α
while the 1/e crossing does not move. Event detection is thresholded in MAD units, which
also scale, so the same events are found. Measured at α = 0.3:

| | change |
|---|---|
| SDNN | **−70.0%** |
| worst recovery feature | **3.1%** |
| τ counted in beats | **exactly 0** |

A 23× robustness gap, and exact in the beat-counted channel.

**Theoretical grounding, and a null this does not contradict.** Critical slowing down
predicts that a system approaching a transition recovers more slowly — rising τ is the
canonical early-warning indicator. This project already tried CSD and measured ~0. That
is not evidence against this module: CSD is defined at the system's *relaxation
timescale*, which for cardiovascular control is seconds, and the earlier attempt sampled
hourly, aliasing the phenomenon away before any estimator saw it. The previous null was
a statement about sampling rate, not physiology.

## What is validated, on real data

`make verify-recovery` — 14 checks, all passing. Nothing here is a detection claim.

| check | result |
|---|---|
| recovers a τ we chose | monotone across 2→12 s, r = 0.986 |
| absolute accuracy | median relative error 29.5% (causal baseline compresses long τ — reported, not hidden) |
| amplitude invariance | 3.1% drift vs SDNN −70%; exactly 0 in beats |
| reads dynamics, not the histogram | SDNN-preserving shuffle moves τ 3.83 s → 1.57 s |
| strictly causal | rewriting the future leaves earlier perturbations identical |
| within-patient stability | within-patient SD 0.85 s vs between-patient 1.27 s |
| runs on the credentialed contract | 28 features, shapes verified, **no synthetic accuracy computed** |

Two of these were failures first, and both taught something. The estimator originally
took a **single MAD over the whole window** — ordinary offline practice, and a leak: the
detection threshold at beat *i* depended on beats after *i*, which the causality check
caught. Fixing it to a past-only running scale also cut amplitude drift from 12.6% to
3.1%, so the leak had been *causing* the imprecision. And the first event threshold
(k = 2.5) yielded ~3 events per 8-minute record; a power analysis put the knee at ~15
events (CV 15.0% at 6 events, 8.9% at 10, 5.5% at 15), which set both the threshold
(k = 1.5, measured to give 2.08 events/min on real records) and the window (512 beats).

## The biggest negative: the camera path is not cleared

The hope was that these features would agree between an ECG-derived and a PPG-derived
pulse, which is what would let them run on the contactless path the OS is built around.
Measured on the same BIDMC patients at the same moments, **they do not**:

| | ECG vs PPG |
|---|---|
| τ concordance | r = 0.042 |
| short-term variability inflation | 1.50× |

And the obvious fix fails too. Switching the fiducial from the systolic **peak** to the
pulse **foot** — the standard choice precisely because it is meant to resist
transit-time jitter — moved concordance to r = 0.024 and made inflation slightly *worse*
(1.62×). Pulse transit time jitters ~10–30 ms independently of the cardiac cycle, which
is the same scale as the dynamics being measured. The limitation is intrinsic to
pulse-derived beat timing, not a matter of which point on the wave is picked.

**Consequence: this algorithm needs ECG-grade beat timing — a bedside monitor feed or a
chest patch, not the camera.** `assert_source_cleared` refuses a pulse source rather than
emitting numbers that look reasonable. The camera path keeps everything else it does
(perfusion, thermal, distress, rPPG heart and respiratory rate); it cannot carry this.

## What is needed next, and what it is blocked on

The detection claim needs beat-to-beat waveforms **with sepsis labels on the same
patients**, which nothing in this repository has — BIDMC has waveforms and no labels,
CinC 2019 has labels at hourly resolution. `backend/physio/data/mimic_waveform.py`
already implements the adapter, and the recovery pipeline consumes the same
`RRLabeledSource` contract, so validation runs unchanged on arrival.

**The target is MIMIC-IV Waveform, not HiRID.** HiRID stores derived heart rate at low
cadence rather than beat-to-beat ECG, so despite its 2-minute resolution it cannot yield
true beat-interval features. Getting there requires, in order: CITI "Data or Specimens
Only Research" training → PhysioNet credentialing → the MIMIC-IV DUA → deriving a
Sepsis-3 onset table from the clinical modules and aligning it to the waveform records.

Open problems, in priority order:

1. **No labelled beat-to-beat data.** Everything else is downstream of this.
2. **Atrial fibrillation and ectopy.** Amplitude invariance addresses confounds that
   *scale* the response. AF *destroys* RR structure and needs detection and exclusion,
   which is not built.
3. **Ventilator-imposed rhythm.** A mandatory mode drives respiration externally; the
   RSA gain channel needs a ventilation-aware guard.
4. **The τ compression bias.** A causal baseline tracks into long perturbations and
   compresses them ~30%. Ordering is preserved, so it does not threaten a detector, but
   an absolute τ should not be quoted as physiology yet.

## Reproducing

```bash
make verify-recovery
```
