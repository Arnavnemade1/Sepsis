# Physio Subsystem — HRV Analysis & Camera-Path Validation

This subsystem closes (as far as open data allows) the honesty gap flagged for the
camera path: *does contactless sensing actually measure sepsis-relevant
physiology?* It splits the claim into two separable links and validates each with
the strongest available **real** data.

## The two links

```
 video ──(optics)──► pulse ──(HRV math)──► HRV features ──(physiology)──► sepsis
        Link A0                Link A1                       Link B
```

| Link | Claim | Validation | Status |
|---|---|---|---|
| **A1** pulse → HRV | HRV from a *pulse* signal matches gold-standard ECG HRV | **BIDMC** (real PPG+ECG, 20 patients) | ✅ done |
| **A0** video → pulse | rPPG optics recover the pulse from face video | synthetic ground truth (real face-video sets are tens of GB) | ⚠️ synthetic only |
| **B** HRV → sepsis | HRV collapse predicts sepsis | HeRO precedent + pipeline; **needs credentialed labels** | 🔬 pipeline ready |

## What's validated, on real data

`validate_ppg_hrv.py` runs the **BIDMC** dataset (PhysioNet, open) — simultaneous
PPG and ECG on the same ICU patients — and compares HRV derived from the PPG
(camera-class pulse) against HRV from the ECG (gold standard):

- **Heart rate:** MAE ≈ 1.2 bpm (≈0.1 bpm on the cleanest recordings)
- **SDNN:** correlation ≈ 0.71, MAE ≈ 7 ms
- **RMSSD:** noisier (corr ≈ 0.5) — a known PPG limitation (pulse-transit jitter)

Conclusion: a pulse signal — the same class a camera produces — yields
trustworthy HR and good SDNN, with RMSSD as a noisier trend. This is real
evidence the camera's HRV is usable.

## Modules

| File | Role |
|---|---|
| `hrv.py` | ECG R-peak + PPG peak detection, NN cleaning, full HRV features (time/freq/nonlinear + HeRO sample-entropy & sample-asymmetry) |
| `data/bidmc.py` | Downloader + loader for BIDMC PPG+ECG (open, ~140 MB) |
| `validate_ppg_hrv.py` | PPG-vs-ECG HRV agreement on real patients |
| `sepsis_hrv.py` | Windowed RR → HRV features → calibrated detector (Link-B pipeline) |

## Link B: HRV → sepsis (credential-ready)

`sepsis_hrv.py` is the full pipeline: slice an RR series into windows, compute HRV
features, train a calibrated GBM with the same patient-grouped, leakage-free
harness used for the clinical model. It consumes any `RRLabeledSource`, so a
MIMIC-IV-Waveform or HiRID adapter is the only missing piece.

```bash
python -m backend.physio.data.bidmc --download --n 20   # fetch real PPG+ECG
python -m backend.physio.validate_ppg_hrv               # PPG vs ECG HRV (real)
python -m backend.physio.sepsis_hrv                     # HRV->sepsis pipeline (proxy)
```

### The honest gap
There is **no open dataset with continuous HRV + sepsis labels**. The HRV→sepsis
*model* is therefore validated against a synthetic cohort whose septic physiology
follows the published **HeRO** signature (reduced variability, reduced sample
entropy, transient decelerations; Moorman et al., proven in a 3,000-infant RCT).
Validation on **real** sepsis labels requires credentialed access (CITI training
+ data-use agreement) to MIMIC-IV Waveform or HiRID — a user action. The pipeline
runs unchanged once that data is available.
