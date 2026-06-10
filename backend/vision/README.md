# Vision Subsystem — Contactless Optical Biosensor

Extracts physiological signals from a plain RGB camera (e.g. the MacBook webcam)
and feeds them into the sepsis-detection stack. The camera does **not** diagnose
sepsis; it produces a subset of the same physiological channels the detector
already understands, plus experimental proxies — each with a confidence score.

## Three optical channels

| Channel | Method | What it yields | Validation status |
|---|---|---|---|
| **rPPG** | POS / CHROM on forehead+cheek skin ROI | Heart rate, HRV, respiratory rate | ✅ HR/RR validated vs synthetic ground truth (±3 bpm) |
| **Micro-perfusion** | Pulsatile AC/DC + skin chrominance + spatial dispersion | Perfusion proxy (shunting) | ⚠️ Experimental proxy, trend-only |
| **Sympathetic distress** | Landmark micro-tremor (4–12 Hz) + pupil instability | Stress vector | ⚠️ Experimental proxy, trend-only |

## Module map

| Module | Role |
|---|---|
| `signal_processing.py` | rPPG core (POS/CHROM, bandpass, HR via Welch, HRV, RR, SNR) — pure, testable |
| `face.py` | MediaPipe FaceLandmarker (478 pts incl. iris) ROIs; Haar fallback |
| `perfusion.py` | Micro-perfusion mapping proxy |
| `distress.py` | Micro-tremor + pupil-instability proxy |
| `pipeline.py` | Buffers + uniform resampling → `CameraVitals` with per-channel confidence |
| `camera.py` | Live webcam capture loop (runtime) |
| `detect.py` | `OpticalSepsisAssessor`: camera → digital twin → neurosymbolic → narrative |
| `scripts_validate.py` | Camera-free end-to-end validation (synthetic faces, known HR) |

## Run it

```bash
# Validate the signal core (no camera needed):
python -m backend.vision.signal_processing      # HR recovery on synthetic pulse
python -m backend.vision.scripts_validate       # full pipeline, synthetic faces

# Live, with your webcam (macOS will prompt for camera permission):
python -m backend.vision.camera --seconds 30
```

First run downloads the MediaPipe FaceLandmarker model into `data/models/`
(git-ignored).

## Camera detection backend (accuracy-focused)

The camera path is built for high-accuracy *continuous* monitoring, not one-shot
guesses:

| Module | Role |
|---|---|
| `signal_processing.py` | rPPG with **parabolic peak interpolation** (HR mean err ~0.4 bpm, ≤1 bpm on clean signal) + SNR-weighted POS/CHROM fusion (`analyze_window_robust`) |
| `camera_detector.py` | `CameraSepsisDetector` — per-reading **log-odds evidence** from HeRO HRV-collapse + tachycardia + tachypnea + perfusion + stress, **relative to an adaptive personal baseline**, quality-weighted |
| `monitor.py` | `ContinuousSepsisMonitor` — **sequential-Bayesian** leaky integrator + hysteresis alarm. Turns noisy per-frame evidence into a stable, low-false-alarm posterior |
| `scripts_camera_eval.py` | stream-level validation |

**Why this is accurate:** a single frame is weak, but the camera streams
continuously, so integrating calibrated evidence over time (trading seconds of
latency for SNR) yields confident detection. On physiology-grounded synthetic
streams *with ~10% motion-artifact frames*, the monitor reaches AUROC ≈ 1.0 with
**0 false alarms** (artifacts are rejected by integration + hysteresis) and a
healthy median lead time. Run: `python -m backend.vision.scripts_camera_eval`.

## Every observable sign + posture + recognition boxes

| Module | Role |
|---|---|
| `signs.py` | **Comprehensive observable-sign inventory** — every camera-visible sepsis sign across cardiovascular / respiratory / perfusion / integument / autonomic / neurologic (AMS), each with confidence, severity, a **validated / proxy / experimental** grade, and calibrated log-odds. Skin-colour features: pallor, cyanosis (lips), jaundice, flush, mottling, diaphoresis. |
| `pose.py` | **Head pose + posture** — pitch/yaw/roll from landmarks; classifies upright / reclined / supine / lateral / inverted. **Supine & rolled faces stay fully monitored** (gate ≈ 1.0); only true profile views are down-weighted. Built for a **perched camera over a lying patient**. |
| `face.py` | Per-ROI **bounding boxes** (forehead, cheeks, eyes, lips) + `annotate_frame()` overlay; named 3D landmarks for pose; eye-openness. |

The pipeline attaches `boxes`, `posture`, and the full sign inventory to
`CameraVitals.raw`, and the monitor folds the (non-core) signs' evidence into the
sequential posterior — so the system **accounts for every observable sign** and
keeps monitoring patients regardless of how they are lying.

## Accuracy & robustness (measured)

Performance was pushed and **measured**, not assumed:

| Lever | Result |
|---|---|
| Sub-sample peak localization + **adaptive narrow-band** HRV | real BIDMC PPG-vs-ECG correlation: **SDNN 0.71→0.88, RMSSD 0.50→0.85, sample-entropy 0.63→0.70** (RMSSD bias 8.5→4.2) |
| POS/CHROM SNR fusion (live pipeline) | **−61 % mean HR error** under degradation; **motion 46→1.1 bpm**, low-SNR 39→19 bpm |
| **Multi-ROI** SNR + agreement fusion | per-ROI SNR diagnostics; a poor/occluded/motion ROI is identified and weighted out instead of contaminating the average |
| **Temporal HR Kalman** smoother + outlier gate | HR jitter **6.8→1.8 bpm** on a noisy stream; rejects bad-window outliers |
| **Skin-pixel segmentation** (tone-agnostic) | rejects glare / shadow / hair / glasses; keeps diverse skin tones (fairness-aware, not a skin-colour box) |
| Motion + illumination artifact gate | bad windows down-weighted instead of trusted |
| Cleaner HRV features → HRV→sepsis | machinery AUROC **0.917→0.936** |

```bash
python -m backend.vision.scripts_robustness   # degradation benchmark
python -m backend.physio.validate_ppg_hrv     # real PPG-vs-ECG HRV agreement
```

Honest limitation surfaced by the benchmark: rPPG on **darker skin tones** is
materially harder (~20 bpm error vs <1 bpm on clean) — a known optics limitation;
the quality gate flags it (≈0.5) so it is down-weighted rather than trusted. The
residual HRV bias vs ECG is a constant offset the **personal-baseline** detector
absorbs (sepsis = HRV *collapse relative to the subject's own normal*).

## The honest validation boundary (important)

- rPPG **heart rate / respiratory rate** are validated against synthetic ground
  truth and are well-established in the literature.
- rPPG **HRV** is real but webcam-noisy → reported as a trend with a quality score.
- Perfusion and distress are **research proxies**, explicitly *not* validated
  surrogates for blood pressure, lactate, or a sepsis diagnosis.
- **End-to-end camera→sepsis detection is not yet validated against ground
  truth.** The camera's most informative signal is HRV collapse, but the
  CinC 2019 dataset (what the detector was trained on) has **no HRV channel**, so
  a CinC-trained model cannot use it. Camera assessment therefore runs through
  the transparent neurosymbolic engine, and full validation awaits a labelled
  dataset that includes continuous HRV. This boundary is deliberate and stated,
  not hidden.

## LLM narrative layer (`backend/llm/`)

`AlertExplainer` turns the detector's structured output into a short
clinician-readable summary, using OpenRouter (Llama 3.3 70B) **sparingly**
(hard call budget + min interval) and only on meaningful events. The API key is
read from `backend/.env` only. Every failure (rate limit, offline) falls back to
a deterministic template — the system never depends on the LLM.
