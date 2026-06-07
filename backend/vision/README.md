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
