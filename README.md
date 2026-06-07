# AegisOS — HyperSepsisTwin-Ω

> **A unified software operating system for continuous critical-care monitoring.**
> AegisOS turns ordinary sensors — a wearable band and a plain camera — into a
> real-time digital twin of the patient that predicts sepsis early, explains its
> reasoning, and synthesizes **formally-verified** treatment protocols.

This is a year-long research build. Its guiding principle is **honesty over
demo-shine**: every detection number is reported with the real dataset and metric
that earned it, and every capability the data can't yet justify is labelled as
such — never faked.

---

## Why this exists

Sepsis kills ~11 million people a year; survival drops every hour treatment is
delayed. The earliest signs are physiological (autonomic, perfusion, inflammatory)
and show up in signals a wearable or camera can capture continuously — long
before a clinician is at the bedside. AegisOS is the software that listens to
those signals around the clock and reasons about them like a careful clinician.

A blunt early finding set the whole direction: the project's original hand-tuned
detector, when finally measured on **real** ICU data, scored barely above chance
(AUROC ≈ 0.64, essentially uncalibrated). Everything since has been rebuilt on
real labelled data with leakage-free validation.

---

## Architecture

```
            ┌──────────────── INPUTS ────────────────┐
            │  Wearable band (BLE)     MacBook camera │
            │  HR·HRV·SpO₂·temp·BP     rPPG·perfusion │
            └───────────────┬────────────────────────┘
                            ▼
        ┌──────────────  AegisOS CORE  ──────────────┐
        │  PatientDigitalTwin   (HDC state, baseline)│
        │  StreamingFeatureBuilder (no train/serve skew)
        │  Calibrated detector  (CinC-2019-grounded) │
        │  NeurosymbolicEngine  (SIRS/qSOFA overrides)│
        │  Risk fusion → BodyState (organ systems)   │
        │  GA + Z3 verifier     (safe protocols)     │
        │  LLM explainer        (clinician narrative)│
        └───────────────┬────────────────────────────┘
                        ▼
        WebSocket / REST snapshot  ──►  (future) body-twin frontend
```

### Subsystems

| Package | Role | Docs |
|---|---|---|
| `backend/ml/` | Data-grounded detection trained/validated on PhysioNet CinC 2019 | [ml/README](backend/ml/README.md) |
| `backend/physio/` | HRV analysis + camera-path validation (BIDMC PPG vs ECG) | [physio/README](backend/physio/README.md) |
| `backend/vision/` | Contactless optical biosensor (rPPG / perfusion / distress) | [vision/README](backend/vision/README.md) |
| `backend/aegis/` | Unified runtime, body-state model, streaming API | [aegis/README](backend/aegis/README.md) |
| `backend/llm/` | Sparing, rate-limited LLM clinical-narrative layer | — |
| `backend/core/` | HDC engine, digital twin, neurosymbolic rules, GA, Z3 verifier | — |

---

## What's validated (real data, honest numbers)

| Capability | Result | Data | Status |
|---|---|---|---|
| Sepsis detection (wearable channels) | **AUROC 0.774**, AUPRC 0.080, utility 0.353 | CinC 2019 (40,335 pts) | ✅ validated |
| Probability calibration | **ECE 0.0008**, Brier 0.017 | CinC 2019 | ✅ validated |
| rPPG heart-rate recovery | within ~3 bpm | synthetic ground truth | ✅ validated |
| Pulse→HRV chain | HR MAE **~1.2 bpm**, SDNN corr 0.71 | BIDMC (real PPG+ECG) | ✅ validated |
| HRV→sepsis pipeline | AUROC ~0.9 | HeRO-grounded synthetic | 🔬 machinery only |
| Treatment-protocol safety | Z3-proven invariants (allergy, fluid caps, vaso) | — | ✅ verified |

**Honest boundaries (not hidden):**
- Camera **optics** (video→pulse) validated only on synthetic faces — real rPPG
  video datasets are tens of GB; closing this needs a UBFC/PURE subset.
- **HRV→sepsis on real labels** needs credentialed data (MIMIC-IV Waveform); the
  plug-and-play adapter is built and waiting.
- The grounded HDC detector (AUROC ~0.70) trails the gradient-boosting benchmark
  (0.77); closing that gap is on the roadmap.

Run it yourself: `python verify_full_system.py` (12 subsystem checks).

---

## Build plan & status

**Phase 1 — Real, measurable detection** ✅ *done*
Data pipeline, leakage-free features, evaluation harness (AUROC/AUPRC/calibration/
utility), data-grounded HDC + GBM models, honest baseline of the old detector.

**Phase 2 — Contactless optical biosensor** ✅ *done*
rPPG (HR/HRV/RR), micro-perfusion + sympathetic-distress proxies, face-mesh ROIs,
camera→detection integration.

**Phase 3 — Trustworthy probabilities + unified runtime** ✅ *done*
Post-hoc calibration (ECE 0.39→0.0008), AegisCore orchestrator, end-to-end
verification suite.

**Phase 4 — Camera-path validation + clinical depth** ✅ *done*
HRV feature suite (incl. HeRO descriptors), real PPG-vs-ECG validation on BIDMC,
HRV→sepsis pipeline + MIMIC-IV adapter, organ-system **BodyState** model.

**Phase 5 — Integration surface** ✅ *done*
Versioned snapshot contract, signal-history buffers, WebSocket + REST AegisOS
server (`backend/aegis/server.py`).

**Phase 6 — The frontend** ⏳ *next, intentionally deferred*
A high-tech 3D/2D **recreation of the body** with live signal overlays, organ-
system lighting driven by `BodyState`, and the risk/protocol timeline — built on
the stable streaming contract once the backend is finalized.

**Phase 7 — Real HRV→sepsis + accuracy** 🔭 *planned*
Credentialed MIMIC-IV Waveform training, temporal/sequence modeling to push past
0.77 AUROC, external validation, real-camera optics validation.

**Phase 8 — On-device AegisOS + hardware** 🔭 *vision*
Run the runtime on the wearable; close the loop band ⇄ backend ⇄ camera.

---

## The vision

AegisOS is meant to feel like a living **mission-control for one human body**. The
backend already emits everything that view needs as one JSON snapshot per tick:
per-organ severity, anatomical regions, every signal's live waveform and source,
the calibrated risk and its drivers, the verified protocol, and a plain-language
narrative. The frontend — deliberately not built yet — will render that as an
extraordinary, fully-integrateable digital twin: a body that lights up where it's
struggling, signals that pulse in real time, and a system that not only says
"sepsis risk rising" but shows *why*, *where*, and *what to do* — with a
mathematical proof the proposed treatment is safe.

---

## Getting started

```bash
git clone <your-repo-url> && cd Sepsis
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# 1. Real clinical data (open, resumable)
python -m backend.ml.data.cinc2019 --download           # detection data (~40k pts)
python -m backend.physio.data.bidmc --download --n 20    # PPG+ECG for HRV validation

# 2. Verify everything works (12 subsystem checks)
python verify_full_system.py

# 3. Train + calibrate the detector
python -m backend.ml.scripts.train_and_compare --feature-set wearable --save

# 4. Run the AegisOS backend (WebSocket + REST contract)
python -m backend.aegis.server      # http://127.0.0.1:8100/aegis/snapshot
```

> `backend/api/app.py` + `frontend/` are the **legacy demo** (simulator + old
> heuristic). The real system is `backend/aegis/` on the data-grounded models.

---

## Safety invariants proven by Z3

| Rule | Constraint |
|---|---|
| CHF fluid cap | `CHF ⇒ fluid_rate ≤ 5.0 mL/kg/hr` |
| Renal fluid cap | `Renal_Failure ⇒ fluid_rate ≤ 4.0 mL/kg/hr` |
| Vasopressor max | `vasopressor_dose ≤ 0.4 mcg/kg/min` |
| Vasopressor justification | `vaso > 0.1 ⇒ MAP < 65 mmHg` |
| Drug allergy | `Penicillin_Allergy ⇒ antibiotic ≠ Pip/Tazo` |
| Sepsis intervention | `Risk_HIGH ⇒ antibiotic ≠ none` |

---

## License

Research and educational use only. **Not a medical device; not for clinical use.**
