# AegisOS — HyperSepsisTwin-Omega

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
        WebSocket / REST snapshot  ──►  AegisOS web shell (`webui/`)
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

For a practical maintainer map, see [docs/REPO_MAP.md](docs/REPO_MAP.md).

---

## What's validated (real data, honest numbers)

| Capability | Result | Data | Status |
|---|---|---|---|
| Sepsis detection (within-hospital) | **AUROC 0.774**, AUPRC 0.080, utility 0.353 | CinC 2019 (40,335 pts) | validated |
| Sepsis detection (**cross-hospital**) | **AUROC 0.689**, calibration holds | CinC A<->B (BIDMC/Emory) | validated |
| Probability calibration | **ECE 0.0008**, Brier 0.017; ECE ~0.005 cross-site | CinC 2019 | validated |
| rPPG heart-rate recovery | within ~3 bpm | synthetic ground truth | validated |
| Pulse-to-HRV chain | HR MAE **~1.2 bpm**, SDNN corr 0.71 | BIDMC (real PPG+ECG) | validated |
| HRV-to-sepsis pipeline | AUROC ~0.9 | HeRO-grounded synthetic | machinery only |
| Treatment-protocol safety | Z3-proven invariants (allergy, fluid caps, vaso) | - | verified |

The **honest deployment number is the cross-hospital AUROC ≈ 0.69** (not the
optimistic within-hospital 0.77) — and probabilities stay trustworthy across
sites. Full analysis: [docs/PHASE7_REPORT.md](docs/PHASE7_REPORT.md).

**Honest boundaries (not hidden):**
- Detection plateaus ~0.77 within-site / ~0.69 cross-site. The bottleneck is
  **data, not model**: labs add only ~0.02 AUROC and richer temporal features add
  nothing — the real levers are continuous HRV + multi-site training.
- Camera **optics** (video→pulse) validated only on synthetic faces — real rPPG
  video datasets are tens of GB; closing this needs a UBFC/PURE subset.
- **HRV→sepsis on real labels** needs credentialed data (MIMIC-IV Waveform); the
  plug-and-play adapter is built and waiting.
- Standalone HDC trails tree ensembles on raw tabular discrimination. HDC remains
  valuable as interpretable case memory, resonance/consensus evidence, and an
  optional stack member rather than the primary accuracy path.
- The single GBM is the stable deployment-compatible artifact path
  (`artifacts/gbm_wearable.joblib`). The current research trail shows the stacked
  ensemble as the stronger production candidate; see `backend/ml/README.md`.

Run it yourself: `python verify_full_system.py` (full subsystem verification).

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

**Phase 7 — Accuracy, generalization & feature science** ✅ *done*
Multi-scale temporal features (tested — no gain, not adopted), lab-augmented
ceiling (+0.02), feature cache, and **cross-hospital external validation**
(AUROC ~0.69, calibration holds). Established that detection is data-limited, not
model-limited. Full write-up: [docs/PHASE7_REPORT.md](docs/PHASE7_REPORT.md).

**Phase 8 — The AegisOS shell (frontend)** ✅ *done*
A full **operating-system shell** (not a website): cold-boot sequence, system bar,
desktop workspace of windowed panels, and a dock. Centerpiece is a live anatomical
**digital twin** whose organs (brain · lungs · heart · liver · kidneys · vascular ·
skin) light by `BodyState` severity, with an HR-synced heartbeat and animated ECG/
PPG traces. Panels: sepsis-risk core, live telemetry, the multi-agent debate, the
what-if treatment projections, the Z3-verified protocol + narrative, and the live
8-layer kernel monitor. Pure rendering against the v1.2 snapshot — `webui/`, served
by the AegisOS server.

**Phase 9 — Real HRV→sepsis + multi-site (gated on credentialed data)** 🔭
Train the HRV→sepsis model on credentialed MIMIC-IV Waveform (adapter ready),
multi-site training to shrink the cross-hospital gap, real-camera optics validation.

**Phase 10 — On-device AegisOS + hardware** 🔭 *vision*
Run the runtime on the wearable; close the loop band ⇄ backend ⇄ camera.

---

## Developer Quickstart

The repo now has a small command surface for the common paths:

```bash
make setup
make verify
make serve
```

Then open `http://127.0.0.1:8100`.

Manual equivalent:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
python verify_full_system.py
python -m uvicorn backend.aegis.server:app --host 127.0.0.1 --port 8100
```

Data-dependent workflows:

```bash
python -m backend.ml.data.cinc2019 --download
python -m backend.physio.data.bidmc --download --n 20
python -m backend.ml.scripts.train_and_compare --feature-set wearable --save
python -m backend.ml.scripts.train_stacked --feature-set wearable
python -m backend.ml.scripts.rigorous_eval --feature-set wearable --boot 500
```

`backend/api/app.py` + `frontend/` are the **legacy demo** (simulator + old
heuristic). The real system is `backend/aegis/` served with `webui/`.

---

## The vision

AegisOS is meant to feel like a living **mission-control for one human body**.
The backend emits one JSON snapshot per tick: per-organ severity, anatomical
regions, every signal's live waveform and source, calibrated risk and drivers,
the verified protocol, and a plain-language narrative. The `webui/` shell renders
that as a digital twin: a body that lights up where it is struggling, signals
that pulse in real time, and a system that shows *why*, *where*, and *what to do*
with a mathematical proof that the proposed treatment is safe.

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
