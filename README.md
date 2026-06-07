# HyperSepsisTwin-Ω

> A Hyperdimensional Neurosymbolic Wearable Digital Twin Framework for Real-Time Early Sepsis Detection and Verifiable Personalized Treatment Protocol Synthesis.

## Overview

HyperSepsisTwin-Ω is a novel systems software framework that:

- 📡 **Ingests real-time physiological telemetry** (HR, HRV, SpO₂, skin temperature, motion, sweat lactate) from a wearable wristband via Bluetooth Low Energy (BLE).
- 🧮 **Encodes patient state** into 10,000-dimensional bipolar hypervectors using a Vector Symbolic Architecture (VSA) / Hyperdimensional Computing (HDC) engine.
- 🧠 **Applies neurosymbolic reasoning** — combining HDC vector-space similarity with strict clinical rule evaluation (SIRS criteria, qSOFA score, Surviving Sepsis Campaign guidelines).
- 🧬 **Evolves personalized treatment protocols** (IV fluid strategy, vasopressor dosing, antibiotic selection/timing) using a multi-objective genetic algorithm with pharmacokinetic projection models.
- 🔒 **Formally verifies treatment safety** using the Z3 SMT (Satisfiability Modulo Theories) solver — mathematically proving that every generated recommendation satisfies clinical invariants before it is applied.
- ☁️ **Persists all data to Supabase** — storing anonymized telemetry logs, sepsis risk alerts, treatment protocols, and Z3 safety proofs in a real-time cloud PostgreSQL database.

---

> **Detection is now data-grounded.** The original hand-tuned heuristic scored
> near-chance on real patients (AUROC ≈ 0.64). Detection has been rebuilt on real
> labelled clinical data (PhysioNet/CinC 2019) with leakage-free, patient-level
> validation — see [`backend/ml/README.md`](backend/ml/README.md). A contactless
> optical biosensor (rPPG / perfusion / sympathetic-distress from a webcam) is in
> [`backend/vision/README.md`](backend/vision/README.md).

## Architecture

```
Wearable (BLE)  →  HDC Vector Engine  →  Digital Twin
                                              │
                                    Neurosymbolic Classifier
                                    (SIRS + qSOFA + HDC sim)
                                              │
                                  Evolutionary Protocol Synthesizer
                                     (Genetic Algorithm + PK model)
                                              │
                                  Formal Verification (Z3 SMT Solver)
                                              │
                              Supabase (patients, telemetry, proofs)
                                              │
                                  Real-Time Web Dashboard
```

---

## Project Structure

```
Sepsis/
├── backend/
│   ├── .env.example            # Environment variable template
│   ├── requirements.txt        # Python dependencies
│   ├── run.py                  # Server startup script
│   └── core/
│       ├── hdc.py              # Hyperdimensional Computing engine (D=10,000)
│       ├── twin.py             # Real-time patient Digital Twin
│       ├── neurosymbolic.py    # SIRS + qSOFA + HDC pattern fusion
│       ├── evolution.py        # Genetic algorithm treatment optimizer
│       ├── verifier.py         # Z3 SMT formal safety prover
│       ├── database.py         # Supabase client + SQLite fallback
│       ├── ble_receiver.py     # BLE wristband client (bleak)
│       └── simulator.py        # Physiological vitals simulator
├── frontend/
│   ├── index.html              # Real-time control console
│   ├── styles.css              # Premium dark glassmorphism UI
│   └── main.js                 # WebSocket client + Chart.js + Canvas HDC map
├── supabase/
│   └── schema.sql              # Full PostgreSQL schema for Supabase
├── verify_system.py            # Automated backend verification suite
└── .gitignore
```

---

## Getting Started

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd Sepsis
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Configure environment variables

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and add your Supabase URL and service role key
```

### 3. Run the database migration

In your [Supabase SQL Editor](https://supabase.com/dashboard), run the contents of `supabase/schema.sql` to create all required tables.

### 4. Run the automated verification suite

```bash
python3 verify_system.py
```

All tests should pass:
- ✅ HDC vector math (orthogonality, binding, bundling, continuous level encoding)
- ✅ Z3 SMT safety invariants (allergy checks, fluid caps, vasopressor limits)
- ✅ Evolutionary optimizer (genetic algorithm convergence and boundary safety)

### 5. Start the server

```bash
python3 backend/run.py
```

Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to access the real-time control console.

---

## Core Technologies

| Component | Technology |
|---|---|
| Hyperdimensional Computing | NumPy (custom VSA/HDC engine, D=10,000) |
| Formal Safety Verification | Z3 SMT Solver (`z3-solver`) |
| Evolutionary Optimization | Custom Genetic Algorithm (multi-objective) |
| BLE Wristband Receiver | `bleak` (GATT notification client) |
| Backend Server | FastAPI + Uvicorn + WebSockets |
| Database | Supabase (PostgreSQL) + SQLite fallback |
| Frontend Dashboard | Vanilla JS + Chart.js + HTML5 Canvas |

---

## Safety Invariants Verified by Z3

| Rule | Constraint |
|---|---|
| CHF Fluid Cap | `CHF ⇒ fluid_rate ≤ 5.0 mL/kg/hr` |
| Renal Fluid Cap | `Renal_Failure ⇒ fluid_rate ≤ 4.0 mL/kg/hr` |
| Vasopressor Max | `vasopressor_dose ≤ 0.4 mcg/kg/min` |
| Vasopressor Justification | `vaso > 0.1 ⇒ MAP < 65 mmHg` |
| Drug Allergy | `Penicillin_Allergy ⇒ antibiotic_code ≠ 1` |
| Sepsis Intervention | `Risk_HIGH ⇒ antibiotic_code ≠ 0` |

---

## License

Research and educational use only. Not for clinical deployment.
