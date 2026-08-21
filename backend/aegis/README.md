# AegisOS — Unified Critical-Care Operating System

`AegisOS` (`os.py`) is the top-level OS that **boots and owns the layers**;
`AegisCore` (`core.py`) is the runtime it composes. Use `AegisOS` as the single
entry point — it adds the OS framing + the predictive what-if scheduler on top of
the core assessment pipeline.

```python
from backend.aegis.os import AegisOS
osys = AegisOS(use_llm=False)
for layer in osys.boot():                 # 8-layer boot sequence + status
    print(layer.index, layer.name, layer.ready)
a = osys.assess({"heart_rate": 128, ...}, source="wearable", whatif="auto")
a.risk_level, a.body_state, a.whatif       # detection + body model + projections
osys.status()                              # live OS layer map (for the UI)
```

### The 8 layers (`os.py`)
HAL · Hyperdimensional Kernel · Digital-Twin + What-if Scheduler · Neurosymbolic
Reasoning Plane · Evolutionary Synthesis · Formal Verification (Z3) · Persistence ·
Interface. `status()`/`layers()` expose them live.

### Multi-agent reasoning plane (`agents.py`)
Five explicit agents examine each tick through one clinical lens and state an
opinion with a confidence and **named evidence**: **Cardiac**, **Inflammatory/
Immune**, **Infection/Pathogen**, **Metabolic/Organ**, and a non-scoring **Safety
Oracle**. They run a transparent one-round *debate* (independent corroboration
raises confidence), reach a confidence-weighted **consensus** with an agreement/
dissent metric, and the Safety Oracle can **escalate** the risk tier on hard
danger criteria (never downgrades). The full debate ships in every snapshot
(`reasoning`) — explainable by construction.

### What-if Twin Scheduler (`scheduler.py` + `physiology.py`)
On HIGH risk (or on demand via `osys.whatif()`), projects the twin forward under
candidate treatments (no-op, antibiotics-only, standard bundle, vasopressor-
forward, and the GA+Z3 *recommended* plan), scores each projected risk trajectory
with the neurosymbolic engine, Z3-verifies safety, and recommends the best safe
option. The forward physiology is an **illustrative** mechanistic model (clearly
labelled) for decision *exploration*, not a validated forecast.

---

## AegisCore — the runtime layer

`AegisCore` is the single coherent brain that ingests streaming telemetry and
produces a complete assessment. It is dashboard-independent — drive it from a
test, a CLI, or the API.

```
telemetry (wearable BLE / camera rPPG)
   -> PatientDigitalTwin           (HDC state + personal baseline + trends)
   -> StreamingFeatureBuilder      (train/serve-consistent causal features)
   -> calibrated trained detector  (backend/ml, real CinC-grounded model)
   -> NeurosymbolicEngine          (transparent SIRS/qSOFA safety cross-check)
   -> unified risk fusion          (ML drives tiers; rules for hard overrides)
   -> BodyState model              (organ systems / regions / channels)
   -> EvolutionaryOptimizer + Z3   (verified treatment protocol on HIGH)
   -> AlertExplainer               (LLM-optional clinician narrative)
```

```python
from backend.aegis.core import AegisCore
core = AegisCore(use_llm=False)                  # loads artifacts/gbm_wearable.joblib
a = core.ingest({"heart_rate": 122, "spo2": 90, ...}, source="wearable",
                camera_vitals={"perfusion_index": 0.4})
a.risk_level, a.ml_probability, a.body_state     # everything a UI needs
```

Degrades gracefully: with no trained model file it runs on the neurosymbolic
engine alone.

## Body-state model (`body_state.py`) — visualization data layer

`build_body_state(...)` (attached to every assessment as `a.body_state`) is the
backend functionality behind the planned anatomical "digital twin" view. It is
pure data — **no rendering** — and is JSON-serializable for any frontend.

It produces:
- **organ_systems** — Cardiovascular, Respiratory, Neurological, Autonomic,
  Peripheral Perfusion (monitored from wearable/camera) + Renal, Hepatic,
  Coagulation (lab-only). Each has a status tier, 0–1 severity, human-readable
  drivers, SOFA points where applicable, and a `monitored` flag.
- **regions** — anatomical keys (heart, lungs, brain, kidneys, liver, skin,
  blood) with aggregated severity/status, for mapping onto a body model.
- **channels** — signal streams (HR, HRV, SpO2, RR, BP, MAP, temp, lactate +
  optical perfusion/stress) with value, normal range, status, and source tag
  (wearable / camera / lab / derived).

**Honesty rule baked in:** systems/regions that need labs we don't have are
reported `UNMONITORED` — the body view must show "no data", never a false green.

> The frontend that renders this (3D/2D body + live signal overlays) is
> intentionally **not implemented yet** — only the functionality is built, so the
> visualization can be layered on once the whole backend is finalized.

---

## Scenario Lab (`scenarios.py`) — the AI poses the case, the OS gets graded

```bash
make scenarios          # run the standing suite against the shipped detector
make verify-scenarios    # the adversarial audit (10 checks)
```

Dr. Aegis could already read every surface of this OS and drive its actuators. It
could not **pose a situation**: build a patient who deteriorates in a particular way,
run the system against them, and see whether the OS did the right thing. The lab is
that missing half plus the sandbox it runs in — seven agent tools
(`scenario_vocabulary`, `create_scenario`, `run_lab_scenario`, `run_scenario_suite`, …)
over eight `/aegis/scenarios/*` routes.

A scenario is a small declarative spec — arc, onset hour, severity, ramp, lab
sparsity, interventions, seed — synthesized into a `PatientRecord` that flows through
the **real** feature pipeline and the **shipped** detector artifact at the **shipped**
operating point. Spec + seed is a complete description, so a failing case is
reproducible from the spec alone.

**The line this cannot cross.** A synthetic patient is written by hand, by us, so
anything measured on one measures our generator. This project has already been burned
by exactly that — the original detector looked flawless on a simulator tuned alongside
it and scored ~chance (AUROC 0.64) the first time it met real patients. So the split
is enforced, not remembered:

- Scenarios grade **behaviour** — did the alarm fire, how early, did the confounder
  stay quiet. A hand-written patient is a legitimate probe for that, like any test input.
- Scenarios never touch **accuracy**. `ScenarioResult` has no AUROC/AUPRC/utility field
  and no place to put one; synthetic records carry `set_name="synthetic"` forever; and
  `build_dataset` **refuses them** unless a caller opts in explicitly, in one line, in
  the diff. Accuracy stays in `docs/FRONTIER_REPORT.md`, on real patients.

The suite is half deteriorations the OS must catch and half confounders it must ignore
— a suite of only sick patients scores a detector that alarms on everyone as perfect.

**Current standing result: 7 / 8.** `cold-sepsis` fails and is *meant to*. That case is
unambiguously shocked (qSOFA 2, MAP 60, SBP 88, lactate 5.3) but hypothermic at 35.3 °C;
the shipped detector peaks at ~0.033 against a 0.055 alarm line and never fires, while
`textbook-shock` — differing essentially only in the **sign of the temperature** — peaks
at 0.101 and alarms 4 h early. That is a **hypothesis about fever-leaning raised on a
patient we wrote**, not a measured weakness. It is settled on real hypothermic septic
patients in CinC 2019, a run nobody has done yet. Do not "fix" it by relaxing the
expectation: the check failing is the check working.

`treated-early` / `untreated-late` is a matched pair (identical seed, one treated 10 h
pre-onset) recorded but deliberately **ungraded** — the treated arm currently peaks
*higher*, plausibly an artifact of how the generator applies an intervention to the
slope features, and asserting a direction nobody has established on real data would
bake a guess into the suite.
