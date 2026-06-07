# AegisOS Core — Unified Critical-Care Runtime

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
