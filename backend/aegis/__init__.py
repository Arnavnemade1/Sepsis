"""AegisOS core — the unified critical-care monitoring runtime.

This package is the single coherent brain that the wearable and backend share.
It binds together every subsystem the project has built into one streaming
pipeline:

    telemetry (wearable BLE / camera rPPG)
        -> PatientDigitalTwin            (per-patient HDC state + baseline + trends)
        -> StreamingFeatureBuilder       (train/serve-consistent causal features)
        -> calibrated trained detector   (backend/ml, real CinC-grounded model)
        -> NeurosymbolicEngine           (transparent SIRS/qSOFA safety cross-check)
        -> EvolutionaryOptimizer + Z3    (verified treatment protocol synthesis)
        -> AlertExplainer                (LLM-optional clinician narrative)

Everything is dashboard-independent: AegisCore is a plain Python object you can
drive from a test, a CLI, the FastAPI app, or (eventually) the on-device runtime.
"""
