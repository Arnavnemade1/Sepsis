# AegisOS Repo Map

This map is the pragmatic path through the repository: what is production-facing,
what is experimental, and what exists to preserve research history.

## Start Here

| Path | Use |
|---|---|
| `README.md` | Project overview, validated claims, quickstart. |
| `Makefile` | Common local workflows: setup, verify, serve, train, evaluate. |
| `verify_full_system.py` | End-to-end smoke and subsystem verification. |
| `backend/aegis/server.py` | Main FastAPI app and static `webui/` shell server. |
| `webui/` | Current AegisOS shell frontend. |

## Backend Surfaces

| Path | Status | Notes |
|---|---|---|
| `backend/aegis/` | Active runtime | Unified session, stream contract, body state, OS/kernel layer, cohort simulator, protocol proof path. |
| `backend/ml/` | Active ML subsystem | CinC data loader, causal features, calibrated GBM/stacked detectors, rigorous evaluation scripts. |
| `backend/physio/` | Active validation subsystem | HRV, PPG/ECG agreement, MIMIC-IV waveform adapter. |
| `backend/hostresponse/` | Offline research study | Pre-registered test of whether blood expression separates infection from STERILE inflammation across public GEO cohorts and array platforms (`make geo-manifest`, `make hostresponse PHASE=discovery|locked`, `make verify-hostresponse`). Never per-patient: no bed carries transcriptomics. Rules in `preregistration.json`; verdict and numbers in `artifacts/hostresponse.json`. |
| `backend/vision/` | Active sensor subsystem | rPPG, camera vitals, motion robustness, distress/perfusion signs. |
| `backend/llm/` | Optional narrative layer | Deterministic fallback is required; the system must not depend on LLM availability. |
| `backend/core/` | Legacy + primitives | Original HDC, simulator, neurosymbolic rules, Z3 verifier used by tests/runtime. |
| `backend/api/` + `frontend/` | Legacy demo | Simulator-era UI/API. Keep for comparison, but do not treat as the primary app. |

## ML Truth Table

| Component | Current role |
|---|---|
| `models/stacked.py` | The production detector: hgb, hgb2, et, lr and xgb@utility, trained on the patient label. `artifacts/detector_manifest.json` records its promotion; it ships as `artifacts/gbm_wearable.joblib`. |
| `models/gbm_baseline.py` | The calibrated single-GBM benchmark. |
| `models/hdc_detector.py` | Data-grounded HDC detector and interpretable HDC signals; not the raw-accuracy winner. |
| `models/augmented.py` | Documented negative result: HDC+GBM gain did not replicate under CV. |
| `quantum/` | Verified QHDC research sandbox; not production accuracy path. |
| `genesis/` | The all-variable research line on CinC 2019 (locked test set, hour-matched AUROC). Not on the serving path. See `docs/GENESIS.md`. |
| `mimic/` | The MIMIC-IV line, verified on the open demo (`make verify-mimic`); the full run waits on credentialed access. See `docs/MIMIC.md`. |
| `scripts/rigorous_eval.py` | Best judge-facing evaluation: cross-hospital, CIs, calibration, utility, decision curves. |

## Data And Artifacts

| Path | Policy |
|---|---|
| `data/cohort_sample/` | Small committed sample for smoke tests. |
| `data/cinc2019/` | Local PhysioNet cache; gitignored. |
| `data/bidmc/` | Local BIDMC cache; gitignored. |
| `artifacts/gbm_wearable.joblib` | Shipped detector artifact path loaded by runtime. |
| `artifacts/feature_cache/` | Regenerable cache; gitignored. |
| `artifacts/rigorous_eval.json` | Judge-facing evaluation output when regenerated locally. |

## Recommended Workflows

```bash
make setup
make verify
make serve
```

For model work:

```bash
make train-gbm
make train-stack
make rigorous
```

For a clean review, start with `verify_full_system.py`, then inspect
`backend/ml/README.md` before changing detector claims. That file carries the
most current research caveats and negative results.
