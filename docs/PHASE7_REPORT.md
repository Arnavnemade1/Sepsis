# Phase 7 — Accuracy & Generalization Report

Goal: push detection accuracy as far as the data and method honestly allow, and
measure how the model generalizes to a *new hospital* — before any frontend work.
All results are leakage-free (patient-grouped), calibrated, on PhysioNet CinC 2019.

## TL;DR

- **Production model:** full-data, calibrated wearable GBM — within-distribution
  **AUROC 0.774**, **ECE 0.0008**.
- **Cross-hospital (the honest deployment number):** **AUROC ≈ 0.69** when trained
  on one hospital and tested on the other — and **calibration holds** (ECE ≈ 0.005).
- **The bottleneck is not the model.** Labs add only ~0.02 AUROC, and richer
  temporal features add nothing. Channel availability and cross-site distribution
  shift dominate.

---

## 1. Temporal feature expansion — tested, not adopted

Added multi-scale causal context (extra rolling windows 4/16/32 h, multi-horizon
lags, expanding extremes, cumulative deterioration burden): 94 → 185 features.

| Feature set | AUROC | AUPRC | ECE | Utility |
|---|---|---|---|---|
| wearable (94) | 0.7273 ± 0.013 | 0.0705 | 0.0033 | 0.320 |
| wearable_temporal (185) | 0.7336 ± 0.010 | 0.0695 | 0.0023 | 0.293 |

*6,000 patients, 4-fold CV.* The +0.006 AUROC is within noise and AUPRC/utility
are flat-to-worse, for ~2× the compute. **Conclusion: not adopted as default.**
The base features already capture the temporal dynamics. (The code remains
available via `FeatureConfig.wearable_temporal()` for future use with labs/more data.)

## 2. Lab-augmented ceiling

| Feature set | AUROC | AUPRC | ECE | Utility |
|---|---|---|---|---|
| wearable (94) | 0.7281 ± 0.014 | 0.0661 | 0.0031 | 0.281 |
| lab-augmented (214) | 0.7474 ± 0.010 | 0.0687 | 0.0032 | 0.321 |

*8,000 patients, 4-fold CV.* Adding sparsely-sampled labs (lactate, WBC,
creatinine, …) lifts AUROC ~0.02. Real but modest — **wearable-only detection is
close to the clinical ceiling for this approach**, so the lack of labs is not the
main limitation.

## 3. Cross-hospital external validation ⭐

CinC set A = Beth Israel Deaconess (BIDMC), set B = Emory. Train on one hospital,
test on the *other* — the strongest generalization test (different population,
equipment, charting).

| Train → Test | AUROC | AUPRC | ECE | Utility | Early-warning lead |
|---|---|---|---|---|---|
| BIDMC → Emory | 0.680 | 0.045 | 0.004 | 0.176 | 36 h |
| Emory → BIDMC | 0.698 | 0.053 | 0.007 | 0.195 | 35 h |
| **mean** | **0.689** | — | — | — | — |

**Interpretation.** Discrimination drops from ~0.77 (within-distribution) to
~0.69 across institutions — a real, well-documented generalization gap that an
honest project must report. Crucially, **probability calibration survives the
domain shift** (ECE stays ≈ 0.005), so the model stays *trustworthy* (its stated
probabilities remain accurate) even where it is less *discriminative*. The
median early-warning lead time (~35 h) also holds across sites.

---

## What this means for the build

- The deployed AegisOS detector should advertise the **cross-hospital** number
  (~0.69 AUROC, well-calibrated), not the optimistic within-hospital 0.77.
- Closing the gap is a *data/representation* problem, not a model-capacity one:
  the genuine next levers are (a) **real continuous HRV** (the camera/wearable
  signal CinC lacks — needs credentialed MIMIC-IV Waveform, adapter already
  built), (b) **multi-site training** to reduce domain shift, and (c)
  **temporal/sequence models trained with labs**, where the extra context may
  finally pay off. These are flagged for Phase 7+ / future data access.

## Reproduce

```bash
python -m backend.ml.scripts.phase7_eval --limit 8000 --folds 4 \
    --feature-sets wearable lab            # ceiling comparison
python -m backend.ml.scripts.external_validation --feature-set wearable  # cross-hospital
```
Artifacts: `artifacts/external_validation_wearable.json`, `artifacts/phase7_results.log`.
