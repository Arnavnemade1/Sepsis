"""GENESIS — the all-variable, label-genesis sepsis architecture.

The previous detection line (``backend/ml/models``) was built for a wrist wearable:
seven vital signs plus demographics. That is the honest product constraint, and every
number it published is a wearable number. This package answers a different question —
*how far can hourly early-warning go on CinC 2019 when the model may use everything the
ICU record contains* — and keeps its evaluation separate so the two are never mixed.

Layout:
    rawstore  — every patient packed into one array (loads in < 1 s)
    features  — vectorized, strictly causal features over all 40 variables
    harness   — the fixed-fold referee on the full two-hospital cohort, with the
                between/within AUROC decomposition and patient-bootstrap comparison
"""
