"""MIMIC-IV line: the GENESIS findings carried to a dataset that records the whole of Sepsis-3.

CinC 2019 taught four things that shape every design choice here (docs/GENESIS.md):

1. The tuned booster's score is SUFFICIENT for CinC's 40 variables — only new information
   can move the number. MIMIC-IV adds the three SOFA organ systems CinC cannot score
   (Glasgow Coma Scale, urine output, vasopressors) and the suspicion record (cultures,
   antibiotics).
2. Dataset construction leaked the label through record length (never-septic stays capped
   at ~60 h). Here the cohort is built by us, with one censoring rule for every stay, and a
   tripwire that fails the build if length alone predicts the label.
3. A model that reads clinicians' orders detects their suspicion (the Epic Sepsis Model
   criticism). Every variable carries an input TIER — P physiology, T treatment,
   S suspicion — and the harness refuses to let a lower-tier model see a higher tier.
4. Pooled AUROC hides a clock and a patient-vs-timing split: every number is reported with
   its hour-matched AUROC and its between / within decomposition.

Modules:
    concepts   the variable catalog (itemids validated against MIMIC's own dictionaries)
    sources    data access: the open demo / full CSVs, or BigQuery with server-side hourly binning
    extract    events -> the hourly grid (a RawStore the GENESIS stack runs on unchanged)
    sepsis3    hourly SOFA (all six organ systems), suspicion of infection, Sepsis-3 onset
    ledger     the Ledger harness
    twin       the Sepsis-3 Process Twin — the temporal model
"""
