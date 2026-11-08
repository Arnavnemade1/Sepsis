"""Our model under YAIB's protocol.

YAIB ("Yet Another ICU Benchmark", van de Water et al., ICLR 2024; github.com/rvandewater/YAIB)
defines an hourly sepsis task (Sepsis-3 onset within the next hours) on MIMIC-IV, eICU,
HiRID and AmsterdamUMCdb, with one shared cohort format and published baselines. Its own
finding is that cohort and preprocessing move results more than the model does, so a
number is comparable only when it was produced on THEIR cohort, THEIR splits and THEIR
metric. This package exists to make that true for our model:

    cohort    reads a YAIB cohort directory (dyn / sta / outc parquet) as it is
    splits    YAIB's split, reproduced (sorted stay ids, stratified outer K-fold for the
              test set, stratified inner K-fold for train / validation, the run's seed),
              with a conformance check against YAIB's own function when it is installed
    features  our causal hourly features from the same 48 dynamic + 4 static variables
    bench     the 5 x 5 runs, the metric YAIB computes (AUROC / AUPRC over every labelled
              test hour of a run, then mean and std over runs), and a paired comparison
              against YAIB's saved per-hour baseline predictions on identical rows

Baselines are never reimplemented here: they come from running YAIB itself.
"""
