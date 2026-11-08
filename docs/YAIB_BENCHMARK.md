# Our model under YAIB's protocol

[YAIB](https://github.com/rvandewater/YAIB) ("Yet Another ICU Benchmark", van de Water et
al., ICLR 2024) defines an hourly sepsis task on MIMIC-IV, eICU, HiRID and AmsterdamUMCdb.
All four datasets share one cohort format, and the benchmark publishes baselines. YAIB's
own finding is that the cohort and preprocessing move results more than the choice of
model does. A number is therefore comparable only if it was made on YAIB's cohort, YAIB's
splits and YAIB's metric. `backend/ml/yaib/` makes that true for our model.

No result on credentialed data is claimed here. The only numbers below come from YAIB's
open eICU demo, where they show that the pipeline works, not that one model beats another.

## What is YAIB's, and what is ours

| Piece | Source |
|---|---|
| Cohort (`dyn` / `sta` / `outc` parquet) | YAIB's cohort generation. It is read as is: nothing is added, dropped or relabelled. |
| Splits | YAIB's `make_single_split_polars`, reproduced in `splits.py` |
| Metric | YAIB's: AUROC and AUPRC over every labelled test hour, per run; then the mean, the field YAIB labels "std", and a t interval over the 25 runs |
| Baselines | YAIB itself, run by `run_baseline.py`. The driver only turns on saving per-hour predictions. |
| Features and learner | Ours: causal hourly features from the same 48 dynamic and 4 static variables, then LightGBM |

## Evidence that the protocol is YAIB's

| Check | Result |
|---|---|
| Our split vs YAIB's own function, every (repetition, fold) | 25/25 identical on the eICU demo and on the synthetic cohort (YAIB af9a5b3, seed 1234) |
| Under YAIB's scikit-learn 1.5.2 and our 1.6.1 | Identical split fingerprints |
| YAIB's logged mean AUROC, recomputed from its saved predictions | 0.6495 = 0.6495 |
| Saving predictions changes YAIB's result | No (0.6495 with and without) |

`verify_yaib` pins three of those fingerprints, so CI checks the split against YAIB without
YAIB installed. When YAIB is installed, it also runs the live 25-run conformance check.

## Two things to know about YAIB's output

1. **Its "std" is a standard error.** `aggregate_results` reports the population SD over
   runs divided by √n. We report the same field under the name `*_yaib_std_field`, keep
   the plain SD as `*_sd_over_runs`, and add YAIB's t interval.
2. **Its saved row ids are out of order.** In `pred_indicators.csv`, the labels and
   predictions are in (stay, hour) order, but the stay/hour ids written beside them are
   not. The polars loader takes the ids before it sorts the features. YAIB's own metric is
   unaffected, because the labels and predictions stay aligned. But anyone who joins on
   the written ids pairs 324 of 7,970 eICU-demo rows with another hour's label.
   `bench.compare` therefore pairs rows by sorted position. It accepts that pairing only
   when the two id sets are identical and every label agrees position by position, and it
   refuses anything else.

## The eICU demo (pipeline check, not a claim)

896 stays, 63 of them septic, seed 1234, 5 × 5 runs:

| Model | AUROC mean | YAIB "std" |
|---|---|---|
| YAIB LightGBM (YAIB's own run) | 0.649 | 0.021 |
| Ours (LightGBM, our features) | 0.661 | 0.015 |
| Paired, ours − YAIB, bootstrap over test stays | +0.011 [−0.028, +0.050] | |

The interval includes zero, so there is no difference to report. That is expected with 63
septic stays.

The paired test was also calibrated on null data. On 200 synthetic seeds where neither
model carries signal, its 95% interval excluded zero 7% of the time, close to the nominal
5%.

## The bar, fixed before any credentialed run

For each dataset, we call our model better than a YAIB baseline only if the paired AUROC
difference (bootstrap over test stays, every run) has a 95% interval entirely above zero.
The baseline is YAIB's own run of that model, with YAIB's tuning.

"Best" means better than every YAIB baseline we ran on that dataset. A tie or a loss is
reported as it comes out.

## Running it on the AWS box

This step needs the credentialed MIMIC-IV and eICU data, the eICU data-use agreement, and
YAIB's cohort generation, which runs in R (ricu).

```
# 1. YAIB at the pinned commit, in its own environment
git clone https://github.com/rvandewater/YAIB && git -C YAIB checkout af9a5b3
pip install -e YAIB

# 2. YAIB's baselines, with per-hour predictions (run from inside the YAIB checkout)
python -m backend.ml.yaib.run_baseline train -d <cohort> -n miiv -t BinaryClassification \
    -tn Sepsis -m LGBMClassifier --seed 1234 --tune -l <logs>

# 3. ours, on the same cohort and seed (predictions stay in the lake)
python -m backend.ml.yaib.bench run <cohort> --seed 1234 --out ours.json --predictions <lake>/ours

# 4. the paired comparison (aggregates only)
python -m backend.ml.yaib.bench compare --ours <lake>/ours --yaib <logs>/miiv/Sepsis/LGBMClassifier \
    --out compare.json
```

Memory: the feature matrix is 443 float32 columns, about 1.8 kB per stay-hour, and a fit
also holds the training rows' copy and LightGBM's binned data. Price a cohort before
running, and keep the matrix on disk on the 7.6 GB worker:

```
python -m backend.ml.yaib.bench plan <cohort> --features     # planned peak vs MemAvailable
python -m backend.ml.yaib.bench run <cohort> ... --features <lake>/X.npy
```

`run` refuses up front when the planned peak exceeds 85% of available memory. One heavy
job at a time.

Compute: YAIB's LR and LightGBM baselines run on the CPU worker. Its neural baselines
(GRU, LSTM, TCN, Transformer) with `--tune` were trained on GPUs. Their cost on the
2-vCPU worker is unmeasured: time them on the open demo first and scale by row count.
If a tuned 25-run protocol does not fit the time available, compare paired with LR and
LightGBM, and cite YAIB's published neural-model numbers as published (not paired, and
labelled so).
