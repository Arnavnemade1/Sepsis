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

### The published numbers to beat

YAIB's paper (ICLR 2024, sepsis, AUROC × 100, mean ± its "std" field):

| Model | eICU | MIMIC-IV |
|---|---|---|
| LR | 71.8 ± 0.3 | 77.1 ± 0.4 |
| LightGBM | 69.1 ± 0.3 | 77.5 ± 0.3 |
| GRU | **77.4 ± 0.2** | **83.6 ± 0.3** |
| LSTM | 74.0 ± 0.2 | 82.0 ± 0.3 |
| TCN | 76.7 ± 0.1 | 82.7 ± 0.3 |
| Transformer | 76.2 ± 0.1 | 80.0 ± 0.8 |

The recurrent and convolutional models beat the tree models by 5-8 points. YAIB's tree
models see only the history's min, max, mean and count of each variable, so that gap
measures features rather than learners. Our learner is also a tree model, so this is the
gap our features have to close.

A comparison with this table is unpaired, and it rests on a cohort regenerated with
today's ricu. We regenerated YAIB's eICU demo with ricu at YAIB-cohorts' pinned commit and
got 873 of YAIB's 896 shipped stays, with 62 of 63 septic stays. The README of
YAIB-cohorts warns of this ("might differ marginally"). The paired claims use YAIB's own
code on our regenerated cohort, so both sides see the same rows.

### Model selection

The `rich` feature set (`--feature-set rich`) adds 6-hour extremes, 1- and 24-hour
changes, change from the stay's first value, and a SOFA proxy with its 24-hour change
and its rise above the stay's best value. Sepsis-3 is an acute SOFA rise, and no YAIB
model is handed one. Whether `rich` replaces `base`, and any learner setting, is chosen
on the validation AUROC of repetition 0's first folds, which every run now reports. The
test folds are scored once, with the chosen configuration.

### Baselines without re-tuning

YAIB tunes each model once, on its first folds, then fixes the result for all 25 runs. On
the 2-vCPU worker that search does not fit: logistic-regression tuning alone took 3 hours
on the 896-stay demo. So the baselines run YAIB's own code with the values of YAIB's own
published eICU models (YAIB-models, repetition 0 fold 0), fixed in
`backend/ml/yaib/gin/*_eicu_published.gin`:

| Model | Fixed values |
|---|---|
| LightGBM | 52 leaves, depth 7, min 1,000 rows per leaf, column sample 0.897, row sample 0.604 |
| LR | elastic net (l1 ratio 0.620), C 0.00198, saga, 100 iterations as published |
| GRU | 2 layers, hidden size 77, Adam lr 8.51e-5, weight decay 1e-6 |

The values were read without executing the published files: the pickle was
disassembled, and the torch file was loaded with `weights_only=True`.

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


## Resumable private experiments

Use `--resume-dir` instead of `--features` when a benchmark must survive worker stops:

```sh
python -m backend.ml.yaib.bench run /lake/yaib-cohorts/sepsis/eicu \
  --seed 1234 --feature-set base \
  --resume-dir /lake/yaib-work/checkpoints/base-fixed \
  --predictions /lake/yaib-work/ours --out /output/ours.json
```

Rerun the same command after interruption. The first call creates a private experiment
manifest, a validated row-major feature matrix, and one checkpoint after each completed
fit. Committed fits are loaded without training again; any missing prediction export is
regenerated from that fit's saved probabilities. Only the unfinished fit is repeated.
A feature build interrupted before its commit receipt is rebuilt, rather than trusted.
There is no resume within a single booster fit or unfinished matrix build.

The manifest fixes the content hashes of dynamic/static rows and labels, the feature
set, training configuration, cross-validation counts and seed, all YAIB module sources,
and the numeric-library versions. Changing these inputs refuses reuse and requires a
new directory. The requested run subset may grow (for example, from two development
folds to the full protocol) only with those fixed settings and original splits. Select
the feature set on validation AUROC only. Reuse does not authorize test-set selection.

Receipts, matrix data and per-run probabilities commit through atomic replacement and
file/directory synchronization. Checksums reject torn or changed files. A nonblocking
file lock prevents two coordinators from writing the same experiment. Before building,
free disk must cover both the column-major scratch and row-major matrix, plus receipt
headroom. Partial owned feature files are removed on a failed build. The independent
`aegis-heavy` lock still governs shared-host compute; checkpoint locking does not grant
a host reservation or defeat the six-hour cap of other experiment lanes.

Keep the entire resume directory and prediction exports inside the credentialed lake.
New resume directories are mode 0700; an existing directory that permits group or
other access is refused, without changing its permissions.
The JSON manifest contains hashes, not patient rows; its binary matrix and probability
files are nevertheless private derivatives. No patient artifact is sent to a hosted
model or committed to GitHub. For API calls, reload the cohort before each `run`: the
existing streamed path drops dynamic columns after building to release memory.

Verification uses invented patients only:

```sh
python -m backend.ml.scripts.verify_yaib
python -m backend.ml.scripts.verify_yaib_resume
```

The resume checks include interruption/replay, zero-fit completed recovery, feature
and probability corruption, edited validation receipts, changed input/configuration/
source/environment, concurrent writers, disk exhaustion before work, and parity of
real streamed LightGBM predictions and metrics. This infrastructure preserves training
semantics; it does not establish a clinical AUROC improvement.
