# The PhysioNet data platform

How credentialed data gets onto a machine, how we know it's the right data, and how anything identifiable is kept from leaving.

**Status:** built and verified on the open MIMIC-IV demo, including a live end-to-end run against physionet.org. The login works: on 2026-11-02 `access` read MIMIC-IV 3.1's 33-file hash list with a real credentialed account. No data file has been downloaded yet, so two things remain unproven: the real speed of a full download, and whether the size projection holds.

## Start here

```bash
make physionet-login USER_NAME=<your-physionet-username>
```

That prints a Keychain command. Run it yourself; it prompts on your terminal, so the password never reaches a shell history or an AI transcript. Your username or the email you log in with both work. Type the password rather than pasting it: a paste can carry a stray space. Then:

```bash
make physionet-doctor
```

```bash
make physionet-access
```

```bash
make physionet-plan
```

`doctor` checks the machine is fit to hold the data. `access` shows which datasets your account can really read, because each one has its own agreement. `plan` says whether it fits. If all three are fine:

```bash
make physionet-up
```

It's safe to stop and re-run at any point. It stops by itself at pre-registration, with the lock unread.

Output is deliberately quiet: table sizes, hash results and aggregates only. Model output goes to a log inside the lake.

## Which route

| | `lake` (default) | `bigquery` | AWS ([deploy/physionet](../deploy/physionet/README.md)) |
|---|---|---|---|
| Needs | ~22 GB free | a Google Cloud project linked on PhysioNet, ~17 GB free | an AWS account |
| Raw data on disk | none: streamed and filtered | none: hourly aggregates only | none: streamed to an encrypted disk |
| Contract checks | yes | no local tables, so SQL parity instead | yes |
| Best when | you have an external encrypted drive, or can free the space | you want the least local data | the laptop shouldn't be tied up |

**This laptop today:** 14 GB free. The lake and grid would just fit (a projected 11.3 GB with the reserve), but the features need about 6 GB more and the planner keeps its own 6 GB reserve, so the full run will refuse here. Use an external encrypted drive, free about 10 GB, or use AWS:

```bash
PHYSIONET_LAKE=/Volumes/<drive>/physionet-lake make physionet-up
```

## What runs

```
doctor → access → capacity → ingest → contract → canary → parity → plan → dev_run → release → HOLD
```

| Stage | Question it answers | It refuses when |
|---|---|---|
| `doctor` | Is this machine fit to hold the data? | disk unencrypted, lake in the repo or a synced folder, lake readable by others |
| `access` | Can this account read it? Is it the release we pinned? | no access, or the published hashes changed since first seen |
| `capacity` | Will it fit? | not enough disk, before a byte is downloaded |
| `ingest` | Are these exactly PhysioNet's bytes? | a SHA-256 mismatch or corrupt gzip; the table is then never published |
| `contract` | Does our catalog still fit this data? | a repointed itemid, a unit change, orphans, a partial cohort |
| `canary` | | records salted hashes of every real identifier, for the scanner |
| `parity` | Does the lake read like the reference? | under 99% agreement on the demo's patients (exact when the dataset is the demo itself) |
| `plan` | Which run can this machine afford? | no plan fits |
| `dev_run` | The full harness, on dev folds only | a failed model audit, per [MIMIC.md](MIMIC.md) |
| `release` | May this report leave the lake? | a real identifier, shifted dates or row-level columns |

A stage is skipped only if what it read hasn't changed. New data or new harness code reruns what depends on it, and every stage is written to a hash-chained journal beside the lake. One heavy job runs at a time; the lake's file lock refuses a second.

**The hold is deliberate.** Choosing claims and spending the lock is a person's decision, and the pipeline has no code path that does it. See [MIMIC.md §8](MIMIC.md).

## The lake

Full MIMIC-IV compresses to roughly 8 GB across the tables we read, and most of `chartevents` is documentation no model uses. So a table can flow straight from the socket:

```
HTTP → SHA-256 → gunzip → CSV parser → keep the catalog's items and columns → Parquet buckets
```

- **Exact.** `LakeSource` reuses every line of `CsvSource`'s binning and only changes where rows come from. The finished grid must equal the CSV path's byte for byte, and it does on the demo.
- **Bounded memory.** Rows go to `key % n_buckets`, and a stay's rows live in one bucket, so the grid is built one bucket at a time. No global sort, so no temporary disk.
- **Nothing unverified is published.** A table is written to a temporary directory and renamed into place only after its hash matches. A dropped connection resumes mid-gzip without the parser noticing; a killed process restarts only the unfinished table.
- **It can't go quietly stale.** A concept added to the catalog changes that table's `spec_hash`, and the lake refuses to serve until it's rebuilt.

Size figures are projections from the demo with a 3× margin. The real sizes replace them on ingest.

**Faster on a dedicated machine (`--parallel N`).** physionet.org caps each connection. Measured on 2026-11-04 with a 183 MB open file: one connection ran at 192–478 KB/s, four at 699–804 KB/s in total, six at 846. So `--parallel 4` fetches each table as four byte ranges, each from a fresh session. Every answer must be a 206 for exactly the range asked, from the same ETag. The joined file is hashed against the published SHA-256 before it's used, then ingested from the lake's own `staging/` directory and deleted whatever happens. A server that ignores ranges falls back to streaming. The default stays 1, so a laptop never holds a raw table; the AWS run uses 4. That's about twice as fast, and four connections, not more, out of courtesy to a free service.

## What building this found

These were all live in the code that was waiting for the data (the last was found with the first real login):

| Finding | Consequence if the data had simply arrived |
|---|---|
| BigQuery dataset names were `mimiciv_3_1_*`; PhysioNet publishes `mimiciv_v3_1_*` | day-one failure on every query |
| The two paths broke same-timestamp ties differently (arterial vs cuff pressure, two specimens): 0.93% of chart cells disagreed | passed the 0.99 parity bar at 0.9907, then two "identical" runs differ. Now one rule: lowest itemid, then highest value |
| Infusions starting after the 336-hour grid were clipped into hour 335 | a look-ahead leak in the reference path |
| pandas' default float parser is off by an ulp on some values | the paths could never match byte for byte. The reference now uses `round_trip` |
| The run wrote its report only at the very end | a crash at hour 8 of 9 cost all 8 hours. Now every fit is checkpointed |
| A cached store was keyed by tag alone | a new lake could silently reuse an old grid. The tag now derives from the lake's fingerprint |
| PhysioNet reads a login only from wget, the WFDB tools and its own Python client, and serves every other client as anonymous (`http_auth_allowed` in physionet-build) | a correct password got 403 on every credentialed file, from our client and from curl alike. Requests now go through the official `physionet` package's session |

The grid's contents changed, so `STORE_VERSION` is 5. Refreshed demo numbers moved noticeably (primary 0.755 to 0.790), and that is not an improvement: flipping the tie rule to its arbitrary opposite gives 0.733. See [MIMIC.md §1](MIMIC.md). A kill-and-resume test matched the uninterrupted report with no non-timing differences. These findings came from the new offline SQL check: `sqlcheck.py` transpiles our BigQuery SQL to DuckDB and runs it on the open demo. It can't prove BigQuery's own behaviour, table names or permissions; `parity` still has to run against the real service.

## The Data Use Agreement, as code

[PhysioNet's position](https://physionet.org/news/post/gpt-responsible-use): sending credentialed data to third-party LLM APIs is prohibited. It names Azure OpenAI (with human review opted out), Amazon Bedrock, Gemini on Vertex AI and Anthropic Claude as acceptable options.

**So in this repository:**

- OpenRouter and the consumer Gemini API (`backend/llm/`) never see a row. The deployed backend doesn't read credentialed data at all.
- No model, from any provider, gets rows through this tooling. Agents work from code, the demo and aggregates. That's stricter than PhysioNet requires, on purpose.
- Your outside models get code and aggregate reports, unless they run on one of the routes above.

**What's enforced:**

| Mechanism | What it does |
|---|---|
| Lake location | defaults to `~/physionet-lake`: outside the repository, private from creation, excluded from Time Machine and Spotlight |
| Doctor | refuses an unencrypted disk, the repo, Dropbox/iCloud/Drive/OneDrive paths, `~/Desktop` and `~/Documents` (iCloud can sync those without changing their path) |
| Pre-commit hook | scans the **staged bytes** of every commit. Installed automatically before the first credentialed byte |
| Identifier canary | salted hashes of every real `subject_id`, `hadm_id` and `stay_id`. Recognises a real identifier in any format, and reveals nothing itself. The demo's open identifiers are excluded |
| Shape rules | shifted dates (21xx, a calendar only MIMIC uses), row-level identifier columns, Parquet under a false name |
| Report release | a report reaches `artifacts/mimic/` only through the scanner |
| Agent fence | `.claude/settings.json` denies Claude Code's Read, Edit and Write tools on `~/physionet-lake`. It doesn't cover shell commands, and a custom lake path needs its own entry |
| Credentials | environment, Keychain or owner-only `.netrc`. Never argv, never written by us, hidden from `repr`, never sent in clear text or across a redirect to another host |

Counts under 10 in a report produce a warning, not a block. PhysioNet sets no number; 10 is this project's policy.

**What it can't do:** a scanner can't catch every way to leak something, and it can't stop someone reading a file and retyping it. The agreement still binds you personally, including reporting anything identifying to `PHI-report@physionet.org` (item 5) and publishing the code behind any released result (item 9). The agreement also says access is for research; whether a trained model may ship in a product is a question to put to PhysioNet before any MIMIC-trained model goes into the deployed app. Nothing here does that automatically.

## What the first full run found

MIMIC-IV 3.1 on a free-tier AWS machine (2 vCPU, 8 GB, 24 GB swap), November 2026. None of these showed on the demo:

| Finding | What happened | Now |
|---|---|---|
| A ratio can't judge a signed lab | Base excess moved −1 → 0 between the demo's median and the full one: sampling noise near zero, and a ratio of 0. The contract refused | Signed concepts compare the shift against 10% of their plausible range; positive ones keep ×2 |
| Releases revise patients | Parity refused at 0.977 (vasopressors) and 0.989 (IV antibiotics). Every value present in both agreed; 2 of the demo's 137 stays had been revised between 2.2 and 3.1 (20,398 → 20,245 raw rows) | Across releases, stays whose raw row counts differ are left out and reported; more than 5% refuses |
| float16 tops out at 65,504 | Ferritin, CK, NT-proBNP and MAP's 24 h variation overflowed to inf: 7,559 of 2.9 billion cells. The demo builds float32, so nothing saw it | Saturated at 65,504; `make mimic-demo-prod` runs the demo through the full run's exact path |
| CPU isn't idleness | A download throttled to 0.4 MB/s sits at 2–3% CPU, right on the idle alarm's line | The machine stops two hours after its last job; the CPU alarm is a 12 h backstop |
| The memory model was 40% short | 10.1 GB used where 7.3 was planned. LightGBM matched its estimate (3.5 vs 3.8 GB); the error was cold memory (every model's scores, the heap) that grows with the rows kept, at about 1,200 bytes a row. It sat in swap untouched | Planned as hot (must be RAM) and cold (may swap); column-wise histograms save about 20% per fit |
| Tuning was nearly flat | 32 trials: 0.8131 → 0.8163, every setting within 0.004 | 8 trials |
| A silent log isn't a stalled run | 6.5 h without a line while LightGBM fitted a tier; the plan said 29 h and the machine fitted a third slower | A numbers-only progress journal, an ETA in `status`, and the next plan calibrated from this machine's measured speed |
| The demo hadn't run the full run's path | Running it there (`make mimic-demo-prod`) crashed at once. LightGBM drops weights that are all exactly 1, and a validation set built by reference to a weighted training set is then left with weight 0: AUC reads 1.0 every round (early stopping follows a constant, silently) and a cross-entropy metric aborts. It needs a validation slice of only septic stays, so it took a small fold; the full run's slices always hold weighted never-septic stays | All-1 validation weights are handed over scaled by 1 + 2⁻²⁰, which changes no AUC or weighted mean |

## Watching a run

`make physionet-status` (or `run.sh status` on AWS) prints each stage, then the modelling run's progress from its own journal. The journal is one line per fit with counts, seconds and memory only, never a value:

```
  modelling: 54% of the planned fitting (612 fits, 3 resumed), 14.2 h in, last fit 12 min ago, 5.4 GB RAM + 4.7 GB swap
             about 13.1 h to go (twin included), around 2026-11-05 14:40 UTC
```

Progress counts **work**, not fits: rows × columns, the planner's own cost model. A run makes hundreds of tiny audit fits beside about 80 big ones; on the demo, 95 of 815 fits were 92% of the work. Counted as fits, the plan was overrun almost at once. A finished demo run had done 81% of its planned work, so the ETA errs long.

The same journal calibrates the next plan on the same machine: measured over predicted time, summed over its fitted boosters so the big ones dominate, after at least five.

## Crash recovery

Inside a run, each booster fit and twin is checkpointed by **what it read**: a content sample of the features, rows, target, weights, parameters, seed, and the harness code. A re-run walks back to where it stopped. A cached model predicts bit for bit what the fresh one did, and a counterfactual world with edited values never reuses a model. Changing harness code invalidates every checkpoint, by design.

## Other datasets

```bash
make physionet-access
```

```bash
PYTHONPATH=. .venv/bin/python -m backend.ml.physionet --dataset mimic4wdb fetch --list 'waves/p100/*'
```

`fetch` downloads whole files with the same verification, for what the lake doesn't model: MIMIC-IV-Note, eICU, ED and waveforms. **It's acquisition only.** Extractors and models for those datasets are not built. The catalog records which open demo verifies each one first. MIMIC-IV-Note has no demo, so an extractor for it can't be checked before it touches real data.

## Verifying this

```bash
make verify-physionet
```

The real service is impersonated by a local server that misbehaves on request: dropped connections, short bodies, ignored ranges, flipped bits, files that change mid-transfer, login pages served as 200. Everything downstream runs on the open demo. Each mechanism is then broken on purpose, to confirm its check goes red.

```bash
make physionet-smoke
```

This one is live: real physionet.org, open data, no credentials, a throwaway lake.

`--no-mutations` skips the slow part, and it's what CI runs on each push (`.github/workflows/verify-physionet.yml`).

| Module | Job |
|---|---|
| `backend/ml/physionet/catalog.py` | datasets, versions and access classes |
| `.../auth.py` | where a password comes from |
| `.../fetch.py` | verified transport |
| `.../lake.py` | csv.gz to bucketed Parquet |
| `.../guard.py` | the posture doctor, the canary, the egress scanner, the hook |
| `.../pipeline.py` | stages, gates, resume, journal |
| `backend/ml/mimic/lake.py` | `project()` (what the lake keeps) and `LakeSource` |
| `backend/ml/mimic/contracts.py` | is this the data we think it is |
| `backend/ml/mimic/sqlcheck.py` | the BigQuery SQL, run without BigQuery |
| `backend/ml/mimic/checkpoint.py` | crash recovery |
