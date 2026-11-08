# System replay

Every result this project has published so far is a model result: AUROC, or detection
at a fixed alert rate. A hospital does not run a model. It runs a paging system, and the
paging system decides what reaches a person. `backend/ml/system_replay.py` measures that.

It has not been run on MIMIC yet. No result is claimed here.

## What it does

Each held-out ICU stay is replayed hour by hour, with the detector's out-of-fold score
for that hour, through two arms. Both arms see the same scores, so any difference between
them comes from the system, not the model.

| Arm | What pages |
|---|---|
| `bare` | Every upward crossing of the alarm line. This is a model behind a threshold, with nothing around it. |
| `system` | The paging OS's detector episodes. This is `paging/detector_episode.step`, the function the live engine runs, imported rather than copied. It closes an episode only after three readings below the clear line, sends one page per patient while that page is being handled, and raises P2 to P1 on Sepsis-3 shock criteria. |

The scores are placed on the ward's risk index with the same `IndexMap` the live ward
uses, pinned at the operating point. Without a given operating point, the lines sit at a
fixed share of all replayed hours (HIGH at 5%, MEDIUM at 15%). That choice is label-free,
so it reads nothing about which stays became septic.

## What it reports

| Metric | Meaning |
|---|---|
| Pages per 100 patient-days | The alert load a unit carries. |
| Sensitivity | Share of septic stays paged within the 48 hours before onset. |
| False pages per caught case | Pages on never-septic stays, plus pages earlier than the lead window, for each septic stay caught. This is the alert-fatigue headline. |
| Warned ≥ k h (k = 1, 3, 6) | Of the septic stays observed at least k hours before onset, the share paged at least k hours early. The fixed denominator stops stay length from posing as lead time. The median warning is also reported and labelled as caught-only. |
| P1 share, late share, pages while uncertain, upgrades | What the pages are, and when they fire. |
| Page-handling sensitivity | The system arm rerun with a page counted as handled after 1, 4 and 12 hours. |

## The bar, fixed before any held-out run

The system counts as better than the bare threshold only if both of these hold on a paired
bootstrap over subjects (1,000 draws):

1. False pages per caught case falls: the 95% interval of (system − bare) is below zero.
2. Sensitivity is non-inferior: the interval's lower end is at or above −0.02.

Sending fewer pages is mechanically easy, because a system can simply never page again.
The second condition is what stops a quiet system from winning. If the system misses this
bar, it is reported as a miss.

## What it cannot measure

- **Routing.** Who gets paged runs on a synthetic roster. MIMIC records no pager, so
  "the right clinician was paged" has no ground truth and is not reported.
- **Acknowledgement and escalation timing**, for the same reason. In the replay nobody
  acknowledges a page, so a page counts as "being handled" for `page_active_h` hours
  (default 4). That assumption appears in every result, together with the sensitivity
  analysis over it.

## Guards

`make verify-replay` runs 33 planted-failure checks, and `verify-os`, which CI runs on
every push, includes them. The checks are:

- **Conformance.** The live `PagingOS` and the replay are fed the same 16 readings,
  including a resolved page and two shock upgrades. They must open the same pages at the
  same final priorities, with the same number of upgrades. Changing the engine's clear
  count must change the replay as well.
- **Crediting.** A page earlier than the window counts as false. A page after onset counts
  as late, which is neither false nor a catch.
- **Lead-time denominator.** A stay observed only 2 hours before onset cannot count toward
  "warned ≥ 3 h".
- **Verdict.** A planted quiet system that loses a third of its catches is refused on
  sensitivity. Episodes on hovering noise are accepted.
- **Release.** Counts under 10 are written as `<10`, the rates that depend on them are
  withheld, and no subject id leaves.

Each guard was also mutation-checked. Each of these bugs turns the intended check red: a
drifted copy of the recross rule, late pages credited as catches, and a verdict without
the sensitivity condition.

## Running it on the AWS box

The replay needs the full hourly series of each held-out stay, plus a score from a model
that never trained on that stay:

1. Inside the lake, build the stays with `stays_from_ledger(led, oof)` on the existing
   dev folds and write them with `export_rows(path, stays)`. These are patient rows: they
   stay in the lake.
2. `python -m backend.ml.system_replay <rows.npz> --out <agg.json>` writes aggregates only.
3. Run the guard scan, then release the JSON.

`stays_from_ledger` refuses a design that sampled rows within a stay. The full run sampled
whole stays (never-septic stays kept at a rate, with a weight of 1/rate), and the weight
carries into every metric.
