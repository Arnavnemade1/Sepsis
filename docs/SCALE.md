# Scale — how many beds this OS actually runs

**Question.** AegisOS had only ever been demonstrated at twelve beds, because
`build_cohort` defaults to twelve and nothing had asked for more. "I built an
operating system for critical care" and "it runs a 250-bed hospital on one core" are
different claims, and only the second is a claim about scale.

**Method.** `backend/aegis/scale/loadgen.py` drives the **real** `CohortCluster.tick()`
over real CinC patients — device bus, memory fabric, fusion engine, per-patient
learner, HDC associative memory, cross-modal resonance gate, Z3-verified cognitive
scheduler. No mock anywhere in the path. It times each phase of the tick separately,
with the instrumentation living in the measurement code so the clinical path carries
no profiling hooks.

```bash
python -m backend.aegis.scale.loadgen --beds 1,50,100,250,400 --ticks 25 --repeats 3
```

---

## Result

Median of 3 rebuilt runs per level, one core of an ARM laptop. Tick budget 600 ms.

| beds | tick p50 | run spread | tick p99 | ms/bed | peak RSS | dominant phase |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.6 ms | ±3% | 1.7 ms | 1.57 | 148 MB | sched_decide 54% |
| 50 | 66.6 ms | ±8% | 71.7 ms | 1.32 | 575 MB | resonance_gate 35% |
| 100 | 132.7 ms | ±6% | 140.4 ms | 1.31 | 1325 MB | resonance_gate 34% |
| 250 | 372.4 ms | ±3% | 391.6 ms | 1.44 | 2239 MB | hdc_memory 38% |
| 400 | 535.9 ms | ±6% | **624.0 ms** | 1.29 | 2541 MB | hdc_memory 44% |

**250 beds within the real-time budget at p99.** 400 beds fits on the median (536 ms)
and breaches at p99 (624 ms), so the crossing lies between the two measured levels.
The **per-bed cost is flat** — 1.29 to 1.57 ms across a 400× range in ward size.

---

## The bottleneck was not the prover

The plan for this work committed, in advance, to "breaking the Z3 wall": a certificate
cache and a solver process pool, on the reasoning that every SMT call in the OS
serialises through one process-wide lock. Profiling said that was wrong.

Scheduler verification runs **once per tick**, not once per bed, so its share of the
tick *falls* as the ward grows: **53% at one bed, 20% at five hundred.** The lock also
already takes a non-blocking acquire and defers when contended, and `VERIFY_EVERY`
throttles it further — the tick loop was designed around exactly that hazard. Both
planned optimisations would have bought close to nothing at the scale that matters.

The measured cost was **HDC associative retrieval**: every bed querying, separately,
a case store that itself grows with the ward. Quadratic in ward size, and 83% of the
tick at 100 beds. Two fixes:

- **One GEMM for the ward.** B separate `(N,D)@(D,)` matvecs are B passes over the
  same N·D floats. One `(N,D)@(D,B)` GEMM reads the matrix once. Results identical.
- **`argpartition` instead of a full sort.** `query` sorted the entire store to take
  the top 16. Exact (200/200 identical to the full ordering) but worth only ~2% —
  ranking was never the cost. Kept because it is free and correct.

100-bed tick: **454 ms → 133 ms.** Per-bed cost: **climbing (1.40 → 4.59 ms) → flat.**

## The hard ceiling behind it

`MemFabric`'s arena is a fixed `1 << 20` cells (8 MiB) regardless of how many beds are
mounted. At a measured 6,144 cells per bed that capped the OS at **170 beds** — and it
surfaced as a `MemoryError` out of a bump allocator during boot, not as a capacity
anyone could read in advance. The arena is now sized to the ward
(`cohort.arena_cells_for`), and the allocator's error names the fix.

## A correctness bug the profiler surfaced

The batched retrieval disagreed with the per-bed one on 14 of 30 beds. The cause was
not the batching.

`HDCEngine.bundle()` resolved ties — dimensions summing to exactly zero — by drawing
**fresh randomness on every call**, so `encode()` was non-deterministic. Ties occur
only when an **even** number of features contribute, which is to say **on beds with
missing data**: 16 of 30 beds on a real cohort, ~12% of each of their vectors,
`top_sim` swinging 0.705–0.721 across identical queries.

The associative memory was returning different precedents for an unchanged patient,
concentrated on the least-instrumented ones. Tie-breaking is now a fixed pattern drawn
once, from its own generator so every previously generated id/level vector stays
bit-identical.

> **Open:** the memory's published retrieval AUROC of 0.626 was measured through that
> noise. It needs re-measuring, and the number may move in either direction.

---

## Two flaws in the harness itself

Both were caught by a contradiction in its own output — a 450-bed run measuring
*slower* than a 500-bed one — and both were fixed before any number here was published.

**`ru_maxrss` is a process high-water mark, not a per-level reading.** It never falls
when a cluster is released, so every RSS figure was "peak after building every earlier
level too". The tell was three consecutive levels reporting an identical 2699 MB. Now
reported as `peak_rss_mb` with a separate `rss_growth_mb` delta.

**Single-run timings vary by ~25% between invocations** — larger than several of the
between-row differences a scaling table invites you to draw conclusions from.
`--repeats` now reports the median with the spread beside it.

---

## What these numbers are not

- **One machine, one core, one tick rate.** These are numbers for an ARM laptop at the
  simulator's tick cadence, not a deployment guarantee.
- **Memory is the uncomfortable axis.** ~2.5 GB peak at 400 beds, growing roughly
  linearly. A large-ward claim on a free-tier host is not credible on memory grounds
  even where latency fits.
- **Capacity is not clinical validation.** Running 250 beds says nothing about whether
  the detector helps any of them; that question lives in `docs/SUBGROUP_AUDIT.md` and
  the research page.

---

## Where this is shown

`site/app/research/page.tsx` §11 "Does it scale" renders this table with the run-to-run
spread beside each row and the 400-bed row flagged for breaching p99, plus the two
findings above. §10 "Who it serves worse" renders the subgroup audit alongside it.

Both were added because the two heaviest ISEF judging categories are *Execution: data
collection, analysis and interpretation* and *Creativity & potential impact* (20 points
each), and judges specifically probe whether a student understands the **limitations**
of their results — so the fairness audit, the flagged exploratory cell, and the
harness's own corrected flaws belong on the page rather than in a footnote.

Wiring them in pushed `site/data/results.json` past its 60 KB bundle guard. Rather than
raise the ceiling, the ROC/PR curves were downsampled to 88 points with endpoints
preserved: they were 22 KB of the bundle — more than every headline number combined —
at a density no chart that size can render. 62.8 KB → 54.0 KB.

Sources for the judging criteria: [Society for Science — Grand Award
criteria](https://www.societyforscience.org/isef/grand-award/criteria/), [Judging at
your Fair](https://www.societyforscience.org/isef/affiliated-fair-network/judging-at-your-fair/).
