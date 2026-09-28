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
python -m backend.aegis.scale.loadgen --beds 1,50,100,250,400,600,800,1000 --ticks 25 --repeats 3
```

---

## Result

Median of 3 rebuilt runs per level, one core of an ARM laptop. Tick budget 600 ms.
Measured 2026-09-27, after the tick was squeezed (next section).

| beds | tick p50 | run spread | tick p99 | ms/bed | peak RSS | dominant phase |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.3 ms | ±6% | 0.7 ms | 0.30 | 189 MB | sched_decide 58% |
| 50 | 9.7 ms | ±1% | 10.3 ms | 0.19 | 540 MB | hdc_memory 45% |
| 100 | 18.0 ms | ±5% | 20.4 ms | 0.18 | 850 MB | hdc_memory 43% |
| 250 | 55.6 ms | ±8% | 60.2 ms | 0.22 | 1370 MB | hdc_memory 46% |
| 400 | 97.2 ms | ±17% | 105.8 ms | 0.24 | 1529 MB | hdc_memory 51% |
| 600 | 166.0 ms | ±9% | 370.3 ms | 0.29 | 1910 MB | hdc_memory 53% |
| 800 | 186.1 ms | ±15% | 256.7 ms | 0.24 | 2333 MB | hdc_memory 55% |
| 1000 | 253.7 ms | ±3% | **267.2 ms** | 0.25 | 2946 MB | hdc_memory 59% |

**1,000 beds within the real-time budget at p99**, from 250 before. The 600-bed p99 is
one slow tick in one repeat (its p50 spread is ±9%); on a laptop doing other work that
is the machine, not the ward. The remaining growth is the precedent retrieval: every
recomputing bed scores against a case store that grows with the ward, so that one
phase is quadratic and everything else is linear. Extrapolating it puts the ceiling
near 1,400 beds; nothing past 1,000 was measured, because memory (2.9 GB there) is the
next wall on this machine.

Before (2026-08-26, same harness):

| beds | tick p50 | tick p99 | ms/bed | dominant phase |
|---:|---:|---:|---:|---|
| 1 | 1.6 ms | 1.7 ms | 1.57 | sched_decide 54% |
| 50 | 66.6 ms | 71.7 ms | 1.32 | resonance_gate 35% |
| 100 | 132.7 ms | 140.4 ms | 1.31 | resonance_gate 34% |
| 250 | 372.4 ms | 391.6 ms | 1.44 | hdc_memory 38% |
| 400 | 535.9 ms | **624.0 ms** | 1.29 | hdc_memory 44% |

## Squeezing the tick (2026-09-27)

Six changes, each exact — the same bits, verdicts and results as the code it replaced,
held to that reference by `make verify-fastpaths` (23 checks, 6 of them planted defects
the suite must catch). At 400 beds: p50 **528 → 97 ms**, p99 **664 → 106 ms**, boot
**13.1 → 7.1 s**.

- **Recompute on change, not on tick.** The clock advances 0.12 ICU hours per tick and
  a bed's HDC query and resonance verdict read its *hourly* frame, so 88% of ticks
  re-encoded, re-queried and re-gated an identical input. Frames, verdicts and
  precedents are now memoised per bed and recomputed exactly when the hour changes.
- **Bed clocks were phase-locked.** Every bed started on a whole hour, so the whole
  ward crossed each hour on the same tick: memoisation alone would have left a
  full-ward recompute spike every ~8 ticks, and the p99 with it. Real admissions are
  not synchronised; each bed now carries a golden-ratio fractional-hour phase, so ~12%
  of the ward changes hour per tick (worst tick 15%, against 100% phase-locked).
- **An encoder that looks up instead of multiplying.** A hypervector was 17 binds (a
  10,000-D multiply each), a stack, a sum and a sign per state. There are only 17 × 100
  distinct binds, so they are a 17 MB table built once; encoding is a gather, an int16
  sum and a sign, batched across states. The resonance gate encodes all three
  modalities from one level lookup, and the frame enriches only the current hour.
- **The scheduler proof as one SMT-LIB2 script.** Building ~4 Z3 terms per bed through
  the Python API cost 107 ms a tick at 400 beds; the same invariants over the same named
  lanes, parsed by Z3 itself, cost 3. A refutation used to pretty-print the solver's
  whole model — every lane of every bed, ~15 ms — and now names the failing invariants
  and beds. So the server's `VERIFY_EVERY` throttle (1 tick in 30) is gone: **every
  scheduling decision is proven** again, at about the cost the throttled schedule had.
- **The boot heap is frozen out of the collector.** The precedent bank alone is tens of
  thousands of long-lived records; each gen-2 pass re-walked them, measured at up to
  35 ms mid-tick at 800 beds. `gc.freeze()` once the kernel is up leaves a pass only
  what the running system allocates (worst pause 2 ms, p50 −11%).
- **Boot does each thing once.** The resonance fit re-derived every cohort row the bank
  had just built, and computed its sepsis axis from two float32 copies of the bank
  (~1.3 GB at 400 beds); it now reuses the rows and sums the ±1 bank in int32, which is
  bit-identical to the float32 mean and allocates nothing.

On the wire: a broadcast was JSON-encoded once per connected socket; it is encoded
once and sent as the same bytes to every socket.

### The server around the tick

The live session (the demo patient the shell streams every 3 s) cost **109.5 ms** a tick,
and 89 ms of it was the production detector scoring one row: each of the histogram
boosters' ~700 trees opened an OpenMP parallel region for that row, and the extra-trees
member spun up a joblib thread pool to average it. Scored single-threaded, the ensemble
is bit-identical to the threaded batch path (300/300 rows) and costs ~6 ms; the session
tick is **19.2 ms**. On a 0.1-CPU host the server's steady load falls from roughly 40% of
the machine to roughly 10%.

The session's clinical story is a hash-chained causal graph that is persisted and
rehydrated on boot, so it only grows — and every snapshot replayed its whole chain
twice and serialised every node to show six. The live integrity flag is now
incremental: each new entry is verified on arrival and a rolling scrub re-hashes 256
older ones per call, so a tampered past entry is reported within ceil(n / 256) snapshots
and the flag latches until a full replay passes (`verify_chain`, unchanged, is still the
full replay every tamper test calls). At 6,000 entries — about a day of uptime — the
snapshot's check costs 1 ms against 21 ms for one replay.

Found on the way: at 400 beds the default 160-lane budget cannot give every bed its
safety lane, so the prover correctly refutes every decision there. The ward is sized
for its budget in production (12 beds); a large ward needs a larger `lane_budget`.

## The bottleneck was not the prover

The plan for this work committed, in advance, to "breaking the Z3 wall": a certificate
cache and a solver process pool, on the reasoning that every SMT call in the OS
serialises through one process-wide lock. Profiling said that was wrong.

Scheduler verification runs **once per tick**, not once per bed, so its share of the
tick *falls* as the ward grows: **53% at one bed, 20% at five hundred.** The lock also
already takes a non-blocking acquire and defers when contended (the `VERIFY_EVERY`
throttle that once sat on top of it is gone; see above) — the tick loop was designed
around exactly that hazard. Both
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
- **Memory is the uncomfortable axis.** 1.5 GB peak at 400 beds and 2.9 GB at 1,000,
  growing roughly linearly (the float32 precedent bank is most of it). A large-ward
  claim on a free-tier host is not credible on memory grounds even where latency fits.
- **Capacity is not clinical validation.** Running 1,000 beds says nothing about whether
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
