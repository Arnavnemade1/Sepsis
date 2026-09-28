"""Load harness — how many beds can this OS actually run, and what gives way first?

AegisOS has always been demonstrated at twelve beds, because
:func:`backend.aegis.cohort.build_cohort` defaults to twelve and nothing ever asked
for more. "I built an operating system for critical care" and "it runs a 400-bed
hospital on one core" are different claims, and only the second is a claim about
scale. This measures which one is true.

It drives the **real** stack — the same ``CohortCluster.tick()`` the live server runs,
over real CinC patients, through the device bus, memory fabric, fusion engine,
per-patient learner, HDC associative memory, cross-modal resonance gate and the
Z3-verified cognitive scheduler. There is no mock anywhere in the path; a number
produced here is a number the product produces.

**It profiles rather than assumes.** The obvious suspect is the process-wide Z3 lock
in :mod:`backend.aegis.z3lock`, since every solver call in the OS serialises through
it. But the scheduler's per-tick verification already takes that lock
*non-blocking* and defers when it is contended, and ``VERIFY_EVERY`` throttles it
further — so the tick loop was designed around exactly that hazard. Rather than
"optimise" a bottleneck chosen in advance, the harness times each phase of the tick
separately and reports where the cost actually is. Wrapping happens here, in the
measurement code, so the production path carries no instrumentation.

    python -m backend.aegis.scale.loadgen --beds 1,10,50,100 --ticks 40
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from typing import Any, Callable, Dict, List

from backend.ml import config


def _peak_rss_mb() -> float:
    """**Peak** resident set since process start, in MB.

    ``ru_maxrss`` is a high-water mark for the whole process, not a current reading:
    it never falls when a cluster is released. Reporting it per level implies a
    per-level memory cost, which it is not — a 500-bed row measured after 1/10/50/100
    /250-bed rows carries their peak too. The field is named for what it is, and the
    per-level cost is reported separately as a delta.

    (bytes on macOS, kilobytes on Linux)
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 ** 2) if sys.platform == "darwin" else raw / 1024


class Probe:
    """Times named phases by wrapping bound methods on the live objects.

    The instrumentation lives here rather than in ``cohort.py`` on purpose: a
    profiler that ships inside the thing it profiles changes the thing it profiles,
    and leaves timing hooks in a clinical path that has no use for them.
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        self._undo: List[Callable[[], None]] = []

    def wrap(self, obj: Any, attr: str, label: str) -> None:
        fn = getattr(obj, attr, None)
        if fn is None or not callable(fn):
            return                                   # optional phase; absent is fine
        totals, calls = self.totals, self.calls

        def timed(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                dt = time.perf_counter() - t0
                totals[label] = totals.get(label, 0.0) + dt
                calls[label] = calls.get(label, 0) + 1

        setattr(obj, attr, timed)
        self._undo.append(lambda: setattr(obj, attr, fn))

    def restore(self) -> None:
        for u in reversed(self._undo):
            u()
        self._undo.clear()

    def reset(self) -> None:
        self.totals.clear()
        self.calls.clear()


#: Phases that run INSIDE another phase, so their time is already counted in the
#: parent. Kept explicit rather than inferred: a profiler that silently sums nested
#: timings reports a breakdown adding up to more than the thing it broke down.
NESTING = {"z3_verify": "sched_decide"}


def instrument(cluster) -> Probe:
    """Wrap every phase of one tick. Names match the reader's mental model of the
    tick, not the call graph, so the report says where the time goes clinically."""
    p = Probe()
    p.wrap(cluster.pump, "step", "device_pump")
    p.wrap(cluster.fusion, "update", "fusion")
    p.wrap(cluster.learner, "get", "learner_lookup")
    p.wrap(cluster.sched, "observe", "sched_observe")
    p.wrap(cluster.sched, "schedule", "sched_decide")
    p.wrap(cluster.sched, "verify_decision_z3", "z3_verify")   # nested inside sched_decide
    p.wrap(cluster, "_sample_vitals", "sample_vitals")
    p.wrap(cluster, "_hdc_query_all", "hdc_memory")     # batched: once per tick
    p.wrap(cluster, "_hdc_query", "hdc_memory_per_bed")  # legacy path, if still used
    p.wrap(cluster, "_resonance_query", "resonance_gate")
    return p


def measure(n_beds: int, ticks: int, warmup: int, seed: int,
            repeats: int = 1) -> Dict[str, Any]:
    """Build a cluster of ``n_beds`` real patients and time ``ticks`` of the real loop.

    ``repeats`` rebuilds and re-times the whole level. A single run of this harness
    varies by tens of percent between invocations — enough that two levels measured
    in different runs cannot be compared, which is exactly the trap a scaling table
    invites. The reported latency is the median across repeats and the spread is
    reported beside it, so a reader can see whether a difference between rows is
    bigger than the noise between runs of the same row.
    """
    import gc
    runs = [_measure_once(n_beds, ticks, warmup, seed) for _ in range(max(repeats, 1))]
    gc.collect()
    med = sorted(runs, key=lambda r: r["tick_ms"]["p50"])[len(runs) // 2]
    p50s = [r["tick_ms"]["p50"] for r in runs]
    med["repeats"] = len(runs)
    med["p50_spread"] = {"min": min(p50s), "max": max(p50s),
                         "rel_range": (max(p50s) - min(p50s)) / max(min(p50s), 1e-9)}
    return med


def _measure_once(n_beds: int, ticks: int, warmup: int, seed: int) -> Dict[str, Any]:
    from backend.aegis.cohort import CohortCluster

    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    # A cohort of N needs at least N usable records to draw from; the default scan
    # of 1500 quietly caps the wing well below 1000 beds otherwise.
    cluster = CohortCluster(n=n_beds, seed=seed)
    build_s = time.perf_counter() - t0
    actual = len(cluster.node_ids)
    # The server freezes its boot heap out of the cyclic collector once the kernel is
    # up (server.py); measure the tick the product runs, not a different heap policy.
    import gc
    gc.collect()
    gc.freeze()

    for _ in range(warmup):                          # first ticks pay import + JIT costs
        cluster.tick()

    probe = instrument(cluster)
    lat: List[float] = []
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    for _ in range(ticks):
        t = time.perf_counter()
        cluster.tick()
        lat.append((time.perf_counter() - t) * 1000.0)
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    probe.restore()
    gc.unfreeze()                                    # the next level starts clean

    lat.sort()
    def pct(q: float) -> float:
        return lat[min(int(q * len(lat)), len(lat) - 1)]

    # Shares are of measured TICK WALL TIME, not of the sum of phases. Phases nest —
    # z3_verify runs inside sched_decide — so a breakdown normalised to its own sum
    # would double-count the nested time and quietly total more than the tick. Read
    # as fractions of the tick, the numbers stay honest and the nesting stays visible
    # (a nested phase's share is a subset of its parent's, not an addition to it).
    tick_total = wall
    phases = {k: {"ms_per_tick": 1000.0 * v / ticks,
                  "share_of_tick": v / max(tick_total, 1e-12),
                  "calls_per_tick": probe.calls[k] / ticks,
                  **({"nested_in": NESTING[k]} if k in NESTING else {})}
              for k, v in sorted(probe.totals.items(), key=lambda kv: -kv[1])}
    accounted = sum(v for k, v in probe.totals.items() if k not in NESTING)

    return {
        "beds_requested": n_beds,
        "beds_built": actual,
        "build_seconds": round(build_s, 2),
        "ticks": ticks,
        "tick_ms": {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
                    "mean": statistics.fmean(lat), "max": lat[-1]},
        "bed_ticks_per_s": actual * ticks / wall,
        "ms_per_bed_tick": 1000.0 * wall / max(actual * ticks, 1),
        "cpu_utilisation": cpu / wall,
        "peak_rss_mb": _peak_rss_mb(),
        # How much this level added to the high-water mark. Zero means an earlier,
        # larger level already claimed the peak — not that this level is free.
        "rss_growth_mb": max(0.0, _peak_rss_mb() - rss_before),
        "phases": phases,
        # What fraction of the tick the top-level phases explain. A low number means
        # the cost is somewhere this harness is not looking, which is worth knowing
        # before anyone optimises what it CAN see.
        "phase_coverage": accounted / max(tick_total, 1e-12),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beds", default="1,10,50,100",
                    help="comma-separated bed counts to measure")
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=1,
                    help="rebuild and re-time each level N times; the table reports the "
                         "median and the spread, because run-to-run variance on this "
                         "harness is tens of percent")
    ap.add_argument("--out", default=os.path.join(config.ARTIFACTS_DIR, "scale.json"))
    args = ap.parse_args()

    levels = [int(b) for b in args.beds.split(",") if b.strip()]
    print(f"host: {platform.processor() or platform.machine()} · "
          f"python {sys.version.split()[0]} · driving the real CohortCluster.tick()\n")
    rows: List[Dict[str, Any]] = []
    for n in levels:
        print(f"building {n} beds ...", flush=True)
        r = measure(n, args.ticks, args.warmup, args.seed, args.repeats)
        rows.append(r)
        t = r["tick_ms"]
        spread = (f" +/-{r['p50_spread']['rel_range']*100:.0f}%"
                  if r.get("repeats", 1) > 1 else "")
        print(f"  {r['beds_built']:>5} beds  tick p50 {t['p50']:8.1f} ms{spread}  "
              f"p99 {t['p99']:8.1f}  "
              f"{r['ms_per_bed_tick']:6.2f} ms/bed  "
              f"peak RSS {r['peak_rss_mb']:.0f} MB (+{r['rss_growth_mb']:.0f})")
        top = [(k, v) for k, v in r["phases"].items()][:5]
        print("        " + "  ".join(
            f"{k}{'*' if 'nested_in' in v else ''} {v['share_of_tick']*100:.0f}%"
            for k, v in top)
            + f"   [top-level phases explain {r['phase_coverage']*100:.0f}% of the tick;"
              f" * = nested]")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "host": {"machine": platform.machine(), "python": sys.version.split()[0],
                 "cpu_count": os.cpu_count()},
        "note": "real CohortCluster.tick() over real CinC patients; no mocks in the path",
        "levels": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
