"""Verify the single-writer discipline — and prove each check can FAIL.

    .venv/bin/python -m backend.aegis.kernel.verify_executor

The bug this guards: ``_cluster_loop`` ticks the cluster from the event-loop
thread while ``POST /aegis/orchestrator/scenario`` ticks the SAME cluster from a
threadpool worker. Two writers into ring buffers whose entire correctness
argument (kernel/memfabric.py header) is "each ring has exactly one writer".

Two further failures shipped with the first version of the fix and are checked
here so they cannot come back:

  * **Over-reach.** Ownership was process-global, so the boot warm-up's writes
    into its OWN private 2-bed cluster were refused. The warm-up silently skipped
    on every boot, and the first clinician click paid the full cold-start cost.
    Check 3 writes an unowned arena from a foreign thread while the wing's arena
    is owned, and requires it to be accepted.
  * **Deadlock.** With the over-reach fixed, the warm-up — which holds the Z3 lock
    — would have called ``run_scenario``, which handed the work to the writer,
    whose job then waited on the Z3 lock the caller held. Check 7 runs exactly that
    shape under a timeout.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import List, Tuple

from backend.aegis.kernel import executor as ex
from backend.aegis.kernel.executor import KernelExecutor, WriterViolation
from backend.aegis.kernel.memfabric import MemoryArena, LocklessRing

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def _push_from_thread(ring: LocklessRing, value: float, name: str = "rogue-scenario-thread") -> str:
    out = {}

    def run() -> None:
        try:
            ring.push(time.time() * 1e6, {"hr": value, "spo2": 0.0})
            out["r"] = "ACCEPTED"
        except WriterViolation:
            out["r"] = "REFUSED"

    t = threading.Thread(target=run, name=name)
    t.start()
    t.join(5.0)
    return out.get("r", "TIMEOUT")


def main() -> int:
    """Run the checks and leave the process as it was found (stop() releases every
    arena the executor owned, so nothing leaks into later suites)."""
    _results.clear()
    del ex._VIOLATIONS[:]
    kx = KernelExecutor(name="verify-writer").start()
    try:
        return _run_checks(kx)
    finally:
        kx.stop()
        del ex._VIOLATIONS[:]


def _run_checks(kx: KernelExecutor) -> int:
    print("=" * 64)
    print("  KERNEL EXECUTOR — single-writer enforcement, per arena")
    print("=" * 64)

    wing = MemoryArena(4096)
    ring = LocklessRing(wing, "verify/bed_0/optical", 64, ["hr", "spo2"])

    # -- 1. unowned: nothing enforced -------------------------------------- #
    check("an unowned arena accepts a write from any thread",
          _push_from_thread(ring, 70.0) == "ACCEPTED" and ring.write_index == 1)

    # -- 2. THE BUG: once owned, a second thread's write is refused -------- #
    kx.own(wing)
    kx.call(ring.push, 1.0, {"hr": 80.0, "spo2": 98.0})
    check("the owning writer thread can push", ring.write_index == 2)
    check("a second thread's write into the OWNED arena is REFUSED",
          _push_from_thread(ring, 999.0) == "REFUSED")
    check("the refused write did not land", ring.write_index == 2,
          f"write_index={ring.write_index}")
    check("the violation is recorded for /proc", len(ex.violations()) == 1,
          str(len(ex.violations())))
    # CONTROL: release ownership and the identical write succeeds — so REFUSED
    # above is the guard, not a broken thread or a typo'd ring.
    wing.writer_ident = None
    check("CONTROL: released, the identical write is ACCEPTED (the guard is load-bearing)",
          _push_from_thread(ring, 999.0) == "ACCEPTED" and ring.write_index == 3)
    kx.own(wing)

    # -- 3. REGRESSION: ownership is per arena, not per process ------------- #
    private = MemoryArena(1024)
    warm_ring = LocklessRing(private, "warmup/node_0/optical", 16, ["hr", "spo2"])
    check("while the wing is owned, a private arena is still writable from another thread "
          "(the boot warm-up's own cluster)",
          _push_from_thread(warm_ring, 72.0, name="orch-warmup") == "ACCEPTED")
    check("owns() is true for the wing and false for the private arena",
          kx.owns(wing) and not kx.owns(private))

    # -- 4. work submitted from elsewhere runs ON the writer thread ---------- #
    seen, out = {}, {}

    def who() -> int:
        seen["ident"] = threading.get_ident()
        return 42

    t = threading.Thread(target=lambda: out.__setitem__("v", kx.call(who)))
    t.start(); t.join(5.0)
    check("call() from a foreign thread runs on the writer and returns its value",
          out.get("v") == 42 and seen.get("ident") != threading.get_ident())

    # -- 5. a nested write does not wait on itself --------------------------- #
    done = {}
    t = threading.Thread(target=lambda: done.__setitem__(
        "v", kx.call(lambda: kx.call(lambda: "inner"))))
    t.start(); t.join(5.0)
    check("a nested call runs inline instead of deadlocking (scenario -> tick -> push)",
          done.get("v") == "inner", "timed out" if t.is_alive() else "")

    # -- 6. periodic work is DROPPED, not queued, while the writer is busy --- #
    release, started = threading.Event(), threading.Event()

    def hog() -> None:
        started.set()
        release.wait(5.0)

    threading.Thread(target=lambda: kx.call(hog)).start()
    started.wait(5.0)
    time.sleep(0.02)
    before, res = kx.skipped, {}
    t = threading.Thread(target=lambda: res.__setitem__("v", kx.try_call(lambda: "ticked")))
    t.start(); t.join(5.0)
    check("try_call refuses rather than queues while the writer is busy",
          res.get("v") is KernelExecutor.BUSY and kx.skipped == before + 1)
    release.set()
    time.sleep(0.05)

    err = {}

    def boom_caller() -> None:
        try:
            kx.call(lambda: (_ for _ in ()).throw(ValueError("planted")))
        except ValueError as exc:
            err["v"] = str(exc)

    t = threading.Thread(target=boom_caller); t.start(); t.join(5.0)
    check("an exception on the writer thread propagates to the caller",
          err.get("v") == "planted")

    # -- 7. REGRESSION: the warm-up shape must not deadlock ------------------ #
    # A thread holding Z3 runs a scenario on a PRIVATE cluster while the executor
    # owns another arena. Routing that through the writer would deadlock (caller
    # holds Z3, waits on writer; writer's job waits on Z3).
    from backend.aegis.kernel import executor as exmod
    from backend.aegis.kernel.neurosym import NeuroSymbolicRuntime
    from backend.aegis.kernel.bundle import SepsisBundleEngine
    from backend.aegis.kernel.scheduler import BioCluster
    from backend.aegis.orchestrator import Orchestrator
    from backend.aegis.z3lock import Z3_LOCK

    saved = exmod.EXEC
    exmod.EXEC = kx                  # run_scenario reads the module-level EXEC
    try:
        warm = BioCluster(n_nodes=2, lane_budget=32, seed=1)
        orch = Orchestrator(warm, cage=NeuroSymbolicRuntime(), bundle=SepsisBundleEngine())
        outcome = {}

        def warmup_shape() -> None:
            try:
                with Z3_LOCK:
                    warm.run(2)
                    r = orch.run_scenario(warm.node_ids[0], warmup=0, crisis_ticks=4)
                outcome["ok"] = isinstance(r, dict)
            except Exception as exc:                    # noqa: BLE001
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        calls_before = kx.calls
        t = threading.Thread(target=warmup_shape, name="orch-warmup", daemon=True)
        t.start()
        t.join(60.0)
        check("a Z3-holding thread runs a private-cluster scenario without deadlock",
              not t.is_alive() and outcome.get("ok") is True,
              "DEADLOCK (still running after 60 s)" if t.is_alive()
              else outcome.get("error", ""))
        check("…and it did NOT go through the writer (nothing to serialise against)",
              kx.calls == calls_before, f"{kx.calls - calls_before} writer calls")

        # CONTROL: a cluster whose arena the executor DOES own is routed through it.
        kx.own(warm.fabric.arena)
        routed_before = kx.calls
        t = threading.Thread(target=lambda: orch.run_scenario(
            warm.node_ids[0], warmup=0, crisis_ticks=2), daemon=True)
        t.start(); t.join(60.0)
        check("CONTROL: the owned cluster's scenario IS routed through the writer",
              not t.is_alive() and kx.calls > routed_before,
              f"{kx.calls - routed_before} writer calls")
    finally:
        exmod.EXEC = saved

    # -- 8. stop() releases what it owned ------------------------------------ #
    probe = KernelExecutor(name="verify-release").start()
    a = MemoryArena(64)
    probe.own(a)
    probe.stop()
    check("stop() releases every owned arena (nothing leaks into later suites)",
          a.writer_ident is None)

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print(f"  ALL {len(_results)} CHECKS PASSED — each arena has exactly one owner")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
