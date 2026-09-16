"""Verify the single-writer discipline — and prove the check can FAIL.

    .venv/bin/python -m backend.aegis.kernel.verify_executor

The bug this guards: ``_cluster_loop`` ticks the cluster from the event-loop
thread while ``POST /aegis/orchestrator/scenario`` ticks the SAME cluster from a
threadpool worker. Two writers into ring buffers whose entire correctness
argument (kernel/memfabric.py header) is "each ring has exactly one writer".

Check 2 is the one that matters, so it is written adversarially: it first shows
the foreign write being REFUSED, then unbinds enforcement and shows the very same
write SUCCEEDING. A guard that cannot be observed failing is not evidence.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import List

from backend.aegis.kernel import executor as ex
from backend.aegis.kernel.executor import KernelExecutor, WriterViolation, assert_writer
from backend.aegis.kernel.memfabric import MemoryArena, LocklessRing

FAILURES: List[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def main() -> int:
    """Run the checks, then hand the process back exactly as it was found.

    Enforcement is a process-global binding, so leaving it set poisons every
    later ring write in the same interpreter — which is precisely what happened
    when this suite first ran inside verify_kernel: five subsequent suites failed
    with WriterViolation from MainThread.
    """
    kx = None
    try:
        kx, code = _run_checks()
        return code
    finally:
        if kx is not None:
            kx.stop()
        ex._WRITER_IDENT = None
        del ex._VIOLATIONS[:]


def _run_checks():
    print("=" * 64)
    print("  KERNEL EXECUTOR — single-writer enforcement")
    print("=" * 64)

    # -- 1. unbound: enforcement is off, single-threaded callers unaffected --
    ex._WRITER_IDENT = None
    del ex._VIOLATIONS[:]
    ok = True
    try:
        assert_writer("unbound probe")
    except WriterViolation:
        ok = False
    check("unbound executor does not enforce (verify suites, offline tools)", ok)

    kx = KernelExecutor(name="verify-writer").start()

    # -- 2. THE BUG: a foreign thread writing a real ring is refused ---------
    arena = MemoryArena(4096)
    ring = LocklessRing(arena, "verify/bed_0/optical", 64, ["hr", "spo2"])

    kx.call(ring.push, 1.0, {"hr": 80.0, "spo2": 98.0})
    check("the writer thread can push", ring.write_index == 1)

    caught = {}

    def foreign_push() -> None:
        try:
            ring.push(2.0, {"hr": 999.0, "spo2": 0.0})
            caught["result"] = "ACCEPTED"
        except WriterViolation:
            caught["result"] = "REFUSED"

    t = threading.Thread(target=foreign_push, name="rogue-scenario-thread")
    t.start(); t.join(5.0)
    check("a second thread's ring write is REFUSED", caught.get("result") == "REFUSED",
          f"got {caught.get('result')}")
    check("the refused write did not land", ring.write_index == 1,
          f"write_index={ring.write_index}")
    check("the violation is recorded for /proc", len(ex.violations()) == 1)

    # ...and the same write SUCCEEDS with enforcement unbound, so we have seen
    # this check go red. Without this, "REFUSED" could be an artifact of a broken
    # thread or a typo'd ring and the check would pass for the wrong reason.
    bound = ex._WRITER_IDENT
    ex._WRITER_IDENT = None
    caught.clear()
    t2 = threading.Thread(target=foreign_push, name="rogue-scenario-thread")
    t2.start(); t2.join(5.0)
    ex._WRITER_IDENT = bound
    check("CONTROL: unenforced, the identical write is ACCEPTED (the guard is load-bearing)",
          caught.get("result") == "ACCEPTED" and ring.write_index == 2,
          f"got {caught.get('result')}, write_index={ring.write_index}")

    # -- 3. work submitted from elsewhere runs ON the writer thread ----------
    seen = {}

    def who() -> int:
        seen["ident"] = threading.get_ident()
        return 42

    out = {}

    def caller() -> None:
        out["v"] = kx.call(who)

    t3 = threading.Thread(target=caller); t3.start(); t3.join(5.0)
    check("call() from a foreign thread runs on the writer and returns its value",
          out.get("v") == 42 and seen.get("ident") != threading.get_ident())

    # -- 4. a nested write does not wait on itself --------------------------
    def nested() -> str:
        return kx.call(lambda: "inner")          # already on the writer thread

    done = {}

    def nested_caller() -> None:
        done["v"] = kx.call(nested)

    t4 = threading.Thread(target=nested_caller); t4.start(); t4.join(5.0)
    check("a nested call runs inline instead of deadlocking (scenario -> tick -> push)",
          done.get("v") == "inner", "timed out" if t4.is_alive() else "")

    # -- 5. periodic work is DROPPED, not queued, while the writer is busy ---
    release = threading.Event()
    started = threading.Event()

    def hog() -> None:
        started.set()
        release.wait(5.0)

    threading.Thread(target=lambda: kx.call(hog)).start()
    started.wait(5.0)
    time.sleep(0.02)
    before = kx.skipped
    res = {}

    def tick_attempt() -> None:
        res["v"] = kx.try_call(lambda: "ticked")

    t5 = threading.Thread(target=tick_attempt); t5.start(); t5.join(5.0)
    check("try_call refuses rather than queues while the writer is busy",
          res.get("v") is KernelExecutor.BUSY and kx.skipped == before + 1)
    release.set()
    time.sleep(0.05)

    # -- 6. an exception on the writer reaches the caller -------------------
    def boom() -> None:
        raise ValueError("planted")

    err = {}

    def boom_caller() -> None:
        try:
            kx.call(boom)
        except ValueError as exc:
            err["v"] = str(exc)

    t6 = threading.Thread(target=boom_caller); t6.start(); t6.join(5.0)
    check("an exception on the writer thread propagates to the caller",
          err.get("v") == "planted")

    print("\n" + "=" * 64)
    if FAILURES:
        print(f"  FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        print("=" * 64)
        return kx, 1
    print("  ALL CHECKS PASSED — the write path has exactly one owner")
    print("=" * 64)
    return kx, 0


if __name__ == "__main__":
    sys.exit(main())
