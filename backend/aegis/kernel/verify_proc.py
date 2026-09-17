"""Verify the process table — lifecycle, signal semantics, real accounting.

    .venv/bin/python -m backend.aegis.kernel.verify_proc

The claim that needs adversarial treatment is the signal interface. A process
table whose ``kill`` is accepted and ignored is worse than none, so the checks
send signals a process cannot honour and require EINVAL, then send one it can and
require its handler to run — the refusal has to be selective, not universal.

One refusal is a safety property rather than a missing feature: a patient's
monitoring pipeline honours no signal at all. That is checked with a CONTROL on
the same table, so it cannot pass merely because signals are broken.
"""
from __future__ import annotations

import sys
import time
from typing import Any, List, Tuple

from backend.aegis.kernel.journal import Journal
from backend.aegis.kernel.proc import (DRIVER, JOB, KTHREAD, MONITOR, SERVICE,
                                       BLOCKED, READY, ZOMBIE,
                                       NoSuchProcess, ProcessTable,
                                       SIGCONT, SIGHUP, SIGKILL, SIGSTOP, SIGTERM,
                                       SIGUSR1, SignalFailed, SignalUnsupported)

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def _refuses(t: ProcessTable, pid: int, sig: str) -> Tuple[bool, str]:
    try:
        t.signal(pid, sig)
        return False, "accepted"
    except SignalUnsupported as exc:
        return True, str(exc)[:70]


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  PROCESS TABLE — lifecycle, signals, measured accounting")
    print("=" * 64)
    j = Journal(ring=512)
    t = ProcessTable(journal=j, reap_after_s=0.05)

    paused = {"loop": False}

    def _loop_handler(sig: str) -> str:
        paused["loop"] = sig == SIGSTOP
        return "paused" if paused["loop"] else "running"

    # -- 1. one pid space across every kind -------------------------------- #
    loop = t.spawn("drive-loop", KTHREAD, on_signal=_loop_handler,
                   signals=(SIGSTOP, SIGCONT))
    bed = t.spawn("bed_3 monitor", MONITOR, bed="bed_3")
    svc = t.spawn("resonance", SERVICE)
    drv = t.spawn("bed_3 optical", DRIVER)
    job = t.spawn("train sepsis-v2", JOB, uid="okafor",
                  on_signal=lambda s: "cancelled", signals=(SIGTERM, SIGKILL))
    pids = [p.pid for p in (loop, bed, svc, drv, job)]
    check("every kind lands in one pid space", pids == [1, 2, 3, 4, 5], str(pids))
    check("ps lists them all", len(t.ps()) == 5)
    check("ps filters by kind", [p["name"] for p in t.ps(kind=SERVICE)] == ["resonance"])
    check("ps filters by owner", [p["name"] for p in t.ps(uid="okafor")] == ["train sepsis-v2"])
    check("find() locates a process by its detail",
          t.find(kind=MONITOR, bed="bed_3") is bed)

    # -- 2. an unknown pid is ESRCH, not a silent None ---------------------- #
    try:
        t.get(999)
        check("an unknown pid raises NoSuchProcess (ESRCH)", False)
    except NoSuchProcess:
        check("an unknown pid raises NoSuchProcess (ESRCH)", True)

    # -- 3. THE CLAIM: signals are honoured or refused, never ignored ------- #
    ok, why = _refuses(t, bed.pid, SIGSTOP)
    check("a patient's monitor refuses SIGSTOP (never paused from kill)", ok, why)
    ok, why = _refuses(t, bed.pid, SIGKILL)
    check("a patient's monitor refuses SIGKILL", ok, why)
    check("the refused signals left the monitor running",
          t.get(bed.pid).state not in (ZOMBIE, BLOCKED), t.get(bed.pid).state)

    # CONTROL: a process that DOES declare SIGSTOP accepts it and its handler acts,
    # so the refusal above is selective rather than a table rejecting everything.
    t.signal(loop.pid, SIGSTOP)
    check("CONTROL: a loop with a pause handler accepts SIGSTOP",
          t.get(loop.pid).state == BLOCKED and paused["loop"] is True)
    t.signal(loop.pid, SIGCONT)
    check("SIGCONT resumes it through the same handler",
          t.get(loop.pid).state == READY and paused["loop"] is False)

    ok, why = _refuses(t, svc.pid, SIGTERM)
    check("a service registered with no handler refuses even a signal its kind allows",
          ok, why)
    ok, why = _refuses(t, loop.pid, SIGTERM)
    check("a loop that declared only pause/resume refuses SIGTERM", ok, why)

    # -- 4. declaring a signal nobody will act on is refused at spawn ------- #
    try:
        t.spawn("liar", SERVICE, signals=(SIGTERM,))
        check("spawn refuses a declared signal with no handler", False, "accepted")
    except ValueError as exc:
        check("spawn refuses a declared signal with no handler", True, str(exc)[:60])
    try:
        t.spawn("bed_9 monitor", MONITOR, on_signal=lambda s: None, signals=(SIGSTOP,))
        check("spawn refuses SIGSTOP on a monitor even with a handler", False, "accepted")
    except ValueError as exc:
        check("spawn refuses SIGSTOP on a monitor even with a handler", True, str(exc)[:60])

    # -- 5. the subsystem's own handler runs ------------------------------- #
    seen = {}

    def _svc_handler(sig: str) -> str:
        # NOT a lambda with `or`: `seen.setdefault(...) or "restarted"` returns the
        # signal name, so the check below passed against its own fixture instead
        # of against the relayed return value.
        seen["sig"] = sig
        return "restarted"

    svc2 = t.spawn("radar", SERVICE, on_signal=_svc_handler, signals=(SIGTERM, SIGHUP))
    out = t.signal(svc2.pid, SIGHUP)
    check("the registered handler receives the signal", seen.get("sig") == SIGHUP)
    check("its return value reaches the caller", out["result"] == "restarted", str(out))

    drv_seen = {}
    drv2 = t.spawn("bed_7 optical", DRIVER,
                   on_signal=lambda s: drv_seen.setdefault("sig", s),
                   signals=(SIGHUP, SIGUSR1))
    t.signal(drv2.pid, SIGUSR1)                  # inject fault, the driver's verb
    check("a driver honours SIGUSR1 (fault injection)", drv_seen.get("sig") == SIGUSR1)

    # -- 6. a failing handler leaves the state alone ------------------------ #
    def _broken(sig: str) -> Any:
        raise RuntimeError("planted: the loop would not pause")

    bad = t.spawn("cron", KTHREAD, on_signal=_broken, signals=(SIGSTOP, SIGCONT))
    try:
        t.signal(bad.pid, SIGSTOP)
        check("a handler that raises surfaces SignalFailed", False, "no error")
    except SignalFailed:
        check("a handler that raises surfaces SignalFailed", True)
    check("…and the table does NOT claim the loop is paused",
          t.get(bad.pid).state != BLOCKED, t.get(bad.pid).state)

    # -- 7. a job can be cancelled and becomes a zombie, not a hole --------- #
    t.signal(job.pid, SIGTERM, uid="okafor")
    z = t.get(job.pid)
    check("a terminated job becomes ZOMBIE with an exit code",
          z.state == ZOMBIE and z.exit_code == 0, f"{z.state}/{z.exit_code}")
    check("a zombie is still answerable in ps",
          any(p["pid"] == job.pid for p in t.ps()))
    t.exit(job.pid, code=3)
    check("exit is idempotent (a second exit does not rewrite the code)",
          t.get(job.pid).exit_code == 0)
    time.sleep(0.06)
    check("reap collects it after the grace", t.reap() == 1 and job.pid not in t.procs)

    # -- 8. reconciliation journals transitions, not restatements ----------- #
    before = sum(1 for e in j.entries if e.event == "state")
    t.set_state(drv2.pid, "DEGRADED", slot="fallback")
    t.set_state(drv2.pid, "DEGRADED", slot="fallback")
    after = sum(1 for e in j.entries if e.event == "state")
    check("set_state journals a transition once, not every restatement",
          after - before == 1, f"{after - before} entries")

    # -- 9. accounting is measured, not declared ---------------------------- #
    work = t.spawn("bed_9 monitor", MONITOR)
    for _ in range(3):
        with t.slice(work.pid):
            time.sleep(0.01)
    d = t.proc(work.pid)
    check("three slices were counted", d["slices"] == 3, str(d["slices"]))
    check("measured CPU is at least the work done (~30ms)", d["cpu_ms"] >= 28.0,
          f"{d['cpu_ms']}ms")
    check("it is not wildly more than the work done", d["cpu_ms"] < 200.0, f"{d['cpu_ms']}ms")
    check("duty is a fraction of its own lifetime, not a fabricated share",
          0.0 < d["duty"] <= 1.0, str(d["duty"]))
    idle = t.proc(drv.pid)
    check("a process that never ran reports zero, not an estimate",
          idle["cpu_ms"] == 0.0 and idle["slices"] == 0)

    # -- 10. top ranks by measured cost ------------------------------------- #
    top = t.top()
    check("top ranks the busiest process first",
          top["top"][0]["pid"] == work.pid, str(top["top"][0]["name"]))
    check("cpu shares sum to about 1",
          abs(sum(r["cpu_share"] for r in top["top"]) - 1.0) < 1e-6)

    # -- 11. the journal carries the lifecycle, and still verifies ---------- #
    events = [e.event for e in j.entries if e.facility == "proc"]
    for want in ("spawn", "signal", "signal.rejected", "signal.failed", "exit", "reap"):
        check(f"journalled: {want}", want in events, str(sorted(set(events))))
    check("the process journal chain verifies", j.verify()["ok"])

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print(f"  ALL {len(_results)} CHECKS PASSED — one table, real signals, measured cost")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
