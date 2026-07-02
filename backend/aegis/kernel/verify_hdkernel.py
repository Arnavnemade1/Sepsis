"""Invariant + behaviour harness for HDKernel (Layer 3c live kernel).

ASSERTED (kernel correctness — always true):
  * no process leaks: every spawned process reaches DONE (barring an intentional
    deadlock test)
  * CPU-time conservation: Σ per-process cpu_ns == kernel clock (single core)
  * causal ordering: no process runs before it is created
  * a SLEEPING process never runs before its wakeup
  * IPC integrity: a consumer receives exactly what the producer sent, in order
  * Spawn/join: a parent fan-out/fan-in over children completes
  * deadlock is DETECTED (two tasks each blocked on an empty channel)
  * determinism: identical run twice -> identical summary

MEASURED (printed, not asserted):
  * the same workload under FIFO/RR/CFS/EDF/HDSched produces different process
    tables — i.e. swapping the policy genuinely changes what the kernel does

Run:
    .venv/bin/python -m backend.aegis.kernel.verify_hdkernel
"""
from __future__ import annotations

import sys
from typing import List

from backend.aegis.kernel.hdkernel import (
    Channel, Compute, Exit, HDKernel, POLICIES, Recv, Send, Sleep, State, US,
    Yield, consumer, cpu_bound, periodic, producer, supervisor,
)


def _mixed_kernel(policy: str) -> HDKernel:
    k = HDKernel(policy=policy)
    ch = Channel("pipe")
    k.spawn(cpu_bound, name="batch-A", priority=0, args=(600 * US,))
    k.spawn(cpu_bound, name="batch-B", priority=0, args=(600 * US,))
    k.spawn(periodic, name="rt-sensor", priority=3, deadline_in=200 * US,
            args=(80 * US, 20 * US, 6))
    k.spawn(producer, name="producer", priority=1, args=(ch, 5))
    k.spawn(consumer, name="consumer", priority=1, args=(ch,))
    return k


def check_core_invariants() -> None:
    print("[1] Core kernel invariants")
    for policy in POLICIES:
        k = _mixed_kernel(policy)
        summ = k.run()
        assert summ["completed"] == summ["processes"], f"{policy}: process leak"
        assert not summ["deadlocked"], f"{policy}: spurious deadlock"
        assert all(p.state is State.DONE for p in k.procs.values()), f"{policy}: not all DONE"
        # CPU-time conservation (single core): Σ cpu_ns + idle == clock
        cpu_sum = sum(p.cpu_ns for p in k.procs.values())
        assert cpu_sum + k.idle_ns == k.clock, \
            f"{policy}: cpu {cpu_sum} + idle {k.idle_ns} != clock {k.clock}"
        # causal ordering: first_run >= created for every process
        assert all(p.first_run is not None and p.first_run >= p.created
                   for p in k.procs.values()), f"{policy}: ran before creation"
        assert k.total_switches <= k.dispatches, f"{policy}: more switches than dispatches"
        print(f"      [PASS] {policy:<8} 5/5 done, cpu+idle==clock ({k.clock//US}us, "
              f"idle {k.idle_ns//US}us), {k.total_switches} ctx switches, causal")
    print()


def check_sleep_semantics() -> None:
    print("[2] SLEEPING processes do not run before wakeup")
    seen: List[int] = []

    def sleeper():
        yield Compute(10 * US)
        t_before = yield Sleep(500 * US)      # Sleep returns None; kernel time advances
        yield Compute(10 * US)
        seen.append("woke")
        yield Exit()

    k = HDKernel(policy="RR")
    # heavy background so the clock advances well past the sleeper's wakeup window
    k.spawn(cpu_bound, name="bg", priority=0, args=(2000 * US,))
    sp = k.spawn(sleeper, name="sleeper", priority=1)
    # step the loop, recording that the sleeper is never RUNNING while SLEEPING
    violated = {"v": False}
    orig = k._dispatch
    def guarded(pcb):
        if pcb.pid == sp and pcb.state is State.SLEEPING:
            violated["v"] = True
        return orig(pcb)
    k._dispatch = guarded
    k.run()
    assert not violated["v"], "a SLEEPING process was dispatched"
    assert seen == ["woke"], "sleeper never woke / woke wrong"
    woke_at = k.procs[sp].finished
    assert woke_at >= 500 * US, f"woke too early at {woke_at}"
    print(f"      [PASS] sleeper woke at t={woke_at//US}us (>=510us), never ran while asleep")
    print()


def check_ipc() -> None:
    print("[3] IPC integrity — consumer receives exactly what producer sent")
    got: List[int] = []
    ch = Channel("data")

    def counting_consumer():
        while True:
            item = yield Recv(ch)
            if item is None:
                break
            got.append(item)
            yield Compute(5 * US)
        yield Exit()

    k = HDKernel(policy="CFS")
    k.spawn(producer, name="prod", args=(ch, 8))
    k.spawn(counting_consumer, name="cons")
    k.run()
    assert got == list(range(8)), f"IPC corrupted: {got}"
    print(f"      [PASS] delivered {got} in order, no loss/dupes")
    print()


def check_spawn_join() -> None:
    print("[4] Spawn + join — parent fans out workers and joins on IPC")
    done = Channel("join")
    k = HDKernel(policy="HDSched")
    k.spawn(supervisor, name="super", args=(done, 4, 150 * US))
    k.run()
    # 1 supervisor + 4 workers = 5 processes, all DONE
    assert len(k.procs) == 5, f"expected 5 procs, got {len(k.procs)}"
    assert all(p.state is State.DONE for p in k.procs.values()), "join incomplete"
    workers = [p for p in k.procs.values() if p.name.startswith("worker")]
    assert len(workers) == 4, f"expected 4 workers, got {len(workers)}"
    print(f"      [PASS] supervisor spawned + joined 4 workers; all 5 DONE")
    print()


def check_deadlock_detection() -> None:
    print("[5] Deadlock detection — two tasks each block on an empty channel")
    a, b = Channel("a"), Channel("b")

    def waiter(chan):
        _ = yield Recv(chan)          # blocks forever (no sender)
        yield Exit()

    k = HDKernel(policy="CFS")
    k.spawn(waiter, name="w1", args=(a,))
    k.spawn(waiter, name="w2", args=(b,))
    summ = k.run()
    assert summ["deadlocked"], "deadlock not detected"
    assert summ["completed"] == 0, "blocked tasks should not complete"
    print(f"      [PASS] deadlock detected; 0/2 completed; log: {k.log[-1]}")
    print()


def check_determinism() -> None:
    print("[6] Determinism — identical workload runs identically")
    s1 = _mixed_kernel("HDSched").run()
    s2 = _mixed_kernel("HDSched").run()
    assert s1 == s2, f"nondeterministic:\n  {s1}\n  {s2}"
    print(f"      [PASS] two runs identical: {s1['dispatches']} dispatches, "
          f"{s1['context_switches']} switches, clock={s1['clock_us']}us")
    print()


def show_policy_effect() -> None:
    print("[7] Swapping the policy changes what the kernel does (measured)")
    print("    same 5-process workload; note turnaround of the RT sensor + batch order\n")
    for policy in ("FIFO", "RR", "CFS", "EDF", "HDSched"):
        k = _mixed_kernel(policy)
        k.run()
        rt = next(r for r in k.ps() if r["name"] == "rt-sensor")
        batch = [r for r in k.ps() if r["name"].startswith("batch")]
        bturn = max(r["turnaround_us"] for r in batch)
        print(f"      {policy:<8} switches={k.total_switches:<3} "
              f"rt_sensor_turn={rt['turnaround_us']:>6}us  "
              f"batch_worst_turn={bturn:>6}us  clock={k.clock//US}us")
    print()


def run() -> int:
    print("=" * 68)
    print("AegisOS Layer 3c — HDKernel live cooperative kernel verification")
    print("=" * 68)
    check_core_invariants()
    check_sleep_semantics()
    check_ipc()
    check_spawn_join()
    check_deadlock_detection()
    check_determinism()
    show_policy_effect()
    print("ALL HDKERNEL INVARIANTS HOLD ✓  (cooperative, non-clairvoyant, honest)")
    print("=" * 68)
    return 0


def main() -> None:
    run()


if __name__ == "__main__":
    sys.exit(run())
