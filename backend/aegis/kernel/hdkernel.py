"""Layer 3c — HDKernel: a live cooperative-multitasking kernel that RUNS.

``hdsched.py`` proved the scheduling *policies* on a discrete-event simulator —
it scores FIFO/RR/CFS/EDF/HDSched but never executes anything. This module is the
other half of "genuine OS": a real kernel that actually *runs tasks*. Processes
are coroutines (Python generators) that make **syscalls** by yielding; the kernel
owns a process table, dispatches with the same pluggable policy family, accounts
CPU time, counts context switches, and delivers inter-process messages.

Honesty rules (stated up front, same as the rest of the kernel):
  * This is **cooperative** multitasking, not preemptive hardware IRQs. A task
    holds a core until it yields a syscall — exactly how classic Mac OS, Windows
    3.x, and modern async/green-thread runtimes work. We do not claim to preempt
    CPU threads or field timer interrupts, and the code never pretends to.
  * The kernel clock is **CPU-time**: it advances by the work each task actually
    consumes (declared per ``Compute`` burst, so the run is deterministic and
    testable). ``sleep``/deadlines live in that same time base.
  * The live scheduler is **non-clairvoyant** — unlike the simulator, it does not
    know a task's remaining service. HDSched here estimates burst length from the
    process's own history (an exponential average, the classic SJF predictor).
    That is the honest way to schedule work whose length you cannot see.

What is real and demonstrable: coroutine processes that spawn children, sleep,
block on IPC channels, and exit; a scheduler that reorders them under load; exact
CPU-time and context-switch accounting; and ``ps``/``top``/``proc`` views a
console can render. It shares HDSched's encoder, so the policy you benchmark is
the policy that runs.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Generator, List, Optional, Tuple

import numpy as np

from backend.aegis.kernel.hdsched import HDEncoder, US

# A task is a generator yielding Syscall objects; the kernel .send()s results back.
Task = Generator[Any, Any, Any]
TaskFn = Callable[..., Task]


# --------------------------------------------------------------------------- #
# Syscalls — the only way a process talks to the kernel
# --------------------------------------------------------------------------- #
class Syscall:
    """Base class. A process performs a syscall by ``yield``-ing an instance."""


@dataclass
class Compute(Syscall):
    """Ask for ``ns`` nanoseconds of CPU. The kernel may split it across quanta
    (the task is re-dispatched until the burst is done). This is the unit of
    work; a task interleaves Compute with sleeps/IPC like any real program."""
    ns: int


@dataclass
class Sleep(Syscall):
    ns: int                                # relinquish the CPU for ns of kernel time


class Yield(Syscall):
    """Voluntarily give up the CPU and go to the back of the ready set."""


@dataclass
class Spawn(Syscall):
    fn: TaskFn
    name: str = "task"
    priority: int = 1
    weight: float = 1.0
    deadline_in: Optional[int] = None      # ns from now; None => best-effort
    args: tuple = ()


@dataclass
class Send(Syscall):
    chan: "Channel"
    item: Any


@dataclass
class Recv(Syscall):
    chan: "Channel"


class Exit(Syscall):
    """Terminate voluntarily (equivalent to the generator returning)."""


class Channel:
    """A blocking IPC mailbox. Send never blocks (buffered); Recv blocks the
    caller until a message is available. The kernel wakes blocked receivers."""
    _ids = itertools.count()

    def __init__(self, name: str = ""):
        self.id = next(self._ids)
        self.name = name or f"chan{self.id}"
        self.buffer: Deque[Any] = deque()
        self.waiters: Deque[int] = deque()     # pids blocked in Recv


# --------------------------------------------------------------------------- #
# Process control block
# --------------------------------------------------------------------------- #
class State(str, Enum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    SLEEPING = "SLEEPING"
    BLOCKED = "BLOCKED"       # waiting on a Channel
    ZOMBIE = "ZOMBIE"         # finished, awaiting reap
    DONE = "DONE"


@dataclass
class PCB:
    pid: int
    name: str
    gen: Task
    priority: int = 1
    weight: float = 1.0
    deadline: Optional[int] = None          # absolute kernel-ns
    ppid: Optional[int] = None
    state: State = State.NEW
    # scheduling / accounting
    ready_since: int = 0                    # kernel-ns it last became READY
    cpu_ns: int = 0                         # total CPU consumed (accounting)
    vruntime: float = 0.0                   # CFS weighted virtual runtime
    ctx_switches: int = 0
    wakeup: int = 0                         # kernel-ns to wake if SLEEPING
    cur_burst_left: int = 0                 # ns remaining in the active Compute
    est_burst: float = 50 * US              # exponential burst-length estimate (SJF)
    _burst_start_total: int = 0             # cpu_ns when the current burst began
    created: int = 0
    first_run: Optional[int] = None
    finished: Optional[int] = None
    send_val: Any = None                    # value to inject on next resume
    recv_on: Optional[Channel] = None

    # -- encoder-compatible view (HDSched reuses hdsched.HDEncoder.sims) -- #
    @property
    def arrival(self) -> int:               # "how long it has waited" axis
        return self.ready_since

    @property
    def remaining(self) -> float:           # non-clairvoyant burst ESTIMATE
        return self.est_burst


# --------------------------------------------------------------------------- #
# Live scheduling policies over the ready set (PCBs)
# --------------------------------------------------------------------------- #
class LivePolicy:
    name = "base"
    quantum = 50 * US

    def pick(self, ready: List[PCB], now: int) -> PCB: ...


class LiveFIFO(LivePolicy):
    name = "FIFO"
    def __init__(self, quantum=1 << 62): self.quantum = quantum   # run-to-completion
    def pick(self, ready, now): return min(ready, key=lambda p: (p.created, p.pid))


class LiveRR(LivePolicy):
    name = "RR"
    def __init__(self, quantum=50 * US): self.quantum = quantum
    def pick(self, ready, now): return min(ready, key=lambda p: (p.ready_since, p.pid))


class LiveCFS(LivePolicy):
    name = "CFS"
    def __init__(self, quantum=50 * US): self.quantum = quantum
    def pick(self, ready, now): return min(ready, key=lambda p: (p.vruntime, p.pid))


class LiveEDF(LivePolicy):
    name = "EDF"
    def __init__(self, quantum=50 * US): self.quantum = quantum
    def pick(self, ready, now):
        INF = 1 << 62
        return min(ready, key=lambda p: (p.deadline if p.deadline is not None else INF, p.pid))


class LiveHDSched(LivePolicy):
    """The HDC policy, live and non-clairvoyant. Reuses hdsched.HDEncoder; the
    'remaining' axis is fed the per-process burst ESTIMATE, not an oracle."""
    name = "HDSched"

    def __init__(self, encoder: Optional[HDEncoder] = None,
                 theta: Optional[np.ndarray] = None, quantum: int = 50 * US):
        self.enc = encoder or HDEncoder(dim=8192, levels=16, seed=7)
        # EDF-flavoured prior, more urgency-weighting under load; override to tune.
        t = theta if theta is not None else np.array([0, 0, 1.0, 1.0, 0, 0, 0.6, 0.4])
        self.theta = np.asarray(t, dtype=np.float32).reshape(2, 4)
        self.quantum = quantum

    def pick(self, ready, now):
        S = self.enc.sims(ready, now)                  # (k,4) HDC similarities
        load = 0.5 * (float(np.tanh(S.sum(axis=1).mean())) + 1.0)
        theta = self.theta[0] + load * self.theta[1]
        return ready[int(np.argmax(S @ theta))]


POLICIES: Dict[str, Callable[[], LivePolicy]] = {
    "FIFO": LiveFIFO, "RR": LiveRR, "CFS": LiveCFS, "EDF": LiveEDF,
    "HDSched": LiveHDSched,
}


# --------------------------------------------------------------------------- #
# The kernel
# --------------------------------------------------------------------------- #
class HDKernel:
    """A live cooperative multitasking kernel. Feed it root tasks, then ``run``."""

    def __init__(self, policy: str = "HDSched", quantum: int = 50 * US,
                 encoder: Optional[HDEncoder] = None):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; have {list(POLICIES)}")
        self.policy_name = policy
        self.policy: LivePolicy = (LiveHDSched(encoder, quantum=quantum)
                                   if policy == "HDSched" else POLICIES[policy](quantum))
        self.quantum = quantum
        self.clock = 0                       # kernel-ns (CPU-time base)
        self.procs: Dict[int, PCB] = {}
        self._pids = itertools.count(1)
        self.running: Optional[int] = None
        self.total_switches = 0
        self.dispatches = 0
        self.idle_ns = 0                     # clock advanced with no task running
        self.log: List[str] = []

    # -- process creation ------------------------------------------------- #
    def spawn(self, fn: TaskFn, name: str = "task", priority: int = 1,
              weight: float = 1.0, deadline_in: Optional[int] = None,
              ppid: Optional[int] = None, args: tuple = ()) -> int:
        pid = next(self._pids)
        pcb = PCB(pid=pid, name=name, gen=fn(*args), priority=int(priority),
                  weight=float(weight), ppid=ppid, created=self.clock,
                  ready_since=self.clock,
                  deadline=(self.clock + deadline_in) if deadline_in is not None else None)
        pcb.state = State.READY
        self.procs[pid] = pcb
        return pid

    # -- the run loop ----------------------------------------------------- #
    def _wake_sleepers(self) -> None:
        for p in self.procs.values():
            if p.state is State.SLEEPING and self.clock >= p.wakeup:
                p.state = State.READY
                p.ready_since = self.clock

    def _ready(self) -> List[PCB]:
        return [p for p in self.procs.values() if p.state is State.READY]

    def run(self, max_dispatches: int = 1_000_000) -> Dict[str, Any]:
        """Run until every process is finished (or a deadlock/step cap). Returns a
        summary; per-process detail is in ``ps()``."""
        while self.dispatches < max_dispatches:
            self._wake_sleepers()
            ready = self._ready()
            if not ready:
                # Nobody runnable: fast-forward the clock to the next wakeup.
                sleepers = [p for p in self.procs.values() if p.state is State.SLEEPING]
                if sleepers:
                    nxt = min(p.wakeup for p in sleepers)
                    self.idle_ns += max(0, nxt - self.clock)   # honest CPU idle time
                    self.clock = max(self.clock, nxt)
                    self.running = None                        # CPU goes idle
                    continue
                # No ready, no sleepers: either all done, or a real deadlock.
                if any(p.state is State.BLOCKED for p in self.procs.values()):
                    self.log.append(f"t={self.clock} DEADLOCK: blocked procs, no senders")
                break

            pcb = self.policy.pick(ready, self.clock)
            self._dispatch(pcb)

        return self.summary()

    def _dispatch(self, pcb: PCB) -> None:
        self.dispatches += 1
        if self.running != pcb.pid:
            self.total_switches += 1
            pcb.ctx_switches += 1
            self.running = pcb.pid
        if pcb.first_run is None:
            pcb.first_run = self.clock
        pcb.state = State.RUNNING

        # Resume the coroutine. If it is mid-burst, we service the burst here in
        # the kernel (no re-entry needed) so a long Compute is sliced by quantum.
        if pcb.cur_burst_left > 0:
            self._service_burst(pcb)
            return
        try:
            call = pcb.gen.send(pcb.send_val)
            pcb.send_val = None
        except StopIteration:
            self._terminate(pcb)
            return
        self._syscall(pcb, call)

    def _service_burst(self, pcb: PCB) -> None:
        slice_ns = min(self.quantum, pcb.cur_burst_left)
        self.clock += slice_ns
        pcb.cpu_ns += slice_ns
        pcb.vruntime += slice_ns / max(1e-6, pcb.weight)
        pcb.cur_burst_left -= slice_ns
        if pcb.cur_burst_left > 0:
            pcb.state = State.READY                # preempted at quantum boundary
            pcb.ready_since = self.clock
        else:
            # burst done -> update the SJF estimator, resume the coroutine
            actual = pcb.cpu_ns - pcb._burst_start_total
            pcb.est_burst = 0.5 * pcb.est_burst + 0.5 * actual
            pcb.state = State.READY                # re-dispatch to run its next stmt
            pcb.ready_since = self.clock

    def _syscall(self, pcb: PCB, call: Any) -> None:
        if call is None or isinstance(call, Yield) or call is Yield:
            pcb.state = State.READY
            pcb.ready_since = self.clock
        elif isinstance(call, Compute):
            pcb.cur_burst_left = int(call.ns)
            pcb._burst_start_total = pcb.cpu_ns
            self._service_burst(pcb)
        elif isinstance(call, Sleep):
            pcb.state = State.SLEEPING
            pcb.wakeup = self.clock + int(call.ns)
        elif isinstance(call, Spawn):
            child = self.spawn(call.fn, call.name, call.priority, call.weight,
                               call.deadline_in, ppid=pcb.pid, args=call.args)
            pcb.send_val = child                   # return child pid to parent
            pcb.state = State.READY
            pcb.ready_since = self.clock
        elif isinstance(call, Send):
            call.chan.buffer.append(call.item)
            # wake one blocked receiver, if any
            if call.chan.waiters:
                wpid = call.chan.waiters.popleft()
                w = self.procs.get(wpid)
                if w and w.state is State.BLOCKED:
                    w.send_val = call.chan.buffer.popleft()
                    w.recv_on = None
                    w.state = State.READY
                    w.ready_since = self.clock
            pcb.state = State.READY
            pcb.ready_since = self.clock
        elif isinstance(call, Recv):
            if call.chan.buffer:
                pcb.send_val = call.chan.buffer.popleft()
                pcb.state = State.READY
                pcb.ready_since = self.clock
            else:
                pcb.state = State.BLOCKED
                pcb.recv_on = call.chan
                call.chan.waiters.append(pcb.pid)
        elif isinstance(call, Exit) or call is Exit:
            self._terminate(pcb)
        else:
            self.log.append(f"t={self.clock} pid={pcb.pid} unknown syscall {call!r}")
            pcb.state = State.READY
            pcb.ready_since = self.clock

    def _terminate(self, pcb: PCB) -> None:
        pcb.state = State.DONE
        pcb.finished = self.clock
        if self.running == pcb.pid:
            self.running = None

    # -- introspection: the OS control surface ---------------------------- #
    def ps(self) -> List[Dict[str, Any]]:
        rows = []
        for p in sorted(self.procs.values(), key=lambda p: p.pid):
            rows.append({
                "pid": p.pid, "ppid": p.ppid, "name": p.name,
                "state": p.state.value, "prio": p.priority,
                "cpu_us": round(p.cpu_ns / US, 1),
                "vruntime_us": round(p.vruntime / US, 1),
                "ctx": p.ctx_switches,
                "deadline_us": round(p.deadline / US, 1) if p.deadline is not None else None,
                "turnaround_us": (round((p.finished - p.created) / US, 1)
                                  if p.finished is not None else None),
                "est_burst_us": round(p.est_burst / US, 1),
            })
        return rows

    def top(self) -> Dict[str, Any]:
        procs = list(self.procs.values())
        by_state: Dict[str, int] = {}
        for p in procs:
            by_state[p.state.value] = by_state.get(p.state.value, 0) + 1
        total_cpu = sum(p.cpu_ns for p in procs) or 1
        hogs = sorted(procs, key=lambda p: p.cpu_ns, reverse=True)[:5]
        busy = sum(p.cpu_ns for p in procs)
        return {
            "policy": self.policy_name, "clock_us": round(self.clock / US, 1),
            "procs": len(procs), "by_state": by_state,
            "dispatches": self.dispatches, "context_switches": self.total_switches,
            "utilization": round(busy / max(1, self.clock), 3),
            "idle_us": round(self.idle_ns / US, 1),
            "cpu_shares": [{"pid": p.pid, "name": p.name,
                            "pct": round(100 * p.cpu_ns / total_cpu, 1)} for p in hogs],
        }

    def proc(self, pid: int) -> Optional[Dict[str, Any]]:
        p = self.procs.get(pid)
        if p is None:
            return None
        d = next((r for r in self.ps() if r["pid"] == pid), {})
        d.update({"created_us": round(p.created / US, 1),
                  "first_run_us": round(p.first_run / US, 1) if p.first_run else None,
                  "waiting_us": round((p.first_run - p.created) / US, 1)
                  if p.first_run else None})
        return d

    def summary(self) -> Dict[str, Any]:
        done = [p for p in self.procs.values() if p.state is State.DONE]
        turn = [(p.finished - p.created) for p in done if p.finished is not None]
        return {
            "policy": self.policy_name,
            "clock_us": round(self.clock / US, 1),
            "processes": len(self.procs),
            "completed": len(done),
            "dispatches": self.dispatches,
            "context_switches": self.total_switches,
            "total_cpu_us": round(sum(p.cpu_ns for p in self.procs.values()) / US, 1),
            "idle_us": round(self.idle_ns / US, 1),
            "mean_turnaround_us": round(float(np.mean(turn)) / US, 1) if turn else 0.0,
            "deadlocked": any(p.state is State.BLOCKED for p in self.procs.values()),
        }


# --------------------------------------------------------------------------- #
# A small standard task library (real programs for the kernel to run)
# --------------------------------------------------------------------------- #
def cpu_bound(total_ns: int, chunk_ns: int = 40 * US) -> Task:
    """A CPU-bound task: consume ``total_ns`` in Compute chunks."""
    left = total_ns
    while left > 0:
        n = min(chunk_ns, left)
        yield Compute(n)
        left -= n
    yield Exit()


def periodic(period_ns: int, work_ns: int, reps: int) -> Task:
    """A periodic real-time task: wake, do a little work, sleep, repeat."""
    for _ in range(reps):
        yield Compute(work_ns)
        yield Sleep(period_ns)
    yield Exit()


def producer(chan: Channel, n: int, work_ns: int = 30 * US) -> Task:
    for i in range(n):
        yield Compute(work_ns)
        yield Send(chan, i)
    yield Send(chan, None)                          # poison pill
    yield Exit()


def consumer(chan: Channel, work_ns: int = 30 * US) -> Task:
    while True:
        item = yield Recv(chan)
        if item is None:
            break
        yield Compute(work_ns)
    yield Exit()


def supervisor(chan: Channel, n_workers: int, job_ns: int = 200 * US) -> Task:
    """A parent that spawns workers then waits for their 'done' messages (fan-out
    / fan-in — exercises Spawn + IPC join)."""
    for i in range(n_workers):
        yield Spawn(_worker, name=f"worker{i}", priority=1, args=(chan, job_ns))
    for _ in range(n_workers):
        yield Recv(chan)
    yield Exit()


def _worker(chan: Channel, job_ns: int) -> Task:
    yield Compute(job_ns)
    yield Send(chan, "done")
    yield Exit()


# --------------------------------------------------------------------------- #
# Named workloads + a one-call runner (shared by the server /proc surface, the
# REST endpoint, the console, and the verify harness — one source of truth).
# --------------------------------------------------------------------------- #
KWORKLOADS = ("mixed", "pipeline", "fanout", "stress")


def build_workload(k: "HDKernel", name: str) -> None:
    """Spawn a named workload's processes into kernel ``k`` (does not run it)."""
    if name == "mixed":
        ch = Channel("pipe")
        k.spawn(cpu_bound, name="batch-A", priority=0, args=(600 * US,))
        k.spawn(cpu_bound, name="batch-B", priority=0, args=(600 * US,))
        k.spawn(periodic, name="rt-sensor", priority=3, deadline_in=200 * US,
                args=(80 * US, 20 * US, 6))
        k.spawn(producer, name="producer", priority=1, args=(ch, 5))
        k.spawn(consumer, name="consumer", priority=1, args=(ch,))
    elif name == "pipeline":                       # 3-stage producer→filter→sink
        a, b = Channel("stage1"), Channel("stage2")
        def filt():
            while True:
                x = yield Recv(a)
                if x is None:
                    yield Send(b, None); break
                yield Compute(20 * US)
                yield Send(b, x * 2)
            yield Exit()
        k.spawn(producer, name="source", priority=1, args=(a, 8))
        k.spawn(filt, name="filter", priority=2)
        k.spawn(consumer, name="sink", priority=1, args=(b,))
    elif name == "fanout":                          # parent fans out + joins workers
        done = Channel("join")
        k.spawn(supervisor, name="supervisor", priority=2, args=(done, 6, 150 * US))
    elif name == "stress":                          # many hogs + several RT deadlines
        for i in range(6):
            k.spawn(cpu_bound, name=f"batch-{i}", priority=0, args=((400 + 80 * i) * US,))
        for i in range(4):
            k.spawn(periodic, name=f"rt-{i}", priority=3, deadline_in=180 * US,
                    args=((60 + 10 * i) * US, 25 * US, 5))
    else:
        raise ValueError(f"unknown workload {name!r}; have {KWORKLOADS}")


def run_workload(name: str = "mixed", policy: str = "HDSched",
                 max_dispatches: int = 200_000) -> Dict[str, Any]:
    """Build + run a named workload under a policy; return summary/ps/top/log.
    The kernel runs to completion (cooperative, CPU-time clock), so this returns
    the final process table — a genuine 'run a program on the OS and inspect it'."""
    k = HDKernel(policy=policy)
    build_workload(k, name)
    summ = k.run(max_dispatches=max_dispatches)
    return {"workload": name, "policy": policy, "summary": summ,
            "ps": k.ps(), "top": k.top(), "log": k.log[-20:]}


# --------------------------------------------------------------------------- #
# Standalone demo
# --------------------------------------------------------------------------- #
def _demo() -> None:
    k = HDKernel(policy="HDSched")
    # A mixed workload: two CPU hogs, a periodic RT task, a producer/consumer pair.
    ch = Channel("pipe")
    k.spawn(cpu_bound, name="batch-A", priority=0, args=(600 * US,))
    k.spawn(cpu_bound, name="batch-B", priority=0, args=(600 * US,))
    k.spawn(periodic, name="rt-sensor", priority=3, deadline_in=200 * US,
            args=(80 * US, 20 * US, 6))
    k.spawn(producer, name="producer", priority=1, args=(ch, 5))
    k.spawn(consumer, name="consumer", priority=1, args=(ch,))
    summ = k.run()
    print("summary:", summ)
    print("\nps:")
    for r in k.ps():
        print(f"  pid={r['pid']:<2} {r['name']:<10} {r['state']:<6} "
              f"cpu={r['cpu_us']:>6}us ctx={r['ctx']:<3} turn={r['turnaround_us']}us")
    print("\ntop:", k.top())


if __name__ == "__main__":
    _demo()
