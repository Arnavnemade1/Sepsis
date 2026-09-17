"""The process table — one list of what is actually running, with real numbers.

WHY THIS EXISTS
---------------
AegisOS had six things that each called themselves a process table and shared
nothing: HDKernel's ``procs`` (synthetic demo tasks, created on first read of
``/proc/kthreads``), the cognitive scheduler's ``nodes`` (beds, called "procs" in
``/proc/sched``), aegisd's ``services``, two DriverSupervisor ``slots``, DevBio's
fd tables, and mlctl's ``jobs``. Meanwhile the things genuinely executing — two
asyncio loops, the warmup threads, the keep-alive, the per-bed pipeline, the
supervised drivers, the training threads — appeared in none of them.

So ``ps`` reported a demo workload while the real system was invisible. This is
the table that lists what is running, who owns it, what it has cost, and what can
be done to it.

WHAT A "PROCESS" IS HERE (the honest definition, and the one the UI must use):
a supervised, accounted unit of execution with an identity, an owner, a state and
a signal interface. It is NOT an address space. There is no memory isolation
between these, no preemption of one by another, and no per-process RSS — CPython
gives none of that, and claiming it would be the exact failure this project keeps
auditing itself for. What IS real: the pid, the owner, the lifecycle, the signal
semantics, and the CPU time, which is measured with a monotonic clock around the
work itself rather than estimated.

DESIGN NOTE: this module imports nothing from the subsystems it tracks. A
process carries an ``on_signal`` callback supplied at registration, so aegisd
knows how to restart a service and the supervisor knows how to re-arm a driver,
while the kernel only knows that a signal was requested, permitted and delivered.
"""
from __future__ import annotations

import itertools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from backend.aegis.kernel.journal import INFO, NOTICE, WARN, Journal, KJOURNAL

# -- kinds -------------------------------------------------------------------
KTHREAD = "kthread"     # the loops and daemons this process actually runs
SERVICE = "service"     # an aegisd-managed app
MONITOR = "monitor"     # one patient's monitoring pipeline (a scheduler node)
DRIVER = "driver"       # a supervised, verified device driver
JOB = "job"             # bounded background work: training, a scenario, an export
SESSION = "session"     # a signed-in user's session (its shell and windows)
AGENT = "agent"         # Dr. Aegis acting on a clinician's behalf
KINDS = (KTHREAD, SERVICE, MONITOR, DRIVER, JOB, SESSION, AGENT)

# -- states ------------------------------------------------------------------
NEW, READY, RUNNING, SLEEPING, BLOCKED, ZOMBIE, EXITED = (
    "NEW", "READY", "RUNNING", "SLEEPING", "BLOCKED", "ZOMBIE", "EXITED")

# -- signals -----------------------------------------------------------------
SIGTERM, SIGKILL, SIGSTOP, SIGCONT, SIGHUP, SIGUSR1 = (
    "SIGTERM", "SIGKILL", "SIGSTOP", "SIGCONT", "SIGHUP", "SIGUSR1")
SIGNALS = (SIGTERM, SIGKILL, SIGSTOP, SIGCONT, SIGHUP, SIGUSR1)

#: The MOST each kind could ever be told to do. A process honours only the subset
#: it declares at spawn, and it may only declare signals it registered a handler
#: for — so a signal is accepted exactly when something will act on it. A kind
#: (or a process) that cannot honour a signal says so (EINVAL) instead of accepting
#: it and doing nothing, which is the failure mode that makes a process table
#: decorative.
SUPPORTED: Dict[str, tuple] = {
    KTHREAD: (SIGTERM, SIGSTOP, SIGCONT),              # stop / pause / resume a loop
    SERVICE: (SIGTERM, SIGHUP),                        # stop / restart (aegisd verbs)
    # A patient's monitoring pipeline honours NOTHING. Pausing or killing a bed's
    # monitoring from `kill` is a clinical hazard, not an administrative action;
    # the scheduler already reallocates compute, and a bed is never switched off.
    MONITOR: (),
    DRIVER: (SIGHUP, SIGUSR1),                         # re-verify / inject fault
    JOB: (SIGTERM, SIGKILL),                           # cancel
    SESSION: (SIGTERM,),                               # sign out
    AGENT: (SIGTERM,),                                 # end the consult
}


class SignalUnsupported(ValueError):
    """EINVAL — this kind of process has no meaning for that signal."""


class NoSuchProcess(KeyError):
    """ESRCH."""


class SignalFailed(RuntimeError):
    """The process accepted the signal but its handler raised. State is unchanged."""


@dataclass
class Process:
    pid: int
    name: str
    kind: str
    uid: str = "kernel"
    ppid: int = 0
    state: str = NEW
    started_at: float = field(default_factory=time.time)
    cpu_ns: int = 0                       # measured, not estimated
    slices: int = 0
    last_slice_ns: int = 0
    max_slice_ns: int = 0
    exit_code: Optional[int] = None
    exited_at: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    on_signal: Optional[Callable[[str], Any]] = None
    signals: tuple = ()                   # what THIS process honours (⊆ its kind's max)

    @property
    def cpu_ms(self) -> float:
        return self.cpu_ns / 1e6

    def supported_signals(self) -> tuple:
        allowed = SUPPORTED.get(self.kind, ())
        return tuple(sg for sg in self.signals if sg in allowed)

    def to_dict(self) -> Dict[str, Any]:
        now = time.time()
        wall = (self.exited_at or now) - self.started_at
        return {
            "pid": self.pid, "ppid": self.ppid, "name": self.name, "kind": self.kind,
            "uid": self.uid, "state": self.state,
            "cpu_ms": round(self.cpu_ms, 3), "slices": self.slices,
            "last_ms": round(self.last_slice_ns / 1e6, 3),
            "max_ms": round(self.max_slice_ns / 1e6, 3),
            # share of its own lifetime spent executing — the honest per-process
            # cost. Not a share of a CPU: several of these run on one thread.
            "duty": round(self.cpu_ns / 1e9 / wall, 4) if wall > 0 else 0.0,
            "uptime_s": round(wall, 1),
            "exit_code": self.exit_code,
            "signals": list(self.supported_signals()),
            "detail": self.detail,
        }


class ProcessTable:
    """Every supervised unit of execution in this kernel, with one pid space."""

    def __init__(self, journal: Optional[Journal] = None, reap_after_s: float = 300.0):
        self._pids = itertools.count(1)
        self.procs: Dict[int, Process] = {}
        self.journal = journal if journal is not None else KJOURNAL
        self.reap_after_s = float(reap_after_s)
        self.booted_at = time.time()
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------- #
    def spawn(self, name: str, kind: str, uid: str = "kernel", ppid: int = 0,
              state: str = RUNNING, on_signal: Optional[Callable[[str], Any]] = None,
              signals: tuple = (), **detail: Any) -> Process:
        """Register a unit of execution.

        ``signals`` is what this process will honour. Every one must be permitted
        for its kind AND backed by ``on_signal`` — declaring a signal with no
        handler is refused at registration, because that is precisely how a
        table ends up accepting a kill it never delivers.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown process kind {kind!r}")
        signals = tuple(sg.upper() for sg in signals)
        bad = [sg for sg in signals if sg not in SUPPORTED[kind]]
        if bad:
            raise ValueError(f"a {kind} process cannot honour {bad} "
                             f"(its kind allows: {list(SUPPORTED[kind]) or 'nothing'})")
        if signals and on_signal is None:
            raise ValueError(f"{name!r} declares {list(signals)} with no handler to act on them")
        with self._lock:
            p = Process(pid=next(self._pids), name=name, kind=kind, uid=uid,
                        ppid=ppid, state=state, on_signal=on_signal,
                        signals=signals, detail=detail)
            self.procs[p.pid] = p
        self.journal.append("proc", "spawn", severity=INFO, uid=uid, pid=p.pid,
                            name=name, kind=kind)
        return p

    def exit(self, pid: int, code: int = 0) -> Process:
        p = self.get(pid)
        if p.state in (ZOMBIE, EXITED):
            return p                                  # exit is idempotent
        p.state = ZOMBIE
        p.exit_code = int(code)
        p.exited_at = time.time()
        self.journal.append("proc", "exit", severity=INFO if code == 0 else WARN,
                            uid=p.uid, pid=pid, name=p.name, code=code)
        return p

    def reap(self, pid: Optional[int] = None) -> int:
        """Collect zombies. Without a pid, collects every one past the grace.

        Zombies are kept deliberately: a training job that failed at 03:00 has to
        still be answerable at 09:00, and a process that vanishes on exit cannot
        be asked why it stopped.
        """
        now = time.time()
        gone = []
        with self._lock:
            for p in list(self.procs.values()):
                if p.state != ZOMBIE:
                    continue
                if pid is not None and p.pid != pid:
                    continue
                if pid is None and (now - (p.exited_at or now)) < self.reap_after_s:
                    continue
                p.state = EXITED
                gone.append(p.pid)
                del self.procs[p.pid]
        for g in gone:
            self.journal.append("proc", "reap", severity=INFO, pid=g)
        return len(gone)

    def get(self, pid: int) -> Process:
        p = self.procs.get(int(pid))
        if p is None:
            raise NoSuchProcess(f"no such process: {pid}")
        return p

    def set_state(self, pid: int, state: str, **detail: Any) -> Process:
        """Reconcile a process with what its subsystem reports (e.g. a driver that
        failed over). Journals only transitions, never a restated state."""
        p = self.get(pid)
        if detail:
            p.detail.update(detail)
        if state != p.state:
            prev, p.state = p.state, state
            self.journal.append("proc", "state", severity=INFO, pid=pid, name=p.name,
                                frm=prev, to=state)
        return p

    def find(self, kind: Optional[str] = None, live: bool = True,
             **match: Any) -> Optional[Process]:
        """First process whose detail matches — how a subsystem finds its pid."""
        for p in self._rows():
            if kind is not None and p.kind != kind:
                continue
            if live and p.state in (ZOMBIE, EXITED):
                continue
            if all(p.detail.get(k) == v for k, v in match.items()):
                return p
        return None

    def _rows(self) -> List[Process]:
        # Spawns happen on training threads and the executor while the API reads,
        # so iterate a copy taken under the lock ("dict changed size" otherwise).
        with self._lock:
            return list(self.procs.values())

    # -- accounting ------------------------------------------------------- #
    @contextmanager
    def slice(self, pid: int):
        """Measure one unit of work against a process.

        ``perf_counter_ns`` around the call: wall time spent inside this slice on
        whichever thread ran it. It is NOT per-thread CPU time (``thread_time``
        would exclude the I/O and the GIL waits that a bed's tick genuinely
        spends), and it is not a share of a core.
        """
        p = self.procs.get(int(pid))
        t0 = time.perf_counter_ns()
        prev = p.state if p else None
        if p is not None:
            p.state = RUNNING
        try:
            yield
        finally:
            if p is not None:
                d = time.perf_counter_ns() - t0
                p.cpu_ns += d
                p.slices += 1
                p.last_slice_ns = d
                p.max_slice_ns = max(p.max_slice_ns, d)
                p.state = prev if prev not in (None, RUNNING) else READY

    @contextmanager
    def job(self, name: str, uid: str = "kernel", ppid: int = 0, **detail: Any):
        """Run bounded work as a JOB process: spawned, measured, and exited with
        code 0 — or 1 if the work raised (the exception still propagates). The
        zombie stays answerable in ``ps`` until the reap grace passes."""
        p = self.spawn(name, JOB, uid=uid, ppid=ppid, **detail)
        code = 0
        try:
            with self.slice(p.pid):
                yield p
        except BaseException:
            code = 1
            raise
        finally:
            self.exit(p.pid, code=code)

    # -- signals ---------------------------------------------------------- #
    def signal(self, pid: int, sig: str, uid: str = "kernel") -> Dict[str, Any]:
        """Deliver a signal. Raises SignalUnsupported (EINVAL) if this process
        cannot honour it — never silently accepts one it will not act on.

        The handler runs FIRST and the state changes only if it succeeds: a
        SIGSTOP whose handler raised must not leave the table claiming the loop is
        paused while it keeps running.
        """
        p = self.get(pid)
        sig = sig.upper()
        if sig not in SIGNALS:
            raise SignalUnsupported(f"unknown signal {sig!r}")
        if sig not in p.supported_signals():
            self.journal.append("proc", "signal.rejected", severity=WARN, uid=uid,
                                pid=pid, signal=sig, kind=p.kind)
            why = ("a patient's monitoring is never paused or killed from the process table"
                   if p.kind == MONITOR else
                   f"supports: {', '.join(p.supported_signals()) or 'nothing'}")
            raise SignalUnsupported(f"{p.kind} process {p.name!r} does not honour {sig} ({why})")
        try:
            result = p.on_signal(sig) if p.on_signal is not None else None
        except Exception as exc:
            self.journal.append("proc", "signal.failed", severity=WARN, uid=uid, pid=pid,
                                signal=sig, name=p.name, error=f"{type(exc).__name__}: {exc}")
            raise SignalFailed(f"{sig} to {p.name!r} failed: {exc}") from exc
        if sig == SIGSTOP:
            p.state = BLOCKED
        elif sig == SIGCONT:
            p.state = READY
        self.journal.append("proc", "signal", severity=NOTICE, uid=uid, pid=pid,
                            signal=sig, name=p.name, kind=p.kind)
        if sig in (SIGTERM, SIGKILL) and p.state not in (ZOMBIE, EXITED):
            self.exit(pid, code=0 if sig == SIGTERM else 137)
        return {"pid": pid, "signal": sig, "state": p.state, "result": result}

    # -- views ------------------------------------------------------------ #
    def ps(self, kind: Optional[str] = None, uid: Optional[str] = None
           ) -> List[Dict[str, Any]]:
        rows = [p for p in self._rows()
                if (kind is None or p.kind == kind) and (uid is None or p.uid == uid)]
        return [p.to_dict() for p in sorted(rows, key=lambda p: p.pid)]

    def top(self, limit: int = 12) -> Dict[str, Any]:
        allp = self._rows()
        rows = sorted(allp, key=lambda p: p.cpu_ns, reverse=True)
        total = sum(p.cpu_ns for p in allp) or 1
        return {
            "uptime_s": round(time.time() - self.booted_at, 1),
            "processes": len(allp),
            "by_kind": {k: sum(1 for p in allp if p.kind == k)
                        for k in KINDS if any(p.kind == k for p in allp)},
            "by_state": {st: sum(1 for p in allp if p.state == st)
                         for st in (RUNNING, READY, SLEEPING, BLOCKED, ZOMBIE)
                         if any(p.state == st for p in allp)},
            "cpu_ms_total": round(total / 1e6, 2),
            "top": [dict(p.to_dict(), cpu_share=round(p.cpu_ns / total, 4))
                    for p in rows[:limit]],
            # Sources of these numbers, stated where they are shown.
            "note": "cpu = measured wall time inside each unit's own work "
                    "(perf_counter_ns); several units share one thread",
        }

    def proc(self, pid: int) -> Dict[str, Any]:
        """``/proc/<pid>`` — one process in full."""
        p = self.get(pid)
        d = p.to_dict()
        d["started"] = round(p.started_at, 3)
        d["supported_signals"] = list(p.supported_signals())
        return d

    def status(self) -> Dict[str, Any]:
        allp = self._rows()
        return {"processes": len(allp),
                "uptime_s": round(time.time() - self.booted_at, 1),
                "by_kind": {k: sum(1 for p in allp if p.kind == k) for k in KINDS},
                "cpu_ms_total": round(sum(p.cpu_ns for p in allp) / 1e6, 2)}


#: The process-wide table. server.py registers the real subsystems into it.
PROCS = ProcessTable()
