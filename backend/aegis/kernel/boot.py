"""The boot sequencer — what actually happened while the kernel came up.

WHY THIS EXISTS
---------------
The shell's boot screen prints eight hardcoded lines on random timers
(``webui/os.js``) and ``AegisOS.boot()`` returns eight hardcoded rows with
``ready=True`` written into the literal. Neither has ever been connected to
anything the server does. Meanwhile the real boot — loading a 7.7 MB detector,
mounting twelve beds' worth of drivers, carving the arena, verifying eBPF
bytecode, starting services, warming the prover — takes 30-60 s on a cold free
tier and is completely invisible.

This records the real thing: each phase's duration, what it produced, and what
failed. The boot screen can then show what the kernel is doing instead of a
loading animation, which is both more honest and more useful, because when a
deploy hangs the log says which phase it hung in.

USAGE (the sequencer wraps construction that already exists; it does not move it):

    with BOOT.phase("devbio", "mount the patient buses") as p:
        cluster = CohortCluster(...)
        p.detail(beds=len(cluster.node_ids), drivers=3)

A phase that raises is recorded as failed, with the exception, and the exception
still propagates — this observes the boot, it does not swallow it.

For long runs of module-level construction, where wrapping every block in a
``with`` would re-indent hundreds of lines, use checkpoints instead:

    BOOT.checkpoint("session", "load the calibrated detector")
    session = AegisSession(...)
    BOOT.checkpoint("wing", "mount the ICU wing")      # closes "session"
    ...
    BOOT.ready()                                       # closes the last one

If the process dies mid-phase (an import error kills uvicorn before it listens),
an ``atexit`` hook records the open phase as failed and flushes the journal, so
the one boot that most needs a log still leaves one.
"""
from __future__ import annotations

import atexit
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.aegis.kernel.journal import ERR, INFO, Journal, KJOURNAL


@dataclass
class Phase:
    name: str
    summary: str
    started_at: float
    duration_ms: float = 0.0
    ok: bool = True
    error: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)
    t0: float = 0.0                       # perf_counter at start (not serialised)
    closed: bool = False

    def detail(self, **kw: Any) -> "Phase":
        """Record what this phase produced — counts, sizes, versions."""
        self.info.update(kw)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"phase": self.name, "summary": self.summary,
                "ms": round(self.duration_ms, 1), "ok": self.ok,
                "error": self.error, "detail": self.info,
                "at": round(self.started_at, 3)}


class BootSequencer:
    """Records the phases of one boot, in order, with real durations."""

    def __init__(self, journal: Optional[Journal] = None) -> None:
        self.started_at = time.time()
        self.phases: List[Phase] = []
        self.journal = journal if journal is not None else KJOURNAL
        self._ready_at: Optional[float] = None
        self._open: Optional[Phase] = None

    @contextmanager
    def phase(self, name: str, summary: str = ""):
        p = Phase(name=name, summary=summary, started_at=time.time())
        self.phases.append(p)
        t0 = time.perf_counter()
        try:
            yield p
        except BaseException as exc:                  # noqa: BLE001 - re-raised below
            p.ok = False
            p.error = f"{type(exc).__name__}: {exc}"
            p.duration_ms = (time.perf_counter() - t0) * 1000.0
            self.journal.append("boot", "phase.failed", severity=ERR, phase=name,
                                ms=round(p.duration_ms, 1), error=p.error)
            raise
        else:
            p.duration_ms = (time.perf_counter() - t0) * 1000.0
            self.journal.append("boot", "phase", severity=INFO, phase=name,
                                ms=round(p.duration_ms, 1), **p.info)

    # -- checkpoint style (module-level construction) --------------------- #
    def checkpoint(self, name: str, summary: str = "", **detail: Any) -> Phase:
        """Close the open checkpoint (successfully) and open ``name``."""
        self._close_open()
        p = Phase(name=name, summary=summary, started_at=time.time(),
                  info=dict(detail), t0=time.perf_counter())
        self.phases.append(p)
        self._open = p
        return p

    def note(self, **detail: Any) -> None:
        """Attach what the open checkpoint produced."""
        if self._open is not None:
            self._open.info.update(detail)

    def _close_open(self) -> None:
        p = self._open
        if p is None or p.closed:
            return
        p.duration_ms = (time.perf_counter() - p.t0) * 1000.0
        p.closed = True
        self._open = None
        self.journal.append("boot", "phase", severity=INFO, phase=p.name,
                            ms=round(p.duration_ms, 1), **p.info)

    def abandon(self, reason: str = "process exited during this phase") -> Optional[str]:
        """Record the open checkpoint as FAILED. Called at exit if boot never
        finished; returns the phase name it marked, or None."""
        p = self._open
        if p is None or p.closed or self._ready_at is not None:
            return None
        p.duration_ms = (time.perf_counter() - p.t0) * 1000.0
        p.ok, p.error, p.closed = False, reason, True
        self._open = None
        self.journal.append("boot", "phase.failed", severity=ERR, phase=p.name,
                            ms=round(p.duration_ms, 1), error=reason)
        return p.name

    def ready(self) -> float:
        """Mark the kernel up. Returns total boot seconds."""
        self._close_open()
        self._ready_at = time.time()
        total = self._ready_at - self.started_at
        self.journal.append("boot", "ready", severity=INFO,
                            seconds=round(total, 2), phases=len(self.phases))
        return total

    # -- views ------------------------------------------------------------ #
    def log(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.phases]

    def summary(self) -> Dict[str, Any]:
        done = self._ready_at is not None
        slowest = max(self.phases, key=lambda p: p.duration_ms) if self.phases else None
        return {
            "booted": done,
            "boot_seconds": round((self._ready_at or time.time()) - self.started_at, 2),
            "phases": len(self.phases),
            "failed": [p.name for p in self.phases if not p.ok],
            "slowest": None if slowest is None else
                       {"phase": slowest.name, "ms": round(slowest.duration_ms, 1)},
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "executable": sys.executable,
            "log": self.log(),
        }


#: The process-wide boot record.
BOOT = BootSequencer()


@atexit.register
def _boot_died() -> None:                          # pragma: no cover - exit path
    if BOOT.abandon() is not None and BOOT.journal.store is not None:
        try:
            BOOT.journal.store.flush()
        except Exception:
            pass
