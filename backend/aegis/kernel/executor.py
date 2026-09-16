"""The kernel executor — the single thread that owns every write into the wing.

WHY THIS EXISTS
---------------
``kernel/memfabric.py`` states the correctness premise of the whole memory fabric
in its header: each ring has exactly ONE writer, readers trail the write pointer
and never block it. That premise was not enforced, and it was being violated in
production: ``_cluster_loop`` ticks the cluster from the event-loop thread every
2 s, while ``POST /aegis/orchestrator/scenario`` is a sync handler — so Starlette
runs it on a threadpool worker, where it ticks the SAME cluster (orchestrator.py
``_run_scenario``). Two threads, one ring, interleaved scalar stores and a shared
``write_index``. Under CPython the GIL keeps each store atomic, so this corrupts
ordering rather than memory: a reader can observe a slot whose fields come from
two different samples, which is exactly the "torn read" the sequence stamps are
supposed to make impossible.

This module fixes it the way an OS does: writes do not happen wherever the caller
happens to be running. They are funnelled onto one owner thread.

  * :meth:`KernelExecutor.call` runs ``fn`` on the writer thread and waits for it.
    Called FROM the writer thread it runs inline, so a nested write (scenario →
    tick → push) does not deadlock on itself.
  * :meth:`KernelExecutor.try_call` refuses rather than queues when the writer is
    busy. The live cluster loop uses this: if a 30 s scenario owns the writer, the
    loop drops its tick and says so, instead of building a backlog of stale ticks
    that would all execute in a burst afterwards.
  * :func:`assert_writer` is the enforcement. ``LocklessRing.push`` calls it, so a
    second writer now raises :class:`WriterViolation` instead of silently tearing
    a sample.

LOCK ORDER (there are exactly two locks in the write path, and this is the rule):
**executor → Z3, never the reverse.** A caller wanting both takes the executor
first (``run_scenario`` does). The writer thread only ever acquires Z3
non-blockingly (``CognitiveScheduler.verify_decision_z3`` defers when the prover
is busy), so it cannot sit blocked on a lock that a thread waiting for the writer
is holding. That is what makes the pair acyclic.

HONEST FRAMING: this is a serialization boundary in one process, not a hardware
memory model. It does not make the rings lock-free across cores; it makes the
single-writer discipline that the lock-free design depends on *checked* rather
than *assumed*.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional


class WriterViolation(RuntimeError):
    """Raised when a thread that does not own the write path tries to write."""


# The identity of the bound writer thread. ``None`` means no executor has been
# started (unit tests, verify suites, offline tools) and enforcement is off — a
# single-threaded caller is trivially a single writer.
_WRITER_IDENT: Optional[int] = None
_VIOLATIONS: list = []


def assert_writer(what: str = "ring write") -> None:
    """Fail loudly if the calling thread is not the bound writer.

    Deliberately a module-level function with a module-level global rather than a
    method: this sits in ``LocklessRing.push``, the hottest path in the fabric, so
    it must cost one global load and one integer compare (~100 ns) and must not
    require the ring to hold a reference to anything.
    """
    ident = _WRITER_IDENT
    if ident is not None and threading.get_ident() != ident:
        cur = threading.current_thread().name
        _VIOLATIONS.append({"what": what, "thread": cur, "ts": time.time()})
        raise WriterViolation(
            "%s attempted from thread %r, which does not own the write path. "
            "Route it through KernelExecutor.call() — see kernel/executor.py."
            % (what, cur))


def violations() -> list:
    """Every refused write since boot (empty is the healthy state)."""
    return list(_VIOLATIONS)


class KernelExecutor:
    """A single-threaded executor that owns the wing's write path."""

    def __init__(self, name: str = "aegis-writer") -> None:
        self._q: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._ready = threading.Event()
        self._busy = False
        self._started = False
        self.calls = 0
        self.skipped = 0
        self.max_wait_ms = 0.0
        self.max_run_ms = 0.0

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> "KernelExecutor":
        global _WRITER_IDENT
        if self._started:
            return self
        self._thread.start()
        # Wait for THIS thread to publish its identity, so the first call cannot
        # race enforcement (it would look like a foreign writer and raise). Waiting
        # on the global instead would return immediately if another executor had
        # already been started in this process — which is exactly what the verify
        # suite does — and hand back an executor whose thread is not yet running.
        self._ready.wait(5.0)
        self._started = True
        return self

    def stop(self) -> None:
        """Stop the writer thread and UNBIND enforcement.

        The bound identity is process-global, so a test that starts an executor
        and walks away leaves every later ring write in that process looking like
        a foreign writer. That is not hypothetical — it broke five kernel suites
        the first time this ran inside the aggregate runner. Anything that starts
        an executor outside the server must stop it in a ``finally``.
        """
        global _WRITER_IDENT
        if not self._started:
            return
        self._q.put(None)
        self._thread.join(5.0)
        self._started = False
        _WRITER_IDENT = None

    def _run(self) -> None:
        global _WRITER_IDENT
        _WRITER_IDENT = threading.get_ident()
        self._ready.set()
        while True:
            job = self._q.get()
            if job is None:
                break
            fn, args, kwargs, box, done = job
            self._busy = True
            t0 = time.perf_counter()
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:           # noqa: BLE001 - relayed to caller
                box["error"] = exc
            finally:
                self.max_run_ms = max(self.max_run_ms,
                                      (time.perf_counter() - t0) * 1000.0)
                self._busy = False
                done.set()

    # -- submission ------------------------------------------------------- #
    def is_writer(self) -> bool:
        return _WRITER_IDENT is not None and threading.get_ident() == _WRITER_IDENT

    @property
    def busy(self) -> bool:
        return self._busy or not self._q.empty()

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` on the writer thread and return its result (or re-raise)."""
        if not self._started or self.is_writer():
            # Not started: the caller IS the only thread. Already the writer: a
            # nested write, which must run inline or it would wait on itself.
            return fn(*args, **kwargs)
        self.calls += 1
        box: Dict[str, Any] = {}
        done = threading.Event()
        t0 = time.perf_counter()
        self._q.put((fn, args, kwargs, box, done))
        done.wait()
        self.max_wait_ms = max(self.max_wait_ms, (time.perf_counter() - t0) * 1000.0)
        if "error" in box:
            raise box["error"]
        return box.get("result")

    #: Sentinel returned by :meth:`try_call` when the writer was busy.
    BUSY = object()

    def try_call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` if the writer is free; otherwise return :data:`BUSY`.

        For periodic work whose value expires: a dropped tick is honest, a queued
        one is a lie told 30 seconds late.
        """
        if not self._started or self.is_writer():
            return fn(*args, **kwargs)
        if self.busy:
            self.skipped += 1
            return KernelExecutor.BUSY
        return self.call(fn, *args, **kwargs)

    def status(self) -> Dict[str, Any]:
        return {
            "started": self._started,
            "busy": self.busy,
            "calls": self.calls,
            "skipped_ticks": self.skipped,
            "max_wait_ms": round(self.max_wait_ms, 2),
            "max_run_ms": round(self.max_run_ms, 2),
            "violations": len(_VIOLATIONS),
        }


#: The process-wide writer. ``server.py`` starts it; everything else imports it.
EXEC = KernelExecutor()
