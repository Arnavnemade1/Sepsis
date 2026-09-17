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
  * Ownership is PER ARENA. :meth:`KernelExecutor.own` binds one memory arena (the
    live wing's) to the writer thread; ``LocklessRing.push`` checks its own arena's
    owner, so a second writer into the wing raises :class:`WriterViolation`
    instead of silently tearing a sample — while a throwaway cluster somewhere else
    in the process (the orchestrator warm-up, a verify suite, the load harness)
    has an unowned arena and writes freely, because nothing else can reach it.

    The first version bound ONE thread for the whole process. That refused the
    warm-up's writes into its own private 2-bed cluster, so the synthesis warm-up
    silently skipped on every boot after it shipped and the first clinician click
    paid the full cold-start cost again. It also made every suite that started an
    executor responsible for un-binding the whole interpreter afterwards.

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


_VIOLATIONS: list = []


def refuse_write(what: str) -> None:
    """Record and raise a refused write. Only reached on an actual violation — the
    hot path in ``LocklessRing.push`` is one attribute load and one compare, and
    formats nothing unless the check fails."""
    cur = threading.current_thread().name
    _VIOLATIONS.append({"what": what, "thread": cur, "ts": time.time()})
    raise WriterViolation(
        "%s attempted from thread %r, which does not own this arena's write path. "
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
        self._ident: Optional[int] = None
        self._owned: list = []

    # -- ownership -------------------------------------------------------- #
    def own(self, arena: Any) -> "KernelExecutor":
        """Make this executor's thread the only writer into ``arena``."""
        if not self._started:
            raise RuntimeError("start() the executor before it can own an arena")
        arena.writer_ident = self._ident
        if arena not in self._owned:
            self._owned.append(arena)
        return self

    def owns(self, arena: Any) -> bool:
        return (arena is not None and self._started
                and getattr(arena, "writer_ident", None) == self._ident)

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> "KernelExecutor":
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
        """Stop the writer thread and release every arena it owned."""
        if not self._started:
            return
        self._q.put(None)
        self._thread.join(5.0)
        self._started = False
        for arena in self._owned:
            if getattr(arena, "writer_ident", None) == self._ident:
                arena.writer_ident = None
        self._owned = []

    def _run(self) -> None:
        self._ident = threading.get_ident()
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
        return self._ident is not None and threading.get_ident() == self._ident

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
            "owned_arenas": len(self._owned),
            "violations": len(_VIOLATIONS),
        }


#: The process-wide writer. ``server.py`` starts it; everything else imports it.
EXEC = KernelExecutor()
