"""A single process-wide lock serialising all Z3 solver access.

The z3-solver Python bindings default to ONE global Z3 context, and that context
is **not thread-safe**: two threads building/solving at the same time corrupt it and
raise ``Z3Exception: b'unreachable'`` (or segfault). AegisOS touches Z3 from several
threads at once — the cognitive scheduler verifies every tick on the cluster loop,
the orchestrator warms up on a daemon thread, and FastAPI runs sync request handlers
(treat/proof/consensus) in a threadpool. On a slow host those overlap and crash the
process (observed: it killed the Render deploy at import, in cluster.run's per-tick
scheduler verification racing the boot warmup thread).

Every site that creates/uses a Z3 ``Solver`` must hold this re-entrant lock. It is
an ``RLock`` so a guarded call that nests another guarded call on the same thread
(e.g. run_scenario → cluster.tick → scheduler verify) does not deadlock.
"""
from __future__ import annotations

import functools
import os
import threading
import time
from typing import Any, Dict, List, Optional

Z3_LOCK = threading.RLock()

# --------------------------------------------------------------------------- #
# Prover health
# --------------------------------------------------------------------------- #
# A Z3Exception from a guarded call is not an ordinary error. The bindings share
# one global context, and once it is corrupted it stays corrupted: every later
# solve on that context is unreliable, and some fail in a finalizer where there is
# no stack to read. Treating it like a transient tick failure — log it, sleep, try
# again — is what let a broken prover keep running in production.
#
# It matters more than an ordinary crash because this prover is what the OS uses to
# decide a treatment is safe. A corrupted context does not reliably raise; it can
# also return a WRONG verdict. So the escalation is not "shout louder", it is:
#
#   1. mark the prover unhealthy,
#   2. FAIL CLOSED — refuse to certify anything at all, rather than emit a
#      certificate whose provenance we no longer trust. Default-deny is the same
#      rule the egress cage already applies to every write.
#   3. surface it in /aegis/health so the state is visible rather than inferred
#      from log noise,
#   4. and after repeated failures, exit so the platform restarts the process. A
#      fresh process is the only way back to a clean context; there is no in-place
#      repair.
#
# Set AEGIS_Z3_EXIT_ON_CORRUPTION=0 to keep a degraded process alive for debugging
# (it stays fail-closed either way).
_EXIT_AFTER = int(os.environ.get("AEGIS_Z3_FAIL_LIMIT", "3"))
_EXIT_ENABLED = os.environ.get("AEGIS_Z3_EXIT_ON_CORRUPTION", "1") != "0"


class _ProverHealth:
    """Process-wide health of the shared Z3 context."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.healthy = True
        self.failures = 0
        self.total_failures = 0
        self.first_error: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_error_at: float = 0.0
        self.log: List[Dict[str, Any]] = []

    def record_failure(self, exc: BaseException, where: str = "") -> None:
        with self._lock:
            self.failures += 1
            self.total_failures += 1
            msg = f"{type(exc).__name__}: {exc}"
            self.last_error, self.last_error_at = msg, time.time()
            if self.first_error is None:
                self.first_error = msg
            self.healthy = False
            self.log = (self.log + [{"at": round(time.time(), 3), "where": where,
                                     "error": msg}])[-16:]
            n = self.failures
        try:
            from backend.aegis.kernel.journal import KJOURNAL, CRIT
            KJOURNAL.append("kern", "prover_corrupted", severity=CRIT,
                            where=where or "?", error=msg, failures=n)
            if _EXIT_ENABLED and n >= _EXIT_AFTER and KJOURNAL.store is not None:
                KJOURNAL.store.flush()                 # os._exit skips atexit
        except Exception:
            pass
        print(f"[aegis] Z3 PROVER CORRUPTED ({n}/{_EXIT_AFTER}) in {where or '?'}: {msg}\n"
              f"[aegis] the shared Z3 context cannot be repaired in-process — the "
              f"safety cage is now FAILING CLOSED (nothing will be certified).",
              flush=True)
        if _EXIT_ENABLED and n >= _EXIT_AFTER:
            print("[aegis] exiting so the platform restarts with a clean Z3 context.",
                  flush=True)
            # os._exit, not sys.exit: this runs on worker threads where a raised
            # SystemExit would simply be swallowed by the thread, leaving a broken
            # process alive — which is the exact failure being escalated.
            os._exit(70)                       # EX_SOFTWARE

    def record_success(self) -> None:
        # Deliberately does NOT clear `healthy`. A context that has thrown once is
        # not trustworthy again because the next call happened to work; corruption
        # is silent between symptoms. Only a fresh process clears this.
        pass

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"healthy": self.healthy, "failures": self.total_failures,
                    "first_error": self.first_error, "last_error": self.last_error,
                    "last_error_at": round(self.last_error_at, 3) or None,
                    "fail_closed": not self.healthy,
                    "exit_after": _EXIT_AFTER if _EXIT_ENABLED else None,
                    "recent": list(self.log)}


PROVER = _ProverHealth()


def prover_healthy() -> bool:
    return PROVER.healthy


class ProverUnavailable(RuntimeError):
    """Raised when the shared Z3 context is known-corrupted. Callers must treat
    this as 'not certified', never as 'certified'."""


def z3_guarded(fn):
    """Decorator: run ``fn`` holding the global Z3 lock (blocking). Use on any
    function/method that builds or solves a Z3 problem. Re-entrant, so a guarded
    call that nests another on the same thread is fine. Do NOT put this on code
    that runs on the asyncio event loop — that would freeze the loop; those paths
    (the scheduler's per-tick verify) use a non-blocking acquire instead, and the
    simulator drive loop hops to a worker thread before touching Z3."""
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        if not PROVER.healthy:
            # Fail closed rather than solve on a context we no longer trust.
            raise ProverUnavailable(
                f"Z3 context corrupted earlier ({PROVER.last_error}); refusing to "
                "produce a safety verdict from it")
        with Z3_LOCK:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:                      # noqa: BLE001
                if _is_z3_corruption(exc):
                    PROVER.record_failure(exc, getattr(fn, "__qualname__", str(fn)))
                raise
    return _wrapped


def _is_z3_corruption(exc: BaseException) -> bool:
    """Context corruption, as opposed to a legitimate modelling error.

    Z3Exception covers both — asking for a model when the result was unsat is a
    Z3Exception too, and that is a caller bug, not a broken context. Only the
    failure modes that indicate the context itself is unusable escalate."""
    try:
        from z3.z3types import Z3Exception
    except Exception:                                     # pragma: no cover
        return False
    if not isinstance(exc, Z3Exception):
        return False
    text = str(exc).lower()
    # Specific markers only. A bare "context" match was too broad — it would swallow
    # legitimate modelling errors that merely mention the word and take the whole
    # process down for a caller bug. Asking for a model after an unsat result also
    # raises Z3Exception, and that is a bug in the caller, not a poisoned context.
    return any(m in text for m in (
        "unreachable",
        "ast is not an expression",
        "sort mismatch",
        "invalid context",
        "context mismatch",
        "assertion violation",
        "canceled",              # solver aborted mid-solve on a broken context
    ))
