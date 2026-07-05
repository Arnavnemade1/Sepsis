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

import threading

Z3_LOCK = threading.RLock()
