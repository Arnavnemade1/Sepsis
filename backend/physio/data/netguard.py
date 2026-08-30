"""One place that makes network-backed data loading interruptible.

WHY THIS IS ITS OWN MODULE
--------------------------
The same failure has now happened twice, in two different adapters, for the same reason.

  * ``mimic3_open`` — a retrieval run sat ALIVE for 4h32m having made no progress for the
    last 4h22m, parked inside a ``wfdb`` HTTP read with no timeout.
  * ``vitaldb_open`` — after fixing the above, this module's docstring claimed the lesson
    was "applied here from the start rather than after the second incident". It was not:
    only the retry wrapper was copied across, not the socket deadline. A retry does
    nothing against a hang, because a wedged call never raises and so is never retried.
    Discovery duly hung at 1541 of 1914 cases and lost 3h32m.

Two adapters, two hangs, one root cause and one forgotten line. That is a sign the
safety belongs in a shared place rather than in a convention each new adapter is
trusted to remember, so it lives here and both import it.

THE THREE LAYERS, AND WHY EACH IS NEEDED
-----------------------------------------
1. ``install_timeouts()`` — a process-wide socket deadline. This is the layer that
   actually addresses the observed hangs, and it is the one that was missed.
2. ``retry()`` — for reads that *fail* rather than hang. Bounded attempts, then give up
   and return ``None``: a skipped item costs a row, a retried-forever item costs the job.
3. ``deadline()`` — a SIGALRM watchdog around a whole unit of work, because a socket
   deadline covers a stalled read and not every way a library can wedge (a decompression
   loop, a retry buried inside the client, a DNS resolver).
"""
from __future__ import annotations

import signal
import socket
import time
from contextlib import contextmanager
from typing import Callable, Optional

#: Process-wide socket deadline, in seconds.
NET_TIMEOUT_S = 30.0

#: Courtesy pause after a successful read. These are free public research resources
#: serving thousands of requests from this pipeline.
REQUEST_PAUSE_S = 0.2


def install_timeouts(seconds: float = NET_TIMEOUT_S) -> None:
    """Set the process-wide socket deadline. Call at adapter import, not at use."""
    socket.setdefaulttimeout(seconds)


class Timeout(Exception):
    """A unit of work exceeded its wall-clock budget."""


@contextmanager
def deadline(seconds: int):
    """Abort the enclosed block after ``seconds``.

    SIGALRM because it is the only thing that reliably interrupts a blocked C-level
    read. Requires the main thread of a single-threaded loop, which is what every
    retrieval script here is.
    """
    def _fire(signum, frame):
        raise Timeout(f"exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def retry(fn: Callable, attempts: int = 2, backoff: float = 2.0,
          pause: float = REQUEST_PAUSE_S):
    """Run a network read; return ``None`` rather than raising or hanging forever."""
    for i in range(attempts):
        try:
            out = fn()
            if pause:
                time.sleep(pause)
            return out
        except Timeout:
            raise                      # a watchdog firing must not be swallowed here
        except Exception:
            if i + 1 >= attempts:
                return None
            time.sleep(backoff * (i + 1))
    return None
