"""Strict-JSON helpers for artifacts that leave Python.

WHY THIS EXISTS
---------------
``json.dump`` happily writes bare ``NaN`` and ``Infinity``. Python reads them back
without complaint, so an artifact written that way looks fine from inside Python and
is unreadable to every strict parser outside it — Node's ``JSON.parse`` included.
``artifacts/transfer.json`` was written that way once and broke the website's own
sync script, which is exactly the kind of failure that surfaces far from its cause.

NaN is usually MEANINGFUL when it appears here — "this statistic does not exist for
this row" (a median over an empty set, a bootstrap that could not resolve) — so it
becomes ``null`` rather than being coerced to a number. A null survives the round
trip and still reads as absent; a zero would be a lie.

This lived inside ``backend/ml/scripts/derive_transfer.py`` until a second and third
artifact writer needed it. A guard that only one writer can reach is not a guard.
"""
from __future__ import annotations

import math
from typing import Any


def json_safe(o: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Handles dicts, lists and tuples; anything else is returned unchanged. Numpy
    scalars are covered because ``np.float64`` is a subclass of ``float``.
    """
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o
