"""radar.app — Critical-Slowing-Down early-warning, as a stateful service.

Wraps the CSD indicators from :mod:`backend.aegis.resilience`. Unlike a stateless
detector, this app keeps its OWN rolling window across ticks and reports a
loss-of-resilience score (rising autocorrelation + widening variance vs the
patient's own early baseline). It declares it reads only the CSD vital channels
and emits NO egress — it is an honest *research signal*, surfaced through
``/proc/radar`` for downstream services (e.g. triage), never a calibrated alarm.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict

import numpy as np

from backend.aegis.apps.sdk import AegisApp, AppContext, AppManifest
from backend.aegis.resilience import BASE_W, CSD_CHANNELS, WIN, _indicators


class RadarApp(AegisApp):
    manifest = AppManifest(
        name="radar",
        version="1.0.0",
        author="aegis-core",
        summary="Critical-Slowing-Down resilience radar (research signal).",
        reads=tuple(CSD_CHANNELS),
        emits=(),                                   # research signal — no calibrated alarm
        lane_class="ml_tracking",
        lane_budget=1,
    )

    def __init__(self) -> None:
        self._win: Dict[str, deque] = {c: deque(maxlen=WIN * 3) for c in CSD_CHANNELS}
        self._base: Dict[str, tuple] = {}           # frozen (ar1, var) baseline per channel
        self._base_buf: Dict[str, list] = {c: [] for c in CSD_CHANNELS}
        self._n = 0
        self._last_ews = 0.0

    def on_tick(self, ctx: AppContext) -> Dict[str, Any]:
        self._n += 1
        rises = []
        for ch in CSD_CHANNELS:
            v = ctx.read(ch)
            if v is None or (isinstance(v, float) and v != v):
                continue
            w = self._win[ch]
            w.append(float(v))
            if len(w) < WIN:
                continue
            ind = _indicators(np.asarray(list(w)[-WIN:], dtype=float))
            if ind is None:
                continue
            ar1, var = ind
            if self._n <= WIN + BASE_W:              # still collecting baseline
                self._base_buf[ch].append((ar1, var))
                continue
            if ch not in self._base and self._base_buf[ch]:
                arr = np.asarray(self._base_buf[ch])
                self._base[ch] = (float(np.median(arr[:, 0])),
                                  float(np.median(arr[:, 1])) or 1e-6)
            if ch in self._base:
                b_ar1, b_var = self._base[ch]
                rises.append(max(0.0, ar1 - b_ar1) + max(0.0, var / b_var - 1.0))
        warming = self._n <= WIN + BASE_W
        ews = float(np.clip(np.mean(rises) / 2.0, 0.0, 1.0)) if rises else 0.0
        out = {"ews": round(ews, 3), "rising": ews > self._last_ews + 1e-3,
               "warming": warming, "n": self._n}
        self._last_ews = ews
        return out
