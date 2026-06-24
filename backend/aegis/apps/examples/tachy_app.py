"""tachy.app — a small, well-behaved third-party app (the SDK in <40 lines).

Declares it reads only heart rate + MAP and may emit only an ALERT, computes a
shock index, and raises a calibrated alert through the cage. It imports nothing
dangerous, so the package registry's static audit admits it; the capability
context confines it at runtime. This is the template a third party copies.
"""
from __future__ import annotations

from typing import Any, Dict

from backend.aegis.apps.sdk import AegisApp, AppContext, AppManifest
from backend.aegis.kernel.neurosym import ALERT


class TachyApp(AegisApp):
    manifest = AppManifest(
        name="tachy",
        version="0.1.0",
        author="community",
        summary="Shock-index alert from heart rate + MAP.",
        reads=("hr", "map"),
        emits=(ALERT,),
        lane_class="ml_tracking",
        lane_budget=1,
    )

    def on_tick(self, ctx: AppContext) -> Dict[str, Any]:
        hr = ctx.read("hr") or 0.0
        mapv = ctx.read("map") or 1.0
        shock = hr / max(mapv, 1.0)                     # shock index
        p = max(0.0, min(0.99, (shock - 0.6)))          # crude calibrated proxy
        severity = 3 if p >= 0.70 else 2 if p >= 0.40 else 1 if p >= 0.20 else 0
        cert = ctx.emit(ALERT, {"severity": severity},
                        {"probability": p, "calibrated": True, "evidence_count": 1})
        return {"shock_index": round(shock, 3), "severity": severity,
                "alert_certified": cert.ok}
