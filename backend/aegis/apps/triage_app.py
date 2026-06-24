"""triage.app — a service that DEPENDS on other services (inter-app IPC).

Proves the platform's dependency model: triage declares ``depends=(resonance,
radar)`` so aegisd starts it after both, and it reads their published outputs from
``/proc/resonance`` and ``/proc/radar`` (capability-checked) to emit a single
consolidated clinical NOTE through the Z3 cage. It reads no raw sensors itself —
its only inputs are other services' certified results.
"""
from __future__ import annotations

from typing import Any, Dict

from backend.aegis.apps.sdk import AegisApp, AppContext, AppManifest
from backend.aegis.kernel.neurosym import NOTE

_LEVEL_RISK = {"CRITICAL": "HIGH", "WATCH": "MEDIUM", "STABLE": "LOW"}


class TriageApp(AegisApp):
    manifest = AppManifest(
        name="triage",
        version="1.0.0",
        author="aegis-core",
        summary="Consolidates resonance + radar into one certified triage note.",
        reads=(),
        emits=(NOTE,),
        proc_reads=("/proc/resonance", "/proc/radar"),
        depends=("resonance", "radar"),
        lane_class="logging",
        lane_budget=1,
    )

    def on_tick(self, ctx: AppContext) -> Dict[str, Any]:
        res = ctx.read_proc("/proc/resonance") or {}
        rad = ctx.read_proc("/proc/radar") or {}
        risk = float(res.get("gated_risk", 0.0) or 0.0)
        ews = float(rad.get("ews", 0.0) or 0.0)
        level = ("CRITICAL" if risk >= 0.70 else
                 "WATCH" if (risk >= 0.40 or ews >= 0.50) else "STABLE")
        risk_level = _LEVEL_RISK[level]
        text = (f"Triage {level}: fused sepsis risk {risk:.2f}"
                + (f" (artifact-rejected on {res['uncorroborated_pathway']})"
                   if res.get("isolated") else "")
                + f"; resilience-loss {ews:.2f}"
                + (" ▲rising" if rad.get("rising") and not rad.get("warming") else "") + ".")
        cert = ctx.emit(NOTE, {"risk_level": risk_level, "text": text,
                               "recommends_treatment": False},
                        {"risk_level": risk_level})
        return {"level": level, "fused_risk": round(risk, 3),
                "resilience_loss": round(ews, 3), "note_certified": cert.ok,
                "note": text}
