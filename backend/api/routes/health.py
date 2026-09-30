"""Liveness, subsystem health, prover health and the model audit.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

import backend.aegis.server as S
from backend.aegis.server import (
    _PROVER,
    _live_mode,
    cluster,
    session,
)

router = APIRouter()


@router.get("/aegis/health")
def health() -> Dict[str, Any]:
    # The prover's state is part of health, not a log line. If the shared Z3
    # context is corrupted the OS can still serve reads, but it can no longer
    # certify anything — and a monitor that only ever sees "ok" would never learn
    # that the safety gate has stopped issuing verdicts.
    prover = _PROVER.status()
    return {"status": "ok" if prover["healthy"] else "degraded",
            "source": session.source,
            "mode": "live" if _live_mode() else "demo",
            "persistence": bool(S._db is not None and getattr(S._db, "use_supabase", False)),
            "model_loaded": session.core.model is not None,
            "prover": {"healthy": prover["healthy"],
                       "fail_closed": prover["fail_closed"],
                       "failures": prover["failures"],
                       "last_error": prover["last_error"]},
            "schema_version": session.snapshot()["schema_version"]}


@router.get("/aegis/health/system")
def health_system() -> Dict[str, Any]:
    """How well the detector actually performs, and where it falls short.

    Reads the cross-hospital evaluation artefact rather than measuring anything
    live: the demo cohort is twelve stays, which cannot support an AUROC. Strengths
    and gaps are derived from those numbers, so the page cannot drift into being a
    brochure."""
    from backend.aegis.system_health import report
    out = report()
    # The live half: subsystem state the artefact cannot know about.
    prover = _PROVER.status()
    out["subsystems"] = [
        {"name": "Detector", "ok": session.core.model is not None,
         "note": ("stacked ensemble loaded" if session.core.model is not None
                  else "no model artefact — running on rules only")},
        {"name": "Safety prover", "ok": prover["healthy"],
         "note": ("Z3 context healthy" if prover["healthy"]
                  else "CORRUPTED — failing closed, nothing is being certified")},
        {"name": "Data source", "ok": True,
         "note": f"{session.source} · {'live' if _live_mode() else 'demo'}"},
        {"name": "Persistence", "ok": bool(S._db is not None and getattr(S._db, "use_supabase", False)),
         "note": ("Supabase connected" if (S._db is not None and getattr(S._db, "use_supabase", False))
                  else "local only — nothing is being retained")},
    ]
    try:
        out["cohort"] = cluster.validation(alarm=0.70) if hasattr(cluster, "validation") else None
    except Exception:
        out["cohort"] = None
    return out


@router.get("/aegis/audit")
def model_audit() -> Dict[str, Any]:
    """What the shipped detector's performance is actually made of.

    Serves ``artifacts/model_audit.json`` rather than measuring live, for two reasons.
    The suspicion-echo probe fits two models, which does not belong in a request handler.
    And a number a judge reads should be the number that was committed and can be
    reproduced from the repository, not one a warm cache happened to produce.

    Regenerate with ``python -m backend.ml.scripts.run_audit``.
    """
    import json as _json
    import os as _os
    from backend.ml import config as _cfg
    path = _os.path.join(_cfg.ARTIFACTS_DIR, "model_audit.json")
    try:
        with open(path) as fh:
            payload = _json.load(fh)
    except (OSError, ValueError):
        # Absent is a normal state, not an error: a machine without the artefact should
        # say the audit has not been run rather than invent one.
        return {"available": False,
                "reason": "artifacts/model_audit.json not present — run "
                          "python -m backend.ml.scripts.run_audit"}
    payload["available"] = True
    return payload


@router.get("/aegis/health/prover")
def health_prover() -> Dict[str, Any]:
    """Full detail on the shared Z3 context, including the recent failure log.

    Separate from /aegis/health because the failure log is the thing you want when
    diagnosing, and noise in a liveness probe is the thing you do not."""
    st = _PROVER.status()
    st["note"] = (
        "A corrupted Z3 context cannot be repaired in-process. While unhealthy the "
        "safety cage fails CLOSED — no protocol is certified — and the simulator "
        "drive loop halts rather than keep asking a prover it does not trust. "
        "Recovery is a process restart.")
    return st

