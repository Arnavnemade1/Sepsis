"""The Temporal Causal Graph: each patient's clinical story, incidents, replay and mined patterns.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter

from backend.api.schemas import (
    ReplayIn,
)
from backend.aegis.server import (
    _causal_for,
    _replay_for,
    session,
)

router = APIRouter()


@router.get("/aegis/causal")
def causal_graph(bed: Optional[str] = None) -> Dict[str, Any]:
    """The tamper-evident causal graph for a patient (live session or any cohort bed,
    built on demand from its real recorded history). Each edge carries its
    evidence_type (z3_proven | model_predicted | observed_association)."""
    ing = _causal_for(bed)
    if ing is None:
        return {"status": "error", "detail": f"no such patient '{bed}'"}
    return {"patient": bed or session.patient_id,
            "summary": ing.summary(), "graph": ing.graph.to_dict()}


@router.get("/aegis/causal/incident/{node_id}")
def causal_incident(node_id: str, bed: Optional[str] = None) -> Dict[str, Any]:
    """Auto-generated incident report: traverse backward from an outcome node to
    the earliest point an intervention could have changed the trajectory."""
    ing = _causal_for(bed)
    if ing is None:
        return {"status": "error", "detail": f"no such patient '{bed}'"}
    try:
        return ing.graph.incident_report(node_id)
    except KeyError:
        return {"status": "error", "detail": f"no such node '{node_id}'"}


@router.get("/aegis/causal/patterns")
def causal_patterns(min_support: int = 5, min_lift: float = 1.2,
                    window_h: float = 4.0) -> Dict[str, Any]:
    """Epidemiological pattern mining across the persisted cohort of causal graphs.
    Returns observed associations with support, Wilson CI, baseline + lift — NOT
    causal guidelines (confounded by indication; disclosed in the payload)."""
    from backend.aegis.causal_mine import PatternMiner
    store = getattr(session.causal, "store", None)
    graphs = []
    if store is not None:
        try:
            graphs = store.load_all()
        except Exception as e:
            print(f"[patterns] cohort load failed: {e}")
    # Always include the live graph (it may not be persisted in this config).
    live = session.causal.graph
    if all(g.bed != live.bed for g in graphs):
        graphs.append(live)
    res = PatternMiner(graphs, outcome_window_h=window_h).mine(
        min_support=min_support, min_lift=min_lift)
    return res


@router.post("/aegis/causal/replay")
def causal_replay(payload: ReplayIn) -> Dict[str, Any]:
    """Counterfactual replay: 'what would have happened under an alternative action?'
    Treatment effect is measured model-internally (alternative vs action-taken),
    scored by the live model; recorded reality is returned for context."""
    eng = _replay_for(payload.bed)
    if eng is None:
        return {"status": "error", "detail": f"no such patient '{payload.bed}'"}
    try:
        if payload.outcome_id:
            return eng.replay_earliest(payload.outcome_id, payload.treatment,
                                       horizon=payload.horizon)
        if payload.node_id:
            return eng.replay(payload.node_id, payload.treatment, horizon=payload.horizon)
        return {"status": "error", "detail": "provide outcome_id or node_id"}
    except (KeyError, ValueError) as e:
        return {"status": "error", "detail": str(e)}

