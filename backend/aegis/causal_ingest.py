"""Causal-graph ingestor — distills the live assessment stream into the clinical story.

``AegisSession.tick`` produces one :class:`AegisAssessment` per sample (potentially
many per second). Writing a graph node per tick would bury the signal in noise and
blow up the ledger. This ingestor instead records **clinically meaningful events**,
the things a chart review or an incident report actually turns on:

  * **VITAL**    — a snapshot, sampled on a risk-band change or every ``vital_every_s``
                   (so the graph keeps a sparse, faithful trajectory, not raw ticks).
  * **ALERT**    — emitted when the risk band *escalates* (LOW→MED→HIGH).
  * **DECISION** — a verified protocol was applied (treatment ordered). The Z3 safety
                   certificate is recorded as node *provenance*.
  * **OMISSION** — HIGH risk has persisted ``omission_after_s`` with no treatment: the
                   indicated action was not taken (the actionable node a report flags).
  * **OUTCOME**  — a terminal state, supplied by the caller (``mark_outcome``).

Edges chain consecutive events in time. They are all ``model_predicted`` — temporal-
succession inferences, honestly typed. We deliberately do **not** mint a ``z3_proven``
edge from a treatment: the protocol's certificate proves it was *within a safe
envelope*, not that it *caused* the downstream change. Borrowing it as causal proof
is exactly what the evidence_type firewall exists to prevent — so the cert lives on
the DECISION node as provenance, never on an edge as proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from backend.aegis.kernel.causal_graph import (
    ALERT, DECISION, MODEL_PREDICTED, OMISSION, OUTCOME, VITAL,
    Node, TemporalCausalGraph)

_BAND = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _risk_value(a: Any) -> float:
    """A [0,1] risk score for a node: prefer the calibrated probability."""
    p = getattr(a, "ml_probability", None)
    if p is not None:
        return float(p)
    det = getattr(a, "detection", None) or {}
    if det.get("sepsis_probability") is not None:
        return float(det["sepsis_probability"])
    return {0: 0.10, 1: 0.45, 2: 0.80}[_BAND.get(getattr(a, "risk_level", "LOW"), 0)]


def _mechanism(prev_kind: Optional[str], new_kind: str) -> str:
    if prev_kind == OMISSION and new_kind in (ALERT, VITAL, OUTCOME):
        return "untreated deterioration"
    if prev_kind == DECISION and new_kind == VITAL:
        return "treatment response"
    return {
        ALERT: "risk escalation",
        OMISSION: "indicated action not taken",
        DECISION: "intervention ordered",
        OUTCOME: "progression to outcome",
        VITAL: "temporal progression",
    }[new_kind]


@dataclass
class CausalGraphIngestor:
    """Stateful distiller: feed it assessments, it grows the patient's causal graph."""

    bed: str = "bed"
    vital_every_s: float = 60.0       # min spacing between sampled VITAL nodes
    omission_after_s: float = 180.0   # HIGH-risk dwell before an omission is logged
    store: Optional[Any] = None       # CausalGraphStore for durable persistence

    graph: TemporalCausalGraph = field(init=False)
    _last_node: Optional[Node] = field(init=False, default=None)
    _last_band: int = field(init=False, default=-1)
    _last_vital_t: float = field(init=False, default=float("-inf"))
    _high_since: Optional[float] = field(init=False, default=None)
    _omission_logged: bool = field(init=False, default=False)
    _treated: bool = field(init=False, default=False)
    ticks: int = field(init=False, default=0)
    rehydrated: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        loaded = None
        if self.store is not None:
            try:
                loaded = self.store.load(self.bed)   # prior episodes, durable
            except Exception as e:                   # pragma: no cover
                print(f"[causal-ingest] rehydrate skipped: {e}")
        if loaded is not None:
            # Resume an existing story: new events chain onto the persisted tail.
            self.graph = loaded
            self.rehydrated = len(loaded._nodes)
            self._last_node = max(loaded._nodes.values(), key=lambda n: n.seq)
        else:
            self.graph = TemporalCausalGraph(bed=self.bed)
        if self.store is not None:
            try:
                self.store.bind_to(self.graph)        # persist all future appends
            except Exception as e:                    # pragma: no cover
                print(f"[causal-ingest] persistence bind skipped: {e}")

    # ------------------------------------------------------------------ #
    def _add(self, kind: str, t: float, payload: Dict[str, Any],
             risk: float, confidence: float) -> Node:
        node = self.graph.add_event(kind, t, payload, risk=risk, confidence=confidence)
        if self._last_node is not None:
            self.graph.link(self._last_node.id, node.id,
                            _mechanism(self._last_node.kind, kind),
                            MODEL_PREDICTED,
                            confidence=round(min(self._last_node.confidence, confidence), 4))
        self._last_node = node
        return node

    def ingest(self, a: Any, vitals: Dict[str, float], t: float) -> Optional[Node]:
        """Fold one assessment into the graph. Returns the node created this tick
        (the most salient one), or None if the tick produced no clinical event."""
        self.ticks += 1
        band = _BAND.get(getattr(a, "risk_level", "LOW"), 0)
        risk = _risk_value(a)
        conf = float(getattr(a, "signal_quality", 1.0) or 1.0)
        salient: Optional[Node] = None

        # -- treatment: a verified protocol was applied this episode -------- #
        proto = getattr(a, "protocol", None)
        ver = getattr(a, "verification", None) or {}
        applied = bool(proto) and bool(ver.get("safe", ver.get("verified", False)))
        if applied and not self._treated:
            self._treated = True
            self._high_since = None          # treatment closes the omission window
            salient = self._add(DECISION, t, {
                "protocol": proto,
                "z3_certified": True,
                "z3_status": ver.get("z3_status"),   # provenance, NOT edge proof
            }, risk=risk, confidence=conf)

        # -- escalation: band stepped up ----------------------------------- #
        if band > self._last_band and self._last_band >= 0:
            salient = self._add(ALERT, t, {
                "from": self._last_band, "to": band,
                "risk_level": getattr(a, "risk_level", "?"),
                "triggered_rules": list(getattr(a, "triggered_rules", []) or [])[:6],
            }, risk=risk, confidence=conf)

        # -- omission: sustained HIGH with no treatment -------------------- #
        if band == _BAND["HIGH"] and not self._treated:
            if self._high_since is None:
                self._high_since = t
            elif (t - self._high_since) >= self.omission_after_s and not self._omission_logged:
                self._omission_logged = True
                salient = self._add(OMISSION, t, {
                    "indicated": "sepsis_bundle", "given": False,
                    "high_dwell_s": round(t - self._high_since, 1),
                }, risk=risk, confidence=conf)
        elif band < _BAND["HIGH"]:
            self._high_since = None

        # -- sampled vital snapshot ---------------------------------------- #
        band_changed = band != self._last_band
        due = (t - self._last_vital_t) >= self.vital_every_s
        if self._last_node is None or band_changed or due:
            v = self._add(VITAL, t, dict(vitals), risk=risk, confidence=conf)
            self._last_vital_t = t
            salient = salient or v

        self._last_band = band
        return salient

    # ------------------------------------------------------------------ #
    def mark_outcome(self, state: str, t: float, vitals: Optional[Dict[str, Any]] = None,
                     risk: float = 0.9) -> Node:
        """Record a terminal outcome node (e.g. 'organ_failure', 'recovery')."""
        payload = {"state": state}
        if vitals:
            payload.update(vitals)
        return self._add(OUTCOME, t, payload, risk=risk, confidence=1.0)

    # ------------------------------------------------------------------ #
    def summary(self, recent: int = 6) -> Dict[str, Any]:
        """Compact, snapshot-friendly view for the frontend contract."""
        st = self.graph.stats()
        d = self.graph.to_dict()
        return {
            **st,
            "ticks_ingested": self.ticks,
            "treated": self._treated,
            "omission_logged": self._omission_logged,
            "recent_events": d["nodes"][-recent:],
        }


# --------------------------------------------------------------------------- #
# Standalone demo.
# --------------------------------------------------------------------------- #
def _demo() -> None:
    from types import SimpleNamespace as A

    ing = CausalGraphIngestor(bed="bed_3", vital_every_s=60, omission_after_s=180)
    t0 = 1000.0
    # A deteriorating, untreated course over ~70 min, sampled each minute.
    for i in range(70):
        t = t0 + i * 60
        if i < 10:
            band, mp, mp_map = "LOW", 0.08, 85
        elif i < 25:
            band, mp, mp_map = "MEDIUM", 0.35, 70
        else:
            band, mp, mp_map = "HIGH", 0.78, 58
        a = A(risk_level=band, ml_probability=mp, detection={}, protocol=None,
              verification=None, signal_quality=0.95, triggered_rules=["qSOFA>=2"])
        ing.ingest(a, {"map": mp_map, "lactate": 1.5 + 0.06 * i}, t)
    out = ing.mark_outcome("organ_failure", t0 + 70 * 60, {"map": 52})

    print("graph:", ing.graph.stats())
    rep = ing.graph.incident_report(out.id)
    ea = rep["earliest_actionable"]
    print(f"earliest actionable: {ea['kind']}@{ea['t']}  lead_time={rep['lead_time_s']}s")


if __name__ == "__main__":
    _demo()
