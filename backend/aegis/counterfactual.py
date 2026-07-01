"""Counterfactual Replay Engine — "what would have happened if…?" over the causal graph.

This binds two pieces that already exist:

  * the **Temporal Causal Graph** (``kernel/causal_graph.py``) — the recorded,
    tamper-evident clinical story, which can tell us the *branch points*
    (``counterfactual_frontier``) and the *actual* downstream trajectory; and
  * the **forward physiology model** (``physiology.forward_project``) — the
    transparent, monotonic pharmacodynamic projector behind the what-if twin.

Given a branch-point node ("at T-6h we omitted antibiotics"), the engine:

  1. reconstructs the vitals state recorded at that point,
  2. replays the forward model from there under an *alternative* action,
  3. lines the projection up against what **actually** happened downstream, and
  4. reports the delta — did the alternative diverge the patient toward recovery?

The honesty boundary is the whole point and is carried in every result:

  * the **actual** path is *recorded data* (real, from the graph);
  * the **counterfactual** path is a *model projection* — illustrative direction,
    not a validated outcome forecast (same caveat the forward model already carries).

So this answers "which way would the alternative have pushed the patient, relative
to what we observed", and never "this patient would have lived." Overclaiming a
counterfactual is exactly the trap the evidence_type firewall guards against on the
graph side; here we guard it with explicit labeling + a confidence that *decays*
with how far we extrapolate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from backend.aegis.kernel.causal_graph import (
    DECISION, DRUG, OMISSION, OUTCOME, VITAL, TemporalCausalGraph)
from backend.aegis.physiology import forward_project

# The "do nothing" action — the factual baseline when an action was omitted.
_NO_ACTION = {"fluid_rate": 0.0, "vasopressor_dose": 0.0, "antibiotic_code": 0}


def _proxy_risk(vitals: Dict[str, float]) -> float:
    """Transparent, monotonic risk proxy used when no model risk_fn is injected.

    Rises as MAP falls below 65 mmHg and as lactate climbs above 2 mmol/L — the
    two anchors of the sepsis perfusion picture. Deterministic and dependency-free
    so the engine (and its verify harness) runs without the HDC/neuro stack; the
    live system injects ``WhatIfScheduler._risk_of`` for the real scorer.
    """
    mp = float(vitals.get("map", 90.0))
    lact = float(vitals.get("sweat_lactate", vitals.get("lactate", 1.0)))
    map_term = max(0.0, (65.0 - mp) / 45.0)          # 0 at MAP 65, 1 at MAP 20
    lact_term = max(0.0, (lact - 2.0) / 6.0)          # 0 at 2, 1 at 8 mmol/L
    return float(max(0.0, min(1.0, 0.6 * map_term + 0.4 * lact_term)))


@dataclass
class CounterfactualReplay:
    """Replays alternative actions from a branch point in a clinical causal graph."""

    graph: TemporalCausalGraph
    risk_fn: Callable[[Dict[str, float]], float] = _proxy_risk
    horizon: int = 16

    # ------------------------------------------------------------------ #
    def _vitals_at(self, t: float) -> Optional[Dict[str, Any]]:
        """The most recent recorded VITAL snapshot at or before time ``t``."""
        vitals = [n for n in self.graph._nodes.values()
                  if n.kind == VITAL and n.t <= t]
        if not vitals:
            return None
        return max(vitals, key=lambda n: n.t)

    def _actual_downstream(self, t: float) -> List[Any]:
        """Recorded VITAL/OUTCOME nodes strictly after ``t`` (what really happened)."""
        return sorted(
            [n for n in self.graph._nodes.values()
             if n.t > t and n.kind in (VITAL, OUTCOME)],
            key=lambda n: n.t)

    def _factual_action(self, branch) -> Dict[str, float]:
        """The treatment actually taken at the branch — the model-internal baseline.

        An OMISSION node means the indicated action was *not* taken → no action. A
        DRUG/DECISION node carries the treatment that *was* given in its payload;
        we read whatever pharmacodynamic fields it recorded and default the rest.
        """
        if branch.kind == OMISSION:
            return dict(_NO_ACTION)
        if branch.kind in (DRUG, DECISION):
            p = branch.payload
            return {k: float(p.get(k, _NO_ACTION[k])) if k != "antibiotic_code"
                    else int(p.get(k, 0)) for k in _NO_ACTION}
        return dict(_NO_ACTION)

    def _confidence(self, steps: int) -> float:
        """Projection confidence decays with extrapolation depth (honest, not a CI).

        Starts near 1.0 and halves roughly every ``horizon`` steps — a transparent
        stand-in for "trust this less the further out it runs", not a calibrated
        interval. We never present a counterfactual as certain.
        """
        return round(0.5 ** (steps / max(1, self.horizon)), 4)

    # ------------------------------------------------------------------ #
    def _project(self, base_vitals, treatment, sev, steps):
        traj = forward_project(base_vitals, treatment, sev, steps)
        path = [{"step": i + 1, "map": v["map"],
                 "sweat_lactate": v["sweat_lactate"], "spo2": v["spo2"],
                 "projected_risk": round(self.risk_fn(v), 4)}
                for i, v in enumerate(traj)]
        final = traj[-1]
        return path, float(final["map"]), float(self.risk_fn(final))

    def replay(self, branch_node_id: str, alternative_treatment: Dict[str, float],
               severity: Optional[float] = None,
               horizon: Optional[int] = None) -> Dict[str, Any]:
        """Replay ``alternative_treatment`` from the branch point.

        Treatment effect is measured **model-internally**: we project both the
        alternative and the action actually taken at the branch under the *same*
        forward model, so the delta isolates the intervention from the model's own
        bias. The recorded downstream path is reported alongside as context, with an
        explicit *model-fidelity* note where the model and reality disagree.
        """
        if branch_node_id not in self.graph._nodes:
            raise KeyError(f"no such node {branch_node_id!r}")
        branch = self.graph._nodes[branch_node_id]
        steps = int(horizon or self.horizon)

        base_node = self._vitals_at(branch.t)
        if base_node is None:
            raise ValueError(
                f"no recorded vitals at or before branch t={branch.t} to replay from")
        base_vitals = dict(base_node.payload)
        # Carry lactate under the forward model's key if recorded as 'lactate'.
        if "lactate" in base_vitals and "sweat_lactate" not in base_vitals:
            base_vitals["sweat_lactate"] = base_vitals["lactate"]

        # Severity drive: explicit, else the risk recorded on the branch vitals.
        sev = float(severity) if severity is not None else float(base_node.risk)

        # --- counterfactual vs factual baseline, same model -------------- #
        factual_action = self._factual_action(branch)
        cf_path, cf_map, cf_risk = self._project(base_vitals, alternative_treatment, sev, steps)
        fb_path, fb_map, fb_risk = self._project(base_vitals, factual_action, sev, steps)

        d_risk = round(cf_risk - fb_risk, 4)          # negative = treatment lowers risk
        d_map = round(cf_map - fb_map, 2)             # positive = better perfusion
        diverges_to_recovery = cf_risk < fb_risk and d_risk <= -0.10

        # --- recorded reality (context) + model-fidelity disclosure ------ #
        actual_nodes = self._actual_downstream(branch.t)
        outcome = next((n for n in reversed(actual_nodes) if n.kind == OUTCOME), None)
        actual_terminal = actual_nodes[-1] if actual_nodes else base_node
        actual_final_risk = float(actual_terminal.risk)
        actual_final_map = float(actual_terminal.payload.get("map", base_vitals.get("map", float("nan"))))

        return {
            "branch": branch.to_dict(),
            "branch_vitals": {k: base_vitals.get(k) for k in
                              ("map", "sweat_lactate", "spo2", "heart_rate")},
            "alternative_treatment": dict(alternative_treatment),
            "factual_action": factual_action,
            "severity": round(sev, 3),
            "horizon_steps": steps,
            "factual_baseline": {                      # same model, action actually taken
                "path": fb_path,
                "final_map": round(fb_map, 2),
                "final_risk": round(fb_risk, 4),
            },
            "counterfactual": {                        # same model, the alternative
                "path": cf_path,
                "final_map": round(cf_map, 2),
                "final_risk": round(cf_risk, 4),
                "confidence": self._confidence(steps),
            },
            "delta": {                                 # treatment effect, model-internal
                "final_risk": d_risk,
                "final_map": d_map,
                "diverges_toward_recovery": bool(diverges_to_recovery),
            },
            "actual": {                                # recorded reality, for context only
                "path": [n.to_dict() for n in actual_nodes],
                "outcome": outcome.to_dict() if outcome else None,
                "final_map": round(actual_final_map, 2),
                "final_risk": round(actual_final_risk, 4),
            },
            "model_fidelity": {
                # How the factual-baseline projection compared to what really happened.
                # A large gap means the illustrative model under/over-states deterioration
                # — disclosed, not hidden, so the counterfactual isn't over-read.
                "baseline_vs_actual_risk": round(fb_risk - actual_final_risk, 4),
                "note": ("Forward model under-predicts the recorded deterioration"
                         if fb_risk + 0.10 < actual_final_risk else
                         "Forward model roughly tracks the recorded trajectory"),
            },
            "validity": ("Treatment effect is the model-internal delta (alternative vs "
                         "action-actually-taken, same model); recorded reality is shown "
                         "for context. Illustrative direction, not a validated outcome "
                         "forecast. Confidence decays with horizon."),
            "integrity_ok": self.graph.verify_chain(),
        }

    # ------------------------------------------------------------------ #
    def replay_earliest(self, outcome_id: str, alternative_treatment: Dict[str, float],
                        **kw) -> Dict[str, Any]:
        """Headline counterfactual: replay the alternative at the *earliest actionable*
        node upstream of a bad outcome ("what if we'd acted at the first opportunity?").

        Composes the graph's incident report (which already finds that node) with a
        replay, so the answer is anchored to the same branch point a clinician would
        be shown in the audit trail.
        """
        report = self.graph.incident_report(outcome_id)
        earliest = report["earliest_actionable"]
        if earliest is None:
            return {"branch": None,
                    "note": "No actionable node upstream of this outcome to replay.",
                    "incident_report": report}
        out = self.replay(earliest["id"], alternative_treatment, **kw)
        out["incident_report"] = report
        return out


# --------------------------------------------------------------------------- #
# Standalone demo.
# --------------------------------------------------------------------------- #
def _demo() -> None:
    from backend.aegis.kernel.causal_graph import (
        ALERT, MODEL_PREDICTED, OBSERVED_ASSOCIATION, OMISSION)

    g = TemporalCausalGraph(bed="bed_3")
    v0 = g.add_event(VITAL, 0.0, {"map": 78, "lactate": 1.6}, risk=0.10)
    v1 = g.add_event(VITAL, 3600.0, {"map": 62, "lactate": 4.2}, risk=0.55)
    om = g.add_event(OMISSION, 3700.0, {"indicated": "antibiotics", "given": False}, risk=0.6)
    a = g.add_event(ALERT, 3900.0, {"severity": 2}, risk=0.62)
    v2 = g.add_event(VITAL, 5400.0, {"map": 55, "lactate": 5.8}, risk=0.8)
    out = g.add_event(OUTCOME, 7200.0, {"state": "organ_failure", "map": 50}, risk=0.9)
    g.link(v0.id, v1.id, "perfusion decline", MODEL_PREDICTED, confidence=0.7)
    g.link(v1.id, om.id, "indicated abx", OBSERVED_ASSOCIATION, confidence=0.8, support=312)
    g.link(om.id, a.id, "untreated deterioration", MODEL_PREDICTED, confidence=0.65)
    g.link(a.id, v2.id, "worsening", MODEL_PREDICTED, confidence=0.6)
    g.link(v2.id, out.id, "organ failure", OBSERVED_ASSOCIATION, confidence=0.6, support=188)

    engine = CounterfactualReplay(g, horizon=16)
    abx_and_fluids = {"fluid_rate": 5.0, "vasopressor_dose": 0.0, "antibiotic_code": 3}
    res = engine.replay_earliest(out.id, abx_and_fluids)
    print(f"branch: {res['branch']['kind']} @ t={res['branch']['t']}  "
          f"(lead {res['incident_report']['lead_time_s']}s before outcome)")
    print(f"actual final:        MAP={res['actual']['final_map']}  risk={res['actual']['final_risk']}")
    print(f"counterfactual final: MAP={res['counterfactual']['final_map']}  "
          f"risk={res['counterfactual']['final_risk']}  conf={res['counterfactual']['confidence']}")
    print(f"delta: {res['delta']}")


if __name__ == "__main__":
    _demo()
