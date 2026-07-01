"""Verify harness for the Temporal Causal Graph — every claim asserted, by measurement.

We build a realistic septic-trajectory graph and then prove the four properties the
layer exists to guarantee:

  1. INTEGRITY    — the hash chain verifies, and mutating any past node/edge breaks it.
  2. FIREWALL     — z3_proven requires a real certificate hash; model/association
                    edges cannot borrow a proof; associations require support.
  3. DAG          — a cause cannot post-date its effect, and no edge may form a cycle.
  4. REASONING    — backward incident traversal finds the earliest actionable node
                    and the correct lead time; the counterfactual frontier returns
                    exactly the editable action nodes upstream of an outcome.

    python -m backend.aegis.kernel.verify_causal_graph
"""
from __future__ import annotations

from dataclasses import replace

from backend.aegis.kernel.causal_graph import (
    ALERT, DRUG, MODEL_PREDICTED, OBSERVED_ASSOCIATION, OMISSION, OUTCOME,
    VITAL, Z3_PROVEN, TemporalCausalGraph, cert_hash)
from backend.aegis.kernel.neurosym import NeuroSymbolicRuntime, PROTOCOL


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def _expect_raises(label: str, fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    print(f"    (expected {label} to be rejected, but it was allowed)")
    return False


def main() -> int:
    failures = 0
    checks = []

    # ---- build a septic trajectory --------------------------------------- #
    g = TemporalCausalGraph(bed="bed_3")
    v0 = g.add_event(VITAL, 0.0, {"map": 78, "lactate": 1.6}, risk=0.10)
    v1 = g.add_event(VITAL, 3600.0, {"map": 62, "lactate": 4.2}, risk=0.55)
    om = g.add_event(OMISSION, 3700.0, {"indicated": "antibiotics", "given": False}, risk=0.6)
    a = g.add_event(ALERT, 3900.0, {"severity": 2}, risk=0.62)
    out = g.add_event(OUTCOME, 7200.0, {"state": "organ_failure"}, risk=0.9)

    g.link(v0.id, v1.id, "perfusion decline", MODEL_PREDICTED, confidence=0.7)
    g.link(v1.id, om.id, "low MAP+lactate indicated abx", OBSERVED_ASSOCIATION,
           confidence=0.8, support=312)
    g.link(om.id, a.id, "untreated deterioration", MODEL_PREDICTED, confidence=0.65)
    g.link(a.id, out.id, "progression to organ failure", OBSERVED_ASSOCIATION,
           confidence=0.6, support=188)

    # ---- 1. INTEGRITY ---------------------------------------------------- #
    _hr("1. INTEGRITY (tamper-evident hash chain)")
    checks.append(("clean chain verifies", g.verify_chain()))

    # Tamper: rewrite a past node's payload in place; the chain must reject it.
    tampered = TemporalCausalGraph(bed="bed_3")
    n0 = tampered.add_event(VITAL, 0.0, {"map": 78}, risk=0.10)
    n1 = tampered.add_event(VITAL, 60.0, {"map": 62}, risk=0.55)
    tampered.link(n0.id, n1.id, "decline", MODEL_PREDICTED, confidence=0.7)
    # Forge: swap the stored node for one with an altered risk but same hash fields.
    tampered._nodes[n0.id] = replace(tampered._nodes[n0.id], risk=0.99)
    checks.append(("tampering with a past node breaks the chain", not tampered.verify_chain()))

    # ---- 2. FIREWALL ----------------------------------------------------- #
    _hr("2. EVIDENCE-TYPE FIREWALL")
    f = TemporalCausalGraph(bed="bed_x")
    x0 = f.add_event(VITAL, 0.0, {"map": 60}, risk=0.6)
    x1 = f.add_event(DRUG, 100.0, {"drug": "vasopressor"}, risk=0.5)
    x2 = f.add_event(VITAL, 200.0, {"map": 72}, risk=0.3)

    checks.append(("z3_proven without a proof is rejected",
                   _expect_raises("unproven z3 edge",
                                  lambda: f.link(x0.id, x1.id, "m", Z3_PROVEN))))
    checks.append(("model_predicted citing a proof is rejected",
                   _expect_raises("model edge with proof",
                                  lambda: f.link(x0.id, x1.id, "m", MODEL_PREDICTED,
                                                 proven_by="z3:deadbeef"))))
    checks.append(("observed_association without support is rejected",
                   _expect_raises("association w/o support",
                                  lambda: f.link(x0.id, x1.id, "m", OBSERVED_ASSOCIATION))))

    # A real Layer-4 certificate makes a z3_proven edge legal, and it is recorded.
    cage = NeuroSymbolicRuntime()
    ok, cert = cage.dispatch(PROTOCOL,
                             {"fluid_rate": 30.0, "vasopressor_dose": 0.1, "antibiotic_code": 1},
                             {"risk_level": "HIGH", "vitals": {"map": 60.0}, "clinical_history": {}})
    e = f.link(x1.id, x2.id, "vasopressor → MAP rise", Z3_PROVEN,
               confidence=0.95, proven_by=cert)
    checks.append(("z3_proven edge with a real cert is accepted + cites its hash",
                   e.evidence_type == Z3_PROVEN and e.proven_by == cert_hash(cert)
                   and e.proven_by.startswith("z3:")))

    # ---- 3. DAG ---------------------------------------------------------- #
    _hr("3. DAG INVARIANTS (time-order + acyclicity)")
    d = TemporalCausalGraph(bed="bed_y")
    y0 = d.add_event(VITAL, 0.0, {}, risk=0.1)
    y1 = d.add_event(VITAL, 100.0, {}, risk=0.2)
    y2 = d.add_event(VITAL, 200.0, {}, risk=0.3)
    d.link(y0.id, y1.id, "a", MODEL_PREDICTED)
    d.link(y1.id, y2.id, "b", MODEL_PREDICTED)
    checks.append(("a cause cannot post-date its effect",
                   _expect_raises("back-in-time edge",
                                  lambda: d.link(y2.id, y0.id, "c", MODEL_PREDICTED))))
    # y2 -> y0 is blocked by time order; force a same-time cycle to test acyclicity.
    z0 = d.add_event(VITAL, 300.0, {}, node_id="z0", risk=0.1)
    z1 = d.add_event(VITAL, 300.0, {}, node_id="z1", risk=0.1)
    d.link(z0.id, z1.id, "p", MODEL_PREDICTED)
    checks.append(("an edge that would form a cycle is rejected",
                   _expect_raises("cycle edge",
                                  lambda: d.link(z1.id, z0.id, "q", MODEL_PREDICTED))))

    # ---- 4. REASONING ---------------------------------------------------- #
    _hr("4. REASONING (incident report + counterfactual frontier)")
    rep = g.incident_report(out.id)
    earliest = rep["earliest_actionable"]
    print(f"    earliest actionable: {earliest['kind']} @ t={earliest['t']}  "
          f"lead_time={rep['lead_time_s']}s")
    print(f"    evidence mix on path: {rep['evidence_mix']}")
    checks.append(("incident report flags the omission as earliest actionable",
                   earliest is not None and earliest["id"] == om.id))
    checks.append(("lead time = outcome.t - omission.t",
                   abs(rep["lead_time_s"] - (out.t - om.t)) < 1e-6))
    checks.append(("report integrity_ok", rep["integrity_ok"] is True))

    frontier = g.counterfactual_frontier(out.id)
    fkinds = sorted({n.kind for n in frontier})
    print(f"    counterfactual frontier: {[n.id for n in frontier]} kinds={fkinds}")
    checks.append(("frontier is exactly the upstream action nodes (the omission)",
                   [n.id for n in frontier] == [om.id]))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — the clinical story is an integrity-checked, "
          "evidence-typed kernel object." if failures == 0
          else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
