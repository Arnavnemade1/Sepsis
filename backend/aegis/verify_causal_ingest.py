"""Verify harness for the causal-graph ingestor — claims asserted by measurement.

Part 1 (DISTILLATION) drives a synthetic deteriorating-then-untreated course
through the ingestor and proves it records the *right* events:

  * exactly one ALERT per band escalation (LOW→MED→HIGH), not one per tick;
  * an OMISSION once HIGH risk persists past the dwell threshold, with no treatment;
  * no OMISSION when a verified treatment is applied in time;
  * every edge honestly typed model_predicted (no treatment masquerades as proof);
  * the grown graph is integrity-intact and feeds incident_report + replay.

Part 2 (LIVE) boots a real AegisSession, ticks it, and proves the causal graph
grows on the live path and serializes inside the frontend snapshot contract.

    python -m backend.aegis.verify_causal_ingest
"""
from __future__ import annotations

import json
from types import SimpleNamespace as A

from backend.aegis.causal_ingest import CausalGraphIngestor
from backend.aegis.counterfactual import CounterfactualReplay
from backend.aegis.kernel.causal_graph import (
    ALERT, DECISION, MODEL_PREDICTED, OMISSION)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def _assess(band, mp, sq=0.95, protocol=None, verification=None):
    return A(risk_level=band, ml_probability=mp, detection={}, protocol=protocol,
             verification=verification, signal_quality=sq, triggered_rules=["qSOFA>=2"])


def main() -> int:
    failures = 0
    checks = []

    # ===== Part 1: DISTILLATION ========================================== #
    _hr("1. DISTILLATION (untreated deteriorating course)")
    ing = CausalGraphIngestor(bed="bed_3", vital_every_s=60, omission_after_s=180)
    t0 = 1000.0
    for i in range(70):
        t = t0 + i * 60
        band, mp, mp_map = (("LOW", 0.08, 85) if i < 10 else
                            ("MEDIUM", 0.35, 70) if i < 25 else ("HIGH", 0.78, 58))
        ing.ingest(_assess(band, mp), {"map": mp_map, "lactate": 1.5 + 0.06 * i}, t)
    out = ing.mark_outcome("organ_failure", t0 + 70 * 60, {"map": 52})
    g = ing.graph
    nodes = g.to_dict()["nodes"]
    kinds = [n["kind"] for n in nodes]
    # escalation alerts are now a TYPED subset of ALERT (the ingestor also emits
    # sign/organ/mods/trajectory ALERTs for a far richer story) — count by type.
    n_alert = sum(1 for n in nodes if n["kind"] == ALERT
                  and (n.get("payload") or {}).get("type") == "escalation")
    n_om = kinds.count(OMISSION)
    print(f"    {len(nodes)} nodes from 71 ticks  escalations={n_alert} omissions={n_om} "
          f"(total alerts={kinds.count(ALERT)})")

    checks.append(("exactly 2 escalation alerts (LOW→MED, MED→HIGH)", n_alert == 2))
    checks.append(("exactly 1 omission logged", n_om == 1))
    checks.append(("all edges are honestly typed model_predicted",
                   g.stats()["evidence_mix"][MODEL_PREDICTED] == g.stats()["edges"]
                   and g.stats()["evidence_mix"]["z3_proven"] == 0))
    checks.append(("grown graph is integrity-intact", g.verify_chain()))

    rep = g.incident_report(out.id)
    ea = rep["earliest_actionable"]
    print(f"    incident: earliest={ea['kind']}@{ea['t']} lead={rep['lead_time_s']}s")
    checks.append(("incident report flags the omission as earliest actionable",
                   ea is not None and ea["kind"] == OMISSION))

    eng = CounterfactualReplay(g, horizon=16)
    res = eng.replay_earliest(out.id, {"fluid_rate": 5.0, "vasopressor_dose": 0.0, "antibiotic_code": 3})
    print(f"    replay Δrisk={res['delta']['final_risk']} "
          f"recovery={res['delta']['diverges_toward_recovery']}")
    checks.append(("counterfactual replay runs on the ingested graph + diverges to recovery",
                   res["delta"]["diverges_toward_recovery"] is True))

    # -- treated-in-time variant: no omission, a DECISION instead --------- #
    _hr("2. TREATED IN TIME (no omission; a verified DECISION is recorded)")
    ing2 = CausalGraphIngestor(bed="bed_7", vital_every_s=60, omission_after_s=180)
    for i in range(20):
        t = t0 + i * 60
        band, mp, mp_map = (("LOW", 0.08, 85) if i < 5 else
                            ("MEDIUM", 0.35, 70) if i < 10 else ("HIGH", 0.78, 60))
        # Treatment applied right at HIGH onset (i==10), verified safe.
        proto = ({"fluid_rate": 30.0, "antibiotic_code": 3} if i == 10 else None)
        ver = ({"safe": True, "z3_status": "unsat"} if i == 10 else None)
        ing2.ingest(_assess(band, mp, protocol=proto, verification=ver),
                    {"map": mp_map, "lactate": 2.0}, t)
    k2 = [n["kind"] for n in ing2.graph.to_dict()["nodes"]]
    print(f"    kinds={sorted(set(k2))}  treated={ing2._treated}  omission_logged={ing2._omission_logged}")
    checks.append(("a verified DECISION node was recorded", DECISION in k2))
    checks.append(("no omission when treated in time", OMISSION not in k2 and not ing2._omission_logged))
    dec = next(n for n in ing2.graph.to_dict()["nodes"] if n["kind"] == DECISION)
    checks.append(("DECISION carries Z3 provenance, not an edge proof",
                   dec["payload"].get("z3_certified") is True))

    # ===== Part 3: LIVE SESSION ========================================== #
    _hr("3. LIVE (graph grows on a real AegisSession + serializes in the snapshot)")
    try:
        from backend.aegis.stream import AegisSession
        sess = AegisSession(patient_id="bed_live", use_llm=False, persist_causal=False)
        snap = None
        for i in range(8):
            tele = {"heart_rate": 95 + i * 4, "respiratory_rate": 20 + i,
                    "skin_temp": 37.5 + i * 0.1, "spo2": 97 - i,
                    "systolic_bp": 120 - i * 5, "diastolic_bp": 78 - i * 2}
            snap = sess.tick(tele, source="wearable", timestamp=t0 + i * 30)
        cg = snap["causal_graph"]
        print(f"    schema={snap['schema_version']}  nodes={cg['nodes']} "
              f"edges={cg['edges']} ticks={cg['ticks_ingested']}")
        checks.append(("snapshot schema is current (>=1.3)", snap["schema_version"] >= "1.3"))
        checks.append(("causal graph grew on the live tick path", cg["nodes"] >= 1))
        checks.append(("live graph integrity-intact", cg["integrity_ok"] is True))
        json.dumps(snap)   # the whole contract must remain JSON-serializable
        checks.append(("snapshot (incl. causal_graph) is JSON-serializable", True))
    except Exception as e:
        print(f"    LIVE session failed: {type(e).__name__}: {e}")
        checks.append(("LIVE session ran", False))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — the live stream now writes a faithful, "
          "integrity-checked clinical story the OS can reason over."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
