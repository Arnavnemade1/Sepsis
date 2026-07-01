"""Verify harness for the Autonomous ICU Orchestrator — claims asserted by measurement.

The capstone integration claim is an *architectural* invariant, not a clinical one:
no action reaches a patient without a safety certificate, and that property holds
continuously and immutably. We prove exactly that:

  1. HIERARCHY  — the orchestrator reports all four layers (sensor/bed/wing/global)
                  composed from real subsystems, under one named invariant.
  2. GOVERNANCE — a guideline-safe order is certified; a CHF-violating fluid bolus,
                  an alarm downgrade on a critical patient, a de-prioritization of a
                  critical bed, and an unknown action are ALL hard-blocked.
  3. NO UNCERTIFIED EGRESS — every governed action (allowed or blocked) is on the
                  attestation ledger; the ledger is hash-chain intact; tampering with
                  any entry is detected.
  4. CONTINUOUS ATTESTATION — the invariant holds across the wing; a forced breach
                  (a critical bed dropped below its priority floor) is caught.
  5. DEMO CASCADE — inducing a sepsis cascade walks sensor→bed→wing→global→governance
                  and ends in a signed, Z3-certified, immutably-logged order.
  6. PORTABLE PROOF — that order also emits an Ed25519-signed proof certificate bound
                  to its ledger entry, independently re-checkable by any third party;
                  a tampered copy is rejected on re-check.

    python -m backend.aegis.verify_orchestrator
"""
from __future__ import annotations

from backend.aegis.kernel.scheduler import BioCluster
from backend.aegis.orchestrator import (Orchestrator, SENSOR, BED, WING, GLOBAL)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def _crisis_bed(orch, bed="node_2", warmup=20, ticks=60):
    orch.cluster.run(warmup)
    orch.cluster.inject_crisis(bed, 1.0)
    for _ in range(ticks):
        orch.cluster.tick()


def main() -> int:
    failures = 0
    checks = []

    cl = BioCluster(n_nodes=4, lane_budget=64, seed=1)
    orch = Orchestrator(cl, histories={"node_2": {"chf": True}})
    _crisis_bed(orch, "node_2")
    st = orch._state("node_2")
    print(f"  node_2: tier={st['tier']} risk={st['risk_level']} "
          f"fused={st['fused_risk']:.3f} MAP={st['vitals']['map']:.0f}")

    # ---- 1. HIERARCHY ---------------------------------------------------- #
    _hr("1. HIERARCHY (four composed layers under one invariant)")
    s = orch.status()
    h = s["hierarchy"]
    checks.append(("invariant is the named non-maleficence invariant",
                   "no action reaches a patient without a safety certificate" in s["invariant"]))
    checks.append(("all four layers reported",
                   all(k in h for k in (SENSOR, BED, WING, GLOBAL))))
    checks.append(("wing layer exposes a real scheduler + flags the critical bed",
                   h[WING]["scheduler"] is not None and "node_2" in h[WING]["critical"]))
    checks.append(("bed layer exposes per-bed causal graphs",
                   isinstance(h[BED]["causal_graphs"], dict) and "node_2" in h[BED]["causal_graphs"]))

    # ---- 2. GOVERNANCE --------------------------------------------------- #
    _hr("2. GOVERNANCE (certify the safe; block the harmful)")
    safe = orch.govern("bed", "treat",
                       {"fluid_rate": 5.0, "vasopressor_dose": 0.0, "antibiotic_code": 3},
                       target="node_2")
    chf = orch.govern("bed", "treat",
                      {"fluid_rate": 30.0, "vasopressor_dose": 0.0, "antibiotic_code": 3},
                      target="node_2")
    downgrade = orch.govern("wing", "escalate", {"severity": 0}, target="node_2")
    deprioritize = orch.govern("wing", "pin", {"pin": 0.5}, target="node_2")
    unknown = orch.govern("bed", "frobnicate", {}, target="node_2")
    for label, c, expect in [
        ("guideline-safe order is CERTIFIED", safe, True),
        ("CHF-violating fluid bolus is BLOCKED", chf, False),
        ("alarm downgrade on a critical patient is BLOCKED", downgrade, False),
        ("de-prioritizing a critical bed is BLOCKED", deprioritize, False),
        ("unknown action is default-denied", unknown, False),
    ]:
        ok = (c.ok == expect)
        print(f"    {label}: ok={c.ok}" + (f" {c.violations}" if not c.ok else ""))
        checks.append((label, ok))
    checks.append(("blocks carry explicit violations",
                   bool(chf.violations) and bool(downgrade.violations)
                   and bool(deprioritize.violations) and bool(unknown.violations)))
    checks.append(("a blocked order does NOT reach the bed's causal graph",
                   "node_2" in orch.graphs and
                   orch.graphs["node_2"].graph.stats()["nodes"] == 1))  # only the safe one

    # ---- 3. NO UNCERTIFIED EGRESS --------------------------------------- #
    _hr("3. NO UNCERTIFIED EGRESS (every action on the tamper-evident ledger)")
    n_actions = len(orch.ledger.entries)
    print(f"    attestation ledger holds {n_actions} governed actions; "
          f"head {orch.ledger.head()[:12]}")
    checks.append(("every governed action recorded (5 issued)", n_actions == 5))
    checks.append(("ledger hash-chain verifies", orch.ledger.verify() is True))

    # ---- 4. CONTINUOUS ATTESTATION ------------------------------------- #
    _hr("4. CONTINUOUS ATTESTATION (invariant holds across the wing)")
    att = orch.attest()
    print(f"    attest ok={att['ok']} breaches={att['breaches']}")
    checks.append(("invariant currently holds across the wing", att["ok"] is True))
    # Force a breach: drop the critical bed below its priority floor out-of-band.
    cl.sched.nodes["node_2"].pin = 0.5
    att2 = orch.attest()
    print(f"    after forced floor breach: attest ok={att2['ok']} breaches={att2['breaches']}")
    checks.append(("a critical bed dropped below floor is caught by attestation",
                   att2["ok"] is False and any(b["bed"] == "node_2" for b in att2["breaches"])))
    cl.sched.nodes["node_2"].pin = 1.0  # restore

    # tamper detection on the ledger
    orch.ledger.entries[0].ok = (not orch.ledger.entries[0].ok)
    checks.append(("tampering with a ledger entry is detected", orch.ledger.verify() is False))

    # ---- 5. DEMO CASCADE ------------------------------------------------ #
    _hr("5. DEMO CASCADE (sensor → bed → wing → global → governed order)")
    cl2 = BioCluster(n_nodes=4, lane_budget=64, seed=3)
    orch2 = Orchestrator(cl2, histories={"node_2": {"chf": False}})
    res = orch2.run_scenario("node_2")
    layers_hit = [t["layer"] for t in res["timeline"]]
    for t in res["timeline"]:
        print(f"    [{t['layer']:10}] {t['event']}")
    print(f"    committed plan: {res['committed_plan']}  cert={res['certificate']['hash']}")
    checks.append(("cascade traverses all layers",
                   set([SENSOR, BED, WING, GLOBAL, "governance"]).issubset(set(layers_hit))))
    checks.append(("global brain recognized at least one signature",
                   any(t["layer"] == GLOBAL and t.get("signatures") for t in res["timeline"])))
    checks.append(("synthesized plan is Z3-certified", res["certificate"]["ok"] is True
                   and res["certificate"]["z3_status"] == "unsat"))
    checks.append(("committed order recorded to the causal graph", res["causal_events"] >= 1))
    checks.append(("scenario ends with the invariant holding", res["attestation"]["ok"] is True))

    # ---- 6. PORTABLE PROOF (the (decision, signed_cert) pair) ----------- #
    _hr("6. PORTABLE PROOF (third-party-checkable certificate for the order)")
    pc = res.get("proof_certificate")
    audit = res.get("proof_audit")
    print(f"    proof emitted: {pc is not None}; audit trusted: "
          f"{(audit or {}).get('trusted')}")
    if pc:
        print(f"    drug certified: {pc['subject']['protocol'].get('antibiotic_name')} "
              f"· obligations {pc['verdict']}")
        print(f"    signed by Ed25519 key {pc['signature'].get('key_id')} "
              f"· bound to ledger seq {pc['subject'].get('ledger_seq')}")
    checks.append(("a governed order emits a portable proof certificate", pc is not None))
    checks.append(("the proof is independently audited as trusted",
                   bool(audit) and audit["trusted"]))
    checks.append(("the proof's math is re-checked (not trusted) and passes",
                   bool(audit) and audit["math_ok"]))
    checks.append(("the proof is Ed25519-signed and bound to its ledger entry",
                   bool(pc) and pc["signature"].get("alg") == "Ed25519"
                   and pc["subject"].get("ledger_seq") == res["certificate"]["seq"]))
    # The cage and the portable proof must AGREE on a safe order.
    checks.append(("cage verdict and portable proof agree the order is safe",
                   bool(pc) and res["certificate"]["ok"]
                   and pc["verdict"]["all_satisfied"]))
    # And a tampered copy of the exported proof is rejected on independent re-check.
    if pc:
        from backend.aegis.proofexport import verify_certificate
        import copy as _copy
        bad = _copy.deepcopy(pc)
        bad["obligations"][0]["smt2"] = bad["obligations"][0]["smt2"].replace(
            "unsat", "sat")
        checks.append(("a tampered exported proof fails independent re-check",
                       not verify_certificate(bad).trusted))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — the orchestrator governs every layer under one "
          "invariant: no uncertified action can reach a patient."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
