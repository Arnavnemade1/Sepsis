"""Verify harness for the Distributed Consensus Clinical Brain — asserted by measurement.

The patient is in septic shock (MAP 58, HIGH risk) with CHF + penicillin allergy —
so the safe plan is tightly constrained. We prove the five guarantees that make this
a real BFT system rather than a voting gimmick:

  1. SAFETY (Byzantine) — with a malfunctioning replica that proposes a dangerous
     plan and rubber-stamps everything, the dangerous plan can NEVER commit; an
     honest quorum still commits a cage-safe plan.
  2. LIVENESS (physiologic) — when deterioration outruns the debate (deadline admits
     too few views to rotate past the faulty proposer), the kernel commits the
     PRE-CERTIFIED emergency order instead of deadlocking — fully multi-signed, Z3-safe.
  3. QUORUM MATH — N=3f+1 ⇒ quorum 2f+1, across panel sizes.
  4. SIGNATURES — the quorum certificate is independently verifiable; tampering the
     committed plan or forging signatures past the quorum margin is detected.
  5. FORMAL BACKSTOP — whatever commits (quorum or fallback) passes the Z3 gate;
     and a remote consultant sees only a need-to-know projection (no PHI).

    python -m backend.aegis.verify_consensus
"""
from __future__ import annotations

from dataclasses import replace

from backend.aegis.consensus import (
    ClinicalBFT, EMERGENCY_PLAN, SpecialistAgent, default_panel, plan_hash)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


SHOCK = {"map": 58.0, "risk_level": "HIGH",
         "clinical_history": {"chf": True, "renal_failure": False, "penicillin_allergy": True}}
DANGEROUS = {"fluid_rate": 60.0, "vasopressor_dose": 0.9, "antibiotic_code": 3}


def _byz_panel():
    panel = default_panel()
    panel[0].byzantine = True          # the intensivist replica is faulty
    return panel


def main() -> int:
    failures = 0
    checks = []

    # ---- 1. SAFETY under a Byzantine replica ----------------------------- #
    _hr("1. SAFETY (a faulty replica cannot force a dangerous plan)")
    bft = ClinicalBFT(_byz_panel())
    out = bft.run_round(SHOCK, deterioration=0.0)     # low deterioration ⇒ views can rotate
    print(f"    path={out.path} plan={out.committed_plan} views={out.views_used}")
    print(f"    {out.summary}")
    checks.append(("a quorum was reached despite the faulty replica", out.path == "quorum"))
    checks.append(("the dangerous plan did NOT commit",
                   plan_hash(out.committed_plan) != plan_hash(DANGEROUS)))
    checks.append(("committed plan respects the CHF fluid cap (≤5)",
                   out.committed_plan["fluid_rate"] <= 5.0))
    checks.append(("committed plan passed the Z3 gate", out.safety_ok is True))
    # the faulty replica's dangerous proposal never gained quorum at any view
    danger_accepts = max(
        (sum(1 for v in out.ledger if v.view == vw and v.plan_hash == plan_hash(DANGEROUS) and v.accept)
         for vw in {v.view for v in out.ledger}), default=0)
    checks.append(("the dangerous proposal never reached quorum",
                   danger_accepts < bft.quorum))

    # ---- 2. LIVENESS (deterioration outruns the debate) ------------------ #
    _hr("2. LIVENESS (crashing patient ⇒ pre-certified emergency, never deadlock)")
    bft2 = ClinicalBFT(_byz_panel())
    out2 = bft2.run_round(SHOCK, deterioration=1.0)   # max deterioration ⇒ 1 view only
    print(f"    deadline={out2.deadline_s}s max_views={out2.max_views} path={out2.path}")
    print(f"    {out2.summary}")
    checks.append(("deterioration collapsed the deadline to a single view", out2.max_views == 1))
    checks.append(("system did NOT deadlock — it committed", out2.committed_plan is not None))
    checks.append(("committed the pre-certified emergency order",
                   out2.path == "emergency_fallback"
                   and plan_hash(out2.committed_plan) == plan_hash(EMERGENCY_PLAN)))
    checks.append(("emergency order is multi-signed ≥ quorum + Z3-safe",
                   bft2.verify_certificate(out2.certificate) and out2.safety_ok))

    # ---- 3. QUORUM MATH -------------------------------------------------- #
    _hr("3. QUORUM MATH (N=3f+1 ⇒ quorum 2f+1)")
    def panel(n):
        specs = ["Intensivist", "Cardiology", "Nephrology", "InfectiousDisease"]
        return [SpecialistAgent(f"a{i}", specs[i % len(specs)], {}) for i in range(n)]
    for n, ef, eq in [(4, 1, 3), (7, 2, 5), (10, 3, 7)]:
        b = ClinicalBFT(panel(n))
        ok = (b.f == ef and b.quorum == eq)
        print(f"    N={n}: f={b.f} quorum={b.quorum} (expected f={ef} quorum={eq})")
        checks.append((f"N={n} ⇒ f={ef}, quorum={eq}", ok))

    # ---- 4. SIGNATURES (tamper detection) -------------------------------- #
    _hr("4. SIGNATURES (certificate independently verifiable; tampering caught)")
    # verify with the SAME instance that produced it (the keys live on its agents)
    base = ClinicalBFT(default_panel())
    g = base.run_round(SHOCK, deterioration=0.0)
    checks.append(("honest certificate verifies", base.verify_certificate(g.certificate)))
    # tamper the committed plan → plan-hash mismatch detected
    tampered_plan = replace(g.certificate, plan={**g.certificate.plan, "fluid_rate": 99.0})
    checks.append(("tampering the committed plan is detected",
                   not base.verify_certificate(tampered_plan)))
    # forge signatures on enough votes to drop below quorum
    votes = list(g.certificate.votes)
    margin = len([v for v in votes if v.accept]) - g.certificate.quorum
    for i in range(margin + 1):
        votes[i] = replace(votes[i], signature="deadbeef")
    forged = replace(g.certificate, votes=votes)
    checks.append(("forging signatures past the quorum margin is detected",
                   not base.verify_certificate(forged)))

    # ---- 5. FORMAL BACKSTOP + need-to-know projection -------------------- #
    _hr("5. FORMAL BACKSTOP + need-to-know projection (not ZK, not HIPAA)")
    checks.append(("every committed plan in this run passed Z3",
                   out.safety_ok and out2.safety_ok and g.safety_ok))
    from backend.aegis.consensus import redact_for
    panel_r = default_panel(remote_id="cardiology")
    remote = next(a for a in panel_r if a.agent_id == "cardiology")
    local = next(a for a in panel_r if a.agent_id == "intensivist")
    phi_state = {**SHOCK, "name": "Jane Doe", "mrn": "A12345", "patient_id": "p1"}
    seen_remote = redact_for(phi_state, remote)
    seen_local = redact_for(phi_state, local)
    print(f"    remote consultant sees keys: {sorted(seen_remote)}")
    checks.append(("remote consultant cannot see PHI (name/mrn/patient_id)",
                   all(k not in seen_remote for k in ("name", "mrn", "patient_id"))))
    checks.append(("remote consultant still sees the physiology it needs to vote",
                   "map" in seen_remote and "clinical_history" in seen_remote))
    checks.append(("on-site clinician sees the full record",
                   "name" in seen_local and "mrn" in seen_local))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — disagreement is reconciled by BFT consensus; the "
          "OS stays safe under faults and live while the patient crashes."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
