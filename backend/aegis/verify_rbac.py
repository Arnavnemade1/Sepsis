"""Verify harness for the Clinical RBAC shell — claims asserted by measurement.

  1. POLICY SOUND  — Z3 proves every grant respects its capability's minimum ring
                     (unsat); a deliberately broken policy is caught (sat + the hole).
  2. CAPABILITY    — deny-by-default; separation of duties (only the attending may
                     order treatment — the chief, though higher-ring, may not).
  3. NO ESCALATION — Z3 proves a nurse can never reach a treatment/cohort capability;
                     a legitimately-granted capability is (correctly) not "blocked".
  4. SESSIONS      — a valid token authorizes; a forged or expired token is rejected.
  5. AUDIT         — every decision (grant + deny) is hash-chain logged; tamper caught.
  6. GATE          — the RBAC gate lets an attending's syscall through and blocks a
                     nurse's with EPERM, never invoking the underlying syscall.

    python -m backend.aegis.verify_rbac
"""
from __future__ import annotations

from backend.aegis.rbac import (GRANTS, RING, RBAC, RBACGate, prove_policy_sound,
                                prove_no_escalation)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def main() -> int:
    failures = 0
    checks = []
    rbac = RBAC()

    # ---- 1. POLICY SOUND ------------------------------------------------- #
    _hr("1. POLICY SOUNDNESS (Z3 over the whole grant matrix)")
    sound = prove_policy_sound()
    print(f"    sound={sound['sound']} z3={sound['z3_status']} "
          f"grants_checked={sound['grants_checked']}")
    broken = prove_policy_sound(grants={"nurse": frozenset({"order:treat"})})
    print(f"    broken policy (nurse↦order:treat): sound={broken['sound']} "
          f"z3={broken['z3_status']} violations={broken['violations']}")
    checks.append(("real policy is proven sound (unsat)",
                   sound["sound"] and sound["z3_status"] == "unsat"))
    checks.append(("a ring-violating grant is caught by Z3 (sat + hole)",
                   broken["sound"] is False and len(broken["violations"]) == 1))

    # ---- 2. CAPABILITY MATRIX ------------------------------------------- #
    _hr("2. CAPABILITY MODEL (deny-by-default + separation of duties)")
    att = rbac.login("dr_house", "attending")
    chief = rbac.login("dr_chief", "chief")
    nurse = rbac.login("rn_joy", "nurse")
    pharm = rbac.login("rx_sam", "pharmacist")
    checks.append(("attending MAY order treatment", rbac.authorize(att.token, "order:treat")[0]))
    checks.append(("chief (higher ring) may NOT order treatment — separation of duties",
                   not rbac.authorize(chief.token, "order:treat")[0]))
    checks.append(("only the chief sees the whole cohort",
                   rbac.authorize(chief.token, "view:cohort")[0]
                   and not rbac.authorize(att.token, "view:cohort")[0]))
    checks.append(("nurse may acknowledge alarms but not treat",
                   rbac.authorize(nurse.token, "ack:alarm")[0]
                   and not rbac.authorize(nurse.token, "order:treat")[0]))
    checks.append(("pharmacist may veto meds but not order treatment",
                   rbac.authorize(pharm.token, "pharm:veto")[0]
                   and not rbac.authorize(pharm.token, "order:treat")[0]))
    checks.append(("deny-by-default: an unknown capability is denied",
                   not rbac.authorize(att.token, "launch:missiles")[0]))

    # ---- 3. NO ESCALATION (Z3) ------------------------------------------ #
    _hr("3. NO PRIVILEGE ESCALATION (Z3 per request)")
    e1 = prove_no_escalation("nurse", "order:treat")
    e2 = prove_no_escalation("nurse", "view:cohort")
    e3 = prove_no_escalation("attending", "order:treat")
    print(f"    nurse→order:treat blocked={e1['escalation_blocked']}")
    print(f"    nurse→view:cohort blocked={e2['escalation_blocked']}")
    print(f"    attending→order:treat blocked={e3['escalation_blocked']} (legit grant)")
    checks.append(("Z3 proves nurse cannot reach order:treat", e1["escalation_blocked"]))
    checks.append(("Z3 proves nurse cannot reach view:cohort", e2["escalation_blocked"]))
    checks.append(("a legitimately granted capability is NOT blocked",
                   not e3["escalation_blocked"]))

    # ---- 4. SESSIONS ----------------------------------------------------- #
    _hr("4. SIGNED SESSIONS (forged / expired rejected)")
    forged = nurse.token[:-4] + "0000"
    expired = RBAC(token_ttl_s=-1).login("ghost", "attending")
    checks.append(("valid token authorizes", rbac.authorize(nurse.token, "ack:alarm")[0]))
    checks.append(("forged token rejected", not rbac.authorize(forged, "ack:alarm")[0]))
    checks.append(("expired token rejected",
                   rbac.verify_token(expired.token) is None))

    # ---- 5. AUDIT -------------------------------------------------------- #
    _hr("5. IMMUTABLE AUDIT LEDGER")
    n = len(rbac.audit.entries)
    grants = sum(1 for e in rbac.audit.entries if e.decision == "GRANT")
    denies = sum(1 for e in rbac.audit.entries if e.decision == "DENY")
    print(f"    {n} decisions logged ({grants} grant / {denies} deny); "
          f"head {rbac.audit.head()[:12]}")
    checks.append(("every authorization decision was logged (grant + deny)",
                   n >= 11 and grants >= 1 and denies >= 1))
    checks.append(("audit hash-chain verifies", rbac.audit.verify()))
    rbac.audit.entries[2].decision = "GRANT" if rbac.audit.entries[2].decision == "DENY" else "DENY"
    checks.append(("tampering with an audit entry is detected", not rbac.audit.verify()))

    # ---- 6. GATE over the syscall trap --------------------------------- #
    _hr("6. RBAC GATE (over the write-syscall trap)")
    calls = []
    class _Res:
        def to_dict(self): return {"ok": True, "syscall_ran": True}
    def fake_syscall(syscall, bed=None, args=None, caller=None):
        calls.append((syscall, caller)); return _Res()
    gate = RBACGate(RBAC(), fake_syscall)
    att2 = gate.rbac.login("dr_a", "attending")
    nrs2 = gate.rbac.login("rn_b", "nurse")
    r_ok = gate.invoke(att2.token, "treat", bed="bed_3", args={})
    r_no = gate.invoke(nrs2.token, "treat", bed="bed_3", args={})
    print(f"    attending treat: ok={r_ok.get('ok')} ran={r_ok.get('syscall_ran')}")
    print(f"    nurse treat: ok={r_no.get('ok')} errno={r_no.get('errno')}")
    checks.append(("attending's syscall passes the gate and runs",
                   r_ok.get("ok") and r_ok.get("syscall_ran")))
    checks.append(("nurse's syscall is blocked with EPERM",
                   r_no.get("ok") is False and r_no.get("errno") == "EPERM"))
    checks.append(("a blocked syscall never reaches the kernel",
                   calls == [("treat", "dr_a")]))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — a multi-user OS with formally-verified privilege "
          "rings and a tamper-evident audit of every clinical action."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
