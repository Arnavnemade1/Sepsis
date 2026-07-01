"""Verify harness for the Clinical Safety-Logic DSL → Z3 compiler — claims asserted.

We author rules in the DSL, compile them to Z3, and prove the compiler + prover
behave exactly as a formal safety language must:

  1. COMPILE    — the DSL parses and compiles to one Z3 BoolRef per rule.
  2. PROVE      — a guideline-compliant protocol is proven (negation unsat) on ALL rules.
  3. CATCH      — each violation class is caught with a counterexample: missing
                  antibiotic on HIGH risk, the WITHIN-1hr temporal bound, a penicillin
                  allergy + PipTazo, a CHF fluid overload, a renal vasopressor overdose.
  4. TEMPORAL   — WITHIN binds the planned administration time: same drug, in-time vs
                  late, flips proven→violated.
  5. CONSISTENCY— a self-contradictory policy is detected (jointly unsat); a sane one
                  is satisfiable with a concrete example protocol.
  6. CERTIFICATE— every verdict carries a machine-readable Z3 s-expression formula.

    python -m backend.aegis.verify_clinical_dsl
"""
from __future__ import annotations

from backend.aegis.clinical_dsl import (DEFAULT_POLICY, compile_rules,
                                        check_consistency, verify_protocol)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def _viol(cert):
    return {c.rule for c in cert.rules if not c.satisfied}


def main() -> int:
    failures = 0
    checks = []
    rules = compile_rules(DEFAULT_POLICY)

    # ---- 1. COMPILE ------------------------------------------------------ #
    _hr("1. COMPILE (DSL → Z3)")
    print(f"    compiled {len(rules)} rules: {[r.name for r in rules]}")
    checks.append(("DSL compiles to 4 rules", len(rules) == 4))
    checks.append(("each rule is a real Z3 formula (has an s-expression)",
                   all(r.sexpr() for r in rules)))

    # ---- 2. PROVE -------------------------------------------------------- #
    _hr("2. PROVE (compliant protocol satisfies all rules)")
    ctx = {"risk_level": "HIGH", "chf": True, "penicillin_allergy": True, "renal_failure": True}
    safe = {"antibiotic_code": 1, "antibiotic_time_hr": 0.5,   # Cefepime within 30 min
            "fluid_rate": 4.0, "vasopressor_dose": 0.1}
    cert = verify_protocol(rules, safe, ctx)
    print(f"    ok={cert.ok} ({cert.n_satisfied}/{cert.n_rules}); "
          f"all z3_status={[c.z3_status for c in cert.rules]}")
    checks.append(("compliant protocol satisfies ALL rules", cert.ok))
    checks.append(("every satisfied rule was proven via unsat-negation",
                   all(c.z3_status == "unsat" for c in cert.rules)))

    # ---- 3. CATCH (violation classes) ----------------------------------- #
    _hr("3. CATCH (each violation class, with counterexample)")
    # missing antibiotic on HIGH risk
    v_abx = verify_protocol(rules, {"antibiotic_code": 0, "antibiotic_time_hr": 0.5,
                                    "fluid_rate": 4.0, "vasopressor_dose": 0.1}, ctx)
    # penicillin allergy + PipTazo (code 2)
    v_all = verify_protocol(rules, {"antibiotic_code": 2, "antibiotic_time_hr": 0.5,
                                    "fluid_rate": 4.0, "vasopressor_dose": 0.1}, ctx)
    # CHF fluid overload
    v_chf = verify_protocol(rules, {"antibiotic_code": 1, "antibiotic_time_hr": 0.5,
                                    "fluid_rate": 30.0, "vasopressor_dose": 0.1}, ctx)
    # renal failure vasopressor overdose
    v_ren = verify_protocol(rules, {"antibiotic_code": 1, "antibiotic_time_hr": 0.5,
                                    "fluid_rate": 4.0, "vasopressor_dose": 0.9}, ctx)
    print(f"    no-abx violations: {_viol(v_abx)}")
    print(f"    allergy violations: {_viol(v_all)}")
    print(f"    CHF violations: {_viol(v_chf)}")
    print(f"    renal violations: {_viol(v_ren)}")
    checks.append(("HIGH risk + no antibiotic → sepsis_antibiotic violated",
                   "sepsis_antibiotic" in _viol(v_abx)))
    checks.append(("penicillin allergy + PipTazo → allergy_guard violated",
                   "allergy_guard" in _viol(v_all)))
    checks.append(("CHF + fluid 30 → fluid_cap violated", "fluid_cap" in _viol(v_chf)))
    checks.append(("renal failure + vaso 0.9 → renal_vaso_cap violated",
                   "renal_vaso_cap" in _viol(v_ren)))
    checks.append(("a violation carries a counterexample model",
                   any(c.counterexample for c in v_chf.rules if not c.satisfied)))

    # ---- 4. TEMPORAL (WITHIN binds the administration time) ------------- #
    _hr("4. TEMPORAL (WITHIN 1hr binds planned administration time)")
    in_time = verify_protocol(rules, {"antibiotic_code": 1, "antibiotic_time_hr": 0.75,
                                      "fluid_rate": 4.0, "vasopressor_dose": 0.1}, ctx)
    too_late = verify_protocol(rules, {"antibiotic_code": 1, "antibiotic_time_hr": 3.0,
                                       "fluid_rate": 4.0, "vasopressor_dose": 0.1}, ctx)
    print(f"    antibiotic @0.75h → sepsis_antibiotic satisfied: "
          f"{'sepsis_antibiotic' not in _viol(in_time)}")
    print(f"    antibiotic @3.0h  → sepsis_antibiotic satisfied: "
          f"{'sepsis_antibiotic' not in _viol(too_late)}")
    checks.append(("same drug within 1hr passes the temporal rule",
                   "sepsis_antibiotic" not in _viol(in_time)))
    checks.append(("same drug at 3hr FAILS the temporal rule",
                   "sepsis_antibiotic" in _viol(too_late)))

    # ---- 5. CONSISTENCY (policy self-contradiction detection) ---------- #
    _hr("5. CONSISTENCY (detect a self-contradictory policy)")
    sane = check_consistency(rules, ctx)
    contradictory = compile_rules("""
        RULE must_treat:  ALWAYS (risk = HIGH -> antibiotic != NONE)
        RULE never_treat: ALWAYS (risk = HIGH -> antibiotic = NONE)
    """)
    bad = check_consistency(contradictory, {"risk_level": "HIGH"})
    print(f"    default policy consistent={sane['consistent']} "
          f"(example protocol keys: {list(sane.get('example_protocol', {}))[:3]}…)")
    print(f"    contradictory policy consistent={bad['consistent']} (z3={bad['z3_status']})")
    checks.append(("a sane policy is jointly satisfiable", sane["consistent"] is True))
    checks.append(("a sane policy yields a concrete example protocol",
                   bool(sane.get("example_protocol"))))
    checks.append(("a self-contradictory policy is detected (unsat)",
                   bad["consistent"] is False))

    # ---- 6. CERTIFICATE -------------------------------------------------- #
    _hr("6. CERTIFICATE (machine-readable proof artifact)")
    c0 = cert.rules[0]
    print(f"    {c0.rule}: status={c0.z3_status}")
    print(f"    formula s-expr: {c0.formula[:78]}…")
    checks.append(("certificate carries a Z3 s-expression formula",
                   c0.formula.startswith("(") and len(c0.formula) > 10))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — hospitals author safety rules; the OS compiles "
          "them to Z3 and proves every protocol before it is suggested."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
