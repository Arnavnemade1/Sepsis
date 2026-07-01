from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from z3 import *


@dataclass
class SafetyConstraint:
    """One named clinical safety invariant the protocol must satisfy.

    ``name`` is the stable identifier (also the obligation name when exported);
    ``desc`` is the clinician-readable statement; ``formula`` is the Z3 BoolRef that
    must hold. The protocol *satisfies* the constraint iff its negation is unsat."""
    name: str
    desc: str
    formula: "BoolRef"


class ProtocolVerifier:
    # Map antibiotic codes to names (this verifier's codebook — note it differs from
    # the Clinical-Safety DSL's enum; callers that cross the boundary must translate).
    ANTIBIOTICS = {
        0: "None",
        1: "Piperacillin/Tazobactam (Penicillin-based, Broad Spectrum)",
        2: "Vancomycin (Glycopeptide, targets MRSA)",
        3: "Cefepime (Cephalosporin, targets Gram-negative)",
        4: "Meropenem (Carbapenem, Ultra-broad spectrum)",
        5: "Levofloxacin (Fluoroquinolone)"
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    # The single source of truth for the safety model: the bound patient/protocol
    # context plus the named invariants. Both the live verifier and the portable
    # proof-export path consume THIS, so they can never drift apart.
    # ------------------------------------------------------------------ #
    def safety_model(
        self,
        clinical_history: Dict[str, Any],
        vitals: Dict[str, float],
        risk_level: str,
        proposed_protocol: Dict[str, Any],
    ) -> Tuple[List["BoolRef"], List[SafetyConstraint], Dict[str, Any]]:
        """Return (bindings, constraints, vars).

        Patient context is modelled as *declared, bound* Z3 variables (not baked-in
        constants) so an exported SMT-LIB2 obligation is self-documenting: an auditor
        sees the full context as explicit assertions."""
        # Declared variables — protocol parameters AND patient context.
        fluid_rate = Real('fluid_rate')          # mL/kg/hr
        vaso_dose = Real('vaso_dose')            # mcg/kg/min (Norepinephrine)
        ab_selection = Int('ab_selection')       # Antibiotic code [0-5]
        has_chf = Bool('has_chf')
        has_renal_failure = Bool('has_renal_failure')
        has_penicillin_allergy = Bool('has_penicillin_allergy')
        is_sepsis_high = Bool('is_sepsis_high')
        map_val = Real('map')                    # mean arterial pressure (mmHg)

        fallback_map = (vitals.get("systolic_bp", 120.0) + (2.0 * vitals.get("diastolic_bp", 80.0))) / 3.0
        bindings: List[BoolRef] = [
            fluid_rate == float(proposed_protocol.get("fluid_rate", 0.0)),
            vaso_dose == float(proposed_protocol.get("vasopressor_dose", 0.0)),
            ab_selection == int(proposed_protocol.get("antibiotic_code", 0)),
            has_chf == bool(clinical_history.get("chf", False)),
            has_renal_failure == bool(clinical_history.get("renal_failure", False)),
            has_penicillin_allergy == bool(clinical_history.get("penicillin_allergy", False)),
            is_sepsis_high == (risk_level == "HIGH"),
            map_val == float(vitals.get("map", fallback_map)),
        ]

        constraints = [
            SafetyConstraint("chf_fluid_cap",
                "CHF ⇒ fluid_rate ≤ 5.0 mL/kg/hr (avoid acute pulmonary edema)",
                Implies(has_chf, fluid_rate <= 5.0)),
            SafetyConstraint("renal_fluid_cap",
                "renal failure ⇒ fluid_rate ≤ 4.0 mL/kg/hr (avoid fluid overload)",
                Implies(has_renal_failure, fluid_rate <= 4.0)),
            SafetyConstraint("fluid_abs_cap",
                "fluid_rate ≤ 25.0 mL/kg/hr (absolute ceiling)",
                fluid_rate <= 25.0),
            SafetyConstraint("fluid_nonneg",
                "fluid_rate ≥ 0", fluid_rate >= 0.0),
            SafetyConstraint("vaso_abs_cap",
                "vasopressor ≤ 0.4 mcg/kg/min (toxic-dose ceiling)",
                vaso_dose <= 0.4),
            SafetyConstraint("vaso_nonneg",
                "vasopressor ≥ 0", vaso_dose >= 0.0),
            SafetyConstraint("vaso_map_justified",
                "vasopressor > 0.1 ⇒ MAP < 65 mmHg (dose justified by hypotension)",
                Implies(vaso_dose > 0.1, map_val < 65.0)),
            SafetyConstraint("allergy_guard",
                "penicillin allergy ⇒ antibiotic ≠ Pip-Tazo (code 1)",
                Implies(has_penicillin_allergy, ab_selection != 1)),
            SafetyConstraint("sepsis_antibiotic",
                "HIGH-risk sepsis ⇒ an antibiotic is administered (code ≠ 0)",
                Implies(is_sepsis_high, ab_selection != 0)),
            SafetyConstraint("antibiotic_valid",
                "antibiotic code within catalog [0,5]",
                And(ab_selection >= 0, ab_selection <= 5)),
        ]
        vars = {"fluid_rate": fluid_rate, "vaso_dose": vaso_dose,
                "ab_selection": ab_selection, "has_chf": has_chf,
                "has_renal_failure": has_renal_failure,
                "has_penicillin_allergy": has_penicillin_allergy,
                "is_sepsis_high": is_sepsis_high, "map": map_val}
        return bindings, constraints, vars

    def verify_protocol(
        self,
        clinical_history: Dict[str, Any],
        vitals: Dict[str, float],
        risk_level: str,
        proposed_protocol: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Formally verifies treatment parameters using the Z3 SMT Solver.
        Returns:
            (is_safe, status_message, proof_metadata)
        """
        bindings, constraints, _vars = self.safety_model(
            clinical_history, vitals, risk_level, proposed_protocol)

        solver = Solver()
        solver.add(*bindings)

        # A violation occurs if ANY of the safety constraints are NOT met.
        solver.add(Or(*[Not(c.formula) for c in constraints]))

        # Check satisfiability of a violation.
        result = solver.check()

        # If a violation is UNSATISFIABLE, the proposed protocol is formally certified safe.
        # If a violation is SATISFIABLE, Z3 has found a scenario where the protocol violates safety guidelines.
        if result == unsat:
            # Reconstruct the Z3 proof formula text for display
            formula_text = (
                f"Assert(fluid_rate == {proposed_protocol.get('fluid_rate')}) ∧ "
                f"Assert(vaso_dose == {proposed_protocol.get('vasopressor_dose')}) ∧ "
                f"Assert(ab_selection == {proposed_protocol.get('antibiotic_code')}) ∧ "
                f"Verify(CHF_Fluid_Cap: CHF ⇒ fluid_rate ≤ 5.0) ∧ "
                f"Verify(Renal_Fluid_Cap: Renal ⇒ fluid_rate ≤ 4.0) ∧ "
                f"Verify(Vaso_Cap: vaso_dose ≤ 0.4) ∧ "
                f"Verify(Vaso_MAP: vaso_dose > 0.1 ⇒ MAP < 65) ∧ "
                f"Verify(Allergy: Penicillin_Allergy ⇒ ab_selection ≠ 1) ∧ "
                f"Verify(Sepsis_AB: Sepsis_High ⇒ ab_selection ≠ 0)"
            )
            return True, "FORMALLY VERIFIED: Protocol satisfies all clinical safety invariants.", {
                "formula": formula_text,
                "z3_status": "unsat",
                "proof_details": {
                    "chf_check": "passed",
                    "renal_check": "passed",
                    "vasopressor_cap_check": "passed",
                    "vasopressor_perfusion_check": "passed",
                    "allergy_check": "passed",
                    "antibiotic_timing_check": "passed"
                }
            }
        else:
            # Satisfiable means a violation exists. Find which safety rule is violated.
            model = solver.model()
            violated_rules = []
            
            # Helper to check individual violation states
            if clinical_history.get("chf", False) and float(proposed_protocol.get("fluid_rate", 0)) > 5.0:
                violated_rules.append("CHF_Fluid_Cap: Fluid rate exceeds 5.0 mL/kg/hr for patient with Congestive Heart Failure.")
            if clinical_history.get("renal_failure", False) and float(proposed_protocol.get("fluid_rate", 0)) > 4.0:
                violated_rules.append("Renal_Fluid_Cap: Fluid rate exceeds 4.0 mL/kg/hr for patient with Renal Failure.")
            if float(proposed_protocol.get("fluid_rate", 0)) > 25.0:
                violated_rules.append("Fluid_Max_Cap: Absolute fluid rate exceeds 25.0 mL/kg/hr ceiling limit.")
            if float(proposed_protocol.get("fluid_rate", 0)) < 0.0:
                violated_rules.append("Fluid_Min_Cap: Fluid rate cannot be negative.")
            if float(proposed_protocol.get("vasopressor_dose", 0)) > 0.4:
                violated_rules.append("Vasopressor_Max_Cap: Norepinephrine dose exceeds maximum safe limit of 0.4 mcg/kg/min.")
            if float(proposed_protocol.get("vasopressor_dose", 0)) < 0.0:
                violated_rules.append("Vasopressor_Min_Cap: Vasopressor dose cannot be negative.")
            if float(proposed_protocol.get("vasopressor_dose", 0)) > 0.1 and float(vitals.get("map", 80.0)) >= 65.0:
                violated_rules.append("Vasopressor_MAP_Cap: Vasopressor dose > 0.1 is not justified because MAP is >= 65 mmHg.")
            if clinical_history.get("penicillin_allergy", False) and int(proposed_protocol.get("antibiotic_code", 0)) == 1:
                violated_rules.append("Drug_Allergy_Violation: Penicillin allergy detected, but Piperacillin/Tazobactam (Code 1) was prescribed.")
            if risk_level == "HIGH" and int(proposed_protocol.get("antibiotic_code", 0)) == 0:
                violated_rules.append("Sepsis_Antibiotic_Missing: Sepsis risk is HIGH, but antibiotic treatment was omitted.")
            if int(proposed_protocol.get("antibiotic_code", 0)) < 0 or int(proposed_protocol.get("antibiotic_code", 0)) > 5:
                violated_rules.append("Invalid_Antibiotic_Code: Prescribed antibiotic code is outside of catalog limits.")

            status_msg = f"VERIFICATION REJECTED: Clinical safety invariants violated: {', '.join(violated_rules)}"
            
            # Convert model details to JSON compatible dict
            model_dump = {}
            for d in model.decls():
                model_dump[d.name()] = str(model[d])

            return False, status_msg, {
                "formula": "Violation condition is satisfiable.",
                "z3_status": "sat",
                "violated_rules": violated_rules,
                "z3_model": model_dump
            }
