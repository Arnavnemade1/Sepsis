from typing import Dict, Any, Tuple
from z3 import *

class ProtocolVerifier:
    # Map antibiotic codes to names
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
        # Create Z3 Solver
        solver = Solver()

        # 1. Declare Z3 variables for the protocol parameters
        fluid_rate = Real('fluid_rate')          # mL/kg/hr
        vaso_dose = Real('vaso_dose')            # mcg/kg/min (Norepinephrine)
        ab_selection = Int('ab_selection')        # Antibiotic code [0-5]

        # 2. Add proposed protocol parameters as assertions
        solver.add(fluid_rate == float(proposed_protocol.get("fluid_rate", 0.0)))
        solver.add(vaso_dose == float(proposed_protocol.get("vasopressor_dose", 0.0)))
        solver.add(ab_selection == int(proposed_protocol.get("antibiotic_code", 0)))

        # 3. Extract patient clinical history flags
        has_chf = BoolVal(bool(clinical_history.get("chf", False)))
        has_renal_failure = BoolVal(bool(clinical_history.get("renal_failure", False)))
        has_penicillin_allergy = BoolVal(bool(clinical_history.get("penicillin_allergy", False)))
        
        # Extract patient vitals
        fallback_map = (vitals.get("systolic_bp", 120.0) + (2.0 * vitals.get("diastolic_bp", 80.0))) / 3.0
        map_val = RealVal(float(vitals.get("map", fallback_map)))
        is_sepsis_high = BoolVal(risk_level == "HIGH")

        # 4. Formulate safety constraints (invariants)
        
        # Constraint A: Fluid administration limit for Congestive Heart Failure (CHF) patients
        # Fluid rate must not exceed 5.0 mL/kg/hr to avoid acute pulmonary edema
        fluid_chf_safe = Implies(has_chf, fluid_rate <= 5.0)

        # Constraint B: Fluid administration limit for Renal Failure patients
        # Fluid rate must not exceed 4.0 mL/kg/hr to avoid severe fluid overload
        fluid_renal_safe = Implies(has_renal_failure, fluid_rate <= 4.0)

        # Constraint C: Absolute maximum fluid rate for any patient
        fluid_max_safe = fluid_rate <= 25.0

        # Constraint D: Non-negative fluid rate
        fluid_min_safe = fluid_rate >= 0.0

        # Constraint E: Vasopressor absolute safety cap
        # Norepinephrine dose must never exceed a toxic threshold of 0.4 mcg/kg/min
        vaso_max_safe = vaso_dose <= 0.4

        # Constraint F: Non-negative vasopressor dose
        vaso_min_safe = vaso_dose >= 0.0

        # Constraint G: Vasopressor dosing conditions
        # Norepinephrine dosing > 0.1 mcg/kg/min is ONLY allowed if Mean Arterial Pressure (MAP) is < 65 mmHg
        vaso_map_safe = Implies(vaso_dose > 0.1, map_val < 65.0)

        # Constraint H: Drug-Allergy Safety
        # If the patient has a penicillin allergy, they CANNOT receive Piperacillin/Tazobactam (Code 1)
        allergy_safe = Implies(has_penicillin_allergy, ab_selection != 1)

        # Constraint I: Clinical intervention timing
        # If the patient is determined to be in HIGH risk sepsis, an antibiotic must be administered (Code != 0)
        antibiotic_needed = Implies(is_sepsis_high, ab_selection != 0)

        # Constraint J: Valid antibiotic selection range
        antibiotic_valid = And(ab_selection >= 0, ab_selection <= 5)

        # 5. Define the "VIOLATION" condition
        # A violation occurs if ANY of the safety constraints are NOT met.
        violation = Or(
            Not(fluid_chf_safe),
            Not(fluid_renal_safe),
            Not(fluid_max_safe),
            Not(fluid_min_safe),
            Not(vaso_max_safe),
            Not(vaso_min_safe),
            Not(vaso_map_safe),
            Not(allergy_safe),
            Not(antibiotic_needed),
            Not(antibiotic_valid)
        )

        # Add the violation assertion to the solver
        solver.add(violation)

        # 6. Check satisfiability of a violation
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
