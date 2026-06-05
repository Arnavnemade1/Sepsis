import sys
import os
import numpy as np

# Add workspace to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.core.hdc import HDCEngine
from backend.core.verifier import ProtocolVerifier
from backend.core.evolution import EvolutionaryOptimizer
from backend.core.neurosymbolic import NeurosymbolicEngine

def test_hdc_engine():
    print("[TEST] Testing HDC Engine...")
    engine = HDCEngine(dimensions=10000)
    
    # 1. Random vector properties
    v1 = engine.random_vector()
    v2 = engine.random_vector()
    
    assert len(v1) == 10000, "Vector dimension should be 10,000"
    assert engine.similarity(v1, v1) == 1.0, "Self-similarity must be 1.0"
    assert abs(engine.similarity(v1, v2)) < 0.05, "Orthogonal similarity should be near 0.0"
    
    # 2. Binding and Bundling properties
    bound = engine.bind(v1, v2)
    assert abs(engine.similarity(bound, v1)) < 0.05, "Bound vector should be orthogonal to constituent vectors"
    
    bundled = engine.bundle([v1, v2])
    assert engine.similarity(bundled, v1) > 0.4, "Bundled vector should be similar to constituents"
    assert engine.similarity(bundled, v2) > 0.4, "Bundled vector should be similar to constituents"

    # 3. Continuous Level Encoding
    v_low = engine.encode_value("heart_rate", 60.0)
    v_mid = engine.encode_value("heart_rate", 120.0)
    v_high = engine.encode_value("heart_rate", 180.0)
    
    sim_low_mid = engine.similarity(v_low, v_mid)
    sim_low_high = engine.similarity(v_low, v_high)
    
    assert sim_low_mid > sim_low_high, "Similarity should decay continuously with value distance"
    print("      HDC Engine tests: PASSED")

def test_z3_verification():
    print("[TEST] Testing Z3 SMT Formal Safety Verifier...")
    verifier = ProtocolVerifier()
    
    # Clinical history and vitals templates
    history = {
        "chf": True,
        "renal_failure": False,
        "penicillin_allergy": True
    }
    vitals = {
        "map": 75.0,
        "systolic_bp": 110.0,
        "diastolic_bp": 70.0
    }
    
    # CASE 1: Allergen prescription (Penicillin allergy + PipTazo prescribed) -> SHOULD FAIL
    bad_allergy_protocol = {
        "fluid_rate": 3.0,
        "vasopressor_dose": 0.0,
        "antibiotic_code": 1 # PIPTAZO
    }
    is_safe, msg, details = verifier.verify_protocol(history, vitals, "HIGH", bad_allergy_protocol)
    assert not is_safe, "Z3 should reject Penicillin-based antibiotic when Penicillin allergy is present"
    assert "Drug_Allergy_Violation" in details["violated_rules"][0], "Expected allergy violation flag"
    print("      Allergy constraint check: PASSED (Z3 blocked Pip/Tazo)")

    # CASE 2: Fluid Overload (CHF patient + Fluid rate > 5.0 mL/kg/hr) -> SHOULD FAIL
    bad_fluid_protocol = {
        "fluid_rate": 8.0, # CHF limit is 5.0
        "vasopressor_dose": 0.05,
        "antibiotic_code": 3 # CEFEPIME (No allergy conflict)
    }
    is_safe, msg, details = verifier.verify_protocol(history, vitals, "HIGH", bad_fluid_protocol)
    assert not is_safe, "Z3 should reject fluid rate > 5.0 for CHF patients"
    assert "CHF_Fluid_Cap" in details["violated_rules"][0], "Expected CHF fluid cap violation flag"
    print("      CHF Fluid cap constraint check: PASSED (Z3 blocked fluid overload)")

    # CASE 3: Unjustified Vasopressor (MAP is healthy, but high vaso dose proposed) -> SHOULD FAIL
    bad_vaso_protocol = {
        "fluid_rate": 2.0,
        "vasopressor_dose": 0.25, # High dose, only justified if MAP < 65
        "antibiotic_code": 3
    }
    # MAP is 80 (healthy)
    healthy_bp_vitals = {"map": 80.0}
    is_safe, msg, details = verifier.verify_protocol(history, healthy_bp_vitals, "HIGH", bad_vaso_protocol)
    assert not is_safe, "Z3 should reject high vasopressor dosage if MAP is healthy"
    assert "Vasopressor_MAP_Cap" in details["violated_rules"][0], "Expected MAP justification violation"
    print("      Vasopressor MAP constraint check: PASSED (Z3 blocked unnecessary vasopressor)")

    # CASE 4: Certified Safe Protocol -> SHOULD PASS
    safe_protocol = {
        "fluid_rate": 3.5, # Under 5.0 for CHF
        "vasopressor_dose": 0.0,
        "antibiotic_code": 3 # CEFEPIME (No allergy conflict)
    }
    is_safe, msg, details = verifier.verify_protocol(history, vitals, "HIGH", safe_protocol)
    assert is_safe, f"Z3 should approve safe protocol. Reason: {msg}"
    assert details["z3_status"] == "unsat", "Verified protocol must have UNSAT violation status"
    print("      Safe intervention blueprint check: PASSED (Z3 certified safe)")
    print("      Z3 Formal Verification tests: PASSED")

def test_evolutionary_optimization():
    print("[TEST] Testing Evolutionary Protocol Synthesizer...")
    optimizer = EvolutionaryOptimizer(pop_size=20, generations=10)
    
    # Sepsis patient vitals
    sepsis_vitals = {
        "heart_rate": 115.0,
        "spo2": 91.0,
        "skin_temp": 38.9,
        "systolic_bp": 88.0,
        "diastolic_bp": 55.0,
        "sweat_lactate": 3.2,
        "map": 68.2,
        "sepsis_similarity": 0.52
    }
    history = {
        "chf": True,
        "renal_failure": False,
        "penicillin_allergy": True
    }
    
    best_protocol, fitness = optimizer.optimize(sepsis_vitals, history, "HIGH")
    
    # Verify bounds of output
    assert 0.0 <= best_protocol["fluid_rate"] <= 25.0, "Fluid rate out of bounds"
    assert 0.0 <= best_protocol["vasopressor_dose"] <= 0.4, "Vasopressor out of bounds"
    assert 1 <= best_protocol["antibiotic_code"] <= 5, "Sepsis protocol must prescribe an antibiotic"
    
    print(f"      GA proposed protocol: Fluids={best_protocol['fluid_rate']} mL/kg/h, Vaso={best_protocol['vasopressor_dose']} mcg/kg/min, Antibiotic Code={best_protocol['antibiotic_code']}")
    print("      Evolutionary Optimizer tests: PASSED")

if __name__ == "__main__":
    print("==================================================")
    print("HYPERSEPSISTWIN-Ω: AUTOMATED BACKEND VERIFIER")
    print("==================================================")
    try:
        test_hdc_engine()
        print("-" * 50)
        test_z3_verification()
        print("-" * 50)
        test_evolutionary_optimization()
        print("==================================================")
        print("SYSTEM VERIFICATION SUCCESSFUL: ALL TESTS PASSED.")
        print("==================================================")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[TEST FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[TEST ERROR] Unexpected Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
