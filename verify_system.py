import sys
import os
import numpy as np

# Add workspace to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.core.hdc import HDCEngine
from backend.core.verifier import ProtocolVerifier
from backend.core.evolution import EvolutionaryOptimizer
from backend.core.neurosymbolic import NeurosymbolicEngine
from backend.core.twin import PatientDigitalTwin

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

    # 4. Temporal HDC encoding should preserve trajectory class evidence.
    stable_window = [{
        "heart_rate": 72.0, "hrv": 85.0, "spo2": 98.5, "skin_temp": 36.6,
        "motion": 12.0, "systolic_bp": 120.0, "diastolic_bp": 80.0,
        "map": 93.3, "sweat_lactate": 0.8, "respiratory_rate": 14.0
    }] * 8
    septic_window = [{
        "heart_rate": 126.0, "hrv": 12.0, "spo2": 88.0, "skin_temp": 39.1,
        "motion": 1.0, "systolic_bp": 78.0, "diastolic_bp": 46.0,
        "map": 55.3, "sweat_lactate": 5.1, "respiratory_rate": 28.0
    }] * 8
    stable_temporal = engine.encode_temporal_window(stable_window)
    septic_temporal = engine.encode_temporal_window(septic_window)
    stable_h, stable_s = engine.classify_temporal_state(stable_temporal)
    septic_h, septic_s = engine.classify_temporal_state(septic_temporal)
    assert stable_h > stable_s, "Stable temporal trajectory should favor healthy temporal archetype"
    assert septic_s > septic_h, "Septic temporal trajectory should favor sepsis temporal archetype"
    print("      HDC Engine tests: PASSED")

def test_digital_twin_features():
    print("[TEST] Testing personalized digital twin features...")
    engine = HDCEngine(dimensions=10000)
    twin = PatientDigitalTwin("test-patient", engine)

    stable = {
        "heart_rate": 72.0,
        "hrv": 85.0,
        "spo2": 98.5,
        "skin_temp": 36.6,
        "motion": 12.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "map": 93.3,
        "sweat_lactate": 0.8,
        "respiratory_rate": 14.0
    }
    for _ in range(12):
        twin.update_telemetry(stable)

    stable_features = twin.get_current_twin_state()["features"]
    assert stable_features["instability_score"] < 2.0, "Stable baseline should have low instability"
    assert stable_features["perfusion_index"] > 0.9, "Stable baseline should have strong perfusion index"

    septic = {
        "heart_rate": 126.0,
        "hrv": 12.0,
        "spo2": 88.0,
        "skin_temp": 39.1,
        "motion": 1.0,
        "systolic_bp": 78.0,
        "diastolic_bp": 46.0,
        "map": 55.3,
        "sweat_lactate": 5.1,
        "respiratory_rate": 28.0
    }
    twin.update_telemetry(septic)
    septic_features = twin.get_current_twin_state()["features"]
    assert septic_features["instability_score"] > stable_features["instability_score"], "Septic state should diverge from personal baseline"
    assert septic_features["perfusion_index"] < 0.6, "Septic shock should degrade perfusion index"
    print("      Digital twin feature extraction tests: PASSED")

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
    verifier = ProtocolVerifier()
    is_safe, msg, _ = verifier.verify_protocol(history, sepsis_vitals, "HIGH", best_protocol)
    
    # Verify bounds of output
    assert 0.0 <= best_protocol["fluid_rate"] <= 25.0, "Fluid rate out of bounds"
    assert 0.0 <= best_protocol["vasopressor_dose"] <= 0.4, "Vasopressor out of bounds"
    assert 1 <= best_protocol["antibiotic_code"] <= 5, "Sepsis protocol must prescribe an antibiotic"
    assert is_safe, f"Optimizer should propose a verifier-safe protocol. Reason: {msg}"
    
    print(f"      GA proposed protocol: Fluids={best_protocol['fluid_rate']} mL/kg/h, Vaso={best_protocol['vasopressor_dose']} mcg/kg/min, Antibiotic Code={best_protocol['antibiotic_code']}")
    print("      Evolutionary Optimizer tests: PASSED")

def test_neurosymbolic_detection():
    print("[TEST] Testing calibrated neurosymbolic detection...")
    hdc = HDCEngine(dimensions=10000)
    detector = NeurosymbolicEngine()
    history = {"wbc": 7.5, "gcs": 15}

    stable_vitals = {
        "heart_rate": 72.0,
        "hrv": 85.0,
        "spo2": 98.5,
        "skin_temp": 36.6,
        "motion": 12.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 80.0,
        "map": 93.3,
        "sweat_lactate": 0.8,
        "respiratory_rate": 14.0
    }
    healthy_sim, sepsis_sim = hdc.classify_state(hdc.encode_patient_state(stable_vitals))
    risk, sirs, qsofa, _, details = detector.evaluate_patient(
        stable_vitals,
        sepsis_sim,
        history,
        healthy_sim,
        {"hrv": "STABLE", "temp": "STABLE"}
    )
    assert risk == "LOW", f"Stable physiology should not trigger sepsis risk; got {risk}"
    assert sirs == 0 and qsofa == 0, "Stable physiology should have zero SIRS/qSOFA"
    assert details["sepsis_probability"] < 0.35, "Stable probability should remain low"
    print("      Stable false-positive guard: PASSED")

    early_vitals = {
        "heart_rate": 102.0,
        "hrv": 32.0,
        "spo2": 95.0,
        "skin_temp": 38.4,
        "motion": 5.0,
        "systolic_bp": 104.0,
        "diastolic_bp": 66.0,
        "map": 77.3,
        "sweat_lactate": 2.2,
        "respiratory_rate": 21.0
    }
    healthy_sim, sepsis_sim = hdc.classify_state(hdc.encode_patient_state(early_vitals))
    risk, sirs, qsofa, _, details = detector.evaluate_patient(
        early_vitals,
        sepsis_sim,
        history,
        healthy_sim,
        {"hrv": "DECLINING", "temp": "RISING"}
    )
    assert risk in {"MEDIUM", "HIGH"}, f"Early deterioration should escalate risk; got {risk}"
    assert details["sepsis_probability"] >= 0.38, "Early deterioration probability should be elevated"
    print("      Early deterioration escalation: PASSED")

    shock_vitals = {
        "heart_rate": 126.0,
        "hrv": 12.0,
        "spo2": 88.0,
        "skin_temp": 39.1,
        "motion": 1.0,
        "systolic_bp": 78.0,
        "diastolic_bp": 46.0,
        "map": 55.3,
        "sweat_lactate": 5.1,
        "respiratory_rate": 28.0
    }
    healthy_sim, sepsis_sim = hdc.classify_state(hdc.encode_patient_state(shock_vitals))
    risk, sirs, qsofa, rules, details = detector.evaluate_patient(
        shock_vitals,
        sepsis_sim,
        history,
        healthy_sim,
        {"hrv": "STEEP_DECLINE", "temp": "FEVER"}
    )
    assert risk == "HIGH", f"Septic shock physiology should be HIGH risk; got {risk}"
    assert qsofa >= 2, "Shock physiology should trigger qSOFA organ dysfunction"
    assert details["shock_signal"], "Shock signal should be true for low MAP plus elevated lactate"
    assert any("Organ dysfunction" in rule for rule in rules), "Organ dysfunction rationale should be emitted"
    print("      Septic shock detection: PASSED")
    print("      Neurosymbolic detection tests: PASSED")

if __name__ == "__main__":
    print("==================================================")
    print("HYPERSEPSISTWIN-Ω: AUTOMATED BACKEND VERIFIER")
    print("==================================================")
    try:
        test_hdc_engine()
        print("-" * 50)
        test_digital_twin_features()
        print("-" * 50)
        test_z3_verification()
        print("-" * 50)
        test_evolutionary_optimization()
        print("-" * 50)
        test_neurosymbolic_detection()
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
