"""Verification for the Hour-1 Sepsis Bundle engine (Module C payload).

Proves the bundle is patient-specific AND formally gated:

  1. A critical bed yields all 5 Hour-1 items, an antibiotic, and every output
     (alert / protocol / note) certified UNSAT by the Z3 cage.
  2. Comorbidity caps hold and are PROVEN: CHF -> fluid ≤ 5, renal -> fluid ≤ 4.
  3. Penicillin allergy -> Cefepime (never Pip-Tazo).
  4. Vasopressor logic: engaged below MAP 65, withheld at/above 65.

    .venv/bin/python -m backend.aegis.kernel.verify_bundle
"""
from __future__ import annotations

from backend.aegis.kernel.bundle import SepsisBundleEngine
from backend.aegis.kernel.fusion import BedProfile


def _ok(msg): print(f"      [PASS] {msg}")


def _profile(map_v=58.0, lact=4.6, **over):
    en = {"temp_c": 39.3, "resp_rate": 30.0, "spo2": 89.0, "map": map_v,
          "abp_var": 7.0, "wbc": 18.0, "lactate": lact, "lactate_clearance": -120.0}
    en.update(over)
    return BedProfile(
        bed_id="node_0", name="Test, P.", fused_risk=0.95,
        edge={"hr": 126.0, "hrv_ms": 6.0, "perfusion": 0.6, "vasomotion": 0.05},
        enterprise=en,
        components={"cardiac": 0.9, "inflammatory": 1.1, "perfusion": 1.0, "metabolic": 1.4},
        drivers=["Lactate 4.6 mmol/L (severe)", "Hypotension MAP 58 mmHg", "Fever 39.3°C"],
        primary_system="metabolic")


def test_critical_certified() -> None:
    print("[1] critical bed -> full bundle, all outputs Z3-certified")
    b = SepsisBundleEngine().generate(_profile(), {})
    assert len(b["checklist"]) == 5
    assert b["protocol"]["antibiotic"] != "None"
    assert b["alert"]["z3"] == "unsat" and b["protocol"]["z3"] == "unsat" and b["note"]["z3"] == "unsat"
    assert b["all_verified"] is True
    _ok(f"5 items; {b['protocol']['antibiotic']}; all UNSAT; all_verified")


def test_comorbidity_caps() -> None:
    print("[2] comorbidity fluid caps hold and are proven")
    eng = SepsisBundleEngine()
    chf = eng.generate(_profile(), {"chf": True})
    renal = eng.generate(_profile(), {"renal_failure": True})
    plain = eng.generate(_profile(), {})
    assert chf["protocol"]["fluid_rate"] <= 5.0 and chf["protocol"]["certified"]
    assert renal["protocol"]["fluid_rate"] <= 4.0 and renal["protocol"]["certified"]
    assert plain["protocol"]["fluid_rate"] == 10.0
    _ok(f"CHF≤5 ({chf['protocol']['fluid_rate']}), renal≤4 "
        f"({renal['protocol']['fluid_rate']}), default 10 — all certified")


def test_allergy_substitution() -> None:
    print("[3] penicillin allergy -> Cefepime, never Pip-Tazo")
    b = SepsisBundleEngine().generate(_profile(), {"penicillin_allergy": True})
    assert b["protocol"]["antibiotic"] == "Cefepime"
    assert b["protocol"]["certified"]
    _ok("allergic patient prescribed Cefepime (certified)")


def test_vasopressor_logic() -> None:
    print("[4] vasopressors engage below MAP 65, withheld at/above")
    eng = SepsisBundleEngine()
    low = eng.generate(_profile(map_v=52.0), {})
    ok = eng.generate(_profile(map_v=78.0, lact=1.2), {})
    assert low["protocol"]["vasopressor_dose"] > 0
    assert ok["protocol"]["vasopressor_dose"] == 0.0
    v5_low = [i for i in low["checklist"] if i["n"] == 5][0]
    v5_ok = [i for i in ok["checklist"] if i["n"] == 5][0]
    assert v5_low["status"] == "indicated" and v5_ok["status"] == "not_indicated"
    _ok(f"MAP 52 -> norepi {low['protocol']['vasopressor_dose']}; MAP 78 -> withheld")


def main() -> None:
    print("=" * 64)
    print("AegisOS — Hour-1 Sepsis Bundle engine verification")
    print("=" * 64)
    test_critical_certified()
    test_comorbidity_caps()
    test_allergy_substitution()
    test_vasopressor_logic()
    print("-" * 64)
    print("ALL BUNDLE CHECKS PASSED")


if __name__ == "__main__":
    main()
