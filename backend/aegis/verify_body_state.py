"""Verification of the multi-organ dysfunction engine (body_state.py).

Proves the clinical scoring against known SOFA/qSOFA cases and exercises the MODS
+ trajectory logic. No network, deterministic.

    .venv/bin/python -m backend.aegis.verify_body_state
"""
from __future__ import annotations

import sys

from backend.aegis.body_state import (
    OrganFailureTracker, build_body_state,
    sofa_cardiovascular, sofa_cns, sofa_coagulation, sofa_liver,
    sofa_renal, sofa_respiration,
)


def check_sofa_subscores() -> None:
    print("[1] SOFA sub-scores match the published thresholds")
    # (fn, input, expected points)
    cases = [
        ("respiration", sofa_respiration, 98, 0), ("respiration", sofa_respiration, 93, 1),
        ("respiration", sofa_respiration, 88, 2),
        ("coagulation", sofa_coagulation, 200, 0), ("coagulation", sofa_coagulation, 120, 1),
        ("coagulation", sofa_coagulation, 80, 2), ("coagulation", sofa_coagulation, 40, 3),
        ("coagulation", sofa_coagulation, 15, 4),
        ("liver", sofa_liver, 0.8, 0), ("liver", sofa_liver, 1.5, 1), ("liver", sofa_liver, 4.0, 2),
        ("liver", sofa_liver, 8.0, 3), ("liver", sofa_liver, 15.0, 4),
        ("cns", sofa_cns, 15, 0), ("cns", sofa_cns, 13, 1), ("cns", sofa_cns, 11, 2),
        ("cns", sofa_cns, 7, 3), ("cns", sofa_cns, 4, 4),
        ("renal", sofa_renal, 1.0, 0), ("renal", sofa_renal, 1.5, 1), ("renal", sofa_renal, 3.0, 2),
        ("renal", sofa_renal, 4.0, 3), ("renal", sofa_renal, 6.0, 4),
    ]
    for organ, fn, val, exp in cases:
        got = fn(val)
        assert got is not None and got[0] == exp, f"{organ}({val}) → {got}, expected {exp}"
    # cardiovascular: MAP + pressor
    assert sofa_cardiovascular(80, None)[0] == 0
    assert sofa_cardiovascular(60, None)[0] == 1
    assert sofa_cardiovascular(60, 0.05)[0] == 3
    assert sofa_cardiovascular(60, 0.3)[0] == 4
    # unmonitored → None
    assert sofa_renal(None) is None and sofa_respiration(None) is None
    print(f"      [PASS] {len(cases)+4} threshold cases + cardio pressor tiers + None-handling")


def check_honest_monitoring() -> None:
    print("[2] Honest monitoring — labs absent → UNMONITORED, never a false green")
    # wearable-only vitals: no creatinine/bilirubin/platelets
    bs = build_body_state(
        {"heart_rate": 88, "spo2": 97, "respiratory_rate": 16, "map": 80},
        {"ml_probability": 0.2}, {}, "LOW", 1000.0)
    by = {s.name: s for s in bs.organ_systems}
    for lab in ("Renal", "Hepatic", "Coagulation"):
        assert by[lab].monitored is False and by[lab].status == "UNMONITORED", f"{lab} not honest"
        assert by[lab].sofa_points is None
    assert by["Respiratory"].monitored and by["Cardiovascular"].monitored
    print("      [PASS] Renal/Hepatic/Coagulation = UNMONITORED (no labs); "
          "Respiratory/Cardiovascular scored")


def check_septic_case() -> None:
    print("[3] Full septic case — SOFA total, qSOFA, MODS, Sepsis-3 flag")
    # a deteriorating septic patient WITH labs + vitals
    vit = {"heart_rate": 122, "spo2": 90, "respiratory_rate": 28,
           "systolic_bp": 92, "map": 60, "skin_temp": 38.9, "sweat_lactate": 4.5}
    hist = {"gcs": 12, "creatinine": 2.4, "bilirubin": 3.0, "platelets": 90,
            "vasopressor_dose": 0.08}
    bs = build_body_state(vit, {"ml_probability": 0.83}, {}, "HIGH", 2000.0,
                          clinical_history=hist, infection_suspected=True)
    d = bs.to_dict()
    sofa, qsofa, mods, s3 = d["sofa"], d["qsofa"], d["mods"], d["sepsis3"]
    # expected SOFA: resp2 + cardio3(pressor low) + cns2 + renal2 + liver2 + coag2 = 13
    print(f"      SOFA total={sofa['total']} per_organ={sofa['per_organ']} "
          f"band={sofa['mortality_band']}")
    assert sofa["total"] == 13, sofa
    assert sofa["organs_scored"] == 6
    assert qsofa["score"] == 3 and qsofa["high_risk"], qsofa
    assert mods["is_mods"] and mods["count"] >= 2, mods
    assert s3["flag"] is True, s3
    print(f"      [PASS] SOFA 13 (mortality {sofa['mortality_band']}), qSOFA 3/3, "
          f"MODS {mods['count']} organs (dominant {mods['dominant']}), Sepsis-3 flag set")


def check_trajectory() -> None:
    print("[4] Trajectory tracker — a worsening organ is flagged RISING with an ETA")
    tr = OrganFailureTracker(window=8, horizon_min=15.0)
    last = None
    # simulate MAP falling 82 → 52 over 10 min (cardio worsening), labs steady-bad
    for i in range(10):
        t = 3000.0 + i * 60.0                # 1-min steps
        mapv = 82 - i * 3.2
        vit = {"map": mapv, "spo2": 96, "respiratory_rate": 18, "sweat_lactate": 2.0 + i * 0.25}
        hist = {"gcs": 15}
        bs = build_body_state(vit, {"ml_probability": 0.5}, {}, "MEDIUM", t,
                              clinical_history=hist)
        last = tr.update(bs)
    cardio = next(s for s in last.organ_systems if s.name == "Cardiovascular")
    print(f"      cardio: sev={cardio.severity} trend={cardio.trend} "
          f"slope/min={cardio.slope_per_min} eta_crit={cardio.eta_critical_min}min "
          f"proj={cardio.projected_severity}")
    assert cardio.trend == "RISING", cardio.trend
    assert cardio.slope_per_min > 0
    assert cardio.projected_severity >= cardio.severity
    print("      [PASS] worsening cardiovascular flagged RISING with a forward projection")


def check_mods_onset() -> None:
    print("[5] MODS-onset edge detection — fires exactly when 2nd organ starts failing")
    tr = OrganFailureTracker()
    onsets = []
    # step 0: only cardio failing (MAP low + lactate → sev≥0.45); step 1: renal
    # ALSO fails (creatinine 4.0 → SOFA 3) → 2 organs → MODS onset fires here.
    base = {"map": 58, "spo2": 97, "respiratory_rate": 16, "sweat_lactate": 4.0}
    seqs = [
        (base, {"gcs": 15}),                            # 1 failing organ (cardio)
        (base, {"gcs": 15, "creatinine": 4.0}),         # +renal → 2 organs → MODS
        (base, {"gcs": 15, "creatinine": 4.0}),         # steady → no new onset
    ]
    for i, (vit, hist) in enumerate(seqs):
        bs = build_body_state(vit, {"ml_probability": 0.6}, {}, "HIGH", 4000.0 + i * 60,
                              clinical_history=hist)
        bs = tr.update(bs)
        onsets.append(bs.mods["onset_now"])
    assert onsets == [False, True, False], f"onset sequence wrong: {onsets}"
    assert tr.mods_onset_ts == 4000.0 + 60, tr.mods_onset_ts
    print(f"      [PASS] MODS onset edge fired once, at the 2nd-organ transition "
          f"(not before, not repeated): {onsets}")


def run() -> int:
    print("=" * 66)
    print("AegisOS — multi-organ dysfunction engine verification")
    print("=" * 66)
    check_sofa_subscores()
    check_honest_monitoring()
    check_septic_case()
    check_trajectory()
    check_mods_onset()
    print("-" * 66)
    print("ALL ORGAN-FAILURE ENGINE CHECKS PASSED ✓  (SOFA · qSOFA · MODS · trajectory)")
    print("=" * 66)
    return 0


def main() -> None:
    run()


if __name__ == "__main__":
    sys.exit(run())
