"""Verify the per-patient tracking terminal (offline, deterministic, no LLM spend).

    .venv/bin/python -m backend.aegis.verify_patient_terminal
"""
from __future__ import annotations

import sys

from backend.aegis.patient_terminal import (
    PatientTerminal, TrackingProfile, _keyword_directive, _sanitize, resolve_metric,
    resolve_organ,
)


def check_vocab() -> None:
    print("[1] Vocabulary resolution (aliases → canonical)")
    assert resolve_metric("lactate") == "sweat_lactate"
    assert resolve_metric("hr") == "heart_rate" and resolve_metric("o2") == "spo2"
    assert resolve_metric("bp") == "systolic_bp" and resolve_metric("temp") == "skin_temp"
    assert resolve_metric("nonsense") is None
    assert resolve_organ("kidneys") == "Renal" and resolve_organ("heart") == "Cardiovascular"
    assert resolve_organ("lungs") == "Respiratory" and resolve_organ("wbc") == "Immune"
    assert resolve_organ("spleen") is None
    print("      [PASS] metric + organ aliases resolve; unknowns → None")


def check_structured_commands() -> None:
    print("[2] Structured commands per patient (independent profiles)")
    pt = PatientTerminal()
    pt.exec("bed_1", "focus kidneys")
    pt.exec("bed_1", "watch lactate > 3")
    pt.exec("bed_1", "watch map < 60 critical")
    r = pt.exec("bed_1", "status")
    p1 = pt.profile("bed_1").to_dict()
    assert p1["foci"] == ["Renal"], p1["foci"]
    assert len(p1["watches"]) == 2
    assert any(w["severity"] == "critical" and w["metric"] == "map" for w in p1["watches"])
    # a different bed is untouched
    assert pt.profile("bed_2").to_dict()["foci"] == []
    print(f"      [PASS] bed_1 → focus {p1['foci']}, {len(p1['watches'])} watches; bed_2 independent")


def check_watch_evaluation() -> None:
    print("[3] Watch evaluation is edge-triggered (fires once per crossing)")
    pt = PatientTerminal()
    pt.exec("b", "watch lactate > 3")
    assert pt.evaluate("b", {"sweat_lactate": 2.0}) == []          # below → nothing
    fired = pt.evaluate("b", {"sweat_lactate": 4.1})               # crosses → fires
    assert len(fired) == 1 and fired[0]["metric"] == "sweat_lactate" and fired[0]["value"] == 4.1
    assert pt.evaluate("b", {"sweat_lactate": 4.5}) == []          # still high → no re-fire
    assert pt.evaluate("b", {"sweat_lactate": 2.0}) == []          # normalises (re-arm)
    assert len(pt.evaluate("b", {"sweat_lactate": 5.0})) == 1       # crosses again → fires
    print("      [PASS] fires once on crossing, re-arms after it normalises")


def check_sanitize_boundary() -> None:
    print("[4] Sanitize is the trust boundary — drops anything off-vocabulary / unsafe")
    dirty = {
        "focus": ["kidneys", "pancreas", "toenail"],       # only Renal is valid
        "watches": [
            {"metric": "lactate", "op": ">", "threshold": 3},        # ok
            {"metric": "unicorns", "op": ">", "threshold": 1},       # bad metric
            {"metric": "map", "op": "DROP TABLE", "threshold": 5},   # bad op
            {"metric": "hr", "op": ">", "threshold": "fast"},        # bad threshold
        ],
        "reply": "ordered 2g ceftriaxone IV now",     # a treatment order — must NOT act on it
        "give_drug": "ceftriaxone",                    # unknown field — ignored
    }
    s = _sanitize(dirty)
    assert s["focus"] == ["Renal"], s["focus"]
    assert len(s["watches"]) == 1 and s["watches"][0]["metric"] == "sweat_lactate"
    assert "give_drug" not in s
    # applying it changes only tracking, never places an order (no order concept exists here)
    prof = TrackingProfile(bed_id="x")
    pt = PatientTerminal()
    applied = pt.apply(prof, s)
    assert prof.foci == ["Renal"] and len(prof.watches) == 1
    print(f"      [PASS] kept only Renal + 1 valid watch; dropped bad metric/op/threshold + "
          f"unknown fields; no order path exists")


def check_keyword_fallback() -> None:
    print("[5] Keyword fallback parses NL when the LLM is offline")
    d = _keyword_directive("keep an eye on her kidneys and alert if lactate goes above 3")
    assert d is not None and "Renal" in d["focus"]
    assert any(w["metric"] == "sweat_lactate" and w["op"] == ">" and w["threshold"] == 3.0
               for w in d["watches"]), d["watches"]
    d2 = _keyword_directive("watch the heart, tell me if MAP drops below 60")
    assert "Cardiovascular" in d2["focus"]
    assert any(w["metric"] == "map" and w["op"] == "<" and w["threshold"] == 60.0 for w in d2["watches"])
    assert _keyword_directive("how is the weather") is None
    print("      [PASS] NL → focus + watch without any LLM; irrelevant text → None")


def check_nl_exec_offline() -> None:
    print("[6] exec() routes NL through the parser when no LLM client is attached")
    pt = PatientTerminal(llm_client=None)
    out = pt.exec("bed_9", "watch her lungs and warn me if SpO2 under 92")
    assert out["source"] == "parser"
    prof = pt.profile("bed_9").to_dict()
    assert "Respiratory" in prof["foci"]
    assert any(w["metric"] == "spo2" and w["op"] == "<" for w in prof["watches"]), prof["watches"]
    print(f"      [PASS] plain-English line → {prof['foci']} + watch {[w['label'] for w in prof['watches']]}")


def run() -> int:
    print("=" * 66)
    print("AegisOS — per-patient tracking terminal verification")
    print("=" * 66)
    check_vocab()
    check_structured_commands()
    check_watch_evaluation()
    check_sanitize_boundary()
    check_keyword_fallback()
    check_nl_exec_offline()
    print("-" * 66)
    print("PER-PATIENT TERMINAL VERIFIED ✓  (foci · watches · edge-eval · safe LLM boundary)")
    print("=" * 66)
    return 0


def main() -> None:
    run()


if __name__ == "__main__":
    sys.exit(run())
