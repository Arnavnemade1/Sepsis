"""Verification suite for Layer 4 — the Formally-Verified Neurosymbolic Runtime.

Proves the egress cage actually gates on formal proofs (default-deny, total):

  1. ALERT: a severity coherent with the calibrated probability is certified;
     an under-alert (quiet at high risk) and a baseless high alert are blocked.
  2. ALERT: an uncalibrated / NaN-probability output is rejected up front — the
     "never dispatch from an uncalibrated state" invariant.
  3. ACTUATOR: a command inside its envelope is certified; out-of-envelope and
     non-finite commands are blocked.
  4. PROTOCOL: composes the existing Z3 prover — safe protocol passes, a CHF
     fluid-overload protocol is rejected.
  5. NOTE: a consistent note with a certified-protocol citation passes; a note
     contradicting the live risk, or recommending treatment with no certified
     protocol, is blocked.
  6. Gate is total + default-deny: an unknown egress class is blocked, and the
     dispatched/blocked counters + quarantine reflect exactly the verdicts.

    .venv/bin/python -m backend.aegis.kernel.verify_neurosym
"""
from __future__ import annotations

from backend.aegis.kernel.neurosym import (
    ACTUATOR, ALERT, NOTE, PROTOCOL, NeuroSymbolicRuntime,
)


def _ok(msg): print(f"      [PASS] {msg}")


def test_alert_coherence() -> None:
    print("[1] ALERT — severity must cohere with calibrated risk")
    rt = NeuroSymbolicRuntime()
    good, _ = rt.dispatch(ALERT, {"severity": 3},
                          {"probability": 0.85, "evidence_count": 3, "calibrated": True})
    under, cu = rt.dispatch(ALERT, {"severity": 0},
                            {"probability": 0.90, "evidence_count": 2, "calibrated": True})
    over, co = rt.dispatch(ALERT, {"severity": 3},
                           {"probability": 0.05, "evidence_count": 0, "calibrated": True})
    assert good and not under and not over
    assert any("Under_Alert" in v for v in cu.violations)
    assert any("Over_Alert" in v for v in co.violations)
    _ok("coherent HIGH certified; under-alert + baseless-HIGH both blocked")


def test_alert_uncalibrated() -> None:
    print("[2] ALERT — uncalibrated/NaN state is default-denied")
    rt = NeuroSymbolicRuntime()
    nan_ok, c1 = rt.dispatch(ALERT, {"severity": 2},
                             {"probability": float("nan"), "calibrated": False})
    oob_ok, c2 = rt.dispatch(ALERT, {"severity": 2},
                             {"probability": 1.7, "calibrated": True})
    miss_ok, c3 = rt.dispatch(ALERT, {"severity": 1}, {"calibrated": True})  # no prob
    assert not nan_ok and not oob_ok and not miss_ok
    assert all("Uncalibrated_Or_Invalid_Probability" in c.violations
               for c in (c1, c2, c3))
    _ok("NaN, out-of-range, and missing probability all rejected up front")


def test_actuator_envelope() -> None:
    print("[3] ACTUATOR — bounded-command envelope (Z3)")
    rt = NeuroSymbolicRuntime()
    inside, _ = rt.dispatch(ACTUATOR,
        {"channel": "infusion_pump", "value": 12.0, "min": 0.0, "max": 25.0}, {})
    edge, _ = rt.dispatch(ACTUATOR,
        {"channel": "infusion_pump", "value": 25.0, "min": 0.0, "max": 25.0}, {})
    over, co = rt.dispatch(ACTUATOR,
        {"channel": "infusion_pump", "value": 80.0, "min": 0.0, "max": 25.0}, {})
    inf, ci = rt.dispatch(ACTUATOR,
        {"channel": "o2", "value": float("inf"), "min": 0.0, "max": 100.0}, {})
    assert inside and edge and not over and not inf
    assert any("OutOfEnvelope" in v for v in co.violations)
    assert any("NonFinite" in v for v in ci.violations)
    _ok("in-envelope + boundary certified; over-range + non-finite blocked")


def test_protocol_delegation() -> None:
    print("[4] PROTOCOL — composes the existing Z3 protocol prover")
    rt = NeuroSymbolicRuntime()
    ctx = {"risk_level": "HIGH", "vitals": {"map": 58.0},
           "clinical_history": {"chf": True}}
    safe, _ = rt.dispatch(PROTOCOL,
        {"fluid_rate": 4.0, "vasopressor_dose": 0.2, "antibiotic_code": 4}, ctx)
    chf, cc = rt.dispatch(PROTOCOL,
        {"fluid_rate": 20.0, "vasopressor_dose": 0.2, "antibiotic_code": 4}, ctx)
    assert safe and not chf
    assert any("CHF" in v for v in cc.violations)
    _ok("safe protocol certified; CHF fluid-overload rejected via Z3 prover")


def test_note_chain_of_trust() -> None:
    print("[5] NOTE — consistency + certified-protocol chain of trust")
    rt = NeuroSymbolicRuntime()
    good, _ = rt.dispatch(NOTE,
        {"risk_level": "HIGH", "text": "Septic shock; abx started.",
         "recommends_treatment": True, "protocol_certified": True},
        {"risk_level": "HIGH"})
    mism, cm = rt.dispatch(NOTE,
        {"risk_level": "LOW", "text": "Stable."}, {"risk_level": "HIGH"})
    unc, cuq = rt.dispatch(NOTE,
        {"risk_level": "HIGH", "text": "Start vasopressors.",
         "recommends_treatment": True}, {"risk_level": "HIGH"})
    assert good and not mism and not unc
    assert any("Risk_Mismatch" in v for v in cm.violations)
    assert any("Uncertified_Treatment" in v for v in cuq.violations)
    _ok("consistent+cited note certified; contradiction + uncited treatment blocked")


def test_gate_total_default_deny() -> None:
    print("[6] gate is total + default-deny; counters reconcile")
    rt = NeuroSymbolicRuntime()
    unk, cu = rt.dispatch("TELEPORT", {}, {})
    assert not unk and "Unknown_Egress_Class" in cu.violations
    # one certified + one blocked
    rt.dispatch(ACTUATOR, {"channel": "x", "value": 1.0, "min": 0, "max": 10}, {})
    rt.dispatch(ACTUATOR, {"channel": "x", "value": 99.0, "min": 0, "max": 10}, {})
    st = rt.status()
    assert st["certified"] == 1 and st["blocked"] == 2
    assert st["quarantined"] == 2
    _ok(f"unknown class denied; counters: {st['certified']} certified / "
        f"{st['blocked']} blocked / {st['quarantined']} quarantined")


def main() -> None:
    print("=" * 64)
    print("AegisOS Layer 4 — Neurosymbolic Egress Cage verification")
    print("=" * 64)
    test_alert_coherence()
    test_alert_uncalibrated()
    test_actuator_envelope()
    test_protocol_delegation()
    test_note_chain_of_trust()
    test_gate_total_default_deny()
    print("-" * 64)
    print("ALL LAYER-4 CHECKS PASSED")


if __name__ == "__main__":
    main()
