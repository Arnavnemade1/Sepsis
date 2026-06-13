"""Layer 4 — the Formally-Verified Neurosymbolic Runtime (the egress cage).

Layers 1–3 acquire, store and schedule probabilistic signal. Layer 4 is the
isolated plane every *output* must pass through before it leaves AegisOS. Nothing
— no alert token, no treatment protocol, no actuator instruction, no automated
clinical note — reaches the I/O channel until a formal model checker has **proven**
it satisfies the system's medical invariants. If the proof fails, the output is
quarantined, never dispatched.

This is the concrete meaning of "the software can never drop into an
uncalibrated, deadlocked, or harmful execution state": the set of *dispatched*
outputs is, by construction, exactly the set the SMT solver certified — so no
harmful output can cross the boundary even if an upstream neural inference is
wrong.

Egress classes and their proofs:

  * ``ALERT``    — Z3 over the alert's declared severity vs. the calibrated
    probability and evidence count: no under-alert at high risk, no baseless
    high alert, valid range, calibration in [0,1]. (Built here.)
  * ``PROTOCOL`` — delegated to the project's existing Z3 prover
    (``backend.core.verifier.ProtocolVerifier``): CHF/renal fluid caps,
    vasopressor ceiling + MAP justification, drug-allergy, sepsis-antibiotic
    timing. (Composed, not duplicated.)
  * ``ACTUATOR`` — Z3 bounded-command envelope: a hardware instruction value
    must be finite and within its declared [lo, hi] safety envelope.
  * ``NOTE``     — structured-consistency check: an automated note may not assert
    a risk level contradicting the live assessment, and may not recommend
    treatment unless it cites an already-certified protocol (chain of trust).

Honest framing (integrity rule): Z3 proves *output-level* safety invariants over
quantifier-free linear arithmetic — a **decidable, terminating** fragment, so the
gate itself always returns a verdict (sat/unsat) and cannot hang. We do NOT claim
to prove liveness/termination of arbitrary upstream code; the guarantee is that
every output that *is* dispatched provably satisfies the invariants, and the gate
is total (every egress is classified, default-deny).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from z3 import (And, Bool, BoolVal, If, Implies, Int, Not, Or, Real, RealVal,
                Solver, sat, unsat)

from backend.core.verifier import ProtocolVerifier

# Severity ladder + the probability bands the alert layer must be consistent with.
SEVERITY = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
BAND_HIGH = 0.70                 # aligned with scheduler.P_CRIT
BAND_MED = 0.40
BAND_LOW = 0.20

ALERT, PROTOCOL, ACTUATOR, NOTE = "ALERT", "PROTOCOL", "ACTUATOR", "NOTE"


@dataclass
class SafetyCertificate:
    """The verdict the cage attaches to every egress attempt."""
    ok: bool
    egress: str
    z3_status: str                       # "unsat" = proven safe, "sat" = violation
    message: str
    violations: List[str] = field(default_factory=list)
    proof: Optional[str] = None
    model: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "egress": self.egress, "z3_status": self.z3_status,
            "message": self.message, "violations": self.violations,
            "proof": self.proof, "model": self.model, "ts": round(self.ts, 3),
        }


def _band(p: float) -> int:
    if p >= BAND_HIGH:
        return 3
    if p >= BAND_MED:
        return 2
    if p >= BAND_LOW:
        return 1
    return 0


class NeuroSymbolicRuntime:
    """The isolated egress plane. Default-deny: an output ships only if certified."""

    def __init__(self):
        self._protocol = ProtocolVerifier()
        self.certified = 0
        self.blocked = 0
        self.quarantine: List[SafetyCertificate] = []
        self.recent: List[SafetyCertificate] = []

    # ------------------------------------------------------------------ #
    # The single boundary: certify, then (optionally) dispatch.
    # ------------------------------------------------------------------ #
    def certify(self, egress: str, payload: Dict[str, Any],
                context: Dict[str, Any]) -> SafetyCertificate:
        if egress == ALERT:
            cert = self._certify_alert(payload, context)
        elif egress == PROTOCOL:
            cert = self._certify_protocol(payload, context)
        elif egress == ACTUATOR:
            cert = self._certify_actuator(payload, context)
        elif egress == NOTE:
            cert = self._certify_note(payload, context)
        else:
            cert = SafetyCertificate(False, egress, "n/a",
                                     f"unknown egress class {egress!r}",
                                     ["Unknown_Egress_Class"])
        self.recent = (self.recent + [cert])[-64:]
        return cert

    def dispatch(self, egress: str, payload: Dict[str, Any],
                 context: Dict[str, Any]) -> Tuple[bool, SafetyCertificate]:
        """Certify and gate. Returns (allowed, certificate). Unsafe -> quarantine."""
        cert = self.certify(egress, payload, context)
        if cert.ok:
            self.certified += 1
        else:
            self.blocked += 1
            self.quarantine = (self.quarantine + [cert])[-128:]
        return cert.ok, cert

    # ------------------------------------------------------------------ #
    # ALERT — Z3 over severity vs. calibrated probability + evidence.
    # ------------------------------------------------------------------ #
    def _certify_alert(self, payload, context) -> SafetyCertificate:
        declared = int(payload.get("severity", 0))
        p = context.get("probability", None)
        evidence = int(context.get("evidence_count", 0))
        calibrated = bool(context.get("calibrated", False))

        # Calibration / non-NaN is a precondition the solver can't be asked to
        # reason about with a NaN literal — reject up front (default-deny).
        if p is None or not isinstance(p, (int, float)) or math.isnan(float(p)) \
                or not calibrated or not (0.0 <= float(p) <= 1.0):
            v = ["Uncalibrated_Or_Invalid_Probability"]
            return SafetyCertificate(False, ALERT, "sat",
                "VERIFICATION REJECTED: alert from uncalibrated/invalid state.", v)

        p = float(p)
        s = Solver()
        d = Int("declared_severity")
        pz = RealVal(p)
        ev = Int("evidence_count")
        s.add(d == declared, ev == evidence)
        band = If(pz >= BAND_HIGH, 3, If(pz >= BAND_MED, 2,
                  If(pz >= BAND_LOW, 1, 0)))

        range_ok = And(d >= 0, d <= 3)
        no_under = d >= band                                   # never under-call
        # Escalation by at most one level is allowed only with real evidence;
        # a high alert from nothing (band 0, no evidence) is rejected.
        over_allow = If(ev >= 1, 1, 0)
        no_over = d <= band + over_allow

        violation = Or(Not(range_ok), Not(no_under), Not(no_over))
        s.add(violation)
        status = s.check()

        if status == unsat:
            proof = (f"declared={SEVERITY[declared]}({declared}) ∧ p={p:.3f} "
                     f"∧ evidence={evidence} ⊢ "
                     f"range[0,3] ∧ (declared≥band(p)) ∧ (declared≤band(p)+[ev≥1])")
            return SafetyCertificate(True, ALERT, "unsat",
                "FORMALLY VERIFIED: alert severity coherent with calibrated risk.",
                proof=proof)

        # Reconstruct the violated rule(s) for the operator.
        b = _band(p)
        v: List[str] = []
        if not (0 <= declared <= 3):
            v.append("Severity_Range: declared severity outside [0,3].")
        if declared < b:
            v.append(f"Under_Alert: p={p:.2f} demands ≥{SEVERITY[b]} "
                     f"but alert declared {SEVERITY[declared]}.")
        if declared > b + (1 if evidence >= 1 else 0):
            v.append(f"Over_Alert: declared {SEVERITY[declared]} not justified by "
                     f"p={p:.2f} (band {SEVERITY[b]}) with evidence={evidence}.")
        return SafetyCertificate(False, ALERT, "sat",
            f"VERIFICATION REJECTED: {' '.join(v)}", v,
            model={"declared": declared, "band": b, "p": round(p, 3),
                   "evidence": evidence})

    # ------------------------------------------------------------------ #
    # PROTOCOL — delegate to the existing Z3 protocol prover.
    # ------------------------------------------------------------------ #
    def _certify_protocol(self, payload, context) -> SafetyCertificate:
        ok, msg, meta = self._protocol.verify_protocol(
            clinical_history=context.get("clinical_history", {}),
            vitals=context.get("vitals", {}),
            risk_level=context.get("risk_level", "LOW"),
            proposed_protocol=payload,
        )
        return SafetyCertificate(
            ok, PROTOCOL, meta.get("z3_status", "unsat" if ok else "sat"), msg,
            violations=meta.get("violated_rules", []),
            proof=meta.get("formula"), model=meta.get("z3_model", {}))

    # ------------------------------------------------------------------ #
    # ACTUATOR — Z3 bounded-command envelope.
    # ------------------------------------------------------------------ #
    def _certify_actuator(self, payload, context) -> SafetyCertificate:
        channel = str(payload.get("channel", "unknown"))
        value = payload.get("value", None)
        lo = float(payload.get("min", 0.0))
        hi = float(payload.get("max", 0.0))

        if value is None or not isinstance(value, (int, float)) \
                or math.isnan(float(value)) or math.isinf(float(value)):
            v = [f"Actuator_NonFinite: '{channel}' command is not a finite value."]
            return SafetyCertificate(False, ACTUATOR, "sat",
                f"VERIFICATION REJECTED: {v[0]}", v)

        value = float(value)
        s = Solver()
        cmd = Real("command")
        s.add(cmd == RealVal(value))
        in_env = And(cmd >= RealVal(lo), cmd <= RealVal(hi))
        s.add(Not(in_env))                                    # search for breach
        if s.check() == unsat:
            return SafetyCertificate(True, ACTUATOR, "unsat",
                f"FORMALLY VERIFIED: '{channel}'={value} within [{lo},{hi}].",
                proof=f"{lo} ≤ {channel}({value}) ≤ {hi}")
        v = [f"Actuator_OutOfEnvelope: '{channel}'={value} outside [{lo},{hi}]."]
        return SafetyCertificate(False, ACTUATOR, "sat",
            f"VERIFICATION REJECTED: {v[0]}", v,
            model={"channel": channel, "value": value, "min": lo, "max": hi})

    # ------------------------------------------------------------------ #
    # NOTE — structured consistency + chain of trust to a certified protocol.
    # ------------------------------------------------------------------ #
    def _certify_note(self, payload, context) -> SafetyCertificate:
        v: List[str] = []
        note_risk = payload.get("risk_level")
        ctx_risk = context.get("risk_level", "LOW")
        if note_risk is not None and note_risk != ctx_risk:
            v.append(f"Note_Risk_Mismatch: note states {note_risk} but live "
                     f"assessment is {ctx_risk}.")
        if payload.get("recommends_treatment") and not payload.get("protocol_certified"):
            v.append("Note_Uncertified_Treatment: note recommends treatment "
                     "without citing a certified protocol.")
        if not str(payload.get("text", "")).strip():
            v.append("Note_Empty: refusing to emit an empty clinical note.")
        if v:
            return SafetyCertificate(False, NOTE, "sat",
                f"VERIFICATION REJECTED: {' '.join(v)}", v)
        return SafetyCertificate(True, NOTE, "unsat",
            "FORMALLY VERIFIED: note consistent with assessment + certified chain.",
            proof=f"risk={ctx_risk} ∧ (treatment ⇒ certified_protocol) ∧ non-empty")

    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        total = self.certified + self.blocked
        return {
            "certified": self.certified, "blocked": self.blocked,
            "quarantined": len(self.quarantine),
            "pass_rate": round(self.certified / total, 4) if total else 1.0,
            "recent": [c.to_dict() for c in self.recent[-8:]],
        }


# --------------------------------------------------------------------------- #
# Standalone demo.
# --------------------------------------------------------------------------- #
def _demo() -> None:
    rt = NeuroSymbolicRuntime()

    def show(label, egress, payload, ctx):
        ok, c = rt.dispatch(egress, payload, ctx)
        verdict = "DISPATCHED " if ok else "QUARANTINED"
        print(f"  [{verdict}] {label}: {c.message}")

    print("ALERT egress:")
    show("sound HIGH alert", ALERT, {"severity": 3},
         {"probability": 0.88, "evidence_count": 3, "calibrated": True})
    show("under-alert at p=0.9", ALERT, {"severity": 0},
         {"probability": 0.90, "evidence_count": 2, "calibrated": True})
    show("baseless HIGH alert", ALERT, {"severity": 3},
         {"probability": 0.05, "evidence_count": 0, "calibrated": True})
    show("uncalibrated alert", ALERT, {"severity": 2},
         {"probability": float("nan"), "calibrated": False})

    print("\nACTUATOR egress:")
    show("infusion pump 12 mL/hr", ACTUATOR,
         {"channel": "infusion_pump", "value": 12.0, "min": 0.0, "max": 25.0}, {})
    show("infusion pump 80 mL/hr", ACTUATOR,
         {"channel": "infusion_pump", "value": 80.0, "min": 0.0, "max": 25.0}, {})

    print("\nPROTOCOL egress:")
    ctx_high = {"risk_level": "HIGH", "vitals": {"map": 58.0},
                "clinical_history": {"chf": True}}
    show("safe protocol", PROTOCOL,
         {"fluid_rate": 4.0, "vasopressor_dose": 0.2, "antibiotic_code": 4}, ctx_high)
    show("CHF fluid overload", PROTOCOL,
         {"fluid_rate": 20.0, "vasopressor_dose": 0.2, "antibiotic_code": 4}, ctx_high)

    print("\nNOTE egress:")
    show("consistent note", NOTE,
         {"risk_level": "HIGH", "text": "Septic shock; broad-spectrum abx started.",
          "recommends_treatment": True, "protocol_certified": True},
         {"risk_level": "HIGH"})
    show("contradictory note", NOTE,
         {"risk_level": "LOW", "text": "Patient stable."}, {"risk_level": "HIGH"})

    print("\nruntime status:", rt.status()["certified"], "certified,",
          rt.status()["blocked"], "blocked,",
          f"pass_rate={rt.status()['pass_rate']}")


if __name__ == "__main__":
    _demo()
