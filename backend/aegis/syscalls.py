"""The write-side syscall interface — how the console *drives* the OS.

Everything else the cockpit does is a **read**: ps, top, kstat, dmesg, who,
explain, radar all inspect kernel state. This module is the narrow, audited,
formally-gated boundary through which userspace (the command console) is allowed
to *mutate* the running system — order a treatment, escalate a bed's priority,
run a crisis drill, recover a bed.

The design mirrors a real OS trap gate, with three non-negotiable properties:

  1. **Enumerated.** There is a fixed, small syscall table. An unknown syscall
     traps to ``EINVAL``; there is no generic "run this" escape hatch.

  2. **Z3-safety-gated (default-deny).** A write that has an external clinical
     consequence (treat / escalate / crisis / clear) is first routed through the
     Layer-4 neurosymbolic egress cage (``NeuroSymbolicRuntime``). The mutation is
     applied *only if* the SMT solver returns a safety certificate. A denied
     syscall returns ``EPERM`` and changes nothing — e.g. you cannot escalate a
     stable patient to CRITICAL (baseless high alert), and you cannot order a
     fluid bolus that violates a CHF/renal cap. The formal gate, not operator
     discretion, decides what is allowed to happen.

  3. **Audited.** Every invocation — allowed or denied — is appended to a
     monotonic, append-only audit log carrying the syscall number, caller,
     arguments, the resulting Z3 status, and the certificate message.

     Two limits on that, stated because the word "audit" invites the assumption
     that neither applies. The log is an in-memory list capped at the last 256
     entries and is gone on restart, so it is a live view, not a system of
     record. And ``caller`` is whatever string the request body supplied — the
     gate authenticates nothing, so the log answers "what was ordered and did the
     prover agree", not "who ordered it". The persistent hash-chained journal and
     the authenticated caller are what turn this into a record.

The actual mutations are real: ``escalate`` writes a clinician priority *pin*
onto the cognitive scheduler (so certified escalations genuinely shift compute
across the cohort), ``treat`` records a certified, applied Hour-1 protocol, and
``crisis``/``clear`` drive the cluster's perturbation actuators. The console is
no longer just a window onto the kernel — it is a governed control surface.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.aegis.kernel.neurosym import (ACTUATOR, ALERT, NeuroSymbolicRuntime,
                                           PROTOCOL, SafetyCertificate)

# errno-style return codes (a denied write is a first-class, inspectable outcome).
EOK, EPERM, ENOENT, EINVAL = 0, 1, 2, 22
ERRNO_NAME = {EOK: "OK", EPERM: "EPERM", ENOENT: "ENOENT", EINVAL: "EINVAL"}

# The enumerated write syscall table. Anything not here traps to EINVAL.
SYSCALLS = ("treat", "escalate", "crisis", "clear")

# Severity ladder shared with the egress cage.
_SEV_NAME = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}


def _band(risk: float) -> int:
    if risk >= 0.70:
        return 3
    if risk >= 0.40:
        return 2
    if risk >= 0.20:
        return 1
    return 0


@dataclass
class SyscallResult:
    """The trap return value: errno + the safety certificate + any payload."""
    sid: int
    syscall: str
    bed: Optional[str]
    bed_name: Optional[str]
    errno: int
    allowed: bool
    z3_status: str
    message: str
    egress: Optional[str] = None
    violations: List[str] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sid": self.sid, "syscall": self.syscall, "bed": self.bed,
            "bed_name": self.bed_name, "errno": self.errno,
            "errno_name": ERRNO_NAME.get(self.errno, str(self.errno)),
            "allowed": self.allowed, "z3_status": self.z3_status,
            "egress": self.egress, "message": self.message,
            "violations": self.violations, "result": self.result,
            "ts": round(self.ts, 3),
        }


@dataclass
class Intervention:
    """The live write-state the OS now carries for a bed because of past syscalls."""
    escalated: bool = False
    pin: float = 1.0
    treated: bool = False
    protocol: Optional[Dict[str, Any]] = None
    last_sid: int = 0
    ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"escalated": self.escalated, "pin": round(self.pin, 2),
                "treated": self.treated, "protocol": self.protocol,
                "last_sid": self.last_sid, "ts": round(self.ts, 3)}


class SyscallTable:
    """The trap gate. Composes (does not duplicate) the cluster, the Z3 egress
    cage and the Hour-1 bundle engine; owns only the audit log + write-state."""

    def __init__(self, cluster, bundle_engine, cage: Optional[NeuroSymbolicRuntime] = None):
        self.cluster = cluster
        self.bundle = bundle_engine
        self.cage = cage or NeuroSymbolicRuntime()
        self._sid = 0
        self.audit: List[SyscallResult] = []
        self.interventions: Dict[str, Intervention] = {}

    # ------------------------------------------------------------------ #
    def _next_sid(self) -> int:
        self._sid += 1
        return self._sid

    def _record(self, res: SyscallResult) -> SyscallResult:
        self.audit = (self.audit + [res])[-256:]
        return res

    def _iv(self, bed: str) -> Intervention:
        return self.interventions.setdefault(bed, Intervention())

    def _profile(self, bed: str):
        return getattr(self.cluster, "profiles", {}).get(bed)

    # ------------------------------------------------------------------ #
    # The public trap entry point.
    # ------------------------------------------------------------------ #
    def invoke(self, syscall: str, bed: Optional[str] = None,
               args: Optional[Dict[str, Any]] = None,
               caller: str = "console") -> SyscallResult:
        args = args or {}
        sid = self._next_sid()
        name = (syscall or "").lower().strip()

        if name not in SYSCALLS:
            return self._record(SyscallResult(
                sid, name or "?", bed, None, EINVAL, False, "n/a",
                f"unknown syscall {syscall!r} — table: {', '.join(SYSCALLS)}"))

        prof = self._profile(bed) if bed else None
        if bed is None or prof is None:
            return self._record(SyscallResult(
                sid, name, bed, None, ENOENT, False, "n/a",
                f"no such bed {bed!r}"))

        handler = getattr(self, f"_sys_{name}")
        res = handler(sid, bed, prof, args, caller)
        return self._record(res)

    # ------------------------------------------------------------------ #
    # sys_treat — order the certified Hour-1 bundle (PROTOCOL egress gate).
    # ------------------------------------------------------------------ #
    def _sys_treat(self, sid, bed, prof, args, caller) -> SyscallResult:
        hist = getattr(self.cluster, "histories", {}).get(bed, {})
        base = self.bundle._derive_protocol(prof, hist)
        # An operator MAY override the order; the override is then subject to the
        # exact same formal gate — e.g. forcing a 30 mL/kg bolus on a CHF patient
        # is what the Z3 fluid cap exists to quarantine.
        proto = dict(base)
        for k in ("fluid_rate", "vasopressor_dose", "antibiotic_code"):
            if k in args and args[k] is not None:
                proto[k] = float(args[k]) if k != "antibiotic_code" else int(args[k])
        overridden = any(k in args for k in ("fluid_rate", "vasopressor_dose", "antibiotic_code"))

        risk = float(prof.fused_risk)
        risk_level = "HIGH" if risk >= 0.70 else "MEDIUM" if risk >= 0.40 else "LOW"
        ctx = {"risk_level": risk_level, "vitals": {"map": prof.enterprise.get("map", 75.0)},
               "clinical_history": hist}
        ok, cert = self.cage.dispatch(PROTOCOL, proto, ctx)

        bundle = None
        if ok:
            iv = self._iv(bed)
            iv.treated = True
            iv.protocol = {"fluid_rate": proto["fluid_rate"],
                           "vasopressor_dose": proto["vasopressor_dose"],
                           "antibiotic_code": int(proto["antibiotic_code"])}
            iv.last_sid, iv.ts = sid, time.time()
            # A bed under active treatment holds a monitoring priority floor so the
            # kernel keeps eyes on it through the response window.
            iv.pin = max(iv.pin, 1.3)
            self._apply_pin(bed, iv.pin)
            bundle = self.bundle.generate(prof, hist)
        return SyscallResult(
            sid, "treat", bed, prof.name, EOK if ok else EPERM, ok,
            cert.z3_status,
            ("PROTOCOL CERTIFIED — order dispatched"
             + (" (operator override)" if overridden else "")) if ok
            else f"PROTOCOL QUARANTINED — {cert.message}",
            egress=PROTOCOL, violations=cert.violations,
            result={"protocol": proto, "overridden": overridden,
                    "bundle": bundle, "proof": cert.proof})

    # ------------------------------------------------------------------ #
    # sys_escalate — raise priority (ALERT egress gate). A certified escalation
    # writes a real pin onto the scheduler; an unjustified one is denied.
    # ------------------------------------------------------------------ #
    def _sys_escalate(self, sid, bed, prof, args, caller) -> SyscallResult:
        risk = float(prof.fused_risk)
        band = _band(risk)
        # Target severity: explicit, else a one-step escalation above the band.
        target = args.get("level", args.get("severity"))
        target = int(target) if target is not None else min(3, band + 1)
        target = max(0, min(3, target))
        evidence = len(getattr(prof, "drivers", []) or [])

        ok, cert = self.cage.dispatch(ALERT, {"severity": target}, {
            "probability": risk, "calibrated": True, "evidence_count": evidence})

        pin = None
        if ok:
            iv = self._iv(bed)
            iv.escalated = target >= 2
            iv.pin = round(1.0 + 0.6 * target, 2)        # sev3 -> 2.8x compute weight
            iv.last_sid, iv.ts = sid, time.time()
            pin = self._apply_pin(bed, iv.pin)
        return SyscallResult(
            sid, "escalate", bed, prof.name, EOK if ok else EPERM, ok,
            cert.z3_status,
            (f"ESCALATED to {_SEV_NAME[target]} — pin {pin:.2f}× compute") if ok
            else f"ESCALATION DENIED — {cert.message}",
            egress=ALERT, violations=cert.violations,
            result={"target_severity": target, "band": band,
                    "evidence": evidence, "pin": pin, "proof": cert.proof})

    # ------------------------------------------------------------------ #
    # sys_crisis — privileged crisis drill (ACTUATOR envelope gate).
    # ------------------------------------------------------------------ #
    def _sys_crisis(self, sid, bed, prof, args, caller) -> SyscallResult:
        intensity = float(args.get("intensity", 1.0))
        ok, cert = self.cage.dispatch(ACTUATOR, {
            "channel": "crisis_drill", "value": intensity, "min": 0.0, "max": 1.0}, {})
        if ok:
            # Through the writer thread: this mutates the bed's driver bias, which
            # lands in the rings on the next tick (kernel/executor.py).
            from backend.aegis.kernel.executor import EXEC
            EXEC.call(self.cluster.inject_crisis, bed, intensity=intensity)
        return SyscallResult(
            sid, "crisis", bed, prof.name, EOK if ok else EINVAL, ok,
            cert.z3_status,
            f"CRISIS DRILL injected (intensity {intensity:.2f})" if ok
            else f"DRILL REJECTED — {cert.message}",
            egress=ACTUATOR, violations=cert.violations,
            result={"intensity": intensity})

    # ------------------------------------------------------------------ #
    # sys_clear — recover a bed: drop the pin + clear intervention state.
    # ------------------------------------------------------------------ #
    def _sys_clear(self, sid, bed, prof, args, caller) -> SyscallResult:
        ok, cert = self.cage.dispatch(ACTUATOR, {
            "channel": "recover", "value": 0.0, "min": 0.0, "max": 1.0}, {})
        if ok:
            from backend.aegis.kernel.executor import EXEC
            EXEC.call(self.cluster.clear_crisis, bed)
            self._apply_pin(bed, 1.0)
            self.interventions.pop(bed, None)
        return SyscallResult(
            sid, "clear", bed, prof.name, EOK if ok else EINVAL, ok,
            cert.z3_status,
            "RECOVERED — pin released, intervention state cleared" if ok
            else f"REJECTED — {cert.message}",
            egress=ACTUATOR, result={})

    # ------------------------------------------------------------------ #
    def _apply_pin(self, bed: str, pin: float) -> float:
        # A pin changes the next allocation the scheduler computes, so it is a
        # write to wing state and belongs on the writer thread with the ticks it
        # competes with — otherwise a pin can land halfway through one.
        from backend.aegis.kernel.executor import EXEC
        try:
            return EXEC.call(self.cluster.sched.set_pin, bed, pin)
        except Exception:
            return pin

    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        allowed = sum(1 for r in self.audit if r.allowed)
        denied = len(self.audit) - allowed
        return {
            "syscalls": list(SYSCALLS),
            "invoked": self._sid, "allowed": allowed, "denied": denied,
            "active_interventions": {b: iv.to_dict()
                                     for b, iv in self.interventions.items()},
            "recent": [r.to_dict() for r in self.audit[-12:]],
        }

    def audit_log(self, limit: int = 64) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.audit[-limit:]]
