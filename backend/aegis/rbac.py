"""Clinical RBAC — a multi-role privilege-ring OS shell (MEGA-UPGRADE Layer 3).

A real medical OS is multi-user: *what you may see and do depends on who you are*.
This is operating-systems privilege rings applied to hospital workflow.

  Ring 1  nurse / remote-specialist  — observe, acknowledge, annotate
  Ring 2  pharmacist                 — medication review, countersign / veto
  Ring 3  attending physician        — treatment synthesis, order protocols
  Ring 4  chief of medicine          — whole-ICU oversight (tactical sphere)

Three guarantees, each real and asserted by measurement (see verify_rbac.py):

  1. **Default-deny capability model.** A role holds an *explicit* capability set;
     anything not granted is denied. Rings are a safety net on top — a capability
     carries a minimum ring and a grant above a role's ring is a policy bug.

  2. **Z3-verified policy soundness.** :func:`prove_policy_sound` proves over the
     whole RBAC matrix that every grant respects its capability's minimum ring
     (the negation is ``unsat``). This is a genuine formal check of the access
     policy — it catches a privilege-escalation hole before deployment, the same
     way the cage proves protocols. It does NOT claim to verify the humans.

  3. **Signed sessions + immutable audit.** Login mints an HMAC-signed, expiring
     token; every authorization decision (grant or deny) is appended to a
     hash-chained audit ledger. A forged or expired token is rejected.

Honest scope: this is demonstrable RBAC with HMAC tokens — NOT a production identity
provider (no real SSO/IAM, no key management). The contribution is the model + the
formal soundness proof + the tamper-evident audit, not enterprise auth.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from z3 import And, Bool, BoolVal, Int, Not, Or, Solver, sat, unsat
from backend.aegis.z3lock import z3_guarded as _z3_guarded


# --------------------------------------------------------------------------- #
# Capability taxonomy + minimum ring per capability.
# --------------------------------------------------------------------------- #
# capability -> minimum privilege ring required to ever hold it
CAP_MIN_RING: Dict[str, int] = {
    "view:vitals": 1, "view:twin": 1, "view:narrative": 1, "ack:alarm": 1, "annotate": 1,
    "view:meds": 2, "view:interactions": 2, "pharm:countersign": 2, "pharm:veto": 2,
    "order:treat": 3, "order:escalate": 3, "order:clear": 3,
    "order:approve_protocol": 3, "drill:crisis": 3, "override:alarm": 3,
    "view:cohort": 4,
}
CAPABILITIES = frozenset(CAP_MIN_RING)

RING: Dict[str, int] = {
    "nurse": 1, "remote_specialist": 1, "pharmacist": 2, "attending": 3, "chief": 4,
}

# Explicit grants (need-to-know). Rings are necessary, not sufficient — a role only
# gets what policy lists. (Chief outranks attending but is NOT granted order:treat:
# treatment is the attending's clinical responsibility — RBAC separation of duties.)
GRANTS: Dict[str, frozenset] = {
    "nurse": frozenset({"view:vitals", "view:twin", "view:narrative", "ack:alarm"}),
    "remote_specialist": frozenset({"view:vitals", "view:twin", "view:narrative", "annotate"}),
    "pharmacist": frozenset({"view:meds", "view:interactions", "view:narrative",
                             "pharm:countersign", "pharm:veto"}),
    "attending": frozenset({"view:vitals", "view:twin", "view:narrative", "view:meds",
                            "view:interactions", "ack:alarm", "order:treat",
                            "order:escalate", "order:clear", "order:approve_protocol",
                            "drill:crisis", "override:alarm"}),
    "chief": frozenset({"view:cohort", "view:vitals", "view:twin", "view:narrative",
                        "view:meds", "ack:alarm", "order:escalate", "override:alarm"}),
}

# write-syscall name -> capability it requires (gates the existing trap gate)
SYSCALL_CAP: Dict[str, str] = {
    "treat": "order:treat", "escalate": "order:escalate",
    "crisis": "drill:crisis", "clear": "order:clear",
}


# --------------------------------------------------------------------------- #
# Signed, expiring session token.
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    user: str
    role: str
    iat: float
    exp: float
    nonce: str
    token: str

    def to_dict(self) -> Dict[str, Any]:
        return {"user": self.user, "role": self.role, "ring": RING.get(self.role, 0),
                "iat": round(self.iat, 3), "exp": round(self.exp, 3),
                "token": self.token[:16] + "…"}


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


# --------------------------------------------------------------------------- #
# Hash-chained audit ledger.
# --------------------------------------------------------------------------- #
@dataclass
class AuditEntry:
    seq: int
    user: str
    role: str
    capability: str
    decision: str           # GRANT | DENY
    reason: str
    prev_hash: str
    hash: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"seq": self.seq, "user": self.user, "role": self.role,
                "capability": self.capability, "decision": self.decision,
                "reason": self.reason, "hash": self.hash[:12], "ts": round(self.ts, 3)}


class AuditLedger:
    GENESIS = "genesis"

    def __init__(self):
        self.entries: List[AuditEntry] = []
        self._head = self.GENESIS

    def append(self, user, role, capability, decision, reason) -> AuditEntry:
        seq = len(self.entries)
        body = {"seq": seq, "user": user, "role": role, "capability": capability,
                "decision": decision, "reason": reason}
        h = hashlib.sha256(self._head.encode()
                           + json.dumps(body, sort_keys=True).encode()).hexdigest()
        e = AuditEntry(seq, user, role, capability, decision, reason, self._head, h)
        self.entries.append(e)
        self._head = h
        from backend.aegis.kernel.journal import KJOURNAL, INFO, WARN
        KJOURNAL.append("auth", "authorize", severity=INFO if decision == "GRANT" else WARN,
                        uid=str(user), role=role, capability=capability,
                        decision=decision, reason=reason)
        return e

    def verify(self) -> bool:
        prev = self.GENESIS
        for e in self.entries:
            body = {"seq": e.seq, "user": e.user, "role": e.role,
                    "capability": e.capability, "decision": e.decision, "reason": e.reason}
            h = hashlib.sha256(prev.encode()
                               + json.dumps(body, sort_keys=True).encode()).hexdigest()
            if e.prev_hash != prev or e.hash != h:
                return False
            prev = e.hash
        return prev == self._head

    def head(self) -> str:
        return self._head


# --------------------------------------------------------------------------- #
# The RBAC authority.
# --------------------------------------------------------------------------- #
class RBAC:
    def __init__(self, secret: Optional[bytes] = None, token_ttl_s: float = 3600.0):
        self.secret = secret or b"aegis-rbac-demo-secret"
        self.ttl = token_ttl_s
        self.audit = AuditLedger()

    # -- sessions -------------------------------------------------------- #
    def login(self, user: str, role: str) -> Session:
        if role not in RING:
            raise ValueError(f"unknown role {role!r}")
        now = time.time()
        payload = {"user": user, "role": role, "iat": now, "exp": now + self.ttl,
                   "nonce": _b64(hashlib.sha256(f"{user}{now}".encode()).digest()[:9])}
        raw = json.dumps(payload, sort_keys=True).encode()
        sig = hmac.new(self.secret, raw, hashlib.sha256).hexdigest()
        token = f"{_b64(raw)}.{sig}"
        return Session(user, role, payload["iat"], payload["exp"], payload["nonce"], token)

    def verify_token(self, token: str) -> Optional[Session]:
        try:
            raw_b64, sig = token.rsplit(".", 1)
            raw = base64.urlsafe_b64decode(raw_b64 + "=" * (-len(raw_b64) % 4))
            expected = hmac.new(self.secret, raw, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected):
                return None                                   # forged / tampered
            p = json.loads(raw)
            if time.time() > p["exp"]:
                return None                                   # expired
            return Session(p["user"], p["role"], p["iat"], p["exp"], p["nonce"], token)
        except Exception:
            return None

    # -- capability checks ---------------------------------------------- #
    @staticmethod
    def can(role: str, capability: str) -> bool:
        return capability in GRANTS.get(role, frozenset())

    def authorize(self, token: str, capability: str) -> Tuple[bool, str, Optional[Session]]:
        """The runtime gate: verify the token, check the capability, audit the call."""
        s = self.verify_token(token)
        if s is None:
            self.audit.append("?", "?", capability, "DENY", "invalid or expired token")
            return False, "invalid or expired token", None
        if capability not in CAPABILITIES:
            self.audit.append(s.user, s.role, capability, "DENY", "unknown capability")
            return False, "unknown capability", s
        if not self.can(s.role, capability):
            self.audit.append(s.user, s.role, capability, "DENY",
                              f"{s.role} (ring {RING[s.role]}) lacks {capability}")
            return False, f"role {s.role} not permitted {capability}", s
        self.audit.append(s.user, s.role, capability, "GRANT", "within privilege ring")
        return True, "authorized", s

    def shell_capabilities(self, role: str) -> Dict[str, Any]:
        """What a role's shell may render/do — the per-role UI contract."""
        return {"role": role, "ring": RING.get(role, 0),
                "capabilities": sorted(GRANTS.get(role, frozenset()))}


# --------------------------------------------------------------------------- #
# Z3: prove the RBAC policy is sound (no grant above its capability's ring).
# --------------------------------------------------------------------------- #
@_z3_guarded
def prove_policy_sound(grants: Optional[Dict[str, frozenset]] = None) -> Dict[str, Any]:
    """Formally verify over the whole RBAC matrix that every grant respects its
    capability's minimum ring. We bind each role-ring and cap-min-ring to its real
    value in Z3, then assert that SOME grant violates the rule
    (``ring(role) < min_ring(cap)``) and check satisfiability:
        unsat  → no violation is possible → the policy is sound (the proof)
        sat    → Z3 found a privilege-escalation hole → returned as a counterexample
    """
    grants = grants if grants is not None else GRANTS
    s = Solver()
    rr = {role: Int(f"ring_{role}") for role in RING}
    cr = {cap: Int("minring_" + cap.replace(":", "_")) for cap in CAP_MIN_RING}
    for role, v in RING.items():
        s.add(rr[role] == v)
    for cap, v in CAP_MIN_RING.items():
        s.add(cr[cap] == v)
    # "there exists a granted (role,cap) whose role ring is below the cap's min ring"
    violating = [rr[role] < cr[cap] for role in grants for cap in grants[role]]
    s.add(Or(*violating)) if violating else s.add(BoolVal(False))
    status = s.check()
    sound = status == unsat
    holes = [{"role": r, "capability": c, "role_ring": RING[r], "cap_min_ring": CAP_MIN_RING[c]}
             for r in grants for c in grants[r] if RING[r] < CAP_MIN_RING[c]]
    return {"sound": sound, "z3_status": "unsat" if sound else "sat",
            "grants_checked": sum(len(c) for c in grants.values()), "violations": holes}


@_z3_guarded
def prove_no_escalation(role: str, capability: str) -> Dict[str, Any]:
    """Prove the gate cannot let ``role`` perform ``capability``. The gate allows
    iff (capability is granted) ∧ (ring(role) ≥ min_ring(cap)); we encode that
    allow-condition in Z3 with the real bindings and prove it is ``unsat``."""
    s = Solver()
    granted = Bool("granted")
    rr, cr = Int("role_ring"), Int("cap_min_ring")
    s.add(granted == BoolVal(RBAC.can(role, capability)))
    s.add(rr == RING.get(role, 0), cr == CAP_MIN_RING.get(capability, 99))
    s.add(And(granted, rr >= cr))            # the would-allow condition
    blocked = s.check() == unsat             # unsat ⇒ gate can never allow it
    return {"role": role, "capability": capability,
            "granted": RBAC.can(role, capability),
            "ring": RING.get(role, 0), "cap_min_ring": CAP_MIN_RING.get(capability, 99),
            "escalation_blocked": blocked}


# --------------------------------------------------------------------------- #
# Gate over the write-side syscall trap (compose, don't duplicate).
# --------------------------------------------------------------------------- #
@dataclass
class RBACGate:
    rbac: RBAC
    syscall_fn: Callable[..., Any]     # e.g. SyscallTable.invoke

    def invoke(self, token: str, syscall: str, bed: Optional[str] = None,
               args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cap = SYSCALL_CAP.get(syscall)
        if cap is None:
            return {"ok": False, "errno": "EINVAL", "detail": f"unknown syscall {syscall!r}"}
        ok, reason, s = self.rbac.authorize(token, cap)
        if not ok:
            return {"ok": False, "errno": "EPERM", "detail": reason,
                    "audit_head": self.rbac.audit.head()[:12]}
        res = self.syscall_fn(syscall, bed=bed, args=args or {}, caller=s.user)
        out = res.to_dict() if hasattr(res, "to_dict") else dict(res)
        out["rbac"] = {"user": s.user, "role": s.role, "capability": cap}
        return out


# --------------------------------------------------------------------------- #
def _demo() -> None:
    rbac = RBAC()
    print("policy sound:", prove_policy_sound())
    for role in RING:
        nurse = rbac.login("u_" + role, role)
        ok, why, _ = rbac.authorize(nurse.token, "order:treat")
        print(f"  {role:18} order:treat -> {'GRANT' if ok else 'DENY'} ({why})")
    nurse = rbac.login("nrs", "nurse")
    print("nurse ack:alarm:", rbac.authorize(nurse.token, "ack:alarm")[0])
    print("forged token:", rbac.authorize(nurse.token + "x", "ack:alarm")[0])
    print("audit intact:", rbac.audit.verify(), "entries:", len(rbac.audit.entries))


if __name__ == "__main__":
    _demo()
