"""Kernel policy proofs — the access policy, checked against the whole syscall table.

WHAT IS PROVEN, AND HOW
-----------------------
The policy is data: roles and their rings (``rbac.RING``), capabilities and their
minimum rings (``rbac.CAP_MIN_RING``), grants (``rbac.GRANTS``), the break-glass
sets, the agent deny-set, and the syscall table (``abi_table.json``: which
capability each of the kernel's routes needs, and which ones mutate). This module
encodes all of it in Z3 over ENUMERATED sorts — Role, Cap, Syscall — as function
tables, then states each property's NEGATION with free symbolic variables and asks
the solver for a witness:

  P1  guest_reaches_no_write     ∃ s. mutating(s) ∧ granted(guest, cap(s))
  P2  grants_respect_rings       ∃ r c. granted(r, c) ∧ ring(r) < minring(c)
  P3  separation_of_duties       ∃ r. granted(r, order:treat) ∧ granted(r, policy:write)
  P4  breakglass_bounded         ∃ r c. elevated(r, c) ∧ forbidden(c) ∧ ¬granted(r, c)
  P5  agent_within_clinician     ∃ r c. agent(r, c) ∧ (¬granted(r, c) ∨ agent_deny(c))
  P6  every_write_is_guarded     ∃ s. mutating(s) ∧ minring(cap(s)) = 0 ∧ ¬public_write(s)

``unsat`` is the proof; ``sat`` returns the model — the role, capability or syscall
that breaks the property — so a bad grant is reported as the exact hole.

Two tables in P4/P5 are NOT the definitions restated: they are the capability sets
the running code actually COMPUTES (``users.elevated_caps``, ``users.agent_caps``),
so the proof checks the implementation's output against the policy, not the
policy against itself.

HONEST SCOPE: a finite-domain proof. Six roles, ~41 capabilities and ~130 syscalls
could also be enumerated in Python and would reach the same verdicts; the solver
adds one formal statement per property and counterexample extraction. It proves
the POLICY and the gate's inputs; it does not prove the gate code routes every
call — that is ``verify_gate`` / ``verify_abi``, by testing. It runs once at boot on
a background thread, never per request.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from z3 import (And, BoolSort, Const, EnumSort, Function, If, IntSort, Not, Or,
                Solver, sat, unsat)

from backend.aegis.rbac import (AGENT_DENY, BREAKGLASS_CAPS, CAP_MIN_RING,
                                FORBIDDEN_ELEVATION, GRANTS, RING)
from backend.aegis.z3lock import z3_guarded

#: Mutating syscalls deliberately open to ring 0. Each needs a reason on record.
PUBLIC_WRITES = {
    "POST /aegis/ingest": "a visitor streaming their OWN camera; nothing else is changed",
    "POST /aegis/sys/kill": "route open so owners can signal their own processes; the "
                            "handler refuses anyone else's unless they hold proc:signal",
    "POST /aegis/fs/write": "the filesystem checks each node's own write capability in "
                            "every gate mode; a guest can write only /tmp scratch",
    "POST /aegis/fs/mkdir": "as fs/write — only /home (owner-only) accepts a directory",
    "POST /aegis/fs/rm": "as fs/write — owners remove their own /home files and /tmp files",
}


def _key(e: Any) -> str:
    return f"{e.method} {e.path}"


@z3_guarded
def prove_kernel_policy(abi: Any, grants: Optional[Dict[str, frozenset]] = None,
                        agent_caps: Optional[Dict[str, frozenset]] = None,
                        elevated_caps: Optional[Dict[str, frozenset]] = None,
                        cap_min_ring: Optional[Dict[str, int]] = None,
                        public_writes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Prove P1–P6. The keyword overrides exist so the verify suite can plant a hole
    (e.g. grant guest ``pkg:install``) and require the proof to FIND it."""
    t0 = time.perf_counter()
    grants = GRANTS if grants is None else grants
    cap_min_ring = CAP_MIN_RING if cap_min_ring is None else cap_min_ring
    public_writes = PUBLIC_WRITES if public_writes is None else public_writes
    if agent_caps is None or elevated_caps is None:
        from backend.aegis.kernel import users as _u
        agent_caps = agent_caps or {r: _u.agent_caps(r) for r in grants}
        elevated_caps = elevated_caps or {r: _u.elevated_caps(r) for r in grants}

    roles = sorted(grants)
    caps = sorted(set(cap_min_ring) | {c for g in grants.values() for c in g})
    entries = list(abi.entries)
    unknown = sorted({e.cap for e in entries} - set(cap_min_ring))
    if unknown:
        return {"sound": False, "error": f"syscalls name unknown capabilities: {unknown}",
                "properties": []}

    # Enumeration sort names are GLOBAL in Z3's shared context — redeclaring "Role"
    # on the second proof in a process raises — so every run gets its own names. A
    # counter is not enough: reloading this module resets it while Z3 still holds
    # the old names (the falsification engine reloads it between runs).
    k = uuid.uuid4().hex[:10]
    RoleS, role_c = EnumSort(f"Role{k}", [f"r{k}_{r}" for r in roles])
    CapS, cap_c = EnumSort(f"Cap{k}", [f"c{k}_" + c.replace(":", "_") for c in caps])
    SysS, sys_c = EnumSort(f"Sys{k}", [f"s{k}_{e.nr}" for e in entries])
    R = dict(zip(roles, role_c))
    C = dict(zip(caps, cap_c))
    S = {e.nr: sc for e, sc in zip(entries, sys_c)}

    granted = Function("granted", RoleS, CapS, BoolSort())
    agent = Function("agent", RoleS, CapS, BoolSort())
    elevated = Function("elevated", RoleS, CapS, BoolSort())
    ring = Function("ring", RoleS, IntSort())
    minring = Function("minring", CapS, IntSort())
    forbidden = Function("forbidden", CapS, BoolSort())
    agent_deny = Function("agent_deny", CapS, BoolSort())
    needs = Function("needs", SysS, CapS)
    mutating = Function("mutating", SysS, BoolSort())
    public_write = Function("public_write", SysS, BoolSort())

    base: List[Any] = []
    for r in roles:
        base.append(ring(R[r]) == RING.get(r, 0))
        for c in caps:
            base.append(granted(R[r], C[c]) == (c in grants.get(r, frozenset())))
            base.append(agent(R[r], C[c]) == (c in agent_caps.get(r, frozenset())))
            base.append(elevated(R[r], C[c]) == (c in elevated_caps.get(r, frozenset())))
    for c in caps:
        base.append(minring(C[c]) == cap_min_ring.get(c, 99))
        base.append(forbidden(C[c]) == (c in FORBIDDEN_ELEVATION))
        base.append(agent_deny(C[c]) == (c in AGENT_DENY))
    for e in entries:
        base.append(needs(S[e.nr]) == C[e.cap])
        base.append(mutating(S[e.nr]) == (e.klass in ("write", "egress")))
        base.append(public_write(S[e.nr]) == (_key(e) in public_writes))

    r, c, s = Const("r", RoleS), Const("c", CapS), Const("s", SysS)
    guest = R.get("guest")
    treat, pol = C.get("order:treat"), C.get("policy:write")

    negations = {
        "guest_reaches_no_write": (
            "a guest can reach no write or egress syscall (except the declared public writes)",
            And(mutating(s), Not(public_write(s)), granted(guest, needs(s)))
            if guest is not None else None),
        "grants_respect_rings": (
            "no role holds a capability above its privilege ring",
            And(granted(r, c), ring(r) < minring(c))),
        "separation_of_duties": (
            "nobody may both prescribe and rewrite the safety policy",
            And(granted(r, treat), granted(r, pol)) if treat is not None and pol is not None
            else None),
        "breakglass_bounded": (
            "break-glass never confers a forbidden capability the role did not already hold",
            And(elevated(r, c), forbidden(c), Not(granted(r, c)))),
        "agent_within_clinician": (
            "Dr. Aegis never holds a capability its clinician lacks, nor a denied one",
            And(agent(r, c), Or(Not(granted(r, c)), agent_deny(c)))),
        "every_write_is_guarded": (
            "every mutating syscall needs a ring ≥ 1 capability unless declared public",
            And(mutating(s), minring(needs(s)) == 0, Not(public_write(s)))),
    }

    names = {str(v): k for k, v in R.items()}
    cnames = {str(v): k for k, v in C.items()}
    snames = {str(S[e.nr]): e for e in entries}
    #: the variables each property actually binds — a witness reports only those
    binds = {"guest_reaches_no_write": ("s",), "grants_respect_rings": ("r", "c"),
             "separation_of_duties": ("r",), "breakglass_bounded": ("r", "c"),
             "agent_within_clinician": ("r", "c"), "every_write_is_guarded": ("s",)}
    out: List[Dict[str, Any]] = []
    for name, (claim, neg) in negations.items():
        if neg is None:
            out.append({"property": name, "claim": claim, "proven": False,
                        "z3": "n/a", "counterexample": "property not applicable"})
            continue
        sol = Solver()
        sol.add(*base)
        sol.add(neg)
        res = sol.check()
        row: Dict[str, Any] = {"property": name, "claim": claim,
                               "proven": res == unsat, "z3": str(res)}
        if res == sat:
            m = sol.model()
            cex: Dict[str, Any] = {}
            if "r" in binds[name]:
                cex["role"] = names.get(str(m.eval(r, model_completion=True)))
            if "c" in binds[name]:
                cex["capability"] = cnames.get(str(m.eval(c, model_completion=True)))
            if "s" in binds[name]:
                ent = snames.get(str(m.eval(s, model_completion=True)))
                if ent is not None:
                    cex["syscall"] = f"{ent.nr} {ent.method} {ent.path} needs {ent.cap}"
            row["counterexample"] = cex
        out.append(row)

    return {"sound": all(p["proven"] for p in out),
            "properties": out,
            "domain": {"roles": len(roles), "capabilities": len(caps), "syscalls": len(entries)},
            "public_writes": public_writes,
            "ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "scope": "finite-domain proof over the committed policy and ABI tables"}
