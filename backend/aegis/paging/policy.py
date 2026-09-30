"""Paging policy: who needs a human's review, how long a page may ring, and the
invariants every dispatch must satisfy — proven by Z3 before a page leaves.

The router ranks and chooses; this module decides whether that choice may be
sent. It is the paging counterpart of the egress cage: a decision is turned into
facts, the policy into constraints, and Z3 is asked whether the facts can violate
the policy. ``unsat`` means no invariant is broken and the page is certified.
``sat`` means a violation, and the page is held for the operator with the broken
rules named. If the prover itself is unhealthy the decision fails closed.

The invariants are stated in ``INVARIANTS`` so the Policy screen and the
verification suite read the same text the prover checks.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.aegis.z3lock import z3_guarded as _z3_guarded

PRIORITIES = (1, 2, 3, 4)

# zone kind -> [P1, P2, P3, P4]; "review" = the operator confirms before the page is sent
DEFAULT_AUTONOMY: Dict[str, List[str]] = {
    "icu": ["review", "review", "auto", "auto"],
    "ed": ["auto", "auto", "auto", "auto"],
    "stepdown": ["auto", "auto", "auto", "auto"],
    "ward": ["auto", "auto", "auto", "auto"],
    "obs": ["auto", "auto", "auto", "auto"],
    "pacu": ["review", "auto", "auto", "auto"],
    "or": ["review", "auto", "auto", "auto"],
}
DEFAULT_SETTINGS: Dict[str, Any] = {
    "autonomy": DEFAULT_AUTONOMY,
    # a machine-raised P1 always gets a human look, wherever it is
    "detector_p1_review": True,
    # seconds a page rings before it escalates to the next backup
    "ack_timeout_s": {"1": 45, "2": 90, "3": 240, "4": 480},
    # review must never block care: an unreviewed P1/P2 goes out on its own after this
    "review_failsafe_s": {"1": 30, "2": 90},
    "consecutive_window_s": 300,        # no clinician paged twice in 5 min for different cases
    "workload_max_per_hour": 3,         # P3/P4 skip anyone holding more than this in the last hour
    "near_tie_pct": 0.10,               # top-3 within 10% -> the lightest caseload gets it
    "hold_window_s": 900,               # time-queued dispatch: wait up to 15 min for the right person
    "max_escalations": 3,
    "simulated_responders": True,       # demo: staff with no device open answer on their own
    "demo_traffic": True,               # demo: a synthetic nurse alert every few minutes
}

INVARIANTS: List[Dict[str, str]] = [
    {"id": "on_shift", "text": "Never page an off-shift clinician, unless it is a P1 and no on-shift clinician can take it."},
    {"id": "consecutive", "text": "No clinician is paged twice within the consecutive window for different cases, unless it is a P1 with no alternative."},
    {"id": "workload", "text": "A P3 or P4 never goes to someone already holding more than the hourly workload limit."},
    {"id": "specialty", "text": "The assignee's specialty is one the case accepts."},
    {"id": "autonomy", "text": "Where policy requires review, nothing is sent until the operator approves (or the review fail-safe fires for a P1/P2)."},
    {"id": "p1_timeout", "text": "A P1 page escalates within 90 seconds if unanswered."},
]


def zone_kind_for_policy(kind: str) -> str:
    return kind if kind in DEFAULT_AUTONOMY else "ward"


def requires_review(settings: Dict[str, Any], zone_kind: str, priority: int, source: str) -> bool:
    if source == "detector" and priority == 1 and settings.get("detector_p1_review", True):
        return True
    row = settings["autonomy"].get(zone_kind_for_policy(zone_kind), ["auto"] * 4)
    return row[priority - 1] == "review"


def ack_timeout(settings: Dict[str, Any], priority: int) -> int:
    return int(settings["ack_timeout_s"][str(priority)])


def merge_settings(current: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a partial update, keeping only known keys and sane values."""
    out = copy.deepcopy(current)
    for k, v in (patch or {}).items():
        if k == "autonomy" and isinstance(v, dict):
            for kind, row in v.items():
                if kind in out["autonomy"] and isinstance(row, list) and len(row) == 4 \
                        and all(x in ("review", "auto") for x in row):
                    out["autonomy"][kind] = list(row)
        elif k in ("ack_timeout_s", "review_failsafe_s") and isinstance(v, dict):
            for pr, secs in v.items():
                if str(pr) in out[k] and isinstance(secs, (int, float)) and 10 <= secs <= 3600:
                    out[k][str(pr)] = int(secs)
        elif k in ("detector_p1_review", "simulated_responders", "demo_traffic") and isinstance(v, bool):
            out[k] = v
        elif k in ("consecutive_window_s", "hold_window_s") and isinstance(v, (int, float)) and 0 <= v <= 7200:
            out[k] = int(v)
        elif k == "workload_max_per_hour" and isinstance(v, int) and 1 <= v <= 20:
            out[k] = v
        elif k == "near_tie_pct" and isinstance(v, (int, float)) and 0 <= v <= 0.5:
            out[k] = float(v)
        elif k == "max_escalations" and isinstance(v, int) and 1 <= v <= 6:
            out[k] = v
    # the P1 invariant is policy, not preference: the timeout cannot be set past it
    out["ack_timeout_s"]["1"] = min(out["ack_timeout_s"]["1"], 90)
    return out


@dataclass
class DispatchFacts:
    """Everything the invariants talk about, for one proposed page."""
    priority: int
    mode: str                          # dispatch | review | hold
    requires_review: bool
    operator_approved: bool
    failsafe_fired: bool
    assignee_on_shift: bool
    on_shift_alternative: bool         # an eligible on-shift clinician exists
    recently_paged_other: bool
    consecutive_alternative: bool      # an eligible clinician exists who was not just paged
    pages_last_hour: int
    workload_limit: int
    specialty_allowed: bool
    timeout_s: int


@_z3_guarded
def certify(f: DispatchFacts) -> Dict[str, Any]:
    """Prove the proposed page satisfies every invariant. unsat = certified."""
    from z3 import And, Bool, BoolVal, Implies, Int, Not, Or, Solver, sat, unsat
    t0 = time.perf_counter()
    P, T, H = Int("priority"), Int("timeout_s"), Int("pages_last_hour")
    L = Int("workload_limit")
    on_shift, alt_shift = Bool("assignee_on_shift"), Bool("on_shift_alternative")
    recent, alt_recent = Bool("recently_paged_other"), Bool("consecutive_alternative")
    spec, review, approved = Bool("specialty_allowed"), Bool("requires_review"), Bool("operator_approved")
    failsafe, sent = Bool("failsafe_fired"), Bool("sent_now")
    facts = [P == f.priority, T == f.timeout_s, H == f.pages_last_hour, L == f.workload_limit,
             on_shift == BoolVal(f.assignee_on_shift), alt_shift == BoolVal(f.on_shift_alternative),
             recent == BoolVal(f.recently_paged_other), alt_recent == BoolVal(f.consecutive_alternative),
             spec == BoolVal(f.specialty_allowed), review == BoolVal(f.requires_review),
             approved == BoolVal(f.operator_approved), failsafe == BoolVal(f.failsafe_fired),
             sent == BoolVal(f.mode == "dispatch")]
    inv = {
        "on_shift": Or(on_shift, And(P == 1, Not(alt_shift))),
        "consecutive": Or(Not(recent), And(P == 1, Not(alt_recent))),
        "workload": Or(P <= 2, H <= L),
        "specialty": spec,
        "autonomy": Implies(And(review, sent), Or(approved, And(failsafe, P <= 2))),
        "p1_timeout": Implies(P == 1, T <= 90),
    }
    s = Solver()
    s.add(*facts)
    s.add(Not(And(*inv.values())))
    status = s.check()
    violations: List[str] = []
    if status == sat:
        for name, cond in inv.items():
            s2 = Solver()
            s2.add(*facts)
            s2.add(Not(cond))
            if s2.check() == sat:
                violations.append(name)
    ok = status == unsat
    return {
        "certified": ok, "z3_status": "unsat" if ok else str(status), "violations": violations,
        "checked": [i["id"] for i in INVARIANTS],
        "proof": ("facts ⊢ " + " ∧ ".join(inv.keys())) if ok else
                 ("counterexample: facts violate " + ", ".join(violations)),
        "ms": round((time.perf_counter() - t0) * 1000, 2),
    }
