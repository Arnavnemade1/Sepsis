"""Verify the kernel policy proofs — and that each property can FIND a hole.

    .venv/bin/python -m backend.aegis.kernel.verify_policy

"Proven" is only evidence if the same query reports ``sat`` with the exact
counterexample when the policy is broken. Each property here is shown green on
the shipped tables and red on a table with one planted hole — and the hole it
reports must be the one that was planted.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

from backend.aegis.kernel.abi import ABI
from backend.aegis.kernel.policy import prove_kernel_policy
from backend.aegis.kernel import users as U
from backend.aegis.rbac import CAP_MIN_RING, GRANTS

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def _holes(res):
    return {p["property"]: p.get("counterexample") for p in res["properties"] if not p["proven"]}


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  POLICY — six properties over the whole syscall table")
    print("=" * 64)
    abi = ABI.load()
    clean = prove_kernel_policy(abi)
    for p in clean["properties"]:
        check(f"PROVEN on the shipped tables: {p['property']}", p["proven"], p["claim"])
    check("the proof covers every syscall in the committed table",
          clean["domain"]["syscalls"] == len(abi.entries), str(clean["domain"]))

    # -- planted holes: each must be found, and named -------------------------- #
    g = dict(GRANTS)
    g["guest"] = GRANTS["guest"] | {"pkg:install"}
    h = _holes(prove_kernel_policy(abi, grants=g))
    cex = (h.get("guest_reaches_no_write") or {}).get("syscall", "")
    check("a guest granted pkg:install is caught, at the install syscall",
          "/aegis/pkg/install" in cex, cex)

    g = dict(GRANTS)
    g["chief"] = GRANTS["chief"] | {"order:treat"}
    h = _holes(prove_kernel_policy(abi, grants=g))
    check("a chief granted order:treat breaks separation of duties, and is named",
          (h.get("separation_of_duties") or {}).get("role") == "chief", str(h))

    g = dict(GRANTS)
    g["nurse"] = GRANTS["nurse"] | {"svc:control"}
    h = _holes(prove_kernel_policy(abi, grants=g))
    check("a nurse granted a ring-4 capability is caught by the ring rule",
          h.get("grants_respect_rings") == {"role": "nurse", "capability": "svc:control"}, str(h))

    leaky_bg = {r: U.elevated_caps(r) | {"policy:write"} for r in GRANTS}
    h = _holes(prove_kernel_policy(abi, elevated_caps=leaky_bg))
    check("break-glass that confers policy:write is caught",
          (h.get("breakglass_bounded") or {}).get("capability") == "policy:write", str(h))

    leaky_agent = {r: U.agent_caps(r) | {"order:treat"} for r in GRANTS}
    h = _holes(prove_kernel_policy(abi, agent_caps=leaky_agent))
    check("an agent holding order:treat for a clinician who lacks it is caught",
          (h.get("agent_within_clinician") or {}).get("capability") == "order:treat", str(h))

    lowered = dict(CAP_MIN_RING)
    lowered["pkg:install"] = 0
    g = dict(GRANTS)
    h = _holes(prove_kernel_policy(abi, cap_min_ring=lowered))
    cex = (h.get("every_write_is_guarded") or {}).get("syscall", "")
    check("a write whose capability is lowered to ring 0 is caught",
          "pkg:install" in cex, cex)

    check("the proof is a one-off at boot, not a per-request cost (< 5 s)",
          clean["ms"] < 5000, f"{clean['ms']} ms")

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print(f"  ALL {len(_results)} CHECKS PASSED — every property proven, every planted hole found")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
