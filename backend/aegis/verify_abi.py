"""Verify the REAL server behind the gate: the table is complete, and it bites.

    .venv/bin/python -m backend.aegis.verify_abi

``verify_gate`` proves the middleware on a toy app. This suite imports the actual
AegisOS server and asks the questions that only it can answer:

  1. Does every live /aegis route have a syscall-table entry, and does the check
     notice when one does not? (A synthetic route is added to the live list.)
  2. With the gate ENFORCING, is a real write refused for the wrong person — with
     an observable consequence that does not happen — and allowed for the right one?
  3. Does the per-trap refinement hold on the real syscall trap (separation of
     duties: the chief may escalate but not treat)?
  4. Does Dr. Aegis — through the real tool registry, including the generic
     ``os_write`` primitive — get refused exactly what its clinician lacks?

TestClient is used WITHOUT its context manager, so the server's startup loops do
not start; the import-time kernel is what is tested.
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

os.environ["SUPABASE_URL"] = ""                   # never the production database
os.environ.setdefault("AEGIS_JOURNAL_DB", os.path.join(__import__("tempfile").mkdtemp(),
                                                       "verify_abi.db"))

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  THE REAL SERVER BEHIND THE GATE")
    print("=" * 64)
    import backend.aegis.server as S
    from fastapi.testclient import TestClient
    from backend.aegis.kernel.abi import ABI, live_routes
    from backend.aegis.kernel.policy import PUBLIC_WRITES
    from backend.aegis.kernel.users import GUEST
    from backend.aegis.agent import DYNAMIC_CAP_TOOLS, TOOL_CAPS
    from backend.aegis.rbac import CAP_MIN_RING

    abi = ABI.load()
    routes = live_routes(S.app)

    # -- 1. completeness ------------------------------------------------------ #
    cov = abi.coverage(routes)
    check("every live /aegis route has a syscall-table entry, and none is stale",
          cov["complete"], f"{cov['live']} routes · missing {cov['missing']} · stale {cov['stale']}")
    planted = abi.coverage(routes + [("POST", "/aegis/planted/unguarded")])
    check("CONTROL: a route added without an entry is reported missing",
          not planted["complete"] and ("POST", "/aegis/planted/unguarded") in planted["missing"])
    unknown = sorted({e.cap for e in abi.entries} - set(CAP_MIN_RING))
    check("every entry names a known capability", not unknown, str(unknown))
    open_writes = [f"{e.method} {e.path}" for e in abi.entries
                   if e.klass in ("write", "egress") and CAP_MIN_RING.get(e.cap, 0) == 0]
    check("every ring-0 write is declared, with a reason, as a public write",
          set(open_writes) <= set(PUBLIC_WRITES), str(sorted(set(open_writes) - set(PUBLIC_WRITES))))
    check("the server booted with complete coverage", bool(S.GATE_COVERAGE and S.GATE_COVERAGE["complete"]))
    missing_caps = sorted(n for n in S._tools._tools
                          if n not in TOOL_CAPS and n not in DYNAMIC_CAP_TOOLS)
    check("every agent tool declares the capability it needs", not missing_caps, str(missing_caps))
    check("the agent registry is wired to the gate (authorizer + principal installed)",
          S._tools.authorizer is not None and S._tools.principal is not None)

    client = TestClient(S.app)
    saved_mode = S.GATE.mode
    S.GATE.mode = "enforce"
    try:
        # -- 2. a real write, refused and allowed ----------------------------- #
        nurse = client.post("/aegis/sys/login", json={"username": "rn.patel"}).json()["token"]
        chief = client.post("/aegis/sys/login", json={"username": "dr.whitfield"}).json()["token"]
        HN = {"Authorization": f"Bearer {nurse}"}
        HC = {"Authorization": f"Bearer {chief}"}
        check("the nurse's session is live", client.get("/aegis/sys/whoami", headers=HN)
              .json()["credential"]["username"] == "rn.patel")
        r = client.post("/aegis/sys/revoke", json={"sid": ""})            # guest, no token
        check("a guest cannot revoke sessions (403 EPERM)", r.status_code == 403
              and r.json().get("errno") == "EPERM", str(r.status_code))
        r = client.post("/aegis/sys/revoke", json={"sid": ""}, headers=HN)
        check("a nurse cannot revoke sessions (403 EPERM)", r.status_code == 403)
        check("THE CLAIM: after both refusals the nurse's session still works",
              client.get("/aegis/sys/whoami", headers=HN).json()["credential"]["username"] == "rn.patel")
        nurse_sid = S.USERS.credential(nurse).sid
        r = client.post("/aegis/sys/revoke", json={"sid": nurse_sid[:12]}, headers=HC)
        check("the chief CAN revoke it", r.status_code == 200 and r.json().get("revoked") == 1, r.text[:80])
        r = client.get("/aegis/sys/whoami", headers=HN)
        check("…and the revoked token is then refused (401 EAUTH)", r.status_code == 401, str(r.status_code))

        # -- 3. the trap's refinement: separation of duties -------------------- #
        bed = S.cluster.node_ids[0]
        r = client.post("/aegis/syscall", json={"syscall": "treat", "bed": bed}, headers=HC).json()
        check("the chief passes the route (order:escalate) but is refused TREAT (order:treat)",
              r.get("errno_name") == "EPERM" and r.get("capability") == "order:treat", str(r)[:100])
        att = client.post("/aegis/sys/login", json={"username": "dr.alvarez"}).json()["token"]
        before = len(S.syscalls.audit)
        r = client.post("/aegis/syscall", json={"syscall": "treat", "bed": bed},
                        headers={"Authorization": f"Bearer {att}"}).json()
        check("CONTROL: the attending's identical treat reaches the trap and is audited as her",
              len(S.syscalls.audit) == before + 1 and S.syscalls.audit[-1].sid == r.get("sid"),
              f"errno={r.get('errno_name')}")
        check("the trap's journal entry carries the SESSION's identity, not the body's caller",
              any(e["event"] == "treat" and e["uid"] == "dr.alvarez"
                  for e in S.KJOURNAL.tail(limit=200)))

        # -- 4. Dr. Aegis through the real registry --------------------------- #
        out = S._tools.invoke("manage_package", {"action": "install", "name": "tachy"},
                              on_behalf_of=GUEST)
        check("the agent acting for a guest is refused a package install",
              out.get("errno") == "EPERM" and out.get("capability") == "pkg:install", str(out)[:100])
        out = S._tools.invoke("os_write", {"path": "/aegis/pkg/uninstall",
                                           "body": {"name": "tachy"}}, on_behalf_of=GUEST)
        check("…and the generic os_write primitive is refused the same thing (no side door)",
              out.get("errno") == "EPERM", str(out)[:100])
        att_cred = S.USERS.credential(att)
        out = S._tools.invoke("run_kernel_workload", {"workload": "mixed", "policy": "FIFO"},
                              on_behalf_of=att_cred)
        check("CONTROL: acting for an attending, the agent may run a workload (sim:control)",
              out.get("ok") is True, str(out)[:100])
        rows = [e for e in S.KJOURNAL.tail(limit=300) if e["uid"] == "dr-aegis"
                and e["event"] in ("gate.deny", "gate.would_deny")]
        check("agent refusals are journalled as dr-aegis", len(rows) >= 1, f"{len(rows)} rows")
        overhead = S.GATE.status()["overhead_us"]
        check("the gate's measured overhead on the real server is microseconds",
              0 < overhead["p50"] < 1000, f"p50={overhead['p50']}µs p99={overhead['p99']}µs")
    finally:
        S.GATE.mode = saved_mode

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print(f"  ALL {len(_results)} CHECKS PASSED — the real kernel refuses what its policy says")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)                          # the imported server started daemon threads
