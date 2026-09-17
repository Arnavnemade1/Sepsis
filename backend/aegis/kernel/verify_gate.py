"""Verify the gate — refusals that stop the handler, audits that do not, and the cost.

    .venv/bin/python -m backend.aegis.kernel.verify_gate

The claim is "a caller without the capability cannot make the kernel do it". The
evidence is a side effect that does NOT happen: the handler increments a counter,
and a refused call must leave it untouched. Then the CONTROL removes the
middleware and requires the identical call to succeed — a 403 that would have
happened anyway (a broken route, a typo) proves nothing.
"""
from __future__ import annotations

import sys
import time
from typing import List, Tuple

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.aegis.kernel import gate as G
from backend.aegis.kernel.abi import ABI, ABIEntry
from backend.aegis.kernel.journal import Journal
from backend.aegis.kernel.proc import ProcessTable
from backend.aegis.kernel.users import PermissionDenied, UserDB, current_credential
from backend.aegis.rbac import RBAC

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def _app(side: dict, with_gate: bool = True) -> FastAPI:
    app = FastAPI()

    @app.get("/aegis/ward")
    def ward():
        return {"who": current_credential().username}

    @app.post("/aegis/pkg/install")
    def install():
        side["installs"] += 1
        return {"ok": True, "by": current_credential().username}

    @app.post("/aegis/sys/login")
    def login(body: dict):
        tok, _, cred = side["users"].login(body["username"])
        return {"token": tok, "role": cred.role}

    @app.websocket("/aegis/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"who": current_credential().username})
        await ws.close()

    if with_gate:
        app.add_middleware(G.GateMiddleware)                 # inside CORS, as in server.py
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    return app


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  GATE — one reference monitor for every call")
    print("=" * 64)
    saved = G.GATE
    try:
        return _run()
    finally:
        G.GATE = saved


def _run() -> int:
    j = Journal(ring=2048)
    users = UserDB(RBAC(), procs=ProcessTable(journal=j), journal=j, demo_login=True)
    abi = ABI([ABIEntry(1, "ward", "GET", "/aegis/ward", "read", "view:public"),
               ABIEntry(2, "pkg.install", "POST", "/aegis/pkg/install", "write", "pkg:install"),
               ABIEntry(3, "sys.login", "POST", "/aegis/sys/login", "auth", "view:public"),
               ABIEntry(4, "stream", "WS", "/aegis/stream", "stream", "view:public")])
    side = {"installs": 0, "users": users}
    client = TestClient(_app(side))
    gate = G.configure(abi, users, journal=j, mode="enforce")

    # -- 1. enforce: the refusal stops the handler ---------------------------- #
    r = client.post("/aegis/pkg/install")
    check("a guest write is refused with 403 EPERM",
          r.status_code == 403 and r.json().get("errno") == "EPERM", f"{r.status_code}")
    check("THE CLAIM: the refused handler never ran (no side effect)", side["installs"] == 0,
          f"installs={side['installs']}")
    check("the refusal names the capability and the principal",
          r.json().get("capability") == "pkg:install" and r.json().get("uid") == "guest")
    check("a 403 still carries CORS headers (gate sits inside CORS)",
          client.post("/aegis/pkg/install", headers={"Origin": "https://aegis.example"})
          .headers.get("access-control-allow-origin") in ("*", "https://aegis.example"))
    check("a guest read is served", client.get("/aegis/ward").json() == {"who": "guest"})

    tok = client.post("/aegis/sys/login", json={"username": "dr.whitfield"}).json()["token"]
    H = {"Authorization": f"Bearer {tok}"}
    r = client.post("/aegis/pkg/install", headers=H)
    check("the chief's identical write is allowed and runs as the chief",
          r.status_code == 200 and r.json().get("by") == "dr.whitfield" and side["installs"] == 1)
    tok_n = client.post("/aegis/sys/login", json={"username": "rn.patel"}).json()["token"]
    r = client.post("/aegis/pkg/install", headers={"Authorization": f"Bearer {tok_n}"})
    check("a signed-in nurse is still refused (identity is not permission)",
          r.status_code == 403 and side["installs"] == 1)
    r = client.post("/aegis/pkg/install", headers={"Authorization": f"Bearer {tok}x"})
    check("a forged token is 401 EAUTH, not a silent guest", r.status_code == 401
          and r.json().get("errno") == "EAUTH")
    check("…and sign-in stays reachable with a stale token (auth is never refused)",
          client.post("/aegis/sys/login", json={"username": "rn.patel"},
                      headers={"Authorization": "Bearer stale.token"}).status_code == 200)
    users.logout(users.credential(tok))
    r = client.post("/aegis/pkg/install", headers=H)
    check("after sign-out the chief's token is refused", r.status_code == 401
          and side["installs"] == 1, f"{r.status_code}")

    # -- 2. CONTROL: without the gate the same guest write lands --------------- #
    bare = TestClient(_app(side, with_gate=False))
    before = side["installs"]
    r = bare.post("/aegis/pkg/install")
    check("CONTROL: with the middleware removed, the identical guest write SUCCEEDS",
          r.status_code == 200 and side["installs"] == before + 1)

    # -- 3. audit: served, counted, journalled once per window ---------------- #
    gate.mode = "audit"
    before = side["installs"]
    for _ in range(5):
        client.post("/aegis/pkg/install")
    check("audit mode serves what it would refuse", side["installs"] == before + 5)
    st = gate.status()
    check("…and counts every would-be refusal", st["would_deny"] >= 5, str(st["would_deny"]))
    rows = [e for e in j.entries if e.event == "gate.would_deny" and e.uid == "guest"]
    check("…but journals a repeated one once per window (not per poll)", len(rows) == 1,
          f"{len(rows)} rows")

    # -- 4. the WebSocket path ------------------------------------------------- #
    gate.mode = "enforce"
    tok2 = client.post("/aegis/sys/login", json={"username": "dr.alvarez"}).json()["token"]
    with client.websocket_connect("/aegis/stream",
                                  subprotocols=["aegis", "aegis.session." + tok2]) as ws:
        msg = ws.receive_json()
        proto = ws.accepted_subprotocol
    check("a socket authenticates by subprotocol and runs as that user",
          msg == {"who": "dr.alvarez"}, str(msg))
    check("the server selects the plain 'aegis' subprotocol (browsers require one)",
          proto == "aegis", str(proto))
    with client.websocket_connect("/aegis/stream") as ws:
        check("a socket with no token is a guest", ws.receive_json() == {"who": "guest"})

    # -- 5. in-process entry: the agent plane -------------------------------- #
    agent = users.agent_credential(users.credential(tok_n))           # acts for the nurse
    try:
        gate.enter(agent, "POST", "/aegis/pkg/install")
        check("gate.enter refuses the agent what its clinician lacks", False, "allowed")
    except PermissionDenied as exc:
        check("gate.enter refuses the agent what its clinician lacks", True, str(exc)[:60])
    d = gate.enter(agent, "GET", "/aegis/ward")
    check("…and allows what the clinician holds", d.allowed)
    try:
        gate.require("order:treat", agent, what="agent syscall treat")
        check("require() refuses a refined capability", False, "allowed")
    except PermissionDenied:
        check("require() refuses a refined capability", True)
    gate.mode = "audit"
    check("in audit mode require() reports False instead of raising",
          gate.require("order:treat", agent, what="audit probe") is False)

    # -- 6. what it costs ------------------------------------------------------ #
    gate.mode = "enforce"
    hot = TestClient(_app(side))
    for _ in range(300):
        hot.get("/aegis/ward", headers={"Authorization": f"Bearer {tok2}"})
    ov = gate.status()["overhead_us"]
    check("gate overhead is measured, in microseconds (resolve + identify + decide)",
          0 < ov["p50"] < 500 and ov["samples"] >= 300, f"p50={ov['p50']}µs p99={ov['p99']}µs")
    check("the gate journal chain verifies", j.verify()["ok"])

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print(f"  ALL {len(_results)} CHECKS PASSED — refusals stop the handler; the gate is load-bearing")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
