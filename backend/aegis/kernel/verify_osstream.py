"""Verify the OS stream — one socket, capability-filtered topics, bounded delivery.

    .venv/bin/python -m backend.aegis.kernel.verify_osstream

The claim that needs adversarial treatment is the journal filter: a guest on the
stream must not receive the auth, syscall, agent, egress or clinical facilities,
and an auditor on the SAME stream must. Both are checked against entries planted
in the journal before either subscribes.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import List, Tuple

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from backend.aegis.kernel import gate as G
from backend.aegis.kernel.abi import ABI, ABIEntry
from backend.aegis.kernel.journal import Journal
from backend.aegis.kernel.osstream import COALESCE, QUEUE_MAX, Conn, OSHub, fingerprint
from backend.aegis.kernel.proc import ProcessTable
from backend.aegis.kernel.users import GUEST, UserDB, current_credential
from backend.aegis.kernel.vfs import TmpFS, VFS
from backend.aegis.rbac import RBAC

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def _until(ws, kind: str, limit: int = 40):
    """Every message up to and including the first of ``kind``."""
    got = []
    for _ in range(limit):
        m = ws.receive_json()
        got.append(m)
        if m.get("type") == kind:
            break
    return got


def _drain(ws, want: int, limit: int = 40):
    got = []
    for _ in range(limit):
        m = ws.receive_json()
        got.append(m)
        if len(got) >= want:
            break
    return got


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  OS STREAM — one socket, filtered by capability, bounded")
    print("=" * 64)
    saved = G.GATE
    try:
        return _run()
    finally:
        G.GATE = saved


def _run() -> int:
    j = Journal(ring=512)
    procs = ProcessTable(journal=j)
    users = UserDB(RBAC(), procs=procs, journal=j)
    G.configure(ABI([ABIEntry(1, "os.stream", "WS", "/aegis/os/stream", "stream", "view:public")]),
                users, journal=j, mode="enforce")
    vfs = VFS(journal=j)
    vfs.mount("/tmp", TmpFS())
    hub = OSHub(journal=j)
    hub.ps = lambda: procs.ps()
    hub.boot = lambda: {"booted": True, "phases": 3}
    hub.vfs_fingerprint = lambda cred, path: fingerprint(vfs, cred, path)

    async def op_tick(conn, msg):            # lets the test drive the periodic publisher
        await hub.tick()
        hub.send(conn, {"type": "ticked"})
    hub.register_op("test.tick", op_tick)

    app = FastAPI()

    @app.websocket("/aegis/os/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        await hub.serve(ws, current_credential())

    app.add_middleware(G.GateMiddleware)
    client = TestClient(app)

    # plant journal entries across facilities BEFORE anyone subscribes
    j.append("auth", "login", uid="dr.alvarez")
    j.append("syscall", "treat", uid="dr.alvarez", bed="bed_3")
    j.append("kern", "boot.start")
    j.append("clinical", "identity_fault")
    j.append("sched", "hard_preempt", owner="bed_3")

    # -- 1. identity and topics ------------------------------------------------ #
    with client.websocket_connect("/aegis/os/stream") as ws:
        hello = ws.receive_json()
        check("an unauthenticated socket is greeted as guest", hello.get("uid") == "guest")
        ws.send_json({"op": "sub", "topic": "journal", "backlog": 50})
        msgs = _drain(ws, 2)
        rows = [r for m in msgs if m.get("type") == "journal" for r in m["data"]]
        facs = {r["facility"] for r in rows}
        check("THE CLAIM: a guest's journal feed excludes auth, syscall and clinical",
              not (facs & {"auth", "syscall", "clinical"}), str(sorted(facs)))
        check("…but still carries the kernel's public facilities",
              {"kern", "sched"} <= facs, str(sorted(facs)))
        ws.send_json({"op": "sub", "topic": "nonsense"})
        err = ws.receive_json()
        check("an unknown topic is refused (EINVAL)", err.get("errno") == "EINVAL")

    tok, _, _ = users.login("dr.alvarez")
    with client.websocket_connect("/aegis/os/stream",
                                  subprotocols=["aegis", "aegis.session." + tok]) as ws:
        hello = ws.receive_json()
        check("a session subprotocol is greeted as that user", hello.get("uid") == "dr.alvarez")
        ws.send_json({"op": "sub", "topic": "journal", "backlog": 50})
        rows = [r for m in _drain(ws, 2) if m.get("type") == "journal" for r in m["data"]]
        facs = {r["facility"] for r in rows}
        check("CONTROL: an auditor on the same stream DOES receive auth and syscall",
              {"auth", "syscall", "clinical"} <= facs, str(sorted(facs)))
        ws.send_json({"op": "sub", "topic": "proc"})
        ws.send_json({"op": "sub", "topic": "boot"})
        got = _drain(ws, 4)
        types = [m.get("type") for m in got]
        check("proc and boot are delivered on subscribe", "proc" in types and "boot" in types,
              str(types))

        # -- 2. watches ----------------------------------------------------- #
        # Only the watch is under test: the periodic tick would otherwise also
        # publish proc and journal traffic into the same window.
        ws.send_json({"op": "unsub", "topic": "proc"})
        ws.send_json({"op": "unsub", "topic": "journal"})
        vfs.write(GUEST, "/tmp/watched", "v1")
        ws.send_json({"op": "watch", "path": "/tmp"})
        _until(ws, "watching")
        ws.send_json({"op": "test.tick"})
        quiet = _until(ws, "ticked")
        check("an unchanged path produces no event on tick",
              not any(m.get("type") == "vfs" for m in quiet), str([m.get("type") for m in quiet]))
        vfs.write(GUEST, "/tmp/other", "new file")
        ws.send_json({"op": "test.tick"})
        after = _until(ws, "ticked")
        check("a change under a watched path produces a vfs event",
              any(m.get("type") == "vfs" and m.get("path") == "/tmp" for m in after),
              str([m.get("type") for m in after]))

    # -- 3. backpressure --------------------------------------------------------- #
    loop = asyncio.new_event_loop()
    try:
        async def fill():
            c = Conn(ws=None, cred=GUEST)
            for i in range(QUEUE_MAX):
                c.offer({"type": "proc", "i": i}, coalesce=True)
            ok = c.offer({"type": "proc", "i": "overflow"}, coalesce=True)
            closed_after_coalesce = c.closed
            c2 = Conn(ws=None, cred=GUEST)
            for i in range(QUEUE_MAX):
                c2.offer({"type": "tty"}, coalesce=False)
            c2.offer({"type": "tty"}, coalesce=False)
            return ok, c.dropped, closed_after_coalesce, c2.closed
        ok, dropped, closed_c, closed_t = loop.run_until_complete(fill())
    finally:
        loop.close()
    check("a full queue DROPS coalescable traffic and counts it (never blocks the kernel)",
          ok is False and dropped == 1 and not closed_c, f"dropped={dropped}")
    check("a client that cannot take control traffic is disconnected, not waited on", closed_t)
    check("snapshot, cluster and proc are the coalescable topics",
          set(COALESCE) == {"snapshot", "cluster", "proc"})
    check("open and close are journalled", any(e.event == "os.stream.open" for e in j.entries)
          and any(e.event == "os.stream.close" for e in j.entries))

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print(f"  ALL {len(_results)} CHECKS PASSED — one socket, and it shows each viewer what they may see")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
