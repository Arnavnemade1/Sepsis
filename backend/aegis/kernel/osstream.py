"""The OS stream — one socket per desktop, carrying every topic the kernel publishes.

WHY THIS EXISTS
---------------
The clinical console opened ``/aegis/stream``, the tactical board opened
``/aegis/cluster/stream``, and everything else polled (a patient snapshot every
4 s, the devices every 2.6 s, a dossier's chart, bundle and tracking every 4 s).
Each surface re-invented its own feed, and none of them could see the process
table, the journal or a file changing.

A desktop now opens ONE socket and subscribes to what it shows:

    snapshot   the live session's clinical snapshot      (was /aegis/stream)
    cluster    the ward payload                          (was /aegis/cluster/stream)
    proc       the process table, every 2 s
    journal    new journal entries — filtered by capability: auth, syscalls, the
               agent, egress and clinical facilities need view:audit
    boot       the boot record, once, on subscribe
    vfs        change notifications for up to 16 watched paths
    tty        a shell session's input and output (the shell registers its ops)

Messages are JSON. Client → server: ``{"op": "sub", "topic": "proc"}``,
``unsub``, ``watch``/``unwatch`` ``{"path": ...}``, ``ping``, and any op a
subsystem registers (``tty.*``). Server → client: ``{"type": <topic>, ...}``.

BACKPRESSURE. Each connection has a bounded queue. A slow client does not slow the
kernel or other clients: coalescable topics (snapshot, cluster, proc) are dropped
when its queue is full and the drop is counted; the connection is closed if even
the control messages cannot be delivered. The old sockets stay, for the pages that
have not moved.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from backend.aegis.kernel.journal import Journal, KJOURNAL

TOPICS = ("snapshot", "cluster", "proc", "journal", "boot", "vfs", "tty")
COALESCE = ("snapshot", "cluster", "proc")
SENSITIVE_FACILITIES = frozenset({"auth", "syscall", "agent", "egress", "clinical", "vfs"})
MAX_WATCHES = 16
QUEUE_MAX = 128


class Conn:
    def __init__(self, ws: Any, cred: Any) -> None:
        self.ws = ws
        self.cred = cred
        self.topics: Set[str] = set()
        self.queue: "asyncio.Queue" = asyncio.Queue(maxsize=QUEUE_MAX)
        self.journal_seq = -1
        self.watches: Dict[str, str] = {}
        self.dropped = 0
        self.sent = 0
        self.opened = time.time()
        self.closed = False

    def offer(self, msg: Dict[str, Any], coalesce: bool) -> bool:
        try:
            self.queue.put_nowait(msg)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            if not coalesce:
                self.closed = True                  # cannot deliver control traffic
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {"uid": self.cred.username, "topics": sorted(self.topics),
                "watches": sorted(self.watches), "sent": self.sent, "dropped": self.dropped,
                "open_s": round(time.time() - self.opened, 1)}


class OSHub:
    def __init__(self, journal: Optional[Journal] = None, period_s: float = 2.0) -> None:
        self.journal = journal if journal is not None else KJOURNAL
        self.period_s = period_s
        self.conns: Set[Conn] = set()
        self.ops: Dict[str, Callable[[Conn, Dict[str, Any]], Awaitable[None]]] = {}
        # Providers the server installs (kept as callables so this module imports
        # nothing from the subsystems it streams).
        self.ps: Callable[[], Any] = lambda: []
        self.boot: Callable[[], Any] = lambda: {}
        self.vfs_fingerprint: Optional[Callable[[Any, str], str]] = None
        self.on_disconnect: List[Callable[[Conn], None]] = []

    # -- publishing (event-loop thread) ------------------------------------- #
    def publish(self, topic: str, data: Any) -> None:
        coalesce = topic in COALESCE
        for c in list(self.conns):
            if topic in c.topics:
                c.offer({"type": topic, "data": data}, coalesce)

    def send(self, conn: Conn, msg: Dict[str, Any]) -> None:
        conn.offer(msg, coalesce=False)

    def register_op(self, op: str, fn: Callable[[Conn, Dict[str, Any]], Awaitable[None]]) -> None:
        self.ops[op] = fn

    # -- the periodic publisher ---------------------------------------------- #
    def _journal_for(self, conn: Conn) -> List[Dict[str, Any]]:
        audit = conn.cred.has("view:audit")
        out = []
        for e in list(self.journal.entries):
            if e.seq <= conn.journal_seq:
                continue
            conn.journal_seq = e.seq
            if not audit and e.facility in SENSITIVE_FACILITIES:
                continue
            out.append(e.to_dict())
        return out[-200:]

    async def tick(self) -> None:
        if not self.conns:
            return
        want_proc = any("proc" in c.topics for c in self.conns)
        if want_proc:
            self.publish("proc", self.ps())
        for c in list(self.conns):
            if "journal" in c.topics:
                rows = self._journal_for(c)
                if rows:
                    c.offer({"type": "journal", "data": rows}, coalesce=False)
            if c.watches and self.vfs_fingerprint is not None:
                for path, old in list(c.watches.items()):
                    new = await asyncio.to_thread(self.vfs_fingerprint, c.cred, path)
                    if new != old:
                        c.watches[path] = new
                        c.offer({"type": "vfs", "path": path, "event": "changed"}, coalesce=False)

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception as exc:                   # pragma: no cover
                print(f"[osstream] tick error: {exc}")
            await asyncio.sleep(self.period_s)

    # -- one connection -------------------------------------------------------- #
    async def serve(self, ws: Any, cred: Any) -> None:
        conn = Conn(ws, cred)
        self.conns.add(conn)
        self.journal.append("net", "os.stream.open", uid=cred.username, pid=getattr(cred, "pid", 0))

        async def pump() -> None:
            while not conn.closed:
                msg = await conn.queue.get()
                await ws.send_json(msg)
                conn.sent += 1

        sender = asyncio.create_task(pump())
        try:
            await ws.send_json({"type": "hello", "uid": cred.username, "role": cred.role,
                                "topics": list(TOPICS), "ops": sorted(self.ops)})
            while not conn.closed:
                msg = await ws.receive_json()
                await self._handle(conn, msg if isinstance(msg, dict) else {})
        except Exception:
            pass
        finally:
            conn.closed = True
            sender.cancel()
            self.conns.discard(conn)
            for fn in self.on_disconnect:
                try:
                    fn(conn)
                except Exception:
                    pass
            self.journal.append("net", "os.stream.close", uid=cred.username,
                                sent=conn.sent, dropped=conn.dropped)

    async def _handle(self, conn: Conn, msg: Dict[str, Any]) -> None:
        op = str(msg.get("op") or "")
        if op == "ping":
            self.send(conn, {"type": "pong", "t": time.time()})
        elif op == "sub":
            topic = str(msg.get("topic") or "")
            if topic not in TOPICS:
                self.send(conn, {"type": "error", "errno": "EINVAL", "detail": f"no topic {topic!r}"})
                return
            conn.topics.add(topic)
            if topic == "journal":
                conn.journal_seq = max((e.seq for e in self.journal.entries), default=-1) - int(
                    min(max(int(msg.get("backlog", 50)), 0), 500))
                rows = self._journal_for(conn)
                if rows:
                    self.send(conn, {"type": "journal", "data": rows})
            elif topic == "boot":
                self.send(conn, {"type": "boot", "data": self.boot()})
            elif topic == "proc":
                self.send(conn, {"type": "proc", "data": self.ps()})
            self.send(conn, {"type": "subscribed", "topic": topic})
        elif op == "unsub":
            conn.topics.discard(str(msg.get("topic") or ""))
        elif op in ("watch", "unwatch"):
            path = str(msg.get("path") or "")
            if op == "unwatch":
                conn.watches.pop(path, None)
                return
            if len(conn.watches) >= MAX_WATCHES:
                self.send(conn, {"type": "error", "errno": "ENOSPC",
                                 "detail": f"at most {MAX_WATCHES} watches"})
                return
            if self.vfs_fingerprint is not None:
                conn.watches[path] = await asyncio.to_thread(self.vfs_fingerprint, conn.cred, path)
                self.send(conn, {"type": "watching", "path": path})
        elif op in self.ops:
            await self.ops[op](conn, msg)
        else:
            self.send(conn, {"type": "error", "errno": "EINVAL", "detail": f"no op {op!r}"})

    def status(self) -> Dict[str, Any]:
        return {"connections": [c.to_dict() for c in self.conns], "ops": sorted(self.ops),
                "topics": list(TOPICS)}


def fingerprint(vfs: Any, cred: Any, path: str) -> str:
    """A cheap change detector for a watched path: a hash of what the caller could
    read there (the listing for a directory, the content for a file), or the error."""
    try:
        node = vfs.stat(cred, path)
        if node.kind == "dir":
            body = "|".join(f"{n.name}:{n.size}:{int(n.mtime)}" for n in vfs.ls(cred, path))
        else:
            body = vfs.read(cred, path)["content"]
    except Exception as exc:
        body = f"error:{exc}"
    return hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()[:16]
