"""Outbound events: AegisOS telling the hospital's other systems what happened.

A page raised, escalated, accepted or resolved; a feed quarantined; a mapping adopted.
A hospital wires these into its own messaging (Vocera, TigerConnect, a nurse-call
integration, a data warehouse) by subscribing a URL. Deliveries follow the Standard
Webhooks specification:

    webhook-id:        msg_<uuid>            (the receiver de-duplicates on it)
    webhook-timestamp: <unix seconds>         (the receiver rejects stale ones)
    webhook-signature: v1,<base64 HMAC-SHA256(key, "<id>.<timestamp>.<body>")>

The subscription's secret is ``whsec_<base64 key>``, shown once at creation.

Delivery is at-least-once with exponential backoff (0 s, 5 s, 30 s, 2 min, 10 min,
30 min) and a dead-letter list after the last attempt. A receiver that answers 2xx
has the event; anything else is retried.

**The server will not be turned into a probe of its own network.** A subscription's
URL must be https and must resolve to a public address; loopback, private, link-local
and metadata addresses are refused unless ``AEGIS_WEBHOOK_ALLOW_PRIVATE=1`` (a
hospital LAN deployment). ``internal://echo`` delivers in-process to a built-in
receiver that verifies the signature the way any receiver should, so the whole path
can be shown without an external service.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse

EVENT_TYPES = ("page.created", "page.escalated", "page.accepted", "page.resolved", "page.cancelled",
               "interface.quarantined", "mapping.adopted", "mapping.review")
BACKOFF_S = (0, 5, 30, 120, 600, 1800)
TOLERANCE_S = 300


def new_secret() -> str:
    return "whsec_" + base64.b64encode(secrets.token_bytes(24)).decode()


def _key(secret: str) -> bytes:
    return base64.b64decode(secret.split("_", 1)[1] if secret.startswith("whsec_") else secret)


def sign(secret: str, msg_id: str, ts: int, body: str) -> str:
    mac = hmac.new(_key(secret), f"{msg_id}.{ts}.{body}".encode(), hashlib.sha256).digest()
    return "v1," + base64.b64encode(mac).decode()


def verify(secret: str, headers: Dict[str, str], body: str, now: Optional[float] = None) -> bool:
    """What a receiver does: recompute, compare in constant time, reject stale timestamps."""
    h = {k.lower(): v for k, v in headers.items()}
    try:
        msg_id, ts = h["webhook-id"], int(h["webhook-timestamp"])
    except (KeyError, ValueError):
        return False
    if abs((time.time() if now is None else now) - ts) > TOLERANCE_S:
        return False
    want = sign(secret, msg_id, ts, body).split(",", 1)[1]
    for sig in h.get("webhook-signature", "").split():
        v, _, s = sig.partition(",")
        if v == "v1" and hmac.compare_digest(s, want):
            return True
    return False


def check_url(url: str) -> Optional[str]:
    """None if a subscription may target ``url``; otherwise the reason it may not."""
    if url == "internal://echo":
        return None
    u = urlparse(url)
    allow_private = os.getenv("AEGIS_WEBHOOK_ALLOW_PRIVATE") == "1"
    if u.scheme != "https" and not (allow_private and u.scheme == "http"):
        return "webhook URLs must be https"
    host = u.hostname or ""
    if not host:
        return "no host in the URL"
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(host, u.port or 443, proto=socket.IPPROTO_TCP)}
    except OSError:
        return f"{host} does not resolve"
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if not allow_private and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                                  or ip.is_multicast or ip.is_unspecified):
            return f"{host} resolves to a non-public address ({ip}); refused"
    return None


@dataclass
class Subscription:
    id: str
    url: str
    events: List[str]
    secret: str
    created_by: str
    created_at: float = field(default_factory=time.time)
    active: bool = True
    description: str = ""
    delivered: int = 0
    failed: int = 0

    def public(self) -> Dict[str, Any]:
        return {"id": self.id, "url": self.url, "events": self.events, "created_by": self.created_by,
                "created_at": self.created_at, "active": self.active, "description": self.description,
                "delivered": self.delivered, "failed": self.failed, "secret": self.secret[:10] + "…"}


@dataclass
class Delivery:
    id: str
    sub: str
    event: str
    body: str
    attempts: int = 0
    next_at: float = 0.0
    status: str = "pending"          # pending | delivered | failed | dead
    last_code: Optional[int] = None
    last_error: str = ""
    created_at: float = field(default_factory=time.time)
    delivered_at: Optional[float] = None
    latency_ms: Optional[float] = None

    def public(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in ("id", "sub", "event", "attempts", "status", "last_code", "last_error",
                                           "created_at", "delivered_at", "latency_ms")}
        try:
            d["data"] = json.loads(self.body).get("data")
        except ValueError:
            d["data"] = None
        return d


class Webhooks:
    def __init__(self, post: Optional[Callable[[str, Dict[str, str], str], Tuple[int, str]]] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._lock = threading.RLock()
        self.subs: Dict[str, Subscription] = {}
        self.queue: Deque[Delivery] = deque()
        self.log: Deque[Delivery] = deque(maxlen=200)
        self.dead: Deque[Delivery] = deque(maxlen=100)
        self.echo: Deque[Dict[str, Any]] = deque(maxlen=50)
        self.clock = clock
        self._post = post or self._http_post
        self._page_status: Dict[str, str] = {}
        self.counters = {"events": 0, "deliveries": 0, "delivered": 0, "retries": 0, "dead": 0}

    # ------------------------------------------------------------------ #
    def subscribe(self, url: str, events: List[str], by: str, description: str = "") -> Dict[str, Any]:
        why = check_url(url)
        if why:
            return {"status": "error", "detail": why}
        ev = [e for e in (events or list(EVENT_TYPES)) if e in EVENT_TYPES]
        if not ev:
            return {"status": "error", "detail": f"no known event types; choose from {list(EVENT_TYPES)}"}
        s = Subscription(f"sub_{uuid.uuid4().hex[:12]}", url, ev, new_secret(), by, description=description[:120])
        with self._lock:
            self.subs[s.id] = s
        # the secret is returned once, here; afterwards only its prefix is shown
        return {"status": "ok", "subscription": {**s.public(), "secret": s.secret}}

    def test(self, sid: str) -> Dict[str, Any]:
        """Queue a signed ``ping`` to one subscription, whatever events it chose."""
        with self._lock:
            s = self.subs.get(sid)
            if s is None:
                return {"status": "error", "detail": "no such subscription"}
            now = self.clock()
            body = json.dumps({"type": "ping", "timestamp": now, "data": {"subscription": sid}},
                              separators=(",", ":"))
            d = Delivery(f"msg_{uuid.uuid4().hex}", sid, "ping", body, next_at=now)
            self.queue.append(d)
            self.counters["deliveries"] += 1
        return {"status": "ok", "delivery": d.id}

    def unsubscribe(self, sid: str) -> Dict[str, Any]:
        with self._lock:
            s = self.subs.pop(sid, None)
        return {"status": "ok"} if s else {"status": "error", "detail": "no such subscription"}

    def publish(self, event: str, data: Dict[str, Any]) -> int:
        """Queue ``event`` for every subscription that wants it. Returns how many."""
        if event not in EVENT_TYPES:
            return 0
        now = self.clock()
        n = 0
        with self._lock:
            self.counters["events"] += 1
            for s in self.subs.values():
                if not s.active or event not in s.events:
                    continue
                mid = f"msg_{uuid.uuid4().hex}"
                body = json.dumps({"type": event, "timestamp": now, "data": data}, separators=(",", ":"),
                                  default=str)
                self.queue.append(Delivery(mid, s.id, event, body, next_at=now))
                self.counters["deliveries"] += 1
                n += 1
        return n

    # paging events → webhook events (status transitions, de-duplicated per page)
    def from_paging(self, events: List[Dict[str, Any]]) -> None:
        for ev in events or []:
            if ev.get("type") != "page":
                continue
            p = ev.get("data") or {}
            pid, st = p.get("id"), p.get("status")
            if not pid or not st:
                continue
            prev = self._page_status.get(pid)
            if prev == st:
                continue
            self._page_status[pid] = st
            brief = {"id": pid, "priority": p.get("priority"), "status": st, "room": p.get("room"),
                     "category": p.get("category"), "text": p.get("text"), "assignee": p.get("assignee"),
                     "source": p.get("source"), "synthetic": p.get("synthetic"), "escalations": p.get("escalations")}
            if prev is None:
                self.publish("page.created", brief)
            elif st == "accepted":
                self.publish("page.accepted", brief)
            elif st == "resolved":
                self.publish("page.resolved", brief)
            elif st == "cancelled":
                self.publish("page.cancelled", brief)
            elif st == "paging" and (p.get("escalations") or 0) > 0:
                self.publish("page.escalated", brief)
        if len(self._page_status) > 5000:
            for k in list(self._page_status)[:1000]:
                self._page_status.pop(k, None)

    # integration-plane events → webhook events
    def from_interop(self, ev: Dict[str, Any]) -> None:
        kind, d = ev.get("type"), ev.get("data") or {}
        if kind == "quarantine":
            self.publish("interface.quarantined", {k: d.get(k) for k in ("id", "interface", "code", "concept",
                                                                          "fault", "counts")})
        elif kind == "mapping" and d.get("status") in ("adopted", "review"):
            self.publish("mapping." + d["status"], {k: d.get(k) for k in ("id", "interface", "code", "concept",
                                                                           "conversion", "proposer", "reason")})

    # ------------------------------------------------------------------ #
    def pump(self, max_items: int = 20) -> int:
        """Attempt the deliveries that are due. Called from a worker thread."""
        now = self.clock()
        due: List[Delivery] = []
        with self._lock:
            keep: Deque[Delivery] = deque()
            while self.queue:
                d = self.queue.popleft()
                (due if d.next_at <= now and len(due) < max_items else keep).append(d)
            self.queue = keep
        for d in due:
            s = self.subs.get(d.sub)
            if s is None:
                continue
            ts = int(self.clock())
            headers = {"content-type": "application/json", "webhook-id": d.id, "webhook-timestamp": str(ts),
                       "webhook-signature": sign(s.secret, d.id, ts, d.body), "user-agent": "AegisOS-Webhooks/1"}
            t0 = time.perf_counter()
            if s.url == "internal://echo":
                code, err = self._echo(s, headers, d.body)
            else:
                code, err = self._post(s.url, headers, d.body)
            d.attempts += 1
            d.last_code, d.last_error = code, err
            d.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            with self._lock:
                if code and 200 <= code < 300:
                    d.status, d.delivered_at = "delivered", self.clock()
                    s.delivered += 1
                    self.counters["delivered"] += 1
                    self.log.append(d)
                elif d.attempts >= len(BACKOFF_S):
                    d.status = "dead"
                    s.failed += 1
                    self.counters["dead"] += 1
                    self.dead.append(d)
                    self.log.append(d)
                else:
                    d.status = "failed"
                    d.next_at = self.clock() + BACKOFF_S[d.attempts]
                    self.counters["retries"] += 1
                    self.queue.append(d)
        return len(due)

    def _echo(self, s: Subscription, headers: Dict[str, str], body: str) -> Tuple[int, str]:
        ok = verify(s.secret, headers, body, self.clock())
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        self.echo.append({"at": self.clock(), "verified": ok, "id": headers["webhook-id"],
                          "type": (data or {}).get("type"), "data": (data or {}).get("data")})
        return (200, "") if ok else (401, "signature did not verify")

    @staticmethod
    def _http_post(url: str, headers: Dict[str, str], body: str) -> Tuple[int, str]:
        why = check_url(url)            # re-checked at send time: DNS can change after subscribing
        if why:
            return 0, why
        try:
            import httpx
            r = httpx.post(url, content=body.encode(), headers=headers, timeout=5.0, follow_redirects=False)
            return r.status_code, "" if r.status_code < 300 else r.text[:120]
        except Exception as exc:
            return 0, f"{type(exc).__name__}: {str(exc)[:120]}"

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"subscriptions": [s.public() for s in self.subs.values()], "queued": len(self.queue),
                    "counters": dict(self.counters), "event_types": list(EVENT_TYPES),
                    "recent": [d.public() for d in list(self.log)[-30:]][::-1],
                    "dead": [d.public() for d in list(self.dead)[-10:]][::-1],
                    "echo": list(self.echo)[-20:][::-1]}
