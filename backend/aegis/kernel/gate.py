"""The gate — one reference monitor that every call to the kernel crosses.

WHY THIS EXISTS
---------------
Of the kernel's ~47 write routes, exactly one used to pass through a gate (the
Z3-checked syscall trap), and nothing checked WHO was calling anything. This
module is the single place that decides, for every HTTP request and WebSocket
under ``/aegis`` and for every in-process call the agent plane makes:

    resolve   which syscall is this?          (the committed ABI table)
    identify  who is calling?                 (session token → Credential)
    decide    may they?                        (capability in the credential)
    account   count it, time it                (per-syscall counters, µs overhead)
    record    journal writes and refusals      (reads are counted, not journalled)

MODES (``AEGIS_GATE_MODE``)
    enforce  refusals are returned (403 EPERM / 401 EAUTH) and the handler never runs
    audit    every call is served; what WOULD have been refused is counted and
             journalled. This is how the gate ships first: the public deployment's
             pages do not send tokens yet, and a gate that broke them on day one
             would be switched off rather than fixed.
    off      no checks (tests only)

WHAT THE GATE IS NOT. It is a reference monitor in one process, enforced by
routing, not by hardware; a handler reached WITHOUT the middleware (a unit test
calling a function directly) is not gated. The in-process path the agent plane
uses (:meth:`Gate.enter`) closes the one production route that bypassed HTTP.
Its overhead is measured and reported, not assumed.

The credential for the current call is published in ``users.CURRENT`` so a
handler can refine the decision — the syscall trap checks the per-trap capability
(``order:treat`` for treat) after the route-level check has passed.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from backend.aegis.kernel.abi import ABI, ABIEntry
from backend.aegis.kernel.journal import INFO, NOTICE, WARN, Journal, KJOURNAL
from backend.aegis.kernel.users import (CURRENT, GUEST, AuthError, Credential,
                                        PermissionDenied, UserDB)
from backend.aegis.rbac import CAP_MIN_RING

MODES = ("off", "audit", "enforce")
WS_PROTOCOL = "aegis"                       # the subprotocol the server selects
WS_TOKEN_PREFIX = "aegis.session."          # a client offers "aegis.session.<token>"


@dataclass
class Decision:
    allowed: bool                           # the policy's verdict
    served: bool                            # whether the call proceeds (audit serves refusals)
    entry: Optional[ABIEntry]
    cred: Credential
    reason: str = ""
    errno: str = ""

    def to_dict(self) -> Dict[str, Any]:
        e = self.entry
        return {"allowed": self.allowed, "served": self.served, "errno": self.errno,
                "reason": self.reason, "uid": self.cred.username, "role": self.cred.role,
                "syscall": e.name if e else None, "nr": e.nr if e else None,
                "capability": e.cap if e else None}


class _Stat:
    __slots__ = ("calls", "denied", "would_deny", "overhead_ns")

    def __init__(self) -> None:
        self.calls = 0
        self.denied = 0
        self.would_deny = 0
        self.overhead_ns: List[int] = []


class Gate:
    def __init__(self, abi: ABI, users: UserDB, journal: Optional[Journal] = None,
                 mode: Optional[str] = None) -> None:
        self.abi = abi
        self.users = users
        self.journal = journal if journal is not None else KJOURNAL
        m = (mode or os.getenv("AEGIS_GATE_MODE") or "audit").strip().lower()
        self.mode = m if m in MODES else "audit"
        self.stats: Dict[int, _Stat] = {}
        self.unmapped: Dict[str, int] = {}
        self.policy_proof: Optional[Dict[str, Any]] = None
        self._overhead: List[int] = []                # rolling sample, all syscalls
        self._dedupe: Dict[Tuple[str, int, str], float] = {}
        self._lock = threading.Lock()

    # -- decision ---------------------------------------------------------- #
    def credential_from_token(self, token: Optional[str]) -> Credential:
        """Resolve a token; a bad one becomes an ``auth_error`` guest credential so
        the caller (middleware / enter) can refuse or downgrade by mode."""
        try:
            return self.users.credential(token)
        except AuthError as exc:
            c = GUEST
            return Credential(**{**c.__dict__, "auth_error": str(exc)})

    def decide(self, cred: Credential, entry: Optional[ABIEntry]) -> Decision:
        if self.mode == "off":
            return Decision(True, True, entry, cred)
        if entry is None:
            # No entry = not in the table. The router 404s unknown paths; a route
            # that EXISTS without an entry is what verify_abi fails the build on.
            return Decision(True, True, None, cred, reason="unmapped")
        if cred.auth_error and entry.klass != "auth":
            ok = False
            reason, errno = f"presented session rejected: {cred.auth_error}", "EAUTH"
        elif cred.has(entry.cap):
            return Decision(True, True, entry, cred)
        else:
            ok = False
            need = CAP_MIN_RING.get(entry.cap, "?")
            reason = (f"{cred.username} ({cred.role}, ring {cred.ring}) lacks "
                      f"{entry.cap} (ring {need})"
                      + (" — the session is locked" if cred.locked else ""))
            errno = "EPERM"
        return Decision(ok, self.mode == "audit", entry, cred, reason, errno)

    def record(self, d: Decision, overhead_ns: int, status: int = 0) -> None:
        e = d.entry
        if e is None:
            return
        with self._lock:
            st = self.stats.get(e.nr)
            if st is None:
                st = self.stats[e.nr] = _Stat()
            st.calls += 1
            st.overhead_ns.append(overhead_ns)
            del st.overhead_ns[:-128]
            self._overhead.append(overhead_ns)
            del self._overhead[:-2048]
            if not d.allowed:
                if d.served:
                    st.would_deny += 1
                else:
                    st.denied += 1
        cred = d.cred
        if not d.allowed:
            # One row per (who, which syscall, verdict) per window: a page polling a
            # refused endpoint every 4 s must not become 900 journal rows an hour.
            key = (cred.username, e.nr, "would_deny" if d.served else "deny")
            now = time.time()
            window = 300.0 if d.served else 60.0
            if now - self._dedupe.get(key, 0.0) >= window:
                self._dedupe[key] = now
                self.journal.append("syscall", "gate.would_deny" if d.served else "gate.deny",
                                    severity=NOTICE if d.served else WARN,
                                    uid=cred.username, pid=cred.pid, nr=e.nr, syscall=e.name,
                                    capability=e.cap, role=cred.role, errno=d.errno,
                                    reason=d.reason, mode=self.mode)
        elif e.klass in ("write", "egress") and e.refine != "syscall":
            # The syscall trap journals itself with its certificate; everything else
            # that mutates is recorded here, with who did it.
            self.journal.append("syscall", e.name, severity=INFO, uid=cred.username,
                                pid=cred.pid, nr=e.nr, status=status,
                                breakglass=bool(cred.breakglass), acting_for=cred.ruid)
            if cred.breakglass:
                self.users.note_elevated_call(cred, e.name)

    # -- in-process entry (the agent plane) ------------------------------- #
    def enter(self, cred: Credential, method: str, path: str) -> Decision:
        """Gate a call that does not arrive over HTTP. Raises PermissionDenied when
        the gate would refuse and is enforcing."""
        t0 = time.perf_counter_ns()
        d = self.decide(cred, self.abi.resolve(method, path.split("?")[0]))
        self.record(d, time.perf_counter_ns() - t0)
        if not d.served:
            raise PermissionDenied(d.reason, d.entry.cap if d.entry else "", cred)
        return d

    def require(self, cap: str, cred: Optional[Credential] = None, what: str = "") -> bool:
        """A handler's refinement: the route passed, but THIS operation needs ``cap``.
        Returns True if allowed; raises PermissionDenied when enforcing a refusal;
        returns False (and journals) when auditing one."""
        cred = cred if cred is not None else CURRENT.get() or GUEST
        if self.mode == "off" or cred.has(cap):
            return True
        reason = f"{cred.username} ({cred.role}) lacks {cap}" + (f" for {what}" if what else "")
        key = (cred.username, -1, cap)
        now = time.time()
        if now - self._dedupe.get(key, 0.0) >= 60.0:
            self._dedupe[key] = now
            self.journal.append("syscall", "gate.deny" if self.mode == "enforce"
                                else "gate.would_deny",
                                severity=WARN if self.mode == "enforce" else NOTICE,
                                uid=cred.username, pid=cred.pid, capability=cap,
                                what=what, mode=self.mode)
        if self.mode == "enforce":
            raise PermissionDenied(reason, cap, cred)
        return False

    # -- views --------------------------------------------------------------- #
    @staticmethod
    def _pct(xs: List[int], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, int(q * (len(s) - 1)))] / 1000.0      # ns → µs

    def status(self, top: int = 12) -> Dict[str, Any]:
        with self._lock:
            ov = list(self._overhead)
            rows = [(nr, st.calls, st.denied, st.would_deny, list(st.overhead_ns))
                    for nr, st in self.stats.items()]
        calls = sum(r[1] for r in rows)
        busiest = sorted(rows, key=lambda r: r[1], reverse=True)[:top]
        refused = sorted([r for r in rows if r[2] or r[3]],
                         key=lambda r: r[2] + r[3], reverse=True)[:top]

        def row(r):
            e = self.abi.by_nr.get(r[0])
            return {"nr": r[0], "syscall": e.name if e else "?", "class": e.klass if e else "?",
                    "cap": e.cap if e else "?", "calls": r[1], "denied": r[2],
                    "would_deny": r[3], "p50_us": round(self._pct(r[4], 0.5), 1)}

        return {
            "mode": self.mode,
            "calls": calls,
            "denied": sum(r[2] for r in rows),
            "would_deny": sum(r[3] for r in rows),
            "overhead_us": {"p50": round(self._pct(ov, 0.50), 1),
                            "p99": round(self._pct(ov, 0.99), 1),
                            "samples": len(ov),
                            "measures": "resolve + authenticate + decide, per call "
                                        "(excludes the handler)"},
            "abi": self.abi.summary(),
            "busiest": [row(r) for r in busiest],
            "refusals": [row(r) for r in refused],
            "policy": None if self.policy_proof is None else {
                "sound": self.policy_proof.get("sound"),
                "properties": [{k: p[k] for k in ("property", "proven")}
                               for p in self.policy_proof.get("properties", [])],
                "ms": self.policy_proof.get("ms")},
        }


#: The kernel's gate. ``server.py`` configures it at boot; until then the
#: middleware passes everything through (the ASGI app is built before the kernel).
GATE: Optional[Gate] = None


def configure(abi: ABI, users: UserDB, journal: Optional[Journal] = None,
              mode: Optional[str] = None) -> Gate:
    global GATE
    GATE = Gate(abi, users, journal=journal, mode=mode)
    return GATE


# --------------------------------------------------------------------------- #
# The ASGI middleware
# --------------------------------------------------------------------------- #
def _headers(scope: Dict[str, Any]) -> Dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers") or []}


def _token_from(scope: Dict[str, Any], headers: Dict[str, str]) -> Tuple[Optional[str], bool]:
    """(token, offered_ws_protocol). Tokens travel in the Authorization header over
    HTTP, and in a WebSocket subprotocol (browsers cannot set headers on a socket).
    Never in the query string: URLs end up in access logs."""
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None, False
    x = headers.get("x-aegis-session")
    if x:
        return x.strip(), False
    if scope["type"] == "websocket":
        offered = [p.strip() for p in headers.get("sec-websocket-protocol", "").split(",")
                   if p.strip()]
        for p in offered:
            if p.startswith(WS_TOKEN_PREFIX):
                return p[len(WS_TOKEN_PREFIX):], WS_PROTOCOL in offered
        return None, WS_PROTOCOL in offered
    return None, False


class GateMiddleware:
    """Pure ASGI (not BaseHTTPMiddleware: that buffers streaming responses, and the
    consult streams). Add it BEFORE CORS so CORS stays outermost and a 403 still
    carries the headers the browser needs to read it."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        gate = GATE
        if (gate is None or scope["type"] not in ("http", "websocket")
                or not scope.get("path", "").startswith("/aegis")
                or scope.get("method") == "OPTIONS"):
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter_ns()
        headers = _headers(scope)
        token, offered_proto = _token_from(scope, headers)
        method = "WS" if scope["type"] == "websocket" else scope.get("method", "GET")
        entry = gate.abi.resolve(method, scope["path"])
        cred = gate.credential_from_token(token)
        if cred.auth_error and gate.mode != "enforce":
            # audit mode: a stale token is served as the guest it now is, but counted
            cred_for_call = cred
        else:
            cred_for_call = cred
        d = gate.decide(cred_for_call, entry)
        overhead = time.perf_counter_ns() - t0

        if not d.served:
            gate.record(d, overhead, status=403)
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401 if d.errno == "EAUTH" else 4403,
                            "reason": d.errno})
                return
            body = json.dumps({"status": "error", **d.to_dict()}).encode()
            await send({"type": "http.response.start",
                        "status": 401 if d.errno == "EAUTH" else 403,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return

        scope.setdefault("state", {})["aegis_cred"] = cred_for_call
        token_ctx = CURRENT.set(cred_for_call)
        status_box = {"status": 0}

        async def send_wrapper(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_box["status"] = message.get("status", 0)
            elif (message["type"] == "websocket.accept" and offered_proto
                  and not message.get("subprotocol")):
                # The client offered "aegis" alongside its token; a browser drops a
                # socket whose accept does not select one of the offered protocols.
                message = {**message, "subprotocol": WS_PROTOCOL}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            CURRENT.reset(token_ctx)
            gate.record(d, overhead, status=status_box["status"])
