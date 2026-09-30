"""Users and sessions — who is calling, established once and carried everywhere.

WHY THIS EXISTS
---------------
``rbac.py`` had a complete role model and nothing to attach it to. Its login
minted a token for whatever role the request named, with no credential; no route
read a token; the frontend never sent one; and the gate object built around it was
constructed and never called. Every write in the OS was reachable by anyone who
could reach the URL.

This module is the part that was missing: ACCOUNTS (the role belongs to the
account, never to the request), SESSIONS the kernel can see, revoke and lock, a
CREDENTIAL that the gate attaches to every call, and the two principals that are
not people — the view-only ``guest`` and the AI agent ``dr-aegis``.

THE AGENT IS A USER, NOT ROOT. When Dr. Aegis acts, it acts on a clinician's
behalf: its effective capabilities are the clinician's capabilities minus
``AGENT_DENY`` (euid = dr-aegis, ruid = the clinician). It can never do more than
the human who asked, and never the few things a model must not do at all.

BREAK-GLASS is the hospital pattern for emergency access: a stated reason, a hard
expiry (≤ 15 min), a fixed set of added capabilities that can never include
prescribing, policy, software or session administration (``FORBIDDEN_ELEVATION``,
proven disjoint at boot), every call during it flagged, and a review queue for the
chief afterwards.

HONEST SCOPE: demo-grade authentication. The seeded staff accounts sign in without
a passphrase while ``AEGIS_DEMO_LOGIN`` is on (the default, and the sign-in screen
says so). Accounts supplied through ``AEGIS_ACCOUNTS`` use PBKDF2-SHA256
passphrases. Tokens are HMAC-signed and expire; revocations are held in memory, so
a restart forgets them (tokens still expire). This is not SSO, not MFA, not IAM.
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.aegis.kernel.journal import CRIT, INFO, NOTICE, WARN, Journal, KJOURNAL
from backend.aegis.kernel.proc import SESSION, SIGTERM, ProcessTable, PROCS
from backend.aegis.rbac import (AGENT_DENY, BREAKGLASS_CAPS, FORBIDDEN_ELEVATION,
                                GRANTS, PUBLIC_CAPS, RBAC, RING)

GUEST_USER = "guest"
AGENT_USER = "dr-aegis"
KERNEL_USER = "root"

BREAKGLASS_MAX_MIN = 15.0
BREAKGLASS_MIN_REASON = 20


class AuthError(Exception):
    """EAUTH — no valid identity (bad passphrase, forged/expired/revoked token)."""


class PermissionDenied(Exception):
    """EPERM — a valid identity that lacks the capability."""

    def __init__(self, message: str, capability: str = "", cred: Optional["Credential"] = None):
        super().__init__(message)
        self.capability = capability
        self.cred = cred


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Account:
    uid: int
    username: str
    display: str
    role: str
    kind: str = "human"            # human | guest | service | kernel
    loginable: bool = True
    demo: bool = False             # signs in without a passphrase while demo login is on
    title: str = ""
    pbkdf2: str = ""               # "pbkdf2_sha256$iterations$salt_b64$hash_b64"

    def public(self) -> Dict[str, Any]:
        return {"uid": self.uid, "username": self.username, "display": self.display,
                "role": self.role, "ring": RING.get(self.role, 0), "title": self.title,
                "kind": self.kind, "demo": self.demo, "passphrase": bool(self.pbkdf2)}


#: Fictional demo staff — deliberately no overlap with the ward's patient aliases.
DEMO_STAFF = (
    Account(1001, "rn.patel", "Priya Patel, RN", "nurse", demo=True, title="Charge nurse"),
    Account(1002, "rx.chen", "Wei Chen, PharmD", "pharmacist", demo=True,
            title="Clinical pharmacist"),
    Account(1003, "dr.alvarez", "Dr. Sofia Alvarez", "attending", demo=True,
            title="Attending intensivist"),
    Account(1004, "dr.whitfield", "Dr. James Whitfield", "chief", demo=True,
            title="Chief of critical care"),
    Account(1005, "dr.lindqvist", "Dr. Anna Lindqvist", "remote_specialist", demo=True,
            title="Remote infectious-disease consult"),
    # the rest of the paging roster (backend/aegis/paging/hospital.py): every
    # clinician who can receive a page is a user of the OS, so answering one is an
    # authenticated act the gate can check and the journal can attribute
    Account(1006, "dr.okafor", "Dr. Daniel Okafor", "attending", demo=True, title="Intensivist"),
    Account(1007, "dr.nakamura", "Dr. Kenji Nakamura", "attending", demo=True, title="Hospitalist"),
    Account(1008, "dr.reyes", "Dr. Maria Reyes", "attending", demo=True, title="Hospitalist"),
    Account(1009, "dr.shah", "Dr. Anika Shah", "attending", demo=True, title="Emergency physician"),
    Account(1010, "dr.brennan", "Dr. Liam Brennan", "attending", demo=True, title="Emergency physician"),
    Account(1011, "dr.adeyemi", "Dr. Tolu Adeyemi", "attending", demo=True, title="Cardiologist"),
    Account(1012, "dr.kowalski", "Dr. Ewa Kowalski", "attending", demo=True, title="Neurologist"),
    Account(1013, "dr.moreau", "Dr. Julien Moreau", "attending", demo=True, title="General surgeon"),
    Account(1014, "dr.fischer", "Dr. Hannah Fischer", "attending", demo=True, title="Anesthesiologist"),
    Account(1015, "dr.haddad", "Dr. Rami Haddad", "attending", demo=True, title="Infectious disease"),
    Account(1016, "rn.ortiz", "Luis Ortiz, RN", "nurse", demo=True, title="Rapid response nurse"),
    Account(1017, "rn.kim", "Grace Kim, RN", "nurse", demo=True, title="Rapid response nurse"),
    Account(1018, "rx.silva", "Ana Silva, PharmD", "pharmacist", demo=True, title="ICU pharmacist"),
)
SYSTEM_ACCOUNTS = (
    Account(0, KERNEL_USER, "AegisOS kernel", "chief", kind="kernel", loginable=False),
    Account(900, AGENT_USER, "Dr. Aegis (AI agent)", "guest", kind="service", loginable=False,
            title="Acts only on a clinician's behalf"),
    Account(65534, GUEST_USER, "Guest (view only)", "guest", kind="guest", loginable=True,
            demo=True, title="View the ward; change nothing"),
)


def hash_passphrase(passphrase: str, iterations: int = 200_000) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, iterations)
    return "pbkdf2_sha256$%d$%s$%s" % (iterations, base64.b64encode(salt).decode(),
                                       base64.b64encode(dk).decode())


def check_passphrase(passphrase: str, encoded: str) -> bool:
    try:
        algo, iters, salt_b64, hash_b64 = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", passphrase.encode(),
                                 base64.b64decode(salt_b64), int(iters))
        return hmac.compare_digest(dk, base64.b64decode(hash_b64))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Credentials — what the gate attaches to every call
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Credential:
    uid: int
    username: str
    display: str
    role: str
    ring: int
    caps: frozenset
    kind: str = "human"
    sid: str = ""
    pid: int = 0
    locked: bool = False
    breakglass: Optional[Dict[str, Any]] = None
    ruid: Optional[str] = None          # for the agent: the human it acts for
    auth_error: Optional[str] = None    # set when a presented token was rejected

    def has(self, cap: str) -> bool:
        return cap in self.caps

    def to_dict(self) -> Dict[str, Any]:
        return {"uid": self.uid, "username": self.username, "display": self.display,
                "role": self.role, "ring": self.ring, "kind": self.kind,
                "capabilities": sorted(self.caps), "sid": self.sid[:8] if self.sid else "",
                "pid": self.pid, "locked": self.locked, "breakglass": self.breakglass,
                "acting_for": self.ruid, "auth_error": self.auth_error}


def _cred(acct: Account, caps: frozenset, **kw: Any) -> Credential:
    return Credential(uid=acct.uid, username=acct.username, display=acct.display,
                      role=acct.role, ring=RING.get(acct.role, 0), caps=caps,
                      kind=acct.kind, **kw)


_GUEST_ACCT = next(a for a in SYSTEM_ACCOUNTS if a.username == GUEST_USER)
_KERNEL_ACCT = next(a for a in SYSTEM_ACCOUNTS if a.username == KERNEL_USER)
_AGENT_ACCT = next(a for a in SYSTEM_ACCOUNTS if a.username == AGENT_USER)

GUEST = _cred(_GUEST_ACCT, PUBLIC_CAPS)
#: Ring-0 kernel code (the loops). Never attached to an external request.
KERNEL = Credential(uid=0, username=KERNEL_USER, display="AegisOS kernel", role="kernel",
                    ring=5, caps=frozenset().union(*GRANTS.values()), kind="kernel")

#: The caller of the current request. Set by the gate middleware for every HTTP and
#: WebSocket scope; FastAPI's threadpool and asyncio.to_thread copy it into the
#: handler. Plain threads and run_in_executor do NOT — pass a credential explicitly.
CURRENT: "contextvars.ContextVar[Optional[Credential]]" = contextvars.ContextVar(
    "aegis_credential", default=None)


def current_credential(default: Optional[Credential] = None) -> Credential:
    c = CURRENT.get()
    return c if c is not None else (default if default is not None else GUEST)


def agent_caps(role: str) -> frozenset:
    """What Dr. Aegis may do on behalf of a clinician holding ``role``."""
    return frozenset(GRANTS.get(role, frozenset()) - AGENT_DENY)


def elevated_caps(role: str) -> frozenset:
    """A role's capabilities during break-glass: its own, plus the emergency set
    minus anything forbidden to confer."""
    return frozenset(GRANTS.get(role, frozenset())
                     | (BREAKGLASS_CAPS - FORBIDDEN_ELEVATION))


# --------------------------------------------------------------------------- #
# Sessions + break-glass
# --------------------------------------------------------------------------- #
@dataclass
class BreakGlass:
    id: str
    username: str
    role: str
    reason: str
    started: float
    until: float
    calls: int = 0
    ended_early: bool = False
    reviewed: bool = False
    verdict: str = ""
    reviewer: str = ""
    note: str = ""

    def active(self, now: float) -> bool:
        return not self.ended_early and now < self.until

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        return {"id": self.id, "username": self.username, "role": self.role,
                "reason": self.reason, "started": round(self.started, 1),
                "until": round(self.until, 1), "active": self.active(now),
                "remaining_s": max(0, round(self.until - now)), "calls": self.calls,
                "reviewed": self.reviewed, "verdict": self.verdict,
                "reviewer": self.reviewer, "note": self.note}


@dataclass
class SessionState:
    sid: str
    username: str
    role: str
    created: float
    expires: float
    last_seen: float
    pid: int = 0
    locked: bool = False
    client: str = ""
    breakglass: Optional[BreakGlass] = None

    def to_dict(self, now: float) -> Dict[str, Any]:
        return {"sid": self.sid[:8], "username": self.username, "role": self.role,
                "created": round(self.created, 1), "idle_s": round(now - self.last_seen),
                "expires_in_s": max(0, round(self.expires - now)), "pid": self.pid,
                "locked": self.locked, "client": self.client,
                "breakglass": self.breakglass.to_dict(now) if self.breakglass else None}


class UserDB:
    """Accounts, sessions, break-glass. One per kernel (``server.py`` builds it)."""

    def __init__(self, rbac: RBAC, procs: Optional[ProcessTable] = None,
                 journal: Optional[Journal] = None, demo_login: Optional[bool] = None,
                 session_ttl_s: float = 8 * 3600.0, idle_lock_s: float = 3600.0,
                 clock: Callable[[], float] = time.time) -> None:
        self.rbac = rbac
        self.procs = procs if procs is not None else PROCS
        self.journal = journal if journal is not None else KJOURNAL
        self.demo_login = (os.getenv("AEGIS_DEMO_LOGIN", "1") != "0"
                           if demo_login is None else bool(demo_login))
        self.session_ttl_s = float(session_ttl_s)
        # A backstop only: the desktop locks on INPUT idleness and calls lock().
        # The server cannot tell a human who walked away from a screen still polling.
        self.idle_lock_s = float(idle_lock_s)
        self.clock = clock
        self.accounts: Dict[str, Account] = {}
        for a in SYSTEM_ACCOUNTS + DEMO_STAFF + tuple(self._env_accounts()):
            self.accounts[a.username] = a
        self.sessions: Dict[str, SessionState] = {}
        self.revoked: set = set()
        self.breakglass_log: List[BreakGlass] = []
        self._lock = threading.RLock()

    @staticmethod
    def _env_accounts() -> List[Account]:
        raw = os.getenv("AEGIS_ACCOUNTS", "").strip()
        if not raw:
            return []
        out = []
        try:
            for i, d in enumerate(json.loads(raw)):
                if d.get("role") not in RING or not d.get("pbkdf2"):
                    continue
                out.append(Account(2000 + i, d["username"], d.get("display", d["username"]),
                                   d["role"], title=d.get("title", ""), pbkdf2=d["pbkdf2"]))
        except Exception as exc:                       # a bad env var must not stop boot
            print(f"[users] AEGIS_ACCOUNTS ignored: {exc}")
        return out

    # -- authentication --------------------------------------------------- #
    def authenticate(self, username: str, passphrase: Optional[str] = None) -> Account:
        acct = self.accounts.get((username or "").strip())
        if acct is None or not acct.loginable:
            raise AuthError("no such account")
        if acct.pbkdf2:
            if not passphrase or not check_passphrase(passphrase, acct.pbkdf2):
                raise AuthError("wrong passphrase")
            return acct
        if acct.demo and self.demo_login:
            return acct
        raise AuthError("this account requires a passphrase (demo sign-in is disabled)")

    def login(self, username: str, passphrase: Optional[str] = None,
              client: str = "") -> Tuple[str, SessionState, Credential]:
        try:
            acct = self.authenticate(username, passphrase)
        except AuthError as exc:
            self.journal.append("auth", "login.failed", severity=WARN,
                                uid=str(username or "?"), reason=str(exc))
            raise
        now = self.clock()
        sid = uuid.uuid4().hex
        token = self.rbac.login(acct.username, acct.role, sid=sid,
                                ttl_s=self.session_ttl_s).token
        state = SessionState(sid=sid, username=acct.username, role=acct.role,
                             created=now, expires=now + self.session_ttl_s,
                             last_seen=now, client=client[:80])
        state.pid = self._spawn_session_process(state)
        with self._lock:
            self.sessions[sid] = state
        self.journal.append("auth", "login", severity=NOTICE, uid=acct.username,
                            pid=state.pid, role=acct.role, demo=acct.demo and not acct.pbkdf2)
        return token, state, self.credential_for(state)

    def login_role(self, role: str, client: str = "") -> Tuple[str, SessionState, Credential]:
        """Sign in as the demo account FOR a role (the legacy /aegis/rbac/login).
        The role picks an account; the account — not the request — carries the role."""
        acct = next((a for a in DEMO_STAFF if a.role == role), None)
        if acct is None:
            if role == "guest":
                acct = _GUEST_ACCT
            else:
                raise AuthError(f"no demo account for role {role!r}")
        return self.login(acct.username, client=client)

    def _spawn_session_process(self, state: SessionState) -> int:
        def on_signal(sig: str, _sid: str = state.sid) -> str:
            self._end(_sid, reason="SIGTERM")
            return "signed out"
        return self.procs.spawn(f"session {state.username}", SESSION, uid=state.username,
                                sid=state.sid[:8], role=state.role,
                                on_signal=on_signal, signals=(SIGTERM,)).pid

    # -- the credential the gate attaches ------------------------------------ #
    def credential(self, token: Optional[str]) -> Credential:
        """Resolve a presented token. No token → guest. A token that fails →
        AuthError (the gate decides whether that is refused or downgraded)."""
        if not token:
            return GUEST
        s = self.rbac.verify_token(token)
        if s is None:
            raise AuthError("invalid or expired session token")
        sid = getattr(s, "sid", "")
        if not sid or sid in self.revoked:
            raise AuthError("session has ended")
        acct = self.accounts.get(s.user)
        if acct is None or acct.role != s.role:
            raise AuthError("token does not match its account")
        with self._lock:
            state = self.sessions.get(sid)
            if state is None:
                # A valid token from before a restart (AEGIS_SESSION_SECRET kept the
                # signature valid): re-materialise the session rather than force a
                # sign-out on every free-tier spin-down.
                now = self.clock()
                state = SessionState(sid=sid, username=acct.username, role=acct.role,
                                     created=s.iat, expires=s.exp, last_seen=now,
                                     client="resumed")
                state.pid = self._spawn_session_process(state)
                self.sessions[sid] = state
                self.journal.append("auth", "session.resumed", severity=INFO,
                                    uid=acct.username, pid=state.pid)
            now = self.clock()
            if not state.locked and now - state.last_seen > self.idle_lock_s:
                state.locked = True
                self.journal.append("auth", "lock.idle", severity=NOTICE,
                                    uid=state.username, pid=state.pid)
            if not state.locked:
                state.last_seen = now
            return self.credential_for(state)

    def credential_for(self, state: SessionState) -> Credential:
        acct = self.accounts[state.username]
        now = self.clock()
        bg = state.breakglass if state.breakglass and state.breakglass.active(now) else None
        if state.locked:
            caps = PUBLIC_CAPS                 # a locked screen may be looked at, not used
        elif bg is not None:
            caps = elevated_caps(acct.role)
        else:
            caps = frozenset(GRANTS.get(acct.role, frozenset()))
        return _cred(acct, caps, sid=state.sid, pid=state.pid, locked=state.locked,
                     breakglass=bg.to_dict(now) if bg else None)

    def agent_credential(self, on_behalf_of: Optional[Credential] = None) -> Credential:
        """Dr. Aegis acting for a clinician: euid dr-aegis, caps = theirs − AGENT_DENY."""
        ruid = on_behalf_of if on_behalf_of is not None else GUEST
        caps = frozenset(ruid.caps - AGENT_DENY)
        return Credential(uid=_AGENT_ACCT.uid, username=AGENT_USER, display=_AGENT_ACCT.display,
                          role=ruid.role, ring=ruid.ring, caps=caps, kind="agent",
                          sid=ruid.sid, pid=ruid.pid, locked=ruid.locked,
                          breakglass=None, ruid=ruid.username)

    # -- lifecycle ---------------------------------------------------------- #
    def _end(self, sid: str, reason: str) -> bool:
        with self._lock:
            state = self.sessions.pop(sid, None)
            self.revoked.add(sid)
        if state is None:
            return False
        if state.breakglass and state.breakglass.active(self.clock()):
            state.breakglass.ended_early = True
        try:
            p = self.procs.get(state.pid)
            if p.state not in ("ZOMBIE", "EXITED"):
                self.procs.exit(state.pid, code=0)
        except KeyError:
            pass
        self.journal.append("auth", "logout", severity=NOTICE, uid=state.username,
                            pid=state.pid, reason=reason)
        return True

    def logout(self, cred: Credential) -> bool:
        return bool(cred.sid) and self._end(cred.sid, reason="signed out")

    def revoke(self, actor: Credential, sid_prefix: str) -> int:
        """session:admin — end someone else's session."""
        if not actor.has("session:admin"):
            raise PermissionDenied("revoking another session needs session:admin",
                                   "session:admin", actor)
        n = 0
        for sid in [s for s in list(self.sessions) if s.startswith(sid_prefix)]:
            n += self._end(sid, reason=f"revoked by {actor.username}")
        return n

    def lock(self, cred: Credential) -> bool:
        state = self.sessions.get(cred.sid)
        if state is None:
            return False
        state.locked = True
        self.journal.append("auth", "lock", severity=INFO, uid=state.username, pid=state.pid)
        return True

    def unlock(self, cred: Credential, passphrase: Optional[str] = None) -> Credential:
        state = self.sessions.get(cred.sid)
        if state is None:
            raise AuthError("no session to unlock")
        self.authenticate(state.username, passphrase)        # the same proof as sign-in
        state.locked = False
        state.last_seen = self.clock()
        self.journal.append("auth", "unlock", severity=INFO, uid=state.username, pid=state.pid)
        return self.credential_for(state)

    # -- break-glass ---------------------------------------------------------- #
    def breakglass(self, cred: Credential, reason: str, minutes: float = 10.0) -> BreakGlass:
        state = self.sessions.get(cred.sid)
        if state is None or cred.kind != "human":
            raise PermissionDenied("break-glass needs a signed-in person", "breakglass:invoke", cred)
        base = GRANTS.get(state.role, frozenset())
        if "breakglass:invoke" not in base:
            self.journal.append("auth", "breakglass.denied", severity=WARN, uid=cred.username,
                                pid=cred.pid, reason="role lacks breakglass:invoke")
            raise PermissionDenied(f"{state.role} cannot invoke break-glass",
                                   "breakglass:invoke", cred)
        reason = (reason or "").strip()
        if len(reason) < BREAKGLASS_MIN_REASON:
            raise ValueError(f"state the emergency (at least {BREAKGLASS_MIN_REASON} characters)")
        minutes = max(1.0, min(float(minutes), BREAKGLASS_MAX_MIN))
        now = self.clock()
        bg = BreakGlass(id=uuid.uuid4().hex[:10], username=state.username, role=state.role,
                        reason=reason[:500], started=now, until=now + minutes * 60.0)
        state.breakglass = bg
        self.breakglass_log.append(bg)
        self.journal.append("auth", "breakglass.invoke", severity=CRIT, uid=state.username,
                            pid=state.pid, id=bg.id, minutes=minutes, reason=bg.reason,
                            adds=sorted(elevated_caps(state.role) - base))
        return bg

    def note_elevated_call(self, cred: Credential, syscall: str) -> None:
        state = self.sessions.get(cred.sid)
        if state and state.breakglass and state.breakglass.active(self.clock()):
            state.breakglass.calls += 1

    def review_breakglass(self, cred: Credential, bg_id: str, verdict: str,
                          note: str = "") -> BreakGlass:
        if not cred.has("breakglass:review"):
            raise PermissionDenied("reviewing break-glass needs breakglass:review",
                                   "breakglass:review", cred)
        bg = next((b for b in self.breakglass_log if b.id == bg_id), None)
        if bg is None:
            raise KeyError(f"no break-glass record {bg_id!r}")
        if bg.username == cred.username:
            raise PermissionDenied("nobody reviews their own break-glass",
                                   "breakglass:review", cred)
        if verdict not in ("appropriate", "inappropriate"):
            raise ValueError("verdict must be 'appropriate' or 'inappropriate'")
        bg.reviewed, bg.verdict, bg.reviewer, bg.note = True, verdict, cred.username, note[:500]
        self.journal.append("auth", "breakglass.review", severity=NOTICE, uid=cred.username,
                            id=bg.id, subject=bg.username, verdict=verdict)
        return bg

    # -- views ---------------------------------------------------------------- #
    def public_accounts(self) -> List[Dict[str, Any]]:
        return [a.public() for a in self.accounts.values()
                if a.loginable and a.kind in ("human", "guest")]

    def who(self) -> List[Dict[str, Any]]:
        now = self.clock()
        with self._lock:
            rows = list(self.sessions.values())
        return [s.to_dict(now) for s in sorted(rows, key=lambda s: s.created)]

    def status(self) -> Dict[str, Any]:
        now = self.clock()
        return {"accounts": len(self.accounts), "sessions": len(self.sessions),
                "demo_login": self.demo_login,
                "breakglass_active": sum(1 for b in self.breakglass_log if b.active(now)),
                "breakglass_unreviewed": sum(1 for b in self.breakglass_log if not b.reviewed),
                "secret": "AEGIS_SESSION_SECRET" if os.getenv("AEGIS_SESSION_SECRET")
                          else "random per boot (a restart signs everyone out)"}


if __name__ == "__main__":                        # pragma: no cover
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "hash":
        print(hash_passphrase(sys.argv[2]))
    else:
        print("usage: python -m backend.aegis.kernel.users hash <passphrase>")
