"""The OS surface: accounts and sessions, break-glass, the process table, journal, boot record and filesystem.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, WebSocket

import backend.aegis.server as S
from backend.api.schemas import (
    BreakGlassIn,
    BreakGlassReviewIn,
    FsPathIn,
    FsWriteIn,
    GateModeIn,
    KillIn,
    LoginIn,
    RevokeIn,
    UnlockIn,
)
from backend.aegis.server import (
    ABI_TABLE,
    BOOT,
    GATE,
    KJOURNAL,
    KVFS,
    NoSuchProcess,
    OS_HUB,
    PROCS,
    SignalFailed,
    SignalUnsupported,
    USERS,
    _AuthError,
    _PermissionDenied,
    _cred,
    _fs_error,
    _session_payload,
    procfs,
)

router = APIRouter()


@router.get("/aegis/proc")
def proc_list() -> Dict[str, Any]:
    """The /proc introspection namespace (every live path + value)."""
    return {"paths": procfs.list(), "tree": procfs.tree()}


@router.get("/aegis/proc/{path:path}")
def proc_read(path: str) -> Dict[str, Any]:
    """Read one /proc path (e.g. services, triage, sched, mem, audit, cage)."""
    full = "/proc/" + path.lstrip("/")
    try:
        return {"path": full, "value": procfs.read(full)}
    except KeyError:
        return {"path": full, "error": "no such proc path",
                "available": procfs.list()}


@router.get("/aegis/sys/ps")
def sys_ps(kind: str = "", uid: str = "") -> Dict[str, Any]:
    """The process table: everything this kernel is actually running."""
    return {"processes": PROCS.ps(kind=kind or None, uid=uid or None),
            "status": PROCS.status()}


@router.get("/aegis/sys/proc/{pid}")
def sys_proc(pid: int) -> Dict[str, Any]:
    """One process in full — /proc/<pid>."""
    try:
        return PROCS.proc(pid)
    except Exception as exc:
        return {"status": "error", "errno": "ESRCH", "detail": str(exc)}


@router.post("/aegis/sys/kill")
def sys_kill(payload: KillIn) -> Dict[str, Any]:
    """Signal a process. A kind that cannot honour the signal returns EINVAL
    rather than accepting it and doing nothing. You may signal your OWN processes
    (sign out your session, cancel your job); anyone else's needs proc:signal."""
    cred = _cred()
    try:
        target = PROCS.get(payload.pid)
        if target.uid != cred.username and not cred.has("proc:signal"):
            GATE.require("proc:signal", cred, what=f"signal pid {payload.pid} ({target.uid})")
    except _PermissionDenied as exc:
        return {"status": "error", "errno": "EPERM", "detail": str(exc)}
    except NoSuchProcess as exc:
        return {"status": "error", "errno": "ESRCH", "detail": str(exc)}
    try:
        return {"status": "ok", **PROCS.signal(payload.pid, payload.signal, uid=cred.username)}
    except SignalUnsupported as exc:
        return {"status": "error", "errno": "EINVAL", "detail": str(exc)}
    except SignalFailed as exc:
        return {"status": "error", "errno": "EIO", "detail": str(exc)}
    except NoSuchProcess as exc:
        return {"status": "error", "errno": "ESRCH", "detail": str(exc)}


@router.get("/aegis/sys/journal")
def sys_journal(limit: int = 80, facility: str = "", severity: int = 7,
                full: bool = False) -> Dict[str, Any]:
    """The one ordered record, plus the chain's integrity verdict."""
    return {"status": KJOURNAL.status(),
            "entries": KJOURNAL.tail(limit=limit, facility=facility or None,
                                     min_severity=severity, full=full)}


@router.get("/aegis/sys/boot")
def sys_boot() -> Dict[str, Any]:
    """What happened while this kernel came up, with measured durations."""
    return BOOT.summary()


@router.get("/aegis/sys/accounts")
def sys_accounts() -> Dict[str, Any]:
    """Who can sign in at this workstation (no secrets)."""
    return {"accounts": USERS.public_accounts(), "demo_login": USERS.demo_login,
            "note": ("Demo staff sign in without a passphrase while AEGIS_DEMO_LOGIN is "
                     "on. This is demo-grade authentication, not a hospital IdP.")}


@router.post("/aegis/sys/login")
def sys_login(p: LoginIn) -> Dict[str, Any]:
    try:
        token, state, cred = USERS.login(p.username, p.passphrase, client="desktop")
    except _AuthError as exc:
        return {"status": "error", "errno": "EAUTH", "detail": str(exc)}
    return _session_payload(token, cred)


@router.post("/aegis/sys/logout")
def sys_logout() -> Dict[str, Any]:
    cred = _cred()
    return {"status": "ok", "signed_out": USERS.logout(cred), "uid": cred.username}


@router.post("/aegis/sys/lock")
def sys_lock() -> Dict[str, Any]:
    return {"status": "ok", "locked": USERS.lock(_cred())}


@router.post("/aegis/sys/unlock")
def sys_unlock(p: UnlockIn) -> Dict[str, Any]:
    try:
        return _session_payload(None, USERS.unlock(_cred(), p.passphrase))
    except _AuthError as exc:
        return {"status": "error", "errno": "EAUTH", "detail": str(exc)}


@router.get("/aegis/sys/whoami")
def sys_whoami() -> Dict[str, Any]:
    return {"credential": _cred().to_dict(), "gate": GATE.mode}


@router.get("/aegis/sys/who")
def sys_who() -> Dict[str, Any]:
    return {"sessions": USERS.who(), "status": USERS.status()}


@router.post("/aegis/sys/revoke")
def sys_revoke(p: RevokeIn) -> Dict[str, Any]:
    try:
        return {"status": "ok", "revoked": USERS.revoke(_cred(), p.sid)}
    except _PermissionDenied as exc:
        return {"status": "error", "errno": "EPERM", "detail": str(exc)}


@router.get("/aegis/sys/breakglass")
def sys_breakglass_log() -> Dict[str, Any]:
    return {"records": [b.to_dict() for b in USERS.breakglass_log][-50:],
            "unreviewed": sum(1 for b in USERS.breakglass_log if not b.reviewed)}


@router.post("/aegis/sys/breakglass")
def sys_breakglass(p: BreakGlassIn) -> Dict[str, Any]:
    """Emergency elevation: a stated reason, ≤ 15 minutes, a fixed set of added
    capabilities that never includes prescribing, policy or administration."""
    try:
        bg = USERS.breakglass(_cred(), p.reason, p.minutes)
    except _PermissionDenied as exc:
        return {"status": "error", "errno": "EPERM", "detail": str(exc)}
    except ValueError as exc:
        return {"status": "error", "errno": "EINVAL", "detail": str(exc)}
    return {"status": "ok", "breakglass": bg.to_dict()}


@router.post("/aegis/sys/breakglass/review")
def sys_breakglass_review(p: BreakGlassReviewIn) -> Dict[str, Any]:
    try:
        return {"status": "ok",
                "breakglass": USERS.review_breakglass(_cred(), p.id, p.verdict, p.note).to_dict()}
    except _PermissionDenied as exc:
        return {"status": "error", "errno": "EPERM", "detail": str(exc)}
    except (KeyError, ValueError) as exc:
        return {"status": "error", "errno": "EINVAL", "detail": str(exc)}


@router.get("/aegis/sys/gate")
def sys_gate() -> Dict[str, Any]:
    """The reference monitor: mode, measured overhead, refusals, table coverage,
    and the boot-time proof of the policy it enforces."""
    return {**GATE.status(), "coverage": S.GATE_COVERAGE, "users": USERS.status()}


@router.post("/aegis/sys/gate/mode")
def sys_gate_mode(p: GateModeIn) -> Dict[str, Any]:
    """Switch audit <-> enforce. Checked strictly, in every mode: in audit mode the
    gate serves what it would refuse, and a gate that let anyone switch itself off
    would audit its own removal. 'off' is not reachable from the API."""
    cred = _cred()
    if not cred.has("session:admin"):
        return {"status": "error", "errno": "EPERM",
                "detail": f"{cred.username} lacks session:admin"}
    if p.mode not in ("audit", "enforce"):
        return {"status": "error", "errno": "EINVAL", "detail": "mode must be audit or enforce"}
    prev, GATE.mode = GATE.mode, p.mode
    KJOURNAL.append("policy", "gate.mode", severity=4, uid=cred.username, frm=prev, to=p.mode)
    return {"status": "ok", "mode": GATE.mode, "was": prev}


@router.get("/aegis/sys/abi")
def sys_abi(cls: str = "") -> Dict[str, Any]:
    rows = [e.to_json() for e in ABI_TABLE.entries if not cls or e.klass == cls]
    return {**ABI_TABLE.summary(), "coverage": S.GATE_COVERAGE, "syscalls": rows}


@router.get("/aegis/sys/policy")
def sys_policy() -> Dict[str, Any]:
    return GATE.policy_proof or {"status": "pending",
                                 "detail": "the policy prover runs shortly after boot"}


@router.get("/aegis/fs/ls")
def fs_ls(path: str = "/") -> Dict[str, Any]:
    try:
        return {"path": KVFS.normalize(path),
                "entries": [n.to_dict() for n in KVFS.ls(_cred(), path)]}
    except Exception as exc:
        return _fs_error(exc)


@router.get("/aegis/fs/stat")
def fs_stat(path: str = "/") -> Dict[str, Any]:
    try:
        return KVFS.stat(_cred(), path).to_dict()
    except Exception as exc:
        return _fs_error(exc)


@router.get("/aegis/fs/read")
def fs_read(path: str) -> Dict[str, Any]:
    try:
        return KVFS.read(_cred(), path)
    except Exception as exc:
        return _fs_error(exc)


@router.post("/aegis/fs/write")
def fs_write(p: FsWriteIn) -> Dict[str, Any]:
    try:
        return {"status": "ok", "node": KVFS.write(_cred(), p.path, p.content,
                                                   append=p.append).to_dict()}
    except Exception as exc:
        return _fs_error(exc)


@router.post("/aegis/fs/mkdir")
def fs_mkdir(p: FsPathIn) -> Dict[str, Any]:
    try:
        return {"status": "ok", "node": KVFS.mkdir(_cred(), p.path).to_dict()}
    except Exception as exc:
        return _fs_error(exc)


@router.post("/aegis/fs/rm")
def fs_rm(p: FsPathIn) -> Dict[str, Any]:
    try:
        KVFS.rm(_cred(), p.path)
        return {"status": "ok", "removed": KVFS.normalize(p.path)}
    except Exception as exc:
        return _fs_error(exc)


@router.get("/aegis/sys/streams")
def sys_streams() -> Dict[str, Any]:
    return OS_HUB.status()


@router.websocket("/aegis/os/stream")
async def os_stream(ws: WebSocket) -> None:
    """One socket per desktop: subscribe to snapshot, cluster, proc, journal, boot,
    vfs watches and (with the shell) tty. Authenticated by session subprotocol."""
    await ws.accept()
    await OS_HUB.serve(ws, _cred())

