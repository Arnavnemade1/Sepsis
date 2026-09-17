"""The virtual filesystem — the kernel's objects as one namespace of paths.

WHY THIS EXISTS
---------------
AegisOS had a ``/proc`` that was a flat dictionary, two browser "filesystems" that
were a handful of hardcoded strings, and no place a person could keep a file. The
state a clinician or an operator needs — a patient's vitals, a signed proof, the
journal, the safety policy, a training run's result — lived behind fifty
differently shaped REST calls.

This is one tree:

    /proc      the process table and every kernel provider      (read)
    /dev       /dev/bio/<bed>/<feed>: the LIVE ring sample, drivers, devices
    /patients  one directory per bed: record, vitals, chart, bundle, causal,
               history (rw), tracking (rw), notes/ (append-only)
    /proofs    signed, independently re-checkable certificates
    /var/log   views of the one journal
    /etc       os-release, the RBAC policy, the syscall table, policy.dsl (rw)
    /models    trained detectors      /apps  packages      /artifacts  results
    /home      each user's own persistent files        /tmp  scratch, in memory

WHAT IT IS AND IS NOT. A Plan 9-style namespace over live kernel objects plus a
small persistent file store in the database. File operations are stateless path
syscalls (no descriptors, no mmap, no partial writes); content is text or JSON; a
virtual file is computed when it is read. It is not a block filesystem.

PERMISSIONS ARE ENFORCED HERE, IN EVERY GATE MODE. Each node names the capability
needed to read it and (if writable) to write it. The route-level gate may still be
in audit mode for the legacy pages; this surface is new, nothing depends on it
being open, so it is born enforcing. /home is owner-only; /tmp is shared scratch
where you may not overwrite someone else's file.
"""
from __future__ import annotations

import json
import os
import posixpath
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from backend.aegis.kernel.journal import INFO, NOTICE, WARN, Journal, KJOURNAL

MAX_READ_BYTES = 512 * 1024
HOME_FILE_MAX = 64 * 1024
HOME_USER_MAX = 1024 * 1024
HOME_USER_FILES = 256
TMP_TOTAL_MAX = 256 * 1024

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


class VFSError(Exception):
    def __init__(self, errno: str, message: str) -> None:
        super().__init__(f"{errno}: {message}")
        self.errno = errno
        self.message = message


@dataclass
class Node:
    path: str
    name: str
    kind: str = "file"                 # dir | file | dev
    size: int = 0
    owner: str = "root"
    mtime: float = 0.0
    cap_read: str = "view:public"
    cap_write: Optional[str] = None    # None = read-only
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        mode = ("d" if self.kind == "dir" else "c" if self.kind == "dev" else "-") + \
               "r" + ("w" if self.cap_write else "-")
        return {"path": self.path, "name": self.name, "kind": self.kind, "size": self.size,
                "owner": self.owner, "mtime": round(self.mtime, 1), "mode": mode,
                "cap_read": self.cap_read, "cap_write": self.cap_write, "note": self.note}


# --------------------------------------------------------------------------- #
# Virtual trees: nested dicts of names → Leaf | dict, built when accessed
# --------------------------------------------------------------------------- #
@dataclass
class Leaf:
    read: Callable[[Any], Any]
    write: Optional[Callable[[Any, str, bool], Any]] = None
    cap_read: str = "view:public"
    cap_write: Optional[str] = None
    kind: str = "file"
    note: str = ""
    append_only: bool = False


Tree = Dict[str, Union["Tree", Leaf]]


class FS:
    """Interface. Paths given to a filesystem are relative to its mount ('' = root)."""
    label = "fs"

    def stat(self, rel: str, cred: Any) -> Node:
        raise NotImplementedError

    def list(self, rel: str, cred: Any) -> List[Node]:
        raise NotImplementedError

    def read(self, rel: str, cred: Any) -> Any:
        raise NotImplementedError

    def write(self, rel: str, data: str, cred: Any, append: bool = False) -> Node:
        raise VFSError("EROFS", "read-only filesystem")

    def mkdir(self, rel: str, cred: Any) -> Node:
        raise VFSError("EROFS", "read-only filesystem")

    def unlink(self, rel: str, cred: Any) -> None:
        raise VFSError("EROFS", "read-only filesystem")


class VirtualFS(FS):
    """A tree computed from live kernel objects each time it is walked."""
    label = "virtual"

    def __init__(self, tree: Callable[[], Tree], mount: str, dir_cap: str = "view:public",
                 note: str = "") -> None:
        self.tree = tree
        self.mount = mount
        self.dir_cap = dir_cap
        self.note = note

    def _walk(self, rel: str) -> Union[Tree, Leaf]:
        cur: Union[Tree, Leaf] = self.tree()
        for part in [p for p in rel.split("/") if p]:
            if not isinstance(cur, dict) or part not in cur:
                raise VFSError("ENOENT", f"no such file or directory: {self.mount}/{rel}")
            cur = cur[part]
        return cur

    def _node(self, rel: str, obj: Union[Tree, Leaf]) -> Node:
        path = posixpath.join(self.mount, rel) if rel else self.mount
        name = posixpath.basename(path) or "/"
        if isinstance(obj, dict):
            return Node(path, name, "dir", size=len(obj), cap_read=self.dir_cap, note=self.note)
        return Node(path, name, obj.kind, cap_read=obj.cap_read, cap_write=obj.cap_write,
                    note=obj.note, mtime=time.time())

    def stat(self, rel: str, cred: Any) -> Node:
        return self._node(rel, self._walk(rel))

    def list(self, rel: str, cred: Any) -> List[Node]:
        obj = self._walk(rel)
        if not isinstance(obj, dict):
            raise VFSError("ENOTDIR", f"not a directory: {self.mount}/{rel}")
        return [self._node(posixpath.join(rel, k), v) for k, v in sorted(obj.items())]

    def read(self, rel: str, cred: Any) -> Any:
        obj = self._walk(rel)
        if isinstance(obj, dict):
            raise VFSError("EISDIR", f"is a directory: {self.mount}/{rel}")
        return obj.read(cred)

    def write(self, rel: str, data: str, cred: Any, append: bool = False) -> Node:
        obj = self._walk(rel)
        if isinstance(obj, dict) or obj.write is None:
            raise VFSError("EROFS", f"{self.mount}/{rel} is read-only")
        if obj.append_only and not append:
            raise VFSError("EPERM", f"{self.mount}/{rel} is append-only (use >>)")
        obj.write(cred, data, append)
        return self._node(rel, obj)


# --------------------------------------------------------------------------- #
# Persistent file store (SQLite always, Supabase when configured)
# --------------------------------------------------------------------------- #
class FileStore:
    """Owner-scoped text files. Mirrors causal_store.py: SQLite is always written;
    Supabase is written too when configured, and preferred for the first load of
    an owner (Render's disk does not survive a restart)."""

    def __init__(self, sqlite_path: Optional[str] = None, use_supabase: Optional[bool] = None,
                 journal: Optional[Journal] = None) -> None:
        from backend.aegis.kernel.journal import default_sqlite_path
        self.sqlite_path = sqlite_path or default_sqlite_path()
        env = bool(SUPABASE_URL and SUPABASE_KEY)
        self.use_supabase = env if use_supabase is None else (use_supabase and env)
        self.client = None
        if self.use_supabase:
            try:
                from supabase import create_client
                self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception:                              # pragma: no cover
                self.use_supabase = False
        self.journal = journal if journal is not None else KJOURNAL
        self.supabase_errors = 0
        self._cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS aegis_files (
                owner TEXT NOT NULL, path TEXT NOT NULL, content TEXT, is_dir INTEGER DEFAULT 0,
                size INTEGER DEFAULT 0, mtime REAL NOT NULL, PRIMARY KEY (owner, path))""")
            conn.commit()
        finally:
            conn.close()

    def _load(self, owner: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            if owner in self._cache:
                return self._cache[owner]
        rows: List[Dict[str, Any]] = []
        loaded = False
        if self.use_supabase:
            try:
                res = (self.client.table("aegis_files").select("*").eq("owner", owner)
                       .limit(HOME_USER_FILES + 8).execute())
                rows = list(getattr(res, "data", None) or [])
                loaded = True
            except Exception:                              # pragma: no cover
                self.supabase_errors += 1
        if not loaded:
            conn = sqlite3.connect(self.sqlite_path)
            try:
                cur = conn.execute("SELECT owner, path, content, is_dir, size, mtime "
                                   "FROM aegis_files WHERE owner = ?", (owner,))
                rows = [dict(zip(("owner", "path", "content", "is_dir", "size", "mtime"), r))
                        for r in cur.fetchall()]
            finally:
                conn.close()
        files = {r["path"]: {"content": r.get("content"), "is_dir": bool(r.get("is_dir")),
                             "size": int(r.get("size") or 0), "mtime": float(r.get("mtime") or 0)}
                 for r in rows}
        with self._lock:
            self._cache.setdefault(owner, files)
            return self._cache[owner]

    def files(self, owner: str) -> Dict[str, Dict[str, Any]]:
        return self._load(owner)

    def put(self, owner: str, path: str, content: Optional[str], is_dir: bool = False) -> Dict[str, Any]:
        rec = {"content": content, "is_dir": is_dir,
               "size": len(content.encode()) if content else 0, "mtime": time.time()}
        self._load(owner)[path] = rec
        row = {"owner": owner, "path": path, "content": content, "is_dir": int(is_dir),
               "size": rec["size"], "mtime": rec["mtime"]}
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute("INSERT OR REPLACE INTO aegis_files (owner, path, content, is_dir, size, mtime) "
                         "VALUES (:owner, :path, :content, :is_dir, :size, :mtime)", row)
            conn.commit()
        finally:
            conn.close()
        if self.use_supabase:
            try:
                self.client.table("aegis_files").upsert(row).execute()
            except Exception:                              # pragma: no cover
                self.supabase_errors += 1
        return rec

    def delete(self, owner: str, path: str) -> None:
        self._load(owner).pop(path, None)
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute("DELETE FROM aegis_files WHERE owner = ? AND path = ?", (owner, path))
            conn.commit()
        finally:
            conn.close()
        if self.use_supabase:
            try:
                self.client.table("aegis_files").delete().eq("owner", owner).eq("path", path).execute()
            except Exception:                              # pragma: no cover
                self.supabase_errors += 1


class HomeFS(FS):
    """/home/<user>: owner-only, persisted, with quotas. The shared anonymous
    ``guest`` identity gets no writable home — a home every visitor shares is a
    public noticeboard, not a home."""
    label = "home"

    def __init__(self, store: FileStore, users_fn: Callable[[], List[str]]) -> None:
        self.store = store
        self.users_fn = users_fn

    @staticmethod
    def _split(rel: str) -> Tuple[str, str]:
        parts = [p for p in rel.split("/") if p]
        return (parts[0], "/".join(parts[1:])) if parts else ("", "")

    def _check_owner(self, user: str, cred: Any, write: bool) -> None:
        if not user:
            return
        if user == "guest" and write:
            raise VFSError("EROFS", "the shared guest identity has no writable home; sign in")
        if cred.username != user and cred.kind != "kernel":
            raise VFSError("EACCES", f"/home/{user} belongs to {user}")

    def stat(self, rel: str, cred: Any) -> Node:
        user, sub = self._split(rel)
        if not user:
            return Node("/home", "home", "dir", size=len(self.users_fn()))
        if not sub:
            return Node(f"/home/{user}", user, "dir", owner=user,
                        cap_write="view:public" if user != "guest" else None)
        self._check_owner(user, cred, write=False)
        rec = self.store.files(user).get(sub)
        if rec is None:
            if any(p.startswith(sub + "/") for p in self.store.files(user)):
                return Node(f"/home/{user}/{sub}", posixpath.basename(sub), "dir", owner=user,
                            cap_write="view:public")
            raise VFSError("ENOENT", f"no such file or directory: /home/{user}/{sub}")
        return Node(f"/home/{user}/{sub}", posixpath.basename(sub),
                    "dir" if rec["is_dir"] else "file", size=rec["size"], owner=user,
                    mtime=rec["mtime"], cap_write="view:public")

    def list(self, rel: str, cred: Any) -> List[Node]:
        user, sub = self._split(rel)
        if not user:
            return [Node(f"/home/{u}", u, "dir", owner=u) for u in sorted(self.users_fn())]
        self._check_owner(user, cred, write=False)
        files = self.store.files(user)
        prefix = (sub + "/") if sub else ""
        if sub and sub not in files and not any(p.startswith(prefix) for p in files):
            raise VFSError("ENOENT", f"no such directory: /home/{user}/{sub}")
        names: Dict[str, Node] = {}
        for p, rec in files.items():
            if not p.startswith(prefix) or p == sub:
                continue
            head, _, rest = p[len(prefix):].partition("/")
            is_dir = bool(rest) or rec["is_dir"]
            if head not in names or is_dir:
                names[head] = Node(f"/home/{user}/{prefix}{head}", head,
                                   "dir" if is_dir else "file",
                                   size=0 if is_dir else rec["size"], owner=user,
                                   mtime=rec["mtime"], cap_write="view:public")
        return [names[k] for k in sorted(names)]

    def read(self, rel: str, cred: Any) -> Any:
        user, sub = self._split(rel)
        self._check_owner(user, cred, write=False)
        rec = self.store.files(user).get(sub) if user else None
        if rec is None or rec["is_dir"]:
            node = self.stat(rel, cred)
            if node.kind == "dir":
                raise VFSError("EISDIR", f"is a directory: /home/{rel}")
            raise VFSError("ENOENT", f"no such file: /home/{rel}")
        return rec["content"] or ""

    def write(self, rel: str, data: str, cred: Any, append: bool = False) -> Node:
        user, sub = self._split(rel)
        if not user or not sub:
            raise VFSError("EISDIR", "write a file inside your home directory")
        self._check_owner(user, cred, write=True)
        files = self.store.files(user)
        prev = files.get(sub)
        if prev is not None and prev["is_dir"]:
            raise VFSError("EISDIR", f"is a directory: /home/{rel}")
        content = ((prev or {}).get("content") or "") + data if append else data
        size = len(content.encode())
        if size > HOME_FILE_MAX:
            raise VFSError("EFBIG", f"file would be {size} bytes (limit {HOME_FILE_MAX})")
        used = sum(r["size"] for p, r in files.items() if p != sub)
        if used + size > HOME_USER_MAX:
            raise VFSError("EDQUOT", f"home quota is {HOME_USER_MAX // 1024} KiB")
        if prev is None and len(files) >= HOME_USER_FILES:
            raise VFSError("EDQUOT", f"home is limited to {HOME_USER_FILES} files")
        parent = posixpath.dirname(sub)
        if parent and parent not in files and not any(p.startswith(parent + "/") for p in files):
            raise VFSError("ENOENT", f"no such directory: /home/{user}/{parent} (mkdir it first)")
        self.store.put(user, sub, content)
        return self.stat(rel, cred)

    def mkdir(self, rel: str, cred: Any) -> Node:
        user, sub = self._split(rel)
        if not user or not sub:
            raise VFSError("EEXIST", "home directories exist per account")
        self._check_owner(user, cred, write=True)
        if sub in self.store.files(user):
            raise VFSError("EEXIST", f"already exists: /home/{rel}")
        self.store.put(user, sub, None, is_dir=True)
        return self.stat(rel, cred)

    def unlink(self, rel: str, cred: Any) -> None:
        user, sub = self._split(rel)
        if not user or not sub:
            raise VFSError("EPERM", "a home directory cannot be removed")
        self._check_owner(user, cred, write=True)
        files = self.store.files(user)
        if sub not in files:
            raise VFSError("ENOENT", f"no such file: /home/{rel}")
        if any(p.startswith(sub + "/") for p in files):
            raise VFSError("ENOTEMPTY", f"directory not empty: /home/{rel}")
        self.store.delete(user, sub)


class TmpFS(FS):
    """/tmp: shared scratch in memory, cleared on restart, size-capped, sticky."""
    label = "tmp"

    def __init__(self) -> None:
        self.files: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def stat(self, rel: str, cred: Any) -> Node:
        if not rel:
            return Node("/tmp", "tmp", "dir", size=len(self.files), cap_write="view:public",
                        note="shared scratch · in memory · cleared on restart")
        rec = self.files.get(rel)
        if rec is None:
            raise VFSError("ENOENT", f"no such file: /tmp/{rel}")
        return Node(f"/tmp/{rel}", rel, "file", size=len(rec["content"].encode()),
                    owner=rec["owner"], mtime=rec["mtime"], cap_write="view:public")

    def list(self, rel: str, cred: Any) -> List[Node]:
        if rel:
            raise VFSError("ENOTDIR", "/tmp is flat")
        return [self.stat(k, cred) for k in sorted(self.files)]

    def read(self, rel: str, cred: Any) -> Any:
        return self.files[rel]["content"] if rel in self.files else self.stat(rel, cred)

    def write(self, rel: str, data: str, cred: Any, append: bool = False) -> Node:
        if not rel or "/" in rel:
            raise VFSError("EINVAL", "/tmp is flat: /tmp/<name>")
        with self._lock:
            rec = self.files.get(rel)
            if rec is not None and rec["owner"] != cred.username:
                raise VFSError("EACCES", f"/tmp/{rel} belongs to {rec['owner']} (sticky)")
            content = (rec["content"] if (rec and append) else "") + data
            total = sum(len(r["content"].encode()) for k, r in self.files.items() if k != rel)
            if total + len(content.encode()) > TMP_TOTAL_MAX:
                raise VFSError("ENOSPC", f"/tmp is limited to {TMP_TOTAL_MAX // 1024} KiB")
            self.files[rel] = {"content": content, "owner": cred.username, "mtime": time.time()}
        return self.stat(rel, cred)

    def unlink(self, rel: str, cred: Any) -> None:
        with self._lock:
            rec = self.files.get(rel)
            if rec is None:
                raise VFSError("ENOENT", f"no such file: /tmp/{rel}")
            if rec["owner"] != cred.username:
                raise VFSError("EACCES", f"/tmp/{rel} belongs to {rec['owner']} (sticky)")
            del self.files[rel]


# --------------------------------------------------------------------------- #
# The namespace
# --------------------------------------------------------------------------- #
class VFS:
    def __init__(self, journal: Optional[Journal] = None) -> None:
        self.mounts: List[Tuple[str, FS]] = []
        self.journal = journal if journal is not None else KJOURNAL
        self.ops = {"stat": 0, "list": 0, "read": 0, "write": 0, "denied": 0}

    def mount(self, prefix: str, fs: FS) -> None:
        self.mounts.append((prefix.rstrip("/") or "/", fs))
        self.mounts.sort(key=lambda m: len(m[0]), reverse=True)

    @staticmethod
    def normalize(path: str, cwd: str = "/") -> str:
        """Absolute, collapsed, and unable to climb above the root."""
        path = (path or "").strip() or "."
        joined = path if path.startswith("/") else posixpath.join(cwd or "/", path)
        parts: List[str] = []
        for p in joined.split("/"):
            if p in ("", "."):
                continue
            if p == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(p)
        return "/" + "/".join(parts)

    def _resolve(self, path: str) -> Tuple[Optional[FS], str, str]:
        for prefix, fs in self.mounts:
            if prefix == "/":
                continue
            if path == prefix or path.startswith(prefix + "/"):
                return fs, path[len(prefix):].lstrip("/"), prefix
        return None, path, "/"

    def _root_children(self) -> List[Node]:
        seen: Dict[str, Node] = {}
        for prefix, fs in self.mounts:
            head = prefix.strip("/").split("/")[0]
            if head and head not in seen:
                seen[head] = Node("/" + head, head, "dir",
                                  note=getattr(fs, "note", "") if prefix.count("/") == 1 else "")
        return [seen[k] for k in sorted(seen)]

    def _deny(self, cred: Any, cap: str, path: str, op: str) -> VFSError:
        self.ops["denied"] += 1
        self.journal.append("vfs", f"{op}.denied", severity=WARN, uid=cred.username,
                            pid=getattr(cred, "pid", 0), path=path, capability=cap)
        return VFSError("EACCES", f"{cred.username} ({cred.role}) lacks {cap} for {path}")

    def stat(self, cred: Any, path: str, cwd: str = "/") -> Node:
        path = self.normalize(path, cwd)
        self.ops["stat"] += 1
        fs, rel, prefix = self._resolve(path)
        if fs is None:
            if path == "/":
                return Node("/", "/", "dir", size=len(self._root_children()))
            kids = [n for n in self._root_children() if n.path == path or path.startswith(n.path + "/")]
            if kids and kids[0].path == path:
                return kids[0]
            # intermediate directories of nested mounts (e.g. /var)
            if any(p.startswith(path + "/") for p, _ in self.mounts):
                return Node(path, posixpath.basename(path), "dir")
            raise VFSError("ENOENT", f"no such file or directory: {path}")
        return fs.stat(rel, cred)

    def ls(self, cred: Any, path: str, cwd: str = "/") -> List[Node]:
        path = self.normalize(path, cwd)
        self.ops["list"] += 1
        fs, rel, prefix = self._resolve(path)
        if fs is None:
            if path == "/":
                return self._root_children()
            nested = sorted({p[len(path) + 1:].split("/")[0] for p, _ in self.mounts
                             if p.startswith(path + "/")})
            if nested:
                return [Node(f"{path}/{n}", n, "dir") for n in nested]
            raise VFSError("ENOENT", f"no such directory: {path}")
        node = fs.stat(rel, cred)
        if node.kind != "dir":
            return [node]
        if not cred.has(node.cap_read):
            raise self._deny(cred, node.cap_read, path, "list")
        return fs.list(rel, cred)

    def read(self, cred: Any, path: str, cwd: str = "/") -> Dict[str, Any]:
        path = self.normalize(path, cwd)
        self.ops["read"] += 1
        fs, rel, _ = self._resolve(path)
        if fs is None:
            raise VFSError("EISDIR", f"is a directory: {path}")
        node = fs.stat(rel, cred)
        if node.kind == "dir":
            raise VFSError("EISDIR", f"is a directory: {path}")
        if not cred.has(node.cap_read):
            raise self._deny(cred, node.cap_read, path, "read")
        content = fs.read(rel, cred)
        if isinstance(content, (dict, list)):
            kind, text = "json", json.dumps(content, indent=1, default=str)
        else:
            kind, text = "text", "" if content is None else str(content)
        if len(text) > MAX_READ_BYTES:
            text = text[:MAX_READ_BYTES] + f"\n… truncated at {MAX_READ_BYTES // 1024} KiB"
        return {"path": path, "kind": kind, "content": text, "size": len(text),
                "node": node.to_dict()}

    def write(self, cred: Any, path: str, data: str, append: bool = False,
              cwd: str = "/") -> Node:
        path = self.normalize(path, cwd)
        fs, rel, _ = self._resolve(path)
        if fs is None:
            raise VFSError("EROFS", f"{path} is not writable")
        try:
            node = fs.stat(rel, cred)
            cap = node.cap_write
        except VFSError as exc:
            if exc.errno != "ENOENT":
                raise
            parent = posixpath.dirname(path)
            pfs, prel, _ = self._resolve(parent)
            cap = (pfs.stat(prel, cred).cap_write if pfs is not None else None)
        if cap is None:
            raise VFSError("EROFS", f"{path} is read-only")
        if not cred.has(cap):
            raise self._deny(cred, cap, path, "write")
        out = fs.write(rel, data if isinstance(data, str) else str(data), cred, append=append)
        self.ops["write"] += 1
        self.journal.append("vfs", "append" if append else "write", severity=INFO,
                            uid=cred.username, pid=getattr(cred, "pid", 0), path=path,
                            bytes=len(str(data).encode()), acting_for=getattr(cred, "ruid", None))
        return out

    def mkdir(self, cred: Any, path: str, cwd: str = "/") -> Node:
        path = self.normalize(path, cwd)
        fs, rel, _ = self._resolve(path)
        if fs is None:
            raise VFSError("EROFS", f"{path} is not writable")
        out = fs.mkdir(rel, cred)
        self.journal.append("vfs", "mkdir", severity=INFO, uid=cred.username, path=path)
        return out

    def rm(self, cred: Any, path: str, cwd: str = "/") -> None:
        path = self.normalize(path, cwd)
        fs, rel, _ = self._resolve(path)
        if fs is None:
            raise VFSError("EPERM", f"{path} cannot be removed")
        fs.unlink(rel, cred)
        self.journal.append("vfs", "rm", severity=NOTICE, uid=cred.username, path=path)

    def status(self) -> Dict[str, Any]:
        return {"mounts": [p for p, _ in sorted(self.mounts)], "ops": dict(self.ops)}
