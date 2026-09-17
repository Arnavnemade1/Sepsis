"""The kernel journal — one ordered, tamper-evident record of what happened.

WHY THIS EXISTS
---------------
AegisOS keeps NINETEEN separate logs: the syscall audit (capped at 256, in
memory), the RBAC ledger, aegisd's events, the package audit, two supervisor
event lists, the device registry's log, the scheduler's interrupt list, the cage's
quarantine, the agent's activity feed, mlctl's job log, and more. They share no
sequence, no clock and no bus, so there is no way to ask the only question that
matters after an incident: *what happened, in what order, and who caused it?*

A `treat` syscall, for example, writes the syscall audit and two cages' counters
— but no RBAC entry and no attestation. `POST /aegis/cluster/chaos` writes
nothing anywhere. And almost all of it dies with the process.

This module is the single journal those become views of. It is deliberately
modest in what it claims:

  * **Ordered.** One monotonic ``seq`` and one wall clock for every facility, so
    entries from the scheduler, the prover and a clinician's order interleave
    correctly instead of sitting in separate lists with separate counters.
  * **Tamper-EVIDENT, not tamper-proof.** Each entry hashes the previous hash
    together with its own canonical body, so editing or deleting an entry breaks
    every hash after it and :meth:`Journal.verify` says where. Anyone who can
    write the database can also recompute a whole chain — what this defeats is
    silent edits, not a determined administrator. Publishing the head elsewhere
    (an attestation, a signed proof bundle) is what would close that.
  * **Durable across restarts.** The chain rehydrates from the store's last row,
    so a reboot continues the chain rather than starting a second one.
  * **Never in the way.** Appends go to an in-memory ring and a bounded queue; a
    background thread does the writing. If the store is down or the queue is
    full, entries are DROPPED and counted — the kernel does not block a clinical
    tick on a database.

WHAT IS NOT JOURNALLED: reads. Every GET would be a row, which on a free-tier
500 MB Postgres buys nothing but cost. Writes, denials, auth, and lifecycle
events are the record; read volume belongs in counters. Per-tick events (the
headline certificate every 2 s, every scheduler decision) are journalled on
CHANGE, not on every tick — 43,000 identical rows a day would exhaust the free
tier in under a fortnight and bury the entries that matter.

ONE WRITER PER STORE. The chain is a single sequence, so two kernels appending to
the same database (a dev server plus a verify script that imports it) interleave
two chains and ``verify_chain`` over the rows will fail. Each boot stamps a
``boot_id`` so the rows can be told apart; point a second process at its own file
with ``AEGIS_JOURNAL_DB``.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

GENESIS = "genesis"

# syslog severities, because inventing a private ladder helps nobody.
EMERG, ALERT, CRIT, ERR, WARN, NOTICE, INFO, DEBUG = range(8)
SEVERITY_NAME = {EMERG: "emerg", ALERT: "alert", CRIT: "crit", ERR: "err",
                 WARN: "warn", NOTICE: "notice", INFO: "info", DEBUG: "debug"}

#: Facilities. Adding one is fine; renaming one breaks every stored query.
FACILITIES = ("kern", "boot", "auth", "syscall", "proc", "sched", "driver",
              "device", "svc", "pkg", "vfs", "irq", "cron", "agent", "policy",
              "egress", "clinical", "net")

#: Identifies this process's boot. Rows from different boots of the same store
#: are distinguishable, which is what makes a forked chain diagnosable.
BOOT_ID = uuid.uuid4().hex[:12]

_COLS = ("hash", "seq", "ts_us", "facility", "severity", "event", "uid", "pid",
         "detail", "prev_hash")


def schema_sql() -> str:
    """Postgres DDL — run once on Supabase for durable journalling."""
    return """
create table if not exists aegis_journal (
  hash text primary key, seq bigint not null, ts_us bigint not null,
  facility text not null, severity int not null, event text not null,
  uid text not null, pid bigint not null default 0,
  detail jsonb not null default '{}'::jsonb,
  prev_hash text not null,
  created_at timestamptz default now()
);
create index if not exists aegis_journal_seq on aegis_journal (seq);
create index if not exists aegis_journal_facility on aegis_journal (facility, seq);
""".strip()


def _canonical(body: Dict[str, Any]) -> bytes:
    # sort_keys + separators so the same entry hashes identically on every
    # interpreter; floats are kept OUT of the body (ts is integer microseconds)
    # because float repr is not a stable wire format.
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      default=str).encode()


@dataclass
class JournalEntry:
    seq: int
    ts_us: int
    facility: str
    severity: int
    event: str
    uid: str
    pid: int
    detail: Dict[str, Any]
    prev_hash: str
    hash: str

    def body(self) -> Dict[str, Any]:
        return {"seq": self.seq, "ts_us": self.ts_us, "facility": self.facility,
                "severity": self.severity, "event": self.event, "uid": self.uid,
                "pid": self.pid, "detail": self.detail}

    def to_dict(self, full: bool = False) -> Dict[str, Any]:
        d = dict(self.body())
        d["ts"] = round(self.ts_us / 1e6, 3)
        d["severity_name"] = SEVERITY_NAME.get(self.severity, str(self.severity))
        d["hash"] = self.hash if full else self.hash[:12]
        if full:
            d["prev_hash"] = self.prev_hash
        return d


def default_sqlite_path() -> str:
    """backend/sepsis_twin.db — the same gitignored file database.py and
    causal_store.py use. (Two dirnames from this file is backend/aegis/, which
    silently created an untracked database inside the package.)"""
    return os.getenv("AEGIS_JOURNAL_DB") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "sepsis_twin.db")


class JournalStore:
    """Durable sink — Supabase when configured, SQLite always, never blocking."""

    def __init__(self, sqlite_path: Optional[str] = None,
                 use_supabase: Optional[bool] = None,
                 flush_interval: float = 2.0, batch: int = 200,
                 queue_max: int = 10000) -> None:
        self.sqlite_path = sqlite_path or default_sqlite_path()
        env_has = bool(SUPABASE_URL and SUPABASE_KEY)
        self.use_supabase = env_has if use_supabase is None else (use_supabase and env_has)
        self.supabase_client = None
        if self.use_supabase:
            try:
                from supabase import create_client
                self.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as exc:                  # pragma: no cover
                print(f"[journal] Supabase init failed ({exc}); using SQLite.")
                self.use_supabase = False
        self._init_sqlite()
        self.flush_interval = float(flush_interval)
        self.batch = int(batch)
        self.dropped = 0
        self.written = 0
        # Which sink last actually accepted a write. `use_supabase` is only what
        # was CONFIGURED: with the table missing, the first run reported
        # "backend: supabase" while every row went to SQLite. A status line that
        # names the wrong store is worse than none, because it is what you would
        # check before trusting the record.
        self.sink = "none"
        self.supabase_errors = 0
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _init_sqlite(self) -> None:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aegis_journal (
                    hash TEXT PRIMARY KEY, seq INTEGER NOT NULL, ts_us INTEGER NOT NULL,
                    facility TEXT NOT NULL, severity INTEGER NOT NULL, event TEXT NOT NULL,
                    uid TEXT NOT NULL, pid INTEGER NOT NULL DEFAULT 0,
                    detail TEXT NOT NULL DEFAULT '{}', prev_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute("CREATE INDEX IF NOT EXISTS aegis_journal_seq ON aegis_journal (seq)")
            conn.commit()
        finally:
            conn.close()

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> "JournalStore":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="journald", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)
            self._thread = None
        self.flush()

    def enqueue(self, entry: JournalEntry) -> bool:
        try:
            self._q.put_nowait(entry)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.flush_interval)
            self.flush()

    def flush(self) -> int:
        rows: List[JournalEntry] = []
        while len(rows) < self.batch:
            try:
                rows.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not rows:
            return 0
        payload = [{"hash": e.hash, "seq": e.seq, "ts_us": e.ts_us,
                    "facility": e.facility, "severity": e.severity, "event": e.event,
                    "uid": e.uid, "pid": e.pid, "detail": e.detail,
                    "prev_hash": e.prev_hash} for e in rows]
        if self.use_supabase:
            try:
                self.supabase_client.table("aegis_journal").upsert(payload).execute()
                self.written += len(payload)
                self.sink = "supabase"
                return len(payload)
            except Exception as exc:                  # pragma: no cover
                self.supabase_errors += 1
                if self.supabase_errors == 1:         # say it once, not every flush
                    print(f"[journal] supabase write failed ({exc}); using SQLite. "
                          "Apply supabase/migrations/*_kernel_journal.sql to enable it.")
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO aegis_journal (%s) VALUES (%s)"
                % (", ".join(_COLS), ", ".join("?" for _ in _COLS)),
                [tuple(json.dumps(r[c]) if c == "detail" else r[c] for c in _COLS)
                 for r in payload])
            conn.commit()
            self.written += len(payload)
            self.sink = "sqlite"
        finally:
            conn.close()
        return len(payload)

    # -- reading back ----------------------------------------------------- #
    def head(self) -> Optional[Dict[str, Any]]:
        """The last persisted (seq, hash) — what a rebooting kernel resumes from."""
        if self.use_supabase:
            try:
                res = (self.supabase_client.table("aegis_journal")
                       .select("seq,hash").order("seq", desc=True).limit(1).execute())
                rows = getattr(res, "data", None) or []
                return {"seq": rows[0]["seq"], "hash": rows[0]["hash"]} if rows else None
            except Exception as exc:                  # pragma: no cover
                print(f"[journal] supabase head failed: {exc}")
        conn = sqlite3.connect(self.sqlite_path)
        try:
            row = conn.execute(
                "SELECT seq, hash FROM aegis_journal ORDER BY seq DESC LIMIT 1").fetchone()
        finally:
            conn.close()
        return {"seq": int(row[0]), "hash": row[1]} if row else None

    def load(self, limit: int = 500, since_seq: int = -1) -> List[JournalEntry]:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            rows = conn.execute(
                "SELECT %s FROM aegis_journal WHERE seq > ? ORDER BY seq LIMIT ?"
                % ", ".join(_COLS), (since_seq, limit)).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = dict(zip(_COLS, r))
            out.append(JournalEntry(
                seq=int(d["seq"]), ts_us=int(d["ts_us"]), facility=d["facility"],
                severity=int(d["severity"]), event=d["event"], uid=d["uid"],
                pid=int(d["pid"]), detail=json.loads(d["detail"]),
                prev_hash=d["prev_hash"], hash=d["hash"]))
        return out


class Journal:
    """The in-memory chain. One per process; ``KJOURNAL`` below is that one."""

    def __init__(self, ring: int = 2048, store: Optional[JournalStore] = None) -> None:
        self.entries: deque = deque(maxlen=int(ring))
        self.store = store
        self._head = GENESIS
        self._seq = 0
        self._lock = threading.Lock()
        self.counts: Dict[str, int] = {}
        self._last_verify: Optional[Dict[str, Any]] = None
        self._last_verify_at = 0.0

    def rehydrate(self) -> Optional[Dict[str, Any]]:
        """Continue the persisted chain instead of starting a parallel one."""
        if self.store is None:
            return None
        head = self.store.head()
        if head:
            with self._lock:
                self._head = head["hash"]
                self._seq = int(head["seq"]) + 1
        return head

    def append(self, facility: str, event: str, severity: int = INFO,
               uid: str = "kernel", pid: int = 0, **detail: Any) -> JournalEntry:
        with self._lock:
            seq = self._seq
            self._seq += 1
            prev = self._head
            body = {"seq": seq, "ts_us": int(time.time() * 1e6), "facility": facility,
                    "severity": int(severity), "event": event, "uid": uid,
                    "pid": int(pid), "detail": detail}
            h = hashlib.sha256(prev.encode() + _canonical(body)).hexdigest()
            e = JournalEntry(prev_hash=prev, hash=h, **body)
            self._head = h
            self.entries.append(e)
            self.counts[facility] = self.counts.get(facility, 0) + 1
        if self.store is not None:
            self.store.enqueue(e)
        return e

    # -- integrity -------------------------------------------------------- #
    def _snapshot(self):
        # Copy the ring and the head TOGETHER. Iterating the deque while another
        # thread appends raises "deque mutated during iteration", and reading the
        # head after the copy makes a concurrent append look like a broken chain.
        with self._lock:
            return list(self.entries), self._head

    def verify(self) -> Dict[str, Any]:
        """Recompute the retained window's chain.

        Reports the window explicitly: the ring holds the last N entries, so a
        pass means "nothing in the retained window was altered", not "the whole
        history is intact". Verifying the whole history means reading the store.
        """
        ents, head = self._snapshot()
        if not ents:
            out = {"ok": True, "window": 0, "from_seq": None, "to_seq": None}
        else:
            out = None
            prev = ents[0].prev_hash
            for e in ents:
                h = hashlib.sha256(prev.encode() + _canonical(e.body())).hexdigest()
                if e.prev_hash != prev or e.hash != h:
                    out = {"ok": False, "window": len(ents), "broken_at": e.seq,
                           "from_seq": ents[0].seq, "to_seq": ents[-1].seq}
                    break
                prev = e.hash
            if out is None:
                out = {"ok": prev == head, "window": len(ents),
                       "from_seq": ents[0].seq, "to_seq": ents[-1].seq}
        self._last_verify, self._last_verify_at = out, time.time()
        return out

    @staticmethod
    def verify_chain(entries: List[JournalEntry], start_hash: str = GENESIS) -> bool:
        """Verify an arbitrary loaded run (e.g. rows read back from the store)."""
        prev = start_hash
        for e in entries:
            h = hashlib.sha256(prev.encode() + _canonical(e.body())).hexdigest()
            if e.prev_hash != prev or e.hash != h:
                return False
            prev = e.hash
        return True

    # -- views ------------------------------------------------------------ #
    def head(self) -> str:
        return self._head

    def tail(self, limit: int = 50, facility: Optional[str] = None,
             min_severity: int = DEBUG, full: bool = False) -> List[Dict[str, Any]]:
        ents, _ = self._snapshot()
        out = [e for e in ents
               if (facility is None or e.facility == facility)
               and e.severity <= min_severity]
        return [e.to_dict(full=full) for e in out[-limit:]]

    def dmesg(self, limit: int = 80) -> List[Dict[str, Any]]:
        """Kernel-facility lines only — the boot/driver/scheduler narrative."""
        return self.tail(limit=limit, facility=None, min_severity=NOTICE)

    def status(self, max_age_s: float = 5.0) -> Dict[str, Any]:
        """Cheap enough to embed in a snapshot. Re-hashing the whole ring is ~30 ms
        on a throttled core, so the verdict is re-computed at most every
        ``max_age_s`` and reported with its age rather than recomputed per call."""
        v = self._last_verify
        if v is None or (time.time() - self._last_verify_at) > max_age_s:
            v = self.verify()
        return {
            "boot_id": BOOT_ID,
            "seq": self._seq, "head": self._head[:12], "retained": len(self.entries),
            "ring": self.entries.maxlen, "by_facility": dict(self.counts),
            "chain_ok": v["ok"], "verified_window": v["window"],
            "verified_age_s": round(time.time() - self._last_verify_at, 1),
            "persisted": None if self.store is None else {
                "configured": "supabase" if self.store.use_supabase else "sqlite",
                "sink": self.store.sink,          # where rows ACTUALLY landed
                "written": self.store.written, "dropped": self.store.dropped,
                "supabase_errors": self.store.supabase_errors,
            },
        }


#: The process-wide journal. ``server.py`` gives it a store and rehydrates it.
KJOURNAL = Journal()
