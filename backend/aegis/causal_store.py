"""Durable persistence for the Temporal Causal Graph — append-only, integrity-preserving.

The causal graph is an append-only, hash-chained ledger (``kernel/causal_graph.py``).
That maps one-to-one onto durable storage: every node and edge is one row keyed by
its content hash, and the stored hashes preserve the chain so :meth:`verify_chain`
still validates *after a reload* — tamper-evidence survives a restart.

Mirrors ``backend/core/database.py``: Supabase when ``SUPABASE_URL``/``SUPABASE_KEY``
are set, else a local SQLite file, with per-write fallback to SQLite so a flaky
network never drops clinical events. Two tables, ``causal_nodes`` and
``causal_edges``, keyed by ``hash`` (PRIMARY KEY) → writes are idempotent: re-saving
the same entry is a no-op, so the live path may persist freely without dedup logic.

Free-tier discipline (see the Supabase-persistence direction): the *ingestor* already
distills the tick stream down to clinically meaningful events, so write volume here is
low (tens of rows per patient-episode, not thousands per minute). :meth:`prune_before`
is provided for the retention window.

Supabase note: this writes to two tables it does **not** create remotely (creating
remote schema is an outward-facing action left to the operator). Run ``schema_sql()``
once against the project; until then writes transparently fall back to SQLite.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from backend.aegis.kernel.causal_graph import Node, Edge, TemporalCausalGraph

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_NODE_COLS = ("hash", "bed", "seq", "id", "kind", "t", "payload",
              "risk", "confidence", "prev_hash")
_EDGE_COLS = ("hash", "bed", "seq", "id", "src", "dst", "mechanism",
              "evidence_type", "confidence", "proven_by", "support", "prev_hash")


def schema_sql() -> str:
    """The Postgres DDL to run once on Supabase to enable durable graph storage."""
    return """
create table if not exists causal_nodes (
  hash text primary key, bed text not null, seq bigint not null,
  id text not null, kind text not null, t double precision not null,
  payload jsonb not null default '{}'::jsonb,
  risk double precision not null, confidence double precision not null,
  prev_hash text not null,
  created_at timestamptz default now()
);
create index if not exists causal_nodes_bed_seq on causal_nodes (bed, seq);

create table if not exists causal_edges (
  hash text primary key, bed text not null, seq bigint not null,
  id text not null, src text not null, dst text not null,
  mechanism text not null, evidence_type text not null,
  confidence double precision not null, proven_by text, support bigint,
  prev_hash text not null,
  created_at timestamptz default now()
);
create index if not exists causal_edges_bed_seq on causal_edges (bed, seq);
""".strip()


class CausalGraphStore:
    """Reads/writes a patient's causal-graph ledger to Supabase or SQLite."""

    def __init__(self, sqlite_path: Optional[str] = None,
                 use_supabase: Optional[bool] = None):
        self.sqlite_path = sqlite_path or os.getenv("AEGIS_CAUSAL_DB") or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "sepsis_twin.db")
        env_has = bool(SUPABASE_URL and SUPABASE_KEY)
        self.use_supabase = env_has if use_supabase is None else (use_supabase and env_has)
        self.supabase_client = None
        if self.use_supabase:
            try:
                from supabase import create_client
                self.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:   # pragma: no cover
                print(f"[causal-store] Supabase init failed ({e}); using SQLite.")
                self.use_supabase = False
        self._init_sqlite()          # always available as the durable fallback

    # ------------------------------------------------------------------ #
    def _init_sqlite(self) -> None:
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS causal_nodes (
                hash TEXT PRIMARY KEY, bed TEXT NOT NULL, seq INTEGER NOT NULL,
                id TEXT NOT NULL, kind TEXT NOT NULL, t REAL NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}', risk REAL NOT NULL,
                confidence REAL NOT NULL, prev_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        cur.execute("CREATE INDEX IF NOT EXISTS causal_nodes_bed_seq ON causal_nodes (bed, seq)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS causal_edges (
                hash TEXT PRIMARY KEY, bed TEXT NOT NULL, seq INTEGER NOT NULL,
                id TEXT NOT NULL, src TEXT NOT NULL, dst TEXT NOT NULL,
                mechanism TEXT NOT NULL, evidence_type TEXT NOT NULL,
                confidence REAL NOT NULL, proven_by TEXT, support INTEGER,
                prev_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        cur.execute("CREATE INDEX IF NOT EXISTS causal_edges_bed_seq ON causal_edges (bed, seq)")
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    def append_node(self, node: Node) -> None:
        r = node.full()
        row = {"hash": r["hash"], "bed": r["bed"], "seq": r["seq"], "id": r["id"],
               "kind": r["kind"], "t": r["t"], "payload": r["payload"],
               "risk": r["risk"], "confidence": r["confidence"],
               "prev_hash": r["prev_hash"]}
        if self.use_supabase:
            try:
                self.supabase_client.table("causal_nodes").upsert(row).execute()
                return
            except Exception as e:   # pragma: no cover
                print(f"[causal-store] supabase append_node failed: {e}")
        self._sqlite_insert("causal_nodes", _NODE_COLS, {**row, "payload": json.dumps(row["payload"])})

    def append_edge(self, edge: Edge) -> None:
        r = edge.full()
        row = {"hash": r["hash"], "bed": self._bed_for_edge(r), "seq": r["seq"],
               "id": r["id"], "src": r["src"], "dst": r["dst"],
               "mechanism": r["mechanism"], "evidence_type": r["evidence_type"],
               "confidence": r["confidence"], "proven_by": r["proven_by"],
               "support": r["support"], "prev_hash": r["prev_hash"]}
        if self.use_supabase:
            try:
                self.supabase_client.table("causal_edges").upsert(row).execute()
                return
            except Exception as e:   # pragma: no cover
                print(f"[causal-store] supabase append_edge failed: {e}")
        self._sqlite_insert("causal_edges", _EDGE_COLS, row)

    def _bed_for_edge(self, r: Dict[str, Any]) -> str:
        # Edges don't carry a bed field; the store is per-graph so we stamp the
        # bed at bind time. Set via bind_to(); default to "bed".
        return getattr(self, "_bed", "bed")

    def _sqlite_insert(self, table: str, cols, row: Dict[str, Any]) -> None:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            placeholders = ", ".join("?" for _ in cols)
            conn.execute(
                f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
                tuple(row.get(c) for c in cols))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    def bind_to(self, graph: TemporalCausalGraph) -> TemporalCausalGraph:
        """Wire this store as the graph's persistence sink (best-effort writes)."""
        self._bed = graph.bed

        def _node_sink(n: Node) -> None:
            try:
                self.append_node(n)
            except Exception as e:   # pragma: no cover
                print(f"[causal-store] node persist skipped: {e}")

        def _edge_sink(e: Edge) -> None:
            try:
                self.append_edge(e)
            except Exception as ex:  # pragma: no cover
                print(f"[causal-store] edge persist skipped: {ex}")

        graph.on_node = _node_sink
        graph.on_edge = _edge_sink
        return graph

    # ------------------------------------------------------------------ #
    def load(self, bed: str) -> Optional[TemporalCausalGraph]:
        """Rehydrate a bed's graph from storage (None if nothing persisted yet)."""
        nodes, edges = self._read(bed)
        if not nodes:
            return None
        return TemporalCausalGraph.from_records(bed, nodes, edges)

    def _read(self, bed: str):
        if self.use_supabase:
            try:
                nr = (self.supabase_client.table("causal_nodes").select("*")
                      .eq("bed", bed).order("seq").execute().data or [])
                er = (self.supabase_client.table("causal_edges").select("*")
                      .eq("bed", bed).order("seq").execute().data or [])
                if nr:
                    return nr, er
            except Exception as e:   # pragma: no cover
                print(f"[causal-store] supabase read failed: {e}")
        # SQLite
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            nrows = conn.execute(
                "SELECT * FROM causal_nodes WHERE bed=? ORDER BY seq", (bed,)).fetchall()
            erows = conn.execute(
                "SELECT * FROM causal_edges WHERE bed=? ORDER BY seq", (bed,)).fetchall()
        finally:
            conn.close()
        nodes = [{**dict(r), "payload": json.loads(r["payload"])} for r in nrows]
        edges = [dict(r) for r in erows]
        return nodes, edges

    # ------------------------------------------------------------------ #
    def list_beds(self) -> List[str]:
        """Every bed (patient/episode) with at least one persisted node."""
        if self.use_supabase:
            try:
                rows = (self.supabase_client.table("causal_nodes")
                        .select("bed").execute().data or [])
                beds = sorted({r["bed"] for r in rows})
                if beds:
                    return beds
            except Exception as e:   # pragma: no cover
                print(f"[causal-store] supabase list_beds failed: {e}")
        conn = sqlite3.connect(self.sqlite_path)
        try:
            rows = conn.execute("SELECT DISTINCT bed FROM causal_nodes").fetchall()
        finally:
            conn.close()
        return sorted(r[0] for r in rows)

    def load_all(self) -> List[TemporalCausalGraph]:
        """Rehydrate every persisted patient graph — the cohort for pattern mining."""
        out: List[TemporalCausalGraph] = []
        for bed in self.list_beds():
            g = self.load(bed)
            if g is not None:
                out.append(g)
        return out

    # ------------------------------------------------------------------ #
    def prune_before(self, bed: str, cutoff_t: float) -> int:
        """Delete events older than ``cutoff_t`` (the retention window). Returns rows
        removed from SQLite. NOTE: pruning the middle of a hash chain breaks
        verify_chain by design — intended for retiring whole closed episodes, not
        trimming a live one."""
        conn = sqlite3.connect(self.sqlite_path)
        try:
            n = conn.execute("DELETE FROM causal_nodes WHERE bed=? AND t < ?",
                             (bed, cutoff_t)).rowcount
            conn.execute("DELETE FROM causal_edges WHERE bed=? AND seq IN ("
                         "SELECT e.seq FROM causal_edges e WHERE e.bed=? AND e.src NOT IN "
                         "(SELECT id FROM causal_nodes WHERE bed=?))", (bed, bed, bed))
            conn.commit()
        finally:
            conn.close()
        return n


# --------------------------------------------------------------------------- #
def _demo() -> None:
    import tempfile
    from backend.aegis.kernel.causal_graph import (
        ALERT, MODEL_PREDICTED, OMISSION, OUTCOME, VITAL)

    path = os.path.join(tempfile.mkdtemp(), "causal_demo.db")
    store = CausalGraphStore(sqlite_path=path, use_supabase=False)
    g = store.bind_to(TemporalCausalGraph(bed="bed_demo"))
    v0 = g.add_event(VITAL, 0.0, {"map": 78}, risk=0.1)
    v1 = g.add_event(VITAL, 60.0, {"map": 60}, risk=0.6)
    om = g.add_event(OMISSION, 70.0, {"given": False}, risk=0.65)
    out = g.add_event(OUTCOME, 200.0, {"state": "organ_failure"}, risk=0.9)
    g.link(v0.id, v1.id, "decline", MODEL_PREDICTED, confidence=0.7)
    g.link(v1.id, om.id, "indicated", MODEL_PREDICTED, confidence=0.7)
    g.link(om.id, out.id, "progression", MODEL_PREDICTED, confidence=0.6)

    reloaded = store.load("bed_demo")
    print("persisted:", g.stats())
    print("reloaded :", reloaded.stats())
    print("chain valid after reload:", reloaded.verify_chain())
    print("heads match:", g.head() == reloaded.head())


if __name__ == "__main__":
    _demo()
