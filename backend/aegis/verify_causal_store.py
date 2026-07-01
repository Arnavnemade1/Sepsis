"""Verify harness for durable causal-graph persistence — claims asserted by measurement.

Using an isolated temp SQLite file (no network, no shared DB), we prove the
persistence layer keeps the ledger's defining property — tamper-evidence — across a
process boundary, and that the system genuinely *resumes* a patient's story:

  1. ROUND-TRIP   — a graph grown live is reloaded with identical head hash, node/
                    edge counts, and a verify_chain() that still passes.
  2. TAMPER       — corrupting a persisted row is detected by verify_chain on reload
                    (the hash chain survives storage, it isn't just recomputed clean).
  3. IDEMPOTENT   — re-persisting the same entries (hash PK) adds no duplicate rows.
  4. RESUME       — a fresh ingestor bound to the store rehydrates the prior episode
                    and chains NEW events onto the persisted tail, still intact, and
                    an incident report spans the whole (old + new) story.

    python -m backend.aegis.verify_causal_store
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from types import SimpleNamespace as A

from backend.aegis.causal_ingest import CausalGraphIngestor
from backend.aegis.causal_store import CausalGraphStore
from backend.aegis.kernel.causal_graph import (
    ALERT, MODEL_PREDICTED, OMISSION, OUTCOME, VITAL, TemporalCausalGraph)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def _assess(band, mp, sq=0.95):
    return A(risk_level=band, ml_probability=mp, detection={}, protocol=None,
             verification=None, signal_quality=sq, triggered_rules=[])


def main() -> int:
    failures = 0
    checks = []
    path = os.path.join(tempfile.mkdtemp(), "causal_store_test.db")

    # ===== 1. ROUND-TRIP ================================================= #
    _hr("1. ROUND-TRIP (grow → persist → reload)")
    store = CausalGraphStore(sqlite_path=path, use_supabase=False)
    g = store.bind_to(TemporalCausalGraph(bed="bed_p"))
    v0 = g.add_event(VITAL, 0.0, {"map": 80, "lactate": 1.4}, risk=0.1)
    v1 = g.add_event(VITAL, 3600.0, {"map": 60, "lactate": 4.5}, risk=0.6)
    om = g.add_event(OMISSION, 3700.0, {"given": False}, risk=0.65)
    out = g.add_event(OUTCOME, 7200.0, {"state": "organ_failure", "map": 50}, risk=0.9)
    g.link(v0.id, v1.id, "decline", MODEL_PREDICTED, confidence=0.7)
    g.link(v1.id, om.id, "indicated", MODEL_PREDICTED, confidence=0.7)
    g.link(om.id, out.id, "progression", MODEL_PREDICTED, confidence=0.6)

    re = store.load("bed_p")
    print(f"    original head={g.head()[:12]} reloaded head={re.head()[:12]}")
    checks.append(("reloaded head hash matches", re.head() == g.head()))
    checks.append(("reloaded node/edge counts match",
                   len(re._nodes) == len(g._nodes) and len(re._edges) == len(g._edges)))
    checks.append(("verify_chain passes after reload", re.verify_chain() is True))
    checks.append(("incident report works on reloaded graph",
                   re.incident_report(out.id)["earliest_actionable"]["kind"] == OMISSION))

    # ===== 2. TAMPER ===================================================== #
    _hr("2. TAMPER (corrupt a stored row → detected on reload)")
    conn = sqlite3.connect(path)
    conn.execute("UPDATE causal_nodes SET risk=0.999 WHERE bed='bed_p' AND kind=?", (VITAL,))
    conn.commit()
    conn.close()
    tampered = store.load("bed_p")
    print(f"    verify_chain after row tamper: {tampered.verify_chain()}")
    checks.append(("tampering with a persisted row breaks the chain on reload",
                   tampered.verify_chain() is False))

    # ===== 3. IDEMPOTENT ================================================= #
    _hr("3. IDEMPOTENT (hash PK → re-persist adds no duplicates)")
    path2 = os.path.join(tempfile.mkdtemp(), "idem.db")
    s2 = CausalGraphStore(sqlite_path=path2, use_supabase=False)
    g2 = s2.bind_to(TemporalCausalGraph(bed="bed_i"))
    a0 = g2.add_event(VITAL, 0.0, {"map": 70}, risk=0.3)
    a1 = g2.add_event(VITAL, 60.0, {"map": 68}, risk=0.35)
    g2.link(a0.id, a1.id, "t", MODEL_PREDICTED)
    for n in g2._nodes.values():     # re-append everything a second time
        s2.append_node(n)
    for e in g2._edges.values():
        s2.append_edge(e)
    conn = sqlite3.connect(path2)
    nrows = conn.execute("SELECT COUNT(*) FROM causal_nodes WHERE bed='bed_i'").fetchone()[0]
    conn.close()
    print(f"    node rows after double-write: {nrows} (expected 2)")
    checks.append(("re-persisting identical entries is a no-op", nrows == 2))

    # ===== 4. RESUME ===================================================== #
    _hr("4. RESUME (new ingestor rehydrates + extends the persisted story)")
    path3 = os.path.join(tempfile.mkdtemp(), "resume.db")
    s3 = CausalGraphStore(sqlite_path=path3, use_supabase=False)
    # First episode: a partial, untreated deterioration.
    ing_a = CausalGraphIngestor(bed="bed_r", store=s3, vital_every_s=60, omission_after_s=180)
    t0 = 1000.0
    for i in range(30):
        band, mp, mp_map = (("LOW", 0.08, 84) if i < 8 else
                            ("MEDIUM", 0.35, 70) if i < 18 else ("HIGH", 0.78, 58))
        ing_a.ingest(_assess(band, mp), {"map": mp_map, "lactate": 1.5 + 0.05 * i}, t0 + i * 60)
    n_after_a = len(ing_a.graph._nodes)
    print(f"    episode A nodes={n_after_a} (persisted)")

    # New process: a fresh ingestor on the same store must resume the story.
    ing_b = CausalGraphIngestor(bed="bed_r", store=s3, vital_every_s=60, omission_after_s=180)
    print(f"    ingestor B rehydrated {ing_b.rehydrated} prior nodes")
    checks.append(("fresh ingestor rehydrates the persisted episode",
                   ing_b.rehydrated == n_after_a))
    checks.append(("rehydrated graph is integrity-intact", ing_b.graph.verify_chain()))

    # Continue the story; the outcome arrives in the second episode.
    for i in range(30, 45):
        ing_b.ingest(_assess("HIGH", 0.82), {"map": 54, "lactate": 4.5}, t0 + i * 60)
    out_r = ing_b.mark_outcome("organ_failure", t0 + 45 * 60, {"map": 50})
    checks.append(("new events chained onto the persisted tail",
                   len(ing_b.graph._nodes) > n_after_a and ing_b.graph.verify_chain()))
    rep = ing_b.graph.incident_report(out_r.id)
    print(f"    incident spans both episodes: earliest={rep['earliest_actionable']['kind']}"
          f"@{rep['earliest_actionable']['t']} lead={rep['lead_time_s']}s")
    checks.append(("incident report spans old+new story (earliest is the episode-A omission)",
                   rep["earliest_actionable"]["kind"] == OMISSION
                   and rep["earliest_actionable"]["t"] < t0 + 30 * 60))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — the clinical story is durable, tamper-evident "
          "across restarts, and genuinely resumable."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
