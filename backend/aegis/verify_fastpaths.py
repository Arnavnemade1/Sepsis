"""The OS's fast paths, held to the slow paths they replaced.

    python -m backend.aegis.verify_fastpaths          (make verify-fastpaths)

Every optimisation in the ward tick is claimed to be EXACT: the same bits, the same
verdicts, the same results. A speedup that silently changes an answer is the worst
kind of bug in a clinical system, because nothing else would notice. So each fast
path is checked against the reference implementation it replaced (kept in the code for
exactly this purpose), and each check is first shown to be able to FAIL: a planted
defect has to turn it red before its green is believed.

  1. encoder     ``encode_many`` (bind tables, gather, sign) vs the per-feature encoder,
                 on cohort states and adversarial ones (missing, NaN, +/-inf, out of
                 range, a numpy NaN, an empty state and its random draw)
  2. bank        the precedent bank vs per-state reference encodes
  3. resonance   sepsis axis (with and without a held-out patient), calibration and
                 every assessment vs the float32-mean / per-modality reference
  4. retrieval   batched queries vs reference encode + matrix-vector + ranking
  5. Z3          the SMT-LIB2 scheduler proof vs the Python-API construction, on random
                 and planted-violation decisions
  6. memo        cached frame / resonance / precedent state vs fresh recomputation on
                 every tick, through a camera artifact and a crisis jump
  7. clocks      bed clocks are phase-staggered: no tick recomputes the whole ward
  8. wire        one encode per broadcast is byte-identical to Starlette's send_json
  9. detector    the single-threaded one-row path scores exactly what the threaded batch
                 path scores, and restores the forest's thread setting
 10. chain       the live incremental integrity check agrees with the full replay, reports
                 a tampered past entry within its scrub bound, and the recent-events read
                 equals the slice of the full serialisation
"""
from __future__ import annotations

import asyncio
import copy
import json
import random
import sys
from typing import Any, Callable, Dict, List

import numpy as np

from backend.aegis import resonance as R
from backend.aegis.cohort import CHANNELS, CohortCluster
from backend.aegis.hdc_memory import SOON_H, TREND_WIN, HDCBioMemory, enrich_series
from backend.aegis.kernel import osstream
from backend.aegis.kernel import scheduler as S

_results: List[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def can_fail(name: str, predicate: Callable[[], bool]) -> None:
    """``predicate`` must return False on a planted defect: the check is non-vacuous."""
    try:
        caught = not predicate()
    except Exception:
        caught = True
    check(f"planted defect caught: {name}", caught)


def _adversarial(states: List[Dict[str, float]], rng) -> List[Dict[str, Any]]:
    out = []
    for st in states:
        d = dict(st)
        for f in list(d):
            u = rng.random()
            if u < 0.15:
                d[f] = None
            elif u < 0.25:
                d[f] = float("nan")
            elif u < 0.30:
                d[f] = d[f] * 50 if d[f] else 1e6
            elif u < 0.33:
                d.pop(f)
        out.append(d)
    return out + [{}, {"hr": None}, {"lactate": float("inf")}, {"map": -float("inf")},
                  {"hr": np.float32("nan")}, {"spo2": 150.0, "hr": -5.0}]


def _fresh_frame(cl: CohortCluster, nid: str) -> Dict[str, float]:
    p = cl._by[nid]
    h = min(p.icu_hour, p.los_total - 1)
    win = [{ch: float(p._ff[t, k]) for k, ch in enumerate(CHANNELS)}
           for t in range(max(0, h - TREND_WIN), h + 1)]
    return enrich_series(win)[-1]


def main() -> int:
    print("=" * 72)
    print("FAST PATHS vs THEIR REFERENCES — exactness, and that each check can fail")
    print("=" * 72)
    cl = CohortCluster(n=20, seed=7)
    hdc, gate = cl.hdc, cl.resonance
    rows = []
    for p in cl.patients:
        rows.extend(HDCBioMemory.rows_from_patient(p))
    states = [r[0] for r in rows]
    rng = np.random.default_rng(0)
    odd = _adversarial(states[:250], rng)

    print("\n1. encoder")
    snap = hdc.eng.rng.bit_generator.state
    ref = np.stack([hdc.encode_reference(s) for s in states + odd])
    hdc.eng.rng.bit_generator.state = snap
    new = hdc.encode_many(states + odd)
    check("encode_many == per-feature reference, bit for bit",
          np.array_equal(ref, new), f"{len(states) + len(odd)} states incl. adversarial")
    hdc.eng.rng.bit_generator.state = snap
    can_fail("one flipped tie-break bit", lambda: np.array_equal(
        ref, _with_flipped_tie(hdc, states + odd)))

    print("\n2. precedent bank")
    check("bank rows == reference encodes",
          np.array_equal(hdc.mem, np.stack([hdc.encode_reference(s) for s in states])))

    print("\n3. resonance gate")
    soon = np.array([1 if (m["label"] == 1 or (m["tto"] is not None and 0 <= m["tto"] <= SOON_H))
                     else 0 for m in hdc.meta], dtype=np.int8)

    def old_axis(exclude):
        M = hdc.mem.astype(np.float32)
        keep = np.array([m["pid"] != exclude for m in hdc.meta], dtype=bool)
        M, s = M[keep], soon[keep]
        a = M[s == 1].mean(0) - M[s == 0].mean(0)
        return a / (np.linalg.norm(a) + 1e-9)
    same = all(np.array_equal(R.CrossModalResonance(hdc).fit(states, exclude_pid=pid).axis,
                              old_axis(pid))
               for pid in [None] + [p.bed_id for p in cl.patients[:4]])
    check("sepsis axis == float32 class-mean reference (and leave-one-patient-out)", same)

    def proj_ref(st, m):
        v = gate._encode_modality(st, m)
        if v is None:
            return 0.0
        vf = v.astype(np.float32)
        n = np.linalg.norm(vf)
        return 0.0 if n < 1e-9 else float(np.dot(vf / n, gate.axis))
    cal_ok = True
    for m in R.MODALITIES:
        a = np.asarray([proj_ref(st, m) for st in states], dtype=np.float64)
        cal_ok &= (float(a.mean()), float(a.std()) or 1.0) == gate.cal[m]
    check("per-modality calibration == reference projections", cal_ok)

    def assess_ref(g, st):
        h = copy.copy(g)
        h._z_all = lambda s: {m: (proj_ref(s, m) - g.cal[m][0]) / g.cal[m][1] for m in R.MODALITIES}
        return R.CrossModalResonance.assess(h, st)
    probe = states[::2] + [s for s in odd if s][:80]
    bad = sum(assess_ref(gate, st) != gate.assess(st) for st in probe)
    check("assess == reference on every probe state", bad == 0, f"{len(probe)} states, {bad} differ")
    g_bad = copy.copy(gate)
    g_bad.cal = dict(gate.cal)
    g_bad.cal["camera"] = (gate.cal["camera"][0] + 0.5, gate.cal["camera"][1])
    can_fail("a shifted camera calibration", lambda: all(
        assess_ref(g_bad, st) == gate.assess(st) for st in probe[:60]))

    print("\n4. precedent retrieval")
    qs, pids = states[::5][:40], [r[3] for r in rows[::5][:40]]
    got = hdc.query_batch(qs, k=16, exclude_pids=pids)

    def query_ref(st, pid):
        q = hdc.encode_reference(st).astype(np.float32)
        with np.errstate(all="ignore"):
            return hdc._finish((hdc._M @ q) / float(hdc.eng.D), 16, pid)
    check("batched queries == reference encode + matvec + ranking",
          got == [query_ref(s, p) for s, p in zip(qs, pids)], f"{len(qs)} queries")

    print("\n5. scheduler proof")
    rr = random.Random(1)
    agree = total = violated = proven = 0
    for trial in range(150):
        sch = S.CognitiveScheduler(lane_budget=rr.choice([8, 30, 64, 160]))
        for i in range(rr.choice([1, 2, 5, 12, 40])):
            sch.register_node(f"bed_{i}", weight=rr.uniform(0.5, 2))
            for _ in range(rr.randint(1, 6)):
                sch.observe(f"bed_{i}", rr.random(), {"hr": 80})
        d = sch.schedule()
        if trial % 3 == 1:
            nid, kind = rr.choice(list(sch.nodes)), rr.randint(0, 3)
            if kind == 0:
                d.lanes[nid]["safety"] = 0
            elif kind == 1:
                d.lanes[nid]["ml_tracking"] += 500
            elif kind == 2:
                d.lanes[nid]["render"], d.mode = 3, "PREEMPT"
            else:
                d.lanes[nid]["ml_tracking"] = -2
        a, b = sch._verify_decision_z3_api(d)[0], sch._verify_decision_z3(d)[0]
        total += 1
        agree += a == b
        violated += b == "violated"
        proven += b == "proven"
    check("SMT-LIB2 verdict == Python-API verdict on every decision", agree == total,
          f"{agree}/{total} ({proven} proven, {violated} violated)")
    check("both verdicts occur (the comparison is not one-sided)", violated > 0 and proven > 0)

    print("\n6. per-bed memo")
    cl2 = CohortCluster(n=20, seed=7)
    cl2.tick()
    cl2.inject_noise_surge("bed_2")
    cl2.inject_crisis("bed_5")
    mism = artifact_ticks = 0
    for _ in range(60):
        was = {n: cl2._artifact.get(n, 0) for n in cl2.node_ids}
        cl2.tick()
        fresh_q = cl2.hdc.query_batch([_fresh_frame(cl2, n) for n in cl2.node_ids], k=16,
                                      exclude_pids=list(cl2.node_ids))
        for nid, q in zip(cl2.node_ids, fresh_q):
            est = _fresh_frame(cl2, nid)
            mism += cl2._frame(nid)[1] != est
            if was[nid] > 0:
                artifact_ticks += 1
                est = dict(est)
                est["hr__tr"] = 185.0 - float(est.get("hr", 80.0))
                est["hr"] = 185.0
            mism += cl2.resonance.assess(est) != cl2.resonance_state[nid]
            mism += cl2.hdc_state[nid] != q
    check("memoised frame / resonance / precedent == fresh recomputation every tick",
          mism == 0, f"60 ticks x 20 beds, {artifact_ticks} artifact bed-ticks, {mism} differ")
    check("the camera-artifact path was exercised", artifact_ticks >= 10)

    def stale_memo_detected() -> bool:
        c = CohortCluster(n=8, seed=7)
        c._frame = _stale(c._frame)                     # planted: never refresh the frame
        for _ in range(40):
            c.tick()
        return all(c._frame(n)[1] == _fresh_frame(c, n) for n in c.node_ids)
    can_fail("a frame memo that ignores the hour", stale_memo_detected)

    print("\n7. clock phases")
    frac = _max_hour_change_fraction(CohortCluster(n=40, seed=7))
    check("no tick recomputes most of the ward (phase-staggered clocks)", frac <= 0.35,
          f"worst tick: {frac:.0%} of beds changed hour")
    locked = CohortCluster(n=40, seed=7)
    for p in locked.patients:
        p.phase, p.pos = 0.0, float(p.win0)
    can_fail("phase-locked clocks", lambda: _max_hour_change_fraction(locked) <= 0.35)

    print("\n8. broadcast encoding")
    payload = cl.snapshot() if hasattr(cl, "snapshot") else {"x": 1}
    sent: List[str] = []

    class FakeWS:
        async def send(self, message):
            sent.append(message["text"])

        async def send_text(self, text):
            sent.append(text)
    from starlette.websockets import WebSocket
    fake = FakeWS()
    asyncio.run(WebSocket.send_json(fake, payload))      # Starlette's own encoding
    asyncio.run(osstream.send(fake, osstream.encode(payload)))
    check("one encode == Starlette send_json bytes", len(sent) == 2 and sent[0] == sent[1],
          f"{len(sent[0]) if sent else 0} bytes")
    check("an unencodable payload falls back to send_json", not isinstance(
        osstream.encode({"x": object()}), osstream.Wire))

    print("\n9. detector serving path")
    import os
    import joblib
    import pandas as pd
    from backend.ml import config as mlcfg
    path = os.path.join(mlcfg.ARTIFACTS_DIR, "gbm_wearable.joblib")
    det = joblib.load(path)
    if hasattr(det, "base_models_"):
        rr_ = np.random.default_rng(1)
        X = pd.DataFrame(rr_.normal(size=(300, len(det.feature_names_))) * 0.6,
                         columns=det.feature_names_)
        batch = det.predict_proba(X)                       # > SERIAL_ROWS rows: threaded
        one = np.array([det.predict_proba(X.iloc[[i]])[0] for i in range(300)])
        jobs = [getattr(m, "n_jobs", None) for m in det.base_models_.values()]
        check("one-row serial scores == threaded batch scores, bit for bit",
              np.array_equal(one, batch), f"300 rows, max |diff| {np.abs(one - batch).max():.1e}")
        det2 = joblib.load(path)
        check("the forest's thread setting is restored after a serial call",
              jobs == [getattr(m, "n_jobs", None) for m in det2.base_models_.values()])
    else:                                                  # pragma: no cover
        check("shipped detector is a stacked ensemble", False, type(det).__name__)

    print("\n10. causal chain, live")
    from dataclasses import replace
    from backend.aegis.kernel.causal_graph import (ALERT, MODEL_PREDICTED, VITAL,
                                                   TemporalCausalGraph)
    g = TemporalCausalGraph("bed_fp")
    prev = None
    for i in range(1500):
        node = g.add_event(VITAL if i % 3 else ALERT, float(i), {"map": 60.0 + i % 30}, risk=0.1)
        if prev is not None:
            g.link(prev.id, node.id, "m", MODEL_PREDICTED)
        prev = node
    full = g.to_dict()
    check("recent events == the tail of the full serialisation",
          all(g.recent_nodes(k) == full["nodes"][-k:] for k in (1, 6, 40)))
    check("clean chain: live integrity == full replay == True",
          g.stats()["integrity_ok"] is True and g.verify_chain() is True)
    victim = next(k for kind, k in g._order[4:] if kind == "n")
    g._nodes[victim] = replace(g._nodes[victim], risk=0.99)      # tamper an OLD entry
    bound = -(-len(g._order) // 256)
    calls = 1
    while g.stats()["integrity_ok"] and calls <= bound + 1:
        calls += 1
    check("a tampered past entry is reported within the scrub bound",
          calls <= bound and g.verify_chain() is False, f"{calls} calls, bound {bound}")
    can_fail("a scrub that never re-hashes old entries", lambda: _scrubless_misses(replace))

    ok = all(_results)
    print("\n" + ("ALL %d FAST-PATH CHECKS PASSED" % len(_results) if ok else
                  "FAILED: %d of %d" % (len(_results) - sum(_results), len(_results))))
    return 0 if ok else 1


def _scrubless_misses(replace) -> bool:
    """True when a live check WITHOUT the scrub still reports a tampered past entry —
    it must not: that is what the scrub is for (so this returning False proves the
    tamper check above is not vacuous)."""
    from backend.aegis.kernel.causal_graph import MODEL_PREDICTED, VITAL, TemporalCausalGraph
    g = TemporalCausalGraph("bed_ns")
    prev = None
    for i in range(600):
        node = g.add_event(VITAL, float(i), {"map": 70.0}, risk=0.1)
        if prev is not None:
            g.link(prev.id, node.id, "m", MODEL_PREDICTED)
        prev = node
    g.integrity(scrub=0)
    victim = next(k for kind, k in g._order[4:] if kind == "n")
    g._nodes[victim] = replace(g._nodes[victim], risk=0.99)
    return all(g.integrity(scrub=0) is False for _ in range(20))


def _with_flipped_tie(hdc: HDCBioMemory, states) -> np.ndarray:
    tie = hdc.eng._tie_break.copy()
    try:
        hdc.eng._tie_break[0] = -hdc.eng._tie_break[0]
        return hdc.encode_many(states)
    finally:
        hdc.eng._tie_break[:] = tie


def _stale(fn):
    memo: Dict[str, Any] = {}

    def frame(nid):
        if nid not in memo:
            memo[nid] = fn(nid)
        return memo[nid]
    return frame


def _max_hour_change_fraction(cl: CohortCluster, ticks: int = 30) -> float:
    worst = 0.0
    prev = {p.bed_id: p.icu_hour for p in cl.patients}
    for _ in range(ticks):
        for p in cl.patients:
            p.advance(cl.HOURS_PER_TICK)
        now = {p.bed_id: p.icu_hour for p in cl.patients}
        worst = max(worst, sum(now[b] != prev[b] for b in now) / len(now))
        prev = now
    return worst


if __name__ == "__main__":
    sys.exit(main())
