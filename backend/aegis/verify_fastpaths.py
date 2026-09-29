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
 11. metrics     the per-patient metrics grouped by one sort (early warning, utility,
                 alarm burden, patient bootstrap, alarm-policy series, conformal macro
                 coverage) vs the scan-per-patient loops they replaced, on shuffled
                 patients with NaN scores; and the alarm-budget threshold on tied scores
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

    print("\n11. per-patient metrics")
    _check_metrics()

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




# --------------------------------------------------------------------------- #
# 11. the scan-per-patient loops the metrics used to run, kept as references
# --------------------------------------------------------------------------- #
def _ref_early_warning(y, p, groups, onset_hour, hour_index, threshold, strict=False):
    lead, det, sep, fa, ns = [], 0, 0, 0, 0
    for g in set(groups.tolist()):
        m = groups == g
        order = np.argsort(hour_index[m])
        pg, hg = p[m][order], hour_index[m][order]
        onset = int(onset_hour[m][0])
        alarm_hours = hg[pg >= threshold]
        if onset >= 0:
            sep += 1
            pre = alarm_hours[(alarm_hours < onset) if strict else (alarm_hours <= onset)]
            if pre.size:
                det += 1
                lead.append(float(onset - pre.min()))
        else:
            ns += 1
            fa += bool(alarm_hours.size)
    return {"detection_rate": det / max(sep, 1),
            "median_lead_time_hours": float(np.median(lead)) if lead else 0.0,
            "mean_lead_time_hours": float(np.mean(lead)) if lead else 0.0,
            "false_alarm_rate": fa / max(ns, 1), "septic_patients": sep, "nonseptic_patients": ns}


def _ref_utility(y, pred, groups, hour_index):
    from backend.ml.eval.metrics import compute_prediction_utility as cpu
    obs = inact = best = 0.0
    for g in set(groups.tolist()):
        m = groups == g
        order = np.argsort(hour_index[m])
        yl, pl = y[m][order].astype(int), pred[m][order].astype(int)
        obs += cpu(yl, pl)
        inact += cpu(yl, np.zeros_like(yl))
        best += cpu(yl, yl)
    return float((obs - inact) / (best - inact))


def _ref_burden(y, p, groups, thresholds, peak=None):
    uniq = list(set(groups.tolist()))
    pos = {g: int(y[groups == g].max()) for g in uniq}
    det, far = [], []
    for t in thresholds:
        a_pos = a_neg = 0
        for g in uniq:
            m = groups == g
            hit = (np.max(p[m]) >= t) if peak == "max" else bool((p[m] >= t).any())
            a_pos += bool(pos[g] and hit)
            a_neg += bool(not pos[g] and hit)
        det.append(a_pos / max(sum(1 for g in uniq if pos[g]), 1))
        far.append(a_neg / max(sum(1 for g in uniq if not pos[g]), 1))
    return det, far


def _ref_series(scores, groups, hours, onset, sort_hours=True):
    out = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        order = np.argsort(hours[m]) if sort_hours else np.arange(int(m.sum()))
        out.append((scores[m][order], hours[m][order], int(onset[m][0])))
    return out


def _synthetic_ward(seed: int = 7, n: int = 400):
    """Shuffled rows of ``n`` string-id patients: unique hours per stay, ~15% septic
    (some with onset past their last hour), scores with NaN holes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in rng.permutation(n):
        L = int(rng.integers(3, 70))
        onset = int(rng.integers(0, L + 6)) if rng.random() < 0.15 else -1
        h = np.arange(L) + int(rng.integers(0, 3))
        y = ((h >= onset - 6) & (onset >= 0)).astype(int)
        rows.append((np.full(L, f"p{i:06d}"), h, y, np.full(L, onset)))
    g, h, y, o = (np.concatenate(c) for c in zip(*rows))
    perm = rng.permutation(len(g))
    g, h, y, o = g[perm], h[perm], y[perm], o[perm]
    p = rng.random(len(g)) * 0.2 + 0.1 * y
    p[rng.random(len(g)) < 0.03] = np.nan
    return y, p, g, h, o


def _check_metrics() -> None:
    from backend.ml import alarm_policy as AP
    from backend.ml import conformal as cf
    from backend.ml.eval import metrics as M
    y, p, g, h, o = _synthetic_ward()
    grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 1.0]
    check("early warning == scan-per-patient reference at every threshold",
          all(M.early_warning(y, p, g, o, h, t) == _ref_early_warning(y, p, g, o, h, t) for t in grid),
          f"{len(set(g.tolist()))} patients, {len(y):,} shuffled rows, {int(np.isnan(p).sum())} NaN")
    can_fail("an alarm AT onset not counted (< for <=)", lambda: all(
        M.early_warning(y, p, g, o, h, t) == _ref_early_warning(y, p, g, o, h, t, strict=True)
        for t in grid))
    pred = (np.nan_to_num(p) >= 0.1).astype(int)
    u_new, u_ref = M.normalized_utility(y, pred, g, h), _ref_utility(y, pred, g, h)
    check("normalized utility == reference (to summation order)", abs(u_new - u_ref) < 1e-12,
          f"|diff| {abs(u_new - u_ref):.1e}")
    b = M.alarm_burden(y, p, g, np.array(grid))
    check("alarm burden == reference, NaN rows ignored as before",
          (b["detection_rate"], b["false_alarm_rate"]) == _ref_burden(y, p, g, grid))
    can_fail("a peak that lets NaN win (max for fmax)",
             lambda: _ref_burden(y, p, g, grid, peak="max") == (b["detection_rate"], b["false_alarm_rate"]))
    q = np.nan_to_num(p)
    ref_ci = _ref_bootstrap(y, q, g)
    check("patient bootstrap == reference draw for draw", M.bootstrap_ci(y, q, g, n_boot=40) == ref_ci)
    new_s = [(s.scores, s.hours, s.onset) for s in AP.to_series(q, g, h, o)]

    def same_series(a, c):
        return len(a) == len(c) and all(np.array_equal(x[0], z[0]) and np.array_equal(x[1], z[1])
                                        and x[2] == z[2] for x, z in zip(a, c))
    check("alarm-policy series == reference (id order, hour order, first-row onset)",
          same_series(new_s, _ref_series(q, g, h, o)))
    can_fail("series left in stored row order", lambda: same_series(new_s, _ref_series(q, g, h, o, False)))
    cov = np.random.default_rng(2).random(len(y)) < 0.9
    macro_ref = round(float(np.mean([cov[g == u].mean() for u in set(g.tolist())])), 4)
    check("conformal macro coverage == reference", cf._cov(cov, y, g)["macro"] == macro_ref)

    # Not an optimisation but a correction, held here with the rest of the metric
    # layer: the alarm-budget threshold must honour its budget when scores TIE, which
    # calibrated stacks make the rule (a few hundred distinct scores per fit).
    rr = np.random.default_rng(5)
    cases = [(np.round(rr.random(int(rr.integers(1, 60))), int(rr.integers(0, 3))),
              float(rr.choice([0.0, 0.05, 0.1, 0.15, 0.2, 1.0]))) for _ in range(2000)]

    def lawful(rule) -> bool:
        for pk, far in cases:
            t, cap = rule(pk, far), np.floor(far * len(pk) + 1e-9)
            n_alarm = (pk >= t).sum()
            if n_alarm > cap or any(n_alarm < (pk >= c).sum() <= cap for c in np.unique(pk)):
                return False
        return True

    def kth_peak(pk, far):                      # the rule four modules used to share
        srt = np.sort(pk)
        return srt[min(max(int(np.ceil((1 - far) * len(srt))) - 1, 0), len(srt) - 1)]
    check("budget threshold: never over budget, and the largest lawful alarm set, on ties",
          lawful(M.budget_threshold), "2000 tied cases")
    can_fail("the k-th sorted peak as threshold", lambda: lawful(kth_peak))


def _ref_bootstrap(y, p, groups, n_boot=40, seed=0, alpha=0.05):
    from backend.ml.eval.metrics import auroc
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        rows = np.concatenate([idx_by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[rows])) > 1:
            v = auroc(y[rows], p[rows])
            if not np.isnan(v):
                vals.append(v)
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))


if __name__ == "__main__":
    sys.exit(main())
