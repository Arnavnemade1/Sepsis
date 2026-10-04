"""The ward is scored by the shipped detector: verify the wiring, not the model.

The detector's accuracy is the promotion gate's business (detector_manifest.json); the
ward's own before/after is backend/ml/scripts/ward_auroc.py. What can go wrong in the
WIRING is different, and each check below plants that failure and shows it is caught:

  [1] the index is pinned at the operating point and strictly monotone
  [2] a monotone map leaves AUROC exactly where it was (a non-monotone one does not)
  [3] scores are causal: rewriting a record's future moves no past score
      (and a backward-filled feature, planted, is caught)
  [4] every ward patient is one the serving fit never trained on
      (a known training patient, planted, is refused)
  [5] the replay reads the CURRENT hour's score, never the next one's
  [6] the tick's risk is the detector's score of that hour, exactly
  [7] one copy of the detector in the process
  [8] an imported chart's prior shift moves probability and index together; zero is identity
  [9] conformal: a bed's set is the detector's own; a shifted bed carries no claim
  [10] without a detector the ward still runs, on the fusion engine, and says so
  [11] the measurement artifact describes THIS detector

    python -m backend.aegis.verify_ward_risk
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys

import numpy as np

from backend.aegis.cohort import CohortCluster, RealPatient, build_cohort
from backend.aegis.ward_risk import (HELDOUT_PATH, INDEX_HIGH, INDEX_MEDIUM, IndexMap,
                                     WardDetector, heldout, measured, shared_model)
from backend.ml.eval.metrics import auroc


def _ok(msg: str) -> None:
    print(f"      [PASS] {msg}")


def _caught(msg: str) -> None:
    print(f"      [PASS] planted: {msg}")


def check_pinned_and_monotone(ward: WardDetector) -> None:
    print("[1] the index is pinned at the operating point and strictly monotone")
    m = ward.map
    assert abs(m.index(m.p_medium) - INDEX_MEDIUM) < 1e-12, m.index(m.p_medium)
    assert abs(m.index(m.p_high) - INDEX_HIGH) < 1e-12, m.index(m.p_high)
    grid = np.concatenate([np.logspace(-6, -1, 400), np.linspace(0.1, 0.999, 401)[1:]])
    idx = m.index_array(grid)
    assert np.all(np.diff(idx) > 0), "index not strictly increasing"
    for p in (1e-4, m.p_medium, 0.04, m.p_high, 0.2):
        assert abs(m.probability(m.index(p)) - p) < 1e-9 * max(p, 1e-3), p
    bad = IndexMap(-m.a, m.c, m.p_medium, m.p_high, "planted")
    assert not np.all(np.diff(bad.index_array(grid)) > 0)
    _ok(f"index(MEDIUM {m.p_medium}) = 0.35, index(HIGH {m.p_high}) = 0.70; "
        f"increasing on 800 points; inverse round-trips")
    _caught("a sign-flipped map is not monotone")


def check_auroc_preserved(ward: WardDetector) -> None:
    print("[2] a monotone map leaves AUROC exactly where it was")
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.03).astype(int)
    p = np.clip(rng.beta(1.2, 40, 20000) + y * rng.beta(1.5, 30, 20000), 1e-6, 1 - 1e-6)
    a_p, a_i = auroc(y, p), auroc(y, ward.map.index_array(p))
    assert abs(a_p - a_i) < 1e-12, (a_p, a_i)
    a_bad = auroc(y, np.abs(p - ward.map.p_high))
    assert abs(a_bad - a_p) > 1e-3, (a_bad, a_p)
    _ok(f"AUROC {a_p:.6f} on p and on the index")
    _caught(f"a non-monotone transform moves it ({a_bad:.4f})")


def _first_held_out_record():
    p = build_cohort(n=12, seed=7)[0]
    return p.record


def check_causal(ward: WardDetector) -> None:
    print("[3] scores are causal: rewriting the future moves no past score")
    rec = _first_held_out_record()
    full = ward.series(rec)
    n = len(full)
    cuts = sorted({max(2, n // 4), max(3, n // 2), max(4, (3 * n) // 4)})
    for h in cuts:
        fut = copy.deepcopy(rec)
        f = fut.frame.copy()
        f.loc[h + 1:, ["HR", "Resp", "MAP", "SBP", "DBP"]] = [190.0, 44.0, 45.0, 70.0, 30.0]
        f.loc[h + 1:, "O2Sat"] = 80.0
        fut.frame = f
        moved = np.max(np.abs(ward.series(fut)[:h + 1] - full[:h + 1]))
        assert moved == 0.0, (h, moved)
    # the planted leak: a feature backward-filled from the future
    from backend.ml.data import features as F
    real = F.build_patient_features

    def leaky(r, cfg):
        df = real(r, cfg)
        df["HR"] = r.frame["HR"].astype(float).bfill().fillna(75.0).to_numpy(dtype=np.float32)
        return df
    F.build_patient_features = leaky
    try:
        h = cuts[1]
        fut = copy.deepcopy(rec)
        f = fut.frame.copy()
        f.loc[h + 1:, "HR"] = 190.0
        f.loc[:h, "HR"] = np.nan           # the past has no HR: bfill reaches forward
        fut.frame = f
        base = copy.deepcopy(rec)
        b = base.frame.copy()
        b.loc[:h, "HR"] = np.nan
        base.frame = b
        moved = np.max(np.abs(ward.series(fut)[:h + 1] - ward.series(base)[:h + 1]))
    finally:
        F.build_patient_features = real
    assert moved > 0.0, "a backward fill went unnoticed"
    _ok(f"{rec.patient_id}: future rewritten after hours {cuts}; every earlier score identical")
    _caught(f"a backward-filled HR moves past scores by up to {moved:.4f}")


def check_held_out(ward: WardDetector) -> None:
    print("[4] every ward patient is one the serving fit never trained on")
    ho = heldout()
    have = getattr(ward.model, "fit_digest_", None)
    assert ho["ids"], f"no held-out list at {HELDOUT_PATH}"
    assert ho.get("fit_digest") == have, (ho.get("fit_digest"), have)
    pts = build_cohort(n=32, seed=7)
    outside = [p.record.patient_id for p in pts if p.record.patient_id not in ho["ids"]]
    assert not outside, f"ward patients the detector trained on: {outside}"
    # p000019 was in the old bundled sample, and in the detector's training set
    assert "p000019" not in ho["ids"], "a training patient is on the held-out list"
    _ok(f"all {len(pts)} sample patients are on the list of {len(ho['ids']):,} held out "
        f"from fit {have}")
    _caught("p000019 (a training patient of this fit) is not on the list")


def check_current_hour(cl: CohortCluster) -> None:
    print("[5] the replay reads the current hour's score, never the next one's")
    p = cl.patients[0]
    distinct = 0
    for frac in (0.0, 0.4, 0.8):
        p.pos = float(p.win0 + 2) + frac          # the tick advances, then scores
        cl.tick()
        h = int(p.pos)
        assert abs(cl.profiles[p.bed_id].fused_risk - p.index_at(h)) < 1e-12
        # a reader of the NEXT hour (interpolating toward it, or rounding pos) would
        # show this value instead; the check can only tell them apart where they differ
        distinct += int(abs(p.index_at(h + 1) - p.index_at(h)) > 1e-6)
    assert distinct, "every next-hour score equalled the current one; the check is vacuous"
    _ok(f"at three fractional positions the risk is the score of floor(pos); the next "
        f"hour's score differed in {distinct} of them")


def check_tick_is_detector(cl: CohortCluster, ward: WardDetector) -> None:
    print("[6] the tick's risk is the detector's score of that hour, exactly")
    import pandas as pd
    from backend.ml.data.features import build_patient_features
    cl.run(25)
    worst = 0.0
    for p in cl.patients:
        h = min(p.icu_hour, p.los_total - 1)
        feats = build_patient_features(p.record, ward.cfg).iloc[h:h + 1]
        with np.errstate(all="ignore"):
            one = float(ward.model.predict_proba(pd.DataFrame(feats))[0])
        worst = max(worst, abs(cl.profiles[p.bed_id].probability - one),
                    abs(cl.profiles[p.bed_id].fused_risk - ward.map.index(one)))
    # one row is scored serially, the batch in parallel: raw sums may differ in the
    # last bits (extra-trees sums votes in thread order), never more
    assert worst < 1e-9, worst
    _ok(f"{len(cl.patients)} beds: tick == single-row detector score (max |diff| {worst:.1e})")


def check_one_copy(ward: WardDetector) -> None:
    print("[7] one copy of the detector in the process")
    assert shared_model() is ward.model
    again = WardDetector()
    assert again.model is ward.model
    _ok("every WardDetector shares the process's detector object")


def check_uplift(cl: CohortCluster) -> None:
    print("[8] a prior shift moves probability and index together; zero is identity")
    nid = cl.node_ids[0]
    base_r, base_p = cl.effective_risk(nid), cl.effective_probability(nid)
    cl.susceptibility_provider = lambda n: 0.0
    assert cl.effective_risk(nid) == base_r and cl.effective_probability(nid) == base_p
    cl.susceptibility_provider = lambda n: 1.0 if n == nid else 0.0
    r, p = cl.effective_risk(nid), cl.effective_probability(nid)
    assert r > base_r and p > base_p, (r, base_r, p, base_p)
    assert abs(cl.ward.map.index(p) - r) < 1e-12
    want = 1 / (1 + math.exp(-(math.log(base_p / (1 - base_p)) + 1.0)))
    assert abs(p - want) < 1e-12
    cl.susceptibility_provider = None
    _ok(f"{nid}: p {base_p:.4f} -> {p:.4f} (log-odds +1), index {base_r:.3f} -> {r:.3f}")


def check_conformal(cl: CohortCluster) -> None:
    print("[9] conformal: the bed's set is the detector's own; a shifted bed has no claim")
    if cl.ward.conformal is None:
        print("      [SKIP] no conformal bands for the served detector")
        return
    from backend.ml import conformal
    nid = cl.node_ids[1]
    p = cl._by[nid]
    h = p.icu_hour
    got = cl.certainty(nid)
    want = conformal.predict_set(cl.ward.conformal, p.p_at(h), p.stratum_at(h))
    assert got and got["state"] == want["state"] and got["stratum"] == want["stratum"]
    cl.susceptibility_provider = lambda n: 1.0 if n == nid else 0.0
    assert cl.certainty(nid) is None, "a shifted score carried the detector's claim"
    cl.susceptibility_provider = None
    _ok(f"{nid}: {got['state']} in cell {got['stratum']}; withdrawn under a prior shift")


def check_fallback() -> None:
    print("[10] without a detector the ward runs on the fusion engine, and says so")
    off = WardDetector()
    off.model = None
    cl = CohortCluster(n=4, seed=7, ward=off)
    cl.run(5)
    b = cl.snapshot()["beds"][0]
    assert "probability" not in b and cl.chart(b["bed_id"])["source"] == "fusion"
    assert cl.validation(alarm=0.70)["summary"]["model"] == "personalized"
    _ok("risk from fusion, no probability claimed, chart and validation labelled")


def check_artifact(ward: WardDetector) -> None:
    print("[11] the measurement artifact describes THIS detector")
    m = measured()
    if m is None:
        print("      [SKIP] artifacts/ward_risk.json not generated "
              "(python -m backend.ml.scripts.ward_auroc)")
        return
    assert m["detector"] == ward.op.detector, (m["detector"], ward.op.detector)
    assert m["index_preserves_auroc"]
    d = m["delta_detector_minus_rules"]
    assert d["lo"] > 0, d
    sc = m["scores"]
    _ok(f"{m['patients']:,} held-out patients: ward {sc['rules']['auroc']:.3f} -> "
        f"{sc['detector']['auroc']:.3f} ({d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]); "
        f"with own-baseline learning on top {sc['index_pers']['auroc']:.3f}")


def report_sample(ward: WardDetector) -> None:
    print("[--] the bundled sample, for the record (32 patients: too few to judge by)")
    pts = build_cohort(n=32, seed=7)
    tmp = CohortCluster.__new__(CohortCluster)
    y, det, rules = [], [], []
    for p in pts:
        p.score(ward)
        y.extend(p.record.labels.astype(int).tolist())
        det.extend(p._idx.tolist())
        rules.extend(CohortCluster._rules_series(tmp, p))
    y = np.array(y)
    print(f"      {len(pts)} patients, {len(y):,} hours: detector {auroc(y, np.array(det)):.3f}, "
          f"old rules path {auroc(y, np.array(rules)):.3f}")


def main() -> int:
    print("=" * 70)
    print("AegisOS — the ward is scored by the shipped detector")
    print("=" * 70)
    ward = WardDetector()
    if not ward.ready:
        print("detector or operating point unavailable; nothing to verify")
        return 1
    assert ward.fit_mismatch is None, ward.fit_mismatch
    check_pinned_and_monotone(ward)
    check_auroc_preserved(ward)
    check_causal(ward)
    check_held_out(ward)
    cl = CohortCluster(n=12, seed=7, ward=ward)
    check_current_hour(cl)
    check_tick_is_detector(cl, ward)
    check_one_copy(ward)
    check_uplift(cl)
    check_conformal(cl)
    check_fallback()
    check_artifact(ward)
    report_sample(ward)
    print("-" * 70)
    print("ALL WARD-RISK CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
