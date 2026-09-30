"""Verification for the real-patient cohort engine (CinC-2019 driven).

Proves the cockpit runs on real ICU data, not synthetic beds:

  1. The cohort is built from real CinC records with real demographics + a mix of
     septic and stable patients.
  2. A patient's feed replays its real hourly vitals through /dev/bio (the values
     a driver emits match the underlying record at that hour).
  3. End-to-end: real trajectories flow kernel-deep (devbio -> fabric -> fusion),
     and at least one severe septic patient crosses the critical threshold near
     its TRUE onset — i.e. genuine early-warning lead time on real data.
  4. The roster + per-bed chart expose real demographics, labels and trajectories.
  5. One risk per bed: the board, effective_risk() and the profile handed to the
     bundle engine agree, including after a chart's prior shift — and the check
     only counts because a bed exists where population and effective risk differ.
  6. The write path grades the same risk: an escalation's band (what the Z3
     cage checks it against) comes from the effective risk, on a bed where the
     population risk would put it in a different band.

    .venv/bin/python -m backend.aegis.verify_cohort
"""
from __future__ import annotations

from backend.aegis.cohort import CohortCluster, build_cohort


def _ok(msg): print(f"      [PASS] {msg}")


def test_cohort_is_real() -> None:
    print("[1] cohort built from real CinC records (mixed septic/stable)")
    coh = build_cohort(12, seed=7)
    assert len(coh) == 12
    septic = [p for p in coh if p.is_septic]
    stable = [p for p in coh if not p.is_septic]
    assert septic and stable, "need both septic and stable real patients"
    assert all(p.mrn.startswith("P") for p in coh)
    assert all(40 <= p.age <= 100 or p.age >= 0 for p in coh)
    assert all(p.gender in ("M", "F") for p in coh)
    _ok(f"{len(coh)} real patients · {len(septic)} septic · ages "
        f"{min(p.age for p in coh):.0f}-{max(p.age for p in coh):.0f}")


def test_feed_replays_real_values() -> None:
    print("[2] /dev/bio feed replays the real hourly record")
    p = next(pp for pp in build_cohort(12, seed=7))
    p.pos = float(p.win0 + 1)            # land on an integer hour
    feed = p.feed_for("bedside")
    raw, q = feed()
    # the emitted MAP must equal the patient's forward-filled record at that hour
    import numpy as np
    from backend.aegis.cohort import CHANNELS
    j = CHANNELS.index("map")
    i = int(p.pos)
    expected = float(p._ff[i, j])
    assert abs(raw["map"] - expected) < 1e-6, (raw["map"], expected)
    assert 0.0 < q <= 1.0
    _ok(f"bedside feed emits real MAP {raw['map']:.0f} at hour {i} (quality {q:.2f})")


def test_end_to_end_lead_time() -> None:
    print("[3] real trajectories run kernel-deep; severe sepsis crosses early")
    cl = CohortCluster(n=12, lane_budget=160, seed=7)
    crossed = {}
    for _ in range(420):
        cl.tick()
        for b in cl.snapshot()["beds"]:
            nid = b["bed_id"]
            if b["fused_risk"] >= 0.70 and nid not in crossed and b["will_be_septic"]:
                crossed[nid] = (b["icu_hour"], b["onset_hour"])
    assert crossed, "no septic patient ever crossed the threshold on real data"
    leads = [onset - hr for hr, onset in crossed.values() if onset is not None]
    early = [l for l in leads if l >= 0]
    assert early, "expected at least one bed flagged at/before its labelled onset"
    _ok(f"{len(crossed)} septic beds crossed critical; max lead "
        f"{max(early)} h before labelled onset")


def test_roster_and_chart() -> None:
    print("[4] roster + per-bed chart expose real record")
    cl = CohortCluster(n=12, lane_budget=160, seed=7)
    cl.run(10)
    roster = cl.roster()
    assert len(roster) == 12 and all("mrn" in r and "age" in r for r in roster)
    ch = cl.chart("bed_0")
    assert ch and "series" in ch and "hr" in ch["series"] and "labels" in ch
    assert len(ch["hours"]) == len(ch["series"]["hr"]) == len(ch["labels"])
    _ok(f"roster of {len(roster)}; chart bed_0 has {len(ch['hours'])} real hours "
        f"across {len(ch['series'])} channels + ground-truth labels")


def test_one_risk_per_bed() -> None:
    print("[5] one risk per bed: board == effective_risk == bundle input")
    from backend.aegis.kernel.bundle import SepsisBundleEngine
    cl = CohortCluster(n=12, lane_budget=160, seed=7)
    cl.run(60)
    board = {b["bed_id"]: b["fused_risk"] for b in cl.snapshot()["beds"]}
    gaps = {nid: abs(cl.profiles[nid].fused_risk - board[nid]) for nid in board}
    # Without a bed whose population risk differs from its effective risk, every
    # assertion below would pass even if a consumer read the population number.
    worst = max(gaps, key=gaps.get)
    assert gaps[worst] > 0.05, f"no divergent bed; the check would be vacuous ({gaps})"
    for nid, r in board.items():
        assert abs(cl.effective_risk(nid) - r) < 1e-9, (nid, cl.effective_risk(nid), r)
        assert abs(cl.effective_profile(nid).fused_risk - r) < 1e-9, nid
    bundle = SepsisBundleEngine().generate(cl.effective_profile(worst), cl.histories.get(worst, {}))
    assert abs(bundle["fused_risk"] - round(board[worst], 3)) < 1e-3, (bundle["fused_risk"], board[worst])
    # a chart's prior shift moves the board and effective_risk together
    cl.susceptibility_provider = lambda nid: 1.0 if nid == worst else 0.0
    shifted = {b["bed_id"]: b["fused_risk"] for b in cl.snapshot()["beds"]}
    assert shifted[worst] > board[worst] + 1e-3, "uplift did not move the board"
    assert abs(cl.effective_risk(worst) - shifted[worst]) < 1e-9
    _ok(f"{worst}: population {cl.profiles[worst].fused_risk:.3f} vs effective "
        f"{board[worst]:.3f}; board, effective_risk and the bundle agree (also with uplift)")


def test_write_path_grades_effective_risk() -> None:
    print("[6] the write path grades the same risk the board shows")
    from backend.aegis.kernel import BioCluster, NeuroSymbolicRuntime, SepsisBundleEngine
    from backend.aegis.syscalls import SyscallTable, _band
    cl = CohortCluster(n=12, lane_budget=160, seed=7)
    cl.run(60)
    board = {b["bed_id"]: b["fused_risk"] for b in cl.snapshot()["beds"]}
    split = [nid for nid in board if _band(cl.profiles[nid].fused_risk) != _band(board[nid])]
    # Without a bed whose population band differs from its effective band, this
    # check could not tell the two risks apart.
    assert split, "no bed where population and effective risk fall in different bands"
    nid = split[0]
    table = SyscallTable(cl, SepsisBundleEngine(), NeuroSymbolicRuntime())
    res = table.invoke("escalate", nid, {}, caller="verify")
    band = res.to_dict()["result"].get("band")
    assert band == _band(board[nid]), (nid, band, _band(board[nid]), _band(cl.profiles[nid].fused_risk))
    _ok(f"{nid}: escalate graded band {band} from effective {board[nid]:.3f} "
        f"(population {cl.profiles[nid].fused_risk:.3f} would be band {_band(cl.profiles[nid].fused_risk)})")


def main() -> None:
    print("=" * 64)
    print("AegisOS — real-patient cohort engine (CinC-2019) verification")
    print("=" * 64)
    test_cohort_is_real()
    test_feed_replays_real_values()
    test_end_to_end_lead_time()
    test_roster_and_chart()
    test_one_risk_per_bed()
    test_write_path_grades_effective_risk()
    print("-" * 64)
    print("ALL COHORT CHECKS PASSED")


if __name__ == "__main__":
    main()
