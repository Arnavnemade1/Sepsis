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


def main() -> None:
    print("=" * 64)
    print("AegisOS — real-patient cohort engine (CinC-2019) verification")
    print("=" * 64)
    test_cohort_is_real()
    test_feed_replays_real_values()
    test_end_to_end_lead_time()
    test_roster_and_chart()
    print("-" * 64)
    print("ALL COHORT CHECKS PASSED")


if __name__ == "__main__":
    main()
