#!/usr/bin/env python3
"""Adversarial audit of the Scenario Lab (make verify-scenarios).

The lab's value rests on negative claims — "a synthetic patient cannot reach an
accuracy number", "the sandbox is not the ward", "a scenario is reproducible", "the
grader can actually fail something". Negative claims are only falsifiable, so each
check below plants the pathology it guards against and asserts the guard fires.

The load-bearing one is the firewall. A generator good enough to be useful is good
enough to be mistaken for data, and this project has already shipped a detector that
looked flawless against its own simulator and scored near chance on real patients.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np

from backend.aegis import scenarios as sc

PASS, FAIL = "  \033[32mPASS\033[0m", "  \033[31mFAIL\033[0m"
_results: List[Tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def wrapped(*a, **k):
            print(f"\n[{name}]")
            try:
                fn(*a, **k)
                print(PASS)
                _results.append((name, True, ""))
            except AssertionError as exc:
                print(f"{FAIL} {exc}")
                _results.append((name, False, str(exc)))
        return wrapped
    return deco


@check("a spec plus a seed is a complete description")
def check_determinism():
    spec = sc.ScenarioSpec(name="d", arc="septic_shock", los_hours=40, onset_hour=26, seed=3)
    a, b = sc.synthesize(spec), sc.synthesize(spec)
    same = np.allclose(a.frame["HR"].to_numpy(), b.frame["HR"].to_numpy(), equal_nan=True)
    other = sc.synthesize(sc.ScenarioSpec(name="d", arc="septic_shock", los_hours=40,
                                          onset_hour=26, seed=4))
    differs = not np.allclose(a.frame["HR"].to_numpy(), other.frame["HR"].to_numpy(),
                              equal_nan=True)
    print(f"    planted: the same spec twice, then a one-digit seed change")
    print(f"    identical under the same seed: {same}; different under seed+1: {differs}")
    assert same, "the same spec produced two different patients — nothing is reproducible"
    assert differs, "changing the seed changed nothing — the seed is not wired in"
    assert a.patient_id == b.patient_id, "fingerprint is not stable across builds"


@check("synthetic records cannot reach the evaluation harness")
def check_firewall():
    from backend.ml.data.features import FeatureConfig, build_dataset
    rec = sc.synthesize(sc.ScenarioSpec(name="f", arc="septic_shock",
                                        los_hours=32, onset_hour=20))
    refused = False
    try:
        build_dataset([rec], FeatureConfig.wearable_only())
    except ValueError as exc:
        refused = "synthetic" in str(exc).lower()
    print("    planted: a synthetic record handed straight to build_dataset")
    print(f"    refused: {refused}")
    assert refused, "a hand-written patient reached the accuracy pipeline unchallenged"
    ds = build_dataset([rec], FeatureConfig.wearable_only(), allow_synthetic=True)
    print(f"    explicit opt-in still builds ({ds.X.shape[0]} rows) — the door is a "
          f"door, not a wall")
    assert ds.X.shape[0] == rec.n_hours


@check("the provenance marker cannot be lost in transit")
def check_provenance():
    rec = sc.synthesize(sc.ScenarioSpec(name="p", arc="stable", los_hours=24,
                                        onset_hour=None))
    print(f"    set_name={rec.set_name!r}  id={rec.patient_id!r}")
    assert sc.is_synthetic(rec), "is_synthetic does not recognise its own output"
    assert rec.patient_id.startswith("SYN-"), "the id does not announce itself"
    assert rec.set_name == sc.SYNTHETIC_SET
    # A real record must NOT trip the marker, or the firewall blocks everything and
    # gets disabled within a week.
    class _Fake:
        set_name = "training_setA"
    assert not sc.is_synthetic(_Fake()), "a real record was flagged synthetic"


@check("the sandbox is not the ward")
def check_isolation():
    lab = sc.ScenarioLab()
    bed = lab.create(sc.ScenarioSpec(name="i", arc="septic_shock", los_hours=30,
                                     onset_hour=20))
    print(f"    handle {bed!r} — prefix {sc.ScenarioLab.PREFIX!r}")
    assert bed.startswith(sc.ScenarioLab.PREFIX), "a sandbox handle looks like a real bed"
    import backend.aegis.server as srv
    live = set(getattr(srv.cluster, "node_ids", []) or [])
    assert bed not in live, "a scenario bed appeared in the live cohort"
    assert not any(b.startswith(sc.ScenarioLab.PREFIX) for b in live), \
        "the live wing contains sandbox beds"
    print(f"    live wing holds {len(live)} beds, none of them sandbox")
    # And the cap is real, or the sandbox is a memory leak with a nice name.
    lab2 = sc.ScenarioLab()
    for i in range(sc.ScenarioLab.MAX_SCENARIOS):
        lab2.create(sc.ScenarioSpec(name=f"s{i}", arc="stable", los_hours=12,
                                    onset_hour=None))
    capped = False
    try:
        lab2.create(sc.ScenarioSpec(name="over", arc="stable", los_hours=12,
                                    onset_hour=None))
    except ValueError:
        capped = True
    assert capped, f"the sandbox accepted more than {sc.ScenarioLab.MAX_SCENARIOS}"


@check("a synthetic label means what a real one means")
def check_label_semantics():
    onset = 30
    rec = sc.synthesize(sc.ScenarioSpec(name="l", arc="septic_shock", los_hours=48,
                                        onset_hour=onset))
    expected = onset - sc.LABEL_LEAD
    print(f"    clinical onset h{onset}, label flips h{rec.label_onset_hour} "
          f"(challenge lead {sc.LABEL_LEAD}h)")
    assert rec.label_onset_hour == expected, (
        f"label flips at {rec.label_onset_hour}, not {expected} — every lead-time "
        f"number computed on a scenario would be off by the difference")
    quiet = sc.synthesize(sc.ScenarioSpec(name="q", arc="stable", los_hours=24,
                                          onset_hour=None))
    assert not quiet.is_septic, "a never-septic scenario carries positive labels"


@check("the generator cannot emit a patient physiology forbids")
def check_bounds():
    worst = []
    for arc in sc.ARCS:
        for sev in (0.4, 1.0, 1.6):
            spec = sc.ScenarioSpec(name="b", arc=arc, los_hours=72,
                                   onset_hour=None if arc in ("stable", "recovering",
                                                              "post_op_fever") else 40,
                                   severity=sev, seed=int(sev * 100),
                                   interventions=[(20, "vasopressors"), (30, "fluids")])
            f = sc.synthesize(spec).frame
            for ch, (lo, hi) in sc._BOUNDS.items():
                col = f[ch].to_numpy(dtype=float)
                col = col[~np.isnan(col)]
                if col.size and (col.min() < lo - 1e-6 or col.max() > hi + 1e-6):
                    worst.append(f"{arc}/{ch}: [{col.min():.1f},{col.max():.1f}] "
                                 f"outside [{lo},{hi}]")
    print(f"    planted: every arc at 3 severities with vasopressors AND fluids stacked")
    print(f"    out-of-range channels: {worst or 'none'}")
    assert not worst, f"impossible physiology emitted: {worst[:3]}"


@check("the suite grades confounders, not just sick patients")
def check_negative_controls():
    cases = sc.suite()
    negatives = [s for s, e in cases
                 if any(x.kind in ("never_alarms", "peak_risk_at_most") for x in e)]
    positives = [s for s, e in cases if any(x.kind in ("alarms_at_all",
                                                       "alarm_lead_at_least") for x in e)]
    print(f"    {len(positives)} must-catch, {len(negatives)} must-ignore, "
          f"{len(cases)} total")
    assert negatives, ("a suite with no must-stay-quiet case scores a detector that "
                       "alarms on everyone as perfect")
    assert positives, "a suite with nothing to catch proves nothing"
    assert all(e for s, e in cases if s.name not in ("treated-early",)), \
        "a graded scenario carries no expectations"


@check("the grader can actually fail something")
def check_grader_discriminates():
    spec = sc.ScenarioSpec(name="g", arc="septic_shock", los_hours=40, onset_hour=26)
    thr = {"high": 0.5, "medium": 0.2}
    exps = [sc.Expectation("alarms_at_all", 1, "must alarm"),
            sc.Expectation("alarm_lead_at_least", 5, "must be early")]
    # Planted oracle: alarms from hour 10, i.e. 16h of lead.
    oracle = np.zeros(40); oracle[10:] = 0.9
    good, _ = sc._grade(spec, oracle, exps, thr)
    # Planted blind detector: never alarms at all.
    blind, _ = sc._grade(spec, np.full(40, 0.01), exps, thr)
    print(f"    planted an oracle -> {sum(g.passed for g in good)}/2 pass")
    print(f"    planted a blind detector -> {sum(g.passed for g in blind)}/2 pass")
    assert all(g.passed for g in good), "the grader failed a detector that did everything right"
    assert not any(g.passed for g in blind), "the grader passed a detector that never alarmed"
    # And a late alarm must fail the lead check specifically, not pass on a technicality.
    late = np.zeros(40); late[30:] = 0.9
    res, _ = sc._grade(spec, late, exps, thr)
    by = {g.kind: g for g in res}
    assert by["alarms_at_all"].passed and not by["alarm_lead_at_least"].passed, \
        "an alarm 4h AFTER onset satisfied a 5h-lead requirement"
    print("    planted an alarm 4h after onset -> alarms_at_all ok, lead check fails")


@check("a scenario result has nowhere to put an accuracy number")
def check_no_accuracy_fields():
    from dataclasses import fields
    names = {f.name.lower() for f in fields(sc.ScenarioResult)}
    banned = {"auroc", "auprc", "utility", "accuracy", "sensitivity", "specificity",
              "precision", "recall", "f1", "ece", "brier"}
    found = names & banned
    print(f"    result fields: {sorted(names)}")
    assert not found, (
        f"ScenarioResult exposes {sorted(found)} — a field like that is an invitation "
        "to quote a synthetic accuracy, which is the exact mistake this lab exists "
        "to avoid")


@check("two agent tools cannot share a name")
def check_no_duplicate_tools():
    from backend.aegis.agent import Tool, _assert_unique, _catalogue
    cat = _catalogue()
    _assert_unique(cat)                              # the real catalogue must be clean
    print(f"    live catalogue: {len(cat)} tools, no duplicates")
    raised = False
    try:
        _assert_unique(list(cat) + [Tool(cat[0].name, cat[0].kind, "shadow")])
    except ValueError as exc:
        raised = cat[0].name in str(exc)
    print(f"    planted a shadow of {cat[0].name!r} -> refused: {raised}")
    assert raised, ("a duplicate tool name was accepted; bind() keys by name, so one "
                    "of the pair becomes dead code the model can still see")


def main() -> int:
    print("=" * 78)
    print("SCENARIO LAB INTEGRITY AUDIT — every check plants a known pathology")
    print("=" * 78)
    check_determinism()
    check_firewall()
    check_provenance()
    check_isolation()
    check_label_semantics()
    check_bounds()
    check_negative_controls()
    check_grader_discriminates()
    check_no_accuracy_fields()
    check_no_duplicate_tools()

    ok = sum(1 for _, p, _ in _results if p)
    print("\n" + "=" * 78)
    print(f"{ok}/{len(_results)} checks passed")
    for nm, p, err in _results:
        if not p:
            print(f"  FAILED: {nm} — {err}")
    print("=" * 78)
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
