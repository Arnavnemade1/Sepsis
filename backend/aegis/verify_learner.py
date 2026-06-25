"""Verification for the per-patient behavioural learner.

Proves the learning is real and honest — it does what it claims and its effect on
detection is measured, not asserted:

  1. The baseline tracks a patient's *own* stable norm; a value normal for the
     population but abnormal for THIS patient yields a positive deviation while a
     value at their (even chronically deranged) baseline yields ~0.
  2. The baseline gate freezes during a sustained deviation, so a slow drift can't
     drag the reference along with it (no boiling-frog).
  3. Deviation is direction-aware (only worsening counts; temp two-sided) and
     smoothed, so a single-hour blip is damped but a sustained shift builds up.
  4. personalized_risk is two-sided: it DAMPENS at-baseline risk and LIFTS on
     deviation, and is gated by maturity (immature -> trust population).
  5. End-to-end on real CinC patients: personalization does not hurt detection and
     surfaces some patients earlier. Reported as MEASURED — the honest finding is
     it is roughly a wash on the global alarm metric (its value is the per-patient
     interpretable signal, not a global threshold shift).

    .venv/bin/python -m backend.aegis.verify_learner
"""
from __future__ import annotations

from backend.aegis.learner import PatientBaselineLearner, personalized_risk


def _ok(msg): print(f"      [PASS] {msg}")


def _settle(lz, vals, n=14):
    for _ in range(n):
        lz.observe(vals)


def test_personal_baseline() -> None:
    print("[1] baseline learns the patient's OWN norm")
    # a patient whose resting HR is genuinely 52 (athlete) — population-normal 78
    lz = PatientBaselineLearner()
    _settle(lz, {"hr": 52, "temp_c": 37.0, "resp_rate": 13, "spo2": 98,
                 "map": 88, "wbc": 7, "lactate": 1.0})
    at_base = lz.per_channel({"hr": 52, "temp_c": 37.0, "resp_rate": 13, "spo2": 98,
                              "map": 88, "wbc": 7, "lactate": 1.0})["hr"]
    drift = lz.per_channel({"hr": 78, "temp_c": 37.0, "resp_rate": 13, "spo2": 98,
                            "map": 88, "wbc": 7, "lactate": 1.0})["hr"]
    assert at_base < 0.5 and drift > 2.0, (at_base, drift)
    _ok(f"HR 78 (population-normal) reads {drift:.1f}σ above this patient's own "
        f"baseline of 52; at-baseline reads {at_base:.1f}σ")


def test_gate_freezes() -> None:
    print("[2] baseline gate freezes during sustained deviation (no boiling-frog)")
    lz = PatientBaselineLearner()
    base = {"hr": 70, "temp_c": 37.0, "resp_rate": 14, "spo2": 98,
            "map": 86, "wbc": 8, "lactate": 1.1}
    _settle(lz, base)
    frozen = lz.mean["hr"]
    # now a sustained climb — the baseline must NOT chase it up
    for hr in range(75, 110, 3):
        lz.observe({**base, "hr": hr})
    drift = lz.mean["hr"]
    assert drift - frozen < 6.0, (frozen, drift)
    _ok(f"baseline HR held at {drift:.1f} despite a sustained climb to 108 "
        f"(moved <6 bpm) — slow drift can't capture the reference")


def test_smoothing_and_direction() -> None:
    print("[3] deviation is smoothed + direction-aware")
    base = {"hr": 70, "temp_c": 37.0, "resp_rate": 14, "spo2": 98,
            "map": 86, "wbc": 8, "lactate": 1.1}
    derang = {**base, "hr": 120, "resp_rate": 30, "lactate": 4.0}
    # single-hour blip
    lz = PatientBaselineLearner(); _settle(lz, base)
    lz.observe(derang)
    spike = lz.deviation
    # same derangement, but sustained for several hours -> must build HIGHER
    lz2 = PatientBaselineLearner(); _settle(lz2, base)
    for _ in range(5):
        lz2.observe(derang)
    sustained = lz2.deviation
    # a LOWER-than-baseline HR is the non-worsening direction -> no HR contribution
    z_low = lz.per_channel({**base, "hr": 50})["hr"]
    assert sustained > spike + 1.0, (spike, sustained)
    assert z_low < 0.5, z_low
    _ok(f"one blip reads {spike:.2f}; the same shift sustained 5 h builds to "
        f"{sustained:.2f} — persistence is required. A below-baseline HR "
        f"contributes {z_low:.2f} (only worsening counts)")


def test_two_sided_risk() -> None:
    print("[4] personalized_risk dampens at baseline, lifts on deviation")
    # chronically high population risk, patient at their own baseline -> dampened
    damp = personalized_risk(pop=0.80, deviation=0.0, maturity=1.0)
    # population-normal, but patient strongly deviating from their own norm -> lifted
    lift = personalized_risk(pop=0.30, deviation=2.4, maturity=1.0)
    # immature baseline -> just trust the population value
    immature = personalized_risk(pop=0.62, deviation=2.4, maturity=0.0)
    assert damp < 0.80, damp
    assert lift > 0.30, lift
    assert abs(immature - 0.62) < 1e-6, immature
    _ok(f"chronic-abnormal at baseline 0.80 -> {damp:.2f} (dampened); "
        f"pop-normal 0.30 but deviating -> {lift:.2f} (lifted); "
        f"immature -> {immature:.2f} (trusts population)")


def test_real_no_harm() -> None:
    print("[5] real CinC cohort — personalization does not hurt detection (measured)")
    from backend.aegis.cohort import CohortCluster
    cl = CohortCluster(n=12, lane_budget=160, seed=7)
    s = cl.validation(0.75)["summary"]
    pop = s["population"]
    assert s["detection_rate"] >= pop["detection_rate"] - 1e-9
    assert s["fp_rate"] <= pop["fp_rate"] + 0.21       # within one patient on this n
    _ok(f"personalized detect {s['detection_rate']*100:.0f}% / FP {s['fp_rate']*100:.0f}% "
        f"vs population {pop['detection_rate']*100:.0f}% / FP {pop['fp_rate']*100:.0f}% "
        f"(sustained-2h alarms; honest — see learner-validation)")


def test_narrative_describes_learned() -> None:
    print("[6] narrative reports what was ACTUALLY learned (no invented tendencies)")
    from backend.aegis.learner import PatientBaselineLearner
    # a patient who genuinely runs tachycardic + hypotensive, then desaturates
    lz = PatientBaselineLearner()
    base = {"hr": 104, "temp_c": 37.0, "resp_rate": 16, "spo2": 98, "map": 66,
            "wbc": 8.0, "lactate": 1.0}
    _settle(lz, base, n=20)
    nar = lz.narrative(base)
    txt = " ".join(nar["learned"]).lower()
    assert nar["status"] == "mature"
    assert "tachycardic" in txt and "hypotensive" in txt, txt
    assert "bradycardic" not in txt and "low spo₂" not in txt, "invented a tendency"
    assert not nar["drifting"], "at baseline there should be no drift"
    _ok(f"learned: {nar['summary']}")
    # now desaturate + drop MAP and confirm drift is reported in the RIGHT direction
    nar2 = lz.narrative({**base, "spo2": 90, "map": 54})
    d = " ".join(nar2["drifting"]).lower()
    assert "spo₂" in d and "below" in d and "map" in d, d
    _ok(f"drift: {nar2['drifting'][0]}")


def main() -> None:
    print("=" * 64)
    print("AegisOS — per-patient behavioural learner verification")
    print("=" * 64)
    test_personal_baseline()
    test_gate_freezes()
    test_smoothing_and_direction()
    test_two_sided_risk()
    test_real_no_harm()
    test_narrative_describes_learned()
    print("-" * 64)
    print("ALL LEARNER CHECKS PASSED")


if __name__ == "__main__":
    main()
