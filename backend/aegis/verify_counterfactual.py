"""Verify harness for the Counterfactual Replay Engine — claims asserted by measurement.

A recorded septic trajectory deteriorated to organ failure because antibiotics were
omitted at T-~1h. We replay the alternative (abx + fluids) from the earliest
actionable branch point and prove:

  1. ANCHORING   — the replay branches at the same node the incident report flags,
                   and rebuilds the vitals state recorded there.
  2. DIRECTION   — measured model-internally (alternative vs action-actually-taken,
                   same forward model): treating yields lower final risk + higher
                   final MAP than the factual baseline.
  3. MONOTONE    — replaying the action that was actually taken against itself has
                   ZERO effect (no free lunch); the delta is purely the treatment.
  4. HONESTY     — model-fidelity gap vs recorded reality is disclosed; the validity
                   caveat is present; confidence strictly decays with horizon.

    python -m backend.aegis.verify_counterfactual
"""
from __future__ import annotations

from backend.aegis.counterfactual import CounterfactualReplay
from backend.aegis.kernel.causal_graph import (
    ALERT, MODEL_PREDICTED, OBSERVED_ASSOCIATION, OMISSION, OUTCOME, VITAL,
    TemporalCausalGraph)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def _build():
    g = TemporalCausalGraph(bed="bed_3")
    v0 = g.add_event(VITAL, 0.0, {"map": 78, "lactate": 1.6}, risk=0.10)
    v1 = g.add_event(VITAL, 3600.0, {"map": 62, "lactate": 4.2}, risk=0.55)
    om = g.add_event(OMISSION, 3700.0, {"indicated": "antibiotics", "given": False}, risk=0.6)
    a = g.add_event(ALERT, 3900.0, {"severity": 2}, risk=0.62)
    v2 = g.add_event(VITAL, 5400.0, {"map": 55, "lactate": 5.8}, risk=0.8)
    out = g.add_event(OUTCOME, 7200.0, {"state": "organ_failure", "map": 50}, risk=0.9)
    g.link(v0.id, v1.id, "perfusion decline", MODEL_PREDICTED, confidence=0.7)
    g.link(v1.id, om.id, "indicated abx", OBSERVED_ASSOCIATION, confidence=0.8, support=312)
    g.link(om.id, a.id, "untreated deterioration", MODEL_PREDICTED, confidence=0.65)
    g.link(a.id, v2.id, "worsening", MODEL_PREDICTED, confidence=0.6)
    g.link(v2.id, out.id, "organ failure", OBSERVED_ASSOCIATION, confidence=0.6, support=188)
    return g, om.id, out.id


def main() -> int:
    failures = 0
    checks = []

    g, om_id, out_id = _build()
    engine = CounterfactualReplay(g, horizon=16)

    treat = {"fluid_rate": 5.0, "vasopressor_dose": 0.0, "antibiotic_code": 3}
    nothing = {"fluid_rate": 0.0, "vasopressor_dose": 0.0, "antibiotic_code": 0}

    # ---- 1. ANCHORING ---------------------------------------------------- #
    _hr("1. ANCHORING (branches at the earliest actionable node)")
    res = engine.replay_earliest(out_id, treat)
    print(f"    branch={res['branch']['kind']}@{res['branch']['t']}  "
          f"branch_vitals={res['branch_vitals']}")
    checks.append(("replay branches at the omission node", res["branch"]["id"] == om_id))
    # The vitals reconstructed are the last recorded snapshot at/before the branch
    # (v1 @ t=3600 with MAP 62), not the post-hoc crash values.
    checks.append(("branch vitals = recorded state at the branch (MAP 62)",
                   res["branch_vitals"]["map"] == 62))

    # ---- 2. DIRECTION (model-internal: cf vs factual baseline) ----------- #
    _hr("2. DIRECTION (alternative vs action-actually-taken, same model)")
    print(f"    factual baseline (omission→no action): "
          f"MAP={res['factual_baseline']['final_map']} risk={res['factual_baseline']['final_risk']}")
    print(f"    counterfactual (abx+fluids):           "
          f"MAP={res['counterfactual']['final_map']} risk={res['counterfactual']['final_risk']}")
    print(f"    delta (treatment effect): {res['delta']}")
    checks.append(("treatment lowers risk vs factual baseline",
                   res["counterfactual"]["final_risk"] < res["factual_baseline"]["final_risk"]))
    checks.append(("treatment raises MAP vs factual baseline",
                   res["counterfactual"]["final_map"] > res["factual_baseline"]["final_map"]))
    checks.append(("flagged as diverging toward recovery",
                   res["delta"]["diverges_toward_recovery"] is True))

    # ---- 3. MONOTONE (no free lunch) ------------------------------------- #
    _hr("3. MONOTONE (replaying the taken action against itself = zero effect)")
    res_none = engine.replay(om_id, nothing)   # alternative == factual (both no-action)
    print(f"    delta when alternative == action taken: {res_none['delta']}")
    checks.append(("replaying the omitted action against itself has zero risk delta",
                   abs(res_none["delta"]["final_risk"]) < 1e-9))
    checks.append(("and is NOT flagged as diverging toward recovery",
                   res_none["delta"]["diverges_toward_recovery"] is False))

    # ---- 4. HONESTY ------------------------------------------------------ #
    _hr("4. HONESTY (model-fidelity disclosure + caveat + decaying confidence)")
    print(f"    model_fidelity: {res['model_fidelity']}")
    # In this trajectory reality crashed harder than the illustrative model — that
    # gap must be DISCLOSED, so the counterfactual isn't read as "would have lived".
    checks.append(("model-fidelity gap vs recorded reality is disclosed",
                   res["model_fidelity"]["baseline_vs_actual_risk"] < 0))
    short = engine.replay(om_id, treat, horizon=4)
    long = engine.replay(om_id, treat, horizon=32)
    print(f"    confidence: horizon4={short['counterfactual']['confidence']}  "
          f"horizon32={long['counterfactual']['confidence']}")
    checks.append(("confidence strictly decays with horizon",
                   long["counterfactual"]["confidence"] < short["counterfactual"]["confidence"]))
    checks.append(("validity caveat present", "not a validated outcome forecast" in res["validity"]))
    checks.append(("graph integrity intact through replay", res["integrity_ok"] is True))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — counterfactual replay turns the recorded story "
          "into a 'what if we'd acted sooner' the audit trail can anchor."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
