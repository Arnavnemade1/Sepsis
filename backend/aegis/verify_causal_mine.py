"""Verify harness for epidemiological pattern mining — claims asserted by measurement.

We plant a deterministic 60-patient cohort with a KNOWN ground truth and prove the
miner recovers it without being fooled by base rates, small N, or confounding:

  Group A (20) untreated deranged (MAP<65 ∧ lactate>4, no antibiotics) → 80% fail
  Group B (20) treated deranged   (same derangement, abx within 2h)    → 20% fail
  Group C (20) healthy                                                  →  5% fail
  A benign marker (SpO2<92) is sprinkled on even-indexed patients in ALL groups,
  uncorrelated with the outcome — a planted NULL pattern.

Asserted:
  1. RECOVERY   — the untreated-perfusion-failure rule is found with confidence 0.80,
                  support n=20, and the cohort baseline ≈ 0.35.
  2. CONFOUNDING— the untreated rule has strictly higher confidence/lift than the
                  same derangement without the treatment filter (treated cases dilute).
  3. NULL       — the uncorrelated marker has lift ≈ 1 and is filtered out by min_lift.
  4. RIGOR      — every rule carries a Wilson CI bracketing its confidence, respects
                  min_support, is stamped observed_association, and the result
                  discloses the candidate count (multiple-comparisons honesty).

    python -m backend.aegis.verify_causal_mine
"""
from __future__ import annotations

from types import SimpleNamespace as A

from backend.aegis.causal_ingest import CausalGraphIngestor
from backend.aegis.causal_mine import (
    Antecedent, BASE_CONDITIONS, PatternMiner)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def _assess(band, mp, proto=None, ver=None):
    return A(risk_level=band, ml_probability=mp, detection={}, protocol=proto,
             verification=ver, signal_quality=1.0, triggered_rules=[])


def _patient(bed, deranged, treated, marker, fails):
    ing = CausalGraphIngestor(bed=bed, vital_every_s=60, omission_after_s=1e9)
    onset_t = 300.0
    for j in range(40):
        t = j * 60.0
        if deranged and j >= 5:
            vit = {"map": 58, "lactate": 5.0, "heart_rate": 118, "respiratory_rate": 26,
                   "spo2": 90 if marker else 97}
            band, mp = "HIGH", 0.8
        else:
            vit = {"map": 82, "lactate": 1.5, "heart_rate": 78, "respiratory_rate": 16,
                   "spo2": 90 if marker else 98}
            band, mp = "LOW", 0.08
        proto = ver = None
        if deranged and treated and j == 6:        # abx at t=360s, within 2h of onset
            proto, ver = {"antibiotic_code": 3}, {"safe": True, "z3_status": "unsat"}
        ing.ingest(_assess(band, mp, proto, ver), vit, t)
    if fails:                                       # outcome after last tick, within 4h of onset
        ing.mark_outcome("organ_failure", 2400.0, {"map": 50})   # onset 300 → 35min later
    return ing.graph


def build_cohort():
    graphs = []
    for i in range(60):
        marker = (i % 2 == 0)
        if i < 20:        # Group A: untreated deranged, 80% fail (i<16)
            graphs.append(_patient(f"A{i}", True, False, marker, i < 16))
        elif i < 40:      # Group B: treated deranged, 20% fail (i in 20..23)
            graphs.append(_patient(f"B{i}", True, True, marker, i < 24))
        else:             # Group C: healthy, 5% fail (i == 40)
            graphs.append(_patient(f"C{i}", False, False, marker, i == 40))
    return graphs


def main() -> int:
    failures = 0
    checks = []
    miner = PatternMiner(build_cohort())
    res = miner.mine(min_support=5, min_lift=1.2)

    _hr("MINED RULES")
    print(f"  cohort={res['cohort_size']} outcome={res['outcome']} "
          f"tested={res['candidates_tested']}")
    for r in res["rules"]:
        print(f"  {r['antecedent']:34} → {int(r['confidence']*100)}% "
              f"(n={r['n_antecedent']}, CI {int(r['ci_low']*100)}–{int(r['ci_high']*100)}%, "
              f"base {int(r['baseline']*100)}%, lift {r['lift']}×)")

    by_ante = {r["antecedent"]: r for r in res["rules"]}
    untreated = by_ante.get("MAP<65 ∧ lactate>4 ∧ no-abx<2h")
    plain = by_ante.get("MAP<65 ∧ lactate>4")

    # ---- 1. RECOVERY ----------------------------------------------------- #
    _hr("1. RECOVERY (planted untreated rule)")
    checks.append(("untreated-perfusion-failure rule was mined", untreated is not None))
    if untreated:
        checks.append(("confidence ≈ 0.80", abs(untreated["confidence"] - 0.80) < 0.01))
        checks.append(("support n == 20 (Group A only)", untreated["n_antecedent"] == 20))
        checks.append(("cohort baseline ≈ 0.35", abs(untreated["baseline"] - 0.35) < 0.02))
        checks.append(("lift ≈ 2.3× (0.80 / 0.35)", untreated["lift"] > 2.0))

    # ---- 2. CONFOUNDING (treatment filter matters) ----------------------- #
    _hr("2. CONFOUNDING (untreated >> same derangement diluted by treated)")
    checks.append(("plain derangement rule was mined", plain is not None))
    if untreated and plain:
        print(f"    untreated conf={untreated['confidence']} lift={untreated['lift']}  |  "
              f"plain conf={plain['confidence']} lift={plain['lift']}")
        checks.append(("untreated confidence > plain (treated cases dilute the rate)",
                       untreated["confidence"] > plain["confidence"] + 0.1))
        checks.append(("untreated lift > plain lift", untreated["lift"] > plain["lift"]))
        checks.append(("plain rule spans A+B (n == 40)", plain["n_antecedent"] == 40))

    # ---- 3. NULL (uncorrelated marker filtered) -------------------------- #
    _hr("3. NULL (uncorrelated marker has lift≈1, filtered out)")
    spo2 = next(c for c in BASE_CONDITIONS if c.label == "SpO2<92")
    null_rule = miner.evaluate(Antecedent([spo2]))
    print(f"    SpO2<92: n={null_rule.n_antecedent} conf={null_rule.confidence:.3f} "
          f"lift={null_rule.lift:.3f}")
    checks.append(("uncorrelated marker has lift ≈ 1", abs(null_rule.lift - 1.0) < 0.2))
    checks.append(("uncorrelated marker is filtered from results (lift < min_lift)",
                   "SpO2<92" not in by_ante))

    # ---- 4. RIGOR -------------------------------------------------------- #
    _hr("4. RIGOR (CI, support gate, honesty stamps)")
    checks.append(("every rule's Wilson CI brackets its confidence",
                   all(r["ci_low"] <= r["confidence"] <= r["ci_high"] for r in res["rules"])))
    checks.append(("every rule respects min_support",
                   all(r["n_antecedent"] >= 5 for r in res["rules"])))
    checks.append(("every rule stamped observed_association",
                   all(r["evidence_type"] == "observed_association" for r in res["rules"])))
    checks.append(("result discloses candidates_tested + disclaimer",
                   res["candidates_tested"] >= 10 and "NOT causal" in res["disclaimer"]))
    # support gate actually excludes thin rules
    strict = miner.mine(min_support=25, min_lift=1.2)
    checks.append(("raising min_support to 25 drops the n=20 untreated rule",
                   "MAP<65 ∧ lactate>4 ∧ no-abx<2h" not in {r["antecedent"] for r in strict["rules"]}))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — the cohort yields honest, baseline-anchored "
          "associations, not causal guidelines."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
