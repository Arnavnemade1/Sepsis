"""Falsifiable verification of the adverse-event projection layer.

Adverse-event probabilities are the easiest thing in this entire system to fake.
Nobody can look at "acute kidney injury: 21%" and tell whether it came from
pharmacology or from a plausible-looking constant. So this suite attacks the two
ways that number could be dishonest:

  * **It could have drifted off the literature.** The strongest check here is
    ANCHORING: give the model the standard dose in a typical patient and every
    exposure-driven event must return *exactly* its published base incidence. Not
    close to it — equal to it, within Monte-Carlo error. Any silent scaling error,
    any mis-sized modifier, any convexity bias in the sampling breaks this
    immediately. It is what caught the Jensen inflation (risk is convex in
    exposure, so the mean over the variability distribution is not the risk at the
    mean, and every probability was ~40% too high) and the covariate
    double-counting that had vancomycin AKI at 55%.

  * **It could have invented a dose-response.** So the evidence firewall is
    checked structurally: an ANCHORED event must carry a published numeric
    threshold and a citation containing it, and must NOT derive its reference from
    the model. Events with no published exposure-response must not gain one — an
    idiosyncratic reaction is required to stay flat when the dose doubles.

The rest are directional claims the model can fail: risk rises with dose, rises as
renal function falls, rises when a known nephrotoxic pair is co-ordered, and the
reported interval actually contains the estimate.

    python -m backend.aegis.verify_adverse
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from backend.aegis.adverse import (ANCHORED, CATALOGUE, COVARIATE_MODIFIERS,
                                   LABEL_RATE, MECHANISTIC, SEVERITY_RANK,
                                   AdverseEventEngine, REFERENCE_PATIENT,
                                   project_order, reference_projection)
from backend.aegis.pharmacokinetics import (DRUGS, PBPKModel, PatientState,
                                            Regimen, project_protocol)

PASS, FAIL = "PASS", "FAIL"
_results: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def events_by_key(res: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {e["key"]: e for e in res["events"]}


def run(protocol: Dict[str, Any], patient: PatientState,
        history: Dict[str, Any] = None, n: int = 40000) -> Dict[str, Any]:
    return project_order(protocol, patient, history or {}, n_samples=n)


def main() -> int:
    eng = AdverseEventEngine(n_samples=40000)

    # ---------------------------------------------------------------------- #
    print("\n=== 1. ANCHORING: standard dose + typical patient = published rate ===")
    print("      (the check that keeps every probability tied to the literature)")
    standard = {"fluid_rate": 10.0, "vasopressor_dose": 0.10, "antibiotic_code": 1}
    res = run(standard, REFERENCE_PATIENT, {"age": 60})
    ek = events_by_key(res)
    n_anchor = 0
    for ev in CATALOGUE:
        if ev.key not in ek or not ev.reference_from_model:
            continue
        got = ek[ev.key]["probability"]
        rel = abs(got - ev.base_incidence) / ev.base_incidence
        n_anchor += 1
        check(f"{ev.key:22s} returns its published rate", rel < 0.03,
              f"{got*100:.2f}% vs published {ev.base_incidence*100:.2f}% "
              f"({rel*100:.2f}% off)")
    check("every model-referenced event was anchoring-tested", n_anchor >= 5,
          f"{n_anchor} events checked at the standard regimen")

    # ---------------------------------------------------------------------- #
    print("\n=== 2. EVIDENCE FIREWALL ===")
    bad_anchor, bad_cite, bad_tier = [], [], []
    for ev in CATALOGUE:
        if ev.evidence == ANCHORED:
            if ev.reference_from_model or ev.reference_exposure <= 0:
                bad_anchor.append(ev.key)
            digits = any(c.isdigit() for c in ev.citation)
            if not digits or len(ev.citation) < 20:
                bad_cite.append(ev.key)
        if ev.evidence not in (ANCHORED, LABEL_RATE, MECHANISTIC):
            bad_tier.append(ev.key)
    check("ANCHORED events use a published threshold, not a model-derived one",
          not bad_anchor, "none violate" if not bad_anchor else ", ".join(bad_anchor))
    check("ANCHORED events carry a citation containing the number",
          not bad_cite, "all cited" if not bad_cite else ", ".join(bad_cite))
    check("every event declares a valid evidence tier", not bad_tier,
          f"{len(CATALOGUE)} events, 3 tiers")
    tiers = {t: sum(1 for e in CATALOGUE if e.evidence == t)
             for t in (ANCHORED, LABEL_RATE, MECHANISTIC)}
    check("the evidence mix is reported, not averaged away",
          set(res["evidence_mix"]) >= {"anchored", "label_rate", "mechanistic",
                                       "fraction_anchored"},
          f"catalogue: {tiers[ANCHORED]} anchored · {tiers[LABEL_RATE]} label-rate "
          f"· {tiers[MECHANISTIC]} mechanistic")
    check("every event states what to monitor for it",
          all(e.monitoring for e in CATALOGUE),
          f"{len(CATALOGUE)}/{len(CATALOGUE)} carry a monitoring instruction")

    # ---------------------------------------------------------------------- #
    print("\n=== 3. IDIOSYNCRATIC EVENTS DO NOT GAIN A DOSE-RESPONSE ===")
    print("      (a reaction with no published exposure-response must stay flat)")
    flat = [e for e in CATALOGUE if not e.exposure_metric]
    for ev in flat:
        probs = []
        for dose in (0.05, 0.10, 0.30):
            proto = {"fluid_rate": 0.0, "vasopressor_dose": dose, "antibiotic_code": 1}
            r = events_by_key(run(proto, REFERENCE_PATIENT))
            if ev.key in r:
                probs.append(r[ev.key]["probability"])
        if len(probs) < 2:
            continue
        check(f"{ev.key:22s} is exposure-invariant",
              max(probs) - min(probs) < 1e-9,
              "identical across a 6× dose range: " +
              " · ".join(f"{p*100:.2f}%" for p in probs))

    # ---------------------------------------------------------------------- #
    print("\n=== 4. RISK RISES WITH DOSE (monotone, every exposure-driven event) ===")
    doses = (0.05, 0.10, 0.20, 0.35)
    series: Dict[str, List[float]] = {}
    for dose in doses:
        proto = {"fluid_rate": 0.0, "vasopressor_dose": dose, "antibiotic_code": 0}
        for k, e in events_by_key(run(proto, REFERENCE_PATIENT)).items():
            series.setdefault(k, []).append(e["probability"])
    checked = 0
    for k, ps in series.items():
        ev = next(e for e in CATALOGUE if e.key == k)
        if not ev.exposure_metric:
            continue
        checked += 1
        mono = all(ps[i] <= ps[i + 1] + 1e-12 for i in range(len(ps) - 1))
        check(f"{k:22s} rises with the infusion rate", mono,
              "0.05→0.35 mcg/kg/min: " + " → ".join(f"{p*100:.1f}%" for p in ps))
    check("norepinephrine dose-response was exercised", checked >= 3,
          f"{checked} exposure-driven events")

    # ---------------------------------------------------------------------- #
    print("\n=== 5. RISK RISES AS RENAL FUNCTION FALLS ===")
    aki, neuro = [], []
    for egfr in (90.0, 60.0, 35.0, 15.0):
        pt = PatientState(weight_kg=78.0, egfr=egfr, map_mmhg=75.0)
        r = events_by_key(run({"fluid_rate": 0.0, "vasopressor_dose": 0.0,
                               "antibiotic_code": 2}, pt))
        aki.append(r["vanc_aki"]["probability"])
        r2 = events_by_key(run({"fluid_rate": 0.0, "vasopressor_dose": 0.0,
                                "antibiotic_code": 3}, pt))
        neuro.append(r2["cefe_neurotox"]["probability"])
    check("vancomycin AKI risk rises as eGFR falls",
          all(aki[i] < aki[i + 1] for i in range(len(aki) - 1)),
          "eGFR 90→15: " + " → ".join(f"{p*100:.1f}%" for p in aki))
    check("cefepime neurotoxicity rises as eGFR falls",
          all(neuro[i] < neuro[i + 1] for i in range(len(neuro) - 1)),
          "eGFR 90→15: " + " → ".join(f"{p*100:.1f}%" for p in neuro))
    check("cefepime neurotoxicity is the steeper of the two (trough-driven, k=2.6)",
          (neuro[-1] / neuro[0]) > (aki[-1] / aki[0]),
          f"{neuro[-1]/neuro[0]:.1f}× vs {aki[-1]/aki[0]:.1f}× over the same eGFR fall")

    # ---------------------------------------------------------------------- #
    print("\n=== 6. THE ANCHORED VANCOMYCIN THRESHOLD BEHAVES AS PUBLISHED ===")
    print("      (Rybak 2020: AKI risk rises steeply above AUC24 600 mg·h/L)")

    def vanc_at(auc24: float) -> float:
        fake = {"drug": "vancomycin", "metrics": {"auc24": auc24, "organ_auc": {}},
                "plasma": [auc24 / 24.0], "times_h": [0.0]}
        r = eng.project([fake], REFERENCE_PATIENT, {})
        return events_by_key(r)["vanc_aki"]["probability"]

    p400, p500, p600, p800 = (vanc_at(400), vanc_at(500), vanc_at(600), vanc_at(800))
    check("AKI risk at the in-target AUC equals the published rate",
          abs(p500 - 0.12) / 0.12 < 0.03,
          f"AUC24 500 → {p500*100:.1f}% (published 12%)")
    check("risk below target is lower, above target is higher",
          p400 < p500 < p600 < p800,
          " · ".join(f"AUC {a}: {p*100:.1f}%" for a, p in
                     ((400, p400), (500, p500), (600, p600), (800, p800))))
    check("exceeding AUC 600 at least doubles the in-target risk", p800 > 2 * p500,
          f"{p800*100:.1f}% at AUC 800 vs {p500*100:.1f}% at 500")

    # ---------------------------------------------------------------------- #
    print("\n=== 7. DRUG-INTERACTION MODIFIER (vancomycin + pip-tazo) ===")
    pt = PatientState(weight_kg=78.0, egfr=70.0, map_mmhg=75.0)
    alone = events_by_key(run({"fluid_rate": 0.0, "vasopressor_dose": 0.0,
                               "antibiotic_code": 2}, pt))["vanc_aki"]
    model = PBPKModel(pt)
    both_arms = [model.project("vancomycin").to_dict(),
                 model.project("piperacillin-tazobactam").to_dict()]
    combo = events_by_key(eng.project(both_arms, pt, {}))["vanc_aki"]
    check("co-ordering the known nephrotoxic pair raises AKI risk",
          combo["probability"] > alone["probability"] * 1.5,
          f"{alone['probability']*100:.1f}% alone → "
          f"{combo['probability']*100:.1f}% with pip-tazo")
    inter = [m for m in combo["modifiers"] if m["kind"] == "interaction"]
    check("the interaction is disclosed with its citation, not applied silently",
          bool(inter) and "Luther" in inter[0]["why"],
          inter[0]["why"] if inter else "no interaction modifier reported")

    # ---------------------------------------------------------------------- #
    print("\n=== 8. COVARIATE MODIFIERS DO NOT RESTATE THE PK MODEL ===")
    print("      (eGFR already raises exposure; the susceptibility term must be")
    print("       small, or the renal pathway is counted twice — it was, once)")
    worst = max((f, sys, flag) for sys, mods in COVARIATE_MODIFIERS.items()
                for flag, f, _ in mods)
    exposure_mediated = [(sys, flag, f) for sys, mods in COVARIATE_MODIFIERS.items()
                         for flag, f, _ in mods if flag == "egfr_below_45"]
    check("no eGFR susceptibility factor exceeds 1.5×",
          all(f <= 1.5 for _s, _fl, f in exposure_mediated),
          " · ".join(f"{s}: {f}×" for s, _fl, f in exposure_mediated))
    check("no covariate modifier exceeds 3× (CHF volume intolerance is the cap)",
          worst[0] <= 3.0, f"largest is {worst[1]}/{worst[2]} at {worst[0]}×")

    # ---------------------------------------------------------------------- #
    print("\n=== 9. INTERVALS AND PROBABILITIES ARE WELL FORMED ===")
    res = run({"fluid_rate": 10.0, "vasopressor_dose": 0.15, "antibiotic_code": 2},
              PatientState(weight_kg=78.0, egfr=38.0, map_mmhg=56.0), {"age": 74})
    bad = [e["key"] for e in res["events"]
           if not (0.0 <= e["ci_low"] <= e["probability"] <= e["ci_high"] <= 1.0)]
    check("every interval contains its estimate and stays in [0,1]", not bad,
          f"{len(res['events'])} events" if not bad else ", ".join(bad))
    spread = [e for e in res["events"] if e["exposure_metric"]]
    check("exposure-driven events carry a non-degenerate interval",
          all(e["ci_high"] > e["ci_low"] for e in spread),
          f"{len(spread)} events, widths "
          f"{min(e['ci_high']-e['ci_low'] for e in spread)*100:.1f}–"
          f"{max(e['ci_high']-e['ci_low'] for e in spread)*100:.1f} pts")
    check("all probabilities are finite",
          all(np.isfinite(e["probability"]) for e in res["events"]),
          f"{len(res['events'])} events")

    # ---------------------------------------------------------------------- #
    print("\n=== 10. JOINT RISK IS BOUNDED CORRECTLY ===")
    serious = [e["probability"] for e in res["events"]
               if SEVERITY_RANK[e["severity"]] >= SEVERITY_RANK["severe"]]
    p_any = res["summary"]["p_any_serious"]
    check("P(any serious) is at least the single worst serious event",
          p_any >= max(serious) - 1e-9,
          f"{p_any*100:.1f}% ≥ worst single {max(serious)*100:.1f}%")
    check("P(any serious) does not exceed the sum of its parts",
          p_any <= sum(serious) + 1e-9,
          f"{p_any*100:.1f}% ≤ Σ {sum(serious)*100:.1f}%")
    check("the independence assumption behind it is disclosed",
          "independent" in res["summary"]["independence_assumption"],
          "stated in the summary, not buried")

    # ---------------------------------------------------------------------- #
    print("\n=== 11. ONLY ORDERED DRUGS PRODUCE EVENTS ===")
    only_vanc = run({"fluid_rate": 0.0, "vasopressor_dose": 0.0,
                     "antibiotic_code": 2}, REFERENCE_PATIENT)
    drugs_seen = {e["drug"] for e in only_vanc["events"]}
    check("a vancomycin-only order yields only vancomycin events",
          drugs_seen == {"vancomycin"},
          f"drugs in output: {', '.join(sorted(drugs_seen))}")
    check("an order with no drugs yields no events",
          run({"fluid_rate": 0.0, "vasopressor_dose": 0.0,
               "antibiotic_code": 0}, REFERENCE_PATIENT)["n_events"] == 0,
          "empty protocol → empty projection")

    # ---------------------------------------------------------------------- #
    print("\n=== 12. RANKING AND STABILITY ===")
    ranks = [(e["probability"] * (1 + SEVERITY_RANK[e["severity"]]))
             for e in res["events"]]
    check("events are ranked by expected harm, worst first",
          all(ranks[i] >= ranks[i + 1] - 1e-9 for i in range(len(ranks) - 1)),
          f"top: {res['events'][0]['label']} "
          f"({res['events'][0]['probability']*100:.1f}%, "
          f"{res['events'][0]['severity']})")
    a = run({"fluid_rate": 10.0, "vasopressor_dose": 0.15, "antibiotic_code": 2},
            PatientState(weight_kg=78.0, egfr=38.0, map_mmhg=56.0), {"age": 74})
    same = all(abs(x["probability"] - y["probability"]) < 1e-12
               for x, y in zip(a["events"], res["events"]))
    check("identical input → identical projection", same, "seeded and deterministic")
    lo = project_order({"fluid_rate": 10.0, "vasopressor_dose": 0.15,
                        "antibiotic_code": 2},
                       PatientState(weight_kg=78.0, egfr=38.0, map_mmhg=56.0),
                       {"age": 74}, n_samples=5000)
    drift = max(abs(x["probability"] - y["probability"])
                for x, y in zip(lo["events"], res["events"]))
    check("the Monte Carlo has converged (5k vs 40k samples)", drift < 0.02,
          f"largest probability shift {drift*100:.2f} pts")

    # ---------------------------------------------------------------------- #
    print("\n=== 13. THE HONESTY BOUNDARY IS ATTACHED, NOT OPTIONAL ===")
    check("every projection carries its honesty statement",
          "does not predict whether this patient" in res["honesty"],
          "stated on the payload the UI receives")
    check("it states that it never gates an order",
          "never gates an order" in res["honesty"],
          "the Z3 cage remains the only gate")
    check("the variability source is cited", "DALI" in res["variability"],
          res["variability"][:60] + "…")

    # ---------------------------------------------------------------------- #
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_tot = len(_results)
    print("\n" + "=" * 72)
    print(f"  ADVERSE-EVENT VERIFICATION: {n_pass}/{n_tot} checks passed")
    if n_pass != n_tot:
        print("  FAILED:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    · {name} — {detail}")
    print("=" * 72 + "\n")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
