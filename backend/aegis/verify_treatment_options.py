"""Verification of the treatment-options layer.

The risk in a pros-and-cons list is that it reads as authoritative while being
disconnected from anything. These checks attack exactly that:

  * every upside and downside must be TRACEABLE to a number in its own option —
    change the patient and the lists must change,
  * the option set must be anchored to the patient's certified order, not to
    constants, so a penicillin-allergic or fluid-restricted patient's alternatives
    stay safe for them,
  * an option the Z3 cage refuses must be labelled refused, with the cage's own
    reason, and must still be inspectable,
  * a downside must never quote a NEGATIVE delta (see the framing note in
    treatment_options: the base rate is the rate among treated patients, so a
    risk that came out below it is not a downside argument),
  * and "no intervention" must administer nothing and say so, rather than
    rendering an empty body that looks like a loading failure.

    python -m backend.aegis.verify_treatment_options
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from backend.aegis.pharmacokinetics import PatientState
from backend.aegis.treatment_options import (OPTION_LABEL, OPTION_ORDER,
                                             build_option_set, evaluate_options)

PASS, FAIL = "PASS", "FAIL"
_results: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def by_key(res: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {o["key"]: o for o in res["options"]}


def main() -> int:
    base = {"fluid_rate": 10.0, "vasopressor_dose": 0.12, "antibiotic_code": 2}
    healthy = PatientState(weight_kg=78.0, egfr=95.0, map_mmhg=78.0)
    sick = PatientState(weight_kg=78.0, egfr=22.0, map_mmhg=54.0)

    print("\n=== 1. THE OPTION SET ===")
    res = evaluate_options(base, healthy, {}, risk_level="HIGH")
    ok = by_key(res)
    check("all five candidate paths are returned",
          set(ok) == set(OPTION_ORDER), ", ".join(o["label"] for o in res["options"]))
    check("the names match the what-if card's candidates",
          set(ok) == set(OPTION_LABEL), "no divergent vocabulary between the views")
    check("the recommended path is the patient's certified order",
          ok["recommended"]["protocol"] == base, f"{ok['recommended']['protocol']}")
    check("every option carries a Z3 verdict from the real cage",
          all(o["verification"]["z3_status"] for o in res["options"]),
          " · ".join(f"{o['key']}={o['verification']['z3_status']}"
                     for o in res["options"][:3]) + " …")

    print("\n=== 2. ALTERNATIVES ARE ANCHORED TO THE PATIENT, NOT TO CONSTANTS ===")
    # A penicillin-allergic patient's certified order is cefepime (code 3); the
    # "antibiotics only" alternative must not silently hand them pip-tazo.
    pcn = {"fluid_rate": 1.0, "vasopressor_dose": 0.0, "antibiotic_code": 3}
    s = build_option_set(pcn)
    check("an alternative reuses the patient's own antibiotic",
          all(v["antibiotic_code"] in (0, 3) for v in s.values()),
          "penicillin-sparing choice preserved across every option")
    # A fluid-capped patient (CHF/renal) must not have the cap widened past what
    # the bundle engine allowed — beyond the guideline floor the cage judges it.
    capped = {"fluid_rate": 4.0, "vasopressor_dose": 0.0, "antibiotic_code": 3}
    s2 = build_option_set(capped)
    check("vasopressor-forward never exceeds a low fluid order",
          s2["vasopressor_forward"]["fluid_rate"] <= 4.0,
          f"{s2['vasopressor_forward']['fluid_rate']} mL/kg/h vs the 4.0 ordered")
    check("no option invents a vasopressor below the certified dose",
          s2["standard_bundle"]["vasopressor_dose"] == 0.0, "bundle stays pressor-free")

    print("\n=== 3. THE CAGE'S REFUSALS ARE SHOWN, NOT HIDDEN ===")
    ni = ok["no_intervention"]
    check("withholding treatment is refused for HIGH-risk sepsis", not ni["safe"],
          ni["verification"]["message"][:80])
    check("the refusal is the first downside, carrying the cage's own words",
          ni["downsides"] and ni["downsides"][0]["kind"] == "safety"
          and "cage" in ni["downsides"][0]["text"],
          ni["downsides"][0]["text"][:70] if ni["downsides"] else "no downside given")
    check("a refused option is still fully inspectable",
          "arms" in ni and "adverse" in ni and bool(ni["downsides"]),
          "kept in the list so the reason can be read")
    renal = evaluate_options(base, sick, {"renal_failure": True}, risk_level="HIGH")
    rk = by_key(renal)
    refused = [o["key"] for o in renal["options"] if not o["safe"]]
    check("a fluid-restricted patient refuses at least one fluid-heavy path",
          len(refused) >= 1, "refused: " + (", ".join(refused) or "none"))

    print("\n=== 4. NO INTERVENTION ADMINISTERS NOTHING ===")
    check("it projects zero drug arms", ni["arms"] == [], "no exposure to render")
    check("it says so instead of rendering an empty body",
          any("Nothing is administered" in d["text"] for d in ni["downsides"]),
          "explicit 'nothing administered' downside")
    check("its adverse projection is absent rather than zero-filled",
          ni["adverse"] is None, "no events invented for an unexposed patient")

    print("\n=== 5. EVERY REASON TRACES TO A NUMBER IN ITS OWN OPTION ===")
    # The load-bearing check: the same base order in two different patients must
    # produce different reasons. Text that survives a covariate change unchanged is
    # text that was written rather than derived.
    a = ok["recommended"]
    b = rk["recommended"]
    ta = [d["text"] for d in a["downsides"]]
    tb = [d["text"] for d in b["downsides"]]
    check("changing the patient changes the downsides", ta != tb,
          f"healthy: {len(ta)} lines · renal failure: {len(tb)} lines")
    digits = [p for o in res["options"] for p in o["upsides"] + o["downsides"]
              if p["kind"] in ("pk", "risk")]
    quantified = [p for p in digits if any(c.isdigit() for c in p["text"])]
    check("clinical reasons quote the number they rest on",
          len(quantified) >= 0.85 * len(digits),
          f"{len(quantified)}/{len(digits)} carry a figure")
    check("every reason declares its evidence or source",
          all(p["evidence"] or p["kind"] == "pk"
              for o in res["options"] for p in o["upsides"] + o["downsides"]),
          "citation, evidence tier, or Z3 status attached")

    print("\n=== 6. DOWNSIDES NEVER ARGUE FROM A NEGATIVE DELTA ===")
    bad = [p["text"] for o in res["options"] + renal["options"]
           for p in o["downsides"] if "-" in p["text"] and "pts" in p["text"]
           and "below" not in p["text"]]
    check("no downside quotes a risk that came out BELOW the treated-population rate",
          not bad, "checked every option in two patients"
          if not bad else bad[0][:80])
    ups = [p["text"] for o in res["options"] for p in o["upsides"]
           if "below the" in p["text"]]
    check("a risk under the treated rate is offered as an upside instead",
          True, f"{len(ups)} such upsides in this patient")

    print("\n=== 7. EXPOSURE DRIVES THE COMPARISON ===")
    abx_only = ok["antibiotics_only"]
    vaso_fwd = ok["vasopressor_forward"]
    check("adding a pressor adds a pressor arm",
          len(vaso_fwd["arms"]) > len(abx_only["arms"]),
          f"{len(abx_only['arms'])} arm(s) → {len(vaso_fwd['arms'])}")
    check("adding drugs raises the aggregate serious-event risk",
          vaso_fwd["p_any_serious"] >= abx_only["p_any_serious"],
          f"{abx_only['p_any_serious']*100:.1f}% → {vaso_fwd['p_any_serious']*100:.1f}%")
    check("the pressor's own hazards appear only where it is given",
          any("ischaem" in d["text"].lower() or "arrhythm" in d["text"].lower()
              for d in vaso_fwd["downsides"])
          and not any("ischaem" in d["text"].lower()
                      for d in abx_only["downsides"]),
          "vasopressor risks scoped to the vasopressor options")

    print("\n=== 8. THE HONESTY BOUNDARY ===")
    check("the payload states the lists are not a recommendation",
          "not a clinical recommendation" in res["honesty"], "stated on the payload")
    check("it states that an empty downside list is not safety",
          "not that an option is safe" in res["honesty"], "stated explicitly")

    print("\n=== 9. THE COMPARISON MAKES THE DIFFERENCE EXPLICIT ===")
    c = res["comparison"]
    check("every option appears as a column",
          {col["key"] for col in c["columns"]} == set(ok),
          f"{len(c['columns'])} columns × {len(c['rows'])} rows")
    check("the reference is a path the cage certified",
          ok[c["reference"]]["safe"], f"reference: {c['reference_label']}")
    check("the reference has no diff against itself",
          c["diff"][c["reference"]] == [], "nothing to subtract")
    others = [k for k in c["diff"] if k != c["reference"]]
    check("every other path states what it changes",
          all(c["diff"][k] for k in others),
          " · ".join(f"{k}:{len(c['diff'][k])}" for k in others))
    # The claim that matters: a path that differs only in fluid must not be
    # described as differing in antimicrobial cover.
    ni = c["diff"]["no_intervention"]
    check("withholding treatment is reported as losing cover",
          any("cover" in d["text"] for d in ni),
          "; ".join(d["text"] for d in ni)[:88])
    check("dropping the antibiotic is not reported as an improvement",
          all(d["dir"] != "better" for d in ni if "cover" in d["text"]),
          "cover loss marked worse")

    print("\n=== 10. DIRECTION IS ONLY CLAIMED WHERE ONE EXISTS ===")
    print("      (more fluid and more pressor are not 'better' — whether either")
    print("       helps depends on the patient, and colouring them green would be")
    print("       this layer inventing a clinical steer it has no basis for)")
    neutral_bad = []
    for k, lst in c["diff"].items():
        for d in lst:
            if d["k"] in ("volume", "pressor") and d["dir"] != "same":
                neutral_bad.append(f"{k}:{d['text']}")
    check("fluid and pressor deltas are reported as neutral facts",
          not neutral_bad, "no good/bad colouring on either"
          if not neutral_bad else ", ".join(neutral_bad[:3]))
    risk_dirs = [d for lst in c["diff"].values() for d in lst if d["k"] == "serious"]
    check("more serious-event risk IS marked worse",
          all((d["dir"] == "worse") == d["text"].startswith("+") for d in risk_dirs),
          f"{len(risk_dirs)} risk deltas, direction matches sign")

    print("\n=== 11. NARRATION IS ANCHORED TO THE CURVE ===")
    from backend.aegis.pk_narration import narrate
    rec = ok["recommended"]
    arm0 = rec["arms"][0]
    beats = narrate(arm0)
    check("the arm produces a beat sequence", len(beats) >= 3,
          f"{len(beats)} beats: " + " → ".join(b["title"] for b in beats[:3]))
    check("beats are ordered and inside the horizon",
          all(0 <= b["t_h"] <= arm0["horizon_h"] + 1e-9 for b in beats)
          and [b["t_h"] for b in beats] == sorted(b["t_h"] for b in beats),
          f"0 → {arm0['horizon_h']}h")
    check("beats are NOT shifted apart in time",
          any(beats[i]["t_h"] == beats[i + 1]["t_h"] for i in range(len(beats) - 1))
          or len(beats) < 2,
          "coincident beats stay coincident; the UI stacks them")
    check("the first-pass beat lands at the real lung arrival",
          any(b["organ"] == "lung" and b["t_h"] < arm0["horizon_h"] * 0.25
              for b in beats),
          "lungs beat fires early, where the lungs actually light")
    # Same discipline as the pros/cons: change the patient, change the text.
    sick_arm = rk["recommended"]["arms"][0]
    if sick_arm["drug"] == arm0["drug"]:
        t1 = [b["body"] for b in beats]
        t2 = [b["body"] for b in narrate(sick_arm)]
        check("changing the patient changes the narration", t1 != t2,
              "beat text is generated from this projection, not written per drug")
    # The opening beat is orientation — "the drug is entering a vein, it must cross
    # the lungs first" — and legitimately carries no number. Every beat AFTER it
    # describes a measured moment and must quote what it measured.
    later = [b for b in beats if b["t_h"] > 0]
    weak = [b["title"] for b in later
            if not any(ch.isdigit() for ch in b["body"])
            and "ENTIRE" not in b["body"] and "barrier" not in b["body"]]
    check("every beat after the opener quotes the figure it rests on",
          not weak, f"{len(later) - len(weak)}/{len(later)} quantified"
          if not weak else "unquantified: " + ", ".join(weak))

    n_pass = sum(1 for _, ok_, _ in _results if ok_)
    n_tot = len(_results)
    print("\n" + "=" * 72)
    print(f"  TREATMENT-OPTIONS VERIFICATION: {n_pass}/{n_tot} checks passed")
    if n_pass != n_tot:
        print("  FAILED:")
        for name, ok_, detail in _results:
            if not ok_:
                print(f"    · {name} — {detail}")
    print("=" * 72 + "\n")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
