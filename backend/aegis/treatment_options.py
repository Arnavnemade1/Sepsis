"""Treatment options — several candidate orders, each projected through the body.

The pharmacokinetic twin could only ever answer one question: "where does THIS
order go". But the decision a clinician is actually making is a comparison — this
bundle or that one, antibiotics alone or antibiotics with pressors, and what each
choice costs. The what-if scheduler already enumerates those candidate paths and
scores their projected risk; this layer takes the same candidate set and pushes
each one through the PBPK model and the adverse-event engine, so every option can
be inspected as a body, a set of exposures, and a list of consequences.

WHAT AN OPTION CARRIES
----------------------
  * the protocol itself (fluid / vasopressor / antibiotic),
  * a Z3 verdict from the SAME safety cage the write path uses, so an option this
    system would refuse to dispatch is labelled refused here too,
  * a full PK projection per drug arm — the 3-D progression,
  * the adverse-event projection for that exposure,
  * and upsides / downsides.

WHY THE UPSIDES AND DOWNSIDES ARE DERIVED, NOT WRITTEN
------------------------------------------------------
The tempting way to build a pros-and-cons list is to write the sentences by hand
per option. That produces text that cannot be wrong, because it is not connected
to anything — it says the same thing whether the patient has normal kidneys or is
anuric. Every line here is generated from a number the model just computed, and
carries the number with it: "%fT>MIC 87% (target ≥60%)" is a claim about this
projection that changes when the projection changes, and "no antimicrobial cover"
only appears when the antibiotic code really is zero.

Consequences of that choice worth stating plainly: the lists are as good as the
thresholds behind them, those thresholds are the published targets already cited
in the drug library and the adverse-event catalogue, and an option with an empty
downside list means nothing crossed a threshold — not that it is risk-free.

    python -m backend.aegis.treatment_options          # demo
    python -m backend.aegis.verify_treatment_options   # the checks
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.aegis.adverse import AdverseEventEngine, SEVERITY_RANK
from backend.aegis.kernel.bundle import ANTIBIOTICS
from backend.aegis.kernel.neurosym import PROTOCOL, NeuroSymbolicRuntime
from backend.aegis.pharmacokinetics import (ANTIBIOTIC_CODE_TO_DRUG, DRUGS,
                                            PatientState, project_protocol)

# The option set. Names match the what-if scheduler's candidates so the two views
# of the same decision cannot disagree about what the choices are called.
OPTION_LABEL = {
    "no_intervention": "No intervention",
    "antibiotics_only": "Antibiotics only",
    "standard_bundle": "Standard bundle",
    "vasopressor_forward": "Vasopressor forward",
    "recommended": "Recommended",
}
OPTION_NOTE = {
    "no_intervention": "Withhold treatment — the comparison baseline, not a plan.",
    "antibiotics_only": "Source control without haemodynamic support.",
    "standard_bundle": "Guideline crystalloid + broad-spectrum cover.",
    "vasopressor_forward": "Earlier pressor support, less volume.",
    "recommended": "What this patient's certified protocol actually orders.",
}
OPTION_ORDER = ["recommended", "standard_bundle", "antibiotics_only",
                "vasopressor_forward", "no_intervention"]


def build_option_set(base: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Derive the candidate protocols around the patient's certified order.

    The alternatives are anchored to ``base`` rather than to fixed numbers: the
    antibiotic an option "only" gives is the one this patient can actually receive
    (penicillin-sparing where the history demands it), and the fluid a bundle gives
    respects the cap the bundle engine already applied for CHF or renal failure.
    An option set built from constants would quietly re-introduce the exact hazards
    the protocol derivation exists to avoid.
    """
    abx = int(base.get("antibiotic_code", 0) or 0)
    fluid = float(base.get("fluid_rate", 0.0) or 0.0)
    vaso = float(base.get("vasopressor_dose", 0.0) or 0.0)
    # If the certified order carries no antibiotic, the alternatives still need one
    # to be meaningful comparisons; cefepime is the penicillin-sparing default.
    abx_alt = abx or 3
    return {
        "no_intervention": {"fluid_rate": 0.0, "vasopressor_dose": 0.0,
                            "antibiotic_code": 0},
        "antibiotics_only": {"fluid_rate": 0.0, "vasopressor_dose": 0.0,
                             "antibiotic_code": abx_alt},
        "standard_bundle": {"fluid_rate": max(fluid, 5.0), "vasopressor_dose": 0.0,
                            "antibiotic_code": abx_alt},
        "vasopressor_forward": {"fluid_rate": min(max(fluid, 1.0), 3.0),
                                "vasopressor_dose": max(vaso, 0.10),
                                "antibiotic_code": abx_alt},
        "recommended": dict(base),
    }


# --------------------------------------------------------------------------- #
# Upsides / downsides, generated from the projection.
# --------------------------------------------------------------------------- #
def _point(text: str, kind: str, evidence: str = "") -> Dict[str, str]:
    return {"text": text, "kind": kind, "evidence": evidence}


def _pk_points(arms: List[Dict[str, Any]], protocol: Dict[str, Any],
               risk_level: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    up: List[Dict[str, str]] = []
    down: List[Dict[str, str]] = []
    septic = risk_level in ("HIGH", "MEDIUM")

    abx_arm = next((a for a in arms if a["drug_class"] == "antibiotic"), None)
    if abx_arm is None:
        if septic:
            down.append(_point(
                "No antimicrobial cover — the single intervention with a mortality "
                "benefit in sepsis is absent", "pk",
                "Surviving Sepsis Campaign Hour-1 bundle"))
        else:
            up.append(_point("No antibiotic exposure — no selection pressure and no "
                             "drug toxicity", "pk"))
    else:
        m = abx_arm["metrics"]
        spec = DRUGS.get(abx_arm["drug"])
        pct = m.get("pct_ft_above_mic")
        if pct is not None and spec and spec.mic_mg_l:
            # Target attainment is the point of giving the drug, so it is the first
            # thing said about it either way.
            target = 50.0
            if "≥60" in (spec.target_note or ""):
                target = 60.0
            elif "≥40" in (spec.target_note or ""):
                target = 40.0
            line = (f"{abx_arm['label']} reaches {pct:.0f}% fT>MIC "
                    f"(target ≥{target:.0f}% at MIC {spec.mic_mg_l:g} mg/L)")
            (up if pct >= target else down).append(_point(
                line if pct >= target else line + " — under-exposed",
                "pk", spec.citation))
        auc24 = m.get("auc24")
        if abx_arm["drug"] == "vancomycin" and auc24:
            if 400.0 <= auc24 <= 600.0:
                up.append(_point(f"Vancomycin AUC24 {auc24:.0f} mg·h/L — inside the "
                                 "400–600 target band", "pk",
                                 "Rybak et al., consensus guideline 2020"))
            elif auc24 > 600.0:
                down.append(_point(f"Vancomycin AUC24 {auc24:.0f} mg·h/L — above 600, "
                                   "where AKI risk climbs steeply", "pk",
                                   "Rybak et al., consensus guideline 2020"))
            else:
                down.append(_point(f"Vancomycin AUC24 {auc24:.0f} mg·h/L — below the "
                                   "400 efficacy target", "pk",
                                   "Rybak et al., consensus guideline 2020"))

    vaso = float(protocol.get("vasopressor_dose", 0.0) or 0.0)
    fluid = float(protocol.get("fluid_rate", 0.0) or 0.0)
    ne_arm = next((a for a in arms if a["drug_class"] == "vasopressor"), None)
    if ne_arm is not None:
        t90 = ne_arm["metrics"].get("t_to_90pct_steady_state_h")
        up.append(_point(
            f"Norepinephrine {vaso:.2f} mcg/kg/min — perfusion support reaching "
            f"steady state in {(t90 or 0)*60:.0f} min", "pk",
            "Levophed FDA label; titrated to MAP ≥ 65"))
    fl_arm = next((a for a in arms if a["drug_class"] == "fluid"), None)
    if fl_arm is not None:
        fm = fl_arm["metrics"]
        up.append(_point(
            f"Crystalloid {fluid:.0f} mL/kg/h — {fm.get('infused_ml', 0):.0f} mL "
            "of volume resuscitation", "pk", "Hahn volume kinetics"))
        ret = fm.get("plasma_retention_1h_post")
        if ret is not None and ret < 0.30:
            down.append(_point(
                f"Only {ret*100:.0f}% of that volume is still intravascular an hour "
                f"later — {fm.get('interstitial_ml_end', 0):.0f} mL goes to tissue",
                "pk", "Chowdhury et al., Ann Surg 2012"))
    if not arms:
        down.append(_point("Nothing is administered, so there is no exposure to "
                           "project and no drug effect to expect", "pk"))
    return up, down


def _risk_points(adverse: Optional[Dict[str, Any]]
                 ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    up: List[Dict[str, str]] = []
    down: List[Dict[str, str]] = []
    if not adverse or not adverse.get("events"):
        return up, down
    events = adverse["events"]
    # Downsides: anything serious and non-trivial, or anything this exposure has
    # materially RAISED above its published population rate.
    for e in events:
        p, base = e["probability"], e["base_incidence"]
        serious = SEVERITY_RANK[e["severity"]] >= SEVERITY_RANK["severe"]
        delta = p - base
        if (serious and p >= 0.05) or delta >= 0.05:
            line = f"{e['label']} {p*100:.0f}%"
            # Only ever quote the delta UPWARD here. The base incidence is the rate
            # among patients who receive this drug, so a patient not receiving it
            # carries roughly none of this risk — printing "-4 pts vs the 10%
            # population rate" beside a downside reads as a complaint about
            # something that improved, when the actual downside is that the risk
            # exists at all once you give the drug.
            if delta >= 0.01:
                line += (f" (+{delta*100:.0f} pts above the {base*100:.0f}% rate "
                         f"seen at a standard dose)")
            down.append(_point(line, "risk", e["evidence"]))
    # An upside worth stating: this exposure LOWERS a risk the population carries.
    for e in events:
        if e["base_incidence"] - e["probability"] >= 0.05:
            down_by = (e["base_incidence"] - e["probability"]) * 100
            up.append(_point(
                f"{e['label']} {e['probability']*100:.0f}% — {down_by:.0f} pts below "
                f"the {e['base_incidence']*100:.0f}% rate seen at a standard dose",
                "risk", e["evidence"]))
    p_any = adverse.get("summary", {}).get("p_any_serious")
    if p_any is not None and p_any < 0.05:
        up.append(_point(f"Low aggregate serious-event risk ({p_any*100:.1f}%)",
                         "risk", "sum over the catalogued serious events"))
    return up, down


# --------------------------------------------------------------------------- #
def evaluate_options(base_protocol: Dict[str, Any], patient: PatientState,
                     history: Optional[Dict[str, Any]] = None,
                     risk_level: str = "MEDIUM", map_mmhg: Optional[float] = None,
                     cage: Optional[NeuroSymbolicRuntime] = None,
                     n_points: int = 121, n_samples: int = 12000) -> Dict[str, Any]:
    """Project every candidate order and return them ranked, with reasons."""
    history = history or {}
    cage = cage or NeuroSymbolicRuntime()
    engine = AdverseEventEngine(n_samples=n_samples)
    options = build_option_set(base_protocol)
    mapv = float(map_mmhg if map_mmhg is not None else patient.map_mmhg)

    out: List[Dict[str, Any]] = []
    for key in OPTION_ORDER:
        proto = options.get(key)
        if proto is None:
            continue
        # Same cage, same context as the write path: an option this OS would refuse
        # to dispatch must be shown as refused, not quietly offered.
        ok, cert = cage.dispatch(PROTOCOL, proto, {
            "risk_level": risk_level, "vitals": {"map": mapv},
            "clinical_history": history})
        pk = project_protocol(proto, patient, n_points=n_points)
        adverse = (engine.project(pk["arms"], patient, history)
                   if pk["arms"] else None)
        up, down = _pk_points(pk["arms"], proto, risk_level)
        rup, rdown = _risk_points(adverse)
        up += rup
        down += rdown
        if not ok:
            down.insert(0, _point(
                f"Refused by the safety cage — {cert.message}", "safety",
                f"Z3 {cert.z3_status}"))
        out.append({
            "key": key, "label": OPTION_LABEL.get(key, key),
            "note": OPTION_NOTE.get(key, ""),
            "protocol": proto,
            "antibiotic": ANTIBIOTICS.get(int(proto.get("antibiotic_code", 0) or 0), "None"),
            "safe": bool(ok),
            "verification": {"z3_status": cert.z3_status, "message": cert.message,
                             "violations": list(cert.violations)},
            "arms": pk["arms"],
            "adverse": adverse,
            "upsides": up, "downsides": down,
            "n_serious": (adverse or {}).get("summary", {}).get("n_serious", 0),
            "p_any_serious": (adverse or {}).get("summary", {}).get("p_any_serious", 0.0),
        })

    # The recommended option is the certified one when the cage allows it. This
    # layer deliberately does NOT invent its own ranking of clinical merit: it
    # compares exposures and consequences, and the protocol derivation plus the
    # formal gate remain what decide.
    rec = next((o["key"] for o in out if o["key"] == "recommended" and o["safe"]), None)
    if rec is None:
        rec = next((o["key"] for o in out if o["safe"]), None)
    return {
        "options": out, "recommended": rec,
        "patient": patient.to_dict(), "risk_level": risk_level,
        "base_protocol": dict(base_protocol),
        "honesty": (
            "Each option is a real PBPK projection of that order plus its "
            "population adverse-event profile. Upsides and downsides are generated "
            "from those numbers against published targets — they are not a "
            "clinical recommendation, and an empty downside list means nothing "
            "crossed a threshold, not that an option is safe."),
    }


def _demo() -> None:
    pt = PatientState(weight_kg=78.0, egfr=40.0, map_mmhg=58.0)
    base = {"fluid_rate": 10.0, "vasopressor_dose": 0.12, "antibiotic_code": 2}
    res = evaluate_options(base, pt, {"age": 72}, risk_level="HIGH")
    print(f"recommended: {res['recommended']}\n")
    for o in res["options"]:
        flag = "OK " if o["safe"] else "REF"
        print(f"[{flag}] {o['label']:22s} {o['antibiotic']:24s} "
              f"fluid {o['protocol']['fluid_rate']:4.1f} vaso "
              f"{o['protocol']['vasopressor_dose']:.2f} · "
              f"P(serious) {o['p_any_serious']*100:5.1f}%")
        for u in o["upsides"][:3]:
            print(f"        + {u['text']}")
        for d in o["downsides"][:3]:
            print(f"        - {d['text']}")
        print()


if __name__ == "__main__":
    _demo()
