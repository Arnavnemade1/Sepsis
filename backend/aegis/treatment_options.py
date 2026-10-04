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
    "extended_infusion": "Extended infusion",
    "recommended": "Recommended",
}
OPTION_NOTE = {
    "no_intervention": "Withhold treatment — the comparison baseline, not a plan.",
    "antibiotics_only": "Source control without haemodynamic support.",
    "standard_bundle": "Guideline crystalloid + broad-spectrum cover.",
    "vasopressor_forward": "Earlier pressor support, less volume.",
    "extended_infusion": "Same dose and same bundle — run over 4 h instead of 30 min.",
    "recommended": "What this patient's certified protocol actually orders.",
}
OPTION_ORDER = ["recommended", "extended_infusion", "standard_bundle",
                "antibiotics_only", "vasopressor_forward", "no_intervention"]

# A beta-lactam given over four hours instead of thirty minutes is the same total
# dose with a different exposure shape: lower peak, far more time above the MIC.
# For a time-dependent antibiotic that is the property that kills bacteria, so it
# is a live clinical strategy rather than a modelling curiosity — and it is the
# option that makes the comparison mean something, because every other path gives
# the same drug at the same rate and therefore lands almost the same curve.
EXTENDED_INFUSION_MIN = 240.0


def supports_extended_infusion(abx_code: int) -> bool:
    """Is prolonged infusion a real lever for this antibiotic?

    Only for TIME-DEPENDENT agents with a short half-life — the beta-lactams,
    whose killing tracks the fraction of the interval spent above the MIC.
    Stretching the infusion trades peak for exactly that fraction, which is why
    prolonged-infusion piperacillin and meropenem are an actual ICU strategy.

    It is deliberately NOT offered for vancomycin or a fluoroquinolone. Vancomycin
    is driven by AUC24, which prolonging barely moves, and a fluoroquinolone is
    concentration-dependent — flattening its peak makes it WORSE. Presenting the
    option there would imply a lever that does not exist, which is a more expensive
    kind of wrong than not offering it.
    """
    drug = ANTIBIOTIC_CODE_TO_DRUG.get(int(abx_code or 0))
    spec = DRUGS.get(drug or "")
    return bool(spec and spec.mic_mg_l and spec.t_half_h <= 3.0)


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
    opts: Dict[str, Dict[str, Any]] = {
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
    # Same bundle as the certified order, delivered slowly — only where that is a
    # real lever for the drug in question.
    if supports_extended_infusion(abx_alt):
        opts["extended_infusion"] = dict(base, antibiotic_code=abx_alt,
                                         infusion_min=EXTENDED_INFUSION_MIN)
    return opts


# --------------------------------------------------------------------------- #
# The haemodynamic consequence of an option, fed back into its own PK.
# --------------------------------------------------------------------------- #
# Coefficients shared with backend/aegis/physiology.py so the what-if projection
# and this one cannot disagree about what a pressor or a fluid bolus does:
#   MAP = base + 0.8·(mL/kg/h crystalloid) + 130·(mcg/kg/min norepinephrine)
MAP_PER_FLUID = 0.8
MAP_PER_VASO = 130.0


def patient_under(protocol: Dict[str, Any], base: PatientState) -> PatientState:
    """The patient state an option PRODUCES — not the one it starts from.

    This is the fix for a real defect: every option used to be projected into the
    same baseline patient, so the antibiotic curve came out byte-identical whether
    you gave it alone, with five litres of crystalloid, or with a pressor running.
    Four of the five paths rendered the same body, which is exactly what it looked
    like.

    Treatments are not independent. Two couplings are modelled, both standard:

      * A VASOPRESSOR RAISES MAP, which reverses the shock redistribution the model
        already applies — renal and splanchnic beds reopen, so a drug cleared
        renally is cleared faster and reaches the gut and liver differently.
      * CRYSTALLOID EXPANDS THE VOLUME OF DISTRIBUTION of a hydrophilic drug,
        because it distributes into the same extracellular water the drug does.
        This is the documented mechanism of beta-lactam under-exposure after
        aggressive fluid resuscitation in sepsis.

    What is NOT modelled, and is stated rather than hidden: antibiotics do not
    feed back on haemodynamics here, and there is no pharmacodynamic interaction
    between drugs. The couplings run one way, from haemodynamic support into drug
    kinetics.
    """
    fluid = float(protocol.get("fluid_rate", 0.0) or 0.0)
    vaso = float(protocol.get("vasopressor_dose", 0.0) or 0.0)
    # MAP the option achieves, capped at a normal ceiling — a pressor titrated to
    # MAP ≥ 65 restores perfusion, it does not make a patient hypertensive here.
    new_map = min(95.0, base.map_mmhg + MAP_PER_FLUID * fluid + MAP_PER_VASO * vaso)
    # Extracellular water gains the retained fraction of one hour of crystalloid.
    litres = fluid * base.weight_kg / 1000.0
    ecw = 0.2 * base.weight_kg                      # ~20% of body weight, litres
    expansion = 1.0 + (litres * 0.75) / max(ecw, 1e-6)
    # eGFR is deliberately NOT raised when MAP improves. It is tempting — restoring
    # perfusion does restore filtration — but the PBPK model already routes renal
    # blood flow off MAP, so crediting the pressor a second time in the clearance
    # term would count the same recovery twice. Same error as the covariate
    # double-count in the adverse layer; the flow model owns this pathway.
    return PatientState(
        weight_kg=base.weight_kg, egfr=base.egfr, hepatic_factor=base.hepatic_factor,
        map_mmhg=new_map, cardiac_output_factor=base.cardiac_output_factor,
        albumin_g_dl=base.albumin_g_dl, vd_expansion=expansion)


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
        # Project this option in the patient IT produces. Without this every arm
        # is solved in the same baseline body and the antibiotic curve comes out
        # identical across every path that gives one.
        pstate = patient_under(proto, patient)
        pk = project_protocol(proto, pstate, n_points=n_points)
        adverse = (engine.project(pk["arms"], pstate, history)
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
            # The haemodynamic state this option creates, so the UI can show WHY
            # two paths giving the same antibiotic produce different exposures.
            "patient_under": pstate.to_dict(),
        })

    # The recommended option is the certified one when the cage allows it. This
    # layer deliberately does NOT invent its own ranking of clinical merit: it
    # compares exposures and consequences, and the protocol derivation plus the
    # formal gate remain what decide.
    rec = next((o["key"] for o in out if o["key"] == "recommended" and o["safe"]), None)
    if rec is None:
        rec = next((o["key"] for o in out if o["safe"]), None)
    from backend.aegis.pk_narration import narrate_arms
    for o in out:
        o["narration"] = narrate_arms(o["arms"]) if o["arms"] else {}
    return {
        "options": out, "recommended": rec,
        "comparison": compare(out, rec),
        "patient": patient.to_dict(), "risk_level": risk_level,
        "base_protocol": dict(base_protocol),
        "honesty": (
            "Each option is a real PBPK projection of that order plus its "
            "population adverse-event profile. Upsides and downsides are generated "
            "from those numbers against published targets — they are not a "
            "clinical recommendation, and an empty downside list means nothing "
            "crossed a threshold, not that an option is safe."),
    }


# --------------------------------------------------------------------------- #
# The comparison. Options side by side, on the axes that actually differ.
# --------------------------------------------------------------------------- #
COMPARE_ROWS = [
    ("cover", "Antimicrobial cover", "the drug and whether it hits its target"),
    ("target", "%fT>MIC", "time free drug spends above the breakpoint"),
    ("exposure", "Peak plasma", "highest concentration the body sees"),
    ("volume", "Volume given", "crystalloid over the first hour"),
    ("pressor", "Pressor support", "norepinephrine rate"),
    ("serious", "P(any serious event)", "aggregate over catalogued serious events"),
    ("worst", "Largest single risk", "the highest-probability serious event"),
    ("safety", "Safety cage", "the same Z3 gate the write path uses"),
]


#: how each row reads inside a sentence (a label lower-cased whole turned %fT>MIC into
#: "%ft>mic", and "identical on safety cage" is not English)
_IN_SENTENCE = {"cover": "antimicrobial cover", "target": "%fT>MIC", "exposure": "peak plasma",
                "volume": "volume given", "pressor": "pressor support",
                "serious": "P(any serious event)", "worst": "the largest single risk",
                "safety": "the safety cage"}


def _listed(keys: List[str]) -> str:
    words = [_IN_SENTENCE.get(k, k) for k in keys]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def compare(options: List[Dict[str, Any]], reference: Optional[str] = None
            ) -> Dict[str, Any]:
    """Build the side-by-side table, plus an explicit DIFF against a reference.

    A rail of options each described on its own terms still leaves the reader
    doing the subtraction. This does the subtraction: for every option it states
    what changes relative to the reference path, in the units the change is
    actually in — 5 mL/kg/h more fluid, 8 points more AKI, cover unchanged.
    """
    ref = None
    for o in options:
        if o["key"] == reference:
            ref = o
    if ref is None:
        ref = next((o for o in options if o["arms"]), options[0] if options else None)
    if ref is None:
        return {"rows": [], "columns": [], "reference": None}

    def cell(o: Dict[str, Any]) -> Dict[str, Any]:
        abx = next((a for a in o["arms"] if a["drug_class"] == "antibiotic"), None)
        ne = next((a for a in o["arms"] if a["drug_class"] == "vasopressor"), None)
        fl = next((a for a in o["arms"] if a["drug_class"] == "fluid"), None)
        ev = ((o.get("adverse") or {}).get("events") or [])
        serious = [e for e in ev
                   if SEVERITY_RANK[e["severity"]] >= SEVERITY_RANK["severe"]]
        worst = max(serious, key=lambda e: e["probability"]) if serious else None
        return {
            "cover": {"text": abx["label"] if abx else "none",
                      "v": 1.0 if abx else 0.0,
                      "bad": abx is None},
            "target": {"text": (f"{abx['metrics']['pct_ft_above_mic']:.0f}%"
                                if abx and abx["metrics"].get("pct_ft_above_mic") is not None
                                else "—"),
                       "v": (abx["metrics"].get("pct_ft_above_mic")
                             if abx else None)},
            "exposure": {"text": (f"{abx['metrics']['cmax']:.0f} {abx['unit']}"
                                  if abx else "—"),
                         "v": abx["metrics"]["cmax"] if abx else None},
            "volume": {"text": (f"{fl['metrics']['infused_ml']:.0f} mL" if fl else "none"),
                       "v": fl["metrics"]["infused_ml"] if fl else 0.0},
            "pressor": {"text": (f"{o['protocol']['vasopressor_dose']:.2f} mcg/kg/min"
                                 if ne else "none"),
                        "v": float(o["protocol"].get("vasopressor_dose", 0) or 0)},
            "serious": {"text": f"{o['p_any_serious']*100:.0f}%",
                        "v": o["p_any_serious"], "lower_better": True},
            "worst": {"text": (f"{worst['label']} {worst['probability']*100:.0f}%"
                               if worst else "none catalogued"),
                      "v": worst["probability"] if worst else 0.0,
                      "lower_better": True},
            "safety": {"text": "certified" if o["safe"] else "REFUSED",
                       "v": 1.0 if o["safe"] else 0.0, "bad": not o["safe"]},
        }

    cells = {o["key"]: cell(o) for o in options}
    rc = cells[ref["key"]]

    def diff_for(o: Dict[str, Any]) -> List[Dict[str, str]]:
        """What changes versus the reference, in the change's own units."""
        if o["key"] == ref["key"]:
            return []
        c = cells[o["key"]]
        out: List[Dict[str, str]] = []
        if bool(c["cover"]["bad"]) != bool(rc["cover"]["bad"]):
            out.append({"k": "cover", "dir": "worse" if c["cover"]["bad"] else "better",
                        "text": ("loses antimicrobial cover" if c["cover"]["bad"]
                                 else "adds antimicrobial cover")})
        elif c["cover"]["text"] != rc["cover"]["text"]:
            out.append({"k": "cover", "dir": "same",
                        "text": f"{rc['cover']['text']} → {c['cover']['text']}"})
        # Direction is only claimed where one genuinely exists. More antimicrobial
        # target attainment is better and more serious-event risk is worse, so those
        # get a good/bad reading. MORE FLUID AND MORE PRESSOR ARE NOT BETTER OR
        # WORSE — whether either helps depends on the patient in front of you, and
        # colouring "+312 mL" green would be this layer inventing a clinical steer
        # it has no basis for. Those changes are reported as neutral facts.
        for key, unit, fmt, direction in (
                ("target", " pts %fT>MIC", "{:+.0f}", "higher_better"),
                ("volume", " mL fluid", "{:+.0f}", "neutral"),
                ("pressor", " mcg/kg/min pressor", "{:+.2f}", "neutral"),
                ("serious", " pts serious risk", "{:+.0f}", "lower_better")):
            a_, b_ = c[key]["v"], rc[key]["v"]
            if a_ is None or b_ is None:
                continue
            d = a_ - b_
            if key == "serious":
                d *= 100.0
            if abs(d) < (0.005 if key == "pressor" else 0.5):
                continue
            if direction == "neutral":
                dirn = "same"
            elif direction == "lower_better":
                dirn = "better" if d < 0 else "worse"
            else:
                dirn = "better" if d > 0 else "worse"
            out.append({"k": key, "dir": dirn, "text": fmt.format(d) + unit})
        if o["safe"] != ref["safe"]:
            out.append({"k": "safety", "dir": "worse" if not o["safe"] else "better",
                        "text": "refused by the safety cage" if not o["safe"]
                                else "passes the safety cage"})
        if not out:
            out.append({"k": "none", "dir": "same",
                        "text": "no material difference on these axes"})
        return out

    # Where do the options ACTUALLY diverge? Stating this is the difference between
    # a comparison and a table that merely looks like one. Four paths giving the
    # same antibiotic at the same dose produce nearly the same antibiotic curve —
    # fluid flattens the peak a little, a pressor changes how fast tissue
    # equilibrates but not the total, because AUC is dose/clearance and clearance
    # is not flow-limited for these drugs. Saying so is more useful than implying a
    # difference the model did not find.
    def spread(key: str) -> Optional[float]:
        vals = [cells[o["key"]][key]["v"] for o in options
                if isinstance(cells[o["key"]][key]["v"], (int, float))
                and cells[o["key"]][key]["v"] is not None]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return None
        lo, hi = min(vals), max(vals)
        if hi == 0:
            return 0.0
        return (hi - lo) / abs(hi)

    diverge: List[Dict[str, str]] = []
    for key, lab, _n in COMPARE_ROWS:
        sp = spread(key)
        if sp is None:
            continue
        if sp >= 0.15:
            diverge.append({"k": key, "level": "major", "text": lab})
        elif sp >= 0.02:
            diverge.append({"k": key, "level": "minor", "text": lab})
        else:
            diverge.append({"k": key, "level": "none", "text": lab})
    major = [d["k"] for d in diverge if d["level"] == "major"]
    same = [d["k"] for d in diverge if d["level"] == "none"]
    if major:
        headline = "These paths differ mainly in " + _listed(major[:3]) + "."
    else:
        headline = "These paths are close on every axis measured here."
    if same:
        headline += " They are effectively identical on " + _listed(same[:3]) + "."

    return {
        "reference": ref["key"],
        "reference_label": ref["label"],
        "divergence": diverge,
        "headline": headline,
        "columns": [{"key": o["key"], "label": o["label"], "safe": o["safe"]}
                    for o in options],
        "rows": [{"key": k, "label": lab, "note": note,
                  "cells": {o["key"]: cells[o["key"]][k] for o in options}}
                 for k, lab, note in COMPARE_ROWS],
        "diff": {o["key"]: diff_for(o) for o in options},
        "honesty": ("Differences are computed between projections, not between "
                    "outcomes. Two options differing by 8 points of projected AKI "
                    "risk is a statement about modelled exposure, not a trial."),
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
