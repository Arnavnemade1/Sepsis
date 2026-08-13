"""CaseDossier — the grounded briefing the AI doctor is allowed to talk about.

The consult call is the one place in AegisOS where a language model speaks freely
to a clinician. That is only safe if the model is arguing from *this* patient's
real state and knows exactly how much that state is worth. This module builds
both halves of that:

  * **The case** — patient, calibrated risk, vitals with units, SIRS/qSOFA and the
    criteria that actually fired, organ-system state, active tracking foci, and
    whatever advisory protocol/what-if the OS has already synthesized.
  * **The provenance** — what the deployed detector scored on *external* data,
    read live from ``artifacts/rigorous_eval.json`` and
    ``artifacts/operating_point.json``. Not from anyone's memory of them.

That second half is the point. An AI doctor that says "risk is 71%" and stops is
worse than useless; this one is required to also say that the detector it is
quoting scores AUROC ~0.72 across hospitals, that roughly a third of its alarms
are false, and which values in front of it are measured versus imputed. The
dossier carries the caveats *with* the numbers so the model cannot quote one
without the other.

``UNMEASURED`` is deliberate: the fields a wearable-feature detector never sees
(cultures, lactate assay, imaging) are listed explicitly, so "we don't have that"
is a fact in the prompt rather than something the model has to infer — inference
is where confabulation starts.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ARTIFACTS = os.path.join(_ROOT, "artifacts")

# Inputs the deployed wearable-feature detector structurally cannot see. Stated
# up front so the doctor declines them as missing instead of inventing them.
UNMEASURED = [
    "blood cultures / microbiology",
    "serum lactate assay (the streamed lactate is a wearable proxy, not a lab draw)",
    "white-cell count and differential unless present in clinical_history",
    "imaging (CXR, CT, ultrasound)",
    "urine output, fluid balance and vasopressor dose",
    "antibiotic administration times",
]

_PROV_CACHE: Dict[str, Any] = {"t": 0.0, "v": None}


def _load(name: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(_ARTIFACTS, name), "r") as fh:
            return json.load(fh)
    except Exception:
        return None


def provenance(ttl_s: float = 600.0) -> Dict[str, Any]:
    """Real, external-validation performance of the deployed detector.

    Read from the committed evaluation artifacts so the doctor can never quote a
    number that is not reproducible from this repository. If the artifacts are
    missing the result says ``known: False`` and the doctor is instructed to say
    the performance is unverified rather than guess at it."""
    now = time.time()
    if _PROV_CACHE["v"] is not None and now - _PROV_CACHE["t"] < ttl_s:
        return _PROV_CACHE["v"]

    out: Dict[str, Any] = {"known": False}
    ev = _load("rigorous_eval.json")
    if ev and isinstance(ev.get("directions"), dict):
        dirs = []
        for key, d in ev["directions"].items():
            m = (d.get("models") or {}).get("stacked") or (d.get("models") or {}).get("gbm")
            if not m:
                continue
            ew = m.get("early_warning") or {}
            dirs.append({
                "direction": key, "train": d.get("train"), "test": d.get("test"),
                "auroc": m.get("auroc"), "ece": m.get("ece"),
                "detection_rate": ew.get("detection_rate"),
                "median_lead_hours": ew.get("median_lead_time_hours"),
                "false_alarm_rate": ew.get("false_alarm_rate"),
            })
        if dirs:
            aur = [d["auroc"] for d in dirs if d.get("auroc") is not None]
            far = [d["false_alarm_rate"] for d in dirs if d.get("false_alarm_rate") is not None]
            lead = [d["median_lead_hours"] for d in dirs if d.get("median_lead_hours") is not None]
            det = [d["detection_rate"] for d in dirs if d.get("detection_rate") is not None]
            out = {
                "known": True,
                "protocol": (ev.get("meta") or {}).get(
                    "protocol", "cross-hospital external validation"),
                "directions": dirs,
                "auroc_range": [min(aur), max(aur)] if aur else None,
                "false_alarm_range": [min(far), max(far)] if far else None,
                "median_lead_range": [min(lead), max(lead)] if lead else None,
                "detection_range": [min(det), max(det)] if det else None,
            }
    op = _load("operating_point.json")
    if op:
        out["operating_point"] = {
            "high": op.get("high"), "medium": op.get("medium"),
            "derived_from": "out-of-fold thresholds, constrained to "
                            f"<= {op.get('max_false_alarm_rate')} false-alarm rate",
            "oof_detection_rate": op.get("oof_detection_rate"),
            "oof_median_lead_hours": op.get("oof_median_lead_hours"),
        }
    _PROV_CACHE.update({"t": now, "v": out})
    return out


def provenance_text() -> str:
    """The provenance block, as prose the model must keep attached to its numbers."""
    p = provenance()
    if not p.get("known"):
        return ("DETECTOR PROVENANCE: the external-validation artifacts are not "
                "loadable on this host. Say the detector's performance is "
                "unverified here; do NOT quote accuracy figures from memory.")
    a, f = p.get("auroc_range"), p.get("false_alarm_range")
    l, d = p.get("median_lead_range"), p.get("detection_range")
    op = p.get("operating_point") or {}
    lines = [
        "DETECTOR PROVENANCE (real, from this system's committed evaluation — "
        f"{p.get('protocol')}):",
        f"  - AUROC {a[0]:.3f}-{a[1]:.3f} when trained on one hospital and tested "
        "on the other. That is modest. It is a triage prior, not a diagnosis."
        if a else "",
        f"  - Catches {d[0]*100:.0f}-{d[1]*100:.0f}% of septic patients at the "
        f"deployed threshold, median {l[0]:.0f}-{l[1]:.0f} h before onset."
        if d and l else "",
        f"  - False-alarm rate {f[0]*100:.0f}-{f[1]*100:.0f}% of non-septic "
        "patients. Roughly one in three alarms is wrong; a HIGH tier means look, "
        "not treat." if f else "",
    ]
    if op.get("high") is not None:
        lines.append(
            f"  - On its own calibrated out-of-fold scale the detector alarms at "
            f"HIGH>={op['high']:.4f} MEDIUM>={op['medium']:.4f} ({op.get('derived_from')}). "
            "Those numbers are small because sepsis prevalence in the validation "
            "cohort is ~2%.")
    return "\n".join(x for x in lines if x)


def _scale_note(prob: Optional[float], bands: Optional[Dict[str, float]]) -> List[str]:
    """Guard against the one cross-comparison that would produce nonsense.

    The tier shown on screen is banded on the fused per-bed risk (HIGH>=0.6), while
    the detector's published operating point lives on its calibrated out-of-fold
    scale (HIGH>=0.0725). They are different quantities. Without this note the
    model sees "tier LOW" beside "P=0.31" beside "alarm at 0.0725" and reconciles
    them by inventing a story. State the banding that actually produced the tier,
    and forbid the comparison."""
    if not bands:
        return []
    out = [f"  TIER BANDING IN EFFECT for the probability above: "
           f"HIGH>={bands.get('high')}, MEDIUM>={bands.get('medium')}, else LOW."]
    op = (provenance().get("operating_point") or {}).get("high")
    if op is not None and bands.get("high") is not None and abs(float(bands["high"]) - float(op)) > 1e-6:
        out.append("  DO NOT compare this probability against the detector's "
                   f"published operating point ({op}) quoted below — that threshold "
                   "is on the detector's own calibrated out-of-fold scale, and this "
                   "number is the fused per-bed risk. Different scales. Quoting one "
                   "against the other is meaningless; if asked, say so.")
    return out


# ------------------------------------------------------------------ #
def _fmt(v: Any, nd: int = 1) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "—"


def _vitals_block(v: Dict[str, Any]) -> List[str]:
    order = [
        ("heart_rate", "HR", "bpm", 0), ("respiratory_rate", "RR", "/min", 0),
        ("skin_temp", "Temp", "°C", 1), ("spo2", "SpO2", "%", 0),
        ("systolic_bp", "SBP", "mmHg", 0), ("diastolic_bp", "DBP", "mmHg", 0),
        ("map", "MAP", "mmHg", 0), ("hrv", "HRV", "ms", 0),
        ("sweat_lactate", "lactate (wearable proxy)", "mmol/L", 2),
    ]
    out = []
    for key, label, unit, nd in order:
        if v.get(key) is not None:
            out.append(f"{label} {_fmt(v[key], nd)} {unit}")
    return out


def _sirs_detail(v: Dict[str, Any], hist: Dict[str, Any]) -> List[str]:
    """Which SIRS criteria fired — the model must be able to name them, not just
    quote the count."""
    fired = []
    t = v.get("skin_temp")
    if t is not None and (float(t) > 38 or float(t) < 36):
        fired.append(f"temp {_fmt(t)}°C")
    hr = v.get("heart_rate")
    if hr is not None and float(hr) > 90:
        fired.append(f"HR {_fmt(hr, 0)}")
    rr = v.get("respiratory_rate")
    if rr is not None and float(rr) > 20:
        fired.append(f"RR {_fmt(rr, 0)}")
    wbc = hist.get("wbc")
    if wbc is not None and (float(wbc) > 12 or float(wbc) < 4):
        fired.append(f"WBC {_fmt(wbc)}")
    return fired


def _qsofa_detail(v: Dict[str, Any], hist: Dict[str, Any]) -> List[str]:
    fired = []
    rr = v.get("respiratory_rate")
    if rr is not None and float(rr) >= 22:
        fired.append(f"RR {_fmt(rr, 0)} >= 22")
    sbp = v.get("systolic_bp")
    if sbp is not None and float(sbp) <= 100:
        fired.append(f"SBP {_fmt(sbp, 0)} <= 100")
    gcs = hist.get("gcs")
    if gcs is not None and int(gcs) < 15:
        fired.append(f"GCS {gcs} < 15")
    return fired


def _record_block(record: Optional[Dict[str, Any]]) -> List[str]:
    """The patient's imported prior record — who they were before this admission.

    A sepsis score computed from six vitals is a thin thing to reason from. The
    same soft pressure means something different in a dialysis-dependent
    72-year-old than in a healthy adult, and a proposed antibiotic means
    something different when the allergy list says penicillin. This block is what
    lets the consult say "given she's CKD stage 4" instead of reciting numbers at
    a stranger.

    Kept explicitly separate from the live vitals: the record is HISTORY, and the
    doctor must not present it as something measured today."""
    if not record:
        return ["PRIOR RECORD: none imported for this patient. If the clinician "
                "refers to past admissions, allergies or comorbidities, say the "
                "system has no chart on file and that one can be imported or "
                "linked from a record source."]
    out: List[str] = ["", "PRIOR RECORD (imported chart — HISTORY, not measured today):"]
    for key, label in (("comorbidities", "Comorbidities"), ("medications", "Medications"),
                       ("allergies", "ALLERGIES"), ("priorities", "Care priorities")):
        vals = record.get(key) or []
        if vals:
            out.append(f"  {label}: {', '.join(str(v) for v in vals[:12])}")
    if record.get("summary"):
        out.append(f"  Summary: {str(record['summary'])[:400]}")
    susc = record.get("susceptibility") or {}
    if susc.get("logit_uplift"):
        out.append(
            f"  Susceptibility: {susc.get('level', 'raised')} — logit uplift "
            f"{susc['logit_uplift']:+.2f} (odds ratio {susc.get('odds_ratio', '?')})"
            + (", capped" if susc.get("capped") else "") + ". This shifts WHEN the "
            "system alerts for this host; it is not a claim that their measured "
            "risk changed.")
    src = record.get("source")
    if src:
        out.append(f"  Source: {src}. Treat it as reported history, not verified "
                   "fact — say where it came from if it matters to the decision.")
    return out


def _organ_block(body: Dict[str, Any]) -> List[str]:
    """Non-normal organ systems only — a wall of 'normal' buries the one that isn't."""
    out = []
    systems = body.get("systems") or body.get("organs") or {}
    if isinstance(systems, dict):
        items = systems.items()
    elif isinstance(systems, list):
        items = [(s.get("name") or s.get("id"), s) for s in systems]
    else:
        return out
    for name, s in items:
        if not isinstance(s, dict):
            continue
        sev = s.get("severity", s.get("score"))
        state = str(s.get("state") or s.get("status") or "").lower()
        try:
            sev_f = float(sev)
        except (TypeError, ValueError):
            sev_f = 0.0
        if sev_f >= 0.25 or state in ("stress", "dysfunction", "critical"):
            out.append(f"{name}: {state or 'elevated'} ({sev_f:.2f})")
    return out


def build_dossier(snapshot: Dict[str, Any],
                  cohort: Optional[List[Dict[str, Any]]] = None,
                  tier_bands: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Assemble the grounded case briefing for one patient.

    ``snapshot`` is a ``/aegis/patient/snapshot`` payload. ``tier_bands`` is the
    banding that actually produced the tier (see :func:`_scale_note`). Returns
    ``{bed, headline, text, facts, unmeasured}``; ``text`` is what the model is
    given and is the ONLY clinical content it may speak from."""
    snapshot = snapshot or {}
    pat = snapshot.get("patient") or {}
    risk = snapshot.get("risk") or {}
    v = snapshot.get("vitals") or {}
    det = snapshot.get("detection") or {}
    hist = pat.get("clinical_history") or {}
    body = snapshot.get("body_state") or {}

    bed = str(pat.get("id") or snapshot.get("bed") or "—")
    name = pat.get("name") or bed
    level = str(risk.get("level") or "—").upper()
    prob = risk.get("ml_probability", det.get("sepsis_probability"))

    L: List[str] = []
    L.append(f"PATIENT: {name} (bed {bed})"
             + (f", age {pat['age']}" if pat.get("age") else "")
             + (f", {pat['gender']}" if pat.get("gender") else ""))
    if hist:
        keep = {k: hist[k] for k in list(hist)[:10]}
        L.append(f"CHART FLAGS (from the monitored record): {json.dumps(keep, default=str)}")
    L.extend(_record_block(snapshot.get("record")))

    L.append("")
    L.append(f"CURRENT RISK: tier {level}, calibrated P(sepsis) = "
             f"{_fmt(prob, 3) if prob is not None else '—'}")
    if risk.get("population_risk") is not None and risk.get("personalized_risk") is not None:
        pop, per = risk["population_risk"], risk["personalized_risk"]
        if abs(float(pop) - float(per)) > 1e-4:
            L.append(f"  (population {_fmt(pop, 3)} -> personalized {_fmt(per, 3)}; the "
                     "personalized value shifts the alerting prior for a high-risk host "
                     "and is NOT a claim that this patient's population risk changed)")
    L.extend(_scale_note(prob, tier_bands))
    sirs = _sirs_detail(v, hist)
    qsofa = _qsofa_detail(v, hist)
    L.append(f"  SIRS {risk.get('sirs_score', '—')}/4"
             + (f" — firing: {', '.join(sirs)}" if sirs else " — none firing"))
    L.append(f"  qSOFA {risk.get('qsofa_score', '—')}/3"
             + (f" — firing: {', '.join(qsofa)}" if qsofa else " — none firing"))
    sq = risk.get("signal_quality", det.get("sensor_quality"))
    if sq is not None:
        L.append(f"  signal quality {float(sq)*100:.0f}%"
                 + ("  <- degraded; weight the optical channels accordingly"
                    if float(sq) < 0.6 else ""))

    vb = _vitals_block(v)
    if vb:
        L.append("")
        L.append("VITALS: " + " · ".join(vb))
    if snapshot.get("recorded"):
        L.append("  NOTE: this bed is a REPLAYED recorded record from the validation "
                 "cohort, not a live bedside monitor. Say so if asked what you are "
                 "looking at.")

    ob = _organ_block(body)
    if ob:
        L.append("")
        L.append("ORGAN-SYSTEM STATE (non-normal only): " + "; ".join(ob))

    track = (body.get("tracking") or {})
    foci = track.get("foci") or []
    if foci:
        L.append(f"ACTIVE TRACKING FOCI (set by the attending): {', '.join(map(str, foci))}")

    proto = snapshot.get("protocol")
    if proto:
        L.append("")
        L.append("OS-SYNTHESIZED ADVISORY PROTOCOL (already generated by the "
                 "certified layer, advisory only): "
                 + json.dumps(proto, default=str)[:600])
    narr = snapshot.get("narrative")
    if narr:
        L.append(f"EXISTING SYSTEM NARRATIVE: {str(narr)[:400]}")

    cg = snapshot.get("causal_graph")
    if cg:
        L.append(f"CAUSAL GRAPH SUMMARY: {json.dumps(cg, default=str)[:400]}")

    if cohort:
        crit = [c for c in cohort if str(c.get("tier", "")).upper() == "CRITICAL"]
        watch = [c for c in cohort if str(c.get("tier", "")).upper() == "WATCH"]
        L.append("")
        L.append(f"UNIT CONTEXT: {len(cohort)} monitored beds, {len(crit)} critical, "
                 f"{len(watch)} watch.")

    L.append("")
    L.append("NOT MEASURED BY THIS SYSTEM (never assert or estimate these — if the "
             "clinician asks, say the system does not have it): "
             + "; ".join(UNMEASURED))
    L.append("")
    L.append(provenance_text())

    text = "\n".join(L)
    # "bed_0" already contains the word, and the headline is read ALOUD on the
    # voice call — "bed bed_0" comes out as "bed bed zero".
    bed_label = bed[4:] if bed.lower().startswith("bed_") else bed
    headline = f"{name} · bed {bed_label} · {level}" + (
        f" · P={_fmt(prob, 3)}" if prob is not None else "")
    return {
        "bed": bed,
        "name": name,
        "headline": headline,
        "text": text,
        "facts": {
            "tier": level, "probability": prob,
            "sirs": risk.get("sirs_score"), "qsofa": risk.get("qsofa_score"),
            "sirs_firing": sirs, "qsofa_firing": qsofa,
            "signal_quality": sq, "recorded": bool(snapshot.get("recorded")),
            "tier_bands": tier_bands,
            "vitals": {k: v.get(k) for k in
                       ("heart_rate", "respiratory_rate", "skin_temp", "spo2",
                        "systolic_bp", "map") if v.get(k) is not None},
        },
        "unmeasured": list(UNMEASURED),
        "provenance": provenance(),
    }
