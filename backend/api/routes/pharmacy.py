"""The Living Pharmacopoeia, the pharmacokinetic treatment twin and the clinical safety DSL.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

from fastapi import APIRouter

from backend.api.schemas import (
    DslVerifyIn,
    PKProjectIn,
    PharmaScreenIn,
)
from backend.aegis.server import (
    DEFAULT_POLICY,
    _PK_CACHE,
    _pk_context,
    _pk_key,
    _pk_resolve,
    _pk_store,
    bundle_engine,
    check_consistency,
    cluster,
    compile_rules,
    dsl_verify,
    pharmacopoeia,
    session,
    syscalls,
)

router = APIRouter()


@router.get("/aegis/pharmacopoeia")
def pharmacopoeia_stats() -> Dict[str, Any]:
    """The fused knowledge base: DDInter pairwise graph + curated CYP/QT/renal layer."""
    return pharmacopoeia.stats()


@router.post("/aegis/pharmacopoeia/screen")
def pharmacopoeia_screen(p: PharmaScreenIn) -> Dict[str, Any]:
    """Traverse the interaction graph for a regimen, compose a Z3 constraint per
    discovered hazard, and return the proof tree (globally safe or the violations)."""
    return pharmacopoeia.verify(p.drugs, p.context)


@router.get("/aegis/dsl/default")
def dsl_default() -> Dict[str, Any]:
    """The example hospital policy + a sample protocol/context to verify against."""
    return {"policy": DEFAULT_POLICY,
            "example_protocol": {"antibiotic_code": 1, "antibiotic_time_hr": 0.5,
                                 "fluid_rate": 4.0, "vasopressor_dose": 0.1},
            "example_context": {"risk_level": "HIGH", "chf": True,
                                "penicillin_allergy": True, "renal_failure": False}}


@router.post("/aegis/dsl/verify")
def dsl_verify_endpoint(p: DslVerifyIn) -> Dict[str, Any]:
    """Compile a hospital's safety rules to Z3 and prove a protocol satisfies them.
    Returns a per-rule proof certificate (with the compiled Z3 s-expression) plus a
    consistency check (is the rule set jointly satisfiable for this patient?)."""
    try:
        rules = compile_rules(p.policy)
    except SyntaxError as e:
        return {"status": "error", "detail": f"parse/compile error: {e}"}
    cert = dsl_verify(rules, p.protocol, p.context)
    consistency = check_consistency(rules, p.context)
    return {"status": "ok", "certificate": cert.to_dict(), "consistency": consistency,
            "n_rules": len(rules)}


@router.get("/aegis/pk/drugs")
def pk_drugs() -> Dict[str, Any]:
    """The drug library behind the treatment twin, with every PK parameter's
    citation and the adverse-event catalogue's evidence tiers. This is the
    provenance surface: everything the projection is built from, inspectable."""
    from backend.aegis.pharmacokinetics import (DRUGS, ANTIBIOTIC_CODE_TO_DRUG,
                                                FLUID_SOURCE, ORGAN_MESH,
                                                PHYSIOLOGY_SOURCE)
    from backend.aegis.adverse import CATALOGUE, VARIABILITY_SOURCE
    return {
        "drugs": [{
            "key": s.key, "label": s.label, "class": s.drug_class,
            "vd_l_per_kg": s.vd_l_per_kg, "cl_l_per_h_per_kg": s.cl_l_per_h_per_kg,
            "t_half_h": s.t_half_h, "f_renal": s.f_renal,
            "protein_binding": s.protein_binding, "regimen": s.regimen.label,
            "horizon_h": s.horizon_h, "unit": s.conc_unit,
            "mic_mg_l": s.mic_mg_l, "target": s.target_note, "citation": s.citation,
        } for s in DRUGS.values()],
        "fluid": {"key": "crystalloid", "label": "Crystalloid (balanced)",
                  "class": "fluid", "citation": FLUID_SOURCE},
        "antibiotic_codes": ANTIBIOTIC_CODE_TO_DRUG,
        "organ_mesh": ORGAN_MESH,
        "adverse_events": [e.to_dict() for e in CATALOGUE],
        "provenance": {"physiology": PHYSIOLOGY_SOURCE,
                       "variability": VARIABILITY_SOURCE},
        "honesty": ("Literature-parameterised population PK and population "
                    "adverse-event rates. Not therapeutic drug monitoring, and "
                    "not a prediction for an individual patient."),
    }


@router.post("/aegis/pk/project")
def pk_project(payload: PKProjectIn) -> Dict[str, Any]:
    """Project a certified order through the body: per-organ drug concentration
    over each arm's own timeframe, plus (by default) the side-effect projection.

    This is what the treatment twin renders. It never modifies anything — a
    projection is a read, and the Z3 cage remains the only thing that can gate an
    order."""
    from backend.aegis.pharmacokinetics import project_protocol
    protocol, patient, hist, err = _pk_resolve(payload)
    if err:
        return {"status": "error", "detail": err}

    key = _pk_key(protocol, patient, payload.adverse)
    if key in _PK_CACHE:
        return _PK_CACHE[key]

    out = project_protocol(protocol, patient, n_points=max(41, min(401, payload.n_points)))
    out["status"] = "ok"
    out["bed"] = payload.bed
    if payload.adverse and out["arms"]:
        from backend.aegis.adverse import AdverseEventEngine
        out["adverse"] = AdverseEventEngine().project(out["arms"], patient, hist)
    return _pk_store(key, out)


@router.get("/aegis/pk/bed/{bed_id}")
def pk_bed(bed_id: str, adverse: bool = True, n_points: int = 121) -> Dict[str, Any]:
    """The one-call form the dossier uses: project this bed's order as it stands.

    If the bed has been treated, this projects what was actually administered; if
    not, it projects the protocol the bundle engine would certify — and says which,
    so the UI never implies a drug is running when none was ordered."""
    if bed_id not in getattr(cluster, "profiles", {}):
        return {"status": "error", "detail": f"unknown bed '{bed_id}'"}
    iv = syscalls.interventions.get(bed_id)
    administered = bool(iv is not None and iv.treated and iv.protocol)
    out = pk_project(PKProjectIn(bed=bed_id, adverse=adverse, n_points=n_points))
    if out.get("status") == "ok":
        # Copied, not mutated in place: the cached payload is shared, and the
        # administered flag depends on syscall state that can change under it.
        out = dict(out)
        out["administered"] = administered
        out["source"] = "administered order" if administered else "certified protocol (not yet ordered)"
        prof = cluster.profiles[bed_id]
        out["bed_name"] = getattr(prof, "name", bed_id)
    return out


@router.get("/aegis/pk/options")
def pk_options(bed: Optional[str] = None, n_points: int = 121) -> Dict[str, Any]:
    """Several candidate orders, each projected through the body.

    Same candidate names the what-if card already shows, but each one carries a
    real PBPK projection, its adverse-event profile, a verdict from the same Z3
    cage the write path uses, and upsides/downsides generated from those numbers.
    This is what makes the treatment twin a comparison rather than a single
    scenario."""
    from backend.aegis.treatment_options import evaluate_options
    ctx = _pk_context(bed)
    if ctx is None:
        return {"status": "error", "detail": f"unknown bed '{bed}'"}
    proto, patient, hist, level, name = ctx
    key = "opt|" + (bed or "session") + "|" + _pk_key(proto, patient, True) + f"|{level}"
    if key in _PK_CACHE:
        return _PK_CACHE[key]
    out = evaluate_options(proto, patient, hist, risk_level=level,
                           n_points=max(41, min(401, n_points)))
    out["status"] = "ok"
    out["bed"] = bed
    out["bed_name"] = name
    iv = syscalls.interventions.get(bed) if bed else None
    out["administered"] = bool(iv is not None and iv.treated and iv.protocol)
    return _pk_store(key, out)


@router.get("/aegis/pk/session")
def pk_session(adverse: bool = True, n_points: int = 121) -> Dict[str, Any]:
    """The same projection for the LIVE single-patient session the clinical console
    watches, rather than for a cohort bed.

    The clinical console has one patient and no bed id, so it cannot use
    ``/aegis/pk/bed/{id}``. Same contract, same honesty: if a protocol has actually
    been dispatched into the session it projects that, otherwise it projects what
    the current assessment would certify and says so."""
    from backend.aegis.pharmacokinetics import PatientState, project_protocol
    snap = session.snapshot()
    vitals = snap.get("vitals") or {}
    hist = (snap.get("patient") or {}).get("clinical_history") or {}
    proto_block = snap.get("protocol") or {}
    proto = dict(proto_block.get("protocol") or {})
    administered = bool(proto_block.get("applied"))
    if not proto:
        # Nothing certified yet — derive what would be ordered for this patient, so
        # the twin can still answer "where would this go" before anyone commits.
        prof = SimpleNamespace(
            enterprise={"map": float(vitals.get("map", 80.0) or 80.0),
                        "lactate": float(vitals.get("sweat_lactate",
                                                    vitals.get("lactate", 1.0)) or 1.0)},
            fused_risk=float((snap.get("risk") or {}).get("ml_probability", 0.0) or 0.0),
            name=(snap.get("patient") or {}).get("name", "patient"),
            drivers=[], primary_system="")
        proto = dict(bundle_engine._derive_protocol(prof, hist))
        administered = False

    mapv = float(vitals.get("map", 80.0) or 80.0)
    hr = float(vitals.get("heart_rate", 90.0) or 90.0)
    egfr = 25.0 if hist.get("renal_failure") else 90.0
    patient = PatientState(
        weight_kg=float(hist.get("weight_kg", 78.0) or 78.0), egfr=egfr,
        hepatic_factor=0.6 if hist.get("cirrhosis") else 1.0, map_mmhg=mapv,
        cardiac_output_factor=float(min(1.6, max(0.7, 0.85 + (hr - 80.0) / 200.0))))

    key = "session|" + _pk_key(proto, patient, adverse)
    if key in _PK_CACHE:
        out = dict(_PK_CACHE[key])
    else:
        out = project_protocol(proto, patient,
                               n_points=max(41, min(401, n_points)))
        out["status"] = "ok"
        if adverse and out["arms"]:
            from backend.aegis.adverse import AdverseEventEngine
            out["adverse"] = AdverseEventEngine().project(out["arms"], patient, hist)
        _pk_store(key, out)
        out = dict(out)
    out["administered"] = administered
    out["source"] = ("administered order" if administered
                     else "certified protocol (not yet ordered)")
    out["bed_name"] = (snap.get("patient") or {}).get("name", "Monitored patient")
    return out

