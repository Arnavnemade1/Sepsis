"""Model training from the OS, the validation lab and the Scenario Lab.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from backend.api.schemas import (
    MLTrainIn,
    MLUseIn,
    ScenarioIn,
)
from backend.aegis.server import (
    _lab_job,
    _read_json,
    mlctl,
)

router = APIRouter()


@router.get("/aegis/models")
def models() -> Dict[str, Any]:
    """Every model the system runs, by name: what it reads, where it serves, whether
    it can page anyone, and its measured result (backend/aegis/models_registry.py)."""
    from backend.aegis import models_registry
    from backend.aegis.server import _llm_client, cluster
    try:
        llm = _llm_client.status()
    except Exception:
        llm = None
    return models_registry.registry(getattr(cluster, "ward", None), llm)


@router.get("/aegis/lab/evidence")
def lab_evidence() -> Dict[str, Any]:
    """What is deployed, and the evidence behind it, read from the artifacts.

    The detector block comes from ``detector_manifest.json``, the record the
    promotion gate writes when it ships a model, so it describes the model that
    is actually loaded. Every other block carries the detector it was measured
    on and ``matches_detector``: an artifact scored on an earlier fit is still
    shown, but never silently presented as describing this one.
    """
    from backend.ml import config as _cfg
    import os as _os

    def art(name: str) -> Any:
        return _read_json(_os.path.join(_cfg.ARTIFACTS_DIR, name))

    man = art("detector_manifest.json") or {}
    det_id = man.get("detector_id")
    keep = ("auroc", "auprc", "ece", "utility", "detect_at_far15", "lead_at_far15",
            "detection_at_spec90", "false_alarm_at_spec90")
    hold = man.get("holdout_metrics") or {}
    out: Dict[str, Any] = {"detector": None, "operating_point": None,
                           "conformal": None, "falsification": None, "ward": None}
    if man:
        out["detector"] = {
            "id": det_id, "promoted": bool(man.get("promoted")), "created": man.get("created"),
            "feature_set": man.get("feature_set"),
            "roster": (man.get("configs") or {}).get("candidate", {}).get("roster"),
            "calibration": (man.get("configs") or {}).get("candidate", {}).get("calibration"),
            "train_patients": (man.get("shipped_fit") or {}).get("train_patients"),
            "judged_patients": man.get("judged_patients"),
            "judged_on": man.get("incumbent_source"),
            "shipped": {k: (hold.get("candidate") or {}).get(k) for k in keep},
            "predecessor": {k: (hold.get("incumbent") or {}).get(k) for k in keep},
            "gates": man.get("gates"),
            # the share of septic patients caught when the alarm may fire on at most
            # k% of non-septic patients (the README's "inside a 15% alarm budget")
            "alarm_budget": {
                side: {k: v for k, v in ((man.get("alarm_budget") or {}).get(side) or {}).items()}
                for side in ("candidate", "incumbent")},
        }
    op = art("operating_point.json")
    if op:
        out["operating_point"] = {
            "high": op.get("high"), "medium": op.get("medium"),
            "max_false_alarm_rate": op.get("max_false_alarm_rate"),
            "detection_rate": op.get("oof_detection_rate"),
            "false_alarm_rate": op.get("oof_false_alarm_rate"),
            "false_alarm_upper": op.get("oof_false_alarm_upper"),
            "median_lead_hours": op.get("oof_median_lead_hours"),
            "n_patients": op.get("n_patients"),
            "detector": op.get("detector"), "matches_detector": op.get("detector") == det_id,
        }
    cf = art("conformal.json")
    if cf:
        locked = (cf.get("coverage") or {}).get("locked") or {}
        overall = locked.get("overall") or {}
        out["conformal"] = {
            "alpha": cf.get("alpha"), "nominal": locked.get("nominal"),
            "coverage_septic": overall.get("micro_pos"), "coverage_non_septic": overall.get("micro_neg"),
            "abstain_rate": locked.get("abstain_rate"), "n_patients": locked.get("n_patients"),
            "detector": cf.get("detector"), "matches_detector": cf.get("detector") == det_id,
        }
    # Which beds the detector actually scores, and what the ward read before it did
    # (backend/aegis/ward_risk.py; measured by backend/ml/scripts/ward_auroc.py).
    wr = art("ward_risk.json")
    if wr:
        sc = wr.get("scores") or {}
        out["ward"] = {
            "detector": wr.get("detector"), "matches_detector": wr.get("detector") == det_id,
            "patients": wr.get("patients"), "rows": wr.get("rows"), "split": wr.get("split"),
            "before": sc.get("rules"), "after": sc.get("detector"),
            "personalised": sc.get("index_pers"),
            "delta": wr.get("delta_detector_minus_rules"),
            "index_map": wr.get("index_map"),
            "scores": ["ICU rooms 301-312 (replayed CinC stays the detector never trained on)",
                       "room 402 (the live session)"],
        }
    # The all-variable detector that now serves the ICU replay beds, judged on the same
    # patients (backend/ml/scripts/train_icu_detector.py, registered beforehand).
    icu = art("icu_detector.json")
    op_icu = art("operating_point_icu.json")
    if icu and out["ward"] is not None:
        sc = icu.get("scores") or {}
        out["ward"]["icu"] = {
            "detector": icu.get("detector"), "shipped": bool(icu.get("shipped")),
            "matches_operating_point": bool(op_icu) and op_icu.get("detector") == icu.get("detector"),
            "score": sc.get("candidate"), "incumbent": sc.get("incumbent"), "delta": icu.get("delta"),
            "alarms": icu.get("alarms_half_b"), "budget15": icu.get("budget15_half_b"),
            "high": (op_icu or {}).get("high"), "medium": (op_icu or {}).get("medium"),
            "trees": icu.get("trees"), "train_patients": icu.get("train_patients"),
        }
        out["ward"]["scores"] = ["ICU rooms 301-312: the all-variable detector",
                                 "room 402 (the live session): the wearable detector"]
    fz = art("falsification.json")
    if fz:
        out["falsification"] = {
            "generated": fz.get("generated"), "mutations": fz.get("n_mutations"),
            "killed": fz.get("n_killed"), "skipped": fz.get("n_skipped"),
            "coverage": fz.get("coverage"), "survivors": fz.get("survivors"),
        }
    return out


@router.get("/aegis/lab")
def lab_state() -> Dict[str, Any]:
    """The Lab: agents propose, a locked gate adjudicates, refusals are the record.

    Read-only. Binding a real adjudicator fits models on held-out patients and takes
    minutes, so it is a job (`make lab-run`), not a request handler — a cheap gate is
    one somebody runs a thousand times until something passes.
    """
    from backend.aegis import lab as _lab
    from backend.ml import config as _cfg
    import os as _os, json as _json
    path = _os.path.join(_cfg.ARTIFACTS_DIR, "lab_ledger.json")
    ledger = None
    if _os.path.exists(path):
        try:
            with open(path) as fh:
                ledger = _json.load(fh)
            # Sanitise on the way OUT too. An older ledger on disk may predate the
            # json_safe fix, and FastAPI's encoder would happily re-emit its NaN tokens
            # to a browser that cannot parse them.
            ledger = _lab.Lab.json_safe(ledger)
        except Exception:
            ledger = None
    # The PROGRAMME, not just the last campaign. A ledger that is overwritten every run
    # cannot show whether a finding came from one campaign or from the tenth attempt, and
    # that is the number a reader most needs. The registry carries every campaign ever
    # run, hash-chained, and reports cumulative alpha as the expected false-finding count.
    programme, campaigns, chain, record = None, [], None, []
    try:
        from backend.aegis.lab_registry import LabRegistry as _Reg
        reg = _Reg.load()
        programme = reg.programme()
        campaigns = reg.campaigns[-12:]
        # The permanent record, newest first. The console showed only the last
        # campaign's ledger, which is the one thing a research record must not do —
        # a reader cannot tell a finding from the tenth attempt at one.
        record = [{"campaign": e.campaign, "agent": e.agent, "kind": e.kind,
                   "title": e.title, "status": e.status, "accepted": e.accepted,
                   "observed_delta": e.observed_delta, "mde": e.mde,
                   "alpha_spent": e.alpha_spent, "protocol": e.protocol,
                   "change_key": e.change_key}
                  for e in reversed(reg.entries[-80:])]
        ok, bad = reg.verify_chain()
        chain = {"ok": ok, "broken_at": bad, "n_entries": len(reg.entries),
                 "note": ("every recorded verdict re-hashes to its stored value"
                          if ok else
                          f"BROKEN at seq {bad} — a verdict has been edited or removed "
                          f"since it was recorded")}
    except Exception:
        pass

    return {
        "proposers": [{"name": p.name, "prior": p.prior} for p in _lab.default_proposers()],
        "roles": {
            "Proposer": "generates a change from a stated prior; never judges it",
            "Critic": "argues against it from this project's recorded nulls and the "
                      "Lab's own refusals; cannot veto",
            "SafetyOfficer": "absolute veto over anything that weakens a safety property",
            "Gate": "held-out patients, paired bootstrap; the only thing that can accept",
            "Replication": "a SECOND pool, disjoint from the gate by construction; the "
                           "only thing that can turn a candidate into a finding",
        },
        "programme": programme,
        "campaigns": campaigns,
        "record": record,
        "chain": chain,
        "partition": _read_json(_os.path.join(_cfg.ARTIFACTS_DIR, "lab_partition.json")),
        "min_effect": _lab.MIN_EFFECT,
        "claim_types": {
            "HELD_OUT": "measured on patients no agent could see or fit on",
            "MECHANISM": "PBPK twin only — a simulator result, never an efficacy claim",
            "REFUSED": "did not clear the gate, or was vetoed",
        },
        "ledger": ledger,
        "note": "A low acceptance rate is the expected outcome. selfimprove.py accepted "
                "0 of 16 under this same protocol; the gate refusing is the gate working.",
    }


@router.get("/aegis/ml")
def ml_status() -> Dict[str, Any]:
    """The ML control plane: registered models + running training jobs."""
    return mlctl.status()


@router.post("/aegis/ml/train")
def ml_train(payload: MLTrainIn) -> Dict[str, Any]:
    """Launch a REAL training/specialisation job (patient-grouped CV on CinC-2019).
    Returns the job id immediately; poll /aegis/ml/job/{id} for progress + metrics."""
    job = mlctl.train(payload.name, feature_set=payload.feature_set,
                      horizon=payload.horizon, limit=payload.limit, folds=payload.folds)
    return {"launched": True, "job": job.to_dict()}


@router.get("/aegis/ml/job/{job_id}")
def ml_job(job_id: str) -> Dict[str, Any]:
    j = mlctl.job(job_id)
    return j if j is not None else {"error": "no such job", "id": job_id}


@router.post("/aegis/ml/use")
def ml_use(payload: MLUseIn) -> Dict[str, Any]:
    return mlctl.use(payload.name)


@router.get("/aegis/scenarios/vocabulary")
def scenarios_vocabulary() -> Dict[str, Any]:
    """Everything needed to author a scenario: the deterioration arcs, the gradeable
    expectations, the interventions, and the limits. Read this before creating one."""
    from backend.aegis import scenarios as sc
    return {
        "arcs": sc.ARCS,
        "expectations": sc.EXPECTATION_KINDS,
        "interventions": {"fluids": "raises MAP/SBP, lowers HR",
                          "antibiotics": "4h lag, then bends the whole arc back",
                          "vasopressors": "raises MAP/SBP, raises HR"},
        "limits": {"max_scenarios": sc.ScenarioLab.MAX_SCENARIOS,
                   "los_hours": [8, 336], "label_lead_hours": sc.LABEL_LEAD},
        "synthetic": True,
        "caveat": "Scenarios grade BEHAVIOUR (did the alarm fire, in what order). "
                  "They can never measure accuracy — the patients are written by "
                  "hand, so any rate computed on them measures the generator.",
    }


@router.get("/aegis/scenarios")
def scenarios_list() -> Dict[str, Any]:
    """Every scenario in the sandbox, plus the pass/fail report for those that ran."""
    from backend.aegis.scenarios import LAB
    return {"scenarios": LAB.list(), "report": LAB.report()}


@router.get("/aegis/scenarios/{bed}")
def scenarios_get(bed: str) -> Dict[str, Any]:
    """One scenario's full result, including the hour-by-hour risk trace."""
    from dataclasses import asdict
    from backend.aegis.scenarios import LAB
    res = LAB.result(bed)
    if res is None:
        return {"bed": bed, "ran": False,
                "note": "not run yet — POST /aegis/scenarios/{bed}/run"}
    return asdict(res)


@router.post("/aegis/scenarios/create")
async def scenarios_create(payload: ScenarioIn) -> Dict[str, Any]:
    """Author a scenario. Returns its sandbox bed handle (always 'sim_'-prefixed)."""
    from backend.aegis.scenarios import LAB, Expectation, ScenarioSpec
    try:
        spec = ScenarioSpec(
            name=payload.name, arc=payload.arc, los_hours=payload.los_hours,
            onset_hour=payload.onset_hour, ramp_hours=payload.ramp_hours,
            severity=payload.severity, age=payload.age, gender=payload.gender,
            weight=payload.weight, lab_interval_hours=payload.lab_interval_hours,
            vitals_dropout=payload.vitals_dropout,
            interventions=[(int(h), str(k)) for h, k in payload.interventions],
            seed=payload.seed, notes=payload.notes)
        exps = [Expectation(kind=e.get("kind", ""), value=float(e.get("value", 0)),
                            because=str(e.get("because", "")))
                for e in payload.expectations]
        bed = LAB.create(spec, exps)
    except (ValueError, TypeError) as exc:
        # Returned rather than raised: the agent gets a correctable message instead
        # of a stack trace, and a rejected spec is a normal outcome here.
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "bed": bed, "fingerprint": spec.fingerprint(),
            "synthetic": True}


@router.post("/aegis/scenarios/{bed}/run")
async def scenarios_run(bed: str) -> Dict[str, Any]:
    """Run one scenario against the SHIPPED detector and grade its expectations."""
    from dataclasses import asdict
    from backend.aegis.scenarios import LAB
    try:
        return asdict(await _lab_job(f"scenario {bed}", LAB.run, bed, scenario=bed))
    except KeyError as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/aegis/scenarios/suite")
async def scenarios_suite() -> Dict[str, Any]:
    """Install and run the standing suite: deteriorations the OS must catch and
    confounders it must ignore. Replaces whatever is in the sandbox."""
    from backend.aegis.scenarios import LAB
    LAB.load_suite()
    await _lab_job("scenario suite", LAB.run_all)
    return LAB.report()


@router.post("/aegis/scenarios/run_all")
async def scenarios_run_all() -> Dict[str, Any]:
    """Run every scenario currently in the sandbox."""
    from backend.aegis.scenarios import LAB
    await _lab_job("scenario run_all", LAB.run_all)
    return LAB.report()


@router.post("/aegis/scenarios/reset")
async def scenarios_reset() -> Dict[str, Any]:
    """Empty the sandbox."""
    from backend.aegis.scenarios import LAB
    return {"ok": True, "cleared": LAB.reset()}

