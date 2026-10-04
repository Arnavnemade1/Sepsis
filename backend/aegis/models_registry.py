"""Every model AegisOS runs, by name: what it reads, what it scores, whether it decides.

The system runs a dozen models and they had been named by their file or their role
("the detector", "the stack", "fusion"), which stopped being enough once two detectors
served different beds. This is the one list. Each entry says:

    code       the short name the UI uses (A40, V7, ...)
    name       what it is, in words
    kind       detector | support | language
    decides    True only for the models whose output can page someone
    serves     where it runs
    reads      its inputs
    measured   its validation, read from the artifact that holds it where one exists,
               else quoted with the document or suite that measured it
    status     serving | fallback | explains | retired

Numbers here are never recomputed; a stale artifact shows its own detector id.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from backend.ml import config as mlcfg

#: detector id prefix -> code, so any payload can say which model scored it
CODES = {"icu:": "A40", "patient:": "V7"}


def code_for(detector_id: Optional[str]) -> Optional[str]:
    for prefix, code in CODES.items():
        if detector_id and detector_id.startswith(prefix):
            return code
    return None


def _art(name: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(mlcfg.ARTIFACTS_DIR, name)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def registry(ward: Any = None, llm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    icu, man, wr = _art("icu_detector.json"), _art("detector_manifest.json"), _art("ward_risk.json")
    op_icu, op, cf = _art("operating_point_icu.json"), _art("operating_point.json"), _art("conformal.json")
    hold = (man or {}).get("holdout_metrics", {}).get("candidate", {})
    ward_id = getattr(getattr(ward, "model", None), "detector_id", None) or getattr(getattr(ward, "op", None), "detector", None)
    a40_serving = code_for(ward_id) == "A40" and bool(getattr(ward, "ready", False))
    sc = (wr or {}).get("scores", {})

    models: List[Dict[str, Any]] = [
        {
            "code": "A40", "name": "All-variable detector", "kind": "detector", "decides": True,
            "status": "serving" if a40_serving else "unavailable",
            "serves": "ICU rooms 301–312 (replayed CinC records)",
            "reads": "all 40 CinC variables: vitals, labs, blood gases, demographics (510 causal features)",
            "method": "one LightGBM, tuned GENESIS parameters, Platt-calibrated",
            "id": (icu or {}).get("detector"),
            "trained_on": (icu or {}).get("train_patients"),
            "measured": None if not icu else {
                "auroc": icu["scores"]["candidate"]["auroc"],
                "hour_matched": icu["scores"]["candidate"]["hour_matched"],
                "ece": icu["scores"]["candidate"]["ece"],
                "patients": icu.get("judged_patients"),
                "vs": f"V7 {icu['scores']['incumbent']['auroc']:.3f} on the same patients; "
                      f"+{icu['delta']['delta']:.3f} [+{icu['delta']['lo']:.3f}, +{icu['delta']['hi']:.3f}]",
                "note": "inside a fixed 15% alarm budget, detection ties V7",
                "source": "artifacts/icu_detector.json (registered: icu_detector_prereg.json)"},
            "alarm": None if not op_icu else {"high": op_icu["high"], "medium": op_icu["medium"]},
            "artifact": "artifacts/gbm_icu.joblib",
        },
        {
            "code": "V7", "name": "Vital-sign detector", "kind": "detector", "decides": True,
            "status": "serving",
            "serves": "room 402, the live session" + ("" if a40_serving else "; and the ICU rooms (A40 unavailable)"),
            "reads": "7 vital signs a wearable or camera can stream (94 features)",
            "method": "stacked ensemble: 2 histogram boosters, extra-trees, logistic regression, utility-aimed XGBoost; "
                      "patient label; centred isotonic calibration",
            "id": (man or {}).get("detector_id"),
            "trained_on": ((man or {}).get("shipped_fit") or {}).get("train_patients"),
            "measured": None if not man else {
                "auroc": hold.get("auroc"), "ece": hold.get("ece"),
                "hour_matched": sc.get("detector", {}).get("hour_matched"),
                "patients": (wr or {}).get("patients"),
                "note": "cross-hospital AUROC 0.68–0.70 (trained at one hospital, scored at the other)",
                "source": "artifacts/detector_manifest.json, artifacts/ward_risk.json"},
            "alarm": None if not op else {"high": op["high"], "medium": op["medium"]},
            "conformal": None if not cf else {"alpha": cf.get("alpha")},
            "artifact": "artifacts/gbm_wearable.joblib",
        },
        {
            "code": "QR", "name": "Bedside rules", "kind": "support", "decides": False,
            "status": "retired",
            "serves": "every ward bed: names the derangements behind its risk (drivers, organ components)",
            "reads": "HR, temperature, respiratory rate, SpO₂, MAP, WBC, lactate",
            "method": "hand-weighted qSOFA/SIRS logistic (kernel/fusion.py)",
            "measured": None if not sc else {
                "auroc": sc.get("rules", {}).get("auroc"), "hour_matched": sc.get("rules", {}).get("hour_matched"),
                "patients": (wr or {}).get("patients"),
                "note": "scored the ICU rooms' alarms until 2026-10-03; now explains only",
                "source": "artifacts/ward_risk.json"},
        },
        {
            "code": "BL", "name": "Baseline learner", "kind": "support", "decides": False, "status": "explains",
            "serves": "every ward bed: how far this patient is from their own baseline",
            "reads": "the bed's own recent vitals and labs",
            "method": "per-patient running mean and spread; sustained deviation",
            "measured": None if not sc else {
                "note": f"layered on V7 it lowered AUROC (0.800 → {sc.get('index_pers', {}).get('auroc', 0):.3f}), "
                        "so it is kept out of the alarm path",
                "source": "artifacts/ward_risk.json"},
        },
        {
            "code": "HM", "name": "Case memory", "kind": "support", "decides": False, "status": "explains",
            "serves": "every ward bed: the closest real precedents",
            "reads": "the bed's vitals, labs and trends, encoded as a hypervector",
            "method": "hyperdimensional associative memory over real CinC patient-hours",
            "measured": {"auroc": 0.626, "note": "standalone retrieval, patient-level and leakage-free",
                         "source": "backend/aegis/verify_hdc_memory.py"},
        },
        {
            "code": "XG", "name": "Resonance gate", "kind": "support", "decides": False, "status": "explains",
            "serves": "every ward bed: whether a rise is seen by more than one sensor pathway",
            "reads": "camera, wearable and record pathways, each projected onto a sepsis axis",
            "method": "per-pathway HDC projection, median consensus, isolated-spike rejection",
            "measured": {"note": "removes 79% of camera-artifact false alarms on 32 held-out patients, short of its 80% bar",
                         "source": "backend/aegis/verify_resonance.py"},
        },
        {
            "code": "EW", "name": "Early-warning radar", "kind": "support", "decides": False, "status": "explains",
            "serves": "the scheduler: which beds get compute first",
            "reads": "variance and autocorrelation of each bed's recent window",
            "method": "critical-slowing-down indicators",
            "measured": {"note": "no anticipatory compute gain on held-out patients; guarded against starving a bed",
                         "source": "backend/aegis/verify_predictive_sched.py"},
        },
        {
            "code": "IG", "name": "Identity guard", "kind": "support", "decides": False, "status": "explains",
            "serves": "the live stream: detects a swapped or reassigned sensor before a dose is released",
            "reads": "the stream's physiologic fingerprint",
            "method": "Mahalanobis distance from the patient's enrolled baseline",
            "measured": {"note": "tells a swapped stream from the same patient at AUROC ~0.88 on real CinC (EER ~0.18); "
                                 "not biometric identity",
                         "source": "backend/aegis/verify_identity.py"},
        },
        {
            "code": "PK", "name": "Drug twin", "kind": "support", "decides": False, "status": "explains",
            "serves": "pharmacy: projected drug levels and adverse events for each treatment path",
            "reads": "weight, renal function, MAP and the candidate orders",
            "method": "physiologically based pharmacokinetic model with cited adverse-event rates",
            "measured": {"note": "anchored to published PK parameters; no outcome data here to validate against, "
                                 "and it never gates an order",
                         "source": "backend/aegis/verify_pharmacokinetics.py"},
        },
        {
            "code": "LM", "name": (llm or {}).get("primary_model") or "Language model", "kind": "language", "decides": False,
            "status": "serving" if (llm or {}).get("available") else "unavailable",
            "serves": "triage review (may only raise a priority), SBAR briefs, the Dr. Aegis consult",
            "reads": "the page text and the bed's grounded dossier",
            "method": "multi-provider failover with cooldowns, a cache and a budget; deterministic fallback",
            "measured": {"note": "never lowers a priority and paging never waits on it",
                         "source": "backend/llm/verify_doctor.py, backend/aegis/paging/verify_paging.py"},
        },
    ]
    return {"models": models,
            "deciding": [m["code"] for m in models if m["decides"]],
            "note": "Only A40 and V7 can page anyone, each at its own validated operating point. "
                    "Every other model explains, routes compute or assists, and none of them can lower a page."}
