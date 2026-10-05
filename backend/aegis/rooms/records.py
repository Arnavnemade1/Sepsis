"""The record each room carries: who the patient is, what came before, what is ordered.

Three kinds of occupant, three honest provenances:

* **Synthetic rooms** (every room outside 301–312 and 402). The occupant from
  ``hospital.synthetic_patients`` is expanded into a longitudinal record: problems
  with the year they began, home medications, the inpatient orders with doses,
  allergies with the reaction, code status, earlier encounters and the vitals taken
  at each, and recent labs. Everything is generated from a fixed per-room seed and
  marked ``synthetic``. The personal baseline is the mean of the vitals recorded at
  those earlier encounters, so a detector that compares against it is reading the
  record, not a constant.
* **ICU rooms 301–312** replay real, de-identified PhysioNet/CinC 2019 stays. Their
  record is the real stay up to the current replay hour: demographics, unit,
  hospital admission time and every lab and vital measured so far. Nothing after the
  current hour is read, and the challenge's sepsis label is never read at all.
  CinC carries no prior history or medications, and the record says so rather than
  inventing them; a clinician can link an external record (FHIR) to fill the gap.
* **Room 402** is the live monitored session, with the history the session holds.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Any, Dict, Iterable, List, Optional

from backend.aegis.rooms.formulary import FORMULARY, NON_DRUG_ORDERS

DAY = 86400.0

# --------------------------------------------------------------------------- #
# What a problem implies, for detectors and for the record's history
# --------------------------------------------------------------------------- #
# keyword -> flags it sets
_PROBLEM_FLAGS = [
    ("copd", {"copd"}), ("heart failure", {"hf"}), ("ckd", {"ckd"}), ("atrial fibrillation", {"af"}),
    ("cad", {"cad"}), ("type 2 diabetes", {"diabetes"}), ("type 1 diabetes", {"diabetes", "t1dm"}),
    ("lymphoma on chemotherapy", {"chemo", "immunocompromised"}), ("colon cancer", {"cancer"}),
    ("dementia", {"dementia"}), ("osteoporosis", {"osteoporosis"}), ("epilepsy", {"epilepsy"}),
    ("hypertension", {"hypertension"}), ("obesity", {"obesity"}), ("smoker", {"smoker"}),
    ("prior abdominal surgery", {"abdominal_surgery"}), ("hyperlipidemia", set()),
]
_DX_FLAGS = [
    ("pneumonia", {"infection"}), ("cellulitis", {"infection"}), ("pyelonephritis", {"infection"}),
    ("urinary tract infection", {"infection"}), ("febrile neutropenia", {"infection", "neutropenia"}),
    ("post-op", {"post_op"}), ("nstemi", {"acs"}), ("rule out acs", {"acs_ro"}),
    ("atrial fibrillation", {"af"}), ("heart failure", {"hf"}), ("stroke", {"stroke"}),
    ("hip fracture", {"fall_history", "post_op_soon"}), ("diabetic ketoacidosis", {"dka"}),
    ("seizure", {"epilepsy"}), ("small bowel obstruction", {"npo"}), ("appendicitis", {"npo"}),
    ("syncope", {"fall_risk_event"}),
]

# chronic problem -> (prior encounter reason, setting, length of stay in days or None)
_PRIOR_ENCOUNTER = {
    "copd": ("COPD exacerbation", "Inpatient", 4),
    "hf": ("Heart failure exacerbation", "Inpatient", 5),
    "ckd": ("Nephrology clinic review", "Clinic", None),
    "diabetes": ("Diabetes review", "Primary care", None),
    "af": ("Atrial fibrillation, rate control", "Cardiology clinic", None),
    "cad": ("Elective PCI to the LAD", "Inpatient", 2),
    "chemo": ("Chemotherapy, cycle 3", "Oncology day unit", None),
    "dementia": ("Fall at home, no fracture", "Emergency", None),
    "osteoporosis": ("Bone density scan", "Clinic", None),
    "epilepsy": ("Seizure review", "Neurology clinic", None),
    "hypertension": ("Annual review", "Primary care", None),
    "cancer": ("Oncology follow-up", "Clinic", None),
    "abdominal_surgery": ("Open appendectomy", "Inpatient", 3),
}

# chronic problem -> home medications (name, dose, frequency)
_HOME_MEDS = {
    "hypertension": [("lisinopril", "10 mg", "daily")],
    "diabetes": [("metformin", "1000 mg", "BID")],
    "t1dm": [("insulin glargine", "20 units", "nightly"), ("insulin lispro", "sliding scale", "with meals")],
    "hf": [("carvedilol", "6.25 mg", "BID"), ("furosemide", "40 mg", "daily")],
    "af": [("apixaban", "5 mg", "BID"), ("metoprolol", "25 mg", "BID")],
    "cad": [("aspirin", "81 mg", "daily"), ("atorvastatin", "40 mg", "nightly")],
    "copd": [("tiotropium", "18 mcg inhaled", "daily"), ("albuterol", "2 puffs", "PRN")],
    "epilepsy": [("levetiracetam", "500 mg", "BID")],
    "dementia": [("donepezil", "10 mg", "nightly")],
    "osteoporosis": [("alendronate", "70 mg", "weekly")],
}

_BETA_BLOCKERS = {"carvedilol", "metoprolol"}

# admission diagnosis -> what the illness does to the vitals today (offsets from baseline)
_ADMISSION_STATE = {
    "pneumonia": {"temp_c": 0.6, "hr": 6, "rr": 1, "spo2": -2, "o2": 2.0},
    "cellulitis": {"temp_c": 0.6, "hr": 6},
    "pyelonephritis": {"temp_c": 0.9, "hr": 8, "rr": 1},
    "post-op day 1": {"hr": 6, "temp_c": 0.3},
    "post-op day 0": {"hr": 8, "sbp": -6},
    "nstemi": {"hr": 4},
    "atrial fibrillation": {"hr": 14},
    "decompensated heart failure": {"rr": 2, "spo2": -1, "hr": 4},
    "ischemic stroke": {"sbp": 14},
    "febrile neutropenia": {"temp_c": 0.6, "hr": 10},
    "small bowel obstruction": {"hr": 8},
    "chest pain": {"hr": 6, "sbp": 6},
    "abdominal pain": {"hr": 10},
    "hip fracture": {"hr": 9},
    "diabetic ketoacidosis": {"hr": 14, "rr": 5},
}

_PLANNED = {
    "nstemi": "Cardiac catheterization",
    "rule out acs": "Exercise stress test",
    "hip fracture": "Hip hemiarthroplasty",
    "appendicitis": "CT abdomen with contrast",
    "small bowel obstruction": "General surgery review",
}

_REACTIONS = {
    "penicillin": [("anaphylaxis", "severe"), ("hives", "moderate"), ("rash", "mild")],
    "sulfa": [("rash", "mild"), ("Stevens-Johnson syndrome", "severe")],
    "iodinated contrast": [("hives", "moderate"), ("bronchospasm", "severe")],
    "vancomycin (red man)": [("flushing with rapid infusion", "mild")],
}


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _flags(problems: Iterable[str], dx: str) -> set:
    out: set = set()
    for p in problems:
        pl = p.lower()
        for kw, fl in _PROBLEM_FLAGS:
            if kw in pl:
                out |= fl
    dl = (dx or "").lower()
    for kw, fl in _DX_FLAGS:
        if kw in dl:
            out |= fl
    return out


_ENCOUNTER_SPREAD = {"hr": 4.0, "sbp": 6.0, "dbp": 4.0, "rr": 1.0, "spo2": 0.8, "temp_c": 0.2}


def _encounter_vitals(base: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    """One set of vitals as an earlier encounter recorded it: the baseline, give or take a visit's variation."""
    v = {k: base[k] + rng.uniform(-d, d) for k, d in _ENCOUNTER_SPREAD.items()}
    v["spo2"] = min(100.0, v["spo2"])
    return {k: round(x, 1) if k == "temp_c" else float(round(x)) for k, x in v.items()}


def _admission_offsets(dx: str) -> Dict[str, float]:
    dl = dx.lower()
    for kw, off in _ADMISSION_STATE.items():
        if kw in dl:
            return dict(off)
    return {}


# --------------------------------------------------------------------------- #
# Synthetic occupants
# --------------------------------------------------------------------------- #
def synthetic_record(p: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    """Expand a synthetic occupant into a longitudinal record (deterministic per room)."""
    now = time.time() if now is None else now
    rng = random.Random(_seed("record", p["room"], p["mrn"]))
    age, sex, dx = int(p["age"]), p["sex"], p["diagnosis"]
    problems_in: List[str] = list(p.get("history") or [])
    flags = _flags(problems_in, dx)

    orders: List[Dict[str, Any]] = []
    admitted_at = now - p.get("admitted_days", 0) * DAY - rng.uniform(2, 20) * 3600
    for name in p.get("medications") or []:
        d = FORMULARY.get(name)
        if d is None:
            orders.append({"name": name, "detail": NON_DRUG_ORDERS.get(name, name), "kind": "order",
                           "status": "active", "classes": [], "started_at": admitted_at})
            continue
        started = admitted_at + rng.uniform(0.2, 3) * 3600
        last = None if d.prn and rng.random() < 0.5 else now - rng.uniform(0.3, 5.5) * 3600
        orders.append({"name": d.name, "generic": d.generic, "dose": d.dose, "route": d.route, "freq": d.freq,
                       "classes": sorted(d.classes), "prn": d.prn, "infusion": d.infusion, "renal": d.renal,
                       "qt": d.qt, "kind": "drug", "status": "active", "started_at": started,
                       "last_given_at": max(started, last) if last else None})
    for o in orders:
        if "opioid" in o.get("classes", []):
            flags.add("opioid")
        if "benzodiazepine" in o.get("classes", []):
            flags.add("benzo")
        if "anticoagulant" in o.get("classes", []):
            flags.add("anticoagulant")
        if "antiplatelet" in o.get("classes", []):
            flags.add("antiplatelet")
        if "insulin" in o.get("classes", []):
            flags.add("insulin")
        if "beta_blocker" in o.get("classes", []) and not o.get("prn"):
            flags.add("beta_blocker")

    allergies = []
    for a in p.get("allergies") or []:
        reaction, sev = rng.choice(_REACTIONS.get(a, [("unknown", "unknown")]))
        allergies.append({"substance": a, "reaction": reaction, "severity": sev,
                          "kind": "intolerance" if "red man" in a else "allergy",
                          "recorded": f"{rng.randint(2006, 2024)}"})

    # -- baseline: what earlier encounters measured --------------------------------
    base = {"hr": 72.0 + rng.uniform(-6, 8), "sbp": 124.0 + rng.uniform(-8, 10), "rr": 15.0 + rng.uniform(-1, 2),
            "spo2": 97.0 + rng.uniform(-1, 1), "temp_c": 36.7 + rng.uniform(-0.2, 0.2)}
    if "hypertension" in flags:
        base["sbp"] += rng.uniform(16, 30)
    if "hf" in flags:
        base["sbp"] = 110 + rng.uniform(-3, 8)
        base["hr"] = 76 + rng.uniform(-4, 6)
        base["spo2"] = 96 + rng.uniform(-0.5, 1)
    if "af" in flags:
        base["hr"] = 80 + rng.uniform(-4, 8)
    if "copd" in flags:
        base["spo2"] = 91 + rng.uniform(-1.5, 1)
        base["rr"] = 16 + rng.uniform(-1, 1)
    if age >= 80:
        base["sbp"] += 6
    base["dbp"] = base["sbp"] * 0.6 + rng.uniform(-4, 4)
    mental = "confused at baseline" if "dementia" in flags else "alert"

    encounters: List[Dict[str, Any]] = []
    for fl in sorted(flags):
        if fl in _PRIOR_ENCOUNTER and len(encounters) < 3:
            reason, setting, los = _PRIOR_ENCOUNTER[fl]
            at = now - rng.uniform(70, 1100) * DAY
            v = _encounter_vitals(base, rng)
            note = None
            if fl == "copd":
                note = "ABG: PaCO₂ 6.9 kPa (chronic CO₂ retention). Discharged on a target SpO₂ of 88–92%."
            elif fl == "hf":
                note = "Echo: LVEF reduced. Discharge weight recorded."
            elif fl == "dementia":
                note = "Baseline: oriented to person only; mobilises with a walker."
            elif fl == "diabetes":
                note = f"HbA1c {rng.uniform(6.8, 8.9):.1f}%."
            encounters.append({"at": at, "setting": setting, "reason": reason, "los_days": los, "vitals": v,
                               "note": note})
    if not encounters:
        at = now - rng.uniform(120, 900) * DAY
        v = _encounter_vitals(base, rng)
        encounters.append({"at": at, "setting": "Primary care", "reason": "Annual review", "los_days": None,
                           "vitals": v, "note": None})
    encounters.sort(key=lambda e: -e["at"])
    measured = {k: round(sum(e["vitals"][k] for e in encounters) / len(encounters), 1)
                for k in ("hr", "sbp", "dbp", "rr", "spo2", "temp_c")}
    baseline = {**measured, "mental": mental, "n": len(encounters),
                "source": f"mean of {len(encounters)} earlier encounter{'s' if len(encounters) != 1 else ''}"}

    # every COPD patient here has a documented target: without one the ward would score
    # chronic hypoxaemia as an emergency on every observation (see the NEWS2 detector)
    hypercapnic = "copd" in flags
    if hypercapnic:
        flags.add("hypercapnic")
    spo2_target = ({"lo": 88, "hi": 92, "scale": 2,
                    "why": "COPD with documented chronic CO₂ retention; target prescribed at the last admission"}
                   if hypercapnic else None)

    problems = []
    for prob in problems_in:
        problems.append({"name": prob, "status": "chronic", "since": str(rng.randint(2004, 2024))})
    problems.insert(0, {"name": dx, "status": "active", "since": "this admission"})

    home = []
    seen = set()
    for fl in sorted(flags):
        for name, dose, freq in _HOME_MEDS.get(fl, []):
            if name in seen or (name in _BETA_BLOCKERS and seen & _BETA_BLOCKERS):
                continue                          # one beta-blocker at home, not two
            seen.add(name)
            home.append({"name": name, "dose": dose, "freq": freq})

    labs = _synthetic_labs(rng, flags, dx, now)
    planned = [v for k, v in _PLANNED.items() if k in dx.lower()]
    mobility = ("walker" if ("dementia" in flags or age >= 82) else
                "bed rest" if "post_op_soon" in flags else
                "assist of one" if "post_op" in flags else "independent")
    weight = round(rng.uniform(52, 82) if sex == "F" else rng.uniform(64, 104), 0)

    return {
        "room": p["room"], "kind": "synthetic",
        "provenance": {"synthetic": True, "label": "Synthetic demo record",
                       "detail": "Generated from a fixed seed for the demo hospital. Not a real person."},
        "patient": {"name": p["name"], "age": age, "sex": sex, "mrn": p["mrn"], "weight_kg": weight},
        "admission": {"reason": dx, "admitted_at": admitted_at, "day": int((now - admitted_at) // DAY) + 1},
        "problems": problems, "orders": orders, "home_meds": home, "allergies": allergies,
        "code_status": p.get("code_status") or "Full code", "encounters": encounters,
        "baseline": baseline, "spo2_target": spo2_target, "planned": planned, "labs": labs,
        "mobility": mobility, "flags": sorted(flags), "linked": [],
        "admission_state": _admission_offsets(dx),
    }


def _synthetic_labs(rng: random.Random, flags: set, dx: str, now: float) -> Dict[str, List[Dict[str, float]]]:
    """Recent labs: this morning's draw and yesterday's, plus the creatinine baseline."""
    am, yday = now - rng.uniform(4, 9) * 3600, now - rng.uniform(26, 31) * 3600
    cr0 = rng.uniform(1.5, 1.9) if "ckd" in flags else rng.uniform(0.7, 1.05)
    hb = rng.uniform(11, 14.2)
    # white count: an infection raises it, unless the marrow cannot (febrile neutropenia sits near its
    # nadir, below the ANC it carries); one draw per entry either way, so no other record shifts
    wbc_y = (0.8, 2.2) if "neutropenia" in flags else (12.5, 16) if "infection" in flags else (5.5, 9.5)
    wbc_am = (0.5, 1.8) if "neutropenia" in flags else (11.5, 15) if "infection" in flags else (5.5, 9.5)
    labs: Dict[str, List[Dict[str, float]]] = {
        "creatinine": [{"t": now - rng.uniform(120, 600) * DAY, "v": round(cr0, 2), "baseline": True},
                       {"t": yday, "v": round(cr0 * rng.uniform(0.98, 1.08), 2)},
                       {"t": am, "v": round(cr0 * rng.uniform(1.0, 1.1), 2)}],
        "wbc": [{"t": yday, "v": round(rng.uniform(*wbc_y), 1)},
                {"t": am, "v": round(rng.uniform(*wbc_am), 1)}],
        "hemoglobin": [{"t": yday, "v": round(hb, 1)}, {"t": am, "v": round(hb * rng.uniform(0.96, 1.02), 1)}],
        "platelets": [{"t": am, "v": round(rng.uniform(55, 110) if "chemo" in flags else rng.uniform(170, 320))}],
        "potassium": [{"t": yday, "v": round(rng.uniform(4.6, 5.2) if "ckd" in flags else rng.uniform(3.7, 4.6), 1)},
                      {"t": am, "v": round(rng.uniform(4.7, 5.3) if "ckd" in flags else rng.uniform(3.7, 4.6), 1)}],
    }
    if "diabetes" in flags or "dka" in flags:
        labs["glucose"] = [{"t": am, "v": round(rng.uniform(150, 230) if "dka" not in flags else rng.uniform(190, 260))}]
    if "infection" in flags:
        labs["lactate"] = [{"t": am, "v": round(rng.uniform(1.2, 1.8), 1)}]
    if "neutropenia" in flags:
        labs["anc"] = [{"t": am, "v": round(rng.uniform(250, 450))}]
    for k in labs:
        labs[k].sort(key=lambda r: r["t"])
    return labs


# --------------------------------------------------------------------------- #
# ICU replay rooms: the real stay up to the current hour
# --------------------------------------------------------------------------- #
# CinC column -> (record key, unit)
CINC_LABS = {
    "Creatinine": ("creatinine", "mg/dL"), "BUN": ("bun", "mg/dL"), "WBC": ("wbc", "×10⁹/L"),
    "Platelets": ("platelets", "×10⁹/L"), "Hgb": ("hemoglobin", "g/dL"), "Potassium": ("potassium", "mmol/L"),
    "Glucose": ("glucose", "mg/dL"), "Lactate": ("lactate", "mmol/L"), "Bilirubin_total": ("bilirubin", "mg/dL"),
    "FiO2": ("fio2", "fraction"), "pH": ("ph", ""), "PaCO2": ("paco2", "mmHg"), "HCO3": ("hco3", "mmol/L"),
    "TroponinI": ("troponin", "ng/mL"), "PTT": ("ptt", "s"), "Fibrinogen": ("fibrinogen", "mg/dL"),
}
CINC_VITALS = {"HR": "hr", "O2Sat": "spo2", "Temp": "temp_c", "SBP": "sbp", "MAP": "map", "DBP": "dbp", "Resp": "rr"}
# the challenge label must never reach the room; listed here so the firewall test can name it
FORBIDDEN_COLUMNS = ("SepsisLabel",)


def _num(x: Any) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def stay_to_date(frame: Any, hour: int) -> Dict[str, Any]:
    """Every measured lab and vital in the real stay up to and including ``hour``.

    Reads only the whitelisted columns, never a row after ``hour``: a record that
    peeked at later hours would hand the detectors the future."""
    hour = max(0, int(hour))
    rows = frame.iloc[: hour + 1]
    labs: Dict[str, List[Dict[str, float]]] = {}
    for col, (key, _unit) in CINC_LABS.items():
        if col not in rows.columns:
            continue
        for h, val in enumerate(rows[col].tolist()):
            v = _num(val)
            if v is not None:
                labs.setdefault(key, []).append({"t": float(h), "v": round(v, 2)})
    vitals: List[Dict[str, Any]] = []
    for h in range(len(rows)):
        r = rows.iloc[h]
        v = {key: _num(r.get(col)) for col, key in CINC_VITALS.items()}
        if any(x is not None for x in v.values()):
            vitals.append({"t": float(h), **{k: round(x, 1) for k, x in v.items() if x is not None}})
    return {"labs": labs, "vitals": vitals, "hours": len(rows)}


def icu_record(room: str, mon: Dict[str, Any], patient: Any, imported: Optional[Dict[str, Any]],
               linked_label: Optional[str] = None) -> Dict[str, Any]:
    """The record of a replayed CinC stay, built only from what the stay has measured so far."""
    hour = int(mon.get("icu_hour") or 0)
    stay = stay_to_date(patient.record.frame, hour) if patient is not None else {"labs": {}, "vitals": [], "hours": 0}
    r0 = patient.record.frame.iloc[0] if patient is not None else {}
    unit = "MICU" if _num(r0.get("Unit1")) == 1 else "SICU" if _num(r0.get("Unit2")) == 1 else "ICU"
    hosp_adm = _num(r0.get("HospAdmTime"))
    flags: set = set()
    problems: List[Dict[str, Any]] = [{"name": "ICU stay", "status": "active", "since": "this stay",
                                        "note": f"recorded unit {unit} · PhysioNet/CinC 2019"}]
    cr = [x["v"] for x in stay["labs"].get("creatinine", [])]
    if cr and max(cr) > 1.5:
        flags.add("renal_impairment")
        problems.append({"name": "Renal impairment", "status": "measured", "since": "this stay",
                         "note": f"creatinine up to {max(cr):.1f} mg/dL so far"})
    plt = [x["v"] for x in stay["labs"].get("platelets", [])]
    if plt and min(plt) < 100:
        problems.append({"name": "Thrombocytopenia", "status": "measured", "since": "this stay",
                         "note": f"platelets down to {min(plt):.0f} so far"})
    linked = []
    allergies: List[Dict[str, Any]] = []
    home: List[Dict[str, Any]] = []
    if imported:
        for c in imported.get("comorbidities") or []:
            problems.append({"name": c, "status": "imported", "since": "external record"})
            flags |= _flags([c], "")
        for a in imported.get("allergies") or []:
            allergies.append({"substance": a, "reaction": "recorded externally", "severity": "unknown", "kind": "allergy"})
        for m in imported.get("medications") or []:
            home.append({"name": m, "dose": "", "freq": ""})
        linked.append({"source": imported.get("source"), "label": linked_label or imported.get("source"),
                       "imported_at": imported.get("imported"), "summary": imported.get("summary")})
    return {
        "room": room, "kind": "replay",
        "provenance": {"synthetic": False, "label": "Replayed real record",
                       "detail": "PhysioNet/CinC 2019, de-identified. Shown up to the current replay hour; "
                                 "the dataset has no prior history, medications or code status."},
        "patient": {"name": mon.get("name"), "age": mon.get("age"), "sex": mon.get("sex"), "mrn": mon.get("mrn"),
                    "bed": mon.get("bed")},
        # what the stay itself shows so far; the source is on the provenance, and the board
        # already marks the room as a real replay
        "admission": {"reason": ", ".join(["ICU stay"] + [p["name"].lower() for p in problems[1:3]]),
                      "icu_hour": hour, "unit": unit,
                      "hospital_admit_h": hosp_adm,
                      "day": hour // 24 + 1},
        "problems": problems, "orders": [], "home_meds": home, "allergies": allergies,
        "code_status": None, "encounters": [], "baseline": None, "spo2_target": None, "planned": [],
        "labs": stay["labs"], "replay_vitals": stay["vitals"], "mobility": None, "flags": sorted(flags),
        "linked": linked, "missing": ["prior encounters", "medications", "code status"],
    }


# --------------------------------------------------------------------------- #
# The live session
# --------------------------------------------------------------------------- #
def live_record(room: str, mon: Dict[str, Any], history: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    h = history or {}
    problems = [{"name": "Live monitored session", "status": "active", "since": "now"}]
    flags: set = set()
    if h.get("chf"):
        problems.append({"name": "Heart failure", "status": "chronic", "since": "session record"})
        flags.add("hf")
    if h.get("renal_failure"):
        problems.append({"name": "Renal failure", "status": "chronic", "since": "session record"})
        flags.add("ckd")
    allergies = [{"substance": "penicillin", "reaction": "recorded in the session", "severity": "unknown",
                  "kind": "allergy"}] if h.get("penicillin_allergy") else []
    labs = {}
    if h.get("wbc") is not None:
        labs["wbc"] = [{"t": time.time(), "v": float(h["wbc"])}]
    return {
        "room": room, "kind": "live",
        "provenance": {"synthetic": False, "label": "Live session",
                       "detail": "Camera, wearable or the simulator feeding the calibrated detector."},
        "patient": {"name": mon.get("name"), "age": mon.get("age"), "sex": mon.get("sex"), "mrn": mon.get("mrn")},
        "admission": {"reason": "Live monitored session"},
        "problems": problems, "orders": [], "home_meds": [], "allergies": allergies, "code_status": None,
        "encounters": [], "baseline": None, "spo2_target": None, "planned": [], "labs": labs,
        "mobility": None, "flags": sorted(flags), "linked": [], "missing": ["medications", "code status"],
    }


# --------------------------------------------------------------------------- #
# Interface-fed rooms: what the hospital's ADT feed told us, and nothing more
# --------------------------------------------------------------------------- #
def interface_record(room: str, p: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    """The record of a patient admitted over HL7 (``backend/aegis/interop``).

    Only what the feed carried: demographics (PID), the admitting and secondary
    diagnoses (DG1), allergies (AL1), code status and weight (OBX). Problems are read
    for the same context flags a synthetic record gets, so the detectors read a COPD
    patient's saturations on scale 2 whichever way the patient arrived. What no feed
    has sent (medications without a pharmacy interface, earlier encounters, a measured
    baseline) is listed as missing rather than invented."""
    now = time.time() if now is None else now
    dx_list = [d for d in (p.get("diagnoses") or []) if d]
    reason = dx_list[0] if dx_list else "Admission (no diagnosis on the feed)"
    problems = [{"name": reason, "status": "active", "since": "this admission"}]
    problems += [{"name": d, "status": "chronic", "since": "ADT feed"} for d in dx_list[1:]]
    flags = _flags(dx_list[1:], reason)
    hypercapnic = "copd" in flags
    if hypercapnic:
        flags.add("hypercapnic")
    allergies = [{"substance": a.get("substance", ""), "reaction": a.get("reaction") or "not stated",
                  "severity": a.get("severity") or "unknown", "kind": "allergy", "recorded": "ADT feed (AL1)"}
                 for a in p.get("allergies") or [] if a.get("substance")]
    admitted = p.get("admitted_at") or now
    simulated = bool(p.get("simulated"))
    feed = p.get("interface_name") or "an ADT interface"
    return {
        "room": room, "kind": "interface",
        "provenance": {"synthetic": simulated,
                       "label": "Interface feed" + (" (simulated hospital)" if simulated else ""),
                       "detail": f"Admitted by {feed} over HL7 (ADT^A01). Vitals and results arrive as HL7 ORU "
                                 "messages, each code read through a mapping the integration engine verified."
                                 + (" The sending systems are simulated; the patient is not a real person."
                                    if simulated else "")},
        "patient": {"name": p.get("name"), "age": p.get("age"), "sex": p.get("sex"), "mrn": p.get("mrn"),
                    "weight_kg": p.get("weight_kg")},
        "admission": {"reason": reason, "admitted_at": admitted, "day": int((now - admitted) // DAY) + 1,
                      "attending": p.get("attending") or None},
        "problems": problems, "orders": [], "home_meds": [], "allergies": allergies,
        "code_status": p.get("code_status"), "encounters": [], "baseline": None,
        "spo2_target": ({"lo": 88, "hi": 92, "scale": 2, "why": "COPD on the ADT feed's diagnoses"}
                        if hypercapnic else None),
        "planned": [], "labs": {}, "mobility": None, "flags": sorted(flags), "linked": [],
        "missing": ["medications (no pharmacy interface)", "earlier encounters", "a measured baseline"],
        "interface": {"name": feed, "mrn": p.get("mrn")},
    }
