"""The hospital the paging OS runs: floors, units, rooms, patients and staff.

Everything here is a DEMO hospital ("Aegis Memorial") with one exception that
matters: the twelve ICU rooms 301–312 hold the real, de-identified PhysioNet/
CinC 2019 stays the cohort cluster replays, and room 402 holds the live
monitored session (camera or wearable, or the simulator when neither is
connected). Every other occupant is a synthetic record, generated from a fixed
seed and marked ``synthetic`` so no screen can present it as a real chart.

Geometry is in plan units on a 100 × 40 floor plate (x across, y deep). The
frontend projects it isometrically; the backend uses it only for proximity.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

HOSPITAL_NAME = "Aegis Memorial"
HOSPITAL_NOTE = ("Demo hospital. ICU rooms 301–312 replay real de-identified PhysioNet/CinC 2019 "
                 "stays; room 402 is the live monitored session; every other patient is synthetic.")

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
# (floor id, label, [(zone id, name, kind, (x, y, w, h), room names or None)])
# kind: ed | icu | stepdown | ward | or | pacu | obs | support | imaging
_LAYOUT: List[Tuple[str, str, List[Tuple[str, str, str, Tuple[int, int, int, int], Optional[List[str]]]]]] = [
    ("A", "Emergency", [
        ("ed", "Emergency Department", "ed", (0, 0, 58, 40), [f"A{i}" for i in range(1, 11)]),
        ("triage", "Triage", "ed", (58, 0, 22, 40), ["T1", "T2", "T3", "T4"]),
        ("imaging", "Imaging", "imaging", (80, 0, 20, 40), None),
    ]),
    ("1", "Admissions", [
        ("obs", "Observation Unit", "obs", (0, 0, 56, 40), [str(101 + i) for i in range(10)]),
        ("pharmacy", "Central Pharmacy", "support", (56, 0, 24, 40), None),
        ("admitting", "Admitting", "support", (80, 0, 20, 40), None),
    ]),
    ("2", "Surgery", [
        ("or", "Operating Rooms", "or", (0, 0, 48, 40), ["OR1", "OR2", "OR3", "OR4"]),
        ("pacu", "Recovery (PACU)", "pacu", (48, 0, 52, 40), [str(201 + i) for i in range(8)]),
    ]),
    ("3", "Critical Care", [
        ("micu", "Medical ICU", "icu", (0, 0, 44, 40), [str(301 + i) for i in range(6)]),
        ("station3", "ICU Station", "support", (44, 0, 14, 40), None),
        ("sicu", "Surgical ICU", "icu", (58, 0, 42, 40), [str(307 + i) for i in range(6)]),
    ]),
    ("4", "Cardiac", [
        ("stepdown", "Step-down", "stepdown", (0, 0, 50, 40), [str(401 + i) for i in range(8)]),
        ("tele", "Cardiac Telemetry", "stepdown", (50, 0, 50, 40), [str(409 + i) for i in range(8)]),
    ]),
    ("5", "Medical–Surgical", [
        ("5e", "5 East", "ward", (0, 0, 50, 40), [str(501 + i) for i in range(10)]),
        ("5w", "5 West", "ward", (50, 0, 50, 40), [str(511 + i) for i in range(10)]),
    ]),
    ("6", "Neuro & Oncology", [
        ("neuro", "Neuroscience", "ward", (0, 0, 46, 40), [str(601 + i) for i in range(6)]),
        ("onc", "Oncology", "ward", (46, 0, 54, 40), [str(607 + i) for i in range(8)]),
    ]),
]
FLOOR_ORDER = [f[0] for f in _LAYOUT]
REMOTE_ZONE = "remote"                     # telehealth: on no floor


@dataclass
class Room:
    id: str
    zone: str
    floor: str
    rect: Tuple[float, float, float, float]


@dataclass
class Zone:
    id: str
    name: str
    kind: str
    floor: str
    rect: Tuple[int, int, int, int]
    rooms: List[str] = field(default_factory=list)


def _grid(n: int) -> Tuple[int, int]:
    """Rooms laid out in two rows along the zone (one row if only a few)."""
    rows = 2 if n >= 4 else 1
    return rows, -(-n // rows)


def build_layout() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Zone], Dict[str, Room]]:
    floors: Dict[str, Dict[str, Any]] = {}
    zones: Dict[str, Zone] = {}
    rooms: Dict[str, Room] = {}
    for idx, (fid, label, zs) in enumerate(_LAYOUT):
        floors[fid] = {"id": fid, "label": label, "index": idx, "zones": []}
        for zid, name, kind, rect, room_names in zs:
            z = Zone(zid, name, kind, fid, rect)
            if room_names:
                rows, cols = _grid(len(room_names))
                x0, y0, w, h = rect
                pad = 1.2
                cw, ch = (w - pad * 2) / cols, (h - pad * 2) / rows
                for i, rn in enumerate(room_names):
                    r, c = divmod(i, cols)
                    rooms[rn] = Room(rn, zid, fid, (round(x0 + pad + c * cw, 2), round(y0 + pad + r * ch, 2),
                                                    round(cw, 2), round(ch, 2)))
                    z.rooms.append(rn)
            zones[zid] = z
            floors[fid]["zones"].append(zid)
    zones[REMOTE_ZONE] = Zone(REMOTE_ZONE, "Telehealth (remote)", "remote", "", (0, 0, 0, 0))
    return floors, zones, rooms


FLOORS, ZONES, ROOMS = build_layout()

# Which zones count as critical care for policy purposes (autonomy, zone escalation).
CRITICAL_ZONES = {"micu", "sicu", "ed", "triage"}


def zone_of(room: Optional[str]) -> Optional[str]:
    r = ROOMS.get(room or "")
    return r.zone if r else None


def floor_distance(zone_a: Optional[str], zone_b: Optional[str]) -> int:
    """Floors between two zones (same zone 0 is handled by the caller). Remote
    staff are 'near' everywhere for consults and far for bedside work."""
    if not zone_a or not zone_b or zone_a == REMOTE_ZONE or zone_b == REMOTE_ZONE:
        return 99
    fa, fb = ZONES[zone_a].floor, ZONES[zone_b].floor
    return abs(FLOOR_ORDER.index(fa) - FLOOR_ORDER.index(fb))


# --------------------------------------------------------------------------- #
# Staff
# --------------------------------------------------------------------------- #
SPECIALTY_LABEL = {
    "critical_care": "Critical care", "hospitalist": "Hospital medicine", "emergency": "Emergency medicine",
    "cardiology": "Cardiology", "neurology": "Neurology", "surgery": "General surgery",
    "anesthesia": "Anesthesiology", "infectious_disease": "Infectious disease", "pharmacy": "Clinical pharmacy",
    "rapid_response": "Rapid response", "charge_nurse": "Charge nurse",
}


@dataclass
class Clinician:
    id: str                       # the kernel account's username
    name: str
    role: str                     # kernel RBAC role
    specialty: str
    home_zone: str
    title: str
    on_call: bool = False
    shift: str = "day"            # day | night | off
    status: str = "available"     # available | paging | in_procedure | on_case | off_shift
    zone: str = ""                # where they are now
    busy_until: Optional[float] = None
    pageable: bool = True         # the charge nurse runs the console; she is not paged

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["specialty_label"] = SPECIALTY_LABEL.get(self.specialty, self.specialty)
        return d


# id, name, role, specialty, home zone, title, on_call, shift, initial status
_ROSTER = [
    ("rn.patel", "Priya Patel, RN", "nurse", "charge_nurse", "station3", "Charge nurse · operator", False, "day", "available", False),
    ("dr.alvarez", "Dr. Sofia Alvarez", "attending", "critical_care", "micu", "Attending intensivist", True, "day", "available", True),
    ("dr.okafor", "Dr. Daniel Okafor", "attending", "critical_care", "sicu", "Intensivist", False, "day", "on_case", True),
    ("dr.whitfield", "Dr. James Whitfield", "chief", "critical_care", "station3", "Chief of critical care", False, "day", "available", True),
    ("dr.nakamura", "Dr. Kenji Nakamura", "attending", "hospitalist", "5e", "Hospitalist", True, "day", "available", True),
    ("dr.reyes", "Dr. Maria Reyes", "attending", "hospitalist", "5w", "Hospitalist", False, "day", "available", True),
    ("dr.shah", "Dr. Anika Shah", "attending", "emergency", "ed", "Emergency physician", True, "day", "on_case", True),
    ("dr.brennan", "Dr. Liam Brennan", "attending", "emergency", "ed", "Emergency physician", False, "night", "off_shift", True),
    ("dr.adeyemi", "Dr. Tolu Adeyemi", "attending", "cardiology", "tele", "Cardiologist", True, "day", "available", True),
    ("dr.kowalski", "Dr. Ewa Kowalski", "attending", "neurology", "neuro", "Neurologist", True, "day", "in_procedure", True),
    ("dr.moreau", "Dr. Julien Moreau", "attending", "surgery", "or", "General surgeon", True, "day", "in_procedure", True),
    ("dr.fischer", "Dr. Hannah Fischer", "attending", "anesthesia", "pacu", "Anesthesiologist", False, "day", "available", True),
    ("dr.haddad", "Dr. Rami Haddad", "attending", "infectious_disease", "obs", "Infectious disease", False, "day", "available", True),
    ("dr.lindqvist", "Dr. Anna Lindqvist", "remote_specialist", "infectious_disease", REMOTE_ZONE, "Remote ID consult", True, "day", "available", True),
    ("rn.ortiz", "Luis Ortiz, RN", "nurse", "rapid_response", "stepdown", "Rapid response nurse", True, "day", "available", True),
    ("rn.kim", "Grace Kim, RN", "nurse", "rapid_response", "5w", "Rapid response nurse", False, "day", "available", True),
    ("rx.chen", "Wei Chen, PharmD", "pharmacist", "pharmacy", "pharmacy", "Clinical pharmacist", True, "day", "available", True),
    ("rx.silva", "Ana Silva, PharmD", "pharmacist", "pharmacy", "station3", "ICU pharmacist", False, "day", "available", True),
]


def build_roster(now: float) -> Dict[str, Clinician]:
    rng = random.Random(7)
    out: Dict[str, Clinician] = {}
    for cid, name, role, spec, home, title, on_call, shift, status, pageable in _ROSTER:
        c = Clinician(cid, name, role, spec, home, title, on_call, shift, status, home, None, pageable)
        if shift == "off" or status == "off_shift":
            c.status, c.shift = "off_shift", "night" if shift == "night" else shift
        if status in ("in_procedure", "on_case"):
            c.busy_until = now + rng.uniform(4, 11) * 60
        out[cid] = c
    return out


# --------------------------------------------------------------------------- #
# Patients
# --------------------------------------------------------------------------- #
# ICU rooms map onto the cohort's replayed CinC beds, in order.
ICU_ROOMS = [str(301 + i) for i in range(12)]
LIVE_ROOM = "402"

_SURNAMES = ["Morgan", "Tanaka", "Silva", "Novak", "Hughes", "Ibrahim", "Lindgren", "Costa", "Walsh",
             "Petrov", "Mensah", "Duarte", "Kaur", "Olsen", "Rossi", "Byrne", "Haas", "Yilmaz",
             "Park", "Laurent", "Owusu", "Castillo", "Weber", "Ng", "Fontaine", "Abara", "Moreno", "Kessler",
             "Adler", "Barros", "Chaudhry", "Dalton", "Eriksen", "Farouk", "Galli", "Hale", "Iyer", "Jansen",
             "Keane", "Lozano", "Maddox", "Nair", "Ortega", "Pryce", "Quinlan", "Rahman", "Sato", "Tobin",
             "Uddin", "Vance", "Wolfe", "Xu", "Yoon", "Zamora", "Brandt", "Cole", "Dunn", "Ekwueme"]
# (admitting diagnosis, pmh, meds, zone kinds it fits)
_CASES = [
    ("Community-acquired pneumonia", ["COPD", "Hypertension"], ["ceftriaxone", "azithromycin", "albuterol"], {"ward", "obs"}),
    ("Cellulitis, left lower leg", ["Type 2 diabetes"], ["cefazolin", "metformin", "insulin glargine"], {"ward", "obs"}),
    ("Urinary tract infection, pyelonephritis", ["CKD stage 3"], ["ceftriaxone", "amlodipine"], {"ward", "obs"}),
    ("Post-op day 1, laparoscopic cholecystectomy", ["Obesity"], ["acetaminophen", "enoxaparin", "ondansetron"], {"pacu", "ward"}),
    ("Post-op day 0, colectomy", ["Colon cancer"], ["hydromorphone PCA", "piperacillin-tazobactam", "enoxaparin"], {"pacu"}),
    ("NSTEMI, awaiting catheterization", ["CAD", "Hyperlipidemia"], ["aspirin", "heparin infusion", "atorvastatin", "metoprolol"], {"stepdown"}),
    ("Atrial fibrillation with RVR", ["Heart failure (EF 35%)"], ["diltiazem", "apixaban", "furosemide"], {"stepdown"}),
    ("Decompensated heart failure", ["Heart failure (EF 25%)", "CKD stage 3"], ["furosemide infusion", "carvedilol", "spironolactone"], {"stepdown"}),
    ("Ischemic stroke, day 2", ["Atrial fibrillation", "Hypertension"], ["aspirin", "atorvastatin", "labetalol"], {"ward"}),
    ("Febrile neutropenia", ["Lymphoma on chemotherapy"], ["cefepime", "filgrastim", "ondansetron"], {"ward"}),
    ("Small bowel obstruction", ["Prior abdominal surgery"], ["NG decompression", "IV fluids", "ondansetron"], {"ward", "obs"}),
    ("Chest pain, rule out ACS", ["Hypertension", "Smoker"], ["aspirin", "nitroglycerin"], {"ed", "obs"}),
    ("Abdominal pain, query appendicitis", [], ["morphine", "ondansetron"], {"ed"}),
    ("Syncope", ["Hypertension"], ["IV fluids"], {"ed", "obs"}),
    ("Fall with hip fracture", ["Osteoporosis", "Dementia"], ["acetaminophen", "enoxaparin"], {"ed", "ward"}),
    ("Diabetic ketoacidosis", ["Type 1 diabetes"], ["insulin infusion", "IV fluids", "potassium chloride"], {"ed", "stepdown"}),
    ("Seizure, known epilepsy", ["Epilepsy"], ["levetiracetam", "lorazepam PRN"], {"ward", "ed"}),
]
_ALLERGIES = [[], [], [], ["penicillin"], ["sulfa"], ["iodinated contrast"], ["vancomycin (red man)"], []]
_CODE = ["Full code", "Full code", "Full code", "DNR/DNI", "Full code"]


def synthetic_patients() -> Dict[str, Dict[str, Any]]:
    """Occupants for the non-ICU rooms, from a fixed seed so every boot is the same
    ward. About 60% of rooms are occupied."""
    rng = random.Random(2019)
    out: Dict[str, Dict[str, Any]] = {}
    names = list(_SURNAMES)
    rng.shuffle(names)
    for rid, room in ROOMS.items():
        zone = ZONES[room.zone]
        if rid in ICU_ROOMS or rid == LIVE_ROOM or zone.kind in ("or", "imaging", "support"):
            continue
        if rng.random() > 0.62:
            continue
        fits = [c for c in _CASES if zone.kind in c[3]] or _CASES
        dx, pmh, meds, _ = rng.choice(fits)
        sex = rng.choice(["F", "M"])
        name = f"{names.pop()}, {rng.choice('ABCDEGHJKLMNRST')}."
        out[rid] = {
            "room": rid, "name": name, "age": rng.randint(24, 91), "sex": sex,
            "mrn": f"S{rng.randint(100000, 999999)}", "diagnosis": dx, "history": pmh,
            "medications": meds, "allergies": rng.choice(_ALLERGIES), "code_status": rng.choice(_CODE),
            "admitted_days": rng.randint(0, 6), "source": "synthetic demo record", "synthetic": True,
            "monitored": False,
        }
    return out


def layout_payload() -> Dict[str, Any]:
    """The static building the frontend draws (sent once, not on every tick)."""
    return {
        "name": HOSPITAL_NAME, "note": HOSPITAL_NOTE, "plate": {"w": 100, "d": 40},
        "floors": [{**FLOORS[f], "zones": FLOORS[f]["zones"]} for f in FLOOR_ORDER],
        "zones": {zid: {"id": z.id, "name": z.name, "kind": z.kind, "floor": z.floor, "rect": list(z.rect),
                        "rooms": z.rooms, "critical": zid in CRITICAL_ZONES} for zid, z in ZONES.items()},
        "rooms": {rid: {"id": r.id, "zone": r.zone, "floor": r.floor, "rect": list(r.rect)} for rid, r in ROOMS.items()},
        "specialties": SPECIALTY_LABEL,
    }
