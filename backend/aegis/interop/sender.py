"""The other side of the interface: a simulated hospital's systems, speaking HL7 v2.

AegisOS's integration plane has to be exercised by something, and a real hospital's
feeds cannot be put in a public demo. These are four simulated systems with the habits
real ones have, and they talk to the engine ONLY in HL7 text, through the same entry
point as a real interface engine (``Integration.ingest_hl7``):

* **Epic ADT** (``EPICADT``): admits a few patients into empty rooms, updates them,
  discharges them and admits new ones. PID, PV1, AL1, DG1, a code-status OBX.
* **Philips IntelliBridge** (``IBRIDGE``): monitor vitals every few seconds with IEEE
  11073 codes (``147842^MDC_ECG_HEART_RATE^MDC``), temperature in °F, cuff pressure
  every cycle; and ORU^R40 alarms (IHE PCD-04 subset) when a patient crosses a limit.
* **Sunquest laboratory** (``SQLAB``): local mnemonics in SI units, the way a lab with
  a European-trained configuration writes them (creatinine µmol/L, glucose mmol/L,
  haemoglobin g/L, haematocrit L/L, urea mmol/L), plus two analytes AegisOS has no
  concept for (procalcitonin, anion gap) and a high-sensitivity troponin in ng/L.
* **Reference laboratory** (``NORDLAB``): send-outs coded the Nordic way (``P-Kreat``,
  ``B-Natr``, ``B-Lkc``, ``B-Hb`` in mmol/L). No lexical rule recognises them; they are
  what the language-model proposer is for.

Each patient's physiology runs here, on the sender's side (``rooms.physiology``):
AegisOS sees the patient only through what the feeds send. The simulated courses
(sepsis, a bleed...) therefore reach the room monitor the way they would in a real
hospital, through mapped vitals and results.

Fault drills reproduce interface failures that happen in real hospitals: a silent unit
change after an analyser swap, a relabelled unit, two fields swapped in a gateway, a
gateway resending a cached cuff reading, a temperature switched to Celsius under a
Fahrenheit label.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.aegis.interop import hl7
from backend.aegis.interop.engine import Integration, Interface

N_PATIENTS = 4
VITALS_S = 10.0
NIBP_S = 30.0
LAB_S = (240.0, 420.0)
STAY_S = (40 * 60.0, 70 * 60.0)
READMIT_S = (60.0, 150.0)
ALARM_COOLDOWN_S = 600.0

ADT = ("EPICADT", "AEGISMEM")
GW = ("IBRIDGE", "AEGISMEM")
LAB = ("SQLAB", "AEGISMEM")
REF = ("NORDLAB", "REFLAB")

# (code, text, system, concept key it really is, unit written, factor from canonical)
SQLAB_PANEL: List[Tuple[str, str, str, Optional[str], str, float]] = [
    ("NA", "Sodium", "L", "sodium", "mmol/L", 1.0),
    ("K", "Potassium", "L", "potassium", "mmol/L", 1.0),
    ("CL", "Chloride", "L", "chloride", "mmol/L", 1.0),
    ("CO2", "Carbon Dioxide, Total", "L", "hco3", "mmol/L", 1.0),
    ("UREA", "Urea", "L", "bun", "mmol/L", 1 / 2.801),
    ("CREA", "Creatinine", "L", "creatinine", "umol/L", 88.42),
    ("GLU", "Glucose", "L", "glucose", "mmol/L", 1 / 18.016),
    ("CA", "Calcium", "L", "calcium", "mmol/L", 1 / 4.008),
    ("MG", "Magnesium", "L", "magnesium", "mmol/L", 1 / 2.431),
    ("PHOS", "Phosphate", "L", "phosphate", "mmol/L", 1 / 3.097),
    ("WBC", "WBC Count", "L", "wbc", "10*9/L", 1.0),
    ("ANC", "Neutrophils, Absolute", "L", "anc", "10*9/L", 0.001),
    ("HGB", "Hemoglobin", "L", "hemoglobin", "g/L", 10.0),
    ("HCT", "Hematocrit", "L", "hematocrit", "L/L", 0.01),
    ("PLT", "Platelet Count", "L", "platelets", "10*9/L", 1.0),
    ("LACT", "Lactate", "L", "lactate", "mmol/L", 1.0),
    ("TBIL", "Bilirubin, Total", "L", "bilirubin", "umol/L", 17.104),
    ("TNIHS", "Troponin I, High Sensitivity", "L", "troponin", "ng/L", 1000.0),
    ("PCT", "Procalcitonin", "L", None, "ng/mL", 1.0),
    ("AGAP", "Anion Gap", "L", None, "mmol/L", 1.0),
]
NORDLAB_PANEL: List[Tuple[str, str, str, Optional[str], str, float]] = [
    ("RL101", "P-Kreat", "99RL", "creatinine", "umol/L", 88.42),
    ("RL102", "B-Natr", "99RL", "sodium", "mmol/L", 1.0),
    ("RL103", "B-Lkc", "99RL", "wbc", "10*9/L", 1.0),
    ("RL104", "B-Hb", "99RL", "hemoglobin", "mmol/L", 1 / 1.611),
]
# IEEE 11073 codes the gateway sends: (numeric code, reference id, concept, unit written)
GW_VITALS = [("147842", "MDC_ECG_HEART_RATE", "hr", "/min"), ("150456", "MDC_PULS_OXIM_SAT_O2", "spo2", "%"),
             ("151562", "MDC_RESP_RATE", "rr", "/min"), ("150364", "MDC_TEMP_BODY", "temp", "[degF]")]
GW_NIBP = [("150021", "MDC_PRESS_BLD_NONINV_SYS", "sbp"), ("150022", "MDC_PRESS_BLD_NONINV_DIA", "dbp"),
           ("150023", "MDC_PRESS_BLD_NONINV_MEAN", "map")]

DRILLS = {
    "glucose_units": ("Analyser swap: glucose now sent in mg/dL, still labelled mmol/L", "lab"),
    "creatinine_label": ("Creatinine relabelled mg/dL while the values stay µmol/L", "lab"),
    "hgb_scale": ("Haemoglobin sent in g/dL, still labelled g/L", "lab"),
    "bp_swap": ("Gateway swaps the cuff systolic and mean fields", "gateway"),
    "nibp_frozen": ("Gateway resends a cached cuff reading", "gateway"),
    "temp_celsius": ("Gateway switches temperature to °C, still labelled °F", "gateway"),
}

_ICD = {"Community-acquired pneumonia": "J18.9", "Cellulitis, left lower leg": "L03.116",
        "Urinary tract infection, pyelonephritis": "N10", "NSTEMI, awaiting catheterization": "I21.4",
        "Atrial fibrillation with RVR": "I48.91", "Decompensated heart failure": "I50.9",
        "Ischemic stroke, day 2": "I63.9", "Febrile neutropenia": "D70.9", "COPD": "J44.9",
        "Hypertension": "I10", "Type 2 diabetes": "E11.9", "CKD stage 3": "N18.30", "CAD": "I25.10",
        "Heart failure (EF 35%)": "I50.22", "Heart failure (EF 25%)": "I50.22", "Atrial fibrillation": "I48.91",
        "Lymphoma on chemotherapy": "C85.90", "Hyperlipidemia": "E78.5", "Smoker": "F17.210"}
_CASES = [
    ("Community-acquired pneumonia", ["COPD", "Hypertension"], ["penicillin"], "Full code"),
    ("Urinary tract infection, pyelonephritis", ["CKD stage 3", "Type 2 diabetes"], [], "Full code"),
    ("Decompensated heart failure", ["Heart failure (EF 25%)", "CKD stage 3"], ["sulfa"], "DNR/DNI"),
    ("Cellulitis, left lower leg", ["Type 2 diabetes"], [], "Full code"),
    ("Febrile neutropenia", ["Lymphoma on chemotherapy"], ["vancomycin (red man)"], "Full code"),
    ("NSTEMI, awaiting catheterization", ["CAD", "Hyperlipidemia"], [], "Full code"),
]
_NAMES = [("Okonkwo", "Ada"), ("Lindholm", "Erik"), ("Ferreira", "Lucia"), ("Brennan", "Thomas"), ("Takahashi", "Yuki"),
          ("Abernathy", "Grace"), ("Moretti", "Paolo"), ("Haddad", "Leila"), ("Kowal", "Marek"), ("Asante", "Kofi")]


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


@dataclass
class SimPatient:
    mrn: str
    family: str
    given: str
    sex: str
    dob: float
    room: str
    dx: str
    pmh: List[str]
    allergies: List[str]
    code_status: str
    weight: float
    phys: Any
    labs: Dict[str, float]
    admitted_at: float
    discharge_at: float
    next_vitals: float = 0.0
    next_nibp: float = 0.0
    next_lab: float = 0.0
    last_alarm: float = -1e9
    last_nibp: Optional[Dict[str, int]] = None
    scenario_started: bool = False


class SimulatedHospital:
    def __init__(self, engine: Integration, free_rooms: Callable[[], List[str]], seed: int = 2026,
                 clock: Callable[[], float] = time.time) -> None:
        self.engine = engine
        self.free_rooms = free_rooms
        self.clock = clock
        self.rng = random.Random(seed)
        self.patients: Dict[str, SimPatient] = {}
        self.pending_admits: List[float] = []
        self.faults: Dict[str, float] = {}
        self.sent = 0
        self.refused = 0
        self._n = 0
        self.started = False
        self.next_scenario = 0.0
        self.log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def register(self) -> None:
        now = self.clock()
        e = self.engine
        e.register(Interface("epic-adt", "Epic ADT", "HL7 v2.5.1", "inbound", "MLLP", ADT[0], ADT[1], "adt",
                             simulated=True, note="Admissions, transfers and discharges for the interface-fed rooms.",
                             created_at=now))
        e.register(Interface("intellibridge", "Philips IntelliBridge gateway", "HL7 v2.5.1 · IHE PCD-01/04", "inbound",
                             "MLLP", GW[0], GW[1], "vitals", simulated=True,
                             note="Monitor vitals (IEEE 11073 codes) and device alarms (ORU^R40).", created_at=now))
        e.register(Interface("sunquest-lis", "Sunquest laboratory", "HL7 v2.5.1", "inbound", "MLLP", LAB[0], LAB[1],
                             "labs", simulated=True, note="Chemistry and haematology in SI units, local mnemonics.",
                             created_at=now))
        e.register(Interface("reference-lab", "Reference laboratory", "HL7 v2.5.1", "inbound", "HTTP", REF[0], REF[1],
                             "labs", simulated=True, note="Send-out results coded the Nordic way (P-Kreat, B-Natr).",
                             created_at=now))

    def _send(self, segs: List[hl7.Segment], who: Tuple[str, str], mtype: Tuple[str, str, str],
              transport: str = "mllp", t: Optional[float] = None) -> Dict[str, Any]:
        now = self.clock() if t is None else t
        self._n += 1
        ctl = f"{who[0][:4]}{self._n:07d}"
        text = hl7.build([hl7.msh(who[0], who[1], "AEGISOS", "AEGISMEM", mtype, ctl, now)] + segs)
        ack, rec = self.engine.ingest_hl7(text, transport=transport, principal="svc.hl7 (simulated)")
        self.sent += 1
        code = hl7.parse_ack(ack)["code"]
        if code != "AA":
            self.refused += 1
        return {"ack": code, "rec": rec}

    # ------------------------------------------------------------------ #
    # patients
    # ------------------------------------------------------------------ #
    def _new_patient(self, room: str, now: float) -> SimPatient:
        from backend.aegis.rooms.physiology import Physiology
        from backend.aegis.rooms.records import synthetic_record
        k = self.rng.randrange(10 ** 6)
        dx, pmh, allergies, code = self.rng.choice(_CASES)
        taken = {(p.family, p.given) for p in self.patients.values()}
        family, given = self.rng.choice([n for n in _NAMES if n not in taken] or _NAMES)
        sex = "F" if given in ("Ada", "Lucia", "Yuki", "Grace", "Leila") else "M"
        age = self.rng.randint(38, 88)
        mrn = f"E{_seed('mrn', room, k) % 900000 + 100000}"
        p = {"room": room, "name": f"{family}, {given[0]}.", "age": age, "sex": sex, "mrn": mrn, "diagnosis": dx,
             "history": pmh, "medications": [], "allergies": allergies, "code_status": code, "admitted_days": 0}
        rec = synthetic_record(p, now)
        phys = Physiology(rec, _seed("sim-phys", mrn), now - 60)
        flags = set(rec.get("flags") or [])
        r = random.Random(_seed("labs", mrn))
        cr = (rec.get("labs", {}).get("creatinine") or [{"v": 0.9}])[-1]["v"]
        wbc = (rec.get("labs", {}).get("wbc") or [{"v": 8.0}])[-1]["v"]
        hb = (rec.get("labs", {}).get("hemoglobin") or [{"v": 12.5}])[-1]["v"]
        plt = (rec.get("labs", {}).get("platelets") or [{"v": 230}])[-1]["v"]
        labs = {"sodium": r.gauss(139, 2.5), "potassium": r.gauss(4.9 if "ckd" in flags else 4.1, 0.3),
                "chloride": r.gauss(103, 2.5), "hco3": r.gauss(24, 2.0),
                "bun": math.exp(r.gauss(math.log(34 if "ckd" in flags else 15), 0.25)), "creatinine": cr,
                "glucose": r.uniform(150, 220) if "diabetes" in flags else r.uniform(88, 128),
                "calcium": r.gauss(9.1, 0.35), "magnesium": r.gauss(2.0, 0.15), "phosphate": r.gauss(3.5, 0.5),
                "wbc": wbc, "anc": (300.0 if "neutropenia" in flags else wbc * 1000 * r.uniform(0.55, 0.72)),
                "hemoglobin": hb, "hematocrit": hb * r.uniform(2.9, 3.1), "platelets": plt,
                "lactate": r.uniform(1.4, 1.9) if "infection" in flags else r.uniform(0.8, 1.4),
                "bilirubin": math.exp(r.gauss(math.log(0.6), 0.35)),
                "troponin": math.exp(r.gauss(math.log(140 if "acs" in flags else 7), 0.5)) / 1000.0}
        stay = self.rng.uniform(*STAY_S)
        sp = SimPatient(mrn, family, given, sex, now - age * 365.25 * 86400 - self.rng.uniform(0, 3e7), room, dx, pmh,
                        allergies, code, rec["patient"].get("weight_kg") or 75.0, phys, labs, now, now + stay,
                        next_vitals=now + self.rng.uniform(0, VITALS_S), next_nibp=now + self.rng.uniform(2, NIBP_S),
                        next_lab=now + self.rng.uniform(*LAB_S))
        return sp

    def _admit(self, now: float) -> Optional[SimPatient]:
        from backend.aegis.paging.hospital import ROOMS, ZONES
        inpatient = [r for r in self.free_rooms() if r in ROOMS and ZONES[ROOMS[r].zone].kind in ("ward", "stepdown", "obs")]
        rooms = [r for r in inpatient if r not in {p.room for p in self.patients.values()}]
        if not rooms:
            return None
        room = self.rng.choice(sorted(rooms))
        p = self._new_patient(room, now)
        segs = [hl7.segment("EVN", "A01", hl7.format_ts(now)),
                hl7.segment("PID", "1", "", (p.mrn, "", "", "AEGISMEM", "MR"), "", (p.family, p.given),
                            "", time.strftime("%Y%m%d", time.gmtime(p.dob)), p.sex),
                hl7.segment("PV1", "1", "I", ("WARD", p.room, "1", "AEGISMEM"), "", "", "",
                            ("1008", "Reyes", "Maria", "", "", "Dr"), *([""] * 36), hl7.format_ts(now))]
        for i, a in enumerate(p.allergies, 1):
            segs.append(hl7.segment("AL1", str(i), "DA", ("", a.title()), "MO", "documented reaction"))
        for i, d in enumerate([p.dx] + p.pmh, 1):
            segs.append(hl7.segment("DG1", str(i), "", (_ICD.get(d, ""), d, "I10"), d, "", "A" if i == 1 else "F"))
        segs.append(hl7.segment("OBX", "1", "ST", ("CODESTAT", "Code status", "L"), "", p.code_status, "", "", "",
                                "", "", "F"))
        segs.append(hl7.segment("OBX", "2", "NM", ("WT", "Weight", "L"), "", float(p.weight), "kg", "", "", "", "",
                                "F"))
        r = self._send(segs, ADT, ("ADT", "A01", "ADT_A01"))
        if r["ack"] != "AA":
            return None
        self.patients[p.mrn] = p
        # the lab's historical load for a newly interfaced patient: the last day's panels
        for hours_ago in (22, 14, 6):
            self._lab_panel(p, now - hours_ago * 3600, backlog=True)
        self._ref_panel(p, now - 20 * 3600)
        self._ref_panel(p, now - 8 * 3600)
        self._ref_panel(p, now - 2 * 3600)
        return p

    def _discharge(self, p: SimPatient, now: float) -> None:
        segs = [hl7.segment("EVN", "A03", hl7.format_ts(now)),
                hl7.segment("PID", "1", "", (p.mrn, "", "", "AEGISMEM", "MR"), "", (p.family, p.given)),
                hl7.segment("PV1", "1", "I", ("WARD", p.room, "1", "AEGISMEM"))]
        self._send(segs, ADT, ("ADT", "A03", "ADT_A03"))
        self.patients.pop(p.mrn, None)
        self.pending_admits.append(now + self.rng.uniform(*READMIT_S))

    # ------------------------------------------------------------------ #
    # the feeds
    # ------------------------------------------------------------------ #
    def _pid(self, p: SimPatient) -> List[hl7.Segment]:
        return [hl7.segment("PID", "1", "", (p.mrn, "", "", "AEGISMEM", "MR"), "", (p.family, p.given)),
                hl7.segment("PV1", "1", "I", ("WARD", p.room, "1", "AEGISMEM"))]

    def _vitals(self, p: SimPatient, now: float) -> None:
        v = p.phys.sample(now)
        ts = hl7.format_ts(now)
        obx: List[hl7.Segment] = []
        temp_f = v["temp_c"] * 9 / 5 + 32
        if "temp_celsius" in self.faults:
            temp_f = v["temp_c"]                  # Celsius under a Fahrenheit label
        vals = {"hr": v["hr"], "spo2": v["spo2"], "rr": v["rr"], "temp": round(temp_f, 1)}
        for i, (code, ref, key, unit) in enumerate(GW_VITALS, 1):
            obx.append(hl7.segment("OBX", str(i), "NM", (code, ref, "MDC"), "", vals[key], (unit, unit, "UCUM"),
                                   "", "", "", "", "F", "", "", ts))
        if now >= p.next_nibp:
            p.next_nibp = now + NIBP_S
            bp = {"sbp": v["sbp"], "dbp": v["dbp"], "map": v["map"]}
            if "nibp_frozen" in self.faults and p.last_nibp:
                bp = dict(p.last_nibp)
            p.last_nibp = dict(bp)
            if "bp_swap" in self.faults:
                bp["sbp"], bp["map"] = bp["map"], bp["sbp"]
            for j, (code, ref, key) in enumerate(GW_NIBP, len(obx) + 1):
                obx.append(hl7.segment("OBX", str(j), "NM", (code, ref, "MDC"), "", bp[key], ("mm[Hg]", "mmHg", "UCUM"),
                                       "", "", "", "", "F", "", "", ts))
        self._send(self._pid(p) + [hl7.segment("OBR", "1", "", "", ("VITALS", "Monitor vitals", "L"), "", "", ts)]
                   + obx, GW, ("ORU", "R01", "ORU_R01"))
        self._maybe_alarm(p, v, now)

    def _maybe_alarm(self, p: SimPatient, v: Dict[str, Any], now: float) -> None:
        if now - p.last_alarm < ALARM_COOLDOWN_S:
            return
        crossing = None
        if v["spo2"] < 88:
            crossing = ("150456", "MDC_PULS_OXIM_SAT_O2", v["spo2"], "%", "LL", "SpO2 low")
        elif v["hr"] > 135:
            crossing = ("147842", "MDC_ECG_HEART_RATE", v["hr"], "/min", "HH", "Heart rate high")
        elif v["sbp"] < 85:
            crossing = ("150021", "MDC_PRESS_BLD_NONINV_SYS", v["sbp"], "mm[Hg]", "LL", "Systolic pressure low")
        if not crossing:
            return
        p.last_alarm = now
        code, ref, val, unit, flag, title = crossing
        ts = hl7.format_ts(now)
        segs = self._pid(p) + [hl7.segment("OBR", "1", "", "", ("ALARM", title, "L"), "", "", ts),
                               hl7.segment("OBX", "1", "NM", (code, ref, "MDC"), "", val, unit, "", flag, "", "", "F",
                                           "", "", ts)]
        self._send(segs, GW, ("ORU", "R40", "ORU_R40"))

    def _lab_value(self, p: SimPatient, key: str) -> float:
        base = p.labs[key]
        s = p.phys.scenario
        prog = s.progress(self.clock()) if s else 0.0
        kind = s.kind if s else None
        if kind == "sepsis":
            base *= {"lactate": 1 + 1.6 * prog, "wbc": 1 + 0.9 * prog, "creatinine": 1 + 0.6 * prog,
                     "bun": 1 + 0.4 * prog, "hco3": 1 - 0.25 * prog, "platelets": 1 - 0.3 * prog,
                     "anc": 1 + 0.9 * prog}.get(key, 1.0)
        elif kind == "bleed":
            base *= {"hemoglobin": 1 - 0.3 * prog, "hematocrit": 1 - 0.3 * prog}.get(key, 1.0)
        elif kind == "hyperkalemia" and key == "potassium":
            base += 2.2 * prog
        noise = {"sodium": 1.2, "potassium": 0.12, "chloride": 1.2, "hco3": 0.8}.get(key)
        x = base + self.rng.gauss(0, noise) if noise else base * math.exp(self.rng.gauss(0, 0.05))
        return max(1e-3, x)

    def _lab_panel(self, p: SimPatient, t: float, backlog: bool = False) -> None:
        ts = hl7.format_ts(t)
        obx = []
        for i, (code, text, system, key, unit, factor) in enumerate(SQLAB_PANEL, 1):
            if key is None:
                val = round(self.rng.uniform(0.05, 0.4), 2) if code == "PCT" else round(self.rng.uniform(8, 14))
            else:
                val = self._lab_value(p, key) * factor
                if not backlog:
                    if code == "GLU" and "glucose_units" in self.faults:
                        val = self._lab_value(p, key)            # mg/dL values, still labelled mmol/L
                    if code == "HGB" and "hgb_scale" in self.faults:
                        val = self._lab_value(p, key)            # g/dL values, still labelled g/L
                val = round(val, 3 if val < 1 else 2 if val < 20 else 1)
            u = unit
            if code == "CREA" and "creatinine_label" in self.faults and not backlog:
                u = "mg/dL"                                       # relabelled; the values stay µmol/L
            obx.append(hl7.segment("OBX", str(i), "NM", (code, text, system), "", val, u, "", "", "", "", "F",
                                   "", "", ts))
        self._send(self._pid(p) + [hl7.segment("OBR", "1", "", f"LAB{self._n}", ("CHEM_CBC", "Chemistry + CBC", "L"),
                                               "", ts, ts)] + obx, LAB, ("ORU", "R01", "ORU_R01"))

    def _ref_panel(self, p: SimPatient, t: float) -> None:
        ts = hl7.format_ts(t)
        obx = []
        for i, (code, text, system, key, unit, factor) in enumerate(NORDLAB_PANEL, 1):
            val = self._lab_value(p, key) * factor
            obx.append(hl7.segment("OBX", str(i), "NM", (code, text, system), "", round(val, 2), unit, "", "", "",
                                   "", "F", "", "", ts))
        self._send(self._pid(p) + [hl7.segment("OBR", "1", "", f"RL{self._n}", ("SENDOUT", "Send-out panel", "99RL"),
                                               "", ts, ts)] + obx, REF, ("ORU", "R01", "ORU_R01"), transport="http")

    # ------------------------------------------------------------------ #
    # the clock
    # ------------------------------------------------------------------ #
    def tick(self) -> None:
        if not self.engine.settings.get("simulated_feeds", True):
            return
        now = self.clock()
        if not self.started:
            self.started = True
            self.pending_admits = [now + 2 + 3 * i for i in range(N_PATIENTS)]
            self.next_scenario = now + self.rng.uniform(360, 600)
        due = [t for t in self.pending_admits if t <= now]
        if due:
            self.pending_admits = [t for t in self.pending_admits if t > now]
            for _ in due:
                if len(self.patients) < N_PATIENTS and self._admit(now) is None:
                    self.pending_admits.append(now + 30)
        for p in list(self.patients.values()):
            if now >= p.discharge_at and not (p.phys.scenario and p.phys.scenario.recover_t0 is None):
                self._discharge(p, now)
                continue
            if now >= p.next_vitals:
                p.next_vitals = now + VITALS_S
                self._vitals(p, now)
            if now >= p.next_lab:
                p.next_lab = now + self.rng.uniform(*LAB_S)
                self._lab_panel(p, now)
                if self.rng.random() < 0.35:
                    self._ref_panel(p, now)
            s = p.phys.scenario
            if s and s.recover_t0 is None and now - s.t0 > s.ramp + 240:
                p.phys.recover(now)               # the sending hospital treats its patients too
        if now >= self.next_scenario and self.patients:
            self.next_scenario = now + self.rng.uniform(600, 900)
            from backend.aegis.rooms.physiology import fitting
            cands = [p for p in self.patients.values() if p.phys.scenario is None]
            if cands:
                p = self.rng.choice(cands)
                kinds = fitting(p.phys.flags) or ["sepsis"]
                p.phys.start(self.rng.choice(kinds), now)

    # ------------------------------------------------------------------ #
    # fault drills
    # ------------------------------------------------------------------ #
    def drill(self, kind: str, on: bool = True) -> Dict[str, Any]:
        if kind == "clear":
            self.faults.clear()
            return {"status": "ok", "faults": {}}
        if kind not in DRILLS:
            return {"status": "error", "detail": f"unknown drill {kind!r}", "drills": list(DRILLS)}
        now = self.clock()
        if on:
            self.faults[kind] = now
            if DRILLS[kind][1] == "lab":
                # an analyser swap releases its queue at once: one panel per patient now
                for p in list(self.patients.values()):
                    self._lab_panel(p, now)
            elif kind in ("bp_swap", "nibp_frozen"):
                for p in self.patients.values():
                    p.next_nibp = min(p.next_nibp, now)
        else:
            self.faults.pop(kind, None)
        return {"status": "ok", "faults": self.status()["faults"]}

    def status(self) -> Dict[str, Any]:
        now = self.clock()
        return {"patients": [{"mrn": p.mrn, "room": p.room, "name": f"{p.family}, {p.given[0]}.", "dx": p.dx,
                              "scenario": p.phys.scenario.to_dict(now) if p.phys.scenario else None,
                              "discharge_in_s": round(p.discharge_at - now)} for p in self.patients.values()],
                "faults": {k: {"label": DRILLS[k][0], "since": t} for k, t in self.faults.items()},
                "drills": [{"kind": k, "label": v[0], "system": v[1]} for k, v in DRILLS.items()],
                "sent": self.sent, "refused": self.refused}
