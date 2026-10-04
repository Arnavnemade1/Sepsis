"""FHIR R4: AegisOS's view of the hospital, in the resources an EHR already reads.

An EHR, a data warehouse or a SMART app asks "what does AegisOS see about this
patient?" in FHIR, not in AegisOS's own JSON. This module is that read facade, built
from the same live state the console shows. Nothing here is a second copy.

    Patient          one per occupied room (the occupant), with the MRN as identifier
    Encounter        the inpatient stay, located in the room
    Observation      latest vitals (vital-signs profile) and laboratory results, LOINC
                     coded, UCUM units, each from the room's own record
    RiskAssessment   the sepsis detector's calibrated probability (ICU and live rooms)
    Flag             each active detection in the room, with its severity
    Communication    each page: priority (stat/asap/urgent/routine), status, recipient

Synthetic demo patients carry the HL7 security label ``HTEST`` ("test health data") in
``meta.security``, the standard way to keep test data from being mistaken for a
chart. Writes come in through ``Observation`` POSTs and transaction Bundles, which are
normalised into the integration engine's ORU path (``engine.ingest_hl7``), so a FHIR
feed earns its mappings on evidence exactly as an HL7 feed does.

Scope, stated plainly: a read facade plus inbound observations, enough for an EHR to
pull findings and push results. It is not a conformant FHIR server: no history, no
_include, no conditional writes, no terminology services.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.aegis.interop import hl7
from backend.aegis.interop.vocab import BY_KEY, BY_LOINC, CONCEPTS

FHIR_VERSION = "4.0.1"
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
SNOMED = "http://snomed.info/sct"
HTEST = {"system": "http://terminology.hl7.org/CodeSystem/v3-ActReason", "code": "HTEST",
         "display": "test health data"}
OBS_CATEGORY = "http://terminology.hl7.org/CodeSystem/observation-category"
PRIORITY = {1: "stat", 2: "asap", 3: "urgent", 4: "routine"}
UCUM_DISPLAY = {"Cel": "°C", "mm[Hg]": "mmHg", "10*3/uL": "10*3/uL", "/min": "/min", "1": "1", "[pH]": "[pH]"}

# room-record keys -> concept keys (the vitals channels and the lab keys rooms use)
_ROOM_TO_CONCEPT = {c.room_key: c.key for c in CONCEPTS if c.room_key}


def _iso(t: Optional[float]) -> Optional[str]:
    if t is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _ref(kind: str, rid: str) -> Dict[str, str]:
    return {"reference": f"{kind}/{rid}"}


def patient_id(room: str) -> str:
    return f"room-{room}"


def room_of(pid: str) -> Optional[str]:
    return pid[5:] if pid.startswith("room-") else None


def outcome(severity: str, code: str, text: str) -> Dict[str, Any]:
    return {"resourceType": "OperationOutcome",
            "issue": [{"severity": severity, "code": code, "diagnostics": text}]}


def bundle(entries: List[Dict[str, Any]], kind: str = "searchset", base: str = "") -> Dict[str, Any]:
    return {"resourceType": "Bundle", "type": kind, "total": len(entries), "timestamp": _iso(time.time()),
            "entry": [{"fullUrl": f"{base}/{e['resourceType']}/{e['id']}" if base and e.get("id") else None,
                       "resource": e, "search": {"mode": "match"}} for e in entries]}


class FHIRFacade:
    """Builds resources from the paging OS's rooms. ``units`` is the RoomMonitor-backed
    reader (``PAGING.room_unit`` / ``rooms_board``); ``pages`` lists pages."""

    def __init__(self, room_unit: Callable[[str], Optional[Dict[str, Any]]], board: Callable[[], Dict[str, Any]],
                 pages: Callable[[], List[Dict[str, Any]]], base_url: str = "/aegis/fhir/R4") -> None:
        self.room_unit = room_unit
        self.board = board
        self.pages = pages
        self.base_url = base_url

    # ------------------------------------------------------------------ #
    def capability(self) -> Dict[str, Any]:
        search = {
            "Patient": [("_id", "token"), ("identifier", "token")],
            "Encounter": [("patient", "reference")],
            "Observation": [("patient", "reference"), ("subject", "reference"), ("category", "token"),
                            ("code", "token")],
            "RiskAssessment": [("patient", "reference"), ("subject", "reference")],
            "Flag": [("patient", "reference"), ("subject", "reference")],
            "Communication": [("patient", "reference"), ("subject", "reference"), ("status", "token")],
        }
        res = []
        for kind, params in search.items():
            inter = [{"code": "read"}, {"code": "search-type"}]
            if kind == "Observation":
                inter.append({"code": "create"})
            res.append({"type": kind, "interaction": inter,
                        "searchParam": [{"name": n, "type": t} for n, t in params]})
        return {"resourceType": "CapabilityStatement", "status": "active", "kind": "instance",
                "date": _iso(time.time()), "fhirVersion": FHIR_VERSION, "format": ["json"],
                "software": {"name": "AegisOS", "version": "interop-1"},
                "implementation": {"description": "AegisOS read facade over live rooms; inbound Observations "
                                                  "are mapped on evidence by the integration engine",
                                   "url": self.base_url},
                "rest": [{"mode": "server", "resource": res,
                          "interaction": [{"code": "transaction"}, {"code": "batch"}]}]}

    # ------------------------------------------------------------------ #
    def _occupied(self) -> List[Dict[str, Any]]:
        return [r for r in (self.board().get("rooms") or []) if r.get("kind")]

    def patient(self, room: str, unit: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        u = unit or self.room_unit(room)
        if not u or not u.get("occupied"):
            return None
        rec = u["record"]
        pt = rec.get("patient") or {}
        name = (pt.get("name") or "").split(", ")
        res: Dict[str, Any] = {
            "resourceType": "Patient", "id": patient_id(room),
            "meta": {"lastUpdated": _iso(u.get("now") if u.get("clock") == "wall" else time.time())},
            "identifier": [{"use": "usual", "type": {"text": "MRN"}, "value": pt.get("mrn") or patient_id(room)}],
            "active": True,
            "name": [{"use": "official", "family": name[0] if name else None,
                      "given": [name[1].rstrip(".")] if len(name) > 1 else []}],
            "gender": {"F": "female", "M": "male"}.get(pt.get("sex") or "", "unknown"),
        }
        if pt.get("age") is not None:
            res["extension"] = [{"url": "https://aegis.local/fhir/StructureDefinition/age-years",
                                 "valueInteger": int(pt["age"])}]
        if (rec.get("provenance") or {}).get("synthetic"):
            res["meta"]["security"] = [HTEST]
        return res

    def patients(self, identifier: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for r in self._occupied():
            if identifier and identifier.split("|")[-1] not in (r.get("mrn"), patient_id(r["room"])):
                continue
            p = self.patient(r["room"])
            if p:
                out.append(p)
        return out

    def encounter(self, room: str) -> Optional[Dict[str, Any]]:
        u = self.room_unit(room)
        if not u or not u.get("occupied"):
            return None
        rec = u["record"]
        adm = rec.get("admission") or {}
        res = {"resourceType": "Encounter", "id": f"enc-{room}", "status": "in-progress",
               "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "IMP",
                         "display": "inpatient encounter"},
               "subject": _ref("Patient", patient_id(room)),
               "period": {"start": _iso(adm.get("admitted_at"))} if adm.get("admitted_at") else {},
               "reasonCode": [{"text": adm.get("reason")}] if adm.get("reason") else [],
               "location": [{"location": {"display": f"Room {room}, {u.get('zone_name')}"}, "status": "active"}]}
        if (rec.get("provenance") or {}).get("synthetic"):
            res["meta"] = {"security": [HTEST]}
        return res

    def _obs(self, room: str, key: str, value: float, t: Optional[float], category: str, idx: str,
             synthetic: bool) -> Optional[Dict[str, Any]]:
        ck = _ROOM_TO_CONCEPT.get(key)
        c = BY_KEY.get(ck or "")
        if c is None or value is None:
            return None
        res: Dict[str, Any] = {
            "resourceType": "Observation", "id": f"obs-{room}-{key}-{idx}", "status": "final",
            "category": [{"coding": [{"system": OBS_CATEGORY, "code": category}]}],
            "code": {"coding": [{"system": LOINC, "code": c.loinc, "display": c.loinc_display}], "text": c.label},
            "subject": _ref("Patient", patient_id(room)),
            "effectiveDateTime": _iso(t),
            "valueQuantity": {"value": round(float(value), 4), "unit": UCUM_DISPLAY.get(c.unit, c.unit),
                              "system": UCUM, "code": c.unit},
        }
        if synthetic:
            res["meta"] = {"security": [HTEST]}
        return res

    def observations(self, room: str, category: Optional[str] = None, code: Optional[str] = None) -> List[Dict[str, Any]]:
        u = self.room_unit(room)
        if not u or not u.get("occupied"):
            return []
        synthetic = bool((u["record"].get("provenance") or {}).get("synthetic"))
        replay = u.get("clock") == "icu_hours"
        out: List[Dict[str, Any]] = []
        if category in (None, "vital-signs"):
            v = u.get("vitals") or {}
            t = None if replay else v.get("t")
            for k in ("hr", "sbp", "dbp", "map", "rr", "spo2", "temp_c"):
                if v.get(k) is not None:
                    o = self._obs(room, k, v[k], t, "vital-signs", "latest", synthetic)
                    if o:
                        out.append(o)
        if category in (None, "laboratory"):
            for k, rows in (u.get("labs") or {}).items():
                if rows:
                    last = rows[-1]
                    o = self._obs(room, k, last.get("v"), None if replay else last.get("t"), "laboratory", "latest",
                                  synthetic)
                    if o:
                        out.append(o)
        if code:
            want = code.split("|")[-1]
            out = [o for o in out if any(cd.get("code") == want for cd in o["code"]["coding"])]
        return out

    def risk(self, room: str) -> List[Dict[str, Any]]:
        u = self.room_unit(room)
        if not u or not u.get("occupied") or u.get("probability") is None:
            return []
        res = {"resourceType": "RiskAssessment", "id": f"risk-{room}", "status": "final",
               "subject": _ref("Patient", patient_id(room)),
               "method": {"text": "AegisOS calibrated sepsis detector (research use; not a medical device)"},
               "occurrenceDateTime": _iso(time.time()),
               "prediction": [{"outcome": {"coding": [{"system": SNOMED, "code": "91302008", "display": "Sepsis"}]},
                               "probabilityDecimal": round(float(u["probability"]), 4),
                               "qualitativeRisk": {"text": "high" if (u.get("risk") or 0) >= 0.70 else
                                                   "moderate" if (u.get("risk") or 0) >= 0.35 else "low"}}],
               "note": [{"text": "The 0-1 index shown in AegisOS is pinned so 0.70 is this detector's own validated "
                                 "alarm point; probabilityDecimal is the calibrated probability."}]}
        if (u["record"].get("provenance") or {}).get("synthetic"):
            res["meta"] = {"security": [HTEST]}
        return [res]

    def flags(self, room: str) -> List[Dict[str, Any]]:
        u = self.room_unit(room)
        if not u or not u.get("occupied"):
            return []
        out = []
        for f in u.get("findings") or []:
            if f.get("severity") == "info":
                continue
            out.append({"resourceType": "Flag", "id": f"flag-{room}-{f['key'].replace(':', '-')}", "status": "active",
                        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/flag-category",
                                                  "code": "clinical"}]}],
                        "code": {"text": f["title"]}, "subject": _ref("Patient", patient_id(room)),
                        "period": {"start": _iso(f.get("first_seen")) if u.get("clock") == "wall" else None},
                        "extension": [{"url": "https://aegis.local/fhir/StructureDefinition/severity",
                                       "valueCode": f["severity"]},
                                      {"url": "https://aegis.local/fhir/StructureDefinition/detector",
                                       "valueString": f.get("detector")}]})
        return out

    def communications(self, room: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for p in self.pages():
            if room and p.get("room") != room:
                continue
            st = "completed" if p["status"] in ("resolved", "cancelled") else "in-progress"
            if status and st != status:
                continue
            res = {"resourceType": "Communication", "id": p["id"].lower(), "status": st,
                   "priority": PRIORITY.get(p.get("priority"), "routine"),
                   "category": [{"text": p.get("category") or "general"}],
                   "sent": _iso(p.get("created_at")),
                   "payload": [{"contentString": p.get("text")}],
                   "medium": [{"text": "AegisOS pager"}],
                   "note": [{"text": f"source: {p.get('source')}; status: {p.get('status')}"}]}
            if p.get("room"):
                res["subject"] = _ref("Patient", patient_id(p["room"]))
            if p.get("assignee"):
                res["recipient"] = [{"reference": f"Practitioner/{p['assignee']}"}]
            if p.get("synthetic"):
                res["meta"] = {"security": [HTEST]}
            out.append(res)
        return out

    # ------------------------------------------------------------------ #
    # inbound: Observations → the integration engine's ORU path
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_oru(observations: List[Dict[str, Any]], sender: Tuple[str, str] = ("FHIR", "EXTERNAL"),
               ctl: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
        """One ORU^R01 per patient, from FHIR Observations. Returns (ER7, problems)."""
        problems: List[str] = []
        by_patient: Dict[str, List[hl7.Segment]] = {}
        for i, o in enumerate(observations):
            if (o or {}).get("resourceType") != "Observation":
                problems.append(f"entry {i}: not an Observation")
                continue
            subj = ((o.get("subject") or {}).get("reference") or "")
            mrn = subj.split("/")[-1] if subj else ""
            if mrn.startswith("room-"):
                problems.append(f"entry {i}: subject must be the sending system's patient id (MRN), not a room")
                continue
            q = o.get("valueQuantity") or {}
            coding = next((c for c in ((o.get("code") or {}).get("coding") or []) if c.get("code")), None)
            if not mrn or coding is None or q.get("value") is None:
                problems.append(f"entry {i}: needs subject, code.coding and valueQuantity")
                continue
            system = "LN" if coding.get("system") == LOINC else "L"
            unit = q.get("code") or q.get("unit") or ""
            ts = None
            if o.get("effectiveDateTime"):
                try:
                    import calendar
                    ts = calendar.timegm(time.strptime(o["effectiveDateTime"][:19], "%Y-%m-%dT%H:%M:%S"))
                except ValueError:
                    ts = None
            segs = by_patient.setdefault(mrn, [])
            segs.append(hl7.segment("OBX", str(len(segs) + 1), "NM",
                                    (coding["code"], coding.get("display") or (o.get("code") or {}).get("text") or "",
                                     system), "", float(q["value"]), unit, "", "", "", "", "F", "", "",
                                    hl7.format_ts(ts) if ts else ""))
        if not by_patient:
            return None, problems
        msgs = []
        for k, (mrn, obx) in enumerate(by_patient.items()):
            msgs.append(hl7.build([hl7.msh(sender[0], sender[1], "AEGISOS", "AEGISMEM", ("ORU", "R01", "ORU_R01"),
                                           f"{ctl or hl7.control_id('FH')}-{k}"),
                                   hl7.segment("PID", "1", "", (mrn, "", "", sender[1], "MR"))] + obx))
        return "\x00".join(msgs), problems
