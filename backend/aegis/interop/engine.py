"""The integration engine: how a hospital's feeds become what AegisOS watches.

A hospital does not hand an early-warning system clean data. It hands it HL7 v2
feeds: admissions from the ADT system, vitals from a monitor gateway, results from
the laboratory, alarms from bedside monitors, each in its own dialect of local codes
and units. This engine is the inbound half of that integration:

    HL7 text (MLLP or HTTP)
      → parse, deduplicate by control id, acknowledge (AA / AE / AR with a 0357 code)
      → ADT: the census (admit, transfer, discharge into the hospital's rooms)
      → ORU: each OBX through its MAPPING (local code → concept, unit → conversion)
             adopted     → converted, delivered to the room's record and detectors
             quarantined → withheld, and the room says so
             learning    → buffered until there is enough data to judge a proposal
      → ORU^R40: a device alarm becomes a page through the paging OS

**Mappings are earned, not declared.** A new local code collects values until there
are enough to judge; proposers (LOINC, IEEE 11073, lexical, a language model) suggest
what it is; the evidence engine (``evidence``) decides whether the data agrees. A
proposal whose declared unit passes is adopted automatically (policy ``evidence``);
one whose unit had to be inferred, that is ambiguous, or that the evidence refused,
waits for a person. A language model's proposal goes through the same checks as
anyone else's.

**Adopted mappings keep being checked.** Every few values the live window is
re-assessed under the adopted conversion, with the physiological identities across
co-observed fields and a stuck-feed test for vitals. A failure quarantines the
mapping: values are withheld from that moment, the values delivered since the
estimated change point are withdrawn from the room, the Sentinel raises it on the
operator console, and the evidence names a fix when it can ("these look like µmol/L")
for a person to accept.

Threading: all state sits behind one lock. Effects on rooms (admit, deliver, withdraw,
alarm) are collected under the lock and applied AFTER it is released, through the
``rooms`` port (the paging OS), so this lock is never held while the paging OS's is
taken. The paging OS may call ``sentinel_recs`` under its own lock; that order
(paging → interop) is the only one that exists.
"""
from __future__ import annotations

import copy
import math
import itertools
import statistics
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from backend.aegis.interop import evidence as E
from backend.aegis.interop import hl7
from backend.aegis.interop.mapping import Mapping, Sample, key_of
from backend.aegis.interop.proposer import Proposal, ask_model, coded, lexical
from backend.aegis.interop.units import IDENTITY, Conversion, norm
from backend.aegis.interop.vocab import BY_KEY, CONCEPTS, Concept, reference_meta, references, resolve_unit, transform

DEFAULT_SETTINGS: Dict[str, Any] = {
    "adoption": "evidence",          # evidence | review
    "unknown_senders": "register",   # register | reject
    "drift_monitor": True,
    "quarantine": True,
    "model_proposer": True,
    "simulated_feeds": True,
}
N_PROPOSE = E.N_MIN                  # values before a proposal is judged
_PROPOSER_NAMES = {"loinc": "the LOINC code", "mdc": "the IEEE 11073 code", "lexical": "the lexical rules"}
DRIFT_EVERY = 4                      # re-assess an adopted mapping every this many values
DRIFT_RECENT = 12                    # ... over its last 12 values (fast)
DRIFT_WINDOW = 40                    # ... and its last 40 (sensitive)
STUCK_RUN = 8                        # identical consecutive cuff pressures = a frozen feed
# cuff pressures only: a paced heart rate, a ventilated respiratory rate and a probe
# temperature legitimately repeat; eight identical oscillometric readings in a row do not
STUCK_CONCEPTS = ("sbp", "dbp", "map")
ALARM_COOLDOWN_S = 600.0
MESSAGES_N = 400
GROUPS_N = 160
VITAL_KEYS = ("hr", "sbp", "dbp", "map", "rr", "spo2", "temp_c", "o2_lpm", "etco2")
ADT_EVENTS = ("A01", "A02", "A03", "A04", "A08", "A11", "A13")
E_RECORD_LOCKED = 206


def _journal(event: str, severity: str = "info", uid: str = "interop", **detail: Any) -> None:
    try:
        from backend.aegis.kernel.journal import INFO, KJOURNAL, NOTICE, WARN
        KJOURNAL.append("interop", event, severity={"info": INFO, "notice": NOTICE, "warn": WARN}[severity],
                        uid=uid, **detail)
    except Exception:
        pass


@dataclass
class Interface:
    id: str
    name: str
    standard: str
    direction: str
    transport: str
    sending_app: str
    facility: str
    kind: str
    simulated: bool = False
    status: str = "idle"
    note: str = ""
    created_at: float = field(default_factory=time.time)
    last_message_at: Optional[float] = None
    stats: Dict[str, int] = field(default_factory=lambda: {
        "messages": 0, "AA": 0, "AE": 0, "AR": 0, "duplicates": 0, "observations": 0, "delivered": 0,
        "withheld": 0, "bytes": 0})
    minutes: Deque[List[int]] = field(default_factory=lambda: deque(maxlen=60))   # [minute, messages, errors]
    latency_us: Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def bump(self, now: float, ok: bool) -> None:
        m = int(now // 60)
        if not self.minutes or self.minutes[-1][0] != m:
            self.minutes.append([m, 0, 0])
        self.minutes[-1][1] += 1
        if not ok:
            self.minutes[-1][2] += 1

    def rate(self, now: float, window_min: int = 5) -> float:
        m = int(now // 60)
        return sum(x[1] for x in self.minutes if m - x[0] < window_min) / float(window_min)

    def series(self, now: float, n: int = 30) -> List[int]:
        m = int(now // 60)
        got = {x[0]: x[1] for x in self.minutes}
        return [got.get(m - n + 1 + i, 0) for i in range(n)]

    def to_dict(self, now: float) -> Dict[str, Any]:
        lat = sorted(self.latency_us)
        return {"id": self.id, "name": self.name, "standard": self.standard, "direction": self.direction,
                "transport": self.transport, "sending_app": self.sending_app, "facility": self.facility,
                "kind": self.kind, "simulated": self.simulated, "status": self.status, "note": self.note,
                "created_at": self.created_at, "last_message_at": self.last_message_at, "stats": dict(self.stats),
                "per_min": round(self.rate(now), 2), "series": self.series(now),
                "latency_us": {"p50": lat[len(lat) // 2] if lat else None,
                               "p99": lat[min(len(lat) - 1, int(len(lat) * 0.99))] if lat else None}}


class Integration:
    def __init__(self, rooms: Any = None, llm: Any = None, clock: Callable[[], float] = time.time,
                 hospital_rooms: Optional[Sequence[str]] = None) -> None:
        self._lock = threading.RLock()
        self.rooms = rooms
        self.llm = llm
        self.clock = clock
        self.settings = dict(DEFAULT_SETTINGS)
        self.hospital_rooms = set(hospital_rooms) if hospital_rooms is not None else None
        self.interfaces: Dict[str, Interface] = {}
        self.by_sender: Dict[Tuple[str, str], str] = {}
        self.mappings: Dict[Tuple[str, str, str], Mapping] = {}
        self.by_id: Dict[str, Mapping] = {}
        self.patients: Dict[str, Dict[str, Any]] = {}
        self.room_of: Dict[str, str] = {}
        self.messages: Deque[Dict[str, Any]] = deque(maxlen=MESSAGES_N)
        self._acks: "OrderedDict[Tuple[str, str], str]" = OrderedDict()
        self.groups: Dict[str, Deque[Dict[str, Any]]] = {}
        self.events: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []
        self.alarm_last: Dict[str, float] = {}
        self._mid = itertools.count(1)
        self._msg = itertools.count(1)
        self._obs = itertools.count(1)
        self._seq = itertools.count(1)
        self._model_queue: List[str] = []
        self._model_busy = False
        self.sync_model = False                  # tests call the model inline
        self.model_calls: List[Dict[str, Any]] = []
        self.started_at = clock()
        self.counters = {"messages": 0, "acks_aa": 0, "acks_ae": 0, "acks_ar": 0, "duplicates": 0,
                         "observations": 0, "delivered": 0, "withheld": 0, "backfilled": 0, "retracted": 0,
                         "adopted_auto": 0, "adopted_human": 0, "review": 0, "quarantines": 0, "alarms": 0,
                         "alarms_suppressed": 0, "model_proposals": 0}

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    def now(self) -> float:
        return self.clock()

    def _emit(self, kind: str, data: Any) -> None:
        ev = {"type": kind, "data": data, "ts": self.now()}
        self.events.append(ev)
        for fn in list(self.listeners):
            try:
                fn(ev)
            except Exception:
                pass

    def _apply(self, effects: List[Tuple[Any, ...]]) -> None:
        """Effects on rooms, applied with this lock RELEASED (see the module docstring)."""
        port = self.rooms
        if port is None or not effects:
            return
        for eff in effects:
            kind, args = eff[0], eff[1:]
            try:
                getattr(port, f"interface_{kind}")(*args)
            except Exception as exc:  # a room-side failure must not break the feed
                _journal("effect.error", "warn", effect=kind, error=str(exc)[:200])

    def register(self, iface: Interface) -> Interface:
        with self._lock:
            self.interfaces[iface.id] = iface
            self.by_sender[(iface.sending_app.upper(), iface.facility.upper())] = iface.id
            self.groups.setdefault(iface.id, deque(maxlen=GROUPS_N))
            return iface

    def _interface_for(self, msg: hl7.Message, transport: str) -> Optional[Interface]:
        key = (msg.sending_app.upper(), msg.sending_facility.upper())
        iid = self.by_sender.get(key)
        if iid:
            return self.interfaces[iid]
        if self.settings["unknown_senders"] != "register":
            return None
        app = msg.sending_app or "UNKNOWN"
        iid = f"{app.lower()}-{(msg.sending_facility or 'site').lower()}"
        kind = {"ADT": "adt", "ORU": "results"}.get(msg.message_type[0], "mixed")
        iface = Interface(iid, f"{app} ({msg.sending_facility or 'unknown facility'})", f"HL7 v{msg.version}",
                          "inbound", transport.upper(), app, msg.sending_facility, kind,
                          note="Registered on its first message; its codes are learned from its traffic.")
        _journal("interface.registered", "notice", interface=iid, sender=app, facility=msg.sending_facility)
        self.register(iface)
        self._emit("interface", iface.to_dict(self.now()))
        return iface

    # ------------------------------------------------------------------ #
    # intake
    # ------------------------------------------------------------------ #
    def ingest_hl7(self, text: str, transport: str = "http", principal: str = "svc.hl7") -> Tuple[str, Dict[str, Any]]:
        """One inbound message → (ACK text, the message record)."""
        t0 = time.perf_counter()
        now = self.now()
        effects: List[Tuple[Any, ...]] = []
        try:
            msg = hl7.Message.parse(text)
        except hl7.HL7Error as e:
            ack = hl7.ack(None, "AR", str(e), e)
            with self._lock:
                rec = self._record(None, None, text, "AR", str(e), e, [], "unparseable message", now, t0, transport)
                self.counters["acks_ar"] += 1
            return ack, rec
        with self._lock:
            iface = self._interface_for(msg, transport)
            if iface is None:
                err = hl7.HL7Error(f"sender {msg.sending_app}/{msg.sending_facility} is not a registered interface",
                                   hl7.E_UNKNOWN_KEY, "MSH^1^3")
                ack = hl7.ack(msg, "AR", str(err), err)
                rec = self._record(msg, None, text, "AR", str(err), err, [], "unknown sender", now, t0, transport)
                self.counters["acks_ar"] += 1
                return ack, rec
            dup = self._acks.get((iface.id, msg.control_id))
            if dup is not None:
                # a resend after a lost ACK: answer the same way, process nothing twice
                iface.stats["duplicates"] += 1
                self.counters["duplicates"] += 1
                rec = self._record(msg, iface, text, hl7.parse_ack(dup)["code"], "duplicate control id: not reprocessed",
                                   None, [], "duplicate (resend)", now, t0, transport, duplicate=True)
                return dup, rec
            outcomes: List[Dict[str, Any]] = []
            summary = ""
            code, note, err = "AA", "", None
            try:
                if msg.version not in hl7.SUPPORTED_VERSIONS:
                    raise hl7.HL7Error(f"HL7 version {msg.version} is not supported", hl7.E_UNSUPPORTED_VERSION,
                                       "MSH^1^12")
                if msg.processing_id not in ("P", "T"):
                    raise hl7.HL7Error(f"processing id {msg.processing_id} is not accepted (P or T)",
                                       hl7.E_UNSUPPORTED_PROCESSING, "MSH^1^11")
                mtype, event, _ = msg.message_type
                if mtype == "ADT":
                    if event not in ADT_EVENTS:
                        raise hl7.HL7Error(f"ADT event {event} is not routed", hl7.E_UNSUPPORTED_EVENT, "MSH^1^9^2")
                    summary = self._adt(msg, iface, event, now, effects)
                elif mtype == "ORU" and event == "R01":
                    summary, outcomes = self._oru(msg, iface, now, effects)
                elif mtype == "ORU" and event == "R40":
                    summary = self._alarm(msg, iface, now, effects)
                else:
                    raise hl7.HL7Error(f"message type {msg.type_name} is not routed", hl7.E_UNSUPPORTED_TYPE, "MSH^1^9")
            except hl7.HL7Error as e:
                err = e
                code = "AR" if e.code in (hl7.E_UNSUPPORTED_TYPE, hl7.E_UNSUPPORTED_EVENT, hl7.E_UNSUPPORTED_VERSION,
                                          hl7.E_UNSUPPORTED_PROCESSING, hl7.E_UNKNOWN_KEY) else "AE"
                note = str(e)
                summary = summary or note
                effects = []
            except Exception as e:  # never let one message take the feed down
                err = hl7.HL7Error(f"internal error: {e}", hl7.E_INTERNAL)
                code, note, summary, effects = "AE", str(err), "internal error", []
            ack = hl7.ack(msg, code, note or summary[:80], err)
            self._acks[(iface.id, msg.control_id)] = ack
            while len(self._acks) > 5000:
                self._acks.popitem(last=False)
            rec = self._record(msg, iface, text, code, note, err, outcomes, summary, now, t0, transport)
            self.counters[{"AA": "acks_aa", "AE": "acks_ae", "AR": "acks_ar"}[code]] += 1
        self._apply(effects)
        return ack, rec

    def _record(self, msg: Optional[hl7.Message], iface: Optional[Interface], text: str, code: str, note: str,
                err: Optional[hl7.HL7Error], outcomes: List[Dict[str, Any]], summary: str, now: float, t0: float,
                transport: str, duplicate: bool = False) -> Dict[str, Any]:
        us = (time.perf_counter() - t0) * 1e6
        rec = {"id": f"MSG-{next(self._msg):06d}", "at": now, "interface": iface.id if iface else None,
               "interface_name": iface.name if iface else None, "transport": transport,
               "type": msg.type_name if msg else None, "control_id": msg.control_id if msg else None,
               "version": msg.version if msg else None, "ack": code, "ack_text": note,
               "error": ({"code": err.code, "text": hl7.ERROR_TEXT.get(err.code, ""), "where": err.where,
                          "message": str(err)} if err else None),
               "summary": summary, "outcomes": outcomes, "raw": text[:6000], "bytes": len(text),
               "latency_us": round(us, 1), "duplicate": duplicate}
        self.messages.append(rec)
        self.counters["messages"] += 1
        if iface is not None:
            iface.stats["messages"] += 1
            iface.stats[code] = iface.stats.get(code, 0) + 1
            iface.stats["bytes"] += len(text)
            iface.last_message_at = now
            iface.bump(now, code == "AA")
            iface.latency_us.append(us)
        if err is not None:
            _journal("message.refused", "notice", interface=iface.id if iface else None, ack=code,
                     code=err.code, detail=str(err)[:160])
        self._emit("message", {k: rec[k] for k in ("id", "at", "interface", "type", "ack", "summary", "latency_us")})
        return rec

    # ------------------------------------------------------------------ #
    # ADT: the census
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mrn(pid: hl7.Segment) -> str:
        for i in range(1, len(pid.reps(3)) + 1):
            if pid.get(3, 5, rep=i).upper() in ("MR", "MRN", "PI"):
                return pid.get(3, 1, rep=i)
        return pid.get(3, 1)

    def _patient_from(self, msg: hl7.Message, iface: Interface, now: float) -> Dict[str, Any]:
        pid, pv1 = msg.first("PID"), msg.first("PV1")
        if pid is None:
            raise hl7.HL7Error("PID segment missing", hl7.E_SEGMENT_SEQUENCE, "PID")
        mrn = self._mrn(pid)
        if not mrn:
            raise hl7.HL7Error("PID-3 patient identifier missing", hl7.E_REQUIRED_FIELD, "PID^1^3")
        family, given = pid.get(5, 1), pid.get(5, 2)
        dob = hl7.parse_ts(pid.get(7, 1))
        age = int((now - dob) // (365.25 * 86400)) if dob else None
        room = (pv1.get(3, 2) or pv1.get(3, 1)) if pv1 else ""
        allergies = [{"substance": (a.get(3, 2) or a.get(3, 1)).lower(), "severity": a.get(4, 2) or a.get(4, 1) or "",
                      "reaction": a.get(5, 1) or ""} for a in msg.all("AL1") if a.get(3, 1) or a.get(3, 2)]
        dx = [d.get(4) or d.get(3, 2) or d.get(3, 1) for d in msg.all("DG1")]
        code_status, weight = None, None
        for o in msg.all("OBX"):
            c, t = o.get(3, 1).upper(), o.get(3, 2).lower()
            if c in ("CODESTAT", "CODESTATUS", "CODE_STATUS") or "code status" in t:
                code_status = o.get(5, 2) or o.get(5, 1)
            elif c in ("WT", "WEIGHT", "29463-7") or t.startswith("weight"):
                weight = hl7.number(o.get(5, 1))
        return {"mrn": mrn, "name": ", ".join(x for x in (family, (given[:1] + ".") if given else "") if x) or mrn,
                "age": age, "sex": pid.get(8, 1) or None, "room": room,
                "class": pv1.get(2, 1) if pv1 else "", "attending": (pv1.get(7, 2) + " " + pv1.get(7, 3)).strip()
                if pv1 else "", "allergies": allergies, "diagnoses": [d for d in dx if d],
                "code_status": code_status, "weight_kg": weight,
                "admitted_at": (hl7.parse_ts(pv1.get(44, 1)) if pv1 else None) or now,
                "interface": iface.id, "interface_name": iface.name, "simulated": iface.simulated}

    def _check_room(self, room: str, mrn: str) -> None:
        if not room:
            raise hl7.HL7Error("PV1-3 assigned location (room) missing", hl7.E_REQUIRED_FIELD, "PV1^1^3")
        if self.hospital_rooms is not None and room not in self.hospital_rooms:
            raise hl7.HL7Error(f"location {room} is not a room in this hospital", hl7.E_TABLE_VALUE, "PV1^1^3^2")
        holder = self.room_of.get(room)
        if holder and holder != mrn:
            raise hl7.HL7Error(f"room {room} is occupied by another interface patient", E_RECORD_LOCKED, "PV1^1^3")
        # interface_room_free reads the paging OS's occupancy WITHOUT its lock (dict membership
        # is atomic); taking that lock here, under this one, could deadlock with the Sentinel
        if not holder and self.rooms is not None and hasattr(self.rooms, "interface_room_free") \
                and not self.rooms.interface_room_free(room):
            raise hl7.HL7Error(f"room {room} is occupied", E_RECORD_LOCKED, "PV1^1^3")

    def _adt(self, msg: hl7.Message, iface: Interface, event: str, now: float, effects: List[Tuple[Any, ...]]) -> str:
        p = self._patient_from(msg, iface, now)
        mrn = p["mrn"]
        cur = self.patients.get(mrn)
        if event in ("A01", "A04", "A13"):
            if cur and cur.get("room") and cur["room"] != p["room"]:
                raise hl7.HL7Error(f"patient {mrn} is already admitted to room {cur['room']}",
                                   hl7.E_DUPLICATE_KEY, "PID^1^3")
            self._check_room(p["room"], mrn)
            self.patients[mrn], self.room_of[p["room"]] = p, mrn
            effects.append(("admit", p["room"], copy.deepcopy(p)))
            _journal("adt.admit", "info", interface=iface.id, mrn=mrn, room=p["room"])
            self._emit("census", {"event": event, "mrn": mrn, "room": p["room"]})
            return f"{event}: admitted {p['name']} ({mrn}) to room {p['room']}"
        if cur is None:
            raise hl7.HL7Error(f"patient {mrn} is not on the census", hl7.E_UNKNOWN_KEY, "PID^1^3")
        if event == "A08":
            keep = cur["room"]
            cur.update({k: v for k, v in p.items() if v not in (None, "", []) and k != "room"})
            cur["room"] = keep
            effects.append(("update", keep, copy.deepcopy(cur)))
            return f"A08: updated {cur['name']} ({mrn})"
        if event == "A02":
            new, old = p["room"], cur["room"]
            if new == old:
                return f"A02: {mrn} already in room {new}"
            self._check_room(new, mrn)
            self.room_of.pop(old, None)
            cur["room"] = new
            self.room_of[new] = mrn
            effects.append(("discharge", old, f"transferred to room {new}"))
            effects.append(("admit", new, copy.deepcopy(cur)))
            return f"A02: transferred {cur['name']} ({mrn}) from room {old} to {new}"
        if event in ("A03", "A11"):
            room = cur["room"]
            self.room_of.pop(room, None)
            self.patients.pop(mrn, None)
            effects.append(("discharge", room, "discharged" if event == "A03" else "admission cancelled"))
            _journal("adt.discharge", "info", interface=iface.id, mrn=mrn, room=room)
            self._emit("census", {"event": event, "mrn": mrn, "room": room})
            return f"{event}: {'discharged' if event == 'A03' else 'admission cancelled for'} {cur['name']} ({mrn}) from room {room}"
        raise hl7.HL7Error(f"ADT event {event} is not routed", hl7.E_UNSUPPORTED_EVENT, "MSH^1^9^2")

    # ------------------------------------------------------------------ #
    # ORU: observations through their mappings
    # ------------------------------------------------------------------ #
    def _mapping(self, iface: Interface, system: str, code: str, text: str, unit: str, now: float) -> Mapping:
        k = key_of(iface.id, system, code)
        m = self.mappings.get(k)
        if m is None:
            m = Mapping(f"M-{next(self._mid):04d}", iface.id, code, (system or "").upper(), text, unit, created_at=now,
                        updated_at=now)
            m.history.append({"at": now, "from": None, "to": "learning", "by": "engine",
                              "why": f"first seen on {iface.name}"})
            self.mappings[k] = m
            self.by_id[m.id] = m
        elif text and not m.text:
            m.text = text
        return m

    def _conversion_for(self, m: Mapping, written: str) -> Optional[Conversion]:
        """The conversion an adopted mapping applies to a value written in ``written``. The
        unit it was adopted under keeps its adopted conversion (which may be an inferred
        fix); a feed that relabels a value in another known unit is converted by that unit."""
        c = BY_KEY.get(m.concept or "")
        if c is None or m.conversion is None:
            return None
        if norm(written) == norm(m.unit_written_adopted or ""):
            return m.conversion
        _tok, conv = resolve_unit(c, written)
        return conv

    def _oru(self, msg: hl7.Message, iface: Interface, now: float,
             effects: List[Tuple[Any, ...]]) -> Tuple[str, List[Dict[str, Any]]]:
        pid = msg.first("PID")
        if pid is None:
            raise hl7.HL7Error("PID segment missing", hl7.E_SEGMENT_SEQUENCE, "PID")
        mrn = self._mrn(pid)
        patient = self.patients.get(mrn)
        if patient is None:
            raise hl7.HL7Error(f"patient {mrn} is not on the census (no ADT received)", hl7.E_UNKNOWN_KEY, "PID^1^3")
        room = patient["room"]
        ctl = msg.control_id
        t_msg = msg.timestamp or now
        outcomes: List[Dict[str, Any]] = []
        vitals: Dict[str, float] = {}
        labs: Dict[str, List[Tuple[str, float]]] = {}
        withheld: List[Dict[str, Any]] = []
        group: Dict[str, float] = {}
        gseq: Dict[str, int] = {}
        held: set = set()
        touched: List[Mapping] = []
        obr_time = None
        for seg in msg.segments:
            if seg.name == "OBR":
                obr_time = hl7.parse_ts(seg.get(7, 1))
                continue
            if seg.name != "OBX":
                continue
            vtype = seg.get(2, 1).upper()
            code, text, system = seg.get(3, 1), seg.get(3, 2), seg.get(3, 3)
            unit = seg.get(6, 1) or seg.get(6, 2)
            status = seg.get(11, 1).upper() or "F"
            t = hl7.parse_ts(seg.get(14, 1)) or obr_time or t_msg
            out = {"set_id": seg.get(1), "code": code, "text": text, "system": system, "value": seg.get(5, 1),
                   "unit": unit, "flag": seg.get(8, 1), "status": "", "concept": None, "canonical": None,
                   "mapping": None}
            if not code:
                out["status"] = "no code"
                outcomes.append(out)
                continue
            if status in ("X", "D", "W"):
                out["status"] = {"X": "result cannot be obtained", "D": "deletion (not applied)",
                                 "W": "marked wrong by sender (not applied)"}[status]
                outcomes.append(out)
                continue
            num = hl7.number(seg.get(5, 1)) if vtype in ("NM", "SN", "ST", "") else None
            m = self._mapping(iface, system, code, text, unit, now)
            out["mapping"] = m.id
            m.counts["seen"] += 1
            if num is None:
                m.counts["non_numeric"] += 1
                out["status"] = "not numeric"
                outcomes.append(out)
                continue
            m.unit_written = unit
            iface.stats["observations"] += 1
            self.counters["observations"] += 1
            s = Sample(t, num, unit, ctl, room, seq=next(self._seq))
            m.samples.append(s)
            group[m.id] = num
            gseq[m.id] = s.seq
            touched.append(m)
            if m.status == "adopted":
                c = BY_KEY[m.concept]
                conv = self._conversion_for(m, unit)
                if conv is None:
                    s.withheld = True
                    m.counts["withheld"] += 1
                    out.update(status="withheld: unit changed to one with no rule", concept=c.key)
                    withheld.append({"concept": c.key, "label": c.label, "value": num, "unit": unit,
                                     "why": f"unit changed to {unit!r}"})
                    self._quarantine([m], now, effects, f"the feed relabelled {c.label.lower()} in {unit!r}, "
                                                         f"which has no conversion to {c.unit}", [])
                    outcomes.append(out)
                    continue
                v = conv.apply(num)
                lo, hi = c.window
                if not lo <= v <= hi:
                    # one impossible value is never delivered, whatever the window says
                    s.withheld = True
                    m.counts["withheld"] += 1
                    iface.stats["withheld"] += 1
                    self.counters["withheld"] += 1
                    out.update(status="withheld: physiologically impossible", concept=c.key, canonical=round(v, 4))
                    withheld.append({"concept": c.key, "label": c.label, "value": round(v, 4), "unit": c.unit,
                                     "why": "outside the physiological window"})
                    m.since_check = DRIFT_EVERY        # an impossible value is evidence: judge now
                else:
                    oid = f"OBS-{next(self._obs):07d}"
                    s.obs_id, s.delivered = oid, True
                    m.counts["delivered"] += 1
                    iface.stats["delivered"] += 1
                    self.counters["delivered"] += 1
                    if c.room_key in VITAL_KEYS:
                        vitals[c.room_key] = round(v, 2)
                    elif c.room_key:
                        labs.setdefault(c.room_key, []).append((oid, round(v, 4)))
                    out.update(status="delivered", concept=c.key, canonical=round(v, 4))
                m.since_check += 1
            elif m.status == "quarantined":
                s.withheld = True
                m.counts["withheld"] += 1
                iface.stats["withheld"] += 1
                self.counters["withheld"] += 1
                c = BY_KEY.get(m.concept or "")
                out.update(status="withheld: mapping quarantined", concept=m.concept)
                withheld.append({"concept": m.concept, "label": c.label if c else m.text, "value": num, "unit": unit,
                                 "why": "the mapping is quarantined"})
            else:
                out["status"] = {"learning": "learning (not delivered)", "proposing": "awaiting a proposal",
                                 "review": "awaiting review", "unmapped": "not in the vocabulary",
                                 "rejected": "rejected mapping"}.get(m.status, m.status)
            outcomes.append(out)
        self.groups.setdefault(iface.id, deque(maxlen=GROUPS_N)).append(
            {"group": ctl, "t": t_msg, "room": room, "vals": group, "seq": gseq, "held": held})
        held.update(mid for mid in group if self.by_id[mid].samples and any(
            x.withheld and x.group == ctl for x in list(self.by_id[mid].samples)[-3:]))
        for m in {id(x): x for x in touched}.values():
            if m.status == "quarantined" and m.fault is not None:
                self._refresh_suggestion(m)
        # judge what is ready: new codes with enough data, adopted codes due a live check
        for m in {id(x): x for x in touched}.values():
            if m.status == "learning" and len(m.samples) >= N_PROPOSE and (
                    len(m.samples) == N_PROPOSE or len(m.samples) % 4 == 0):
                self._propose(m, now, effects)
            elif m.status == "adopted" and m.since_check >= DRIFT_EVERY and self.settings["drift_monitor"]:
                self._drift(m, now, effects)
        if vitals or labs:
            effects.append(("observe", room, t_msg, vitals, labs, {"interface": iface.id, "name": iface.name}))
        if withheld:
            effects.append(("withheld", room, t_msg, withheld, {"interface": iface.id, "name": iface.name}))
        n_del = sum(1 for o in outcomes if o["status"] == "delivered")
        return (f"{len(outcomes)} result{'s' if len(outcomes) != 1 else ''} for {patient['name']} (room {room}): "
                f"{n_del} delivered, {len(withheld)} withheld, {len(outcomes) - n_del - len(withheld)} not mapped yet",
                outcomes)

    # ------------------------------------------------------------------ #
    # ORU^R40: a device alarm (IHE PCD-04 subset)
    # ------------------------------------------------------------------ #
    def _alarm(self, msg: hl7.Message, iface: Interface, now: float, effects: List[Tuple[Any, ...]]) -> str:
        pid = msg.first("PID")
        mrn = self._mrn(pid) if pid else ""
        patient = self.patients.get(mrn) if mrn else None
        pv1 = msg.first("PV1")
        room = (patient or {}).get("room") or ((pv1.get(3, 2) or pv1.get(3, 1)) if pv1 else "")
        if not room:
            raise hl7.HL7Error("alarm names no patient on the census and no location", hl7.E_UNKNOWN_KEY, "PV1^1^3")
        parts = []
        for o in msg.all("OBX"):
            what = o.get(3, 2) or o.get(3, 1)
            val, unit, flag = o.get(5, 1), o.get(6, 1) or o.get(6, 2), o.get(8, 1)
            if what:
                parts.append(f"{_alarm_label(what)} {val}{(' ' + unit) if unit and unit not in ('%',) else unit}"
                             + (f" ({flag})" if flag else ""))
        obr = msg.first("OBR")
        title = (obr.get(4, 2) or obr.get(4, 1)) if obr else ""
        text = f"Monitor alarm, room {room}: " + ("; ".join(parts) or title or "alarm")
        last = self.alarm_last.get(room)
        if last is not None and now - last < ALARM_COOLDOWN_S:
            self.counters["alarms_suppressed"] += 1
            effects.append(("event", room, "alarm", "watch", f"{text} (suppressed: an alarm from this room was paged "
                                                             f"{int(now - last)} s ago)"))
            return f"R40: alarm in room {room} suppressed (repeat within {int(ALARM_COOLDOWN_S // 60)} min)"
        self.alarm_last[room] = now
        self.counters["alarms"] += 1
        effects.append(("alarm", room, text, f"{iface.name} (IHE PCD-04)"))
        _journal("alarm", "notice", interface=iface.id, room=room)
        return f"R40: alarm in room {room} sent to the paging OS"

    # ------------------------------------------------------------------ #
    # proposals and adoption
    # ------------------------------------------------------------------ #
    def _partners(self, m: Mapping, adopted_only: bool = False) -> Dict[str, Tuple[str, Conversion]]:
        """Other mappings on the same interface whose reading is known: adopted, or (at
        onboarding) the one viable reading the evidence found."""
        out: Dict[str, Tuple[str, Conversion]] = {}
        for o in self.by_id.values():
            if o.interface != m.interface or o.id == m.id:
                continue
            if o.status == "adopted" and o.concept and o.conversion:
                out[o.id] = (o.concept, o.conversion)
            elif not adopted_only and o.tentative and o.status in ("review", "proposing", "learning"):
                out[o.id] = o.tentative
        return out

    def _joint_rows(self, m: Mapping, concept: str, conv: Conversion, adopted_only: bool = False,
                    since_seq: int = 0) -> List[Tuple[float, Dict[str, float]]]:
        """Co-observed samples (group time, {concept: canonical value}) reading ``m`` as
        (concept, conv) and each partner as it is known. A partner's value is left out when
        it was withheld, or (for live checks) when it predates that partner's own evidence
        window: a value a mapping no longer vouches for cannot convict its neighbour."""
        partners = self._partners(m, adopted_only)
        rows: List[Tuple[float, Dict[str, float]]] = []
        for g in self.groups.get(m.interface, ()):
            if m.id not in g["vals"] or g["seq"].get(m.id, 0) <= since_seq or m.id in g["held"]:
                continue
            row = {concept: conv.apply(g["vals"][m.id])}
            for pid, raw in g["vals"].items():
                if pid not in partners or pid in g["held"]:
                    continue
                if adopted_only and g["seq"].get(pid, 0) <= self.by_id[pid].drift_from:
                    continue
                pc, pconv = partners[pid]
                if pc != concept:
                    row[pc] = pconv.apply(raw)
            rows.append((g["t"], row))
        return rows

    def _joint(self, m: Mapping, concept: str, conv: Conversion, adopted_only: bool = False,
               since_seq: int = 0, last: Optional[int] = None, tag: str = "") -> List[E.Check]:
        """The physiological identities that involve ``concept``, over this interface's recent
        messages, reading ``m`` as (concept, conv) and its partners as they are known."""
        partners = self._partners(m, adopted_only)
        keys_avail = {concept} | {c for c, _ in partners.values()}
        invs = [inv for inv in E.INVARIANTS if concept in inv.concepts and set(inv.concepts) <= keys_avail]
        if not invs:
            return []
        rows = self._joint_rows(m, concept, conv, adopted_only, since_seq)
        samples = [r for _t, r in rows]
        if last is not None:
            samples = samples[-last:]
        checks = [E.check_invariant(inv, samples) for inv in invs]
        if tag:
            for ch in checks:
                ch.name = f"{ch.name}:{tag}"
        return checks

    def _joint_live(self, m: Mapping, concept: str, conv: Conversion) -> List[E.Check]:
        """The identities involving ``concept``, over every co-observed message since the
        feed's evidence window began, judged by Page's CUSUM (``evidence.cusum_identity``)."""
        partners = self._partners(m, adopted_only=True)
        keys = {concept} | {c for c, _ in partners.values()}
        invs = [inv for inv in E.INVARIANTS if concept in inv.concepts and set(inv.concepts) <= keys]
        if not invs:
            return []
        rows = [r for _t, r in self._joint_rows(m, concept, conv, adopted_only=True, since_seq=m.drift_from)]
        return [E.cusum_identity(inv, rows) for inv in invs]

    def _explain_swap(self, m: Mapping, failing: List[str]) -> Optional[Dict[str, Any]]:
        """Does exchanging two adopted fields' readings restore every identity they take part
        in, over the messages since the violations began? Then those two were swapped."""
        maps = {o.concept: o for o in self.by_id.values()
                if o.interface == m.interface and o.status == "adopted" and o.concept and o.conversion}
        hot = set()
        for n in failing:
            hot |= set(E.INVARIANT_BY_NAME[n].concepts)
        invs = [inv for inv in E.INVARIANTS if set(inv.concepts) & hot and all(k in maps for k in inv.concepts)]
        fields = sorted({k for inv in invs for k in inv.concepts})
        if len(fields) < 2:
            return None
        rows: List[Dict[str, float]] = []
        for g in self.groups.get(m.interface, ()):
            row = {}
            for k in fields:
                o = maps[k]
                if o.id in g["vals"] and o.id not in g["held"] and g["seq"].get(o.id, 0) > o.drift_from:
                    row[k] = o.conversion.apply(g["vals"][o.id])
            if len(row) >= 2:
                rows.append(row)
        # only the current excursion: from where the failing identities' CUSUM last left zero
        start = len(rows)
        for n in failing:
            inv = E.INVARIANT_BY_NAME[n]
            lp, lq = math.log(E.CUSUM_P1 / inv.p0), math.log((1 - E.CUSUM_P1) / (1 - inv.p0))
            S, st = 0.0, len(rows)
            for i, r in enumerate(rows):
                if not all(k in r for k in inv.concepts):
                    continue
                try:
                    v = bool(inv.violated(r))
                except (ZeroDivisionError, ValueError, KeyError):
                    v = True
                S2 = max(0.0, S + (lp if v else lq))
                if S == 0.0 and S2 > 0.0:
                    st = i
                S = S2
            start = min(start, st)
        recent = rows[start:]

        def rate(inv: E.Invariant, rs: List[Dict[str, float]]) -> Tuple[int, int]:
            n = k = 0
            for r in rs:
                if all(x in r for x in inv.concepts):
                    n += 1
                    try:
                        k += bool(inv.violated(r))
                    except (ZeroDivisionError, ValueError, KeyError):
                        k += 1
            return k, n

        best = None
        for i, a in enumerate(fields):
            for b in fields[i + 1:]:
                swapped = []
                for r in recent:
                    r2 = dict(r)
                    if a in r and b in r:
                        r2[a], r2[b] = r[b], r[a]
                    swapped.append(r2)
                touched = [inv for inv in invs if a in inv.concepts or b in inv.concepts]
                ok, worst, told = True, 0.0, []
                for inv in touched:
                    k0, n0 = rate(inv, recent)
                    k1, n1 = rate(inv, swapped)
                    if n1 < 3:
                        ok = False
                        break
                    if k1 / n1 > max(1.5 * inv.p0, 1.0 / n1) or k1 >= k0:
                        ok = False
                        break
                    worst = max(worst, k1 / n1)
                    told.append(f"{inv.statement} holds in {n1 - k1} of {n1}")
                if ok and touched and (best is None or worst < best[0]):
                    best = (worst, a, b, told)
        if best is None:
            return None
        return {"a": maps[best[1]], "b": maps[best[2]], "restored": "; ".join(best[3])}

    def _joint_change(self, m: Mapping, inv_names: List[str]) -> Optional[float]:
        """Group time where the trailing run of identity violations begins (≥ 75% violated)."""
        rows = self._joint_rows(m, m.concept, m.conversion, adopted_only=True, since_seq=m.drift_from)
        invs = [E.INVARIANT_BY_NAME[n] for n in inv_names if n in E.INVARIANT_BY_NAME]
        flags: List[Tuple[float, bool]] = []
        for t, row in rows:
            app = [inv for inv in invs if all(k in row for k in inv.concepts)]
            if not app:
                continue
            bad = False
            for inv in app:
                try:
                    bad = bad or bool(inv.violated(row))
                except (ZeroDivisionError, ValueError, KeyError):
                    bad = True
            flags.append((t, bad))
        for i in range(len(flags)):
            if flags[i][1]:
                tail = [f for _t, f in flags[i:]]
                if sum(tail) >= 0.75 * len(tail):
                    return flags[i][0]
        return None

    def _propose(self, m: Mapping, now: float, effects: List[Tuple[Any, ...]],
                 model: Optional[Dict[str, Proposal]] = None, force: bool = False,
                 outcomes: Optional[Dict[str, str]] = None) -> None:
        props: List[Proposal] = []
        cp = coded(m.code, m.text, m.system)
        if cp:
            props.append(cp)
        for p in lexical(m.code, m.text, m.unit_written):
            if all(q.concept != p.concept for q in props):
                props.append(p)
        if model and m.code in model:
            m.model_props = [model[m.code]]
            self.counters["model_proposals"] += 1
        for mp in m.model_props:
            if all(q.concept != mp.concept for q in props):
                props.append(mp)
            else:
                for q in props:
                    if q.concept == mp.concept and "model agrees" not in q.rationale:
                        q.rationale += f"; the model agrees ({mp.proposer})"
        strong = any(p.score >= 0.75 for p in props)
        if (not strong and not force and model is None and not m.model_asked and self.settings["model_proposer"]
                and self.llm is not None and getattr(self.llm, "available", False)):
            m.model_asked = True
            self._model_queue.append(m.id)
            m.transition("proposing", "engine", "no confident lexical match: asking the language model", now)
            self._emit("mapping", m.to_dict())
            return
        values = m.values()
        results: List[Dict[str, Any]] = []
        for p in props:
            c = BY_KEY[p.concept]
            tok, declared = resolve_unit(c, m.unit_written)
            if tok and declared is None:
                # the feed wrote a unit and it cannot be this concept (a pressure written "/min",
                # a count written "mg/dL"): evidence AGAINST the proposal, not missing evidence.
                # (Found by the registered benchmark's smoke run; fixed after the registered run.)
                why = (f"the feed writes {m.unit_written!r}, which is not a unit {c.label.lower()} can be reported in"
                       if not tok.startswith("?") else f"the unit {m.unit_written!r} is not recognised")
                results.append({"proposal": p.to_dict(), "concept": c.key, "label": c.label, "decision": "unit mismatch",
                                "declared": None, "conversion": None, "_conv": None, "_declared_missing": False,
                                "inference": {"decision": "unit mismatch", "declared": None, "best": None,
                                              "alternatives": [], "why": why},
                                "joint": [], "viable": False})
                continue
            inf = E.infer(c, values, declared)
            conv = inf.conversion
            rival_why = None
            if inf.decision == "ambiguous" and declared is not None and inf.declared is not None \
                    and inf.declared.status == "pass":
                alts = [f.conversion.label for f in inf.passing if f.conversion.factor != declared.factor
                        and not f.conversion.label.startswith("decade shift")]
                rival_why = (f"the values fit the written unit ({declared.label}) and also "
                             f"{' / '.join(alts[:2])}: the data cannot tell them apart, so a person confirms the unit")
            joint = self._joint(m, c.key, conv) if conv is not None else []
            joint_fail = [j for j in joint if j.status == "fail"]
            results.append({"proposal": p.to_dict(), "concept": c.key, "label": c.label, "decision": inf.decision,
                            "declared": declared.to_dict() if declared else None,
                            "conversion": conv.to_dict() if conv else None, "_conv": conv,
                            "_declared_missing": declared is None,
                            "inference": {**inf.to_dict(), **({"why": rival_why} if rival_why else {})},
                            "joint": [j.to_dict() for j in joint],
                            "viable": inf.decision in ("declared", "inferred") and not joint_fail})
        results.sort(key=lambda r: (not r["viable"], -r["proposal"]["score"]))
        m.candidates = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        viable = [r for r in results if r["viable"]]
        concepts_viable = list(dict.fromkeys(r["concept"] for r in viable))
        if len(concepts_viable) == 1:
            r = viable[0]
            m.tentative = (r["concept"], r["_conv"])
            src = r["proposal"]["proposer"]
            who = _PROPOSER_NAMES.get(src) or ("the language model" if src.startswith("model") else src)
            # only a unit the feed WROTE can be confirmed: a feed with no unit waits for a person,
            # whatever fits (pH and INR, unitless by convention, resolve to their unit above)
            auto = self.settings["adoption"] == "evidence" and r["decision"] == "declared"
            if auto:
                self._adopt(m, BY_KEY[r["concept"]], r["_conv"], "loinc" if src == "loinc" else "declared", src,
                            "engine (evidence)", now, effects,
                            why=f"{who} proposed {r['label']}; the data agrees "
                                + ("in the written unit" if r["decision"] == "declared" else "with no unit written"))
                self.counters["adopted_auto"] += 1
            else:
                why = (f"{who} proposed {r['label']}; the values only fit read as "
                       f"{r['conversion']['label']} (written {m.unit_written!r}): a person confirms the unit"
                       if r["decision"] == "inferred" else
                       f"{who} proposed {r['label']}; policy: every mapping is adopted by a person")
                m.transition("review", "engine", why, now)
                self.counters["review"] += 1
        elif len(concepts_viable) > 1:
            m.tentative = None
            m.transition("review", "engine", "ambiguous: the data fits " + " and ".join(
                BY_KEY[k].label for k in concepts_viable[:3]), now)
            self.counters["review"] += 1
        elif results:
            if any(r["decision"] == "insufficient" for r in results) and len(m.samples) < 3 * N_PROPOSE:
                m.transition("learning", "engine", "not enough data to judge the proposals yet", now)
            else:
                m.tentative = None
                top = results[0]
                fit = top["inference"].get("declared") or top["inference"].get("best") or {}
                why = [c["detail"] for c in (fit.get("checks") or []) if c["status"] in ("fail", "unsure")][:2]
                if top["inference"].get("why"):
                    why.insert(0, top["inference"]["why"])
                why += [j["detail"] for j in top["joint"] if j["status"] == "fail"][:1]
                m.transition("review", "engine", f"the evidence could not confirm any proposal ({top['label']}"
                             + (f" read in {m.unit_written!r}" if m.unit_written else "") + ": "
                             + ("; ".join(why) or "no reading passes") + ")", now)
                self.counters["review"] += 1
        elif m.model_asked and m.status == "proposing" and (outcomes or {}).get(m.code, "").startswith("outside:"):
            m.tentative = None
            m.transition("unmapped", "engine", f"not in the vocabulary: the language model named "
                                               f"{outcomes[m.code][8:]!r}, which AegisOS does not carry", now)
        elif m.model_asked and m.status == "proposing" and (outcomes or {}).get(m.code) == "declined":
            m.tentative = None
            m.transition("unmapped", "engine", "no match in the vocabulary: the lexical rules and the language model "
                                               "both declined", now)
        elif m.model_asked and m.status == "proposing":
            # no answer is not a decline: a failed call (quota, timeout) or a skipped code goes
            # to a person, never silently out of the review queue
            m.tentative = None
            m.transition("review", "engine", "no proposer recognised this code and the language model gave no "
                                             "answer for it: map it by hand", now)
            self.counters["review"] += 1
        else:
            m.tentative = None
            m.transition("review", "engine", "no proposer recognised this code"
                         + ("" if self.llm is not None and getattr(self.llm, "available", False)
                            else " (the language-model proposer is unavailable)") + ": map it by hand", now)
            self.counters["review"] += 1
        if m.status != "adopted":
            _journal("mapping." + m.status, "info", interface=m.interface, mapping=m.id, code=m.code)
        self._emit("mapping", m.to_dict())

    def _adopt(self, m: Mapping, c: Concept, conv: Conversion, unit_source: str, proposer: Optional[str], by: str,
               now: float, effects: List[Tuple[Any, ...]], why: str, backfill: bool = True) -> None:
        m.concept, m.conversion, m.unit_source, m.proposer = c.key, conv, unit_source, proposer
        m.unit_written_adopted = m.unit_written
        m.version += 1
        m.adopted_at, m.adopted_by = now, by
        m.fault, m.since_check, m.tentative = None, 0, None
        m.drift_from = max((s.seq for s in m.samples), default=0) if not backfill else 0
        m.baseline = E.feed_baseline(c, [conv.apply(s.raw) for s in m.samples])
        m.evidence = {"at": now, "onboarding": next((x for x in m.candidates if x["concept"] == c.key), None),
                      "live": None}
        m.transition("adopted", by, why, now, concept=c.key, conversion=conv.to_dict())
        self.quarantine_clear(m.id)
        _journal("mapping.adopted", "notice", uid=by.split(" ")[0], interface=m.interface, mapping=m.id,
                 code=m.code, concept=c.key, conversion=conv.label, version=m.version)
        if backfill and c.room_key and c.room_key not in VITAL_KEYS:
            by_room: Dict[str, List[Tuple[str, float]]] = {}
            for s in m.samples:
                if s.delivered or s.room is None or self.room_of.get(s.room) is None:
                    continue
                v = conv.apply(s.raw)
                if not c.window[0] <= v <= c.window[1]:
                    continue
                oid = f"OBS-{next(self._obs):07d}"
                s.obs_id, s.delivered, s.withheld = oid, True, False
                by_room.setdefault(s.room, []).append((oid, round(v, 4), s.t))
                m.counts["backfilled"] += 1
                self.counters["backfilled"] += 1
            for room, rows in by_room.items():
                effects.append(("backfill", room, c.room_key, rows, f"{c.label} · mapping {m.id} adopted"))
        self._emit("mapping", m.to_dict())

    # ------------------------------------------------------------------ #
    # live evidence, quarantine and recovery
    # ------------------------------------------------------------------ #
    def _canonical_window(self, m: Mapping, c: Concept, n: int) -> List[Tuple[Sample, float]]:
        out = []
        for s in list(m.samples):
            if s.seq <= m.drift_from:
                continue
            conv = self._conversion_for(m, s.unit) or m.conversion
            out.append((s, conv.apply(s.raw)))
        return out[-n:]

    def _drift(self, m: Mapping, now: float, effects: List[Tuple[Any, ...]]) -> None:
        """Live evidence for an adopted mapping (see ``evidence.drift_checks``): impossible
        values and unit-size jumps against the feed's own baseline, over the last 12 and the
        last 40 values; the identities over its recent co-observed messages; and, for cuff
        pressures, a frozen feed."""
        m.since_check = 0
        m.checks_run += 1
        c = BY_KEY[m.concept]
        win = self._canonical_window(m, c, DRIFT_WINDOW)
        if len(win) < 4:
            return
        vals = [v for _s, v in win]
        if m.baseline is None and len(vals) >= DRIFT_RECENT:
            # after a person's fix the baseline is re-learned from what arrives under the fix,
            # never from values the new reading would misread
            m.baseline = E.feed_baseline(c, vals)
        checks = E.drift_checks(c, vals[-DRIFT_RECENT:], vals, m.baseline)
        joint = self._joint_live(m, c.key, m.conversion)
        live: Dict[str, Any] = {"at": now, "n": len(win), "checks": [ch.to_dict() for ch in checks],
                                "invariants": [j.to_dict() for j in joint], "liveness": None}
        failing = [ch for ch in checks if ch.status == "fail"]
        if c.key in STUCK_CONCEPTS:
            by_room: Dict[Optional[str], List[float]] = {}
            for smp, _v in win:
                by_room.setdefault(smp.room, []).append(smp.raw)
            run = max((E.stuck_run(xs) for xs in by_room.values()), default=0)
            live["liveness"] = {"longest_identical_run": run, "limit": STUCK_RUN}
            if run >= STUCK_RUN:
                failing.append(E.Check("liveness", "fail", float(run), float(STUCK_RUN), len(win),
                                       f"{run} identical consecutive readings: the feed looks frozen"))
        joint_fail = [j for j in joint if j.status == "fail"]
        m.evidence = {**(m.evidence or {}), "live": live}
        if not (failing or joint_fail) or not self.settings["quarantine"]:
            return
        if joint_fail and not failing:
            seen, causes = set(), []
            for j in joint_fail:
                base = j.name.split(":")[1]
                if base not in seen:
                    seen.add(base)
                    causes.append(j.detail)
            since_t = self._joint_change(m, sorted(seen))
            swap = self._explain_swap(m, sorted(seen))
            if swap is not None:
                # exchanging two fields' readings restores every identity they take part in:
                # that pair is the fault, and only that pair is withheld
                a, b = swap["a"], swap["b"]
                cause = (f"{a.code} ({BY_KEY[a.concept].label}) and {b.code} ({BY_KEY[b.concept].label}) look exchanged: "
                         + "; ".join(causes) + f"; read the other way round, {swap['restored']}")
                self._quarantine([a, b], now, effects, cause, joint_fail, since_t=since_t,
                                 swap={"a": a.id, "b": b.id, "codes": [a.code, b.code]})
                return
            # an identity failed and no single field moved, and no exchange explains it: it
            # cannot say which field is wrong, so every adopted field in the identity is withheld
            names = set()
            for j in joint_fail:
                names |= set(E.INVARIANT_BY_NAME[j.name.split(":")[1]].concepts)
            group = [o for o in self.by_id.values() if o.interface == m.interface and o.status == "adopted"
                     and o.concept in names]
            self._quarantine(group or [m], now, effects, "; ".join(causes), joint_fail, since_t=since_t)
        else:
            seen, causes = set(), []
            for ch in failing:
                base = ch.name.split(":")[0]
                if base not in seen:
                    seen.add(base)
                    causes.append(ch.detail)
            self._quarantine([m], now, effects, "; ".join(causes), failing + joint_fail)

    def _suspicious(self, m: Mapping, c: Concept, win: List[Tuple[Sample, float]]) -> List[bool]:
        """Per value: does it no longer read as this feed? Impossible for the concept, or a
        unit-size jump from the feed's own baseline. Values that look fine are left alone,
        even if they arrive after the change (a backlog of results from before it)."""
        tol = E.jump_tolerance(c)
        base = m.baseline

        def one(v: float) -> bool:
            if not c.window[0] <= v <= c.window[1]:
                return True
            if base is None or c.scale == "none":
                return False
            t = transform(c, v)
            return t is None or abs(t - base["loc"]) > tol

        return [one(v) for _s, v in win]

    def _suggest(self, m: Mapping, c: Concept, suspects: List[Sample]) -> Tuple[Optional[Conversion], Any]:
        """The evidence's fix for a quarantined mapping: the one reading of the suspect values
        that passes, when it differs from the conversion that was actually applied to them."""
        raw = [s.raw for s in suspects]
        if len(raw) < 4:
            return None, None
        inf = E.infer(c, raw, None, n_min=min(8, len(raw)))
        if inf.decision != "inferred" or inf.best is None:
            return None, inf
        applied = self._conversion_for(m, suspects[-1].unit) or m.conversion
        best = inf.best.conversion
        if applied and abs(best.factor - applied.factor) < 1e-9 and abs(best.offset - applied.offset) < 1e-9:
            return None, inf
        return best, inf

    def _refresh_suggestion(self, m: Mapping) -> None:
        """While quarantined, keep reading what the feed sends: a fix the first few suspect
        values could not support may become clear as more arrive."""
        f = m.fault or {}
        if f.get("suggested") or m.concept is None or f.get("by_identity"):
            return
        c = BY_KEY[m.concept]
        since = f.get("first_suspect_seq") or 0
        win = [(x, (self._conversion_for(m, x.unit) or m.conversion).apply(x.raw))
               for x in m.samples if x.seq >= since][-DRIFT_WINDOW:]
        sus = [x for (x, _v), bad in zip(win, self._suspicious(m, c, win)) if bad]
        if len(sus) < 4 or len(sus) == f.get("suggest_n"):
            return
        sug, inf = self._suggest(m, c, sus)
        f["suggest_n"] = len(sus)
        if sug is not None:
            f["suggested"] = sug.to_dict()
            f["inference"] = inf.to_dict() if inf else None
            self._emit("mapping", m.to_dict())

    def _quarantine(self, group: List[Mapping], now: float, effects: List[Tuple[Any, ...]], cause: str,
                    checks: List[E.Check], since_t: Optional[float] = None,
                    swap: Optional[Dict[str, Any]] = None) -> None:
        for m in group:
            if m.status == "quarantined":
                continue
            c = BY_KEY[m.concept]
            win = self._canonical_window(m, c, DRIFT_WINDOW)
            if since_t is not None:
                # an identity failed: no single value looks wrong, so everything from the
                # onset of the violations is unreliable, and no unit fix is inferred
                suspects = [s for s, _v in win if s.t >= since_t] or [s for s, _v in win[-DRIFT_RECENT:]]
                suggested, inf = None, None
            else:
                flags = self._suspicious(m, c, win)
                suspects = [s for (s, _v), bad in zip(win, flags) if bad]
                if not suspects:                     # frozen feed: the repeated readings
                    suspects = [s for s, _v in win[-DRIFT_RECENT:]]
                    suggested, inf = None, None
                else:
                    suggested, inf = self._suggest(m, c, suspects)
            retract: Dict[str, List[str]] = {}
            vit_since: Dict[str, float] = {}
            n_retracted = 0
            for s in suspects:
                if s.delivered and s.room:
                    if c.room_key in VITAL_KEYS:
                        vit_since[s.room] = min(vit_since.get(s.room, s.t), s.t)
                    elif s.obs_id:
                        retract.setdefault(s.room, []).append(s.obs_id)
                    s.delivered = False
                    n_retracted += 1
            m.counts["retracted"] += n_retracted
            self.counters["retracted"] += n_retracted
            m.fault = {"at": now, "cause": cause, "checks": [ch.to_dict() for ch in checks],
                       "by_identity": since_t is not None,
                       "first_suspect_t": suspects[0].t if suspects else None,
                       "first_suspect_seq": suspects[0].seq if suspects else None, "withdrawn": n_retracted,
                       "suggested": suggested.to_dict() if suggested else None,
                       "inference": inf.to_dict() if inf else None}
            if swap is not None:
                other = swap["b"] if swap["a"] == m.id else swap["a"]
                om = self.by_id.get(other)
                m.fault["swap"] = {"with": other, "code": om.code if om else None,
                                   "concept": om.concept if om else None,
                                   "label": f"exchange {m.code} and {om.code if om else other}"}
            m.transition("quarantined", "drift monitor", cause, now)
            self.counters["quarantines"] += 1
            iface = self.interfaces.get(m.interface)
            rooms = set(retract) | set(vit_since) | {s.room for s in suspects if s.room}
            for room in rooms:
                effects.append(("retract", room, retract.get(room, []),
                                {c.room_key: vit_since[room]} if room in vit_since and c.room_key else {}))
                effects.append(("event", room, "interface", "watch",
                                f"{c.label} feed from {iface.name if iface else m.interface} quarantined: {cause}. "
                                + (f"{len(retract.get(room, []))} result(s) withdrawn. " if retract.get(room) else "")
                                + "New values are withheld until it is reviewed."))
            _journal("mapping.quarantined", "warn", interface=m.interface, mapping=m.id, code=m.code,
                     concept=m.concept, cause=cause[:200], withdrawn=n_retracted,
                     suggested=suggested.label if suggested else None)
            self._emit("quarantine", m.to_dict(detail=False))

    def quarantine_clear(self, mapping_id: str) -> None:
        pass  # quarantines are derived from mapping status; kept for symmetry with the Sentinel hook

    # ------------------------------------------------------------------ #
    # people acting on mappings
    # ------------------------------------------------------------------ #
    def adopt(self, mapping_id: str, by: str, concept: Optional[str] = None,
              conversion: Any = None) -> Dict[str, Any]:
        """A person adopts a mapping. ``conversion`` is "declared" (the written unit),
        "suggested" (the evidence's inferred fix), "identity", or {"factor", "offset"}.
        The evidence for the chosen reading is recorded beside the decision; a person may
        adopt against it, and the record then says so."""
        effects: List[Tuple[Any, ...]] = []
        with self._lock:
            m = self.by_id.get(mapping_id)
            if m is None:
                return {"status": "error", "detail": "no such mapping"}
            key = concept or m.concept or (m.tentative[0] if m.tentative else None) or \
                (m.candidates[0]["concept"] if m.candidates else None)
            c = BY_KEY.get(key or "")
            if c is None:
                return {"status": "error", "detail": "name a concept to adopt this mapping as"}
            now = self.now()
            sw = (m.fault or {}).get("swap") if conversion == "swap" else None
            if conversion == "swap":
                partner = self.by_id.get((sw or {}).get("with") or "")
                if partner is None or partner.concept is None or m.concept is None:
                    return {"status": "error", "detail": "no exchange to accept"}
                pairs = [(m, BY_KEY[partner.concept], partner.conversion or IDENTITY),
                         (partner, BY_KEY[m.concept], m.conversion or IDENTITY)]
                for x, cx, cv in pairs:
                    x.unit_written_adopted = x.samples[-1].unit if x.samples else x.unit_written
                    self._adopt(x, cx, cv, "human", f"human:{by}", f"human:{by}", now, effects,
                                why=f"accepted by {by}: {m.code} and {partner.code} were exchanged at the sender; "
                                    f"{x.code} is now read as {cx.label}", backfill=False)
                    x.drift_from = max((smp.seq for smp in x.samples), default=0)
                    x.baseline = None
                self.counters["adopted_human"] += 2
                result = {"status": "ok", "mapping": m.to_dict(detail=True), "against_evidence": False,
                          "exchanged": [m.code, partner.code]}
        if conversion == "swap":
            self._apply(effects)
            return result
        with self._lock:
            m = self.by_id.get(mapping_id)
            if m is None:
                return {"status": "error", "detail": "no such mapping"}
            now = self.now()
            conv = self._resolve_choice(m, c, conversion)
            if conv is None:
                return {"status": "error", "detail": "no conversion for that choice (the written unit has no rule "
                                                     "for this concept; pick 'suggested' or give a factor)"}
            fit = E.assess(c, m.values(), conv)
            against = fit.status == "fail"
            was_q = m.status == "quarantined"
            withheld = [s for s in m.samples if s.withheld and not s.delivered] if was_q else []
            m.candidates = m.candidates or []
            self._adopt(m, c, conv, "human", f"human:{by}", f"human:{by}", now, effects,
                        why=f"adopted by {by} as {c.label} ({conv.label})" + (" against the evidence" if against else ""),
                        backfill=not was_q)
            if was_q and m.samples:
                m.unit_written_adopted = m.samples[-1].unit      # the fix reads the label as it is now written
            if was_q:
                # after a fix: read only what arrives from now on, and deliver what was withheld
                # during the quarantine if it now reads as the concept
                m.drift_from = max((s.seq for s in m.samples), default=0)
                m.baseline = None
                rows: Dict[str, List[Tuple[str, float, float]]] = {}
                if c.room_key and c.room_key not in VITAL_KEYS:
                    for s in withheld:
                        v = conv.apply(s.raw)
                        if s.room and self.room_of.get(s.room) and c.window[0] <= v <= c.window[1]:
                            oid = f"OBS-{next(self._obs):07d}"
                            s.obs_id, s.delivered, s.withheld = oid, True, False
                            rows.setdefault(s.room, []).append((oid, round(v, 4), s.t))
                            m.counts["backfilled"] += 1
                            self.counters["backfilled"] += 1
                for room, rr in rows.items():
                    effects.append(("backfill", room, c.room_key, rr, f"{c.label} · withheld values released after fix"))
                # a room whose withheld values were ALL released stops listing them as withheld
                iface = self.interfaces.get(m.interface)
                for room in {s.room for s in withheld if s.room and self.room_of.get(s.room)}:
                    if all(s.delivered for s in withheld if s.room == room):
                        effects.append(("released", room, c.key, iface.name if iface else m.interface))
            m.evidence = {**(m.evidence or {}), "human": {"fit": fit.to_dict(), "against_evidence": against}}
            self.counters["adopted_human"] += 1
            out = m.to_dict(detail=True)
        self._apply(effects)
        return {"status": "ok", "mapping": out, "against_evidence": against}

    def _resolve_choice(self, m: Mapping, c: Concept, choice: Any) -> Optional[Conversion]:
        if isinstance(choice, dict) and "factor" in choice:
            try:
                return Conversion(float(choice["factor"]), float(choice.get("offset", 0.0)),
                                  str(choice.get("label") or "set by hand"))
            except (TypeError, ValueError):
                return None
        if choice in (None, "suggested"):
            sug = (m.fault or {}).get("suggested")
            if sug:
                return Conversion(sug["factor"], sug["offset"], sug["label"])
            for cand in m.candidates:
                if cand["concept"] == c.key and cand.get("conversion"):
                    cv = cand["conversion"]
                    return Conversion(cv["factor"], cv["offset"], cv["label"])
            if choice == "suggested":
                return None
        if choice == "identity":
            return IDENTITY
        _tok, conv = resolve_unit(c, m.unit_written)
        return conv

    def reject(self, mapping_id: str, by: str, reason: str = "") -> Dict[str, Any]:
        with self._lock:
            m = self.by_id.get(mapping_id)
            if m is None:
                return {"status": "error", "detail": "no such mapping"}
            m.transition("rejected", f"human:{by}", reason or f"rejected by {by}", self.now())
            m.tentative = None
            _journal("mapping.rejected", "notice", uid=by, mapping=m.id, code=m.code)
            self._emit("mapping", m.to_dict())
            return {"status": "ok", "mapping": m.to_dict(detail=True)}

    def restore(self, mapping_id: str, by: str) -> Dict[str, Any]:
        """Put a quarantined mapping back as it was (the fault was the sender's and is fixed).
        Live evidence then reads only what arrives from now on."""
        with self._lock:
            m = self.by_id.get(mapping_id)
            if m is None or m.status != "quarantined":
                return {"status": "error", "detail": "only a quarantined mapping can be restored"}
            now = self.now()
            m.drift_from = max((s.seq for s in m.samples), default=0)
            m.fault = None
            m.since_check = 0
            m.transition("adopted", f"human:{by}", f"restored by {by}: the sender fixed the feed", now)
            _journal("mapping.restored", "notice", uid=by, mapping=m.id, code=m.code)
            self._emit("mapping", m.to_dict())
            return {"status": "ok", "mapping": m.to_dict(detail=True)}

    def reevaluate(self, mapping_id: str) -> Dict[str, Any]:
        effects: List[Tuple[Any, ...]] = []
        with self._lock:
            m = self.by_id.get(mapping_id)
            if m is None:
                return {"status": "error", "detail": "no such mapping"}
            if m.status in ("adopted",):
                self._drift(m, self.now(), effects)
            elif m.status in ("learning", "review", "unmapped", "proposing"):
                self._propose(m, self.now(), effects, force=True)
            out = m.to_dict(detail=True)
        self._apply(effects)
        return {"status": "ok", "mapping": out}

    def update_settings(self, patch: Dict[str, Any], by: str) -> Dict[str, Any]:
        with self._lock:
            for k, v in (patch or {}).items():
                if k == "adoption" and v in ("evidence", "review"):
                    self.settings[k] = v
                elif k == "unknown_senders" and v in ("register", "reject"):
                    self.settings[k] = v
                elif k in ("drift_monitor", "quarantine", "model_proposer", "simulated_feeds") and isinstance(v, bool):
                    self.settings[k] = v
            _journal("settings", "notice", uid=by, settings=dict(self.settings))
            self._emit("settings", dict(self.settings))
            return dict(self.settings)

    # ------------------------------------------------------------------ #
    # the clock: the model proposer, interface health
    # ------------------------------------------------------------------ #
    def tick(self) -> None:
        now = self.now()
        with self._lock:
            for iface in self.interfaces.values():
                if iface.direction != "inbound":
                    continue
                q = sum(1 for m in self.by_id.values() if m.interface == iface.id and m.status == "quarantined")
                recent_err = sum(x[2] for x in iface.minutes if int(now // 60) - x[0] < 5)
                status = ("idle" if not iface.last_message_at or now - iface.last_message_at > 300 else
                          "degraded" if q or recent_err else "live")
                if status != iface.status:
                    iface.status = status
                    self._emit("interface", iface.to_dict(now))
            batch: List[str] = []
            if self._model_queue and not self._model_busy:
                batch, self._model_queue = self._model_queue[:12], self._model_queue[12:]
                self._model_busy = True
                items = [{"code": self.by_id[i].code, "text": self.by_id[i].text, "unit": self.by_id[i].unit_written,
                          "values": self.by_id[i].values(60)} for i in batch]
        if not batch:
            return
        if self.sync_model:
            self._pump(batch, items, now)
        else:      # a model call can take seconds; the feeds keep flowing meanwhile
            threading.Thread(target=self._pump, args=(batch, items, now), daemon=True,
                             name="interop-model").start()

    def _pump(self, batch: List[str], items: List[Dict[str, Any]], now: float) -> None:
        t0 = time.perf_counter()
        outcomes: Dict[str, str] = {}
        try:
            props, model_id = ask_model(self.llm, items, outcomes=outcomes)
        except Exception:
            props, model_id = {}, None
        effects: List[Tuple[Any, ...]] = []
        with self._lock:
            self._model_busy = False
            self.model_calls.append({"at": now, "codes": [it["code"] for it in items], "model": model_id,
                                     "answered": sorted(props), "outcomes": dict(outcomes),
                                     "ms": round((time.perf_counter() - t0) * 1000)})
            del self.model_calls[:-30]
            for i in batch:
                m = self.by_id.get(i)
                if m is not None and m.status == "proposing":
                    self._propose(m, self.now(), effects, model=props, force=True, outcomes=outcomes)
        self._apply(effects)

    # ------------------------------------------------------------------ #
    # the Sentinel's view (called under the paging OS's lock: never call back into it)
    # ------------------------------------------------------------------ #
    def sentinel_recs(self, now: float) -> List[Any]:
        from backend.aegis.paging.sentinel import Rec
        out = []
        with self._lock:
            for m in self.by_id.values():
                if m.status != "quarantined":
                    continue
                c = BY_KEY.get(m.concept or "")
                iface = self.interfaces.get(m.interface)
                f = m.fault or {}
                sug = f.get("suggested")
                vital = bool(c and c.kind == "vital")
                out.append(Rec(f"interop:{m.id}", "interface_fault", "critical" if vital else "warn",
                               f"{c.label if c else m.code} feed quarantined on {iface.name if iface else m.interface}",
                               (f.get("cause") or "live evidence failed")[:240]
                               + (f" Withdrew {f.get('withdrawn')} delivered value(s)." if f.get("withdrawn") else ""),
                               ("Accept the evidence's fix (" + sug["label"] + ") in Integrations, or restore once the "
                                "sender is corrected." if sug else
                                "Accept the exchange (" + f["swap"]["label"] + ") in Integrations, or restore once the "
                                "sender is corrected." if f.get("swap") else
                                "Check the sender's configuration, then restore the mapping in Integrations.")
                               + " New values stay withheld meanwhile.",
                               [m.id, m.code]))
            review = [m for m in self.by_id.values() if m.status == "review"]
            if review:
                out.append(Rec("interop:review", "interface_review", "info",
                               f"{len(review)} interface code{'s' if len(review) != 1 else ''} waiting for review",
                               "Codes the evidence could not adopt on its own: an inferred unit, an ambiguous "
                               "match, or a proposal the data refused. Their values are not delivered.",
                               "Review them in Integrations → Mappings.", [m.id for m in review[:6]]))
        return out

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def overview(self) -> Dict[str, Any]:
        with self._lock:
            now = self.now()
            by_status: Dict[str, int] = {}
            by_iface: Dict[str, Dict[str, int]] = {}
            for m in self.by_id.values():
                by_status[m.status] = by_status.get(m.status, 0) + 1
                bi = by_iface.setdefault(m.interface, {"total": 0})
                bi["total"] += 1
                bi[m.status] = bi.get(m.status, 0) + 1
            return copy.deepcopy({
                "now": now, "since": self.started_at, "settings": self.settings,
                "interfaces": [i.to_dict(now) for i in self.interfaces.values()],
                "mappings": {"total": len(self.by_id), "by_status": by_status, "by_interface": by_iface},
                "review": [m.to_dict() for m in self.by_id.values() if m.status == "review"],
                "quarantined": [m.to_dict() for m in self.by_id.values() if m.status == "quarantined"],
                "census": len(self.patients), "counters": dict(self.counters),
                "model": {"available": bool(self.llm is not None and getattr(self.llm, "available", False)),
                          "queue": len(self._model_queue), "busy": self._model_busy, "calls": self.model_calls[-8:]},
                "reference": {**reference_meta(), "concepts": len(references())},
                "vocabulary": len(CONCEPTS), "invariants": len(E.INVARIANTS),
                "recent": [{k: r[k] for k in ("id", "at", "interface", "type", "ack", "summary", "latency_us")}
                           for r in list(self.messages)[-12:]],
            })

    def interface(self, iid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            i = self.interfaces.get(iid)
            if i is None:
                return None
            now = self.now()
            return copy.deepcopy({**i.to_dict(now),
                                  "mappings": [m.to_dict() for m in self.by_id.values() if m.interface == iid],
                                  "messages": [self._msg_summary(r) for r in self.messages if r["interface"] == iid][-50:]})

    @staticmethod
    def _msg_summary(r: Dict[str, Any]) -> Dict[str, Any]:
        return {k: r[k] for k in ("id", "at", "interface", "interface_name", "type", "control_id", "ack", "summary",
                                  "latency_us", "bytes", "duplicate")} | {"error": r["error"]}

    def message_list(self, interface: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            rows = [self._msg_summary(r) for r in self.messages if interface in (None, r["interface"])]
            return copy.deepcopy(rows[-limit:][::-1])

    def message(self, mid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for r in self.messages:
                if r["id"] == mid:
                    return copy.deepcopy(r)
            return None

    def mapping_list(self, interface: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy([m.to_dict() for m in self.by_id.values() if interface in (None, m.interface)])

    def mapping(self, mid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            m = self.by_id.get(mid)
            if m is None:
                return None
            d = m.to_dict(detail=True)
            c = BY_KEY.get(m.concept or (m.tentative[0] if m.tentative else "") or "")
            ref = references().get(c.key) if c else None
            d["concept_info"] = c.to_dict() if c else None
            d["reference"] = ref.to_dict() if ref else None
            return copy.deepcopy(d)

    def census(self) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(sorted(self.patients.values(), key=lambda p: p["room"]))

    def room_feed(self, room: str) -> Optional[Dict[str, Any]]:
        """What the integration plane knows about one room: the patient's feed and the
        state of every mapping that delivers into it. Read by the room view."""
        with self._lock:
            mrn = self.room_of.get(room)
            if mrn is None:
                return None
            p = self.patients[mrn]
            chans = []
            for m in self.by_id.values():
                if not any(s.room == room for s in list(m.samples)[-30:]):
                    continue
                c = BY_KEY.get(m.concept or "")
                last = next((s for s in reversed(m.samples) if s.room == room), None)
                chans.append({"mapping": m.id, "interface": m.interface,
                              "interface_name": (self.interfaces.get(m.interface).name
                                                 if m.interface in self.interfaces else m.interface),
                              "code": m.code, "text": m.text,
                              "concept": m.concept, "label": c.label if c else m.text, "status": m.status,
                              "unit_written": m.unit_written, "conversion": m.conversion.label if m.conversion else None,
                              "last_t": last.t if last else None, "fault": (m.fault or {}).get("cause")})
            return copy.deepcopy({"mrn": mrn, "interface": p["interface"], "interface_name": p["interface_name"],
                                  "simulated": p["simulated"], "admitted_at": p["admitted_at"],
                                  "channels": sorted(chans, key=lambda x: (x["status"] != "quarantined", x["label"]))})

    def vocabulary(self) -> Dict[str, Any]:
        refs = references()
        return {"concepts": [{**c.to_dict(), "reference": refs[c.key].to_dict() if c.key in refs else None}
                             for c in CONCEPTS],
                "invariants": [i.to_dict() for i in E.INVARIANTS], "reference_meta": reference_meta(),
                "thresholds": {"alpha": E.ALPHA, "n_min": E.N_MIN, "n_min_invariant": E.N_MIN_INV,
                               "p_implausible": E.P_IMPLAUSIBLE, "p_tail": E.P_TAIL, "drift_every": DRIFT_EVERY,
                               "drift_window": DRIFT_WINDOW, "stuck_run": STUCK_RUN}}


def _alarm_label(what: str) -> str:
    """A readable name for an alarm's measurement (the triage rules read these words)."""
    w = what.upper()
    for ref, label in (("SAT_O2", "SpO2"), ("HEART_RATE", "HR"), ("PULS_RATE", "HR"), ("RESP_RATE", "RR"),
                       ("NONINV_SYS", "BP systolic"), ("ART_SYS", "BP systolic"), ("NONINV_MEAN", "MAP"),
                       ("TEMP", "temperature")):
        if ref in w:
            return label
    return what
