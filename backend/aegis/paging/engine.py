"""The paging OS: the Operator agent and the escalation queue.

One alert, from a nurse or from the sepsis detector, runs one pipeline:

    triage (Priority Handler) → locate room, unit, patient → rank staff (Case
    Handler) → apply guardrails → autonomy policy → Z3 certificate → page

and every page then lives in a timeout-tracked queue: no answer in time, or a
decline, and it moves to the next-best clinician without anyone babysitting it.
A Sentinel watches the whole system on a 30 s timer and never pages anyone.

All state sits behind one lock. The server calls ``tick()`` once a second from a
worker thread and ``observe_ward()`` from the cluster loop; request handlers call
the rest. Events accumulate in a queue the server drains onto the WebSocket, so
nothing here touches the event loop.
"""
from __future__ import annotations

import copy
import itertools
import random
import statistics
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from backend.aegis.paging import demo, policy, sbar
from backend.aegis.paging.hospital import (ICU_ROOMS, LIVE_ROOM, REMOTE_ZONE, ROOMS, SPECIALTY_LABEL, ZONES,
                                           build_roster, layout_payload, synthetic_patients, zone_of)
from backend.aegis.paging.router import RouteDecision, rank
from backend.aegis.paging.sentinel import Sentinel
from backend.aegis.paging.triage import (DETECTOR_ALARM, DETECTOR_CLEAR, REFINE_TIMEOUT_S, Triage, model_available,
                                         triage_detector, triage_text)

ACTIVE = ("review", "held", "paging", "accepted", "unassigned")
CLOSED = ("resolved", "cancelled")
SENTINEL_PERIOD_S = 30.0


def _journal(event: str, severity: str = "info", uid: str = "paging", **detail: Any) -> None:
    try:
        from backend.aegis.kernel.journal import INFO, KJOURNAL, NOTICE, WARN
        sev = {"info": INFO, "notice": NOTICE, "warn": WARN}[severity]
        KJOURNAL.append("paging", event, severity=sev, uid=uid, **detail)
    except Exception:
        pass


class PagingOS:
    def __init__(self, llm: Any = None, clock: Callable[[], float] = time.time, seed: Optional[int] = None,
                 demo_history: bool = True) -> None:
        self._lock = threading.RLock()
        self.llm = llm
        self.clock = clock
        self.settings = copy.deepcopy(policy.DEFAULT_SETTINGS)
        now = self.clock()
        self.started_at = now
        self.roster = build_roster(now)
        self.synthetic = synthetic_patients()
        self.monitored: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.sentinel = Sentinel()
        self.events: Deque[Dict[str, Any]] = deque(maxlen=4000)
        self.presence: Dict[str, int] = {}
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self._pid = itertools.count(1041)
        self._aid = itertools.count(2201)
        self.rng = random.Random(seed)
        self.next_demo_at = now + 40
        self.next_roster_churn = now + 120
        self.last_sentinel = 0.0
        self.bundle_started: Optional[Callable[[Optional[str]], Optional[bool]]] = None
        self.dossier_text: Optional[Callable[[Optional[str]], Optional[str]]] = None
        self.bed_rooms: Dict[str, str] = {}      # bed_id -> room
        self.sync_sbar = False                   # tests write the brief inline
        self.sync_refine = False                 # tests refine inline
        if demo_history:
            self._seed_history(now)

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    def now(self) -> float:
        return self.clock()

    def emit(self, kind: str, data: Any) -> None:
        self.events.append({"type": kind, "data": data, "ts": self.now()})

    def drain(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = list(self.events)
            self.events.clear()
            return out

    def _patient(self, room: Optional[str]) -> Optional[Dict[str, Any]]:
        if not room:
            return None
        return self.monitored.get(room) or self.synthetic.get(room)

    def _load(self, now: float) -> Dict[str, Dict[str, Any]]:
        load: Dict[str, Dict[str, Any]] = {c: {"active": 0, "last_hour": 0, "last_paged_at": None, "last_case": None}
                                           for c in self.roster}
        for p in self.pages.values():
            if p["status"] in ("paging", "accepted") and p.get("assignee") in load:
                load[p["assignee"]]["active"] += 1
            for a in p["attempts"]:
                l = load.get(a["clinician"])
                if l is None:
                    continue
                if now - a["sent_at"] <= 3600:
                    l["last_hour"] += 1
                if l["last_paged_at"] is None or a["sent_at"] > l["last_paged_at"]:
                    l["last_paged_at"], l["last_case"] = a["sent_at"], p["id"]
        return load

    def _clin(self, cid: str) -> Dict[str, Any]:
        c = self.roster[cid].to_dict()
        c["present"] = self.presence.get(cid, 0) > 0
        return c

    def _emit_clin(self, cid: str) -> None:
        if cid in self.roster:
            self.emit("clinician", self._clin(cid))

    # ------------------------------------------------------------------ #
    # intake
    # ------------------------------------------------------------------ #
    def raise_alert(self, text: str, room: Optional[str] = None, source: str = "nurse",
                    reporter: Optional[str] = None, synthetic: bool = False) -> Dict[str, Any]:
        """A person raises an alert. The page goes out on the rules at once; the model
        then refines it in the background and may only raise it (see _refine)."""
        text = (text or "").strip()
        if not text:
            return {"status": "error", "detail": "empty alert"}
        tri = triage_text(text, room, client=None)
        refine = (source in ("nurse", "voice") and not synthetic and self.llm is not None
                  and model_available(self.llm) and not (tri.red_flags and tri.priority == 1))
        with self._lock:
            page = self._open_case(tri, text, source, reporter, synthetic)
            page["refining"] = refine
            out = copy.deepcopy(page)
        if refine:
            if self.sync_refine:
                self._refine(page["id"], text)
            else:
                threading.Thread(target=self._refine, args=(page["id"], text), daemon=True).start()
        return {"status": "ok", "page": out}

    def _refine(self, page_id: str, text: str) -> None:
        """Ask the model what the rules may have missed, after the page is already out.
        It may RAISE the priority (and re-route to the right service); it may never
        lower a page that has been sent. Its opinion is recorded either way."""
        with self._lock:
            page = self.pages.get(page_id)
            room = page.get("room") if page else None
        if not page:
            return
        llm_tri = triage_text(text, room, client=self.llm, timeout=REFINE_TIMEOUT_S)
        with self._lock:
            page = self.pages.get(page_id)
            if not page:
                return
            page["refining"] = False
            now = self.now()
            if llm_tri.source != "llm+rules":
                page["refinement"] = {"outcome": "unavailable", "at": now,
                                      "note": "The model did not answer in time; the rules stand."}
            elif llm_tri.priority < page["priority"] and page["status"] in ACTIVE:
                old_pr, old_cat = page["priority"], page["category"]
                page["priority"], page["category"] = llm_tri.priority, llm_tri.category
                page["triage"] = {**llm_tri.to_dict(), "rules_priority": old_pr}
                page["refinement"] = {"outcome": "raised", "at": now, "from": old_pr, "to": llm_tri.priority,
                                      "model": llm_tri.model, "note": llm_tri.rationale}
                page["escalation_log"].append({"at": now, "why": f"model raised P{old_pr}→P{llm_tri.priority}"
                                               + (f" ({old_cat}→{llm_tri.category})" if old_cat != llm_tri.category else ""),
                                               "to": page.get("assignee")})
                _journal("refined", "notice", page=page_id, frm=old_pr, to=llm_tri.priority, model=llm_tri.model)
                self._reroute_after_refine(page, now)
            else:
                page["refinement"] = {"outcome": "agrees" if llm_tri.priority == page["priority"] else "kept",
                                      "at": now, "model": llm_tri.model, "suggested": llm_tri.priority,
                                      "note": (llm_tri.rationale if llm_tri.priority == page["priority"] else
                                               f"Model suggested P{llm_tri.priority}; a sent page is never lowered.")}
            self._set_status(page, page["status"])
            self.emit("page", page)

    def _reroute_after_refine(self, page: Dict[str, Any], now: float) -> None:
        """A raised page rings on the shorter timeout of its new priority, and moves
        to the right service if the one holding it does not cover the new category."""
        dec = rank(page["priority"], page["category"], page["room"], self.roster, self._load(now), self.settings,
                   page["id"], now)
        allowed = set(dec.primary) | set(dec.acceptable)
        page["decision"] = {**dec.to_dict(), "previous_reasoning": page["decision"].get("reasoning")}
        holder = page.get("assignee") or page.get("planned")
        if page["status"] == "paging":
            att = page["attempts"][-1]
            att["deadline"] = min(att["deadline"], att["sent_at"] + policy.ack_timeout(self.settings, page["priority"]))
            if holder and self.roster[holder].specialty not in allowed:
                att.update(outcome="rerouted", at=now, reason="model identified a different service")
                self._release(holder, att)
                self._escalate(page, now, "re-routed after the model's triage")
        elif page["status"] in ("review", "held", "unassigned") and dec.assignee:
            page["planned"] = dec.assignee
            if page["status"] == "held":            # an upgraded page no longer waits
                self._send(page, dec.assignee, now)
        self.emit("notify", {"to": page.get("assignee"), "page": page["id"], "priority": page["priority"],
                             "room": page.get("room"), "text": f"Upgraded to P{page['priority']}", "kind": "upgrade"})

    def _open_case(self, tri: Triage, text: str, source: str, reporter: Optional[str], synthetic: bool,
                   bed: Optional[str] = None) -> Dict[str, Any]:
        now = self.now()
        t0 = time.perf_counter()
        room = tri.room
        zone = zone_of(room)
        z = ZONES.get(zone) if zone else None
        patient = self._patient(room)
        t_locate = time.perf_counter()
        pid, aid = f"PG-{next(self._pid)}", f"AL-{next(self._aid)}"
        dec = rank(tri.priority, tri.category, room, self.roster, self._load(now), self.settings, pid, now)
        t_route = time.perf_counter()
        review = room is None or policy.requires_review(self.settings, z.kind if z else "ward", tri.priority, source)
        page: Dict[str, Any] = {
            "id": pid, "alert_id": aid, "priority": tri.priority, "category": tri.category,
            "room": room, "zone": zone, "floor": z.floor if z else None, "zone_name": z.name if z else None,
            "bed": bed or (patient or {}).get("bed"), "patient": _patient_brief(patient),
            "text": text, "source": source, "reporter": reporter, "synthetic": synthetic,
            "status": "new", "assignee": None, "planned": dec.assignee, "backups": dec.backups,
            "attempts": [], "escalations": 0, "escalation_log": [],
            "created_at": now, "accepted_at": None, "resolved_at": None, "hold_until": dec.hold_until,
            "review_deadline": None, "requires_review": review, "review_reason": None,
            "operator_approved": False, "failsafe": False, "blocked": [],
            "triage": tri.to_dict(), "decision": dec.to_dict(), "certificate": None,
            "co_notified": [], "sbar": None,
            "timings": {"triage_ms": tri.latency_ms, "locate_ms": round((t_locate - t0) * 1000, 1),
                        "route_ms": round((t_route - t_locate) * 1000, 1)},
        }
        if review:
            page["review_reason"] = ("Room not identified from the alert." if room is None else
                                     "Machine-raised P1: a person confirms before it is sent." if source == "detector"
                                     and tri.priority == 1 and self.settings.get("detector_p1_review") else
                                     f"Policy: {z.name if z else 'this unit'} P{tri.priority} pages need operator review.")
        self.pages[pid] = page
        alert = {"id": aid, "page_id": pid, "kind": "alert", "text": text, "priority": tri.priority,
                 "category": tri.category, "room": room, "zone": zone, "floor": page["floor"],
                 "patient": (patient or {}).get("name"), "source": source, "reporter": reporter,
                 "created_at": now, "status": "new", "synthetic": synthetic, "history": False}
        self.alerts.append(alert)
        del self.alerts[:-300]
        _journal("alert", "notice" if tri.priority <= 2 else "info", uid=reporter or source,
                 alert=aid, page=pid, priority=tri.priority, category=tri.category, room=room, source=source)
        self._decide(page, dec, now)
        self._co_notify(page, dec)
        page["timings"]["total_ms"] = round((time.perf_counter() - t0) * 1000 + tri.latency_ms, 1)
        self.emit("alert", alert)
        self.emit("page", page)
        return page

    def _decide(self, page: Dict[str, Any], dec: RouteDecision, now: float) -> None:
        if dec.mode == "unassigned" or not dec.assignee:
            self._set_status(page, "unassigned")
            _journal("unassigned", "warn", page=page["id"], priority=page["priority"])
        elif dec.mode == "hold":
            page["hold_until"] = dec.hold_until
            self._set_status(page, "held")
        elif page["requires_review"]:
            fs = self.settings["review_failsafe_s"].get(str(page["priority"]))
            page["review_deadline"] = now + fs if fs else None
            self._set_status(page, "review")
        else:
            self._send(page, dec.assignee, now)

    def _co_notify(self, page: Dict[str, Any], dec: RouteDecision) -> None:
        for spec in dec.co_notify:
            staff = [c for c in self.roster.values() if c.specialty == spec and c.pageable
                     and c.status not in ("off_shift",) and c.id != page.get("planned")]
            if not staff:
                continue
            same = [c for c in staff if c.zone == page.get("zone")]
            c = (same or sorted(staff, key=lambda c: c.status != "available"))[0]
            note = {"clinician": c.id, "name": c.name, "specialty": SPECIALTY_LABEL[spec],
                    "why": "Hour-1 sepsis bundle: antibiotics within the hour" if page["category"] == "sepsis"
                    else "Code team notification"}
            page["co_notified"].append(note)
            self.emit("notify", {"to": c.id, "page": page["id"], "priority": page["priority"],
                                 "room": page["room"], "text": note["why"], "kind": "fyi"})

    # ------------------------------------------------------------------ #
    # sending, with the certificate
    # ------------------------------------------------------------------ #
    def _facts(self, page: Dict[str, Any], cid: str, now: float) -> policy.DispatchFacts:
        c = self.roster[cid]
        load = self._load(now).get(cid, {})
        d = page["decision"]
        recent = bool(load.get("last_paged_at") and now - load["last_paged_at"] < self.settings["consecutive_window_s"]
                      and load.get("last_case") != page["id"])
        return policy.DispatchFacts(
            priority=page["priority"], mode="dispatch", requires_review=page["requires_review"],
            operator_approved=page["operator_approved"], failsafe_fired=page["failsafe"],
            assignee_on_shift=c.status != "off_shift", on_shift_alternative=d.get("on_shift_alternative", True),
            recently_paged_other=recent, consecutive_alternative=d.get("consecutive_alternative", True),
            pages_last_hour=load.get("last_hour", 0), workload_limit=self.settings["workload_max_per_hour"],
            specialty_allowed=c.specialty in (d.get("primary", []) + d.get("acceptable", [])),
            timeout_s=policy.ack_timeout(self.settings, page["priority"]))

    def _send(self, page: Dict[str, Any], cid: str, now: float) -> bool:
        t0 = time.perf_counter()
        try:
            cert = policy.certify(self._facts(page, cid, now))
        except Exception as exc:                       # prover unhealthy -> fail closed
            cert = {"certified": False, "z3_status": "unavailable", "violations": ["prover_unavailable"],
                    "proof": f"prover unavailable ({type(exc).__name__}); failing closed", "ms": 0.0}
        page["timings"]["certify_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        page["certificate"] = {**cert, "at": now, "clinician": cid}
        if not cert["certified"]:
            page["blocked"] = cert["violations"]
            page["review_reason"] = "Held: the proposed page violates " + ", ".join(cert["violations"]) + "."
            page["planned"] = cid
            self._set_status(page, "review")
            _journal("blocked", "warn", page=page["id"], clinician=cid, violations=cert["violations"])
            return False
        c = self.roster[cid]
        timeout = policy.ack_timeout(self.settings, page["priority"])
        page["attempts"].append({"clinician": cid, "name": c.name, "specialty": SPECIALTY_LABEL[c.specialty],
                                 "sent_at": now, "deadline": now + timeout, "outcome": None, "at": None,
                                 "reason": None, "sim_at": None, "sim_action": None})
        page["assignee"], page["planned"], page["blocked"] = cid, cid, []
        prev = c.status
        page["attempts"][-1]["prior_status"] = prev
        if c.status == "available":
            c.status = "paging"
        self._set_status(page, "paging")
        self._emit_clin(cid)
        self.emit("notify", {"to": cid, "page": page["id"], "priority": page["priority"], "room": page["room"],
                             "text": page["text"], "kind": "page"})
        _journal("page", "notice" if page["priority"] <= 2 else "info", page=page["id"], clinician=cid,
                 priority=page["priority"], z3=cert["z3_status"], attempt=len(page["attempts"]))
        return True

    def _set_status(self, page: Dict[str, Any], status: str) -> None:
        page["status"] = status
        for a in reversed(self.alerts):
            if a.get("page_id") == page["id"]:
                a["status"] = status
                a["priority"] = page["priority"]
                self.emit("alert", a)
                break

    def _release(self, cid: Optional[str], attempt: Optional[Dict[str, Any]] = None) -> None:
        """Put a clinician back to what they were doing before a page rang."""
        if not cid or cid not in self.roster:
            return
        c = self.roster[cid]
        if c.status == "paging":
            still = any(p["status"] == "paging" and p.get("assignee") == cid for p in self.pages.values())
            if not still:
                prior = (attempt or {}).get("prior_status") or "available"
                c.status = prior if prior != "paging" else "available"
        self._emit_clin(cid)

    # ------------------------------------------------------------------ #
    # the queue: responses, escalation, operator actions
    # ------------------------------------------------------------------ #
    def respond(self, page_id: str, cid: str, action: str, reason: Optional[str] = None,
                actor: Optional[str] = None, simulated: bool = False) -> Dict[str, Any]:
        with self._lock:
            page = self.pages.get(page_id)
            if not page:
                return {"status": "error", "detail": "no such page"}
            if page["status"] != "paging" or not page["attempts"] or page["attempts"][-1]["clinician"] != cid:
                return {"status": "error", "detail": "this page is no longer ringing for you"}
            now = self.now()
            att = page["attempts"][-1]
            who = actor or cid
            if action == "accept":
                att.update(outcome="accepted", at=now, reason=reason)
                page["accepted_at"] = now
                c = self.roster[cid]
                c.status = "on_case"
                if c.zone != REMOTE_ZONE:
                    c.zone = page.get("zone") or c.zone
                c.busy_until = now + {1: 1500, 2: 1200, 3: 720, 4: 480}[page["priority"]]
                self._set_status(page, "accepted")
                self._emit_clin(cid)
                _journal("accepted", uid=("sim:" + cid) if simulated else who, page=page_id,
                         seconds=round(now - att["sent_at"], 1))
                self.emit("page", page)
                if self.sync_sbar:
                    self._write_sbar(page_id, not simulated)
                else:
                    threading.Thread(target=self._write_sbar, args=(page_id, not simulated), daemon=True).start()
            elif action == "decline":
                att.update(outcome="declined", at=now, reason=reason or "declined")
                self._release(cid, att)
                _journal("declined", "notice", uid=("sim:" + cid) if simulated else who, page=page_id, reason=reason)
                self._escalate(page, now, f"{self.roster[cid].name} declined" + (f": {reason}" if reason else ""))
            else:
                return {"status": "error", "detail": f"unknown action {action!r}"}
            return {"status": "ok", "page": copy.deepcopy(page)}

    def _escalate(self, page: Dict[str, Any], now: float, why: str) -> None:
        page["escalations"] += 1
        tried = [a["clinician"] for a in page["attempts"]]
        if page["escalations"] > self.settings["max_escalations"]:
            page["escalation_log"].append({"at": now, "why": why, "to": None})
            page["review_reason"] = "Escalation chain exhausted: the operator must assign this page."
            self._set_status(page, "unassigned")
            _journal("exhausted", "warn", page=page["id"])
            self.emit("page", page)
            return
        dec = rank(page["priority"], page["category"], page["room"], self.roster, self._load(now), self.settings,
                   page["id"], now, exclude=tried)
        page["decision"] = {**dec.to_dict(), "previous_reasoning": page["decision"].get("reasoning")}
        page["backups"] = dec.backups
        if dec.mode == "unassigned" or not dec.assignee:
            page["escalation_log"].append({"at": now, "why": why, "to": None})
            page["review_reason"] = "No one left who covers this case: the operator must assign it."
            self._set_status(page, "unassigned")
        else:
            page["escalation_log"].append({"at": now, "why": why, "to": dec.assignee,
                                           "name": self.roster[dec.assignee].name})
            self._send(page, dec.assignee, now)
        _journal("escalated", "notice", page=page["id"], why=why, to=dec.assignee)
        self.emit("page", page)

    def escalate_now(self, page_id: str, actor: str) -> Dict[str, Any]:
        with self._lock:
            page = self.pages.get(page_id)
            if not page or page["status"] not in ("paging", "review", "held", "unassigned"):
                return {"status": "error", "detail": "nothing to escalate"}
            now = self.now()
            if page["status"] == "paging":
                att = page["attempts"][-1]
                att.update(outcome="escalated", at=now, reason=f"operator {actor}")
                self._release(att["clinician"], att)
            else:
                page["operator_approved"] = True
            self._escalate(page, now, f"escalated by {actor}")
            return {"status": "ok", "page": copy.deepcopy(page)}

    def approve(self, page_id: str, actor: str, assignee: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            page = self.pages.get(page_id)
            if not page or page["status"] not in ("review", "held", "unassigned"):
                return {"status": "error", "detail": "this page is not waiting for approval"}
            now = self.now()
            target = assignee or page.get("planned")
            if target and target not in self.roster:
                return {"status": "error", "detail": f"unknown clinician {target!r}"}
            if not target:
                dec = rank(page["priority"], page["category"], page["room"], self.roster, self._load(now),
                           self.settings, page["id"], now, exclude=[a["clinician"] for a in page["attempts"]])
                page["decision"] = dec.to_dict()
                target = dec.assignee
            if not target:
                return {"status": "error", "detail": "no eligible clinician; choose one"}
            # an override outside the case's specialties is not blocked here: the
            # certificate names the violated invariant and the page stays in review
            page["operator_approved"] = True
            _journal("approved", "notice", uid=actor, page=page_id, clinician=target)
            ok = self._send(page, target, now)
            self.emit("page", page)
            return {"status": "ok" if ok else "blocked", "page": copy.deepcopy(page)}

    def cancel(self, page_id: str, actor: str, reason: str = "") -> Dict[str, Any]:
        with self._lock:
            page = self.pages.get(page_id)
            if not page or page["status"] in CLOSED:
                return {"status": "error", "detail": "already closed"}
            now = self.now()
            if page["status"] == "paging":
                att = page["attempts"][-1]
                att.update(outcome="cancelled", at=now)
                self._release(att["clinician"], att)
            if page["status"] == "accepted":
                self._finish_case(page)
            page["resolved_at"] = now
            self._set_status(page, "cancelled")
            _journal("cancelled", uid=actor, page=page_id, reason=reason)
            self.emit("page", page)
            return {"status": "ok", "page": copy.deepcopy(page)}

    def resolve(self, page_id: str, actor: str) -> Dict[str, Any]:
        with self._lock:
            page = self.pages.get(page_id)
            if not page or page["status"] != "accepted":
                return {"status": "error", "detail": "only an accepted page can be resolved"}
            page["resolved_at"] = self.now()
            self._finish_case(page)
            self._set_status(page, "resolved")
            _journal("resolved", uid=actor, page=page_id)
            self.emit("page", page)
            return {"status": "ok", "page": copy.deepcopy(page)}

    def _finish_case(self, page: Dict[str, Any]) -> None:
        cid = page.get("assignee")
        if cid in self.roster:
            c = self.roster[cid]
            others = any(p["status"] == "accepted" and p.get("assignee") == cid and p["id"] != page["id"]
                         for p in self.pages.values())
            if not others and c.status == "on_case":
                c.status, c.busy_until = "available", None
            self._emit_clin(cid)

    def set_status(self, cid: str, status: str, actor: str) -> Dict[str, Any]:
        if status not in ("available", "in_procedure", "on_case", "off_shift"):
            return {"status": "error", "detail": f"unknown status {status!r}"}
        with self._lock:
            c = self.roster.get(cid)
            if not c:
                return {"status": "error", "detail": "no such clinician"}
            now = self.now()
            c.status = status
            c.busy_until = now + 15 * 60 if status in ("in_procedure", "on_case") else None
            if status in ("available", "in_procedure", "on_case"):
                c.zone = c.zone or c.home_zone
            _journal("status", uid=actor, clinician=cid, status=status)
            self._emit_clin(cid)
            if status == "off_shift":
                for p in self.pages.values():
                    if p["status"] == "paging" and p.get("assignee") == cid:
                        p["attempts"][-1].update(outcome="off_shift", at=now)
                        self._escalate(p, now, f"{c.name} went off shift")
            return {"status": "ok", "clinician": self._clin(cid)}

    def acknowledge_rec(self, rec_id: str, actor: str) -> Dict[str, Any]:
        with self._lock:
            rec = self.sentinel.acknowledge(rec_id)
            if not rec:
                return {"status": "error", "detail": "no such recommendation"}
            self.emit("rec", rec.to_dict())
            _journal("rec_ack", uid=actor, rec=rec_id)
            return {"status": "ok", "rec": rec.to_dict()}

    def update_settings(self, patch: Dict[str, Any], actor: str) -> Dict[str, Any]:
        with self._lock:
            self.settings = policy.merge_settings(self.settings, patch)
            self.emit("settings", self.settings)
            _journal("settings", "notice", uid=actor, keys=sorted((patch or {}).keys()))
            return {"status": "ok", "settings": copy.deepcopy(self.settings)}

    def presence_join(self, cid: Optional[str]) -> None:
        if not cid or cid not in self.roster:
            return
        with self._lock:
            self.presence[cid] = self.presence.get(cid, 0) + 1
            self._emit_clin(cid)

    def presence_leave(self, cid: Optional[str]) -> None:
        if not cid or cid not in self.roster:
            return
        with self._lock:
            self.presence[cid] = max(0, self.presence.get(cid, 0) - 1)
            self._emit_clin(cid)

    # ------------------------------------------------------------------ #
    # SBAR (off the lock: it may call the model)
    # ------------------------------------------------------------------ #
    def _write_sbar(self, page_id: str, human: bool) -> None:
        with self._lock:
            page = copy.deepcopy(self.pages.get(page_id))
        if not page:
            return
        patient = self._patient(page.get("room"))
        dossier = None
        if human and page.get("bed") and self.dossier_text:
            try:
                dossier = self.dossier_text(page["bed"])
            except Exception:
                dossier = None
        # the template is instant, so the clinician never waits on a network call;
        # when a person is holding the device the model's version replaces it later
        briefs = [sbar.template(page, patient)]
        with self._lock:
            live = self.pages.get(page_id)
            if live is None:
                return
            live["sbar"] = briefs[0]
            live["sbar_refining"] = bool(human and self.llm is not None and model_available(self.llm))
            self.emit("page", live)
            self.emit("notify", {"to": live.get("assignee"), "page": page_id, "kind": "sbar",
                                 "priority": live["priority"], "room": live.get("room"), "text": "Brief ready"})
            refine = live["sbar_refining"]
        if not refine:
            return
        brief = sbar.generate(page, patient, self.llm, dossier, timeout=REFINE_TIMEOUT_S)
        with self._lock:
            live = self.pages.get(page_id)
            if live is not None:
                live["sbar_refining"] = False
                if brief.get("source") == "llm":
                    live["sbar"] = brief
                self.emit("page", live)

    # ------------------------------------------------------------------ #
    # the detector as an alert source
    # ------------------------------------------------------------------ #
    def observe_ward(self, beds: List[Dict[str, Any]], topology: Optional[List[Dict[str, Any]]] = None,
                     history: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Called by the cluster loop with the ward payload (effective risk per bed)."""
        tiers = {t["node_id"]: t.get("tier") for t in (topology or [])}
        ordered = sorted(beds, key=lambda b: int(str(b["bed_id"]).split("_")[-1]))
        with self._lock:
            now = self.now()
            for i, b in enumerate(ordered):
                if i >= len(ICU_ROOMS):
                    break
                room = ICU_ROOMS[i]
                bid = b["bed_id"]
                self.bed_rooms[bid] = room
                risk = float(b.get("fused_risk") or 0.0)
                flags = [k.replace("_", " ") for k, v in ((history or {}).get(bid) or {}).items() if v is True]
                self.monitored[room] = {
                    "room": room, "bed": bid, "name": b.get("name"), "age": int(b.get("age") or 0) or None,
                    "sex": b.get("gender"), "mrn": b.get("mrn"), "icu_hour": b.get("icu_hour"),
                    "risk": round(risk, 4), "population_risk": b.get("population_risk"), "tier": tiers.get(bid),
                    "vitals": b.get("enterprise") or {}, "drivers": b.get("drivers") or [],
                    "history": flags, "monitored": True, "synthetic": False,
                    "source": "PhysioNet/CinC 2019 · replayed", "watch": DETECTOR_CLEAR <= risk < DETECTOR_ALARM,
                }
                self._episode(room, bid, risk, b, now)
            self.emit("monitor", self.monitor_state())

    def observe_live(self, snap: Dict[str, Any]) -> None:
        """The live session (room 402): graded on the calibrated detector's own tier."""
        if not snap:
            return
        r = snap.get("risk") or {}
        v = snap.get("vitals") or {}
        with self._lock:
            now = self.now()
            level = r.get("level", "LOW")
            p = r.get("ml_probability")
            pt = snap.get("patient") or {}
            hist = [k.replace("_", " ") for k, val in (pt.get("clinical_history") or {}).items() if val is True]
            self.monitored[LIVE_ROOM] = {
                "room": LIVE_ROOM, "bed": None, "name": pt.get("name"), "age": pt.get("age"),
                "sex": (pt.get("gender") or "")[:1] or None, "mrn": pt.get("id"), "icu_hour": None,
                "risk": p, "level": level, "tier": level, "vitals": {k: v.get(k) for k in ("map", "heart_rate", "spo2",
                                                                                           "respiratory_rate", "skin_temp")},
                "drivers": list(snap.get("triggered_rules") or [])[:4], "history": hist, "monitored": True,
                "synthetic": False, "live": True, "source": f"live session · {snap.get('source', 'idle')}",
                "watch": level == "MEDIUM",
            }
            ep = self.episodes.get(LIVE_ROOM) or {"open": False}
            mp = v.get("map")
            shock = mp is not None and mp < 65
            if level == "HIGH" and not ep["open"]:
                tri = Triage(1 if shock else 2, "sepsis", LIVE_ROOM, ["calibrated risk HIGH"],
                             {"map": mp} if mp is not None else {},
                             [f"calibrated detector tier HIGH (p={p:.3f})" if p is not None else "detector tier HIGH"]
                             + (["MAP < 65"] if shock else []),
                             "The calibrated detector's tier reached HIGH on the live monitored patient.",
                             "detector", None, {"shock_signs": shock})
                page = self._open_case(tri, f"Live patient {pt.get('name')} (room {LIVE_ROOM}): calibrated sepsis risk "
                                            f"HIGH{', MAP ' + format(mp, '.0f') if mp else ''}.",
                                       "detector", "AegisOS detector", False)
                self.episodes[LIVE_ROOM] = {"open": True, "priority": tri.priority, "page": page["id"],
                                            "opened_at": now, "below": 0}
            elif level == "LOW" and ep.get("open"):
                ep["below"] = ep.get("below", 0) + 1
                if ep["below"] >= 3:
                    ep["open"] = False
                    self._note(f"Live patient back to LOW risk; detector episode closed.", LIVE_ROOM)
                self.episodes[LIVE_ROOM] = ep

    def _episode(self, room: str, bed: str, risk: float, b: Dict[str, Any], now: float) -> None:
        ep = self.episodes.get(room) or {"open": False, "priority": None, "page": None, "below": 0}
        if risk >= DETECTOR_ALARM:
            tri = triage_detector(room, b.get("name") or room, risk, b.get("enterprise") or {},
                                  b.get("drivers") or [], b.get("population_risk"))
            page = self.pages.get(ep["page"]) if ep.get("page") else None
            if not ep["open"]:
                text = (f"Sepsis risk {risk:.2f} on {b.get('name')} (room {room}). "
                        + ("; ".join((b.get('drivers') or [])[:3]) or "Risk above the alarm line."))
                page = self._open_case(tri, text, "detector", "AegisOS detector", False, bed=bed)
                ep = {"open": True, "priority": tri.priority, "page": page["id"], "opened_at": now, "below": 0}
            elif tri.priority < (ep.get("priority") or 4):
                ep["priority"] = tri.priority
                if page and page["status"] in ACTIVE:
                    page["priority"], page["triage"] = tri.priority, tri.to_dict()
                    page["escalation_log"].append({"at": now, "why": "upgraded to P1: septic-shock criteria met",
                                                   "to": page.get("assignee")})
                    self._set_status(page, page["status"])
                    self.emit("page", page)
                    self.emit("notify", {"to": page.get("assignee"), "page": page["id"], "priority": 1,
                                         "room": room, "text": "Upgraded to P1: septic-shock criteria met",
                                         "kind": "upgrade"})
                    _journal("upgraded", "notice", page=page["id"], room=room)
                else:
                    text = f"Septic-shock criteria on {b.get('name')} (room {room}): risk {risk:.2f}."
                    page = self._open_case(tri, text, "detector", "AegisOS detector", False, bed=bed)
                    ep["page"] = page["id"]
            ep["below"] = 0
        elif ep["open"] and risk < DETECTOR_CLEAR:
            ep["below"] = ep.get("below", 0) + 1
            if ep["below"] >= 3:
                ep["open"] = False
                self._note(f"{b.get('name')} (room {room}): sepsis risk back to {risk:.2f}, below "
                           f"{DETECTOR_CLEAR:.2f}. Detector episode closed.", room)
        self.episodes[room] = ep

    def _note(self, text: str, room: Optional[str]) -> None:
        zone = zone_of(room)
        item = {"id": f"AL-{next(self._aid)}", "page_id": None, "kind": "note", "text": text, "priority": None,
                "category": "sepsis", "room": room, "zone": zone, "floor": ZONES[zone].floor if zone else None,
                "patient": (self._patient(room) or {}).get("name"), "source": "detector",
                "reporter": "AegisOS detector", "created_at": self.now(), "status": "info", "synthetic": False,
                "history": False}
        self.alerts.append(item)
        self.emit("alert", item)

    # ------------------------------------------------------------------ #
    # the clock: timeouts, holds, fail-safes, simulated staff, sentinel
    # ------------------------------------------------------------------ #
    def tick(self) -> None:
        demo_alert = None
        with self._lock:
            now = self.now()
            s = self.settings
            for page in list(self.pages.values()):
                st = page["status"]
                if st == "paging":
                    att = page["attempts"][-1]
                    cid = att["clinician"]
                    if s["simulated_responders"] and not self.presence.get(cid):
                        self._simulate(page, att, now)
                        if page["status"] != "paging":
                            continue
                    if now >= att["deadline"]:
                        att.update(outcome="timeout", at=now)
                        self._release(cid, att)
                        _journal("timeout", "notice", page=page["id"], clinician=cid)
                        self._escalate(page, now, f"no answer from {att['name']} in "
                                                  f"{int(att['deadline'] - att['sent_at'])} s")
                elif st == "held" and page.get("hold_until") and now >= page["hold_until"]:
                    target = page.get("planned")
                    if target and self.roster[target].status in ("available", "on_case"):
                        self._send(page, target, now)
                    else:
                        self._escalate(page, now, "hold expired")
                    self.emit("page", page)
                elif st == "review" and page.get("review_deadline") and now >= page["review_deadline"] \
                        and not page["blocked"]:
                    page["failsafe"] = True
                    target = page.get("planned")
                    _journal("failsafe", "warn", page=page["id"])
                    if target:
                        self._send(page, target, now)
                    else:
                        self._escalate(page, now, "review fail-safe")
                    self.emit("page", page)
                elif st == "accepted" and s["simulated_responders"] and not self.presence.get(page.get("assignee")):
                    done_at = page.setdefault("sim_resolve_at", now + self.rng.uniform(
                        *{1: (150, 300), 2: (120, 240), 3: (60, 180), 4: (40, 120)}[page["priority"]]))
                    if now >= done_at:
                        page["resolved_at"] = now
                        self._finish_case(page)
                        self._set_status(page, "resolved")
                        self.emit("page", page)
            for c in self.roster.values():
                if c.status in ("in_procedure", "on_case") and c.busy_until and now >= c.busy_until:
                    if not any(p["status"] == "accepted" and p.get("assignee") == c.id for p in self.pages.values()):
                        c.status, c.busy_until = "available", None
                        c.zone = c.home_zone
                        self._emit_clin(c.id)
            if s["simulated_responders"] and now >= self.next_roster_churn:
                self._churn(now)
            if s["demo_traffic"] and now >= self.next_demo_at:
                self.next_demo_at = now + self.rng.uniform(75, 150)
                occupied = {r: p for r, p in self.synthetic.items()}
                demo_alert = demo.pick_scenario(self.rng, occupied)
            if now - self.last_sentinel >= SENTINEL_PERIOD_S:
                self.last_sentinel = now
                changed = self.sentinel.scan(now, self.alerts, list(self.pages.values()), self.roster,
                                             self._load(now), self.bundle_started)
                for r in changed:
                    self.emit("rec", r.to_dict())
                if changed:
                    self.emit("stats", self.stats())
        if demo_alert:
            text, room = demo_alert
            self.raise_alert(text, room, source="demo", reporter="demo nurse", synthetic=True)

    def _simulate(self, page: Dict[str, Any], att: Dict[str, Any], now: float) -> None:
        """Staff with no device open answer on their own, realistically: most accept,
        some are in a procedure and decline, some never answer (and the page escalates)."""
        cid = att["clinician"]
        c = self.roster[cid]
        if att.get("sim_action") is None:
            prior = att.get("prior_status")
            if prior == "in_procedure":
                weights = (0.25, 0.5, 0.25)
            else:
                weights = {1: (0.85, 0.05, 0.10), 2: (0.75, 0.10, 0.15), 3: (0.65, 0.15, 0.20),
                           4: (0.55, 0.20, 0.25)}[page["priority"]]
            action = self.rng.choices(["accept", "decline", "none"], weights=weights)[0]
            window = att["deadline"] - att["sent_at"]
            att["sim_action"] = action
            att["sim_at"] = att["sent_at"] + self.rng.uniform(5, max(6, min(35, window * 0.7)))
            if action == "decline":
                att["sim_reason"] = "in a procedure" if prior == "in_procedure" else self.rng.choice(
                    ["with another patient", "not my service", "covering another floor"])
        if att["sim_action"] in ("accept", "decline") and now >= att["sim_at"]:
            self.respond(page["id"], cid, att["sim_action"], att.get("sim_reason"), actor=f"sim:{cid}",
                         simulated=True)

    def _churn(self, now: float) -> None:
        """Keep the roster alive: someone free goes into a procedure now and then."""
        self.next_roster_churn = now + self.rng.uniform(90, 180)
        free = [c for c in self.roster.values() if c.status == "available" and c.pageable
                and c.specialty in ("surgery", "anesthesia", "cardiology", "neurology", "critical_care", "emergency")
                and not any(p["status"] in ("paging", "accepted") and p.get("assignee") == c.id
                            for p in self.pages.values())]
        if len(free) >= 2:
            c = self.rng.choice(free)
            c.status, c.busy_until = "in_procedure", now + self.rng.uniform(6, 14) * 60
            self._emit_clin(c.id)

    # ------------------------------------------------------------------ #
    # views
    # ------------------------------------------------------------------ #
    def monitor_state(self) -> Dict[str, Any]:
        return {room: {"risk": m.get("risk"), "tier": m.get("tier"), "watch": m.get("watch"),
                       "episode": bool((self.episodes.get(room) or {}).get("open")), "bed": m.get("bed"),
                       "live": bool(m.get("live"))}
                for room, m in self.monitored.items()}

    def occupancy(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for room, p in self.synthetic.items():
            out[room] = {"name": p["name"], "age": p["age"], "sex": p["sex"], "diagnosis": p["diagnosis"],
                         "synthetic": True, "monitored": False}
        for room, m in self.monitored.items():
            out[room] = {"name": m.get("name"), "age": m.get("age"), "sex": m.get("sex"),
                         "diagnosis": "Live monitored patient" if m.get("live") else "ICU admission (CinC 2019 replay)",
                         "synthetic": False, "monitored": True, "bed": m.get("bed"), "live": bool(m.get("live"))}
        return out

    def stats(self) -> Dict[str, Any]:
        now = self.now()
        real = [p for p in self.pages.values()]
        by_pr = {str(k): 0 for k in (1, 2, 3, 4)}
        for p in real:
            by_pr[str(p["priority"])] += 1
        ack: Dict[str, List[float]] = {str(k): [] for k in (1, 2, 3, 4)}
        for p in real:
            if p.get("accepted_at") and p["attempts"]:
                ack[str(p["priority"])].append(p["accepted_at"] - p["attempts"][0]["sent_at"])
        attempts = [a for p in real for a in p["attempts"]]
        esc = sum(1 for p in real if p["escalations"] > 0)
        return {
            "since": self.started_at, "pages": len(real), "by_priority": by_pr,
            "active": sum(1 for p in real if p["status"] in ACTIVE),
            "review": sum(1 for p in real if p["status"] == "review"),
            "by_source": {s: sum(1 for p in real if p["source"] == s) for s in ("nurse", "voice", "detector", "demo")},
            "median_ack_s": {k: round(statistics.median(v), 1) if v else None for k, v in ack.items()},
            "escalation_rate": round(esc / len(real), 3) if real else None,
            "timeouts": sum(1 for a in attempts if a["outcome"] == "timeout"),
            "declines": sum(1 for a in attempts if a["outcome"] == "declined"),
            "certificates": sum(1 for p in real if (p.get("certificate") or {}).get("certified")),
            "blocked": sum(1 for p in real if p.get("blocked")),
            "median_dispatch_ms": round(statistics.median([p["timings"].get("total_ms", 0) for p in real]), 1)
            if real else None,
            "monitored_beds": len(self.monitored), "uptime_s": round(now - self.started_at),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = self.now()
            recent = [p for p in self.pages.values() if p["status"] in ACTIVE
                      or now - (p.get("resolved_at") or p["created_at"]) < 1800]
            return copy.deepcopy({
                "now": now, "settings": self.settings,
                "clinicians": [self._clin(c) for c in self.roster],
                "pages": sorted(recent, key=lambda p: p["created_at"]),
                "alerts": self.alerts[-80:], "recs": self.sentinel.active(),
                "monitor": self.monitor_state(), "occupancy": self.occupancy(),
                "stats": self.stats(), "invariants": policy.INVARIANTS,
            })

    def page_detail(self, page_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            p = self.pages.get(page_id)
            if not p:
                return None
            out = copy.deepcopy(p)
            out["patient_record"] = copy.deepcopy(self._patient(p.get("room")))
            return out

    def room_detail(self, room: str) -> Optional[Dict[str, Any]]:
        if room not in ROOMS:
            return None
        with self._lock:
            r = ROOMS[room]
            z = ZONES[r.zone]
            return copy.deepcopy({
                "room": room, "zone": r.zone, "zone_name": z.name, "floor": r.floor,
                "patient": self._patient(room),
                "pages": [p for p in self.pages.values() if p.get("room") == room][-10:],
                "alerts": [a for a in self.alerts if a.get("room") == room][-10:],
                "episode": self.episodes.get(room),
            })

    @staticmethod
    def layout() -> Dict[str, Any]:
        return layout_payload()

    # ------------------------------------------------------------------ #
    def _seed_history(self, now: float) -> None:
        rng = random.Random(11)
        rooms = sorted(self.synthetic)
        for minutes, template, status in demo.HISTORY:
            room = rng.choice(rooms)
            text = template.format(room=room)
            tri = triage_text(text, room, client=None)
            zone = zone_of(room)
            self.alerts.append({
                "id": f"AL-{next(self._aid)}", "page_id": None, "kind": "alert", "text": text,
                "priority": max(tri.priority, 3), "category": tri.category, "room": room, "zone": zone,
                "floor": ZONES[zone].floor if zone else None, "patient": self.synthetic[room]["name"],
                "source": "demo", "reporter": "demo nurse", "created_at": now - minutes * 60,
                "status": status, "synthetic": True, "history": True})


def _patient_brief(p: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not p:
        return None
    return {k: p.get(k) for k in ("name", "age", "sex", "mrn", "diagnosis", "monitored", "synthetic", "source",
                                  "risk", "bed", "live")}
