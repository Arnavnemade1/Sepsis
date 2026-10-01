"""Verification for the paging OS.

Every guardrail is shown doing its job, and the certificate is shown refusing a
planted bad decision (a check that has never failed is not evidence):

  [1] triage       red flags set a floor; the model may raise, never lower; P1 skips it
  [1b] refinement  pages go out on the rules; the model raises (and re-routes) afterwards
  [2] routing      specialty, consecutive paging, workload, near-tie, off-shift, time-queued
  [3] certificate  good pages are certified (unsat); a forced off-shift page and an
                   unreviewed ICU P1 are refused (sat), named, and not sent
  [4] queue        decline -> backup, timeout -> escalate, exhaustion, fail-safe, hold, cancel
  [5] brief        SBAR on accept: four sections, under 100 words, facts only
  [6] detector     one page per episode, P2 -> P1 upgrade, hysteresis, watch without paging
  [7] sentinel     zone surge, overload, sepsis cluster, bundle clock — and each clears
  [8] liveness     with simulated staff the queue never sticks; a present clinician answers for themself

    .venv/bin/python -m backend.aegis.paging.verify_paging
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from backend.aegis.paging import policy
from backend.aegis.paging.engine import PagingOS
from backend.aegis.paging.hospital import Clinician, ICU_ROOMS
from backend.aegis.paging.router import rank
from backend.aegis.paging.triage import LLM_TIMEOUT_S, extract_vitals, triage_text

_n = {"pass": 0}


def ok(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    _n["pass"] += 1
    print(f"      [PASS] {msg}")


class Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def fresh(sim: bool = False) -> tuple:
    clk = Clock()
    os_ = PagingOS(clock=clk, seed=5, demo_history=False)
    os_.sync_sbar = True
    os_.settings["simulated_responders"] = sim
    os_.settings["demo_traffic"] = False
    return os_, clk


class FakeModel:
    def __init__(self, reply: Optional[str], delay: float = 0.0) -> None:
        self.reply, self.delay, self.calls, self._last_model = reply, delay, 0, "fake"

    def available(self) -> bool:
        return True

    def budget_remaining(self) -> int:
        return 999

    def chat(self, system: str, user: str, **_: Any) -> Optional[str]:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return self.reply


# --------------------------------------------------------------------------- #
def test_triage() -> None:
    print("[1] triage: rules set a floor, the model may only raise it")
    cases = [("Code blue room 307, unresponsive", 1, "307"),
             ("STEMI suspected, ST elevation V2-V4, BP 92/58 HR 118 in 409", 1, "409"),
             ("A4 sats 84% on 4L, RR 32", 1, "A4"),
             ("512 BP 84/50, febrile 39.2, new confusion", 1, "512"),
             ("603 new facial droop, NIHSS 14, last known well 35 min ago", 1, "603"),
             ("Family in 503 asking about discharge plans", 4, "503")]
    for text, pr, room in cases:
        t = triage_text(text)
        ok(t.priority == pr and t.room == room, f"P{pr} room {room}: {text[:44]!r} (got P{t.priority} {t.room})")
    ok(triage_text("HR 104, BP 130/80 in bed 507").room == "507", "a heart rate of 104 is not room 104")
    for text, spo2 in (("605 desaturating to 84% on room air", 84), ("409 is 86% on RA", 86), ("O2 at 15 L NRB, sats 91%", 91)):
        got = extract_vitals(text).get("spo2")
        ok(got == spo2, f"SpO2 {spo2} read from {text!r} (got {got})")
    code_model = FakeModel('{"priority": 4, "category": "general", "rationale": "sounds routine"}')
    t = triage_text("code blue in 307", client=code_model)
    ok(t.priority == 1 and code_model.calls == 0, "a P1 red flag goes out on the rule; the model is never asked")
    lower = FakeModel('{"priority": 4, "category": "general", "rationale": "low concern"}')
    t = triage_text("BP 86/50 in 512, febrile 38.9", client=lower)
    ok(t.priority <= 2, f"the model cannot argue a red flag down (P{t.priority})")
    raise_ = FakeModel('{"priority": 2, "category": "cardiac", "rationale": "jaw pain in a diabetic may be ACS"}')
    t = triage_text("503 jaw discomfort, diabetic, feels off", client=raise_)
    ok(t.priority == 2 and t.source == "llm+rules", "the model may raise a priority the rules missed")
    class PropertyModel(FakeModel):          # the real LLMProvider exposes `available` as a property
        available = True                     # type: ignore[assignment]
    prop = PropertyModel('{"priority": 2, "category": "cardiac", "rationale": "possible ACS"}')
    t = triage_text("503 jaw discomfort, diabetic, feels off", client=prop)
    ok(prop.calls == 1 and t.priority == 2, "a provider whose `available` is a property is consulted too")
    slow = FakeModel('{"priority": 1}', delay=LLM_TIMEOUT_S + 2)
    t0 = time.time()
    t = triage_text("503 mild nausea", client=slow)
    ok(time.time() - t0 < LLM_TIMEOUT_S + 1.0 and t.source == "rules", "a slow model times out and the rules stand")


def test_refinement() -> None:
    print("[1b] refinement: the page goes out on the rules; the model may only raise it afterwards")
    def engine(model: Any) -> tuple:
        clk = Clock()
        os_ = PagingOS(llm=model, clock=clk, seed=5, demo_history=False)
        os_.sync_sbar = os_.sync_refine = True
        os_.settings["simulated_responders"] = os_.settings["demo_traffic"] = False
        return os_, clk
    up = FakeModel('{"priority": 2, "category": "cardiac", "rationale": "atypical ACS in a diabetic"}')
    os_, clk = engine(up)
    p = os_.raise_alert("503 diabetic with jaw discomfort and nausea, feels off", source="nurse")["page"]
    ok(p["priority"] == 3 and p["status"] == "paging", "dispatched at once on the rules (P3), before the model answers")
    live = os_.pages[p["id"]]
    att = live["attempts"][-1]
    ok(live["priority"] == 2 and live["refinement"]["outcome"] == "raised"
       and att["deadline"] - att["sent_at"] == policy.ack_timeout(os_.settings, 2),
       "the model raised it to P2 and the page now rings on the P2 timeout")
    move = FakeModel('{"priority": 2, "category": "cardiac", "rationale": "chest tightness after a dosing error"}')
    os_, clk = engine(move)
    p = os_.raise_alert("503 wrong dose of insulin given, now tight chest", source="nurse")["page"]
    first = p["assignee"]
    live = os_.pages[p["id"]]
    ok(os_.roster[first].specialty == "pharmacy" and live["assignee"] != first
       and os_.roster[live["assignee"]].specialty != "pharmacy",
       "a raised page moves from pharmacy to a service that covers the new category")
    down = FakeModel('{"priority": 4, "category": "general", "rationale": "routine"}')
    os_, clk = engine(down)
    p = os_.raise_alert("503 fall, no head strike", source="nurse")["page"]
    live = os_.pages[p["id"]]
    ok(live["priority"] == 3 and live["refinement"]["outcome"] == "kept", "a sent page is never lowered by the model")
    silent = FakeModel(None)
    os_, clk = engine(silent)
    p = os_.raise_alert("503 fall, no head strike", source="nurse")["page"]
    ok(os_.pages[p["id"]]["refinement"]["outcome"] == "unavailable", "a silent model leaves the rules standing, and says so")
    red = FakeModel('{"priority": 1}')
    os_, clk = engine(red)
    os_.raise_alert("code blue 505", source="nurse")
    ok(red.calls == 0, "a P1 red flag is never sent to the model")


# --------------------------------------------------------------------------- #
def _staff(**over: Dict[str, Any]) -> Dict[str, Clinician]:
    base = {
        "a": Clinician("a", "Dr. A", "attending", "hospitalist", "5e", "Hospitalist", True, "day", "available", "5e"),
        "b": Clinician("b", "Dr. B", "attending", "hospitalist", "5e", "Hospitalist", False, "day", "available", "5e"),
        "c": Clinician("c", "Dr. C", "attending", "cardiology", "tele", "Cardiologist", True, "day", "available", "tele"),
    }
    for k, fields in over.items():
        for f, v in fields.items():
            setattr(base[k], f, v)
    return base


def _load(**over: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {k: {"active": 0, "last_hour": 0, "last_paged_at": None, "last_case": None} for k in "abc"}
    for k, v in over.items():
        out[k].update(v)
    return out


def test_routing() -> None:
    print("[2] routing: specialty, guardrails, near-tie, hold")
    s = policy.DEFAULT_SETTINGS
    now = 1_800_000_000.0
    d = rank(2, "cardiac", "503", _staff(), _load(), s, "X", now)
    ok(d.assignee == "c", "chest pain goes to cardiology, not the closer hospitalist")
    d = rank(3, "general", "503", _staff(), _load(a={"last_paged_at": now - 60, "last_case": "OTHER"}), s, "X", now)
    ok(d.assignee == "b" and any("Consecutive" in g for g in d.guardrails),
       "consecutive paging: someone paged a minute ago for another case is skipped")
    d = rank(3, "general", "503", _staff(), _load(a={"last_hour": 4}), s, "X", now)
    ok(d.assignee == "b" and any("Workload" in g for g in d.guardrails), "workload: a P3 skips someone with 4 pages this hour")
    d = rank(2, "general", "503", _staff(), _load(a={"last_hour": 4}), s, "X", now)
    ok(d.assignee in ("a", "b") and not any("Workload" in g for g in d.guardrails), "the workload limit does not apply to P1/P2")
    # a: on call (+10) but 2 pages this hour (-6) -> 129 vs b 125: within 10%, b is lighter
    d = rank(3, "general", "503", _staff(), _load(a={"last_hour": 2}), s, "X", now)
    ok(d.assignee == "b" and any("Near-tie" in g for g in d.guardrails), "near-tie: the lighter caseload gets it")
    d = rank(3, "general", "503", _staff(a={"status": "off_shift"}), _load(), s, "X", now)
    ok(d.assignee == "b", "off-shift staff are excluded")
    only_off = _staff(a={"status": "off_shift"}, b={"status": "off_shift"})
    d = rank(3, "general", "503", only_off, _load(), s, "X", now)
    ok(d.mode == "unassigned", "a P3 with only off-shift staff is left for the operator")
    d = rank(1, "general", "503", only_off, _load(), s, "X", now)
    ok(d.assignee in ("a", "b") and not d.on_shift_alternative, "a P1 with only off-shift staff pages one (last resort)")
    one_busy = _staff(a={"status": "in_procedure", "busy_until": now + 300})
    d = rank(4, "general", "503", one_busy, _load(), s, "X", now)
    ok(d.assignee == "b" and d.mode == "dispatch", "a P4 never interrupts a procedure when someone free covers it")
    busy = _staff(a={"status": "in_procedure", "busy_until": now + 300}, b={"status": "in_procedure",
                                                                           "busy_until": now + 900})
    d = rank(3, "general", "503", busy, _load(), s, "X", now)
    ok(d.mode == "hold" and d.hold_until == now + 300 and d.assignee == "a",
       "time-queued: with nobody free, a P3 waits for the first one out of a procedure")
    d = rank(2, "general", "503", busy, _load(), s, "X", now)
    ok(d.mode == "dispatch", "a P2 does not wait: it pages even someone in a procedure")


# --------------------------------------------------------------------------- #
def test_certificate() -> None:
    print("[3] certificate: good pages proven, planted bad ones refused")
    os_, clk = fresh()
    r = os_.raise_alert("503 c/o chest pain 6/10, BP 150/90", source="nurse", reporter="rn.test")["page"]
    ok(r["status"] == "paging" and r["certificate"]["z3_status"] == "unsat", "a ward P2 is certified (unsat) and sent")
    r = os_.raise_alert("Code blue 305", source="nurse")["page"]
    ok(r["status"] == "review" and r["requires_review"], "an ICU P1 waits for the operator (policy)")
    # planted: send an ICU P1 without review
    page = os_.pages[r["id"]]
    ok_sent = os_._send(page, page["planned"], clk())
    ok(not ok_sent and "autonomy" in page["blocked"], "planted: sending an unreviewed ICU P1 is refused (autonomy)")
    # planted: force an off-shift clinician while on-shift ones exist
    os_2, clk2 = fresh()
    p = os_2.raise_alert("503 c/o chest pain 6/10", source="nurse")["page"]
    page = os_2.pages[p["id"]]
    os_2.roster["dr.brennan"].specialty = "cardiology"         # an off-shift clinician who COULD take it
    page["decision"]["on_shift_alternative"] = True
    sent = os_2._send(page, "dr.brennan", clk2())
    ok(not sent and "on_shift" in page["blocked"] and page["certificate"]["z3_status"] == "sat",
       "planted: paging an off-shift clinician with on-shift alternatives is refused (sat, on_shift)")
    # approving the ICU P1 certifies it
    res = os_.approve(r["id"], "rn.patel")
    ok(res["status"] == "ok" and res["page"]["status"] == "paging", "operator approval certifies and sends the ICU P1")


# --------------------------------------------------------------------------- #
def test_queue() -> None:
    print("[4] queue: decline, timeout, exhaustion, fail-safe, hold, cancel")
    os_, clk = fresh()
    p = os_.raise_alert("512 temp 38.9, HR 118, RR 24, new confusion", source="nurse")["page"]
    first = p["assignee"]
    os_.respond(p["id"], first, "decline", "with another patient")
    p = os_.pages[p["id"]]
    ok(p["status"] == "paging" and p["assignee"] != first and len(p["attempts"]) == 2,
       "a decline moves the page to the next backup")
    second = p["assignee"]
    clk.advance(policy.ack_timeout(os_.settings, p["priority"]) + 1)
    os_.tick()
    ok(os_.pages[p["id"]]["assignee"] not in (first, second), "no answer in time escalates again")
    for _ in range(6):
        clk.advance(policy.ack_timeout(os_.settings, p["priority"]) + 1)
        os_.tick()
    ok(os_.pages[p["id"]]["status"] == "unassigned", "an exhausted chain lands with the operator, not in silence")
    os_, clk = fresh()
    p = os_.raise_alert("Code blue 305", source="nurse")["page"]
    clk.advance(policy.DEFAULT_SETTINGS["review_failsafe_s"]["1"] + 1)
    os_.tick()
    p = os_.pages[p["id"]]
    ok(p["status"] == "paging" and p["failsafe"] and p["certificate"]["certified"],
       "review never blocks care: an unreviewed P1 goes out on the fail-safe, still certified")
    # the planned clinician is paged for ANOTHER case while this P1 waits in review: at
    # the fail-safe the consecutive-paging invariant refuses them, and the page must
    # route around them, not wait in review forever
    os_, clk = fresh()
    p = os_.raise_alert("Code blue 305", source="nurse")["page"]
    planned = p["planned"]
    other = os_.raise_alert("Code blue 307", source="nurse")["page"]
    os_.approve(other["id"], "rn.patel", planned)
    ok(os_.pages[other["id"]]["assignee"] == planned, "setup: the planned clinician is now ringing for another case")
    clk.advance(policy.DEFAULT_SETTINGS["review_failsafe_s"]["1"] + 1)
    os_.tick()
    p = os_.pages[p["id"]]
    ok(p["status"] == "paging" and p["assignee"] not in (None, planned) and p["certificate"]["certified"],
       f"a P1 whose first pick became unavailable is re-routed at the fail-safe (got {p['status']}, "
       f"{p['assignee']}, blocked={p['blocked']})")
    os_, clk = fresh()
    for c in os_.roster.values():            # nobody free anywhere; hospitalists out first
        if c.pageable:
            c.status, c.busy_until = "in_procedure", clk() + (240 if c.specialty == "hospitalist" else 3000)
    p = os_.raise_alert("503 requesting something to help sleep", source="nurse")["page"]
    ok(p["status"] == "held", "a P4 with nobody free is held for the right person")
    clk.advance(300)
    os_.tick()
    ok(os_.pages[p["id"]]["status"] == "paging", "the hold releases when they are free")
    who = os_.pages[p["id"]]["assignee"]
    os_.cancel(p["id"], "rn.patel")
    ok(os_.pages[p["id"]]["status"] == "cancelled" and os_.roster[who].status != "paging",
       "cancel closes the page and frees the clinician")


def test_brief() -> None:
    print("[5] brief: SBAR on accept")
    os_, clk = fresh()
    p = os_.raise_alert("512 BP 86/50, HR 124, temp 39.4, urine output dropping", source="nurse")["page"]
    os_.respond(p["id"], p["assignee"], "accept")
    b = os_.pages[p["id"]]["sbar"]
    ok(b is not None and all(b.get(k) for k in ("situation", "background", "assessment", "request")),
       "accepting writes all four SBAR sections")
    ok(b["words"] <= policy.DEFAULT_SETTINGS["max_escalations"] * 0 + 100, f"under 100 words ({b['words']})")
    ok("Hour-1 bundle" in b["request"], "a sepsis P1 brief asks for the Hour-1 bundle")
    rec = os_.synthetic.get("512")
    ok(rec is None or "synthetic" in b["background"], "a synthetic record says it is synthetic")
    os_.resolve(p["id"], p["assignee"])
    ok(os_.roster[p["assignee"]].status == "available", "resolving frees the clinician")


# --------------------------------------------------------------------------- #
def _bed(i: int, risk: float, mp: float = 72.0) -> Dict[str, Any]:
    return {"bed_id": f"bed_{i}", "name": f"Patient {i}", "fused_risk": risk, "population_risk": risk,
            "age": 70, "gender": "F", "mrn": f"P{i}", "icu_hour": 10,
            "enterprise": {"map": mp, "lactate": 1.5}, "drivers": ["tachycardia"]}


def test_detector() -> None:
    print("[6] detector: one page per episode, upgrade, hysteresis, watch")
    os_, clk = fresh()
    beds = [_bed(i, 0.2) for i in range(12)]
    beds[4] = _bed(4, 0.75)
    for _ in range(5):
        os_.observe_ward(beds)
        clk.advance(2)
    det = [p for p in os_.pages.values() if p["source"] == "detector"]
    ok(len(det) == 1 and det[0]["room"] == ICU_ROOMS[4] and det[0]["priority"] == 2,
       "a bed above the alarm line opens exactly one P2 page across five ticks")
    beds[4] = _bed(4, 0.8, mp=58.0)
    os_.observe_ward(beds)
    ok(os_.pages[det[0]["id"]]["priority"] == 1 and len([p for p in os_.pages.values() if p["source"] == "detector"]) == 1,
       "septic-shock criteria (MAP < 65) upgrade the same page to P1")
    beds[4] = _bed(4, 0.5)
    for _ in range(3):
        os_.observe_ward(beds)
    ok(not os_.episodes[ICU_ROOMS[4]]["open"], "three readings below 0.60 close the episode")
    beds[4] = _bed(4, 0.72)
    os_.observe_ward(beds)
    det = [p for p in os_.pages.values() if p["source"] == "detector"]
    ok(len(det) == 1 and os_.episodes[ICU_ROOMS[4]]["open"]
       and any("re-crossed" in e["why"] for e in det[0]["escalation_log"]),
       "re-crossing while the first page is still open writes on THAT page: one patient, one page")
    os_.cancel(det[0]["id"], "rn.patel", "seen")
    beds[4] = _bed(4, 0.5)
    for _ in range(3):
        os_.observe_ward(beds)
    beds[4] = _bed(4, 0.72)
    os_.observe_ward(beds)
    ok(len([p for p in os_.pages.values() if p["source"] == "detector"]) == 2,
       "once that page is closed, a new episode pages again")
    beds[7] = _bed(7, 0.65)
    os_.observe_ward(beds)
    ok(os_.monitor_state()[ICU_ROOMS[7]]["watch"] and not any(p["room"] == ICU_ROOMS[7] for p in os_.pages.values()),
       "between 0.60 and 0.70 a bed is on watch and pages nobody")


# --------------------------------------------------------------------------- #
def test_sentinel() -> None:
    print("[7] sentinel: patterns surface, then clear")
    os_, clk = fresh()
    for r in ("503", "505", "507"):
        os_.raise_alert(f"{r} fall, no head strike", source="nurse")
    os_.last_sentinel = 0
    os_.tick()
    kinds = {r["kind"] for r in os_.sentinel.active()}
    ok("zone_surge" in kinds, "three alerts on 5 East in 20 min raise a zone surge")
    clk.advance(40 * 60)
    os_.last_sentinel = 0
    os_.tick()
    ok("zone_surge" not in {r["kind"] for r in os_.sentinel.active()}, "the surge clears when the window passes")
    os_, clk = fresh()
    for _ in range(4):                      # one patient alarming over and over is not a short unit
        os_.raise_alert("503 fall, no head strike", source="nurse")
    os_._note("503: the detector closed its episode", "503")
    os_.last_sentinel = 0
    os_.tick()
    ok("zone_surge" not in {r["kind"] for r in os_.sentinel.active()},
       "repeat alerts from ONE room (plus a detector note) are that patient's episode, not a unit surge")
    os_, clk = fresh()
    beds = [_bed(i, 0.2) for i in range(12)]
    beds[0], beds[1] = _bed(0, 0.76), _bed(1, 0.74)
    os_.observe_ward(beds)
    os_.last_sentinel = 0
    os_.tick()
    ok("sepsis_cluster" in {r["kind"] for r in os_.sentinel.active()}, "two detector alerts in the MICU raise a cluster warning")
    os_, clk = fresh()
    started = {"bed_2": False}
    os_.bundle_started = lambda bed: started.get(bed)
    beds = [_bed(i, 0.2) for i in range(12)]
    beds[2] = _bed(2, 0.78)
    os_.observe_ward(beds)
    p = next(p for p in os_.pages.values() if p["source"] == "detector")
    os_.approve(p["id"], "rn.patel")
    p = os_.pages[p["id"]]
    os_.respond(p["id"], p["assignee"], "accept")
    clk.advance(35 * 60)
    os_.last_sentinel = 0
    os_.tick()
    ok(any(r["kind"] == "bundle_clock" for r in os_.sentinel.active()), "an accepted sepsis page with no antibiotic order starts the bundle clock")
    started["bed_2"] = True
    os_.last_sentinel = 0
    os_.tick()
    ok(not any(r["kind"] == "bundle_clock" for r in os_.sentinel.active()), "the bundle clock clears once the order is certified")
    os_, clk = fresh()
    load_page = os_.raise_alert("409 chest pain", source="nurse")["page"]
    who = load_page["assignee"]
    for i in range(5):
        os_.pages[load_page["id"]]["attempts"].append({"clinician": who, "name": who, "sent_at": clk() - i * 60,
                                                       "deadline": clk(), "outcome": "accepted", "at": clk()})
    os_.last_sentinel = 0
    os_.tick()
    ok(any(r["kind"] == "overload" for r in os_.sentinel.active()), "one clinician with six pages this hour is flagged")


# --------------------------------------------------------------------------- #
def test_liveness() -> None:
    print("[8] liveness: simulated staff keep the queue moving; a present clinician answers for themself")
    os_, clk = fresh(sim=True)
    ids = [os_.raise_alert(t, source="nurse")["page"]["id"] for t in (
        "503 fall, no head strike", "409 chest pain 5/10", "512 temp 38.9 HR 118 RR 24 new confusion",
        "605 requesting sleep aid", "A3 laceration 4 cm, bleeding controlled")]
    # the longest chain is a P4: four attempts x an 8-minute ring
    horizon = int(policy.ack_timeout(os_.settings, 4) * (os_.settings["max_escalations"] + 1) + 300)
    for _ in range(horizon):
        clk.advance(1)
        os_.tick()
    stuck = [i for i in ids if os_.pages[i]["status"] in ("paging", "held")]
    ok(not stuck, f"within the longest escalation chain ({horizon // 60} simulated min) no page is still ringing ({stuck})")
    closed = [i for i in ids if os_.pages[i]["status"] in ("resolved", "accepted", "unassigned", "review")]
    ok(len(closed) == len(ids), "every page reached an owner or the operator")
    os_, clk = fresh(sim=True)
    p = os_.raise_alert("503 fall, no head strike", source="nurse")["page"]
    os_.presence_join(p["assignee"])
    first = p["assignee"]
    for _ in range(int(policy.ack_timeout(os_.settings, p["priority"])) - 1):
        clk.advance(1)
        os_.tick()
    ok(os_.pages[p["id"]]["status"] == "paging" and os_.pages[p["id"]]["assignee"] == first,
       "while their device is open, nobody answers for the clinician")
    ev = os_.drain()
    ok(any(e["type"] == "page" for e in ev) and not os_.drain(), "state changes are emitted as events, and drained once")


def main() -> None:
    print("=" * 72)
    print("AegisOS paging OS — verification")
    print("=" * 72)
    for t in (test_triage, test_refinement, test_routing, test_certificate, test_queue, test_brief, test_detector,
              test_sentinel, test_liveness):
        t()
    print("-" * 72)
    print(f"ALL {_n['pass']} PAGING CHECKS PASSED")


if __name__ == "__main__":
    main()
