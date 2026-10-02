"""Verification for the room monitor.

A check that has never failed is not evidence, so every record-aware claim is checked as a
PAIR: the same vitals with and without the piece of the record that should change the
reading, and the reading has to differ.

  [1] NEWS2        known answers; scale 2 only when the record prescribes it; confusion
                   that is the patient's baseline does not score; the single-red band
  [2] screens      qSOFA, SIRS, shock index, MAP, against hand-computed answers
  [3] baseline     the same systolic alarms against one patient's record and not another's
  [4] medication   allergy conflict vs cross-reactivity; expected regimens; duplication;
                   QT; renal dosing from Cockcroft–Gault; reconciliation
  [5] conditional  opioid sedation, neutropenic fever, hypoglycaemia, AF, bleeding, KDIGO:
                   each fires only when the record makes it possible
  [6] firewall     a replayed ICU record never reads a later hour or the sepsis label
  [7] monitor      calm at rest; a deterioration pages once, routes, recovers and closes;
                   a nurse's report becomes the room's measurement; ICU rooms are not paged
                   by the monitor; a broken detector cannot blind the room

    .venv/bin/python -m backend.aegis.rooms.verify_rooms
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.aegis.paging.engine import ACTIVE, PagingOS
from backend.aegis.rooms import detectors as D
from backend.aegis.rooms.formulary import FORMULARY, creatinine_clearance
from backend.aegis.rooms.records import stay_to_date

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


def order(name: str, status: str = "active") -> Dict[str, Any]:
    d = FORMULARY[name]
    return {"name": d.name, "generic": d.generic, "dose": d.dose, "route": d.route, "freq": d.freq,
            "classes": sorted(d.classes), "prn": d.prn, "infusion": d.infusion, "renal": d.renal, "qt": d.qt,
            "kind": "drug", "status": status}


def record(**kw: Any) -> Dict[str, Any]:
    r = {"room": "999", "kind": "synthetic", "provenance": {"synthetic": True},
         "patient": {"name": "Test, A.", "age": 60, "sex": "M", "mrn": "S000000", "weight_kg": 80},
         "admission": {"reason": "Community-acquired pneumonia"}, "problems": [], "orders": [], "home_meds": [],
         "allergies": [], "code_status": "Full code", "encounters": [], "baseline": None, "spo2_target": None,
         "planned": [], "labs": {}, "flags": []}
    r.update(kw)
    return r


def ctx(rec: Dict[str, Any], v: Dict[str, Any], labs: Optional[Dict[str, Any]] = None, kind: str = "synthetic",
        series: Optional[List[Dict[str, Any]]] = None, now: float = 10_000.0, ml: Optional[Dict[str, Any]] = None) -> D.Ctx:
    return D.Ctx("999", kind, rec, v, now, series or [], labs or rec.get("labs") or {}, now, ml=ml,
                 flags=frozenset(rec.get("flags") or []))


def keys(found: List[D.Finding]) -> Dict[str, D.Finding]:
    return {f.key: f for f in found}


NORMAL = {"rr": 16, "spo2": 97, "o2_lpm": 0.0, "sbp": 124, "hr": 76, "avpu": "A", "temp_c": 36.8, "map": 90}


# --------------------------------------------------------------------------- #
def test_news2() -> None:
    print("[1] NEWS2: known answers, and the record changes the reading")
    n = D.news2_score(NORMAL, False, False)
    ok(n["score"] == 0 and n["band"] == "low" and n["complete"], "normal vitals score 0, band low, all 7 parameters present")
    v = {"rr": 24, "spo2": 95, "o2_lpm": 0.0, "sbp": 105, "hr": 115, "avpu": "A", "temp_c": 38.5}
    n = D.news2_score(v, False, False)
    ok(n["score"] == 2 + 1 + 0 + 1 + 2 + 0 + 1 == 7 and n["band"] == "high",
       "RR 24, SpO₂ 95, air, SBP 105, HR 115, alert, 38.5 °C scores 7 (2+1+0+1+2+0+1): high")
    copd = {**NORMAL, "spo2": 89, "o2_lpm": 2.0}
    with_target = record(spo2_target={"lo": 88, "hi": 92, "scale": 2, "why": "documented CO₂ retention"},
                         flags=["copd", "hypercapnic"])
    without = record(flags=["copd"])
    a = keys(D.detect_news2(ctx(with_target, copd)))["news2"]
    b = keys(D.detect_news2(ctx(without, copd)))["news2"]
    ok(a.score == 2 and b.score == 5,
       f"SpO₂ 89% on 2 L: scale 2 from the record scores {a.score}, the same vitals without the target score {b.score}")
    ok(any("Scale 1 would read 5" in x for x in a.context), "the scale-2 reading says what scale 1 would have read")
    conf = {**NORMAL, "avpu": "C"}
    ok(D.news2_score(conf, False, True)["score"] == 0 and D.news2_score(conf, False, False)["score"] == 3,
       "confusion documented as the patient's baseline scores 0; new confusion scores 3")
    red = D.news2_score({**NORMAL, "rr": 7}, False, False)
    ok(red["score"] == 3 and red["band"] == "low-medium", "a single red parameter (RR 7) on an aggregate of 3 is the low-medium band")
    hi = keys(D.detect_news2(ctx(record(), v)))["news2"]
    ok(hi.severity == "critical" and hi.page == 2, "NEWS2 ≥ 7 is critical and pages P2")
    part = {k: x for k, x in v.items() if k != "sbp"}
    inc = keys(D.detect_news2(ctx(record(), part)))["news2"]
    ok(inc.page is None and any("systolic BP" in c for c in inc.context),
       "an incomplete NEWS2 (no systolic) is shown with what is missing and never pages")
    ok(D.detect_news2(ctx(record(kind="replay"), v, kind="replay")) == [], "NEWS2 is not run in the ICU replay rooms")


def test_screens() -> None:
    print("[2] sepsis screens and haemodynamics")
    q = keys(D.detect_qsofa(ctx(record(flags=["infection"]), {**NORMAL, "rr": 22, "sbp": 100, "avpu": "V"})))
    ok(q["qsofa"].score == 3 and q["qsofa"].severity == "critical" and q["qsofa"].page == 2,
       "qSOFA 3 (RR 22, SBP 100, voice) with infection on the record: critical, pages P2 sepsis")
    ok(D.detect_qsofa(ctx(record(), {**NORMAL, "rr": 21, "sbp": 101})) == [], "RR 21 and SBP 101 meet no qSOFA criterion")
    no_inf = keys(D.detect_qsofa(ctx(record(admission={"reason": "Syncope"}), {**NORMAL, "rr": 24, "sbp": 96})))["qsofa"]
    ok(no_inf.page is None and no_inf.severity == "urgent", "qSOFA 2 without infection context is urgent but does not page as sepsis")
    s = keys(D.detect_sirs(ctx(record(flags=["infection"]), {**NORMAL, "temp_c": 38.6, "hr": 104, "rr": 24})))["sirs"]
    ok(s.score == 3 and s.severity == "watch", "SIRS 3 of 4 with infection is a watch")
    s2 = keys(D.detect_sirs(ctx(record(admission={"reason": "Syncope"}), {**NORMAL, "hr": 104, "rr": 24})))["sirs"]
    ok(s2.severity == "info", "SIRS without an infection on the record is information only")
    si = keys(D.detect_shock_index(ctx(record(), {**NORMAL, "hr": 120, "sbp": 100})))["shock_index"]
    ok(abs(si.score - 1.2) < 1e-9 and si.severity == "urgent", "HR 120 / SBP 100 = shock index 1.20, urgent")
    mp = keys(D.detect_shock_index(ctx(record(), {**NORMAL, "map": 61})))
    ok("map_floor" in mp and mp["map_floor"].page == 2, "MAP 61 is below the 65 target and pages P2")


def test_baseline() -> None:
    print("[3] personal baseline: the same number means different things in different records")
    v = {**NORMAL, "sbp": 108}
    hyper = record(baseline={"sbp": 162, "hr": 74, "rr": 15, "spo2": 97, "temp_c": 36.7, "source": "mean of 2 earlier encounters"})
    usual = record(baseline={"sbp": 116, "hr": 74, "rr": 15, "spo2": 97, "temp_c": 36.7, "source": "mean of 2 earlier encounters"})
    a = keys(D.detect_baseline(ctx(hyper, v)))
    b = keys(D.detect_baseline(ctx(usual, v)))
    ok("baseline:sbp" in a and a["baseline:sbp"].severity == "urgent" and "baseline:sbp" not in b,
       "SBP 108 alarms in a patient whose baseline is 162 (−33%) and not in one whose baseline is 116")
    ok(any("Above 100" in c for c in a["baseline:sbp"].context), "the finding says population thresholds would call it normal")
    ok(D.detect_baseline(ctx(record(), v)) == [], "no baseline on the record, no baseline finding")


def test_medication() -> None:
    print("[4] medication safety: the orders read against the record")
    pen = [{"substance": "penicillin", "reaction": "anaphylaxis", "severity": "severe", "kind": "allergy"}]
    a = keys(D.detect_medications(ctx(record(allergies=pen, orders=[order("piperacillin-tazobactam")]), NORMAL)))
    ok("med:allergy:piperacillin-tazobactam" in a and a["med:allergy:piperacillin-tazobactam"].severity == "critical"
       and a["med:allergy:piperacillin-tazobactam"].page == 3 and a["med:allergy:piperacillin-tazobactam"].page_category == "medication",
       "piperacillin-tazobactam with a penicillin allergy: critical, pages pharmacy (P3)")
    b = keys(D.detect_medications(ctx(record(orders=[order("piperacillin-tazobactam")]), NORMAL)))
    ok(not any(k.startswith("med:allergy") for k in b), "the same order with no allergy on the record raises nothing")
    c = keys(D.detect_medications(ctx(record(allergies=pen, orders=[order("ceftriaxone")]), NORMAL)))
    ok("med:cross:ceftriaxone" in c and c["med:cross:ceftriaxone"].severity == "watch" and c["med:cross:ceftriaxone"].page is None,
       "ceftriaxone with a penicillin anaphylaxis is a cross-reactivity watch, not a contraindication")
    mild = [{**pen[0], "reaction": "rash", "severity": "mild"}]
    c2 = keys(D.detect_medications(ctx(record(allergies=mild, orders=[order("ceftriaxone")]), NORMAL)))
    ok(c2["med:cross:ceftriaxone"].severity == "info", "after a mild rash it is information")
    acs = record(admission={"reason": "NSTEMI, awaiting catheterization"}, orders=[order("heparin infusion"), order("aspirin")])
    oth = record(admission={"reason": "Cellulitis"}, orders=[order("heparin infusion"), order("aspirin")])
    sa = [f for f in D.detect_medications(ctx(acs, NORMAL)) if "antiplatelet" in f.title.lower()]
    sb = [f for f in D.detect_medications(ctx(oth, NORMAL)) if "antiplatelet" in f.title.lower()]
    ok(sa and sb and sa[0].severity == "info" and sb[0].severity == "watch",
       "heparin with aspirin is the expected regimen in an NSTEMI (info) and a watch elsewhere")
    dup = [f for f in D.detect_medications(ctx(record(orders=[order("heparin infusion"), order("enoxaparin")]), NORMAL))
           if f.title.startswith("Two anticoagulants")]
    ok(dup and dup[0].severity == "urgent", "heparin infusion with enoxaparin is a duplicate anticoagulant (urgent)")
    qt = keys(D.detect_medications(ctx(record(orders=[order("azithromycin"), order("ondansetron")]), NORMAL)))
    ok("med:qt" in qt, "azithromycin with ondansetron: two QT-prolonging drugs")
    crcl = creatinine_clearance(80, 60, 2.0, True)
    ok(abs(crcl - (140 - 80) * 60 / (72 * 2.0) * 0.85) < 1e-9, f"Cockcroft–Gault: 80 y, 60 kg, Cr 2.0, female = {crcl:.1f} mL/min")
    pt = {"name": "T", "age": 80, "sex": "F", "weight_kg": 60}
    low = keys(D.detect_medications(ctx(record(patient=pt, orders=[order("enoxaparin")],
                                               labs={"creatinine": [{"t": 1.0, "v": 2.0}]}), NORMAL)))
    high = keys(D.detect_medications(ctx(record(patient=pt, orders=[order("enoxaparin")],
                                                labs={"creatinine": [{"t": 1.0, "v": 0.8}]}), NORMAL)))
    ok("med:renal:enoxaparin" in low and "med:renal:enoxaparin" not in high,
       "enoxaparin flags for renal dosing at CrCl 30 (Cr 2.0) and not at CrCl 74 (Cr 0.8)")
    home = [{"name": "levetiracetam", "dose": "500 mg", "freq": "BID"}]
    r1 = keys(D.detect_medications(ctx(record(home_meds=home, orders=[order("acetaminophen")]), NORMAL)))
    r2 = keys(D.detect_medications(ctx(record(home_meds=home, orders=[order("levetiracetam")]), NORMAL)))
    ok("med:recon:levetiracetam" in r1 and "med:recon:levetiracetam" not in r2,
       "home levetiracetam not continued is flagged; continued, it is not")
    held = keys(D.detect_medications(ctx(record(allergies=pen, orders=[order("piperacillin-tazobactam", "held")]), NORMAL)))
    ok(not any(k.startswith("med:allergy") for k in held), "an order already held is not an active conflict")


def test_conditional() -> None:
    print("[5] detectors the record makes possible")
    sed = {**NORMAL, "rr": 7, "spo2": 86, "avpu": "V"}
    a = keys(D.detect_opioid(ctx(record(orders=[order("hydromorphone PCA")], flags=["opioid"]), sed)))
    ok("opioid_rd" in a and a["opioid_rd"].severity == "critical" and a["opioid_rd"].page == 1,
       "RR 7, rousable to voice, on a PCA: critical, pages P1")
    ok(D.detect_opioid(ctx(record(), sed)) == [], "the same vitals with no opioid order: no opioid finding")
    fn = {**NORMAL, "temp_c": 38.5}
    chemo = record(flags=["chemo", "neutropenia"], admission={"reason": "Febrile neutropenia"})
    x = keys(D.detect_neutropenic_fever(ctx(chemo, fn)))["neutropenic_fever"]
    ok(x.severity == "critical" and x.page == 2, "38.5 °C on chemotherapy with no antibiotic: critical, pages P2")
    y = keys(D.detect_neutropenic_fever(ctx(record(flags=["chemo", "neutropenia"], orders=[order("cefepime")]), fn)))["neutropenic_fever"]
    ok(y.severity == "urgent" and y.page == 3, "already on cefepime: an urgent review (P3), not the 60-minute emergency")
    ok(D.detect_neutropenic_fever(ctx(record(), fn)) == [], "the same fever without chemotherapy on the record: none")
    g = lambda v: keys(D.detect_glucose(ctx(record(), NORMAL, labs={"glucose": [{"t": 9_900.0, "v": v}]})))
    ok(g(48)["glucose"].severity == "critical" and g(65)["glucose"].severity == "urgent" and "glucose" not in g(120),
       "glucose 48 is level-2 hypoglycaemia, 65 level 1, 120 nothing")
    stale = keys(D.detect_glucose(ctx(record(), NORMAL, labs={"glucose": [{"t": 1_000.0, "v": 48}]})))
    ok("glucose" not in stale, "a glucose result older than an hour is not a current finding")
    af = keys(D.detect_af(ctx(record(flags=["af"]), {**NORMAL, "hr": 146})))
    ok(af["af_rvr"].severity == "critical" and af["af_rvr"].page == 2, "AF on the record at 146/min: critical")
    ok(D.detect_af(ctx(record(), {**NORMAL, "hr": 146})) == [], "no AF on the record: the rhythm detector does not run")
    hb = {"hemoglobin": [{"t": 1_000.0, "v": 12.4}, {"t": 9_000.0, "v": 9.9}]}
    bl = keys(D.detect_bleeding(ctx(record(orders=[order("enoxaparin")]), NORMAL, labs=hb)))
    ok("bleeding" in bl and bl["bleeding"].page == 2, "haemoglobin down 2.5 g/dL on enoxaparin pages P2")
    ok(D.detect_bleeding(ctx(record(), NORMAL, labs=hb)) == [], "the same fall with no antithrombotic: no bleeding finding")
    cr = lambda v: {"creatinine": [{"t": 1.0, "v": 1.0, "baseline": True}, {"t": 9_000.0, "v": v}]}
    st = [keys(D.detect_kidney(ctx(record(), NORMAL, labs=cr(v)))).get("aki") for v in (1.2, 1.6, 2.2, 3.1)]
    ok(st[0] is None and [f.score for f in st[1:]] == [1.0, 2.0, 3.0],
       "KDIGO against the record's baseline 1.0: 1.2 none, 1.6 stage 1, 2.2 stage 2, 3.1 stage 3")


def test_firewall() -> None:
    print("[6] the replayed ICU record never reads the future or the label")
    n = 30
    frame = pd.DataFrame({"HR": np.full(n, 90.0), "O2Sat": np.full(n, 96.0), "Temp": np.full(n, 37.0),
                          "SBP": np.full(n, 120.0), "MAP": np.full(n, 80.0), "DBP": np.full(n, 60.0),
                          "Resp": np.full(n, 18.0), "Creatinine": np.full(n, np.nan), "Lactate": np.full(n, np.nan),
                          "SepsisLabel": np.zeros(n)})
    frame.loc[3, "Creatinine"] = 1.1
    frame.loc[20, "Creatinine"] = 99.0         # a future value the record must not see at hour 10
    frame.loc[20, "HR"] = 199.0
    frame.loc[15:, "SepsisLabel"] = 1
    s = stay_to_date(frame, 10)
    ok(s["hours"] == 11 and [x["v"] for x in s["labs"]["creatinine"]] == [1.1],
       "at hour 10 the record holds hours 0–10 only: creatinine [1.1], not the 99 measured at hour 20")
    ok(max(r["hr"] for r in s["vitals"]) == 90.0, "the HR of 199 recorded at hour 20 is not visible at hour 10")
    ok("SepsisLabel" not in json.dumps(s) and "label" not in json.dumps(s).lower(),
       "the challenge's sepsis label column is never read into the record")
    s20 = stay_to_date(frame, 20)
    ok(99.0 in [x["v"] for x in s20["labs"]["creatinine"]], "at hour 20 the same value is visible (the check can fail)")


def _os(clk: Clock, demo: bool = False) -> PagingOS:
    os_ = PagingOS(clock=clk, seed=5, demo_history=False)
    os_.settings["demo_traffic"] = demo
    os_.settings["simulated_responders"] = True
    return os_


def _run(os_: PagingOS, clk: Clock, seconds: int) -> None:
    for _ in range(seconds):
        clk.advance(1)
        os_.tick()


def test_monitor() -> None:
    print("[7] the room monitor in the paging OS")
    clk = Clock()
    os_ = _os(clk)
    _run(os_, clk, 600)
    mon = [p for p in os_.pages.values() if p["source"] == "monitor"]
    ok(not mon, f"a resting hospital pages nobody: 0 monitor pages in 10 min across {len(os_.units.phys)} rooms")
    board = os_.rooms_board()
    occ = [r for r in board["rooms"] if r["kind"]]
    ok(len(occ) == len(os_.synthetic) and all(r["scores"].get("news2") for r in occ),
       f"every occupied synthetic room ({len(occ)}) carries a record and a live NEWS2")
    ok(all(os_.units.records[r]["provenance"]["synthetic"] for r in os_.synthetic), "every synthetic record is marked synthetic")

    room = next(r for r, ph in os_.units.phys.items() if "opioid" in ph.flags)
    os_.units.phys[room].start("opioid", clk())
    first = None
    for _ in range(480):
        clk.advance(1)
        os_.tick()
        pg = [p for p in os_.pages.values() if p["room"] == room and p["source"] == "monitor"]
        if pg and first is None:
            first = pg[0]
        live = [p for p in os_.pages.values() if p["room"] == room and p["status"] in ACTIVE]
        if len(live) > 1:
            break
    ok(first is not None and first["priority"] <= 2 and first["category"] == "respiratory",
       f"an opioid course in {room} is detected and paged (P{first['priority'] if first else '?'}, respiratory)")
    ok(first is not None and first["synthetic"] and first["certificate"] is not None,
       "the monitor's page went through the same pipeline: certified, and marked synthetic")
    ok(len([p for p in os_.pages.values() if p["room"] == room and p["status"] in ACTIVE]) <= 1,
       "one room, one page: the monitor never opened a second page for the same episode")
    ok(any(f["key"] == "opioid_rd" for f in first["monitor"]["findings"]), "the page carries the findings that raised it")
    _run(os_, clk, 900)
    ph = os_.units.phys[room]
    ok(ph.scenario is None or ph.scenario.recover_t0 is not None, "after the page was handled the patient recovers")
    ep = os_.units.episodes.get(room) or {}
    ok(not ep.get("open"), "and once nothing calls for a page the monitor episode closes")
    tl = [e["kind"] for e in os_.units.timeline[room]]
    ok("finding" in tl and "page" in tl and "resolved" in tl, "the room's timeline holds the finding, the page and the resolution")

    allergic = next(r for r, rec in os_.units.records.items() if rec["kind"] == "synthetic" and r != room
                    and any(a["substance"] == "penicillin" for a in rec["allergies"])
                    and not any(o["name"] == "piperacillin-tazobactam" for o in rec["orders"]))
    sep = os_.raise_alert(f"Room {allergic} BP 84/50, HR 126, temp 39.2, rigors", source="nurse", reporter="rn.test")["page"]
    _run(os_, clk, 5)
    os_.units._order(allergic, "piperacillin-tazobactam", clk(), "antibiotics broadened for sepsis")
    _run(os_, clk, 30)
    med = [p for p in os_.pages.values() if p["room"] == allergic and p["source"] == "monitor" and p["category"] == "medication"]
    ok(os_.pages[sep["id"]]["status"] in ACTIVE and med and med[0]["priority"] == 3,
       f"an allergy conflict found while a sepsis page is open in {allergic} still pages pharmacy (its own P3 episode)")

    target = next(r for r, ph in os_.units.phys.items() if ph.scenario is None and r not in (room, allergic))
    os_.raise_alert(f"Room {target} SpO2 85% on 2L, RR 30, new confusion", source="nurse", reporter="rn.test")
    _run(os_, clk, 40)
    v = os_.units.series[target][-1]
    ok(abs(v["spo2"] - 85) <= 2 and abs(v["rr"] - 30) <= 3,
       f"a nurse's report becomes the room's measurement (SpO₂ {v['spo2']}, RR {v['rr']})")

    from backend.aegis.paging.hospital import ICU_ROOMS
    beds = [{"bed_id": f"bed_{i}", "name": f"ICU {i}", "fused_risk": 0.2, "population_risk": 0.2, "age": 70,
             "gender": "M", "mrn": f"P{i}", "icu_hour": 5, "enterprise": {"map": 50, "spo2": 82}, "drivers": []}
            for i in range(3)]
    os_.observe_ward(beds)
    _run(os_, clk, 60)
    icu_mon = [p for p in os_.pages.values() if p["room"] in ICU_ROOMS and p["source"] == "monitor"]
    ok(not icu_mon and os_.units.records[ICU_ROOMS[0]]["kind"] == "replay",
       "ICU replay rooms carry a record but are paged only by the sepsis detector, never by the monitor")

    def broken(c: D.Ctx) -> List[D.Finding]:
        raise RuntimeError("planted fault")
    saved = list(D.DETECTORS)
    try:
        D.DETECTORS.insert(0, ("Broken", broken))
        found, faults = D.run_all(ctx(record(flags=["af"]), {**NORMAL, "hr": 150}))
        ok(faults and "planted fault" in faults[0] and any(f.key == "af_rvr" for f in found),
           "a detector that raises is reported as a fault and the others still run")
    finally:
        D.DETECTORS[:] = saved


def main() -> None:
    print("=" * 72)
    print("AegisOS room monitor — verification")
    print("=" * 72)
    for t in (test_news2, test_screens, test_baseline, test_medication, test_conditional, test_firewall, test_monitor):
        t()
    print("-" * 72)
    print(f"ALL {_n['pass']} ROOM CHECKS PASSED")


if __name__ == "__main__":
    main()
