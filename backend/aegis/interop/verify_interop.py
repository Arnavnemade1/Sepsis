"""Verification for the integration plane (backend/aegis/interop).

Every guard is checked as a PAIR, the input that must pass and the input that must fail,
and the guards whose absence would be silent are checked again with the guard removed,
to show the check can go red at all.

  [1] codec        HL7 v2 parse/build/escape round-trips, separators from MSH-2, CR/LF/CRLF,
                   malformed input refused with the right table-0357 code, timestamps, ACKs,
                   MLLP de-framing across split reads and noise
  [2] vocabulary   written units → tokens; analyte conversions against hand-computed values;
                   every concept has a LOINC code, a window and (from CinC) a reference
  [3] evidence     the right unit passes on reference-shaped data; each catalogued unit error
                   fails and unit inference names the fix; a small sample cannot refuse; a
                   mixture fails on its tail (ionized calcium inside total calcium)
  [4] identities   each identity: the clean tuple holds and its catalogued fault breaks it;
                   base-rate violations never fail a window, a burst does
  [5] onboarding   LOINC / IEEE 11073 / lexical / model proposals adopted only on evidence;
                   the model cannot widen the vocabulary or adopt a wrong analyte; an inferred
                   unit waits for a person; policy "review" adopts nothing; values reach a room
                   only once adopted; results are backfilled, vitals are not
  [6] protocol     duplicate control ids are not reprocessed; unknown patient / type / version /
                   sender are refused with their codes; ADT occupancy and location rules
  [7] live         every fault drill is quarantined with the right cause and fix; impossible
                   values are never delivered; withdrawn values leave the room; restore and
                   accept-fix; NO false quarantine on a clean run with deteriorating patients;
                   the feed-relative jump test vs the rejected global-reference design; a
                   withheld partner cannot convict its neighbour
  [8] paging       interface rooms are watched by the room monitor; ICU and live rooms are
                   never admitted over HL7; a device alarm pages once; a quarantined feed
                   reaches the Sentinel; no deadlock between ingest and the Sentinel

    .venv/bin/python -m backend.aegis.interop.verify_interop
"""
from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional

from backend.aegis.interop import evidence as E
from backend.aegis.interop import hl7
from backend.aegis.interop import units as U
from backend.aegis.interop import vocab as V
from backend.aegis.interop.engine import Integration, Interface
from backend.aegis.interop.proposer import coded, lexical, parse_model_answer
from backend.aegis.interop.sender import SimulatedHospital

_results: List[Any] = []        # (check name, passed, detail): read by the falsification audit


def ok(cond: bool, msg: str) -> None:
    """Record a check and keep going: the falsification audit (``make falsify``) needs to
    see WHICH named checks a planted defect breaks, not just the first one."""
    _results.append((msg, bool(cond), ""))
    print(f"      [{'PASS' if cond else 'FAIL'}] {msg}")


def section(title: str) -> None:
    print(f"\n  {title}")


class Clock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeRooms:
    """A rooms port that records every effect and keeps a census of who is where."""

    def __init__(self, occupied: Optional[set] = None) -> None:
        self.calls: List[tuple] = []
        self.occupied = set(occupied or ())

    def interface_room_free(self, room: str) -> bool:
        return room not in self.occupied

    def __getattr__(self, name: str):
        if not name.startswith("interface_"):
            raise AttributeError(name)

        def f(*a: Any) -> None:
            self.calls.append((name[len("interface_"):],) + a)
        return f

    def of(self, kind: str) -> List[tuple]:
        return [c for c in self.calls if c[0] == kind]


class FakeLLM:
    """Answers like a model would, including one wrong analyte and one invented key."""
    available = True
    _last_model = "test-model"

    def __init__(self, answers: Dict[str, Optional[str]]) -> None:
        self.answers = answers
        self.calls = 0

    def chat(self, system: str, user: str, **kw: Any) -> str:
        self.calls += 1
        codes = re.findall(r"code='([^']+)'", user)
        return json.dumps({"mappings": [{"code": c, "concept": self.answers.get(c), "unit": "", "confidence": 0.8,
                                         "why": "test"} for c in codes]})


def ref_sample(key: str, n: int, seed: int = 0) -> List[float]:
    """Values shaped like the reference (inverse CDF through its stored quantiles)."""
    r = V.references()[key]
    ps = sorted(r.quantiles)
    xs = [r.quantiles[p] for p in ps]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        u = rng.uniform(ps[1], ps[-2])
        for i in range(len(ps) - 1):
            if ps[i] <= u <= ps[i + 1]:
                w = (u - ps[i]) / (ps[i + 1] - ps[i])
                out.append(xs[i] + w * (xs[i + 1] - xs[i]))
                break
    return out


# --------------------------------------------------------------------------- #
def check_codec() -> None:
    section("[1] HL7 v2 codec")
    text = ("MSH|^~\\&|LIS|AEGISMEM|AEGISOS|AEGISMEM|20261004101500+0200||ORU^R01^ORU_R01|C123|P|2.5.1\r"
            "PID|1||S123456^^^AEGISMEM^MR~X9^^^OTHER^PI||Morgan^Alex||19610402|F\r"
            "OBX|1|NM|CREA^Creatinine \\T\\ eGFR^L||132|umol/L|62-106|H|||F|||20261004095100\r")
    m = hl7.Message.parse(text)
    ok(m.type_name == "ORU^R01" and m.control_id == "C123" and m.version == "2.5.1", "header fields read from MSH")
    pid = m.first("PID")
    ok(pid.get(3, 1) == "S123456" and pid.get(3, 1, rep=2) == "X9" and pid.get(5, 2) == "Alex",
       "repetitions and components addressed by HL7 numbering")
    ok(m.first("OBX").get(3, 2) == "Creatinine & eGFR", "escape sequence \\T\\ decoded")
    ok(m.timestamp == hl7.parse_ts("20261004081500"), "MSH-7 with a +0200 offset converts to UTC")
    alt = text.replace("|", "#").replace("MSH#^~\\&", "MSH#^~\\&")
    ok(hl7.Message.parse(alt).first("PID").get(5, 1) == "Morgan", "a feed's own field separator is honoured (MSH-1)")
    for sep in ("\n", "\r\n"):
        ok(hl7.Message.parse(text.replace("\r", sep)).first("OBX").get(5) == "132",
           f"segments ending {sep!r} parse")
    hard = "a|b^c&d~e\\f"
    seg = hl7.segment("NTE", "1", "", hard)
    round_trip = hl7.Message.parse(hl7.build([hl7.msh("A", "B", "C", "D", ("ORU", "R01", "ORU_R01"), "X1"), seg]))
    ok(round_trip.first("NTE").get(3) == hard, "every delimiter inside a value survives build → parse")
    ok(hl7.format_number(0.00001) == "0.00001" and "e" not in hl7.format_number(1e-7 + 1e-9),
       "numbers are written without exponent notation")
    for bad, code in (("PID|1||X", hl7.E_SEGMENT_SEQUENCE), ("MSH|^~\\&|A|B|C|D|2026||||P|2.5.1", hl7.E_REQUIRED_FIELD),
                      (text + text, hl7.E_SEGMENT_SEQUENCE)):
        try:
            hl7.Message.parse(bad)
            raised = None
        except hl7.HL7Error as e:
            raised = e.code
        ok(raised == code, f"malformed input refused with table-0357 code {code}")
    ok(hl7.parse_ts("202610") == hl7.parse_ts("20261001000000") and hl7.parse_ts("2026x") is None,
       "partial-precision timestamps parse; garbage does not")
    err = hl7.HL7Error("bad", hl7.E_UNKNOWN_KEY, "PID^1^3")
    a = hl7.parse_ack(hl7.ack(m, "AE", "bad", err))
    ok(a["code"] == "AE" and a["control_id"] == "C123" and a["error"]["code"] == 204 and a["error"]["where"] == "PID^1^3",
       "an ACK carries MSA (code, control id) and ERR (0357 code, location)")
    dec = hl7.MLLPDecoder()
    f1, f2 = hl7.frame("MSH|first"), hl7.frame("MSH|second")
    got = dec.feed(b"noise" + f1 + f2[:5])
    got += dec.feed(f2[5:])
    ok(got == ["MSH|first", "MSH|second"] and dec.noise_bytes == 5,
       "MLLP: two frames across a split read, noise before the first counted not fatal")


def check_vocab() -> None:
    section("[2] vocabulary and units")
    cases = {"K/uL": "10*3/uL", "x10^9/L": "10*9/L", "×10⁹/L": "10*9/L", "µmol/L": "umol/L", "MG/DL": "mg/dL",
             "°F": "[degF]", "G/L": "10*9/L", "g/L": "g/L", "mmHg": "mm[Hg]", "bpm": "/min", "L/L": "1", "": ""}
    for written, tok in cases.items():
        ok(U.norm(written) == tok, f"{written!r} → {tok!r}")
    ok(U.norm("furlongs").startswith("?"), "an unknown unit is marked unknown, never guessed")
    conv = [("glucose", "mmol/L", 5.5, 99.088), ("creatinine", "umol/L", 88.42, 1.0), ("temp", "[degF]", 98.6, 37.0),
            ("hemoglobin", "mmol/L", 7.8, 12.566), ("calcium", "mmol/L", 2.5, 10.02), ("bun", "mmol/L", 7.0, 19.607),
            ("wbc", "/uL", 8200, 8.2), ("anc", "10*9/L", 4.1, 4100.0), ("fio2", "%", 40, 0.4),
            ("hematocrit", "1", 0.42, 42.0), ("troponin", "ng/L", 34, 0.034), ("magnesium", "meq/L", 1.6, 1.9448)]
    for key, unit, x, want in conv:
        c = V.BY_KEY[key].conversion(U.norm(unit))
        ok(c is not None and abs(c.apply(x) - want) < 1e-3 * max(1, abs(want)), f"{key}: {x} {unit} → {want}")
    ok(V.BY_KEY["calcium"].conversion("meq/L").factor == 2.004 and V.BY_KEY["potassium"].conversion("meq/L").factor == 1.0,
       "mEq/L is valence-aware: ×2.004 for divalent calcium, ×1 for potassium")
    ok(all(c.loinc and c.window[0] < c.window[1] for c in V.CONCEPTS), "every concept has a LOINC code and a window")
    refs = V.references()
    ok(all(refs.get(c.key) and refs[c.key].n > 0 for c in V.CONCEPTS if c.cinc),
       f"every CinC-backed concept has a reference built from training patients ({len(refs)} references)")


def check_evidence() -> None:
    section("[3] single-variable evidence and unit inference")
    for c in V.CONCEPTS:
        if c.key not in V.references() or c.scale == "none":
            continue
        xs = ref_sample(c.key, 80, seed=hash(c.key) % 1000)
        f = E.assess(c, xs, U.IDENTITY)
        ok(f.status == "pass", f"{c.key}: reference-shaped values in the canonical unit pass")
    faults = [("glucose", "mg/dL", 1 / 18.016, "mmol/L"), ("creatinine", "mg/dL", 88.42, "umol/L"),
              ("hemoglobin", "g/dL", 10.0, "g/L"), ("wbc", "10*3/uL", 1000.0, "/uL"),
              ("fio2", "1", 100.0, "%"), ("hematocrit", "%", 0.01, "1 →"), ("calcium", "mg/dL", 1 / 4.008, "mmol/L"),
              ("magnesium", "mg/dL", 1 / 2.431, "mmol/L"), ("lactate", "mmol/L", 9.008, "mg/dL"),
              ("bilirubin", "mg/dL", 17.104, "umol/L"), ("troponin", "ng/mL", 1000.0, "ng/L"),
              ("bun", "mg/dL", 1 / 2.801, "mmol/L")]
    ambiguous = []
    for key, declared, factor, fix in faults:
        c = V.BY_KEY[key]
        xs = [x * factor for x in ref_sample(key, 80, seed=7)]
        inf = E.infer(c, xs, c.conversion(U.norm(declared)))
        right = any(fix in f.conversion.label for f in inf.passing)
        if inf.decision == "inferred":
            ok(inf.declared.status == "fail" and fix in inf.best.conversion.label,
               f"{key} sent in the wrong unit fails as {declared} and is read as {inf.best.conversion.label}")
        else:
            ambiguous.append(key)
            ok(inf.declared.status == "fail" and inf.decision == "ambiguous" and right,
               f"{key} fails as {declared}; the right reading passes but so does "
               + " / ".join(f.conversion.label for f in inf.passing if fix not in f.conversion.label)
               + ": named ambiguous, so a person decides")
    ok(ambiguous == ["bilirubin"],
       "the one ambiguity is bilirubin: µmol/L and mg/L differ by 1.7×, inside its population band")
    c = V.BY_KEY["temp"]
    xs = [x * 9 / 5 + 32 for x in ref_sample("temp", 60, seed=3)]
    inf = E.infer(c, xs, U.IDENTITY)
    ok(inf.decision == "inferred" and "[degF]" in inf.best.conversion.label, "°F values under a °C label: the fix is °F → °C")
    # a small sample can never refuse a correct mapping
    cr = V.BY_KEY["creatinine"]
    small = [cr.window[0] * 1.5, 0.9, 1.1, 3.8, 0.6]
    ok(E.assess(cr, small, U.IDENTITY).status != "fail", "a 5-value sample of a correct mapping is never refused")
    # mixture: ionized calcium (mmol/L) recorded under total calcium (mg/dL), as in CinC hospital B
    ca = V.BY_KEY["calcium"]
    total = ref_sample("calcium", 80, seed=11)
    mixed = total[:64] + [random.Random(5).uniform(1.05, 1.3) for _ in range(16)]
    ok(E.assess(ca, total, U.IDENTITY).status == "pass" and E.assess(ca, mixed, U.IDENTITY).status == "fail",
       "pair: total calcium passes; the same feed with 20% ionized values fails")
    ok(E.assess(ca, mixed, U.IDENTITY).check("location").status == "pass",
       "... and it is the tail and the window that catch the mixture, not the median (which barely moves)")
    # named units beat decade guesses
    vals = [x * 88.42 for x in ref_sample("creatinine", 40, seed=2)]
    inf = E.infer(cr, vals, None)
    ok(inf.decision == "inferred" and "umol/L" in inf.best.conversion.label,
       "with no unit written, ÷88.42 (µmol/L) beats the ÷100 decade guess that also fits")


def check_identities() -> None:
    section("[4] physiological identities")
    pairs = {
        "bp_order": ({"sbp": 124, "map": 88, "dbp": 70}, {"sbp": 88, "map": 124, "dbp": 70}),
        "map_estimate": ({"sbp": 124, "map": 88, "dbp": 70}, {"sbp": 124, "map": 54, "dbp": 70}),
        "hct_hgb": ({"hematocrit": 36, "hemoglobin": 12}, {"hematocrit": 12, "hemoglobin": 36}),
        "henderson_hasselbalch": ({"ph": 7.40, "paco2": 40, "hco3": 24}, {"ph": 7.40, "paco2": 24, "hco3": 40}),
        "spo2_sao2": ({"spo2": 96, "sao2": 97}, {"spo2": 96, "sao2": 64}),
        "bun_creatinine": ({"bun": 18, "creatinine": 1.0}, {"bun": 18, "creatinine": 88.4}),
        "bilirubin_fraction": ({"bilirubin_direct": 0.3, "bilirubin": 1.1}, {"bilirubin_direct": 1.1, "bilirubin": 0.3}),
        "anc_wbc": ({"anc": 5200, "wbc": 8.0}, {"anc": 5_200_000, "wbc": 8.0}),
        "anion_gap": ({"sodium": 139, "chloride": 103, "hco3": 24}, {"sodium": 4.1, "chloride": 103, "hco3": 24}),
        "van_slyke": ({"base_excess": 0.0, "hco3": 24, "ph": 7.40}, {"base_excess": 24.0, "hco3": 24, "ph": 7.40}),
    }
    for inv in E.INVARIANTS:
        clean, broken = pairs[inv.name]
        ok(not inv.violated(clean) and inv.violated(broken), f"{inv.name}: holds when clean, breaks on its fault")
    ag = E.INVARIANT_BY_NAME["anion_gap"]
    ok(not ag.violated({"sodium": 139, "chloride": 24, "hco3": 103}),
       "the anion gap is blind to a Cl/HCO₃ swap (its statement says so; the per-variable checks catch it)")
    inv = E.INVARIANT_BY_NAME["bp_order"]
    good = {"sbp": 120, "map": 85, "dbp": 70}
    bad = {"sbp": 85, "map": 120, "dbp": 70}
    base_rate = [good] * 39 + [bad]
    burst = [good] * 8 + [bad] * 4
    ok(E.check_invariant(inv, base_rate).status == "pass" and E.check_invariant(inv, burst).status == "fail",
       "pair: one violation in 40 (the base rate) passes; four in twelve fails")


# --------------------------------------------------------------------------- #
def _sim(llm: Any = None, adoption: str = "evidence", occupied: Optional[set] = None):
    from backend.aegis.paging.hospital import ICU_ROOMS, LIVE_ROOM, ROOMS, synthetic_patients
    occ = set(synthetic_patients()) | set(ICU_ROOMS) | {LIVE_ROOM} if occupied is None else occupied
    clock = Clock()
    rooms = FakeRooms(occ)
    eng = Integration(rooms=rooms, llm=llm, clock=clock, hospital_rooms=list(ROOMS))
    eng.sync_model = True
    eng.settings["adoption"] = adoption
    free = [r for r in ROOMS if r not in occ]
    sim = SimulatedHospital(eng, lambda: free, clock=clock)
    sim.register()

    def run(sec: float) -> None:
        for _ in range(int(sec)):
            sim.tick()
            eng.tick()
            clock.t += 1.0
    return eng, sim, rooms, run, clock


def check_onboarding() -> None:
    section("[5] onboarding: proposals adopted only on evidence")
    llm = FakeLLM({"RL101": "creatinine", "RL102": "sodium", "RL103": "wbc", "RL104": "hematocrit",
                   "PCT": None, "AGAP": "procalcitonin_typo"})
    eng, sim, rooms, run, _ = _sim(llm)
    run(600)
    maps = {m["code"]: m for m in eng.mapping_list()}
    want = {"147842": ("hr", "identity"), "150364": ("temp", "[degF]"), "CREA": ("creatinine", "umol/L"),
            "GLU": ("glucose", "mmol/L"), "HGB": ("hemoglobin", "g/L"), "UREA": ("bun", "mmol/L"),
            "TBIL": ("bilirubin", "umol/L"), "HCT": ("hematocrit", "1 →"), "ANC": ("anc", "10*9/L"),
            "CA": ("calcium", "mmol/L"), "150021": ("sbp", "identity")}
    for code, (concept, conv) in want.items():
        m = maps[code]
        ok(m["status"] == "adopted" and m["concept"] == concept and conv in m["conversion"]["label"],
           f"{code} adopted as {concept} ({m['conversion']['label']}) by {m['proposer']}")
    ok(maps["RL101"]["status"] == "adopted" and maps["RL101"]["proposer"].startswith("model:"),
       "the model's right answer (P-Kreat → creatinine) is adopted, on the data's evidence")
    ok(maps["RL104"]["status"] == "review" and maps["RL104"]["concept"] is None,
       "the model's WRONG answer (B-Hb → haematocrit) is refused by the evidence and waits for a person")
    ok(maps["AGAP"]["status"] == "unmapped" and maps["PCT"]["status"] == "unmapped",
       "a key outside the vocabulary is dropped, and a declined code is left unmapped")
    ok(maps["TNIHS"]["status"] == "review", "hs-troponin, whose ward population sits far from the ICU reference, waits for a person")
    pre = [c for c in rooms.of("observe") if any(k == "creatinine" for k in c[4])]
    ok(all(c[4]["creatinine"][0][0].startswith("OBS-") for c in pre), "only adopted mappings deliver values to rooms")
    bf = rooms.of("backfill")
    ok(any(c[2] == "creatinine" for c in bf) and not any(c[2] in ("hr", "sbp", "spo2") for c in bf),
       "results learned before adoption are backfilled; vitals are not")
    # the model only proposes
    got = parse_model_answer('{"mappings":[{"code":"X","concept":"made_up_key"},{"code":"Y","concept":"sodium"}]}',
                             ["X", "Y"])
    ok("X" not in got and got["Y"].concept == "sodium", "a model answer naming a key outside the vocabulary is discarded")
    ok(coded("2160-0", "", "LN").concept == "creatinine" and coded("150456", "MDC_PULS_OXIM_SAT_O2", "MDC").concept == "spo2",
       "LOINC and IEEE 11073 codes name their concept outright")
    lx = lexical("HGB", "Hemoglobin", "umol/L")
    ok(not lx or lx[0].score < lexical("HGB", "Hemoglobin", "g/L")[0].score,
       "a unit that cannot be the analyte lowers the lexical score")
    # policy review: nothing adopts on its own
    eng2, _sim2, _r2, run2, _ = _sim(None, adoption="review")
    run2(300)
    st = {m["status"] for m in eng2.mapping_list()}
    ok("adopted" not in st and "review" in st, "policy 'review': every mapping waits for a person")
    # an inferred unit at onboarding waits for a person
    clock = Clock()
    fr = FakeRooms(set())
    e3 = Integration(rooms=fr, clock=clock)
    e3.register(Interface("lab2", "Lab 2", "HL7 v2.5.1", "inbound", "HTTP", "LAB2", "SITE", "labs"))
    _admit(e3, "M1", "512")
    vals = [x * 88.42 for x in ref_sample("creatinine", 14, seed=4)]
    for i, v in enumerate(vals):
        _oru(e3, "LAB2", "SITE", "M1", [("CR", "Creatinine", "L", round(v, 1), "mg/dL")], f"C{i}")
    m = e3.mapping_list()[0]
    ok(m["status"] == "review" and ("µmol" in m["reason"] or "umol" in m["reason"]),
       "values that only fit as µmol/L under an mg/dL label: the unit fix is proposed, not adopted")


def _admit(eng: Integration, mrn: str, room: str, app: str = "ADT", fac: str = "SITE", event: str = "A01") -> str:
    if (app.upper(), fac.upper()) not in eng.by_sender:
        eng.register(Interface(f"{app.lower()}-{fac.lower()}", app, "HL7 v2.5.1", "inbound", "HTTP", app, fac, "adt"))
    segs = [hl7.segment("EVN", event), hl7.segment("PID", "1", "", (mrn, "", "", "SITE", "MR"), "", ("Test", "Pat"),
                                                   "", "19700101", "F"),
            hl7.segment("PV1", "1", "I", ("WARD", room, "1", "SITE")), hl7.segment("DG1", "1", "", ("", "Pneumonia", ""),
                                                                                  "Community-acquired pneumonia")]
    ack, _ = eng.ingest_hl7(hl7.build([hl7.msh(app, fac, "AEGISOS", "SITE", ("ADT", event, "ADT_A01"),
                                               f"A{mrn}{event}{room}")] + segs))
    return hl7.parse_ack(ack)["code"]


def _oru(eng: Integration, app: str, fac: str, mrn: str, obs: List[tuple], ctl: str,
         mtype: tuple = ("ORU", "R01", "ORU_R01"), version: str = "2.5.1") -> Dict[str, Any]:
    segs = [hl7.segment("PID", "1", "", (mrn, "", "", "SITE", "MR"))]
    for i, (code, text, system, val, unit) in enumerate(obs, 1):
        segs.append(hl7.segment("OBX", str(i), "NM", (code, text, system), "", val, unit, "", "", "", "", "F"))
    ack, rec = eng.ingest_hl7(hl7.build([hl7.msh(app, fac, "AEGISOS", "SITE", mtype, ctl, eng.now(), version=version)]
                                        + segs))
    return hl7.parse_ack(ack)


def check_protocol() -> None:
    section("[6] protocol: acknowledgements, idempotence, census")
    clock = Clock()
    fr = FakeRooms({"501"})
    e = Integration(rooms=fr, clock=clock, hospital_rooms=["501", "502", "503"])
    e.register(Interface("lab", "Lab", "HL7 v2.5.1", "inbound", "HTTP", "LAB", "SITE", "labs"))
    ok(_admit(e, "M1", "502") == "AA" and fr.of("admit")[0][1] == "502", "ADT^A01 admits into a free room")
    ok(_admit(e, "M2", "501") == "AE", "an A01 into an occupied room is refused (AE 206), the census unchanged")
    ok(_admit(e, "M3", "999") == "AE", "an A01 to a location the hospital does not have is refused (AE 103)")
    a1 = _oru(e, "LAB", "SITE", "M1", [("K", "Potassium", "L", 4.1, "mmol/L")], "DUP1")
    n_obs = e.counters["observations"]
    a2 = _oru(e, "LAB", "SITE", "M1", [("K", "Potassium", "L", 4.1, "mmol/L")], "DUP1")
    ok(a1 == a2 and e.counters["observations"] == n_obs and e.counters["duplicates"] == 1,
       "a resent control id gets the same ACK and is not processed twice")
    ok(_oru(e, "LAB", "SITE", "NOPE", [("K", "Potassium", "L", 4.1, "mmol/L")], "U1")["error"]["code"] == 204,
       "results for a patient not on the census: AE 204")
    ok(_oru(e, "LAB", "SITE", "M1", [], "T1", mtype=("QRY", "A19", "QRY_A19"))["code"] == "AR",
       "an unrouted message type: AR 200")
    ok(_oru(e, "LAB", "SITE", "M1", [], "V1", version="2.1")["error"]["code"] == 203, "an unsupported version: AR 203")
    e.settings["unknown_senders"] = "reject"
    ok(_oru(e, "ROGUE", "ELSEWHERE", "M1", [], "R1")["error"]["code"] == 204, "an unknown sender under policy 'reject': AR 204")
    ack, _ = e.ingest_hl7("PID|1||X\r")
    ok(hl7.parse_ack(ack)["code"] == "AR", "an unparseable message is refused, the engine keeps running")
    seg = [hl7.segment("EVN", "A02"), hl7.segment("PID", "1", "", ("M1", "", "", "SITE", "MR")),
           hl7.segment("PV1", "1", "I", ("WARD", "503", "1", "SITE"))]
    e.ingest_hl7(hl7.build([hl7.msh("ADT", "SITE", "AEGISOS", "SITE", ("ADT", "A02", "ADT_A02"), "X02")] + seg))
    ok(e.room_of.get("503") == "M1" and "502" not in e.room_of and fr.of("discharge")[-1][1] == "502",
       "ADT^A02 moves the patient and frees the old room")


def check_live() -> None:
    section("[7] live evidence, quarantine, recovery")
    eng, sim, rooms, run, clock = _sim(None)
    run(600)
    q0 = eng.counters["quarantines"]
    run(2 * 3600)
    ok(eng.counters["quarantines"] == q0 and eng.counters["delivered"] > 10000,
       f"no false quarantine over two clean hours ({eng.counters['delivered']} values delivered, deteriorations included)")
    expect = {"glucose_units": ("GLU", "window", "identity"), "creatinine_label": ("CREA", "window", "umol/L"),
              "hgb_scale": ("HGB", "window", "identity"), "temp_celsius": ("150364", "window", "identity"),
              "bp_swap": ("150021", "systolic > mean", None), "nibp_frozen": ("150021", "frozen", None)}
    for kind, (code, cause, fix) in expect.items():
        before = {m["id"] for m in eng.mapping_list() if m["status"] == "quarantined"}
        sim.drill(kind, True)
        t0 = clock.t
        for _ in range(600):
            run(1)
            if any(m["status"] == "quarantined" and m["code"] == code for m in eng.mapping_list()):
                break
        m = next((x for x in eng.mapping_list() if x["code"] == code), None)
        f = (m or {}).get("fault") or {}
        ok(m is not None and m["status"] == "quarantined" and cause in f.get("cause", ""),
           f"drill {kind}: {code} quarantined in {int(clock.t - t0)} s ({f.get('cause', '')[:60]})")
        if fix:
            ok((f.get("suggested") or {}).get("label", "").find(fix) >= 0, f"... and the evidence names the fix: {fix}")
        sim.drill(kind, False)
        for x in eng.mapping_list():
            if x["status"] == "quarantined" and x["id"] not in before:
                eng.restore(x["id"], "verify")
        run(90)
    withheld = [c for c in rooms.of("withheld")]
    ok(withheld and all(item.get("why") for c in withheld for item in c[3]),
       "impossible values were withheld with a reason, never delivered")
    retract = rooms.of("retract")
    ok(any(c[3] for c in retract), "a quarantine withdraws the values delivered since its change point")
    # accept the fix: release withheld results under the corrected conversion
    sim.drill("creatinine_label", True)
    run(30)
    m = next(x for x in eng.mapping_list() if x["code"] == "CREA")
    n_bf = eng.counters["backfilled"]
    res = eng.adopt(m["id"], "verify", None, "suggested")
    run(10)
    m2 = next(x for x in eng.mapping_list() if x["code"] == "CREA")
    ok(res["status"] == "ok" and m2["status"] == "adopted" and "umol/L" in m2["conversion"]["label"]
       and eng.counters["backfilled"] > n_bf, "accepting the fix adopts it and releases the withheld results")
    sim.drill("creatinine_label", False)
    # the drift design, as a pair: patients getting sicker must not cut the feed
    hr = V.BY_KEY["hr"]
    base = E.feed_baseline(hr, [78, 82, 75, 88, 80, 84, 77, 90, 81, 79, 86, 83])
    septic = [118, 126, 131, 122, 128, 135, 119, 124, 130, 127, 121, 133]
    live = E.drift_checks(hr, septic, septic, base)
    rejected = E.assess(hr, septic, U.IDENTITY)              # the global-reference test with the population band
    ok(all(c.status == "pass" for c in live) and rejected.check("location").status == "fail",
       "pair: a ward turning septic (HR ~125) passes the feed-relative test, and WOULD have cut the heart-rate "
       "feed under the rejected global-reference design")
    lac = V.BY_KEY["lactate"]
    lbase = E.feed_baseline(lac, [1.1, 1.3, 1.2, 1.4, 1.0, 1.6, 1.3, 1.2, 1.5, 1.1, 1.3, 1.4])
    sick = [3.8, 4.6, 5.2, 4.1, 6.0, 4.4, 3.9, 5.5, 4.8, 4.2, 5.0, 4.7]
    unit_err = [x * 9.008 for x in [1.1, 1.3, 1.2, 1.4, 1.0, 1.6, 1.3, 1.2, 1.5, 1.1, 1.3, 1.4]]
    ok(all(c.status == "pass" for c in E.drift_checks(lac, sick, sick, lbase))
       and any(c.status == "fail" for c in E.drift_checks(lac, unit_err, unit_err, lbase)),
       "pair: septic lactates (median 4.7) keep the feed; the same feed switched to mg/dL (×9) is quarantined")
    # a withheld partner cannot convict its neighbour (the bug this guard fixes)
    hct = next(x for x in eng.by_id.values() if x.code == "HCT")
    hgb = next(x for x in eng.by_id.values() if x.code == "HGB")
    groups = eng.groups["sunquest-lis"]
    g = groups[-1]
    g2 = {**g, "vals": dict(g["vals"]), "seq": dict(g["seq"]), "held": set(g["held"])}
    g2["vals"][hgb.id] = g2["vals"][hgb.id] * 0.1 if hgb.id in g2["vals"] else 1.0
    g2["held"] = {hgb.id}
    groups.append(g2)
    rows = eng._joint_rows(hct, hct.concept, hct.conversion, adopted_only=True, since_seq=0)
    ok(rows and "hemoglobin" not in rows[-1][1], "a withheld haemoglobin is left out of the Hct/Hgb identity")
    g2["held"] = set()                                       # planted: the guard removed
    rows = eng._joint_rows(hct, hct.concept, hct.conversion, adopted_only=True, since_seq=0)
    ok(rows and "hemoglobin" in rows[-1][1] and E.INVARIANT_BY_NAME["hct_hgb"].violated(rows[-1][1]),
       "... planted: without the guard the same value would break the identity")
    groups.pop()


def check_paging() -> None:
    section("[8] the paging OS")
    from backend.aegis.paging.engine import ACTIVE, PagingOS
    from backend.aegis.paging.hospital import ICU_ROOMS, LIVE_ROOM, ROOMS
    clock = Clock()
    P = PagingOS(llm=None, clock=clock, seed=7, demo_history=False)
    P.settings["demo_traffic"] = False
    P.sync_sbar = True
    eng = Integration(rooms=P, clock=clock, hospital_rooms=list(ROOMS))
    eng.sync_model = True
    P.units.feed_status = eng.room_feed
    P.sentinel.external = eng.sentinel_recs
    sim = SimulatedHospital(eng, lambda: [r for r in ROOMS if P.interface_room_free(r)], clock=clock)
    sim.register()
    ok(not P.monitored and not P.interface_room_free("312") and not P.interface_room_free(LIVE_ROOM),
       "before the ward has ticked (no monitored beds yet), ICU and live rooms are still never free")

    def run(sec: float) -> None:
        for _ in range(int(sec)):
            sim.tick()
            eng.tick()
            P.tick()
            clock.t += 1.0
    run(600)
    rooms = sorted(P.interfaced)
    ok(len(rooms) == 4 and not set(rooms) & (set(ICU_ROOMS) | {LIVE_ROOM}), f"four patients admitted over HL7: {rooms}")
    u = P.room_unit(rooms[0])
    ok(u["kind"] == "interface" and u["vitals"] and u["labs"] and "news2" in u["scores"],
       "an interface room is watched like any other: vitals, results, NEWS2")
    ok(u["feed"] and u["feed"]["channels"] and any(d["name"].startswith("Philips") for d in u["devices"]),
       "its feed and devices are shown from the integration plane")
    room = rooms[1]
    mrn = eng.room_of[room]
    alarm = [hl7.segment("PID", "1", "", (mrn, "", "", "AEGISMEM", "MR")),
             hl7.segment("OBR", "1", "", "", ("ALARM", "SpO2 low", "L")),
             hl7.segment("OBX", "1", "NM", ("150456", "MDC_PULS_OXIM_SAT_O2", "MDC"), "", 82, "%", "", "LL", "", "",
                         "F")]
    n0 = len(P.pages)
    eng.ingest_hl7(hl7.build([hl7.msh("IBRIDGE", "AEGISMEM", "AEGISOS", "AEGISMEM", ("ORU", "R40", "ORU_R40"), "AL1")]
                             + alarm))
    pages = [p for p in P.pages.values() if p.get("room") == room and p["source"] == "device"]
    ok(len(P.pages) == n0 + 1 and pages and pages[0]["priority"] <= 2,
       f"a device alarm (ORU^R40, SpO₂ 82) becomes a P{pages[0]['priority'] if pages else '?'} page from source 'device'")
    eng.alarm_last.clear()                                   # past the engine's cooldown
    eng.ingest_hl7(hl7.build([hl7.msh("IBRIDGE", "AEGISMEM", "AEGISOS", "AEGISMEM", ("ORU", "R40", "ORU_R40"), "AL2")]
                             + alarm))
    ok(len(P.pages) == n0 + 1, "a second alarm while that page is open does not open another page")
    sim.drill("creatinine_label", True)
    run(40)
    P.last_sentinel = 0
    run(2)
    recs = P.sentinel.active()
    ok(any(r["kind"] == "interface_fault" and "umol/L" in r["action"] for r in recs),
       "the quarantined feed reaches the Sentinel with the evidence's fix")
    sim.drill("creatinine_label", False)
    # no deadlock: ingest (interop lock → paging lock, released in between) against the
    # Sentinel (paging lock → interop lock)
    stop = time.time() + 3.0
    errs: List[str] = []

    def sentinel_loop() -> None:
        while time.time() < stop:
            P.last_sentinel = 0
            try:
                P.tick()
            except Exception as exc:
                errs.append(str(exc))

    def ingest_loop() -> None:
        k = 0
        while time.time() < stop:
            k += 1
            eng.ingest_hl7(hl7.build([hl7.msh("SQLAB", "AEGISMEM", "AEGISOS", "AEGISMEM", ("ORU", "R01", "ORU_R01"),
                                              f"DL{k}"),
                                      hl7.segment("PID", "1", "", (mrn, "", "", "AEGISMEM", "MR")),
                                      hl7.segment("OBX", "1", "NM", ("K", "Potassium", "L"), "", 4.2, "mmol/L")]))
    ts = [threading.Thread(target=sentinel_loop, daemon=True), threading.Thread(target=ingest_loop, daemon=True)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=12)
    ok(not any(t.is_alive() for t in ts) and not errs, "three seconds of concurrent ingest and Sentinel scans: no deadlock")


def main() -> None:
    t0 = time.time()
    _results.clear()
    print("verify_interop: the integration plane")
    for part in (check_codec, check_vocab, check_evidence, check_identities, check_onboarding, check_protocol,
                 check_live, check_paging):
        try:
            part()
        except Exception as exc:                  # a crashed section is a failed check, never a pass
            ok(False, f"{part.__name__} crashed: {type(exc).__name__}: {exc}")
    failed = [r[0] for r in _results if not r[1]]
    print("-" * 72)
    if failed:
        print(f"{len(failed)} OF {len(_results)} INTEROP CHECKS FAILED  [{time.time() - t0:.0f}s]")
        for f in failed:
            print(f"  FAILED: {f}")
        raise SystemExit(1)
    print(f"ALL {len(_results)} INTEROP CHECKS PASSED  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
