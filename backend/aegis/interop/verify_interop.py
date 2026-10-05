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
  [10] post-hoc    the fixes the registered benchmark motivated (docs/INTEROP.md): a written
                   unit that cannot be the concept refuses; no unit never auto-adopts (pH
                   excepted); a jump back into the population band is not a fault; a burst
                   at the clean rate no longer quarantines (Page's CUSUM) while a swap still
                   does; a swap withholds only the exchanged pair and its un-swap is accepted
  [9] standards    FHIR resources are valid and label synthetic data HTEST; an inbound
                   FHIR bundle is mapped on evidence like HL7; CDS Hooks cards for risk,
                   a quarantined feed, an allergy and an interaction; webhooks sign, a
                   tampered body fails verification, private targets are refused, and a
                   failing receiver retries into the dead-letter list

    .venv/bin/python -m backend.aegis.interop.verify_interop
"""
from __future__ import annotations

import json
import math
import random
import zlib
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
    """Values shaped like the reference: inverse CDF through its stored quantiles, interpolated
    on the concept's own scale (log for skewed analytes; raw-space interpolation distorts an
    analyte spanning four decades, like troponin)."""
    r = V.references()[key]
    log = V.BY_KEY[key].scale == "log"
    ps = sorted(r.quantiles)
    xs = [r.quantiles[p] for p in ps]
    ts = [math.log(max(x, 1e-6)) if log else x for x in xs]
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        u = rng.uniform(ps[1], ps[-2])
        for i in range(len(ps) - 1):
            if ps[i] <= u <= ps[i + 1]:
                w = (u - ps[i]) / (ps[i + 1] - ps[i])
                t = ts[i] + w * (ts[i + 1] - ts[i])
                out.append(math.exp(t) if log else t)
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
        xs = ref_sample(c.key, 80, seed=zlib.crc32(c.key.encode()))     # hash() is salted per process
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
    llm = FakeLLM({"RL101": "creatinine", "RL102": "chloride", "RL103": "wbc", "RL104": "hematocrit",
                   "PCT": None, "AGAP": "procalcitonin_typo"})
    eng, sim, rooms, run, _ = _sim(llm)
    run(600)
    maps = {m["code"]: m for m in eng.mapping_list()}
    want = {"147842": ("hr", "identity"), "150364": ("temp", "[degF]"), "CREA": ("creatinine", "umol/L"),
            "GLU": ("glucose", "mmol/L"), "HGB": ("hemoglobin", "g/L"), "UREA": ("bun", "mmol/L"),
            "HCT": ("hematocrit", "1 →"), "ANC": ("anc", "10*9/L"),
            "CA": ("calcium", "mmol/L"), "150021": ("sbp", "identity")}
    for code, (concept, conv) in want.items():
        m = maps[code]
        ok(m["status"] == "adopted" and m["concept"] == concept and conv in m["conversion"]["label"],
           f"{code} adopted as {concept} ({m['conversion']['label']}) by {m['proposer']}")
    ok(maps["TBIL"]["status"] == "review" and "mg/L" in maps["TBIL"]["reason"],
       "bilirubin in µmol/L waits for a person: the values fit µmol/L and mg/L alike (x1.7, inside its band)")
    ok(maps["RL101"]["status"] == "adopted" and maps["RL101"]["proposer"].startswith("model:"),
       "the model's right answer (P-Kreat → creatinine) is adopted, on the data's evidence")
    ok(maps["RL102"]["status"] == "review" and maps["RL102"]["concept"] is None,
       "the model's WRONG answer in the right unit (B-Natr, a sodium, → chloride) is refused by the data and waits")
    hb = [c for c in (eng.mapping(maps["RL104"]["id"]) or {}).get("candidates", []) if c["concept"] == "hematocrit"]
    ok(maps["RL104"]["status"] == "review" and hb and hb[0]["decision"] == "unit mismatch",
       "a wrong answer whose written unit cannot be the concept (B-Hb in mmol/L → haematocrit %) is refused first")
    ok(maps["AGAP"]["status"] == "unmapped" and maps["PCT"]["status"] == "unmapped"
       and "procalcitonin_typo" in maps["AGAP"]["reason"],
       "a key outside the vocabulary is dropped (and named), and a declined code is left unmapped")

    class DownLLM(FakeLLM):
        def chat(self, system: str, user: str, **kw: Any) -> str:
            self.calls += 1
            raise TimeoutError("quota exceeded")
    down = DownLLM({})
    eng_d, _s, _r, run_d, _ = _sim(down)
    run_d(600)
    md = {m["code"]: m for m in eng_d.mapping_list()}
    ok(down.calls > 0 and all(md[c]["status"] == "review" for c in ("RL101", "PCT", "AGAP")),
       "a failed model call (quota, timeout) is no answer, not a decline: its codes wait for a person "
       "instead of leaving the review queue as 'unmapped'")
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

    def held(r: str) -> bool:
        return any(w.get("concept") == "creatinine" for w in P.room_unit(r)["withheld"])
    held_rooms = [r for r in P.interfaced if held(r)]
    crea = next(x for x in eng.mapping_list() if x["code"] == "CREA")
    eng.adopt(crea["id"], "verify", None, "suggested")
    ok(held_rooms and not any(held(r) for r in held_rooms),
       f"once the fix releases them, no room still lists creatinine as withheld ({len(held_rooms)} did)")
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


def check_standards() -> None:
    section("[9] FHIR, CDS Hooks, webhooks")
    from backend.aegis.interop import fhir as F
    from backend.aegis.interop import webhooks as W
    from backend.aegis.interop.cdshooks import CDSHooks
    from backend.aegis.paging.engine import PagingOS
    from backend.aegis.paging.hospital import ROOMS
    clock = Clock()
    P = PagingOS(llm=None, clock=clock, seed=7, demo_history=False)
    P.settings["demo_traffic"] = False
    eng = Integration(rooms=P, clock=clock, hospital_rooms=list(ROOMS))
    eng.sync_model = True
    P.units.feed_status = eng.room_feed
    sim = SimulatedHospital(eng, lambda: [r for r in ROOMS if P.interface_room_free(r)], clock=clock)
    sim.register()
    for _ in range(420):
        sim.tick()
        eng.tick()
        P.tick()
        clock.t += 1.0
    fx = F.FHIRFacade(P.room_unit, P.rooms_board, lambda: list(P.snapshot()["pages"]))
    cap = fx.capability()
    kinds = {r["type"] for r in cap["rest"][0]["resource"]}
    ok(cap["fhirVersion"] == "4.0.1" and {"Patient", "Observation", "RiskAssessment", "Flag", "Communication"} <= kinds,
       "the CapabilityStatement declares FHIR 4.0.1 and the resources served")
    room = sorted(P.interfaced)[0]
    pt = fx.patient(room)
    ok(pt["resourceType"] == "Patient" and pt["identifier"][0]["value"] == eng.room_of[room]
       and any(sec.get("code") == "HTEST" for sec in pt["meta"].get("security", [])),
       "an HL7-admitted patient reads back as a FHIR Patient with its MRN, labelled HTEST (simulated)")
    obs = fx.observations(room, "vital-signs")
    ok(obs and all(o["code"]["coding"][0]["system"] == F.LOINC and o["valueQuantity"]["system"] == F.UCUM for o in obs),
       f"vital signs come back LOINC-coded with UCUM units ({len(obs)} observations)")
    hr = next((o for o in obs if o["code"]["coding"][0]["code"] == "8867-4"), None)
    ok(hr is not None and hr["valueQuantity"]["code"] == "/min", "heart rate is LOINC 8867-4 in /min")
    syn = next(r for r in P.rooms_board()["rooms"] if r.get("kind") == "synthetic")
    ok(any(s.get("code") == "HTEST" for s in fx.patient(syn["room"])["meta"].get("security", [])),
       "a synthetic demo patient is labelled HTEST too")
    # inbound FHIR: a new sender's LOINC-coded Observations earn their mapping on evidence
    mrn = eng.room_of[room]
    acks = []
    for i, k in enumerate(range(14)):
        o = {"resourceType": "Observation", "status": "final", "subject": {"reference": f"Patient/{mrn}"},
             "code": {"coding": [{"system": F.LOINC, "code": "2823-3", "display": "Potassium"}]},
             "valueQuantity": {"value": 4.0 + 0.05 * (i % 7), "unit": "mmol/L", "code": "mmol/L"}}
        er7, probs = F.FHIRFacade.to_oru([o], sender=("FHIRSRC", "EHR"), ctl=f"FH{i}")
        ack, rec = eng.ingest_hl7(er7)
        acks.append(rec["ack"])
    fm = next(m for m in eng.mapping_list() if m["interface"].startswith("fhirsrc"))
    ok(all(a == "AA" for a in acks) and fm["status"] == "adopted" and fm["proposer"] == "loinc",
       "a FHIR sender's Observations are acknowledged and its LOINC code adopted on evidence")
    bad, probs = F.FHIRFacade.to_oru([{"resourceType": "Observation", "subject": {"reference": "Patient/room-512"},
                                       "code": {"coding": [{"code": "x"}]}, "valueQuantity": {"value": 1}}])
    ok(bad is None and probs, "an Observation naming a room instead of the sender's patient is refused, with a reason")
    # CDS Hooks
    cds = CDSHooks(room_for=lambda pid: F.room_of(pid), room_unit=P.room_unit, feed=eng.room_feed,
                   screen=lambda drugs, ctx: [], app_url="https://app.example")
    ok(len(cds.discovery()["services"]) == 2, "discovery lists patient-view and order-sign")
    cards = cds.call("aegis-patient-view", {"hook": "patient-view", "context": {"patientId": F.patient_id(room)}})["cards"]
    ok(cards and all(c["indicator"] in ("info", "warning", "critical") and len(c["summary"]) <= 140 for c in cards)
       and cards[0]["links"][0]["url"].endswith(f"/room/?id={room}"),
       "patient-view returns spec-shaped cards linking back to the room")
    sim.drill("creatinine_label", True)
    for _ in range(30):
        sim.tick()
        eng.tick()
        P.tick()
        clock.t += 1.0
    q_rooms = [r for r in P.interfaced if any(c["status"] == "quarantined" for c in (eng.room_feed(r) or {}).get("channels", []))]
    qc = cds.call("aegis-patient-view", {"hook": "patient-view", "context": {"patientId": F.patient_id(q_rooms[0])}})["cards"] if q_rooms else []
    ok(any("withheld" in c["summary"] and c["indicator"] == "warning" for c in qc),
       "a quarantined feed reaches the EHR as a warning card: the value it shows may come from that feed")
    sim.drill("creatinine_label", False)
    allergic = next((r for r in P.interfaced if any("penicillin" in a.get("substance", "")
                                                   for a in (P.room_unit(r) or {}).get("record", {}).get("allergies", []))), None)
    if allergic is None:                       # make one: the sim's case mix is random
        allergic = room
        P.units.records[allergic]["allergies"] = [{"substance": "penicillin"}]
    draft = {"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "MedicationRequest", "id": "d1",
                                                               "medicationCodeableConcept": {"text": "piperacillin-tazobactam 4.5 g IV"}}}]}
    oc = cds.call("aegis-order-sign", {"hook": "order-sign", "context": {"patientId": F.patient_id(allergic),
                                                                         "draftOrders": draft}})["cards"]
    ok(any(c["indicator"] == "critical" and c["suggestions"][0]["actions"][0]["type"] == "delete" for c in oc),
       "order-sign: a penicillin-class order for a penicillin-allergic patient is a critical card with a removal")
    from backend.aegis.pharmacopoeia import Interaction
    cds2 = CDSHooks(room_for=lambda pid: F.room_of(pid), room_unit=P.room_unit, feed=eng.room_feed,
                    screen=lambda drugs, ctx: [Interaction("levofloxacin", "amiodarone", "Major", "additive QT", "test")]
                    if "levofloxacin" in drugs else [])
    lc = cds2.call("aegis-order-sign", {"hook": "order-sign", "context": {"patientId": F.patient_id(room), "draftOrders": {
        "resourceType": "Bundle", "entry": [{"resource": {"resourceType": "MedicationRequest", "id": "d2",
                                                          "medicationCodeableConcept": {"text": "Levofloxacin 750 MG Injection"}}}]}}})["cards"]
    ok(any("interaction" in c["summary"] and c["indicator"] == "critical" for c in lc),
       "order-sign normalises an EHR drug string to its generic and surfaces a major interaction")
    cds3 = CDSHooks(room_for=lambda pid: F.room_of(pid), room_unit=P.room_unit, feed=eng.room_feed,
                    screen=lambda drugs, ctx: [Interaction("levofloxacin", "azithromycin", "Moderate", "documented", "DDInter"),
                                               Interaction("levofloxacin, azithromycin", None, "Major", "additive QT",
                                                           "CredibleMeds")] if "levofloxacin" in drugs else [])
    dc = cds3.call("aegis-order-sign", {"hook": "order-sign", "context": {"patientId": F.patient_id(room), "draftOrders": {
        "resourceType": "Bundle", "entry": [{"resource": {"resourceType": "MedicationRequest", "id": "d3",
                                                          "medicationCodeableConcept": {"text": "levofloxacin 750 MG IV"}}}]}}})["cards"]
    ok(len(dc) == 1 and dc[0]["indicator"] == "critical" and "DDInter" in dc[0]["detail"]
       and "CredibleMeds" in dc[0]["detail"],
       "one hazard reported by two knowledge layers (moderate and major) is ONE card at the higher severity, "
       "citing both")
    # webhooks
    wh = W.Webhooks(clock=clock)
    sub = wh.subscribe("internal://echo", [], "verify")["subscription"]
    ok(sub["secret"].startswith("whsec_"), "a subscription's secret is a whsec_ key, returned once")
    wh.from_paging([{"type": "page", "data": {"id": "PG-1", "status": "paging", "priority": 1}},
                    {"type": "page", "data": {"id": "PG-1", "status": "paging", "priority": 1}},
                    {"type": "page", "data": {"id": "PG-1", "status": "accepted", "priority": 1}}])
    wh.pump()
    echo = list(wh.echo)
    ok([e["type"] for e in echo] == ["page.created", "page.accepted"] and all(e["verified"] for e in echo),
       "page transitions become one event each, and the built-in receiver verifies every signature")
    body = '{"type":"page.created"}'
    hdr = {"webhook-id": "msg_1", "webhook-timestamp": str(int(clock.t)),
           "webhook-signature": W.sign(sub["secret"], "msg_1", int(clock.t), body)}
    ok(W.verify(sub["secret"], hdr, body, clock.t) and not W.verify(sub["secret"], hdr, body + " ", clock.t)
       and not W.verify(sub["secret"], hdr, body, clock.t + 3600),
       "pair: the signature verifies; a one-byte change, or a stale timestamp, does not")
    import os as _os
    prev = _os.environ.pop("AEGIS_WEBHOOK_ALLOW_PRIVATE", None)
    ok(W.check_url("https://127.0.0.1/x") and W.check_url("https://10.0.0.5/x") and W.check_url("http://example.com"),
       "loopback, private and plain-http targets are refused (no SSRF through webhooks)")
    if prev is not None:
        _os.environ["AEGIS_WEBHOOK_ALLOW_PRIVATE"] = prev
    calls = []
    failing = W.Webhooks(post=lambda url, h, b: (calls.append(1), (500, "boom"))[1], clock=clock)
    failing.subs["s"] = W.Subscription("s", "https://example.org/hook", list(W.EVENT_TYPES), W.new_secret(), "verify")
    failing.publish("page.created", {"id": "PG-2"})
    for _ in range(len(W.BACKOFF_S) + 1):
        failing.pump()
        clock.t += 2000
    st = failing.status()
    ok(len(calls) == len(W.BACKOFF_S) and st["counters"]["dead"] == 1 and st["dead"],
       f"a receiver that keeps failing is retried {len(W.BACKOFF_S)} times, then dead-lettered")


def check_posthoc() -> None:
    section("[10] post-hoc fixes motivated by the registered run")
    clock = Clock()
    fr = FakeRooms(set())
    e = Integration(rooms=fr, clock=clock)
    e.register(Interface("lab3", "Lab 3", "HL7 v2.5.1", "inbound", "HTTP", "LAB3", "SITE", "mixed"))
    _admit(e, "P1", "512", app="ADT3")
    rng = random.Random(9)
    for i in range(16):
        # heart-rate values written "/min" under a code the lexical rules read as mean arterial pressure
        _oru(e, "LAB3", "SITE", "P1", [("MAP", "Mean arterial pressure", "L", rng.randint(70, 95), "/min"),
                                       ("K", "Potassium", "L", round(rng.uniform(3.7, 4.6), 1), ""),
                                       ("PH", "pH arterial", "L", round(rng.uniform(7.36, 7.44), 2), "")], f"U{i}")
    ms = {m["code"]: m for m in e.mapping_list()}
    cand = [c for c in e.mapping(ms["MAP"]["id"])["candidates"] if c["concept"] == "map"]
    ok(ms["MAP"]["status"] == "review" and cand and not cand[0]["viable"] and cand[0]["decision"] == "unit mismatch"
       and "/min" in ms["MAP"]["reason"],
       "a unit the feed wrote that cannot be the proposed concept refuses it ('/min' is not a pressure): "
       "the console offers no 'adopt' for it")
    ok(ms["K"]["status"] == "review" and ms["PH"]["status"] == "adopted",
       "a feed with no unit waits for a person; pH, unitless by convention, is adopted")
    mg = V.BY_KEY["magnesium"]
    mgv = ref_sample("magnesium", 60, seed=8)
    ok(E.infer(mg, [x / 2.431 for x in mgv], mg.conversion("mmol/L")).decision == "declared"
       and E.infer(mg, mgv, mg.conversion("meq/L")).decision == "ambiguous",
       "pair: magnesium in mmol/L is confirmed; mg/dL values labelled mEq/L (x1.22) are ambiguous, not adopted")
    be = V.BY_KEY["base_excess"]
    bev = ref_sample("base_excess", 60, seed=8)
    ok(E.infer(be, bev, U.IDENTITY).decision == "declared",
       "base excess (linear, centred on zero) is confirmed: reading it x0.001 crushes its spread and is refused")
    bil = V.BY_KEY["bilirubin"]
    high = E.feed_baseline(bil, [7.5, 8.2, 9.1, 6.8, 8.8, 7.9, 9.4, 8.0, 7.2, 8.6, 9.0, 7.7])
    back = [0.6, 0.9, 1.1, 0.7, 0.8, 1.3, 0.5, 0.9, 1.0, 0.7, 0.8, 1.2]
    wild = [x * 20 for x in [7.5, 8.2, 9.1, 6.8, 8.8, 7.9, 9.4, 8.0, 7.2, 8.6, 9.0, 7.7]]
    ok(all(c.status == "pass" for c in E.drift_checks(bil, back, back, high))
       and any(c.status == "fail" for c in E.drift_checks(bil, wild, wild, high)),
       "pair: a feed moving back INTO the population band from an unrepresentative baseline passes; x20 out of it fails")
    inv = E.INVARIANT_BY_NAME["bp_order"]
    good, bad = {"sbp": 120, "map": 85, "dbp": 70}, {"sbp": 85, "map": 120, "dbp": 70}
    burst = [good] * 8 + [bad] * 4
    swapped = [good] * 8 + [bad] * 6
    ok(E.check_invariant(inv, burst).status == "fail" and E.cusum_identity(inv, burst).status != "fail",
       "pair: four violations in twelve failed the old windowed test (the registered H4 false alarms); CUSUM waits")
    ok(E.cusum_identity(inv, swapped).status == "fail", "... while six consecutive violations (a swap) fail it")
    rr = random.Random(3)
    clean = [bad if rr.random() < 0.02 else good for _ in range(20000)]
    peak, S = 0.0, 0.0
    lp, lq = math.log(E.CUSUM_P1 / inv.p0), math.log((1 - E.CUSUM_P1) / (1 - inv.p0))
    for r in clean:
        S = max(0.0, S + (lp if inv.violated(r) else lq))
        peak = max(peak, S)
    ok(0 < peak < E.CUSUM_H, f"20,000 rows at the identity's own allowance (2%) never reach the threshold (peak {peak:.1f})")
    # a PaCO2/HCO3 swap: only the pair is withheld, and the exchange is the fix
    e2 = Integration(rooms=FakeRooms(set()), clock=clock)
    e2.register(Interface("abg", "Blood gas", "HL7 v2.5.1", "inbound", "HTTP", "ABG", "SITE", "gas"))
    _admit(e2, "P2", "513", app="ADT4")
    g = random.Random(5)

    def panel(i: int, swap: bool = False) -> None:
        pco2 = g.uniform(35, 45)
        hco3 = g.uniform(22, 26)
        ph = 6.1 + math.log10(hco3 / (0.0307 * pco2)) + g.uniform(-0.02, 0.02)
        be = 0.93 * (hco3 - 24.4 + 14.8 * (ph - 7.4)) + g.uniform(-1, 1)
        a, b = (hco3, pco2) if swap else (pco2, hco3)
        _oru(e2, "ABG", "SITE", "P2", [("2744-1", "pH", "LN", round(ph, 3), "[pH]"),
                                       ("2019-8", "PaCO2", "LN", round(a, 1), "mm[Hg]"),
                                       ("1963-8", "HCO3", "LN", round(b, 1), "mmol/L"),
                                       ("1925-7", "BE", "LN", round(be, 1), "mmol/L")], f"G{i}")
    for i in range(40):
        panel(i)
    for i in range(40, 60):
        panel(i, swap=True)
    q = {m["code"] for m in e2.mapping_list() if m["status"] == "quarantined"}
    pco2m = next(m for m in e2.mapping_list() if m["code"] == "2019-8")
    ok(q == {"2019-8", "1963-8"} and (pco2m["fault"] or {}).get("swap"),
       "a PaCO2/HCO3 swap withholds exactly those two; pH and base excess keep flowing (the registered run lost 6,821 values here)")
    res = e2.adopt(pco2m["id"], "verify", None, "swap")
    after = {m["code"]: m for m in e2.mapping_list()}
    ok(res["status"] == "ok" and after["2019-8"]["concept"] == "hco3" and after["1963-8"]["concept"] == "paco2"
       and after["2019-8"]["status"] == after["1963-8"]["status"] == "adopted",
       "accepting the exchange re-reads each code as the other's concept")
    for i in range(60, 75):
        panel(i, swap=True)
    ok(all(after_m["status"] == "adopted" for after_m in e2.mapping_list()),
       "... and the exchanged feed then passes its live evidence")


def check_clusters() -> None:
    section("[11] values cluster by patient (post hoc: the febrile-neutropenia demo census)")
    wbc = V.BY_KEY["wbc"]
    g = random.Random(21)
    # the demo hospital's four rooms: two febrile-neutropenic patients (WBC near their nadir),
    # a pyelonephritis (raised) and a heart failure (normal); four results each
    census = {"512": (0.5, 1.8), "505": (0.5, 1.8), "101": (11.5, 15.0), "409": (5.5, 9.5)}
    vals, rooms = [], []
    for _ in range(4):
        for room, (lo, hi) in census.items():
            vals.append(round(g.uniform(lo, hi), 1))
            rooms.append(room)
    pooled = E.infer(wbc, vals, U.IDENTITY)
    clustered = E.infer(wbc, vals, U.IDENTITY, clusters=rooms)
    ok(pooled.decision == "none" and pooled.declared.status == "fail"
       and clustered.decision == "insufficient" and clustered.declared.status == "unsure"
       and clustered.declared.n_eff == 4,
       "pair: two febrile-neutropenic patients among four refuse the white count when 16 values are pooled; "
       "counted as 4 patients the same values are insufficient, never a refusal")
    loc, tail = clustered.declared.check("location"), clustered.declared.check("tail")
    ok(loc.status == "unsure" and "too few patients" in loc.detail and tail.status != "fail",
       f"... the location test says why ({loc.detail}), and the tail counts patients, not values")
    singles = list(range(len(vals)))
    a, b = E.assess(wbc, vals, U.IDENTITY), E.assess(wbc, vals, U.IDENTITY, clusters=singles)
    sig = lambda f: [(c.name, c.status, c.stat, c.limit, c.n) for c in f.checks]
    ok(a.status == b.status and sig(a) == sig(b) and b.n_eff == len(vals),
       "every value its own patient (16 values) reproduces the per-value test (the registered stream's regime)")
    # unit faults on the same four-patient census: refused at the first judgement, as before
    cr, k = V.BY_KEY["creatinine"], V.BY_KEY["potassium"]
    crv = [round(g.uniform(0.6, 1.4) if r != "101" else g.uniform(1.6, 2.4), 2) for r in rooms]
    kv = [round(g.uniform(3.6, 4.8), 1) for _ in rooms]
    faults = [(wbc, [v * 1000 for v in vals], "x1000 white cells"), (cr, [v * 88.42 for v in crv], "x88 creatinine"),
              (k, [v * 10 for v in kv], "x10 potassium"), (cr, [v / 88.42 for v in crv], "/88 creatinine")]
    same = []
    for c, xs, what in faults:
        for n in (12, 16):
            p_, q_ = E.assess(c, xs[:n], U.IDENTITY), E.assess(c, xs[:n], U.IDENTITY, clusters=rooms[:n])
            same.append((what, n, p_.status, q_.status))
    ok(all(p_ == q_ == "fail" for _w, _n, p_, q_ in same),
       "x1000 white cells, x88 and /88 creatinine and x10 potassium from the same four patients are refused at 12 "
       "values, clustered exactly as pooled (they leave the physiological window, which stays per value)")
    # with enough patients the clustered test still refuses a shift no case mix explains
    many = [x * 0.2 for x in ref_sample("wbc", 48, seed=4)]
    mrooms = [str(i % 16) for i in range(48)]
    ok(E.assess(wbc, many, U.IDENTITY, clusters=mrooms).check("location").status == "fail",
       "sixteen patients whose white counts all sit x0.2 below the population are still refused on location")
    # the engine: the demo census keeps the white count learning, then adopts it as patients change
    llm = FakeLLM({"RL101": "creatinine", "RL102": "sodium", "RL103": "wbc", "RL104": "hemoglobin",
                   "PCT": None, "AGAP": None})
    eng, sim, _rooms, run, _clock = _sim(llm)
    seen: Dict[str, set] = {"WBC": set(), "RL103": set()}
    for _ in range(36):
        run(100)
        for m in eng.mapping_list():
            if m["code"] in seen:
                seen[m["code"]].add(m["status"])
    ok(all("review" not in st and "rejected" not in st for st in seen.values())
       and all("learning" in st for st in seen.values()),
       f"the demo hospital (two of four patients febrile-neutropenic): WBC and RL103 learn and never go to "
       f"review over an hour ({ {k: sorted(v) for k, v in seen.items()} })")
    final = {m["code"]: m for m in eng.mapping_list()}
    ok(final["WBC"]["status"] == "adopted" and final["RL103"]["status"] == "adopted",
       "... and both are adopted on the evidence once the census changes (a discharge, a new patient)")


def main() -> None:
    t0 = time.time()
    _results.clear()
    print("verify_interop: the integration plane")
    for part in (check_codec, check_vocab, check_evidence, check_identities, check_onboarding, check_protocol,
                 check_live, check_paging, check_standards, check_posthoc, check_clusters):
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
