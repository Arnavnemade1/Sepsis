"""The detection library a room runs over its patient: published scores, read with the record.

Each detector is a pure function of a :class:`Ctx` (the record, the latest vitals,
the recent series, the labs) and returns :class:`Finding`s. A finding says what was
seen, the evidence (each value with its time), the source the rule comes from, what
in the patient's record changed the reading, and the escalation the rule calls for.

The detectors are deliberately record-aware, because the same numbers mean different
things in different patients:

* **NEWS2** switches to SpO₂ scale 2 when the record documents CO₂ retention with a
  prescribed 88–92% target, and scores confusion only when it is *new* (a patient
  with dementia who is confused at baseline does not score 3 for it).
* **Personal baseline** compares today's vitals with the vitals measured at the
  patient's own earlier encounters: a systolic of 108 is unremarkable on paper and
  alarming in a patient whose baseline is 160.
* **Medication safety** reads the orders against the allergies, the problems and the
  labs (renal function, potassium), and knows when a combination is the expected
  regimen (heparin with aspirin in an NSTEMI) rather than an error.
* **Opioid sedation, neutropenic fever, hypoglycaemia, AF with RVR, bleeding** only
  run when the record makes them possible (an opioid order, chemotherapy, insulin,
  AF, an anticoagulant).

Observation only: a finding can raise a page; it never places or changes an order.
Thresholds and wording follow the cited sources; nothing here is fitted to data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from backend.aegis.rooms.formulary import (ALLERGY_RULES, CONTRAST_PROCEDURES, FORMULARY, PAIR_RULES,
                                           creatinine_clearance, matches)

SEVERITY_RANK = {"info": 0, "watch": 1, "urgent": 2, "critical": 3}


@dataclass
class Finding:
    key: str                      # stable within the room ("news2", "med:allergy:piperacillin-tazobactam")
    detector: str                 # "NEWS2", "qSOFA", "Medication safety", …
    title: str
    severity: str                 # info | watch | urgent | critical
    category: str                 # deterioration | sepsis | respiratory | cardiac | metabolic | renal | medication | safety | neuro
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    context: List[str] = field(default_factory=list)      # what in the record changed the reading
    response: str = ""                                    # the escalation the rule calls for
    source: str = ""
    score: Optional[float] = None
    page: Optional[int] = None    # the priority this finding pages at (None = shown, not paged)
    page_category: str = "general"
    static: bool = False          # from the record alone (orders, allergies): no vitals involved

    def to_dict(self) -> Dict[str, Any]:
        # shallow on purpose: every finding's lists are built fresh by its detector
        # (dataclasses.asdict deep-copies recursively and was most of an evaluation's cost)
        return {"key": self.key, "detector": self.detector, "title": self.title, "severity": self.severity,
                "category": self.category, "evidence": self.evidence, "context": self.context,
                "response": self.response, "source": self.source, "score": self.score, "page": self.page,
                "page_category": self.page_category, "static": self.static}


@dataclass
class Ctx:
    room: str
    kind: str                                  # synthetic | replay | live
    record: Dict[str, Any]
    v: Dict[str, Any]                          # latest (smoothed) vitals
    v_at: Optional[float]
    series: List[Dict[str, Any]]               # [{t, hr, sbp, …}] oldest first
    labs: Dict[str, List[Dict[str, float]]]    # name -> [{t, v}] oldest first
    now: float
    clock: str = "wall"                        # wall | icu_hours (replay rooms)
    ml: Optional[Dict[str, Any]] = None        # the sepsis detector's state for monitored rooms
    flags: frozenset = frozenset()

    def has(self, *flags: str) -> bool:
        return any(f in self.flags for f in flags)

    def lab(self, name: str) -> Optional[Dict[str, float]]:
        xs = self.labs.get(name) or []
        return xs[-1] if xs else None

    def active_orders(self) -> List[Dict[str, Any]]:
        return [o for o in self.record.get("orders") or [] if o.get("status") == "active" and o.get("kind") == "drug"]


def _ev(label: str, value: Any, unit: str = "", pts: Optional[int] = None, at: Optional[float] = None) -> Dict[str, Any]:
    e: Dict[str, Any] = {"label": label, "value": value, "unit": unit}
    if pts is not None:
        e["points"] = pts
    if at is not None:
        e["at"] = at
    return e


def _fmt(x: Optional[float], nd: int = 0) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if nd else f"{round(x):d}")


# --------------------------------------------------------------------------- #
# NEWS2 (Royal College of Physicians, 2017)
# --------------------------------------------------------------------------- #
def _rr_pts(rr: float) -> int:
    return 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3


def _spo2_scale1(s: float) -> int:
    return 3 if s <= 91 else 2 if s <= 93 else 1 if s <= 95 else 0


def _spo2_scale2(s: float, on_o2: bool) -> int:
    if s <= 83:
        return 3
    if s <= 85:
        return 2
    if s <= 87:
        return 1
    if s <= 92 or not on_o2:
        return 0
    return 1 if s <= 94 else 2 if s <= 96 else 3


def _sbp_pts(sbp: float) -> int:
    return 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0 if sbp <= 219 else 3


def _hr_pts(hr: float) -> int:
    return 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3


def _temp_pts(t: float) -> int:
    return 3 if t <= 35.0 else 1 if t <= 36.0 else 0 if t <= 38.0 else 1 if t <= 39.0 else 2


def news2_score(v: Dict[str, Any], scale2: bool, chronic_confusion: bool) -> Dict[str, Any]:
    """The aggregate and each parameter's points. Missing parameters are listed, not guessed."""
    parts: List[Tuple[str, Any, str, int]] = []
    missing: List[str] = []
    on_o2 = (v.get("o2_lpm") or 0) > 0
    if v.get("rr") is not None:
        parts.append(("Respiration rate", v["rr"], "/min", _rr_pts(v["rr"])))
    else:
        missing.append("respiration rate")
    if v.get("spo2") is not None:
        pts = _spo2_scale2(v["spo2"], on_o2) if scale2 else _spo2_scale1(v["spo2"])
        parts.append((f"SpO₂ (scale {2 if scale2 else 1})", v["spo2"], "%", pts))
    else:
        missing.append("SpO₂")
    if v.get("o2_lpm") is not None:
        parts.append(("Air or oxygen", f"O₂ {v['o2_lpm']:g} L/min" if on_o2 else "air", "", 2 if on_o2 else 0))
    else:
        missing.append("oxygen")
    if v.get("sbp") is not None:
        parts.append(("Systolic BP", v["sbp"], "mmHg", _sbp_pts(v["sbp"])))
    else:
        missing.append("systolic BP")
    if v.get("hr") is not None:
        parts.append(("Pulse", v["hr"], "/min", _hr_pts(v["hr"])))
    else:
        missing.append("pulse")
    avpu = v.get("avpu")
    if avpu is not None:
        # new confusion scores 3; confusion the record documents as the patient's baseline does not
        new = avpu in ("V", "P", "U") or (avpu == "C" and not chronic_confusion)
        label = {"A": "alert", "C": "confused", "V": "voice", "P": "pain", "U": "unresponsive"}[avpu]
        if avpu == "A" and chronic_confusion:
            label = "baseline (confused at baseline)"
        parts.append(("Consciousness", label, "", 3 if new else 0))
    else:
        missing.append("consciousness")
    if v.get("temp_c") is not None:
        parts.append(("Temperature", v["temp_c"], "°C", _temp_pts(v["temp_c"])))
    else:
        missing.append("temperature")
    total = sum(p[3] for p in parts)
    red = any(p[3] == 3 for p in parts)
    band = "high" if total >= 7 else "medium" if total >= 5 else "low-medium" if red else "low"
    return {"score": total, "band": band, "parts": parts, "missing": missing, "single_red": red,
            "complete": not missing}


def detect_news2(c: Ctx) -> List[Finding]:
    if c.kind == "replay":
        return []                  # NEWS2 is a ward score; the ICU runs the sepsis detector and SOFA instead
    target = c.record.get("spo2_target")
    scale2 = bool(target and target.get("scale") == 2)
    chronic = bool(c.v.get("chronic_confusion"))
    n = news2_score(c.v, scale2, chronic)
    if not n["parts"]:
        return []
    ctx: List[str] = []
    if scale2:
        alt = news2_score(c.v, False, chronic)
        ctx.append(f"SpO₂ scale 2 from the record: {target['why']} (target {target['lo']}–{target['hi']}%)."
                   + (f" Scale 1 would read {alt['score']}." if alt["score"] != n["score"] else ""))
    if chronic:
        ctx.append("Confusion is documented as this patient's baseline, so it scores only if it worsens.")
    if "beta_blocker" in c.flags:
        ctx.append("On a beta-blocker: the pulse may not rise with deterioration, so a normal pulse reassures less.")
    if c.record.get("code_status") and c.record["code_status"] != "Full code":
        ctx.append(f"Code status {c.record['code_status']}: escalate within the documented goals of care.")
    if n["missing"]:
        ctx.append("Not measured here: " + ", ".join(n["missing"]) + ".")
    ev = [_ev(lbl, val, unit, pts) for lbl, val, unit, pts in n["parts"]]
    s, band = n["score"], n["band"]
    if band == "high":
        return [Finding("news2", "NEWS2", f"NEWS2 {s}: emergency response threshold", "critical", "deterioration", ev, ctx,
                        "Emergency assessment by a clinical team with critical-care competencies; continuous monitoring.",
                        "NEWS2, Royal College of Physicians 2017", s, page=2 if n["complete"] else None,
                        page_category="general")]
    if band == "medium":
        return [Finding("news2", "NEWS2", f"NEWS2 {s}: urgent response threshold", "urgent", "deterioration", ev, ctx,
                        "Urgent review by a clinician competent in acute illness; observations at least hourly.",
                        "NEWS2, Royal College of Physicians 2017", s, page=3 if n["complete"] else None,
                        page_category="general")]
    if band == "low-medium":
        red = next(p for p in n["parts"] if p[3] == 3)
        return [Finding("news2", "NEWS2", f"NEWS2 {s} with a single red score ({red[0].lower()})", "watch",
                        "deterioration", ev, ctx,
                        "Urgent ward-based review of the extreme parameter; observations at least hourly.",
                        "NEWS2, Royal College of Physicians 2017", s)]
    return [Finding("news2", "NEWS2", f"NEWS2 {s}", "info", "deterioration", ev, ctx,
                    "Routine: observations at least every 4–6 hours." if s else "Routine: observations at least every 12 hours.",
                    "NEWS2, Royal College of Physicians 2017", s)]


# --------------------------------------------------------------------------- #
# Sepsis screens
# --------------------------------------------------------------------------- #
def _infection_context(c: Ctx) -> Optional[str]:
    if c.has("infection", "neutropenia"):
        return "suspected infection: " + (c.record.get("admission") or {}).get("reason", "on the record")
    t = c.v.get("temp_c")
    if t is not None and (t >= 38.3 or t < 36.0):
        return f"temperature {t:.1f} °C"
    w = c.lab("wbc")
    if w and (w["v"] > 12 or w["v"] < 4):
        return f"WBC {w['v']:.1f}"
    if c.has("post_op"):
        return "post-operative"
    return None


def detect_qsofa(c: Ctx) -> List[Finding]:
    rr, sbp, avpu = c.v.get("rr"), c.v.get("sbp"), c.v.get("avpu")
    if rr is None and sbp is None:
        return []
    altered = avpu in ("V", "P", "U") or (avpu == "C" and not c.v.get("chronic_confusion"))
    pts = int(rr is not None and rr >= 22) + int(sbp is not None and sbp <= 100) + int(bool(altered))
    measurable = 3 - int(rr is None) - int(sbp is None) - int(avpu is None)
    if pts < 2:
        return []
    inf = _infection_context(c)
    ev = [_ev("Respiratory rate ≥ 22", c.v.get("rr"), "/min", int(rr is not None and rr >= 22)),
          _ev("Systolic ≤ 100", c.v.get("sbp"), "mmHg", int(sbp is not None and sbp <= 100)),
          _ev("Altered mentation", {"A": "no", None: "not assessed"}.get(avpu, "yes"), "", int(bool(altered)))]
    ctx = [f"With {inf}." if inf else "No infection on the record or in the vitals yet."]
    if measurable < 3:
        ctx.append(f"{measurable} of 3 criteria measurable here.")
    sev = "critical" if inf and pts == 3 else "urgent"
    return [Finding("qsofa", "qSOFA", f"qSOFA {pts}" + (" with suspected infection" if inf else ""), sev, "sepsis",
                    ev, ctx, "Assess for sepsis and organ dysfunction now: lactate, cultures, review within the hour.",
                    "qSOFA, Sepsis-3 (Seymour et al., JAMA 2016)", pts,
                    page=(2 if inf and c.kind != "replay" else None), page_category="sepsis")]


def detect_sirs(c: Ctx) -> List[Finding]:
    t, hr, rr = c.v.get("temp_c"), c.v.get("hr"), c.v.get("rr")
    w = c.lab("wbc")
    crit = [
        ("Temperature > 38 or < 36", t, "°C", t is not None and (t > 38.0 or t < 36.0)),
        ("Heart rate > 90", hr, "/min", hr is not None and hr > 90),
        ("Respiratory rate > 20", rr, "/min", rr is not None and rr > 20),
        ("WBC > 12 or < 4", w["v"] if w else None, "×10⁹/L", bool(w and (w["v"] > 12 or w["v"] < 4))),
    ]
    met = sum(1 for x in crit if x[3])
    if met < 2:
        return []
    inf = _infection_context(c)
    ctx = [f"With {inf}." if inf else "No infection on the record: SIRS alone is non-specific here."]
    if "beta_blocker" in c.flags:
        ctx.append("On a beta-blocker: the heart-rate criterion may be blunted.")
    if c.has("post_op"):
        ctx.append("Post-operative inflammation also meets SIRS; read alongside the infection context.")
    return [Finding("sirs", "SIRS", f"SIRS {met} of 4", "watch" if inf and met >= 3 else "info", "sepsis",
                    [_ev(l, v, u, int(m)) for l, v, u, m in crit], ctx,
                    "Low specificity on its own: look for a source and for organ dysfunction.",
                    "SIRS, ACCP/SCCM consensus 1992", met)]


# --------------------------------------------------------------------------- #
# Haemodynamics
# --------------------------------------------------------------------------- #
def detect_shock_index(c: Ctx) -> List[Finding]:
    hr, sbp, mp = c.v.get("hr"), c.v.get("sbp"), c.v.get("map")
    out: List[Finding] = []
    if hr and sbp:
        si = hr / sbp
        age = (c.record.get("patient") or {}).get("age")
        ctx = []
        if age and age >= 65:
            ctx.append(f"Age {int(age)}: age × SI = {age * si:.0f} (above 50 predicts poorer outcome in older adults).")
        if "beta_blocker" in c.flags:
            ctx.append("Beta-blockade lowers the shock index; a normal value reassures less.")
        if si >= 1.0:
            out.append(Finding("shock_index", "Shock index", f"Shock index {si:.2f}", "urgent" if si < 1.3 else "critical",
                               "deterioration", [_ev("Heart rate", hr, "/min"), _ev("Systolic", sbp, "mmHg"),
                                                 _ev("HR / SBP", round(si, 2))], ctx,
                               "Haemodynamic instability: assess volume status and bleeding or sepsis now.",
                               "Shock index (Allgöwer & Burri 1967)", round(si, 2),
                               page=(2 if si >= 1.3 and c.kind == "synthetic" else None), page_category="general"))
        elif si >= 0.9:
            out.append(Finding("shock_index", "Shock index", f"Shock index {si:.2f}", "watch", "deterioration",
                               [_ev("Heart rate", hr, "/min"), _ev("Systolic", sbp, "mmHg"), _ev("HR / SBP", round(si, 2))],
                               ctx, "Borderline: trend it with the next observations.",
                               "Shock index (Allgöwer & Burri 1967)", round(si, 2)))
    if mp is not None and mp < 65:
        out.append(Finding("map_floor", "Perfusion pressure", f"MAP {mp:.0f} below 65", "critical", "deterioration",
                           [_ev("Mean arterial pressure", round(mp), "mmHg")], [],
                           "Below the Surviving Sepsis Campaign perfusion target: bedside assessment now.",
                           "Surviving Sepsis Campaign 2021 (MAP target ≥ 65)", round(mp),
                           page=(2 if c.kind == "synthetic" else None), page_category="general"))
    return out


# --------------------------------------------------------------------------- #
# The record's own baseline
# --------------------------------------------------------------------------- #
def detect_baseline(c: Ctx) -> List[Finding]:
    b = c.record.get("baseline")
    if not b:
        return []
    out: List[Finding] = []
    src = b.get("source", "earlier encounters")
    sbp, hr, rr, spo2 = c.v.get("sbp"), c.v.get("hr"), c.v.get("rr"), c.v.get("spo2")
    if sbp is not None and b.get("sbp"):
        drop = (b["sbp"] - sbp) / b["sbp"]
        if drop >= 0.25:
            plain = sbp > 100
            out.append(Finding("baseline:sbp", "Personal baseline", f"Systolic {sbp:.0f}, {drop * 100:.0f}% below this patient's baseline",
                               "urgent" if drop >= 0.33 else "watch", "deterioration",
                               [_ev("Systolic now", sbp, "mmHg"), _ev("Baseline systolic", round(b["sbp"]), "mmHg"),
                                _ev("Change", f"−{drop * 100:.0f}%")],
                               [f"Baseline is the {src}."] + (["Above 100, so population thresholds score it as normal or near-normal."] if plain else []),
                               "Relative hypotension: review against the patient's usual pressure, not the population's.",
                               "Personal baseline from the record", round(drop * 100)))
    if hr is not None and b.get("hr"):
        rise = hr - b["hr"]
        if rise >= 30 and hr > 95:
            out.append(Finding("baseline:hr", "Personal baseline", f"Pulse {hr:.0f}, {rise:.0f} above this patient's baseline",
                               "watch", "deterioration",
                               [_ev("Pulse now", hr, "/min"), _ev("Baseline pulse", round(b["hr"]), "/min")],
                               [f"Baseline is the {src}."], "Look for the cause: pain, fever, volume, bleeding, arrhythmia.",
                               "Personal baseline from the record", round(rise)))
    if rr is not None and b.get("rr"):
        rise = rr - b["rr"]
        if rise >= 8:
            out.append(Finding("baseline:rr", "Personal baseline", f"Respiratory rate {rr:.0f}, {rise:.0f} above baseline",
                               "watch", "respiratory",
                               [_ev("Rate now", rr, "/min"), _ev("Baseline rate", round(b["rr"]), "/min")],
                               [f"Baseline is the {src}."], "A rising respiratory rate is the earliest sign in many deteriorations.",
                               "Personal baseline from the record", round(rise)))
    if spo2 is not None and b.get("spo2"):
        tgt = c.record.get("spo2_target")
        floor = tgt["lo"] if tgt else None
        fall = b["spo2"] - spo2
        if fall >= 5 and (floor is None or spo2 < floor):
            out.append(Finding("baseline:spo2", "Personal baseline", f"SpO₂ {spo2:.0f}%, {fall:.0f} points below baseline",
                               "watch", "respiratory",
                               [_ev("SpO₂ now", spo2, "%"), _ev("Baseline SpO₂", round(b["spo2"]), "%")]
                               + ([_ev("Prescribed target", f"{tgt['lo']}–{tgt['hi']}", "%")] if tgt else []),
                               [f"Baseline is the {src}."], "Below the patient's own usual saturation.",
                               "Personal baseline from the record", round(fall)))
    return out


# --------------------------------------------------------------------------- #
# Trend: worsening that has not crossed a threshold yet
# --------------------------------------------------------------------------- #
_TREND = {  # channel -> (direction, change over the window that counts, label, unit)
    "hr": (+1, 18.0, "Pulse", "/min"),
    "rr": (+1, 5.0, "Respiratory rate", "/min"),
    "sbp": (-1, 18.0, "Systolic", "mmHg"),
    "spo2": (-1, 3.5, "SpO₂", "%"),
    "temp_c": (+1, 0.9, "Temperature", "°C"),
}


def _slope(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Least-squares slope and R², in one pass over centred sums."""
    n = len(xs)
    x0 = xs[0]
    sx = sy = sxx = sxy = syy = 0.0
    for x, y in zip(xs, ys):
        x -= x0
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
        syy += y * y
    vx = sxx - sx * sx / n
    if vx <= 0:
        return 0.0, 0.0
    cxy = sxy - sx * sy / n
    vy = syy - sy * sy / n
    b = cxy / vx
    r2 = (cxy * cxy) / (vx * vy) if vy > 0 else 0.0
    return b, r2


#: share of the trend window the samples must span before a change over it is reported
TREND_COVERAGE = 0.75


def detect_trend(c: Ctx) -> List[Finding]:
    if c.kind == "replay":
        window, unit_label = 6.0, "6 h"          # hourly real data
    else:
        window, unit_label = 900.0, "15 min"     # one sample every 10 s
    pts = [s for s in c.series if s.get("t") is not None and s["t"] >= c.now - window]
    if len(pts) < (5 if c.kind == "replay" else 30):
        return []
    worsening: List[Tuple[str, float, float]] = []
    for ch, (sign, thresh, label, unit) in _TREND.items():
        xy = [(s["t"], s[ch]) for s in pts if s.get(ch) is not None]
        if len(xy) < (5 if c.kind == "replay" else 30):
            continue
        # The change is reported over the whole window, so the samples must span most of
        # it. The live room samples every ~3 s: thirty samples are 90 s, and a 10 bpm rise
        # across them used to be reported as "+107 over 15 min".
        if xy[-1][0] - xy[0][0] < TREND_COVERAGE * window:
            continue
        b, r2 = _slope([x for x, _ in xy], [y for _, y in xy])
        change = b * window
        if sign * change >= thresh and r2 >= 0.45:
            worsening.append((label, change, r2))
    if not worsening:
        return []
    ev = [_ev(label, f"{'+' if ch > 0 else '−'}{abs(ch):.1f}" if label == "Temperature" else f"{'+' if ch > 0 else '−'}{abs(ch):.0f}",
              f"over {unit_label}") for label, ch, _r in worsening]
    sev = "urgent" if len(worsening) >= 3 else "watch"
    return [Finding("trend", "Trend", f"{len(worsening)} vital sign{'s' if len(worsening) > 1 else ''} worsening over {unit_label}",
                    sev, "deterioration", ev,
                    ["Least-squares slope over the window; counted only when the fit is consistent (R² ≥ 0.45)."],
                    "Worsening before any threshold is crossed: shorten the observation interval and look for the cause.",
                    "Trend detector (least-squares slope)", float(len(worsening)))]


# --------------------------------------------------------------------------- #
# Conditions the record makes possible
# --------------------------------------------------------------------------- #
def detect_opioid(c: Ctx) -> List[Finding]:
    ops = [o for o in c.active_orders() if "opioid" in o.get("classes", [])]
    if not ops:
        return []
    rr, spo2, avpu = c.v.get("rr"), c.v.get("spo2"), c.v.get("avpu")
    sedated = avpu in ("V", "P", "U")
    if rr is None:
        return []
    names = ", ".join(o["name"] for o in ops)
    benzo = [o["name"] for o in c.active_orders() if "benzodiazepine" in o.get("classes", [])]
    ctx = [f"Active opioid order: {names}."] + ([f"Also on {', '.join(benzo)}: additive respiratory depression."] if benzo else [])
    if c.has("copd", "hypercapnic"):
        ctx.append("COPD with CO₂ retention: lower reserve against hypoventilation.")
    if rr <= 8 or (rr <= 10 and sedated):
        return [Finding("opioid_rd", "Opioid safety", f"Respiratory depression on {ops[0]['name']}: RR {rr:.0f}" + (", rousable to voice only" if sedated else ""),
                        "critical", "respiratory",
                        [_ev("Respiratory rate", rr, "/min"), _ev("SpO₂", spo2, "%"), _ev("Consciousness", avpu or "—")], ctx,
                        "Rapid response: stimulate, stop the opioid infusion or PCA, support breathing; the prescriber decides on naloxone.",
                        "Opioid-induced respiratory depression (APSF; POSS sedation scale)", rr,
                        page=1, page_category="respiratory")]
    if rr <= 10 or sedated or (spo2 is not None and spo2 <= 90 and rr <= 12):
        what = "rousable to voice only" if sedated else f"RR {rr:.0f}" if rr <= 10 else f"SpO₂ {spo2:.0f}%"
        return [Finding("opioid_rd", "Opioid safety", f"Early sedation on {ops[0]['name']}: {what}", "urgent", "respiratory",
                        [_ev("Respiratory rate", rr, "/min"), _ev("SpO₂", spo2, "%"), _ev("Consciousness", avpu or "—")], ctx,
                        "Assess sedation (POSS); hold further opioid doses until reviewed; continuous oximetry.",
                        "Opioid-induced respiratory depression (APSF; POSS sedation scale)", rr,
                        page=2, page_category="respiratory")]
    return []


def detect_neutropenic_fever(c: Ctx) -> List[Finding]:
    if not c.has("neutropenia", "chemo"):
        return []
    t = c.v.get("temp_c")
    if t is None or t < 38.0:
        return []
    anc = c.lab("anc")
    abx = [o["name"] for o in c.active_orders() if "antibiotic" in o.get("classes", [])]
    ctx = ["Chemotherapy / neutropenia on the record."] + ([f"ANC {anc['v']:.0f} /µL."] if anc else [])
    ev = [_ev("Temperature", t, "°C")] + ([_ev("ANC", anc["v"], "/µL")] if anc else [])
    if abx:
        # already on empirical cover: a new spike needs review, not the 60-minute antibiotic emergency
        ctx.append(f"Already on empirical antibiotics: {', '.join(abx)}.")
        if t < 38.3:
            return [Finding("neutropenic_fever", "Neutropenic fever", f"Temperature {t:.1f} °C on empirical antibiotics", "watch",
                            "sepsis", ev, ctx, "Recheck within the hour.",
                            "IDSA 2010 / ASCO–IDSA 2018 febrile neutropenia", t)]
        return [Finding("neutropenic_fever", "Neutropenic fever", f"New fever {t:.1f} °C on empirical antibiotics", "urgent",
                        "sepsis", ev, ctx, "Clinical review now: repeat cultures, look for a source, consider broadening cover.",
                        "IDSA 2010 / ASCO–IDSA 2018 febrile neutropenia", t, page=3, page_category="sepsis")]
    return [Finding("neutropenic_fever", "Neutropenic fever", f"Temperature {t:.1f} °C in a neutropenic patient", "critical",
                    "sepsis", ev, ctx + ["No empirical antibiotic is active."],
                    "Time-critical: empirical antibiotics within 60 minutes of recognition; cultures first if no delay.",
                    "IDSA 2010 / ASCO–IDSA 2018 febrile neutropenia", t, page=2, page_category="sepsis")]


def detect_glucose(c: Ctx) -> List[Finding]:
    g = c.lab("glucose")
    if not g:
        return []
    if c.now - g["t"] > (6.0 if c.kind == "replay" else 3600.0):     # a stale value is not a finding
        return []
    v = g["v"]
    insulin = [o["name"] for o in c.active_orders() if "insulin" in o.get("classes", [])]
    ctx = [f"On {', '.join(insulin)}."] if insulin else []
    if v < 54:
        return [Finding("glucose", "Glucose", f"Glucose {v:.0f} mg/dL: level 2 hypoglycaemia", "critical", "metabolic",
                        [_ev("Glucose", v, "mg/dL", at=g["t"])], ctx,
                        "Treat now per the hypoglycaemia protocol and recheck in 15 minutes; review the insulin order.",
                        "ADA Standards of Care: hypoglycaemia levels", v, page=2, page_category="general")]
    if v < 70:
        return [Finding("glucose", "Glucose", f"Glucose {v:.0f} mg/dL: level 1 hypoglycaemia", "urgent", "metabolic",
                        [_ev("Glucose", v, "mg/dL", at=g["t"])], ctx,
                        "Treat per protocol and recheck in 15 minutes.",
                        "ADA Standards of Care: hypoglycaemia levels", v)]
    if v > 300:
        return [Finding("glucose", "Glucose", f"Glucose {v:.0f} mg/dL", "watch", "metabolic",
                        [_ev("Glucose", v, "mg/dL", at=g["t"])], ctx, "Check ketones if type 1 diabetes; review insulin.",
                        "ADA Standards of Care", v)]
    return []


def detect_af(c: Ctx) -> List[Finding]:
    if not c.has("af"):
        return []
    hr, sbp = c.v.get("hr"), c.v.get("sbp")
    if hr is None or hr <= 110:
        return []
    rate = [o["name"] for o in c.active_orders() if set(o.get("classes", [])) & {"beta_blocker", "ccb_nondhp"}]
    ctx = ["Atrial fibrillation on the record."] + ([f"Rate control ordered: {', '.join(rate)}."] if rate else ["No rate-control agent active."])
    unstable = sbp is not None and sbp < 90
    if hr >= 140 or unstable:
        return [Finding("af_rvr", "Rhythm", f"AF with rapid ventricular response: {hr:.0f}/min" + (", hypotensive" if unstable else ""),
                        "critical", "cardiac", [_ev("Ventricular rate", hr, "/min"), _ev("Systolic", sbp, "mmHg")], ctx,
                        "Assess stability now; 12-lead ECG; unstable rapid AF needs immediate senior review.",
                        "AHA/ACC/HRS 2023 AF guideline (lenient rate target < 110)", hr,
                        page=2, page_category="cardiac")]
    return [Finding("af_rvr", "Rhythm", f"AF rate {hr:.0f}/min, above the lenient target", "watch", "cardiac",
                    [_ev("Ventricular rate", hr, "/min")], ctx, "Above the lenient rate-control target of 110.",
                    "AHA/ACC/HRS 2023 AF guideline (lenient rate target < 110)", hr)]


def detect_bleeding(c: Ctx) -> List[Finding]:
    anti = [o["name"] for o in c.active_orders() if set(o.get("classes", [])) & {"anticoagulant", "antiplatelet"}]
    if not anti:
        return []
    hb = c.labs.get("hemoglobin") or []
    drop = None
    if len(hb) >= 2:
        recent = [x for x in hb if c.now - x["t"] <= 36 * 3600] if c.kind != "replay" else hb[-6:]
        if len(recent) >= 2:
            drop = max(x["v"] for x in recent[:-1]) - recent[-1]["v"]
    b = c.record.get("baseline") or {}
    hr, sbp = c.v.get("hr"), c.v.get("sbp")
    haem = bool(b.get("hr") and b.get("sbp") and hr is not None and sbp is not None
                and hr - b["hr"] >= 25 and sbp <= b["sbp"] * 0.8)
    if (drop is not None and drop >= 2.0) or (haem and drop is not None and drop >= 1.0):
        return [Finding("bleeding", "Bleeding", f"Haemoglobin down {drop:.1f} g/dL on {anti[0]}", "critical", "deterioration",
                        [_ev("Haemoglobin", hb[-1]["v"], "g/dL", at=hb[-1]["t"]), _ev("Fall", f"−{drop:.1f}", "g/dL"),
                         _ev("Pulse", hr, "/min"), _ev("Systolic", sbp, "mmHg")],
                        [f"Antithrombotics active: {', '.join(anti)}."],
                        "Possible bleed: look for a source, group and screen, senior review of the antithrombotics.",
                        "Major bleeding definition (ISTH: Hb fall ≥ 2 g/dL)", drop, page=2, page_category="surgical")]
    if haem:
        return [Finding("bleeding", "Bleeding", f"Pulse up and pressure down on {anti[0]}", "urgent", "deterioration",
                        [_ev("Pulse", hr, "/min"), _ev("Baseline pulse", round(b["hr"]), "/min"),
                         _ev("Systolic", sbp, "mmHg"), _ev("Baseline systolic", round(b["sbp"]), "mmHg")],
                        [f"Antithrombotics active: {', '.join(anti)}.", f"Baseline is the {b.get('source', 'record')}."],
                        "Haemodynamic change on antithrombotics: check haemoglobin and look for bleeding.",
                        "Haemodynamics against the record's baseline", None)]
    return []


def detect_kidney(c: Ctx) -> List[Finding]:
    """KDIGO 2012 creatinine criteria against the record's own baseline."""
    cr = c.labs.get("creatinine") or []
    if len(cr) < 2:
        return []
    cur = cr[-1]
    hour = 3600.0 if c.kind != "replay" else 1.0
    # the reference: a documented baseline if the record has one, else the lowest value in the prior 7 days
    base_rows = [x for x in cr[:-1] if x.get("baseline")] or [x for x in cr[:-1] if cur["t"] - x["t"] <= 168 * hour]
    if not base_rows:
        return []
    base = min(x["v"] for x in base_rows)
    ratio = cur["v"] / base if base > 0 else 1.0
    recent = [x for x in cr if cur["t"] - x["t"] <= 48 * hour]
    rise48 = cur["v"] - min(x["v"] for x in recent) if recent else 0.0
    stage = 3 if (ratio >= 3.0 or cur["v"] >= 4.0) else 2 if ratio >= 2.0 else 1 if (ratio >= 1.5 or rise48 >= 0.3) else 0
    if stage == 0:
        return []
    renal = [f"{o['name']} ({o['renal']})" for o in c.active_orders() if o.get("renal")]
    ctx = [f"Baseline creatinine {base:.2f} mg/dL ({'from the record' if any(x.get('baseline') for x in cr) else 'lowest in the prior 7 days'})."]
    if renal:
        ctx.append("Renally cleared orders to review: " + "; ".join(renal) + ".")
    return [Finding("aki", "Acute kidney injury", f"KDIGO stage {stage}: creatinine {cur['v']:.2f} mg/dL", "urgent" if stage >= 2 else "watch",
                    "renal", [_ev("Creatinine", cur["v"], "mg/dL", at=cur["t"]), _ev("Baseline", base, "mg/dL"),
                              _ev("Ratio", f"×{ratio:.1f}"), _ev("Rise in 48 h", f"+{rise48:.2f}", "mg/dL")], ctx,
                    "Review fluids, nephrotoxins and renally cleared doses; urine output.",
                    "KDIGO 2012 AKI creatinine criteria", float(stage))]


def detect_potassium(c: Ctx) -> List[Finding]:
    k = c.lab("potassium")
    if not k or k["v"] < 5.5 or c.now - k["t"] > (24.0 if c.kind == "replay" else 24 * 3600.0):
        return []
    ctx = []
    sparing = [o["name"] for o in c.active_orders() if set(o.get("classes", [])) & {"k_sparing", "potassium"}]
    if sparing:
        ctx.append("Contributing orders: " + ", ".join(sparing) + ".")
    if c.has("ckd", "renal_impairment"):
        ctx.append("Renal impairment on the record.")
    hr = c.v.get("hr")
    sev = "critical" if k["v"] >= 6.0 else "watch"
    return [Finding("potassium", "Potassium", f"Potassium {k['v']:.1f} mmol/L", sev, "metabolic",
                    [_ev("Potassium", k["v"], "mmol/L", at=k["t"])] + ([_ev("Pulse", hr, "/min")] if hr else []), ctx,
                    "12-lead ECG now; treat per the hyperkalaemia protocol; stop potassium-raising orders." if sev == "critical"
                    else "Repeat and review potassium-raising orders.",
                    "UK Renal Association hyperkalaemia guideline (≥ 6.0 severe)", k["v"],
                    page=(2 if sev == "critical" and c.kind == "synthetic" else None), page_category="general")]


def detect_lactate(c: Ctx) -> List[Finding]:
    lac = c.lab("lactate")
    if not lac or lac["v"] < 2.0:
        return []
    if c.now - lac["t"] > (12.0 if c.kind == "replay" else 4 * 3600.0):
        return []
    sev = "critical" if lac["v"] >= 4.0 else "urgent"
    inf = _infection_context(c)
    return [Finding("lactate", "Lactate", f"Lactate {lac['v']:.1f} mmol/L", sev, "sepsis",
                    [_ev("Lactate", lac["v"], "mmol/L", at=lac["t"])], [f"With {inf}."] if inf else [],
                    "Hypoperfusion: remeasure within 2–4 hours; ≥ 4 mmol/L warrants immediate resuscitation review.",
                    "Surviving Sepsis Campaign 2021", lac["v"],
                    page=(2 if sev == "critical" and inf and c.kind == "synthetic" else None), page_category="sepsis")]


def detect_sofa(c: Ctx) -> List[Finding]:
    """SOFA organ points from the real labs of a replayed ICU stay (no GCS or vasopressors in CinC)."""
    if c.kind != "replay":
        return []

    def last(name: str) -> Optional[float]:
        x = c.lab(name)
        return x["v"] if x else None

    def first_day(name: str) -> Optional[float]:
        xs = [x["v"] for x in c.labs.get(name) or [] if x["t"] <= 24]
        return xs[0] if xs else None

    def points(plt: Optional[float], bili: Optional[float], cr: Optional[float], mp: Optional[float]) -> Tuple[int, List[Tuple[str, Any, int]]]:
        rows = []
        if plt is not None:
            rows.append(("Coagulation · platelets", plt, 4 if plt < 20 else 3 if plt < 50 else 2 if plt < 100 else 1 if plt < 150 else 0))
        if bili is not None:
            rows.append(("Liver · bilirubin", bili, 4 if bili >= 12 else 3 if bili >= 6 else 2 if bili >= 2 else 1 if bili >= 1.2 else 0))
        if cr is not None:
            rows.append(("Renal · creatinine", cr, 4 if cr >= 5 else 3 if cr >= 3.5 else 2 if cr >= 2 else 1 if cr >= 1.2 else 0))
        if mp is not None:
            rows.append(("Cardiovascular · MAP", mp, 1 if mp < 70 else 0))
        return sum(r[2] for r in rows), rows

    now_pts, rows = points(last("platelets"), last("bilirubin"), last("creatinine"), c.v.get("map"))
    if not rows:
        return []
    first_map = next((s.get("map") for s in c.series if s.get("map") is not None), None)
    base_pts, _ = points(first_day("platelets"), first_day("bilirubin"), first_day("creatinine"), first_map)
    delta = now_pts - base_pts
    ev = [_ev(l, round(v, 2) if isinstance(v, float) else v, "", p) for l, v, p in rows]
    ctx = ["Respiratory and CNS components need PaO₂/FiO₂ and GCS, which this dataset does not carry.",
           f"First-day score {base_pts}, now {now_pts}."]
    if delta >= 2:
        return [Finding("sofa", "SOFA", f"SOFA up {delta} points from the first day ({now_pts})", "urgent", "sepsis", ev, ctx,
                        "An acute rise of 2 or more is the Sepsis-3 definition of organ dysfunction: review now.",
                        "SOFA (Vincent et al. 1996); Sepsis-3 ΔSOFA ≥ 2", float(now_pts))]
    return [Finding("sofa", "SOFA", f"SOFA {now_pts} (partial)", "info", "sepsis", ev, ctx,
                    "Tracked each time a lab result arrives.", "SOFA (Vincent et al. 1996)", float(now_pts))]


def detect_sepsis_model(c: Ctx) -> List[Finding]:
    ml = c.ml
    if not ml or ml.get("risk") is None:
        return []
    r = float(ml["risk"])
    live = c.kind == "live"
    alarm, watch = (0.70, 0.40) if live else (0.70, 0.60)
    # the risk index (ward_risk) and the calibrated probability it was mapped from
    prob = ml.get("probability")
    head = [_ev("Risk index", round(r, 3))] + ([_ev("Calibrated P(sepsis)", round(float(prob), 4))]
                                                if prob is not None else [])
    if r >= alarm:
        return [Finding("sepsis_model", "Sepsis detector", f"Sepsis risk {r:.2f}, paged", "critical", "sepsis",
                        head + [_ev("Driver", d) for d in (ml.get("drivers") or [])[:3]],
                        ["Paged by the sepsis detector itself; the room monitor does not page twice."],
                        "Sepsis review now; the Hour-1 bundle team is notified with the page.",
                        "AegisOS stacked detector, validated on PhysioNet/CinC 2019", round(r, 3))]
    if r >= watch:
        return [Finding("sepsis_model", "Sepsis detector", f"Sepsis risk {r:.2f}: on watch", "watch", "sepsis",
                        head + [_ev("Driver", d) for d in (ml.get("drivers") or [])[:3]], [],
                        "Below the paging line: trend it.", "AegisOS stacked detector, validated on PhysioNet/CinC 2019",
                        round(r, 3))]
    return []


# --------------------------------------------------------------------------- #
# Medication safety: orders against the record
# --------------------------------------------------------------------------- #
def detect_medications(c: Ctx, screen: Optional[Callable[[List[str], Dict[str, Any]], List[Any]]] = None) -> List[Finding]:
    orders = c.active_orders()
    out: List[Finding] = []
    if not orders and not c.record.get("planned"):
        return out
    dx = ((c.record.get("admission") or {}).get("reason") or "").lower()
    drugs = [(o, FORMULARY.get(o["name"])) for o in orders]
    # 1. allergies
    for a in c.record.get("allergies") or []:
        rule = ALLERGY_RULES.get(a["substance"])
        if not rule:
            continue
        for o, d in drugs:
            if d is None:
                continue
            if rule.direct & d.classes:
                out.append(Finding(f"med:allergy:{o['name']}", "Medication safety",
                                   f"{o['name']} ordered with a documented {a['substance']} allergy", "critical", "medication",
                                   [_ev("Order", f"{o['name']} {o.get('dose', '')} {o.get('route', '')} {o.get('freq', '')}".strip()),
                                    _ev("Allergy", a["substance"]), _ev("Reaction", a.get("reaction")), _ev("Severity", a.get("severity"))],
                                   [f"Allergy recorded {a.get('recorded', 'on the record')}."],
                                   "Hold the next dose and contact the prescriber and pharmacist before it is given.",
                                   "Allergy record against active orders", None, page=3, page_category="medication", static=True))
            else:
                for cls, note in rule.cross.items():
                    if cls in d.classes:
                        sev = "watch" if a.get("severity") == "severe" else "info"
                        out.append(Finding(f"med:cross:{o['name']}", "Medication safety",
                                           f"{o['name']} with a {a['substance']} allergy ({a.get('reaction')})", sev, "medication",
                                           [_ev("Order", o["name"]), _ev("Allergy", a["substance"]), _ev("Reaction", a.get("reaction"))],
                                           [note], "Proceed only if the reaction history supports it; monitor the first dose.",
                                           "Cross-reactivity (AAAAI/ACAAI drug allergy practice parameter 2022)", None, static=True))
        if rule.kind == "procedure":
            for proc in c.record.get("planned") or []:
                if any(k in proc.lower() for k in CONTRAST_PROCEDURES):
                    out.append(Finding("med:contrast", "Medication safety",
                                       f"Contrast allergy with {proc.lower()} planned", "watch", "medication",
                                       [_ev("Planned", proc), _ev("Allergy", a["substance"]), _ev("Reaction", a.get("reaction"))],
                                       [], "Premedication plan and an alternative assessed before the procedure.",
                                       "ACR Manual on Contrast Media", None, static=True))
    # 2. curated pairs
    seen_pairs = set()
    for i in range(len(drugs)):
        for j in range(len(drugs)):
            if i >= j:
                continue
            (oa, da), (ob, db) = drugs[i], drugs[j]
            if da is None or db is None:
                continue
            for r in PAIR_RULES:
                hit = (matches(r.a, da) and matches(r.b, db)) or (matches(r.a, db) and matches(r.b, da))
                if not hit or (r.a == r.b and da.generic == db.generic):
                    continue
                key = f"med:pair:{r.title}:{oa['name']}+{ob['name']}"
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                expected = any(k in dx for k in r.expected_in)
                sev = "info" if expected else r.severity
                ctx = [f"Expected in this admission's regimen ({dx}); monitor for bleeding."] if expected else []
                if r.a == "beta_blocker" and c.v.get("hr") is not None and c.v["hr"] < 55:
                    sev, ctx = "urgent", ctx + [f"Pulse is {c.v['hr']:.0f} now."]
                if r.a == "k_sparing" and c.has("ckd", "renal_impairment"):
                    ctx.append("Renal impairment on the record raises the risk.")
                out.append(Finding(key, "Medication safety", f"{r.title}: {oa['name']} + {ob['name']}", sev, "medication",
                                   [_ev("Orders", f"{oa['name']}, {ob['name']}")], [r.why] + ctx,
                                   "Review the combination with the prescriber and pharmacist.", r.source, None,
                                   page=(3 if sev == "critical" else None), page_category="medication", static=True))
    # 3. QT (CredibleMeds known risk)
    qt = [o["name"] for o, d in drugs if d is not None and d.qt]
    if len(qt) >= 2:
        out.append(Finding("med:qt", "Medication safety", f"Two QT-prolonging drugs: {', '.join(qt)}", "watch", "medication",
                           [_ev("Orders", ", ".join(qt))], ["Additive QT prolongation; worse with low potassium or magnesium."],
                           "Baseline ECG and electrolytes.", "CredibleMeds QTdrugs (known risk)", None, static=True))
    # 4. renal dosing against this patient's creatinine clearance (Cockcroft–Gault from the record)
    cr = c.lab("creatinine")
    renal = c.has("ckd", "renal_impairment") or bool(cr and cr["v"] >= 1.5)
    pt = c.record.get("patient") or {}
    crcl = None
    if cr and pt.get("age") and pt.get("weight_kg"):
        crcl = creatinine_clearance(float(pt["age"]), float(pt["weight_kg"]), float(cr["v"]), pt.get("sex") == "F")
    for o, d in drugs:
        if d is None or not d.renal_below or crcl is None or crcl >= d.renal_below:
            continue
        out.append(Finding(f"med:renal:{o['name']}", "Medication safety",
                           f"{o['name']}: creatinine clearance {crcl:.0f} mL/min, below {d.renal_below:.0f}",
                           "watch", "medication",
                           [_ev("Order", f"{o['name']} {o.get('dose', '')} {o.get('freq', '')}".strip()),
                            _ev("Creatinine", cr["v"], "mg/dL", at=cr["t"]), _ev("CrCl (Cockcroft–Gault)", round(crcl), "mL/min"),
                            _ev("Weight", pt.get("weight_kg"), "kg")],
                           [f"{d.renal.capitalize()}."] + (["CKD on the record."] if c.has("ckd") else []),
                           "Confirm the dose against current renal function.",
                           "Product labelling; Cockcroft–Gault (1976)", round(crcl), static=True))
    # 5. reconciliation: home medications that should not stop silently
    active = {o["name"] for o in orders} | {FORMULARY[o["name"]].generic for o in orders if o["name"] in FORMULARY}
    classes = {cl for o in orders for cl in o.get("classes", [])}
    for h in c.record.get("home_meds") or []:
        n = h["name"].lower()
        if n in active or any(n in a for a in active):
            continue
        why = None
        if n in ("carvedilol", "metoprolol") and not classes & {"beta_blocker", "ccb_nondhp"}:
            why = "Abrupt beta-blocker withdrawal can cause rebound tachycardia and ischaemia."
        elif n == "levetiracetam" and "antiepileptic" not in classes:
            why = "Stopping an antiepileptic risks breakthrough seizures."
        elif n == "apixaban" and "anticoagulant" not in classes and "stroke" not in dx and "bleed" not in dx:
            why = "Anticoagulation for AF interrupted without a documented reason."
        elif n == "insulin glargine" and c.has("t1dm") and "insulin" not in classes:
            why = "Type 1 diabetes: basal insulin must never stop (diabetic ketoacidosis)."
        if why:
            out.append(Finding(f"med:recon:{n}", "Medication safety", f"Home {h['name']} not continued", "watch", "medication",
                               [_ev("Home medication", f"{h['name']} {h['dose']} {h['freq']}".strip())], [why],
                               "Reconcile: continue, or document why it is held.", "Medication reconciliation (Joint Commission NPSG 03.06.01)",
                               None, static=True))
    # 6. the general interaction knowledge base, for pairs the curated rules did not cover
    if screen is not None and len(orders) >= 2:
        try:
            covered = {frozenset(k.split(":")[-1].split("+")) for k in seen_pairs}
            generic = {FORMULARY[o["name"]].generic: o["name"] for o in orders if o["name"] in FORMULARY}
            for ix in screen(list(generic), {"renal_failure": renal}):
                if ix.b is None or ix.severity not in ("Major", "Moderate"):
                    continue
                pair = frozenset({generic.get(ix.a, ix.a), generic.get(ix.b, ix.b)})
                if pair in covered:
                    continue
                covered.add(pair)
                # a Moderate interaction in a large knowledge base is usually the expected regimen
                # (carvedilol with furosemide in heart failure): shown, not flagged
                out.append(Finding(f"med:kb:{'+'.join(sorted(pair))}", "Medication safety",
                                   f"Interaction: {' + '.join(sorted(pair))}",
                                   "watch" if ix.severity == "Major" else "info", "medication",
                                   [_ev("Orders", ", ".join(sorted(pair))), _ev("Knowledge base", ix.source)], [ix.mechanism],
                                   "Review with the pharmacist.", f"{ix.source} (via the pharmacopoeia)", None, static=True))
        except Exception:
            pass
    return out


def detect_falls(c: Ctx) -> List[Finding]:
    """Morse Fall Scale, scored from the record and the orders (Morse 1989)."""
    if c.kind != "synthetic":
        return []
    items: List[Tuple[str, int]] = []
    if c.has("fall_history", "dementia") or any("fall" in (e.get("reason") or "").lower() for e in c.record.get("encounters") or []):
        items.append(("History of falling", 25))
    if len([p for p in c.record.get("problems") or [] if p.get("status") == "chronic"]) >= 1:
        items.append(("Secondary diagnosis", 15))
    mob = c.record.get("mobility")
    if mob == "walker":
        items.append(("Ambulatory aid", 15))
    if any(o.get("route") == "IV" for o in c.active_orders()):
        items.append(("IV access", 20))
    age = (c.record.get("patient") or {}).get("age") or 0
    if mob in ("walker", "assist of one") or c.has("post_op") or age >= 80:
        items.append(("Impaired gait", 10 if age < 80 else 20))
    if c.has("dementia"):
        items.append(("Forgets limitations", 15))
    score = sum(x[1] for x in items)
    if score < 45:
        return []
    sed = [o["name"] for o in c.active_orders() if "sedating" in o.get("classes", [])]
    return [Finding("falls", "Falls risk", f"High falls risk: Morse {score}", "info", "safety",
                    [_ev(l, "", "", p) for l, p in items], ([f"Sedating orders: {', '.join(sed)}."] if sed else []),
                    "Falls bundle: bed low, call bell in reach, non-slip footwear, assisted toileting.",
                    "Morse Fall Scale (Morse 1989), ≥ 45 high risk", float(score), static=True)]


def detect_signal(c: Ctx) -> List[Finding]:
    """The room's own feed: a monitor that stops sending is itself a finding."""
    if c.v_at is None or c.kind == "replay":
        return []
    gap = c.now - c.v_at
    if gap < 90:
        return []
    return [Finding("signal", "Monitoring", f"No vitals for {int(gap // 60)} min {int(gap % 60)} s", "watch", "safety",
                    [_ev("Last sample", "", "", at=c.v_at)], [],
                    "Check the device and the connection; take a manual set of observations.", "Room device health", gap)]


DETECTORS: List[Tuple[str, Callable[..., List[Finding]]]] = [
    ("NEWS2", detect_news2), ("qSOFA", detect_qsofa), ("SIRS", detect_sirs), ("Shock index", detect_shock_index),
    ("Personal baseline", detect_baseline), ("Trend", detect_trend), ("Opioid safety", detect_opioid),
    ("Neutropenic fever", detect_neutropenic_fever), ("Glucose", detect_glucose), ("Rhythm", detect_af),
    ("Bleeding", detect_bleeding), ("Acute kidney injury", detect_kidney), ("Potassium", detect_potassium),
    ("Lactate", detect_lactate), ("SOFA", detect_sofa), ("Sepsis detector", detect_sepsis_model),
    ("Medication safety", detect_medications), ("Falls risk", detect_falls), ("Monitoring", detect_signal),
]


def run_all(c: Ctx, screen: Optional[Callable[..., List[Any]]] = None) -> Tuple[List[Finding], List[str]]:
    """Every detector over the room. A detector that raises is reported, never allowed to stop the rest."""
    out: List[Finding] = []
    faults: List[str] = []
    for name, fn in DETECTORS:
        try:
            out.extend(fn(c, screen) if fn is detect_medications else fn(c))
        except Exception as exc:  # a broken detector must not blind the room
            faults.append(f"{name}: {type(exc).__name__}: {exc}")
    out.sort(key=lambda f: (-SEVERITY_RANK[f.severity], f.detector))
    return out, faults
