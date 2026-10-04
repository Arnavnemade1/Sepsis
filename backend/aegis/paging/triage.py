"""Priority Handler: turn an alert into a priority (P1–P4), a category, a room.

Two sources feed it. A person types or says something ("BP 84/50 in 512, febrile,
confused"), or the sepsis detector crosses its alarm line on a monitored bed.

Rules decide first and set a floor. A red flag (a code, a STEMI, SBP < 90 with a
fever, SpO₂ < 88) fixes the priority, and the language model may then only RAISE
it: it can notice what a regex cannot, but it can never argue a red flag down.
A P1 red flag skips the model entirely, so the most urgent pages never wait on a
network call. Every LLM call is bounded by a timeout and falls back to the rules.
"""
from __future__ import annotations

import concurrent.futures as _cf
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.aegis.paging.hospital import ROOMS

CATEGORIES = ("code", "sepsis", "cardiac", "neuro", "respiratory", "surgical", "medication", "trauma", "general")
LLM_TIMEOUT_S = 6.0          # when a caller waits inline
REFINE_TIMEOUT_S = 45.0      # the background refinement after a page is already out


@dataclass
class Triage:
    priority: int
    category: str
    room: Optional[str]
    symptoms: List[str]
    vitals: Dict[str, float]
    red_flags: List[str]
    rationale: str
    source: str                          # rules | llm+rules | detector
    model: Optional[str] = None
    sepsis_screen: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
_ROOM_RX = [
    re.compile(r"\b(?:room|rm\.?|bed|bay)\s*#?\s*([A-Za-z]{0,2}\d{1,3})\b", re.I),
    re.compile(r"\b(OR\s?\d|T\d|A\d{1,2})\b", re.I),
    re.compile(r"\b(\d{3})\b"),
]


_VITAL_BEFORE = re.compile(r"(?:hr|bp|rr|map|sats?|spo2|o2|temp|glucose|bg|lactate|pulse|gcs|nihss|weight|age|x)\s*[:=]?\s*$", re.I)


def extract_room(text: str) -> Optional[str]:
    for i, rx in enumerate(_ROOM_RX):
        for m in rx.finditer(text):
            tok = m.group(1).upper().replace(" ", "")
            if tok not in ROOMS:
                continue
            if i == len(_ROOM_RX) - 1:
                # a bare number is a room only if it is not a reading ("HR 104", "BP 104/60", "104%")
                before, after = text[max(0, m.start() - 12):m.start()], text[m.end():m.end() + 2]
                if _VITAL_BEFORE.search(before) or after[:1] in ("/", "%", ".") or before.endswith("/"):
                    continue
            return tok
    return None


def _f(x: str) -> float:
    return float(x)


def extract_vitals(text: str) -> Dict[str, float]:
    t = text.lower()
    v: Dict[str, float] = {}
    m = re.search(r"\b(?:bp|b/p|blood pressure)?\s*(\d{2,3})\s*/\s*(\d{2,3})\b", t)
    if m and 40 <= _f(m.group(1)) <= 260:
        v["sbp"], v["dbp"] = _f(m.group(1)), _f(m.group(2))
        v.setdefault("map", round((v["sbp"] + 2 * v["dbp"]) / 3, 1))
    # (key, pattern, plausible range): the first reading in range wins, so "O2 at 15 L, sats 91%"
    # reads the saturation, not the flow
    for key, rx, lo, hi in (
        ("hr", r"\b(?:hr|heart rate|pulse)\s*(?:of|is)?\s*[:=]?\s*(\d{2,3})\b", 20, 250),
        ("spo2", r"\b(?:spo2|sp02|o2 sat(?:uration)?s?|sats?|satt?ing|saturation|saturating|desat(?:urat(?:ing|ed|ion)|ting|s)?|pulse ox|o2)\s*(?:(?:is|was|now|of|at|to|down to|dropp(?:ing|ed) to|falling to|fell to)\s+)*[:=]?\s*(\d{2,3})\s*%?", 50, 100),
        ("rr", r"\b(?:rr|resp(?:iratory)? rate|resps?)\s*[:=]?\s*(\d{1,2})\b", 4, 70),
        ("map", r"\bmap\s*[:=]?\s*(\d{2,3})\b", 20, 200),
        ("lactate", r"\b(?:lactate|lac)\s*(?:of|is)?\s*[:=]?\s*(\d{1,2}(?:\.\d)?)\b", 0.1, 30),
        ("gcs", r"\bgcs\s*[:=]?\s*(\d{1,2})\b", 3, 15),
        ("glucose", r"\b(?:glucose|bg|sugar)\s*[:=]?\s*(\d{2,3})\b", 10, 1000),
        ("nihss", r"\bnihss\s*[:=]?\s*(\d{1,2})\b", 0, 42),
    ):
        for m in re.finditer(rx, t):
            if lo <= _f(m.group(1)) <= hi:
                v[key] = _f(m.group(1))
                break
    if "spo2" not in v:  # an unlabelled reading with its oxygen context: "84% on room air", "88% on 4L"
        m = re.search(r"\b(\d{2,3})\s*%\s*(?:on\s+(?:ra|room air|\d+\s*l|nrb|nc|high[- ]flow|bipap|cpap)\b|ra\b)", t)
        if m and 50 <= _f(m.group(1)) <= 100:
            v["spo2"] = _f(m.group(1))
    m = re.search(r"\b(?:temp|t|temperature|febrile(?: to)?|fever(?: of)?|tmax)\s*[:=]?\s*(\d{2,3}(?:\.\d)?)\s*(°?\s*[cf])?\b", t)
    if m:
        val = _f(m.group(1))
        unit = (m.group(2) or "").replace("°", "").strip()
        if unit == "f" or val > 45:
            val = round((val - 32) * 5 / 9, 1)
        if 30 <= val <= 44:
            v["temp_c"] = val
    return v


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
# (pattern, category, priority, label) — the lowest priority number that matches wins.
_KEYWORDS: List[Tuple[str, str, int, str]] = [
    (r"\bcode blue\b|\bcardiac arrest\b|\bpulseless\b|\bnot breathing\b|\bapn(?:e|o)ic\b|\bunresponsive\b", "code", 1, "unresponsive / arrest"),
    (r"\bstemi\b|\bst[- ]elevation\b", "cardiac", 1, "STEMI pattern"),
    (r"\bseptic shock\b", "sepsis", 1, "septic shock stated"),
    (r"\banaphyla|\bstridor\b|\bairway (?:compromise|obstruct)|\bangioedema\b", "respiratory", 1, "airway threat"),
    (r"\bstroke\b|\bfacial droop\b|\baphasi|\bhemipare|\bslurred speech\b|\btpa\b|\bnihss\b|\blast known well\b", "neuro", 2, "stroke signs"),
    (r"\bsepsis\b|\bseptic\b|\brigors?\b|\bsource of infection\b", "sepsis", 2, "sepsis concern"),
    (r"\bchest pain\b|\bangina\b|\btroponin\b|\bpalpitations?\b|\bheart block\b|\bbradycard", "cardiac", 2, "cardiac symptoms"),
    (r"\bshort(?:ness)? of breath\b|\bsob\b|\bdesat|\bhypoxi|\brespiratory distress\b|\bdyspn", "respiratory", 2, "respiratory distress"),
    (r"\bseizure|\bpost[- ]?ictal\b|\bstatus epilepticus\b", "neuro", 2, "seizure"),
    (r"\bbleed|\bhemorrhag|\bmelena\b|\bhematemesis\b|\bhematochezia\b", "surgical", 2, "bleeding"),
    (r"\bconfus|\baltered\b|\bdelirium\b|\blethargic\b|\bobtunded\b", "neuro", 3, "altered mental status"),
    (r"\bfever|\bfebrile\b|\bpyrexi", "sepsis", 3, "fever"),
    (r"\bfall\b|\bfell\b", "trauma", 3, "fall"),
    (r"\blaceration\b|\bwound\b|\bdehiscence\b|\babscess\b", "trauma", 3, "wound"),
    (r"\breaction\b|\brash\b|\bhives\b|\bmedication error\b|\bwrong dose\b|\binfusion\b", "medication", 3, "medication issue"),
    (r"\babdominal pain\b|\bdistension\b|\bvomit|\bileus\b|\bacute abdomen\b", "surgical", 3, "abdominal"),
    (r"\bpain\b", "general", 3, "pain"),
    (r"\brefill\b|\bfamily\b|\bdischarge\b|\bquestion\b|\bdiet\b|\bsleep\b|\bconstipat|\bpaperwork\b|\bconsent\b", "general", 4, "routine request"),
]
_INFECTION = re.compile(r"\bfever|\bfebrile\b|\binfect|\bpneumonia\b|\buti\b|\bcellulitis\b|\bsepsis\b|\bseptic\b|"
                        r"\brigors?\b|\bwbc\b|\bpurulent\b|\bcultures?\b|\bneutropeni|\babscess\b", re.I)
_ALTERED = re.compile(r"\bconfus|\baltered\b|\bdelirium\b|\blethargic\b|\bobtunded\b|\bunresponsive\b|\bdisoriented\b", re.I)


def sepsis_screen(text: str, v: Dict[str, float]) -> Dict[str, Any]:
    """qSOFA from what the text states, and the Sepsis-3 shock criteria."""
    rr, sbp = v.get("rr"), v.get("sbp")
    altered = bool(_ALTERED.search(text)) or (v.get("gcs", 15) < 15)
    q = int(rr is not None and rr >= 22) + int(sbp is not None and sbp <= 100) + int(altered)
    infection = bool(_INFECTION.search(text)) or (v.get("temp_c") is not None and (v["temp_c"] >= 38.3 or v["temp_c"] < 36))
    shock = (v.get("map") is not None and v["map"] < 65) or (v.get("lactate") is not None and v["lactate"] >= 4) \
        or (sbp is not None and sbp < 90)
    return {"qsofa": q, "infection_signs": infection, "shock_signs": bool(shock), "altered": altered}


def rules(text: str) -> Tuple[int, str, List[str], List[str], Dict[str, float], Dict[str, Any], bool]:
    """(priority, category, symptoms, red_flags, vitals, screen, matched_anything)."""
    t = text.lower()
    v = extract_vitals(text)
    hits = [(p, cat, label) for rx, cat, p, label in _KEYWORDS if re.search(rx, t)]
    symptoms = [label for _, _, label in hits]
    red: List[Tuple[int, str, str]] = []           # (priority, category, flag)

    sbp, hr, spo2, rr = v.get("sbp"), v.get("hr"), v.get("spo2"), v.get("rr")
    scr = sepsis_screen(text, v)
    if spo2 is not None and spo2 < 88:
        red.append((1, "respiratory", f"SpO₂ {spo2:.0f}% < 88%"))
    elif spo2 is not None and spo2 < 92:
        red.append((2, "respiratory", f"SpO₂ {spo2:.0f}% < 92%"))
    if sbp is not None and sbp < 80:
        red.append((1, "code" if not scr["infection_signs"] else "sepsis", f"SBP {sbp:.0f} < 80"))
    elif sbp is not None and sbp < 90:
        red.append((2, "sepsis" if scr["infection_signs"] else "general", f"SBP {sbp:.0f} < 90"))
    if hr is not None and (hr >= 140 or hr < 40):
        pr = 1 if (sbp is not None and sbp < 90) else 2
        red.append((pr, "cardiac", f"HR {hr:.0f}"))
    if rr is not None and rr >= 30:
        red.append((2, "respiratory", f"RR {rr:.0f} ≥ 30"))
    if v.get("gcs") is not None and v["gcs"] <= 8:
        red.append((1, "neuro", f"GCS {v['gcs']:.0f} ≤ 8"))
    stroke = any(cat == "neuro" and label == "stroke signs" for _, cat, label in hits)
    lkw = re.search(r"last known well\s*(?:at|was)?\s*(\d+(?:\.\d)?)\s*(min|minutes|mins|h|hr|hrs|hours?)\b", t)
    if stroke:
        hours = None
        if lkw:
            hours = float(lkw.group(1)) / (60.0 if lkw.group(2).startswith("m") else 1.0)
        in_window = (hours is not None and hours <= 4.5) or bool(re.search(r"\btpa window\b|\bwindow open\b", t))
        if in_window or v.get("nihss", 0) >= 6:
            red.append((1, "neuro", "stroke within the thrombolysis window" if in_window else f"NIHSS {v['nihss']:.0f}"))
    if scr["infection_signs"] and scr["shock_signs"]:
        red.append((1, "sepsis", "infection with shock signs (Sepsis-3)"))
    elif scr["infection_signs"] and scr["qsofa"] >= 2:
        red.append((2, "sepsis", f"qSOFA {scr['qsofa']} with infection signs"))
    if v.get("lactate") is not None and v["lactate"] >= 2 and scr["infection_signs"]:
        red.append((1 if v["lactate"] >= 4 else 2, "sepsis", f"lactate {v['lactate']:g}"))
    for p, cat, label in hits:
        if p == 1:
            red.append((1, cat, label))

    if v.get("temp_c") is not None and v["temp_c"] >= 38.3 and not any(cat == "sepsis" for _, cat, _ in hits):
        hits.append((3, "sepsis", f"fever {v['temp_c']:.1f} °C"))
        symptoms.append(f"fever {v['temp_c']:.1f} °C")
    candidates = [(p, cat) for p, cat, _ in hits] + [(p, cat) for p, cat, _ in red]
    if candidates:
        pr, cat = min(candidates, key=lambda x: (x[0], CATEGORIES.index(x[1])))
    else:
        pr, cat = 3, "general"
    return pr, cat, symptoms, [f for _, _, f in sorted(red)], v, scr, bool(hits or red)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You triage hospital pages. Reply with ONLY a JSON object: "
    '{"priority": 1-4, "category": one of ' + json.dumps(list(CATEGORIES)) + ', '
    '"room": string or null, "symptoms": [short strings], "rationale": one sentence}. '
    "P1 = immediate threat to life (arrest, shock, STEMI, stroke in window, airway). "
    "P2 = urgent, needs a clinician within minutes (sepsis concern, chest pain, hypoxia, new deficit). "
    "P3 = needs review this hour. P4 = routine. Never invent vitals or history not in the text."
)


def model_available(client: Any) -> bool:
    """LLMProvider exposes ``available`` as a property; test doubles as a method."""
    a = getattr(client, "available", False)
    try:
        return bool(a() if callable(a) else a)
    except Exception:
        return False


def _ask_model(client: Any, text: str, timeout: float = LLM_TIMEOUT_S) -> Optional[Dict[str, Any]]:
    if client is None or not model_available(client):
        return None
    def call() -> Optional[str]:
        return client.chat(_SYSTEM, text, max_tokens=220, temperature=0.0, respect_interval=False)
    pool = _cf.ThreadPoolExecutor(max_workers=1)
    try:
        raw = pool.submit(call).result(timeout=timeout)
    except Exception:
        return None
    finally:
        pool.shutdown(wait=False)
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except Exception:
        return None
    try:
        out["priority"] = int(out.get("priority"))
    except Exception:
        return None
    if out["priority"] not in (1, 2, 3, 4):
        return None
    return out


def triage_text(text: str, room_hint: Optional[str] = None, client: Any = None,
                timeout: float = LLM_TIMEOUT_S) -> Triage:
    t0 = time.perf_counter()
    text = (text or "").strip()
    pr, cat, symptoms, red, v, scr, matched = rules(text)
    room = room_hint if room_hint in ROOMS else extract_room(text)
    source, model, rationale = "rules", None, ""
    if red:
        rationale = "Red flags: " + "; ".join(red)
    elif matched:
        rationale = "Matched: " + ", ".join(symptoms)
    else:
        rationale = "No red flags or known patterns; defaulted to P3 pending review."

    # A P1 red flag goes out on the rule alone. Otherwise the model may refine.
    if not (red and pr == 1):
        llm = _ask_model(client, text, timeout)
        if llm is not None:
            source, model = "llm+rules", getattr(client, "_last_model", None) or "llm"
            llm_pr = llm["priority"]
            # the model may raise priority; it may lower it only when no red flag set the floor
            new_pr = min(llm_pr, pr) if red else llm_pr
            if llm.get("category") in CATEGORIES and (not red or new_pr < pr):
                cat = llm["category"]
            if new_pr != pr:
                rationale += f" Model {'raised' if new_pr < pr else 'lowered'} P{pr}→P{new_pr}: {llm.get('rationale', '')}".rstrip()
            else:
                rationale += f" Model agrees: {llm.get('rationale', '')}".rstrip()
            pr = new_pr
            if not room and isinstance(llm.get("room"), str) and llm["room"].upper() in ROOMS:
                room = llm["room"].upper()
            for s in llm.get("symptoms") or []:
                if isinstance(s, str) and s and s not in symptoms and len(symptoms) < 6:
                    symptoms.append(s[:60])
    return Triage(pr, cat, room, symptoms, v, red, rationale.strip(), source, model, scr,
                  round((time.perf_counter() - t0) * 1000, 1))


# --------------------------------------------------------------------------- #
# The detector as an alert source
# --------------------------------------------------------------------------- #
# The ward index (backend/aegis/ward_risk.py) is the shipped detector's calibrated
# probability mapped so its validated operating point sits on these lines: the paging
# line IS the detector's HIGH point, so a page means "the detector alarmed".
DETECTOR_ALARM = 0.70          # the detector's HIGH operating point, on the ward index
DETECTOR_CLEAR = 0.60          # hysteresis: an episode closes only below this


def triage_detector(room: str, name: str, risk: float, enterprise: Dict[str, float],
                    drivers: List[str], population: Optional[float] = None,
                    probability: Optional[float] = None) -> Triage:
    """A monitored bed crossed the alarm line. P1 when the Sepsis-3 shock criteria
    are met (MAP < 65 or lactate ≥ 4), else P2. The detector never raises P3/P4:
    below the alarm line the bed shows as 'watch' on the floor, it does not page."""
    mp, lac = enterprise.get("map"), enterprise.get("lactate")
    shock = (mp is not None and mp < 65) or (lac is not None and lac >= 4)
    red = [f"risk index {risk:.2f} ≥ {DETECTOR_ALARM:.2f}"
           + (f" (calibrated P {probability:.3f})" if probability is not None else "")]
    if mp is not None and mp < 65:
        red.append(f"MAP {mp:.0f} < 65")
    if lac is not None and lac >= 4:
        red.append(f"lactate {lac:.1f} ≥ 4")
    v = {k: float(enterprise[k]) for k in ("map", "spo2", "resp_rate", "temp_c", "lactate", "wbc") if k in enterprise}
    return Triage(
        priority=1 if shock else 2, category="sepsis", room=room,
        symptoms=list(drivers[:4]) or ["risk above alarm line"],
        vitals=v, red_flags=red,
        rationale=("Septic-shock criteria met (Sepsis-3): " if shock else "Sepsis risk crossed the alarm line: ")
        + "; ".join(red) + (f" (before the chart's prior shift {population:.2f})"
                            if population is not None and abs(population - risk) > 1e-3 else ""),
        source="detector", model=None,
        sepsis_screen={"shock_signs": shock, "detector_risk": round(risk, 4),
                       "probability": round(probability, 5) if probability is not None else None},
    )
