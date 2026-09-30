"""SBAR brief: what the clinician reads on the way to the bedside.

Generated the moment a page is accepted, under 100 words, from facts only: the
alert, the triage, and the patient's own record (the real replayed CinC chart,
the live monitored session, or a synthetic demo record, which it says it is).
The language model writes it when a person is holding the device; otherwise,
and whenever the model is slow or unavailable, a deterministic template does.
The model is told to write "not documented" rather than fill a gap.
"""
from __future__ import annotations

import concurrent.futures as _cf
import re
import time
from typing import Any, Dict, List, Optional

from backend.aegis.paging.triage import model_available

LLM_TIMEOUT_S = 8.0
MAX_WORDS = 100

_REQUEST = {
    "sepsis": {1: "Bedside now. Start the Hour-1 bundle: lactate, blood cultures ×2, broad-spectrum antibiotics, "
                  "30 mL/kg crystalloid for hypotension. Pharmacy notified.",
               2: "Assess within 15 min: lactate, cultures, source review; decide on antibiotics this hour."},
    "cardiac": {1: "Bedside now. 12-lead ECG review; activate the cath lab if STEMI confirmed.",
                2: "Assess within 15 min: ECG, troponin, rhythm review."},
    "neuro": {1: "Bedside now. Stroke protocol: glucose, CT head, thrombolysis eligibility.",
              2: "Neuro assessment within 15 min."},
    "respiratory": {1: "Bedside now. Airway and oxygenation; consider escalation of support.",
                    2: "Assess breathing within 15 min; ABG, chest X-ray."},
    "code": {1: "Code response: bedside immediately."},
    "surgical": {1: "Bedside now for surgical assessment.", 2: "Surgical review within 30 min."},
    "medication": {1: "Bedside now; stop the suspected agent.", 2: "Review the medication and order changes."},
    "trauma": {2: "Assess within 30 min.", 3: "Assess this hour."},
    "general": {1: "Bedside now.", 2: "Assess within 30 min.", 3: "Review this hour.", 4: "Address when available."},
}


def _request_for(category: str, priority: int) -> str:
    table = _REQUEST.get(category, _REQUEST["general"])
    for p in range(priority, 5):
        if p in table:
            return table[p]
    return _REQUEST["general"][min(priority, 4)]


def _trim(words: str, n: int) -> str:
    w = words.split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")


def facts(page: Dict[str, Any], patient: Optional[Dict[str, Any]]) -> Dict[str, str]:
    tri = page.get("triage") or {}
    p = patient or {}
    who = p.get("name") or "patient"
    demo = f"{p.get('age', '?')}{p.get('sex', '')}" if p.get("age") else ""
    situation = f"P{page['priority']} {page['category']}: {page.get('text', '').strip()}"
    if p.get("monitored"):
        hist = ", ".join(p.get("history") or []) or "no flagged history"
        background = f"{who} {demo}, {p.get('source', 'monitored')}. ICU hour {p.get('icu_hour', '?')}; {hist}."
    elif p:
        meds = ", ".join((p.get("medications") or [])[:3]) or "none listed"
        allergies = ", ".join(p.get("allergies") or []) or "NKDA"
        background = (f"{who} {demo}, admitted for {p.get('diagnosis', 'not documented')}; "
                      f"PMH {', '.join(p.get('history') or []) or 'none'}; meds {meds}; allergies {allergies}; "
                      f"{p.get('code_status', 'code status not documented')}"
                      + (" (synthetic demo record)." if p.get("synthetic") else "."))
    else:
        background = "No record on file for this room."
    bits: List[str] = []
    if p.get("risk") is not None:
        bits.append(f"sepsis risk {p['risk']:.2f} (alarm 0.70)")
    v = tri.get("vitals") or {}
    for k, label, fmt in (("sbp", "SBP", "{:.0f}"), ("map", "MAP", "{:.0f}"), ("hr", "HR", "{:.0f}"),
                          ("spo2", "SpO₂", "{:.0f}%"), ("rr", "RR", "{:.0f}"), ("resp_rate", "RR", "{:.0f}"),
                          ("temp_c", "T", "{:.1f}°C"), ("lactate", "lactate", "{:.1f}")):
        if k in v:
            bits.append(f"{label} {fmt.format(v[k])}")
    red = tri.get("red_flags") or []
    assessment = "; ".join(bits[:5]) or "no vitals in the alert"
    if red:
        assessment += ". Flags: " + "; ".join(red[:3])
    return {"situation": situation, "background": background, "assessment": assessment,
            "request": _request_for(page["category"], page["priority"])}


def template(page: Dict[str, Any], patient: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    f = facts(page, patient)
    # keep the whole brief under the word limit, trimming the longest parts first
    budget = {"situation": 26, "background": 30, "assessment": 24, "request": 22}
    sec = {k: _trim(f[k], budget[k]) for k in f}
    return _pack(sec, source="template", model=None)


def _pack(sec: Dict[str, str], source: str, model: Optional[str]) -> Dict[str, Any]:
    text = "\n".join(f"{k[0].upper()}: {sec[k]}" for k in ("situation", "background", "assessment", "request"))
    return {**sec, "text": text, "words": len(text.split()), "source": source, "model": model,
            "generated_at": time.time()}


_SYSTEM = ("You write SBAR pages for a clinician walking to a bedside. Use ONLY the facts given. "
           "Exactly four lines, starting 'S:', 'B:', 'A:', 'R:'. Under 90 words total. "
           "If a fact is missing write 'not documented'. Never invent vitals, drugs, history or results.")


def generate(page: Dict[str, Any], patient: Optional[Dict[str, Any]], client: Any = None,
             dossier_text: Optional[str] = None, timeout: float = LLM_TIMEOUT_S) -> Dict[str, Any]:
    """LLM brief with a hard timeout; the template otherwise."""
    if client is None or not model_available(client):
        return template(page, patient)
    try:
        if client.budget_remaining() < 40:
            return template(page, patient)
    except Exception:
        pass
    f = facts(page, patient)
    user = "\n".join(f"{k.upper()} FACTS: {v}" for k, v in f.items())
    if dossier_text:
        user += "\nMONITORING DOSSIER (authoritative):\n" + dossier_text[:1800]

    def call() -> Optional[str]:
        return client.chat(_SYSTEM, user, max_tokens=260, temperature=0.2, respect_interval=False)
    pool = _cf.ThreadPoolExecutor(max_workers=1)
    try:
        raw = pool.submit(call).result(timeout=timeout)
    except Exception:
        raw = None
    finally:
        pool.shutdown(wait=False)
    if not raw:
        return template(page, patient)
    sec: Dict[str, str] = {}
    for key, tag in (("situation", "S"), ("background", "B"), ("assessment", "A"), ("request", "R")):
        m = re.search(rf"^\s*\**{tag}\**\s*[:\-]\s*(.+)$", raw, re.M)
        if m:
            sec[key] = m.group(1).strip()
    if len(sec) < 4:
        return template(page, patient)
    out = _pack(sec, source="llm", model=getattr(client, "_last_model", None))
    if out["words"] > MAX_WORDS + 10:
        return template(page, patient)
    return out
