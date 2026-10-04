"""Proposers: who suggests what a hospital's local code means.

Onboarding a feed means answering, for every code it sends, "which AegisOS concept is
this, and in what unit?". Integration analysts do it by hand from a spreadsheet of
local codes, and it takes weeks. Four proposers answer it here, from cheapest to most
expensive:

* **loinc**: the sender already codes in LOINC (OBX-3 system ``LN``). Exact.
* **mdc**: a monitor gateway sends IEEE 11073 reference ids (``MDC_PULS_OXIM_SAT_O2``).
* **lexical**: the local mnemonic and display text against each concept's aliases,
  with the written unit as a hint (a ``umol/L`` code is not haemoglobin).
* **model**: a language model reads the codes the others could not place, with the
  vocabulary in front of it, and answers in JSON.

None of them can adopt a mapping. Each returns ranked candidates; the evidence engine
decides whether the DATA agrees (``engine.Integration._decide``). A model that names
the wrong analyte is refused by the same checks as a typo in a spreadsheet, and its
proposal, the evidence, and the refusal are all recorded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.aegis.interop.units import dimension, norm
from backend.aegis.interop.vocab import BY_KEY, BY_LOINC, CONCEPTS, Concept


@dataclass
class Proposal:
    concept: str
    score: float                 # proposer's own confidence, 0–1
    proposer: str                # loinc | mdc | lexical | model:<id>
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"concept": self.concept, "score": round(self.score, 3), "proposer": self.proposer,
                "rationale": self.rationale}


def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("₂", "2").replace("+", " plus ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# local mnemonics labs commonly use, beyond the aliases on each concept
_MNEMONICS = {
    "k": "potassium", "kplus": "potassium", "na": "sodium", "naplus": "sodium", "cl": "chloride", "crea": "creatinine",
    "creat": "creatinine", "cr": "creatinine", "scr": "creatinine", "bun": "bun", "urea": "bun", "glu": "glucose",
    "gluc": "glucose", "lac": "lactate", "lact": "lactate", "hgb": "hemoglobin", "hb": "hemoglobin",
    "hct": "hematocrit", "plt": "platelets", "wbc": "wbc", "anc": "anc", "agc": "anc", "tbil": "bilirubin",
    "tbili": "bilirubin", "dbil": "bilirubin_direct", "dbili": "bilirubin_direct", "ast": "ast", "sgot": "ast",
    "alp": "alp", "alkp": "alp", "tni": "troponin", "ctni": "troponin", "trop": "troponin", "ptt": "ptt",
    "aptt": "ptt", "inr": "inr", "fib": "fibrinogen", "mg": "magnesium", "mag": "magnesium", "phos": "phosphate",
    "po4": "phosphate", "ca": "calcium", "ica": "ionized_calcium", "co2": "hco3", "tco2": "hco3", "hco3": "hco3",
    "ph": "ph", "pco2": "paco2", "paco2": "paco2", "sao2": "sao2", "be": "base_excess", "fio2": "fio2",
    "spo2": "spo2", "hr": "hr", "rr": "rr", "temp": "temp", "sbp": "sbp", "dbp": "dbp", "map": "map",
    "etco2": "etco2", "o2flow": "o2_flow",
}


def _tokens(s: str) -> List[str]:
    return [t for t in _norm_text(s).split() if t]


def lexical(code: str, text: str, unit: str = "", k: int = 3) -> List[Proposal]:
    """Rank concepts by how well the local code and display text match their aliases,
    nudged by whether the written unit can convert to the concept at all."""
    ntext = _norm_text(text)
    ncode = _norm_text(code).replace(" ", "")
    toks = set(_tokens(text)) | ({ncode} if ncode else set())
    unit_tok = norm(unit)
    scores: Dict[str, Tuple[float, str]] = {}

    def bump(key: str, s: float, why: str) -> None:
        if key in BY_KEY and s > scores.get(key, (0.0, ""))[0]:
            scores[key] = (s, why)

    if ncode in _MNEMONICS:
        bump(_MNEMONICS[ncode], 0.80, f"local code {code!r} is a common mnemonic")
    for c in CONCEPTS:
        for a in c.aliases:
            na = _norm_text(a)
            if not na:
                continue
            if na == ntext:
                bump(c.key, 0.95, f"display text matches {a!r}")
            elif na.replace(" ", "") == ncode and len(ncode) >= 2:
                bump(c.key, 0.85, f"code matches {a!r}")
            else:
                at = set(na.split())
                if at and at <= toks and len(" ".join(at)) >= 3:
                    bump(c.key, 0.55 + 0.25 * min(1.0, len(at) / max(1, len(toks))), f"contains {a!r}")
    out = []
    for key, (s, why) in scores.items():
        c = BY_KEY[key]
        if unit_tok and not unit_tok.startswith("?"):
            if c.conversion(unit_tok) is not None:
                s = min(1.0, s + 0.1)
                why += f"; unit {unit!r} converts to {c.unit}"
            elif dimension(unit_tok) is not None:
                s -= 0.4
                why += f"; unit {unit!r} cannot be {c.label.lower()}"
        if s > 0.2:
            out.append(Proposal(key, s, "lexical", why))
    out.sort(key=lambda p: -p.score)
    return out[:k]


def coded(code: str, text: str, system: str) -> Optional[Proposal]:
    """LOINC and IEEE 11073 codes name their concept outright."""
    sysu = (system or "").upper()
    if sysu in ("LN", "LOINC", "HTTP://LOINC.ORG"):
        c = BY_LOINC.get(code)
        if c:
            return Proposal(c.key, 1.0, "loinc", f"LOINC {code} is {c.loinc_display}")
    blob = f"{code} {text}".upper()
    for c in CONCEPTS:
        for ref in c.mdc:
            if re.search(rf"\b{ref}\b", blob):
                return Proposal(c.key, 0.98, "mdc", f"IEEE 11073 {ref}")
    return None


# --------------------------------------------------------------------------- #
# The language-model proposer
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You map a hospital's local laboratory and monitor codes onto a fixed clinical vocabulary. "
    "For each local code you are given its code, display text, the unit the feed writes, and a "
    "summary of the values seen. Choose the ONE vocabulary key it measures, or null if none fits. "
    "Read the values: a 'creatinine' whose median is 130 is reported in umol/L. Never invent a key "
    "that is not in the vocabulary. You only propose; every proposal is checked against the data "
    "before anything uses it, so say null rather than guess. Answer with JSON only: "
    '{"mappings": [{"code": "...", "concept": "<key or null>", "unit": "<the unit you believe the '
    'values are in>", "confidence": 0.0-1.0, "why": "<12 words max>"}]}'
)


def _vocab_table() -> str:
    return "\n".join(f"{c.key} | {c.label} | LOINC {c.loinc} | canonical unit {c.unit}" for c in CONCEPTS)


def model_prompt(items: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for it in items:
        vs = sorted(v for v in it.get("values", []) if v is not None)
        summ = ""
        if vs:
            q = lambda p: vs[min(len(vs) - 1, int(p * (len(vs) - 1)))]  # noqa: E731
            summ = f"n={len(vs)} min={vs[0]:g} p25={q(.25):g} median={q(.5):g} p75={q(.75):g} max={vs[-1]:g}"
        lines.append(f"- code={it['code']!r} text={it.get('text', '')!r} unit={it.get('unit', '')!r} values: {summ}")
    return ("VOCABULARY (key | label | LOINC | canonical unit):\n" + _vocab_table()
            + "\n\nLOCAL CODES TO MAP:\n" + "\n".join(lines))


def _answer_rows(text: Optional[str]) -> List[Dict[str, Any]]:
    if not text:
        return []
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    rows = (data.get("mappings") or []) if isinstance(data, dict) else []
    return [r for r in rows if isinstance(r, dict)]


def model_outcomes(text: Optional[str], codes: Sequence[str]) -> Dict[str, str]:
    """What the model said about each code it did NOT propose a vocabulary key for:
    ``declined`` (it answered null) or ``outside:<key>`` (it named something the vocabulary
    lacks). A code missing here and from the proposals got no answer at all: the call
    failed, or the model skipped it. Only an answer may mark a code as outside the
    vocabulary; no answer sends it to a person."""
    out: Dict[str, str] = {}
    asked = set(codes)
    for row in _answer_rows(text):
        code, key = str(row.get("code", "")), row.get("concept")
        if code not in asked:
            continue
        if key is None or (isinstance(key, str) and key.strip().lower() in ("", "null", "none")):
            out[code] = "declined"
        elif not isinstance(key, str) or key not in BY_KEY:
            out[code] = f"outside:{str(key)[:40]}"
    return out


def parse_model_answer(text: Optional[str], codes: Sequence[str]) -> Dict[str, Proposal]:
    """Proposals from the model's JSON. Anything not in the vocabulary, or about a code we
    did not ask about, is dropped: the model cannot widen what it is choosing from."""
    out: Dict[str, Proposal] = {}
    asked = set(codes)
    for row in _answer_rows(text):
        code, key = str(row.get("code", "")), row.get("concept")
        if code not in asked or not isinstance(key, str) or key not in BY_KEY:
            continue
        try:
            conf = max(0.0, min(1.0, float(row.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        why = str(row.get("why", ""))[:120]
        unit = str(row.get("unit", ""))[:24]
        out[code] = Proposal(key, conf, "model", (why + (f" (values read as {unit})" if unit else "")).strip())
    return out


def ask_model(llm: Any, items: Sequence[Dict[str, Any]], max_batch: int = 12,
              outcomes: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Proposal], Optional[str]]:
    """One call per batch of unmapped codes. Returns (proposals by code, model id); fills
    ``outcomes`` (if given) with what the model said about the codes it did not map."""
    if llm is None or not getattr(llm, "available", False) or not items:
        return {}, None
    out: Dict[str, Proposal] = {}
    model_id = None
    for i in range(0, len(items), max_batch):
        batch = list(items[i:i + max_batch])
        try:
            text = llm.chat(SYSTEM_PROMPT, model_prompt(batch), max_tokens=900, temperature=0.0,
                            respect_interval=False)
        except Exception:
            text = None
        got = parse_model_answer(text, [b["code"] for b in batch])
        if outcomes is not None:
            outcomes.update(model_outcomes(text, [b["code"] for b in batch]))
        model_id = getattr(llm, "_last_model", None) or model_id
        for code, p in got.items():
            p.proposer = f"model:{model_id or 'llm'}"
            out[code] = p
    return out, model_id
