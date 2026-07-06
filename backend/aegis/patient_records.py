"""Per-patient medical-history intake + records-source connectors.

This feeds the per-patient tracking profile (foci + watch-rules + priorities) from
a patient's prior medical history, so AegisOS *specialises* its monitoring toward
that individual's pressing issues — WITHOUT ever crossing the observation-only
safety boundary that the patient terminal enforces.

Two intake paths, one parser:

  * **manual upload / paste** — a clinician pastes (or uploads) the patient's
    history as free text; it is parsed (LLM + deterministic fallback) into a
    structured record *and* a tracking directive over the SAME fixed vocabulary
    the patient terminal allows (organ systems + metrics). Anything outside that
    vocabulary is dropped; it cannot take effect.

  * **records-source connectors** — a small, MCP-shaped adapter registry. A real
    EHR / hospital-database transport slots in behind :class:`RecordsSource`
    (e.g. an MCP client tool); shipped here is an honest :class:`MockHospitalSource`
    (canned demo records, clearly labelled a stub) so the whole pipeline is
    exercisable end-to-end without a real hospital system attached.

Safety boundary (load-bearing, identical to ``patient_terminal``): importing a
history can seed a FOCUS or a WATCH and record NOTES / PRIORITIES. It can NEVER
place an order, prescribe a drug, change a dose, or move a threshold that gates
dispensing — those stay behind the audited, Z3-gated syscall gate.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# The fixed, safe vocabulary + resolvers live in patient_terminal. Importing the
# names (not the module) keeps the dependency one-directional: patient_terminal
# imports THIS module only lazily, from inside its methods.
from backend.aegis.patient_terminal import (
    METRICS, ORGANS, OPS, resolve_metric, resolve_organ,
)

# --------------------------------------------------------------------------- #
# Comorbidity → monitoring specialisation. Each known chronic condition seeds an
# organ focus (and, where clinically sensible, a conservative WATCH). This is the
# deterministic backbone; the LLM refines it but can never exceed the vocabulary.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Spec:
    organ: str
    # optional seed watch: (metric, op, threshold, severity)
    watch: Optional[Tuple[str, str, float, str]] = None
    label: str = ""


# ordered longest-key-first at match time so "chronic kidney disease" wins over "kidney"
COMORBIDITY_MAP: Dict[str, _Spec] = {
    "chronic kidney disease": _Spec("Renal", ("map", "<", 65, "watch"), "CKD"),
    "renal failure":          _Spec("Renal", ("map", "<", 65, "watch"), "renal failure"),
    "ckd":                    _Spec("Renal", ("map", "<", 65, "watch"), "CKD"),
    "dialysis":               _Spec("Renal", None, "dialysis-dependent"),
    "aki":                    _Spec("Renal", None, "prior AKI"),

    "congestive heart failure": _Spec("Cardiovascular", ("map", "<", 60, "watch"), "CHF"),
    "heart failure":          _Spec("Cardiovascular", ("map", "<", 60, "watch"), "heart failure"),
    "chf":                    _Spec("Cardiovascular", ("map", "<", 60, "watch"), "CHF"),
    "cardiomyopathy":         _Spec("Cardiovascular", None, "cardiomyopathy"),
    "coronary artery disease": _Spec("Cardiovascular", None, "CAD"),
    "cad":                    _Spec("Cardiovascular", None, "CAD"),
    "atrial fibrillation":    _Spec("Cardiovascular", ("heart_rate", ">", 110, "watch"), "AF"),
    "hypertension":           _Spec("Cardiovascular", None, "hypertension"),

    "copd":                   _Spec("Respiratory", ("spo2", "<", 92, "watch"), "COPD"),
    "chronic obstructive":    _Spec("Respiratory", ("spo2", "<", 92, "watch"), "COPD"),
    "asthma":                 _Spec("Respiratory", ("spo2", "<", 93, "watch"), "asthma"),
    "pulmonary fibrosis":     _Spec("Respiratory", ("spo2", "<", 92, "watch"), "ILD"),
    "pneumonia":              _Spec("Respiratory", ("respiratory_rate", ">", 24, "watch"), "pneumonia hx"),
    "osa":                    _Spec("Respiratory", None, "sleep apnoea"),

    "cirrhosis":              _Spec("Hepatic", None, "cirrhosis"),
    "liver failure":          _Spec("Hepatic", None, "liver failure"),
    "hepatic":                _Spec("Hepatic", None, "hepatic disease"),

    "diabetes":               _Spec("Metabolic", ("sweat_lactate", ">", 2.5, "watch"), "diabetes"),
    "dka":                    _Spec("Metabolic", ("sweat_lactate", ">", 2.5, "watch"), "prior DKA"),
    "lactic acidosis":        _Spec("Metabolic", ("sweat_lactate", ">", 2.5, "critical"), "prior lactic acidosis"),

    "immunocompromised":      _Spec("Immune", ("skin_temp", ">", 38.0, "watch"), "immunocompromised"),
    "neutropenia":            _Spec("Immune", ("skin_temp", ">", 38.0, "critical"), "neutropenia"),
    "chemotherapy":           _Spec("Immune", ("skin_temp", ">", 38.0, "critical"), "on chemo"),
    "chemo":                  _Spec("Immune", ("skin_temp", ">", 38.0, "critical"), "on chemo"),
    "transplant":             _Spec("Immune", ("skin_temp", ">", 38.0, "watch"), "transplant"),
    "hiv":                    _Spec("Immune", None, "HIV"),
    "cancer":                 _Spec("Immune", None, "malignancy"),
    "malignancy":             _Spec("Immune", None, "malignancy"),

    "coagulopathy":           _Spec("Coagulation", None, "coagulopathy"),
    "dic":                    _Spec("Coagulation", None, "prior DIC"),
    "thrombocytopenia":       _Spec("Coagulation", None, "thrombocytopenia"),
    "anticoagulated":         _Spec("Coagulation", None, "anticoagulated"),

    "stroke":                 _Spec("Neurological", None, "prior stroke"),
    "dementia":               _Spec("Neurological", None, "dementia"),
    "delirium":               _Spec("Neurological", None, "delirium-prone"),
    "seizure":                _Spec("Neurological", None, "seizure disorder"),

    "sepsis":                 _Spec("Immune", ("skin_temp", ">", 38.0, "watch"), "prior sepsis"),
    "septic shock":           _Spec("Cardiovascular", ("map", "<", 65, "critical"), "prior septic shock"),
}


# --------------------------------------------------------------------------- #
@dataclass
class MedicalHistory:
    """A parsed, structured medical history + how it specialised the monitoring."""
    bed_id: str
    comorbidities: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    priorities: List[str] = field(default_factory=list)   # pressing issues (free phrases)
    summary: str = ""
    raw: str = ""
    source: str = "manual"
    imported: float = field(default_factory=time.time)
    # what the import seeded (for the audit trail shown to the clinician)
    seeded_foci: List[str] = field(default_factory=list)
    seeded_watches: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bed_id": self.bed_id, "comorbidities": self.comorbidities,
            "medications": self.medications, "allergies": self.allergies,
            "priorities": self.priorities, "summary": self.summary,
            "source": self.source, "imported": self.imported,
            "seeded_foci": self.seeded_foci, "seeded_watches": self.seeded_watches,
            "raw_len": len(self.raw or ""),
        }


# --------------------------------------------------------------------------- #
# LLM extraction — structured, strict, vocabulary-bounded.
# --------------------------------------------------------------------------- #
_HISTORY_SYSTEM = (
    "You extract a patient's PRIOR MEDICAL HISTORY into strict JSON for an "
    "experimental sepsis early-warning MONITOR. Output ONLY the JSON (no prose, no "
    "code fence). Schema:\n"
    '{"comorbidities":[str],"medications":[str],"allergies":[str],'
    '"priorities":[short phrase describing a pressing issue to watch],'
    '"focus":[organ system to emphasise],'
    '"watches":[{"metric":str,"op":"<|>|<=|>=","threshold":number,"severity":"watch|critical"}],'
    '"summary":"one short clinical sentence"}\n'
    f"Allowed organ systems (focus): {ORGANS}.\n"
    f"Allowed metrics (watches): {list(METRICS)} (map lactate->sweat_lactate, "
    "hr->heart_rate, bp->systolic_bp, temp->skin_temp, o2/sat->spo2, rr->respiratory_rate).\n"
    "Rules: (1) focus/watches MUST use only the allowed organs/metrics; drop anything "
    "else. (2) This configures MONITORING ONLY — never emit a treatment, drug order, "
    "or dose as an action. Listing the patient's home medications under 'medications' "
    "is fine (it's history, not an order). (3) Choose foci/watches that match the "
    "comorbidities so the monitor specialises toward this patient's risks. (4) Keep "
    "'summary' to one sentence."
)


def parse_history(bed_id: str, text: str, source: str = "manual",
                  llm_client: Optional[Any] = None) -> Tuple[MedicalHistory, Dict[str, Any]]:
    """Parse free-text history into a :class:`MedicalHistory` + a tracking directive
    (``{focus, watches}``) ready for ``PatientTerminal.apply``. LLM first when
    available; a deterministic comorbidity scan always backs it up (and its foci/
    watches are unioned in, so a working parse never depends on the network)."""
    hist = MedicalHistory(bed_id=bed_id, raw=text or "", source=source)
    directive: Dict[str, Any] = {"focus": [], "watches": []}

    # -- LLM extraction (optional) -------------------------------------- #
    if llm_client is not None and getattr(llm_client, "available", False) and (text or "").strip():
        try:
            out = llm_client.chat(_HISTORY_SYSTEM, (text or "")[:4000], max_tokens=420,
                                  temperature=0.0, respect_interval=False)
            d = _extract_json(out)
        except Exception:
            d = None
        if isinstance(d, dict):
            hist.comorbidities = _strlist(d.get("comorbidities"))
            hist.medications = _strlist(d.get("medications"))
            hist.allergies = _strlist(d.get("allergies"))
            hist.priorities = _strlist(d.get("priorities"))
            hist.summary = str(d.get("summary", ""))[:280]
            for o in d.get("focus", []) or []:
                org = resolve_organ(str(o))
                if org and org not in directive["focus"]:
                    directive["focus"].append(org)
            for w in d.get("watches", []) or []:
                sw = _sanitize_watch(w)
                if sw:
                    directive["watches"].append(sw)

    # -- deterministic comorbidity backbone (always runs) --------------- #
    det_comorbid, det_focus, det_watches = _scan_comorbidities(text or "")
    for c in det_comorbid:
        if c not in hist.comorbidities:
            hist.comorbidities.append(c)
    for org in det_focus:
        if org not in directive["focus"]:
            directive["focus"].append(org)
    # de-dupe watches by (metric, op) — LLM watch wins, deterministic fills gaps
    seen = {(w["metric"], w["op"]) for w in directive["watches"]}
    for w in det_watches:
        key = (w["metric"], w["op"])
        if key not in seen:
            directive["watches"].append(w)
            seen.add(key)

    if not hist.summary:
        hist.summary = _fallback_summary(hist)
    return hist, directive


def _scan_comorbidities(text: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Deterministic keyword scan → (comorbidity labels, organ foci, seed watches)."""
    t = " " + (text or "").lower() + " "
    comorbid: List[str] = []
    foci: List[str] = []
    watches: List[Dict[str, Any]] = []
    # longest keys first so multi-word conditions win over their substrings
    for key in sorted(COMORBIDITY_MAP, key=len, reverse=True):
        if re.search(r"\b" + re.escape(key) + r"\b", t):
            spec = COMORBIDITY_MAP[key]
            lbl = spec.label or key
            if lbl not in comorbid:
                comorbid.append(lbl)
            if spec.organ not in foci:
                foci.append(spec.organ)
            if spec.watch:
                m, op, th, sev = spec.watch
                if not any(w["metric"] == m and w["op"] == op for w in watches):
                    watches.append({"metric": m, "op": op, "threshold": float(th),
                                    "severity": sev})
    return comorbid, foci, watches


def _sanitize_watch(w: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(w, dict):
        return None
    met = resolve_metric(str(w.get("metric", "")))
    op = str(w.get("op", "")).strip()
    try:
        th = float(w.get("threshold"))
    except (TypeError, ValueError):
        return None
    if not met or op not in OPS:
        return None
    sev = "critical" if str(w.get("severity", "")).lower().startswith("crit") else "watch"
    return {"metric": met, "op": op, "threshold": th, "severity": sev}


def _fallback_summary(h: MedicalHistory) -> str:
    if h.comorbidities:
        return "History of " + ", ".join(h.comorbidities[:4]) + \
               ("…" if len(h.comorbidities) > 4 else "") + "."
    return "No prior comorbidities parsed from the record."


def _strlist(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out = []
    for v in x:
        s = str(v).strip()
        if s and s.lower() not in ("none", "n/a", "nil") and s not in out:
            out.append(s[:80])
    return out[:20]


def _extract_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    if "```" in s:
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s.strip("`\n "), flags=re.I)
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b < a:
        return None
    try:
        d = json.loads(s[a:b + 1])
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Records-source connectors — MCP-shaped adapter registry.
# --------------------------------------------------------------------------- #
class RecordsSource:
    """Adapter interface for a source of patient history. Real EHR / hospital-DB
    transports (e.g. an MCP client tool, an HL7/FHIR endpoint) implement ``fetch``.
    Everything downstream (parse → specialise) is transport-agnostic."""
    id: str = "source"
    label: str = "Records source"
    kind: str = "stub"           # "stub" | "manual" | "mcp" | "fhir" — honesty flag
    live: bool = False           # True only when a real backing system is attached

    def fetch(self, bed_id: str, patient: Optional[Dict[str, Any]] = None) -> Optional[str]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label, "kind": self.kind, "live": self.live}


class MockHospitalSource(RecordsSource):
    """An honest STUB: canned demo histories keyed off the patient's coarse profile,
    so the import→specialise pipeline runs end-to-end with no real hospital link.
    A production deployment registers a live :class:`RecordsSource` (MCP/FHIR) in its
    place; this never claims to have pulled a real chart."""
    id = "mock_hospital"
    label = "Demo Hospital EHR (stub)"
    kind = "stub"
    live = False

    _SEPTIC = (
        "PAST MEDICAL HISTORY: Type 2 diabetes mellitus, chronic kidney disease "
        "(stage 3b, baseline creatinine 1.9), congestive heart failure (EF 35%), "
        "prior admission for pneumonia. Immunocompromised — on low-dose chemotherapy "
        "for lymphoma. Home medications: metformin, furosemide, lisinopril, insulin "
        "glargine. Allergies: penicillin (rash). Priorities: high sepsis risk given "
        "immunosuppression; watch renal perfusion and oxygenation closely."
    )
    _STABLE = (
        "PAST MEDICAL HISTORY: Hypertension, well-controlled asthma. Home medications: "
        "amlodipine, albuterol PRN. No known drug allergies. Priorities: routine "
        "monitoring; low baseline acuity."
    )

    def fetch(self, bed_id: str, patient: Optional[Dict[str, Any]] = None) -> Optional[str]:
        septic = bool((patient or {}).get("septic"))
        return self._SEPTIC if septic else self._STABLE


class ManualSource(RecordsSource):
    """Marker for the clinician paste/upload path (no remote fetch)."""
    id = "manual"
    label = "Manual paste / file upload"
    kind = "manual"
    live = True

    def fetch(self, bed_id: str, patient: Optional[Dict[str, Any]] = None) -> Optional[str]:
        return None


class RecordsRegistry:
    """The pluggable connector registry. Register a real MCP/FHIR source here to make
    the hospital link live; the frontend reads :meth:`list_sources` to offer them."""

    def __init__(self) -> None:
        self._sources: Dict[str, RecordsSource] = {}
        self.register(ManualSource())
        self.register(MockHospitalSource())

    def register(self, source: RecordsSource) -> None:
        self._sources[source.id] = source

    def get(self, source_id: str) -> Optional[RecordsSource]:
        return self._sources.get(source_id)

    def list_sources(self) -> List[Dict[str, Any]]:
        return [s.describe() for s in self._sources.values()]
