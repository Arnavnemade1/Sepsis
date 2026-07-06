"""Per-patient tracking control plane — a terminal for EACH patient.

Every bed gets its own shell in which the attending can shape how AegisOS watches
that specific patient — without touching anyone else's monitoring:

  * **focus** an organ system ("watch her kidneys") so the twin + narrative + causal
    story emphasise it and its watches tighten;
  * define **watch rules** ("alert me if lactate goes over 3", "MAP under 60") that
    are evaluated every tick and fire edge-triggered events into the clinical story;
  * drive all of it in **natural language** — the LLM parses the instruction into a
    validated, structured directive (never a free-text side effect).

Safety boundary (load-bearing): this configures *observation only*. It can add a
focus or a watch/alert; it can NEVER place an order, change a dose, or move a
threshold that gates dispensing — those live behind the audited, Z3-gated syscall
gate. The LLM is constrained to emit a tracking directive over a fixed vocabulary
(known metrics + organ systems); anything else is refused and dropped. When the
LLM is unavailable the whole thing still works through a deterministic parser.

Later: importing a patient's medical history feeds this same profile (seed foci +
watches from prior conditions), so the system specialises toward the patient.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Vocabulary — the fixed, safe surface the terminal (and the LLM) may touch.
# --------------------------------------------------------------------------- #
METRICS: Dict[str, Dict[str, Any]] = {
    "heart_rate":       {"label": "heart rate", "unit": "bpm", "aliases": ["hr", "pulse", "heartrate", "heart"]},
    "hrv":              {"label": "HRV", "unit": "ms", "aliases": ["heart rate variability", "variability"]},
    "spo2":             {"label": "SpO₂", "unit": "%", "aliases": ["sat", "o2", "oxygen", "saturation", "spo2"]},
    "respiratory_rate": {"label": "respiratory rate", "unit": "/min", "aliases": ["rr", "resp", "breathing", "respiration"]},
    "systolic_bp":      {"label": "systolic BP", "unit": "mmHg", "aliases": ["sbp", "bp", "systolic", "blood pressure"]},
    "map":              {"label": "MAP", "unit": "mmHg", "aliases": ["mean arterial", "mean arterial pressure"]},
    "skin_temp":        {"label": "temperature", "unit": "°C", "aliases": ["temp", "temperature", "fever"]},
    "sweat_lactate":    {"label": "lactate", "unit": "mmol/L", "aliases": ["lactate", "lac"]},
}
ORGANS = ["Cardiovascular", "Respiratory", "Neurological", "Autonomic",
          "Peripheral Perfusion", "Renal", "Hepatic", "Coagulation", "Metabolic", "Immune"]
_ORGAN_ALIASES = {
    "heart": "Cardiovascular", "cardiac": "Cardiovascular", "cardio": "Cardiovascular", "cv": "Cardiovascular",
    "lung": "Respiratory", "lungs": "Respiratory", "pulmonary": "Respiratory", "resp": "Respiratory",
    "brain": "Neurological", "neuro": "Neurological", "cns": "Neurological", "mental": "Neurological",
    "autonomic": "Autonomic", "hrv": "Autonomic",
    "skin": "Peripheral Perfusion", "perfusion": "Peripheral Perfusion", "peripheral": "Peripheral Perfusion",
    "kidney": "Renal", "kidneys": "Renal", "renal": "Renal",
    "liver": "Hepatic", "hepatic": "Hepatic",
    "clotting": "Coagulation", "coag": "Coagulation", "platelets": "Coagulation", "coagulation": "Coagulation",
    "metabolic": "Metabolic", "lactate": "Metabolic", "acid": "Metabolic",
    "immune": "Immune", "wbc": "Immune", "infection": "Immune", "inflammatory": "Immune",
}
OPS = {"<", ">", "<=", ">=", "=="}
_OP_WORDS = {"above": ">", "over": ">", "exceeds": ">", "greater": ">", "gt": ">",
             "below": "<", "under": "<", "less": "<", "lt": "<",
             "reaches": ">=", "at least": ">=", "at most": "<="}

# Words that signal a TREATMENT/ORDER intent in a free-text clinician synthesis.
# We never execute these here — we flag them so the reply can redirect to the
# certified, Z3-gated order gate (the load-bearing safety boundary).
_TREATMENT_RE = re.compile(
    r"\b(order|prescrib\w*|administer|give|start|bolus|dose|dosing|titrat\w*|"
    r"infus\w*|push|stat|mg|mcg|units?|ml/hr|antibiotic\w*|vasopressor\w*|"
    r"norepinephrine|levophed|fluids?|abx|treat\w*|intubat\w*)\b", re.I)


def resolve_metric(token: str) -> Optional[str]:
    t = (token or "").strip().lower()
    if t in METRICS:
        return t
    for key, spec in METRICS.items():
        if t == key or t in spec["aliases"]:
            return key
    return None


def resolve_organ(token: str) -> Optional[str]:
    t = (token or "").strip().lower()
    for o in ORGANS:
        if t == o.lower():
            return o
    return _ORGAN_ALIASES.get(t)


# --------------------------------------------------------------------------- #
@dataclass
class WatchRule:
    metric: str
    op: str
    threshold: float
    severity: str = "watch"           # watch | critical
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:6])
    created: float = field(default_factory=time.time)
    _fired: bool = field(default=False, repr=False)

    @property
    def label(self) -> str:
        m = METRICS.get(self.metric, {"label": self.metric, "unit": ""})
        return f"{m['label']} {self.op} {self.threshold:g} {m['unit']}".strip()

    def matches(self, value: float) -> bool:
        if value is None:
            return False
        v, th = float(value), float(self.threshold)
        return {"<": v < th, ">": v > th, "<=": v <= th, ">=": v >= th,
                "==": abs(v - th) < 1e-6}[self.op]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "metric": self.metric, "op": self.op,
                "threshold": self.threshold, "severity": self.severity,
                "label": self.label, "active": self._fired}


@dataclass
class TrackingProfile:
    bed_id: str
    foci: List[str] = field(default_factory=list)       # organ systems emphasised
    watches: List[WatchRule] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    updated: float = field(default_factory=time.time)

    # -- mutation (returns a human summary line) ------------------------- #
    def add_focus(self, organ: str) -> str:
        o = resolve_organ(organ)
        if not o:
            return f"unknown system '{organ}' — try: {', '.join(ORGANS[:5])}…"
        if o not in self.foci:
            self.foci.append(o); self.updated = time.time()
            return f"focusing on {o}"
        return f"already focusing on {o}"

    def remove_focus(self, organ: str) -> str:
        o = resolve_organ(organ)
        if o and o in self.foci:
            self.foci.remove(o); self.updated = time.time()
            return f"released focus on {o}"
        return f"not focusing on '{organ}'"

    def add_watch(self, metric: str, op: str, threshold: float,
                  severity: str = "watch") -> str:
        m = resolve_metric(metric)
        if not m:
            return f"unknown metric '{metric}' — try: {', '.join(list(METRICS)[:5])}…"
        if op not in OPS:
            return f"bad operator '{op}' (use < > <= >= ==)"
        sev = "critical" if str(severity).lower().startswith("crit") else "watch"
        w = WatchRule(metric=m, op=op, threshold=float(threshold), severity=sev)
        self.watches.append(w); self.updated = time.time()
        return f"watching {w.label}" + (" [CRITICAL]" if sev == "critical" else "")

    def remove_watch(self, key: str) -> str:
        if key == "all":
            n = len(self.watches); self.watches.clear(); self.updated = time.time()
            return f"cleared {n} watch{'es' if n != 1 else ''}"
        m = resolve_metric(key)
        before = len(self.watches)
        self.watches = [w for w in self.watches if w.id != key and w.metric != m]
        self.updated = time.time()
        return "removed watch" if len(self.watches) < before else f"no watch matching '{key}'"

    # -- evaluation (edge-triggered) ------------------------------------- #
    def evaluate(self, vitals: Dict[str, float]) -> List[Dict[str, Any]]:
        """Return newly-fired watch events; edge-triggered so each fires once per
        crossing and re-arms when the value returns to normal."""
        fired: List[Dict[str, Any]] = []
        for w in self.watches:
            val = _metric_value(w.metric, vitals)
            hit = w.matches(val)
            if hit and not w._fired:
                w._fired = True
                fired.append({"watch_id": w.id, "metric": w.metric, "op": w.op,
                              "threshold": w.threshold, "value": round(float(val), 2),
                              "label": w.label, "severity": w.severity,
                              "kind": "doctor_watch"})
            elif not hit:
                w._fired = False
        return fired

    def to_dict(self) -> Dict[str, Any]:
        return {"bed_id": self.bed_id, "foci": list(self.foci),
                "watches": [w.to_dict() for w in self.watches],
                "notes": self.notes[-8:], "updated": self.updated,
                "n_watches": len(self.watches)}


def _metric_value(metric: str, vitals: Dict[str, float]) -> Optional[float]:
    v = vitals.get(metric)
    if v is None and metric == "sweat_lactate":
        v = vitals.get("lactate")
    if v is None and metric == "heart_rate":
        v = vitals.get("hr")
    if v is None and metric == "respiratory_rate":
        v = vitals.get("resp_rate")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# The per-patient terminal: structured commands + LLM natural language.
# --------------------------------------------------------------------------- #
_LLM_SYSTEM = (
    "You configure per-patient MONITORING for an experimental sepsis early-warning "
    "system. Convert the clinician's instruction into a STRICT JSON tracking "
    "directive and output ONLY that JSON (no prose, no code fence). Schema:\n"
    '{"focus":[organ...],"unfocus":[organ...],"clear_focus":false,'
    '"watches":[{"metric":str,"op":"<|>|<=|>=","threshold":number,"severity":"watch|critical"}],'
    '"unwatch":[metric...],"reply":"one short confirmation sentence"}\n'
    f"Allowed organ systems: {ORGANS}.\n"
    f"Allowed metrics: {list(METRICS)} (map lactate→sweat_lactate, hr→heart_rate, "
    "bp→systolic_bp, temp→skin_temp, o2/sat→spo2, rr→respiratory_rate).\n"
    "Rules: (1) use ONLY the allowed organs/metrics; drop anything else. (2) You "
    "configure OBSERVATION ONLY — never emit a treatment, drug, dose, or order; if "
    "asked to treat/dose, set every list empty and put a refusal in 'reply' pointing "
    "to the certified order gate. (3) omit fields you don't use. Keep 'reply' short."
)


class PatientTerminal:
    """Registry of per-patient tracking profiles + the command interpreter."""

    def __init__(self, llm_client: Optional[Any] = None):
        self.profiles: Dict[str, TrackingProfile] = {}
        self.client = llm_client            # OpenRouterClient (optional)
        # medical-history intake + clinician synthesis log (lazy registry import to
        # avoid a circular import at module load — patient_records imports us).
        self.histories: Dict[str, Any] = {}          # bed -> MedicalHistory
        self.annotations: Dict[str, List[Dict[str, Any]]] = {}   # bed -> [synthesis notes]
        self._registry = None

    @property
    def registry(self):
        if self._registry is None:
            from backend.aegis.patient_records import RecordsRegistry
            self._registry = RecordsRegistry()
        return self._registry

    def profile(self, bed_id: str) -> TrackingProfile:
        return self.profiles.setdefault(bed_id, TrackingProfile(bed_id=bed_id))

    def evaluate(self, bed_id: str, vitals: Dict[str, float]) -> List[Dict[str, Any]]:
        if bed_id not in self.profiles:
            return []
        return self.profiles[bed_id].evaluate(vitals)

    # -- command entry point --------------------------------------------- #
    def exec(self, bed_id: str, line: str, context: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
        """Run one terminal line for a patient. Structured verbs are handled
        deterministically; anything else is routed to the LLM (with a keyword
        fallback). Returns {ok, reply, source, profile}."""
        prof = self.profile(bed_id)
        raw = (line or "").strip()
        if not raw:
            return self._out(prof, "type `help` for tracking commands, or just tell me "
                              "in plain English what to watch.", "system")
        toks = raw.split()
        verb = toks[0].lower()

        if verb in ("help", "?"):
            return self._out(prof, _HELP, "system")
        if verb == "status":
            return self._out(prof, self._status_text(prof), "system")
        if verb in ("focus", "watchsys") and len(toks) >= 2:
            if toks[1].lower() in ("clear", "none"):
                prof.foci.clear(); prof.updated = time.time()
                return self._out(prof, "cleared all focus", "command")
            org = resolve_organ(" ".join(toks[1:]))
            if org:                                     # a single clean organ → apply directly
                return self._out(prof, prof.add_focus(org), "command")
            # a phrase like "her kidneys and heart" → fall through to natural language
        if verb in ("unfocus", "release") and len(toks) >= 2:
            return self._out(prof, prof.remove_focus(" ".join(toks[1:])), "command")
        if verb == "watch" and len(toks) >= 2:
            parsed = _parse_watch(toks[1:])
            if parsed:
                metric, op, th, sev = parsed
                return self._out(prof, prof.add_watch(metric, op, th, sev), "command")
            # fall through to NL if it wasn't a clean `watch <metric> <op> <val>`
        if verb in ("unwatch", "rmwatch") and len(toks) >= 2:
            return self._out(prof, prof.remove_watch(toks[1]), "command")
        if verb == "note":
            prof.notes.append(raw[4:].strip()); prof.updated = time.time()
            return self._out(prof, "note added", "command")
        if verb in ("clear", "reset"):
            self.profiles[bed_id] = TrackingProfile(bed_id=bed_id)
            return self._out(self.profiles[bed_id], "tracking profile reset", "command")

        # -- natural language -> LLM directive -> apply -------------------- #
        return self._nl(prof, raw, context)

    # -- natural language path ------------------------------------------- #
    def _nl(self, prof: TrackingProfile, text: str,
            context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        directive, source = self._interpret(text, context)
        if directive is None:
            return self._out(prof, "couldn't parse that — try `help`, or e.g. "
                              "\"watch her kidneys and alert if lactate over 3\".", "system")
        applied = self.apply(prof, directive)
        reply = directive.get("reply") or ("; ".join(applied) if applied else "nothing to change")
        out = self._out(prof, reply, source)
        out["applied"] = applied
        return out

    def _interpret(self, text: str, context: Optional[Dict[str, Any]]
                   ) -> Tuple[Optional[Dict[str, Any]], str]:
        # LLM first (if available), else deterministic keyword parser.
        if self.client is not None and getattr(self.client, "available", False):
            ctx = f"\nPatient context (for grounding): {json.dumps(context)[:600]}" if context else ""
            txt = self.client.chat(_LLM_SYSTEM, text + ctx, max_tokens=260,
                                   temperature=0.0, respect_interval=False)
            d = _extract_json(txt)
            if d is not None:
                return _sanitize(d), "llm"
        return _keyword_directive(text), "parser"

    # -- apply a (sanitised) directive ----------------------------------- #
    def apply(self, prof: TrackingProfile, d: Dict[str, Any]) -> List[str]:
        applied: List[str] = []
        if d.get("clear_focus"):
            prof.foci.clear(); prof.updated = time.time(); applied.append("cleared focus")
        for o in d.get("focus", []) or []:
            r = prof.add_focus(o)
            if r.startswith("focus"):
                applied.append(r)
        for o in d.get("unfocus", []) or []:
            applied.append(prof.remove_focus(o))
        for w in d.get("watches", []) or []:
            try:
                applied.append(prof.add_watch(w["metric"], w["op"], w["threshold"],
                                              w.get("severity", "watch")))
            except (KeyError, TypeError, ValueError):
                continue
        for m in d.get("unwatch", []) or []:
            applied.append(prof.remove_watch(m))
        return [a for a in applied if a]

    # -- medical-history intake (specialises the monitor to the patient) -- #
    def import_history(self, bed_id: str, text: str, source: str = "manual"
                       ) -> Dict[str, Any]:
        """Parse a free-text medical history and SEED this patient's monitoring from
        it (foci + watch-rules). Observation-only: it can add foci/watches and record
        priorities; it can never place an order. Returns the parsed record + what it
        specialised."""
        from backend.aegis.patient_records import parse_history
        prof = self.profile(bed_id)
        hist, directive = parse_history(bed_id, text, source=source, llm_client=self.client)
        foci_before = set(prof.foci)
        watches_before = {w.id for w in prof.watches}
        applied = self.apply(prof, directive)
        hist.seeded_foci = [o for o in prof.foci if o not in foci_before]
        hist.seeded_watches = [w.label for w in prof.watches if w.id not in watches_before]
        for p in hist.priorities:
            note = "priority (history): " + p
            if note not in prof.notes:
                prof.notes.append(note)
        prof.updated = time.time()
        self.histories[bed_id] = hist
        return {"ok": True, "bed_id": bed_id, "source": source,
                "history": hist.to_dict(), "applied": applied,
                "profile": prof.to_dict()}

    def link_source(self, bed_id: str, source_id: str,
                    patient: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Pull a history from a registered records source (MCP-shaped stub / future
        live EHR) and import it. Honest about stubs: the returned payload carries the
        source's ``live`` flag so the UI never implies a real chart was fetched."""
        src = self.registry.get(source_id)
        if src is None:
            return {"ok": False, "reply": f"unknown records source '{source_id}'",
                    "sources": self.registry.list_sources()}
        text = src.fetch(bed_id, patient)
        if not text:
            return {"ok": False, "reply": f"{src.label}: no record returned "
                    "(paste/upload manually)", "source": src.describe()}
        out = self.import_history(bed_id, text, source=src.id)
        out["source"] = src.describe()
        return out

    def get_history(self, bed_id: str) -> Optional[Dict[str, Any]]:
        h = self.histories.get(bed_id)
        return h.to_dict() if h else None

    def sources(self) -> List[Dict[str, Any]]:
        return self.registry.list_sources()

    # -- clinician synthesis / instructions ------------------------------ #
    def add_synthesis(self, bed_id: str, text: str,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Record a clinician's detailed synthesis / instruction against this patient.
        The free text is stored verbatim as a timestamped annotation (surfaced in the
        clinical story), and any MONITORING intent in it (focus/watch, safe vocab) is
        applied. Treatment-like requests are NOT executed — the reply points to the
        certified, Z3-gated order gate (the honest safety boundary)."""
        prof = self.profile(bed_id)
        raw = (text or "").strip()
        if not raw:
            return {"ok": False, "reply": "empty synthesis"}
        directive, src = self._interpret(raw, context)
        applied = self.apply(prof, directive) if directive else []
        treat_flag = bool(_TREATMENT_RE.search(raw))
        note = {
            "id": uuid.uuid4().hex[:8], "t": time.time(), "text": raw[:1200],
            "author": "clinician", "applied": applied,
            "monitoring_only": True,
            "order_intent": treat_flag,
        }
        self.annotations.setdefault(bed_id, []).append(note)
        self.annotations[bed_id] = self.annotations[bed_id][-40:]
        parts = []
        if applied:
            parts.append("applied: " + "; ".join(applied))
        else:
            parts.append("logged")
        if treat_flag:
            parts.append("orders/doses were NOT executed — route treatment through the "
                         "certified `treat` gate (audited · Z3-verified)")
        return {"ok": True, "bed_id": bed_id, "reply": " · ".join(parts),
                "note": note, "applied": applied, "order_intent": treat_flag,
                "source": src, "profile": prof.to_dict()}

    def get_annotations(self, bed_id: str) -> List[Dict[str, Any]]:
        return list(self.annotations.get(bed_id, []))

    # -- helpers --------------------------------------------------------- #
    def _out(self, prof: TrackingProfile, reply: str, source: str) -> Dict[str, Any]:
        return {"ok": True, "reply": reply, "source": source,
                "bed_id": prof.bed_id, "profile": prof.to_dict()}

    @staticmethod
    def _status_text(prof: TrackingProfile) -> str:
        foci = ", ".join(prof.foci) if prof.foci else "(none — whole-body)"
        ws = " · ".join(w.label + ("!" if w.severity == "critical" else "") for w in prof.watches) or "(none)"
        return f"focus: {foci}\nwatches: {ws}"


_HELP = ("per-patient tracking · shape how AegisOS watches THIS patient:\n"
         "  focus <system>         emphasise an organ (heart, kidneys, lungs, …)\n"
         "  unfocus <system>       release it   ·   focus clear\n"
         "  watch <metric> <op> <v>  alert rule, e.g.  watch lactate > 3\n"
         "  unwatch <metric|all>   remove a watch\n"
         "  status · note <text> · clear\n"
         "…or just say it: \"keep an eye on her kidneys and tell me if MAP drops below 60\".")


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def _parse_watch(toks: List[str]) -> Optional[Tuple[str, str, float, str]]:
    """`watch lactate > 3` / `watch map < 60 critical` / `watch lactate>3`."""
    s = " ".join(toks)
    m = re.match(r"^([a-z0-9_ ]+?)\s*(<=|>=|==|<|>)\s*(-?\d+(?:\.\d+)?)\s*(crit\w*|watch)?\s*$",
                 s, re.I)
    if not m:
        return None
    metric = resolve_metric(m.group(1).strip())
    if not metric:
        return None
    sev = "critical" if (m.group(4) or "").lower().startswith("crit") else "watch"
    return metric, m.group(2), float(m.group(3)), sev


def _extract_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    if "```" in s:                       # strip a ```json … ``` fence
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s.strip("`\n "), flags=re.I)
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b < a:
        return None
    try:
        d = json.loads(s[a:b + 1])
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def _sanitize(d: Dict[str, Any]) -> Dict[str, Any]:
    """Filter an LLM directive down to the allowed vocabulary — the trust boundary."""
    out: Dict[str, Any] = {"reply": str(d.get("reply", ""))[:200]}
    if d.get("clear_focus"):
        out["clear_focus"] = True
    out["focus"] = [o for o in (resolve_organ(x) for x in d.get("focus", []) or []) if o]
    out["unfocus"] = [o for o in (resolve_organ(x) for x in d.get("unfocus", []) or []) if o]
    out["unwatch"] = [m for m in (resolve_metric(x) for x in d.get("unwatch", []) or []) if m]
    ws = []
    for w in d.get("watches", []) or []:
        if not isinstance(w, dict):
            continue
        met = resolve_metric(str(w.get("metric", "")))
        op = str(w.get("op", "")).strip()
        try:
            th = float(w.get("threshold"))
        except (TypeError, ValueError):
            continue
        if met and op in OPS:
            sev = "critical" if str(w.get("severity", "")).lower().startswith("crit") else "watch"
            ws.append({"metric": met, "op": op, "threshold": th, "severity": sev})
    out["watches"] = ws
    return out


def _keyword_directive(text: str) -> Optional[Dict[str, Any]]:
    """Deterministic fallback when the LLM is unavailable — best-effort NL parse."""
    t = text.lower()
    d: Dict[str, Any] = {"focus": [], "watches": [], "unwatch": [], "unfocus": []}
    # organ foci by keyword
    for alias, organ in _ORGAN_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", t) and organ not in d["focus"]:
            d["focus"].append(organ)
    # watch rules: "<metric> <above/below/over/under/>/<> <number>"
    for m in re.finditer(r"([a-z0-9_ ]{2,28}?)\s*(above|over|below|under|exceeds|less than|greater than|>=|<=|>|<|reaches)\s*(-?\d+(?:\.\d+)?)", t):
        phrase = m.group(1).strip()
        # try the whole phrase (catches multi-word aliases like "blood pressure"),
        # else scan its words nearest-to-operator first ("lactate goes" → lactate)
        metric = resolve_metric(phrase)
        if not metric:
            for word in reversed(phrase.split()):
                metric = resolve_metric(word)
                if metric:
                    break
        op = _OP_WORDS.get(m.group(2).strip(), m.group(2).strip())
        if metric and op in OPS:
            sev = "critical" if ("critical" in t or "urgent" in t) else "watch"
            d["watches"].append({"metric": metric, "op": op,
                                 "threshold": float(m.group(3)), "severity": sev})
    if not d["focus"] and not d["watches"]:
        return None
    parts = []
    if d["focus"]:
        parts.append("focusing on " + ", ".join(d["focus"]))
    if d["watches"]:
        parts.append("added " + str(len(d["watches"])) + " watch(es)")
    d["reply"] = "[parser] " + "; ".join(parts)
    return d
