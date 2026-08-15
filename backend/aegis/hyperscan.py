"""HYPERSCAN — sweep every subsystem, then say what it adds up to.

The consult answers questions. This answers the question nobody asked yet: given
everything AegisOS knows about this patient right now, what is the situation?

It is a staged sweep — vitals, prior record, organ state, the causal graph, the
consensus chamber, the identity guard, the trend, the candidate treatments, the
drug-interaction screen, the detector's own track record — each stage emitting a
progress event as it lands, so the body on screen can flag the finding while the
collection is still running. Then one synthesis pass turns the pile into a
situation report with a recommendation.

Three things are structural rather than requested of the model:

  * **The review notice is appended in code.** A prompt can ask for a caveat and
    a model can forget it. This one cannot be omitted, because it is not the
    model's to write — see ``REVIEW_NOTICE``.
  * **Refused options stay refused.** The treatment options carry the Z3 cage's
    verdicts, and a recommendation is only accepted if it names an option the
    cage allowed. If the model picks a refused one, that is reported as a refusal
    rather than quietly promoted.
  * **Every stage is attributed.** Each finding carries the subsystem it came
    from, so a clinician reading the report can go and check the panel it came
    out of instead of taking the summary's word for it.

This is decision SUPPORT over existing model output. It does not order anything;
it never touches the syscall gate.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

# Appended by this module, always, to every report. Not a prompt instruction —
# a prompt instruction is a request, and this is a requirement.
REVIEW_NOTICE = (
    "This is an automated synthesis from an EXPERIMENTAL research system, not a "
    "diagnosis and not a clinical decision. Every finding, option and figure "
    "above must be independently reviewed and verified by the responsible "
    "clinician against the patient and the primary record before it informs any "
    "action. The detector behind it is modest (see its measured performance "
    "below), roughly one alarm in three is a false alarm, and several inputs a "
    "clinician would consider are not measured by this system at all."
)

# stage key -> (label, body region the finding hangs off, tool to call)
STAGES: List[Dict[str, Any]] = [
    {"key": "vitals", "label": "Vitals and calibrated risk", "region": "heart",
     "tool": "get_patient"},
    {"key": "history", "label": "Prior record", "region": "brain",
     "tool": "get_patient_history"},
    {"key": "trend", "label": "Trajectory", "region": "heart",
     "tool": "get_vitals_trend"},
    {"key": "causal", "label": "Causal graph", "region": "brain",
     "tool": "get_causal_graph"},
    {"key": "consensus", "label": "Specialist consensus", "region": "brain",
     "tool": "run_consensus"},
    {"key": "identity", "label": "Physiologic identity", "region": "vessels",
     "tool": "check_identity"},
    {"key": "options", "label": "Treatment options", "region": "liver",
     "tool": "show_treatment_options"},
    {"key": "validation", "label": "Detector track record", "region": "metabolic",
     "tool": "get_validation"},
]

_SYSTEM = """You are DR. AEGIS performing a HYPERSCAN: a full sweep of every \
subsystem in an EXPERIMENTAL sepsis early-warning research system, synthesised \
into one situation report for the responsible clinician.

You are given the raw output of every subsystem. Produce STRICT JSON, no prose \
outside it, with exactly these keys:

{
 "situation": "3-5 sentences. What is going on with this patient right now, in \
plain clinical language. Lead with what matters. Name the patient, never a bed id.",
 "key_findings": [{"finding": "one sentence", "source": "which subsystem", \
"region": "heart|lungs|kidneys|liver|brain|blood|metabolic|vessels|immune|skin", \
"weight": "high|medium|low"}],
 "consistent_with": ["what the pattern is consistent with — NOT diagnoses"],
 "against": ["what argues against it — be honest, include the weak points"],
 "options": [{"label": "as given", "allowed": true/false, "verdict": "the cage's \
own words if refused", "tradeoff": "one sentence"}],
 "recommendation": {"option": "the EXACT label, copied character for character, \
of ONE option from the supplied list that the cage ALLOWED — never a label you \
invent", "why": "2-3 sentences tied to this patient's numbers", "watch": "what \
would change this"},
 "not_measured": ["inputs a clinician would want that this system does not have"],
 "confidence": "one honest sentence about how much weight this deserves"
}

Rules. Ground every claim in the supplied subsystem output; invent nothing. \
Attribute findings to the subsystem they came from. NEVER recommend an option the \
Z3 cage refused — if the clinically obvious choice was refused, say so in \
"against" and recommend from what remains. Quote AUROC as discrimination, never \
as accuracy. Distinguish measured values from the prior record, which is reported \
history. Give no doses, rates or timings. Output JSON only."""


class Hyperscan:
    def __init__(self, invoke: Callable[[str, Dict[str, Any]], Dict[str, Any]],
                 client: Any):
        self.invoke = invoke
        self.client = client

    # ------------------------------------------------------------------ #
    def collect(self, bed: Optional[str],
                emit: Optional[Callable[[Dict[str, Any]], None]] = None
                ) -> Dict[str, Any]:
        """Run every stage, announcing each before and after."""
        out: Dict[str, Any] = {}
        for i, st in enumerate(STAGES):
            if emit:
                emit({"type": "stage", "state": "running", "key": st["key"],
                      "label": st["label"], "region": st["region"],
                      "i": i, "n": len(STAGES)})
            args = {"bed": bed} if bed and st["tool"] != "get_validation" else {}
            try:
                res = self.invoke(st["tool"], args)
            except Exception as exc:               # a stage failing is data, not fatal
                res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            payload = res.get("data") if res.get("data") is not None else res.get("result")
            out[st["key"]] = payload if res.get("ok") else {"unavailable": res.get("error")}
            if emit:
                emit({"type": "stage", "state": "done", "key": st["key"],
                      "label": st["label"], "region": st["region"],
                      "ok": bool(res.get("ok")), "i": i, "n": len(STAGES),
                      "headline": _headline(st["key"], out[st["key"]])})
        return out

    # ------------------------------------------------------------------ #
    def synthesize(self, bed: Optional[str], collected: Dict[str, Any],
                   dossier: Dict[str, Any],
                   emit: Optional[Callable[[Dict[str, Any]], None]] = None
                   ) -> Dict[str, Any]:
        if emit:
            emit({"type": "stage", "state": "running", "key": "synthesis",
                  "label": "Synthesising", "region": "brain",
                  "i": len(STAGES), "n": len(STAGES) + 1})

        blob = json.dumps(collected, default=str)[:22000]
        # The candidate paths, pulled OUT of the blob and put in front of the
        # model. Buried at depth in 22 KB of JSON they were reliably missed —
        # the synthesis invented a label instead, which the enforcement below
        # then had to reject, leaving a safe but empty recommendation. A short
        # explicit menu is the difference between a guard that fires and a guard
        # that never has to.
        real = ((collected.get("options") or {}).get("options") or [])
        menu = "\n".join(
            f"  - {o.get('label')!r} — {'ALLOWED' if o.get('allowed') else 'REFUSED BY THE CAGE'}"
            f"{'' if o.get('allowed') else ': ' + str(o.get('verdict'))[:120]}"
            for o in real) or "  (none produced for this patient)"
        prompt = (f"CASE DOSSIER:\n{dossier.get('text', '')}\n\n"
                  f"CANDIDATE TREATMENT PATHS — your recommendation.option MUST be one "
                  f"of these labels, copied exactly, and MUST be one marked ALLOWED:\n"
                  f"{menu}\n\n"
                  f"SUBSYSTEM SWEEP (raw):\n{blob}\n\n"
                  "Produce the hyperscan JSON now.")
        text = None
        if getattr(self.client, "available", False):
            text = self.client.chat(_SYSTEM, prompt, max_tokens=2200,
                                    temperature=0.25, respect_interval=False)
        report = _parse(text) or _template(dossier, collected)
        report = self._enforce(report, collected)
        report.update({
            "bed": dossier.get("bed"), "patient": dossier.get("name"),
            "headline": dossier.get("headline"),
            "generated": time.time(),
            "source": "llm" if text else "template",
            "provenance": dossier.get("provenance"),
            # never the model's to omit
            "review_notice": REVIEW_NOTICE,
        })
        if emit:
            emit({"type": "stage", "state": "done", "key": "synthesis",
                  "label": "Synthesising", "region": "brain", "ok": True,
                  "i": len(STAGES), "n": len(STAGES) + 1,
                  "headline": (report.get("situation") or "")[:90]})
        return report

    # ------------------------------------------------------------------ #
    @staticmethod
    def _enforce(report: Dict[str, Any], collected: Dict[str, Any]) -> Dict[str, Any]:
        """The options list and the cage's verdicts are FACTS, not model output.

        Left to author the options itself, the synthesis returned a single
        invented option ("escalate_to_high") while the treatment engine had
        produced six real ones with one refused by the cage — so the report
        showed a recommendation the safety layer had never seen, and the refusal
        check had nothing real to compare against. The candidate set and every
        verdict are therefore rebuilt here from the subsystem output; the model
        contributes only the trade-off wording and WHICH of the real options it
        would pick, and that pick has to match one the cage allowed."""
        real = ((collected.get("options") or {}).get("options") or [])
        by_label = {str(o.get("label", "")).strip().lower(): o for o in real}
        allowed = [o.get("label") for o in real if o.get("allowed")]
        refused = [o.get("label") for o in real if not o.get("allowed")]

        # keep the model's tradeoff prose, but only where it names a REAL option
        said = {str(o.get("label", "")).strip().lower(): o
                for o in (report.get("options") or []) if isinstance(o, dict)}
        report["options"] = [{
            "label": o.get("label"),
            "allowed": bool(o.get("allowed")),
            "verdict": o.get("verdict"),
            "violations": o.get("violations") or [],
            "tradeoff": (said.get(str(o.get("label", "")).strip().lower()) or {}).get("tradeoff", ""),
        } for o in real]
        report["options_source"] = "treatment engine + Z3 cage (verbatim)"

        rec = report.get("recommendation") or {}
        pick = str(rec.get("option") or "").strip().lower()
        if not real:
            rec["option"] = None
            rec["note"] = "No candidate treatment paths were produced for this patient."
        elif pick not in by_label:
            rec["note"] = (f"The synthesis named {rec.get('option')!r}, which is not one "
                           "of the candidate paths this system evaluated. Only the "
                           "options listed above were assessed by the safety cage.")
            rec["option"] = None
            rec["allowed_options"] = allowed
        elif not by_label[pick].get("allowed"):
            rec["refused_by_cage"] = True
            rec["note"] = (f"The synthesis selected {rec.get('option')!r}, which the Z3 "
                           f"safety cage REFUSED: {by_label[pick].get('verdict')}. The "
                           "cage's verdict stands; choose from the permitted options.")
            rec["option"] = None
            rec["allowed_options"] = allowed
        else:
            rec["option"] = by_label[pick].get("label")     # canonical casing
            rec["cage_verdict"] = by_label[pick].get("verdict")
        report["recommendation"] = rec
        report["refused_options"] = refused
        return report


# ---------------------------------------------------------------------- #
def _headline(key: str, payload: Any) -> str:
    """A one-line 'what this stage found', for the flag on the body."""
    if not isinstance(payload, dict):
        return ""
    if payload.get("unavailable"):
        return "unavailable"
    try:
        if key == "vitals":
            r = payload.get("risk") or {}
            return f"{r.get('level', '—')} · P={r.get('ml_probability', '—')}"
        if key == "history":
            return ("prior record on file" if payload.get("has_prior_record")
                    else "no prior chart")
        if key == "causal":
            s = payload.get("summary") or {}
            return f"{s.get('nodes', 0)} nodes · {s.get('edges', 0)} edges"
        if key == "consensus":
            return f"{payload.get('path', 'consensus')} reached"
        if key == "identity":
            return "identity fault" if payload.get("fault") else "identity consistent"
        if key == "options":
            return f"{payload.get('n_options', 0)} options · {payload.get('n_refused', 0)} refused"
        if key == "validation":
            s = payload.get("summary") or {}
            return f"detection {s.get('detection_rate', '—')}"
        if key == "trend":
            return f"{len(payload.get('hours') or [])} h of record"
    except Exception:
        pass
    return "collected"


def _parse(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):                        # models fence JSON habitually
        s = s.split("```", 2)[1]
        s = s[4:] if s.lower().startswith("json") else s
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        got = json.loads(s[a:b + 1])
        return got if isinstance(got, dict) else None
    except Exception:
        return None


def _template(dossier: Dict[str, Any], collected: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback. The sweep is real even when the model is not
    available, so the report degrades to the retrieved facts rather than to
    nothing."""
    f = dossier.get("facts") or {}
    opts = ((collected.get("options") or {}).get("options") or [])
    return {
        "situation": (f"[synthesis offline] {dossier.get('headline', 'this patient')}. "
                      f"SIRS {f.get('sirs', '—')}/4, qSOFA {f.get('qsofa', '—')}/3. "
                      "The subsystem sweep below completed; the narrative layer did not."),
        "key_findings": [{"finding": x, "source": "dossier", "region": "heart",
                          "weight": "medium"}
                         for x in (f.get("qsofa_firing") or [])[:4]],
        "consistent_with": [], "against": [],
        "options": [{"label": o.get("label"), "allowed": o.get("allowed"),
                     "verdict": o.get("verdict"), "tradeoff": ""} for o in opts],
        "recommendation": {"option": None, "why": "Synthesis unavailable.",
                           "watch": ""},
        "not_measured": dossier.get("unmeasured") or [],
        "confidence": "Narrative synthesis was unavailable; only retrieved facts shown.",
    }
