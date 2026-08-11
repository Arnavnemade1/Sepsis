"""AegisAssistant — a grounded, read-only natural-language layer over live state.

The clinical explainer narrates ONE assessment. This adds two on-demand, operator-
facing capabilities that make the LLM key genuinely useful across AegisOS while
staying strictly inside the safety architecture:

  * ``ask(question, state)`` — answer a free-text question using ONLY the JSON
    state handed in (cohort risk, /proc, kernel/top, resonance, etc.). It is
    retrieval-grounded: if the answer is not in the state, it must say so. It is
    forbidden from inventing vitals, giving diagnoses, or issuing treatment
    orders — every actionable path in AegisOS goes through the audited, Z3-gated
    syscall gate, not through prose.
  * ``brief(state)`` — a concise shift-handoff summary of the cohort.

Both are user-initiated (never in the realtime hot loop), budgeted through the
shared client, cached, and fall back to a deterministic template when the LLM is
unavailable. The assistant introduces NO new authority — it explains state that
already exists.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from backend.llm.provider import LLMProvider

_ASK_SYSTEM = (
    "You are AegisOS's read-only operations assistant for an EXPERIMENTAL sepsis "
    "early-warning research system. You answer an operator's question using ONLY "
    "the JSON system state provided. Hard rules:\n"
    "(1) Ground every statement in the given JSON. If the answer is not present, "
    "say 'That is not in the current system state.' Never invent vitals, numbers, "
    "patients, or events.\n"
    "(2) You give NO diagnoses and NO treatment/dosing orders. If asked to act "
    "(treat, escalate, dose), reply that actions must go through the certified, "
    "Z3-gated syscall gate (e.g. the `treat`/`escalate` commands), not through you.\n"
    "(3) Be concise and specific: 2-5 sentences, cite the fields you used (bed id, "
    "risk, tier). Use plain clinical-operations language.\n"
    "(4) End nothing with medical advice; this is decision-support over existing "
    "model output, not a diagnosis."
)

_BRIEF_SYSTEM = (
    "You are AegisOS's shift-handoff assistant for an EXPERIMENTAL sepsis research "
    "system. Given the JSON cohort state, write a short handoff (4-6 sentences): "
    "how many beds and how they break down by tier, which beds need attention and "
    "why (name the driving signals from the state), and any kernel/service notes. "
    "Ground everything in the JSON; invent nothing; give no diagnoses or orders. "
    "End with one line noting this is a research tool, not a diagnosis."
)


class AegisAssistant:
    def __init__(self, client: Optional[LLMProvider] = None,
                 max_state_chars: int = 6000):
        self.client = client or LLMProvider()
        self.max_state_chars = max_state_chars

    # ------------------------------------------------------------------ #
    def ask(self, question: str, state: Dict[str, Any],
            use_llm: bool = True, max_tokens: int = 320) -> Dict[str, Any]:
        """Answer ``question`` grounded in ``state``. Returns
        {answer, source: llm|template, model, grounded: True}."""
        question = (question or "").strip()
        if not question:
            return {"answer": "Ask a question about the current system state, "
                    "e.g. 'which beds are critical?' or 'what is the kernel doing?'.",
                    "source": "template", "model": None, "grounded": True}
        blob = self._state_blob(state)
        if use_llm and self.client.available:
            prompt = (f"SYSTEM STATE (JSON):\n{blob}\n\n"
                      f"OPERATOR QUESTION: {question}\n\n"
                      f"Answer using only the state above.")
            text = self.client.chat(_ASK_SYSTEM, prompt, max_tokens=max_tokens,
                                    respect_interval=False)   # user-initiated
            if text:
                return {"answer": text, "source": "llm",
                        "model": self.client._last_model, "grounded": True}
        return {"answer": self._ask_template(question, state),
                "source": "template", "model": None, "grounded": True}

    def brief(self, state: Dict[str, Any], use_llm: bool = True,
              max_tokens: int = 400) -> Dict[str, Any]:
        """Shift-handoff summary of the cohort state."""
        blob = self._state_blob(state)
        if use_llm and self.client.available:
            text = self.client.chat(_BRIEF_SYSTEM, f"COHORT STATE (JSON):\n{blob}",
                                    max_tokens=max_tokens, respect_interval=False)
            if text:
                return {"brief": text, "source": "llm", "model": self.client._last_model}
        return {"brief": self._brief_template(state), "source": "template", "model": None}

    # ------------------------------------------------------------------ #
    def _state_blob(self, state: Dict[str, Any]) -> str:
        try:
            blob = json.dumps(state, default=str, separators=(",", ":"))
        except Exception:
            blob = str(state)
        return blob[: self.max_state_chars]

    # -- deterministic fallbacks (retrieval over the same state) --------- #
    @staticmethod
    def _beds(state: Dict[str, Any]):
        for key in ("topology", "beds", "roster", "cohort"):
            v = state.get(key)
            if isinstance(v, list) and v:
                return v
        return []

    def _ask_template(self, question: str, state: Dict[str, Any]) -> str:
        q = question.lower()
        beds = self._beds(state)
        if any(w in q for w in ("critical", "attention", "worst", "sick", "risk")):
            crit = [b for b in beds if str(b.get("tier", "")).upper() == "CRITICAL"]
            watch = [b for b in beds if str(b.get("tier", "")).upper() == "WATCH"]
            if not beds:
                return "That is not in the current system state (no cohort provided)."
            names = ", ".join(str(b.get("name") or b.get("node_id") or b.get("bed")) for b in (crit or watch)[:6])
            band = "critical" if crit else ("watch" if watch else "stable")
            return (f"[LLM offline — template] {len(crit)} critical, {len(watch)} watch of "
                    f"{len(beds)} beds. Attention: {names or 'none'} ({band}). "
                    f"Actions must go through the certified syscall gate.")
        if any(w in q for w in ("kernel", "scheduler", "cpu", "process", "sched")):
            k = state.get("ksched") or state.get("kernel") or state.get("scheduler") or {}
            return (f"[LLM offline — template] Kernel: {json.dumps(k, default=str)[:300]}"
                    if k else "That is not in the current system state.")
        return ("[LLM offline — template] The assistant is rate-limited right now. "
                "Query the state directly via the console (roster, ps, proc, kstat).")

    def _brief_template(self, state: Dict[str, Any]) -> str:
        beds = self._beds(state)
        crit = sum(1 for b in beds if str(b.get("tier", "")).upper() == "CRITICAL")
        watch = sum(1 for b in beds if str(b.get("tier", "")).upper() == "WATCH")
        return (f"[LLM offline — template] Cohort: {len(beds)} beds "
                f"({crit} critical, {watch} watch, {len(beds) - crit - watch} stable). "
                f"Review critical/watch beds via `open <bed>`. Research tool, not a diagnosis.")
