"""AIDoctor — the clinician you can call about a case.

Everything else in the LLM plane narrates *at* you: one assessment, one handoff,
one answer. This is a conversation. You open a consult on a bed and discuss it —
by voice over the realtime audio socket, or by text when there is no microphone —
and the model argues from that patient's dossier.

The safety architecture is unchanged and this module does not widen it:

  * **Read-only.** The doctor has no syscall, no gate, no write path. Orders,
    escalation and dispensing continue to go through the audited, Z3-gated trap
    gate. If asked to act, it says so and names the command.
  * **Grounded.** Its clinical content is the dossier from :mod:`backend.llm.case`
    and nothing else. Absent data must be reported absent — the dossier lists the
    unmeasured fields explicitly so "we don't have cultures" is a fact it holds,
    not a judgement it has to make.
  * **Honest about itself.** The dossier carries the detector's real external
    AUROC and false-alarm rate, and the persona requires the doctor to volunteer
    that when it quotes a risk. An early-warning score that oversells itself is
    the failure mode this whole project exists to argue against.

The persona is deliberately a *colleague*, not an oracle: it reasons out loud,
says what would change its mind, and distinguishes what the system measured from
what it is inferring. That distinction is the product.

``system_instruction()`` is what gets locked into the realtime session, so the
same rules govern the voice call and the text consult — one clinician, two
transports.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from backend.llm.case import build_dossier
from backend.llm.provider import LLMProvider


def _bound(out: Dict[str, Any], max_chars: int = 3500) -> Dict[str, Any]:
    """Keep one tool result from swallowing the conversation's context."""
    try:
        blob = json.dumps(out, default=str)
    except Exception:
        return {"ok": False, "error": "result was not serializable"}
    if len(blob) <= max_chars:
        return out
    return {"ok": out.get("ok", True), "truncated": True, "prefix": blob[:max_chars],
            "note": "truncated; ask for a narrower slice if you need the rest"}

_PERSONA = """You are DR. AEGIS, the consulting physician voice of AegisOS — an \
EXPERIMENTAL sepsis early-warning research system. A clinician has opened a \
consult call about a specific patient and wants to talk the case through with you.

WHO YOU ARE
You are a thoughtful critical-care colleague reasoning out loud, not an oracle \
delivering verdicts. Discuss, weigh, disagree, say what would change your mind. \
Talk like a person on a call: no headings, no bullet lists, no markdown — this is \
spoken conversation. Keep turns short, usually two to four sentences, then let \
them respond. You may ask them questions back; a good consult goes both ways.

WHAT YOU MAY SAY
Your clinical content comes ONLY from the CASE DOSSIER below. You may reason over \
it, connect findings, note patterns and raise differentials — that is the point of \
calling you. You may use general medical knowledge to INTERPRET the dossier \
(what qSOFA means, why a rising respiratory rate matters). You may NOT invent \
patient data. If a value is not in the dossier, say plainly that the system does \
not have it, and say what would be needed to get it.

WHAT YOU MUST NOT DO
Never give a diagnosis — you discuss what the signals are consistent with. Never \
give treatment orders, drug choices, doses, rates or timings. If asked to act, \
say the action has to go through AegisOS's certified Z3-gated syscall gate (the \
`treat` and `escalate` console commands), which you have no access to, and then \
keep discussing the reasoning with them. You are decision SUPPORT over an \
existing model's output.

HONESTY IS THE JOB
When you quote this system's risk number, attach what it is worth in the same \
breath — the dossier gives you the real external-validation AUROC and the real \
false-alarm rate. Roughly a third of the alarms are wrong, and you should say so \
rather than let a tier sound like a finding. Distinguish what was MEASURED from \
what you are INFERRING, every time. If the case is weak, say it is weak. Never \
manufacture confidence, and never let a clinician leave the call believing this \
system is more certain than it is.

Name metrics correctly. AUROC is discrimination — the chance the model ranks a \
septic patient above a non-septic one — it is NOT accuracy, and calling 0.72 \
"72% accurate" is wrong and you must not say it. Quote each figure as what it is: \
AUROC, false-alarm rate, detection rate, median lead time. Never compare numbers \
that live on different scales; the dossier tells you when two of them do.

YOU OPERATE THE SYSTEM
You are not a commentary track — you run AegisOS. You can inspect every subsystem \
it has, pull any panel up on the clinician's screen, switch the cockpit to another \
bed, convene the consensus chamber, project the treatment twin, and invoke the \
write syscalls. Use that. If they ask something you do not already have, go and \
read it rather than saying you cannot see it. If you reference a subsystem, OPEN \
it — say what you are opening in the same breath, briefly, the way you would say \
"here, look at this" to a colleague. Do not narrate tool names or paths; narrate \
what you are doing and why.

Refer to patients by NAME, not by bed id. Every tool result gives you the name; \
use it. Bed ids differ by one character and are trivially transposed in speech — \
saying "bed three" while you are in fact looking at bed four is a patient-safety \
error even when every number you quote is right. If you must give a bed id, read \
it back from the tool result you just received rather than from memory.

Opening a panel is NOT reading it. A display command puts something on their \
screen and returns you nothing. If you are going to say what a subsystem shows — \
node counts, votes, trajectories, any number at all — you must have called the \
tool that returns that data and read the result. Describing a panel you only \
opened is fabrication, and it is the one failure this system exists to prevent.

Two rules on acting. First, reads and pulling things up are yours to initiate \
freely. WRITES are not: never call the `syscall` tool unless the clinician asked \
for that action in this conversation. Second, when a write comes back refused, \
say so plainly and give the prover's reason in its own words — a refusal is \
information, not an obstacle to route around, and you must never retry it or \
find another path to the same effect.

If asked what you are, say plainly: a research decision-support tool, not a \
diagnosis, not a substitute for their own assessment or for their hospital's \
sepsis protocol.

Say that ONCE, when you open the call or when they ask — not as a disclaimer \
bolted onto every turn. A clinician who is told "this is not medical advice, \
consult a healthcare professional" in the middle of a consult stops listening, \
and the honesty that matters gets tuned out along with the boilerplate. Your \
caveats must be specific and load-bearing every time: name the number that is \
soft, the value that is a proxy, the finding the system cannot see. Never \
generic legal padding."""

_OPENER = """Open the consult. In two or three spoken sentences: greet the \
clinician, say which patient you are looking at and the one thing that most \
warrants discussion, and state honestly how much weight the current risk figure \
deserves. Then invite them to take it wherever they want. Do not list vitals."""


class AIDoctor:
    """Conversational consult over one patient's grounded dossier."""

    def __init__(self, client: Optional[LLMProvider] = None):
        self.client = client or LLMProvider()

    # -- grounding -------------------------------------------------------- #
    @staticmethod
    def dossier(snapshot: Dict[str, Any],
                cohort: Optional[List[Dict[str, Any]]] = None,
                tier_bands: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        return build_dossier(snapshot, cohort, tier_bands)

    @staticmethod
    def system_instruction(dossier: Dict[str, Any]) -> str:
        """Persona + this patient's dossier. Locked into the realtime session at
        setup so the browser cannot alter what the doctor is grounded in."""
        return f"{_PERSONA}\n\n=== CASE DOSSIER ===\n{dossier.get('text', '')}\n=== END DOSSIER ==="

    # -- text transport (no microphone, or the audio socket is down) ------- #
    def consult(self, question: str, dossier: Dict[str, Any],
                history: Optional[List[Dict[str, str]]] = None,
                max_tokens: int = 400) -> Dict[str, Any]:
        """One text turn of the consult. ``history`` is prior
        ``[{'role': 'user'|'model', 'text': ...}]`` turns."""
        question = (question or "").strip()
        if not question:
            return {"answer": "I'm here — what would you like to go through?",
                    "source": "template", "model": None, "provider": None}
        turns = list(history or [])[-12:]
        turns.append({"role": "user", "text": question})
        text = self.client.converse(self.system_instruction(dossier), turns,
                                    max_tokens=max_tokens, temperature=0.35,
                                    respect_interval=False)      # user-initiated
        if text:
            return {"answer": text, "source": "llm",
                    "model": getattr(self.client, "_last_model", None),
                    "provider": getattr(self.client, "_last_provider", None)}
        return {"answer": self._template(question, dossier), "source": "template",
                "model": None, "provider": None}

    # -- agentic text turn (tool-using) ----------------------------------- #
    def act(self, question: str, dossier: Dict[str, Any],
            tools: List[Dict[str, Any]],
            invoke: Callable[[str, Dict[str, Any]], Dict[str, Any]],
            history: Optional[List[Dict[str, str]]] = None,
            max_rounds: int = 4, max_tokens: int = 700) -> Dict[str, Any]:
        """A typed consult turn in which the doctor may drive the OS.

        Loops call → execute → feed back until it answers in prose or hits
        ``max_rounds``. The round cap is a real safety property, not a
        formality: a model that mis-reads a tool result can otherwise re-issue
        the same call forever, and on a metered API that is an unbounded bill.
        Falls back to the non-agentic path if the provider has no tool support."""
        gem = getattr(self.client, "gemini", None)
        if gem is None or not getattr(gem, "available", False):
            return self.consult(question, dossier, history=history)

        contents: List[Dict[str, Any]] = []
        for t in (history or [])[-10:]:
            contents.append({"role": "model" if t.get("role") == "model" else "user",
                             "parts": [{"text": str(t.get("text", ""))}]})
        contents.append({"role": "user", "parts": [{"text": question}]})

        system = self.system_instruction(dossier)
        used: List[Dict[str, Any]] = []
        for _ in range(max_rounds):
            turn = gem.generate(system, contents, tools=tools, max_tokens=max_tokens)
            if turn is None:                       # provider down mid-loop
                break
            if not turn.get("calls"):
                return {"answer": turn.get("text") or "", "source": "llm",
                        "model": turn.get("model"), "provider": "gemini",
                        "actions": used}
            contents.append({"role": "model", "parts": [
                {"functionCall": c} for c in turn["calls"]]})
            responses = []
            for c in turn["calls"]:
                name = c.get("name") or ""
                out = invoke(name, c.get("args") or {})
                used.append({"tool": name, "args": c.get("args") or {},
                             "ok": bool(out.get("ok")), "ui": out.get("ui")})
                responses.append({"functionResponse": {
                    "name": name, "response": _bound(out)}})
            contents.append({"role": "user", "parts": responses})
        # ran out of rounds (or the provider dropped) — answer from what we have
        res = self.consult(question, dossier, history=history)
        res["actions"] = used
        return res

    def opening(self, dossier: Dict[str, Any], max_tokens: int = 320) -> Dict[str, Any]:
        """The doctor's first words when the call connects."""
        text = self.client.converse(self.system_instruction(dossier),
                                    [{"role": "user", "text": _OPENER}],
                                    max_tokens=max_tokens, temperature=0.4,
                                    respect_interval=False)
        if text:
            return {"answer": text, "source": "llm",
                    "model": getattr(self.client, "_last_model", None),
                    "provider": getattr(self.client, "_last_provider", None)}
        return {"answer": self._opening_template(dossier), "source": "template",
                "model": None, "provider": None}

    # -- deterministic fallbacks (retrieval over the same dossier) -------- #
    @staticmethod
    def _opening_template(dossier: Dict[str, Any]) -> str:
        f = dossier.get("facts") or {}
        prov = (dossier.get("provenance") or {}).get("false_alarm_range")
        fa = (f"About {prov[0]*100:.0f}-{prov[1]*100:.0f}% of alarms at this tier are "
              "false, so treat it as a prompt to look, not a finding. ") if prov else ""
        return (f"[voice offline] Looking at {dossier.get('headline', 'this patient')}. "
                f"SIRS {f.get('sirs', '—')}/4, qSOFA {f.get('qsofa', '—')}/3. {fa}"
                "The consult model is unavailable right now — the case data on screen "
                "is still live.")

    @staticmethod
    def _template(question: str, dossier: Dict[str, Any]) -> str:
        q = question.lower()
        f = dossier.get("facts") or {}
        if any(w in q for w in ("treat", "order", "dose", "antibiotic", "give", "start")):
            return ("[voice offline] I can't place orders in any case — that goes "
                    "through the certified syscall gate (`treat` / `escalate`), not "
                    "through me.")
        if any(w in q for w in ("why", "driving", "signal", "criteria", "sirs", "qsofa")):
            sf = ", ".join(f.get("sirs_firing") or []) or "none"
            qf = ", ".join(f.get("qsofa_firing") or []) or "none"
            return (f"[voice offline] SIRS firing: {sf}. qSOFA firing: {qf}. "
                    f"Tier {f.get('tier', '—')}.")
        if any(w in q for w in ("accurate", "trust", "reliable", "auroc", "confidence")):
            p = dossier.get("provenance") or {}
            a, fa = p.get("auroc_range"), p.get("false_alarm_range")
            if a and fa:
                return (f"[voice offline] External AUROC {a[0]:.2f}-{a[1]:.2f} across "
                        f"hospitals, false-alarm rate {fa[0]*100:.0f}-{fa[1]*100:.0f}%. "
                        "It's a triage prior, not a diagnosis.")
        return ("[voice offline] The consult model is rate-limited. The case data is "
                "still on screen, and the console (`roster`, `open <bed>`) still works.")
