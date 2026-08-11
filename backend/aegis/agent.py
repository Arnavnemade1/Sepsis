"""The agent plane — how Dr. Aegis actually *operates* AegisOS.

Until now the consult could only talk about a dossier someone else assembled for
it. This is the control surface: a typed tool catalogue the model calls during a
conversation, so "let me pull up her causal graph" opens the causal graph, and
"what does the consensus chamber say" convenes the consensus chamber.

Three kinds of tool, and the distinction is the whole design:

  ``read``   Inspect anything. Every subsystem AegisOS exposes is reachable,
             including ones not written yet — :func:`os_read` is a generic GET
             over the OS's own REST surface, so the agent's reach grows with the
             system instead of being a hand-maintained list that rots. Named
             tools exist for the common paths so the model does not have to guess
             a URL to answer an ordinary question.

  ``ui``     Drive the cockpit. These do not execute here; they are forwarded to
             the browser over the consult socket, which runs them against the
             real shell modules (``AegisPatient.set``, ``AegisStory.open``, …).
             The agent operates the same OS the clinician does — not a shadow
             copy of it.

  ``write``  Mutate the running system. These go through
             :class:`~backend.aegis.syscalls.SyscallTable` — the enumerated,
             Z3-gated, append-only-audited trap gate — exactly as the console
             does. The agent may *invoke* every syscall in the table; the prover
             decides which ones take effect.

That last point is deliberate and it is not a limitation on what the agent can
reach. Routing writes around the gate would not make the agent more capable, it
would only make it unaccountable: the same order, minus the certificate, minus
the audit row, minus the reason. A denied write comes back with the prover's own
explanation, which the agent then has to say out loud — "I tried to escalate her
and the cage refused, because…". An assistant that can tell you why it was not
allowed to do something is worth more than one that quietly did it.

Handlers are injected by :mod:`backend.aegis.server` at import time via
:meth:`ToolRegistry.bind` (the same registration idiom as ``procfs.register``),
so this module stays free of a circular import back into the app.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

KIND_READ, KIND_UI, KIND_WRITE = "read", "ui", "write"

# The cockpit surfaces the agent can pull up. Names match the frontend modules;
# webui/doctor.js maps each to the real shell call.
PANELS = (
    "camera",        # optical biosensor (rPPG / perfusion / distress)
    "story",         # clinical story — temporal causal graph
    "consensus",     # BFT multi-specialist consensus chamber
    "orchestrator",  # autonomous ICU orchestrator + attestation
    "federation",    # federated intelligence
    "rbac",          # clinical role/permission shell
    "dsl",           # hospital safety-rule DSL
    "pharma",        # living pharmacopoeia
    "identity",      # physiologic identity guard
    "twin",          # treatment twin (PK distribution)
    "tactical",      # global ICU tactical sphere
    "consult",       # this call
)

ORGANS = ("heart", "lungs", "kidneys", "liver", "brain", "gut", "vasculature")


@dataclass
class Tool:
    name: str
    kind: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    handler: Optional[Callable[..., Any]] = None

    def declaration(self) -> Dict[str, Any]:
        """Gemini functionDeclaration form (shared by Live and generateContent)."""
        decl: Dict[str, Any] = {"name": self.name, "description": self.description}
        if self.properties:
            decl["parameters"] = {"type": "OBJECT", "properties": self.properties,
                                  "required": list(self.required)}
        return decl


def _s(desc: str, enum: Optional[tuple] = None) -> Dict[str, Any]:
    p: Dict[str, Any] = {"type": "STRING", "description": desc}
    if enum:
        p["enum"] = list(enum)
    return p


def _n(desc: str) -> Dict[str, Any]:
    return {"type": "NUMBER", "description": desc}


def _b(desc: str) -> Dict[str, Any]:
    return {"type": "BOOLEAN", "description": desc}


class ToolCallError(Exception):
    pass


class ToolRegistry:
    """The catalogue plus the dispatch table. One instance per process."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._log: List[Dict[str, Any]] = []
        self._seq = 0
        for t in _catalogue():
            self._tools[t.name] = t

    # -- wiring ----------------------------------------------------------- #
    def bind(self, name: str, handler: Callable[..., Any]) -> None:
        """Attach the implementation for a ``read``/``write`` tool."""
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"no such tool: {name}")
        tool.handler = handler

    def bound(self) -> List[str]:
        return sorted(n for n, t in self._tools.items()
                      if t.kind == KIND_UI or t.handler is not None)

    def unbound(self) -> List[str]:
        return sorted(n for n, t in self._tools.items()
                      if t.kind != KIND_UI and t.handler is None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    # -- what the model is told it can do --------------------------------- #
    def declarations(self, kinds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Only bound tools are advertised — a model must never be offered a
        capability that would fail the moment it reached for it."""
        return [t.declaration() for t in self._tools.values()
                if (kinds is None or t.kind in kinds)
                and (t.kind == KIND_UI or t.handler is not None)]

    def catalogue(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "kind": t.kind, "description": t.description,
                 "params": sorted(t.properties), "required": t.required,
                 "bound": t.kind == KIND_UI or t.handler is not None}
                for t in self._tools.values()]

    # -- dispatch --------------------------------------------------------- #
    def invoke(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a tool. Never raises — a failure is a result the agent must read
        and account for out loud, not an exception that silently ends its turn."""
        args = dict(args or {})
        self._seq += 1
        started = time.time()
        tool = self._tools.get(name)
        if tool is None:
            return self._record(name, args, started,
                                {"ok": False, "error": f"unknown tool {name!r}",
                                 "available": sorted(self._tools)[:40]}, "unknown")
        if tool.kind == KIND_UI:
            # Handed to the browser, which drives the shell — but a UI tool may
            # also carry a COMPANION READ.
            #
            # Two reasons. The plain one: "pull up her causal graph" means show it
            # AND read it, so a tool that only dispatched a command forced a
            # second round-trip to answer the question that was actually asked.
            # The careful one: a display command returning nothing but success is
            # an invitation to describe what was opened, and the gap between
            # "I opened it" and "I have seen it" is exactly where a language model
            # invents. Returning the contents closes the gap structurally instead
            # of relying on the model to respect a disclaimer. Panels with nothing
            # readable behind them (the camera is a live device) still get the
            # blunt note below.
            out: Dict[str, Any] = {"ok": True, "ui": {"action": name, "args": args},
                                   "displayed": True}
            if tool.handler is not None:
                try:
                    data = tool.handler(**args)
                except Exception as exc:
                    out["note"] = (f"Opened on their screen, but its data could not "
                                   f"be read ({type(exc).__name__}). You have NOT seen "
                                   "its contents — do not describe them.")
                    return self._record(name, args, started, out, KIND_UI)
                if data is not None:
                    out["data"] = data
                    out["note"] = ("Opened on their screen; 'data' is what it "
                                   "contains. Describe only what is in 'data'.")
                    return self._record(name, args, started, out, KIND_UI)
            out["note"] = ("This opened it on the CLINICIAN'S screen and returned NO "
                           "data to you. You have not seen its contents and must not "
                           "describe them.")
            return self._record(name, args, started, out, KIND_UI)
        if tool.handler is None:
            return self._record(name, args, started,
                                {"ok": False, "error": f"tool {name!r} is not available "
                                 "on this deployment"}, tool.kind)
        try:
            out = tool.handler(**args)
        except TypeError as exc:                     # wrong/missing arguments
            return self._record(name, args, started,
                                {"ok": False, "error": f"bad arguments: {exc}",
                                 "expects": sorted(tool.properties)}, tool.kind)
        except Exception as exc:
            return self._record(name, args, started,
                                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                                tool.kind)
        return self._record(name, args, started, {"ok": True, "result": out}, tool.kind)

    def _record(self, name: str, args: Dict[str, Any], started: float,
                out: Dict[str, Any], kind: str) -> Dict[str, Any]:
        entry = {"seq": self._seq, "tool": name, "kind": kind, "args": args,
                 "ok": bool(out.get("ok")), "ms": round((time.time() - started) * 1000, 1),
                 "ts": started}
        # A gated write carries the prover's verdict; surface it on the log line
        # so the activity feed can show refusals without digging into the payload.
        res = out.get("result")
        if isinstance(res, dict):
            for k in ("errno", "z3_status", "allowed", "message"):
                if k in res:
                    entry[k] = res[k]
        if not out.get("ok"):
            entry["error"] = out.get("error")
        self._log.append(entry)
        del self._log[:-200]
        return out

    def activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._log[-limit:]


# ---------------------------------------------------------------------- #
#  The catalogue.
# ---------------------------------------------------------------------- #
def _catalogue() -> List[Tool]:
    T: List[Tool] = []

    # ---- READ: the generic primitive ---------------------------------- #
    T.append(Tool(
        "os_read", KIND_READ,
        "Read ANY AegisOS state endpoint by path — the universal inspection "
        "primitive. Use it when no named tool fits: every subsystem the OS "
        "exposes is reachable this way. Paths look like '/aegis/orchestrator', "
        "'/aegis/proc/llm', '/aegis/devices', '/aegis/cohort/validation'. "
        "Call list_surfaces first if you do not know the path. Read-only.",
        {"path": _s("An /aegis/... GET path, e.g. '/aegis/kernel/devices'."),
         "params": _s("Optional query string, e.g. 'bed=bed_3&limit=10'.")},
        ["path"]))
    T.append(Tool(
        "list_surfaces", KIND_READ,
        "List every readable state path in this AegisOS build, with a one-line "
        "description each. Use before os_read when you need to find where "
        "something lives.", {}, []))

    # ---- READ: the paths worth naming --------------------------------- #
    T.append(Tool(
        "list_patients", KIND_READ,
        "The monitored roster: every bed with name, tier, risk and whether it is "
        "a live monitor or a replayed validation record.", {}, []))
    T.append(Tool(
        "get_patient", KIND_READ,
        "Full clinical state for one bed: vitals, calibrated risk and tier, "
        "SIRS/qSOFA, organ-system state, reasoning, advisory protocol, history.",
        {"bed": _s("Bed id, e.g. 'bed_3'. Omit for the currently selected patient.")}, []))
    T.append(Tool(
        "get_case_dossier", KIND_READ,
        "The grounded consult dossier for a bed — the same briefing that grounds "
        "this call, including what the system CANNOT see and the detector's real "
        "external-validation performance. Use when switching to a different bed.",
        {"bed": _s("Bed id. Omit for the current patient.")}, []))
    T.append(Tool(
        "get_vitals_trend", KIND_READ,
        "The recorded time series for a bed — how the vitals actually moved, not "
        "just where they are now. Use before claiming something is 'rising'.",
        {"bed": _s("Bed id.")}, ["bed"]))
    T.append(Tool(
        "get_causal_graph", KIND_READ,
        "The tamper-evident temporal causal graph for a bed: what happened, in "
        "what order, and which links are Z3-proven versus merely associated.",
        {"bed": _s("Bed id. Omit for the current patient.")}, []))
    T.append(Tool(
        "trace_incident", KIND_READ,
        "Trace backwards from one causal-graph node to the chain that produced "
        "it — an incident report for a single event.",
        {"node_id": _s("Node id from get_causal_graph."),
         "bed": _s("Bed id. Omit for the current patient.")}, ["node_id"]))
    T.append(Tool(
        "run_consensus", KIND_READ,
        "Convene the Byzantine-fault-tolerant consensus chamber: independent "
        "specialist models vote on deterioration and return a quorum certificate. "
        "Use when you want a second opinion that is not your own.",
        {"bed": _s("Bed id. Omit for the current patient.")}, []))
    T.append(Tool(
        "run_whatif", KIND_READ,
        "Project the treatment twin forward for a bed: what the modelled "
        "trajectory looks like if the advisory protocol is followed versus not. "
        "A projection, not a recommendation.",
        {"bed": _s("Bed id. Omit for the current patient.")}, []))
    T.append(Tool(
        "project_treatment", KIND_READ,
        "Physiologically-based pharmacokinetic projection for a bed: where a drug "
        "distributes over time and the exposure-driven adverse-event risk.",
        {"bed": _s("Bed id."),
         "adverse": _b("Include adverse-event projection (default true).")}, ["bed"]))
    T.append(Tool(
        "screen_drugs", KIND_READ,
        "Screen a drug list against the living pharmacopoeia: interaction graph, "
        "CYP450, QT prolongation and renal hazards, each returned as a Z3 "
        "constraint with a proof tree.",
        {"drugs": {"type": "ARRAY", "items": {"type": "STRING"},
                   "description": "Drug names, e.g. ['vancomycin','piperacillin']."},
         "bed": _s("Bed id, so renal/hepatic state is taken into account.")}, ["drugs"]))
    T.append(Tool(
        "get_validation", KIND_READ,
        "The detector's REAL measured performance against ground-truth labels — "
        "AUROC, detection rate, lead time, false-alarm rate. Quote this whenever "
        "you are asked how much the system can be trusted.", {}, []))
    T.append(Tool(
        "get_orchestrator", KIND_READ,
        "The autonomous orchestrator's governance state and attestation ledger: "
        "what it has certified, under the non-maleficence invariant.", {}, []))
    T.append(Tool(
        "get_kernel", KIND_READ,
        "Kernel state: the cognitive scheduler's process table, priorities and "
        "top summary — which patients the OS is spending compute on and why.", {}, []))
    T.append(Tool(
        "get_audit", KIND_READ,
        "The append-only syscall audit log: every write anyone has asked the OS "
        "for, allowed or denied, with the prover's verdict on each.",
        {"limit": _n("How many entries (default 24).")}, []))
    T.append(Tool(
        "get_activity", KIND_READ,
        "What you yourself have done this session — your own tool calls and their "
        "outcomes. Use if asked what you just did.",
        {"limit": _n("How many entries (default 20).")}, []))

    # ---- UI: drive the cockpit ---------------------------------------- #
    T.append(Tool(
        "open_panel", KIND_UI,
        "Pull up a subsystem on the clinician's screen AND read what it contains "
        "— the result's 'data' field is the panel's actual contents, so use this "
        "when they ask you to look at something with them. Do it whenever you "
        "reference something they should be seeing; say what you are opening as "
        "you open it. Panels: " + ", ".join(PANELS) + ".",
        {"panel": _s("Which surface to open.", PANELS),
         "bed": _s("Bed id to scope it to. Omit for the current patient.")},
        ["panel"]))
    T.append(Tool(
        "close_panel", KIND_UI, "Close a subsystem panel.",
        {"panel": _s("Which surface to close.", PANELS)}, ["panel"]))
    T.append(Tool(
        "select_patient", KIND_UI,
        "Switch the whole cockpit to another bed — every panel re-scopes to them. "
        "Do this before discussing a different patient so the screen matches the "
        "conversation. IMPORTANT: the case dossier you were given at the start of "
        "this call describes the ORIGINAL patient and does NOT update. This tool "
        "returns the new patient's dossier in 'data' — from that point on, that "
        "is your grounding, and you must not carry the previous patient's numbers, "
        "name or history across.",
        {"bed": _s("Bed id from list_patients.")}, ["bed"]))
    T.append(Tool(
        "enable_camera", KIND_UI,
        "Turn on the optical biosensor (contactless rPPG, perfusion, distress "
        "signs). Needs the clinician to grant camera access.", {}, []))
    T.append(Tool(
        "focus_organ", KIND_UI,
        "Focus and highlight one organ system on the 3D body twin, so they are "
        "looking at the thing you are describing.",
        {"organ": _s("Which organ system.", ORGANS)}, ["organ"]))
    T.append(Tool(
        "highlight", KIND_UI,
        "Flash a short caption over the cockpit to point at what matters right "
        "now. Use sparingly — for the one thing, not a running commentary.",
        {"text": _s("Under ~90 characters.")}, ["text"]))

    # ---- WRITE: through the trap gate --------------------------------- #
    T.append(Tool(
        "syscall", KIND_WRITE,
        "Ask the OS to actually DO something: 'treat' orders the certified Hour-1 "
        "bundle, 'escalate' raises a bed's scheduler priority, 'crisis' runs a "
        "drill, 'clear' recovers a bed. Every call goes through the Z3 egress cage "
        "and is audited. The prover may REFUSE — a refusal returns EPERM with the "
        "reason, and nothing changes. If that happens, tell the clinician plainly "
        "that it was refused and what the certificate said; do not retry it. "
        "Never call this without the clinician asking for it in this conversation.",
        {"syscall": _s("Which write.", ("treat", "escalate", "crisis", "clear")),
         "bed": _s("Bed id the write applies to."),
         "args": {"type": "OBJECT", "description": "Optional syscall arguments."}},
        ["syscall", "bed"]))
    T.append(Tool(
        "set_tracking", KIND_WRITE,
        "Change what AegisOS watches for a patient — add a focus (an organ system "
        "or metric to weight) or a watch rule. Observation-only: it changes what "
        "is monitored, never what is administered.",
        {"bed": _s("Bed id."),
         "command": _s("A tracking command, e.g. 'focus respiratory' or "
                       "'watch lactate > 3'.")}, ["bed", "command"]))
    T.append(Tool(
        "service_control", KIND_WRITE,
        "Start, stop or restart an aegisd service (resonance, radar, triage).",
        {"name": _s("Service name from os_read('/aegis/services')."),
         "action": _s("What to do.", ("start", "stop", "restart"))}, ["name", "action"]))
    T.append(Tool(
        "simulate_stage", KIND_WRITE,
        "Drive the DEMO patient simulator to a clinical stage. This is the "
        "demonstration simulator only — it never touches a recorded cohort bed. "
        "Say that you are simulating when you use it.",
        {"stage": _s("Stage.", ("STABLE", "DETERIORATING", "SEPSIS", "RECOVERING"))},
        ["stage"]))
    return T


# The process-wide registry. server.py binds handlers onto it at import.
registry = ToolRegistry()


def summarize(result: Dict[str, Any], max_chars: int = 3500) -> Dict[str, Any]:
    """Bound a tool result before it goes back to the model.

    A raw AegisOS payload can be hundreds of kilobytes (a causal graph, a full
    snapshot). Handing that to a realtime voice session wastes the context the
    conversation needs and slows the turn. Oversized results are truncated with
    an explicit marker so the model knows it is seeing a prefix and can narrow
    its query rather than confidently summarising a payload it only half saw."""
    try:
        blob = json.dumps(result, default=str)
    except Exception:
        return {"ok": False, "error": "result was not serializable"}
    if len(blob) <= max_chars:
        return result
    return {"ok": result.get("ok", True), "truncated": True,
            "note": f"result was {len(blob)} chars, truncated to {max_chars}. "
                    "Ask for a narrower slice if you need the rest.",
            "prefix": blob[:max_chars]}
