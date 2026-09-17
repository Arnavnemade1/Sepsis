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

  ``write``  Mutate the running system. CLINICAL orders go through
             :class:`~backend.aegis.syscalls.SyscallTable` — the enumerated,
             Z3-gated, append-only-audited trap gate — exactly as the console
             does. The agent may *invoke* every syscall in the table; the prover
             decides which ones take effect.

             Everything else does NOT, and pretending otherwise would be the
             wrong kind of comfortable: ``os_write`` resolves any ``/aegis`` POST
             route and calls its endpoint function directly, so chaos injection,
             package installs, service control, driver faults, ML training and
             the rest reach the system with no certificate, no capability check
             and no audit row. Four syscalls are gated; roughly forty other
             actuation routes are not. One reference monitor that every caller
             crosses — human, console or agent — is what closes that gap.

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
    # How much of a result is worth sending back. Most payloads are reference
    # data where a prefix is fine, but a few tools ARE their payload — the
    # treatment options are six arguments with six verdicts, and truncating them
    # to a prefix leaves the model describing the first two options as if they
    # were all of them. Those get more room; the default stays tight.
    budget: Optional[int] = None

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


def _assert_unique(tools: List[Tool]) -> List[Tool]:
    """Refuse a catalogue with two tools of the same name.

    ``bind`` is keyed by name and ``get`` returns the first match, so a duplicate
    does not raise — it silently makes one of the two dead while both keep appearing
    in the declarations the model sees. It reaches for the shadowed one, gets the
    other one's behaviour, and nothing anywhere reports a problem. That already
    happened once: a scenario-lab tool was named `run_scenario`, which was taken by
    the orchestrator's wing drill.
    """
    seen: Dict[str, int] = {}
    for t in tools:
        seen[t.name] = seen.get(t.name, 0) + 1
    dupes = sorted(n for n, c in seen.items() if c > 1)
    if dupes:
        raise ValueError(
            f"duplicate tool name(s) in the agent catalogue: {dupes}. Two tools with "
            "one name silently shadow each other — rename one.")
    return tools


class ToolRegistry:
    """The catalogue plus the dispatch table. One instance per process."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._log: List[Dict[str, Any]] = []
        self._seq = 0
        # _assert_unique, not a dict comprehension: building the dict silently
        # keeps the last of any duplicate pair, which is precisely the failure we
        # want to be loud about.
        for t in _assert_unique(_catalogue()):
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
        if kind in (KIND_WRITE, KIND_UI) or not entry["ok"]:
            from backend.aegis.kernel.journal import KJOURNAL, INFO, WARN
            KJOURNAL.append("agent", name, severity=INFO if entry["ok"] else WARN,
                            uid="dr-aegis", kind=kind, ok=entry["ok"], ms=entry["ms"],
                            **{k: entry[k] for k in ("errno", "z3_status", "allowed", "error")
                               if k in entry})
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
    T.append(Tool(
        "list_actions", KIND_READ,
        "List every action this OS can be made to take — each with the body it "
        "expects. Use before os_write when you need to find how to drive "
        "something.", {}, []))

    # ---- WRITE: the generic primitive --------------------------------- #
    T.append(Tool(
        "os_write", KIND_WRITE,
        "Drive ANY AegisOS actuation endpoint by path — the universal DO, for "
        "anything the named tools do not cover. Reaches the ML control plane, "
        "the package manager, chaos injection, device pairing, record import, "
        "proof certification and the rest. Call list_actions to find the path "
        "and the body it wants. Anything that would reach a patient still passes "
        "the Z3 gate and can come back refused.",
        {"path": _s("An /aegis/... POST path, e.g. '/aegis/ml/train'."),
         "body": {"type": "OBJECT", "description": "JSON body the endpoint expects."},
         "params": _s("Optional query string.")},
        ["path"]))

    # ---- READ: the paths worth naming --------------------------------- #
    T.append(Tool(
        "list_patients", KIND_READ,
        "The monitored roster, sorted sickest first: every patient with their "
        "NAME, tier, live risk and whether they are a live monitor or a replayed "
        "validation record. Each carries a 'bed_handle' for passing to other "
        "tools — that is an internal identifier, not something to say out loud.",
        {}, []))
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
        {"bed": _s("bed_handle from list_patients. Omit for the current patient.")}, []))
    T.append(Tool(
        "get_patient_history", KIND_READ,
        "The patient's PRIOR RECORD — comorbidities, regular medications, "
        "allergies, care priorities, the imported summary, and how much their "
        "history raises their alerting prior. This is history, not today's "
        "measurements. Reach for it whenever context before this admission would "
        "change your reading: a soft pressure in a dialysis patient, a proposed "
        "antibiotic against a listed allergy, a fever in someone immunosuppressed. "
        "If nothing is on file it says so — offer to import or link one.",
        {"bed": _s("bed_handle from list_patients. Omit for the current patient.")}, []))
    T.append(Tool(
        "list_record_sources", KIND_READ,
        "Where a patient's prior chart can be imported or linked from — manual "
        "paste, a live FHIR endpoint, or a hospital stub. Use when they ask to "
        "pull someone's history in and you need to say what is available.", {}, []))
    T.append(Tool(
        "get_vitals_trend", KIND_READ,
        "The recorded time series for a bed — how the vitals actually moved, not "
        "just where they are now. Use before claiming something is 'rising'.",
        {"bed": _s("bed_handle from list_patients.")}, ["bed"]))
    T.append(Tool(
        "get_causal_graph", KIND_READ,
        "The tamper-evident temporal causal graph for a bed: what happened, in "
        "what order, and which links are Z3-proven versus merely associated.",
        {"bed": _s("bed_handle from list_patients. Omit for the current patient.")}, []))
    T.append(Tool(
        "trace_incident", KIND_READ,
        "Trace backwards from one causal-graph node to the chain that produced "
        "it — an incident report for a single event.",
        {"node_id": _s("Node id from get_causal_graph."),
         "bed": _s("bed_handle from list_patients. Omit for the current patient.")}, ["node_id"]))
    T.append(Tool(
        "run_consensus", KIND_READ,
        "Convene the Byzantine-fault-tolerant consensus chamber: independent "
        "specialist models vote on deterioration and return a quorum certificate. "
        "Use when you want a second opinion that is not your own.",
        {"bed": _s("bed_handle from list_patients. Omit for the current patient.")}, []))
    T.append(Tool(
        "run_whatif", KIND_READ,
        "Project the treatment twin forward for a bed: what the modelled "
        "trajectory looks like if the advisory protocol is followed versus not. "
        "A projection, not a recommendation.",
        {"bed": _s("bed_handle from list_patients. Omit for the current patient.")}, []))
    T.append(Tool(
        "project_treatment", KIND_READ,
        "Physiologically-based pharmacokinetic projection for a bed: where a drug "
        "distributes over time and the exposure-driven adverse-event risk.",
        {"bed": _s("bed_handle from list_patients."),
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
        "hyperscan", KIND_UI,
        "Run a HYPERSCAN: sweep every subsystem for this patient and produce a "
        "full situation report — findings with their sources, the candidate "
        "treatment paths with the cage's verdicts, a recommendation, and what the "
        "system cannot see. Opens the scan on their screen and returns the report "
        "to you. Use when they ask for the whole picture, a workup, or 'what is "
        "going on with this patient'. It takes several seconds; say you are "
        "running it. Always tell them the report needs their own review.",
        {"bed": _s("bed_handle from list_patients. Omit for the current patient.")}, [],
        budget=9000))
    T.append(Tool(
        "show_treatment_options", KIND_UI,
        "Pull up the candidate treatment paths for a patient AND read them — "
        "every option the OS considered, what each one buys, what it costs, its "
        "projected adverse-event risk, and crucially WHICH ONES THE Z3 CAGE "
        "REFUSED and in the prover's own words. Use this whenever they ask what "
        "the choices are, what you would consider, or why something is not on the "
        "table. Do NOT read the whole list aloud — six options recited in order is "
        "a wall of text on a voice call. Lead with the recommended path and why, "
        "give the ONE trade-off that actually differs between the realistic "
        "choices, and name any refusal in the cage's own words. Let them ask "
        "about the rest. You are explaining options that already exist — you are "
        "still not ordering them.",
        {"bed": _s("bed_handle from list_patients. Omit for the current patient.")}, [],
        budget=8000))
    # ---- SCENARIO LAB: pose a situation, then be graded on it ---------- #
    T.append(Tool(
        "scenario_vocabulary", KIND_READ,
        "What can be built in the scenario sandbox: the deterioration arcs "
        "(septic shock, cold/hypothermic sepsis, occult, stable, recovering, "
        "post-op fever), the gradeable expectations, the interventions and the "
        "limits. Read this BEFORE create_scenario — the arc and expectation names "
        "are a fixed vocabulary and an invented one is rejected.", {}, []))
    T.append(Tool(
        "list_scenarios", KIND_READ,
        "Every scenario in the sandbox with its pass/fail state, plus the suite "
        "report. Sandbox patients are synthetic and 'sim_'-prefixed; they are not "
        "on the ward and must never be discussed as real people.", {}, []))
    T.append(Tool(
        "get_scenario", KIND_READ,
        "One scenario's full result: every graded expectation with what was "
        "expected and what happened, plus the hour-by-hour risk trace.",
        {"bed": _s("Sandbox handle, e.g. 'sim_2'.")}, ["bed"]))
    T.append(Tool(
        "create_scenario", KIND_WRITE,
        "Author a synthetic patient to test the OS with: choose a deterioration "
        "arc, when it starts, how severe it is, how sparsely labs are drawn, and "
        "what SHOULD happen. Use it to probe a suspicion — a presentation you "
        "think might be missed, or a confounder you think might trigger a false "
        "alarm. This creates a hand-written patient in a sandbox: it tests what "
        "the system DOES, and can never measure how accurate it is.",
        {"name": _s("Short label, e.g. 'hypothermic-elderly'."),
         "arc": _s("Deterioration arc from scenario_vocabulary."),
         "los_hours": _n("Stay length in hours (8-336)."),
         "onset_hour": _n("Clinical onset hour, or omit for a patient who never "
                          "becomes septic. Must exceed 6."),
         "severity": _n("0.4 subtle, 1.0 textbook, 1.6 fulminant."),
         "ramp_hours": _n("Hours from first drift to worst."),
         "age": _n("Patient age."),
         "lab_interval_hours": _n("Hours between lab draws; sparser is harder."),
         "seed": _n("Seed — the same spec and seed always gives the same patient."),
         "expectations": {"type": "ARRAY", "description":
                          "What SHOULD happen: [{kind, value, because}]. Kinds come "
                          "from scenario_vocabulary. 'because' is required reading "
                          "on failure, so state the clinical reason.",
                          "items": {"type": "OBJECT"}},
         "notes": _s("Why this case is worth running.")},
        ["name", "arc"]))
    T.append(Tool(
        "run_lab_scenario", KIND_WRITE,
        "Run one SANDBOX scenario against the shipped detector and grade it. Not "
        "to be confused with run_scenario, which drills the orchestrator on a real "
        "wing bed.",
        {"bed": _s("Sandbox handle from create_scenario, e.g. 'sim_0'.")}, ["bed"]))
    T.append(Tool(
        "run_scenario_suite", KIND_WRITE,
        "Install and run the standing suite — the deteriorations the OS must catch "
        "and the confounders it must ignore — and return the pass/fail report. "
        "Replaces whatever is in the sandbox. One case is a KNOWN failure and is "
        "meant to stay failing; read its note before treating it as a regression.",
        {}, []))
    T.append(Tool(
        "reset_scenarios", KIND_WRITE,
        "Empty the scenario sandbox.", {}, []))
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
        {"bed": _s("The bed_handle from list_patients (internal id — do not say it aloud).")}, ["bed"]))
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
    T.append(Tool(
        "play_treatment_twin", KIND_UI,
        "Drive the 3D treatment-twin playback so they can watch a drug "
        "distribute: play, pause, scrub to a time, change speed, or switch which "
        "candidate order is being shown.",
        {"command": _s("What to do.", ("play", "pause", "restart")),
         "arm": _s("Which candidate order to show."),
         "at_hour": _n("Scrub to this hour."),
         "rate": _n("Playback speed, e.g. 0.5, 1, 3.")}, []))
    # NOTE: there is deliberately no `run_console` tool. The OS console lives on
    # tactical.html, not on the page the consult runs in, so a tool that "typed
    # into the console" would have had nothing to type into — and its commands
    # (ps / top / kstat / roster) are already covered by get_kernel and
    # list_patients. A tool that silently does nothing is worse than no tool.
    T.append(Tool(
        "set_camera_sensitivity", KIND_UI,
        "Retune the optical biosensor's detection sensitivity.",
        {"mode": _s("Sensitivity mode.", ("low", "balanced", "high"))}, ["mode"]))
    T.append(Tool(
        "note", KIND_UI,
        "Write an observation into the patient's Observations pane — a note the "
        "clinician keeps, not an order.",
        {"text": _s("The observation.")}, ["text"]))

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
        {"bed": _s("bed_handle from list_patients."),
         "command": _s("A tracking command, e.g. 'focus respiratory' or "
                       "'watch lactate > 3'.")}, ["bed", "command"]))
    T.append(Tool(
        "service_control", KIND_WRITE,
        "Start, stop or restart an aegisd service (resonance, radar, triage).",
        {"name": _s("Service name from os_read('/aegis/services')."),
         "action": _s("What to do.", ("start", "stop", "restart"))}, ["name", "action"]))
    # ---- WRITE: named actuation, so the model reaches for the right thing --- #
    T.append(Tool(
        "counterfactual_replay", KIND_WRITE,
        "Re-run a patient's recorded course as if a different action had been "
        "taken at a branch point — the 'what if we had caught this sooner' "
        "experiment. Get an outcome_id or node_id from get_causal_graph first.",
        {"bed": _s("bed_handle from list_patients."),
         "outcome_id": _s("Replay at the earliest actionable node upstream of this outcome."),
         "node_id": _s("…or replay at this specific branch point."),
         "treatment": {"type": "OBJECT",
                       "description": "Alternative action, e.g. "
                                      "{'fluid_rate':5,'vasopressor_dose':0,'antibiotic_code':3}."},
         "horizon": _n("Hours to project forward.")}, []))
    T.append(Tool(
        "govern", KIND_WRITE,
        "Put a decision through the autonomous orchestrator's non-maleficence "
        "invariant: it composes the layers and returns either a signed "
        "certificate or a refusal. Use to show whether an action would be "
        "permitted before anyone takes it.",
        {"action": _s("What to govern.", ("treat", "escalate", "pin", "actuator")),
         "target": _s("Bed id the decision applies to."),
         "layer": _s("Which layer, default 'bed'."),
         "payload": {"type": "OBJECT", "description": "Action parameters."}},
        ["action"]))
    T.append(Tool(
        "run_scenario", KIND_WRITE,
        "Run the orchestrator's end-to-end scenario for a bed — the full cascade "
        "from signal to certified decision.",
        {"bed": _s("bed_handle from list_patients. Omit for the current patient.")}, []))
    T.append(Tool(
        "certify_proof", KIND_WRITE,
        "Mint a portable, Ed25519-signed SMT-LIB2 safety certificate for a "
        "protocol — one any third party can re-check WITHOUT trusting AegisOS.",
        {"risk_level": _s("Risk level the protocol is for, e.g. 'HIGH'."),
         "protocol": {"type": "OBJECT", "description": "The protocol to certify."},
         "vitals": {"type": "OBJECT", "description": "Vitals context."},
         "clinical_history": {"type": "OBJECT", "description": "History context."}}, []))
    T.append(Tool(
        "verify_proof", KIND_WRITE,
        "Re-check a signed safety certificate. Use to demonstrate that a forged "
        "or altered certificate fails verification.",
        {"certificate": {"type": "OBJECT", "description": "The certificate to check."},
         "require_signature": _b("Require a valid signature (default true).")},
        ["certificate"]))
    T.append(Tool(
        "verify_driver", KIND_WRITE,
        "Statically verify a device driver with the Aegis-eBPF verifier: proves "
        "termination, memory safety and capability confinement before the driver "
        "is ever allowed to touch a patient's data stream.",
        {"name": _s("A driver from os_read('/aegis/kernel/devices')."),
         "source": _s("Or raw eBPF assembly to verify instead."),
         "win_len": _n("Window length (default 2).")}, []))
    T.append(Tool(
        "verify_rules", KIND_WRITE,
        "Compile hospital-authored safety rules (the clinical DSL) to Z3 and "
        "prove a protocol against them — returns a counterexample for any rule "
        "that does not hold. Read the shipped rule set first with "
        "os_read('/aegis/dsl/default').",
        {"policy": _s("DSL rule text. Omit to use the shipped default policy."),
         "protocol": {"type": "OBJECT", "description": "The protocol to check."},
         "context": {"type": "OBJECT",
                     "description": "Patient context, e.g. {'risk_level':'HIGH','chf':true}."}},
        []))
    T.append(Tool(
        "check_identity", KIND_WRITE,
        "Run the physiologic identity guard: does this waveform still belong to "
        "the same patient? With only 'target' it verifies them against their own "
        "later stream; adding 'source' simulates another patient's stream being "
        "connected to their monitor, which should raise PATIENT_IDENTITY_FAULT.",
        {"target": _s("Whose monitor (patient id from os_read('/aegis/identity'))."),
         "source": _s("Whose stream is (mis)connected — omit for a self-check.")}, []))
    T.append(Tool(
        "train_model", KIND_WRITE,
        "Train or specialize a sepsis detector from inside the OS, with real "
        "grouped cross-validation. Returns a job id — poll it with "
        "os_read('/aegis/ml/job/<id>'). Registers the result; it does NOT "
        "hot-swap the certified production detector, and you must not imply it "
        "did.",
        {"name": _s("A name for the model you are training."),
         "feature_set": _s("Which features.", ("wearable", "lab")),
         "horizon": _n("Early-warning hours before onset, to specialize."),
         "limit": _n("How many patients to train on (default 1000)."),
         "folds": _n("CV folds (default 3).")}, ["name"]))
    T.append(Tool(
        "run_kernel_workload", KIND_WRITE,
        "Run the HDSched cognitive scheduler over a workload and return the "
        "benchmark: how the policy dispatched, with its overhead disclosed.",
        {"workload": _s("Workload.", ("mixed", "pipeline", "fanout", "stress")),
         "policy": _s("Policy.", ("FIFO", "RR", "CFS", "EDF", "HDSched"))}, []))
    T.append(Tool(
        "inject_fault", KIND_WRITE,
        "Inject a device fault — pull a sensor — to demonstrate that the kernel "
        "degrades safely rather than inventing data. Always say you are injecting "
        "a fault; never present the result as a real device failure.",
        {"slot": _s("Device key or bay, e.g. 'dev_0'."),
         "reason": _s("Why (recorded in the fault log).")}, ["slot"]))
    T.append(Tool(
        "recover_device", KIND_WRITE,
        "Bring an injected-fault device back and re-verify its driver.",
        {"slot": _s("Device key or bay.")}, ["slot"]))
    T.append(Tool(
        "run_chaos", KIND_WRITE,
        "Run a chaos experiment against one node of the cognitive scheduler — "
        "demonstrates the OS holding its safety properties under stress. Say "
        "that you are perturbing the system; never present the result as the "
        "patient's own physiology.",
        {"node": _s("Bed id to perturb."),
         "action": _s("What to do.", ("crisis", "clear", "noise")),
         "intensity": _n("How hard (default 1.0).")}, ["node"]))
    T.append(Tool(
        "manage_package", KIND_WRITE,
        "Install or uninstall a capability-confined AegisOS app package.",
        {"action": _s("What to do.", ("install", "uninstall")),
         "name": _s("Package name from os_read('/aegis/pkg').")}, ["action", "name"]))
    T.append(Tool(
        "import_history", KIND_WRITE,
        "Import a patient's clinical history as free text, or link it from a "
        "record source. This personalises their alerting PRIOR — say that it "
        "shifts when the OS alerts, not that their population risk changed.",
        {"bed": _s("bed_handle from list_patients."),
         "text": _s("The chart text to import."),
         "source": _s("Or a source to link from os_read('/aegis/records/sources')."),
         "patient_ref": _s("Source-specific id (FHIR patient id / MRN), when linking.")},
        ["bed"]))
    T.append(Tool(
        "device_action", KIND_WRITE,
        "Drive a device in the fleet: pair it, assign it to a bed, unassign it, "
        "or send it a command.",
        {"device_id": _s("Device id from os_read('/aegis/devices')."),
         "action": _s("What to do.", ("pair", "assign", "unassign", "command")),
         "bed": _s("Bed id, for assign."),
         "command": _s("Command name, for 'command'.")}, ["device_id", "action"]))
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
