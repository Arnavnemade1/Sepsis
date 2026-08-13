"""Verification for the agent plane — the part where Dr. Aegis operates the OS.

Giving a language model a control surface over a clinical system is only
defensible if the surface itself has properties you can state and check. These
are those properties:

  * every advertised tool is bound — a model must never be offered a capability
    that fails the moment it reaches for it;
  * the schemas are valid function declarations, so the model can actually call
    them rather than guessing at shapes;
  * the universal read primitive is a READ: confined to this app's GET routes,
    unable to leave /aegis, unable to reach a mutation;
  * **every write goes through the Z3 trap gate**, and a refused write changes
    nothing — checked by refusing one and then proving the kernel did not move;
  * a UI command reports honestly whether it returned data, because "I opened it"
    and "I have seen it" are different claims;
  * a failing tool returns a result the agent must account for, never an
    exception that silently ends its turn.

Runs in-process against the real app. No LLM, no network, no spend.

    python -m backend.aegis.verify_agent
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List

PASS, FAIL = "[PASS]", "[FAIL]"
_failures: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"      {PASS if ok else FAIL} {detail or name}")
    if not ok:
        _failures.append(name)


def main() -> int:
    print("=" * 70)
    print("AGENT PLANE — VERIFICATION")
    print("=" * 70)

    import backend.aegis.server as srv
    from backend.aegis.agent import KIND_UI, KIND_WRITE, registry as R, summarize

    # -- 1. everything advertised is real ------------------------------- #
    print("[1] The catalogue: nothing is offered that cannot be called")
    cat = R.catalogue()
    check("all bound", not R.unbound(),
          f"{len(R.bound())} tools bound, 0 unbound "
          f"({sum(1 for t in cat if t['kind']=='read')} read / "
          f"{sum(1 for t in cat if t['kind']=='ui')} ui / "
          f"{sum(1 for t in cat if t['kind']=='write')} write)")
    decls = R.declarations()
    check("advertised == bound", len(decls) == len(R.bound()),
          "only bound tools are advertised to the model")
    bad = []
    for d in decls:
        if not d.get("name") or not d.get("description"):
            bad.append(d.get("name"))
        p = d.get("parameters")
        if p and (p.get("type") != "OBJECT" or not isinstance(p.get("properties"), dict)):
            bad.append(d.get("name"))
        for req in (p or {}).get("required", []):
            if req not in (p or {}).get("properties", {}):
                bad.append(f"{d.get('name')}:{req}")
    check("schemas valid", not bad, f"all {len(decls)} declarations well-formed"
          if not bad else f"malformed: {bad}")

    # -- 2. the read primitive is a read -------------------------------- #
    print("[2] os_read reaches everything in AegisOS — and nothing outside it")
    ok = R.invoke("os_read", {"path": "/aegis/orchestrator"})
    check("reads", ok["ok"] and "error" not in (ok.get("result") or {}),
          "reads a named subsystem")
    tmpl = R.invoke("os_read", {"path": "/aegis/cohort/bed_0/bundle"})
    res = tmpl.get("result") or {}
    check("templated", res.get("bed_id") == "bed_0",
          "resolves a templated path correctly (bed_id='bed_0', not 'bed_0/bundle')")
    q = R.invoke("os_read", {"path": "/aegis/syscall/audit", "params": "limit=3"})
    check("query coercion", (q.get("result") or {}).get("log") is not None,
          "coerces query values to the endpoint's types")
    for bad_path in ("/etc/passwd", "../../secret", "http://evil/x", "/aegis/../etc"):
        r = (R.invoke("os_read", {"path": bad_path}).get("result") or {})
        if "error" not in r:
            check("confinement", False, f"escaped via {bad_path!r}")
            break
    else:
        check("confinement", True, "refuses any path outside /aegis")
    ws = (R.invoke("os_read", {"path": "/aegis/stream"}).get("result") or {})
    check("no websockets", "error" in ws, "cannot reach a WebSocket route")
    surfaces = R.invoke("list_surfaces")["result"]
    check("discoverable", surfaces["count"] > 30,
          f"{surfaces['count']} readable surfaces are self-describing")

    # A read tool must not be able to reach a mutation: no POST-only route is
    # in the read index, and the read primitive refuses one by path.
    read_only = [p for p in srv._READ_ROUTES
                 if "GET" not in (getattr(srv._READ_ROUTES[p], "methods", None) or set())]
    check("gets only", not read_only, "every indexed read route is a GET")
    post_only = next((p for p in srv._WRITE_ROUTES if p not in srv._READ_ROUTES), None)
    if post_only:
        r = (R.invoke("os_read", {"path": post_only}).get("result") or {})
        check("read cannot mutate", "error" in r,
              f"os_read refuses a POST-only path ({post_only})")

    # -- 2b. the write primitive reaches the actuation surface ---------- #
    print("[2b] os_write drives the OS — and stays inside it")
    actions = R.invoke("list_actions")["result"]
    check("actions listed", actions["count"] > 25,
          f"{actions['count']} actuation endpoints are discoverable with their bodies")
    check("bodies resolved", any(a["body"] for a in actions["actions"]),
          "body models resolve under Pydantic v2 (field_info.annotation, not .type_)")
    for bad_path in ("/etc/shadow", "http://evil/x", "../../x"):
        r = (R.invoke("os_write", {"path": bad_path}).get("result") or {})
        if "error" not in r:
            check("write confinement", False, f"escaped via {bad_path!r}")
            break
    else:
        check("write confinement", True, "refuses any path outside /aegis")
    reentry = (R.invoke("os_write", {"path": "/aegis/agent/invoke",
                                     "body": {}}).get("result") or {})
    check("no re-entry", "error" in reentry,
          "cannot re-enter the agent plane through itself")
    missing = (R.invoke("os_write", {"path": "/aegis/nope"}).get("result") or {})
    check("unknown action", "error" in missing, "an unknown action path is reported, not crashed")

    # THE invariant: clinical egress is gated no matter which road reaches it.
    via_write = (R.invoke("os_write", {
        "path": "/aegis/syscall",
        "body": {"syscall": "escalate", "bed": "bed_1", "args": {"level": 3}}
    }).get("result") or {})
    check("gate holds via os_write", via_write.get("errno_name") == "EPERM",
          "a baseless escalation through the GENERIC write primitive is still refused "
          "— the gate is in the endpoint, not in the tool wrapper")

    # -- 3. writes go through the gate ---------------------------------- #
    print("[3] Writes: the trap gate decides, not the agent")
    unknown = R.invoke("syscall", {"syscall": "rm_rf", "bed": "bed_0"})["result"]
    check("enumerated", unknown["errno_name"] == "EINVAL" and not unknown["allowed"],
          "an off-table syscall traps to EINVAL — there is no generic escape hatch")

    # a stable bed escalated to CRITICAL is a baseless high alert: must be refused
    before = srv.syscalls.status()
    denied = R.invoke("syscall", {"syscall": "escalate", "bed": "bed_1",
                                  "args": {"level": 3}})["result"]
    after = srv.syscalls.status()
    check("refuses", denied["errno_name"] == "EPERM" and not denied["allowed"],
          f"baseless escalation refused: {denied['message'][:72]}")
    check("z3 decided", denied["z3_status"] in ("sat", "unsat"),
          f"the refusal carries the prover's verdict (z3={denied['z3_status']})")
    iv_before = (before.get("active_interventions") or {}).get("bed_1", {})
    iv_after = (after.get("active_interventions") or {}).get("bed_1", {})
    check("nothing changed", iv_before.get("pin") == iv_after.get("pin"),
          "a denied write left the kernel untouched (pin unchanged)")
    check("audited", after["invoked"] > before["invoked"],
          "the refusal is in the append-only audit log, not just dropped")

    # the agent's own log must carry the verdict so the cockpit can show it
    act = [a for a in R.activity(10) if a["tool"] == "syscall"]
    check("verdict surfaced", act and act[-1].get("z3_status") is not None,
          "the activity feed carries errno + z3 status for every write")

    # -- 3b. the named actuation tools actually run --------------------- #
    print("[3b] Named actuation: every wrapper matches its endpoint's real payload")
    ran = []
    for name, args in (("verify_driver", {"name": "ecg"}),
                       ("verify_rules", {}),
                       ("check_identity", {}),
                       ("run_kernel_workload", {"workload": "mixed", "policy": "HDSched"}),
                       ("run_scenario", {}),
                       ("certify_proof", {"risk_level": "HIGH"}),
                       ("govern", {"action": "treat", "target": "bed_0"}),
                       ("counterfactual_replay", {"bed": "bed_0"})):
        out = R.invoke(name, args)
        bad = (not out.get("ok")) or "bad arguments" in str(out.get("error", ""))
        ran.append((name, not bad, out.get("error")))
    broken = [(n, e) for n, ok, e in ran if not ok]
    check("wrappers match", not broken,
          f"{len(ran)} named actuation tools dispatch cleanly"
          if not broken else f"signature mismatch: {broken}")
    # a wrapper that silently mis-declares its schema is the failure mode here:
    # the model calls it once, gets "bad arguments", and gives up on the tool
    for tname in ("train_model", "run_chaos", "device_action", "import_history",
                  "manage_package", "inject_fault"):
        tool = R.get(tname)
        miss = [r for r in tool.required if r not in tool.properties]
        check(f"schema {tname}", not miss,
              f"{tname}: required args all declared ({', '.join(tool.required) or 'none'})")

    # -- 4. UI commands are honest about what they returned ------------- #
    print("[4] UI commands: 'I opened it' and 'I have seen it' stay distinct")
    story = R.invoke("open_panel", {"panel": "story", "bed": "bed_0"})
    check("dispatch", story["ui"]["action"] == "open_panel",
          "the command is handed to the cockpit, not executed server-side")
    check("companion read", isinstance(story.get("data"), dict),
          "a panel with readable contents returns them, so nothing is left to invent")
    cam = R.invoke("open_panel", {"panel": "camera"})
    check("no data => says so", "data" not in cam and "NO data" in cam["note"],
          "a panel with nothing to read says plainly that it returned no data")
    sel = R.invoke("select_patient", {"bed": "bed_2"})
    check("re-grounds", (sel.get("data") or {}).get("bed") == "bed_2",
          "switching beds hands over the NEW patient's dossier (the locked one is stale)")

    # -- 4b. treatment options: the refusals are the point --------------- #
    print("[4b] Treatment options carry the cage's verdicts, and fit in a turn")
    to = R.invoke("show_treatment_options", {"bed": "bed_0"})
    data = to.get("data") or {}
    opts = data.get("options") or []
    check("options read", len(opts) >= 3,
          f"{len(opts)} candidate paths returned with the panel")
    check("verdicts", all(o.get("verdict") and o.get("z3_status") for o in opts),
          "every option carries the prover's verdict and status")
    refused = [o for o in opts if not o.get("allowed")]
    check("refusal visible", any("REJECT" in str(o.get("verdict", "")).upper()
                                 for o in refused) if refused else True,
          f"{len(refused)} refused option(s) quote the cage verbatim"
          if refused else "no refusals for this patient right now (not a failure)")
    budget = R.get("show_treatment_options").budget
    import json as _json
    sized = summarize(to, budget or 3500)
    check("fits the turn", not sized.get("truncated"),
          f"payload {len(_json.dumps(to, default=str))} chars fits its {budget}-char "
          "budget — a truncated options list would describe the first two paths "
          "as if they were all of them")

    # -- 5. rankable roster --------------------------------------------- #
    print("[5] 'Who is the sickest patient' is answerable from the data")
    roster = R.invoke("list_patients")["result"]
    risks = [p["risk"] for p in roster["patients"] if p["risk"] is not None]
    check("has risk", len(risks) >= 5, f"{len(risks)} beds carry a live risk value")
    check("sorted", risks == sorted(risks, reverse=True),
          "the roster is ordered sickest-first, so ranking is read not guessed")
    check("has tier", any(p.get("tier") for p in roster["patients"]),
          "each bed carries its tier alongside the number")
    first = roster["patients"][0]
    check("name leads", list(first)[0] == "name" and first.get("bed_handle"),
          "the NAME is the first field and the id is demoted to 'bed_handle' — "
          "the model was transposing 'bed three' while acting on bed four")
    check("handle explained", "bed_handle" in roster.get("note", ""),
          "the roster tells the model the handle is internal, not to be spoken")

    # -- 5b. the vitals the mid-call watcher tracks --------------------- #
    print("[5b] The watcher can see vitals move, not just the tier")
    watched = {v[0] for v in srv._WATCHED_VITALS}
    facts = (R.invoke("get_case_dossier", {"bed": "bed_0"})["result"]
             .get("facts") or {})
    present = watched & set((facts.get("vitals") or {}))
    check("watchable", len(present) >= 3,
          f"{len(present)} of the watched vitals are present in the dossier facts "
          f"({', '.join(sorted(present))})")
    check("thresholds sane", all(d > 0 for _, _, _, d in srv._WATCHED_VITALS),
          "every watched vital has a non-zero threshold, so a note means something moved")

    # -- 6. failures are results, not exceptions ------------------------ #
    print("[6] A failing tool returns something the agent has to account for")
    for name, args in (("nonexistent_tool", {}),
                       ("get_patient", {"wrong_arg": 1}),
                       ("os_read", {}),
                       ("syscall", {"syscall": "treat", "bed": "no_such_bed"})):
        out = R.invoke(name, args)
        if not isinstance(out, dict) or "ok" not in out:
            check("no exceptions", False, f"{name}{args} did not return a result dict")
            break
    else:
        check("no exceptions", True,
              "unknown tools, bad arguments and missing beds all return results")
    unknown_tool = R.invoke("nonexistent_tool", {})
    check("names alternatives", not unknown_tool["ok"] and unknown_tool.get("available"),
          "an unknown tool comes back with the list of real ones")

    # -- 7. result bounding --------------------------------------------- #
    print("[7] One tool result cannot swallow the conversation")
    big = summarize({"ok": True, "result": {"x": "y" * 50000}}, max_chars=2000)
    check("bounded", big.get("truncated") and len(str(big)) < 4000,
          "an oversized payload is truncated with an explicit marker")

    print("-" * 70)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {', '.join(_failures)}")
        return 1
    print("ALL AGENT-PLANE CHECKS PASSED ✓  (catalogue · read confinement · "
          "gated writes · honest UI · failure handling)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
