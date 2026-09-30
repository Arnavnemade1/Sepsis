"""Dr. Aegis: the grounded assistant, the consult (text, stream, live voice relay), speech, HyperScan and the agent plane.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.schemas import (
    AgentInvokeIn,
    DoctorConsultIn,
    LLMAskIn,
    SpeakIn,
)
from backend.aegis.server import (
    _CALL_MAX_AUDIO_S,
    _HYPERSCAN_STAGES,
    _LiveConsult,
    _LiveUnavailable,
    _WATCHED_VITALS,
    _cred,
    _doctor,
    _doctor_dossier,
    _gate_verdict,
    _invoke_for,
    _llm,
    _llm_client,
    _llm_state,
    _rtcfg,
    _run_hyperscan,
    _speech,
    _threading,
    _tool_summarize,
    _tools,
)

router = APIRouter()


@router.get("/aegis/llm")
def llm_status() -> Dict[str, Any]:
    """Telemetry for the resilient LLM layer: chain, budget, cooldowns, cache."""
    return _llm_client.status()


@router.post("/aegis/llm/ask")
def llm_ask(payload: LLMAskIn) -> Dict[str, Any]:
    """Grounded natural-language Q&A over the LIVE system state (cohort + kernel).
    Read-only: the assistant answers only from the current state and refuses to
    issue orders — actions go through the certified syscall gate, not prose."""
    return {**_llm.ask(payload.question, _llm_state()),
            "budget_remaining": _llm_client.budget_remaining()}


@router.post("/aegis/llm/brief")
def llm_brief() -> Dict[str, Any]:
    """A shift-handoff briefing of the current cohort (grounded, budgeted)."""
    return {**_llm.brief(_llm_state()),
            "budget_remaining": _llm_client.budget_remaining()}


@router.get("/aegis/doctor/status")
def doctor_status() -> Dict[str, Any]:
    """Can we place a call, by which transport, and if not — why not?"""
    voice_ok, voice_reason = _LiveConsult.availability()
    return {
        "voice": voice_ok,                        # realtime audio socket
        "voice_reason": voice_reason,
        "key_source": _rtcfg.source("GEMINI_API_KEY"),   # environment | supabase | unset
        "text": bool(_llm_client.available),      # typed consult / fallback
        "llm": _llm_client.status(),
    }


@router.get("/aegis/doctor/brief")
def doctor_brief(bed: Optional[str] = None, opening: bool = True) -> Dict[str, Any]:
    """The case dossier the doctor is grounded in, plus its opening statement.

    Returned before the call connects so the UI can show what the doctor can see —
    a consult whose grounding is invisible is impossible to audit."""
    dossier = _doctor_dossier(bed)
    out: Dict[str, Any] = {
        "bed": dossier.get("bed"), "headline": dossier.get("headline"),
        "facts": dossier.get("facts"), "unmeasured": dossier.get("unmeasured"),
        "provenance": dossier.get("provenance"), "dossier_text": dossier.get("text"),
        "voice": _LiveConsult.available(),
        "voice_reason": _LiveConsult.availability()[1],
    }
    if opening:
        out["opening"] = _doctor.opening(dossier)
    return out


@router.get("/aegis/speech")
def speech_status() -> Dict[str, Any]:
    """Can the assistant read something out, and in whose voice?"""
    return _speech.status()


@router.post("/aegis/speech")
def speak(payload: SpeakIn) -> Dict[str, Any]:
    """Render text to speech in the assistant's voice.

    Same voice and same 24 kHz mono PCM as the consult call, so the assistant
    sounds like one clinician wherever you reach it. Returns
    {available:false} rather than an error when speech is unavailable — the
    caller always has the text and must not lose it because audio failed."""
    out = _speech.synthesize(payload.text, payload.voice)
    if not out:
        return {"available": False, "reason": (
            "no GEMINI_API_KEY on the server" if not _speech.available()
            else "speech budget exhausted or the renderer failed")}
    return {"available": True, **out}


@router.get("/aegis/hyperscan")
def hyperscan(bed: Optional[str] = None) -> Dict[str, Any]:
    """Full-subsystem sweep + situation report for one patient (blocking)."""
    return _run_hyperscan(bed, on_behalf_of=_cred())


@router.post("/aegis/hyperscan/stream")
def hyperscan_stream(payload: DoctorConsultIn):
    """The sweep, narrated stage by stage.

    Streamed because the point is watchable: each subsystem lands as its own
    event so the body on screen can flag the finding while the rest is still
    being collected. A nine-stage sweep behind a spinner would be the same work
    and none of the legibility."""
    from fastapi.responses import StreamingResponse
    import queue as _queue

    q: "_queue.Queue" = _queue.Queue()
    asker = _cred()

    def work() -> None:
        try:
            q.put({"type": "begin", "stages": len(_HYPERSCAN_STAGES) + 1})
            rep = _run_hyperscan(payload.bed, q.put, on_behalf_of=asker)
            # Report first, THEN the voice. Rendering speech takes ~12 s; holding
            # the finished report back for it would trade a fast read for a slow
            # listen. This way the report is on screen immediately and the audio
            # arrives while it is being read.
            q.put({"type": "report", "report": rep})
            spoken = rep.get("spoken")
            if spoken:
                q.put({"type": "speech_pending"})
                audio = _speech.synthesize(spoken)
                q.put({"type": "speech", "text": spoken,
                       **(audio or {"available": False})})
        except Exception as exc:                    # pragma: no cover
            q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            q.put(None)

    _threading.Thread(target=work, name="hyperscan", daemon=True).start()

    def events():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/aegis/doctor/consult/stream")
def doctor_consult_stream(payload: DoctorConsultIn):
    """The typed consult, narrated as it works.

    An agentic answer that chains four tools takes about nine seconds. Returned
    as one blob that is nine seconds of nothing, which reads as a hang — the user
    reported it as "takes a long time" long before the model was actually slow.
    Same work, streamed: each tool is announced before it runs, so the wait is
    legible and the cockpit can follow along live instead of catching up at the
    end."""
    from fastapi.responses import StreamingResponse
    import queue as _queue

    q: "_queue.Queue" = _queue.Queue()
    asker = _cred()             # captured here: the worker thread has no request context

    def work() -> None:
        try:
            dossier = _doctor_dossier(payload.bed)
            q.put({"type": "grounded", "bed": dossier.get("bed"),
                   "headline": dossier.get("headline"), "facts": dossier.get("facts")})
            res = _doctor.act(payload.question, dossier, _tools.declarations(),
                              _invoke_for(asker), history=payload.history,
                              budgets={t.name: t.budget for t in _tools._tools.values()
                                       if t.budget},
                              on_event=q.put)
            q.put({"type": "answer", **res})
        except Exception as exc:                       # pragma: no cover
            q.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            q.put(None)

    _threading.Thread(target=work, name="consult-stream", daemon=True).start()

    def events():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/aegis/doctor/consult")
def doctor_consult(payload: DoctorConsultIn) -> Dict[str, Any]:
    """One typed turn of the consult — the transport when there is no microphone,
    and the fallback when the realtime socket is unavailable.

    Agentic by default: the doctor may drive the OS to answer, and any cockpit
    commands it issues come back in ``actions`` for the browser to run, so the
    typed path has the same reach as the voice path."""
    dossier = _doctor_dossier(payload.bed)
    if payload.agentic:
        res = _doctor.act(payload.question, dossier, _tools.declarations(),
                          _invoke_for(_cred()), history=payload.history,
                          budgets={t.name: t.budget for t in _tools._tools.values()
                                   if t.budget})
    else:
        res = _doctor.consult(payload.question, dossier, history=payload.history)
    return {**res, "bed": dossier.get("bed"), "headline": dossier.get("headline"),
            "budget_remaining": _llm_client.budget_remaining()}


@router.get("/aegis/agent/tools")
def agent_tools() -> Dict[str, Any]:
    """The agent's capability catalogue — what Dr. Aegis can reach, read and do,
    and which of those go through the certified gate."""
    cat = _tools.catalogue()
    return {"tools": cat, "bound": len(_tools.bound()), "unbound": _tools.unbound(),
            "counts": {k: sum(1 for t in cat if t["kind"] == k)
                       for k in ("read", "ui", "write")}}


@router.get("/aegis/agent/activity")
def agent_activity(limit: int = 50) -> Dict[str, Any]:
    """What the agent has actually done — every tool call and its outcome."""
    return {"activity": _tools.activity(limit)}


@router.post("/aegis/agent/invoke")
def agent_invoke(payload: AgentInvokeIn) -> Dict[str, Any]:
    """Invoke one tool directly (the text consult and the console use this)."""
    return _tools.invoke(payload.tool, payload.args, on_behalf_of=_cred())


@router.websocket("/aegis/doctor/live")
async def doctor_live(ws: WebSocket) -> None:
    """Relay: browser audio/video  <->  Gemini Live, grounded in the bed's dossier.

    Browser frames (JSON): audio{data} · video{data,mime} · text{text} ·
    end_audio · bye. Server frames: ready · transcript · audio · turn_end ·
    interrupted · case_update · error · closed."""
    await ws.accept()
    bed = ws.query_params.get("bed") or None
    # Who opened the call. Tools run in an executor, which does not inherit context
    # variables, so the credential is captured now and bound to every tool call.
    live_caller = _cred()
    session_live: Optional[_LiveConsult] = None
    try:
        dossier = _doctor_dossier(bed)
    except Exception as exc:
        await ws.send_json({"type": "error", "fatal": True,
                            "message": f"could not assemble the case: {exc}"})
        await ws.close()
        return

    try:
        session_live = _LiveConsult(_doctor.system_instruction(dossier),
                                    tools=_tools.declarations())
        await session_live.connect()
    except _LiveUnavailable as exc:
        # Not an error the clinician can act on — tell the UI to fall back to the
        # typed consult, which is grounded in the identical dossier.
        await ws.send_json({"type": "error", "fatal": True, "fallback": "text",
                            "message": str(exc)})
        await ws.close()
        return

    await ws.send_json({
        "type": "ready", "bed": dossier.get("bed"), "headline": dossier.get("headline"),
        "facts": dossier.get("facts"), "unmeasured": dossier.get("unmeasured"),
        "provenance": dossier.get("provenance"),
        "input_rate": 16000, "output_rate": 24000, "voice": session_live.voice,
        "model": session_live.model, "tools": len(session_live.tools),
    })

    stop = asyncio.Event()
    budget = {"audio_s": 0.0}
    # Which bed the conversation is actually on. The agent can move it mid-call.
    scope: Dict[str, Any] = {
        "bed": bed,
        "tier": (dossier.get("facts") or {}).get("tier"),
        "vitals": dict((dossier.get("facts") or {}).get("vitals") or {}),
        "dossier": dossier,
        # A rolling tail of what was said, so a session that expires can be
        # rebuilt with the conversation intact rather than starting over.
        "tail": [],
        "session": 1,
    }

    def remember(role: str, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        tail = scope["tail"]
        if tail and tail[-1]["role"] == role:
            tail[-1]["text"] = (tail[-1]["text"] + " " + t)[-1200:]
        else:
            tail.append({"role": role, "text": t[:1200]})
        del tail[:-10]

    async def run_tools(calls: List[Dict[str, Any]]) -> None:
        """Execute what the agent asked for and answer every call it made.

        UI tools are not run here — they are forwarded to the cockpit, which
        drives the real shell. The model is told the command was dispatched so it
        can narrate ("pulling that up now") without waiting on the browser."""
        responses: List[Dict[str, Any]] = []
        loop = asyncio.get_running_loop()
        for call in calls:
            name = call.get("name") or ""
            # Off the event loop: tools are synchronous and some are slow (a
            # kernel workload, a training launch), and the ones that wrap async
            # endpoints have to be able to schedule back onto THIS loop and wait.
            # Running them inline would block the audio relay and deadlock those.
            # run_in_executor does not carry context variables, so the caller's
            # credential is bound explicitly (captured when the socket opened).
            out = await loop.run_in_executor(
                None, lambda n=name, a=(call.get("args") or {}): _tools.invoke(
                    n, a, on_behalf_of=live_caller))
            ui = out.get("ui")
            if ui:
                await ws.send_json({"type": "ui", "action": ui.get("action"),
                                    "args": ui.get("args") or {}})
                # the tier watcher has to follow the agent, or it keeps reporting
                # changes on the bed the call opened on rather than the one being
                # discussed now
                if name == "select_patient" and (ui.get("args") or {}).get("bed"):
                    scope["bed"] = ui["args"]["bed"]
                    scope["tier"] = None
            # mirror every call into the cockpit's activity feed: an agent whose
            # actions are invisible is an agent nobody can check
            await ws.send_json({
                "type": "tool", "tool": name, "args": call.get("args") or {},
                "ok": bool(out.get("ok")), "kind": (_tools.get(name).kind
                                                    if _tools.get(name) else "unknown"),
                "error": out.get("error"),
                "verdict": _gate_verdict(out)})
            spec = _tools.get(name)
            responses.append({"id": call.get("id"), "name": name,
                              "response": _tool_summarize(
                                  out, (spec.budget if spec and spec.budget else 3500))})
        if responses:
            await session_live.send_tool_response(responses)

    async def pump_up() -> None:
        """Browser -> Gemini. Forwards chunk-for-chunk; never batches (pacing is
        what the model's turn detection runs on)."""
        from backend.llm.live import pcm_duration_s
        try:
            while not stop.is_set():
                msg = await ws.receive_json()
                kind = msg.get("type")
                if kind == "audio" and msg.get("data"):
                    budget["audio_s"] += pcm_duration_s(msg["data"])
                    if budget["audio_s"] > _CALL_MAX_AUDIO_S:
                        await ws.send_json({"type": "error", "fatal": True,
                                            "message": "call length limit reached"})
                        break
                    await session_live.send_audio(msg["data"])
                elif kind == "video" and msg.get("data"):
                    await session_live.send_video(msg["data"],
                                                  msg.get("mime", "image/jpeg"))
                elif kind == "text" and msg.get("text"):
                    await session_live.send_text(str(msg["text"])[:2000])
                elif kind == "end_audio":
                    await session_live.end_audio()
                elif kind == "bye":
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:                          # pragma: no cover
            print(f"[doctor] uplink ended: {type(exc).__name__}: {exc}")
        finally:
            stop.set()

    async def reconnect() -> bool:
        """Rebuild the realtime session, carrying the conversation across.

        Gemini Live sessions expire (~15 min) and announce it with `goAway`. The
        old behaviour was a banner and then silence mid-sentence, which in front
        of anyone reads as broken. Instead: open a fresh session on the CURRENT
        bed's dossier, replay the last few turns as context, and keep going. The
        clinician sees a short reconnecting blip, not the end of the consult."""
        nonlocal session_live
        old = session_live
        try:
            fresh = _doctor_dossier(scope["bed"])
            scope["dossier"] = fresh
            tail = scope["tail"]
            context = ""
            if tail:
                lines = "\n".join(
                    f"{'You' if t['role'] == 'doctor' else 'The clinician'}: {t['text']}"
                    for t in tail)
                context = (
                    "\n\n=== CONVERSATION SO FAR ===\nThis consult is already in "
                    "progress; the realtime session was renewed underneath it. Do "
                    "NOT greet them again or re-introduce yourself. Pick up exactly "
                    "where this left off:\n" + lines + "\n=== END ===")
            new = _LiveConsult(_doctor.system_instruction(fresh) + context,
                               tools=_tools.declarations())
            await new.connect()
            session_live = new
            scope["session"] += 1
        except Exception as exc:
            print(f"[doctor] reconnect failed: {type(exc).__name__}: {exc}")
            return False
        finally:
            try:
                await old.close()
            except Exception:
                pass
        return True

    async def pump_down() -> None:
        """Gemini -> browser, with tool calls executed and expiry healed in between."""
        try:
            while not stop.is_set():
                renewed = False
                async for ev in session_live.events():
                    if stop.is_set():
                        break
                    kind = ev.get("type")
                    if kind == "tool_call":
                        await run_tools(ev.get("calls") or [])
                        continue                   # the raw call is not the UI's business
                    if kind == "tool_cancel":
                        continue
                    if kind == "transcript":
                        remember("doctor" if ev.get("role") == "doctor" else "clinician",
                                 ev.get("text", ""))
                    if kind == "going_away":
                        # Renew BEFORE it dies, so the seam falls in a gap rather
                        # than in the middle of a sentence.
                        await ws.send_json({"type": "renewing"})
                        renewed = await reconnect()
                        if renewed:
                            await ws.send_json({"type": "renewed",
                                                "session": scope["session"]})
                        break
                    await ws.send_json(ev)
                    if kind == "closed":
                        break
                if not renewed:
                    break
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:                          # pragma: no cover
            print(f"[doctor] downlink ended: {type(exc).__name__}: {exc}")
        finally:
            stop.set()

    async def watch_case() -> None:
        """A call outlives the state it opened on, and the dossier does not move.

        The system instruction is fixed at connect, so ten minutes into a consult
        the doctor is still holding the vitals it was handed at the start. Tier
        changes were caught; a systolic sliding from 104 to 84 was not, and a
        system whose whole argument is honesty about what it knows should not
        state a twenty-minute-old number in a live conversation.

        So watch the numbers too, and only speak up when a move is material —
        a note on every tick would be noise the doctor learns to ignore."""
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=20.0)
                return
            except asyncio.TimeoutError:
                pass
            try:
                fresh = _doctor_dossier(scope["bed"])
                facts = fresh.get("facts") or {}
                changes: List[str] = []

                tier, last = facts.get("tier"), scope["tier"]
                if tier and last and tier != last:
                    changes.append(f"tier {last} -> {tier}")
                scope["tier"] = tier

                now_v = facts.get("vitals") or {}
                was_v = scope["vitals"] or {}
                for key, label, unit, delta in _WATCHED_VITALS:
                    a, b = was_v.get(key), now_v.get(key)
                    if a is None or b is None:
                        continue
                    if abs(float(b) - float(a)) >= delta:
                        changes.append(
                            f"{label} {float(a):.0f} -> {float(b):.0f} {unit}".strip())
                        was_v[key] = b
                scope["vitals"] = {**was_v, **{k: v for k, v in now_v.items()
                                               if k not in was_v}}

                if changes:
                    await session_live.note(
                        f"{fresh.get('headline')} — " + "; ".join(changes)
                        + ". These supersede the numbers in your dossier; use "
                          "these, and say they have moved since you last looked.")
                    await ws.send_json({"type": "case_update", "facts": facts,
                                        "headline": fresh.get("headline"),
                                        "changes": changes, "to_tier": tier})
            except Exception:
                return

    try:
        await asyncio.gather(pump_up(), pump_down(), watch_case(),
                             return_exceptions=True)
    finally:
        stop.set()
        if session_live is not None:
            await session_live.close()
        try:
            await ws.close()
        except Exception:
            pass

