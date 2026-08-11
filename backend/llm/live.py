"""LiveConsult — the realtime audio session behind the AI-doctor call.

Wraps Gemini's ``bidiGenerateContent`` socket and normalizes it into a small,
boring event stream the relay can forward to a browser. The browser never talks
to Google: it talks to AegisOS, and AegisOS holds the key. That is not incidental
— ephemeral auth tokens mint fine on this account but are rejected by the Live
socket, so a relay is the only design in which the credential stays server-side.

Two things learned the hard way, both verified live against the real socket:

  * **``realtimeInput.audio``, not ``mediaChunks``.** The latter is deprecated and
    the server closes the connection with 1007 rather than ignoring it.
  * **Audio must arrive paced, not bursted.** Sending a whole utterance's chunks
    back-to-back leaves voice-activity detection waiting forever and the turn
    never completes — the first probe of this hung with an empty transcript. A
    browser capturing from a microphone paces naturally at 1× realtime, so the
    relay forwards chunk-for-chunk and never batches or buffers ahead.

The session is grounded by a system instruction locked in at setup
(:meth:`AIDoctor.system_instruction`). ``note()`` injects a mid-call state change
as an out-of-band system note, so a call that outlives the patient's tier is told
about it instead of quietly discussing stale numbers.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, AsyncIterator, Dict, Optional

try:
    import websockets
except Exception:  # pragma: no cover
    websockets = None

from backend.llm.gemini import live_model, live_url, live_voice

# Wire formats fixed by the Live API. The browser must match these exactly.
INPUT_RATE = 16000        # mono s16le PCM up to the model
OUTPUT_RATE = 24000       # mono s16le PCM back from the model


class LiveUnavailable(RuntimeError):
    """No key, no websockets library, or the socket refused the session."""


class LiveConsult:
    def __init__(self, system_instruction: str, voice: Optional[str] = None,
                 model: Optional[str] = None, connect_timeout: float = 20.0,
                 tools: Optional[List[Dict[str, Any]]] = None):
        self.system_instruction = system_instruction
        self.voice = voice or live_voice()
        self.model = model or live_model()
        self.connect_timeout = connect_timeout
        self.tools = tools or []            # functionDeclarations the agent may call
        self._ws: Optional[Any] = None
        self._closed = False

    @staticmethod
    def available() -> bool:
        return websockets is not None and bool(
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    # -- lifecycle -------------------------------------------------------- #
    async def connect(self) -> None:
        if not self.available():
            raise LiveUnavailable("realtime consult unavailable (no key or websockets)")
        url = live_url()
        if not url:
            raise LiveUnavailable("no GEMINI_API_KEY configured")
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(url, max_size=None), self.connect_timeout)
            setup: Dict[str, Any] = {
                "model": self.model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": self.voice}}},
                },
                "systemInstruction": {"parts": [{"text": self.system_instruction}]},
                # Both directions transcribed: the call needs a readable record,
                # and a consult you cannot read back is not a clinical artifact.
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
            }
            if self.tools:
                setup["tools"] = [{"functionDeclarations": self.tools}]
            await self._ws.send(json.dumps({"setup": setup}))
            raw = await asyncio.wait_for(self._ws.recv(), self.connect_timeout)
            if "setupComplete" not in self._decode(raw):
                raise LiveUnavailable(f"unexpected setup reply: {str(raw)[:200]}")
        except LiveUnavailable:
            raise
        except Exception as exc:
            await self.close()
            raise LiveUnavailable(f"live connect failed: {type(exc).__name__}: {exc}")

    async def close(self) -> None:
        self._closed = True
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    # -- uplink ----------------------------------------------------------- #
    async def _send(self, msg: Dict[str, Any]) -> None:
        if self._ws is None or self._closed:
            return
        await self._ws.send(json.dumps(msg))

    async def send_audio(self, b64_pcm16: str) -> None:
        """Forward one microphone chunk (base64 s16le mono @16 kHz), unbuffered."""
        await self._send({"realtimeInput": {"audio": {
            "data": b64_pcm16, "mimeType": f"audio/pcm;rate={INPUT_RATE}"}}})

    async def send_video(self, b64_jpeg: str, mime: str = "image/jpeg") -> None:
        """Forward one camera frame — this is what lets the doctor comment on what
        it can actually see of the patient rather than only the telemetry."""
        await self._send({"realtimeInput": {"video": {"data": b64_jpeg, "mimeType": mime}}})

    async def send_text(self, text: str) -> None:
        """A typed turn mid-call (no microphone, or a value easier typed than said)."""
        await self._send({"clientContent": {
            "turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": True}})

    async def note(self, text: str) -> None:
        """Inject an out-of-band state change without it reading as the clinician
        speaking. Marked so the doctor reports it as a system update."""
        await self._send({"clientContent": {
            "turns": [{"role": "user", "parts": [{"text":
                "[SYSTEM UPDATE — not spoken by the clinician. The monitored state "
                f"changed mid-call: {text}. Mention this briefly when it is "
                "relevant; do not read it out verbatim.]"}]}],
            "turnComplete": False}})

    async def end_audio(self) -> None:
        await self._send({"realtimeInput": {"audioStreamEnd": True}})

    async def send_tool_response(self, responses: List[Dict[str, Any]]) -> None:
        """Return tool results. Each entry is ``{id, name, response}``.

        The model is blocked on these — a call it never gets an answer to leaves
        the turn hanging with dead air on a voice line, so the relay must reply
        to EVERY id it was given, failures included."""
        await self._send({"toolResponse": {"functionResponses": responses}})

    # -- downlink --------------------------------------------------------- #
    @staticmethod
    def _decode(raw: Any) -> Dict[str, Any]:
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", "replace")
            return json.loads(raw)
        except Exception:
            return {}

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        """Normalized events: transcript / audio / turn_end / interrupted / going_away."""
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                msg = self._decode(raw)
                if not msg:
                    continue
                if msg.get("goAway"):
                    yield {"type": "going_away",
                           "time_left": (msg["goAway"] or {}).get("timeLeft")}
                    continue
                tc = msg.get("toolCall")
                if tc:
                    calls = [{"id": c.get("id"), "name": c.get("name"),
                              "args": c.get("args") or {}}
                             for c in (tc.get("functionCalls") or [])]
                    if calls:
                        yield {"type": "tool_call", "calls": calls}
                    continue
                if msg.get("toolCallCancellation"):
                    yield {"type": "tool_cancel",
                           "ids": (msg["toolCallCancellation"] or {}).get("ids") or []}
                    continue
                sc = msg.get("serverContent")
                if not sc:
                    continue
                it = sc.get("inputTranscription")
                if it and it.get("text"):
                    yield {"type": "transcript", "role": "clinician", "text": it["text"]}
                ot = sc.get("outputTranscription")
                if ot and ot.get("text"):
                    yield {"type": "transcript", "role": "doctor", "text": ot["text"]}
                for part in ((sc.get("modelTurn") or {}).get("parts") or []):
                    inline = part.get("inlineData")
                    if inline and inline.get("data"):
                        yield {"type": "audio", "data": inline["data"], "rate": OUTPUT_RATE}
                    elif part.get("text"):
                        yield {"type": "transcript", "role": "doctor", "text": part["text"]}
                if sc.get("interrupted"):
                    # Barge-in: the clinician talked over the doctor. The browser
                    # must drop whatever it has queued or it plays a dead answer.
                    yield {"type": "interrupted"}
                if sc.get("turnComplete"):
                    yield {"type": "turn_end"}
        except Exception as exc:
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        finally:
            yield {"type": "closed"}


def pcm_duration_s(b64_pcm16: str, rate: int = INPUT_RATE) -> float:
    """Seconds of audio in a base64 s16le chunk — used to bound a call's uplink."""
    try:
        return len(base64.b64decode(b64_pcm16)) / 2.0 / float(rate)
    except Exception:
        return 0.0
