"""Speech — Dr. Aegis reading something out, outside a live call.

The consult already talks: the Live socket streams native audio both ways. But
the hyperscan is not a conversation, it is a report, and a report that only
exists on screen is one a clinician has to stop and read at exactly the moment
they are doing something else with their hands. So the report gets a voice too.

Synthesis rather than the Live socket, deliberately. Opening a realtime session
to say forty seconds of prose would hold a bidirectional audio channel open for a
monologue, and realtime sessions expire, get interrupted, and cost far more to
keep alive than one render of fixed text. This is a paragraph that is already
written; it wants a renderer, not a conversation.

Same voice as the call (``GEMINI_LIVE_VOICE``, default Charon) and the same
24 kHz mono PCM the consult playback path already handles, so the browser plays
it through machinery that exists rather than a second audio stack. If the two
ever drifted apart, the assistant would sound like two different people depending
on which surface you reached it from.

Failure is silent and safe, like every other LLM path here: no key, no budget, no
network — ``synthesize`` returns None, and the caller shows the text it was
always going to show anyway.
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:  # pragma: no cover
    pass

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
OUT_RATE = 24000                     # mono s16le — identical to the consult downlink

# A rendered paragraph is ~1 MB of base64 and takes ~12 s to generate, so a
# repeat (re-open the report, replay it) must never pay that twice.
_CACHE: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
_CACHE_TTL = 1800.0
_CACHE_MAX = 24

# Speech is the most expensive thing per call in this whole layer. Cap it.
_MAX_CHARS = 1400
_MAX_CALLS = 60
_calls = 0


def voice() -> str:
    return os.getenv("GEMINI_LIVE_VOICE", "Charon")


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) \
        and httpx is not None


def budget_remaining() -> int:
    return max(0, _MAX_CALLS - _calls)


def _key(text: str, v: str) -> str:
    return hashlib.sha256(f"{v}\x00{text}".encode("utf-8")).hexdigest()


def synthesize(text: str, v: Optional[str] = None,
               timeout_s: float = 90.0) -> Optional[Dict[str, Any]]:
    """Render ``text`` to speech. Returns ``{audio, rate, voice, chars}`` (audio is
    base64 s16le PCM), or None on any failure."""
    global _calls
    text = (text or "").strip()
    if not text or not available():
        return None
    if len(text) > _MAX_CHARS:
        # Cut at a sentence end rather than mid-word: a report that stops
        # halfway through a clause sounds broken in a way silence does not.
        cut = text[:_MAX_CHARS]
        stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[:stop + 1] if stop > 200 else cut

    v = v or voice()
    key = _key(text, v)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        _CACHE.move_to_end(key)
        return {"audio": hit[1], "rate": OUT_RATE, "voice": v,
                "chars": len(text), "cached": True}
    if _calls >= _MAX_CALLS:
        return None

    try:
        r = httpx.post(
            f"{_BASE}/models/{_MODEL}:generateContent",
            headers={"x-goog-api-key": os.getenv("GEMINI_API_KEY")
                     or os.getenv("GOOGLE_API_KEY") or ""},
            json={"contents": [{"parts": [{"text": text}]}],
                  "generationConfig": {
                      "responseModalities": ["AUDIO"],
                      "speechConfig": {"voiceConfig": {
                          "prebuiltVoiceConfig": {"voiceName": v}}}}},
            timeout=timeout_s)
        data = r.json()
        part = (((data.get("candidates") or [{}])[0].get("content") or {})
                .get("parts") or [{}])[0]
        audio = (part.get("inlineData") or {}).get("data")
        if r.status_code != 200 or not audio:
            print(f"[speech] unavailable ({r.status_code}): "
                  f"{str(data.get('error', {}).get('message', ''))[:110]}")
            return None
    except Exception as exc:                        # pragma: no cover
        print(f"[speech] failed: {type(exc).__name__}: {exc}")
        return None

    _calls += 1
    _CACHE[key] = (time.time(), audio)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return {"audio": audio, "rate": OUT_RATE, "voice": v,
            "chars": len(text), "cached": False}


def status() -> Dict[str, Any]:
    return {"available": available(), "voice": voice(), "model": _MODEL,
            "calls": _calls, "budget_remaining": budget_remaining(),
            "cached": len(_CACHE), "rate": OUT_RATE}
