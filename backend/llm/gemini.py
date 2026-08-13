"""Resilient Gemini chat client — the PRIMARY provider for the LLM plane.

This mirrors :class:`backend.llm.openrouter.OpenRouterClient` field-for-field
(``available`` / ``budget_remaining`` / ``chat`` / ``status`` / ``_last_model``)
so the two are interchangeable and :class:`backend.llm.provider.LLMProvider` can
fail over between them without any caller knowing which one answered.

Why Gemini leads: it is the only provider wired here that does realtime
bidirectional audio (``bidiGenerateContent``), which is what the AI-doctor
consult call runs on. Keeping the *text* path on the same vendor means the
doctor sounds like one clinician whether you are typing or talking to it.

Three things this client knows that a naive one does not:

  * **Auth is a header, not a query param.** ``?key=`` returns a bodyless 404 on
    ``:generateContent`` for this account; ``x-goog-api-key`` works. Verified live.
  * **Thinking silently eats the answer.** Gemini 2.5/3.x spend ``maxOutputTokens``
    on internal reasoning *before* emitting prose, so a 200-token cap returns
    ``finishReason: MAX_TOKENS`` and ~18 characters of text. Verified live on
    both 2.5-flash and 3.5-flash. We send ``thinkingBudget: 0`` — with that, the
    same cap returns a complete answer. A model that rejects the field is retried
    once without it (older/newer surfaces disagree on the schema).
  * **A truncated answer is a failed answer.** ``MAX_TOKENS`` with a stub body
    falls through to the next model instead of handing the caller half a sentence.

Everything else is the resilience contract the OpenRouter client established:
multi-model fallback, per-model cooldown, bounded TTL cache, hard call budget,
minimum spacing, and always-safe failure (``None`` → the caller templates).
The key is read ONLY from ``backend/.env`` and never leaves the server.
"""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

try:
    from dotenv import load_dotenv
    _ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_ENV_PATH)
except Exception:  # pragma: no cover
    pass

_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Ordered strongest-first; each is a live-verified `generateContent` model on this
# key. Overridable via GEMINI_MODELS (comma-separated); GEMINI_MODEL sets primary.
_DEFAULT_CHAIN = [
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# (status_code, json_or_none) — injectable so the logic is testable without spend.
PostFn = Callable[[str, Dict[str, str], Dict[str, Any], float], Tuple[int, Optional[Dict[str, Any]]]]


def _default_post(url: str, headers: Dict[str, str], payload: Dict[str, Any],
                  timeout: float) -> Tuple[int, Optional[Dict[str, Any]]]:
    if httpx is None:  # pragma: no cover
        return 0, None
    resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, None


class GeminiClient:
    provider = "gemini"

    def __init__(
        self,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
        max_calls: int = 200,
        min_interval_s: float = 2.0,
        timeout_s: float = 45.0,
        cooldown_s: float = 45.0,
        cache_ttl_s: float = 300.0,
        cache_size: int = 128,
        post_fn: Optional[PostFn] = None,
    ):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        chain = models or self._env_chain()
        primary = model or (os.getenv("GEMINI_MODEL") if models is None else None)
        if primary and primary in chain:
            chain = [primary] + [m for m in chain if m != primary]
        elif primary:
            chain = [primary] + chain
        seen: set = set()
        self.models: List[str] = [m for m in chain if not (m in seen or seen.add(m))]
        self.model = self.models[0] if self.models else None

        self.max_calls = max_calls
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.cooldown_s = cooldown_s
        self.cache_ttl_s = cache_ttl_s
        self.cache_size = cache_size
        self._post = post_fn or _default_post

        self._calls = 0
        self._cache_hits = 0
        self._last_call = 0.0
        self._last_model: Optional[str] = None
        self._last_err: Optional[str] = None
        self._cooldown: Dict[str, float] = {}
        self._stats: Dict[str, Dict[str, int]] = {}
        self._no_thinking_cfg: set = set()          # models that reject thinkingBudget
        self._cache: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()

    # -- config ----------------------------------------------------------- #
    @staticmethod
    def _env_chain() -> List[str]:
        raw = os.getenv("GEMINI_MODELS")
        if raw:
            got = [m.strip() for m in raw.split(",") if m.strip()]
            if got:
                return got
        return list(_DEFAULT_CHAIN)

    @property
    def available(self) -> bool:
        return bool(self.api_key) and (httpx is not None or self._post is not _default_post)

    def budget_remaining(self) -> int:
        return max(0, self.max_calls - self._calls)

    # -- cache ------------------------------------------------------------ #
    @staticmethod
    def _key(system: str, user: str, max_tokens: int) -> str:
        return hashlib.sha256(
            f"{max_tokens}\x00{system}\x00{user}".encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        item = self._cache.get(key)
        if item is None:
            return None
        ts, val = item
        if time.time() - ts > self.cache_ttl_s:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return val

    def _cache_put(self, key: str, val: str) -> None:
        self._cache[key] = (time.time(), val)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _bump(self, model: str, field: str) -> None:
        self._stats.setdefault(model, {"ok": 0, "429": 0, "err": 0})[field] += 1

    def _live_models(self, now: float) -> List[str]:
        return [m for m in self.models if self._cooldown.get(m, 0.0) <= now]

    # -- the call --------------------------------------------------------- #
    def chat(self, system: str, user: str, max_tokens: int = 350,
             temperature: float = 0.2, respect_interval: bool = True) -> Optional[str]:
        """Single-turn convenience wrapper (the shared provider contract)."""
        return self.converse(system, [{"role": "user", "text": user}],
                             max_tokens=max_tokens, temperature=temperature,
                             respect_interval=respect_interval)

    def converse(self, system: str, turns: List[Dict[str, str]], max_tokens: int = 350,
                 temperature: float = 0.2, respect_interval: bool = True) -> Optional[str]:
        """Multi-turn chat. ``turns`` is ``[{'role': 'user'|'model', 'text': ...}]``.

        Returns assistant text, or None if unavailable / budgeted-out / failed —
        every caller must have a deterministic fallback for None."""
        if not self.available or not turns:
            return None
        cache_key = self._key(system, repr(turns), max_tokens)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        now = time.time()
        if self._calls >= self.max_calls:
            return None
        if respect_interval and (now - self._last_call) < self.min_interval_s:
            return None

        contents = [{"role": ("model" if t.get("role") == "model" else "user"),
                     "parts": [{"text": str(t.get("text", ""))}]} for t in turns]
        headers = {"x-goog-api-key": self.api_key or "", "Content-Type": "application/json"}

        for model in self._live_models(now):
            for attempt in (0, 1):
                payload: Dict[str, Any] = {
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature,
                    },
                }
                if system:
                    payload["systemInstruction"] = {"parts": [{"text": system}]}
                # Thinking off: it otherwise consumes the whole output budget and
                # returns a truncated stub. Skipped on models known to reject it.
                if attempt == 0 and model not in self._no_thinking_cfg:
                    payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
                url = f"{_BASE}/models/{model}:generateContent"
                try:
                    status, data = self._post(url, headers, payload, self.timeout_s)
                except Exception as exc:
                    self._bump(model, "err")
                    self._last_err = str(exc)
                    break                               # transport down → next model
                if status == 400 and attempt == 0 and self._rejects_thinking(data):
                    self._no_thinking_cfg.add(model)    # retry this model, no cfg
                    continue
                if status in (429, 503):
                    self._bump(model, "429")
                    self._cooldown[model] = time.time() + self.cooldown_s
                    break
                text, truncated = self._extract(data)
                if status == 200 and text and not truncated:
                    self._bump(model, "ok")
                    self._calls += 1
                    self._last_call = time.time()
                    self._last_model = model
                    self._cache_put(cache_key, text)
                    return text
                self._bump(model, "err")
                self._last_err = self._err_msg(data, status, truncated)
                break                                   # 4xx/5xx/empty/truncated
        return None

    # -- agentic turn (function calling) ---------------------------------- #
    def generate(self, system: str, contents: List[Dict[str, Any]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 max_tokens: int = 700, temperature: float = 0.6
                 ) -> Optional[Dict[str, Any]]:
        """One raw turn that may come back as TEXT or as FUNCTION CALLS.

        ``contents`` is Gemini-native (roles + parts), because a tool-using
        conversation carries functionCall/functionResponse parts that the simple
        ``{role, text}`` shape cannot represent. Returns
        ``{text, calls, model}`` or None. Deliberately uncached: a tool call is
        an action, and replaying an action from cache would silently repeat it."""
        if not self.available:
            return None
        if self._calls >= self.max_calls:
            return None
        headers = {"x-goog-api-key": self.api_key or "", "Content-Type": "application/json"}
        now = time.time()
        for model in self._live_models(now):
            for attempt in (0, 1):
                payload: Dict[str, Any] = {
                    "contents": contents,
                    "generationConfig": {"maxOutputTokens": max_tokens,
                                         "temperature": temperature},
                }
                if system:
                    payload["systemInstruction"] = {"parts": [{"text": system}]}
                if tools:
                    payload["tools"] = [{"functionDeclarations": tools}]
                if attempt == 0 and model not in self._no_thinking_cfg:
                    payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
                try:
                    status, data = self._post(
                        f"{_BASE}/models/{model}:generateContent", headers,
                        payload, self.timeout_s)
                except Exception as exc:
                    self._bump(model, "err")
                    self._last_err = str(exc)
                    break
                if status == 400 and attempt == 0 and self._rejects_thinking(data):
                    self._no_thinking_cfg.add(model)
                    continue
                if status in (429, 503):
                    self._bump(model, "429")
                    self._cooldown[model] = time.time() + self.cooldown_s
                    break
                if status == 200 and data:
                    parts = (((data.get("candidates") or [{}])[0]
                              .get("content") or {}).get("parts") or [])
                    calls = [p["functionCall"] for p in parts if p.get("functionCall")]
                    text = "".join(p.get("text", "") for p in parts
                                   if not p.get("thought")).strip()
                    if calls or text:
                        self._bump(model, "ok")
                        self._calls += 1
                        self._last_call = time.time()
                        self._last_model = model
                        return {"text": text, "calls": calls, "model": model}
                self._bump(model, "err")
                self._last_err = self._err_msg(data, status, False)
                break
        return None

    @staticmethod
    def _rejects_thinking(data: Optional[Dict[str, Any]]) -> bool:
        msg = ((data or {}).get("error") or {}).get("message", "")
        return "hinking" in msg          # "Thinking ... not supported for this model"

    @staticmethod
    def _err_msg(data: Optional[Dict[str, Any]], status: int, truncated: bool) -> str:
        if truncated:
            return f"{status} truncated (MAX_TOKENS with stub body)"
        return f"{status} {((data or {}).get('error') or {}).get('message', 'empty body')}"[:200]

    @staticmethod
    def _extract(data: Optional[Dict[str, Any]]) -> Tuple[Optional[str], bool]:
        """Return (text, truncated). Text excludes 'thought' parts; a MAX_TOKENS
        finish with a stub body counts as truncated so the chain keeps walking."""
        if not data:
            return None, False
        try:
            cand = (data.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            truncated = cand.get("finishReason") == "MAX_TOKENS" and len(text) < 80
            return (text or None), truncated
        except Exception:
            return None, False

    # -- telemetry (surfaced at /proc/llm) -------------------------------- #
    def status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "provider": self.provider,
            "available": self.available,
            "primary_model": self.model,
            "chain": self.models,
            "calls": self._calls,
            "budget_remaining": self.budget_remaining(),
            "max_calls": self.max_calls,
            "cache_hits": self._cache_hits,
            "cache_entries": len(self._cache),
            "last_model": self._last_model,
            "last_error": self._last_err,
            "cooling": sorted(m for m, t in self._cooldown.items() if t > now),
            "per_model": self._stats,
        }


# -- realtime (bidiGenerateContent) config ------------------------------- #
def live_model() -> str:
    return os.getenv("GEMINI_LIVE_MODEL", "models/gemini-3.1-flash-live-preview")


def live_voice() -> str:
    return os.getenv("GEMINI_LIVE_VOICE", "Charon")


def live_url() -> Optional[str]:
    """WebSocket URL for the Live API, authenticated with the server-side key.

    Ephemeral tokens (``auth_tokens.create``) mint on this account but are
    rejected by the Live socket, so the browser can never hold a credential of
    its own — the backend relays instead, which is why this URL must stay
    server-side. Returns None when no key is configured."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    return ("wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService."
            f"BidiGenerateContent?key={key}")
