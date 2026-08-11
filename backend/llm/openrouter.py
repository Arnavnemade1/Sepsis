"""Resilient, rate-limited OpenRouter chat client.

The account is on OpenRouter's FREE tier, where any single model is aggressively
rate-limited (429). A one-model client is therefore barely usable. This client is
built to squeeze real capability out of that constraint without ever abusing it:

  * **Multi-model fallback chain** — on a 429/error from one free model, fall
    straight through to the next. (Verified live: the default Llama returns 429
    while gpt-oss-120b / gemma answer fine, and vice-versa minutes later.)
  * **Per-model cooldown** — a model that 429s is skipped for a cooldown window
    so we stop hammering it; the chain routes around it automatically.
  * **Response cache** (TTL, bounded) — identical prompts don't spend budget or
    hit the network; a repeated narrative for an unchanged state is free.
  * **Hard budget + min interval** — a ceiling on calls and a floor on spacing so
    a realtime loop can never run the account into the ground.
  * **Always-safe failure** — any exhaustion/failure returns None so every caller
    falls back to a deterministic template. The system never depends on the LLM.

Design constraints kept from the original: the key is read ONLY from the
environment (backend/.env), never hardcoded; failures are silent-and-safe.

The API (`available`, `budget_remaining`, `chat`) is backward-compatible; the
transport is injectable (`post_fn`) so the logic can be tested offline without
spending a cent of the free budget.
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

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# A fallback chain of strong, general free instruct models. Ordered so a capable
# model leads and lighter/faster ones backstop it. Overridable via env
# OPENROUTER_MODELS (comma-separated); OPENROUTER_MODEL still sets the primary.
_DEFAULT_CHAIN = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
]

# HTTP response type from a transport: (status_code, json_or_none).
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


class OpenRouterClient:
    provider = "openrouter"

    def __init__(
        self,
        model: Optional[str] = None,
        models: Optional[List[str]] = None,
        max_calls: int = 40,
        min_interval_s: float = 8.0,
        timeout_s: float = 20.0,
        cooldown_s: float = 45.0,
        cache_ttl_s: float = 300.0,
        cache_size: int = 128,
        post_fn: Optional[PostFn] = None,
    ):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        # Build the model chain: explicit arg > env list > env single + default chain.
        # An explicit `models` list wins outright (env primary is not merged in).
        chain = models or self._env_chain()
        primary = model or (os.getenv("OPENROUTER_MODEL") if models is None else None)
        if primary and primary in chain:
            chain = [primary] + [m for m in chain if m != primary]
        elif primary:
            chain = [primary] + chain
        seen: set = set()
        self.models: List[str] = [m for m in chain if not (m in seen or seen.add(m))]
        self.model = self.models[0] if self.models else None      # back-compat attr

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
        self._cooldown: Dict[str, float] = {}                     # model -> until ts
        self._stats: Dict[str, Dict[str, int]] = {}               # model -> {ok,429,err}
        self._cache: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()

    # -- config helpers --------------------------------------------------- #
    @staticmethod
    def _env_chain() -> List[str]:
        raw = os.getenv("OPENROUTER_MODELS")
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
        h = hashlib.sha256(f"{max_tokens}\x00{system}\x00{user}".encode("utf-8"))
        return h.hexdigest()

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
        """Return assistant text, or None if unavailable / budgeted-out / failed.

        Walks the model chain, skipping cooled-down models, until one answers.
        Identical prompts are served from cache (free). ``respect_interval=False``
        lets a *user-initiated* call (e.g. the console) bypass the realtime spacing
        floor while still honouring the hard budget."""
        if not self.available:
            return None
        cache_key = self._key(system, user, max_tokens)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        now = time.time()
        if self._calls >= self.max_calls:
            return None
        if respect_interval and (now - self._last_call) < self.min_interval_s:
            return None                                    # too soon — caller templates

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Arnavnemade1/Sepsis",
            "X-Title": "HyperSepsisTwin-Omega",
        }
        for model in self._live_models(now):
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            try:
                status, data = self._post(_ENDPOINT, headers, payload, self.timeout_s)
            except Exception as exc:                        # network/timeout -> next model
                self._bump(model, "err")
                self._last_err = str(exc)
                continue
            if status == 429:
                self._bump(model, "429")
                self._cooldown[model] = time.time() + self.cooldown_s
                continue
            text = self._extract(data)
            if status == 200 and text:
                self._bump(model, "ok")
                self._calls += 1
                self._last_call = time.time()
                self._last_model = model
                self._cache_put(cache_key, text)
                return text
            self._bump(model, "err")                        # 4xx/5xx or empty body
        return None                                         # chain exhausted -> template

    @staticmethod
    def _extract(data: Optional[Dict[str, Any]]) -> Optional[str]:
        if not data:
            return None
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    # -- telemetry (surfaced at /proc/llm) -------------------------------- #
    def status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "available": self.available,
            "primary_model": self.model,
            "chain": self.models,
            "calls": self._calls,
            "budget_remaining": self.budget_remaining(),
            "max_calls": self.max_calls,
            "cache_hits": self._cache_hits,
            "cache_entries": len(self._cache),
            "last_model": self._last_model,
            "cooling": sorted(m for m, t in self._cooldown.items() if t > now),
            "per_model": self._stats,
        }
