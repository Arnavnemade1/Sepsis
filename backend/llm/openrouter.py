"""Minimal, rate-limited OpenRouter chat client.

Design constraints (deliberate):
  * The API key is read ONLY from the environment (backend/.env), never hardcoded.
  * Hard call budget + minimum interval so the free model is used sparingly and
    we never hammer the endpoint or block the realtime loop.
  * Any failure returns None so callers fall back to a deterministic template —
    the system never depends on the LLM being available.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

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


class OpenRouterClient:
    def __init__(
        self,
        model: Optional[str] = None,
        max_calls: int = 25,
        min_interval_s: float = 30.0,
        timeout_s: float = 20.0,
    ):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        self.max_calls = max_calls
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self._calls = 0
        self._last_call = 0.0

    @property
    def available(self) -> bool:
        return bool(self.api_key) and httpx is not None

    def budget_remaining(self) -> int:
        return max(0, self.max_calls - self._calls)

    def chat(self, system: str, user: str, max_tokens: int = 350,
             temperature: float = 0.2) -> Optional[str]:
        """Return assistant text, or None if unavailable / budgeted-out / failed."""
        if not self.available:
            return None
        now = time.time()
        if self._calls >= self.max_calls:
            return None
        if now - self._last_call < self.min_interval_s:
            return None  # too soon — caller uses the template this time

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers (optional but recommended).
            "HTTP-Referer": "https://github.com/Arnavnemade1/Sepsis",
            "X-Title": "HyperSepsisTwin-Omega",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            resp = httpx.post(_ENDPOINT, headers=headers, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            self._calls += 1
            self._last_call = now
            return text
        except Exception as exc:  # pragma: no cover
            print(f"[llm] OpenRouter call failed ({exc}); using template fallback.")
            return None
