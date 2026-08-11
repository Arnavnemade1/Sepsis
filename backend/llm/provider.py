"""LLMProvider — one text endpoint over two vendors, so a quota wall is not an outage.

AegisOS has exactly one rule about the LLM: **nothing may depend on it.** Every
caller already falls back to a deterministic template when ``chat`` returns None.
This class widens the gap before that happens by putting a second vendor behind
the first:

    Gemini (primary)  ──429/exhausted──▶  OpenRouter · Nemotron 3 Ultra (backup)
                                                   │
                                                   └── both down ──▶ None → template

It is a drop-in for either underlying client: same ``available`` /
``budget_remaining`` / ``chat`` / ``status`` / ``_last_model`` surface, so
``AegisAssistant``, ``PatientTerminal`` and ``AlertExplainer`` gained failover
without changing a line.

Ordering is deliberate, not arbitrary. Gemini leads because the AI-doctor consult
call runs on Gemini's realtime audio socket, and a consult that is voiced by one
model and typed by another is two different clinicians. The backup is text-only —
it keeps the *narrative* layer alive when Gemini is walled, and the UI says which
vendor answered rather than pretending the two are the same.

``prefer=`` lets a caller pin a vendor (the verify suites use it); the default
walks the chain.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from backend.llm.gemini import GeminiClient
from backend.llm.openrouter import OpenRouterClient


class LLMProvider:
    provider = "multi"

    def __init__(self, gemini: Optional[GeminiClient] = None,
                 openrouter: Optional[OpenRouterClient] = None,
                 prefer: Optional[str] = None):
        self.gemini = gemini if gemini is not None else GeminiClient()
        self.openrouter = openrouter if openrouter is not None else OpenRouterClient()
        self.prefer = (prefer or os.getenv("LLM_PREFER") or "gemini").lower()
        self._last_model: Optional[str] = None
        self._last_provider: Optional[str] = None

    # -- chain ------------------------------------------------------------ #
    def _chain(self) -> List[Any]:
        order = [self.gemini, self.openrouter]
        if self.prefer.startswith("open") or self.prefer.startswith("nemo"):
            order.reverse()
        return [c for c in order if c is not None]

    @property
    def available(self) -> bool:
        return any(c.available for c in self._chain())

    def budget_remaining(self) -> int:
        return sum(c.budget_remaining() for c in self._chain() if c.available)

    # -- the call --------------------------------------------------------- #
    def chat(self, system: str, user: str, max_tokens: int = 350,
             temperature: float = 0.2, respect_interval: bool = True) -> Optional[str]:
        for client in self._chain():
            if not client.available:
                continue
            text = client.chat(system, user, max_tokens=max_tokens,
                               temperature=temperature, respect_interval=respect_interval)
            if text:
                self._record(client)
                return text
        return None

    def converse(self, system: str, turns: List[Dict[str, str]], max_tokens: int = 350,
                 temperature: float = 0.2, respect_interval: bool = True) -> Optional[str]:
        """Multi-turn. Clients without a native multi-turn path (OpenRouter here)
        get the conversation flattened into one prompt rather than losing it."""
        for client in self._chain():
            if not client.available:
                continue
            if hasattr(client, "converse"):
                text = client.converse(system, turns, max_tokens=max_tokens,
                                       temperature=temperature,
                                       respect_interval=respect_interval)
            else:
                text = client.chat(system, self._flatten(turns), max_tokens=max_tokens,
                                   temperature=temperature,
                                   respect_interval=respect_interval)
            if text:
                self._record(client)
                return text
        return None

    @staticmethod
    def _flatten(turns: List[Dict[str, str]]) -> str:
        lines = []
        for t in turns:
            who = "DOCTOR" if t.get("role") == "model" else "CLINICIAN"
            lines.append(f"{who}: {t.get('text', '')}")
        lines.append("DOCTOR:")
        return "\n\n".join(lines)

    def _record(self, client: Any) -> None:
        self._last_provider = getattr(client, "provider", client.__class__.__name__)
        self._last_model = getattr(client, "_last_model", None)

    # -- telemetry (surfaced at /proc/llm) -------------------------------- #
    def status(self) -> Dict[str, Any]:
        g, o = self.gemini.status(), self.openrouter.status()
        return {
            "provider": "multi",
            "prefer": self.prefer,
            "available": self.available,
            "last_provider": self._last_provider,
            "last_model": self._last_model,
            "budget_remaining": self.budget_remaining(),
            # back-compat: /proc/llm consumers read these top-level fields
            "primary_model": g.get("primary_model") if self.gemini.available else o.get("primary_model"),
            "calls": int(g.get("calls", 0)) + int(o.get("calls", 0)),
            "cache_hits": int(g.get("cache_hits", 0)) + int(o.get("cache_hits", 0)),
            "gemini": g,
            "openrouter": o,
        }
