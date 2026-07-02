"""Offline verification of the resilient LLM layer (spends ZERO free-tier budget).

Every network call is mocked, so this is deterministic and free. It proves the
capability the free tier actually needs:

  * multi-model FALLBACK: a 429 on the lead model routes to the next
  * per-model COOLDOWN: a 429'd model is skipped on the next call
  * response CACHE: an identical prompt is served without touching the transport
  * hard BUDGET: calls past max_calls return None (-> template)
  * interval gating: realtime spacing floor vs user-initiated bypass
  * empty/garbled body -> fall through, never crash
  * AegisAssistant grounding + deterministic template fallback

An optional LIVE smoke test runs only if AEGIS_LLM_LIVE=1 (one real call).

Run:  .venv/bin/python -m backend.llm.verify_llm
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from backend.llm.assistant import AegisAssistant
from backend.llm.openrouter import OpenRouterClient


class MockTransport:
    """Programmable transport: map model -> a queue of (status, body) responses."""
    def __init__(self, script: Dict[str, List[Tuple[int, Optional[dict]]]],
                 default: Tuple[int, Optional[dict]] = (200, None)):
        self.script = {m: list(v) for m, v in script.items()}
        self.default = default
        self.calls: List[str] = []

    def __call__(self, url, headers, payload, timeout):
        model = payload["model"]
        self.calls.append(model)
        q = self.script.get(model)
        if q:
            return q.pop(0)
        return self.default


def _ok(text: str) -> Tuple[int, dict]:
    return 200, {"choices": [{"message": {"content": text}}]}


CHAIN = ["model-A", "model-B", "model-C"]


def _client(transport, **kw) -> OpenRouterClient:
    opts = dict(min_interval_s=0.0, max_calls=40, cooldown_s=1e9)
    opts.update(kw)
    c = OpenRouterClient(models=list(CHAIN), post_fn=transport, **opts)
    c.api_key = "test-key"                      # force available without a real key
    return c


def check_fallback() -> None:
    print("[1] Multi-model fallback: 429 on lead -> next model answers")
    t = MockTransport({"model-A": [(429, None)], "model-B": [_ok("from B")]})
    c = _client(t)
    out = c.chat("sys", "hello")
    assert out == "from B", out
    assert t.calls == ["model-A", "model-B"], t.calls
    assert c._last_model == "model-B"
    print(f"      [PASS] routed A(429)->B; answer={out!r}; last_model={c._last_model}")


def check_cooldown() -> None:
    print("[2] Per-model cooldown: a 429'd model is skipped next call")
    t = MockTransport({"model-A": [(429, None)], "model-B": [_ok("B1"), _ok("B2")]})
    c = _client(t)
    assert c.chat("s", "q1") == "B1"
    # second call: model-A is cooling, should go straight to B (not retried)
    assert c.chat("s", "q2") == "B2"
    assert t.calls == ["model-A", "model-B", "model-B"], t.calls
    assert "model-A" in c.status()["cooling"]
    print(f"      [PASS] model-A cooled after 429; 2nd call skipped it; cooling={c.status()['cooling']}")


def check_cache() -> None:
    print("[3] Response cache: identical prompt served free (no transport hit)")
    t = MockTransport({"model-A": [_ok("cached-answer")]})
    c = _client(t)
    a = c.chat("s", "same question")
    b = c.chat("s", "same question")            # identical -> cache
    assert a == b == "cached-answer"
    assert len(t.calls) == 1, t.calls          # only ONE network call
    assert c._cache_hits == 1 and c._calls == 1
    print(f"      [PASS] 2 asks, 1 network call, {c._cache_hits} cache hit, budget spent={c._calls}")


def check_budget() -> None:
    print("[4] Hard budget: calls past max_calls return None (-> template)")
    t = MockTransport({"model-A": [_ok("x"), _ok("y"), _ok("z")]})
    c = _client(t, max_calls=2)
    assert c.chat("s", "u1") == "x"
    assert c.chat("s", "u2") == "y"
    assert c.chat("s", "u3") is None           # budget exhausted
    assert c.budget_remaining() == 0
    print(f"      [PASS] max_calls=2 enforced; 3rd call -> None; budget_remaining=0")


def check_interval_and_bypass() -> None:
    print("[5] Interval gating: realtime spacing vs user-initiated bypass")
    t = MockTransport({"model-A": [_ok("first"), _ok("second")]})
    c = _client(t, min_interval_s=1e9)         # effectively 'never again soon'
    assert c.chat("s", "a") == "first"
    assert c.chat("s", "b", respect_interval=True) is None       # blocked by interval
    assert c.chat("s", "c", respect_interval=False) == "second"  # user bypass
    print(f"      [PASS] realtime call blocked by interval; user-initiated bypassed it")


def check_bad_body() -> None:
    print("[6] Garbled/empty body -> fall through to next, never crash")
    t = MockTransport({"model-A": [(200, {"unexpected": 1})], "model-B": [_ok("recovered")]})
    c = _client(t)
    assert c.chat("s", "q") == "recovered"
    assert c._stats["model-A"]["err"] == 1
    print(f"      [PASS] empty choices on A handled; recovered on B")


def check_all_down() -> None:
    print("[7] Whole chain down -> None (caller uses deterministic template)")
    t = MockTransport({m: [(429, None)] for m in CHAIN})
    c = _client(t)
    assert c.chat("s", "q") is None
    print(f"      [PASS] all 429 -> None; per_model={ {m: c._stats[m]['429'] for m in CHAIN} }")


def check_assistant() -> None:
    print("[8] AegisAssistant: grounded answer + template fallback")
    state = {"topology": [
        {"node_id": "bed_0", "name": "Bed 01", "tier": "CRITICAL", "p": 0.82},
        {"node_id": "bed_1", "name": "Bed 02", "tier": "WATCH", "p": 0.44},
        {"node_id": "bed_2", "name": "Bed 03", "tier": "STABLE", "p": 0.10}]}
    # LLM path (mocked): assistant should pass grounded state into the prompt
    seen = {}
    def spy(url, headers, payload, timeout):
        seen["user"] = payload["messages"][1]["content"]
        return _ok("Bed 01 is critical (p=0.82).")
    a = AegisAssistant(_client(spy))
    out = a.ask("which beds are critical?", state)
    assert out["source"] == "llm" and "critical" in out["answer"].lower()
    assert "bed_0" in seen["user"] and "CRITICAL" in seen["user"], "state not grounded into prompt"
    # template path (LLM unavailable): still answers from the same state
    a2 = AegisAssistant(OpenRouterClient(models=CHAIN, post_fn=MockTransport({})))
    a2.client.api_key = None                    # force unavailable
    tmpl = a2.ask("which beds are critical?", state)
    assert tmpl["source"] == "template" and "1 critical" in tmpl["answer"]
    print(f"      [PASS] grounded LLM answer + template fallback both cite real beds")


def live_smoke() -> None:
    if os.getenv("AEGIS_LLM_LIVE") != "1":
        print("[9] Live smoke: SKIPPED (set AEGIS_LLM_LIVE=1 to spend 1 real call)")
        return
    print("[9] Live smoke: one real call across the fallback chain")
    c = OpenRouterClient(min_interval_s=0.0)
    r = c.chat("Reply with exactly: OK", "ping", max_tokens=5)
    print(f"      result={r!r} via {c._last_model}; status={c.status()['per_model']}")


def run() -> int:
    print("=" * 66)
    print("AegisOS — resilient LLM layer verification (offline, zero budget)")
    print("=" * 66)
    check_fallback()
    check_cooldown()
    check_cache()
    check_budget()
    check_interval_and_bypass()
    check_bad_body()
    check_all_down()
    check_assistant()
    live_smoke()
    print("-" * 66)
    print("ALL LLM RESILIENCE CHECKS PASSED ✓  (fallback · cooldown · cache · budget)")
    print("=" * 66)
    return 0


def main() -> None:
    run()


if __name__ == "__main__":
    sys.exit(run())
