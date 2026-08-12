"""Server-side runtime config, with a Supabase fallback for hosted deploys.

The AI-doctor voice consult relays through the backend precisely so the Gemini
credential never reaches the browser. That design has one consequence: the
credential must exist *on the server*. On a host where the environment variables
cannot be edited, voice is simply off — which is what the deployed instance was
reporting.

So: environment first, always. If a key is absent from the environment, look it
up once in ``public.runtime_config`` using the service-role Supabase client the
backend already holds. That table has RLS enabled and no policies, so it is
unreachable by anon/authenticated roles by construction, and the browser in this
project never speaks to Supabase at all.

The env var ALWAYS wins. Setting the variable on the host overrides this table
without a code change, so the fallback never becomes a shadow configuration that
disagrees with the platform and is hard to reason about.

Failure is silent and safe: no Supabase, no table, no row, no network — the
lookup returns None and the caller behaves exactly as it did before, which for
the consult means voice stays off and says why.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, Optional

# Load backend/.env here rather than relying on some other module having imported
# first. This module runs EARLY — before the LLM clients that also call
# load_dotenv — so without it, local runs see an empty environment at prime time,
# find no Supabase credentials either, and report "unset" for a key that is in
# fact configured. Same code path locally and on a host, which is the point.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:  # pragma: no cover
    pass

_CACHE: Dict[str, Optional[str]] = {}
# Where each name actually came from. Tracked separately because `prime` writes
# the recovered value INTO the environment, which would otherwise make every
# fallback report itself as "environment" — erasing the one fact the diagnostic
# exists to tell you.
_ORIGIN: Dict[str, str] = {}
_LOCK = threading.Lock()
_TABLE = "runtime_config"


def get(name: str, default: Optional[str] = None) -> Optional[str]:
    """Environment first; Supabase ``runtime_config`` second; ``default`` last."""
    env = os.getenv(name)
    if env:
        _ORIGIN.setdefault(name, "environment")
        return env
    with _LOCK:
        if name in _CACHE:                     # includes cached misses
            return _CACHE[name] or default
        val = _fetch(name)
        _CACHE[name] = val
        _ORIGIN[name] = "supabase:runtime_config" if val else "unset"
    return val or default


def _fetch(name: str) -> Optional[str]:
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not (url and key):
        return None
    try:
        from supabase import create_client
        client = create_client(url, key)
        res = (client.table(_TABLE).select("value").eq("key", name)
               .limit(1).execute())
        rows = getattr(res, "data", None) or []
        if rows and rows[0].get("value"):
            print(f"[config] {name} loaded from Supabase runtime_config "
                  f"(not set in the environment)")
            return str(rows[0]["value"])
    except Exception as exc:                   # pragma: no cover — never fatal
        print(f"[config] runtime_config lookup for {name} unavailable: "
              f"{type(exc).__name__}")
    return None


def source(name: str) -> str:
    """Where a value actually came from — surfaced in status so a deploy can be
    diagnosed without guessing. Reports the true origin even after ``prime`` has
    copied a recovered value into the environment."""
    if name in _ORIGIN:
        return _ORIGIN[name]
    return "environment" if os.getenv(name) else "unset"


def prime(name: str) -> Optional[str]:
    """Resolve a name and publish it into the process environment.

    Everything downstream (``GeminiClient``, ``live_url``, ``LiveConsult``) reads
    ``os.getenv`` directly and is shared with offline verification, so the
    cleanest injection point is the environment itself — one lookup at boot
    rather than a new indirection threaded through every call site."""
    val = get(name)
    if val and not os.getenv(name):
        os.environ[name] = val
    return val
