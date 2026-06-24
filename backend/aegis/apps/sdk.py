"""The AegisOS application SDK — the contract every app runs under.

AegisOS already verifies *drivers* (`kernel/ebpf.py`) and certifies *outputs*
(`kernel/neurosym.py`). This generalizes the same default-deny philosophy to the
computation in between: a **diagnostic application**. An app declares, up front,
exactly what it is allowed to touch — which sensor channels it may read, which
egress classes it may emit, how much compute it may spend — and the kernel hands
it an :class:`AppContext` that is the *only* path to those services. Any access
the manifest did not declare raises :class:`CapabilityViolation`. There is no
ambient authority.

This is a **reference-monitor** enforced capability boundary, stated honestly:
the app receives no handle to the kernel except the context, so it cannot read an
undeclared channel or emit an undeclared egress class through the supported API.
A *bytecode* app additionally inherits the eBPF static guarantee; a *Python* app
is trusted-but-confined (capability-mediated context + the install-time AST checks
added in the package phase) — it is not formally memory-isolated. We claim only
what is enforced.

The result: input (drivers), computation (apps) and output (actions) are all
capability-confined and formally gated, and apps become a uniform, portable unit
a third party can write against.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.aegis.kernel.neurosym import (ACTUATOR, ALERT, NOTE, PROTOCOL,
                                           NeuroSymbolicRuntime, SafetyCertificate)

EGRESS_CLASSES = (ALERT, PROTOCOL, ACTUATOR, NOTE)
LANE_CLASSES = ("safety", "ml_tracking", "render", "logging")


class CapabilityViolation(Exception):
    """An app tried to use authority its manifest never declared (default-deny)."""


# --------------------------------------------------------------------------- #
@dataclass
class AppManifest:
    """The capabilities an app declares — and is then held to. This is the unit
    the package registry verifies and the service manager schedules."""
    name: str
    version: str = "0.1.0"
    author: str = "unknown"
    summary: str = ""
    reads: Tuple[str, ...] = ()            # sensor channels the app may read
    emits: Tuple[str, ...] = ()            # egress classes the app may emit
    lane_class: str = "ml_tracking"        # scheduler class it runs in
    lane_budget: int = 1                   # compute lanes it may request
    depends: Tuple[str, ...] = ()          # other services it needs running first

    def validate(self) -> List[str]:
        errs: List[str] = []
        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            errs.append(f"bad app name {self.name!r}")
        for e in self.emits:
            if e not in EGRESS_CLASSES:
                errs.append(f"unknown egress class {e!r} (allowed: {EGRESS_CLASSES})")
        if self.lane_class not in LANE_CLASSES:
            errs.append(f"unknown lane class {self.lane_class!r}")
        if self.lane_budget < 0:
            errs.append("lane_budget must be >= 0")
        return errs

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "author": self.author,
                "summary": self.summary, "reads": list(self.reads),
                "emits": list(self.emits), "lane_class": self.lane_class,
                "lane_budget": self.lane_budget, "depends": list(self.depends)}

    def integrity_hash(self) -> str:
        """Stable content hash over the declared capabilities — the registry pins
        an installed package to this so a manifest can't be silently widened."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
class AppContext:
    """The reference monitor. The ONLY way an app reaches the kernel; every call
    is checked against the manifest before it touches a real service."""

    def __init__(self, manifest: AppManifest, cage: NeuroSymbolicRuntime,
                 frame: Dict[str, float]):
        self._m = manifest
        self._cage = cage
        self._frame = frame                # this tick's sensor frame (kept private)
        self._reads = set(manifest.reads)
        self._emits = set(manifest.emits)
        self.emitted: List[Dict[str, Any]] = []
        self.violations: List[str] = []
        self.logs: List[str] = []

    # -- capability-checked sensor reads -------------------------------- #
    def read(self, channel: str) -> Optional[float]:
        if channel not in self._reads:
            self._deny(f"read channel {channel!r} not in capabilities "
                       f"{sorted(self._reads)}")
        return self._frame.get(channel)

    def read_all(self) -> Dict[str, float]:
        """Every declared channel present in this frame — nothing more."""
        return {c: self._frame[c] for c in self._reads if c in self._frame}

    # -- capability-checked, cage-gated egress -------------------------- #
    def emit(self, egress_class: str, payload: Dict[str, Any],
             context: Optional[Dict[str, Any]] = None) -> SafetyCertificate:
        if egress_class not in self._emits:
            self._deny(f"emit class {egress_class!r} not in capabilities "
                       f"{sorted(self._emits)}")
        ok, cert = self._cage.dispatch(egress_class, payload, context or {})
        self.emitted.append({"egress": egress_class, "ok": ok,
                             "z3": cert.z3_status, "msg": cert.message})
        return cert

    def log(self, msg: str) -> None:
        self.logs.append(str(msg))

    def _deny(self, msg: str) -> None:
        v = f"[{self._m.name}] {msg}"
        self.violations.append(v)
        raise CapabilityViolation(v)


# --------------------------------------------------------------------------- #
class AegisApp:
    """Base class for an AegisOS application. Subclass, set ``manifest``, and
    implement ``on_tick``. The app never imports a kernel service directly — it
    receives everything through the capability-checked context."""

    manifest: AppManifest = AppManifest(name="unnamed")

    def on_start(self, ctx: AppContext) -> None:
        """Called once when the service starts (optional)."""

    def on_tick(self, ctx: AppContext) -> Dict[str, Any]:
        """Process one frame. Return the app's output (JSON-able)."""
        return {}

    def on_stop(self) -> None:
        """Called when the service is stopped (optional)."""


# --------------------------------------------------------------------------- #
@dataclass
class TickResult:
    """What the host observes after running one app tick — fault-contained."""
    app: str
    ok: bool
    output: Optional[Dict[str, Any]] = None
    emitted: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    error: Optional[str] = None
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"app": self.app, "ok": self.ok, "output": self.output,
                "emitted": self.emitted, "violations": self.violations,
                "error": self.error, "ms": round(self.ms, 3)}


def tick_app(app: AegisApp, cage: NeuroSymbolicRuntime,
             frame: Dict[str, float]) -> TickResult:
    """Run one app tick inside the sandbox. A capability violation or any other
    exception is CONTAINED here (reported, never propagated) — the seed of the
    service supervisor: a misbehaving app cannot take down the kernel."""
    ctx = AppContext(app.manifest, cage, frame)
    t0 = time.perf_counter()
    try:
        out = app.on_tick(ctx)
        return TickResult(app.manifest.name, True, output=out, emitted=ctx.emitted,
                          ms=(time.perf_counter() - t0) * 1e3)
    except CapabilityViolation:
        return TickResult(app.manifest.name, False, violations=ctx.violations,
                          emitted=ctx.emitted, ms=(time.perf_counter() - t0) * 1e3)
    except Exception as exc:
        return TickResult(app.manifest.name, False,
                          error=f"{type(exc).__name__}: {exc}",
                          emitted=ctx.emitted, ms=(time.perf_counter() - t0) * 1e3)
