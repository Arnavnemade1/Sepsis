"""aegisd — the AegisOS service manager (init system).

The scattered diagnostics used to be hand-wired into the cohort tick. aegisd makes
them real **services**: registered with a capability manifest, started in
dependency order, ticked each kernel cycle inside the app sandbox, and — crucially
— **supervised**. This is the driver supervisor (`kernel/supervisor.py`)
generalized from devices to applications: a service that faults is contained,
auto-restarted within a budget, and quarantined if it keeps failing, while every
other service (and every patient bed) keeps ticking. The kernel never crashes
because one app threw.

Each service publishes its latest output at ``/proc/<name>`` so a downstream
service can consume it (capability-checked) — the platform's inter-service IPC.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.aegis.apps.sdk import AegisApp, AppContext, TickResult, tick_app
from backend.aegis.kernel.neurosym import NeuroSymbolicRuntime
from backend.aegis.procfs import ProcFS

REGISTERED, RUNNING, QUARANTINED, STOPPED = "REGISTERED", "RUNNING", "QUARANTINED", "STOPPED"


@dataclass
class Service:
    app: AegisApp
    state: str = REGISTERED
    faults: int = 0
    restarts: int = 0
    ticks: int = 0
    last: Optional[TickResult] = None
    output: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return self.app.manifest.name

    @property
    def depends(self) -> tuple:
        return tuple(self.app.manifest.depends)

    def status(self) -> Dict[str, Any]:
        m = self.app.manifest
        return {"name": self.name, "version": m.version, "state": self.state,
                "depends": list(self.depends), "faults": self.faults,
                "restarts": self.restarts, "ticks": self.ticks,
                "caps": {"reads": len(m.reads), "emits": list(m.emits),
                         "proc_reads": list(m.proc_reads), "lanes": m.lane_budget},
                "healthy": self.state == RUNNING and (self.last is None or self.last.ok),
                "last_ms": round(self.last.ms, 2) if self.last else None}


class ServiceManager:
    """aegisd: registers, dependency-orders, ticks and supervises app services."""

    def __init__(self, cage: NeuroSymbolicRuntime, procfs: ProcFS,
                 max_restarts: int = 3):
        self.cage = cage
        self.procfs = procfs
        self.max_restarts = max_restarts
        self.services: Dict[str, Service] = {}
        self.events: List[Dict[str, Any]] = []
        self.tick_no = 0
        procfs.register("/proc/services", lambda: self.status())

    # -- registration + dependency-ordered lifecycle -------------------- #
    def register(self, app: AegisApp) -> Service:
        errs = app.manifest.validate()
        if errs:
            raise ValueError(f"invalid manifest for {app.manifest.name!r}: {errs}")
        svc = Service(app)
        self.services[svc.name] = svc
        return svc

    def _topo(self) -> List[str]:
        """Topological order over `depends` (start deps before dependents)."""
        order: List[str] = []
        seen: Dict[str, int] = {}                    # 0=visiting, 1=done

        def visit(n: str, stack: tuple) -> None:
            if seen.get(n) == 1:
                return
            if seen.get(n) == 0:
                raise ValueError(f"dependency cycle: {' -> '.join(stack + (n,))}")
            seen[n] = 0
            for d in self.services[n].depends:
                if d not in self.services:
                    raise ValueError(f"service {n!r} depends on missing {d!r}")
                visit(d, stack + (n,))
            seen[n] = 1
            order.append(n)

        for name in self.services:
            visit(name, ())
        return order

    def start_all(self) -> List[str]:
        order = self._topo()
        for name in order:
            self._start(name)
        return order

    def _start(self, name: str) -> None:
        svc = self.services[name]
        try:
            svc.app.on_start(AppContext(svc.app.manifest, self.cage, {}, self.procfs))
        except Exception as exc:                      # a bad on_start can't crash boot
            self._emit("START_FAIL", name, f"{type(exc).__name__}: {exc}")
        svc.state = RUNNING
        self.procfs.register(f"/proc/{name}", lambda s=svc: s.output or {})
        self._emit("START", name, f"v{svc.app.manifest.version}")

    def stop(self, name: str) -> bool:
        svc = self.services.get(name)
        if not svc:
            return False
        try:
            svc.app.on_stop()
        except Exception:
            pass
        svc.state = STOPPED
        self._emit("STOP", name, "operator stop")
        return True

    def restart(self, name: str) -> bool:
        svc = self.services.get(name)
        if not svc:
            return False
        svc.restarts = 0                              # operator restart clears the budget
        self._start(name)
        return True

    # -- the supervised tick -------------------------------------------- #
    def tick(self, frame: Dict[str, float]) -> Dict[str, TickResult]:
        self.tick_no += 1
        results: Dict[str, TickResult] = {}
        for name in self._topo():
            svc = self.services[name]
            if svc.state != RUNNING:
                continue
            r = tick_app(svc.app, self.cage, frame, self.procfs)
            svc.ticks += 1
            svc.last = r
            if r.ok:
                svc.output = r.output                 # published at /proc/<name> NOW
            else:
                self._on_fault(svc, r)
            results[name] = r
        return results

    def _on_fault(self, svc: Service, r: TickResult) -> None:
        svc.faults += 1
        reason = r.error or (r.violations[0] if r.violations else "unhealthy tick")
        self._emit("FAULT", svc.name, reason)
        if svc.restarts < self.max_restarts:
            try:
                svc.app.on_start(AppContext(svc.app.manifest, self.cage, {}, self.procfs))
            except Exception:
                pass
            svc.restarts += 1
            svc.state = RUNNING
            self._emit("RESTART", svc.name,
                       f"auto-restart {svc.restarts}/{self.max_restarts}")
        else:
            svc.state = QUARANTINED
            self._emit("QUARANTINE", svc.name, "exceeded restart budget")

    # -- introspection -------------------------------------------------- #
    def _emit(self, kind: str, svc: str, detail: str) -> None:
        self.events = (self.events + [{"tick": self.tick_no, "kind": kind,
                                       "svc": svc, "detail": detail,
                                       "ts": round(time.time(), 3)}])[-128:]

    def status(self) -> Dict[str, Any]:
        svcs = [s.status() for s in self.services.values()]
        running = sum(1 for s in self.services.values() if s.state == RUNNING)
        return {
            "tick": self.tick_no, "n": len(self.services),
            "running": running, "quarantined":
                sum(1 for s in self.services.values() if s.state == QUARANTINED),
            "services": svcs, "events": self.events[-16:],
        }
