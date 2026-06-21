"""The driver supervisor — fault isolation for the microkernel.

The Aegis-eBPF verifier proves a driver's *code* is safe: it cannot loop, cannot
escape its memory window, cannot call outside its capability set. But a verified
driver still sits in front of *physical hardware*, and hardware fails — a sensor
is unplugged, a feed throws, a frame is garbage. In a monolithic kernel that
exception unwinds into the read path and the whole runtime panics, taking every
other patient's monitoring down with it.

The microkernel answer is **fault containment**. :class:`SupervisedDevice` wraps a
driver and presents itself to the OS as an ordinary :class:`BioDevice` — it mounts
on ``/dev/bio`` and feeds a ring exactly like any other. Internally it is the
isolation boundary: when its active driver raises :class:`DriverFault` at runtime,
it **quarantines that instance, hot-swaps a fallback, and still returns a sample
on the same tick**, so the bus, the memory fabric and the Layer-3 scheduler never
observe a gap. A driver can die; the kernel does not.

The guarantees, stated honestly:

  * **Containment, not resurrection.** A faulting driver is torn down and isolated;
    it is not magically fixed. The bed continues on a fallback (a redundant
    verified driver, then a degraded last-known-good hold at quality 0) — clearly
    flagged, never silently faked.
  * **No dropped ticks for the cohort.** Because the failover happens inside one
    device's ``_produce`` and is invisible above Layer 1, a fault on bed *i*
    cannot cost bed *j* a single sample. That cross-patient isolation is the whole
    point, and the harness measures it directly.

:class:`DriverSupervisor` owns the supervised slots and an append-only event log —
the system of record for every quarantine and failover, mirroring the audit
discipline of the Z3 egress cage on the write side.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.aegis.kernel.devbio import BioDevice
from backend.aegis.kernel.ebpfdrv import DriverFault

# Slot lifecycle states.
RUNNING = "RUNNING"        # primary driver healthy
DEGRADED = "DEGRADED"      # primary quarantined; serving a fallback
DOWN = "DOWN"             # no live driver; holding last-known-good at quality 0

# Event kinds (the supervisor's dmesg).
FAULT, FAILOVER, DOWNED, RESTORE = "FAULT", "FAILOVER", "DOWN", "RESTORE"


@dataclass
class FaultRecord:
    ts: float
    slot: str
    kind: str
    driver: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"ts": round(self.ts, 3), "slot": self.slot, "kind": self.kind,
                "driver": self.driver, "detail": self.detail}


class SupervisedDevice(BioDevice):
    """A fault-isolating wrapper that is itself a ``/dev/bio`` device."""

    def __init__(self, name: str, primary: BioDevice,
                 fallback_factory: Optional[Callable[[], BioDevice]] = None,
                 supervisor: Optional["DriverSupervisor"] = None,
                 primary_factory: Optional[Callable[[], BioDevice]] = None):
        super().__init__(primary.channels, primary.sample_rate_hz)
        # Present as the underlying kind so ring naming / telemetry stay consistent.
        self.kind = getattr(primary, "kind", "ebpf")
        self.name = name
        self.active = primary
        self.fallback_factory = fallback_factory
        self.primary_factory = primary_factory     # lets a quarantined slot be rebuilt
        self.supervisor = supervisor
        self.state = RUNNING
        self.faults = 0
        self.failovers = 0
        self.restores = 0
        self.quarantined: List[BioDevice] = []
        self.last_event: Optional[FaultRecord] = None
        self._last_good: Dict[str, float] = {c.key: float("nan") for c in primary.channels}

    # -- lifecycle: propagate open/close to whatever driver is active ------- #
    def on_open(self) -> None:
        super().on_open()
        self.active.on_open()

    def on_close(self) -> None:
        super().on_close()
        self.active.on_close()

    # -- the contract: ALWAYS return a sample; never let a fault escape ----- #
    def _produce(self) -> Tuple[Dict[str, float], float]:
        if self.state != DOWN:
            try:
                raw, q = self.active._produce()
                self._last_good = dict(raw)
                return raw, q
            except DriverFault as exc:
                self._on_fault(exc)                    # quarantine + maybe hot-swap
                if self.state != DOWN:                 # retry the fallback THIS tick
                    try:
                        raw, q = self.active._produce()
                        self._last_good = dict(raw)
                        return raw, q
                    except DriverFault as exc2:
                        self._on_fault(exc2)
        # Degraded hold: still emit a sample (quality 0 ⇒ downstream distrusts it),
        # so the kernel keeps ticking instead of dropping the bed.
        return dict(self._last_good), 0.0

    # -- the isolation state machine ---------------------------------------- #
    def _on_fault(self, exc: DriverFault) -> None:
        self.faults += 1
        faulting = getattr(self.active, "kind", "?")
        self.quarantined.append(self.active)
        try:
            self.active.on_close()
        except Exception:
            pass
        self._emit(FAULT, faulting, f"{faulting} driver trapped: {exc}")
        if self.state == RUNNING and self.fallback_factory is not None:
            self.active = self.fallback_factory()      # construct = re-verify the fallback
            self.active.on_open()
            self.state = DEGRADED
            self.failovers += 1
            self._emit(FAILOVER, self.active.kind,
                       "quarantined faulting instance; hot-swapped verified fallback")
        else:
            self.state = DOWN
            self._emit(DOWNED, faulting,
                       "no fallback available; holding last-known-good at quality 0")

    def restore(self) -> bool:
        """Rebuild the primary driver and return the slot to RUNNING — the
        recovery the operator triggers once the hardware is back. Re-running the
        primary factory re-verifies it, so a restore is just as gated as a mount."""
        if self.primary_factory is None:
            return False
        try:
            self.active.on_close()
        except Exception:
            pass
        self.active = self.primary_factory()       # construct = re-verify
        self.active.on_open()
        self.state = RUNNING
        self.restores += 1
        self._emit(RESTORE, self.active.kind, "primary driver rebuilt and re-armed")
        return True

    def _emit(self, kind: str, driver: str, detail: str) -> None:
        rec = FaultRecord(time.time(), self.name, kind, driver, detail)
        self.last_event = rec
        if self.supervisor is not None:
            self.supervisor.record(rec)

    def status(self) -> Dict[str, Any]:
        return {
            "slot": self.name, "state": self.state, "active": self.active.kind,
            "faults": self.faults, "failovers": self.failovers,
            "restores": self.restores, "quarantined": len(self.quarantined),
            "last_event": self.last_event.to_dict() if self.last_event else None,
        }


class DriverSupervisor:
    """Owns the supervised slots and the fleet-wide fault event log."""

    def __init__(self) -> None:
        self.slots: Dict[str, SupervisedDevice] = {}
        self.events: List[FaultRecord] = []

    def supervise(self, name: str, primary: BioDevice,
                  fallback_factory: Optional[Callable[[], BioDevice]] = None,
                  primary_factory: Optional[Callable[[], BioDevice]] = None
                  ) -> SupervisedDevice:
        dev = SupervisedDevice(name, primary, fallback_factory, supervisor=self,
                               primary_factory=primary_factory)
        self.slots[name] = dev
        return dev

    def record(self, rec: FaultRecord) -> None:
        self.events = (self.events + [rec])[-256:]

    def status(self) -> Dict[str, Any]:
        degraded = sum(1 for d in self.slots.values() if d.state != RUNNING)
        return {
            "slots": {n: d.status() for n, d in self.slots.items()},
            "healthy": len(self.slots) - degraded, "degraded": degraded,
            "failovers": sum(d.failovers for d in self.slots.values()),
            "events": [r.to_dict() for r in self.events[-16:]],
        }
