"""Slice-3 harness: driver fault isolation under live load.

Scenario — two beds share one bus, one memory fabric, one pump:
  * bed_3/optical: a VERIFIED eBPF driver whose physical sensor dies mid-stream
    (its feed throws at tick 150 — "cable disconnected"). It is supervised, with a
    verified fallback driver behind it.
  * bed_7/optical: an ordinary verified driver that never faults — the bystander.

We pump 300 ticks and then check, by measurement, the microkernel promise:
  1. bed_3 faulted, was quarantined, and failed over to its fallback — recorded.
  2. bed_3 still wrote a sample on EVERY tick (no gap in its ring) despite the fault.
  3. bed_7 dropped ZERO ticks while bed_3 was failing over (cross-patient isolation).

Every claim is asserted; the script exits non-zero if reality disagrees.

    python -m backend.aegis.kernel.verify_supervisor
"""
from __future__ import annotations

import numpy as np

from backend.aegis.kernel.devbio import ChannelSpec, DevBio
from backend.aegis.kernel.ebpfdrv import EbpfDevice, scatter_program
from backend.aegis.kernel.memfabric import FabricPump, MemFabric
from backend.aegis.kernel.supervisor import DEGRADED, DriverSupervisor

OPTICAL = [ChannelSpec("hr", "bpm", q=1e-2, r0=4.0),
           ChannelSpec("perfusion", "au", q=1e-3, r0=1.0)]
PROG = scatter_program(len(OPTICAL))
TICKS = 300
FAIL_AT = 150


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def flaky_feed(fail_at: int, seed: int = 1):
    """A real optical sensor that disconnects at ``fail_at`` (the feed raises)."""
    rng = np.random.default_rng(seed)
    n = {"i": 0}

    def feed():
        n["i"] += 1
        if n["i"] >= fail_at:
            raise RuntimeError("optical sensor disconnected (cable fault)")
        hr = 82.0 + 6.0 * np.sin(n["i"] / 25.0) + rng.normal(0, 3.0)
        return [hr, 1.4 + rng.normal(0, 0.2)], 1.0

    return feed


def safe_feed(seed: int = 2):
    """A redundant, always-available source: coarse baseline at low quality."""
    rng = np.random.default_rng(seed)

    def feed():
        return [80.0 + rng.normal(0, 1.0), 1.2], 0.3      # quality 0.3 ⇒ distrusted
    return feed


def healthy_feed(seed: int = 3):
    rng = np.random.default_rng(seed)

    def feed():
        return [78.0 + rng.normal(0, 2.5), 1.5 + rng.normal(0, 0.2)], 1.0
    return feed


def main() -> int:
    failures = 0
    sup = DriverSupervisor()
    bus = DevBio()
    fabric = MemFabric(capacity_cells=1 << 15)

    # bed_3 — supervised driver with a sensor that dies at tick 150.
    primary = EbpfDevice(OPTICAL, PROG, flaky_feed(FAIL_AT), kind="optical")
    bed3 = sup.supervise(
        "bed_3/optical", primary,
        fallback_factory=lambda: EbpfDevice(OPTICAL, PROG, safe_feed(), kind="optical-bkp"))
    p3 = bus.mount("bed_3", "optical_feed", bed3)
    fd3 = bus.open(p3)
    ring3 = fabric.map_descriptor(fd3, p3, bed3.channel_keys(), capacity=512)

    # bed_7 — the bystander, an ordinary verified driver.
    bed7 = EbpfDevice(OPTICAL, PROG, healthy_feed(), kind="optical")
    p7 = bus.mount("bed_7", "optical_feed", bed7)
    fd7 = bus.open(p7)
    ring7 = fabric.map_descriptor(fd7, p7, bed7.channel_keys(), capacity=512)

    pump = FabricPump(bus, fabric, [fd3, fd7])

    _hr(f"PUMPING {TICKS} TICKS (bed_3 sensor fails at tick {FAIL_AT})")
    pump.run(TICKS)

    st3 = bed3.status()
    print(f"  bed_3: state={st3['state']} active={st3['active']} "
          f"faults={st3['faults']} failovers={st3['failovers']} "
          f"quarantined={st3['quarantined']}")
    print(f"  bed_3 ring wrote {ring3.write_index} blocks over {TICKS} ticks "
          f"(gap = {TICKS - ring3.write_index})")
    print(f"  bed_7 ring wrote {ring7.write_index} blocks over {TICKS} ticks "
          f"(gap = {TICKS - ring7.write_index})")
    hr3 = ring3.field(ring3.latest_slot(), "hr")
    print(f"  bed_3 latest conditioned HR (on fallback) = {hr3:.2f} bpm")

    _hr("SUPERVISOR EVENT LOG (dmesg)")
    for e in sup.status()["events"]:
        print(f"  [{e['kind']:8}] {e['slot']:16} {e['driver']:12} {e['detail']}")

    # ---- assertions: the microkernel promise, measured -------------------- #
    _hr("VERIFYING THE GUARANTEE")
    checks = [
        ("bed_3 faulted and failed over",      st3["faults"] >= 1 and st3["failovers"] >= 1),
        ("bed_3 is DEGRADED (on fallback)",    st3["state"] == DEGRADED),
        ("bed_3 dropped NO ticks",             ring3.write_index == TICKS),
        ("bed_7 dropped NO ticks (isolated)",  ring7.write_index == TICKS),
        ("bed_7 never faulted",                bed7.faults == 0),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — a driver died; the kernel never skipped a beat."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
