"""Slice-2 harness: a VERIFIED eBPF driver feeding a real zero-copy ring.

Proves the full path is real, end to end:
  1. A verified eBPF program is admitted as a ``/dev/bio`` device (load-time gate).
  2. It is mounted, opened, and mapped into the Layer-2 shared-arena ring.
  3. The FabricPump drives it like any trusted driver; a consumer reads the
     conditioned samples back out of the ring by zero-copy view.
  4. A HOSTILE driver (its data plane scribbles outside its window) is REFUSED at
     mount — the kernel will not bind unverified bytecode to a live ring.

Every assertion is checked; the script exits non-zero if reality disagrees.

    python -m backend.aegis.kernel.verify_ebpfdrv
"""
from __future__ import annotations

import numpy as np

from backend.aegis.kernel.devbio import ChannelSpec, DevBio
from backend.aegis.kernel.ebpf import assemble
from backend.aegis.kernel.ebpfdrv import (DriverRejected, EbpfDevice,
                                          load_ebpf_driver, scatter_program)
from backend.aegis.kernel.memfabric import FabricPump, MemFabric

OPTICAL = [ChannelSpec("hr", "bpm", q=1e-2, r0=4.0),
           ChannelSpec("perfusion", "au", q=1e-3, r0=1.0)]


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def make_feed(seed: int = 7):
    """A raw optical hardware FIFO: noisy HR + perfusion, occasional low quality."""
    rng = np.random.default_rng(seed)
    t = {"i": 0}

    def feed():
        t["i"] += 1
        hr = 82.0 + 6.0 * np.sin(t["i"] / 25.0) + rng.normal(0, 3.0)
        perf = 1.4 + rng.normal(0, 0.2)
        quality = 1.0 if rng.random() > 0.12 else float(rng.uniform(0.1, 0.4))
        return [hr, perf], quality

    return feed


def main() -> int:
    failures = 0

    # ---- 1. admit a verified driver --------------------------------------- #
    _hr("LOADING A VERIFIED eBPF DRIVER")
    drv = load_ebpf_driver(OPTICAL, scatter_program(len(OPTICAL)), make_feed(),
                           kind="optical")
    rep = drv.report
    print(f"  admitted: {drv.kind} driver, {len(OPTICAL)} channels "
          f"({', '.join(c.key for c in OPTICAL)})")
    print(f"  ⊢ {rep.proof}")
    failures += 0 if rep.ok else 1

    # ---- 2/3. mount → ring → pump → zero-copy consumer -------------------- #
    _hr("FEEDING A REAL ZERO-COPY RING THROUGH THE VERIFIED DRIVER")
    bus = DevBio()
    fabric = MemFabric(capacity_cells=1 << 14)
    path = bus.mount("bed_3", "optical_feed", drv)
    fd = bus.open(path)
    ring = fabric.map_descriptor(fd, path, drv.channel_keys(), capacity=256)
    cur = fabric.subscribe(fd)
    pump = FabricPump(bus, fabric, [fd])

    pump.run(400)                                       # 400 verified-driver ticks
    slots = cur.poll()
    last_hr = ring.field(slots[-1], "hr")
    shares = bool(np.shares_memory(ring.view(slots[-1]), fabric.arena.buffer))
    lat = bus.latency_stats()

    print(f"  mounted at {path}  (fd={fd})")
    print(f"  pumped {ring.write_index} blocks → consumer drained {len(slots)} slots "
          f"by offset (zero-copy view shares arena = {shares})")
    print(f"  latest conditioned HR off the ring = {last_hr:.2f} bpm "
          f"· inline-Kalman p50 = {lat['p50_us']:.1f}µs")
    print(f"  VM ran {drv.last_steps} bytecode steps on the last tick "
          f"(static bound {rep.max_steps}); driver faults = {drv.faults}")

    flowed = ring.write_index >= 400 and len(slots) > 0 and shares and drv.faults == 0
    failures += 0 if flowed else 1

    # ---- 4. a hostile driver is refused at mount -------------------------- #
    _hr("REJECTING A HOSTILE DRIVER AT MOUNT")
    # Its data plane writes a sample into slot 9999 — far outside its 2-cell
    # window, i.e. into a neighbouring bed's ring in the shared arena.
    hostile_src = """
        LDI  r6, 9999
        CALL bio_input
        STX  r6, 0, r0
        LDI  r0, 0
        EXIT
    """
    refused = False
    try:
        EbpfDevice(OPTICAL, hostile_src, make_feed(), kind="optical")
    except DriverRejected as exc:
        refused = True
        print(f"  ✗ DriverRejected — {exc}")
    if not refused:
        print("  !! hostile driver was admitted — verifier failed to gate the mount")
    print(f"  kernel bound unverified bytecode to a ring: {'NO ✓' if refused else 'YES ✗'}")
    failures += 0 if refused else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓" if failures == 0
          else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
