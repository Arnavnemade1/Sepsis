"""The live Aegis-eBPF device layer — the verifier + supervisor, running.

Slices 1–3 built and unit-proved the pieces (static verifier, ring-bound driver,
fault-isolating supervisor). This module stands them up as a *live subsystem* the
server ticks every cluster cycle and the console can drive:

  * a small fleet of supervised, verified eBPF drivers, each feeding a real
    Layer-2 ring through the Layer-1 ``/dev/bio`` stack;
  * a live verifier the operator can submit driver bytecode to and get an
    auditable ACCEPT / REJECT verdict (the "try to load a malicious driver" demo);
  * on-demand sensor-fault injection, so a driver can be made to die and the
    failover watched in real time, with every other bed provably untouched.

It owns no clinical logic — it is the kernel's verifiable *device plane*, separate
from the patient replay cohort, exposed so the runtime's safety machinery is
something judges can interact with rather than read about.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.aegis.kernel.devbio import ChannelSpec, DevBio
from backend.aegis.kernel.ebpf import assemble, Verifier
from backend.aegis.kernel.ebpfdrv import DEFAULT_HELPERS, EbpfDevice, scatter_program
from backend.aegis.kernel.memfabric import FabricPump, MemFabric
from backend.aegis.kernel.supervisor import DriverSupervisor

OPTICAL = [ChannelSpec("hr", "bpm", q=1e-2, r0=4.0),
           ChannelSpec("perfusion", "au", q=1e-3, r0=1.0)]
WIN = len(OPTICAL)

# A catalogue of driver bytecode the console offers to the live verifier. One is
# safe; the rest each trip a different proof obligation — the verifier's verdict
# on them is computed live, never canned.
PROGRAMS: Dict[str, Tuple[str, str]] = {
    "ecg": (scatter_program(WIN),
            "valid 2-channel optical ingest — bounded, halts, in-window"),
    "loop": ("LDI r0, 0\nspin: ADDI r0, 1\nJMP spin\nEXIT",
             "infinite loop via a backward jump (would starve the scheduler)"),
    "escape": ("LDI r6, 9999\nCALL bio_input\nSTX r6, 0, r0\nLDI r0, 0\nEXIT",
               "writes far outside its window (cross-patient corruption)"),
    "wild": ("CALL bio_input\nMOV r8, r0\nCALL bio_input\nSTX r0, 0, r8\nLDI r0, 0\nEXIT",
             "dereferences an untrusted input with no bound check"),
    "uninit": ("STX r3, 0, r3\nLDI r0, 0\nEXIT",
               "stores through an uninitialized register"),
    "rogue_call": ("CALL system\nLDI r0, 0\nEXIT",
                   "calls a helper outside the capability set"),
}


class InjectableFeed:
    """A hardware FIFO with an operator kill-switch. While ``tripped`` it raises,
    exactly as a disconnected sensor would — which the driver surfaces as a
    DriverFault for the supervisor to contain."""

    def __init__(self, base: Callable[[], Tuple[List[float], float]]):
        self.base = base
        self.tripped = False
        self.reason = ""

    def trip(self, reason: str) -> None:
        self.tripped, self.reason = True, reason

    def reset(self) -> None:
        self.tripped = False

    def __call__(self) -> Tuple[List[float], float]:
        if self.tripped:
            raise RuntimeError(self.reason)
        return self.base()


def _optical_base(seed: int) -> Callable[[], Tuple[List[float], float]]:
    rng = np.random.default_rng(seed)
    n = {"i": 0}

    def feed() -> Tuple[List[float], float]:
        n["i"] += 1
        hr = 80.0 + 7.0 * np.sin(n["i"] / 30.0) + rng.normal(0, 2.6)
        perf = 1.4 + rng.normal(0, 0.18)
        quality = 1.0 if rng.random() > 0.1 else float(rng.uniform(0.2, 0.5))
        return [hr, perf], quality
    return feed


def _safe_base(seed: int) -> Callable[[], Tuple[List[float], float]]:
    rng = np.random.default_rng(seed)

    def feed() -> Tuple[List[float], float]:
        return [80.0 + rng.normal(0, 1.0), 1.3], 0.3      # redundant, low-confidence
    return feed


class KernelDeviceRuntime:
    """The running verified-device fleet + the live verifier."""

    def __init__(self, labels: Optional[List[str]] = None, capacity: int = 512):
        labels = labels or ["Optical · Bay 1", "Optical · Bay 2", "Optical · Bay 3"]
        self.bus = DevBio()
        self.fabric = MemFabric(capacity_cells=1 << 15)
        self.sup = DriverSupervisor()
        self.verifier = Verifier(DEFAULT_HELPERS)
        self.verify_pass = 0
        self.verify_fail = 0
        self.slots: Dict[str, Dict[str, Any]] = {}
        fds: List[int] = []
        for i, label in enumerate(labels):
            key = f"dev_{i}"
            feed = InjectableFeed(_optical_base(seed=11 + i))
            prog = scatter_program(WIN)
            pf = (lambda feed=feed, prog=prog:
                  EbpfDevice(OPTICAL, prog, feed, kind="optical"))
            ff = (lambda i=i, prog=prog:
                  EbpfDevice(OPTICAL, prog, _safe_base(100 + i), kind="optical-bkp"))
            sd = self.sup.supervise(f"{key}/optical", pf(),
                                    fallback_factory=ff, primary_factory=pf)
            path = self.bus.mount(key, "optical_feed", sd)
            fd = self.bus.open(path)
            ring = self.fabric.map_descriptor(fd, path, sd.channel_keys(), capacity)
            self.slots[key] = {"label": label, "feed": feed, "sd": sd,
                               "fd": fd, "ring": ring, "path": path}
            fds.append(fd)
        self.pump = FabricPump(self.bus, self.fabric, fds)

    # -- driven by the server's cluster loop -------------------------------- #
    def tick(self, n: int = 1) -> None:
        self.pump.run(n)

    # -- the live verifier (the "load a driver" trap) ----------------------- #
    def verify_source(self, source: str, win_len: int = WIN) -> Dict[str, Any]:
        try:
            prog = assemble(source)
        except SyntaxError as exc:
            self.verify_fail += 1
            return {"ok": False, "stage": "assemble", "error": str(exc),
                    "source": source}
        rep = self.verifier.verify(prog, win_len)
        self.verify_pass += int(rep.ok)
        self.verify_fail += int(not rep.ok)
        out: Dict[str, Any] = {"ok": rep.ok, "stage": "verify",
                               "report": rep.to_dict(),
                               "listing": [str(p) for p in prog]}
        if rep.ok:
            out["proof"] = rep.proof
        else:
            v = rep.violations[0]
            out["violation"] = {**v.to_dict(), "instr": str(prog[v.index])}
        return out

    def verify_named(self, name: str) -> Dict[str, Any]:
        if name not in PROGRAMS:
            return {"ok": False, "stage": "lookup",
                    "error": f"no such driver {name!r}",
                    "available": list(PROGRAMS)}
        src, desc = PROGRAMS[name]
        res = self.verify_source(src)
        res["name"], res["description"], res["source"] = name, desc, src
        return res

    # -- fault injection / recovery (the "kill a sensor" demo) -------------- #
    def resolve(self, q: str) -> Optional[str]:
        q = (q or "").strip().lower()
        if q in self.slots:
            return q
        if q.isdigit():                              # 1-based bay index
            k = f"dev_{int(q) - 1}"
            return k if k in self.slots else None
        for k, s in self.slots.items():              # label substring
            if q and q in s["label"].lower():
                return k
        return None

    def inject_fault(self, key: str, reason: str = "sensor disconnected (injected)"
                     ) -> Dict[str, Any]:
        s = self.slots.get(key)
        if s is None:
            return {"status": "error", "detail": f"no such device {key!r}"}
        s["feed"].trip(reason)
        self.tick(1)                                 # force the fault to surface now
        return {"status": "ok", "slot": key, **s["sd"].status()}

    def recover(self, key: str) -> Dict[str, Any]:
        s = self.slots.get(key)
        if s is None:
            return {"status": "error", "detail": f"no such device {key!r}"}
        s["feed"].reset()
        s["sd"].restore()
        return {"status": "ok", "slot": key, **s["sd"].status()}

    # -- telemetry ---------------------------------------------------------- #
    def status(self, full: bool = True) -> Dict[str, Any]:
        devices = []
        for key, s in self.slots.items():
            sd, ring = s["sd"], s["ring"]
            rep = getattr(sd.active, "report", None)
            devices.append({
                "key": key, "label": s["label"], "path": s["path"],
                "state": sd.state, "active": sd.active.kind,
                "faults": sd.faults, "failovers": sd.failovers,
                "restores": sd.restores, "quarantined": len(sd.quarantined),
                "blocks": ring.write_index,
                "verified": bool(rep.ok) if rep else None,
                "max_steps": rep.max_steps if rep else None,
                "last_steps": getattr(sd.active, "last_steps", None),
            })
        total = self.verify_pass + self.verify_fail
        out: Dict[str, Any] = {
            "devices": devices,
            "verifier": {"checked": total, "accepted": self.verify_pass,
                         "rejected": self.verify_fail,
                         "helpers": list(DEFAULT_HELPERS)},
            "supervisor": self.sup.status(),
            "latency": self.bus.latency_stats(),
        }
        if full:                                    # static/heavy fields: endpoint only
            out["arena"] = self.fabric.address_matrix()["arena"]
            out["programs"] = [{"name": n, "description": d}
                               for n, (_, d) in PROGRAMS.items()]
        return out
