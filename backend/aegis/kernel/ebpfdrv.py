"""Binding the Aegis-eBPF sandbox to the real ``/dev/bio`` driver stack.

``ebpf.py`` proves a bytecode program safe in the abstract. This module makes a
verified program *the data plane of an actual device*: :class:`EbpfDevice` is a
drop-in :class:`BioDevice` whose ``_produce()`` is computed by **executing
verified bytecode**, not by trusted Python. The program reads raw samples from
the hardware FIFO (the ``bio_input`` helper) and scatters them into its own
output window — one cell per channel — and the verifier has already proven, at
load time, that those writes can only land inside that window.

The key property this buys the kernel:

  * **An untrusted driver cannot be mounted until it is proven safe.** The
    constructor runs the verifier and *raises* :class:`DriverRejected` if the
    program could loop, escape its window, or call an un-whitelisted helper. There
    is no path that binds unverified bytecode to a live ring.

  * **Above Layer 1, nothing changes.** A verified driver flows through the exact
    same ``DevBio.read`` (inline Kalman conditioning) → ``MemFabric.publish``
    (zero-copy ring) → consumer path as a trusted Python driver. The cohort can't
    tell the difference — except that this one carries a machine-checked safety
    certificate.

:class:`DriverFault` is raised when the VM traps at *runtime* (a defense-in-depth
guard that, for verified code, never fires). It exists so the supervisor in the
next slice can quarantine a faulting instance instead of letting it corrupt state.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.aegis.kernel.devbio import BioDevice, ChannelSpec
from backend.aegis.kernel.ebpf import Instr, VM, Verifier, VerdictReport, assemble

DEFAULT_HELPERS: Tuple[str, ...] = ("bio_input", "bio_time")

# A feed is the raw hardware source: returns (one raw value per channel, quality).
Feed = Callable[[], Tuple[Sequence[float], float]]


class DriverRejected(Exception):
    """The static verifier refused to admit this driver to the kernel."""


class DriverFault(Exception):
    """A loaded driver trapped at runtime (caught by the supervisor)."""


def scatter_program(n_channels: int) -> str:
    """The canonical verified ingest data-plane: pull ``n`` raw samples from the
    FIFO and scatter them into window slots ``0..n-1``. r6 is callee-saved so the
    slot address survives each ``CALL`` (using a caller-saved reg would be caught
    as a clobbered read), and every store targets a constant slot inside the
    window — so the verifier admits it."""
    lines: List[str] = []
    for i in range(n_channels):
        lines += [f"LDI r6, {i}", "CALL bio_input", "STX r6, 0, r0"]
    lines += ["LDI r0, 0", "EXIT"]
    return "\n".join(lines)


class EbpfDevice(BioDevice):
    """A ``/dev/bio`` device whose sampling logic is verified eBPF bytecode."""

    kind = "ebpf"

    def __init__(self, channels: List[ChannelSpec], program: Any, feed: Feed,
                 *, kind: Optional[str] = None, sample_rate_hz: float = 1.0,
                 helpers: Sequence[str] = DEFAULT_HELPERS,
                 verifier: Optional[Verifier] = None):
        super().__init__(channels, sample_rate_hz)
        if kind:
            self.kind = kind
        self.win_len = len(channels)
        self.program: List[Instr] = assemble(program) if isinstance(program, str) else list(program)

        # -- LOAD-TIME GATE: refuse to construct an unverified driver --------- #
        self.verifier = verifier or Verifier(tuple(helpers))
        self.report: VerdictReport = self.verifier.verify(self.program, self.win_len)
        if not self.report.ok:
            v = self.report.violations[0]
            raise DriverRejected(
                f"driver refused: instr #{v.index} {self.program[v.index]!s} — "
                f"{v.rule}: {v.detail}")

        self._feed = feed
        self._mem = np.zeros(self.win_len, dtype=np.float64)
        self._fifo: Any = iter(())
        self._t0 = time.perf_counter_ns()
        self._vm = VM({"bio_input": self._bio_input, "bio_time": self._bio_time})
        # runtime telemetry (the supervisor reads these)
        self.faults = 0
        self.last_steps = 0
        self.last_trap: Optional[str] = None

    # -- kernel helpers the bytecode may call (the capability set) ---------- #
    def _bio_input(self) -> float:
        try:
            return float(next(self._fifo))
        except StopIteration:
            return 0.0

    def _bio_time(self) -> float:
        return (time.perf_counter_ns() - self._t0) / 1_000.0

    # -- BioDevice contract: produce one sample by RUNNING the bytecode ----- #
    def _produce(self) -> Tuple[Dict[str, float], float]:
        try:
            raw_vals, quality = self._feed()          # pull from the hardware FIFO
        except Exception as exc:                       # the physical device faulted
            self.faults += 1
            self.last_trap = f"device feed error: {type(exc).__name__}: {exc}"
            raise DriverFault(self.last_trap) from exc
        self._fifo = iter(raw_vals)
        self._mem[:] = np.nan                         # clear the window each tick
        res = self._vm.run(self.program, self._mem)
        self.last_steps, self.last_trap = res.steps, res.trapped
        if res.trapped is not None:                   # never happens for verified code
            self.faults += 1
            raise DriverFault(res.trapped)
        raw = {c.key: float(self._mem[i]) for i, c in enumerate(self.channels)}
        return raw, float(quality)

    # -- introspection (for telemetry / the supervisor / the UI) ------------ #
    def verification(self) -> Dict[str, Any]:
        return {
            "loaded": True, "kind": self.kind, "win_len": self.win_len,
            "report": self.report.to_dict(),
            "runtime": {"last_steps": self.last_steps, "faults": self.faults,
                        "last_trap": self.last_trap},
        }


def load_ebpf_driver(channels: List[ChannelSpec], program: Any, feed: Feed,
                     **kw: Any) -> EbpfDevice:
    """Verify-then-construct. Raises :class:`DriverRejected` if the program is
    not provably safe — the single chokepoint for admitting a driver."""
    return EbpfDevice(channels, program, feed, **kw)
