"""Layer 1 — the ``/dev/bio`` Unified Device Driver Architecture.

In a POSIX kernel *everything is a file*: a SATA disk is ``/dev/sda``, a serial
line is ``/dev/ttyS0``. AegisOS extends that principle to the only peripheral a
critical-care runtime actually cares about — **the patient and the sensors
strapped to them**. A bedside monitor, an optical (rPPG) camera matrix, a
thermal grid: each is mounted into a ``/dev/bio`` namespace and read through an
integer **file descriptor**, exactly like a character device.

What makes this a real OS layer and not a naming gimmick:

  * **A mount table + fd table.** ``mount()`` registers a driver at a path
    (``/dev/bio/node_0/optical_feed``); ``open()`` returns a small integer fd;
    ``read(fd)`` pulls one conditioned data block; ``close()``/``unmount()``
    tear it down. The namespace is introspectable (``listdir``/``stat``) the way
    ``/dev`` is.

  * **An inline stochastic pipeline built into the driver stack.** Every block
    that crosses an fd is pushed through a per-channel adaptive **Kalman
    state-space filter** *before* it is ever handed upward. The filter's
    measurement-noise term is divided by the device's reported signal quality,
    so noisy frames (lighting jitter, motion) are automatically distrusted. This
    mirrors the frontend SHAL (``webui/shal.js``) but lives in the kernel so the
    rest of the OS only ever sees clean, stabilized state.

  * **Conditioning-latency telemetry.** Each read records how long its filtering
    took, in microseconds, from ``time.perf_counter_ns`` — the platform's
    monotonic software clock, not a CPU performance counter (no PMU, no cycle
    counter, no rdtsc). The bus keeps p50/p95/max, which measures this process's
    own work under load and is reported as such.

Drivers are pluggable: subclass :class:`BioDevice`, implement ``_produce()``,
and you can mount anything — a synthetic generator for the ICU demo, a replay of
real MIMIC-IV / eICU arrays, or (eventually) a live hardware bus. Nothing above
this layer knows or cares which.

This module is dependency-light (numpy only) and has no import-time side
effects, so it is trivially unit-testable and safe to compose into ``AegisOS``.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

# Monotonic clock in microseconds — used for block timestamps and the
# conditioning-latency counters. perf_counter is monotonic and high-resolution.
def _now_us() -> float:
    return time.perf_counter_ns() / 1_000.0


# --------------------------------------------------------------------------- #
# The inline stochastic stage: a 1-D adaptive Kalman filter, one per channel.
# --------------------------------------------------------------------------- #
@dataclass
class KalmanScalar:
    """Constant-velocity-free scalar Kalman filter with quality-adaptive noise.

    State is a single scalar ``x`` (the conditioned value) with variance ``P``.
    Process noise ``Q`` admits drift; measurement noise ``R`` is *divided by the
    per-sample quality* so a low-quality reading (quality→0) is trusted far less
    (R→∞ ⇒ gain→0 ⇒ estimate barely moves). This is the same adaptive-R trick
    the frontend SHAL uses, ported to the kernel.
    """

    q: float = 1e-3          # process noise (per step)
    r0: float = 1.0          # base measurement noise (at quality == 1.0)
    x: Optional[float] = None
    p: float = 1.0
    last_gain: float = 0.0

    def update(self, z: float, quality: float = 1.0) -> float:
        if self.x is None or not math.isfinite(self.x):
            # Cold start: adopt the first finite measurement directly.
            self.x = float(z)
            self.p = self.r0
            self.last_gain = 1.0
            return self.x
        quality = min(max(float(quality), 1e-3), 1.0)
        r = self.r0 / quality                      # distrust low-quality samples
        # Predict
        p_pred = self.p + self.q
        # Update
        k = p_pred / (p_pred + r)                  # Kalman gain in [0, 1)
        self.x = self.x + k * (float(z) - self.x)
        self.p = (1.0 - k) * p_pred
        self.last_gain = k
        return self.x


# --------------------------------------------------------------------------- #
# Driver model
# --------------------------------------------------------------------------- #
@dataclass
class ChannelSpec:
    """One scalar signal a device exposes (e.g. heart rate, in bpm)."""
    key: str
    unit: str
    q: float = 1e-3          # Kalman process noise for this channel
    r0: float = 1.0          # Kalman base measurement noise for this channel


class BioDevice(ABC):
    """Abstract bio-peripheral driver — the ``/dev/bio`` analogue of a char device.

    Subclasses declare their ``channels`` and implement :meth:`_produce`, which
    returns ``(raw_values, quality)`` for one sample. The bus owns timing,
    conditioning, fd bookkeeping and telemetry; a driver only has to know how to
    yield its next raw reading.
    """

    kind: str = "generic"

    def __init__(self, channels: List[ChannelSpec], sample_rate_hz: float = 1.0):
        self.channels = channels
        self.sample_rate_hz = float(sample_rate_hz)
        self._opened = False

    # -- lifecycle ------------------------------------------------------- #
    def on_open(self) -> None:
        self._opened = True

    def on_close(self) -> None:
        self._opened = False

    # -- the one thing a driver must implement --------------------------- #
    @abstractmethod
    def _produce(self) -> Tuple[Dict[str, float], float]:
        """Return ``({channel_key: raw_value, ...}, quality in [0, 1])``."""
        raise NotImplementedError

    def channel_keys(self) -> List[str]:
        return [c.key for c in self.channels]


# --------------------------------------------------------------------------- #
# A read result
# --------------------------------------------------------------------------- #
@dataclass
class BioBlock:
    """One conditioned sample crossing a file descriptor."""
    fd: int
    path: str
    seq: int
    t_us: float                       # monotonic acquisition time (µs)
    raw: Dict[str, float]             # what the device produced
    value: Dict[str, float]           # Kalman-conditioned values
    quality: float                    # device-reported signal quality [0, 1]
    gain: Dict[str, float]            # per-channel Kalman gain (0 = ignored)
    cond_latency_us: float            # time spent in the inline pipeline

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fd": self.fd, "path": self.path, "seq": self.seq,
            "t_us": round(self.t_us, 1), "raw": self.raw, "value": self.value,
            "quality": round(self.quality, 4),
            "gain": {k: round(v, 4) for k, v in self.gain.items()},
            "cond_latency_us": round(self.cond_latency_us, 2),
        }


# --------------------------------------------------------------------------- #
# A file descriptor: a device opened for reading, with live filter state.
# --------------------------------------------------------------------------- #
@dataclass
class DeviceDescriptor:
    fd: int
    path: str
    device: BioDevice
    opened_us: float
    seq: int = 0
    filters: Dict[str, KalmanScalar] = field(default_factory=dict)
    reads: int = 0

    def __post_init__(self) -> None:
        for c in self.device.channels:
            self.filters[c.key] = KalmanScalar(q=c.q, r0=c.r0)


# --------------------------------------------------------------------------- #
# The bus: mount table + fd table + namespace + telemetry.
# --------------------------------------------------------------------------- #
class DevBio:
    """The ``/dev/bio`` device bus — Layer 1 of the Aegis bio-kernel.

    Thread-model note (integrity): this is a *single-writer* bus. It does not
    claim kernel-level preemption; it is the deterministic ingress substrate the
    higher layers schedule on top of. Calls are O(1) in the number of channels.
    """

    ROOT = "/dev/bio"

    def __init__(self, latency_window: int = 4096):
        self._mounts: Dict[str, BioDevice] = {}     # path -> device
        self._fds: Dict[int, DeviceDescriptor] = {} # fd  -> descriptor
        self._path_fd: Dict[str, int] = {}          # path -> open fd (if any)
        self._next_fd = 3                            # 0/1/2 reserved, like POSIX
        self._lat_us: Deque[float] = deque(maxlen=latency_window)
        self._total_reads = 0

    # -- namespace / mount table ----------------------------------------- #
    def mount(self, node: str, name: str, device: BioDevice) -> str:
        """Register ``device`` at ``/dev/bio/<node>/<name>`` and return the path."""
        path = f"{self.ROOT}/{node}/{name}"
        if path in self._mounts:
            raise FileExistsError(f"{path} already mounted")
        if not isinstance(device, BioDevice) or not device.channels:
            raise TypeError("mount requires a BioDevice with at least one channel")
        self._mounts[path] = device
        return path

    def unmount(self, path: str) -> None:
        if path in self._path_fd:
            self.close(self._path_fd[path])
        self._mounts.pop(path, None)

    def listdir(self, node: Optional[str] = None) -> List[str]:
        """List mounted device paths, optionally filtered to one node."""
        prefix = f"{self.ROOT}/{node}/" if node else f"{self.ROOT}/"
        return sorted(p for p in self._mounts if p.startswith(prefix))

    def nodes(self) -> List[str]:
        out = set()
        for p in self._mounts:
            rest = p[len(self.ROOT) + 1:]
            out.add(rest.split("/", 1)[0])
        return sorted(out)

    def stat(self, path: str) -> Dict[str, Any]:
        dev = self._mounts.get(path)
        if dev is None:
            raise FileNotFoundError(path)
        fd = self._path_fd.get(path)
        d = self._fds.get(fd) if fd is not None else None
        return {
            "path": path, "kind": dev.kind,
            "channels": [{"key": c.key, "unit": c.unit} for c in dev.channels],
            "sample_rate_hz": dev.sample_rate_hz,
            "open": fd is not None, "fd": fd,
            "reads": d.reads if d else 0,
        }

    # -- public accessors (so callers never reach into the fd/mount tables) -- #
    def open_fds(self) -> List[int]:
        return list(self._fds.keys())

    def path_for_fd(self, fd: int) -> str:
        d = self._fds.get(fd)
        if d is None:
            raise OSError(f"bad file descriptor: {fd}")
        return d.path

    def device_at(self, path: str) -> BioDevice:
        dev = self._mounts.get(path)
        if dev is None:
            raise FileNotFoundError(path)
        return dev

    # -- fd lifecycle ---------------------------------------------------- #
    def open(self, path: str) -> int:
        dev = self._mounts.get(path)
        if dev is None:
            raise FileNotFoundError(path)
        if path in self._path_fd:
            return self._path_fd[path]            # idempotent open
        fd = self._next_fd
        self._next_fd += 1
        dev.on_open()
        self._fds[fd] = DeviceDescriptor(fd=fd, path=path, device=dev,
                                         opened_us=_now_us())
        self._path_fd[path] = fd
        return fd

    def close(self, fd: int) -> None:
        d = self._fds.pop(fd, None)
        if d is None:
            return
        d.device.on_close()
        self._path_fd.pop(d.path, None)

    # -- the hot path: read + inline conditioning ------------------------ #
    def read(self, fd: int) -> BioBlock:
        d = self._fds.get(fd)
        if d is None:
            raise OSError(f"bad file descriptor: {fd}")
        raw, quality = d.device._produce()

        # --- inline stochastic pipeline (timed) ------------------------- #
        t0 = time.perf_counter_ns()
        value: Dict[str, float] = {}
        gain: Dict[str, float] = {}
        for key in d.device.channel_keys():
            z = raw.get(key, float("nan"))
            kf = d.filters[key]
            value[key] = kf.update(z, quality)
            gain[key] = kf.last_gain
        cond_us = (time.perf_counter_ns() - t0) / 1_000.0
        # ---------------------------------------------------------------- #

        d.seq += 1
        d.reads += 1
        self._total_reads += 1
        self._lat_us.append(cond_us)
        return BioBlock(
            fd=fd, path=d.path, seq=d.seq, t_us=_now_us(),
            raw=raw, value=value, quality=float(quality),
            gain=gain, cond_latency_us=cond_us,
        )

    def readv(self, fds: List[int]) -> List[BioBlock]:
        """Scatter-read several descriptors in one pass (one scheduler tick)."""
        return [self.read(fd) for fd in fds]

    # -- telemetry ------------------------------------------------------- #
    def latency_stats(self) -> Dict[str, float]:
        if not self._lat_us:
            return {"count": 0, "p50_us": 0.0, "p95_us": 0.0,
                    "max_us": 0.0, "mean_us": 0.0}
        arr = np.fromiter(self._lat_us, dtype=np.float64)
        return {
            "count": int(arr.size),
            "p50_us": float(np.percentile(arr, 50)),
            "p95_us": float(np.percentile(arr, 95)),
            "max_us": float(arr.max()),
            "mean_us": float(arr.mean()),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "root": self.ROOT,
            "mounted": len(self._mounts),
            "open_fds": len(self._fds),
            "nodes": self.nodes(),
            "total_reads": self._total_reads,
            "latency": self.latency_stats(),
        }


# --------------------------------------------------------------------------- #
# Concrete pluggable drivers (synthetic generators for the ICU demo / replay).
# --------------------------------------------------------------------------- #
class _SeededDriver(BioDevice):
    """Base for synthetic drivers: deterministic RNG, optional external feed.

    Pass ``feed`` to replay real data: a callable returning ``(raw, quality)``.
    With no feed, the driver synthesizes a plausible signal around ``baseline``
    with occasional quality dropouts (the lighting/motion artifacts the inline
    Kalman is there to reject).
    """

    def __init__(self, channels, sample_rate_hz, baseline, noise, seed=0,
                 feed: Optional[Callable[[], Tuple[Dict[str, float], float]]] = None):
        super().__init__(channels, sample_rate_hz)
        self._rng = np.random.default_rng(seed)
        self._baseline = dict(baseline)
        self._noise = dict(noise)
        self._feed = feed
        self._t = 0
        self._bias: Dict[str, float] = {}        # crisis / deterioration offset

    def set_bias(self, bias: Dict[str, float]) -> None:
        """Shift channel baselines (physiologic deterioration / Chaos Controller).

        The offset is applied at the sensor, so it flows honestly through the
        inline Kalman → /dev/bio → memory fabric like any real signal change.
        """
        self._bias = dict(bias)

    def _produce(self) -> Tuple[Dict[str, float], float]:
        if self._feed is not None:
            return self._feed()
        self._t += 1
        # Quality occasionally dips (artifact) — distrusted by the inline filter.
        quality = 1.0 if self._rng.random() > 0.12 else float(self._rng.uniform(0.1, 0.4))
        raw: Dict[str, float] = {}
        for key, base in self._baseline.items():
            base = base + self._bias.get(key, 0.0)
            drift = 0.5 * math.sin(self._t / 40.0)         # slow physiologic drift
            sigma = self._noise.get(key, 1.0) / max(quality, 1e-3)
            raw[key] = float(base + drift + self._rng.normal(0.0, sigma))
        return raw, quality


class OpticalRPPGDriver(_SeededDriver):
    """Optical rPPG camera matrix — heart rate + perfusion index from skin."""
    kind = "optical"

    def __init__(self, seed=0, feed=None):
        super().__init__(
            channels=[ChannelSpec("hr", "bpm", q=1e-2, r0=4.0),
                      ChannelSpec("perfusion", "au", q=1e-3, r0=1.0)],
            sample_rate_hz=30.0,
            baseline={"hr": 82.0, "perfusion": 1.4},
            noise={"hr": 3.0, "perfusion": 0.25}, seed=seed, feed=feed)


class ThermalGridDriver(_SeededDriver):
    """Thermal spatial grid — peripheral skin temperature."""
    kind = "thermal"

    def __init__(self, seed=0, feed=None):
        super().__init__(
            channels=[ChannelSpec("temp_c", "C", q=1e-4, r0=0.5)],
            sample_rate_hz=8.0,
            baseline={"temp_c": 37.0}, noise={"temp_c": 0.2}, seed=seed, feed=feed)


class BedsideTelemetryDriver(_SeededDriver):
    """Bedside monitor block — respiratory rate, SpO2, mean arterial pressure."""
    kind = "bedside"

    def __init__(self, seed=0, feed=None):
        super().__init__(
            channels=[ChannelSpec("resp_rate", "bpm", q=1e-2, r0=2.0),
                      ChannelSpec("spo2", "%", q=1e-3, r0=1.0),
                      ChannelSpec("map", "mmHg", q=1e-2, r0=3.0)],
            sample_rate_hz=1.0,
            baseline={"resp_rate": 16.0, "spo2": 98.0, "map": 88.0},
            noise={"resp_rate": 1.5, "spo2": 0.8, "map": 2.5}, seed=seed, feed=feed)


class LabsDriver(_SeededDriver):
    """Enterprise tier — low-frequency structured labs (MIMIC-IV/eICU shaped).

    Slow-moving clinical chemistry the bedside optics can't see: white-blood-cell
    count and serum lactate. Heavy base measurement noise (r0) makes the inline
    Kalman integrate these toward their true level slowly, so they behave like
    intermittently-sampled lab draws rather than a live waveform.
    """
    kind = "labs"

    def __init__(self, seed=0, feed=None):
        super().__init__(
            channels=[ChannelSpec("wbc", "10^3/uL", q=1e-3, r0=6.0),
                      ChannelSpec("lactate", "mmol/L", q=1e-3, r0=5.0)],
            sample_rate_hz=0.1,
            baseline={"wbc": 8.0, "lactate": 1.0},
            noise={"wbc": 0.6, "lactate": 0.18}, seed=seed, feed=feed)


# A registry so a node can be provisioned by kind name (used by the demo/cluster).
DRIVER_REGISTRY: Dict[str, Callable[..., BioDevice]] = {
    "optical": OpticalRPPGDriver,
    "thermal": ThermalGridDriver,
    "bedside": BedsideTelemetryDriver,
    "labs": LabsDriver,
}


def provision_node(bus: DevBio, node: str, kinds=("optical", "thermal", "bedside"),
                   seed: int = 0) -> Dict[str, int]:
    """Mount a standard sensor set for one patient node and open every fd.

    Returns ``{driver_kind: fd}``. This is the per-bed bring-up the multi-tenant
    cluster (Layer 3) will call once per patient.
    """
    fds: Dict[str, int] = {}
    for i, kind in enumerate(kinds):
        drv = DRIVER_REGISTRY[kind](seed=seed * 100 + i)
        path = bus.mount(node, f"{kind}_feed", drv)
        fds[kind] = bus.open(path)
    return fds


# --------------------------------------------------------------------------- #
# Standalone demo / smoke proof.
# --------------------------------------------------------------------------- #
def _demo() -> None:
    bus = DevBio()
    fds = provision_node(bus, "node_0", seed=1)
    print(f"mounted /dev/bio namespace:")
    for p in bus.listdir():
        s = bus.stat(p)
        chans = ",".join(c["key"] for c in s["channels"])
        print(f"  {p:<34} kind={s['kind']:<8} fd={s['fd']} chans=[{chans}]")

    # Read 300 blocks across all fds; measure how much the inline Kalman
    # suppresses noise on the optical HR channel vs. the raw stream.
    raw_hr, cond_hr = [], []
    for _ in range(300):
        for blk in bus.readv(list(fds.values())):
            if "hr" in blk.raw:
                raw_hr.append(blk.raw["hr"])
                cond_hr.append(blk.value["hr"])
    raw_std = float(np.std(raw_hr))
    cond_std = float(np.std(cond_hr))
    print(f"\ninline-Kalman HR noise: raw std={raw_std:.3f} -> "
          f"conditioned std={cond_std:.3f} ({raw_std / max(cond_std, 1e-9):.1f}x quieter)")
    print("bus status:", bus.status())


if __name__ == "__main__":
    _demo()
