"""Verification suite for Layer 1 — the /dev/bio device-driver bus.

Proves the real, claimable properties (no naming without substance):

  1. Namespace + fd lifecycle behave like a POSIX char device (mount/open/
     read/close/unmount, idempotent open, bad-fd raises).
  2. The inline Kalman pipeline genuinely suppresses sensor noise.
  3. Adaptive R: low-quality samples move the estimate far less than high-quality
     ones (the filter distrusts artifacts), not a no-op.
  4. Replay drivers can be fed real external arrays (MIMIC/eICU path).
  5. Microsecond conditioning-latency telemetry is recorded and bounded.

    .venv/bin/python -m backend.aegis.kernel.verify_devbio
"""
from __future__ import annotations

import numpy as np

from backend.aegis.kernel.devbio import (
    BedsideTelemetryDriver, BioDevice, ChannelSpec, DevBio, KalmanScalar,
    OpticalRPPGDriver, provision_node,
)


def _ok(msg): print(f"      [PASS] {msg}")


def test_namespace_and_fd_lifecycle() -> None:
    print("[1] /dev/bio namespace + fd lifecycle")
    bus = DevBio()
    path = bus.mount("node_0", "optical_feed", OpticalRPPGDriver(seed=1))
    assert path == "/dev/bio/node_0/optical_feed"
    assert bus.listdir() == [path]
    assert bus.nodes() == ["node_0"]

    fd = bus.open(path)
    assert fd >= 3                                  # 0/1/2 reserved, POSIX-style
    assert bus.open(path) == fd                     # idempotent open
    assert bus.stat(path)["open"] is True

    blk = bus.read(fd)
    assert set(blk.value) == {"hr", "perfusion"}
    assert blk.seq == 1 and bus.stat(path)["reads"] == 1

    try:
        bus.read(9999); assert False, "bad fd must raise"
    except OSError:
        pass

    bus.close(fd)
    assert bus.stat(path)["open"] is False
    bus.unmount(path)
    assert bus.listdir() == []
    _ok("mount/open/read/close/unmount, idempotent open, bad-fd raises")


def test_inline_kalman_denoises() -> None:
    print("[2] inline Kalman suppresses sensor noise")
    bus = DevBio()
    fds = provision_node(bus, "node_0", seed=2)
    raw, cond = [], []
    for _ in range(400):
        blk = bus.read(fds["optical"])
        raw.append(blk.raw["hr"]); cond.append(blk.value["hr"])
    ratio = np.std(raw) / max(np.std(cond), 1e-9)
    assert ratio >= 3.0, f"expected >=3x denoise, got {ratio:.2f}x"
    _ok(f"optical HR noise reduced {ratio:.1f}x (raw std "
        f"{np.std(raw):.2f} -> {np.std(cond):.2f})")


def test_adaptive_r_distrusts_low_quality() -> None:
    print("[3] adaptive R distrusts low-quality samples")
    # Same warm filter, same large measurement jump, different quality.
    def jump_move(quality: float) -> float:
        kf = KalmanScalar(q=1e-3, r0=1.0)
        for _ in range(50):
            kf.update(80.0, quality=1.0)            # settle at 80
        before = kf.x
        kf.update(120.0, quality=quality)           # +40 bpm artifact-shaped jump
        return kf.x - before

    hi = jump_move(1.0)
    lo = jump_move(0.1)
    assert hi > lo * 3, f"low-quality should move much less: hi={hi:.3f} lo={lo:.3f}"
    _ok(f"high-quality jump moves estimate {hi:.2f}; low-quality only "
        f"{lo:.2f} ({hi / max(lo, 1e-9):.1f}x less trusted)")


def test_replay_feed_path() -> None:
    print("[4] replay driver accepts real external arrays")
    series = iter([({"map": v}, 1.0) for v in (70, 72, 95, 60, 88)])

    class ReplayMAP(BioDevice):
        kind = "replay"
        def __init__(self):
            super().__init__([ChannelSpec("map", "mmHg", r0=2.0)], 1.0)
        def _produce(self):
            return next(series)

    bus = DevBio()
    fd = bus.open(bus.mount("node_9", "art_line", ReplayMAP()))
    vals = [bus.read(fd).raw["map"] for _ in range(5)]
    assert vals == [70, 72, 95, 60, 88]
    _ok("external feed streamed verbatim through the fd")


def test_latency_telemetry() -> None:
    print("[5] microsecond conditioning-latency telemetry")
    bus = DevBio()
    fds = provision_node(bus, "node_0", seed=3)
    for _ in range(200):
        bus.readv(list(fds.values()))
    s = bus.latency_stats()
    assert s["count"] == 600 and s["p50_us"] > 0.0
    assert s["p95_us"] >= s["p50_us"] <= s["max_us"]
    _ok(f"p50={s['p50_us']:.2f}µs p95={s['p95_us']:.2f}µs "
        f"max={s['max_us']:.1f}µs over {s['count']} conditioning ops")


def main() -> None:
    print("=" * 64)
    print("AegisOS Layer 1 — /dev/bio device-driver bus verification")
    print("=" * 64)
    test_namespace_and_fd_lifecycle()
    test_inline_kalman_denoises()
    test_adaptive_r_distrusts_low_quality()
    test_replay_feed_path()
    test_latency_telemetry()
    print("-" * 64)
    print("ALL LAYER-1 CHECKS PASSED")


if __name__ == "__main__":
    main()
