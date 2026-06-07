"""Camera-free end-to-end validation of the vision pipeline.

Injects synthetic facial observations with a *known* embedded heart rate (and a
controllable micro-tremor) straight into VisionVitalsPipeline, then checks the
recovered HR — proving the whole frames->vitals chain works without a webcam.

    python -m backend.vision.scripts_validate
"""
from __future__ import annotations

import numpy as np

from backend.vision.face import FaceObservation
from backend.vision.pipeline import VisionVitalsPipeline
from backend.vision import signal_processing as sp


def _synthetic_run(true_hr: float, fs: float = 30.0, seconds: float = 14.0,
                   tremor_hz: float = 0.0, tremor_amp: float = 0.0) -> None:
    rgb = sp.synthesize_rgb(true_hr, duration_s=seconds, fs=fs, seed=int(true_hr))
    n = len(rgb)
    t0 = 1000.0
    rng = np.random.default_rng(7)
    pipe = VisionVitalsPipeline(window_seconds=seconds)

    for i in range(n):
        ts = t0 + i / fs
        # Static landmark cloud + optional tremor + tiny noise.
        base = np.array([[100, 100], [200, 100], [150, 160],
                         [120, 200], [180, 200], [150, 130]], dtype=np.float64)
        if tremor_amp > 0:
            base = base + tremor_amp * np.sin(2 * np.pi * tremor_hz * (i / fs))
        base = base + rng.normal(0, 0.05, base.shape)
        roi = rgb[i].astype(np.float64)
        obs = FaceObservation(
            found=True,
            roi_rgb=roi,
            region_rgb={"forehead": roi, "left_cheek": roi * 0.98, "right_cheek": roi * 1.02},
            pupil_ratio=0.12 + rng.normal(0, 0.002),
            landmarks=base,
            roi_fraction=0.05,
            quality=0.9,
        )
        pipe.add_observation(obs, ts)

    v = pipe.estimate()
    err = abs((v.heart_rate or 0) - true_hr)
    flag = "OK" if err <= 5.0 else "FAIL"
    print(
        f"  true_HR={true_hr:6.1f}  est_HR={v.heart_rate}  err={err:4.1f}  "
        f"HRV={v.hrv}ms  RR={v.respiratory_rate}  perfusion={v.perfusion_index}  "
        f"stress={v.stress_vector}  HRconf={v.confidence.get('heart_rate')}  [{flag}]"
    )
    pipe.close()


def main() -> None:
    print("End-to-end vision pipeline validation (synthetic faces, known HR):")
    for hr in (58.0, 72.0, 88.0, 110.0):
        _synthetic_run(hr)
    print("\nWith injected 8 Hz micro-tremor (stress_vector should rise):")
    _synthetic_run(72.0, tremor_hz=8.0, tremor_amp=1.2)


if __name__ == "__main__":
    main()
