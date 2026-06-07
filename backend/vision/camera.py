"""Runtime webcam capture loop for the optical biosensor.

Requires a real camera + a face in view, so this is a runtime tool (not part of
the automated tests). On macOS the first run triggers a camera-permission prompt.

    python -m backend.vision.camera --seconds 30        # live HR/HRV/perfusion
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

from backend.vision.pipeline import VisionVitalsPipeline, CameraVitals


def stream_vitals(
    duration_s: float = 30.0,
    camera_index: int = 0,
    report_every_s: float = 2.0,
    on_reading=None,
):
    """Capture from the webcam and yield/print CameraVitals periodically."""
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"OpenCV not available: {exc}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pipe = VisionVitalsPipeline()
    start = time.time()
    last_report = 0.0
    try:
        while time.time() - start < duration_s:
            ok, frame = cap.read()
            if not ok:
                continue
            pipe.add_frame(frame, time.time())
            now = time.time() - start
            if now - last_report >= report_every_s:
                last_report = now
                v = pipe.estimate()
                _print_reading(now, v)
                if on_reading is not None:
                    on_reading(v)
    finally:
        cap.release()
        pipe.close()


def _print_reading(t: float, v: CameraVitals) -> None:
    if not v.found:
        print(f"[{t:5.1f}s] no stable face (window {v.window_seconds}s)")
        return
    c = v.confidence
    print(
        f"[{t:5.1f}s] HR={v.heart_rate} bpm (c={c.get('heart_rate')}) "
        f"HRV={v.hrv}ms RR={v.respiratory_rate} "
        f"perfusion={v.perfusion_index} stress={v.stress_vector} "
        f"| fps={v.fps} quality={v.overall_quality}"
    )


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    print("Starting optical biosensor — keep your face in view, well lit, still.")
    stream_vitals(args.seconds, args.camera)
