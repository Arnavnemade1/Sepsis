"""Face ROI + landmark extraction for optical biosensing.

Uses MediaPipe FaceMesh (478 landmarks incl. iris) to locate the skin regions
with the strongest rPPG signal (forehead + cheeks), an iris/pupil proxy, and the
full landmark set used for micro-tremor analysis. Falls back to an OpenCV Haar
cascade (forehead-band ROI only) if MediaPipe is unavailable.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# MediaPipe 0.10.x ships the "Tasks" API (mp.tasks), not the legacy mp.solutions.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_MP = True
except Exception:  # pragma: no cover
    _HAS_MP = False

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model() -> Optional[str]:
    """Download the FaceLandmarker model bundle on first use; return its path."""
    from backend.ml import config  # reuse repo paths
    model_dir = os.path.join(config.REPO_ROOT, "data", "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "face_landmarker.task")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
        return path
    except Exception as exc:  # pragma: no cover
        print(f"[vision] could not download FaceLandmarker model: {exc}")
        return None

# FaceMesh landmark index groups (standard 468-point topology).
_FOREHEAD = [67, 109, 10, 338, 297, 332, 333, 299, 296, 336, 9, 107, 66, 105]
_LEFT_CHEEK = [50, 101, 118, 117, 123, 116, 205, 206, 203, 36]
_RIGHT_CHEEK = [280, 330, 347, 346, 352, 345, 425, 426, 423, 266]
_RIGHT_IRIS = [468, 469, 470, 471, 472]
_LEFT_IRIS = [473, 474, 475, 476, 477]
# Stable points for head-motion / tremor reference (nose, eye corners).
_STABLE = [1, 33, 263, 61, 291, 199]


@dataclass
class FaceObservation:
    found: bool
    roi_rgb: Optional[np.ndarray] = None          # (3,) mean RGB over skin ROIs
    region_rgb: Dict[str, np.ndarray] = field(default_factory=dict)  # per-region mean RGB
    pupil_ratio: float = 0.0                       # iris diameter / inter-ocular dist
    landmarks: Optional[np.ndarray] = None         # (K, 2) pixel coords (stable pts)
    roi_fraction: float = 0.0                       # skin pixels / frame (coverage)
    quality: float = 0.0


class FaceROIExtractor:
    """Stateless per-frame extractor. Feed BGR frames (OpenCV convention)."""

    def __init__(self) -> None:
        self._landmarker = None
        self._haar = None
        if _HAS_MP:
            model_path = _ensure_model()
            if model_path:
                opts = mp_vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model_path),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        if self._landmarker is None and cv2 is not None:  # pragma: no cover
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    # ------------------------------------------------------------------ #
    def __call__(self, frame_bgr: np.ndarray) -> FaceObservation:
        if self._landmarker is not None:
            return self._extract_mesh(frame_bgr)
        if self._haar is not None:  # pragma: no cover
            return self._extract_haar(frame_bgr)
        return FaceObservation(found=False)

    # ------------------------------------------------------------------ #
    def _extract_mesh(self, frame_bgr: np.ndarray) -> FaceObservation:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect(mp_image)
        if not res.face_landmarks:
            return FaceObservation(found=False)

        lm = res.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        regions = {
            "forehead": self._region_mean(rgb, pts, _FOREHEAD),
            "left_cheek": self._region_mean(rgb, pts, _LEFT_CHEEK),
            "right_cheek": self._region_mean(rgb, pts, _RIGHT_CHEEK),
        }
        region_rgb = {k: v for k, (v, _) in regions.items() if v is not None}
        coverage = float(np.mean([f for _, f in regions.values()])) if regions else 0.0
        if not region_rgb:
            return FaceObservation(found=False)

        roi_rgb = np.mean(list(region_rgb.values()), axis=0)
        pupil_ratio = self._pupil_ratio(pts)
        stable = pts[_STABLE] if pts.shape[0] > max(_STABLE) else pts[:6]

        quality = float(np.clip(0.6 * min(1.0, coverage * 40) + 0.4 * (len(region_rgb) / 3), 0, 1))
        return FaceObservation(
            found=True,
            roi_rgb=roi_rgb,
            region_rgb=region_rgb,
            pupil_ratio=round(pupil_ratio, 4),
            landmarks=stable,
            roi_fraction=round(coverage, 4),
            quality=round(quality, 3),
        )

    def _region_mean(self, rgb: np.ndarray, pts: np.ndarray, idx: List[int]) -> Tuple[Optional[np.ndarray], float]:
        h, w = rgb.shape[:2]
        valid = [i for i in idx if i < len(pts)]
        if len(valid) < 3:
            return None, 0.0
        poly = pts[valid].astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(poly), 255)
        area = int(mask.sum() // 255)
        if area < 30:
            return None, 0.0
        mean = cv2.mean(rgb, mask=mask)[:3]   # returns RGB means
        return np.array(mean, dtype=np.float64), area / float(h * w)

    def _pupil_ratio(self, pts: np.ndarray) -> float:
        """Iris diameter normalized by inter-ocular distance (scale-invariant
        pupil/iris proxy). Real pupillometry needs IR + close-up; this is a
        coarse webcam proxy for *relative* change tracking only."""
        if pts.shape[0] <= max(_LEFT_IRIS):
            return 0.0
        r_iris, l_iris = pts[_RIGHT_IRIS], pts[_LEFT_IRIS]
        r_d = np.linalg.norm(r_iris[1] - r_iris[3]) + np.linalg.norm(r_iris[2] - r_iris[4])
        l_d = np.linalg.norm(l_iris[1] - l_iris[3]) + np.linalg.norm(l_iris[2] - l_iris[4])
        iris_d = (r_d + l_d) / 4.0
        # inter-ocular distance: right-iris-centre (468) to left-iris-centre (473)
        ocular = np.linalg.norm(pts[468] - pts[473])
        return float(iris_d / ocular) if ocular > 1e-6 else 0.0

    # ------------------------------------------------------------------ #
    def _extract_haar(self, frame_bgr: np.ndarray) -> FaceObservation:  # pragma: no cover
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        if len(faces) == 0:
            return FaceObservation(found=False)
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3])[-1]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        fh = rgb[y:y + int(0.25 * h), x + int(0.2 * w):x + int(0.8 * w)]
        if fh.size == 0:
            return FaceObservation(found=False)
        roi = fh.reshape(-1, 3).mean(axis=0)
        return FaceObservation(
            found=True, roi_rgb=roi, region_rgb={"forehead": roi},
            roi_fraction=(fh.shape[0] * fh.shape[1]) / float(rgb.shape[0] * rgb.shape[1]),
            quality=0.5,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
