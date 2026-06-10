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
from typing import Any, Dict, List, Optional, Tuple

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
_LIPS = [61, 291, 0, 17, 37, 267, 84, 314, 13, 14]
_LEFT_EYE = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Named landmarks used for head-pose estimation (see pose.py).
_POSE_IDX = {"right_eye": 33, "left_eye": 263, "nose": 1,
             "mouth_right": 61, "mouth_left": 291, "chin": 199}
# Box groups drawn as "areas of interest" overlays.
_BOX_GROUPS = {"forehead": _FOREHEAD, "left_cheek": _LEFT_CHEEK, "right_cheek": _RIGHT_CHEEK,
               "lips": _LIPS, "left_eye": _LEFT_EYE, "right_eye": _RIGHT_EYE}


def _bbox(pts: np.ndarray, idx: List[int]) -> Optional[Dict[str, int]]:
    """Axis-aligned bounding box (pixels) around a landmark group."""
    valid = [i for i in idx if i < len(pts)]
    if len(valid) < 2:
        return None
    p = pts[valid]
    x0, y0 = p.min(axis=0); x1, y1 = p.max(axis=0)
    pad = 0.12 * max(x1 - x0, y1 - y0, 1)
    return {"x": int(x0 - pad), "y": int(y0 - pad),
            "w": int((x1 - x0) + 2 * pad), "h": int((y1 - y0) + 2 * pad)}


def _eye_openness(pts: np.ndarray) -> float:
    """Eye aspect ratio proxy (0 closed .. ~0.4 wide) averaged over both eyes."""
    def ear(eye):
        if max(eye) >= len(pts):
            return 0.0
        p = pts[eye]
        w = np.linalg.norm(p[0] - p[3]) + 1e-6
        h = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / 2.0
        return float(h / w)
    return float(np.clip((ear(_LEFT_EYE) + ear(_RIGHT_EYE)) / 2.0, 0.0, 1.0))


@dataclass
class FaceObservation:
    found: bool
    roi_rgb: Optional[np.ndarray] = None          # (3,) mean RGB over skin ROIs
    region_rgb: Dict[str, np.ndarray] = field(default_factory=dict)  # per-region mean RGB
    pupil_ratio: float = 0.0                       # iris diameter / inter-ocular dist
    landmarks: Optional[np.ndarray] = None         # (K, 2) pixel coords (stable pts)
    roi_fraction: float = 0.0                       # skin pixels / frame (coverage)
    quality: float = 0.0
    boxes: List[Dict[str, Any]] = field(default_factory=list)   # areas-of-interest overlay
    pose_landmarks: Dict[str, np.ndarray] = field(default_factory=dict)  # named 3D pts for pose
    eye_openness: float = 0.0


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
        lm3 = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)   # normalized 3D for pose

        regions = {
            "forehead": self._region_mean(rgb, pts, _FOREHEAD),
            "left_cheek": self._region_mean(rgb, pts, _LEFT_CHEEK),
            "right_cheek": self._region_mean(rgb, pts, _RIGHT_CHEEK),
            "lips": self._region_mean(rgb, pts, _LIPS),
        }
        region_rgb = {k: v for k, (v, _) in regions.items() if v is not None}
        skin = {k: v for k, v in region_rgb.items() if k != "lips"}
        coverage = float(np.mean([f for v, f in regions.values() if v is not None])) if region_rgb else 0.0
        if not skin:
            return FaceObservation(found=False)

        roi_rgb = np.mean(list(skin.values()), axis=0)
        pupil_ratio = self._pupil_ratio(pts)
        stable = pts[_STABLE] if pts.shape[0] > max(_STABLE) else pts[:6]

        # Areas-of-interest boxes for the recognition overlay.
        boxes = []
        for name, idx in _BOX_GROUPS.items():
            bb = _bbox(pts, idx)
            if bb:
                bb["name"] = name
                bb["status"] = "tracking" if name in region_rgb or "eye" in name or name == "lips" else "—"
                boxes.append(bb)
        pose_lm = {k: lm3[i] for k, i in _POSE_IDX.items() if i < len(lm3)}

        quality = float(np.clip(0.6 * min(1.0, coverage * 40) + 0.4 * (len(skin) / 3), 0, 1))
        return FaceObservation(
            found=True,
            roi_rgb=roi_rgb,
            region_rgb=region_rgb,
            pupil_ratio=round(pupil_ratio, 4),
            landmarks=stable,
            roi_fraction=round(coverage, 4),
            quality=round(quality, 3),
            boxes=boxes,
            pose_landmarks=pose_lm,
            eye_openness=round(_eye_openness(pts), 4),
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


# --------------------------------------------------------------------------- #
# Recognition overlay: draw the areas-of-interest boxes onto a frame.
# --------------------------------------------------------------------------- #
_BOX_COLOR = {  # BGR
    "forehead": (255, 200, 60), "left_cheek": (255, 200, 60), "right_cheek": (255, 200, 60),
    "lips": (120, 120, 255), "left_eye": (120, 255, 160), "right_eye": (120, 255, 160),
}


def annotate_frame(frame_bgr: "np.ndarray", obs: FaceObservation) -> "np.ndarray":
    """Return a copy of the frame with labelled boxes around each area of
    interest (rPPG skin ROIs, eyes, lips). For the live recognition overlay."""
    if cv2 is None or not obs.found:
        return frame_bgr
    out = frame_bgr.copy()
    for b in obs.boxes:
        col = _BOX_COLOR.get(b["name"], (200, 200, 200))
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 1)
        cv2.putText(out, b["name"], (x, max(10, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    return out
