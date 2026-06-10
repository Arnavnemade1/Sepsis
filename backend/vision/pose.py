"""Head-pose estimation + posture handling for bedside cameras.

Critical real-world fact: most monitored patients are **lying down**, and the
camera is **perched above/around the bed** — so the face arrives rotated,
reclined, or even inverted, never as a tidy upright passport photo. A naive
pipeline rejects these faces; this module makes the system *orientation-robust*:

  • estimate head pitch / yaw / roll from facial landmarks,
  • classify posture (upright / reclined / supine / lateral / inverted) given the
    camera mount,
  • compute a frontal-quality factor that penalizes only PROFILE views (extreme
    yaw/pitch, where the skin ROI genuinely degrades) and NOT in-plane rotation
    (roll), since a rolled/inverted face still yields a perfectly good rPPG ROI.

The rPPG/skin ROIs are landmark-defined, so they are already roll-invariant; this
module adds the explicit posture awareness so a supine patient is monitored
normally instead of being dropped as "no upright face".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class HeadPose:
    pitch: float            # deg: + looking down / chin toward camera
    yaw: float              # deg: + turned to subject's left
    roll: float             # deg: in-plane rotation (0 = upright, ±180 = inverted)
    posture: str            # upright | reclined | lateral | supine | inverted
    frontal_quality: float  # 0..1 — usable skin-ROI quality from this angle
    valid: bool             # a face was located (true even if rotated/inverted)

    def to_dict(self) -> Dict[str, float]:
        return {"pitch": round(self.pitch, 1), "yaw": round(self.yaw, 1), "roll": round(self.roll, 1),
                "posture": self.posture, "frontal_quality": round(self.frontal_quality, 3), "valid": self.valid}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def head_pose(landmarks: Dict[str, np.ndarray], camera_mount: str = "overhead") -> HeadPose:
    """Estimate head pose from named 3D facial landmarks.

    landmarks needs: left_eye, right_eye, nose, mouth_left, mouth_right, chin
    (each a 3-vector; MediaPipe normalized coords are fine — x right, y down,
    z toward camera). Robust to rotation/inversion.
    """
    try:
        le = np.asarray(landmarks["left_eye"], float); re = np.asarray(landmarks["right_eye"], float)
        nose = np.asarray(landmarks["nose"], float)
        ml = np.asarray(landmarks["mouth_left"], float); mr = np.asarray(landmarks["mouth_right"], float)
    except Exception:
        return HeadPose(0, 0, 0, "unknown", 0.0, False)

    eye_mid = (le + re) / 2.0
    mouth_mid = (ml + mr) / 2.0
    x_axis = _unit(re - le)                      # ear-to-ear (subject right)
    down = _unit(mouth_mid - eye_mid)            # eyes -> mouth
    z_axis = _unit(np.cross(x_axis, down))       # out of the face toward camera
    y_axis = _unit(np.cross(z_axis, x_axis))     # true facial up/down

    # roll: rotation of the eye line in the image plane.
    roll = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    # yaw / pitch from the facial forward axis relative to the camera (−z).
    yaw = math.degrees(math.atan2(z_axis[0], abs(z_axis[2]) + 1e-6))
    pitch = math.degrees(math.atan2(z_axis[1], abs(z_axis[2]) + 1e-6))

    # frontal quality: profile (large |yaw|/|pitch|) hurts the skin ROI; roll does not.
    fq = float(np.clip(1.0 - (abs(yaw) / 70.0) ** 2 - (abs(pitch) / 70.0) ** 2, 0.0, 1.0))
    posture = _classify(pitch, yaw, roll, camera_mount)
    return HeadPose(pitch, yaw, roll, posture, fq, True)


def _classify(pitch: float, yaw: float, roll: float, mount: str) -> str:
    """Interpret pose into a bedside posture given the camera mounting.

    For an OVERHEAD/perched camera looking down at a bed, a supine patient's face
    presents near-upright but rolled (head turned on the pillow) or inverted
    (camera at the head of the bed). Lateral = turned to one side (large yaw)."""
    ar = abs(roll)
    if abs(yaw) > 45:
        return "lateral"
    if ar > 135:
        return "inverted" if mount == "overhead" else "supine"
    if mount == "overhead":
        # Bedside: meaningful roll or backward pitch ⇒ reclining/lying.
        if ar > 30 or pitch < -20:
            return "supine" if ar > 60 else "reclined"
        return "upright"
    if pitch < -25:
        return "reclined"
    return "upright"


def posture_signal_gate(pose: Optional[HeadPose]) -> float:
    """Quality multiplier for downstream signal extraction. Any located face —
    upright, rolled, supine, or inverted — is monitored; only true profile views
    (where the skin ROI is genuinely poor) are down-weighted."""
    if pose is None or not pose.valid:
        return 1.0   # no pose info: don't penalize
    return float(np.clip(0.35 + 0.65 * pose.frontal_quality, 0.0, 1.0))
