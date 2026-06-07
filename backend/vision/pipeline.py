"""Vision pipeline: frames -> camera-derived physiological vitals.

Maintains rolling time-stamped buffers of facial ROI colour, landmarks, and the
pupil proxy, resamples them onto a uniform time grid (webcam frame timing is
jittery), and runs the rPPG / perfusion / distress estimators to produce a
single structured ``CameraVitals`` reading with per-channel confidence.

The output uses the same internal channel names as the rest of the system so the
digital twin / detector can consume camera vitals exactly like wearable vitals —
weighted by their confidence.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from backend.vision.face import FaceObservation, FaceROIExtractor
from backend.vision import signal_processing as sp
from backend.vision.perfusion import analyze_perfusion
from backend.vision.distress import analyze_distress


@dataclass
class CameraVitals:
    found: bool
    heart_rate: Optional[float] = None
    hrv: Optional[float] = None                # SDNN (ms) — trend-only on webcam
    hrv_rmssd: Optional[float] = None
    respiratory_rate: Optional[float] = None
    perfusion_index: Optional[float] = None    # experimental proxy 0..1
    stress_vector: Optional[float] = None      # experimental proxy 0..1
    confidence: Dict[str, float] = field(default_factory=dict)
    overall_quality: float = 0.0
    fps: float = 0.0
    window_seconds: float = 0.0
    raw: Dict[str, object] = field(default_factory=dict)

    def to_detection_vitals(self, min_conf: float = 0.3) -> Dict[str, float]:
        """Map confident camera channels to internal vital names for the
        detector. Channels below ``min_conf`` are omitted (treated as missing)."""
        out: Dict[str, float] = {}
        c = self.confidence
        if self.heart_rate and c.get("heart_rate", 0) >= min_conf:
            out["heart_rate"] = self.heart_rate
        if self.hrv and c.get("hrv", 0) >= min_conf:
            out["hrv"] = self.hrv
        if self.respiratory_rate and c.get("respiratory_rate", 0) >= min_conf:
            out["respiratory_rate"] = self.respiratory_rate
        return out


class VisionVitalsPipeline:
    def __init__(self, window_seconds: float = 12.0, target_fps: float = 30.0):
        self.window_seconds = window_seconds
        self.target_fps = target_fps
        maxlen = int(window_seconds * target_fps * 1.5) + 30
        self._t: Deque[float] = deque(maxlen=maxlen)
        self._roi: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._regions: Deque[Dict[str, np.ndarray]] = deque(maxlen=maxlen)
        self._landmarks: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._pupil: Deque[float] = deque(maxlen=maxlen)
        self._found_flags: Deque[int] = deque(maxlen=maxlen)
        self._extractor: Optional[FaceROIExtractor] = None

    # ------------------------------------------------------------------ #
    def add_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> FaceObservation:
        if self._extractor is None:
            self._extractor = FaceROIExtractor()
        obs = self._extractor(frame_bgr)
        self.add_observation(obs, timestamp if timestamp is not None else time.time())
        return obs

    def add_observation(self, obs: FaceObservation, timestamp: float) -> None:
        """Lower-level ingest (also used for camera-free testing)."""
        self._t.append(timestamp)
        self._found_flags.append(1 if obs.found else 0)
        if obs.found and obs.roi_rgb is not None:
            self._roi.append(obs.roi_rgb)
            self._regions.append(obs.region_rgb)
            self._landmarks.append(obs.landmarks if obs.landmarks is not None else np.zeros((6, 2)))
            self._pupil.append(obs.pupil_ratio)
        else:
            # Keep buffers time-aligned by repeating last known / zeros.
            self._roi.append(self._roi[-1] if self._roi else np.zeros(3))
            self._regions.append(self._regions[-1] if self._regions else {})
            self._landmarks.append(self._landmarks[-1] if self._landmarks else np.zeros((6, 2)))
            self._pupil.append(self._pupil[-1] if self._pupil else 0.0)

    # ------------------------------------------------------------------ #
    def estimate(self) -> CameraVitals:
        n = len(self._t)
        if n < 8:
            return CameraVitals(found=False)
        t = np.array(self._t)
        span = float(t[-1] - t[0])
        if span < 4.0:
            return CameraVitals(found=False, window_seconds=round(span, 1))

        found_frac = float(np.mean(self._found_flags))
        fps = (n - 1) / span if span > 0 else self.target_fps

        # Resample ROI + landmarks + pupil onto a uniform grid for clean DSP.
        grid_fs = self.target_fps
        grid = np.arange(t[0], t[-1], 1.0 / grid_fs)
        roi = np.stack(self._roi)                       # (n, 3)
        roi_u = np.stack([np.interp(grid, t, roi[:, c]) for c in range(3)], axis=1)

        # Per-region redness time series (for spatial perfusion dispersion).
        region_redness: Dict[str, List[float]] = {}
        for name in ("forehead", "left_cheek", "right_cheek"):
            vals = []
            for rd in self._regions:
                v = rd.get(name)
                if v is not None:
                    s = float(np.sum(v)) or 1.0
                    vals.append(float(v[0] / s))
            if vals:
                region_redness[name] = vals

        pulse = sp.analyze_window(roi_u, grid_fs, method="pos")
        perf = analyze_perfusion(roi_u, region_redness, grid_fs)

        # Rich HRV feature set from the rPPG pulse (time/freq/nonlinear + HeRO
        # descriptors) — the same features the HRV->sepsis model consumes, so the
        # camera can drive that model directly once it is trained on labelled data.
        from backend.physio import hrv as _hrv
        _pulse_sig = sp.pos_pulse(roi_u, grid_fs)
        _rr = _hrv.rr_intervals_ms(_hrv.detect_ppg_peaks(_pulse_sig, grid_fs), grid_fs)
        hrv_features = _hrv.hrv_features(_rr).as_dict()

        # Landmarks/pupil resampled for distress analysis.
        lm = np.stack(self._landmarks)                  # (n, K, 2)
        K = lm.shape[1]
        lm_u = np.stack([
            np.stack([np.interp(grid, t, lm[:, k, ax]) for ax in range(2)], axis=1)
            for k in range(K)
        ], axis=1)                                      # (len(grid), K, 2)
        pupil_u = np.interp(grid, t, np.array(self._pupil))
        dist = analyze_distress(lm_u, pupil_u, grid_fs)

        # Confidences: rPPG quality gates HR/HRV/RR; coverage gates everything.
        coverage = float(np.clip(found_frac, 0.0, 1.0))
        hr_conf = round(pulse.quality * coverage, 3)
        hrv_conf = round(pulse.quality * coverage * (1.0 if pulse.n_beats >= 6 else 0.4), 3)
        rr_conf = round(0.6 * pulse.quality * coverage, 3)
        perf_conf = round(perf.quality * coverage, 3)
        stress_conf = round(dist.quality * coverage, 3)

        overall = round(float(np.mean([hr_conf, perf_conf, stress_conf])), 3)

        return CameraVitals(
            found=found_frac > 0.3,
            heart_rate=pulse.hr_bpm or None,
            hrv=pulse.hrv_sdnn_ms or None,
            hrv_rmssd=pulse.hrv_rmssd_ms or None,
            respiratory_rate=pulse.resp_bpm or None,
            perfusion_index=perf.perfusion_proxy,
            stress_vector=dist.stress_vector,
            confidence={
                "heart_rate": hr_conf, "hrv": hrv_conf, "respiratory_rate": rr_conf,
                "perfusion_index": perf_conf, "stress_vector": stress_conf,
            },
            overall_quality=overall,
            fps=round(fps, 1),
            window_seconds=round(span, 1),
            raw={
                "pulse": pulse.__dict__, "perfusion": perf.__dict__,
                "distress": dist.__dict__, "found_fraction": round(found_frac, 3),
                "hrv_features": hrv_features,
            },
        )

    def close(self) -> None:
        if self._extractor is not None:
            self._extractor.close()
