"""Continuous sequential-Bayesian sepsis monitor for the camera stream.

A single camera reading is weak and noisy. But the camera observes *continuously*,
so we integrate the per-reading log-odds evidence over time with a forgetting
factor — a leaky sequential-probability-ratio filter. Sustained physiological
deterioration accumulates into a confident posterior, while random noise averages
out. This temporal integration is the main reason a contactless monitor can reach
high accuracy: it trades latency (seconds-to-minutes) for a dramatically better
signal-to-noise ratio, and a hysteresis band keeps the alarm from flapping.

    posterior_logodds_t = prior + decay·(posterior_{t-1} − prior) + evidence_t
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from backend.vision.camera_detector import CameraSepsisDetector, CameraEvidence


@dataclass
class MonitorState:
    timestamp: float
    probability: float                 # integrated posterior P(sepsis)
    log_odds: float
    risk_level: str                    # LOW | MEDIUM | HIGH
    alarm: bool
    trend: str                         # RISING | FALLING | STABLE
    quality: float
    n_updates: int
    instant_prob: float                # single-frame probability (for comparison)
    evidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class ContinuousSepsisMonitor:
    def __init__(
        self,
        detector: Optional[CameraSepsisDetector] = None,
        decay: float = 0.94,           # per-reading forgetting (≈ minutes of memory)
        p_high: float = 0.50,          # alarm raise threshold
        p_low: float = 0.20,           # alarm clear threshold (hysteresis)
        p_med: float = 0.20,
        clip: float = 8.0,
    ):
        self.detector = detector or CameraSepsisDetector()
        self.prior = self.detector.prior_log_odds
        self.decay = decay
        self.p_high, self.p_low, self.p_med = p_high, p_low, p_med
        self.clip = clip
        self.L = self.prior
        self._alarm = False
        self._n = 0
        self._prob_hist: list = []

    # ------------------------------------------------------------------ #
    def update(self, reading: Dict[str, float], quality: float = 1.0,
               timestamp: Optional[float] = None) -> MonitorState:
        ts = timestamp if timestamp is not None else time.time()
        ev: CameraEvidence = self.detector.score(reading, quality)
        # leaky integration toward the prior + new evidence
        self.L = self.prior + self.decay * (self.L - self.prior) + ev.log_odds
        self.L = float(np.clip(self.L, self.prior - self.clip, self.prior + self.clip))
        prob = 1.0 / (1.0 + np.exp(-self.L))
        self._n += 1
        self._prob_hist.append(prob)
        if len(self._prob_hist) > 30:
            self._prob_hist.pop(0)

        # hysteresis alarm
        if prob >= self.p_high:
            self._alarm = True
        elif prob < self.p_low:
            self._alarm = False

        risk = "HIGH" if prob >= self.p_high else ("MEDIUM" if prob >= self.p_med else "LOW")
        return MonitorState(
            timestamp=ts, probability=float(prob), log_odds=float(self.L),
            risk_level=risk, alarm=self._alarm, trend=self._trend(),
            quality=ev.quality, n_updates=self._n, instant_prob=ev.instant_prob,
            evidence=ev.contributions,
        )

    def update_from_camera(self, cam: Any, timestamp: Optional[float] = None) -> MonitorState:
        reading = CameraSepsisDetector.from_camera_vitals(cam)
        return self.update(reading, getattr(cam, "overall_quality", 1.0), timestamp)

    def _trend(self) -> str:
        if len(self._prob_hist) < 6:
            return "STABLE"
        slope = float(np.polyfit(range(len(self._prob_hist)), self._prob_hist, 1)[0])
        return "RISING" if slope > 0.004 else ("FALLING" if slope < -0.004 else "STABLE")

    def reset(self) -> None:
        self.L = self.prior; self._alarm = False; self._n = 0; self._prob_hist = []
