"""Personalized per-reading sepsis evidence from camera-derived vitals.

Sepsis is a *deviation from the patient's own normal*, so this detector scores
each reading **relative to an adaptive personal baseline** and emits calibrated
log-odds evidence — not an absolute label. The features are exactly those a
camera can observe, weighted by the physiology that actually precedes sepsis:

  * HRV collapse        — reduced SDNN/RMSSD, reduced sample entropy, raised
                          sample asymmetry (the published HeRO signature) → the
                          single strongest early signal.
  * Tachycardia         — HR rising above the personal baseline.
  * Tachypnea           — respiratory rate elevated.
  * Perfusion drop      — optical perfusion proxy falling (possible shunting).
  * Sympathetic stress  — optical stress vector elevated.

Every contribution is multiplied by the signal quality, so a noisy frame adds no
evidence rather than a wrong one. The per-reading log-odds are integrated over
time by ``ContinuousSepsisMonitor`` (monitor.py), which is where weak per-frame
signals become a confident, low-false-alarm continuous risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

# Population priors for the personal baseline (adapts toward the subject).
_BASE0 = {"heart_rate": 75.0, "hrv_sdnn": 50.0, "hrv_rmssd": 35.0,
          "sample_entropy": 1.15, "respiratory_rate": 15.0}
# Per-reading evidence weights (log-odds). Modest by design — the monitor
# integrates many readings; one frame should never dominate.
_W = {"hrv_sdnn": 0.45, "hrv_rmssd": 0.30, "sample_entropy": 0.55,
      "sample_asymmetry": 0.40, "heart_rate": 0.40, "respiratory_rate": 0.35,
      "perfusion": 0.40, "stress": 0.25}


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class CameraEvidence:
    log_odds: float                       # total per-reading evidence (log-odds)
    instant_prob: float                   # sigmoid(prior + log_odds), single-frame
    quality: float
    contributions: Dict[str, float] = field(default_factory=dict)
    baseline: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"log_odds": round(self.log_odds, 4), "instant_prob": round(self.instant_prob, 4),
                "quality": round(self.quality, 3),
                "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
                "baseline": {k: round(v, 2) for k, v in self.baseline.items()}}


class CameraSepsisDetector:
    def __init__(self, prior_prevalence: float = 0.03, baseline_lr: float = 0.03):
        self.prior_log_odds = float(np.log(prior_prevalence / (1 - prior_prevalence)))
        self.baseline = dict(_BASE0)
        self.baseline_lr = baseline_lr
        self._warm = 0

    # ------------------------------------------------------------------ #
    def score(self, reading: Dict[str, float], quality: float = 1.0) -> CameraEvidence:
        """Score one camera reading. ``reading`` keys (any subset):
        heart_rate, hrv_sdnn, hrv_rmssd, sample_entropy, sample_asymmetry,
        respiratory_rate, perfusion_index, stress_vector."""
        q = float(np.clip(quality, 0.0, 1.0))
        b = self.baseline
        c: Dict[str, float] = {}

        def have(k):
            v = reading.get(k)
            return v is not None and not (isinstance(v, float) and np.isnan(v)) and v > 0

        # --- HRV collapse (HeRO) --- #
        if have("hrv_sdnn"):
            ratio = reading["hrv_sdnn"] / max(b["hrv_sdnn"], 1e-6)
            c["hrv_sdnn"] = _W["hrv_sdnn"] * _clip((0.8 - ratio) / 0.45)
        if have("hrv_rmssd"):
            ratio = reading["hrv_rmssd"] / max(b["hrv_rmssd"], 1e-6)
            c["hrv_rmssd"] = _W["hrv_rmssd"] * _clip((0.8 - ratio) / 0.45)
        if have("sample_entropy"):
            c["sample_entropy"] = _W["sample_entropy"] * _clip((b["sample_entropy"] - reading["sample_entropy"]) / 0.6)
        if reading.get("sample_asymmetry") is not None:
            c["sample_asymmetry"] = _W["sample_asymmetry"] * _clip((reading["sample_asymmetry"] - 0.55) / 0.25)

        # --- Tachycardia (relative to baseline) --- #
        if have("heart_rate"):
            dev = (reading["heart_rate"] - b["heart_rate"] * 1.12) / (b["heart_rate"] * 0.4)
            c["heart_rate"] = _W["heart_rate"] * _clip(dev)
        # --- Tachypnea --- #
        if have("respiratory_rate"):
            c["respiratory_rate"] = _W["respiratory_rate"] * _clip((reading["respiratory_rate"] - 20.0) / 8.0)
        # --- Perfusion drop (proxy) --- #
        if reading.get("perfusion_index") is not None:
            c["perfusion"] = _W["perfusion"] * _clip((0.6 - reading["perfusion_index"]) / 0.4)
        # --- Sympathetic stress (proxy) --- #
        if reading.get("stress_vector") is not None:
            c["stress"] = _W["stress"] * _clip((reading["stress_vector"] - 0.5) / 0.3)

        log_odds = q * float(sum(c.values()))
        instant = 1.0 / (1.0 + np.exp(-(self.prior_log_odds + log_odds)))

        self._adapt_baseline(reading, log_odds, q)
        return CameraEvidence(log_odds=log_odds, instant_prob=float(instant), quality=q,
                              contributions=c, baseline=dict(b))

    # ------------------------------------------------------------------ #
    def _adapt_baseline(self, reading: Dict[str, float], log_odds: float, q: float) -> None:
        """Learn the subject's resting baseline — but only from good-quality,
        low-evidence (apparently stable) readings, so we never adapt toward the
        septic state itself."""
        if q < 0.5 or log_odds > 0.25:
            return
        lr = self.baseline_lr * q
        for k in _BASE0:
            v = reading.get(k if k != "respiratory_rate" else "respiratory_rate")
            if v is not None and v > 0:
                self.baseline[k] = (1 - lr) * self.baseline[k] + lr * float(v)
        self._warm += 1

    @staticmethod
    def from_camera_vitals(cam: Any) -> Dict[str, float]:
        """Flatten a CameraVitals (+ its rich HRV features) into a reading dict."""
        raw = getattr(cam, "raw", {}) or {}
        hrv = raw.get("hrv_features", {}) or {}
        return {
            "heart_rate": getattr(cam, "heart_rate", None),
            "hrv_sdnn": hrv.get("sdnn") or getattr(cam, "hrv", None),
            "hrv_rmssd": hrv.get("rmssd") or getattr(cam, "hrv_rmssd", None),
            "sample_entropy": hrv.get("sample_entropy"),
            "sample_asymmetry": hrv.get("sample_asymmetry"),
            "respiratory_rate": getattr(cam, "respiratory_rate", None),
            "perfusion_index": getattr(cam, "perfusion_index", None),
            "stress_vector": getattr(cam, "stress_vector", None),
        }
