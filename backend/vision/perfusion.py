"""Micro-perfusion mapping proxy (vascular-shunting signal).

Clinical hypothesis (from the project brief): in sepsis, peripheral
vasoconstriction ("shunting") reduces cutaneous blood volume, so facial skin
loses pulsatile amplitude and reddish chrominance. We approximate this with two
camera-observable quantities:

  * perfusion_index : pulsatile (AC) amplitude / steady (DC) level of the green
                      ROI trace — the optical analogue of a pulse-oximeter's
                      perfusion index. Falls when peripheral perfusion drops.
  * redness_ratio   : R / (R+G+B) over skin — pallor lowers it.
  * spatial_dispersion : variation of redness across forehead vs cheeks —
                      regional shunting raises it.

⚠️ EXPERIMENTAL PROXY. This is NOT a validated surrogate for blood pressure or a
diagnosis. It is a relative, trend-only signal with an attached confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PerfusionEstimate:
    perfusion_index: float       # ~0 (poor) .. higher (good); optical PI analogue
    redness_ratio: float         # fraction of reflectance in the red channel
    spatial_dispersion: float    # cross-region redness variability
    perfusion_proxy: float       # normalized 0..1, LOW => possible shunting
    quality: float


def analyze_perfusion(
    roi_rgb_series: np.ndarray,          # (N, 3) mean RGB over skin ROI
    region_redness: Dict[str, List[float]],   # region -> list of redness over time
    fs: float,
) -> PerfusionEstimate:
    roi = np.asarray(roi_rgb_series, dtype=np.float64)
    if roi.ndim != 2 or roi.shape[1] != 3 or len(roi) < int(fs * 3):
        return PerfusionEstimate(0.0, 0.0, 0.0, 0.0, 0.0)

    green = roi[:, 1]
    dc = float(np.mean(green))
    # AC = pulsatile std after removing slow drift (linear detrend).
    ac = float(np.std(_detrend(green)))
    perfusion_index = (ac / dc) if dc > 1e-6 else 0.0

    rgb_sum = roi.sum(axis=1)
    rgb_sum[rgb_sum == 0] = 1.0
    redness = float(np.mean(roi[:, 0] / rgb_sum))

    region_means = [float(np.mean(v)) for v in region_redness.values() if len(v) > 0]
    spatial_dispersion = float(np.std(region_means)) if len(region_means) >= 2 else 0.0

    # Normalize the optical PI to 0..1 with a soft scale. Healthy facial PI is
    # typically a few parts per thousand to ~1%; map [0, 0.01] -> [0, 1].
    pi_norm = float(np.clip(perfusion_index / 0.01, 0.0, 1.0))
    # Combine: good perfusion = strong pulsatility + adequate redness.
    redness_term = float(np.clip((redness - 0.30) / 0.12, 0.0, 1.0))
    perfusion_proxy = float(np.clip(0.7 * pi_norm + 0.3 * redness_term, 0.0, 1.0))

    quality = float(np.clip(len(roi) / (fs * 8), 0.0, 1.0))
    return PerfusionEstimate(
        perfusion_index=round(perfusion_index, 5),
        redness_ratio=round(redness, 4),
        spatial_dispersion=round(spatial_dispersion, 4),
        perfusion_proxy=round(perfusion_proxy, 3),
        quality=round(quality, 3),
    )


def _detrend(x: np.ndarray) -> np.ndarray:
    n = len(x)
    t = np.arange(n)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)
