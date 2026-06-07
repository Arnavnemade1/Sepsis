"""Sympathetic-distress proxies (micro-tremor + pupil instability).

Clinical hypothesis (from the project brief): systemic inflammation and
metabolic stress perturb autonomic/neurological state, producing involuntary
muscle micro-tremor and pupillary instability ("hippus"). From video we estimate:

  * tremor_power : energy of facial-landmark motion in a tremor band (~4–12 Hz),
                   after removing voluntary low-frequency head motion.
  * pupil_instability : variability of the iris/pupil-diameter proxy over time.

These combine into a single normalized "stress_vector" in 0..1.

⚠️ EXPERIMENTAL PROXY. Webcam pupillometry is coarse (no IR, low resolution) and
the tremor↔sepsis link is unvalidated. Trend-only, confidence-weighted signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

TREMOR_BAND_HZ = (4.0, 12.0)


@dataclass
class DistressEstimate:
    tremor_power: float          # normalized motion energy in the tremor band
    tremor_peak_hz: float        # dominant tremor frequency
    pupil_instability: float     # coefficient of variation of pupil proxy
    stress_vector: float         # normalized 0..1 composite
    quality: float


def analyze_distress(
    landmark_series: np.ndarray,   # (N, K, 2) stable landmark pixel coords over time
    pupil_series: np.ndarray,      # (N,) pupil/iris ratio over time
    fs: float,
) -> DistressEstimate:
    lm = np.asarray(landmark_series, dtype=np.float64)
    pup = np.asarray(pupil_series, dtype=np.float64)
    if lm.ndim != 3 or len(lm) < int(fs * 3):
        return DistressEstimate(0.0, 0.0, 0.0, 0.0, 0.0)

    # Signed per-coordinate displacement signals (N, K*2). Using SIGNED motion
    # (not the norm) avoids the rectification artifact that would double an
    # oscillation's frequency and push real tremor out of the analysis band.
    centered = lm - lm.mean(axis=0, keepdims=True)
    coords = centered.reshape(len(centered), -1)           # (N, K*2)

    band_total = 0.0
    var_total = 0.0
    peak_hz = 0.0
    peak_pow = 0.0
    for ch in range(coords.shape[1]):
        sig = coords[:, ch]
        e, pk = _band_energy(sig, fs, TREMOR_BAND_HZ)
        band_total += e
        var_total += float(np.var(sig))
        if e > peak_pow:
            peak_pow, peak_hz = e, pk
    tremor_peak = peak_hz
    tremor_norm = float(np.clip(band_total / (var_total + 1e-9), 0.0, 1.0))

    if pup.size >= int(fs * 3) and np.mean(pup) > 1e-6:
        pupil_cv = float(np.std(pup) / np.mean(pup))
    else:
        pupil_cv = 0.0
    pupil_norm = float(np.clip(pupil_cv / 0.15, 0.0, 1.0))

    stress_vector = float(np.clip(0.6 * tremor_norm + 0.4 * pupil_norm, 0.0, 1.0))
    quality = float(np.clip(len(lm) / (fs * 8), 0.0, 1.0))

    return DistressEstimate(
        tremor_power=round(tremor_norm, 4),
        tremor_peak_hz=round(tremor_peak, 2),
        pupil_instability=round(pupil_cv, 4),
        stress_vector=round(stress_vector, 3),
        quality=round(quality, 3),
    )


def _band_energy(x: np.ndarray, fs: float, band) -> tuple:
    nperseg = min(len(x), int(fs * 4))
    if nperseg < 16 or fs / 2 <= band[0]:
        return 0.0, 0.0
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= min(band[1], fs / 2 - 1e-3))
    if not mask.any():
        return 0.0, 0.0
    energy = float(np.trapz(psd[mask], freqs[mask]))
    peak = float(freqs[mask][int(np.argmax(psd[mask]))])
    return energy, peak
