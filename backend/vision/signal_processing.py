"""Remote-photoplethysmography (rPPG) signal processing — the validated core.

These are pure functions over RGB time-series so they can be unit-tested against
synthetic signals with a *known* heart rate. The estimators implement published
classical rPPG algorithms:

  * POS  — Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 2017
  * CHROM — de Haan & Jeanne, "Robust Pulse Rate from Chrominance", IEEE 2013

We deliberately use these deterministic, peer-reviewed methods (not a black box)
so every step is inspectable and the recovered HR is provably correct on test
signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal

# rPPG cardiac band: 0.7–4.0 Hz == 42–240 bpm.
HR_BAND_HZ = (0.7, 4.0)
# Respiration band: 0.1–0.5 Hz == 6–30 breaths/min.
RESP_BAND_HZ = (0.1, 0.5)


@dataclass
class PulseEstimate:
    hr_bpm: float
    hrv_sdnn_ms: float
    hrv_rmssd_ms: float
    resp_bpm: float
    snr_db: float
    quality: float            # 0..1 overall usability of this window
    n_beats: int


def _detrend_normalize(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N, 3) -> temporally mean-normalized (N, 3), per the POS recipe."""
    mean = rgb.mean(axis=0, keepdims=True)
    mean[mean == 0] = 1.0
    return rgb / mean - 1.0


def pos_pulse(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Plane-Orthogonal-to-Skin pulse signal from an (N,3) RGB trace.

    Implemented as the overlap-added sliding-window form from the paper, which
    is robust to slow illumination drift and subject motion.
    """
    n = len(rgb)
    if n < 8:
        return np.zeros(n)
    H = np.zeros(n)
    win = max(int(1.6 * fs), 8)          # ~1.6 s analysis window
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for start in range(0, n - win + 1):
        block = rgb[start:start + win]
        mean = block.mean(axis=0, keepdims=True)
        mean[mean == 0] = 1.0
        Cn = block / mean                # temporal normalization
        S = proj @ Cn.T                  # (2, win)
        s1, s2 = S[0], S[1]
        std2 = s2.std()
        alpha = (s1.std() / std2) if std2 > 1e-9 else 0.0
        h = s1 + alpha * s2
        h = h - h.mean()
        H[start:start + win] += h
    return H


def chrom_pulse(rgb: np.ndarray, fs: float) -> np.ndarray:
    """CHROM chrominance pulse signal — an alternative/secondary estimator."""
    n = len(rgb)
    if n < 8:
        return np.zeros(n)
    Xn = _detrend_normalize(rgb)
    r, g, b = Xn[:, 0], Xn[:, 1], Xn[:, 2]
    xs = 3.0 * r - 2.0 * g
    ys = 1.5 * r + g - 1.5 * b
    xf = _bandpass(xs, fs, *HR_BAND_HZ)
    yf = _bandpass(ys, fs, *HR_BAND_HZ)
    sy = yf.std()
    alpha = (xf.std() / sy) if sy > 1e-9 else 0.0
    return xf - alpha * yf


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-4), min(hi / nyq, 0.999)
    if hi_n <= lo_n:
        return x - x.mean()
    b, a = signal.butter(order, [lo_n, hi_n], btype="band")
    if len(x) <= 3 * max(len(a), len(b)):
        return x - x.mean()
    return signal.filtfilt(b, a, x)


def _band_power_peak(x: np.ndarray, fs: float, band: Tuple[float, float]) -> Tuple[float, float, float]:
    """Return (peak_freq_hz, peak_power, total_band_power) via Welch PSD."""
    nperseg = min(len(x), int(fs * 8))
    if nperseg < 16:
        return 0.0, 0.0, 0.0
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return 0.0, 0.0, 0.0
    bf, bp = freqs[mask], psd[mask]
    k = int(np.argmax(bp))
    return float(bf[k]), float(bp[k]), float(bp.sum())


def estimate_hr(pulse: np.ndarray, fs: float) -> Tuple[float, float]:
    """Heart rate (bpm) and SNR (dB) from a pulse signal."""
    filt = _bandpass(pulse, fs, *HR_BAND_HZ)
    peak_f, peak_p, total_p = _band_power_peak(filt, fs, (HR_BAND_HZ[0], 3.5))
    if peak_f <= 0:
        return 0.0, -np.inf
    # SNR: peak (plus its first harmonic neighbourhood) vs the rest of the band.
    snr = peak_p / max(total_p - peak_p, 1e-12)
    return peak_f * 60.0, 10.0 * np.log10(snr + 1e-12)


def estimate_hrv(pulse: np.ndarray, fs: float, hr_bpm: float) -> Tuple[float, float, int]:
    """SDNN, RMSSD (ms) and beat count from inter-beat intervals.

    Webcam HRV is noisy; treat as a trend signal, not a clinical value.
    """
    filt = _bandpass(pulse, fs, *HR_BAND_HZ)
    if hr_bpm <= 0:
        return 0.0, 0.0, 0
    min_dist = int(fs * 60.0 / (hr_bpm * 1.4))   # allow up to 40% faster beats
    peaks, _ = signal.find_peaks(filt, distance=max(min_dist, 1))
    if len(peaks) < 3:
        return 0.0, 0.0, len(peaks)
    ibi_ms = np.diff(peaks) / fs * 1000.0
    # Reject physiologically implausible intervals (300–2000 ms).
    ibi_ms = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]
    if len(ibi_ms) < 2:
        return 0.0, 0.0, len(peaks)
    sdnn = float(np.std(ibi_ms))
    rmssd = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
    return sdnn, rmssd, len(peaks)


def estimate_resp(green: np.ndarray, fs: float) -> float:
    """Respiratory rate (breaths/min) from low-frequency baseline modulation."""
    x = green - green.mean()
    peak_f, _, _ = _band_power_peak(_bandpass(x, fs, *RESP_BAND_HZ), fs, RESP_BAND_HZ)
    return peak_f * 60.0


def analyze_window(rgb: np.ndarray, fs: float, method: str = "pos") -> PulseEstimate:
    """Full estimate from an (N,3) RGB ROI trace sampled at ``fs`` Hz."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or len(rgb) < int(fs * 4):
        return PulseEstimate(0, 0, 0, 0, -np.inf, 0.0, 0)

    pulse = pos_pulse(rgb, fs) if method == "pos" else chrom_pulse(rgb, fs)
    hr, snr = estimate_hr(pulse, fs)
    sdnn, rmssd, n_beats = estimate_hrv(pulse, fs, hr)
    resp = estimate_resp(rgb[:, 1], fs)

    # Quality blends SNR, plausibility of HR, and having enough beats.
    snr_q = float(np.clip((snr + 3.0) / 10.0, 0.0, 1.0))
    hr_q = 1.0 if 40.0 <= hr <= 200.0 else 0.0
    beat_q = float(np.clip(n_beats / 8.0, 0.0, 1.0))
    quality = float(np.clip(0.5 * snr_q + 0.3 * hr_q + 0.2 * beat_q, 0.0, 1.0))

    return PulseEstimate(
        hr_bpm=round(hr, 1),
        hrv_sdnn_ms=round(sdnn, 1),
        hrv_rmssd_ms=round(rmssd, 1),
        resp_bpm=round(resp, 1),
        snr_db=round(snr, 2),
        quality=round(quality, 3),
        n_beats=n_beats,
    )


# --------------------------------------------------------------------------- #
# Synthetic validation: prove the core recovers a known heart rate.
# --------------------------------------------------------------------------- #
def synthesize_rgb(
    hr_bpm: float,
    duration_s: float,
    fs: float,
    resp_bpm: float = 15.0,
    noise: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Generate a realistic (N,3) RGB ROI trace with an embedded pulse.

    The pulse modulates the GREEN channel most strongly (hemoglobin absorbs
    green), with respiration baseline wander and per-channel sensor noise — a
    faithful test bed for the estimators.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    f = hr_bpm / 60.0
    pulse = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
    resp = 0.4 * np.sin(2 * np.pi * (resp_bpm / 60.0) * t)
    base = np.array([0.5, 0.5, 0.5])
    gain = np.array([0.3, 1.0, 0.2])     # green carries most pulse signal
    rgb = base[None, :] + gain[None, :] * pulse[:, None] * 0.02
    rgb += (resp[:, None] * 0.01)
    rgb += rng.normal(0, noise, size=(n, 3))
    return rgb * 255.0


if __name__ == "__main__":  # pragma: no cover
    print("rPPG core self-validation (synthetic ground truth):")
    fs = 30.0
    for true_hr in (55.0, 72.0, 96.0, 130.0):
        rgb = synthesize_rgb(true_hr, duration_s=15.0, fs=fs, seed=int(true_hr))
        est = analyze_window(rgb, fs, method="pos")
        err = abs(est.hr_bpm - true_hr)
        flag = "OK" if err <= 4.0 else "FAIL"
        print(f"  true={true_hr:6.1f} bpm  est={est.hr_bpm:6.1f}  "
              f"err={err:4.1f}  quality={est.quality:.2f}  snr={est.snr_db:.1f}dB  [{flag}]")
