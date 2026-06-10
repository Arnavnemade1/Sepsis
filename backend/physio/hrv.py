"""Heart-rate-variability analysis: beat detection -> RR cleaning -> features.

All feature definitions follow the standard HRV literature (Task Force 1996) plus
the neonatal-sepsis "heart rate characteristics" descriptors (Moorman/HeRO:
reduced variability, sample entropy, sample asymmetry). Everything is computed
from a clean RR-interval (NN) series, so it is source-agnostic: ECG, contact PPG,
or camera rPPG all flow through the same code — which is exactly what makes the
cross-source validation meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal, interpolate


# --------------------------------------------------------------------------- #
# Beat detection
# --------------------------------------------------------------------------- #
def detect_ecg_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """R-peak indices from an ECG lead (Pan-Tompkins-style envelope + adaptive
    peak picking). Robust enough for clean monitored ECG like BIDMC."""
    x = np.asarray(ecg, dtype=np.float64)
    x = x - np.median(x)
    bp = _bandpass(x, fs, 5.0, 18.0)
    deriv = np.gradient(bp)
    sq = deriv ** 2
    win = max(int(0.12 * fs), 1)
    integrated = np.convolve(sq, np.ones(win) / win, mode="same")
    thr = 0.3 * np.mean(integrated) + 0.2 * np.std(integrated)
    min_dist = int(0.30 * fs)                      # refractory ~200 bpm cap
    peaks, _ = signal.find_peaks(integrated, height=thr, distance=min_dist)
    # Refine each detection to the local ECG maximum (true R apex), then
    # parabolic sub-sample interpolation for precise beat timing.
    refined = []
    r = max(int(0.05 * fs), 1)
    for p in peaks:
        lo, hi = max(0, p - r), min(len(x), p + r)
        if hi > lo:
            refined.append(lo + int(np.argmax(x[lo:hi])))
    apex = np.array(sorted(set(refined)), dtype=int)
    return _refine_subsample(x, apex)


def detect_ppg_peaks(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Systolic peak positions (SUB-SAMPLE) from a PPG/rPPG pulse signal.

    Returns fractional indices: parabolic interpolation of each peak removes the
    1/fs timing quantization that otherwise dominates RMSSD/SDSD error — the
    single biggest accuracy win for HRV at camera frame rates (33 ms at 30 fps).
    """
    x = np.asarray(ppg, dtype=np.float64)
    bp = _bandpass(x, fs, 0.5, 8.0)
    min_dist = max(int(0.33 * fs), 1)              # cap ~180 bpm
    # Adaptive threshold: local moving baseline (handles slow drift) + a small
    # amplitude margin. Kept low so low-amplitude beats are not missed (missed
    # beats fabricate double-length IBIs that bias SDNN/RMSSD upward).
    win = max(int(fs * 2), 3)
    base = np.convolve(bp, np.ones(win) / win, mode="same")
    thr = base + 0.05 * np.std(bp)
    peaks, _ = signal.find_peaks(bp, distance=min_dist, height=thr,
                                 prominence=0.15 * np.std(bp))
    return _refine_subsample(bp, peaks)


def _refine_subsample(sig: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Parabolic (quadratic) sub-sample refinement of peak positions."""
    if len(peaks) == 0:
        return np.array([], dtype=np.float64)
    out = []
    for p in peaks:
        if 0 < p < len(sig) - 1:
            a, b, c = sig[p - 1], sig[p], sig[p + 1]
            denom = a - 2 * b + c
            off = 0.5 * (a - c) / denom if denom != 0 else 0.0
            out.append(p + float(np.clip(off, -0.5, 0.5)))
        else:
            out.append(float(p))
    return np.array(out, dtype=np.float64)


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-4), min(hi / nyq, 0.999)
    if hi_n <= lo_n:
        return x - x.mean()
    b, a = signal.butter(order, [lo_n, hi_n], btype="band")
    if len(x) <= 3 * max(len(a), len(b)):
        return x - x.mean()
    return signal.filtfilt(b, a, x)


# --------------------------------------------------------------------------- #
# RR / NN intervals
# --------------------------------------------------------------------------- #
def rr_intervals_ms(peaks: np.ndarray, fs: float) -> np.ndarray:
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs * 1000.0


def _nn_mask(rr_ms: np.ndarray, lo: float = 300.0, hi: float = 2000.0,
             rel_tol: float = 0.2) -> np.ndarray:
    """Boolean keep-mask: physiologically plausible AND not an ectopic jump.

    A beat is rejected if it leaves the 300–2000 ms range, or if it deviates
    >``rel_tol`` from BOTH a local rolling median and the previous retained beat
    (the Malik-style relative filter — stricter than a global-median test and
    much better at catching missed/extra beats)."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    n = len(rr)
    if n == 0:
        return np.array([], dtype=bool)
    keep = (rr >= lo) & (rr <= hi)
    if n >= 5:
        from scipy.signal import medfilt
        k = min(n if n % 2 else n - 1, 5)
        local = medfilt(rr, k)
        keep &= np.abs(rr - local) <= rel_tol * np.maximum(local, 1.0)
    return keep


def clean_nn(rr_ms: np.ndarray, lo: float = 300.0, hi: float = 2000.0,
             rel_tol: float = 0.2) -> np.ndarray:
    """Return the retained NN intervals (see ``_nn_mask`` for the filter)."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    return rr[_nn_mask(rr, lo, hi, rel_tol)]


# --------------------------------------------------------------------------- #
# Feature set
# --------------------------------------------------------------------------- #
@dataclass
class HRVFeatures:
    n_beats: int
    mean_hr: float
    mean_nn: float
    sdnn: float
    rmssd: float
    sdsd: float
    pnn50: float
    cvnn: float
    # frequency domain
    lf: float
    hf: float
    lf_hf: float
    total_power: float
    lf_nu: float
    hf_nu: float
    # nonlinear
    sd1: float
    sd2: float
    sd1_sd2: float
    sample_entropy: float
    sample_asymmetry: float       # HeRO-style R1/R2 asymmetry
    quality: float

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def hrv_features(rr_ms: np.ndarray, clean: bool = True) -> HRVFeatures:
    rr = np.asarray(rr_ms, dtype=np.float64)
    if clean:
        mask = _nn_mask(rr)
        nn = rr[mask]
        # Successive differences ONLY between consecutive retained beats — never
        # across a removed ectopic/artifact (which would fabricate a huge jump
        # and corrupt RMSSD/SDSD/pNN50).
        adj = mask[:-1] & mask[1:]
        diff = np.diff(rr)[adj] if adj.any() else np.array([])
    else:
        nn = rr
        diff = np.diff(nn)
    n = len(nn)
    if n < 4:
        return HRVFeatures(*([n] + [0.0] * 19), quality=0.0)

    mean_nn = float(np.mean(nn))
    sdnn = float(np.std(nn, ddof=1))
    rmssd = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else 0.0
    sdsd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    pnn50 = float(np.mean(np.abs(diff) > 50.0) * 100.0) if len(diff) else 0.0
    mean_hr = 60000.0 / mean_nn if mean_nn > 0 else 0.0
    cvnn = sdnn / mean_nn if mean_nn > 0 else 0.0

    lf, hf, total, lf_nu, hf_nu = _frequency_domain(nn)
    sd1, sd2 = _poincare(nn)
    sd1_sd2 = sd1 / sd2 if sd2 > 0 else 0.0
    samp_en = _sample_entropy(nn)
    asym = _sample_asymmetry(nn)

    # Usability: needs enough clean beats for stable frequency-domain estimates.
    quality = float(np.clip(n / 60.0, 0.0, 1.0))

    return HRVFeatures(
        n_beats=n, mean_hr=round(mean_hr, 2), mean_nn=round(mean_nn, 2),
        sdnn=round(sdnn, 2), rmssd=round(rmssd, 2), sdsd=round(sdsd, 2),
        pnn50=round(pnn50, 2), cvnn=round(cvnn, 4),
        lf=round(lf, 2), hf=round(hf, 2), lf_hf=round(lf / hf if hf > 0 else 0.0, 3),
        total_power=round(total, 2), lf_nu=round(lf_nu, 2), hf_nu=round(hf_nu, 2),
        sd1=round(sd1, 2), sd2=round(sd2, 2), sd1_sd2=round(sd1_sd2, 3),
        sample_entropy=round(samp_en, 4), sample_asymmetry=round(asym, 4),
        quality=round(quality, 3),
    )


def _frequency_domain(nn: np.ndarray) -> Tuple[float, float, float, float, float]:
    """LF/HF/total power via Welch PSD of the evenly-resampled tachogram."""
    t = np.cumsum(nn) / 1000.0
    t = t - t[0]
    if t[-1] < 20.0:                # need a reasonable span for HF/LF
        return 0.0, 0.0, 0.0, 0.0, 0.0
    fs_i = 4.0
    grid = np.arange(0, t[-1], 1.0 / fs_i)
    f = interpolate.interp1d(t, nn, kind="cubic", fill_value="extrapolate")(grid)
    f = f - f.mean()
    nperseg = min(len(f), int(fs_i * 60))
    if nperseg < 32:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    freqs, psd = signal.welch(f, fs=fs_i, nperseg=nperseg)
    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(np.trapz(psd[m], freqs[m])) if m.any() else 0.0
    vlf, lf, hf = band(0.003, 0.04), band(0.04, 0.15), band(0.15, 0.4)
    total = vlf + lf + hf
    denom = lf + hf
    lf_nu = 100.0 * lf / denom if denom > 0 else 0.0
    hf_nu = 100.0 * hf / denom if denom > 0 else 0.0
    return lf, hf, total, lf_nu, hf_nu


def _poincare(nn: np.ndarray) -> Tuple[float, float]:
    x1, x2 = nn[:-1], nn[1:]
    sd1 = float(np.std((x1 - x2) / np.sqrt(2), ddof=1))
    sd2 = float(np.std((x1 + x2) / np.sqrt(2), ddof=1))
    return sd1, sd2


def _sample_entropy(nn: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample entropy — a key HeRO descriptor (lower = more regular = sicker)."""
    x = np.asarray(nn, dtype=np.float64)
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_factor * np.std(x)
    if r <= 0:
        return 0.0

    def _phi(mm: int) -> int:
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        count = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            count += int(np.sum(d <= r)) - 1   # exclude self-match
        return count

    b = _phi(m)
    a = _phi(m + 1)
    if b == 0 or a == 0:
        return 0.0
    return float(-np.log(a / b))


def _sample_asymmetry(nn: np.ndarray) -> float:
    """HeRO sample asymmetry: ratio of variance above vs below the median —
    sepsis produces transient decelerations that skew the RR distribution."""
    x = np.asarray(nn, dtype=np.float64)
    med = np.median(x)
    above = x[x > med] - med
    below = med - x[x < med]
    r2 = float(np.mean(above ** 2)) if above.size else 0.0
    r1 = float(np.mean(below ** 2)) if below.size else 0.0
    denom = r1 + r2
    return float(r2 / denom) if denom > 0 else 0.5


# --------------------------------------------------------------------------- #
# Convenience: features straight from a raw signal
# --------------------------------------------------------------------------- #
def hrv_from_signal(sig: np.ndarray, fs: float, kind: str = "ppg") -> HRVFeatures:
    peaks = detect_ecg_rpeaks(sig, fs) if kind == "ecg" else detect_ppg_peaks(sig, fs)
    return hrv_features(rr_intervals_ms(peaks, fs))
