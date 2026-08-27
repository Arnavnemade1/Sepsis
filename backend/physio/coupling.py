"""Cardio-respiratory coupling — the "decomplexification" pathway.

WHY THIS EXISTS
---------------
Every other detection lever in this project has been measured and found to be at its
ceiling: richer temporal features (Phase 7), a random convolutional basis
(`ml/models/minirocket.py`), tree-leaf topology (`ml/models/forest_vsa.py`), 16
self-improvement mutations, and — most recently — label-free deployment transfer
(`docs/TRANSFER.md`). All of them re-read the same tabular signal. The only honest
routes left are new *signal* and better *decisions*, and this module is the first.

In healthy physiology, heart rate and respiration are coupled: the heart accelerates on
inspiration and decelerates on expiration (respiratory sinus arrhythmia), and the two
rhythms hold a stable phase relationship. Early sepsis degrades that coupling before it
moves any single vital out of its population-normal band — the phenomenon usually
called **decomplexification**. Reduced cardio-respiratory information transfer is
associated with deterioration and poor prognosis in septic patients (Journal of Applied
Physiology, 2024), which makes the coupling itself, not either signal alone, the
quantity worth measuring.

WHAT THIS MODULE MEASURES
-------------------------
  * **Transfer entropy** TE(RESP -> HR) and TE(HR -> RESP): directed information flow.
    Reported **surrogate-corrected**, because a binned TE estimator on finite data is
    positively biased and a raw TE is essentially never zero even for independent
    signals. The corrected value is TE minus the mean TE of phase-randomised
    surrogates, so "no coupling" reads as approximately 0 rather than as a confident
    small positive number.
  * **Phase-synchronisation index** over the Hilbert phases: how tightly the two
    rhythms hold a fixed phase offset, independent of amplitude.
  * **RSA band coherence**: the classical measure — how much HR variance sits in the
    respiratory frequency band.

HONEST SCOPE (the integrity rule this repo runs on)
---------------------------------------------------
This is a **signal extractor with a mechanistic rationale, not a sepsis classifier.**
BIDMC — the only waveform data present here — has **no sepsis labels**, and CinC 2019 is
hourly, which is far too coarse for any of these estimators. **No AUROC, lead-time or
detection claim may be made from this module** on the data currently in the repo.

What CAN be validated here, and is, in ``verify_coupling.py``: that the estimators are
stable within a real record, and that they fall **monotonically** under controlled
synthetic decoupling — the exact change the pathophysiology predicts. That is a
mechanism validation, the same standard ``ppg_morphology.py`` was held to.

The label-bearing path is already written: ``backend/physio/data/mimic_waveform.py``
exposes ``RRLabeledSource`` over MIMIC-IV Waveform. The blocker is CITI training plus a
data-use agreement — a user action, not an engineering one.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import hilbert, welch

from backend.physio.hrv import _bandpass, detect_ecg_rpeaks, detect_ppg_peaks

#: Respiration lives here. Below this is baseline wander, above it is cardiac leakage.
RESP_BAND = (0.1, 0.5)          # Hz — 6 to 30 breaths/min
HR_GRID_HZ = 4.0                 # instantaneous-HR resample rate (standard for HRV)


# --------------------------------------------------------------------------- #
def instantaneous_hr(sig: np.ndarray, fs: float, kind: str = "ecg",
                     grid_hz: float = HR_GRID_HZ) -> Tuple[np.ndarray, float]:
    """Beat-to-beat HR resampled onto a uniform grid.

    Coupling estimators need two series on a **common, evenly spaced** time base; beat
    intervals are neither. Interpolating the tachogram onto a 4 Hz grid is the standard
    HRV treatment and keeps the respiratory band (0.1-0.5 Hz) far below Nyquist.
    """
    peaks = (detect_ecg_rpeaks(sig, fs) if kind == "ecg" else detect_ppg_peaks(sig, fs))
    peaks = np.asarray(peaks, dtype=float)
    if peaks.size < 8:
        return np.zeros(0), grid_hz
    t_beat = peaks / fs
    rr = np.diff(t_beat)
    hr = 60.0 / np.clip(rr, 0.25, 3.0)
    t_mid = t_beat[:-1] + rr / 2.0
    grid = np.arange(t_mid[0], t_mid[-1], 1.0 / grid_hz)
    if grid.size < 16:
        return np.zeros(0), grid_hz
    return np.interp(grid, t_mid, hr), grid_hz


def resp_envelope(resp: np.ndarray, fs: float, grid_hz: float = HR_GRID_HZ,
                  n: Optional[int] = None) -> np.ndarray:
    """Band-limited respiration on the same grid as :func:`instantaneous_hr`."""
    x = _bandpass(np.asarray(resp, dtype=float), fs, RESP_BAND[0], RESP_BAND[1])
    src_t = np.arange(len(x)) / fs
    m = int(len(x) / fs * grid_hz) if n is None else n
    if m < 16:
        return np.zeros(0)
    return np.interp(np.linspace(src_t[0], src_t[-1], m), src_t, x)


# --------------------------------------------------------------------------- #
def _symbolise(x: np.ndarray, bins: int) -> np.ndarray:
    """Equiprobable (quantile) binning.

    Equal-width bins put almost every sample of a peaked distribution into one bin and
    the entropy estimate collapses. Quantile bins keep the marginal uniform by
    construction, which is what makes the conditional entropies comparable across
    records with different amplitudes.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.astype(int)
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(x, edges)


def _te_binned(src: np.ndarray, dst: np.ndarray, bins: int, lag: int) -> float:
    """Raw binned transfer entropy TE(src -> dst), in bits.

    ``TE = sum p(d1, d0, s0) * log2[ p(d1|d0,s0) / p(d1|d0) ]`` with one step of
    history. Positive means knowing the source's past reduces uncertainty about the
    destination's next value *beyond* what the destination's own past already explains
    — which is the property that separates coupling from shared trend.
    """
    s = _symbolise(src, bins)
    d = _symbolise(dst, bins)
    n = min(len(s), len(d)) - lag
    if n < 32:
        return 0.0
    d1, d0, s0 = d[lag:lag + n], d[:n], s[:n]

    joint = np.zeros((bins, bins, bins), dtype=float)
    np.add.at(joint, (d1, d0, s0), 1.0)
    joint /= joint.sum()
    p_d0s0 = joint.sum(axis=0)                     # (d0, s0)
    p_d1d0 = joint.sum(axis=2)                     # (d1, d0)
    p_d0 = joint.sum(axis=(0, 2))                  # (d0,)

    te = 0.0
    idx = np.nonzero(joint)
    for a, b, c in zip(*idx):
        num = joint[a, b, c] / p_d0s0[b, c]        # p(d1 | d0, s0)
        den = p_d1d0[a, b] / p_d0[b]               # p(d1 | d0)
        if num > 0 and den > 0:
            te += joint[a, b, c] * np.log2(num / den)
    return float(max(te, 0.0))


def phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate with the SAME power spectrum and destroyed phase relationships.

    The right null for a coupling statistic. A plain shuffle also destroys coupling but
    destroys the autocorrelation with it, so it under-states the bias of the estimator
    on a smooth signal and the correction comes out too small.
    """
    X = np.fft.rfft(np.asarray(x, dtype=float))
    ph = rng.uniform(0, 2 * np.pi, size=X.shape)
    ph[0] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


def transfer_entropy(src: np.ndarray, dst: np.ndarray, bins: int = 4, lag: int = 1,
                     n_surrogate: int = 12, seed: int = 0) -> Dict[str, float]:
    """Surrogate-corrected TE(src -> dst) plus the raw value and the null it beat."""
    raw = _te_binned(src, dst, bins, lag)
    rng = np.random.default_rng(seed)
    null = np.array([_te_binned(phase_randomise(src, rng), dst, bins, lag)
                     for _ in range(n_surrogate)])
    mu = float(null.mean()) if null.size else 0.0
    sd = float(null.std()) if null.size else 0.0
    return {"te": round(float(max(raw - mu, 0.0)), 5), "te_raw": round(raw, 5),
            "null_mean": round(mu, 5),
            "z": round((raw - mu) / sd, 3) if sd > 1e-12 else 0.0}


def phase_sync(a: np.ndarray, b: np.ndarray, n: int = 1, m: int = 1,
               fs: float = HR_GRID_HZ, band: Tuple[float, float] = RESP_BAND) -> float:
    """n:m phase-synchronisation index in [0, 1] over the analytic-signal phases.

    1.0 is a rigidly held phase offset, 0.0 is a uniformly distributed one. Amplitude
    plays no part, which is the point: coupling can persist while both signals shrink,
    and it can break while both look perfectly normal in size.

    **Both inputs are band-passed to the respiratory band first, and that is not
    optional.** A Hilbert phase is only meaningful for a narrowband signal. Handed
    broadband input the analytic phase is dominated by whatever low-frequency drift
    each series happens to carry, and two *independent* random walks come out at 0.72
    — higher than real coupled physiology scored at the time. That is exactly what
    this estimator measured before the filter was added, and it would have turned
    ordinary baseline wander into a confident coupling finding.
    """
    if len(a) < 16 or len(b) < 16:
        return 0.0
    k = min(len(a), len(b))
    fa = _bandpass(np.asarray(a[:k], dtype=float) - float(np.mean(a[:k])), fs, *band)
    fb = _bandpass(np.asarray(b[:k], dtype=float) - float(np.mean(b[:k])), fs, *band)
    pa, pb = np.angle(hilbert(fa)), np.angle(hilbert(fb))
    return float(np.abs(np.mean(np.exp(1j * (n * pa - m * pb)))))


def rsa_coherence(hr: np.ndarray, grid_hz: float = HR_GRID_HZ) -> float:
    """Fraction of HR spectral power inside the respiratory band.

    The classical respiratory-sinus-arrhythmia readout. Falls when the respiratory
    modulation of heart rate is lost, which is the earliest autonomic sign here.
    """
    if len(hr) < 32:
        return 0.0
    f, p = welch(np.asarray(hr, dtype=float) - np.mean(hr), fs=grid_hz,
                 nperseg=min(256, len(hr)))
    tot = float(np.trapz(p, f))
    if tot <= 0:
        return 0.0
    band = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    return float(np.trapz(p[band], f[band]) / tot)


# --------------------------------------------------------------------------- #
@dataclass
class Coupling:
    te_resp_to_hr: float
    te_hr_to_resp: float
    te_asymmetry: float          # net directed flow; respiration normally drives HR
    phase_sync: float
    rsa_coherence: float
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def analyze(cardiac: np.ndarray, resp: np.ndarray, fs: float, kind: str = "ecg",
            seed: int = 0) -> Dict[str, Any]:
    """Full coupling readout from a raw cardiac waveform plus a respiration channel."""
    hr, grid = instantaneous_hr(cardiac, fs, kind=kind)
    if hr.size < 32:
        return Coupling(0.0, 0.0, 0.0, 0.0, 0.0, 0).to_dict()
    rs = resp_envelope(resp, fs, grid_hz=grid, n=len(hr))
    k = min(len(hr), len(rs))
    hr, rs = hr[:k], rs[:k]
    r2h = transfer_entropy(rs, hr, seed=seed)["te"]
    h2r = transfer_entropy(hr, rs, seed=seed + 1)["te"]
    return Coupling(te_resp_to_hr=r2h, te_hr_to_resp=h2r, te_asymmetry=r2h - h2r,
                    phase_sync=phase_sync(hr, rs, fs=grid),
                    rsa_coherence=rsa_coherence(hr, grid),
                    n_samples=k).to_dict()


def decouple(resp: np.ndarray, alpha: float, seed: int = 0) -> np.ndarray:
    """Synthetically break cardio-respiratory coupling by ``alpha`` in [0, 1].

    Blends respiration toward a phase-randomised surrogate of itself. The power
    spectrum is preserved throughout, so the breathing still *looks* normal in rate and
    depth — only its timing relationship to the heart is destroyed. That is the point:
    a control that also changed amplitude would let the estimators respond to the wrong
    thing and the monotonicity result would prove nothing. Clearly synthetic; used only
    by the validation.
    """
    x = np.asarray(resp, dtype=float)
    sur = phase_randomise(x, np.random.default_rng(seed))
    return (1.0 - alpha) * x + alpha * sur
