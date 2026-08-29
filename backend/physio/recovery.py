"""Autonomic recovery profiling — sepsis as a loss of control-loop *speed*.

WHY THIS MODULE EXISTS
----------------------
This project's detection work has now been measured to its floor. Eight independent
method families — multi-scale temporal features, MiniRocket, Forest-VSA, a dilated TCN,
a Mamba-2 state space model, critical slowing down, quantum HDC, and cross-channel
coordination — all failed to beat a flat-feature booster on hourly vitals, and the
supervision-density work (``docs/SUPERVISION_DENSITY.md``) explained why: essentially
all of the accessible signal is *which patient*, and almost none of it is *when*. The
shipped detector is a risk stratifier, not an early-warning system.

The mechanism behind that is not a modelling failure. The sepsis cascade runs

    pathogen → cytokine surge → endothelial activation → MICROCIRCULATORY failure
             → capillary leak → macrocirculatory compensation → decompensation

and heart rate, blood pressure and respiratory rate only move at the *compensation*
stage. By the time an hourly vital sign changes, the cascade is already advanced. Eight
nulls are one fact measured eight times: the signal arrives late because we are reading
it late.

THE PRECEDENT, AND WHY IT DOES NOT TRANSFER UNCHANGED
------------------------------------------------------
Exactly one sepsis early-warning system has ever reduced mortality in a randomised
trial: HeRO (Moorman et al., ~3000 VLBW infants, >20% mortality reduction), which reads
**beat-to-beat** heart-rate characteristics — reduced variability, transient
decelerations, sample asymmetry — and predicts deterioration ~24h out. Its *model* is
close to a logistic regression. The power is in the signal, not the algorithm, and the
signal is one hourly sampling destroys by construction.

``backend/physio/sepsis_hrv.py`` already implements that feature family. What it cannot
do is survive an adult ICU. Adult heart-rate variability *magnitude* is dominated by
things that have nothing to do with sepsis: beta-blockade, sedation depth, mechanical
ventilation, atrial fibrillation, pacing, and age-related autonomic decline. Measuring
SDNN in an adult ICU is substantially measuring the drug chart, which is the standing
explanation for why adult HeRO analogues have not replicated.

WHAT THIS MODULE MEASURES INSTEAD
---------------------------------
Not the *size* of the autonomic response — the *speed* of it.

An ICU record is full of spontaneous perturbations: transient decelerations, respiratory
excursions, brief pressure dips. Each one is a free stress test the patient performs on
themselves. A healthy cardiovascular control loop is knocked off its operating point and
returns quickly; a failing one returns slowly, or does not return. So for every detected
perturbation this module fits a **recovery time constant** τ — the time to decay to 1/e
of the peak deviation — and profiles the distribution of τ, its asymmetry between
accelerations and decelerations, and the fraction of perturbations that never resolve.

THE PROPERTY THAT MOTIVATES THE WHOLE DESIGN
---------------------------------------------
τ is **amplitude-invariant by construction**, and that is provable rather than hoped
for. Scale every RR deviation by α (which is roughly what beta-blockade, deep sedation
or autonomic ageing do to the response amplitude):

    SDNN, RMSSD, total power   scale by α        → collapse
    peak deviation             scales by α
    the 1/e crossing of that peak                → UNCHANGED

Event detection is thresholded in MAD units, which also scale by α, so the events found
are the same events.

One honest caveat, discovered by the audit rather than reasoned out in advance: in RR
analysis **the signal is also the clock**. Shrinking the deviations shrinks the actual
inter-beat durations, so a τ measured in *seconds* moves slightly even though the
sequence of beats is identical. Measured at α = 0.3: SDNN falls 70% while the recovery
features drift ~6% — a twelvefold difference in robustness, but not the exact zero the
first draft of this docstring claimed. ``tau_median_beats`` is exactly invariant and is
carried alongside the seconds version for that reason.

THE THEORETICAL GROUNDING, AND A NULL THIS DOES NOT CONTRADICT
---------------------------------------------------------------
Critical slowing down predicts that a system approaching a critical transition recovers
more slowly from perturbation — rising τ is the canonical early-warning indicator. This
project already tried critical slowing down and measured ~0 gain, so it is worth being
explicit about why that is not evidence against this module. CSD is defined at the
system's *relaxation timescale*. For cardiovascular control that is seconds; the earlier
attempt sampled hourly, which aliases the entire phenomenon away before any estimator
sees it. The previous null was a statement about sampling rate, not about physiology.

HONEST SCOPE — READ THIS BEFORE QUOTING ANY NUMBER FROM HERE
-------------------------------------------------------------
This is a **signal extractor with a mechanistic rationale, not a validated sepsis
detector**, and the distinction is the same one ``coupling.py`` draws:

  * Amplitude invariance is proven, and it addresses ONE confound class — anything that
    scales response amplitude. It does **not** address atrial fibrillation or frequent
    ectopy (which destroy RR structure rather than scaling it), nor a ventilator
    imposing its own rhythm. Those need detection and exclusion, not invariance, and
    that work is not done here.
  * BIDMC — the only waveform data in this repository — has **no sepsis labels**, and
    CinC 2019 is hourly, which is far too coarse for any estimator in this file. **No
    AUROC, lead-time or detection claim may be made from this module on the data
    currently present.** What CAN be shown on real data is that the estimator recovers a
    known τ and reads dynamics rather than the marginal distribution.
  * **The camera path is not cleared.** The obvious hope was that these features would
    agree between an ECG-derived and a PPG-derived pulse, which is what would let them
    run on the contactless path. Measured on the same BIDMC patients at the same
    moments, they do not: r = 0.17, with PPG short-term variability running 2-4x the
    simultaneous ECG value because pulse transit time jitters independently of the
    cardiac cycle. ``assert_source_cleared`` refuses a pulse source rather than letting
    the pipeline emit numbers that look reasonable. This is the single biggest open
    problem in this module.
  * The detection claim needs credentialed high-resolution labelled data (MIMIC-IV
    Waveform, HiRID). The features here are emitted through the same
    ``RRLabeledSource`` contract ``sepsis_hrv.py`` uses, so that validation runs
    unchanged the day the data arrives.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sps

from backend.physio import hrv

#: Deviation, in robust (MAD) units, that starts an event. Scale-free by construction.
#: Set from measurement, not taste: on real BIDMC ICU records the event rate is
#: 2.08 / 1.13 / 0.44 per minute at k = 1.5 / 2.0 / 2.5, and a power analysis of the
#: estimator (``scripts`` note in verify_recovery) puts the knee at ~15 events per
#: window — CV of the median tau is 15.0% at 6 events, 8.9% at 10, 5.5% at 15, and
#: flat beyond. k = 2.5 yielded ~3 events in an 8-minute record, which is why the first
#: version of this module was too noisy to be a property of the patient at all.
EVENT_K_MAD: float = 1.5

#: An excursion must persist this many beats to count. One aberrant beat is ectopy or a
#: detection miss, not a control-loop perturbation.
EVENT_MIN_BEATS: int = 2

#: Recovery is declared when the deviation falls below this fraction of its peak. 1/e is
#: the definition of a time constant for exponential relaxation, so this is the standard
#: quantity rather than a tuned one.
RECOVERY_FRACTION: float = 1.0 / np.e

#: Pulse sources these features are cleared to run on. ECG only, and that is a measured
#: restriction rather than a cautious default: on the same BIDMC patients at the same
#: moments, PPG-derived RR carries 2-4x the short-term variability of the simultaneous
#: ECG-derived RR (SDNN 12.1 -> 22.9, 6.7 -> 21.6 ms on two records), because pulse
#: transit time jitters independently of the cardiac cycle. That jitter manufactures
#: perturbations this module then measures the recovery of — one record showed 1 event
#: on ECG and 10 on PPG — and the two channels' tau estimates correlate at only r = 0.17.
#: Until rPPG peak timing improves or a denoiser is validated, recovery features from a
#: pulse source are not evidence about the patient, so the pipeline refuses them rather
#: than emitting numbers that look fine.
CLEARED_PULSE_SOURCES = ("ecg",)


class UnclearedPulseSource(ValueError):
    """Raised when recovery features are requested from a pulse source that the
    ECG-vs-PPG concordance check has not cleared."""


def assert_source_cleared(source: str) -> None:
    if source not in CLEARED_PULSE_SOURCES:
        raise UnclearedPulseSource(
            f"pulse source {source!r} is not cleared for recovery features. Measured on "
            f"BIDMC: PPG-derived tau correlates r=0.17 with simultaneous ECG-derived tau, "
            f"and PPG short-term variability runs 2-4x higher from pulse-transit jitter, "
            f"which fabricates perturbations. Cleared sources: {CLEARED_PULSE_SOURCES}. "
            f"Re-run backend.physio.verify_recovery after improving peak timing to change "
            f"this.")


@dataclass
class Perturbation:
    """One spontaneous excursion of the RR series and the return that followed it."""
    start_beat: int
    peak_beat: int
    sign: int                    # +1 deceleration (RR lengthens), -1 acceleration
    amplitude_ms: float          # peak deviation from baseline
    amplitude_mad: float         # the same, in robust units (scale-free)
    tau_s: Optional[float]       # time to decay to 1/e of peak; None if never recovered
    tau_beats: Optional[float]
    recovered: bool
    overshoot: bool              # crossed the baseline on the way back


@dataclass
class RecoveryFeatures:
    """Per-window summary of how fast the control loop returns to its operating point.

    Every field is either a time, a count rate, a ratio or a fraction. None of them is
    an amplitude, which is the point: amplitudes are what the drug chart moves.
    """
    n_events: int = 0
    events_per_min: float = 0.0
    tau_median_s: float = np.nan
    #: The same quantity counted in BEATS rather than seconds. Exactly invariant to
    #: response amplitude, where the seconds version is only approximately so — see
    #: the amplitude-invariance note in the module docstring.
    tau_median_beats: float = np.nan
    tau_iqr_s: float = np.nan
    tau_p90_s: float = np.nan
    tau_decel_s: float = np.nan          # vagal-dominant branch
    tau_accel_s: float = np.nan          # sympathetic-dominant branch
    tau_asymmetry: float = np.nan        # decel/accel ratio — branch imbalance
    incomplete_frac: float = np.nan      # events that never came back
    overshoot_frac: float = np.nan       # damping: does it ring, or crawl?
    amplitude_mad_median: float = np.nan  # kept for diagnostics, NOT scale-free
    quality: float = 0.0                 # 0 = nothing usable in this window

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


#: The ordered feature vector handed to a model. ``amplitude_mad_median`` is deliberately
#: excluded: it is the one field that is not invariant to the confound this module exists
#: to survive, and including it would quietly reintroduce it.
RECOVERY_FEATURE_NAMES: List[str] = [
    "events_per_min", "tau_median_s", "tau_median_beats", "tau_iqr_s", "tau_p90_s",
    "tau_decel_s", "tau_accel_s", "tau_asymmetry",
    "incomplete_frac", "overshoot_frac",
]


# --------------------------------------------------------------------------- #
# Perturbation detection and recovery fitting
# --------------------------------------------------------------------------- #
def _robust_baseline(nn: np.ndarray, halflife_beats: float = 40.0) -> np.ndarray:
    """A slow, causal operating point for the RR series.

    Causal on purpose. A centred filter would let a perturbation's own recovery pull the
    baseline toward it and shrink the very τ being measured — the estimator would read
    faster recovery precisely where recovery was worst.
    """
    alpha = 1.0 - np.exp(-np.log(2.0) / max(halflife_beats, 1.0))
    out = np.empty_like(nn, dtype=np.float64)
    acc = float(nn[0])
    for i, v in enumerate(nn):
        acc = alpha * float(v) + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to be a consistent estimator of sigma."""
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def _causal_scale(dev: np.ndarray, halflife_beats: float = 60.0,
                  warmup: int = 30) -> np.ndarray:
    """A per-beat robust scale for the deviation series, using only the past.

    The first version of this module took a single MAD over the whole window. That is
    the ordinary thing to do offline and it is a **leak**: the detection threshold at
    beat i then depends on beats after i, so rewriting a record's second half changed
    which perturbations were found in its first half — which the causality check caught.
    It matters because this estimator is meant to run on a live stream inside the OS,
    where there is no future to average over.

    E|x| = sigma*sqrt(2/pi) for a Gaussian, so the mean absolute deviation is converted
    with 1.2533. During warmup the scale is held at the first stable estimate rather
    than allowed to be tiny, which would otherwise make the opening beats of every
    record look like a storm of perturbations.
    """
    a = np.abs(np.asarray(dev, dtype=np.float64))
    alpha = 1.0 - np.exp(-np.log(2.0) / max(halflife_beats, 1.0))
    out = np.empty_like(a)
    acc = float(a[0])
    for i, v in enumerate(a):
        acc = alpha * float(v) + (1.0 - alpha) * acc
        out[i] = acc
    scale = 1.2533 * out
    w = min(warmup, len(scale) - 1)
    if w > 0:
        scale[:w] = scale[w]
    return scale


def find_perturbations(nn_ms: np.ndarray,
                       k_mad: float = EVENT_K_MAD,
                       min_beats: int = EVENT_MIN_BEATS,
                       recovery_fraction: float = RECOVERY_FRACTION,
                       ) -> Tuple[List[Perturbation], np.ndarray]:
    """Locate spontaneous excursions and measure the return from each.

    Returns the perturbations and the baseline used, so a caller can plot or audit the
    decomposition rather than take it on trust.
    """
    nn = np.asarray(nn_ms, dtype=np.float64)
    if len(nn) < 16:
        return [], np.array([])
    base = _robust_baseline(nn)
    dev = nn - base
    scale = _causal_scale(dev)                  # per-beat, past-only (see _causal_scale)
    if not np.all(np.isfinite(scale)) or np.all(scale <= 0):
        return [], base

    beat_time_s = np.cumsum(nn) / 1000.0        # beat i occurs at this many seconds
    thr = k_mad * np.maximum(scale, 1e-9)
    above = np.abs(dev) >= thr

    events: List[Perturbation] = []
    i = 0
    n = len(nn)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j] and np.sign(dev[j]) == np.sign(dev[i]):
            j += 1
        if j - i < min_beats:
            i = j
            continue

        seg = dev[i:j]
        sign = int(np.sign(seg[np.argmax(np.abs(seg))]))
        peak_off = int(np.argmax(np.abs(seg)))
        peak = i + peak_off
        amp = float(dev[peak])
        scale_at_peak = float(max(scale[peak], 1e-9))

        # Recovery: walk forward from the peak until the deviation has decayed to 1/e of
        # it. "Never" is a real outcome and is recorded as such rather than imputed —
        # a perturbation that does not resolve is the strongest thing in this file.
        target = abs(amp) * recovery_fraction
        tau_s: Optional[float] = None
        tau_beats: Optional[float] = None
        overshoot = False
        for m in range(peak + 1, n):
            if abs(dev[m]) <= target:
                tau_s = float(beat_time_s[m] - beat_time_s[peak])
                tau_beats = float(m - peak)
                break
            if np.sign(dev[m]) != sign and abs(dev[m]) > target:
                overshoot = True
        events.append(Perturbation(
            start_beat=i, peak_beat=peak, sign=sign, amplitude_ms=amp,
            amplitude_mad=float(amp / scale_at_peak), tau_s=tau_s, tau_beats=tau_beats,
            recovered=tau_s is not None, overshoot=overshoot))
        i = j
    return events, base


def recovery_features(nn_ms: np.ndarray, clean: bool = True) -> RecoveryFeatures:
    """Profile one window of RR intervals.

    ``clean`` runs the project's existing NN filter first, so ectopic beats and detection
    misses do not masquerade as control-loop perturbations. It is shared with
    ``hrv.hrv_features`` deliberately: two estimators disagreeing about which beats are
    real would make their comparison meaningless.
    """
    nn = hrv.clean_nn(np.asarray(nn_ms, dtype=np.float64)) if clean else np.asarray(nn_ms, float)
    f = RecoveryFeatures()
    if len(nn) < 16:
        return f
    events, _ = find_perturbations(nn)
    duration_min = float(np.sum(nn) / 1000.0 / 60.0)
    f.n_events = len(events)
    f.events_per_min = float(len(events) / duration_min) if duration_min > 0 else 0.0
    if not events:
        # No perturbation is not the same as fast recovery, and must not read as a good
        # number. Quality 0 tells the caller this window carries no recovery evidence.
        f.quality = 0.0
        return f

    taus = np.array([e.tau_s for e in events if e.tau_s is not None], dtype=float)
    dec = np.array([e.tau_s for e in events if e.sign > 0 and e.tau_s is not None], float)
    acc = np.array([e.tau_s for e in events if e.sign < 0 and e.tau_s is not None], float)
    f.incomplete_frac = float(np.mean([not e.recovered for e in events]))
    f.overshoot_frac = float(np.mean([e.overshoot for e in events]))
    f.amplitude_mad_median = float(np.median([abs(e.amplitude_mad) for e in events]))
    tb = np.array([e.tau_beats for e in events if e.tau_beats is not None], dtype=float)
    if tb.size:
        f.tau_median_beats = float(np.median(tb))
    if taus.size:
        f.tau_median_s = float(np.median(taus))
        f.tau_iqr_s = float(np.subtract(*np.percentile(taus, [75, 25])))
        f.tau_p90_s = float(np.percentile(taus, 90))
    if dec.size:
        f.tau_decel_s = float(np.median(dec))
    if acc.size:
        f.tau_accel_s = float(np.median(acc))
    if dec.size and acc.size and f.tau_accel_s > 0:
        f.tau_asymmetry = float(f.tau_decel_s / f.tau_accel_s)
    # Quality is the share of events that produced a usable time constant, which is what
    # every headline feature here is computed from.
    f.quality = float(taus.size / len(events)) if events else 0.0
    return f


def feature_vector(f: RecoveryFeatures) -> np.ndarray:
    """The ordered vector, with non-finite entries left as NaN for the model to handle."""
    d = f.as_dict()
    return np.array([d[k] for k in RECOVERY_FEATURE_NAMES], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Respiratory coupling gain — the second confound-resistant channel
# --------------------------------------------------------------------------- #
@dataclass
class RSAGain:
    """Respiratory sinus arrhythmia expressed as a GAIN, not an amplitude.

    Amplitude answers "how much does the heart rate swing", which a ventilator or a
    sedative changes without any change in autonomic health. Gain answers "how much does
    it swing *per unit of respiratory drive*", which is a property of the reflex arc.
    """
    gain: float = np.nan             # ms of RR per unit normalised respiratory excursion
    phase_lag_s: float = np.nan
    coherence: float = np.nan        # how much of the RR variance the respiration explains
    quality: float = 0.0


def rsa_gain(nn_ms: np.ndarray, resp: np.ndarray, fs: float,
             band: Tuple[float, float] = (0.15, 0.40)) -> RSAGain:
    """Estimate the RR-per-respiration transfer gain over a window.

    ``band`` is the adult high-frequency band. The respiration signal is normalised to
    unit robust scale before the regression, which is what turns an amplitude into a
    gain and is the whole reason this is worth computing separately from HF power.
    """
    nn = np.asarray(nn_ms, dtype=np.float64)
    r = np.asarray(resp, dtype=np.float64)
    out = RSAGain()
    if len(nn) < 32 or len(r) < int(8 * fs):
        return out

    # Put RR on an even time grid so it can be compared with the respiration waveform.
    t_beats = np.cumsum(nn) / 1000.0
    t_beats -= t_beats[0]
    fs_i = 4.0
    if t_beats[-1] < 20.0:
        return out
    grid = np.arange(0.0, t_beats[-1], 1.0 / fs_i)
    rr_i = np.interp(grid, t_beats, nn)
    rr_i = rr_i - np.mean(rr_i)

    t_resp = np.arange(len(r)) / fs
    resp_i = np.interp(grid, t_resp, r)
    resp_i = resp_i - np.median(resp_i)
    s = _mad(resp_i)
    if not np.isfinite(s) or s <= 0:
        return out
    resp_i = resp_i / s                      # unit robust scale -> the gain denominator

    b, a = sps.butter(2, [band[0] / (fs_i / 2), band[1] / (fs_i / 2)], btype="band")
    rr_b = sps.filtfilt(b, a, rr_i)
    resp_b = sps.filtfilt(b, a, resp_i)

    denom = float(np.dot(resp_b, resp_b))
    if denom <= 0:
        return out
    out.gain = float(np.dot(rr_b, resp_b) / denom)
    num = float(np.dot(rr_b, resp_b))
    out.coherence = float(num * num / (denom * float(np.dot(rr_b, rr_b)) + 1e-12))
    # Phase lag from the cross-correlation peak, bounded to one respiratory period.
    xc = sps.correlate(rr_b, resp_b, mode="same")
    lags = (np.arange(len(xc)) - len(xc) // 2) / fs_i
    keep = np.abs(lags) <= 1.0 / band[0]
    if keep.any():
        out.phase_lag_s = float(lags[keep][int(np.argmax(np.abs(xc[keep])))])
    out.quality = 1.0
    return out


# --------------------------------------------------------------------------- #
# Windowing — the same contract sepsis_hrv.py uses, so both run on one source
# --------------------------------------------------------------------------- #
@dataclass
class WindowedRecoveryDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[str]


def build_recovery_dataset(source, window_beats: int = 512, step_beats: int = 128,
                           with_hrv: bool = True, pulse_source: str = "ecg",
                           ) -> WindowedRecoveryDataset:
    """Slice an ``RRLabeledSource`` into windows of recovery features.

    ``with_hrv`` appends the existing HeRO-style HRV vector, because the honest
    comparison this module has to win is not "recovery features vs nothing" but
    "recovery features vs the HRV features that already exist, and both together".
    Windows are 512 beats — roughly seven minutes at an adult ICU rate. That length is
    set by the power analysis rather than convenience: at the event rate measured on real
    ICU records it yields ~14 perturbations per window, and the estimator's coefficient
    of variation flattens at ~15 (15.0% at 6 events, 8.9% at 10, 5.5% at 15). Shorter
    windows would sharpen a lead-time claim and were what the first version used; they
    were also too noisy for tau to be a property of the patient rather than of the window.
    """
    assert_source_cleared(pulse_source)
    Xs, ys, gs = [], [], []
    names = list(RECOVERY_FEATURE_NAMES)
    if with_hrv:
        from backend.physio.sepsis_hrv import HRV_FEATURE_NAMES
        names = names + list(HRV_FEATURE_NAMES)

    for pid, rr_ms, labels in source:
        rr = np.asarray(rr_ms, dtype=np.float64)
        if len(rr) < window_beats:
            continue
        per_beat = (np.full(len(rr), int(labels)) if np.ndim(labels) == 0
                    else np.asarray(labels, dtype=int))
        for start in range(0, len(rr) - window_beats + 1, step_beats):
            win = rr[start:start + window_beats]
            rf = recovery_features(win)
            if rf.quality <= 0.0:
                continue
            vec = feature_vector(rf)
            if with_hrv:
                hf = hrv.hrv_features(win)
                if hf.quality <= 0.0:
                    continue
                d = hf.as_dict()
                from backend.physio.sepsis_hrv import HRV_FEATURE_NAMES
                vec = np.concatenate([vec, np.array([d[k] for k in HRV_FEATURE_NAMES])])
            Xs.append(vec)
            ys.append(int(per_beat[start:start + window_beats].max()))
            gs.append(pid)
    if not Xs:
        raise ValueError("no usable recovery windows produced from the source")
    return WindowedRecoveryDataset(np.vstack(Xs), np.array(ys, dtype=int),
                                   np.array(gs), names)


def rr_from_bidmc(path: str, kind: str = "ecg") -> np.ndarray:
    """RR intervals from one BIDMC signals CSV — the only real waveform data present.

    ``kind`` selects the pulse source: ``"ecg"`` uses lead II (gold standard) and
    ``"ppg"`` uses the PLETH channel, which is the camera-class signal. Having both from
    the SAME patient at the SAME time is what makes the concordance check in
    ``verify_recovery.py`` a real test of whether these features survive the camera path.
    """
    import pandas as pd
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    fs = 125.0
    if kind == "ecg":
        col = "II" if "II" in df.columns else ("V" if "V" in df.columns else "AVR")
        peaks = hrv.detect_ecg_rpeaks(df[col].to_numpy(dtype=float), fs)
    else:
        peaks = hrv.detect_ppg_peaks(df["PLETH"].to_numpy(dtype=float), fs)
    return hrv.rr_intervals_ms(peaks, fs)
