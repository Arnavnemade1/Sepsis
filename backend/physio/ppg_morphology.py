"""PPG pulse-wave morphology — the dicrotic-notch / vascular-tone pathway.

A photoplethysmogram is not just a heart-rate metronome; the *shape* of each
pulse encodes the state of the arterial tree. After the systolic upstroke the
wave decelerates at the **dicrotic notch** (aortic-valve closure) and rises again
into a **diastolic peak** built from the reflected pressure wave. As systemic
vascular tone falls — the hallmark of distributive (septic) shock — wave
reflection weakens: the diastolic peak shrinks, the notch flattens, and the pulse
collapses toward a single triangular hump. Tracking that erosion turns a raw PPG
(wearable contact sensor, or a camera rPPG waveform) into a continuous proxy for
arterial tone and fluid responsiveness.

This module extracts, per beat, the established morphology descriptors:

  * **Reflection Index (RI)** — diastolic-peak height ÷ systolic-peak height. The
    direct measure of how much reflected wave survives; it falls as tone drops.
  * **Dicrotic-notch detectability + relative depth** — whether the notch is even
    present, and how pronounced. Erosion → lower detection rate.
  * **Crest time** — foot-to-systolic-peak interval (vascular-tone-sensitive).
  * **b/a ratio** of the acceleration PPG (2nd derivative) — the classic
    stiffness/ageing index, robust because it needs no diastolic-peak search.
  * **PW50** — pulse width at 50% systolic amplitude.

HONEST SCOPE (the integrity rule): this is a *signal extractor + mechanistic
proxy*, not a sepsis classifier trained on labels. It is validated on REAL BIDMC
ICU PPG for the things BIDMC can prove — that notches are detected reliably, that
the features are stable within a record, and that they respond *monotonically* to
controlled notch erosion (the exact change vasodilation causes). Its role in
AegisOS is to be one more INDEPENDENT physical pathway feeding the cross-modal
resonance gate; the gate, not this module, decides corroboration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from backend.physio.hrv import _bandpass, detect_ppg_peaks


@dataclass
class BeatMorphology:
    crest_time: float                 # s, foot → systolic peak
    reflection_index: float           # diastolic-peak height / systolic height
    notch_detected: bool
    notch_depth: float                # dicrotic-notch height / systolic (0 if none)
    ba_ratio: float                   # APG b/a (arterial-stiffness index)
    pw50: float                       # s, pulse width at 50% systolic amplitude

    def to_dict(self) -> Dict[str, Any]:
        return {"crest_time": round(self.crest_time, 4),
                "reflection_index": round(self.reflection_index, 4),
                "notch_detected": self.notch_detected,
                "notch_depth": round(self.notch_depth, 4),
                "ba_ratio": round(self.ba_ratio, 4),
                "pw50": round(self.pw50, 4)}


def _segment_beats(x: np.ndarray, peaks: np.ndarray, fs: float) -> List[tuple]:
    """Return (foot, peak, end) index triples — one analysable beat each. The foot
    is the onset minimum just before a systolic peak; the beat runs foot→next foot."""
    if len(peaks) < 3:
        return []
    ibi = float(np.median(np.diff(peaks)))
    look = max(int(0.5 * ibi), 2)
    out = []
    for i in range(len(peaks) - 1):
        p, pn = int(peaks[i]), int(peaks[i + 1])
        lo = max(0, p - look)
        foot = lo + int(np.argmin(x[lo:p + 1])) if p > lo else p
        nlo = max(p, pn - look)
        end = nlo + int(np.argmin(x[nlo:pn + 1])) if pn > nlo else pn
        if end - foot > 4 and p - foot > 1 and end > p:
            out.append((foot, p, end))
    return out


def _beat_features(x: np.ndarray, foot: int, peak: int, end: int,
                   fs: float) -> Optional[BeatMorphology]:
    seg = x[foot:end + 1].astype(np.float64)
    if seg.size < 6:
        return None
    base = float(seg[0])
    # Re-localize the systolic peak inside THIS filtered segment (the detector's
    # narrowband peak can sit a few ms off) — take the max over the first ~40%.
    sys_lim = max(2, int(0.4 * seg.size))
    sp = int(np.argmax(seg[:sys_lim]))
    amp_s = float(seg[sp]) - base
    if amp_s <= 1e-9 or sp < 1:
        return None
    crest_time = sp / fs

    # --- reflected (diastolic) wave via deviation from the systolic→foot chord -- #
    # A beat with no reflection decays convexly and stays BELOW the straight chord
    # from the systolic peak to the end-diastolic foot. A dicrotic notch, shoulder
    # or diastolic peak lifts the downslope toward/above that chord; the size of
    # that lift is a robust, continuous reflected-wave measure that captures
    # shoulder-type notches (common in damped ICU PPG), not just distinct peaks —
    # and →0 as vascular tone falls (the erosion model blends toward this chord).
    skip = max(1, int(0.05 * fs))
    s0 = min(sp + skip, seg.size - 2)
    down = seg[s0:]
    notch_detected, ri, notch_depth = False, 0.0, 0.0
    if down.size > 3:
        chord = np.linspace(down[0], down[-1], down.size)
        dev = down - chord
        j = int(np.argmax(dev))
        ri = max(0.0, float(dev[j]) / amp_s)             # reflection index (strength)
        notch_detected = ri > 0.04
        if notch_detected:
            n_idx = int(np.argmin(seg[sp:s0 + j + 1]))   # notch = trough before the lift
            notch_depth = (float(seg[sp + n_idx]) - base) / amp_s

    # --- APG b/a (2nd-derivative arterial-stiffness index) ------------------ #
    sm = gaussian_filter1d(seg, sigma=max(1.0, 0.012 * fs))
    apg = np.gradient(np.gradient(sm))
    sys_end = min(seg.size, sp + max(2, int(0.10 * fs)))
    ba = 0.0
    if sys_end > 3:
        a_idx = int(np.argmax(apg[:sys_end]))           # a-wave (early systolic +)
        if a_idx + 1 < sys_end and apg[a_idx] > 1e-12:
            b_idx = a_idx + int(np.argmin(apg[a_idx:sys_end]))
            ba = float(apg[b_idx] / apg[a_idx])          # typically negative

    # --- pulse width at 50% systolic amplitude ------------------------------ #
    half = base + 0.5 * amp_s
    above = np.where(seg >= half)[0]
    pw50 = (above[-1] - above[0]) / fs if above.size >= 2 else 0.0

    return BeatMorphology(crest_time, float(np.clip(ri, 0.0, 1.5)),
                          notch_detected, float(np.clip(notch_depth, -0.2, 1.0)),
                          float(np.clip(ba, -2.0, 1.0)), pw50)


def analyze(ppg: np.ndarray, fs: float, max_beats: int = 600) -> Dict[str, Any]:
    """Extract per-beat morphology + record-level aggregates from a PPG signal."""
    x = _bandpass(np.asarray(ppg, dtype=np.float64), fs, 0.4, 12.0)  # drift + HF out, notch kept
    peaks_f = detect_ppg_peaks(x, fs)
    peaks = np.round(peaks_f).astype(int)
    peaks = peaks[(peaks > 0) & (peaks < len(x) - 1)]
    beats: List[BeatMorphology] = []
    for foot, peak, end in _segment_beats(x, peaks, fs)[:max_beats]:
        bm = _beat_features(x, foot, peak, end, fs)
        # quality gate: physiologic crest time + a real systolic excursion
        if bm is not None and 0.06 <= bm.crest_time <= 0.35 and bm.pw50 > 0:
            beats.append(bm)
    return _aggregate(beats)


def _aggregate(beats: List[BeatMorphology]) -> Dict[str, Any]:
    if not beats:
        return {"n_beats": 0, "notch_rate": 0.0, "reflection_index": None,
                "ba_ratio": None, "crest_time": None, "pw50": None, "beats": []}
    ri_all = np.array([b.reflection_index for b in beats])     # continuous, all beats
    return {
        "n_beats": len(beats),
        # primary pathway feature: continuous mean reflected-wave strength over ALL
        # beats (→0 as the notch erodes, robust to a few missed beats).
        "reflection_index": round(float(np.mean(ri_all)), 4),
        "notch_rate": round(float(np.mean([b.notch_detected for b in beats])), 3),
        "notch_depth": round(float(np.median([b.notch_depth for b in beats])), 4),
        "ba_ratio": round(float(np.median([b.ba_ratio for b in beats])), 4),
        "crest_time": round(float(np.median([b.crest_time for b in beats])), 4),
        "pw50": round(float(np.median([b.pw50 for b in beats])), 4),
        "beats": [b.to_dict() for b in beats],
    }


def erode_notch(ppg: np.ndarray, fs: float, alpha: float) -> np.ndarray:
    """Synthetically erode the dicrotic notch by ``alpha`` ∈ [0,1] — a controlled
    model of falling vascular tone: each beat's diastolic (post-systolic) segment is
    blended toward a monotonic decay, collapsing the reflected wave while leaving
    the systolic upstroke and beat timing untouched. Used only by the sensitivity
    validation; clearly synthetic."""
    x = _bandpass(np.asarray(ppg, dtype=np.float64), fs, 0.4, 12.0).copy()
    peaks = np.round(detect_ppg_peaks(x, fs)).astype(int)
    peaks = peaks[(peaks > 0) & (peaks < len(x) - 1)]
    for foot, peak, end in _segment_beats(x, peaks, fs):
        if end - peak < 3:
            continue
        decay = np.linspace(x[peak], x[end], end - peak + 1)   # monotonic systole→foot
        x[peak:end + 1] = (1 - alpha) * x[peak:end + 1] + alpha * decay
    return x
