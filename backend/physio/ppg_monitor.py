"""Live PPG morphology monitor — the vascular-tone pathway, streaming.

`ppg_morphology.py` validated the feature extractor in batch. This wraps it into a
LIVE signal: a real high-rate PPG waveform (BIDMC ICU record, 125 Hz) is replayed
through a sliding window and morphology is recomputed each tick, yielding a
continuous vascular-tone readout. BIDMC replay stands in for the BLE wearable we
have no hardware for — the same honest-replay stance the CinC cohort takes for
live patients (clearly labelled as replay, never presented as a live sensor).

It also exposes a ``vasodilation`` control: the operator dials up the
:func:`erode_notch` model and watches, in real time, the dicrotic notch flatten
and the tone metrics collapse — a direct, interactive demonstration of the exact
physiology distributive shock produces.

Honest boundary: this is a STANDALONE vascular-tone monitor. It is NOT fused into
the cohort cross-modal resonance gate, because that would require one patient with
synchronized high-rate PPG *and* vitals/labs; mixing a BIDMC waveform with a CinC
bed would be two different people. True fusion is a hardware-deployment step.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from backend.physio.data import bidmc
from backend.physio.hrv import _bandpass, detect_ppg_peaks
from backend.physio.ppg_morphology import analyze, erode_notch

WINDOW_S = 6.0             # analysis + display window
WAVE_POINTS = 220          # downsampled points sent to the UI


def _tone_label(notch_rate: float, ri: float) -> str:
    if notch_rate >= 0.5 or ri >= 0.06:
        return "preserved"
    if notch_rate >= 0.15 or ri >= 0.03:
        return "reduced"
    return "collapsed"


class PPGMorphologyMonitor:
    def __init__(self, record: Optional[bidmc.BidmcRecord] = None):
        if record is None:
            recs = bidmc.load_records(limit=20)
            # pick a record that actually HAS a dicrotic notch, so the vasodilation
            # demo has something to erode (records 9 / 19 are notch-rich).
            record = self._pick_notched(recs) if recs else None
        self.rec = record
        self.fs = record.fs if record else 125.0
        self.win = int(WINDOW_S * self.fs)
        self.pos = 0
        self.alpha = 0.0       # vasodilation (0 = native, 1 = notch fully eroded)
        self.available = record is not None

    @staticmethod
    def _pick_notched(recs: List[bidmc.BidmcRecord]) -> bidmc.BidmcRecord:
        best, best_rate = recs[0], -1.0
        for r in recs:
            seg = r.ppg[:int(WINDOW_S * r.fs * 4)]
            a = analyze(seg, r.fs)
            if a["n_beats"] >= 10 and a["notch_rate"] > best_rate:
                best, best_rate = r, a["notch_rate"]
        return best

    # ------------------------------------------------------------------ #
    def set_vasodilation(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def tick(self, dt_s: float = 1.0) -> None:
        if not self.available:
            return
        self.pos += int(dt_s * self.fs)
        if self.pos + self.win >= len(self.rec.ppg):
            self.pos = 0                                    # loop the record

    # ------------------------------------------------------------------ #
    def state(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False,
                    "note": "BIDMC PPG not present — run backend/physio/data/bidmc.download"}
        seg = self.rec.ppg[self.pos:self.pos + self.win].astype(np.float64)
        if self.alpha > 0:
            seg = erode_notch(seg, self.fs, self.alpha)     # live vasodilation model
        morph = analyze(seg, self.fs)

        # transport waveform: clean, normalize to [0,1], downsample for the UI
        clean = _bandpass(seg, self.fs, 0.5, 8.0)
        peaks = np.round(detect_ppg_peaks(clean, self.fs)).astype(int)
        lo, hi = float(clean.min()), float(clean.max())
        norm = (clean - lo) / (hi - lo + 1e-9)
        idx = np.linspace(0, len(norm) - 1, min(WAVE_POINTS, len(norm))).astype(int)
        wave = [round(float(norm[i]), 3) for i in idx]
        # map systolic peaks into the downsampled coordinate frame
        scale = (len(wave) - 1) / max(1, len(norm) - 1)
        pk = sorted({int(round(p * scale)) for p in peaks if 0 <= p < len(norm)})

        return {
            "available": True, "record": self.rec.record_id, "fs": self.fs,
            "alpha": round(self.alpha, 2),
            "hr": round(60.0 * self.fs / np.median(np.diff(peaks)), 0)
                  if len(peaks) > 2 else None,
            "ba_ratio": morph["ba_ratio"], "reflection_index": morph["reflection_index"],
            "notch_rate": morph["notch_rate"], "crest_time": morph["crest_time"],
            "n_beats": morph["n_beats"],
            "tone": _tone_label(morph["notch_rate"] or 0.0, morph["reflection_index"] or 0.0),
            "wave": wave, "peaks": pk,
        }
