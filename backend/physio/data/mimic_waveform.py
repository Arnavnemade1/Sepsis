"""MIMIC-IV Waveform -> RRLabeledSource adapter (credential-ready).

This is the plug-and-play bridge that lets ``backend.physio.sepsis_hrv`` train and
validate on REAL labelled HRV the moment credentialed data is available. It reads
WFDB waveform records (MIMIC-IV-ECG / MIMIC-IV Waveform), extracts an ECG (or PPG)
channel, detects beats, builds the RR series, and aligns Sepsis-3 onset labels to
each beat — yielding exactly the ``(patient_id, rr_ms, labels)`` tuples the
pipeline expects.

HOW TO GET THE DATA (user action — requires credentialing):
  1. Complete CITI "Data or Specimens Only Research" training.
  2. Get credentialed on PhysioNet and sign the MIMIC-IV data use agreement.
  3. Download MIMIC-IV Waveform (WFDB records) + derive a Sepsis-3 onset table
     from the MIMIC-IV clinical modules (the `sepsis3` derived view from the
     official mimic-code repo gives stay_id + sepsis_onset time).
  4. Build a labels CSV: columns [record_id, onset_s] where onset_s is the Sepsis-3
     onset in SECONDS from the start of the waveform record (or NaN/empty if the
     patient never becomes septic). Point ``labels_csv`` at it.

NOTE on HiRID: HiRID stores derived heart rate at low cadence, not beat-to-beat
ECG, so it cannot yield true HRV — MIMIC-IV Waveform is the correct source here.

``wfdb`` is an optional dependency (``pip install wfdb``); the label-alignment
logic is pure-python and unit-tested without it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from backend.physio import hrv

# Sepsis-3 label lead: mark beats as positive starting this many hours BEFORE
# onset (mirrors the CinC 2019 early-prediction convention). 0 = label at onset.
DEFAULT_LEAD_HOURS = 6.0

# Preferred channel names to search for, in order (case-insensitive substrings).
_ECG_CHANNELS = ["ii", "i", "v", "ecg", "mlii"]
_PPG_CHANNELS = ["pleth", "ppg"]


def align_beat_labels(
    beat_times_s: np.ndarray,
    onset_s: Optional[float],
    lead_hours: float = DEFAULT_LEAD_HOURS,
) -> np.ndarray:
    """Per-beat 0/1 labels given a Sepsis-3 onset time (seconds from record start).

    Beats at or after (onset - lead) are positive. ``onset_s=None`` (never septic)
    -> all zeros. Pure function: the testable core of the adapter.
    """
    n = len(beat_times_s)
    if onset_s is None or not np.isfinite(onset_s):
        return np.zeros(n, dtype=int)
    threshold = float(onset_s) - lead_hours * 3600.0
    return (np.asarray(beat_times_s, dtype=float) >= threshold).astype(int)


def rr_and_labels(
    signal: np.ndarray,
    fs: float,
    onset_s: Optional[float],
    kind: str = "ecg",
    lead_hours: float = DEFAULT_LEAD_HOURS,
) -> Tuple[np.ndarray, np.ndarray]:
    """From a raw waveform + onset, return (rr_ms, per_beat_labels).

    Labels align to the SECOND beat of each RR pair (the interval's end time),
    so an RR interval is labelled by the physiological time it represents.
    """
    peaks = (hrv.detect_ecg_rpeaks(signal, fs) if kind == "ecg"
             else hrv.detect_ppg_peaks(signal, fs))
    if len(peaks) < 2:
        return np.array([]), np.array([], dtype=int)
    rr_ms = hrv.rr_intervals_ms(peaks, fs)
    beat_times_s = peaks[1:] / fs                      # end-of-interval timestamps
    labels = align_beat_labels(beat_times_s, onset_s, lead_hours)
    return rr_ms, labels


@dataclass
class MimicWaveformConfig:
    records_dir: str                       # dir of WFDB records (.hea/.dat)
    labels_csv: Optional[str] = None       # CSV: record_id, onset_s
    channel_kind: str = "ecg"              # "ecg" | "ppg"
    lead_hours: float = DEFAULT_LEAD_HOURS
    record_list: Optional[List[str]] = None  # explicit record ids, else autodiscover


class MimicWaveformRRSource:
    """Implements the ``RRLabeledSource`` protocol over a MIMIC-IV waveform dir."""

    def __init__(self, cfg: MimicWaveformConfig):
        self.cfg = cfg
        self._labels = self._load_labels(cfg.labels_csv)

    def _load_labels(self, path: Optional[str]) -> Dict[str, Optional[float]]:
        if not path or not os.path.exists(path):
            return {}
        import pandas as pd
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        rid_c = cols.get("record_id") or list(df.columns)[0]
        onset_c = cols.get("onset_s")
        out: Dict[str, Optional[float]] = {}
        for _, row in df.iterrows():
            rid = str(row[rid_c])
            onset = float(row[onset_c]) if onset_c and not pd.isna(row[onset_c]) else None
            out[rid] = onset
        return out

    def _discover_records(self) -> List[str]:
        if self.cfg.record_list:
            return list(self.cfg.record_list)
        ids = []
        for f in sorted(os.listdir(self.cfg.records_dir)):
            if f.endswith(".hea"):
                ids.append(f[:-4])
        return ids

    def _pick_channel(self, sig_names: List[str]) -> Optional[int]:
        prefs = _ECG_CHANNELS if self.cfg.channel_kind == "ecg" else _PPG_CHANNELS
        lower = [s.lower() for s in sig_names]
        for pref in prefs:
            for i, name in enumerate(lower):
                if pref in name:
                    return i
        return None

    def __iter__(self) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
        try:
            import wfdb
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "wfdb is required to read MIMIC-IV waveform records: pip install wfdb"
            ) from exc

        for rid in self._discover_records():
            path = os.path.join(self.cfg.records_dir, rid)
            try:
                rec = wfdb.rdrecord(path)
            except Exception as exc:  # pragma: no cover
                print(f"[mimic] skip {rid}: {exc}")
                continue
            ch = self._pick_channel(list(rec.sig_name))
            if ch is None:
                continue
            signal = np.asarray(rec.p_signal[:, ch], dtype=np.float64)
            fs = float(rec.fs)
            onset = self._labels.get(rid, None)
            rr_ms, labels = rr_and_labels(signal, fs, onset, self.cfg.channel_kind,
                                          self.cfg.lead_hours)
            if len(rr_ms) >= 2:
                yield rid, rr_ms, labels


def build_source(records_dir: str, labels_csv: Optional[str] = None,
                 channel_kind: str = "ecg") -> MimicWaveformRRSource:
    return MimicWaveformRRSource(MimicWaveformConfig(
        records_dir=records_dir, labels_csv=labels_csv, channel_kind=channel_kind))
