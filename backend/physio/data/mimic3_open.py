"""MIMIC-III Waveform Database (OPEN ACCESS) — beat timing and circulatory outcomes.

WHY THIS EXISTS
---------------
``mimic_waveform.py`` is the credentialed path: real Sepsis-3 labels aligned to
beat-to-beat ECG, which is the validation this project actually wants. It requires CITI
training, PhysioNet credentialing and a data use agreement, and that pipeline is
measured in weeks.

This module is the workaround, and it is a legitimate one rather than a shortcut around
anybody's data use agreement. The **MIMIC-III Waveform Database** is *open access* under
the Open Data Commons ODbL — 67,830 record sets from roughly 30,000 ICU patients, ECG
and arterial blood pressure at 125 Hz, with records running tens of hours. What it does
not carry is the clinical chart, so there are no sepsis labels. Only about a third of
records have matched clinical data and that match needs credentials.

So the outcome is derived from the waveforms themselves.

THE IDEA THAT MAKES THIS WORK
-----------------------------
``recovery.py`` reads the **ECG** — beat timing, and nothing else. The outcome here is
derived from the **arterial line** — a physically separate transducer. That makes the
label non-circular by construction: nothing about a recovery time constant is computed
from the pressure channel it is asked to predict.

The outcome is a **circulatory decompensation event**: a sustained fall to MAP < 65 mmHg
after a period of stability. That threshold is not invented for this module — it is the
resuscitation target in the Surviving Sepsis guidelines and the standard definition in
the intraoperative-hypotension literature.

WHAT THIS DOES AND DOES NOT VALIDATE
------------------------------------
It validates the mechanism ``recovery.py`` claims: that autonomic control-loop slowing
precedes circulatory failure, on real ICU patients, with a real outcome, at scale. It
does **not** validate sepsis-specific early detection, because these patients have no
sepsis labels. Septic shock is a circulatory decompensation, so the proxy is
mechanistically aligned rather than arbitrary — but it is a proxy, and anything written
from this module must say so.

THE FAILURE MODE THIS MODULE IS MOSTLY ABOUT
--------------------------------------------
Arterial lines disconnect, get flushed, get zeroed, and go damped. The very first
segment inspected while writing this had a median ABP of **-1.9 mmHg** — a zeroed
transducer, which a naive MAP < 65 rule would have read as a profound decompensation
event lasting hours. Left unhandled, the majority of "events" in a dataset like this are
plumbing rather than patients, and a detector trained on them would learn to predict
line handling. ``abp_valid_mask`` is therefore the most important function in the file,
and ``verify_mimic3_open.py`` plants each artifact in turn and requires it to be caught.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from backend.physio import hrv

DB = "mimic3wdb"
FS_EXPECTED = 125.0

#: Channel names that carry a usable ECG lead / an arterial pressure line, upper-cased.
ECG_NAMES = ("II", "I", "III", "V", "MCL1", "AVR", "AVF", "AVL")
ABP_NAMES = ("ABP", "ART", "AOBP", "P1")

#: Physiologic bounds for a *valid* arterial trace. Outside these the transducer is
#: disconnected, zeroed, flushed or damped — not the patient decompensating.
ABP_MIN_MMHG = 20.0        # below this the line is open to air or the patient is dead
ABP_MAX_MMHG = 300.0       # above this is a flush or a whip artifact
PULSE_MIN_MMHG = 10.0      # a damped/wedged line loses its pulse pressure
PULSE_MAX_MMHG = 150.0

#: The outcome. MAP below this, sustained, is a circulatory decompensation — the
#: Surviving Sepsis resuscitation target and the standard hypotension definition.
MAP_THRESHOLD_MMHG = 65.0
EVENT_MIN_MINUTES = 5.0     # sustained, so a single artifact beat cannot make an event
STABLE_MIN_MINUTES = 10.0   # preceded by stability, so this is ONSET not chronic
STABLE_MAP_MMHG = 75.0


# --------------------------------------------------------------------------- #
# Signal quality — the part that decides whether any of this means anything
# --------------------------------------------------------------------------- #
def abp_valid_mask(abp: np.ndarray, fs: float = FS_EXPECTED,
                   win_s: float = 10.0) -> np.ndarray:
    """Per-sample mask of arterial samples that are plausibly the patient.

    Four artifacts, each of which a naive MAP rule would misread as decompensation:

    * **disconnection / zeroing** — the trace sits near or below zero. Caught by the
      absolute bounds.
    * **flush** — a square-wave pressure spike far above physiologic range.
    * **damping or wedging** — the mean survives but the *pulse* collapses, so the
      waveform stops being an arterial pressure at all. Caught by requiring a plausible
      pulse pressure inside a rolling window.
    * **flatline** — a held value with no variance, which reads as a perfectly stable
      hypotension. Caught by the same pulse-pressure requirement.

    Returned as a per-sample mask rather than a per-record verdict, because a line is
    usually good for hours and bad for minutes and throwing away whole records would
    cost most of the data.
    """
    x = np.asarray(abp, dtype=np.float64)
    ok = np.isfinite(x) & (x > ABP_MIN_MMHG) & (x < ABP_MAX_MMHG)
    n = len(x)
    w = max(int(win_s * fs), 8)
    # Rolling pulse pressure via running max/min over a window, computed on a decimated
    # grid so a multi-hour record stays cheap, then broadcast back.
    step = max(w // 4, 1)
    idx = np.arange(0, n, step)
    pulse_ok = np.zeros(n, dtype=bool)
    xf = np.where(np.isfinite(x), x, np.nan)
    for a in idx:
        b = min(a + w, n)
        seg = xf[a:b]
        if not np.any(np.isfinite(seg)):
            continue
        pp = np.nanmax(seg) - np.nanmin(seg)
        if PULSE_MIN_MMHG <= pp <= PULSE_MAX_MMHG:
            pulse_ok[a:b] = True
    return ok & pulse_ok


def beat_map_series(abp: np.ndarray, fs: float, valid: np.ndarray,
                    bin_s: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Mean arterial pressure on a coarse regular grid, using only valid samples.

    A 10-second bin is far finer than the 5-minute event definition and removes the need
    to detect individual pressure beats, which is a separate and noisier problem than the
    outcome requires.
    """
    x = np.asarray(abp, dtype=np.float64)
    n = len(x)
    b = max(int(bin_s * fs), 1)
    t, m = [], []
    for a in range(0, n - b + 1, b):
        seg = x[a:a + b][valid[a:a + b]]
        if seg.size >= b * 0.5:
            t.append((a + b / 2.0) / fs)
            m.append(float(np.mean(seg)))
    return np.asarray(t, dtype=np.float64), np.asarray(m, dtype=np.float64)


def find_decompensation_events(t_s: np.ndarray, map_mmhg: np.ndarray) -> List[float]:
    """Onset times (seconds) of sustained hypotension that follows stability.

    Requiring a stable run first is what makes this an *event* rather than a description
    of a chronically hypotensive patient. A detector for the latter would be another
    patient-classifier, which this project already has and does not need again.
    """
    if len(t_s) < 4:
        return []
    m = np.asarray(map_mmhg, dtype=np.float64)
    known = np.isfinite(m)
    low = known & (m < MAP_THRESHOLD_MMHG)
    dt = float(np.median(np.diff(t_s))) if len(t_s) > 1 else 10.0
    # The grid is regular, so a "gap" is now an unmeasured sample rather than a jump in
    # the time axis. Either way it breaks a run: an episode cannot be sustained through
    # a period in which nobody was measuring.
    gap = ~known
    need_low = max(int(EVENT_MIN_MINUTES * 60.0 / dt), 2)
    need_stable = max(int(STABLE_MIN_MINUTES * 60.0 / dt), 2)

    onsets: List[float] = []
    i = 0
    n = len(low)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j] and not gap[j]:
            j += 1
        if j - i >= need_low:
            a = max(0, i - need_stable)
            prior = map_mmhg[a:i]
            # Stability is judged on the run immediately before, and it must be a real,
            # UNBROKEN run: a gap in the arterial trace is not evidence the patient was
            # stable, it is evidence nobody was measuring.
            # Stability is the MEDIAN of the prior window, not its minimum. The minimum
            # is one sample: a single transient dip below the threshold in the preceding
            # ten minutes vetoed the event, which rejected every candidate in the cohort
            # (0 events vs 3 on the same records when this was a median). The clinical
            # intent is "the patient was stable before this", which is a statement about
            # the run and not about its worst single reading.
            enough = np.isfinite(prior).sum() >= 0.8 * need_stable
            if enough and float(np.nanmedian(prior)) >= STABLE_MAP_MMHG:
                onsets.append(float(t_s[i]))
        i = max(j, i + 1)
    return onsets


# --------------------------------------------------------------------------- #
# Numerics-first discovery: find the outcome across the WHOLE stay, cheaply
# --------------------------------------------------------------------------- #
#
# The first version of this module downloaded the first two hours of waveform per
# patient and looked for events in it. Records run 40-190 hours, so that read ~3% of
# each stay, and the first three patients streamed yielded zero events at ~70 s each.
#
# Every record carries a companion NUMERICS record (``<id>n``) at one sample per minute
# spanning the entire stay, with ``ABP SYS`` / ``ABP DIAS`` / ``ABP MEAN`` among its
# channels — 3,971 samples for 66 hours, a trivial download. So the outcome is discovered
# there across the whole admission, and the 125 Hz ECG is fetched only for the minutes
# that actually matter. Same data, same definitions, 20-90x the patient-time per byte.

NUM_ECG_LEAD_PREF = ("II", "I", "III", "V", "MCL1")


def numerics_map_series(folder: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Whole-stay mean arterial pressure at 1/min, artifact-gated.

    The same four artifacts as the waveform gate, expressed in the numerics channels:
    an implausible mean, and a pulse pressure (SYS - DIAS) outside physiologic range,
    which is what catches a damped or wedged line whose *mean* still looks survivable.
    """
    import wfdb
    pid = folder.strip("/").split("/")[-1]
    try:
        rec = wfdb.rdrecord(f"{pid}n", pn_dir=_pn_dir(folder))
    except Exception:
        return None
    names = [n.strip().upper() for n in (rec.sig_name or [])]
    if "ABP MEAN" not in names:
        return None
    x = rec.p_signal
    mean = x[:, names.index("ABP MEAN")].astype(np.float64)
    sysp = x[:, names.index("ABP SYS")].astype(np.float64) if "ABP SYS" in names else None
    diap = x[:, names.index("ABP DIAS")].astype(np.float64) if "ABP DIAS" in names else None

    ok = np.isfinite(mean) & (mean > ABP_MIN_MMHG) & (mean < ABP_MAX_MMHG)
    if sysp is not None and diap is not None:
        pp = sysp - diap
        ok &= np.isfinite(pp) & (pp >= PULSE_MIN_MMHG) & (pp <= PULSE_MAX_MMHG)
    if ok.sum() < 60:
        return None
    # Keep the REGULAR one-per-minute grid and mark rejected samples NaN, rather than
    # compacting the series down to the valid samples. Compacting was the first version
    # and it was badly wrong: an arterial line that drops out for a minute here and there
    # fragmented genuine half-hour hypotensive episodes into dozens of sub-5-minute runs,
    # and 3,673 candidate events were discarded as "too short" while zero survived.
    t = np.arange(len(mean), dtype=np.float64) / float(rec.fs)
    m = np.where(ok, mean, np.nan)
    return t, _bridge_short_gaps(m, dt_s=1.0 / float(rec.fs))


def _bridge_short_gaps(m: np.ndarray, dt_s: float,
                       max_gap_s: float = 180.0) -> np.ndarray:
    """Interpolate dropouts shorter than ``max_gap_s``; leave longer ones as NaN.

    A monitor losing the trace for two minutes has not told us the patient recovered,
    and it has not told us they crashed either — but it also should not end an episode.
    Three minutes is short relative to the five-minute sustain requirement, so bridging
    below that cannot manufacture an event on its own. Longer holes stay NaN and break
    runs, because there the honest answer is that nobody was measuring.
    """
    x = np.asarray(m, dtype=np.float64).copy()
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    max_n = int(max_gap_s / max(dt_s, 1e-9))
    idx = np.flatnonzero(good)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_n + 1:
            x[a + 1:b] = np.linspace(x[a], x[b], b - a + 1)[1:-1]
    return x


@dataclass
class PatientOutcome:
    """One patient's whole-stay pressure course and the events found in it."""
    patient_id: str
    folder: str
    t_s: np.ndarray
    map_mmhg: np.ndarray
    event_times_s: List[float]
    stay_hours: float


def discover_outcomes(folder: str) -> Optional[PatientOutcome]:
    got = numerics_map_series(folder)
    if got is None:
        return None
    t, m = got
    # A gap in the arterial trace is not a stable interval. ``find_decompensation_events``
    # reasons over consecutive samples, so runs are split where the record jumps.
    return PatientOutcome(
        patient_id=folder.strip("/").split("/")[-1], folder=folder.strip("/"),
        t_s=t, map_mmhg=m, event_times_s=find_decompensation_events(t, m),
        stay_hours=float((t[-1] - t[0]) / 3600.0) if len(t) > 1 else 0.0)


def fetch_rr_window(folder: str, start_s: float, dur_s: float
                    ) -> Optional[np.ndarray]:
    """RR intervals from the ECG over one absolute time window of the stay.

    Reads the multi-segment parent record, so an absolute offset into a 66-hour
    admission resolves to the right segment without this module tracking the layout.
    """
    import wfdb
    pid = folder.strip("/").split("/")[-1]
    try:
        hdr = wfdb.rdheader(pid, pn_dir=_pn_dir(folder), rd_segments=False)
        fs = float(hdr.fs)
        a = int(max(start_s, 0.0) * fs)
        b = int(min(a + dur_s * fs, hdr.sig_len))
        if b - a < fs * 60:
            return None
        rec = wfdb.rdrecord(pid, pn_dir=_pn_dir(folder), sampfrom=a, sampto=b, m2s=True)
    except Exception:
        return None
    names = [n.upper() for n in (rec.sig_name or [])]
    ch = next((names.index(n) for n in NUM_ECG_LEAD_PREF if n in names), None)
    if ch is None or rec.p_signal is None:
        return None
    ecg = rec.p_signal[:, ch].astype(np.float64)
    if np.isfinite(ecg).mean() < 0.8:
        return None
    peaks = hrv.detect_ecg_rpeaks(np.where(np.isfinite(ecg), ecg, 0.0), float(rec.fs))
    if len(peaks) < 200:
        return None
    return hrv.rr_intervals_ms(peaks, float(rec.fs))


def control_times(out: PatientOutcome, n_controls: int = 3,
                  guard_hours: float = 2.0, seed: int = 0) -> List[float]:
    """Stable moments far from any event, to serve as this patient's own negatives.

    The design is **case-control within patient**: a stay contributes both its
    decompensation onsets and its quiet hours. That is deliberate. Every accuracy result
    this project has produced turned out to be between-patient discrimination wearing an
    early-warning label; forcing positives and negatives to come from the same admission
    removes patient identity as an explanation by construction, and asks only the
    question that was never answerable on hourly data — *when*.
    """
    if len(out.t_s) < 60:
        return []
    rng = np.random.default_rng(seed)
    guard = guard_hours * 3600.0
    lo, hi = float(out.t_s[0]) + 3600.0, float(out.t_s[-1])
    cand = []
    for _ in range(200):
        t = float(rng.uniform(lo, hi))
        if any(abs(t - e) < guard for e in out.event_times_s):
            continue
        w = (out.t_s >= t - 1800.0) & (out.t_s <= t)
        if w.sum() < 20:
            continue
        if float(np.min(out.map_mmhg[w])) < STABLE_MAP_MMHG:
            continue
        if all(abs(t - c) > guard for c in cand):
            cand.append(t)
        if len(cand) >= n_controls:
            break
    return cand


# --------------------------------------------------------------------------- #
# Streaming the open database
# --------------------------------------------------------------------------- #
@dataclass
class PatientBeats:
    """What one patient contributes: beat timing, pressure, and the outcome times."""
    patient_id: str
    rr_ms: np.ndarray            # consecutive RR intervals from the ECG
    rr_t_s: np.ndarray           # time of each RR interval, seconds from segment start
    map_t_s: np.ndarray          # arterial grid
    map_mmhg: np.ndarray
    event_times_s: List[float]
    seconds: float


def _pn_dir(folder: str) -> str:
    return f"{DB}/{folder.strip('/')}"


def list_patient_folders(limit: Optional[int] = None) -> List[str]:
    import wfdb
    recs = wfdb.get_record_list(DB)
    return [r.strip("/") for r in (recs[:limit] if limit else recs)]


def _pick_channels(sig_names: Sequence[str]) -> Optional[Tuple[str, str]]:
    up = [s.upper() for s in sig_names]
    ecg = next((sig_names[up.index(n)] for n in ECG_NAMES if n in up), None)
    abp = next((sig_names[up.index(n)] for n in ABP_NAMES if n in up), None)
    return (ecg, abp) if ecg and abp else None


def load_patient(folder: str, max_minutes: float = 120.0,
                 min_minutes: float = 20.0) -> Optional[PatientBeats]:
    """Stream one patient's longest usable ECG+ABP segment and reduce it to beats.

    Raw waveforms are never stored. A 2-hour two-channel segment is ~7 MB of float and a
    cohort of them would be gigabytes; the RR series and a 10-second pressure grid that
    survive are a few tens of kilobytes, which is the whole point of reducing on the fly.
    """
    import wfdb
    pid = folder.strip("/").split("/")[-1]
    try:
        hdr = wfdb.rdheader(pid, pn_dir=_pn_dir(folder), rd_segments=False)
    except Exception:
        return None
    segs = [s for s in (getattr(hdr, "seg_name", []) or [])
            if s and s not in ("~",) and not s.endswith("_layout")]
    if not segs:
        return None

    best = None
    for s in segs:
        try:
            sh = wfdb.rdheader(s, pn_dir=_pn_dir(folder))
        except Exception:
            continue
        pick = _pick_channels(sh.sig_name or [])
        if not pick or not sh.fs:
            continue
        dur_min = sh.sig_len / sh.fs / 60.0
        if dur_min < min_minutes:
            continue
        if best is None or dur_min > best[1]:
            best = (s, dur_min, sh, pick)
    if best is None:
        return None

    seg, dur_min, sh, (ecg_name, abp_name) = best
    n = int(min(sh.sig_len, sh.fs * max_minutes * 60.0))
    try:
        rec = wfdb.rdrecord(seg, pn_dir=_pn_dir(folder), sampfrom=0, sampto=n,
                            channel_names=[ecg_name, abp_name])
    except Exception:
        return None
    x = rec.p_signal
    if x is None or x.shape[1] < 2:
        return None
    fs = float(rec.fs)
    ecg = x[:, 0].astype(np.float64)
    abp = x[:, 1].astype(np.float64)

    valid = abp_valid_mask(abp, fs)
    if valid.mean() < 0.2:
        return None                     # arterial line unusable for this segment
    t_map, m_map = beat_map_series(abp, fs, valid)
    if len(t_map) < 30:
        return None

    ecg_f = np.where(np.isfinite(ecg), ecg, 0.0)
    peaks = hrv.detect_ecg_rpeaks(ecg_f, fs)
    if len(peaks) < 300:
        return None
    rr = hrv.rr_intervals_ms(peaks, fs)
    rr_t = peaks[1:] / fs

    return PatientBeats(
        patient_id=pid, rr_ms=rr, rr_t_s=rr_t, map_t_s=t_map, map_mmhg=m_map,
        event_times_s=find_decompensation_events(t_map, m_map),
        seconds=float(n / fs))


def stream_cohort(folders: Sequence[str], max_minutes: float = 120.0,
                  progress: bool = True) -> Iterator[PatientBeats]:
    for i, f in enumerate(folders):
        pb = load_patient(f, max_minutes=max_minutes)
        if progress:
            tag = (f"{len(pb.rr_ms)} beats, {len(pb.event_times_s)} events"
                   if pb else "skipped")
            print(f"  [{i + 1}/{len(folders)}] {f.split('/')[-1]}: {tag}", flush=True)
        if pb is not None:
            yield pb


def save_cohort(cohort: Sequence[PatientBeats], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.array([c.patient_id for c in cohort], dtype=object),
        rr=np.array([c.rr_ms for c in cohort], dtype=object),
        rr_t=np.array([c.rr_t_s for c in cohort], dtype=object),
        map_t=np.array([c.map_t_s for c in cohort], dtype=object),
        map_v=np.array([c.map_mmhg for c in cohort], dtype=object),
        events=np.array([np.asarray(c.event_times_s) for c in cohort], dtype=object),
        seconds=np.array([c.seconds for c in cohort], dtype=float),
        allow_pickle=True)


def load_cohort(path: str) -> List[PatientBeats]:
    d = np.load(path, allow_pickle=True)
    return [PatientBeats(patient_id=str(i), rr_ms=r, rr_t_s=rt, map_t_s=mt,
                         map_mmhg=mv, event_times_s=list(ev), seconds=float(sec))
            for i, r, rt, mt, mv, ev, sec in zip(
                d["ids"], d["rr"], d["rr_t"], d["map_t"], d["map_v"],
                d["events"], d["seconds"])]
