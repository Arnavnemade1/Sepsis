"""Phase 2 — retrieve the ECG windows the discovered outcomes point at.

    PYTHONPATH=. python -m backend.physio.data.fetch_mimic3_windows [--max-per-patient 3]

Discovery (``build_mimic3_cohort.py``) reads the one-per-minute numerics record across a
whole admission and returns the times of circulatory decompensation onsets plus matched
stable control windows. This phase turns each of those times into a feature row: it
fetches the 125 Hz ECG for the 90 minutes *before* the time, reduces it to an RR series,
and computes recovery + HRV features over windows ending at several lead times.

ONE DOWNLOAD PER TIMEPOINT, several lead times out of it. A 90-minute single-channel
pull is about eight seconds; doing it once per lead time instead would triple a job
already measured in hours for no new information.

BEAT-DETECTION QC, WHICH IS NOT OPTIONAL HERE
---------------------------------------------
The estimator in ``recovery.py`` measures the recovery from RR *deviations*. A missed
beat doubles an interval and a doubled beat halves one, so a detection error is
indistinguishable from an enormous physiologic perturbation — precisely the quantity
being measured. Left unchecked it would fill the dataset with artifact events.

The monitor's own ``HR`` numerics channel is an independent estimate of the same
quantity, produced by different hardware from the same electrodes, and it is free. So
every window is checked against it and dropped when the two disagree. Building this
found a real instance immediately: one patient's ECG-derived rate read 105.4 bpm where
the monitor said 76.0 — T-wave over-detection — while the same patient agreed within
2 bpm at three other times.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.physio import hrv, recovery
from backend.physio.data import mimic3_open as M

OUT_DIR = os.path.join("artifacts", "mimic3_open")
DISCOVERY = os.path.join(OUT_DIR, "discovery.json")
WINDOWS = os.path.join(OUT_DIR, "windows.npz")
PROGRESS = os.path.join(OUT_DIR, "fetch_progress.json")

#: Lead times, in minutes before the outcome, at which features are computed. The whole
#: point of the exercise is early warning, so the result is reported per lead rather than
#: at one flattering horizon.
LEADS_MIN: Tuple[float, ...] = (15.0, 30.0, 60.0)

#: Feature window length. Ten minutes at an adult ICU rate is 600-900 beats, comfortably
#: above the ~15 perturbations the power analysis in ``verify_recovery`` says a stable
#: time constant needs.
WINDOW_MIN = 10.0

#: Fetch span before the timepoint: enough to cover the longest lead plus its window.
FETCH_MIN = 75.0

#: Beat detection is rejected when it disagrees with the monitor's own HR by more than
#: this fraction. 10% is far looser than the 1.5 bpm median agreement measured across
#: aligned windows, so it removes detection failures without trimming real physiology.
HR_QC_TOLERANCE = 0.10


def monitor_hr(folder: str, pid: str) -> Optional[Tuple[np.ndarray, float]]:
    import wfdb
    try:
        rec = wfdb.rdrecord(f"{pid}n", pn_dir=M._pn_dir(folder))
    except Exception:
        return None
    names = [n.strip().upper() for n in (rec.sig_name or [])]
    if "HR" not in names:
        return None
    return rec.p_signal[:, names.index("HR")].astype(np.float64), float(rec.fs)


def _hr_qc(rr_ms: np.ndarray, hr_track: np.ndarray, fs_n: float,
           t0_s: float, t1_s: float) -> Tuple[bool, float, float]:
    """Does ECG-derived rate agree with the monitor over the same span?"""
    nn = hrv.clean_nn(np.asarray(rr_ms, dtype=np.float64))
    if len(nn) < 30:
        return False, float("nan"), float("nan")
    ecg_hr = 60000.0 / float(np.median(nn))
    a, b = int(t0_s * fs_n), int(t1_s * fs_n)
    seg = hr_track[max(a, 0):max(b, 1)]
    seg = seg[np.isfinite(seg)]
    if seg.size < 3:
        return False, ecg_hr, float("nan")
    mon = float(np.median(seg))
    ok = mon > 0 and abs(ecg_hr - mon) / mon <= HR_QC_TOLERANCE
    return bool(ok), ecg_hr, mon


def features_for(rr_ms: np.ndarray, rr_t_s: np.ndarray, end_s: float
                 ) -> Optional[np.ndarray]:
    """Recovery + HRV features over the WINDOW_MIN minutes ending at ``end_s``."""
    m = (rr_t_s >= end_s - WINDOW_MIN * 60.0) & (rr_t_s <= end_s)
    win = rr_ms[m]
    if len(win) < 256:
        return None
    rf = recovery.recovery_features(win)
    if rf.quality <= 0.0:
        return None
    hf = hrv.hrv_features(win)
    if hf.quality <= 0.0:
        return None
    from backend.physio.sepsis_hrv import HRV_FEATURE_NAMES
    d = hf.as_dict()
    return np.concatenate([recovery.feature_vector(rf),
                           np.array([d[k] for k in HRV_FEATURE_NAMES], dtype=float)])


def feature_names() -> List[str]:
    from backend.physio.sepsis_hrv import HRV_FEATURE_NAMES
    return list(recovery.RECOVERY_FEATURE_NAMES) + list(HRV_FEATURE_NAMES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-patient", type=int, default=3,
                    help="cap events and controls per patient so one long stay cannot "
                         "dominate the cohort")
    ap.add_argument("--limit", type=int, default=0, help="stop after N timepoints")
    ap.add_argument("--dual-only", action="store_true",
                    help="keep only patients contributing BOTH an event and a control. "
                         "Controls drawn from patients who never decompensated would "
                         "reintroduce between-patient contrast, which is precisely the "
                         "thing every previous result in this project turned out to be.")
    args = ap.parse_args()

    with open(DISCOVERY) as fh:
        disc = json.load(fh)
    done: Dict[str, bool] = {}
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as fh:
            done = json.load(fh)

    rows: List[np.ndarray] = []
    meta: List[Tuple[str, float, int, float]] = []      # pid, lead, label, timepoint
    if os.path.exists(WINDOWS):
        d = np.load(WINDOWS, allow_pickle=True)
        rows = list(d["X"])
        meta = [tuple(m) for m in d["meta"]]
        print(f"resuming with {len(rows)} rows")

    # Patients that actually had an event are fetched first, and within a patient the
    # events precede their own controls. A run measured in hours will be interrupted, and
    # this ordering means a partial cohort is still a usable case-control study rather
    # than a pile of negatives — the first test run fetched six timepoints and every one
    # of them was a control.
    todo: List[Tuple[str, Dict, float, int]] = []
    items = list(disc["patients"].items())
    if args.dual_only:
        items = [kv for kv in items if kv[1]["events"] and kv[1]["controls"]]
        print(f"dual-only: {len(items)} patients contribute both an event and a control")
    ordered = sorted(items, key=lambda kv: (0 if kv[1]["events"] else 1, kv[0]))
    for pid, v in ordered:
        for t in v["events"][:args.max_per_patient]:
            todo.append((pid, v, float(t), 1))
        for t in v["controls"][:args.max_per_patient]:
            todo.append((pid, v, float(t), 0))
    print(f"{len(todo)} timepoints "
          f"({sum(1 for x in todo if x[3] == 1)} events, "
          f"{sum(1 for x in todo if x[3] == 0)} controls) from "
          f"{len(disc['patients'])} patients")

    hr_cache: Dict[str, Optional[Tuple[np.ndarray, float]]] = {}
    t0 = time.time()
    n_new = n_qc = n_rhythm = 0
    for i, (pid, v, T, label) in enumerate(todo):
        key = f"{pid}:{T:.0f}:{label}"
        if done.get(key):
            continue
        if args.limit and n_new >= args.limit:
            break
        folder = v["folder"]
        start = max(T - FETCH_MIN * 60.0, 0.0)
        rr = M.fetch_rr_window(folder, start, FETCH_MIN * 60.0)
        done[key] = True
        if rr is None or len(rr) < 300:
            continue
        rr_t = start + np.cumsum(rr) / 1000.0

        if pid not in hr_cache:
            hr_cache[pid] = monitor_hr(folder, pid)
        track = hr_cache[pid]
        if track is not None:
            ok, ehr, mhr = _hr_qc(rr, track[0], track[1], start, T)
            if not ok:
                n_qc += 1
                print(f"  [{i+1}/{len(todo)}] {pid} t={T/3600:.1f}h QC REJECT "
                      f"ecg {ehr:.0f} vs monitor {mhr:.0f} bpm", flush=True)
                continue

        # Is there an autonomic control loop here at all? A paced, fibrillating or
        # heavily ectopic rhythm makes a recovery time constant meaningless rather than
        # merely noisy, and the monitor already classifies all three.
        rok, why = M.rhythm_ok_from_numerics(folder, pid, start, T)
        if rok is None:
            rok, why = M.rhythm_ok_from_rr(rr)
        if not rok:
            n_rhythm += 1
            print(f"  [{i+1}/{len(todo)}] {pid} t={T/3600:.1f}h RHYTHM REJECT ({why})",
                  flush=True)
            continue

        added = 0
        for lead in LEADS_MIN:
            vec = features_for(rr, rr_t, T - lead * 60.0)
            if vec is None:
                continue
            rows.append(vec)
            meta.append((pid, float(lead), int(label), float(T)))
            added += 1
        n_new += 1
        if added:
            print(f"  [{i+1}/{len(todo)}] {pid} t={T/3600:.1f}h "
                  f"{'EVENT' if label else 'control'}: {added} leads "
                  f"({len(rows)} rows, {time.time()-t0:.0f}s)", flush=True)
        if n_new % 10 == 0:
            np.savez_compressed(WINDOWS, X=np.array(rows, dtype=float),
                                meta=np.array(meta, dtype=object),
                                names=np.array(feature_names(), dtype=object))
            with open(PROGRESS, "w") as fh:
                json.dump(done, fh)

    np.savez_compressed(WINDOWS, X=np.array(rows, dtype=float),
                        meta=np.array(meta, dtype=object),
                        names=np.array(feature_names(), dtype=object))
    with open(PROGRESS, "w") as fh:
        json.dump(done, fh)
    lab = np.array([m[2] for m in meta]) if meta else np.array([])
    print(f"\nDONE {len(rows)} feature rows from {n_new} timepoints "
          f"({n_qc} rejected by HR QC, {n_rhythm} by rhythm)")
    if lab.size:
        print(f"  event rows {int((lab == 1).sum())}, control rows {int((lab == 0).sum())}, "
              f"{len(set(m[0] for m in meta))} patients -> {WINDOWS}")


if __name__ == "__main__":
    main()
