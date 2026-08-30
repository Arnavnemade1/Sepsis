"""Phase 2 — turn VitalDB decompensation times into feature rows.

    PYTHONPATH=. python -m backend.physio.data.fetch_vitaldb_windows [--definition primary_5min]

CASE-MAJOR, which is the whole design. ``vitaldb.load_case`` returns a whole track, and a
three-hour 500 Hz ECG is about three seconds over the wire, so a case's ECG is fetched
**once** and every window that case contributes — each event, each control, each lead
time — is sliced out of that single array. The MIMIC arm fetched per window over HTTP
range reads and spent hours doing it; this is minutes.

WHAT A ROW IS
-------------
One (case, timepoint, lead) triple. Features come from the ten minutes of beat timing
ending ``lead`` minutes before the timepoint, and nothing else: recovery time constants
plus the HeRO-style HRV magnitude set, so the two can be compared head to head on
identical windows.

THE QC THAT DECIDES WHETHER ROWS ARE REAL
-----------------------------------------
``recovery.py`` measures the return from RR *deviations*, so a missed beat or a doubled
beat is indistinguishable from an enormous physiologic perturbation. Two gates, both
carried over from the MIMIC arm where they earned their place:

* **rhythm** — atrial fibrillation has no operating point to return to, so a recovery
  time constant is not a physiologic quantity there. Rejected on pNN50.
* **plausibility** — a window whose derived rate is outside a survivable range is a
  detection failure, not a patient. VitalDB has no independent monitor-HR track to check
  against the way MIMIC's numerics did, so this gate is the weaker of the two and is
  deliberately conservative.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.physio import hrv, recovery
from backend.physio.data import netguard, vitaldb_open as V

OUT_DIR = os.path.join("artifacts", "vitaldb")
DISCOVERY = os.path.join(OUT_DIR, "discovery.json")

LEADS_MIN: Tuple[float, ...] = (15.0, 30.0, 60.0)
WINDOW_MIN = 10.0
MIN_BEATS = 256

#: Derived heart rate outside this range is a beat-detection failure, not physiology.
HR_PLAUSIBLE = (30.0, 200.0)


def features_for(rr_ms: np.ndarray, rr_t_s: np.ndarray, end_s: float
                 ) -> Optional[np.ndarray]:
    m = (rr_t_s >= end_s - WINDOW_MIN * 60.0) & (rr_t_s <= end_s)
    win = rr_ms[m]
    if len(win) < MIN_BEATS:
        return None
    nn = hrv.clean_nn(np.asarray(win, dtype=np.float64))
    if len(nn) < 60:
        return None
    bpm = 60000.0 / float(np.median(nn))
    if not (HR_PLAUSIBLE[0] <= bpm <= HR_PLAUSIBLE[1]):
        return None
    ok, _ = V.rhythm_ok(win)
    if not ok:
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
    ap.add_argument("--definition", default="primary_5min",
                    choices=["primary_5min", "secondary_3min"])
    ap.add_argument("--max-per-case", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    windows = os.path.join(OUT_DIR, f"windows_{args.definition}.npz")
    progress = os.path.join(OUT_DIR, f"fetch_{args.definition}.json")

    with open(DISCOVERY) as fh:
        disc = json.load(fh)
    done: Dict[str, bool] = {}
    if os.path.exists(progress):
        with open(progress) as fh:
            done = json.load(fh)

    rows: List[np.ndarray] = []
    meta: List[Tuple[str, float, int, float]] = []
    if os.path.exists(windows):
        d = np.load(windows, allow_pickle=True)
        rows = list(d["X"]); meta = [tuple(m) for m in d["meta"]]
        print(f"resuming with {len(rows)} rows")

    # Only cases contributing BOTH an event and a control can serve the within-patient
    # design; a control from a case that never decompensated reintroduces between-case
    # contrast, which is the failure mode this whole line of work exists to escape.
    key_ev = f"{args.definition}_spont"
    cases = [(cid, v) for cid, v in disc["cases"].items()
             if not v.get("skip") and v.get(key_ev) and v.get("controls")]
    n_ev = sum(len(v[key_ev][:args.max_per_case]) for _, v in cases)
    n_ct = sum(len(v["controls"][:args.max_per_case]) for _, v in cases)
    print(f"definition={args.definition}: {len(cases)} dual-label cases, "
          f"{n_ev} events, {n_ct} controls")
    if args.limit:
        cases = cases[:args.limit]

    t0 = time.time()
    n_case = 0
    for ci, (cid, v) in enumerate(cases):
        if done.get(cid):
            continue
        pts = ([(float(t), 1) for t in v[key_ev][:args.max_per_case]]
               + [(float(t), 0) for t in v["controls"][:args.max_per_case]])
        span_lo = min(t for t, _ in pts) - (max(LEADS_MIN) + WINDOW_MIN) * 60.0
        span_hi = max(t for t, _ in pts)
        done[cid] = True
        try:
            with netguard.deadline(240):
                got = V.fetch_rr(int(cid), max(span_lo, 0.0), span_hi)
        except netguard.Timeout:
            print(f"  [{ci+1}/{len(cases)}] case {cid}: TIMEOUT — skipped", flush=True)
            with open(progress, "w") as fh:
                json.dump(done, fh)
            continue
        if got is None:
            continue
        rr, rr_t = got
        added = 0
        for T, label in pts:
            for lead in LEADS_MIN:
                vec = features_for(rr, rr_t, T - lead * 60.0)
                if vec is None:
                    continue
                rows.append(vec)
                meta.append((str(cid), float(lead), int(label), float(T)))
                added += 1
        n_case += 1
        print(f"  [{ci+1}/{len(cases)}] case {cid}: {added} rows "
              f"({len(rows)} total, {time.time()-t0:.0f}s)", flush=True)
        if n_case % 10 == 0:
            np.savez_compressed(windows, X=np.array(rows, dtype=float),
                                meta=np.array(meta, dtype=object),
                                names=np.array(feature_names(), dtype=object))
            with open(progress, "w") as fh:
                json.dump(done, fh)

    np.savez_compressed(windows, X=np.array(rows, dtype=float),
                        meta=np.array(meta, dtype=object),
                        names=np.array(feature_names(), dtype=object))
    with open(progress, "w") as fh:
        json.dump(done, fh)
    lab = np.array([m[2] for m in meta]) if meta else np.array([])
    cid_all = set(m[0] for m in meta)
    dual = sum(1 for c in cid_all
               if len(set(m[2] for m in meta if m[0] == c)) == 2)
    print(f"\nDONE {len(rows)} rows from {n_case} cases")
    if lab.size:
        print(f"  {int((lab==1).sum())} event rows / {int((lab==0).sum())} control rows")
        print(f"  {len(cid_all)} cases, {dual} contributing BOTH -> {windows}")


if __name__ == "__main__":
    main()
