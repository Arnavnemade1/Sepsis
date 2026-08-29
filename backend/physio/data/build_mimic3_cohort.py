"""Build the open-access MIMIC-III decompensation cohort.

    PYTHONPATH=. python -m backend.physio.data.build_mimic3_cohort --target-events 150

Two phases, because they cost very different amounts:

  1. DISCOVERY — for each candidate patient, read the tiny ``_layout`` header to check
     for an arterial line, then the one-per-minute numerics record covering the whole
     admission, and find circulatory decompensation onsets. Cheap, and it sees 100% of
     each stay instead of the first 2 hours.
  2. RETRIEVAL — for each event and each matched control window, fetch ONLY the minutes
     of 125 Hz ECG needed, and reduce them to an RR series on the spot. Raw waveforms
     are never stored.

Checkpoints after every patient, so a run that is interrupted resumes rather than
restarts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

from backend.physio.data import mimic3_open as M

OUT_DIR = os.path.join("artifacts", "mimic3_open")
DISCOVERY = os.path.join(OUT_DIR, "discovery.json")


def _has_abp(folder: str) -> bool:
    import wfdb
    pid = folder.strip("/").split("/")[-1]
    try:
        h = wfdb.rdheader(f"{pid}_layout", pn_dir=M._pn_dir(folder))
        return any(n.upper() in M.ABP_NAMES for n in (h.sig_name or []))
    except Exception:
        return False


def discover(target_events: int, max_folders: int, seed: int = 11) -> Dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    state: Dict = {"patients": {}, "scanned": 0}
    if os.path.exists(DISCOVERY):
        with open(DISCOVERY) as fh:
            state = json.load(fh)
        print(f"resuming: {len(state['patients'])} patients, "
              f"{sum(len(v['events']) for v in state['patients'].values())} events")

    folders = M.list_patient_folders()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(folders))
    t0 = time.time()
    for k in order[state["scanned"]:state["scanned"] + max_folders]:
        f = folders[int(k)]
        state["scanned"] += 1
        pid = f.strip("/").split("/")[-1]
        if pid in state["patients"]:
            continue
        if not _has_abp(f):
            continue
        try:
            o = M.discover_outcomes(f)
        except Exception:
            o = None
        if o is None:
            continue
        state["patients"][pid] = {
            "folder": o.folder, "hours": o.stay_hours,
            "events": [float(e) for e in o.event_times_s],
            "controls": [float(c) for c in M.control_times(o)],
        }
        n_ev = sum(len(v["events"]) for v in state["patients"].values())
        n_pt = len(state["patients"])
        hrs = sum(v["hours"] for v in state["patients"].values())
        print(f"  [{n_pt:4d}p {n_ev:4d}ev {hrs:7.0f}h] {pid}: {o.stay_hours:6.1f}h "
              f"{len(o.event_times_s)} events  (scanned {state['scanned']}, "
              f"{time.time()-t0:.0f}s)", flush=True)
        with open(DISCOVERY, "w") as fh:
            json.dump(state, fh)
        if n_ev >= target_events:
            print(f"\nreached {n_ev} events")
            break
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-events", type=int, default=150)
    ap.add_argument("--max-folders", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    st = discover(args.target_events, args.max_folders, args.seed)
    n_ev = sum(len(v["events"]) for v in st["patients"].values())
    n_ct = sum(len(v["controls"]) for v in st["patients"].values())
    hrs = sum(v["hours"] for v in st["patients"].values())
    withev = sum(1 for v in st["patients"].values() if v["events"])
    print(f"\nDISCOVERY DONE: {len(st['patients'])} patients / {hrs:.0f} patient-hours")
    print(f"  {n_ev} decompensation events across {withev} patients")
    print(f"  {n_ct} control windows")
    print(f"  scanned {st['scanned']} folders -> {DISCOVERY}")


if __name__ == "__main__":
    main()
