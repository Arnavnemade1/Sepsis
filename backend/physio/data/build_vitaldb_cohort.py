"""Build the VitalDB decompensation cohort — discovery phase.

    PYTHONPATH=. python -m backend.physio.data.build_vitaldb_cohort [--limit N]

Loads each case's numerics (arterial mean + both anaesthetic effect-site
concentrations, 2 s grid) and records its decompensation onsets, split into
spontaneous and anaesthetic-driven, at TWO sustain definitions.

PRE-DECLARED, BEFORE ANY OUTCOME ASSOCIATION WAS COMPUTED:

  * **primary = 5 minutes** — byte-identical to the MIMIC-III arm, so the two
    investigations answer the same question and the results are comparable.
    Projected ~151 spontaneous events over the cohort, five times the MIMIC cohort.
  * **secondary = 3 minutes** — the intraoperative-hypotension literature commonly uses
    1-5 minutes, because an anaesthetist is present and treats promptly, so sustained
    episodes are rarer here than in an ICU. Projected ~453 events. This is EXPLORATORY:
    it was added after observing that the 5-minute yield was thin, and it is reported as
    a secondary rather than promoted if it happens to be the one that wins.

Both are stored, so neither can be chosen after the fact without that choice being
visible in the diff.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

from backend.physio.data import netguard, vitaldb_open as V

OUT_DIR = os.path.join("artifacts", "vitaldb")
DISCOVERY = os.path.join(OUT_DIR, "discovery.json")
SUSTAINS = {"primary_5min": 5.0, "secondary_3min": 3.0}


def _events_at(o, sustain_min: float):
    import backend.physio.data.vitaldb_open as VV
    old = VV.EVENT_MIN_MINUTES
    VV.EVENT_MIN_MINUTES = sustain_min
    try:
        sp, dr = VV.find_events(o.t_s, o.map_mmhg, o.ppf_ce, o.rftn_ce)
    finally:
        VV.EVENT_MIN_MINUTES = old
    return sp, dr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    state: Dict = {"cases": {}}
    if os.path.exists(DISCOVERY):
        with open(DISCOVERY) as fh:
            state = json.load(fh)
        print(f"resuming: {len(state['cases'])} cases already discovered")

    ids = V.cohort(require_ce=True)
    print(f"cohort: {len(ids)} cases with ECG + arterial + both effect-site tracks")
    if args.limit:
        ids = ids[:args.limit]

    t0 = time.time()
    for k, cid in enumerate(ids):
        if str(cid) in state["cases"]:
            continue
        try:
            with netguard.deadline(120):
                o = V.discover(int(cid))
        except netguard.Timeout:
            print(f"  case {cid}: TIMEOUT — skipped and marked done", flush=True)
            state["cases"][str(cid)] = {"hours": 0.0, "skip": True, "timeout": True}
            with open(DISCOVERY, "w") as fh:
                json.dump(state, fh)
            continue
        if o is None:
            state["cases"][str(cid)] = {"hours": 0.0, "skip": True}
        else:
            rec = {"hours": o.hours, "skip": False}
            for tag, s in SUSTAINS.items():
                sp, dr = _events_at(o, s)
                rec[f"{tag}_spont"] = sp
                rec[f"{tag}_drug"] = dr
            rec["controls"] = V.control_times(o)
            state["cases"][str(cid)] = rec
            if rec["primary_5min_spont"] or rec["secondary_3min_spont"]:
                print(f"  case {cid}: {o.hours:4.1f}h  5min {len(rec['primary_5min_spont'])}sp/"
                      f"{len(rec['primary_5min_drug'])}dr  3min "
                      f"{len(rec['secondary_3min_spont'])}sp/{len(rec['secondary_3min_drug'])}dr",
                      flush=True)
        if k % 20 == 0:
            with open(DISCOVERY, "w") as fh:
                json.dump(state, fh)
            ok = [v for v in state["cases"].values() if not v.get("skip")]
            n5 = sum(len(v.get("primary_5min_spont", [])) for v in ok)
            n3 = sum(len(v.get("secondary_3min_spont", [])) for v in ok)
            d5 = sum(len(v.get("primary_5min_drug", [])) for v in ok)
            print(f"  [{len(state['cases'])}/{len(ids)}] {sum(v['hours'] for v in ok):.0f}h  "
                  f"5min {n5}sp/{d5}dr  3min {n3}sp  [{time.time()-t0:.0f}s]", flush=True)

    with open(DISCOVERY, "w") as fh:
        json.dump(state, fh)
    ok = [v for v in state["cases"].values() if not v.get("skip")]
    n5 = sum(len(v.get("primary_5min_spont", [])) for v in ok)
    d5 = sum(len(v.get("primary_5min_drug", [])) for v in ok)
    n3 = sum(len(v.get("secondary_3min_spont", [])) for v in ok)
    d3 = sum(len(v.get("secondary_3min_drug", [])) for v in ok)
    print(f"\nDISCOVERY DONE: {len(ok)} usable cases, {sum(v['hours'] for v in ok):.0f} hours")
    print(f"  primary   (5 min): {n5} spontaneous, {d5} anaesthetic-driven "
          f"({d5/max(n5+d5,1):.0%} drug-explained) across "
          f"{sum(1 for v in ok if v.get('primary_5min_spont'))} cases")
    print(f"  secondary (3 min): {n3} spontaneous, {d3} anaesthetic-driven "
          f"across {sum(1 for v in ok if v.get('secondary_3min_spont'))} cases")


if __name__ == "__main__":
    main()
