"""What a long run is doing, in numbers only, and what that says about the next one.

The first full MIMIC-IV run sat 6.5 hours without a log line while it fitted, and ran about a
third slower than planned on a throttled two-vCPU machine. Nothing in it said so.

Every fit the checkpoint layer sees becomes one line of ``progress.jsonl`` beside the
checkpoints: when it ended, whether it was resumed or fitted, its seconds, how many rows and
columns it read, and the process's memory. A run's plan writes a ``start`` line first. No value,
label or identifier is ever written: ``record`` accepts only ``ALLOWED`` keys and scalar values,
and refuses anything else.

``summary``      progress and an ETA from the journal, for ``physionet status``
``calibration``  how much slower or faster this machine fitted than the planner's constants,
                 from earlier runs on the SAME machine (threads and RAM), for the next plan

Both count WORK, not fits: a fit's work is rows x columns, the planner's own cost model, in
billions of cells. A run makes hundreds of tiny audit fits (the information-flow audit alone
made ~760 of the demo run's ~780) beside ~80 big ones. Counted as fits, progress overflowed
the plan at once, and a median of per-fit timings read the tiny fits' fixed overhead as a
machine 100x too slow. Ratios of sums let the big fits, which are the time, dominate.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

JOURNAL = "progress.jsonl"
ALLOWED = frozenset({"t", "event", "kind", "hit", "seconds", "rows", "cols", "rss_gb", "swap_gb", "peak_gb",
                     "threads", "ram_gb", "planned_fits", "planned_gcells", "planned_hours", "twin_hours",
                     "time_factor"})
MIN_FITS = 5            # fewer fitted (not resumed) boosters than this calibrate nothing
SAME_RAM = 0.10         # machines within 10% of RAM, with the same threads, count as the same


def memory() -> Dict[str, float]:
    """This process: resident and swapped memory on Linux; the peak elsewhere."""
    try:
        out = {}
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmSwap:")):
                    out["rss_gb" if line.startswith("VmRSS") else "swap_gb"] = int(line.split()[1]) * 1024 / 1e9
        if out:
            return {k: round(v, 3) for k, v in out.items()}
    except OSError:
        pass
    import resource
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_gb": round(peak / 1e6 if sys.platform.startswith("linux") else peak / 1e9, 3)}


def machine() -> Dict[str, float]:
    from backend.ml.mimic import plan as PL
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    return {"threads": PL.threads(), "ram_gb": round(ram, 2)}


def record(directory: str, **entry) -> None:
    """Append one line. Refuses a key outside ``ALLOWED`` or a value that isn't a number, a
    bool or a short word, so nothing derived from a patient can be journaled by mistake."""
    bad = set(entry) - ALLOWED
    if bad:
        raise ValueError(f"progress journal: refusing keys {sorted(bad)}")
    for k, v in entry.items():
        ok = isinstance(v, (bool, int, float, np.integer, np.floating)) or (isinstance(v, str) and len(v) <= 16)
        if not ok:
            raise ValueError(f"progress journal: {k} must be a number or a short word, not {type(v).__name__}")
    line = {"t": round(time.time(), 1), **{k: (v.item() if isinstance(v, np.generic) else v) for k, v in entry.items()}}
    try:
        with open(os.path.join(directory, JOURNAL), "a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
    except OSError:                        # like a checkpoint, a journal never kills the run
        pass


def read(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:             # a line torn by a crash
                continue
    return out


def _runs(entries: List[Dict]) -> List[Dict]:
    """The journal split at each ``start``: [{"start": {...}, "fits": [...]}, ...]."""
    runs: List[Dict] = []
    for e in entries:
        if e.get("event") == "start":
            runs.append({"start": e, "fits": []})
        elif runs and e.get("kind"):
            runs[-1]["fits"].append(e)
    return runs


def _gcells(f: Dict) -> float:
    return f.get("rows", 0) * f.get("cols", 0) / 1e9


def calibration(path: str, threads: int, ram_gb: float) -> Optional[Dict]:
    """``{"time": factor, "fits": n}``: measured over predicted seconds, summed over every booster
    FITTED (not resumed) by an earlier run on this same machine, so the big fits dominate. None
    until there are ``MIN_FITS`` of them. Predicted is the planner's model for those rows x columns."""
    from backend.ml.mimic import plan as PL
    secs = pred = 0.0
    n = 0
    for r in _runs(read(path)):
        s = r["start"]
        if s.get("threads") != threads or abs(s.get("ram_gb", 0) - ram_gb) > SAME_RAM * ram_gb:
            continue
        for f in r["fits"]:
            if f.get("kind") == "lgb" and not f.get("hit") and f.get("rows") and f.get("cols") and f.get("seconds"):
                secs += f["seconds"]
                pred += PL.SEC_PER_ROW_COL * f["rows"] * f["cols"] * PL.CALIBRATION_THREADS / max(threads, 1)
                n += 1
    if n < MIN_FITS or pred <= 0:
        return None
    return {"time": secs / pred, "fits": n}


def summary(path: str, now: Optional[float] = None) -> Optional[Dict]:
    """The latest run: work done of planned, the fitted rate, memory, and an ETA."""
    runs = _runs(read(path))
    if not runs:
        return None
    now = time.time() if now is None else now
    r = runs[-1]
    s, fits = r["start"], r["fits"]
    lgb = [f for f in fits if f.get("kind") == "lgb"]
    fitted = [f for f in lgb if not f.get("hit")]
    done = sum(_gcells(f) for f in lgb)                  # resumed work is done work
    fitted_cells = sum(_gcells(f) for f in fitted)
    rate = sum(f["seconds"] for f in fitted) / fitted_cells if fitted_cells > 0 else None   # s per Gcell
    planned = float(s.get("planned_gcells", 0.0))
    twin_done = any(f.get("kind") == "twin" for f in fits)
    eta_h = None
    if rate is not None:
        eta_h = max(planned - done, 0.0) * rate / 3600 + (0.0 if twin_done else float(s.get("twin_hours", 0.0)))
    last = fits[-1] if fits else s
    mem = {k: last[k] for k in ("rss_gb", "swap_gb", "peak_gb") if k in last}
    return {"fits": len(lgb), "resumed": sum(1 for f in lgb if f.get("hit")), "done_gcells": done,
            "planned_gcells": planned, "share": done / planned if planned > 0 else None, "s_per_gcell": rate,
            "since_last_s": now - last["t"], "elapsed_h": (now - s["t"]) / 3600, "twin_done": twin_done,
            "eta_h": eta_h, "eta_t": None if eta_h is None else now + eta_h * 3600, **mem}
