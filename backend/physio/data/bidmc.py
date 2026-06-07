"""BIDMC PPG and Respiration dataset (PhysioNet, open access).

53 eight-minute ICU recordings with SIMULTANEOUS photoplethysmogram (PLETH),
ECG (lead II), and respiration, sampled at 125 Hz. Because PPG and gold-standard
ECG are recorded at the same time on the same patient, this is the ideal open
dataset to validate the pulse->HRV chain: HRV computed from the PPG (what a
camera/wearable sees) can be compared directly against HRV from the ECG.

Fetched over plain HTTP from the published CSV export (no credentialing).
"""
from __future__ import annotations

import io
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

BIDMC_FS = 125.0
_BASE = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv"
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "bidmc",
)
_UA = "HyperSepsisTwin-Omega/1.0 (research; BIDMC loader)"


@dataclass
class BidmcRecord:
    record_id: str
    fs: float
    ppg: np.ndarray          # PLETH
    ecg: np.ndarray          # ECG lead II
    resp: np.ndarray         # impedance respiration
    duration_s: float


def _http_get(url: str, timeout: float = 30.0, retries: int = 3) -> bytes:
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # pragma: no cover
            last = exc
    raise RuntimeError(f"GET {url} failed: {last}")


def download(n_records: int = 20, verbose: bool = True) -> int:
    """Download the first ``n_records`` BIDMC signal CSVs (resumable)."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    got = 0
    for i in range(1, n_records + 1):
        rid = f"{i:02d}"
        fname = f"bidmc_{rid}_Signals.csv"
        dest = os.path.join(_DATA_DIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        try:
            data = _http_get(f"{_BASE}/{fname}")
            with open(dest, "wb") as fh:
                fh.write(data)
            got += 1
            if verbose:
                print(f"  [bidmc] fetched {fname}")
        except Exception as exc:  # pragma: no cover
            print(f"  [bidmc] skip {fname}: {exc}")
    return got


def _parse(path: str, rid: str) -> Optional[BidmcRecord]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    need = {"PLETH", "II", "RESP"}
    if not need.issubset(df.columns):
        return None
    n = len(df)
    return BidmcRecord(
        record_id=rid, fs=BIDMC_FS,
        ppg=df["PLETH"].to_numpy(dtype=np.float64),
        ecg=df["II"].to_numpy(dtype=np.float64),
        resp=df["RESP"].to_numpy(dtype=np.float64),
        duration_s=n / BIDMC_FS,
    )


def load_records(limit: Optional[int] = None) -> List[BidmcRecord]:
    if not os.path.isdir(_DATA_DIR):
        return []
    recs = []
    for fname in sorted(os.listdir(_DATA_DIR)):
        if not fname.endswith("_Signals.csv"):
            continue
        if limit is not None and len(recs) >= limit:
            break
        rid = fname.split("_")[1]
        rec = _parse(os.path.join(_DATA_DIR, fname), rid)
        if rec is not None:
            recs.append(rec)
    return recs


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    if args.download:
        download(args.n)
    print(f"local BIDMC records: {len(load_records())}")
