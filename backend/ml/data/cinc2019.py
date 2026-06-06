"""Download and parse the PhysioNet/CinC 2019 sepsis dataset.

The dataset is published as one pipe-delimited ``.psv`` per patient across two
training sets (~40k patients total). There is no official single-file archive,
so we mirror the directory index and fetch files in parallel with resume
support. A ``limit`` argument lets you stand up the full pipeline on a small
subset in seconds before committing to the multi-GB full download.

Each patient file is hourly: one row per ICU hour, columns defined in
``config.CINC_COLUMNS``. We parse into a tidy per-patient object that keeps the
raw hourly matrix, the label vector, and the derived true sepsis-onset hour.
"""
from __future__ import annotations

import os
import re
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.ml import config

_INDEX_RE = re.compile(r'href="(p\d+\.psv)"')
_USER_AGENT = "HyperSepsisTwin-Omega/1.0 (research; PhysioNet CinC2019 loader)"


# --------------------------------------------------------------------------- #
# Patient container
# --------------------------------------------------------------------------- #
@dataclass
class PatientRecord:
    """One patient's hourly ICU stay."""

    patient_id: str
    set_name: str
    frame: pd.DataFrame          # hourly rows, all CINC_COLUMNS
    labels: np.ndarray           # SepsisLabel per hour (already lead-shifted)

    @property
    def n_hours(self) -> int:
        return len(self.frame)

    @property
    def is_septic(self) -> bool:
        return bool(self.labels.any())

    @property
    def label_onset_hour(self) -> Optional[int]:
        """First hour where the (lead-shifted) label flips to 1, else None."""
        idx = np.flatnonzero(self.labels == 1)
        return int(idx[0]) if idx.size else None

    @property
    def true_onset_hour(self) -> Optional[int]:
        """Estimated clinical onset = labelled onset + the 6h lead the challenge
        bakes into the label. Used only for honest lead-time reporting."""
        lo = self.label_onset_hour
        return None if lo is None else lo + config.SEPSIS_LABEL_LEAD_HOURS


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: float = 30.0, retries: int = 4) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to GET {url}: {last_err}")


def list_remote_files(set_name: str) -> List[str]:
    """Scrape the PhysioNet directory index for a training set's patient files."""
    url = f"{config.CINC_BASE_URL}/{set_name}/"
    html = _http_get(url).decode("utf-8", errors="replace")
    files = sorted(set(_INDEX_RE.findall(html)))
    if not files:
        raise RuntimeError(f"No patient files found at {url} (index format change?)")
    return files


def download_set(
    set_name: str,
    limit: Optional[int] = None,
    workers: int = 12,
    verbose: bool = True,
) -> int:
    """Download a training set into ``DATA_DIR/set_name`` with resume support.

    Returns the number of files newly downloaded. Existing files are skipped so
    re-running is cheap and the full download can be resumed after interruption.
    """
    dest_dir = os.path.join(config.DATA_DIR, set_name)
    os.makedirs(dest_dir, exist_ok=True)

    files = list_remote_files(set_name)
    if limit is not None:
        files = files[:limit]

    pending = [f for f in files if not _valid_local(os.path.join(dest_dir, f))]
    if verbose:
        print(f"[{set_name}] {len(files)} target files, {len(pending)} to fetch.")

    downloaded = 0

    def _fetch(fname: str) -> Tuple[str, bool]:
        url = f"{config.CINC_BASE_URL}/{set_name}/{fname}"
        try:
            data = _http_get(url)
            with open(os.path.join(dest_dir, fname), "wb") as fh:
                fh.write(data)
            return fname, True
        except Exception as exc:  # pragma: no cover
            print(f"  ! {fname}: {exc}")
            return fname, False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, f): f for f in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            _, ok = fut.result()
            downloaded += int(ok)
            if verbose and (i % 500 == 0 or i == len(pending)):
                print(f"  [{set_name}] {i}/{len(pending)} fetched")

    return downloaded


def download_all(limit_per_set: Optional[int] = None, workers: int = 12) -> None:
    for set_name in config.CINC_SETS:
        download_set(set_name, limit=limit_per_set, workers=workers)


def _valid_local(path: str) -> bool:
    """A file counts as present only if it is non-empty and has the header."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    return True


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #
def _parse_psv(path: str, patient_id: str, set_name: str) -> Optional[PatientRecord]:
    try:
        frame = pd.read_csv(path, sep="|")
    except Exception:
        return None
    if frame.empty or config.LABEL_COLUMN not in frame.columns:
        return None
    # Guard against partially-written files from an interrupted download.
    missing = [c for c in config.CINC_COLUMNS if c not in frame.columns]
    if missing:
        return None
    labels = frame[config.LABEL_COLUMN].fillna(0).to_numpy(dtype=np.int8)
    return PatientRecord(patient_id, set_name, frame.reset_index(drop=True), labels)


def iter_patients(
    sets: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> Iterable[PatientRecord]:
    """Yield parsed ``PatientRecord``s from the local cache.

    ``limit`` caps the total number of patients across all sets (handy for fast
    development iterations).
    """
    sets = list(sets) if sets is not None else list(config.CINC_SETS)
    count = 0
    for set_name in sets:
        set_dir = os.path.join(config.DATA_DIR, set_name)
        if not os.path.isdir(set_dir):
            continue
        for fname in sorted(os.listdir(set_dir)):
            if not fname.endswith(".psv"):
                continue
            if limit is not None and count >= limit:
                return
            pid = fname[:-4]
            rec = _parse_psv(os.path.join(set_dir, fname), pid, set_name)
            if rec is not None:
                count += 1
                yield rec


def load_patients(
    sets: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> List[PatientRecord]:
    return list(iter_patients(sets=sets, limit=limit))


def dataset_summary(records: List[PatientRecord]) -> Dict[str, float]:
    """Quick sanity statistics — prevalence is the number you must know before
    interpreting any accuracy metric on this dataset."""
    n = len(records)
    if n == 0:
        return {"n_patients": 0}
    septic = sum(r.is_septic for r in records)
    total_hours = sum(r.n_hours for r in records)
    septic_hours = sum(int(r.labels.sum()) for r in records)
    return {
        "n_patients": n,
        "septic_patients": septic,
        "patient_prevalence": round(septic / n, 4),
        "total_hours": total_hours,
        "septic_hours": septic_hours,
        "hour_prevalence": round(septic_hours / max(total_hours, 1), 4),
        "median_stay_hours": float(np.median([r.n_hours for r in records])),
    }


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Download/inspect CinC 2019 data.")
    ap.add_argument("--download", action="store_true", help="download the data")
    ap.add_argument("--limit-per-set", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--summary", action="store_true", help="print cache summary")
    args = ap.parse_args()

    if args.download:
        download_all(limit_per_set=args.limit_per_set, workers=args.workers)
    if args.summary:
        recs = load_patients()
        print(dataset_summary(recs))
