"""Every CinC 2019 patient in one array.

The existing loader parses one ``.psv`` per patient into a pandas frame, which is
the right shape for serving one bed and the wrong shape for 40,335 patients: the
feature builder then loops in Python per patient. Here the whole cohort is packed
once into a single ``float32`` matrix with segment offsets, so every downstream
operation is a vectorized pass over 1.55M rows.

Rows are hourly and patients are contiguous: patient ``k`` owns rows
``starts[k] : starts[k] + lens[k]``, in the order the files sort (set A then set B).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from backend.ml import config

#: The 34 time-varying channels, in file order.
DYNAMIC: List[str] = config.CINC_COLUMNS[:34]
#: Per-patient context. ``Unit1``/``Unit2`` are one-hot MICU/SICU, missing for ~39% of rows.
STATIC: List[str] = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
VITALS: List[str] = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

CACHE = os.path.join(config.ARTIFACTS_DIR, "feature_cache", "genesis_raw.npz")


def _read_psv(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path) as fh:
        header = fh.readline().strip().split("|")
        rows = [ln.strip().split("|") for ln in fh if ln.strip()]
    arr = np.array([[np.nan if v == "NaN" else float(v) for v in r] for r in rows],
                   dtype=np.float32)
    return header, arr


@dataclass
class RawStore:
    X: np.ndarray          # (N, 40) float32 — DYNAMIC + STATIC, label excluded
    y: np.ndarray          # (N,) int8 challenge label
    lens: np.ndarray       # (P,) rows per patient
    pid: np.ndarray        # (P,) patient id ("p000001")
    site: np.ndarray       # (P,) 0 = set A (BIDMC), 1 = set B (Emory)
    columns: List[str]

    # ------------------------------------------------------------------ #
    @property
    def n_rows(self) -> int:
        return int(len(self.y))

    @property
    def n_patients(self) -> int:
        return int(len(self.lens))

    @property
    def starts(self) -> np.ndarray:
        return np.r_[0, np.cumsum(self.lens)[:-1]].astype(np.int64)

    @property
    def seg(self) -> np.ndarray:
        """Patient index of every row."""
        return np.repeat(np.arange(self.n_patients), self.lens)

    @property
    def row_start(self) -> np.ndarray:
        """First row of the row's own patient."""
        return np.repeat(self.starts, self.lens)

    @property
    def hour(self) -> np.ndarray:
        """0-based hour of the row within its record."""
        return np.arange(self.n_rows, dtype=np.int64) - self.row_start

    def col(self, name: str) -> np.ndarray:
        return self.X[:, self.columns.index(name)]

    def patient_label(self) -> np.ndarray:
        """(P,) 1 if the patient's label ever turns on."""
        return np.maximum.reduceat(self.y, self.starts).astype(np.int8)

    def onset_hour(self) -> np.ndarray:
        """(P,) labelled onset hour (first hour with label 1), -1 if never."""
        first = np.full(self.n_patients, -1, dtype=np.int64)
        pos = np.flatnonzero(self.y == 1)
        seg = self.seg[pos]
        # first positive row per patient: rows are sorted, so take the first occurrence
        uniq, idx = np.unique(seg, return_index=True)
        first[uniq] = pos[idx] - self.starts[uniq]
        return first

    def subset(self, patients: np.ndarray) -> "RawStore":
        """A new store holding only ``patients`` (indices), in the given order."""
        patients = np.asarray(patients, dtype=np.int64)
        rows = np.concatenate([np.arange(self.starts[k], self.starts[k] + self.lens[k])
                               for k in patients]) if len(patients) else np.zeros(0, np.int64)
        return RawStore(X=self.X[rows], y=self.y[rows], lens=self.lens[patients],
                        pid=self.pid[patients], site=self.site[patients],
                        columns=list(self.columns))


def build(workers: int = 6) -> RawStore:
    paths: List[str] = []
    sites: List[int] = []
    for s, name in enumerate(config.CINC_SETS):
        d = os.path.join(config.DATA_DIR, name)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".psv"))
        paths += [os.path.join(d, f) for f in files]
        sites += [s] * len(files)
    if not paths:
        raise FileNotFoundError(f"no CinC files under {config.DATA_DIR}; run the downloader")
    with ProcessPoolExecutor(workers) as ex:
        parsed = list(ex.map(_read_psv, paths, chunksize=200))
    header = parsed[0][0]
    if header != config.CINC_COLUMNS:
        raise ValueError("unexpected CinC header")
    keep = [header.index(c) for c in DYNAMIC + STATIC]
    lab = header.index(config.LABEL_COLUMN)
    good = [i for i, (h, a) in enumerate(parsed) if h == header and len(a)]
    arrs = [parsed[i][1] for i in good]
    full = np.concatenate(arrs)
    return RawStore(
        X=np.ascontiguousarray(full[:, keep]),
        y=np.nan_to_num(full[:, lab]).astype(np.int8),
        lens=np.array([len(a) for a in arrs], dtype=np.int64),
        pid=np.array([os.path.basename(paths[i])[:-4] for i in good]),
        site=np.array([sites[i] for i in good], dtype=np.int8),
        columns=DYNAMIC + STATIC,
    )


def load(refresh: bool = False, workers: int = 6) -> RawStore:
    if os.path.exists(CACHE) and not refresh:
        d = np.load(CACHE, allow_pickle=False)
        return RawStore(X=d["X"], y=d["y"], lens=d["lens"], pid=d["pid"], site=d["site"],
                        columns=[str(c) for c in d["columns"]])
    raw = build(workers)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, X=raw.X, y=raw.y, lens=raw.lens, pid=raw.pid, site=raw.site,
             columns=np.array(raw.columns))
    return raw


if __name__ == "__main__":  # pragma: no cover
    import time
    t0 = time.time()
    r = load(refresh=True)
    pl = r.patient_label()
    print(f"{r.n_patients} patients, {r.n_rows} rows, septic {pl.sum()} "
          f"({pl.mean():.4f}), hour prevalence {r.y.mean():.4f}  [{time.time()-t0:.1f}s]")
