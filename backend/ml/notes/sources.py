"""The MIMIC-IV-Note tables, from the published CSVs or the PhysioNet lake.

Layout of the release (``physionet.org/files/mimic-iv-note/2.2/``)::

    note/radiology.csv.gz   note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
    note/discharge.csv.gz   note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text

Only the columns the pipeline reads are kept. ``TABLE_SPECS`` lets the PhysioNet lake ingest
these tables with the same hash-verified streaming as MIMIC-IV, bucketed by subject so one
patient's notes sit together.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

from backend.ml.physionet.lake import TableSpec

TABLES: Dict[str, List[str]] = {
    "radiology": ["note_id", "subject_id", "hadm_id", "charttime", "storetime", "text"],
    "discharge": ["note_id", "subject_id", "hadm_id", "charttime", "storetime", "text"],
}
TIMES = ("charttime", "storetime")


def table_specs() -> List[TableSpec]:
    i, t, s = "int64", "timestamp", "string"
    kinds = {"note_id": s, "subject_id": i, "hadm_id": i, "charttime": t, "storetime": t, "text": s}
    return [TableSpec("note", name, tuple((c, kinds[c]) for c in cols), key="subject_id", buckets=16)
            for name, cols in TABLES.items()]


class NoteSource:
    """Reads the note tables from a directory laid out like the release (``<root>/note/*.csv.gz``)."""

    def __init__(self, root: str):
        self.root = root

    def _path(self, table: str) -> str:
        for ext in (".csv.gz", ".csv"):
            p = os.path.join(self.root, "note", table + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"note/{table} not found under {self.root}")

    def read(self, table: str, subjects: Optional[set] = None) -> pd.DataFrame:
        df = pd.read_csv(self._path(table), usecols=TABLES[table], parse_dates=list(TIMES), low_memory=False)
        if subjects is not None:
            df = df[df.subject_id.isin(subjects)]
        return df.reset_index(drop=True)
