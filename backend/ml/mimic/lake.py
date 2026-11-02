"""MIMIC-IV read from the Parquet lake: ``CsvSource``'s semantics at full-database scale.

``CsvSource`` is the reference, but on full MIMIC it would hold every kept chart row in one
frame, well past a laptop's memory. ``LakeSource`` inherits every line of its binning and only
changes where rows come from: one bucket at a time, each bucket holding all the rows of the
stays hashed to it. Binning is per stay, so the pages are exactly the reference's rows, and
``verify_physionet`` requires the finished grid to match the CSV path byte for byte.

``project`` derives what the lake keeps from the same catalog the binning reads. A concept
added to the catalog changes a table's ``spec_hash``, and the lake then refuses to serve until
that table is rebuilt, rather than quietly returning nothing for the new item.
"""
from __future__ import annotations

import contextlib
import os
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from backend.ml.mimic import concepts as K
from backend.ml.mimic.sources import TIME_COLS, CsvSource
from backend.ml.physionet import lake as PL
from backend.ml.physionet.lake import TableSpec


class LakeStale(RuntimeError):
    """A lake table is missing, or was built for a different catalog."""


def _ids(concepts) -> frozenset:
    return frozenset(i for c in concepts for i in c.itemids)


def project() -> List[TableSpec]:
    """Every table and column ``CsvSource`` reads, and only those rows the catalog can use."""
    i, f, s, t = "int64", "float64", "string", "timestamp"
    interval_items = _ids(K.VENT) | _ids(c for c in K.TREAT_INTERVAL if c.kind != "sedation")
    return [
        TableSpec("icu", "icustays", (("subject_id", i), ("hadm_id", i), ("stay_id", i), ("first_careunit", s),
                                      ("intime", t), ("outtime", t), ("los", f))),
        TableSpec("hosp", "patients", (("subject_id", i), ("gender", s), ("anchor_age", i), ("anchor_year", i),
                                       ("anchor_year_group", s))),
        TableSpec("hosp", "admissions", (("subject_id", i), ("hadm_id", i), ("admittime", t), ("dischtime", t),
                                         ("admission_type", s), ("admission_location", s),
                                         ("hospital_expire_flag", i), ("insurance", s), ("race", s))),
        TableSpec("hosp", "services", (("subject_id", i), ("hadm_id", i), ("transfertime", t), ("curr_service", s))),
        TableSpec("hosp", "diagnoses_icd", (("subject_id", i), ("hadm_id", i), ("icd_code", s), ("icd_version", i))),
        TableSpec("icu", "d_items", (("itemid", i), ("label", s), ("abbreviation", s), ("linksto", s),
                                     ("category", s), ("unitname", s), ("param_type", s))),
        TableSpec("hosp", "d_labitems", (("itemid", i), ("label", s), ("fluid", s), ("category", s))),
        TableSpec("icu", "chartevents", (("stay_id", i), ("charttime", t), ("itemid", i), ("value", s), ("valuenum", f)),
                  key="stay_id", buckets=128, items=_ids(K.CHART_ALL),
                  text_items=frozenset({223900}) | _ids(c for c in K.EXAM if c.kind == "chart_text")),
        TableSpec("hosp", "labevents", (("hadm_id", i), ("charttime", t), ("itemid", i), ("valuenum", f), ("value", s)),
                  key="hadm_id", buckets=64, items=_ids(K.LAB + K.LAB_TEXT), text_items=_ids(K.LAB_TEXT)),
        TableSpec("icu", "outputevents", (("stay_id", i), ("charttime", t), ("itemid", i), ("value", f)),
                  key="stay_id", buckets=16, items=frozenset(K.UO.itemids)),
        # not filtered by item: the first charted weight and the antibiotic category read every row
        TableSpec("icu", "inputevents", (("stay_id", i), ("starttime", t), ("endtime", t), ("itemid", i), ("rate", f),
                                         ("rateuom", s), ("amount", f), ("patientweight", f),
                                         ("statusdescription", s), ("ordercategoryname", s)),
                  key="stay_id", buckets=32),
        TableSpec("icu", "procedureevents", (("stay_id", i), ("starttime", t), ("endtime", t), ("itemid", i)),
                  key="stay_id", buckets=16, items=interval_items),
        TableSpec("hosp", "microbiologyevents", (("hadm_id", i), ("chartdate", t), ("charttime", t), ("spec_type_desc", s)),
                  key="hadm_id", buckets=16),
        TableSpec("hosp", "prescriptions", (("hadm_id", i), ("starttime", t), ("drug", s), ("route", s)),
                  key="hadm_id", buckets=32),
    ]


class LakeSource(CsvSource):
    """The ``CsvSource`` interface over a lake. Tables are read one bucket at a time."""

    def __init__(self, root: str):
        super().__init__(root=root)
        self.specs: Dict[str, TableSpec] = {s.table: s for s in project()}
        self._bucket: Optional[int] = None
        stale = [s.rel for s in self.specs.values() if not PL.is_current(root, s, None)]
        if stale:
            raise LakeStale(f"lake at {root} is missing or stale for: {', '.join(stale)}; rebuild those tables")

    def fingerprint(self) -> str:
        return PL.fingerprint(self.root, list(self.specs.values()))

    def _files(self, spec: TableSpec) -> List[str]:
        d = PL.table_dir(self.root, spec)
        if self._bucket is not None and spec.buckets > 1:
            p = os.path.join(d, f"b{self._bucket % spec.buckets:04d}.parquet")
            return [p] if os.path.exists(p) else []
        return [os.path.join(d, f) for f in PL.read_manifest(self.root, spec)["files"]]

    def _read(self, module: str, table: str, usecols: List[str], keep=None) -> pd.DataFrame:
        import pyarrow as pa
        import pyarrow.parquet as pq
        spec = self.specs[table]
        missing = set(usecols) - set(spec.names)
        if missing:
            raise LakeStale(f"{table} does not keep {sorted(missing)}; add them to mimic.lake.project")
        files = self._files(spec)
        if files:
            tb = pa.concat_tables([pq.read_table(f, columns=usecols) for f in files])
        else:
            tb = pa.schema([(c, PL._arrow_type(pa, dict(spec.columns)[c])) for c in usecols]).empty_table()
        df = tb.to_pandas(coerce_temporal_nanoseconds=True)
        for c in df.columns:                     # None -> NaN, as a CSV read gives
            if df[c].dtype == object:
                df[c] = df[c].where(df[c].notna(), np.nan)
        if keep is not None:
            df = df[keep(df)]
        return df

    def d_items(self) -> pd.DataFrame:
        return self._read("icu", "d_items", self.specs["d_items"].names)

    def d_labitems(self) -> pd.DataFrame:
        return self._read("hosp", "d_labitems", self.specs["d_labitems"].names)

    # ---------------- one bucket at a time ---------------- #
    @contextlib.contextmanager
    def _in_bucket(self, b: int):
        self._bucket = b
        try:
            yield
        finally:
            self._bucket = None

    def _shards(self, stays: pd.DataFrame, table: str) -> Iterator:
        spec = self.specs[table]
        ids = stays[spec.key].to_numpy()
        for b in range(spec.buckets):
            shard = stays[ids % spec.buckets == b]
            if len(shard):
                yield b, shard

    def _paged(self, stays, table, call):
        for b, shard in self._shards(stays, table):
            with self._in_bucket(b):
                frame = call(shard)
            yield frame

    def binned_chart(self, stays):
        return self._paged(stays, "chartevents", super().binned_chart)

    def binned_labs(self, stays):
        return self._paged(stays, "labevents", super().binned_labs)

    def binned_uo(self, stays):
        return self._paged(stays, "outputevents", super().binned_uo)

    def interval_hours(self, stays, kind):
        table = "inputevents" if kind in ("vaso", "sedation") else "procedureevents"
        return self._paged(stays, table, lambda s: super(LakeSource, self).interval_hours(s, kind))

    def hourly_amounts(self, stays):
        return self._paged(stays, "inputevents", super().hourly_amounts)

    def event_times(self, stays, kind):
        table = {"abx_admin": "inputevents", "culture": "microbiologyevents"}.get(kind, "prescriptions")
        parts = list(self._paged(stays, table, lambda s: super(LakeSource, self).event_times(s, kind)))
        return pd.concat(parts, ignore_index=True).sort_values(["stay_id", "t"], kind="stable").reset_index(drop=True)

    def first_weight(self, stays):
        parts = list(self._paged(stays, "inputevents", super().first_weight))
        return pd.concat(parts).sort_index()


def build(root: str, demo_root: str, csv_root: Optional[str] = None) -> Dict:  # pragma: no cover - convenience
    """Build the lake from local CSVs (``fetch`` handles the remote path)."""
    from backend.ml.physionet.fetch import LocalStream
    out = {}
    for spec in project():
        stream = LocalStream(os.path.join(csv_root or demo_root, spec.source))
        try:
            out[spec.rel] = PL.ingest(spec, stream, root, None)
        finally:
            stream.close()
    return out


__all__ = ["LakeSource", "LakeStale", "project", "TIME_COLS"]
