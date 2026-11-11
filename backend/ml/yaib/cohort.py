"""A YAIB cohort directory, read as it is.

YAIB's classification cohorts are three parquet files keyed by ``stay_id``:

    dyn.parquet    one row per stay-hour: ``time`` (a timedelta) and 48 dynamic variables
    sta.parquet    one row per stay: age, sex, height, weight
    outc.parquet   one row per stay-hour: ``time`` and the hourly ``label``

Rows are not added, dropped or relabelled here. Anything that does not match that shape
is refused, because a silently repaired cohort is no longer YAIB's cohort.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

GROUP, SEQUENCE, LABEL = "stay_id", "time", "label"
DYNAMIC: List[str] = [
    "alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir", "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl",
    "crea", "crp", "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact", "lymph", "map", "mch", "mchc",
    "mcv", "methb", "mg", "na", "neut", "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp", "sbp", "temp",
    "tnt", "urine", "wbc"]
STATIC: List[str] = ["age", "sex", "height", "weight"]


@dataclass
class Cohort:
    """Hourly rows sorted by (stay, hour), the labels on those rows, and one static row per stay."""
    dyn: pd.DataFrame          # stay_id, hour, DYNAMIC...
    sta: pd.DataFrame          # stay_id, STATIC... (sex as 1.0 male / 0.0 female / NaN)
    label: np.ndarray          # float per dyn row; NaN where YAIB has no label
    name: str = ""

    @property
    def stays(self) -> np.ndarray:
        return np.unique(self.dyn[GROUP].to_numpy())

    def stay_labels(self) -> pd.Series:
        """Per-stay label as YAIB stratifies on it: the max over the stay's hours."""
        s = pd.Series(np.nan_to_num(self.label, nan=0.0), index=self.dyn[GROUP].to_numpy())
        return s.groupby(level=0).max().sort_index()


def _hours(t: pd.Series) -> np.ndarray:
    if pd.api.types.is_timedelta64_dtype(t):
        return (t / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    return t.to_numpy(dtype=float)


def load(directory: str, *, stage=None) -> Cohort:
    import pyarrow.parquet as pq

    def mark(name):
        if stage is not None:
            stage(name)

    def read(name, columns):
        path = os.path.join(directory, name + ".parquet")
        with pq.ParquetFile(path) as metadata:
            missing = [c for c in columns if c not in metadata.schema_arrow.names]
            if missing:
                raise KeyError(f"{name}.parquet lacks {missing}")
            table = metadata.read(columns=columns, use_threads=False)
        # Avoid pandas' default consolidation of every numeric column into one
        # second full-frame allocation. Never access table after self_destruct.
        frame = table.to_pandas(split_blocks=True, self_destruct=True)
        del table
        mark(name + "_read")
        return frame

    # Project at read time: unrelated parquet columns must never inflate this loader.
    dyn = read("dyn", [GROUP, SEQUENCE] + DYNAMIC)
    sta = read("sta", [GROUP] + STATIC)
    outc = read("outc", [GROUP, SEQUENCE, LABEL])
    # assign(...).drop(...) and selection.copy() made multiple full-size copies on
    # the 6.4M-row cohort. Normalize these owned frames without that transient peak.
    dyn["hour"] = _hours(dyn[SEQUENCE])
    del dyn[SEQUENCE]
    outc["hour"] = _hours(outc[SEQUENCE])
    del outc[SEQUENCE]
    dyn.sort_values([GROUP, "hour"], kind="mergesort", inplace=True, ignore_index=True)
    mark("dynamic_sorted")
    if dyn.duplicated([GROUP, "hour"]).any():
        raise ValueError("dyn.parquet has more than one row for a stay-hour")
    if outc.duplicated([GROUP, "hour"]).any():
        raise ValueError("outc.parquet has more than one row for a stay-hour")
    lab = dyn[[GROUP, "hour"]].merge(outc[[GROUP, "hour", LABEL]], on=[GROUP, "hour"], how="left", sort=False)
    if len(lab) != len(dyn):
        raise ValueError("label join changed the row count")
    extra = len(outc) - lab[LABEL].notna().sum()
    if extra:
        raise ValueError(f"{extra} labelled outc rows have no dyn row")
    label = lab[LABEL].astype(float).to_numpy()
    mark("labels_aligned")

    sta = sta[[GROUP] + STATIC].copy()
    sex = sta["sex"]
    if sex.dtype == object or pd.api.types.is_string_dtype(sex):
        sta["sex"] = sex.map({"Male": 1.0, "Female": 0.0, "M": 1.0, "F": 0.0}).astype(float)
    sta = sta.sort_values(GROUP).reset_index(drop=True)
    if sta[GROUP].duplicated().any():
        raise ValueError("sta.parquet has more than one row for a stay")
    # reindex(copy=False) can still consolidate Arrow's split numeric blocks into
    # another full frame. Moving this one column preserves the established order
    # without materializing every variable again.
    dyn.insert(1, "hour", dyn.pop("hour"))
    return Cohort(dyn, sta, label, os.path.basename(os.path.normpath(directory)))
