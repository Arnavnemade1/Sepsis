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


def load(directory: str) -> Cohort:
    dyn = pd.read_parquet(os.path.join(directory, "dyn.parquet"))
    sta = pd.read_parquet(os.path.join(directory, "sta.parquet"))
    outc = pd.read_parquet(os.path.join(directory, "outc.parquet"))
    for name, df, need in (("dyn", dyn, [GROUP, SEQUENCE] + DYNAMIC), ("sta", sta, [GROUP] + STATIC),
                           ("outc", outc, [GROUP, SEQUENCE, LABEL])):
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"{name}.parquet lacks {missing}")
    dyn = dyn.assign(hour=_hours(dyn[SEQUENCE])).drop(columns=[SEQUENCE])
    outc = outc.assign(hour=_hours(outc[SEQUENCE])).drop(columns=[SEQUENCE])
    dyn = dyn.sort_values([GROUP, "hour"], kind="mergesort").reset_index(drop=True)
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

    sta = sta[[GROUP] + STATIC].copy()
    sex = sta["sex"]
    if sex.dtype == object or pd.api.types.is_string_dtype(sex):
        sta["sex"] = sex.map({"Male": 1.0, "Female": 0.0, "M": 1.0, "F": 0.0}).astype(float)
    sta = sta.sort_values(GROUP).reset_index(drop=True)
    if sta[GROUP].duplicated().any():
        raise ValueError("sta.parquet has more than one row for a stay")
    keep = [GROUP, "hour"] + DYNAMIC
    return Cohort(dyn[keep].copy(), sta, label, os.path.basename(os.path.normpath(directory)))
