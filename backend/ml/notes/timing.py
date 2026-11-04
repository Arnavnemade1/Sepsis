"""Which note may be read at which hour of an ICU stay. Every availability rule lives here.

The grid's convention (``mimic.sources``): something known at time ``t`` lands in hour
``hr = floor((t - intime) / 1 h)``, and the row for ``hr`` may use everything in it.

  * Radiology: available at ``storetime``, the moment the signed read exists. A report stored
    before the ICU stay but within ``LOOKBACK_HOURS`` (the ED film, the floor's chest X-ray)
    is known on arrival and lands in hour 0. A missing ``storetime`` drops the report: a
    ``charttime`` is when the film was taken, which can be hours before anyone read it.
  * Discharge summaries: only those of admissions that ENDED before this stay's ``intime``
    (their ``storetime``, else ``charttime``, before ``intime`` too), so the current
    admission's summary, written after the outcome, can never be read.
  * Nothing past the stay's end or ``MAX_HOURS``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ml.mimic import concepts as K

LOOKBACK_HOURS = 24.0


def _hours(t: pd.Series, intime: pd.Series) -> np.ndarray:
    return ((t - intime).dt.total_seconds() / 3600.0).to_numpy()


def radiology_available(notes: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """Reports joined to the stays they may inform, with ``hr``: the hour the read became known.

    ``stays`` needs ``stay_id, subject_id, intime, outtime``. A patient's report can inform
    several stays (it is matched by subject and time, not by admission, because an ED film
    has the admission's ``hadm_id`` only sometimes)."""
    n = notes.dropna(subset=["storetime"])
    j = n.merge(stays[["stay_id", "subject_id", "intime", "outtime"]], on="subject_id")
    t = _hours(j.storetime, j.intime)
    end = np.minimum(_hours(j.outtime, j.intime), K.MAX_HOURS)
    keep = (t >= -LOOKBACK_HOURS) & (t < end)
    j = j[keep].assign(hr=np.maximum(np.floor(t[keep]), 0).astype(int), t=t[keep])
    return j.sort_values(["stay_id", "t", "note_id"]).reset_index(drop=True)


def prior_discharge(notes: pd.DataFrame, stays: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """Discharge summaries of admissions that ended before each stay began, joined to the stay.

    ``admissions`` needs ``hadm_id, dischtime``. The stay's own admission is excluded twice
    over: its discharge is after ``intime``, and so is the summary's time."""
    adm = admissions[["hadm_id", "dischtime"]]
    n = notes.merge(adm, on="hadm_id", how="inner")
    written = n.storetime.fillna(n.charttime)
    n = n.assign(written=written).dropna(subset=["written"])
    j = n.merge(stays[["stay_id", "subject_id", "hadm_id", "intime"]], on="subject_id", suffixes=("", "_stay"))
    keep = (j.dischtime < j.intime) & (j.written < j.intime) & (j.hadm_id != j.hadm_id_stay)
    return j[keep].sort_values(["stay_id", "written", "note_id"]).reset_index(drop=True)
