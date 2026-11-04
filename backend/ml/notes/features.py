"""Finding scores -> grid rows. Numbers only: a text column here is a bug, and ``_numeric`` refuses it.

``note_rows``     ``(stay_id, hr, name, value)``, the shape of ``mimic.sources``' binned tables:
                  each finding's highest score among the reports that became known in that hour,
                  and ``Note_rad_n``, how many did. Carrying a value forward is the grid builder's job,
                  exactly as for a lab.
``history_rows``  ``(stay_id, name, value)``, static: each finding's highest score over the
                  discharge summaries of admissions that ended before the stay, and
                  ``NoteHist_n``, how many there were.

Every name here is tier S (``TIER``). Each note is read once, however many stays it informs.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from backend.ml.notes import timing
from backend.ml.notes.readers import FINDINGS, Reader

TIER = "S"
COUNT = "Note_rad_n"
HIST_COUNT = "NoteHist_n"


def _numeric(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    for c in df.columns:
        if c not in keys and c != "name" and not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"note features may carry numbers only; column {c!r} is {df[c].dtype}")
    if "name" in df and not df.name.isin(set(names_all())).all():
        raise TypeError("an unexpected feature name: refusing in case it carries text")
    return df


def names_all() -> List[str]:
    from backend.ml.notes.readers import INDICATIONS
    return ([f"Note_{f}" for f in FINDINGS] + [f"Note_ind_{k}" for k in INDICATIONS] + [COUNT]
            + [f"NoteHist_{f}" for f in FINDINGS] + [HIST_COUNT])


def tiers() -> dict:
    return {n: TIER for n in names_all()}


def _score(notes: pd.DataFrame, reader: Reader) -> pd.DataFrame:
    """One row per note_id: its scores, read once."""
    u = notes.drop_duplicates("note_id")
    s = reader.read(u.text.fillna("").tolist())
    return pd.DataFrame(s, columns=reader.names, index=u.note_id.to_numpy())


def note_rows(radiology: pd.DataFrame, stays: pd.DataFrame, reader: Reader) -> pd.DataFrame:
    avail = timing.radiology_available(radiology, stays)
    if avail.empty:
        return pd.DataFrame(columns=["stay_id", "hr", "name", "value"])
    sc = _score(avail, reader)
    wide = sc.loc[avail.note_id.to_numpy()].reset_index(drop=True)
    wide[["stay_id", "hr"]] = avail[["stay_id", "hr"]].to_numpy()
    wide[COUNT] = 1.0
    agg = {c: "max" for c in reader.names}
    agg[COUNT] = "sum"
    g = wide.groupby(["stay_id", "hr"], as_index=False).agg(agg)
    out = g.melt(id_vars=["stay_id", "hr"], var_name="name", value_name="value")
    out = out.astype({"stay_id": np.int64, "hr": np.int64, "value": np.float32})
    return _numeric(out.sort_values(["stay_id", "hr", "name"]).reset_index(drop=True), ["stay_id", "hr"])


def history_rows(discharge: pd.DataFrame, stays: pd.DataFrame, admissions: pd.DataFrame,
                 reader: Reader) -> pd.DataFrame:
    prior = timing.prior_discharge(discharge, stays, admissions)
    cols = [f"Note_{f}" for f in FINDINGS]
    if prior.empty:
        return pd.DataFrame(columns=["stay_id", "name", "value"])
    sc = _score(prior, reader)[cols]
    wide = sc.loc[prior.note_id.to_numpy()].reset_index(drop=True)
    wide.columns = [c.replace("Note_", "NoteHist_", 1) for c in cols]
    wide["stay_id"] = prior.stay_id.to_numpy()
    wide[HIST_COUNT] = 1.0
    agg = {c: "max" for c in wide.columns if c.startswith("NoteHist_") and c != HIST_COUNT}
    agg[HIST_COUNT] = "sum"
    g = wide.groupby("stay_id", as_index=False).agg(agg)
    out = g.melt(id_vars=["stay_id"], var_name="name", value_name="value")
    out = out.astype({"stay_id": np.int64, "value": np.float32})
    return _numeric(out.sort_values(["stay_id", "name"]).reset_index(drop=True), ["stay_id"])
