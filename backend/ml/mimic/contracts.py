"""Before any model sees the data: is it the data we think it is?

A hash proves the bytes are what PhysioNet published. It cannot prove the catalog still fits
them. A new release can repoint an itemid, change a unit or drop a column, and nothing would
crash: SOFA would just quietly become wrong, and every downstream number with it.

Each check here is planted by ``verify_physionet``. ``fail`` stops the pipeline; ``warn`` is
reported. The result is aggregate by construction: counts, shares and medians, with any count
below ``SMALL_CELL`` printed as "<10", so it is safe to read, log and commit.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.ml.mimic import concepts as K
from backend.ml.mimic.lake import LakeSource

SMALL_CELL = 10
SAMPLE_BUCKETS = 8
DRIFT_RATIO = 2.0          # a median this far from the demo's is a unit or a meaning change
DRIFT_MIN_N = 100


def _cell(n: int):
    return f"<{SMALL_CELL}" if 0 < n < SMALL_CELL else int(n)


class _Report:
    def __init__(self):
        self.checks: List[Dict] = []

    def add(self, name: str, ok: bool, detail: str = "", severity: str = "fail") -> None:
        self.checks.append({"check": name, "ok": bool(ok), "severity": severity, "detail": detail})

    def out(self) -> Dict:
        failed = [c["check"] for c in self.checks if not c["ok"] and c["severity"] == "fail"]
        warned = [c["check"] for c in self.checks if not c["ok"] and c["severity"] == "warn"]
        return {"passed": not failed, "failed": failed, "warned": warned, "checks": self.checks}


def _item_medians(src: LakeSource, table: str, value: str, buckets: int) -> Dict[int, Dict]:
    """Per itemid: rows and median of the numeric value, over a sample of buckets. A bucket is
    a hash of the stay, so a sample of buckets is a sample of patients."""
    spec = src.specs[table]
    parts = []
    for b in range(min(buckets, spec.buckets)):
        with src._in_bucket(b * max(spec.buckets // max(buckets, 1), 1)):
            parts.append(src._read(spec.module, table, ["itemid", value]))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["itemid", value])
    g = df.groupby("itemid")[value]
    return {int(k): {"n": int(v.notna().sum()), "median": float(v.median())} for k, v in g}


def check(src: LakeSource, baseline: Optional[LakeSource] = None, min_stays: int = 0) -> Dict:
    r = _Report()
    st, pa, ad = src.icustays(), src.patients(), src.admissions()

    r.add("cohort size", len(st) >= min_stays, f"{_cell(len(st))} ICU stays; this dataset needs at least {min_stays}")
    for name, df, key in (("icustays", st, "stay_id"), ("patients", pa, "subject_id"), ("admissions", ad, "hadm_id")):
        r.add(f"{name}.{key} is unique", df[key].is_unique, f"{_cell(int(df[key].duplicated().sum()))} duplicates")
    orphan_p = ~st.subject_id.isin(pa.subject_id)
    orphan_a = ~st.hadm_id.isin(ad.hadm_id)
    r.add("every stay has its patient", not orphan_p.any(), f"{_cell(int(orphan_p.sum()))} orphans")
    r.add("every stay has its admission", not orphan_a.any(), f"{_cell(int(orphan_a.sum()))} orphans")

    bad_time = (st.outtime <= st.intime) | st.intime.isna() | st.outtime.isna()
    r.add("stays end after they begin", bad_time.mean() < 0.001 if len(st) else False,
          f"{bad_time.mean():.4%} of stays")
    years = st.intime.dt.year.dropna()
    r.add("dates are in MIMIC's shifted range", bool(len(years)) and years.between(2100, 2230).all(),
          "the de-identified calendar runs 2100-2230" if len(years) else "no dates")
    age = pa.anchor_age
    r.add("ages are plausible", bool(len(age)) and age.between(0, 120).all(), f"{age.min()}..{age.max()}" if len(age) else "")

    di, dl = src.d_items(), src.d_labitems()
    labels = dict(zip(di.itemid, di.label)) | dict(zip(dl.itemid, dl.label))
    wrong = [f"{c.name}/{i}" for c in K.all_concepts() for i in c.itemids
             if i not in labels or not re.search(c.label_pattern, str(labels[i]))]
    r.add("every catalog itemid still means the same thing", not wrong,
          f"{sum(len(c.itemids) for c in K.all_concepts())} itemids" + (f"; MISMATCH {wrong[:6]}" if wrong else ""))

    chart = _item_medians(src, "chartevents", "valuenum", SAMPLE_BUCKETS)
    labs = _item_medians(src, "labevents", "valuenum", SAMPLE_BUCKETS)
    data = chart | labs
    cons = [c for c in K.CHART + K.LAB]
    absent = [c.name for c in cons if not any(data.get(i, {}).get("n", 0) for i in c.itemids)]
    r.add("the core chart and lab concepts have data", len(absent) <= len(cons) // 5,
          f"{len(cons) - len(absent)} of {len(cons)} in the sampled patients" + (f"; absent {absent[:8]}" if absent else ""))

    orphans_c = 0
    spec = src.specs["chartevents"]
    seen = 0
    for b in range(min(SAMPLE_BUCKETS, spec.buckets)):
        with src._in_bucket(b * max(spec.buckets // SAMPLE_BUCKETS, 1)):
            ids = src._read("icu", "chartevents", ["stay_id"]).stay_id
        seen += len(ids)
        orphans_c += int((~ids.isin(st.stay_id)).sum())
    r.add("chart rows belong to known stays", seen > 0 and orphans_c / max(seen, 1) < 0.001,
          f"{orphans_c / max(seen, 1):.4%} of {_cell(seen)} sampled rows are orphans")

    if baseline is not None:
        base = _item_medians(baseline, "chartevents", "valuenum", baseline.specs["chartevents"].buckets) | \
            _item_medians(baseline, "labevents", "valuenum", baseline.specs["labevents"].buckets)
        drifted = []
        compared = 0
        for item, cur in data.items():
            ref = base.get(item)
            if ref is None or min(ref["n"], cur["n"]) < DRIFT_MIN_N or abs(ref["median"]) < 1e-6:
                continue
            compared += 1
            ratio = cur["median"] / ref["median"]
            if not (1 / DRIFT_RATIO <= ratio <= DRIFT_RATIO):
                drifted.append(f"{item}: {ref['median']:g} -> {cur['median']:g}")
        r.add("no item moved like a unit change", not drifted and compared > 0,
              f"{compared} items compared with the demo" + (f"; DRIFTED {drifted[:6]}" if drifted else ""))
    else:
        r.add("no item moved like a unit change", False, "no demo baseline to compare with", severity="warn")
    return r.out()
