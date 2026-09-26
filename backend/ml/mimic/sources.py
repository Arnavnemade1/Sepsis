"""Where MIMIC-IV comes from: local CSVs (the open demo, or full CSVs) or BigQuery.

Both sources expose the same BINNED interface — every event is reduced to
``(stay_id, hr, name, value)`` rows, ``hr = floor((time - intime) / 1 h)`` — so the grid
assembly downstream never knows which one it is reading. ``CsvSource`` is the REFERENCE
semantics (pandas, readable, tested on the demo); ``BigQuerySource`` renders the same
binning as SQL and runs it server-side, so a full-cohort extraction downloads hourly
aggregates (hundreds of MB) rather than raw events (tens of GB). Nothing here downloads
anything unless asked.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from backend.ml.mimic import concepts as K

TIME_COLS = {"intime", "outtime", "admittime", "dischtime", "deathtime", "charttime",
             "chartdate", "starttime", "endtime", "storetime", "transfertime"}
PRE_ICU_LAB_HOURS = 24      # labs drawn this long before ICU admission are known at hour 0
EVENT_LOOKBACK_HOURS = 72   # cultures / antibiotics kept this far before admission (for pairing)


def _to_hours(t: pd.Series, intime: pd.Series) -> np.ndarray:
    return ((t - intime).dt.total_seconds() / 3600.0).to_numpy()


# --------------------------------------------------------------------------- #
@dataclass
class CsvSource:
    """MIMIC-IV laid out as PhysioNet ships it: ``<root>/hosp/*.csv.gz``, ``<root>/icu/*``."""
    root: str
    chunksize: int = 2_000_000

    def _path(self, module: str, table: str) -> str:
        for ext in (".csv.gz", ".csv"):
            p = os.path.join(self.root, module, table + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"{module}/{table} not found under {self.root}")

    def _read(self, module: str, table: str, usecols: List[str],
              keep=None) -> pd.DataFrame:
        """Read in chunks (full chartevents is ~3 GB compressed); ``keep`` filters a chunk."""
        path = self._path(module, table)
        parts = []
        for ch in pd.read_csv(path, usecols=usecols, chunksize=self.chunksize, low_memory=False):
            if keep is not None:
                ch = ch[keep(ch)]
            parts.append(ch)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
        for c in set(usecols) & TIME_COLS:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    # ---------------- dimension tables ---------------- #
    def icustays(self) -> pd.DataFrame:
        return self._read("icu", "icustays", ["subject_id", "hadm_id", "stay_id", "first_careunit",
                                              "intime", "outtime", "los"])

    def patients(self) -> pd.DataFrame:
        return self._read("hosp", "patients", ["subject_id", "gender", "anchor_age", "anchor_year",
                                               "anchor_year_group"])

    def admissions(self) -> pd.DataFrame:
        # race / insurance feed the subgroup audit only (extract.AUDIT_ATTRS), never a model
        return self._read("hosp", "admissions", ["subject_id", "hadm_id", "admittime", "dischtime",
                                                 "admission_type", "admission_location",
                                                 "hospital_expire_flag", "insurance", "race"])

    def services(self) -> pd.DataFrame:
        return self._read("hosp", "services", ["subject_id", "hadm_id", "transfertime", "curr_service"])

    def diagnoses(self) -> pd.DataFrame:
        """Every coded diagnosis (``history`` reads only admissions discharged before the stay)."""
        return self._read("hosp", "diagnoses_icd", ["subject_id", "hadm_id", "icd_code", "icd_version"])

    def d_items(self) -> pd.DataFrame:
        return pd.read_csv(self._path("icu", "d_items"))

    def d_labitems(self) -> pd.DataFrame:
        return pd.read_csv(self._path("hosp", "d_labitems"))

    # ---------------- binned event tables ---------------- #
    def binned_chart(self, stays: pd.DataFrame) -> pd.DataFrame:
        ids = {i: c for c in K.CHART_ALL for i in c.itemids}
        sid = set(stays.stay_id)
        ev = self._read("icu", "chartevents", ["stay_id", "charttime", "itemid", "value", "valuenum"],
                        keep=lambda d: d.itemid.isin(ids) & d.stay_id.isin(sid))
        ev["name"] = ev.itemid.map(lambda i: ids[i].name)
        v = ev.valuenum.astype(float)
        # intubated: "No Response-ETT" carries valuenum 1, identical to a truly unresponsive
        # patient. Recoded to 0 (unable to assess) as MIT-LCP mimic-code does.
        v = np.where((ev.itemid == 223900) & (ev.value == "No Response-ETT"), 0.0, v)
        text = ev.value.fillna("").astype(str).str.strip().str.upper()
        for c in K.EXAM:                               # the bedside exam: text -> ordinal
            if c.kind == "chart_text":
                m = (ev.name == c.name).to_numpy()
                v[m] = text[m].map(dict(c.text_map)).astype(float).to_numpy()
        ev["value"] = v
        ev = _plausible(ev, {c.name: c for c in K.CHART_ALL})
        # units: Fahrenheit -> Celsius; FiO2 percent -> fraction
        f = ev.name == "TempF"
        ev.loc[f, "value"] = (ev.loc[f, "value"] - 32.0) * 5.0 / 9.0
        ev.loc[ev.name.isin(["TempF", "TempC"]), "name"] = "Temp"
        fi = ev.name == "FiO2"
        ev.loc[fi, "value"] = np.where(ev.loc[fi, "value"] > 1.0, ev.loc[fi, "value"] / 100.0,
                                       ev.loc[fi, "value"])
        ev = ev[~fi | (ev.value >= 0.21) & (ev.value <= 1.0)]
        worst = {c.name for c in K.CHART_ALL if c.agg == "max"}
        parts = [_bin_last(ev[~ev.name.isin(worst)], stays, "charttime", lo=0),
                 _bin_agg(ev[ev.name.isin(worst)], stays, "charttime", "max")]
        for base, how in K.EXTREMES.items():            # the hour's extreme beside its last value
            parts.append(_bin_agg(ev[ev.name == base], stays, "charttime", how).assign(name=f"{base}_{how}"))
        return pd.concat(parts, ignore_index=True)

    def binned_labs(self, stays: pd.DataFrame) -> pd.DataFrame:
        ids = {i: c for c in K.LAB + K.LAB_TEXT for i in c.itemids}
        hadm = set(stays.hadm_id)
        ev = self._read("hosp", "labevents", ["hadm_id", "charttime", "itemid", "valuenum", "value"],
                        keep=lambda d: d.itemid.isin(ids) & d.hadm_id.isin(hadm))
        ev["name"] = ev.itemid.map(lambda i: ids[i].name)
        num = ev.valuenum.astype(float).to_numpy()
        text = ev.value.fillna("").astype(str).str.strip().str.upper()
        for c in K.LAB_TEXT:                          # urinalysis: text -> ordinal, else dropped
            m = (ev.name == c.name).to_numpy()
            num[m] = text[m].map(dict(c.text_map)).astype(float).to_numpy()
        ev["value"] = num
        ev = _plausible(ev, {c.name: c for c in K.LAB + K.LAB_TEXT})
        # labs belong to the admission; attach to each of its ICU stays by time
        ev = ev.merge(stays[["stay_id", "hadm_id"]], on="hadm_id")
        return _bin_last(ev, stays, "charttime", lo=-PRE_ICU_LAB_HOURS, fold_pre_to_zero=True)

    def binned_uo(self, stays: pd.DataFrame) -> pd.DataFrame:
        sid = set(stays.stay_id)
        ev = self._read("icu", "outputevents", ["stay_id", "charttime", "itemid", "value"],
                        keep=lambda d: d.itemid.isin(K.UO.itemids) & d.stay_id.isin(sid))
        v = ev.value.astype(float)
        ev["value"] = np.where(ev.itemid == K.UO_IRRIGANT_IN, -v, v)   # irrigant in subtracts
        ev["name"] = "UrineOutput"
        ev = ev.merge(stays[["stay_id", "intime"]], on="stay_id")
        ev["hr"] = np.floor(_to_hours(ev.charttime, ev.intime)).astype(int)
        ev = ev[(ev.hr >= 0) & (ev.hr < K.MAX_HOURS)]
        out = ev.groupby(["stay_id", "hr", "name"], as_index=False).value.sum()
        return out

    def interval_hours(self, stays: pd.DataFrame, kind: str) -> pd.DataFrame:
        """Vasopressor rates (hourly max) or ventilation (1) for every hour an interval covers."""
        sid = set(stays.stay_id)
        if kind == "vaso":
            cons = {i: c for c in K.VASO for i in c.itemids}
            ev = self._read("icu", "inputevents",
                            ["stay_id", "starttime", "endtime", "itemid", "rate", "rateuom",
                             "patientweight", "statusdescription"],
                            keep=lambda d: d.itemid.isin(cons) & d.stay_id.isin(sid))
            ev = ev[ev.statusdescription.fillna("") != "Rewritten"]
            rate = ev.rate.astype(float)
            per_min = ev.rateuom.fillna("").str.lower() == "mcg/min"          # not weight-based
            rate = np.where(per_min, rate / ev.patientweight.astype(float), rate)
            ev["value"] = rate
            ev["name"] = ev.itemid.map(lambda i: cons[i].name)
            ev = _plausible(ev, {c.name: c for c in K.VASO})
        elif kind == "sedation":
            cons = {i: c for c in K.TREAT_INTERVAL if c.kind == "sedation" for i in c.itemids}
            ev = self._read("icu", "inputevents", ["stay_id", "starttime", "endtime", "itemid",
                                                   "statusdescription"],
                            keep=lambda d: d.itemid.isin(cons) & d.stay_id.isin(sid))
            ev = ev[ev.statusdescription.fillna("") != "Rewritten"]
            ev["value"] = 1.0
            ev["name"] = "Sedation"
        else:
            src = K.VENT if kind == "vent" else [c for c in K.TREAT_INTERVAL if c.kind == kind]
            cons = {i: c for c in src for i in c.itemids}
            ev = self._read("icu", "procedureevents", ["stay_id", "starttime", "endtime", "itemid"],
                            keep=lambda d: d.itemid.isin(cons) & d.stay_id.isin(sid))
            ev["value"] = 1.0
            ev["name"] = ev.itemid.map(lambda i: cons[i].name)
        return _expand_intervals(ev, stays)

    def hourly_amounts(self, stays: pd.DataFrame) -> pd.DataFrame:
        """Volume STARTED in each hour (mL): crystalloids and blood products."""
        sid = set(stays.stay_id)
        cons = {i: c for c in K.TREAT_AMOUNT for i in c.itemids}
        ev = self._read("icu", "inputevents", ["stay_id", "starttime", "itemid", "amount", "statusdescription"],
                        keep=lambda d: d.itemid.isin(cons) & d.stay_id.isin(sid))
        ev = ev[ev.statusdescription.fillna("") != "Rewritten"]
        ev["value"] = ev.amount.astype(float)
        ev["name"] = ev.itemid.map(lambda i: cons[i].name)
        return _sum_by_hour(_plausible(ev, {c.name: c for c in K.TREAT_AMOUNT}), stays)

    def event_times(self, stays: pd.DataFrame, kind: str) -> pd.DataFrame:
        """``(stay_id, t)`` hours from ICU admission of every culture / antibiotic start, or
        (``abx_admin``) every IV antibiotic administration charted at the bedside."""
        hadm = set(stays.hadm_id)
        if kind == "abx_admin":
            sid = set(stays.stay_id)
            ev = self._read("icu", "inputevents", ["stay_id", "starttime", "ordercategoryname",
                                                   "statusdescription"],
                            keep=lambda d: (d.ordercategoryname == K.ABX_ADMIN_CATEGORY) & d.stay_id.isin(sid))
            ev = ev[ev.statusdescription.fillna("") != "Rewritten"]
            ev = ev.merge(stays[["stay_id", "intime", "outtime"]], on="stay_id")
            ev["t"] = _to_hours(ev.starttime, ev.intime)
            end = _to_hours(ev.outtime, ev.intime)
            ev = ev[(ev.t >= 0) & (ev.t <= np.minimum(end, K.MAX_HOURS))]
            return ev[["stay_id", "t"]].sort_values(["stay_id", "t"]).reset_index(drop=True)
        if kind == "culture":
            ev = self._read("hosp", "microbiologyevents", ["hadm_id", "chartdate", "charttime", "spec_type_desc"],
                            keep=lambda d: d.hadm_id.isin(hadm))
            ev["time"] = ev.charttime.fillna(ev.chartdate)
        else:
            ev = self._read("hosp", "prescriptions", ["hadm_id", "starttime", "drug", "route"],
                            keep=lambda d: d.hadm_id.isin(hadm))
            ev = ev[[K.is_antibiotic(d, r) for d, r in zip(ev.drug, ev.route)]]
            ev["time"] = ev.starttime
        ev = ev.dropna(subset=["time"]).merge(stays[["stay_id", "hadm_id", "intime", "outtime"]],
                                              on="hadm_id")
        ev["t"] = _to_hours(ev.time, ev.intime)
        end = _to_hours(ev.outtime, ev.intime)
        ev = ev[(ev.t >= -EVENT_LOOKBACK_HOURS) & (ev.t <= np.minimum(end, K.MAX_HOURS))]
        return typed_events(ev, kind)

    def first_weight(self, stays: pd.DataFrame) -> pd.Series:
        sid = set(stays.stay_id)
        ev = self._read("icu", "inputevents", ["stay_id", "starttime", "patientweight"],
                        keep=lambda d: d.stay_id.isin(sid))
        ev = ev.dropna(subset=["patientweight"]).sort_values("starttime")
        w = ev.groupby("stay_id").patientweight.first().astype(float)
        return w.where((w > 20) & (w < 400))


# --------------------------------------------------------------------------- #
def _plausible(ev: pd.DataFrame, cons: Dict[str, K.Concept]) -> pd.DataFrame:
    lo = ev.name.map(lambda n: cons[n].lo).astype(float)
    hi = ev.name.map(lambda n: cons[n].hi).astype(float)
    ok = np.isfinite(ev.value) & (ev.value >= lo) & (ev.value <= hi)
    return ev[ok]


def _bin_last(ev: pd.DataFrame, stays: pd.DataFrame, tcol: str, lo: int,
              fold_pre_to_zero: bool = False) -> pd.DataFrame:
    """Last value per (stay, hour, name) by event time, plus the count in that hour."""
    if "intime" not in ev.columns:
        ev = ev.merge(stays[["stay_id", "intime"]], on="stay_id")
    h = _to_hours(ev[tcol], ev["intime"])
    ev = ev.assign(hr=np.floor(h).astype(int), _t=h)
    ev = ev[(ev.hr >= lo) & (ev.hr < K.MAX_HOURS)]
    if fold_pre_to_zero:
        ev.loc[ev.hr < 0, "hr"] = 0          # known at admission; keeps its earlier time order
    ev = ev.sort_values(["stay_id", "name", "hr", "_t"])
    g = ev.groupby(["stay_id", "hr", "name"], sort=False)
    out = g.agg(value=("value", "last"), n=("value", "size")).reset_index()
    return out


def typed_events(ev: pd.DataFrame, kind: str) -> pd.DataFrame:
    """``(stay_id, t)`` plus what the event WAS — the specimen group of a culture (one event per
    specimen group and time, as the label's culture times are one per time), or the spectrum
    of an antibiotic (oral vancomycin treats C. difficile, not MRSA) — one function for both
    sources."""
    if kind == "culture":
        ev = ev.assign(group=[K.specimen_group(x) for x in ev.get("spec_type_desc", pd.Series([""] * len(ev)))])
        ev = ev.drop_duplicates(["stay_id", "t", "group"])
        return ev[["stay_id", "t", "group"]].sort_values(["stay_id", "t"]).reset_index(drop=True)
    cls = [K.abx_classes(d) for d in ev.drug]
    oral = ev.route.fillna("").astype(str).str.upper().str.strip().isin(["PO", "NG", "PO/NG", "OG"]).to_numpy()
    ev = ev.assign(mrsa=np.array([c[0] for c in cls], dtype=bool) & ~(oral & ev.drug.str.lower().str.contains("vanco")),
                   broad=np.array([c[1] for c in cls], dtype=bool))
    return ev[["stay_id", "t", "mrsa", "broad"]].sort_values(["stay_id", "t"]).reset_index(drop=True)


def _bin_agg(ev: pd.DataFrame, stays: pd.DataFrame, tcol: str, how: str) -> pd.DataFrame:
    """min / max per (stay, hour, name) — hours from ICU admission, inside the grid."""
    if not len(ev):
        return pd.DataFrame(columns=["stay_id", "hr", "name", "value"])
    if "intime" not in ev.columns:
        ev = ev.merge(stays[["stay_id", "intime"]], on="stay_id")
    ev = ev.assign(hr=np.floor(_to_hours(ev[tcol], ev["intime"])).astype(int))
    ev = ev[(ev.hr >= 0) & (ev.hr < K.MAX_HOURS)]
    return ev.groupby(["stay_id", "hr", "name"], as_index=False).value.agg(how)


def _sum_by_hour(ev: pd.DataFrame, stays: pd.DataFrame, tcol: str = "starttime") -> pd.DataFrame:
    ev = ev.merge(stays[["stay_id", "intime"]], on="stay_id")
    ev["hr"] = np.floor(_to_hours(ev[tcol], ev.intime))
    ev = ev[(ev.hr >= 0) & (ev.hr < K.MAX_HOURS)]
    ev["hr"] = ev.hr.astype(int)
    return ev.groupby(["stay_id", "hr", "name"], as_index=False).value.sum()


def _expand_intervals(ev: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    ev = ev.merge(stays[["stay_id", "intime"]], on="stay_id")
    s = np.floor(_to_hours(ev.starttime, ev.intime))
    e = np.floor(_to_hours(ev.endtime, ev.intime) - 1e-9)
    s = np.clip(s, 0, K.MAX_HOURS - 1)
    e = np.clip(e, -1, K.MAX_HOURS - 1)
    keep = np.isfinite(s) & np.isfinite(e) & (e >= s)
    ev, s, e = ev[keep], s[keep].astype(int), e[keep].astype(int)
    reps = (e - s + 1)
    rows = np.repeat(np.arange(len(ev)), reps)
    offs = np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps)
    out = pd.DataFrame({"stay_id": ev.stay_id.to_numpy()[rows], "hr": s[rows] + offs,
                        "name": ev.name.to_numpy()[rows], "value": ev.value.to_numpy()[rows]})
    return out.groupby(["stay_id", "hr", "name"], as_index=False).value.max()


# --------------------------------------------------------------------------- #
@dataclass
class BigQuerySource:
    """The same binned interface, computed server-side on ``physionet-data``.

    Requires credentialed PhysioNet access linked to a Google Cloud project, and
    ``pip install google-cloud-bigquery db-dtypes``. The SQL mirrors ``CsvSource``
    exactly (same itemids, conversions, plausibility windows, hour arithmetic); run
    ``verify_mimic.py --bigquery`` once to compare both paths on the demo stays."""
    project: str
    hosp: str = "physionet-data.mimiciv_3_1_hosp"
    icu: str = "physionet-data.mimiciv_3_1_icu"

    def _client(self):
        try:
            from google.cloud import bigquery
        except ImportError as exc:  # pragma: no cover
            raise ImportError("BigQuerySource needs `pip install google-cloud-bigquery db-dtypes` "
                              "and `gcloud auth application-default login`") from exc
        return bigquery.Client(project=self.project)

    def query(self, sql: str) -> pd.DataFrame:  # pragma: no cover - needs credentials
        return self._client().query(sql).to_dataframe()

    #: at most this many stays are named in a query (the parity check's demo cohort); a full
    #: cohort is filtered by the joins instead
    FILTER_MAX = 2000

    @classmethod
    def stay_filter(cls, stays) -> str:
        """``s.stay_id IN (...)`` for a small cohort, else TRUE."""
        if stays is None or len(stays) > cls.FILTER_MAX:
            return "TRUE"
        return f"s.stay_id IN ({', '.join(str(int(x)) for x in stays.stay_id)})"

    def _pages(self, sql: str, page_rows: int = 2_000_000):  # pragma: no cover - needs credentials
        """The result a page at a time: full MIMIC's hourly chart table is ~10^8 rows, too many
        to hold as one frame on a 16 GB laptop; ``extract.build_store`` places each page."""
        for df in self._client().query(sql).result(page_size=page_rows).to_dataframe_iterable():
            for c in set(df.columns) & TIME_COLS:
                df[c] = pd.to_datetime(df[c])
            yield df

    # ---- the same interface as CsvSource, computed server-side (needs credentials) ---- #
    def _q(self, sql: str) -> pd.DataFrame:  # pragma: no cover
        df = self.query(sql)
        for c in set(df.columns) & TIME_COLS:
            df[c] = pd.to_datetime(df[c])
        return df

    def icustays(self):  # pragma: no cover
        return self._q(f"SELECT subject_id, hadm_id, stay_id, first_careunit, intime, outtime, los "
                       f"FROM `{self.icu}.icustays`")

    def patients(self):  # pragma: no cover
        return self._q(f"SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group "
                       f"FROM `{self.hosp}.patients`")

    def admissions(self):  # pragma: no cover
        return self._q(f"SELECT subject_id, hadm_id, admittime, dischtime, admission_type, admission_location, "
                       f"hospital_expire_flag, insurance, race FROM `{self.hosp}.admissions`")

    def services(self):  # pragma: no cover
        return self._q(f"SELECT subject_id, hadm_id, transfertime, curr_service FROM `{self.hosp}.services`")

    def diagnoses(self):  # pragma: no cover
        return self._q(f"SELECT subject_id, hadm_id, icd_code, icd_version FROM `{self.hosp}.diagnoses_icd`")

    def d_items(self):  # pragma: no cover
        return self._q(f"SELECT * FROM `{self.icu}.d_items`")

    def d_labitems(self):  # pragma: no cover
        return self._q(f"SELECT * FROM `{self.hosp}.d_labitems`")

    def binned_chart(self, stays):  # pragma: no cover
        return self._pages(render_chart_sql(self.icu, self.stay_filter(stays)))

    def binned_labs(self, stays):  # pragma: no cover
        return self._pages(render_labs_sql(self.hosp, self.icu, self.stay_filter(stays)))

    def binned_uo(self, stays):  # pragma: no cover
        return self._pages(render_uo_sql(self.icu, self.stay_filter(stays)))

    def interval_hours(self, stays, kind):  # pragma: no cover
        return self._pages(render_interval_sql(self.icu, kind, self.stay_filter(stays)))

    def hourly_amounts(self, stays):  # pragma: no cover
        return self._pages(render_amount_sql(self.icu, self.stay_filter(stays)))

    def event_times(self, stays, kind):  # pragma: no cover
        df = self._q(render_events_sql(self.hosp, self.icu, kind, self.stay_filter(stays)))
        if kind == "abx":   # the antibiotic name / route filter is applied here, identically
            df = df[[K.is_antibiotic(d, r) for d, r in zip(df.drug, df.route)]]
        if kind in ("culture", "abx"):
            return typed_events(df, kind)
        return df[["stay_id", "t"]].sort_values(["stay_id", "t"]).reset_index(drop=True)

    def first_weight(self, stays):  # pragma: no cover
        df = self._q(f"""SELECT stay_id, ARRAY_AGG(patientweight ORDER BY starttime LIMIT 1)[OFFSET(0)] AS w
                         FROM `{self.icu}.inputevents` WHERE patientweight IS NOT NULL GROUP BY stay_id""")
        w = df.set_index("stay_id").w.astype(float)
        return w.where((w > 20) & (w < 400))

    # SQL renderers are plain functions of the catalog so they can be tested offline
    def sql_binned_chart(self, stay_filter: str = "TRUE") -> str:
        return render_chart_sql(self.icu, stay_filter)

    def sql_binned_labs(self, stay_filter: str = "TRUE") -> str:
        return render_labs_sql(self.hosp, self.icu, stay_filter)


def _case_name(concepts: Iterable[K.Concept], col: str = "itemid") -> str:
    whens = [f"WHEN {col} IN ({', '.join(map(str, c.itemids))}) THEN '{c.name}'" for c in concepts]
    return "CASE " + " ".join(whens) + " END"


def _exam_value() -> str:
    """The bedside exam's text -> ordinal coding (the CsvSource mapping), else valuenum."""
    whens = []
    for c in K.EXAM:
        if c.kind == "chart_text":
            cases = " ".join(f"WHEN '{t}' THEN {v}" for t, v in c.text_map)
            whens.append(f"WHEN ce.itemid IN ({', '.join(map(str, c.itemids))}) "
                         f"THEN CASE UPPER(TRIM(ce.value)) {cases} END")
    return " ".join(whens)


def render_chart_sql(icu: str, stay_filter: str = "TRUE") -> str:
    ids = ", ".join(str(i) for c in K.CHART_ALL for i in c.itemids)
    bounds = " OR ".join(f"(name = '{c.name}' AND v BETWEEN {c.lo} AND {c.hi})" for c in K.CHART_ALL)
    worst = ", ".join(f"'{c.name}'" for c in K.CHART_ALL if c.agg == "max")
    extremes = "\n".join(
        f"""UNION ALL
SELECT stay_id, CAST(FLOOR(h) AS INT64) AS hr, '{base}_{how}' AS name, {how.upper()}(value) AS value, COUNT(*) AS n
FROM ok WHERE name = '{base}' AND h >= 0 AND h < {K.MAX_HOURS}
GROUP BY stay_id, hr""" for base, how in K.EXTREMES.items())
    return f"""
WITH ev AS (
  SELECT ce.stay_id, ce.charttime, {_case_name(K.CHART_ALL, 'ce.itemid')} AS name,
         CASE WHEN ce.itemid = 223900 AND ce.value = 'No Response-ETT' THEN 0.0
              {_exam_value()}
              ELSE ce.valuenum END AS v,
         DATETIME_DIFF(ce.charttime, s.intime, SECOND) / 3600.0 AS h
  FROM `{icu}.chartevents` ce JOIN `{icu}.icustays` s USING (stay_id)
  WHERE ce.itemid IN ({ids}) AND ({stay_filter})
), ok AS (
  SELECT stay_id, charttime, h,
         CASE WHEN name IN ('TempF', 'TempC') THEN 'Temp' ELSE name END AS name,
         CASE WHEN name = 'TempF' THEN (v - 32) * 5 / 9
              WHEN name = 'FiO2' AND v > 1 THEN v / 100 ELSE v END AS value
  FROM ev WHERE v IS NOT NULL AND ({bounds})
)
SELECT stay_id, CAST(FLOOR(h) AS INT64) AS hr, name,
       IF(name IN ({worst}), MAX(value),
          ARRAY_AGG(value ORDER BY charttime DESC LIMIT 1)[OFFSET(0)]) AS value, COUNT(*) AS n
FROM ok
WHERE h >= 0 AND h < {K.MAX_HOURS} AND (name != 'FiO2' OR value BETWEEN 0.21 AND 1.0)
GROUP BY stay_id, hr, name
{extremes}
"""


def _text_value(concepts) -> str:
    """valuenum, or the ordinal of a text-coded result (urinalysis) — the CsvSource mapping."""
    whens = []
    for c in concepts:
        cases = " ".join(f"WHEN '{t}' THEN {v}" for t, v in c.text_map)
        whens.append(f"WHEN le.itemid IN ({', '.join(map(str, c.itemids))}) "
                     f"THEN CASE UPPER(TRIM(le.value)) {cases} END")
    return "CASE " + " ".join(whens) + " ELSE le.valuenum END"


def render_labs_sql(hosp: str, icu: str, stay_filter: str = "TRUE") -> str:
    cons = K.LAB + K.LAB_TEXT
    ids = ", ".join(str(i) for c in cons for i in c.itemids)
    bounds = " OR ".join(f"(name = '{c.name}' AND v BETWEEN {c.lo} AND {c.hi})" for c in cons)
    return f"""
WITH ev AS (
  SELECT s.stay_id, le.charttime, {_case_name(cons, 'le.itemid')} AS name, {_text_value(K.LAB_TEXT)} AS v,
         DATETIME_DIFF(le.charttime, s.intime, SECOND) / 3600.0 AS h
  FROM `{hosp}.labevents` le JOIN `{icu}.icustays` s ON le.hadm_id = s.hadm_id
  WHERE le.itemid IN ({ids}) AND ({stay_filter})
), ok AS (SELECT * FROM ev WHERE v IS NOT NULL AND ({bounds})
          AND h >= -{PRE_ICU_LAB_HOURS} AND h < {K.MAX_HOURS})
SELECT stay_id, GREATEST(CAST(FLOOR(h) AS INT64), 0) AS hr, name,
       ARRAY_AGG(v ORDER BY charttime DESC LIMIT 1)[OFFSET(0)] AS value, COUNT(*) AS n
FROM ok GROUP BY stay_id, hr, name
"""


def render_uo_sql(icu: str, stay_filter: str = "TRUE") -> str:
    items = ", ".join(map(str, K.UO_ITEMS + (K.UO_IRRIGANT_IN,)))
    return f"""
SELECT oe.stay_id, CAST(FLOOR(DATETIME_DIFF(oe.charttime, s.intime, SECOND) / 3600.0) AS INT64) AS hr,
       'UrineOutput' AS name,
       SUM(CASE WHEN oe.itemid = {K.UO_IRRIGANT_IN} THEN -oe.value ELSE oe.value END) AS value
FROM `{icu}.outputevents` oe JOIN `{icu}.icustays` s USING (stay_id)
WHERE oe.itemid IN ({items}) AND ({stay_filter})
GROUP BY stay_id, hr, name
HAVING hr >= 0 AND hr < {K.MAX_HOURS}
"""


def render_interval_sql(icu: str, kind: str, stay_filter: str = "TRUE") -> str:
    if kind == "vaso":
        cons, table = K.VASO, "inputevents"
        val = ("CASE WHEN LOWER(e.rateuom) = 'mcg/min' THEN e.rate / e.patientweight ELSE e.rate END")
        extra = "AND COALESCE(e.statusdescription, '') != 'Rewritten'"
    elif kind == "sedation":
        cons = [K.Concept("Sedation", "sedation", c.itemids) for c in K.TREAT_INTERVAL if c.kind == "sedation"]
        table, val = "inputevents", "1.0"
        extra = "AND COALESCE(e.statusdescription, '') != 'Rewritten'"
    elif kind == "dialysis":
        cons, table, val, extra = [c for c in K.TREAT_INTERVAL if c.kind == "dialysis"], "procedureevents", "1.0", ""
    else:
        cons, table, val, extra = K.VENT, "procedureevents", "1.0", ""
    ids = ", ".join(str(i) for c in cons for i in c.itemids)
    bounds = " OR ".join(f"(name = '{c.name}' AND value BETWEEN {c.lo} AND {c.hi})" for c in cons) \
        if kind == "vaso" else "TRUE"
    return f"""
WITH ev AS (
  SELECT e.stay_id, {_case_name(cons, 'e.itemid')} AS name, {val} AS value,
         GREATEST(CAST(FLOOR(DATETIME_DIFF(e.starttime, s.intime, SECOND) / 3600.0) AS INT64), 0) AS h0,
         LEAST(CAST(FLOOR(DATETIME_DIFF(e.endtime, s.intime, SECOND) / 3600.0 - 1e-9) AS INT64),
               {K.MAX_HOURS - 1}) AS h1
  FROM `{icu}.{table}` e JOIN `{icu}.icustays` s USING (stay_id)
  WHERE e.itemid IN ({ids}) {extra} AND ({stay_filter})
)
SELECT stay_id, hr, name, MAX(value) AS value
FROM ev, UNNEST(GENERATE_ARRAY(h0, h1)) AS hr
WHERE h1 >= h0 AND value IS NOT NULL AND ({bounds})
GROUP BY stay_id, hr, name
"""


def render_amount_sql(icu: str, stay_filter: str = "TRUE") -> str:
    """Volume started per hour — the CsvSource ``hourly_amounts`` binning."""
    ids = ", ".join(str(i) for c in K.TREAT_AMOUNT for i in c.itemids)
    bounds = " OR ".join(f"(name = '{c.name}' AND value BETWEEN {c.lo} AND {c.hi})" for c in K.TREAT_AMOUNT)
    return f"""
WITH ev AS (
  SELECT e.stay_id, {_case_name(K.TREAT_AMOUNT, 'e.itemid')} AS name, e.amount AS value,
         CAST(FLOOR(DATETIME_DIFF(e.starttime, s.intime, SECOND) / 3600.0) AS INT64) AS hr
  FROM `{icu}.inputevents` e JOIN `{icu}.icustays` s USING (stay_id)
  WHERE e.itemid IN ({ids}) AND COALESCE(e.statusdescription, '') != 'Rewritten' AND ({stay_filter})
)
SELECT stay_id, hr, name, SUM(value) AS value
FROM ev WHERE value IS NOT NULL AND ({bounds}) AND hr >= 0 AND hr < {K.MAX_HOURS}
GROUP BY stay_id, hr, name
"""


def render_events_sql(hosp: str, icu: str, kind: str, stay_filter: str = "TRUE") -> str:
    if kind == "abx_admin":
        return f"""
SELECT e.stay_id, DATETIME_DIFF(e.starttime, s.intime, SECOND) / 3600.0 AS t
FROM `{icu}.inputevents` e JOIN `{icu}.icustays` s USING (stay_id)
WHERE e.ordercategoryname = '{K.ABX_ADMIN_CATEGORY}' AND COALESCE(e.statusdescription, '') != 'Rewritten'
  AND ({stay_filter})
  AND DATETIME_DIFF(e.starttime, s.intime, SECOND) / 3600.0
      BETWEEN 0 AND LEAST(DATETIME_DIFF(s.outtime, s.intime, SECOND) / 3600.0, {K.MAX_HOURS})
"""
    if kind == "culture":
        return f"""
SELECT DISTINCT s.stay_id, m.spec_type_desc,
       DATETIME_DIFF(COALESCE(m.charttime, m.chartdate), s.intime, SECOND) / 3600.0 AS t
FROM `{hosp}.microbiologyevents` m JOIN `{icu}.icustays` s ON m.hadm_id = s.hadm_id
WHERE ({stay_filter}) AND DATETIME_DIFF(COALESCE(m.charttime, m.chartdate), s.intime, SECOND) / 3600.0
      BETWEEN -{EVENT_LOOKBACK_HOURS} AND LEAST(DATETIME_DIFF(s.outtime, s.intime, SECOND) / 3600.0, {K.MAX_HOURS})
"""
    return f"""
SELECT s.stay_id, p.drug, p.route,
       DATETIME_DIFF(p.starttime, s.intime, SECOND) / 3600.0 AS t
FROM `{hosp}.prescriptions` p JOIN `{icu}.icustays` s ON p.hadm_id = s.hadm_id
WHERE p.starttime IS NOT NULL AND ({stay_filter})
  AND DATETIME_DIFF(p.starttime, s.intime, SECOND) / 3600.0
      BETWEEN -{EVENT_LOOKBACK_HOURS} AND LEAST(DATETIME_DIFF(s.outtime, s.intime, SECOND) / 3600.0, {K.MAX_HOURS})
"""
