"""Events -> the hourly grid: a ``RawStore`` the GENESIS feature engine runs on unchanged.

One row per ICU hour, hour 0 = [intime, intime + 1 h). Every stay — septic or not — is
censored by the same rule (ICU discharge or ``MAX_HOURS``, whichever is first), so record
length cannot carry the label: the CinC 2019 artifact (never-septic records stopped at
~60 h, septic ones at onset + 3 h) is impossible by construction, and the Ledger harness
re-checks it with a tripwire anyway.

Channel conventions: measured variables are NaN when not measured that hour; vasopressor
rates and ventilation are dense (0 = off); Culture / Antibiotic are 1 in the hour an order
happened. ``tiers`` maps every channel to P / T / S.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.genesis.rawstore import RawStore
from backend.ml.mimic import concepts as K
from backend.ml.mimic import sepsis3 as S3
from backend.ml.mimic.sources import CsvSource

CACHE_DIR = os.path.join(config.ARTIFACTS_DIR, "mimic_cache")

VITAL_CH = ["HR", "SBP", "DBP", "MAP", "Resp", "O2Sat", "Temp", "EtCO2"]
GCS_CH = ["GCS_eye", "GCS_verbal", "GCS_motor", "GCS"]
LAB_CH = [c.name for c in K.LAB]
VASO_CH = [c.name for c in K.VASO]
VENT_CH = [c.name for c in K.VENT]
SUSP_CH = ["Culture", "Antibiotic"]
SOFA_CH = ["SOFA_P", "SOFA"]                      # hourly SOFA as a causal input channel
DYNAMIC = VITAL_CH + ["FiO2"] + GCS_CH + LAB_CH + ["UrineOutput"] + VASO_CH + VENT_CH + \
    SUSP_CH + SOFA_CH
UNITS = {"Medical Intensive Care Unit (MICU)": "Unit_MICU",
         "Surgical Intensive Care Unit (SICU)": "Unit_SICU",
         "Cardiac Vascular Intensive Care Unit (CVICU)": "Unit_CVICU",
         "Coronary Care Unit (CCU)": "Unit_CCU",
         "Trauma SICU (TSICU)": "Unit_TSICU",
         "Medical/Surgical Intensive Care Unit (MICU/SICU)": "Unit_MSICU",
         "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Unit_Neuro"}
STATIC = ["Age", "Gender", "HospAdmTime", "Weight", "AdmEmergency"] + \
    sorted(set(UNITS.values())) + ["Unit_Other", "ICULOS"]
YEAR_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]
MIN_HOURS = 4
RULES = ("measured", "rise", "absolute")
PRIMARY_RULE = "measured"


def tier_of(channel: str) -> str:
    if channel in VASO_CH or channel in VENT_CH or channel == "SOFA":
        return "T"
    if channel in SUSP_CH:
        return "S"
    return "P"


@dataclass
class MimicStore(RawStore):
    tiers: Dict[str, str] = field(default_factory=dict)
    static_columns: List[str] = field(default_factory=list)
    meta: Optional[pd.DataFrame] = None           # one row per stay, in store order
    onsets: Optional[pd.DataFrame] = None         # per stay, every label variant
    sofa: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def subject(self) -> np.ndarray:
        return self.meta.subject_id.to_numpy()

    def label_for(self, rule: str = PRIMARY_RULE, anchor: str = "min") -> np.ndarray:
        """Hourly challenge-style label under a chosen onset definition."""
        t = self.onsets[f"t_sepsis_{rule}_{anchor}"].to_numpy()[self.seg]
        return (np.isfinite(t) & (self.hour >= t - 6)).astype(np.int8)


# --------------------------------------------------------------------------- #
def cohort(src) -> pd.DataFrame:
    """Adult ICU stays of at least ``MIN_HOURS``; one grid per stay (subjects may repeat —
    the harness splits by subject)."""
    st = src.icustays().merge(src.patients(), on="subject_id").merge(
        src.admissions(), on=["subject_id", "hadm_id"])
    st["age"] = st.anchor_age + (st.admittime.dt.year - st.anchor_year)
    st["los_hours"] = (st.outtime - st.intime).dt.total_seconds() / 3600.0
    st = st[(st.age >= 18) & (st.los_hours >= MIN_HOURS)].copy()
    st["n_hours"] = np.minimum(np.ceil(st.los_hours), K.MAX_HOURS).astype(int)
    return st.sort_values("stay_id").reset_index(drop=True)


def build_store(src, verbose: bool = True) -> MimicStore:
    t0 = time.time()
    st = cohort(src)
    lens = st.n_hours.to_numpy(dtype=np.int64)
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    pos_of = dict(zip(st.stay_id, range(len(st))))
    N = int(lens.sum())
    cols = DYNAMIC + STATIC
    j = {c: i for i, c in enumerate(cols)}
    X = np.full((N, len(cols)), np.nan, dtype=np.float32)
    for c in VASO_CH + VENT_CH:
        X[:, j[c]] = 0.0

    def place(df: pd.DataFrame) -> None:
        if df is None or not len(df):
            return
        k = df.stay_id.map(pos_of)
        ok = k.notna().to_numpy()
        k = k[ok].astype(int).to_numpy()
        hr = df.hr.to_numpy()[ok].astype(int)
        inside = hr < lens[k]
        rows = starts[k[inside]] + hr[inside]
        names = df.name.to_numpy()[ok][inside]
        vals = df.value.to_numpy()[ok][inside].astype(np.float32)
        for nm in np.unique(names):
            m = names == nm
            X[rows[m], j[nm]] = vals[m]

    stays = st[["stay_id", "hadm_id", "intime", "outtime"]]
    parts = {}
    for name, fn in [("chart", lambda: src.binned_chart(stays)), ("labs", lambda: src.binned_labs(stays)),
                     ("uo", lambda: src.binned_uo(stays)),
                     ("vaso", lambda: src.interval_hours(stays, "vaso")),
                     ("vent", lambda: src.interval_hours(stays, "vent"))]:
        ts = time.time()
        parts[name] = fn()
        place(parts[name])
        if verbose:
            print(f"    {name:<6} {len(parts[name]):>9,} hourly values [{time.time()-ts:.1f}s]", flush=True)
    events = {k: src.event_times(stays, k) for k in ("culture", "abx")}
    for key, ch in (("culture", "Culture"), ("abx", "Antibiotic")):
        ev = events[key]
        hr = np.maximum(np.floor(ev.t.to_numpy()), 0).astype(int)     # pre-ICU orders known at hour 0
        place(pd.DataFrame({"stay_id": ev.stay_id, "hr": hr, "name": ch, "value": 1.0}))

    rs = np.repeat(starts, lens)
    hour = np.arange(N) - rs
    X[:, j["GCS"]] = S3.gcs_total(X[:, j["GCS_eye"]], X[:, j["GCS_verbal"]], X[:, j["GCS_motor"]], rs)

    # static
    per = {c: np.full(len(st), np.nan) for c in STATIC if c != "ICULOS"}
    per["Age"] = st.age.to_numpy(float)
    per["Gender"] = (st.gender == "M").to_numpy(float)
    per["HospAdmTime"] = ((st.admittime - st.intime).dt.total_seconds() / 3600.0).to_numpy()
    w = src.first_weight(stays)
    per["Weight"] = st.stay_id.map(w).to_numpy(float)
    per["AdmEmergency"] = st.admission_type.fillna("").str.contains("EMER|URGENT").to_numpy(float)
    unit = st.first_careunit.map(UNITS).fillna("Unit_Other")
    for u in sorted(set(UNITS.values())) + ["Unit_Other"]:
        per[u] = (unit == u).to_numpy(float)
    seg = np.repeat(np.arange(len(st)), lens)
    for c, v in per.items():
        X[:, j[c]] = v[seg]
    X[:, j["ICULOS"]] = hour + 1

    # SOFA (causal, 24-h worst) — also exposed as input channels by tier
    col = lambda nm: X[:, j[nm]].astype(np.float64)
    sofa = S3.hourly_sofa(col, rs, hour)
    X[:, j["SOFA_P"]] = sofa["sofa_total_P"]
    X[:, j["SOFA"]] = sofa["sofa_total"]

    # onset per stay under every label variant
    rows = []
    sofa_tot = sofa["sofa_total"]
    parts_now = S3.component_points_now(col, rs, hour)
    abx_by = {k: g.t.to_numpy() for k, g in events["abx"].groupby("stay_id")}
    cul_by = {k: g.t.to_numpy() for k, g in events["culture"].groupby("stay_id")}
    for i, sid in enumerate(st.stay_id):
        si = S3.suspicion_times(abx_by.get(sid, []), cul_by.get(sid, []))
        a, b = starts[i], starts[i] + lens[i]
        r = {"stay_id": sid}
        for rule in RULES:
            for anchor in ("min", "si", "sofa"):
                o = S3.onset_for_stay(sofa_tot[a:b], si, rule, anchor, parts_now[a:b])
                r[f"t_sepsis_{rule}_{anchor}"] = o.t_sepsis
                if anchor == "min":
                    r[f"t_si_{rule}"], r[f"t_sofa_{rule}"] = o.t_si, o.t_sofa
        r["first_si"] = float(si[0]) if len(si) else np.nan
        orders = np.r_[abx_by.get(sid, []), cul_by.get(sid, [])]
        r["first_order"] = float(orders.min()) if len(orders) else np.nan
        rows.append(r)
    onsets = pd.DataFrame(rows)

    t = onsets[f"t_sepsis_{PRIMARY_RULE}_min"].to_numpy()[seg]
    y = (np.isfinite(t) & (hour >= t - 6)).astype(np.int8)
    site = st.anchor_year_group.map({g: k for k, g in enumerate(YEAR_GROUPS)}).fillna(-1).to_numpy(np.int8)
    store = MimicStore(X=X, y=y, lens=lens, pid=st.stay_id.to_numpy(), site=site, columns=cols,
                       tiers={c: tier_of(c) for c in DYNAMIC} | {c: "P" for c in STATIC},
                       static_columns=list(STATIC),
                       meta=st[["subject_id", "hadm_id", "stay_id", "first_careunit", "anchor_year_group",
                                "age", "los_hours", "n_hours", "hospital_expire_flag"]].reset_index(drop=True),
                       onsets=onsets, sofa=sofa)
    if verbose:
        sep = np.isfinite(onsets[f"t_sepsis_{PRIMARY_RULE}_min"])
        print(f"    store: {len(st):,} stays ({st.subject_id.nunique():,} subjects), {N:,} hours, "
              f"{len(cols)} channels; Sepsis-3 ({PRIMARY_RULE} rule) in {int(sep.sum())} stays "
              f"({sep.mean():.1%}); label-1 hours {y.mean():.3%} [{time.time()-t0:.1f}s]", flush=True)
    return store


def subset(store: MimicStore, stays: np.ndarray) -> MimicStore:
    """A new store with only ``stays`` (indices, kept in the given order)."""
    stays = np.asarray(stays, dtype=np.int64)
    rows = np.concatenate([np.arange(store.starts[k], store.starts[k] + store.lens[k]) for k in stays]) \
        if len(stays) else np.zeros(0, np.int64)
    return MimicStore(X=store.X[rows], y=store.y[rows], lens=store.lens[stays], pid=store.pid[stays],
                      site=store.site[stays], columns=list(store.columns), tiers=dict(store.tiers),
                      static_columns=list(store.static_columns),
                      meta=store.meta.iloc[stays].reset_index(drop=True),
                      onsets=store.onsets.iloc[stays].reset_index(drop=True),
                      sofa={k: v[rows] for k, v in store.sofa.items()})


def sample_design(store: MimicStore, keep_rate: float, seed: int = 0):
    """Keep every stay that becomes septic under ANY rule and each never-septic stay with
    probability ``keep_rate``. Returns ``(sub_store, row_weights)`` with weight 1/keep_rate
    on kept never-septic rows, so design-weighted metrics estimate the full cohort."""
    rng = np.random.default_rng(seed)
    ever = np.zeros(store.n_patients, dtype=bool)
    for rule in RULES:
        ever |= store.onsets[f"t_sepsis_{rule}_min"].notna().to_numpy()
    keep = ever | (rng.random(store.n_patients) < keep_rate)
    idx = np.flatnonzero(keep)
    sub = subset(store, idx)
    w_stay = np.where(ever[idx], 1.0, 1.0 / keep_rate)
    return sub, w_stay[sub.seg]


def save(store: MimicStore, tag: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{tag}_store.npz")
    np.savez(path, X=store.X, y=store.y, lens=store.lens, pid=store.pid, site=store.site,
             columns=np.array(store.columns), static_columns=np.array(store.static_columns),
             tier_keys=np.array(list(store.tiers)), tier_vals=np.array(list(store.tiers.values())),
             **{f"sofa__{k}": v.astype(np.float32) for k, v in store.sofa.items()})
    store.meta.to_pickle(os.path.join(CACHE_DIR, f"{tag}_meta.pkl"))
    store.onsets.to_pickle(os.path.join(CACHE_DIR, f"{tag}_onsets.pkl"))
    return path


def load(tag: str) -> MimicStore:
    path = os.path.join(CACHE_DIR, f"{tag}_store.npz")
    d = np.load(path, allow_pickle=False)
    return MimicStore(X=d["X"], y=d["y"], lens=d["lens"], pid=d["pid"], site=d["site"],
                      columns=[str(c) for c in d["columns"]],
                      tiers=dict(zip([str(k) for k in d["tier_keys"]], [str(v) for v in d["tier_vals"]])),
                      static_columns=[str(c) for c in d["static_columns"]],
                      meta=pd.read_pickle(os.path.join(CACHE_DIR, f"{tag}_meta.pkl")),
                      onsets=pd.read_pickle(os.path.join(CACHE_DIR, f"{tag}_onsets.pkl")),
                      sofa={k[6:]: d[k] for k in d.files if k.startswith("sofa__")})


DEMO_ROOT = os.path.join(config.REPO_ROOT, "data", "mimic_demo", "mimic-iv-clinical-database-demo-2.2")


if __name__ == "__main__":  # pragma: no cover
    s = build_store(CsvSource(DEMO_ROOT))
    print(save(s, "demo"))
