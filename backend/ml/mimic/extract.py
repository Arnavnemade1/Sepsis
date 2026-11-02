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
from backend.ml.mimic import history as H
from backend.ml.mimic import sepsis3 as S3
from backend.ml.mimic.sources import CsvSource

#: credentialed runs keep their grid beside the lake, outside the repository (backend.ml.physionet)
CACHE_DIR = os.environ.get("MIMIC_CACHE_DIR") or os.path.join(config.ARTIFACTS_DIR, "mimic_cache")

VITAL_CH = ["HR", "SBP", "DBP", "MAP", "Resp", "O2Sat", "Temp", "EtCO2"]
HEMO_CH = [c.name for c in K.HEMO]                # central venous pressure
GCS_CH = ["GCS_eye", "GCS_verbal", "GCS_motor", "GCS"]
LAB_CH = [c.name for c in K.LAB + K.LAB_TEXT]
#: the v3 labs: sparse, so the feature engine gives them its light set (``LIGHT_CH``)
NEW_LAB_CH = [c.name for c in K.LAB_V3 + K.LAB_TEXT]
VENT_SET_CH = [c.name for c in K.VENT_SET]        # ventilator settings (T)
VASO_CH = [c.name for c in K.VASO]
VENT_CH = [c.name for c in K.VENT]
TREAT_CH = [c.name for c in K.TREAT_AMOUNT + K.TREAT_INTERVAL]   # fluids, blood, sedation, dialysis (T)
EXAM_CH = [c.name for c in K.EXAM]                # the bedside exam (P)
EXT_CH = [f"{b}_{h}" for b, h in K.EXTREMES.items()]   # within-hour extremes of the vitals (P)
CULT_CH = list(K.SPECIMENS) + ["CultureOther"]    # which workup (S)
ABX_CLASS_CH = ["AbxMRSA", "AbxBroad"]            # which spectrum (S)
SUSP_CH = ["Culture", "Antibiotic", "AbxAdmin"] + CULT_CH + ABX_CLASS_CH
SOFA_CH = ["SOFA_P", "SOFA"]                      # hourly SOFA as a causal input channel
DYNAMIC = VITAL_CH + EXT_CH + HEMO_CH + ["FiO2"] + GCS_CH + EXAM_CH + LAB_CH + ["UrineOutput"] + \
    VENT_SET_CH + VASO_CH + VENT_CH + TREAT_CH + SUSP_CH + SOFA_CH
#: dense channels: 0 = off / none given, never missing
DENSE_CH = VASO_CH + VENT_CH + TREAT_CH
#: channels whose feature set is the light one (value, age, counts, previous, change)
LIGHT_CH = NEW_LAB_CH + HEMO_CH + VENT_SET_CH + TREAT_CH + EXAM_CH + EXT_CH + ["AbxAdmin"] + CULT_CH + ABX_CLASS_CH
UNITS = {"Medical Intensive Care Unit (MICU)": "Unit_MICU",
         "Surgical Intensive Care Unit (SICU)": "Unit_SICU",
         "Cardiac Vascular Intensive Care Unit (CVICU)": "Unit_CVICU",
         "Coronary Care Unit (CCU)": "Unit_CCU",
         "Trauma SICU (TSICU)": "Unit_TSICU",
         "Medical/Surgical Intensive Care Unit (MICU/SICU)": "Unit_MSICU",
         "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Unit_Neuro"}
#: what was known on arrival (``history``): admission source, admitting service, and
#: comorbidities / prior-admission recency from admissions discharged BEFORE this one
CONTEXT = H.SRC_COLS + H.SVC_COLS + H.HX_COLS
STATIC = ["Age", "Gender", "HospAdmTime", "Weight", "AdmEmergency"] + \
    sorted(set(UNITS.values())) + ["Unit_Other"] + CONTEXT + ["ICULOS"]
YEAR_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]
MIN_HOURS = 4
#: Bumped whenever the grid's contents change (v2: SOFA_P without the ventilator; audit-only
#: attributes in ``meta``), so a stale cache is rebuilt instead of silently reused.
STORE_VERSION = 5                 # v3: inflammation labs, urinalysis, CVP, ventilator settings,
                                  # treatments, IV antibiotic administrations, arrival context;
                                  # v4: the bedside exam, within-hour extremes (SOFA reads the
                                  # hour's lowest MAP), culture specimen groups, antibiotic spectrum;
                                  # v5: same-timestamp ties have one definition; no interval from
                                  # past the grid written into its last hour
#: Per-stay attributes kept for the subgroup audit ONLY. They live in ``meta``, never in the
#: grid ``X``, so no feature — and therefore no model — can read them.
AUDIT_ATTRS = ["sex", "age_band", "race_group", "insurance", "admission_type", "careunit",
               "anchor_year_group"]
RULES = ("measured", "rise", "absolute")
PRIMARY_RULE = "measured"


def tier_of(channel: str) -> str:
    if channel in VASO_CH or channel in VENT_CH or channel in VENT_SET_CH or channel in TREAT_CH \
            or channel == "SOFA":
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
    for c in DENSE_CH:
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
                     ("vent", lambda: src.interval_hours(stays, "vent")),
                     ("sedate", lambda: src.interval_hours(stays, "sedation")),
                     ("dialys", lambda: src.interval_hours(stays, "dialysis")),
                     ("fluids", lambda: src.hourly_amounts(stays))]:
        ts = time.time()
        n = 0
        for page in _pages(fn()):                  # BigQuery hands over pages; a CSV one frame
            place(page)
            n += len(page)
        parts[name] = n
        if verbose:
            print(f"    {name:<6} {n:>9,} hourly values [{time.time()-ts:.1f}s]", flush=True)
    events = {k: src.event_times(stays, k) for k in ("culture", "abx", "abx_admin")}
    typed = {"Culture": events["culture"], "Antibiotic": events["abx"], "AbxAdmin": events["abx_admin"]}
    for g in CULT_CH:                                  # which workup
        typed[g] = events["culture"][events["culture"].group == g]
    typed["AbxMRSA"] = events["abx"][events["abx"].mrsa]
    typed["AbxBroad"] = events["abx"][events["abx"].broad]
    for ch, ev in typed.items():
        hr = np.maximum(np.floor(ev.t.to_numpy()), 0).astype(int)     # pre-ICU orders known at hour 0
        place(pd.DataFrame({"stay_id": ev.stay_id, "hr": hr, "name": ch, "value": 1.0}))

    rs = np.repeat(starts, lens)
    hour = np.arange(N) - rs

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
    ts = time.time()
    ctx = H.admission_context(st, src.services()).set_index("stay_id")
    hist = H.prior_history(st, src.admissions(), src.diagnoses()).set_index("stay_id")
    for c in CONTEXT:
        per[c] = st.stay_id.map((ctx if c in ctx.columns else hist)[c]).to_numpy(float)
    if verbose:
        print(f"    context {len(ctx):>8,} stays (source, service, prior-admission history) "
              f"[{time.time()-ts:.1f}s]", flush=True)
    seg = np.repeat(np.arange(len(st)), lens)
    for c, v in per.items():
        X[:, j[c]] = v[seg]
    X[:, j["ICULOS"]] = hour + 1

    # GCS total and SOFA (causal, 24-h worst) — derived channels, exposed as inputs by tier
    sofa = derive_channels(X, cols, lens)
    col = lambda nm: X[:, j[nm]].astype(np.float64)

    # onset per stay under every label variant
    rows = []
    sofa_tot = sofa["sofa_total"]
    parts_now = S3.component_points_now(col, rs, hour)
    abx_by = {k: g.t.to_numpy() for k, g in events["abx"].groupby("stay_id")}
    cul_by = {k: np.unique(g.t.to_numpy()) for k, g in events["culture"].groupby("stay_id")}
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
                       meta=_meta(st),
                       onsets=onsets, sofa=sofa)
    if verbose:
        sep = np.isfinite(onsets[f"t_sepsis_{PRIMARY_RULE}_min"])
        print(f"    store: {len(st):,} stays ({st.subject_id.nunique():,} subjects), {N:,} hours, "
              f"{len(cols)} channels; Sepsis-3 ({PRIMARY_RULE} rule) in {int(sep.sum())} stays "
              f"({sep.mean():.1%}); label-1 hours {y.mean():.3%} [{time.time()-t0:.1f}s]", flush=True)
    return store


def _pages(result):
    """A frame, or an iterable of frames (BigQuery's paged results), as frames."""
    if isinstance(result, pd.DataFrame):
        yield result
    else:
        yield from result


def derive_channels(X: np.ndarray, columns: List[str], lens: np.ndarray) -> Dict[str, np.ndarray]:
    """Recompute the DERIVED channels of a grid in place — total GCS from its components,
    SOFA_P and SOFA from everything — and return the SOFA components. One function for the
    real grid and every counterfactual one, so an edited input can never leave a stale
    derived channel behind."""
    j = {c: i for i, c in enumerate(columns)}
    starts = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
    rs = np.repeat(starts, lens)
    hour = np.arange(len(rs)) - rs
    X[:, j["GCS"]] = S3.gcs_total(X[:, j["GCS_eye"]], X[:, j["GCS_verbal"]], X[:, j["GCS_motor"]], rs)
    sofa = S3.hourly_sofa(lambda nm: X[:, j[nm]].astype(np.float64), rs, hour)
    X[:, j["SOFA_P"]] = sofa["sofa_total_P"]
    X[:, j["SOFA"]] = sofa["sofa_total"]
    return sofa


def rederive(store: MimicStore, X: np.ndarray) -> MimicStore:
    """The same stays with an edited grid ``X`` (base channels), derived channels recomputed.
    Labels, onsets and metadata are the ORIGINAL ones: a counterfactual changes what a model
    sees, never what happened."""
    X = np.array(X, dtype=np.float32, copy=True)
    sofa = derive_channels(X, store.columns, store.lens)
    return MimicStore(X=X, y=store.y, lens=store.lens, pid=store.pid, site=store.site,
                      columns=list(store.columns), tiers=dict(store.tiers),
                      static_columns=list(store.static_columns), meta=store.meta,
                      onsets=store.onsets, sofa=sofa)


def _race_group(r: str) -> str:
    r = str(r).upper()
    if r.startswith(("WHITE", "PORTUGUESE")):
        return "White"
    if r.startswith("BLACK"):
        return "Black"
    if r.startswith(("HISPANIC", "SOUTH AMERICAN")):
        return "Hispanic"
    if r.startswith("ASIAN"):
        return "Asian"
    if r.startswith(("UNKNOWN", "UNABLE", "PATIENT DECLINED")) or r in ("NAN", ""):
        return "Unknown"
    return "Other"


def _meta(st: pd.DataFrame) -> pd.DataFrame:
    m = st[["subject_id", "hadm_id", "stay_id", "first_careunit", "anchor_year_group",
            "age", "los_hours", "n_hours", "hospital_expire_flag"]].reset_index(drop=True).copy()
    m["sex"] = st.gender.map({"M": "Male", "F": "Female"}).fillna("Unknown").to_numpy()
    m["age_band"] = pd.cut(st.age.to_numpy(float), [0, 45, 65, 80, 200], right=False,
                           labels=["18-44", "45-64", "65-79", "80+"]).astype(str)
    m["race_group"] = [_race_group(r) for r in st.get("race", pd.Series([""] * len(st)))]
    m["insurance"] = st.get("insurance", pd.Series(["Unknown"] * len(st))).fillna("Unknown").to_numpy()
    m["admission_type"] = np.where(st.admission_type.fillna("").str.contains("EMER|URGENT"),
                                   "Emergency/urgent", "Elective/other")
    m["careunit"] = st.first_careunit.map(UNITS).fillna("Unit_Other").str.replace("Unit_", "").to_numpy()
    return m


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


def trim_septic(store: MimicStore, margin: int = 4) -> MimicStore:
    """Drop the hours no model trains on and no metric scores: in a stay septic under the
    PRIMARY rule, everything after its LATEST onset across all 9 definitions + 3 h + ``margin``
    (so label sensitivity keeps every row it scores). Every feature and model is causal, so
    no prediction on a kept row can change (``verify_mimic`` checks it bit for bit) — and a
    septic stay's post-onset days are most of the full cohort's hours. Stays never septic
    under the primary rule are untouched: all their hours are negatives. The original length
    stays in ``meta.n_hours`` (the record-length tripwire reads it)."""
    prim = np.isfinite(store.onsets[f"t_sepsis_{PRIMARY_RULE}_min"].to_numpy())
    cols = [c for c in store.onsets.columns if c.startswith("t_sepsis_")]
    latest = np.nanmax(store.onsets[cols].to_numpy(dtype=float), axis=1)
    keep_len = store.lens.copy()
    cut = prim & np.isfinite(latest)
    keep_len[cut] = np.minimum(store.lens[cut], np.ceil(latest[cut]).astype(np.int64) + 3 + margin + 1)
    keep_len = np.maximum(keep_len, 1)
    rows = np.concatenate([np.arange(store.starts[k], store.starts[k] + keep_len[k]) for k in range(store.n_patients)])
    return MimicStore(X=store.X[rows], y=store.y[rows], lens=keep_len, pid=store.pid, site=store.site,
                      columns=list(store.columns), tiers=dict(store.tiers), static_columns=list(store.static_columns),
                      meta=store.meta, onsets=store.onsets, sofa={k: v[rows] for k, v in store.sofa.items()})


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


def _stem(tag: str) -> str:
    return os.path.join(CACHE_DIR, f"{tag}_v{STORE_VERSION}")


def save(store: MimicStore, tag: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _stem(tag) + "_store.npz"
    np.savez(path, X=store.X, y=store.y, lens=store.lens, pid=store.pid, site=store.site,
             columns=np.array(store.columns), static_columns=np.array(store.static_columns),
             tier_keys=np.array(list(store.tiers)), tier_vals=np.array(list(store.tiers.values())),
             **{f"sofa__{k}": v.astype(np.float32) for k, v in store.sofa.items()})
    store.meta.to_pickle(_stem(tag) + "_meta.pkl")
    store.onsets.to_pickle(_stem(tag) + "_onsets.pkl")
    return path


def load(tag: str) -> MimicStore:
    path = _stem(tag) + "_store.npz"
    d = np.load(path, allow_pickle=False)
    return MimicStore(X=d["X"], y=d["y"], lens=d["lens"], pid=d["pid"], site=d["site"],
                      columns=[str(c) for c in d["columns"]],
                      tiers=dict(zip([str(k) for k in d["tier_keys"]], [str(v) for v in d["tier_vals"]])),
                      static_columns=[str(c) for c in d["static_columns"]],
                      meta=pd.read_pickle(_stem(tag) + "_meta.pkl"),
                      onsets=pd.read_pickle(_stem(tag) + "_onsets.pkl"),
                      sofa={k[6:]: d[k] for k in d.files if k.startswith("sofa__")})


DEMO_ROOT = os.environ.get("MIMIC_DEMO_ROOT") or \
    os.path.join(config.REPO_ROOT, "data", "mimic_demo", "mimic-iv-clinical-database-demo-2.2")


if __name__ == "__main__":  # pragma: no cover
    s = build_store(CsvSource(DEMO_ROOT))
    print(save(s, "demo"))
