"""The Ledger harness — every number carries its provenance, and every input pays its way.

Built from the GENESIS harness (docs/GENESIS.md) and extended by what CinC 2019 taught:

SUBJECT-GROUPED SPLITS. MIMIC patients return; a subject's stays never straddle a split.
The locked set is 20% of SUBJECTS, drawn once with a fixed seed, every access ledgered.

TIER FIREWALL. Every feature column inherits the tier of the channel it came from (P
physiology, T treatment, S suspicion). A model declares its tier; asking for a column
above it raises. The physiology number cannot be contaminated by clinicians' orders.

PRE-SUSPICION AUROC. The same predictions scored only on hours before the stay's first
culture or antibiotic order — the hours in which a model cannot be "detecting the order"
(the Epic Sepsis Model criticism) because there is no order yet.

EARLY-WARNING COHORT. Stays whose onset is within ``ONSET_MIN`` hours of ICU admission
(sepsis on arrival) are scored separately; the primary number is on stays where there was
something to warn about. Septic stays are scored up to onset + 3 h (the CinC utility
horizon); never-septic stays on every hour — no artificial truncation of either.

DESIGN WEIGHTS. Full MIMIC does not fit a 16 GB laptop at hourly density; never-septic
stays can be subsampled at rate r and every metric (AUROC, AUPRC, the decomposition,
hour-matched AUROC, the bootstrap) is computed with weights 1/r, which recovers the
full-cohort metric in expectation (AUROC is a ratio of weighted pair counts).

CLOCK CONTROL + DECOMPOSITION. Hour-matched AUROC and between / within-septic AUROC on
every result, as in GENESIS.

LABEL SENSITIVITY. The same predictions scored against every onset rule / anchor
(``sepsis3``), so a result that only holds under one definition is visible as such.

TEMPORAL VALIDATION. Train on earlier ``anchor_year_group``s, test on later ones — a
deployment question CinC could not ask.

TRIPWIRES. Clock-only AUROC; the share of onsets in the first hours per rule (the
data-arrival artifact); record length as a label predictor.

INFORMATION LEDGER. For each variable family: the AUROC lost when it is removed, and the
sufficiency test (features + score vs score) for the final model.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from backend.ml import config
from backend.ml.genesis import features as feat
from backend.ml.genesis.harness import Ranked
from backend.ml.genesis.rawstore import VITALS
from backend.ml.mimic import concepts as K
from backend.ml.mimic import extract as E

OUT_DIR = os.path.join(config.ARTIFACTS_DIR, "mimic")
ONSET_MIN = 4.0                 # onset earlier than this (hours after ICU admission) = on arrival
SCORE_AFTER_ONSET = 3.0         # septic stays scored up to onset + 3 h
LOCK_SEED = 29
LOCK_FRACTION = 0.2

#: variable families for the information ledger (channel -> family)
FAMILIES = {
    "vitals": E.VITAL_CH + ["FiO2"], "gcs": E.GCS_CH, "labs": E.LAB_CH, "urine": ["UrineOutput"],
    "sofa": E.SOFA_CH, "vasopressors": E.VASO_CH, "ventilation": E.VENT_CH,
    "suspicion": E.SUSP_CH,
}


def feature_spec(channels: Sequence[str]) -> feat.FeatureSpec:
    return feat.FeatureSpec(channels=list(channels), rolling=[c for c in VITALS if c in channels],
                            labs=[c for c in E.LAB_CH if c in channels],
                            static_cols=list(E.STATIC))


def channel_of(feature: str) -> Optional[str]:
    base = feature.split("__")[0]
    return base if base in E.DYNAMIC else None


def weighted_hour_matched(y, p, hour, w) -> float:
    """Pair-weighted mean of per-hour weighted AUROCs (the clock cannot contribute)."""
    order = np.lexsort((p, hour))
    hs, ys, ps, ws = hour[order], y[order], p[order], w[order]
    starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
    ends = np.r_[starts[1:], len(hs)]
    num = den = 0.0
    for a, b in zip(starts, ends):
        yy, ww = ys[a:b], ws[a:b]
        wp, wn = float(ww[yy == 1].sum()), float(ww[yy == 0].sum())
        if wp == 0 or wn == 0:
            continue
        num += Ranked(ps[a:b]).auc(yy, ww) * wp * wn
        den += wp * wn
    return num / den if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
@dataclass
class Ledger:
    store: E.MimicStore
    F: np.ndarray
    names: List[str]
    ftier: np.ndarray                         # (D,) 0/1/2 per feature column
    folds: np.ndarray                         # (P,) dev fold per stay, -1 locked
    rule: str = E.PRIMARY_RULE
    anchor: str = "min"
    design_w: Optional[np.ndarray] = None     # (N,) inverse sampling probability
    y: np.ndarray = field(init=False)
    onset: np.ndarray = field(init=False)     # (P,) onset hour of the stay under the rule
    train_mask: np.ndarray = field(init=False)
    eval_mask: np.ndarray = field(init=False)
    pre_susp: np.ndarray = field(init=False)

    def __post_init__(self):
        s = self.store
        self.onset = s.onsets[f"t_sepsis_{self.rule}_{self.anchor}"].to_numpy()
        self.y = s.label_for(self.rule, self.anchor)
        t = self.onset[s.seg]
        septic = np.isfinite(t)
        window = ~septic | (s.hour <= t + SCORE_AFTER_ONSET)
        self.train_mask = window
        early = septic & (t < ONSET_MIN)
        self.eval_mask = window & ~early
        first = s.onsets.first_order.to_numpy()[s.seg]
        self.pre_susp = ~np.isfinite(first) | (s.hour < first)
        if self.design_w is None:
            self.design_w = np.ones(s.n_rows)

    # ------------------------------------------------------------------ #
    @staticmethod
    def build(store: E.MimicStore, rule: str = E.PRIMARY_RULE, n_folds: int = 5,
              seed: int = 20260925, chunk_stays: Optional[int] = None,
              memmap_path: Optional[str] = None, dtype=np.float32,
              design_w: Optional[np.ndarray] = None) -> "Ledger":
        """Feature matrix over every tier. ``chunk_stays`` builds it a block of stays at a
        time into ``memmap_path`` (``dtype`` float16 halves the disk) — exact, because every
        feature is a function of its own stay's past only."""
        F, names = build_features(store, chunk_stays, memmap_path, dtype)
        tiers = []
        for n in names:
            ch = channel_of(n)
            tiers.append(K.TIER_ORDER[store.tiers[ch]] if ch else 0)
        return Ledger(store=store, F=F, names=names, ftier=np.array(tiers, dtype=np.int8),
                      folds=subject_folds(store, n_folds, seed), rule=rule, design_w=design_w)

    @property
    def n_folds(self) -> int:
        return int(self.folds.max()) + 1

    def columns(self, tier: str) -> np.ndarray:
        return np.flatnonzero(self.ftier <= K.TIER_ORDER[tier])

    def assert_tier(self, cols: np.ndarray, tier: str) -> None:
        bad = cols[self.ftier[cols] > K.TIER_ORDER[tier]]
        if len(bad):
            raise PermissionError(f"tier-{tier} model asked for {len(bad)} higher-tier column(s), "
                                  f"e.g. {self.names[bad[0]]!r}")

    def fold_rows(self, k: int, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        f = self.folds[self.store.seg]
        tr = np.flatnonzero((f >= 0) & (f != k) & self.train_mask)
        te = np.flatnonzero((f == k) & self.train_mask)
        return tr, te

    def lock_rows(self, purpose: str) -> np.ndarray:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "lock_ledger.jsonl"), "a") as fh:
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": purpose,
                                 "lock": lock_fingerprint(self.store, self.folds)}) + "\n")
        return np.flatnonzero((self.folds[self.store.seg] < 0) & self.train_mask)

    # ------------------------------------------------------------------ #
    def run(self, name: str, fit_predict: Callable, tier: str, verbose: bool = True) -> np.ndarray:
        """OOF predictions over the dev folds. ``fit_predict(ledger, cols, tr, te) -> p``."""
        cols = self.columns(tier)
        self.assert_tier(cols, tier)
        oof = np.full(self.store.n_rows, np.nan)
        for k in range(self.n_folds):
            tr, te = self.fold_rows(k)
            if not len(te):
                continue
            oof[te] = fit_predict(self, cols, tr, te)
        if verbose:
            s = self.score(oof)
            print(f"  {name:<28} [{tier}] AUROC {s['auroc']:.4f}  hour-matched {s['auroc_hourmatched']:.4f}  "
                  f"pre-suspicion {s['auroc_presusp']:.4f}  between {s['between']:.4f}  "
                  f"within {s['within_septic']:.4f}  AUPRC {s['auprc']:.4f}", flush=True)
        return oof

    def score(self, p: np.ndarray, rows: Optional[np.ndarray] = None,
              y: Optional[np.ndarray] = None) -> Dict[str, float]:
        s = self.store
        y = self.y if y is None else y
        rows = np.flatnonzero(self.eval_mask & np.isfinite(p) & (self.folds[s.seg] >= 0)) \
            if rows is None else rows
        yy, pp, ww, hh = y[rows], p[rows], self.design_w[rows], s.hour[rows]
        septic = np.isfinite(self.onset[s.seg[rows]])
        out: Dict[str, float] = {"n_rows": int(len(rows)), "n_pos": int(yy.sum()),
                                 "n_stays": int(len(np.unique(s.seg[rows])))}
        if yy.min() == yy.max():
            return out | {k: float("nan") for k in ("auroc", "between", "within_septic", "auprc",
                                                     "auroc_hourmatched", "auroc_presusp")}
        R = Ranked(pp)
        out["auroc"] = R.auc(yy, ww)
        out["between"] = R.auc(yy, np.where((yy == 0) & septic, 0.0, ww))
        out["within_septic"] = R.auc(yy, np.where(septic, ww, 0.0))
        out["auprc"] = float(average_precision_score(yy, pp, sample_weight=ww))
        out["auroc_hourmatched"] = weighted_hour_matched(yy, pp, hh, ww)
        pre = self.pre_susp[rows]
        out["auroc_presusp"] = R.auc(yy, np.where(pre, ww, 0.0)) if (yy[pre].min() < yy[pre].max()
                                                                     if pre.any() else False) else float("nan")
        return out

    def compare(self, a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Dict:
        """Paired bootstrap over SUBJECTS (design-weighted)."""
        s = self.store
        rows = np.flatnonzero(self.eval_mask & np.isfinite(a) & np.isfinite(b) & (self.folds[s.seg] >= 0))
        subj_codes, inv = np.unique(s.subject[s.seg[rows]], return_inverse=True)
        yy, ww = self.y[rows], self.design_w[rows]
        Ra, Rb = Ranked(a[rows]), Ranked(b[rows])
        rng = np.random.default_rng(seed)
        d = np.empty(n_boot)
        for i in range(n_boot):
            m = np.bincount(rng.integers(0, len(subj_codes), len(subj_codes)), minlength=len(subj_codes))
            w = ww * m[inv]
            d[i] = Ra.auc(yy, w) - Rb.auc(yy, w)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"d_auroc": Ra.auc(yy, ww) - Rb.auc(yy, ww), "ci": (float(lo), float(hi)),
                "significant": bool(lo > 0 or hi < 0)}

    # ------------------------------------------------------------------ #
    def label_sensitivity(self, p: np.ndarray) -> Dict[str, float]:
        """AUROC of the same predictions under every onset rule / anchor."""
        out = {}
        for rule in E.RULES:
            for anchor in ("min", "si", "sofa"):
                other = Ledger(self.store, self.F, self.names, self.ftier, self.folds, rule, anchor,
                               self.design_w)
                out[f"{rule}/{anchor}"] = other.score(p)["auroc"]
        return out

    def tripwires(self) -> Dict[str, float]:
        s = self.store
        rows = np.flatnonzero(self.eval_mask & (self.folds[s.seg] >= 0))
        clock = Ranked(s.hour[rows].astype(float)).auc(self.y[rows], self.design_w[rows])
        length = Ranked(s.lens[s.seg[rows]].astype(float)).auc(self.y[rows], self.design_w[rows])
        early = {}
        for rule in E.RULES:
            t = s.onsets[f"t_sepsis_{rule}_min"].to_numpy()
            sep = np.isfinite(t)
            early[rule] = float(np.mean(t[sep] < ONSET_MIN)) if sep.any() else float("nan")
        return {"clock_auroc": clock, "record_length_auroc": length,
                **{f"onset_within_{int(ONSET_MIN)}h_{r}": v for r, v in early.items()}}


def build_features(store: E.MimicStore, chunk_stays: Optional[int] = None,
                   memmap_path: Optional[str] = None, dtype=np.float32):
    spec = feature_spec(E.DYNAMIC)
    if not chunk_stays:
        F, names = feat.build(store, spec)
        return np.ascontiguousarray(F, dtype=dtype), names
    first = E.subset(store, np.arange(min(chunk_stays, store.n_patients)))
    F0, names = feat.build(first, spec)
    shape = (store.n_rows, len(names))
    F = (np.lib.format.open_memmap(memmap_path, mode="w+", dtype=dtype, shape=shape)
         if memmap_path else np.empty(shape, dtype=dtype))
    for a in range(0, store.n_patients, chunk_stays):
        idx = np.arange(a, min(a + chunk_stays, store.n_patients))
        part = E.subset(store, idx)
        Fc, nc = feat.build(part, spec)
        if nc != names:
            raise RuntimeError("feature names changed between chunks")
        r0 = store.starts[idx[0]]
        F[r0:r0 + part.n_rows] = Fc
    if memmap_path:
        F.flush()
    return F, names


def subject_folds(store: E.MimicStore, n_folds: int, seed: int) -> np.ndarray:
    """Per-stay fold id; subjects are the unit. The locked 20% of subjects is drawn first
    with its own seed, stratified by whether the subject ever becomes septic."""
    subj = store.subject
    uniq, inv = np.unique(subj, return_inverse=True)
    t = store.onsets[f"t_sepsis_{E.PRIMARY_RULE}_min"].to_numpy()
    ever = np.zeros(len(uniq), dtype=int)
    np.maximum.at(ever, inv, np.isfinite(t).astype(int))
    n_lock_split = int(round(1 / LOCK_FRACTION))
    lock_split = StratifiedKFold(n_splits=n_lock_split, shuffle=True, random_state=LOCK_SEED)
    _, locked = next(lock_split.split(np.zeros(len(uniq)), ever))
    fold_s = np.full(len(uniq), -1, dtype=np.int16)
    dev = np.setdiff1d(np.arange(len(uniq)), locked)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(dev)), ever[dev])):
        fold_s[dev[te]] = k
    return fold_s[inv]


def lock_fingerprint(store: E.MimicStore, folds: np.ndarray) -> str:
    ids = ",".join(sorted(map(str, np.unique(store.subject[folds < 0]))))
    return hashlib.sha256(ids.encode()).hexdigest()[:16]


def design_weights(store: E.MimicStore, keep_nonseptic: np.ndarray, rate: float) -> np.ndarray:
    """Row weights for a design that kept each never-septic stay with probability ``rate``."""
    t = store.onsets[f"t_sepsis_{E.PRIMARY_RULE}_min"].to_numpy()
    w_stay = np.where(np.isfinite(t), 1.0, 1.0 / rate)
    return w_stay[store.seg]
