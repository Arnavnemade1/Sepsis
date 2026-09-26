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

INFORMATION LEDGER. For each variable family: the AUROC lost when it is removed.

Around this core (docs/MIMIC.md §4-6):

* ``counterfactual`` — every model is re-scored on EDITED patients: the Care-Process
  Transplant (whose schedule, whose values) and the admission audit (rewrite the future,
  truncate the stay, scramble the neighbours — nothing earlier may move);
* ``infoflow`` — the tier firewall is MEASURED, not declared: covert channels from
  physiology columns into treatment state are found and a strict tier drops them;
* ``crossexam`` — the questions a skeptical intensivist would ask of any number;
* ``trust`` — a hash-chained, code-bound ledger, power-gated pre-registration and a
  Ladder-protected lock that can be consulted more than once without being learned.

A model enters as a :class:`Candidate` — a fit function the harness calls per fold, which
returns a predictor it can apply to ANY ledger-shaped view (the real patients or edited
ones). That one interface is what lets every counterfactual test run on every model.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from backend.ml import config
from backend.ml.genesis import features as feat
from backend.ml.genesis.harness import Ranked
from backend.ml.genesis.features import BLOOD_GAS
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
    "vitals": E.VITAL_CH + E.EXT_CH + E.HEMO_CH + ["FiO2"], "gcs": E.GCS_CH, "exam": E.EXAM_CH,
    "labs": [c for c in E.LAB_CH if c not in E.NEW_LAB_CH], "inflammation": E.NEW_LAB_CH,
    "urine": ["UrineOutput"], "sofa": E.SOFA_CH, "vasopressors": E.VASO_CH,
    "ventilation": E.VENT_CH + E.VENT_SET_CH, "treatments": E.TREAT_CH, "suspicion": E.SUSP_CH,
}
#: static families (dropped by column name, not by channel)
STATIC_FAMILIES = {"history": E.CONTEXT}


#: features that describe WHEN a channel was measured, not what it read
PROCESS_SUFFIXES = ("__age", "__n", "__n24", "__dtlast")
_SOFA_PROXY = frozenset({"Platelets", "Bilirubin_total", "Creatinine", "MAP", "FiO2", "SaO2", "O2Sat"})
_SF = frozenset({"FiO2", "SaO2", "O2Sat"})
#: the channels each GENESIS composite is computed from (features.py ``_composites``)
COMPOSITE_SOURCES: Dict[str, FrozenSet[str]] = {
    "sofa_proxy": _SOFA_PROXY, "sofa_rise24": _SOFA_PROXY, "sofa_rise48": _SOFA_PROXY,
    "sofa_rise_all": _SOFA_PROXY, "sofa_mean24": _SOFA_PROXY, "sofa_resp": _SF, "sf_ratio": _SF,
    "sirs": frozenset({"Temp", "HR", "Resp", "PaCO2", "WBC"}), "qsofa_proxy": frozenset({"Resp", "SBP"}),
    "shock_index": frozenset({"HR", "SBP"}), "pulse_pressure": frozenset({"SBP", "DBP"}),
    "bun_cr": frozenset({"BUN", "Creatinine"}),
}


#: channels with rolling-window trends beyond CinC's seven vitals
ROLLING_EXTRA = ["GCS", "SOFA_P", "CVP"]


def feature_spec(channels: Sequence[str]) -> feat.FeatureSpec:
    return feat.FeatureSpec(channels=list(channels),
                            rolling=[c for c in list(VITALS) + ROLLING_EXTRA if c in channels],
                            labs=[c for c in E.LAB_CH if c in channels],
                            static_cols=list(E.STATIC), light=[c for c in E.LIGHT_CH if c in channels])


def channel_of(feature: str) -> Optional[str]:
    base = feature.split("__")[0]
    return base if base in E.DYNAMIC else None


_SOFA_P_SOURCES = frozenset({"PaO2", "FiO2", "Platelets", "Bilirubin_total", "MAP", "GCS_eye", "GCS_verbal",
                             "GCS_motor", "Creatinine", "UrineOutput"})
#: derived channels inherit every channel they are computed from — so the strict tier
#: drops total GCS with a covert verbal score, and SOFA with a covert FiO2
DERIVED_SOURCES: Dict[str, FrozenSet[str]] = {
    "GCS": frozenset({"GCS_eye", "GCS_verbal", "GCS_motor"}),
    "SOFA_P": _SOFA_P_SOURCES,
    "SOFA": _SOFA_P_SOURCES | frozenset(E.VASO_CH) | frozenset({"VentInvasive"}),
}


def sources_of(feature: str) -> FrozenSet[str]:
    """Every raw channel a feature column is computed from (empty for static context)."""
    ch = channel_of(feature)
    if ch:
        src = frozenset({ch}) | DERIVED_SOURCES.get(ch, frozenset())
        if feature.startswith("Neutrophils__nlr"):
            src |= {"Lymphocytes"}
        if feature.startswith("Norepinephrine__nee"):
            src |= frozenset(E.VASO_CH)
        if feature.startswith("Antibiotic__si"):
            src |= {"Culture"}
        if "__levy_" in feature:                       # a signature cross term reads BOTH streams
            partner = feature.split("__levy_", 1)[1].rsplit("_", 1)[0]
            src |= frozenset({partner}) | DERIVED_SOURCES.get(partner, frozenset())
        return src
    if feature in COMPOSITE_SOURCES:
        return COMPOSITE_SOURCES[feature]
    if feature.startswith("proc__gas"):
        return frozenset(c for c in BLOOD_GAS if c in E.LAB_CH)
    if feature.startswith("proc__vitals"):
        return frozenset(c for c in VITALS if c in E.DYNAMIC)
    if feature.startswith("proc__"):
        return frozenset(E.LAB_CH)
    return frozenset()


def is_process_feature(feature: str) -> bool:
    """Does the column describe the measurement PROCESS (when, how often) rather than a value?"""
    return feature.startswith("proc__") or (channel_of(feature) is not None and
                                            feature.endswith(PROCESS_SUFFIXES))


def pairs_auc(s: np.ndarray, wpos: np.ndarray, wneg: np.ndarray) -> float:
    """Tie-exact AUROC where every row carries a positive weight AND a negative weight —
    hard labels (w*y, w*(1-y)), soft labels, or two disjoint row sets."""
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    block = np.cumsum(np.r_[True, ss[1:] != ss[:-1]]) - 1
    nb = int(block[-1]) + 1 if len(block) else 0
    pos = np.bincount(block, weights=np.asarray(wpos, float)[order], minlength=nb)
    neg = np.bincount(block, weights=np.asarray(wneg, float)[order], minlength=nb)
    below = np.r_[0.0, np.cumsum(neg)[:-1]]
    den = pos.sum() * neg.sum()
    return float((pos * (below + 0.5 * neg)).sum() / den) if den > 0 else float("nan")


def stratified_auc(y: np.ndarray, p: np.ndarray, strata: np.ndarray,
                   w: Optional[np.ndarray] = None) -> float:
    """P(a positive outranks a negative FROM THE SAME STRATUM) — the pair-weighted mean of
    per-stratum AUROCs. Strata = ICU hour gives the hour-matched AUROC (the clock cannot
    contribute); strata = bins of a clinician-action score gives the AUROC among hours
    that clinicians' actions made look equally suspicious."""
    y, p, strata = np.asarray(y), np.asarray(p, dtype=np.float64), np.asarray(strata)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    if not len(y):
        return float("nan")
    order = np.lexsort((p, strata))
    hs, ys, ps, ws = strata[order], y[order], p[order], w[order]
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


def weighted_hour_matched(y, p, hour, w) -> float:
    """Pair-weighted mean of per-hour weighted AUROCs (the clock cannot contribute)."""
    return stratified_auc(y, p, hour, w)


# --------------------------------------------------------------------------- #
Predictor = Callable[["Ledger", np.ndarray], np.ndarray]


@dataclass
class Candidate:
    """A model as the harness sees it. ``fit(ledger, cols, train_rows, fold)`` trains on
    ``train_rows`` and returns a PREDICTOR ``(view, rows) -> margins`` that can be applied
    to any ledger-shaped view — the real patients, or counterfactually edited ones."""
    name: str
    tier: str
    fit: Callable[..., Predictor]
    strict: bool = False            # the measured-strict tier (``infoflow`` covert channels out)
    #: a narrower column set than the whole tier (still checked against the tier's firewall)
    columns: Optional[Callable[["Ledger"], np.ndarray]] = None

    def cols(self, led: "Ledger") -> np.ndarray:
        cols = self.columns(led) if self.columns is not None else led.columns(self.tier, strict=self.strict)
        led.assert_tier(np.asarray(cols), self.tier)
        return np.asarray(cols)


@dataclass
class FoldModels:
    """A candidate fitted fold by fold: its out-of-fold scores and the predictor of every
    fold, so the harness can re-score the SAME fitted models on edited patients."""
    name: str
    tier: str
    oof: np.ndarray
    predictors: Dict[int, Predictor]
    cols: Optional[np.ndarray] = None

    def predict(self, view: "Ledger", rows: np.ndarray) -> np.ndarray:
        """Margins for ``rows`` of ``view``, each row scored by ITS OWN fold's model (so an
        edited patient is only ever scored by a model that never trained on them)."""
        rows = np.asarray(rows, dtype=np.int64)
        f = view.folds[view.store.seg[rows]]
        out = np.full(len(rows), np.nan)
        for k in np.unique(f):
            if int(k) in self.predictors:
                m = f == k
                out[m] = self.predictors[int(k)](view, rows[m])
        return out


def _gates(M: np.ndarray) -> np.ndarray:
    """Each margin m as [log sigmoid(m), log sigmoid(-m)]: their difference is m itself, and
    unit weights on the first reproduce the GENESIS product of experts exactly — so a linear
    stacker over these can express both, and learn anything between."""
    from backend.ml.genesis.stack import log_sigmoid
    M = np.clip(M, -30.0, 30.0)
    return np.column_stack([log_sigmoid(M), log_sigmoid(-M)])


def _offset_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray, off: np.ndarray, lam: float,
                  iters: int = 60) -> Tuple[np.ndarray, float]:
    """Weighted logistic regression with a fixed OFFSET and an L2 penalty on the slopes
    (intercept free): damped Newton with a backtracking line search, like ``platt``."""
    n, d = X.shape
    Xb = np.column_stack([X, np.ones(n)])
    pen = np.r_[np.full(d, lam), 0.0]
    sw = float(w.sum())
    beta = np.zeros(d + 1)

    def obj(b):
        z = off + Xb @ b
        return float((w * (np.logaddexp(0.0, z) - y * z)).sum()) / sw + float((pen * b * b).sum())
    f = obj(beta)
    for _ in range(iters):
        z = off + Xb @ beta
        pr = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        g = Xb.T @ (w * (pr - y)) / sw + 2 * pen * beta
        H = (Xb * (w * pr * (1 - pr))[:, None]).T @ Xb / sw + np.diag(2 * pen) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        t, moved = 1.0, False
        while t > 1e-6:
            nb = beta - t * step
            nf = obj(nb)
            if np.isfinite(nf) and nf <= f - 1e-12:
                moved = True
                break
            t *= 0.5
        if not moved:
            break
        beta, f = nb, nf
        if np.max(np.abs(t * step)) < 1e-8:
            break
    return beta[:d], float(beta[d])


SHRINK_GRID = (1.0, 10.0, 100.0, 1000.0)


def stack(led: "Ledger", name: str, parts: Sequence[FoldModels],
          shrink: Optional[float] = None) -> FoldModels:
    """A cross-fitted stacker that STARTS at the product of experts and learns corrections.

    logit = sum_j log sigmoid(m_j)  (the GENESIS product, a fixed offset)
            + sum_j beta_j . [log sigmoid(m_j), log sigmoid(-m_j)]  + b

    with an L2 penalty ``shrink / n_stays`` on beta: with few patients the stack stays at the
    product, with many it learns what the product cannot. ``shrink`` is chosen per outer fold
    by an inner leave-one-fold-out over that fold's TRAINING folds (the grid ``SHRINK_GRID``),
    so the fold-k stack — weights and strength alike — never sees the labels it scores.
    Two simpler stacks were measured first on the demo and lost to the product they were
    meant to generalise: plain logistic regression on raw margins (0.654 vs 0.739) and on
    gated margins (0.628) — a free stacker with ~20 septic stays per fold overfits."""
    s = led.store
    raw = np.clip(np.column_stack([fm.oof for fm in parts]), -30.0, 30.0)
    G = _gates(raw)
    off_all = G[:, :len(parts)].sum(axis=1)
    f = led.folds[s.seg]
    ok = (f >= 0) & led.train_mask & np.all(np.isfinite(raw), axis=1)
    folds = sorted(set.intersection(*(set(fm.predictors) for fm in parts)))
    oof = np.full(s.n_rows, np.nan)
    preds: Dict[int, Predictor] = {}
    def fit_on(rows, strength):
        mu, sd = G[rows].mean(axis=0), G[rows].std(axis=0) + 1e-9
        lam = strength / max(len(np.unique(s.seg[rows])), 1)
        beta, b = _offset_logit((G[rows] - mu) / sd, led.y[rows].astype(float), led.design_w[rows],
                                off_all[rows], lam)
        return beta / sd, b - float((beta * mu / sd).sum())

    chosen = {}
    for k in folds:
        tr, te = np.flatnonzero(ok & (f != k)), np.flatnonzero(ok & (f == k))
        if not len(tr) or not len(te) or led.y[tr].min() == led.y[tr].max():
            continue
        strength = shrink
        if strength is None:                   # inner leave-one-fold-out over the training folds
            score = {}
            for lam in SHRINK_GRID:
                aucs = []
                for j in folds:
                    if j == k:
                        continue
                    itr, ite = tr[f[tr] != j], tr[f[tr] == j]
                    if not len(ite) or led.y[itr].min() == led.y[itr].max() or led.y[ite].min() == led.y[ite].max():
                        continue
                    c, bb = fit_on(itr, lam)
                    aucs.append(Ranked(off_all[ite] + G[ite] @ c + bb).auc(led.y[ite], led.design_w[ite]))
                score[lam] = float(np.mean(aucs)) if aucs else -1.0
            strength = max(score, key=score.get)
        chosen[k] = strength
        coef, b = fit_on(tr, strength)
        oof[te] = off_all[te] + G[te] @ coef + b

        def make(k=k, coef=coef, b=b):
            def predict(view, rows):
                g = _gates(np.column_stack([fm.predictors[k](view, rows) for fm in parts]))
                return g[:, :len(parts)].sum(axis=1) + g @ coef + b
            return predict
        preds[k] = make()
    fm = FoldModels(name, max((fm.tier for fm in parts), key=lambda t: K.TIER_ORDER[t]), oof, preds)
    fm.shrink = chosen
    return fm


def product(name: str, parts: Sequence[FoldModels]) -> FoldModels:
    """The GENESIS product of experts, sum of log-sigmoids, kept re-appliable."""
    from backend.ml.genesis.stack import log_sigmoid
    folds = set.intersection(*(set(fm.predictors) for fm in parts))

    def make(k):
        return lambda view, rows: sum(log_sigmoid(fm.predictors[k](view, rows)) for fm in parts)
    with np.errstate(invalid="ignore"):                   # NaN = rows outside every fold
        oof = sum(log_sigmoid(fm.oof) for fm in parts)
    return FoldModels(name, max((fm.tier for fm in parts), key=lambda t: K.TIER_ORDER[t]),
                      oof, {k: make(k) for k in sorted(folds)})


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
    covert: FrozenSet[str] = frozenset()      # channels the information-flow audit flagged
    parent_rows: Optional[np.ndarray] = None  # a view's rows in the ledger it came from
    cache: Dict = field(default_factory=dict, repr=False)
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

    def columns(self, tier: str, strict: bool = False) -> np.ndarray:
        """Columns a tier may read. ``strict`` also drops every column computed from a
        covert channel (``infoflow``): the declared tier, minus what the audit measured
        leaking into it."""
        ok = self.ftier <= K.TIER_ORDER[tier]
        if strict and self.covert:
            ok &= np.array([not (sources_of(n) & self.covert) for n in self.names])
        return np.flatnonzero(ok)

    def clinician_columns(self) -> np.ndarray:
        """What clinicians' ACTIONS reveal: every treatment / order column, plus the
        measurement process of the physiology channels (when things were measured and
        how often — never what they read)."""
        return np.flatnonzero([(self.ftier[j] > 0) or is_process_feature(n)
                               for j, n in enumerate(self.names)])

    def dev_rows(self, eval_only: bool = True) -> np.ndarray:
        m = self.folds[self.store.seg] >= 0
        return np.flatnonzero(m & (self.eval_mask if eval_only else self.train_mask))

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

    def lock_rows(self, purpose: str, chain=None) -> np.ndarray:
        """The locked rows — every access appended to the hash chain (``trust``)."""
        from backend.ml.mimic import trust
        (chain or trust.Chain()).append("lock_access", {
            "purpose": purpose, "lock": lock_fingerprint(self.store, self.folds)})
        return np.flatnonzero((self.folds[self.store.seg] < 0) & self.train_mask)

    # ------------------------------------------------------------------ #
    def fit_oof(self, cand: Candidate, verbose: bool = True) -> FoldModels:
        """Fit ``cand`` on every dev fold; keep its out-of-fold scores AND its predictors."""
        cols = cand.cols(self)
        oof = np.full(self.store.n_rows, np.nan)
        preds: Dict[int, Predictor] = {}
        for k in range(self.n_folds):
            tr, te = self.fold_rows(k)
            if not len(te):
                continue
            pr = cand.fit(self, cols, tr, k)
            oof[te] = pr(self, te)
            preds[k] = pr
        fm = FoldModels(cand.name, cand.tier, oof, preds, cols)
        if verbose:
            self.report(fm)
        return fm

    def report(self, fm: FoldModels) -> None:
        s = self.score(fm.oof)
        print(f"  {fm.name:<28} [{fm.tier}] AUROC {s['auroc']:.4f}  hour-matched {s['auroc_hourmatched']:.4f}  "
              f"pre-suspicion {s['auroc_presusp']:.4f}  between {s['between']:.4f}  "
              f"within {s['within_septic']:.4f}  AUPRC {s['auprc']:.4f}", flush=True)

    def view(self, stays: np.ndarray, X: Optional[np.ndarray] = None) -> "Ledger":
        """A ledger over ``stays`` of this one — optionally with an EDITED grid ``X`` (their
        rows, base channels) whose derived channels are recomputed. Features are rebuilt
        from the grid; labels, masks, folds and design weights are inherited, because a
        counterfactual changes what a model sees, never what happened to the patient."""
        stays = np.asarray(stays, dtype=np.int64)
        sub = E.subset(self.store, stays)
        if X is not None:
            sub = E.rederive(sub, X)
        F, names = build_features(sub, dtype=self.F.dtype)
        if names != self.names:
            raise RuntimeError("a view's features differ from the ledger's")
        rows = np.concatenate([np.arange(self.store.starts[k], self.store.starts[k] + self.store.lens[k])
                               for k in stays]) if len(stays) else np.zeros(0, np.int64)
        return Ledger(store=sub, F=F, names=names, ftier=self.ftier, folds=self.folds[stays],
                      rule=self.rule, anchor=self.anchor, design_w=self.design_w[rows].copy(),
                      covert=self.covert, parent_rows=rows)

    def score(self, p: np.ndarray, rows: Optional[np.ndarray] = None,
              y: Optional[np.ndarray] = None) -> Dict[str, float]:
        s = self.store
        y = self.y if y is None else y
        default_rows = rows is None
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
        if default_rows:
            # the settings a paper would call "ideal": every stay (sepsis on arrival included),
            # and patient level (a stay's highest score vs whether it became septic)
            every = np.flatnonzero(self.train_mask & np.isfinite(p) & (self.folds[s.seg] >= 0))
            ya = y[every]
            out["auroc_all_stays"] = Ranked(p[every]).auc(ya, self.design_w[every]) \
                if 0 < ya.sum() < len(ya) else float("nan")
            out["patient_auroc"] = self._patient_auc(p, rows)
            out["patient_auroc_all_stays"] = self._patient_auc(p, every)
        return out

    def _patient_auc(self, p: np.ndarray, rows: np.ndarray) -> float:
        """AUROC of each stay's highest score (over ``rows``: scored up to onset + 3 h for a
        septic stay, the whole stay otherwise) against whether the stay became septic."""
        s = self.store
        if not len(rows):
            return float("nan")
        seg = s.seg[rows]
        first = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
        mx = np.maximum.reduceat(p[rows], first)
        lab = np.isfinite(self.onset[seg[first]]).astype(np.int8)
        return Ranked(mx).auc(lab, self.design_w[rows][first]) if 0 < lab.sum() < len(lab) else float("nan")

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
                               self.design_w, self.covert)
                out[f"{rule}/{anchor}"] = other.score(p)["auroc"]
        return out

    def tripwires(self) -> Dict[str, float]:
        s = self.store
        rows = np.flatnonzero(self.eval_mask & (self.folds[s.seg] >= 0))
        clock = Ranked(s.hour[rows].astype(float)).auc(self.y[rows], self.design_w[rows])
        # the ORIGINAL record length (``trim_septic`` shortens the grid, never the stay)
        full_len = s.meta.n_hours.to_numpy() if s.meta is not None and "n_hours" in s.meta else s.lens
        length = Ranked(full_len[s.seg[rows]].astype(float)).auc(self.y[rows], self.design_w[rows])
        early = {}
        for rule in E.RULES:
            t = s.onsets[f"t_sepsis_{rule}_min"].to_numpy()
            sep = np.isfinite(t)
            early[rule] = float(np.mean(t[sep] < ONSET_MIN)) if sep.any() else float("nan")
        return {"clock_auroc": clock, "record_length_auroc": length,
                **{f"onset_within_{int(ONSET_MIN)}h_{r}": v for r, v in early.items()}}


def _features(store: E.MimicStore, spec: feat.FeatureSpec):
    """The GENESIS engine's features plus the MIMIC block (the label's structure, causal)."""
    from backend.ml.mimic import mimicfeat
    F, names = feat.build(store, spec, cap=2000)
    B, bnames = mimicfeat.block(store)
    return np.hstack([F, B]), list(names) + bnames


def build_features(store: E.MimicStore, chunk_stays: Optional[int] = None,
                   memmap_path: Optional[str] = None, dtype=np.float32):
    spec = feature_spec(E.DYNAMIC)
    if not chunk_stays:
        F, names = _features(store, spec)
        return np.ascontiguousarray(F, dtype=dtype), names
    first = E.subset(store, np.arange(min(chunk_stays, store.n_patients)))
    F0, names = _features(first, spec)
    shape = (store.n_rows, len(names))
    F = (np.lib.format.open_memmap(memmap_path, mode="w+", dtype=dtype, shape=shape)
         if memmap_path else np.empty(shape, dtype=dtype))
    for a in range(0, store.n_patients, chunk_stays):
        idx = np.arange(a, min(a + chunk_stays, store.n_patients))
        part = E.subset(store, idx)
        Fc, nc = _features(part, spec)
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
