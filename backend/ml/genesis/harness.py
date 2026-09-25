"""The full-cohort referee for the GENESIS line.

Same contract as :mod:`backend.ml.eval.bench` — the harness owns the folds, a
candidate is only a function ``(cohort, train_rows, test_rows) -> p_test`` — but
built for the whole two-hospital cohort (40,335 patients, 1.55M hours) and for the
question this line is chasing, which is pooled hourly AUROC.

Three things are new here, each because the old harness could not answer a question
this work has to answer:

**The AUROC decomposition.** Pooled hourly AUROC averages over (positive hour,
negative hour) pairs, and the negatives come in two kinds: hours of patients who
never become septic (90.5% of negatives) and the not-yet-labelled hours of patients
who will (9.5%). So pooled AUROC is *exactly* a weighted mean of two AUROCs::

    pooled = 0.905 * between + 0.095 * within_septic

``between`` asks "is this a septic patient near onset, or somebody who never gets
sepsis?"; ``within_septic`` asks "is this septic patient near onset yet?". A model
that only ever learns *which patients* caps out near 0.905 + 0.095 * 0.5 = 0.953 even
with a perfect between-patient score. Every candidate is reported on both, because
the two respond to different levers (see ``docs/SUPERVISION_DENSITY.md``).

**An O(n) weighted AUROC.** Scores are sorted once; any patient-bootstrap draw is
then just a new weight vector, so a 1,000-draw paired bootstrap over 1.55M rows
takes seconds instead of an hour. Ties are split half-and-half, exactly as the
Mann-Whitney statistic (and ``sklearn.roc_auc_score``) does.

**The P19 view.** The irregular-time-series literature (Raindrop, ViTST, ...) reports
"P19" AUROC ~0.87-0.90 on this same dataset, but that task is *patient-level*: one
label per record, scored from the whole record, which for a septic patient includes
the hours after onset. It is a different, much easier question. ``patient_auroc``
(max score over the record vs. the patient label) is reported beside the hourly number
so the two can never be confused.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from backend.ml import config
from backend.ml.eval.utility import best_threshold_utility, utility_terms
from backend.ml.genesis import features as feat
from backend.ml.genesis import rawstore

CACHE_DIR = os.path.join(config.ARTIFACTS_DIR, "feature_cache")
RESULTS_DIR = os.path.join(config.ARTIFACTS_DIR, "genesis")

Candidate = Callable[["Cohort", np.ndarray, np.ndarray], np.ndarray]

#: Hours after which a CinC 2019 record is almost surely septic (see ``score``).
CLOCK_CUT = 60


# --------------------------------------------------------------------------- #
# Weighted, tie-exact AUROC over a fixed score ordering
# --------------------------------------------------------------------------- #
class Ranked:
    """Sort a score vector once; score any (label, weight) combination in O(n)."""

    def __init__(self, s: np.ndarray):
        s = np.asarray(s, dtype=np.float64)
        if not np.all(np.isfinite(s)):
            raise ValueError("scores contain NaN/inf")
        self.order = np.argsort(s, kind="mergesort")
        ss = s[self.order]
        self.block = np.cumsum(np.r_[True, ss[1:] != ss[:-1]]) - 1
        self.nb = int(self.block[-1]) + 1

    def auc(self, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
        ys = np.asarray(y)[self.order]
        ws = np.ones(len(ys)) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        pos = np.bincount(self.block, weights=ws * (ys == 1), minlength=self.nb)
        neg = np.bincount(self.block, weights=ws * (ys == 0), minlength=self.nb)
        below = np.r_[0.0, np.cumsum(neg)[:-1]]
        den = pos.sum() * neg.sum()
        return float((pos * (below + 0.5 * neg)).sum() / den) if den > 0 else float("nan")


def auroc(y: np.ndarray, s: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    return Ranked(s).auc(y, w)


# --------------------------------------------------------------------------- #
@dataclass
class Cohort:
    raw: rawstore.RawStore
    F: np.ndarray
    names: List[str]
    folds: np.ndarray                 # (P,) dev fold id per patient; -1 = LOCKED test set
    y: np.ndarray = field(init=False)
    seg: np.ndarray = field(init=False)
    hour: np.ndarray = field(init=False)
    plabel: np.ndarray = field(init=False)
    onset: np.ndarray = field(init=False)      # (P,) labelled onset hour, -1 never
    onset_row: np.ndarray = field(init=False)  # (N,) the row's patient's onset

    def __post_init__(self):
        r = self.raw
        self.y = r.y
        self.seg = r.seg
        self.hour = r.hour
        self.plabel = r.patient_label()
        self.onset = r.onset_hour()
        self.onset_row = self.onset[self.seg]

    # ------------------------------------------------------------------ #
    @staticmethod
    def load(spec: Optional[feat.FeatureSpec] = None, n_folds: int = 5, seed: int = 20260923,
             refresh: bool = False, mmap: bool = True) -> "Cohort":
        spec = spec or feat.FeatureSpec()
        raw = rawstore.load()
        key = spec.key()
        fpath = os.path.join(CACHE_DIR, f"genesis_{key}_rm.npy")
        npath = os.path.join(CACHE_DIR, f"genesis_{key}_rm.names")
        if not (os.path.exists(fpath) and os.path.exists(npath)) or refresh:
            F, names = feat.build(raw, spec)
            _save_row_major(fpath, F)
            with open(npath, "w") as fh:
                fh.write("\n".join(names))
            del F
        # memory-mapped: file-backed pages the OS can drop under pressure, instead of
        # 3 GB of anonymous memory competing with the fit on a 16 GB laptop
        F = np.load(fpath, mmap_mode="r" if mmap else None)
        names = open(npath).read().split("\n")
        if F.shape[0] != raw.n_rows or F.shape[1] != len(names):
            raise ValueError("feature cache does not match the raw store; rebuild with refresh=True")
        return Cohort(raw=raw, F=F, names=names, folds=patient_folds(raw, n_folds, seed))

    @property
    def n_folds(self) -> int:
        return int(self.folds.max()) + 1

    @property
    def dev_rows(self) -> np.ndarray:
        return np.flatnonzero(self.folds[self.seg] >= 0)

    def lock_rows(self, purpose: str) -> np.ndarray:
        """Rows of the LOCKED test set. Every access is appended to a ledger, so the
        number of times the final set was looked at is itself on the record."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "lock_ledger.jsonl"), "a") as fh:
            fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "purpose": purpose}) + "\n")
        return np.flatnonzero(self.folds[self.seg] < 0)

    def rows_of(self, patients: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.raw.n_patients, dtype=bool)
        mask[patients] = True
        return np.flatnonzero(mask[self.seg])

    def fold_rows(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """(train, test) rows of dev fold ``k``. Locked patients are in neither."""
        f = self.folds[self.seg]
        return np.flatnonzero((f >= 0) & (f != k)), np.flatnonzero(f == k)

    def col(self, name: str) -> np.ndarray:
        return self.F[:, self.names.index(name)]

    def cols(self, names: List[str], step: int = 131072) -> np.ndarray:
        """Several columns in ONE pass over the row-major matrix (``col`` per name would
        scan the whole 3 GB map once per column)."""
        j = np.array([self.names.index(n) for n in names])
        out = np.empty((self.F.shape[0], len(j)), dtype=np.float32)
        for a in range(0, self.F.shape[0], step):
            out[a:a + step] = self.F[a:a + step][:, j]
        return out


#: Seed of the locked test set. Chosen once, before any architecture work, and never
#: changed: the lock is only worth something if it was fixed before it could be tuned to.
LOCK_SEED = 17
LOCK_FRACTION = 0.2


def _save_row_major(path: str, F: np.ndarray, step: int = 65536) -> None:
    """Write ``F`` (built column-major, which is fast to fill) as a ROW-major .npy.

    Every consumer reads rows — a fold, a batch, a patient — and a row of a
    column-major memory map is 510 separate page touches. Converted in row chunks so
    the peak is one chunk, not a second copy of the matrix."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=F.shape)
    for a in range(0, F.shape[0], step):
        out[a:a + step] = F[a:a + step]
    out.flush()
    del out


def patient_folds(raw: rawstore.RawStore, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per patient (-1 = locked), stratified jointly by outcome and hospital so
    the locked set and every dev fold carry the same septic mix from both sites.

    The locked 20% is drawn first with its own fixed seed, so changing the dev fold
    count or seed can never move a patient into or out of the locked set."""
    strata = raw.patient_label().astype(int) * 2 + raw.site.astype(int)
    lock_split = StratifiedKFold(n_splits=int(round(1 / LOCK_FRACTION)), shuffle=True,
                                 random_state=LOCK_SEED)
    _, locked = next(lock_split.split(np.zeros(raw.n_patients), strata))
    folds = np.full(raw.n_patients, -1, dtype=np.int16)
    dev = np.setdiff1d(np.arange(raw.n_patients), locked)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(dev)), strata[dev])):
        folds[dev[te]] = k
    return folds


def lock_fingerprint(c: "Cohort") -> str:
    import hashlib
    ids = ",".join(sorted(c.raw.pid[c.folds < 0].tolist()))
    return hashlib.sha256(ids.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
@dataclass
class Result:
    name: str
    oof: np.ndarray
    per_fold: List[Dict[str, float]]
    pooled: Dict[str, float]
    seconds: float
    meta: Dict[str, object] = field(default_factory=dict)

    def line(self) -> str:
        p = self.pooled
        fa = np.array([f["auroc"] for f in self.per_fold])
        return (f"{self.name:<28} AUROC {p['auroc']:.4f} (folds {fa.mean():.4f}±{fa.std():.4f})  "
                f"<60h {p['auroc_lt60']:.4f}  hourmatched {p['auroc_hourmatched']:.4f}  "
                f"between {p['between']:.4f}  within {p['within_septic']:.4f}  "
                f"stay {p['within_stay']:.4f}  AUPRC {p['auprc']:.4f}  "
                f"P19 {p['patient_auroc']:.4f}  util* {p['utility_best']:+.4f}  "
                f"det@FA15 {p['detect_far15']:.3f}  [{self.seconds:.0f}s]")


def score(c: Cohort, p: np.ndarray, rows: Optional[np.ndarray] = None,
          with_utility: bool = True) -> Dict[str, float]:
    """Every headline number for predictions ``p`` on ``rows`` (default: all rows)."""
    rows = np.arange(len(c.y)) if rows is None else rows
    y = c.y[rows].astype(np.int8)
    seg = c.seg[rows]
    sep_row = c.plabel[seg] == 1
    R = Ranked(p)
    out: Dict[str, float] = {}
    out["auroc"] = R.auc(y)
    out["between"] = R.auc(y, np.where((y == 0) & sep_row, 0.0, 1.0))
    out["within_septic"] = R.auc(y, np.where(sep_row, 1.0, 0.0))
    out["neg_share_septic"] = float(((y == 0) & sep_row).sum() / max((y == 0).sum(), 1))
    out["within_stay"] = _within_stay(y, p, seg)
    # CLOCK CONTROL. Never-septic records in CinC 2019 stop at ~60 h (99th pct 59 h at
    # site A) while septic ones run to 336 h, so past hour 60 a row is ~82% likely to be
    # a septic patient's and the ICU clock alone scores AUROC 0.636. Scoring only the
    # first 60 hours of every record removes that shortcut from the metric.
    early = c.hour[rows] < CLOCK_CUT
    if early.any() and 0 < y[early].sum() < early.sum():
        out["auroc_lt60"] = Ranked(p[early]).auc(y[early])
    else:
        out["auroc_lt60"] = float("nan")
    out["auroc_hourmatched"] = hour_matched_auroc(y, p, c.hour[rows])
    out["auprc"] = _auprc(y, p)
    # patient level (P19 analogue): max over the scored record vs. the patient label
    pts, inv = np.unique(seg, return_inverse=True)
    peak = np.full(len(pts), -np.inf)
    np.maximum.at(peak, inv, p)
    out["patient_auroc"] = auroc(c.plabel[pts], peak)
    # early-warning view: septic patients scored only on hours up to labelled onset
    onset = c.onset_row[rows]
    hour = c.hour[rows]
    pre = np.where((onset < 0) | (hour <= onset), p, -np.inf)
    peak_pre = np.full(len(pts), -np.inf)
    np.maximum.at(peak_pre, inv, pre)
    # a septic patient whose scored rows all lie after onset has nothing to be early on
    has_pre = np.isfinite(peak_pre)
    out["patient_auroc_pre"] = auroc(c.plabel[pts][has_pre], peak_pre[has_pre])
    ns_peak = np.sort(peak[c.plabel[pts] == 0])
    sep_peak = peak_pre[(c.plabel[pts] == 1) & has_pre]
    for far in (0.05, 0.15):
        k = int(np.ceil((1.0 - far) * len(ns_peak))) - 1
        thr = ns_peak[min(max(k, 0), len(ns_peak) - 1)]
        out[f"detect_far{int(far*100)}"] = float((sep_peak >= thr).mean())
    if with_utility:
        terms = utility_terms(y, seg, hour, onset)
        u, t = best_threshold_utility(y, p, terms)
        out["utility_best"], out["thr_best"] = float(u), float(t)
    else:
        out["utility_best"], out["thr_best"] = float("nan"), float("nan")
    return out


def hour_matched_auroc(y: np.ndarray, p: np.ndarray, hour: np.ndarray) -> float:
    """P(score(pos) > score(neg)) over pairs taken at the SAME record hour.

    The strictest clock control available: a model that knows nothing but the hour
    scores exactly 0.5, because every compared pair shares the hour. Equals the
    pair-count-weighted mean of per-hour AUROCs."""
    order = np.lexsort((p, hour))
    hs, ys, ps = hour[order], y[order], p[order]
    starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
    ends = np.r_[starts[1:], len(hs)]
    num = den = 0.0
    for a, b in zip(starts, ends):
        yy = ys[a:b]
        npos = float(yy.sum())
        nneg = float(len(yy) - npos)
        if npos == 0 or nneg == 0:
            continue
        num += Ranked(ps[a:b]).auc(yy) * npos * nneg
        den += npos * nneg
    return num / den if den > 0 else float("nan")


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def _within_stay(y: np.ndarray, p: np.ndarray, seg: np.ndarray) -> float:
    """Mean per-patient AUROC over septic stays that contain both classes."""
    order = np.lexsort((p, seg))
    ys, ss = y[order], seg[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    ends = np.r_[starts[1:], len(ss)]
    ps = p[order]
    vals = []
    for a, b in zip(starts, ends):
        yy = ys[a:b]
        npos = int(yy.sum())
        if npos == 0 or npos == b - a:
            continue
        vals.append(Ranked(ps[a:b]).auc(yy))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
def run(c: Cohort, name: str, cand: Candidate, folds: Optional[List[int]] = None,
        verbose: bool = True, save: bool = True, meta: Optional[Dict] = None) -> Result:
    t0 = time.time()
    oof = np.full(len(c.y), np.nan)
    per_fold = []
    for k in (folds if folds is not None else range(c.n_folds)):
        tr, te = c.fold_rows(k)
        tf = time.time()
        p = np.asarray(cand(c, tr, te), dtype=np.float64)
        if p.shape != te.shape:
            raise ValueError(f"{name}: {p.shape} predictions for {te.shape} rows")
        oof[te] = p
        s = score(c, p, te, with_utility=False)
        per_fold.append(s)
        if verbose:
            print(f"    fold {k}: AUROC {s['auroc']:.4f}  between {s['between']:.4f}  "
                  f"within {s['within_septic']:.4f}  P19 {s['patient_auroc']:.4f}  "
                  f"[{time.time()-tf:.0f}s]", flush=True)
    done = np.isfinite(oof)
    if (c.folds[c.seg[done]] < 0).any():
        raise RuntimeError("a locked patient received a dev prediction")
    pooled = score(c, oof[done], np.flatnonzero(done))
    res = Result(name=name, oof=oof, per_fold=per_fold, pooled=pooled,
                 seconds=time.time() - t0, meta=dict(meta or {}))
    if verbose:
        print("  " + res.line(), flush=True)
    if save:
        save_result(res)
    return res


def save_result(res: Result) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = res.name.replace("/", "_").replace(" ", "_")
    np.save(os.path.join(RESULTS_DIR, f"{safe}_oof.npy"), res.oof.astype(np.float32))
    with open(os.path.join(RESULTS_DIR, f"{safe}.json"), "w") as fh:
        json.dump({"name": res.name, "pooled": res.pooled, "per_fold": res.per_fold,
                   "seconds": res.seconds, "meta": res.meta}, fh, indent=1, default=float)


def load_oof(name: str) -> np.ndarray:
    safe = name.replace("/", "_").replace(" ", "_")
    return np.load(os.path.join(RESULTS_DIR, f"{safe}_oof.npy")).astype(np.float64)


def compare(c: Cohort, a: np.ndarray, b: np.ndarray, n_boot: int = 1000,
            seed: int = 0, rows: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Paired patient-bootstrap of pooled AUROC (and its two components): a - b.

    Patients are the resampling unit. Both candidates are scored on the same draw, so
    the draw's shared difficulty cancels and only the model difference is measured."""
    rows = np.flatnonzero(np.isfinite(a) & np.isfinite(b)) if rows is None else rows
    y = c.y[rows]
    seg = c.seg[rows]
    sep_row = c.plabel[seg] == 1
    Ra, Rb = Ranked(a[rows]), Ranked(b[rows])
    wb = np.where((y == 0) & sep_row, 0.0, 1.0)
    ww = np.where(sep_row, 1.0, 0.0)
    rng = np.random.default_rng(seed)
    P = c.raw.n_patients
    present = np.unique(seg)
    d = np.empty((n_boot, 3))
    for i in range(n_boot):
        m = np.bincount(rng.choice(present, size=len(present), replace=True), minlength=P)
        w = m[seg].astype(np.float64)
        d[i, 0] = Ra.auc(y, w) - Rb.auc(y, w)
        d[i, 1] = Ra.auc(y, w * wb) - Rb.auc(y, w * wb)
        d[i, 2] = Ra.auc(y, w * ww) - Rb.auc(y, w * ww)
    point = [Ra.auc(y) - Rb.auc(y), Ra.auc(y, wb) - Rb.auc(y, wb), Ra.auc(y, ww) - Rb.auc(y, ww)]
    out = {}
    for j, key in enumerate(("auroc", "between", "within")):
        lo, hi = np.percentile(d[:, j], [2.5, 97.5])
        out[f"d_{key}"] = float(point[j])
        out[f"{key}_ci"] = (float(lo), float(hi))
        out[f"{key}_sig"] = bool(lo > 0 or hi < 0)
    return out
