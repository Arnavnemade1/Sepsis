"""Infection vs sterile inflammation from blood expression — the study engine.

THE PIPELINE, AND WHY EACH STEP IS THE WAY IT IS
------------------------------------------------
1. RANK WITHIN SAMPLE. Each sample's genes are ranked against each other, and
   nothing is ever computed across samples or pooled across cohorts. Two reasons,
   both load-bearing:
     * No cross-cohort statistic can leak into a feature, so a batch effect cannot
       enter through the normalisation itself.
     * It is invariant to any monotone per-sample transform — which matters here
       because GSE74224 ships log-scale values (2.07-14.08) and GSE40012 ships
       linear intensities (41.7-34,356). Those never have to be reconciled, because
       the rank does not care.
   After ranking, the z-step is close to an affine no-op (ranks of a fixed gene set
   have a fixed mean and spread); it is kept because ties and missing probes make it
   not quite constant, and because the pre-registration specifies it. The rank is
   doing the work.

2. COLLAPSE PROBES TO SYMBOLS, ON THE RANKS. Several probes map to one gene, and the
   ORDER of these two steps is not arbitrary: averaging raw values first would break
   the invariance above, because a mean is not invariant to a monotone per-sample
   transform while a rank is. So: rank, then average ranks per symbol.

3. INTERSECT. Features are the gene symbols present on every included platform —
   13,785 of them for the two admissible cohorts.

4. LEAVE ONE COHORT OUT. Fit on all-but-one cohort, score the held-out one, repeat.
   EVERY direction is reported, never only the mean: a signature that transfers one
   way and not the other has fitted one cohort's idiosyncrasy, which is the same rule
   `derive_transfer.py` applies to hospitals.

WHAT WOULD MAKE A GOOD NUMBER HERE MEANINGLESS
----------------------------------------------
A model can score well by learning WHICH COHORT a sample came from — different lab,
different scanner, different year — wherever cohorts are POOLED. Three guards exist
for that, all computed in the same run as the primary, never on request:

* :func:`mean_direction_ci` — the interval never pools cohorts, so a platform offset
  cannot enter it. Verified rather than assumed: a +3.0 offset on one cohort leaves
  it identical while a pooled AUROC on the same data moves 0.834 -> 0.714.
* :func:`cohort_identity_auroc` — REPORTED, and under amendment 4 no longer a void
  condition. The primary is a mean of per-direction AUROCs, each computed inside one
  held-out cohort, where a cohort signal is constant and cannot rank patients; the
  frozen rule (>= 0.90 voids) would have voided the study on platform differences
  alone. The artifact reports the verdict under BOTH the frozen and amended rules.
* :func:`permutation_loco` — labels shuffled within cohort over MANY draws. The
  [0.42, 0.58] band applies to the MEAN of that null, not to one draw: a properly
  designed null here has per-draw sd ~0.10, so a lone draw leaves the band often by
  construction. Two thin nulls raised false alarms in this study before that was
  fixed — a single draw, and five draws that all reused one fold split.

A PATIENT IS THE UNIT, NOT A SAMPLE. GSE40012 contributes 190 samples from 58
people at days 1/2/3/5. Every interval clusters on patient, because resampling
samples would treat repeated measurements of one person as independent evidence
and report a narrower interval than the data supports.

HEALTHY SAMPLES ARE EXCLUDED. Infection-vs-healthy is a much easier question and
answering it would inflate the result while addressing nothing this system confuses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from backend.hostresponse import annotation as ann
from backend.hostresponse.data import geo_series as gs
from backend.hostresponse.data.build_geo_manifest import _class_of, label_field
from backend.ml.eval import metrics

#: The only two classes in the primary endpoint. HEALTHY is not a third class; it is
#: dropped, and the count of dropped samples is reported.
INFECTIOUS, STERILE = 1, 0


@dataclass
class Cohort:
    """One cohort, normalised and collapsed to gene symbols."""
    accession: str
    platform: str
    #: symbols x samples, rank-normalised within each sample
    X: pd.DataFrame
    y: np.ndarray
    samples: List[str]
    #: One entry per sample. NOT the same thing as ``samples``: GSE40012 contributes
    #: 190 samples from 58 patients (repeated timepoints, ``day`` 1/2/3/5), so
    #: treating a GSM as a patient would leak one person across a split and would
    #: make every interval too narrow. Where a cohort names no patient key, each
    #: sample is its own patient and that is recorded rather than assumed.
    patients: List[str] = field(default_factory=list)
    n_healthy_dropped: int = 0
    label_key: Optional[str] = None
    patient_key: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.samples)


class ExpressionCohortSource:
    """What a cohort must provide to plug into this study.

    The analogue of ``backend.physio.sepsis_hrv.RRLabeledSource``: implement
    ``load() -> Cohort`` and everything downstream — intersection, LOCO, the
    controls — runs unchanged. A future cohort on a new platform needs an adapter,
    not a change to the pipeline.
    """

    def load(self) -> Cohort:                                   # pragma: no cover
        raise NotImplementedError


def rank_within_sample(mat: pd.DataFrame) -> pd.DataFrame:
    """Rank genes within each sample (column), then z-score those ranks."""
    r = mat.rank(axis=0, method="average")
    mu = r.mean(axis=0)
    sd = r.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (r - mu) / sd


def collapse_to_symbols(ranked: pd.DataFrame, symbols: Dict[str, str]) -> pd.DataFrame:
    """Average the RANKS of probes sharing a gene symbol (see the module docstring)."""
    mapped = [symbols.get(str(i)) for i in ranked.index]
    keep = [j for j, s in enumerate(mapped) if s]
    if not keep:
        raise ValueError("no probe on this platform mapped to a gene symbol")
    out = ranked.iloc[keep].copy()
    out.index = pd.Index([mapped[j] for j in keep], name="symbol")
    return out.groupby(level=0).mean()


@dataclass
class GeoCohort(ExpressionCohortSource):
    """A cohort read from a downloaded GEO series matrix."""
    accession: str
    platform: str
    root: str = "data/geo"
    #: Which class is the NEGATIVE one. "sterile" is the study's question. "healthy"
    #: exists only for the easiness calibrator, which asks the trivial question ON
    #: PURPOSE so the gap between the two can be reported — never quoted as a result.
    contrast: str = "sterile"

    def load(self) -> Cohort:
        series = gs.parse_series_matrix(gs.local_path(self.accession, self.root),
                                        with_matrix=True)
        if series.matrix is None or series.matrix.shape[0] == 0:
            raise ValueError("%s has no expression table in its series matrix"
                             % self.accession)
        key, values = label_field(series)
        classes = [_class_of(v) for v in values]
        if self.contrast not in ("sterile", "healthy"):
            raise ValueError("contrast must be 'sterile' or 'healthy', not %r" % self.contrast)
        keep = [i for i, c in enumerate(classes) if c in ("infectious", self.contrast)]
        if not keep:
            raise ValueError("%s has no infectious/%s samples" % (self.accession, self.contrast))

        cols = [series.samples[i] for i in keep]
        mat = series.matrix.loc[:, [c for c in cols if c in series.matrix.columns]]
        present = [c for c in cols if c in series.matrix.columns]
        y = np.array([INFECTIOUS if classes[series.samples.index(c)] == "infectious"
                      else STERILE for c in present], dtype=int)

        ranked = rank_within_sample(mat)
        X = collapse_to_symbols(ranked, ann.symbol_map(self.accession, self.platform,
                                                       self.root))
        pkey, patients = self._patients(series, present)
        return Cohort(accession=self.accession, platform=self.platform, X=X, y=y,
                      samples=present, patients=patients,
                      n_healthy_dropped=(sum(1 for c in classes if c == "healthy")
                                         if self.contrast == "sterile" else 0),
                      label_key=key, patient_key=pkey)

    #: Characteristics keys that identify a PERSON rather than a sample.
    PATIENT_KEYS = ("patient id", "patient", "subject id", "subject", "id", "donor")

    def _patients(self, series, present: Sequence[str]) -> Tuple[Optional[str], List[str]]:
        """Map each kept sample to a patient, or fall back to one-patient-per-sample.

        A key only counts if it actually collapses samples (fewer distinct values than
        samples); an ``id`` column that is unique per row identifies the sample, not
        the person, and using it would silently re-assert the bug this exists to fix.
        Values are namespaced by accession so two cohorts cannot collide on "id: 3".
        """
        by_sample = {sid: i for i, sid in enumerate(series.samples)}
        for k in self.PATIENT_KEYS:
            vals = series.characteristics.get(k)
            if not vals:
                continue
            got = [(vals[by_sample[sid]] or "") for sid in present]
            if all(got) and len(set(got)) < len(got):
                return k, ["%s:%s" % (self.accession, v) for v in got]
        return None, ["%s:%s" % (self.accession, sid) for sid in present]


@dataclass
class Aligned:
    """Every cohort on one shared feature space, stacked sample-wise."""
    X: np.ndarray                      # samples x symbols
    y: np.ndarray
    cohort: np.ndarray                 # accession per sample
    sample_ids: np.ndarray
    patient_ids: np.ndarray            # the unit every interval must resample
    symbols: List[str]

    def summary(self) -> Dict[str, object]:
        return {"n_samples": int(len(self.y)),
                "n_patients": int(len(set(self.patient_ids.tolist()))),
                "n_symbols": len(self.symbols),
                "n_infectious": int((self.y == INFECTIOUS).sum()),
                "n_sterile": int((self.y == STERILE).sum()),
                "per_cohort": {c: {"samples": int((self.cohort == c).sum()),
                                   "patients": len(set(self.patient_ids[self.cohort == c].tolist()))}
                               for c in sorted(set(self.cohort.tolist()))}}


def align(cohorts: Sequence[Cohort]) -> Aligned:
    """Intersect gene symbols across cohorts and stack the samples."""
    shared = set(cohorts[0].X.index)
    for c in cohorts[1:]:
        shared &= set(c.X.index)
    symbols = sorted(shared)
    if len(symbols) < 100:
        raise ValueError("only %d shared gene symbols — too few to be a feature space; "
                         "report this as the finding rather than modelling it"
                         % len(symbols))
    Xs, ys, cs, ids, pids = [], [], [], [], []
    for c in cohorts:
        sub = c.X.loc[symbols]
        Xs.append(sub.to_numpy(dtype=float).T)          # samples x symbols
        ys.append(c.y)
        cs.append(np.array([c.accession] * c.n))
        ids.append(np.array(c.samples))
        pids.append(np.array(c.patients if c.patients else
                             ["%s:%s" % (c.accession, s) for s in c.samples]))
    return Aligned(np.vstack(Xs), np.concatenate(ys), np.concatenate(cs),
                   np.concatenate(ids), np.concatenate(pids), symbols)


#: L2-regularised logistic regression, no feature scaling, C = 0.01 (amendment 4).
#: Chosen on the DISCOVERY cohort only, by a rule stated before looking: the most
#: regularised C within 0.005 of the best 5-fold x 3-seed CV AUROC.
#:
#: * GBMDetector — the plan's reuse target — is unusable at p >> n. On 13,785 features
#:   and 52 training rows it emits ONE distinct prediction, which is why an early
#:   mechanics check read exactly 0.500.
#: * No StandardScaler, on principle: it fits per-gene statistics on the TRAINING
#:   cohort and applies them to the HELD-OUT one, re-introducing a cross-cohort
#:   statistic immediately after the normalisation was built to exclude one. The
#:   rank-z features already share a bounded per-sample scale (+/- sqrt 3).
L2_C = 0.01


def _fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                 symbols: Sequence[str], seed: int = 42) -> np.ndarray:
    """Fit on the training rows; return P(infectious) for the held-out rows.

    ``symbols`` and ``seed`` are kept for the callers' sake: lbfgs is deterministic,
    and the feature names travel with the Aligned matrix, not the model.
    """
    m = LogisticRegression(C=L2_C, max_iter=5000)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def loco(a: Aligned, seed: int = 42) -> Dict[str, object]:
    """Leave one cohort out. Returns every direction plus the pooled held-out scores.

    Each sample is scored exactly once, by a model that never saw its cohort, so the
    pooled vector is a legitimate basis for one interval over samples.
    """
    per: Dict[str, float] = {}
    pooled = np.full(len(a.y), np.nan)
    for held in sorted(set(a.cohort.tolist())):
        te = a.cohort == held
        tr = ~te
        if len(np.unique(a.y[tr])) < 2 or len(np.unique(a.y[te])) < 2:
            per[held] = float("nan")
            continue
        p = _fit_predict(a.X[tr], a.y[tr], a.X[te], a.symbols, seed)
        pooled[te] = p
        per[held] = metrics.auroc(a.y[te], p)
    vals = [v for v in per.values() if not np.isnan(v)]
    return {"per_cohort": per,
            "mean_auroc": float(np.mean(vals)) if vals else float("nan"),
            "pooled": pooled}


def auroc_ci(y: np.ndarray, p: np.ndarray, patient_ids: np.ndarray,
             n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """95% cluster bootstrap over PATIENTS.

    This function previously resampled SAMPLES and its docstring asserted that one
    sample is one patient. That is false for GSE40012, which contributes 190 samples
    from 58 people at days 1/2/3/5. Resampling samples treats repeated measurements
    of one person as independent evidence and reports an interval narrower than the
    data supports — on the pre-registered headline number.
    """
    ok = ~np.isnan(p)
    return metrics.bootstrap_ci(y[ok], p[ok], patient_ids[ok], stat="auroc",
                                n_boot=n_boot, seed=seed)


def mean_direction_ci(y: np.ndarray, p: np.ndarray, cohort: np.ndarray,
                      patient_ids: np.ndarray, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> Dict[str, object]:
    """The PRIMARY and its interval: the mean of per-direction AUROCs.

    Every bootstrap draw resamples patients WITHIN each held-out cohort independently,
    recomputes that cohort's AUROC on its own samples only, and averages the
    directions. No quantity is ever pooled across cohorts.

    Why that matters, measured rather than assumed: a per-cohort AUROC is invariant to
    any cohort-level score offset, because the offset is constant within the cohort it
    is added to. A POOLED AUROC is not. In a simulation at this study's real cluster
    counts, adding a platform-style +1.5 offset to one cohort's scores dragged the
    pooled interval's expected lower bound below the per-direction design's at every
    true AUROC tested (0.719 vs 0.766 at a true 0.85). So the pooled interval this
    module previously computed was exposed to exactly the batch effect the design
    exists to exclude, and this one is not.
    """
    ok = ~np.isnan(p)
    y, p, cohort, patient_ids = y[ok], p[ok], cohort[ok], patient_ids[ok]
    names = sorted(set(cohort.tolist()))
    # per cohort: patient -> the row indices that patient contributes
    rows_by = {}
    for c in names:
        idx = np.where(cohort == c)[0]
        pats = {}
        for i in idx:
            pats.setdefault(patient_ids[i], []).append(i)
        rows_by[c] = [np.array(v) for v in pats.values()]

    point = {c: metrics.auroc(y[cohort == c], p[cohort == c]) for c in names}
    rng = np.random.default_rng(seed)
    draws: List[float] = []
    for _ in range(n_boot):
        vals = []
        for c in names:
            groups = rows_by[c]
            pick = rng.integers(0, len(groups), size=len(groups))
            r = np.concatenate([groups[k] for k in pick])
            v = metrics.auroc(y[r], p[r])       # nan if a resample lost a class
            vals.append(v)
        if not any(np.isnan(vals)):
            draws.append(float(np.mean(vals)))
    arr = np.array(draws)
    lo, hi = (np.percentile(arr, 100 * alpha / 2), np.percentile(arr, 100 * (1 - alpha / 2))) \
        if arr.size else (float("nan"), float("nan"))
    finite = [v for v in point.values() if not np.isnan(v)]
    return {"point_mean": float(np.mean(finite)) if finite else float("nan"),
            "per_direction": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_boot_used": int(arr.size), "n_boot_requested": int(n_boot)}


# --------------------------------------------------------------------------- #
# pre-registered negative controls — computed every run, never on request
# --------------------------------------------------------------------------- #
def cohort_identity_auroc(a: Aligned, seed: int = 42) -> float:
    """Can the SAME pipeline predict which cohort a sample came from?

    If it can (>= 0.90), the features encode which lab ran the assay and the primary
    endpoint is reported CONFOUNDED at any value. Only defined for two cohorts; with
    more, it is the mean over one-vs-rest.
    """
    names = sorted(set(a.cohort.tolist()))
    scores = []
    for name in names[:1] if len(names) == 2 else names:
        z = (a.cohort == name).astype(int)
        # held-out by sample, not by cohort: the question is whether the FEATURES
        # carry cohort identity at all, so the model must be allowed to see both.
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(z))
        cut = len(z) // 2
        tr, te = idx[:cut], idx[cut:]
        if len(np.unique(z[tr])) < 2 or len(np.unique(z[te])) < 2:
            continue
        p = _fit_predict(a.X[tr], z[tr], a.X[te], a.symbols, seed)
        scores.append(metrics.auroc(z[te], p))
    return float(np.mean(scores)) if scores else float("nan")


def permutation_loco(a: Aligned, seed: int = 7, n_draws: int = 20) -> Dict[str, object]:
    """LOCO with labels shuffled WITHIN cohort, over MANY draws.

    One draw is not a control. Measured on the discovery cohort, a single permutation
    at this sample size has sd ~0.066, so a lone draw lands outside [0.42, 0.58] about
    one time in twenty by construction — which is an alarm that fires on nothing. The
    band therefore applies to the MEAN of the null, and the spread is reported so a
    reader can see how wide "chance" actually is here.

    Shuffling within cohort, not globally, is deliberate: a global shuffle would also
    destroy the cohort/prevalence structure and could collapse for the wrong reason.
    """
    draws: List[float] = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed + d)
        y_perm = a.y.copy()
        for c in sorted(set(a.cohort.tolist())):
            m = a.cohort == c
            vals = y_perm[m].copy()
            rng.shuffle(vals)
            y_perm[m] = vals
        shuffled = Aligned(a.X, y_perm, a.cohort, a.sample_ids, a.patient_ids, a.symbols)
        draws.append(float(loco(shuffled, seed=seed)["mean_auroc"]))
    arr = np.array([v for v in draws if not np.isnan(v)])
    return {"mean": float(arr.mean()) if arr.size else float("nan"),
            "sd": float(arr.std()) if arr.size else float("nan"),
            "min": float(arr.min()) if arr.size else float("nan"),
            "max": float(arr.max()) if arr.size else float("nan"),
            "n_draws": int(arr.size), "draws": [round(v, 4) for v in arr.tolist()]}
