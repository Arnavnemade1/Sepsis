"""Label-free deployment transfer — re-deriving the DECISION at a new hospital.

WHY THIS EXISTS
---------------
Cross-hospital, this detector's ranking half-transfers and its decision does not
transfer at all:

    within-site   AUROC 0.7811   utility +0.3849
    setA -> setB  AUROC 0.6797   utility +0.1759      (-13% vs -54%)

AUROC is rank-only and therefore blind to the thing that actually broke. Hospital
prevalence differs by **1.54x** (setA hour-prevalence 0.0217, setB 0.0141), so a model
calibrated at A systematically over-states risk at B; the operating point and the
conformal bands were both derived against A's score distribution and land in the wrong
place on B's. The conformal layer shows the signature clearly — septic-hour coverage
collapses 0.929 -> 0.739 while non-septic coverage *holds* at 0.971, which is a
per-class collapse that a marginal coverage number would have hidden entirely.

THE DECOMPOSITION IS THE POINT
------------------------------
A cross-site gap has three causes with three different remedies, and one of them has
no remedy at all:

    label (prior) shift    P(Y) moves, P(X|Y) stable   -> prior correction, label-free
    covariate shift        P(X) moves, P(Y|X) stable   -> importance weights, label-free
    concept shift          P(Y|X) itself moves         -> NOTHING label-free fixes this

So this module does not "adapt the model". It estimates which of the three is
happening, applies the remedies the diagnosis licenses, and leaves the residual
explicitly named as concept shift. A number that cannot be fixed is still worth
measuring: if concept shift dominates, that is the explanation for why cross-hospital
sepsis models fail, and no amount of adaptation engineering will change it.

THE LABEL-FREE CONTRACT
-----------------------
Every estimator here takes **unlabeled target covariates only**. Target labels exist in
our evaluation data, and using them anywhere in the adaptation path would silently turn
a deployment method into an oracle. ``verify_shift.py`` enforces this by passing
all-NaN target labels and asserting nothing changes.

THE BOUNDARY, STATED UP FRONT
-----------------------------
Label-free correction is provably limited: no label-free method is uniformly valid and
efficient per class under joint shift, and per-class recovery stops once *marginal*
coverage itself breaks. This module is therefore expected to recover part of the gap,
not all of it, and the honest deliverable is where it stops.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Prior (label) shift
# --------------------------------------------------------------------------- #
def adjust_for_prior(p: np.ndarray, pi_src: float, pi_tgt: float) -> np.ndarray:
    """Re-weight calibrated probabilities from a source prior to a target prior.

    The standard prior-correction identity: multiply the odds by the prior ratio.

        p' = p*r / (p*r + (1-p)*s),   r = pi_t/pi_s,  s = (1-pi_t)/(1-pi_s)

    This is exact when only P(Y) has moved and P(X|Y) is unchanged (label shift). It is
    a *monotone* map, so it cannot change AUROC by a single point — which is precisely
    why the cross-site AUROC drop was never the thing to chase. What it changes is
    where a fixed probability threshold falls, i.e. the decision.
    """
    p = np.asarray(p, dtype=float)
    pi_src = float(np.clip(pi_src, EPS, 1 - EPS))
    pi_tgt = float(np.clip(pi_tgt, EPS, 1 - EPS))
    r, s = pi_tgt / pi_src, (1 - pi_tgt) / (1 - pi_src)
    num = p * r
    return num / (num + (1 - p) * s + EPS)


def estimate_prior_em(p_tgt: np.ndarray, pi_src: float,
                      p_src: Optional[np.ndarray] = None,
                      max_iter: int = 500, tol: float = 1e-10) -> Tuple[float, int]:
    """Saerens-Latinne-Decaestecker EM estimate of the target prior.

    Iterates: recompute the posterior responsibilities under the current prior
    estimate, then set the next estimate to their mean. Uses the target's
    **covariates only** — ``p_tgt`` are the source model's predictions on unlabeled
    target rows.

    ANCHORING, WHICH IS NOT A DETAIL
    --------------------------------
    ``p_src`` is the source model's predictions on held-out SOURCE rows, and passing
    it matters more than it looks. The correction has to be referenced to the prior
    the model's *outputs* actually encode, ``mean(p_src)`` — not to the source label
    prevalence ``mean(y_src)``. When they differ, and they always differ a little, the
    ratio between them becomes a constant multiplicative factor in the fixed-point
    map, and EM has no interior fixed point at all: it slides monotonically to zero.

    That is not hypothetical. On this project's own synthetic label-shift check the
    source model under-predicted its mean by 3.6% (mean p 0.03074 vs prevalence
    0.03189), the map became ``pi -> 0.964 * pi``, and EM returned 0.0005 against a
    true prior of 0.0175. Anchored to ``mean(p_src)`` the same data gives 0.0162.

    Returns ``(pi_tgt, iterations)``. EM suits a well-calibrated source model (ours is,
    ECE ~0.002); :func:`estimate_prior_bbse` never asks for calibration at all, so
    disagreement between the two is itself a diagnostic.
    """
    p_tgt = np.asarray(p_tgt, dtype=float)
    ref = float(np.mean(np.asarray(p_src, dtype=float))) if p_src is not None else float(pi_src)
    ref = float(np.clip(ref, EPS, 1 - EPS))
    pi = ref
    for it in range(1, max_iter + 1):
        nxt = float(np.mean(adjust_for_prior(p_tgt, ref, pi)))
        nxt = float(np.clip(nxt, EPS, 1 - EPS))
        if abs(nxt - pi) < tol:
            return nxt, it
        pi = nxt
    return pi, max_iter


def estimate_prior_bbse(y_src: np.ndarray, p_src: np.ndarray, p_tgt: np.ndarray,
                        threshold: float) -> float:
    """Black-Box Shift Estimation (Lipton et al.) of the target prior.

    Builds the source joint P(yhat, y) at a fixed threshold, then solves
    ``C w = mu_target`` for the class-weight vector ``w = pi_t / pi_s`` and returns the
    implied target prior. Unlike EM this never asks the model's probabilities to be
    calibrated — only that its *hard* decisions behave the same way at both sites — so
    it degrades differently, which is what makes the pair informative.

    ``y_src`` / ``p_src`` must come from held-out source rows (out-of-fold), not the
    fit: an in-sample confusion matrix is sharper than anything seen at serve time.
    """
    y_src = np.asarray(y_src).astype(int)
    yhat_s = (np.asarray(p_src, dtype=float) >= threshold).astype(int)
    yhat_t = (np.asarray(p_tgt, dtype=float) >= threshold).astype(int)

    # C[j, k] = P_source(yhat = j, y = k)
    C = np.zeros((2, 2), dtype=float)
    n = len(y_src)
    for j in (0, 1):
        for k in (0, 1):
            C[j, k] = np.sum((yhat_s == j) & (y_src == k)) / max(n, 1)
    mu = np.array([np.mean(yhat_t == 0), np.mean(yhat_t == 1)], dtype=float)

    pi_src = float(np.mean(y_src))
    if np.linalg.matrix_rank(C) < 2 or abs(np.linalg.det(C)) < 1e-10:
        # A singular confusion matrix means the threshold does not separate the
        # classes at all; BBSE is undefined there. Returning the source prior (i.e.
        # "no detectable shift") is the honest answer and is reported as such rather
        # than dressed up as an estimate.
        return pi_src
    w = np.linalg.solve(C, mu)                       # w[k] = pi_t(k) / pi_s(k)
    pi_t = float(w[1] * pi_src)
    if not np.isfinite(pi_t) or pi_t <= 0.0 or pi_t >= 1.0:
        # BBSE can return a NEGATIVE class weight when the confusion matrix is
        # ill-conditioned — which is what happened on real setA->setB data. Clipping
        # that into range reports it as "prevalence 0.0000", a confident-looking
        # number that is really an estimator failure. NaN says what actually happened
        # so a caller can fall back rather than act on garbage.
        return float("nan")
    return pi_t


# --------------------------------------------------------------------------- #
# Covariate shift
# --------------------------------------------------------------------------- #
def covariate_weights(X_src: np.ndarray, X_tgt: np.ndarray, seed: int = 42,
                      max_rows: int = 60000, clip: float = 20.0
                      ) -> Tuple[np.ndarray, float]:
    """Importance weights w(x) = P_tgt(x)/P_src(x) via a domain classifier.

    Fits source-vs-target on covariates alone and converts the classifier's odds into
    a density ratio, correcting for the sampling ratio of the two pools. Returns
    ``(weights_for_X_src, auc)``.

    **The AUC is not diagnostics decoration — it is the finding.** A domain classifier
    at chance means the two hospitals' covariate distributions are indistinguishable,
    i.e. *there is no covariate shift to correct*, and the weights it produces are
    noise around 1.0. Reporting that honestly is the difference between "we applied
    importance weighting" and "importance weighting had nothing to do".

    Weights are clipped: an unclipped density ratio from a finite classifier puts
    almost all the mass on a handful of rows, and a weighted quantile over three
    effective points is not an estimate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    si = rng.choice(len(X_src), size=min(max_rows, len(X_src)), replace=False)
    ti = rng.choice(len(X_tgt), size=min(max_rows, len(X_tgt)), replace=False)
    Xs, Xt = np.asarray(X_src)[si], np.asarray(X_tgt)[ti]

    def clean(a):
        # Twice: nan_to_num first, then again after scaling. A column that is constant
        # in the sample scales to 0/0, and one inf survivor makes the whole logistic
        # fit emit overflow warnings and return garbage coefficients.
        return np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    X = clean(np.vstack([Xs, Xt]))
    d = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    sc = StandardScaler().fit(X)
    Xz = clean(sc.transform(X))
    clf = LogisticRegression(max_iter=300, C=0.1).fit(Xz, d)

    auc = float(roc_auc_score(d, clf.predict_proba(Xz)[:, 1]))
    pt = clf.predict_proba(clean(sc.transform(clean(X_src))))[:, 1]
    ratio = (len(Xs) / max(len(Xt), 1))
    w = (pt / np.clip(1 - pt, EPS, None)) * ratio
    return np.clip(w, 1.0 / clip, clip), auc


# --------------------------------------------------------------------------- #
def _safe_prior(y: Optional[np.ndarray]) -> float:
    """Mean of a label vector, or NaN when the labels are absent or unusable.

    ``np.asarray([nan]).astype(int)`` does not propagate NaN — it produces a large
    negative integer — so an all-NaN label vector would otherwise be recorded as a
    perfectly plausible prevalence. This is the guard that makes the label-free
    contract checkable.
    """
    if y is None:
        return float("nan")
    a = np.asarray(y, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        return float("nan")
    return float(np.mean(a.astype(int)))


@dataclass
class ShiftReport:
    """What kind of shift this is, measured — plus what that licenses."""
    pi_source: float
    pi_target_true: float          # evaluation only; NEVER used by any estimator
    pi_target_em: float
    pi_target_bbse: float
    em_iterations: int
    domain_auc: float
    weight_mean: float
    weight_ess_frac: float         # effective sample size / n, after weighting

    def prior_ratio_true(self) -> float:
        return self.pi_target_true / max(self.pi_source, EPS)

    def prior_ratio_em(self) -> float:
        return self.pi_target_em / max(self.pi_source, EPS)

    def covariate_shift_detected(self, auc_floor: float = 0.60) -> bool:
        """Is there covariate shift a classifier can actually see?

        0.60 rather than 0.50 because with 60k rows a domain classifier will find
        *some* separability in almost any pair of real datasets; below this the
        weights are not carrying information worth the variance they add.
        """
        return self.domain_auc >= auc_floor

    def to_json(self) -> Dict[str, object]:
        d = asdict(self)
        d["prior_ratio_true"] = self.prior_ratio_true()
        d["prior_ratio_em"] = self.prior_ratio_em()
        d["covariate_shift_detected"] = self.covariate_shift_detected()
        return d


def diagnose(y_src: np.ndarray, p_src: np.ndarray, X_src: np.ndarray,
             p_tgt: np.ndarray, X_tgt: np.ndarray, threshold: float,
             y_tgt_for_report_only: Optional[np.ndarray] = None,
             seed: int = 42) -> ShiftReport:
    """Estimate the shift between two sites from unlabeled target data.

    ``y_tgt_for_report_only`` is exactly what it says: it is recorded in the report so
    the estimate can be scored, and it is passed to no estimator. The argument is named
    to make a misuse read as a misuse at the call site.
    """
    pi_src = float(np.mean(np.asarray(y_src).astype(int)))
    pi_em, iters = estimate_prior_em(p_tgt, pi_src, p_src=p_src)
    pi_bbse = estimate_prior_bbse(y_src, p_src, p_tgt, threshold)
    w, auc = covariate_weights(X_src, X_tgt, seed=seed)
    ess = float((w.sum() ** 2) / max(np.sum(w ** 2), EPS)) / max(len(w), 1)
    return ShiftReport(
        pi_source=pi_src,
        pi_target_true=_safe_prior(y_tgt_for_report_only),
        pi_target_em=pi_em, pi_target_bbse=pi_bbse, em_iterations=iters,
        domain_auc=auc, weight_mean=float(np.mean(w)), weight_ess_frac=ess,
    )


# --------------------------------------------------------------------------- #
def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The q-th weighted quantile — the primitive weighted conformal needs.

    Sorts by value and walks the normalised cumulative weight. With uniform weights
    this reduces to the ordinary empirical quantile, which ``verify_shift.py`` asserts
    rather than assumes.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return float("inf")
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return float(v[-1])
    idx = int(np.searchsorted(cw, q * total, side="left"))
    return float(v[min(idx, len(v) - 1)])
