"""The mutation catalogue — one entry per real failure mode this repo has suffered.

Every mutation here corresponds to a mistake that was actually made in this project and
cost something. They are not mutation *operators* chosen for ease of implementation;
they are the specific ways a scientific claim has been observed to go wrong here.

Reading order, roughly by what each has cost:

    fit-identity        3 false findings in one investigation
    causality           would have posted a lead time made of hindsight
    exchangeability     an interval several times too narrow
    class-conditioning  the whole conformal construction going degenerate at 2%
    naive-statistic     an estimator that reads positive on pure noise
    label-leak          a deployment method silently becoming an oracle
    floor-removal       a quantile estimated from two patients, with a guarantee on it
"""
from __future__ import annotations

import numpy as np

from backend.ml.falsify import mutation

CONF = "backend.ml.scripts.verify_conformal"
SHIFT = "backend.ml.scripts.verify_shift"
MR = "backend.ml.scripts.verify_minirocket"
OP = "backend.ml.scripts.verify_operating_point"
FID = "backend.ml.scripts.verify_fitid"


# --------------------------------------------------------------------------- #
# causality — a transform that reads forward in time
# --------------------------------------------------------------------------- #
@mutation("causality", MR, ["a future spike moves NO earlier hour"],
          note="the textbook centred MiniRocket kernel; reads (K-1)*d/2 hours ahead")
def centred_convolution(mod):
    import backend.ml.models.minirocket as M
    K = M.KERNEL_LEN

    def conv(x, w, d):
        T = x.shape[0]
        y = np.zeros(T, dtype=np.float32)
        half = (K - 1) // 2 * d
        for j in range(K):
            sh = (K - 1 - j) * d - half            # negative shift => future tap
            y += w[j] * x[np.clip(np.arange(T) - sh, 0, T - 1)]
        return y
    M.causal_conv = conv
    mod.causal_conv = conv


@mutation("causality", MR, ["a future spike moves NO earlier hour"],
          note="whole-stay PPV pooling: every hour sees the entire admission")
def whole_series_ppv(mod):
    import backend.ml.models.minirocket as M

    def ppv(ind, window):
        return np.full(ind.shape[0], float(ind.mean()), dtype=np.float32)
    M.trailing_ppv = ppv
    mod.trailing_ppv = ppv


# --------------------------------------------------------------------------- #
# exchangeability — the unit of resampling
# --------------------------------------------------------------------------- #
@mutation("exchangeability", MR, ["rewriting one patient leaves every other patient"],
          note="treat the whole cohort as one series; stays bleed into each other")
def no_patient_split(mod):
    import backend.ml.models.minirocket as M

    def bounds(groups, hour_index):
        order = np.lexsort((hour_index, groups))
        return order, np.array([0, len(groups)])
    M.CausalMiniRocket._patient_bounds = staticmethod(bounds)


@mutation("exchangeability", MR, ["a shuffled row order yields identical per-row"],
          note="drop the hour sort; 'causal' over arrival order is meaningless")
def no_hour_sort(mod):
    import backend.ml.models.minirocket as M

    def bounds(groups, hour_index):
        order = np.argsort(groups, kind="stable")
        g = groups[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        return order, np.r_[starts, len(g)]
    M.CausalMiniRocket._patient_bounds = staticmethod(bounds)


@mutation("exchangeability", CONF, ["row-split calibration overstates coverage"],
          note="split rows, not patients — the leak that flatters every coverage number")
def conformal_row_split(mod):
    mod._cal_test_by_row = mod._cal_test_by_patient


@mutation("exchangeability", CONF, ["macro (per-patient) coverage differs from micro"],
          note="report macro as micro, hiding within-stay correlation entirely")
def macro_equals_micro(mod):
    import backend.ml.conformal as cf
    orig = cf._cov

    def patched(covered, y, groups):
        d = orig(covered, y, groups)
        return dict(d, macro=d["micro"], macro_pos=d["micro_pos"], macro_neg=d["micro_neg"])
    cf._cov = patched
    mod.cf = cf


# --------------------------------------------------------------------------- #
# class-conditioning — the fix that keeps conformal alive at 2% prevalence
# --------------------------------------------------------------------------- #
@mutation("class-conditioning", CONF, ["...while class-conditional does admit it"],
          note="pool both classes into one quantile; label 1 never enters a set again")
def pooled_quantile(mod):
    import backend.ml.conformal as cf

    def pooled(stratum, y, p, groups, alpha, fallback, mh, mp, w=None):
        s_true = np.where(y == 1, 1.0 - p, p)
        q = cf.conformal_quantile(s_true, alpha)
        return cf.CellBand(stratum=stratum, lo=1.0 - q, hi=q, q_pos=q, q_neg=q,
                           n_hours_pos=int((y == 1).sum()), n_hours_neg=int((y == 0).sum()),
                           n_patients_pos=0, n_patients_neg=0,
                           fallback_pos=False, fallback_neg=False)
    cf._cell_band = pooled
    mod.cf = cf


# --------------------------------------------------------------------------- #
# naive-statistic — a corrected estimator replaced by its uncorrected form
# --------------------------------------------------------------------------- #
@mutation("naive-statistic", CONF, ["(n+1) correction reaches nominal"],
          note="the textbook empirical quantile; under-covers by ~1/n forever")
def plain_empirical_quantile(mod):
    import backend.ml.conformal as cf
    cf.conformal_quantile = lambda s, a: float(np.quantile(np.asarray(s, float), 1 - a))
    mod.cf = cf


@mutation("naive-statistic", SHIFT, ["EM recovers the target prior"],
          note="anchor EM to the label prior; a 3.6% calibration gap collapses it to 0")
def em_mis_anchored(mod):
    import backend.ml.shift as sh
    orig = sh.estimate_prior_em
    sh.estimate_prior_em = lambda p_tgt, pi_src, p_src=None, **k: orig(p_tgt, pi_src, None, **k)
    mod.shift = sh


@mutation("naive-statistic", SHIFT, ["...and lopsided weights move it"],
          note="a weighted quantile that ignores its weights. NOTE the target check: "
               "'uniform weights reproduce the empirical quantile' is SATISFIED by this "
               "mutation, since ignoring the weights is exactly what uniform weights do. "
               "The engine reported this as a survivor and the fault was in the mutation "
               "spec, not the suite — which is the audit working on its own author.")
def weighted_quantile_ignores_weights(mod):
    import backend.ml.shift as sh
    sh.weighted_quantile = lambda v, w, q: float(
        np.quantile(np.asarray(v, float), q, method="lower"))
    mod.shift = sh


@mutation("naive-statistic", SHIFT, ["BBSE recovers it too"],
          note="BBSE that just echoes the source prior back")
def bbse_echoes_source(mod):
    import backend.ml.shift as sh
    sh.estimate_prior_bbse = lambda y, ps, pt, threshold: float(
        np.mean(np.asarray(y).astype(int)))
    mod.shift = sh


@mutation("naive-statistic", SHIFT, ["...while changing how many rows cross"],
          note="prior correction as the identity map — moves nothing at all")
def prior_correction_identity(mod):
    import backend.ml.shift as sh
    sh.adjust_for_prior = lambda p, a, b: np.asarray(p, dtype=float)
    mod.shift = sh


# --------------------------------------------------------------------------- #
# label-leak — a label-free estimator handed the answers
# --------------------------------------------------------------------------- #
@mutation("label-leak", SHIFT, ["...and the true target prior is recorded for scoring only"],
          note="_safe_prior invents a prevalence from all-NaN labels")
def safe_prior_ignores_nan(mod):
    import backend.ml.shift as sh
    sh._safe_prior = lambda y: 0.02 if y is not None else float("nan")
    mod.shift = sh


@mutation("label-leak", SHIFT, ["pure label shift is nearly invisible to a domain"],
          note="a domain classifier that always reports shift, so the diagnosis is fixed")
def domain_auc_always_high(mod):
    import backend.ml.shift as sh
    orig = sh.covariate_weights
    sh.covariate_weights = lambda Xs, Xt, **k: (orig(Xs, Xt, **k)[0], 0.95)
    mod.shift = sh


# --------------------------------------------------------------------------- #
# floor-removal — a size guard disabled, leaving a guarantee on nothing
# --------------------------------------------------------------------------- #
@mutation("floor-removal", CONF, ["undersized cell falls back to marginal"],
          note="no fallback: a cell of 60 correlated hours fits its own tail quantile")
def no_small_cell_fallback(mod):
    import backend.ml.conformal as cf
    orig = cf._cell_band
    cf._cell_band = lambda st, y, p, g, a, fb, mh, mp, w=None: orig(st, y, p, g, a, None, mh, mp, w)
    mod.cf = cf


@mutation("floor-removal", MR, ["estimate_bytes matches the real output size"],
          note="a memory guard that lies; the 11.6 GB transform ships unnoticed")
def estimate_bytes_lies(mod):
    import backend.ml.models.minirocket as M
    M.CausalMiniRocket.estimate_bytes = lambda self, n: 1


# --------------------------------------------------------------------------- #
# fit-identity — the family that cost three false findings
# --------------------------------------------------------------------------- #
@mutation("fit-identity", CONF, ["save/load round-trips bands, spec and detector identity"],
          note="drop the detector stamp; bands become readable against any fit")
def conformal_forgets_detector(mod):
    import backend.ml.conformal as cf
    orig = cf.ConformalBands.to_json

    def patched(self):
        d = orig(self)
        d["detector"] = "unknown"
        return d
    cf.ConformalBands.to_json = patched
    mod.cf = cf


@mutation("fit-identity", OP, ["the detector whose scores these are is recorded"],
          note="the operating point forgets which model its threshold was measured on")
def operating_point_forgets_detector(mod):
    import backend.ml.operating_point as op
    orig = op.derive

    def patched(*a, **k):
        k["detector"] = "unknown"
        return orig(*a, **k)
    op.derive = patched
    mod.opmod = op


@mutation("fit-identity", OP, ["the in-range rule pulls it back"],
          note="disable the in-range rule: the threshold wanders into the sparse tail")
def operating_point_no_in_range(mod):
    import backend.ml.operating_point as op
    orig = op.derive

    def patched(*a, **k):
        k["require_in_range"] = False
        return orig(*a, **k)
    op.derive = patched
    mod.opmod = op


@mutation("fit-identity", OP, ["setting it on the upper bound breaches strictly less"],
          note="drop the bootstrap margin; the cap holds only on the patients that set it")
def operating_point_no_margin(mod):
    import backend.ml.operating_point as op
    orig = op.derive

    def patched(*a, **k):
        k["far_margin"] = False
        return orig(*a, **k)
    op.derive = patched
    mod.opmod = op


# --------------------------------------------------------------------------- #
# fit-identity, continued — the guard built BECAUSE of the three false findings
# is itself audited, because a provenance check that cannot fail is the worst
# possible thing to have: it would license exactly the comparisons it exists to stop.
# --------------------------------------------------------------------------- #
@mutation("fit-identity", FID,
          ["an absent fingerprint is refused, not assumed compatible"],
          note="treat a MISSING fingerprint as compatible — the permissive rule that "
               "would have waved through all three of the original mismatches. TARGET "
               "NOTE: this is NOT caught by the three-false-findings check, which "
               "compares real fingerprints and is therefore unaffected. The engine "
               "reported that as a survivor and the fault was the mutation spec naming "
               "the wrong defender — the second time the audit corrected its author, "
               "and the argument for naming a specific check rather than accepting "
               "'something failed'.")
def unknown_reads_as_compatible(mod):
    import backend.ml.fitid as F
    orig = F.require_same

    def lax(a, b, what="these artifacts", strict=True):
        if a is None or b is None:
            return
        return orig(a, b, what, strict)
    F.require_same = lax
    mod.require_same = lax


@mutation("fit-identity", FID,
          ["all three of this session's false findings are now refused"],
          note="a digest that ignores the spec, so dev-vs-setA compares as identical")
def digest_ignores_spec(mod):
    import backend.ml.fitid as F
    F.FIELDS = tuple(f for f in F.FIELDS if f != "spec")
    mod.fitid = F


@mutation("fit-identity", FID, ["identical fits produce identical digests"],
          note="a non-deterministic digest — provenance that cannot be compared at all")
def digest_is_random(mod):
    import backend.ml.fitid as F
    import uuid
    F.FitId.digest = lambda self: uuid.uuid4().hex[:12]
    mod.fitid = F
