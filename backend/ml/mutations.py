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


# ═══════════════════════════════════════════════════════════════════════════ #
# EXTENSION — the remaining suites that share the check contract.
#
# Cost matters here in a way it did not for the ML suites: verify_subgroup takes
# ~130s per run and verify_alarm_policy ~35s, so each mutation against them costs
# that much again. They are tagged SLOW and excluded from the default audit; the
# fast eleven cover 392 additional checks for a handful of seconds each.
# ═══════════════════════════════════════════════════════════════════════════ #
COUP = "backend.physio.verify_coupling"
SELFIMP = "backend.ml.scripts.verify_selfimprove"
Z3S = "backend.aegis.verify_z3_safety"
AGENT = "backend.aegis.verify_agent"
DOCTOR = "backend.llm.verify_doctor"
SUBGRP = "backend.ml.scripts.verify_subgroup"
ALARM = "backend.ml.scripts.verify_alarm_policy"

#: Suites whose single run costs more than the whole rest of the audit combined.
SLOW_SUITES = {SUBGRP, ALARM}


# --------------------------------------------------------------------------- #
# naive-statistic — the coupling estimators, whose whole validity is a correction
# --------------------------------------------------------------------------- #
@mutation("naive-statistic", COUP, ["surrogate-corrected TE ~ 0 on independent signals"],
          note="drop the surrogate null: binned TE reads 0.025 bits on pure noise, so "
               "every pair of unrelated signals becomes a coupling finding")
def te_without_surrogate(mod):
    import backend.physio.coupling as C
    orig = C.transfer_entropy

    def raw(src, dst, bins=4, lag=1, n_surrogate=12, seed=0):
        r = orig(src, dst, bins, lag, n_surrogate, seed)
        return dict(r, te=r["te_raw"])            # uncorrected
    C.transfer_entropy = raw
    mod.cp = C


@mutation("naive-statistic", COUP, ["phase sync ~ 0 on independent signals"],
          note="skip the respiratory band-pass: Hilbert phase is only meaningful for a "
               "narrowband signal, and unfiltered this scored 0.716 on two independent "
               "random walks — higher than real coupled physiology")
def phase_sync_unfiltered(mod):
    import backend.physio.coupling as C
    from scipy.signal import hilbert
    import numpy as _np

    def ps(a, b, n=1, m=1, fs=C.HR_GRID_HZ, band=C.RESP_BAND):
        if len(a) < 16 or len(b) < 16:
            return 0.0
        k = min(len(a), len(b))
        pa = _np.angle(hilbert(_np.asarray(a[:k], float) - _np.mean(a[:k])))
        pb = _np.angle(hilbert(_np.asarray(b[:k], float) - _np.mean(b[:k])))
        return float(_np.abs(_np.mean(_np.exp(1j * (n * pa - m * pb)))))
    C.phase_sync = ps
    mod.cp = C


@mutation("causality", COUP, ["...while the decoupling control leaves the respiratory"],
          note="a decoupling control that also destroys the respiratory spectrum, so a "
               "monotone response would prove the estimator tracks amplitude, not timing")
def decouple_destroys_spectrum(mod):
    import backend.physio.coupling as C
    import numpy as _np
    C.decouple = lambda resp, alpha, seed=0: _np.asarray(resp, float) * (1.0 - alpha)
    mod.cp = C


# --------------------------------------------------------------------------- #
# floor-removal — the self-improvement acceptance rule, i.e. the refusal itself
# --------------------------------------------------------------------------- #
@mutation("floor-removal", SELFIMP, ["a significant but trivial gain is refused (min_gain)"],
          note="weaken the effect-size floor in the rule THE SUITE TESTS",
          targets="suite")
def selfimprove_local_no_min_gain(mod):
    orig = mod.apply_rule
    mod.apply_rule = lambda trials, alpha_c, min_gain: orig(trials, alpha_c, -1.0)


@mutation("floor-removal", SELFIMP, ["a large but noisy gain is refused (alpha)"],
          note="weaken the significance bar in the rule THE SUITE TESTS",
          targets="suite")
def selfimprove_local_no_alpha(mod):
    orig = mod.apply_rule
    mod.apply_rule = lambda trials, alpha_c, min_gain: orig(trials, 1.0, min_gain)


@mutation("fit-identity", SELFIMP, ["a significant but trivial gain is refused (min_gain)"],
          note="THE FINDING. Weaken the SHIPPED acceptance rule — RecursiveSelfImprover's "
               "own min_gain and alpha — and nothing notices, because verify_selfimprove "
               "never references RecursiveSelfImprover at all. It defines a local "
               "apply_rule() whose docstring says 'the shipped rule' and tests that. The "
               "0-of-16 refusal is the crown-jewel evidence for this project's refusal "
               "architecture, and it is verified against a re-implementation in the test "
               "file rather than against the code that produced it. This mutation is "
               "EXPECTED TO SURVIVE; the survival is the result.")
def selfimprove_shipped_rule_drifts(mod):
    import backend.ml.selfimprove as S
    imp = getattr(S, "RecursiveSelfImprover", None)
    if imp is None:
        raise RuntimeError("no RecursiveSelfImprover to weaken")
    orig_init = imp.__init__

    def loose(self, *a, **k):
        orig_init(self, *a, **k)
        self.min_gain = -1.0                       # accept any delta
        self.alpha = 1.0                           # accept any p-value
    imp.__init__ = loose
    mod.selfimprove = S


# --------------------------------------------------------------------------- #
# concurrency — the Z3 safety cage. A lock that is not held is not a cage.
# --------------------------------------------------------------------------- #
@mutation("concurrency", Z3S, ["no public function builds Z3 terms without the lock"],
          note="strip the guard from a public prover entry point; Z3 contexts are not "
               "thread-safe and a corrupted one previously returned garbage rather "
               "than raising")
def z3_unguarded_entrypoint(mod):
    src = getattr(mod, "_unguarded_public", None)
    if src is None:
        raise RuntimeError("suite does not expose _unguarded_public")
    mod._unguarded_public = lambda path: ["verify_protocol"]   # report a real breach


@mutation("concurrency", Z3S, ["no public function builds Z3 terms without the lock"],
          note="report a SECOND unguarded entry point. The earlier version of this "
               "mutation swapped ProverUnavailable for RuntimeError, which broke the "
               "suite's own exception handling and produced an ERROR rather than a "
               "FAIL — correctly scored as a survivor, since a crash is not evidence "
               "that a claim is defended.")
def z3_second_unguarded_entrypoint(mod):
    if not hasattr(mod, "_unguarded_public"):
        raise RuntimeError("suite does not expose _unguarded_public")
    mod._unguarded_public = lambda path: ["safety_model", "verify_protocol"]


# --------------------------------------------------------------------------- #
# confinement — the agent plane's read-only boundary
# --------------------------------------------------------------------------- #
@mutation("label-leak", AGENT, ["gets only"],
          note="index a POST-only route as readable. The read plane's guarantee is that "
               "every route reachable through os_read is a GET; listing a write route "
               "there is exactly how a read tool acquires a mutation.")
def agent_write_route_indexed_as_read(mod):
    import backend.aegis.server as srv
    rw = getattr(srv, "_READ_ROUTES", None)
    ww = getattr(srv, "_WRITE_ROUTES", None)
    if not isinstance(rw, dict) or not ww:
        raise RuntimeError("route indexes not available")
    victim = next((p for p in ww if p not in rw), None)
    if victim is None:
        raise RuntimeError("no write-only route to smuggle in")
    rw[victim] = ww[victim]
    mod.srv = srv
