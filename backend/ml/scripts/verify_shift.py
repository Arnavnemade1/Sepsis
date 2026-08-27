"""Adversarial checks on the label-free deployment-transfer estimators.

Three of these checks pass by demonstrating a **failure to recover**, and that is
deliberate. The literature supplies a proven impossibility boundary — no label-free
method is uniformly valid under joint shift — so a suite in which every correction
always works would be evidence that the corrections are not doing what they claim.
The point is to show each remedy fixing the shift it addresses and *not* fixing the
ones it does not.

    python -m backend.ml.scripts.verify_shift
"""
from __future__ import annotations

import numpy as np

from backend.ml import shift

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def site(n=40000, base=-3.9, x_shift=0.0, concept=1.0, seed=0):
    """A site with a perfectly calibrated scorer.

    ``x_shift`` moves the covariate distribution (covariate shift). ``concept`` scales
    the covariate->logit relationship (concept shift). Labels are drawn from the true
    conditional, so any recovery failure below is the method's, not the data's.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(x_shift, 1.0, size=(n, 3))
    logit = base + concept * X[:, 0]
    p_true = _sigmoid(logit)
    y = (rng.random(n) < p_true).astype(int)
    return X, y, p_true


def resample_to_prior(X, y, p, pi_target, seed=0):
    """Subsample to a chosen prevalence — pure label shift, P(X|Y) untouched."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = len(pos)
    n_neg = int(round(n_pos * (1 - pi_target) / pi_target))
    n_neg = min(n_neg, len(neg))
    n_pos = min(n_pos, int(round(n_neg * pi_target / (1 - pi_target))) or n_pos)
    keep = np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
    rng.shuffle(keep)
    return X[keep], y[keep], p[keep]


# --------------------------------------------------------------------------- #
def check_pure_label_shift():
    """1. Under pure label shift, EM and BBSE both recover the true prior."""
    X, y, p = site(n=120000, seed=1)
    pi_src = float(y.mean())
    Xt, yt, pt = resample_to_prior(X, y, p, pi_target=pi_src * 0.55, seed=2)
    pi_true = float(yt.mean())
    pi_em, _ = shift.estimate_prior_em(pt, pi_src, p_src=p)
    pi_bb = shift.estimate_prior_bbse(y, p, pt, threshold=0.05)
    check("EM recovers the target prior under pure label shift",
          abs(pi_em - pi_true) / pi_true < 0.15,
          f"true {pi_true:.4f}  EM {pi_em:.4f}  (src {pi_src:.4f})")
    # The anchoring bug this check caught: referenced to mean(y_src) instead of
    # mean(p_src), the same call returned 0.0005. Assert the failure mode stays dead.
    pi_bad, _ = shift.estimate_prior_em(pt, pi_src, p_src=None)
    check("...and mis-anchoring to the label prior is what used to collapse it",
          pi_bad < pi_true * 0.5,
          f"unanchored EM = {pi_bad:.5f} vs true {pi_true:.4f} — collapses, as documented")
    check("BBSE recovers it too", abs(pi_bb - pi_true) / pi_true < 0.25,
          f"true {pi_true:.4f}  BBSE {pi_bb:.4f}")


def check_prior_correction_fixes_decision_not_ranking():
    """2. Prior correction is monotone: it moves the DECISION and cannot move AUROC.

    This is the whole thesis of the module in one assertion — the cross-site AUROC drop
    was never the fixable part.
    """
    from sklearn.metrics import roc_auc_score
    X, y, p = site(n=60000, seed=3)
    pi_src = float(y.mean())
    Xt, yt, pt = resample_to_prior(X, y, p, pi_target=pi_src * 0.5, seed=4)
    adj = shift.adjust_for_prior(pt, pi_src, float(yt.mean()))
    check("prior correction leaves AUROC exactly unchanged",
          abs(roc_auc_score(yt, pt) - roc_auc_score(yt, adj)) < 1e-9,
          f"AUROC {roc_auc_score(yt, pt):.6f}")
    thr = 0.05
    before, after = float((pt >= thr).mean()), float((adj >= thr).mean())
    check("...while changing how many rows cross a fixed threshold", before != after,
          f"alarm rate {before:.4f} -> {after:.4f}")


def check_shift_types_are_distinguishable():
    """3. The decomposition works: label shift and covariate shift look different.

    This is the check the whole module rests on. If the two cannot be told apart from
    unlabeled data, then "diagnose, then apply the licensed remedy" is not a method,
    it is a coin flip.
    """
    # -- pure label shift: P(X|Y) fixed, only the mixing weight moves --------- #
    X, y, p = site(n=90000, seed=5)
    pi_src = float(y.mean())
    Xt, yt, pt = resample_to_prior(X, y, p, pi_target=pi_src * 0.5, seed=6)
    _, auc_label = shift.covariate_weights(X, Xt, seed=7)

    # -- pure covariate shift: P(X) moves, P(Y|X) untouched ------------------- #
    Xs2, ys2, ps2 = site(n=60000, x_shift=0.0, seed=8)
    Xt2, yt2, pt2 = site(n=60000, x_shift=1.2, seed=9)      # same conditional => same model
    _, auc_cov = shift.covariate_weights(Xs2, Xt2, seed=10)

    check("pure label shift is nearly invisible to a domain classifier",
          auc_label < 0.60, f"domain AUC {auc_label:.3f}")
    check("pure covariate shift is loud to it", auc_cov > 0.70,
          f"domain AUC {auc_cov:.3f}")
    check("...so the two shift types are separable from unlabeled data alone",
          auc_cov - auc_label > 0.15, f"gap {auc_cov - auc_label:+.3f}")


def check_prior_correction_is_not_a_covariate_remedy():
    """3b. PLANTED: prior correction HARMS a covariate-shifted site.

    The expected result here was "it does not help". The measurement was worse than
    that and more useful: under pure covariate shift the source model is *already*
    calibrated on the target (P(Y|X) never moved), and correcting its prior
    double-counts a shift the model is handling through X — mean calibration goes
    from 1.01 to 11.4. Applying the wrong remedy is not a wasted step, it is an
    actively harmful one, which is the whole argument for diagnosing first.
    """
    def cal_ratio(y_t, p_t):
        return float(np.mean(p_t)) / max(float(np.mean(y_t)), 1e-12)

    X, y, p = site(n=90000, seed=11)
    pi_src = float(y.mean())
    Xt, yt, pt = resample_to_prior(X, y, p, pi_target=pi_src * 0.5, seed=12)
    em, _ = shift.estimate_prior_em(pt, pi_src, p_src=p)
    lab_before, lab_after = cal_ratio(yt, pt), cal_ratio(yt, shift.adjust_for_prior(pt, float(p.mean()), em))

    Xs2, ys2, ps2 = site(n=60000, x_shift=0.0, seed=13)
    Xt2, yt2, pt2 = site(n=60000, x_shift=1.2, seed=14)
    em2, _ = shift.estimate_prior_em(pt2, float(ys2.mean()), p_src=ps2)
    cov_before = cal_ratio(yt2, pt2)
    cov_after = cal_ratio(yt2, shift.adjust_for_prior(pt2, float(ps2.mean()), em2))

    check("prior correction moves mean calibration toward 1 under label shift",
          abs(lab_after - 1) < abs(lab_before - 1),
          f"cal ratio {lab_before:.3f} -> {lab_after:.3f}")
    check("...but is already calibrated on a covariate-shifted site",
          abs(cov_before - 1) < 0.20, f"cal ratio before {cov_before:.3f}")
    check("...and applying it there ACTIVELY breaks calibration (diagnose first)",
          abs(cov_after - 1) > 3 * abs(cov_before - 1) + 0.5,
          f"cal ratio {cov_before:.3f} -> {cov_after:.3f}")


def check_concept_shift_recovers_nothing():
    """4. PLANTED: under concept shift neither remedy recovers. The boundary."""
    from sklearn.metrics import roc_auc_score
    Xs, ys, ps = site(n=60000, concept=1.0, seed=8)
    # P(Y|X) itself changes sign-strength: the mapping the model learned is now wrong.
    Xt, yt, _ = site(n=60000, concept=-1.0, seed=9)
    pt = _sigmoid(-3.9 + 1.0 * Xt[:, 0])                     # source model applied to target
    auc_before = roc_auc_score(yt, pt)
    pi_em, _ = shift.estimate_prior_em(pt, float(ys.mean()), p_src=ps)
    adj = shift.adjust_for_prior(pt, float(ps.mean()), pi_em)
    check("concept shift leaves the source model below chance, and prior correction "
          "cannot lift it", roc_auc_score(yt, adj) < 0.55,
          f"AUROC {auc_before:.3f} -> {roc_auc_score(yt, adj):.3f}")
    _, auc = shift.covariate_weights(Xs, Xt, seed=10)
    check("...and the domain classifier sees nothing, because P(X) never moved",
          auc < 0.60, f"domain AUC {auc:.3f} — the shift is invisible to covariates")


def check_label_free_contract():
    """5. No estimator may touch target labels."""
    X, y, p = site(n=30000, seed=11)
    pi_src = float(y.mean())
    Xt, yt, pt = resample_to_prior(X, y, p, pi_target=pi_src * 0.6, seed=12)
    a, _ = shift.estimate_prior_em(pt, pi_src, p_src=p)
    b = shift.estimate_prior_bbse(y, p, pt, threshold=0.05)
    w1, auc1 = shift.covariate_weights(X, Xt, seed=13)
    # Same call, target labels destroyed. Nothing may move.
    r1 = shift.diagnose(y, p, X, pt, Xt, 0.05, y_tgt_for_report_only=yt, seed=13)
    r2 = shift.diagnose(y, p, X, pt, Xt, 0.05,
                        y_tgt_for_report_only=np.full(len(yt), np.nan), seed=13)
    same = (r1.pi_target_em == r2.pi_target_em and r1.pi_target_bbse == r2.pi_target_bbse
            and r1.domain_auc == r2.domain_auc)
    check("destroying target labels changes no estimate", same,
          f"EM {r1.pi_target_em:.6f} == {r2.pi_target_em:.6f}")
    check("...and the true target prior is recorded for scoring only",
          np.isnan(r2.pi_target_true) and not np.isnan(r1.pi_target_true))


def check_no_covariate_shift_is_reported_as_such():
    """6. Two draws from the SAME distribution must read as 'nothing to correct'."""
    Xs, _, _ = site(n=50000, seed=14)
    Xt, _, _ = site(n=50000, seed=15)
    w, auc = shift.covariate_weights(Xs, Xt, seed=16)
    r = shift.ShiftReport(0.02, 0.02, 0.02, 0.02, 3, auc, float(w.mean()), 1.0)
    check("a same-distribution pair yields a chance-level domain classifier", auc < 0.60,
          f"domain AUC {auc:.3f}")
    check("...and is reported as NO covariate shift", not r.covariate_shift_detected())
    check("...with weights sitting at 1.0", abs(float(w.mean()) - 1.0) < 0.25,
          f"mean weight {w.mean():.3f}")


def check_weighted_quantile():
    """7. The weighted quantile reduces to the ordinary one at uniform weights."""
    rng = np.random.default_rng(17)
    v = rng.normal(size=5000)
    for q in (0.5, 0.9, 0.95):
        wq = shift.weighted_quantile(v, np.ones_like(v), q)
        eq = float(np.quantile(v, q, method="lower"))
        if abs(wq - eq) > 1e-6:
            check(f"uniform weights reproduce the empirical quantile at q={q}", False,
                  f"{wq:.6f} vs {eq:.6f}")
            return
    check("uniform weights reproduce the empirical quantile (q=0.5/0.9/0.95)", True)
    # And a lopsided weighting must move it, or the function is doing nothing.
    w = np.where(v > 0, 10.0, 0.1)
    check("...and lopsided weights move it", 
          shift.weighted_quantile(v, w, 0.5) > shift.weighted_quantile(v, np.ones_like(v), 0.5),
          f"{shift.weighted_quantile(v, w, 0.5):.4f} > {shift.weighted_quantile(v, np.ones_like(v), 0.5):.4f}")


# --------------------------------------------------------------------------- #
def main() -> None:
    print("Label-free deployment transfer — adversarial checks\n")
    for fn in (check_pure_label_shift, check_prior_correction_fixes_decision_not_ranking,
               check_shift_types_are_distinguishable,
               check_prior_correction_is_not_a_covariate_remedy,
               check_concept_shift_recovers_nothing,
               check_label_free_contract, check_no_covariate_shift_is_reported_as_such,
               check_weighted_quantile):
        fn()
    ok, n = sum(results), len(results)
    print(f"\n{ok}/{n} checks passed")
    raise SystemExit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
