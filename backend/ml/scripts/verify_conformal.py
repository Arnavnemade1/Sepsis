"""Adversarial checks on the conformal abstention layer.

A coverage guarantee is the easiest thing in this repo to fake: print ``1-α``, ship
it, and nothing ever contradicts you. So none of these checks assert that the layer
works. Each **plants the specific failure the guarantee is supposed to exclude** and
asserts the machinery reports it — including the two failures that are not bugs at
all but properties of conformal prediction people routinely forget:

* calibrating on rows instead of patients makes coverage look fine while it is not,
* distribution shift breaks the guarantee outright, and it must be seen to break.

Checks 2, 4 and 5 are therefore inverted on purpose: they PASS by demonstrating a
loss of coverage. A version of this file where everything passes by looking healthy
would be evidence of nothing.

    python -m backend.ml.scripts.verify_conformal
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from backend.ml import conformal as cf

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
def cohort(n_patients=1200, hours=30, seed=0, base=-3.9, patient_sd=0.0,
           score_shift=0.0, n_strata=3):
    """Patient-hours with a PERFECTLY CALIBRATED score column.

    ``p`` is the true conditional probability and ``y`` is drawn from it, so any
    coverage shortfall observed below is a property of the calibration procedure,
    never of a miscalibrated model. ``patient_sd`` adds a per-patient latent offset —
    the within-stay correlation that makes an ICU cohort non-exchangeable at the row
    level. ``score_shift`` perturbs the scores AFTER the labels are drawn, which is
    exactly what a new hospital does to a fixed model.
    """
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_patients), hours)
    z = base + rng.normal(0, 1.0, size=n_patients * hours)
    z += np.repeat(rng.normal(0, patient_sd, size=n_patients), hours)
    p_true = _sigmoid(z)
    y = (rng.random(p_true.size) < p_true).astype(int)
    p = np.clip(_sigmoid(z + score_shift), 1e-6, 1 - 1e-6)
    strata = np.array([f"m{i % n_strata}/s0" for i in range(p.size)], dtype=object)
    return y, p, groups, strata


def _cal_test_by_patient(groups, frac=0.5, seed=0):
    """Split PATIENTS. No stay may straddle the boundary."""
    rng = np.random.default_rng(seed)
    pts = np.array(sorted(set(groups.tolist())))
    rng.shuffle(pts)
    cal_pts = set(pts[: int(frac * len(pts))].tolist())
    m = np.array([g in cal_pts for g in groups])
    return np.flatnonzero(m), np.flatnonzero(~m)


def _cal_test_by_row(groups, frac=0.5, seed=0):
    """Split ROWS — the leak. The same patient lands on both sides."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(groups))
    cut = int(frac * len(groups))
    return idx[:cut], idx[cut:]


def _fit_cover(y, p, groups, strata, cal, te, alpha, **kw):
    b = cf.calibrate(y[cal], p[cal], groups[cal], strata[cal], alpha=alpha, **kw)
    return b, cf.coverage_report(b, y[te], p[te], groups[te], strata[te])


# --------------------------------------------------------------------------- #
def check_nominal_coverage():
    """1. On exchangeable data the guarantee holds — at both α, both classes."""
    for alpha in (0.05, 0.10):
        # One hour per patient => rows ARE exchangeable, the theorem's own setting.
        y, p, groups, strata = cohort(n_patients=40000, hours=1, seed=1)
        cal, te = _cal_test_by_patient(groups, seed=1)
        _, rep = _fit_cover(y, p, groups, strata, cal, te, alpha,
                            min_hours=0, min_patients=0)
        nom = 1 - alpha
        ok_pos = rep["overall"]["micro_pos"] >= nom - 0.03
        ok_neg = rep["overall"]["micro_neg"] >= nom - 0.01
        check(f"coverage >= nominal at alpha={alpha} (exchangeable)", ok_pos and ok_neg,
              f"pos {rep['overall']['micro_pos']:.3f} neg {rep['overall']['micro_neg']:.3f} "
              f"vs {nom:.2f}")


def check_plus_one_correction():
    """1b. The (n+1) correction is load-bearing, not cosmetic.

    Calibrate repeatedly on a SMALL set and measure coverage on everything else. The
    textbook empirical quantile under-covers by roughly 1/n; the conformal one does
    not. The assertion is deliberately two-sided — corrected reaches nominal AND
    plain falls short — because a one-sided "corrected >= plain" comparison is
    satisfied by equality and would still pass if the correction were deleted. (It
    did: this check was written that way first, and the mutation audit caught it.)
    """
    y, p, groups, strata = cohort(n_patients=8000, hours=1, seed=7)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(0)
    cov_cf, cov_plain = [], []
    for _ in range(300):
        s = rng.choice(neg, size=40, replace=False)
        rest = np.setdiff1d(neg, s)
        cov_cf.append((p[rest] <= cf.conformal_quantile(p[s], 0.05)).mean())
        cov_plain.append((p[rest] <= float(np.quantile(p[s], 0.95))).mean())
    m_cf, m_pl = float(np.mean(cov_cf)), float(np.mean(cov_plain))
    check("(n+1) correction reaches nominal where the plain quantile under-covers",
          m_cf >= 0.95 > m_pl, f"conformal {m_cf:.4f} vs plain {m_pl:.4f} (nominal 0.95)")


def check_row_split_leaks():
    """2. PLANTED LEAK: splitting rows instead of patients flatters coverage.

    With a strong per-patient offset, a row split puts near-duplicates of every
    calibration hour into the test set. Coverage then looks healthy while the
    honest patient-grouped split shows it is not. The check PASSES when the leak is
    visible — i.e. when the grouped number is the worse one.
    """
    y, p, groups, strata = cohort(n_patients=1500, hours=40, seed=3, patient_sd=2.2)
    a = 0.05
    _, rep_row = _fit_cover(y, p, groups, strata, *_cal_test_by_row(groups, seed=3), a,
                            min_hours=0, min_patients=0)
    _, rep_grp = _fit_cover(y, p, groups, strata, *_cal_test_by_patient(groups, seed=3), a,
                            min_hours=0, min_patients=0)
    row, grp = rep_row["overall"]["micro_pos"], rep_grp["overall"]["micro_pos"]
    check("row-split calibration overstates coverage vs patient-grouped", row > grp,
          f"row {row:.4f} > grouped {grp:.4f}")


def check_macro_exposes_correlation():
    """2b. micro/macro gap is a readout of within-stay correlation.

    Pooling hours lets a few long stays carry the number. Weighting each patient once
    does not. On a strongly correlated cohort the two must disagree; if they never
    did, reporting both would be pointless.
    """
    y, p, groups, strata = cohort(n_patients=1500, hours=40, seed=4, patient_sd=2.2)
    cal, te = _cal_test_by_patient(groups, seed=4)
    _, rep = _fit_cover(y, p, groups, strata, cal, te, 0.05, min_hours=0, min_patients=0)
    gap = abs(rep["overall"]["micro_pos"] - rep["overall"]["macro_pos"])
    check("macro (per-patient) coverage differs from micro under correlation", gap > 0.01,
          f"micro {rep['overall']['micro_pos']:.4f} macro {rep['overall']['macro_pos']:.4f}")


def check_small_cell_fallback():
    """3. An undersized cell falls back to the marginal band AND records it."""
    y, p, groups, strata = cohort(n_patients=1200, hours=30, seed=5)
    strata = strata.copy()
    strata[:60] = "tiny"          # 60 hours, far under the floors
    b = cf.calibrate(y, p, groups, strata, alpha=0.05,
                     min_hours=cf.MIN_CELL_HOURS, min_patients=cf.MIN_CELL_PATIENTS)
    tiny = b.cells["tiny"]
    recorded = tiny.fallback_pos and tiny.fallback_neg
    equal = (tiny.q_pos == b.marginal.q_pos and tiny.q_neg == b.marginal.q_neg)
    check("undersized cell falls back to marginal", recorded and equal,
          f"fallback_pos={tiny.fallback_pos} fallback_neg={tiny.fallback_neg}")
    # And the floor is what causes it: disable the floor and the cell fits its own.
    b0 = cf.calibrate(y, p, groups, strata, alpha=0.05, min_hours=0, min_patients=0)
    check("...and the floor, not chance, is what triggered it",
          not b0.cells["tiny"].fallback_neg, "floor=0 → cell fits its own quantile")


def check_shift_breaks_coverage():
    """4. PLANTED SHIFT: a new distribution breaks the guarantee, visibly.

    Calibrate on one cohort, serve another whose scores are systematically deflated.
    Coverage of the positive class must FALL BELOW nominal. A layer that still
    reported 0.95 here would be reporting the parameter, not the measurement.
    """
    y, p, groups, strata = cohort(n_patients=4000, hours=20, seed=6)
    b = cf.calibrate(y, p, groups, strata, alpha=0.05, min_hours=0, min_patients=0)
    y2, p2, g2, s2 = cohort(n_patients=4000, hours=20, seed=61, score_shift=-1.6)
    rep = cf.coverage_report(b, y2, p2, g2, s2)
    check("distribution shift is seen to break coverage",
          rep["overall"]["micro_pos"] < 0.95,
          f"shifted pos coverage {rep['overall']['micro_pos']:.3f} < 0.95")


def check_class_conditional_is_required():
    """5. PLANTED DEGENERACY: the marginal taxonomy collapses at 2% prevalence.

    Pooling both classes into one quantile makes the negatives set the bar. Label 1
    then needs ``1-p <= q`` with q small — impossible for a detector whose maximum
    output is ~0.17 — so no hour is ever called septic, and the high-scoring hours
    get an EMPTY set instead. This is why the module conditions on the class.
    """
    y, p, groups, strata = cohort(n_patients=6000, hours=20, seed=8)
    s_true = np.where(y == 1, 1.0 - p, p)          # nonconformity of the TRUE label
    q = cf.conformal_quantile(s_true, 0.05)        # one pooled, marginal quantile
    marg_has1 = int((1.0 - p <= q).sum())
    b = cf.calibrate(y, p, groups, strata, alpha=0.05, min_hours=0, min_patients=0)
    # Count hours whose SET CONTAINS 1 — "septic" or "uncertain". Counting
    # "anything but not_septic" also sweeps in the EMPTY set, which the pooled
    # quantile produces in bulk, and the check would then pass on the very
    # degeneracy it exists to expose.
    cc_has1 = int(np.isin(cf.predict_states(b, p, strata), ["septic", "uncertain"]).sum())
    check("marginal (non-class-conditional) never admits label 1", marg_has1 == 0,
          f"q={q:.4f} → needs p >= {1 - q:.4f}, max p = {p.max():.4f}")
    check("...while class-conditional does admit it", cc_has1 > 0,
          f"{cc_has1:,} hours admit label 1")


def check_serve_matches_derivation():
    """6. The serve-time path reproduces the evaluation path exactly."""
    y, p, groups, strata = cohort(n_patients=800, hours=20, seed=9)
    b = cf.calibrate(y, p, groups, strata, alpha=0.05, min_hours=0, min_patients=0)
    vec = cf.predict_states(b, p, strata)
    one = np.array([cf.predict_set(b, float(pi), si)["state"] for pi, si in zip(p, strata)],
                   dtype=object)
    check("predict_set matches vectorised predict_states on every row",
          bool((vec == one).all()), f"{len(p):,} rows")


def check_stored_edges_are_used():
    """7. Bin edges are fixed at calibration, not recomputed at serve.

    Re-fitting the spec on serve-time data reassigns rows to different cells, so the
    stored quantiles would be read from cells that no longer mean what they meant.
    The check plants that mistake and asserts the assignments genuinely differ —
    which is what makes storing the edges load-bearing rather than tidy.
    """
    rng = np.random.default_rng(11)
    n_meas_cal = rng.integers(0, 8, size=20000).astype(float)
    stale_cal = rng.exponential(2.0, size=20000)
    spec = cf.StratumSpec.fit(n_meas_cal, stale_cal, ["HR"], n_bins=3)

    n_meas_serve = rng.integers(4, 8, size=20000).astype(float)   # a better-instrumented ward
    stale_serve = rng.exponential(0.4, size=20000)
    stored = spec.assign(n_meas_serve, stale_serve)
    refit = cf.StratumSpec.fit(n_meas_serve, stale_serve, ["HR"], n_bins=3
                               ).assign(n_meas_serve, stale_serve)
    check("re-fitting edges at serve time would reassign rows", (stored != refit).mean() > 0.1,
          f"{(stored != refit).mean():.1%} of rows change cell")
    check("assign_one agrees with vectorised assign",
          all(spec.assign_one(a, c) == s for a, c, s in
              zip(n_meas_serve[:500], stale_serve[:500], stored[:500])))


def check_unseen_stratum():
    """A live stratum the cohort never produced is served, and flagged."""
    y, p, groups, strata = cohort(n_patients=800, hours=20, seed=12)
    b = cf.calibrate(y, p, groups, strata, alpha=0.05, min_hours=0, min_patients=0)
    out = cf.predict_set(b, 0.05, "m9/s9")
    check("unseen stratum uses the marginal band and says so",
          out["unseen_stratum"] and out["cell"] == "marginal",
          f"state={out['state']}")


def check_input_guards():
    """Bad calibration input fails loudly rather than producing a plausible band."""
    y, p, groups, strata = cohort(n_patients=200, hours=10, seed=13)
    bad = p.copy(); bad[3] = np.nan
    ok_nan = ok_len = ok_alpha = False
    try:
        cf.calibrate(y, bad, groups, strata)
    except ValueError:
        ok_nan = True
    try:
        cf.calibrate(y[:-1], p, groups, strata)
    except ValueError:
        ok_len = True
    try:
        cf.calibrate(y, p, groups, strata, alpha=1.5)
    except ValueError:
        ok_alpha = True
    check("NaN scores rejected", ok_nan)
    check("length mismatch rejected", ok_len)
    check("alpha outside (0,1) rejected", ok_alpha)
    miss = cf.StratumSpec.columns
    try:
        miss(["HR", "HR__age"], ["HR", "MAP"])
        ok_col = False
    except KeyError:
        ok_col = True
    check("missing stratum column rejected (never silently one cell)", ok_col)


def check_roundtrip_and_absence():
    """The artifact survives a round-trip; its absence is not a failure."""
    y, p, groups, strata = cohort(n_patients=900, hours=20, seed=14)
    spec = cf.StratumSpec.fit(np.arange(len(p)) % 7.0, np.zeros(len(p)), ["HR"])
    b = cf.calibrate(y, p, groups, strata, alpha=0.05, spec=spec,
                     min_hours=0, min_patients=0, detector="test")
    b.coverage = cf.coverage_report(b, y, p, groups, strata)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "conformal.json")
        cf.save(b, path)
        back = cf.load(path)
        same = (back is not None and back.alpha == b.alpha
                and back.detector == "test"
                and set(back.cells) == set(b.cells)
                and all(abs(back.cells[k].lo - b.cells[k].lo) < 1e-12 for k in b.cells)
                and back.spec.measured_edges == spec.measured_edges)
        check("save/load round-trips bands, spec and detector identity", same)
        check("load() of a missing artifact returns None, not an error",
              cf.load(os.path.join(d, "nope.json")) is None)


# --------------------------------------------------------------------------- #
def main() -> None:
    print("Conformal abstention layer — adversarial checks\n")
    for fn in (check_nominal_coverage, check_plus_one_correction, check_row_split_leaks,
               check_macro_exposes_correlation, check_small_cell_fallback,
               check_shift_breaks_coverage, check_class_conditional_is_required,
               check_serve_matches_derivation, check_stored_edges_are_used,
               check_unseen_stratum, check_input_guards, check_roundtrip_and_absence):
        fn()
    n = len(results)
    ok = sum(results)
    print(f"\n{ok}/{n} checks passed")
    raise SystemExit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
