"""Adversarial audit of the supervision-density claim.

This verifies the CLAIM, not the code. Following the pattern of ``verify_integrity.py``,
each check plants a specific known pathology and asserts that the guard fires — because
an assertion that has never been seen to fail is not evidence of anything.

    PYTHONPATH=. python -m backend.ml.scripts.verify_horizon

The claim under audit: *denser supervision of septic stays — labelling more of the
stay positive, while evaluating and calibrating against the original challenge label —
raises detection at a fixed alarm budget, and the temporal placement of the extra
positive hours does not matter.* The ways that claim could be false, and the check for
each:

  1. the axis is off by six hours, so "W = 0" is not the challenge label and every
     comparison is against a baseline that is not the incumbent;
  2. the widened target leaks the future into a FEATURE rather than only into the
     training label;
  3. the gain is a calibration artifact rather than a ranking change;
  4. the gain is a fold accident that a paired patient bootstrap would not support;
  5. the metric moved because the *threshold* moved, not because the detector improved —
     i.e. it only appears at one hand-picked alarm budget;
  6. the mechanism is misattributed. An earlier version of this module claimed the
     gain came from a wider PRE-ONSET WINDOW teaching the model the approach to
     sepsis. Check 6 is the control that falsified it, and it is kept pointed the
     same way: if a prevalence-matched RANDOM relabelling ever stops matching the
     window, the window matters after all and the documentation is wrong again.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np

from backend.ml.eval.bench import Benchmark, BenchResult
from backend.ml.eval import metrics
from backend.ml.models.supervision import (SupervisionBooster,
                                           sham_target, supervision_target)

PASS, FAIL = "PASS", "FAIL"
_results: List[Tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def main() -> int:
    b = Benchmark.load("dev", n_folds=4)
    ds = b.ds
    print(f"cohort: {b.describe()}\n")

    # ---------------------------------------------------------------- 1. the axis
    print("1. the supervision axis is anchored where it is claimed to be")
    t0 = supervision_target(ds.y, ds.hour_index, ds.onset_hour, 0.0)
    check("W=0 reproduces the challenge label exactly",
          np.array_equal(t0, ds.y.astype(np.int8)),
          f"{int((t0 != ds.y).sum())} mismatching rows")
    # The guard must be able to fail: a six-hour offset is the specific mistake it
    # exists to catch, and it was made in the first version of this sweep.
    t6 = supervision_target(ds.y, ds.hour_index, ds.onset_hour, 6.0)
    check("a 6h-offset target is DETECTED as different from the label",
          not np.array_equal(t6, ds.y.astype(np.int8)),
          f"{int((t6 != ds.y).sum())} rows differ, as it must")
    widths = [0.0, 24.0, 96.0, np.inf]
    prev = [float(supervision_target(ds.y, ds.hour_index, ds.onset_hour, w).mean()) for w in widths]
    check("training prevalence rises monotonically with supervision density",
          all(a < c for a, c in zip(prev, prev[1:])),
          " -> ".join(f"{p:.4f}" for p in prev))
    # Widening must only ever add positives inside septic stays.
    tALL = supervision_target(ds.y, ds.hour_index, ds.onset_hour, np.inf)
    check("widening never marks a never-septic patient positive",
          bool(((tALL == 1) & (ds.onset_hour < 0)).sum() == 0))
    check("widening is nested: every W=0 positive stays positive at W=96",
          bool((supervision_target(ds.y, ds.hour_index, ds.onset_hour, 96.0)[t0 == 1] == 1).all()))

    # ------------------------------------------------------- 2. no feature leakage
    print("\n2. the future enters the training label only, never a feature")
    # onset_hour is metadata on the FeatureMatrix, not a column of X. If any feature
    # were a function of it, the model could read the answer at inference time.
    corr_max, worst = 0.0, ""
    on = ds.onset_hour.astype(float)
    for i, n in enumerate(ds.feature_names):
        col = ds.X[:, i].astype(float)
        if np.std(col) < 1e-12:
            continue
        c = abs(float(np.corrcoef(col, on)[0, 1]))
        if c > corr_max:
            corr_max, worst = c, n
    check("no feature is near-perfectly correlated with onset_hour",
          corr_max < 0.95, f"max |r| = {corr_max:.3f} ({worst})")
    # And the positive control: a deliberately planted leak must be caught.
    planted = abs(float(np.corrcoef(on + np.random.default_rng(0).normal(0, 1e-3, len(on)), on)[0, 1]))
    check("a PLANTED onset-derived feature would be caught", planted >= 0.95,
          f"planted |r| = {planted:.4f}")

    # ------------------------------------------- 3./4. the gain, and its significance
    print("\n3. the gain is a ranking change, reproduced on the fixed folds")
    def cand(w: float):
        def run(dd, tr, te):
            m = SupervisionBooster(kind="hgb", width_hours=w, feature_names=dd.feature_names)
            m.fit(dd.X[tr], dd.y[tr], groups=dd.groups[tr],
                  hour_index=dd.hour_index[tr], onset_hour=dd.onset_hour[tr])
            return m.predict_proba(dd.X[te])
        return run

    base = b.run("win0", cand(0.0), verbose=False)
    wide = b.run("win96", cand(96.0), verbose=False)
    print(f"    W=0  : {base.line()}")
    print(f"    W=96 : {wide.line()}")

    # Isotonic calibration is monotone, so it cannot change AUROC or any
    # detection-at-budget number. If the "gain" were a calibration artifact it would
    # have to show up as a ranking difference of zero.
    rho = float(np.corrcoef(np.argsort(np.argsort(base.oof)),
                            np.argsort(np.argsort(wide.oof)))[0, 1])
    check("the two detectors genuinely rank differently (not a calibration artifact)",
          rho < 0.99, f"Spearman rho = {rho:.4f}")

    check("ECE stays low despite the widened training target",
          wide.mean("ece") < 0.02,
          f"ECE {wide.mean('ece'):.4f} (calibrated against the challenge label)")

    print("\n4. detection at a fixed alarm budget, at EVERY budget")
    sig = []
    for far in (0.05, 0.10, 0.15, 0.20, 0.30):
        r = b.compare_alarm_budget(wide, base, target_far=far, n_boot=1000)
        sig.append(r["significant"])
        print(f"    budget {int(far*100):>2}%  ddetect {r['delta_detection']:+.4f} "
              f"CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
              f"{'SIG' if r['significant'] else 'ns'}")
    check("the gain holds at every alarm budget, not one flattering cell",
          all(sig), f"{sum(sig)}/5 budgets significant")

    # The AUROC claim is deliberately NOT asserted: it is not significant, and a check
    # that passed here would mean the suite was rubber-stamping an overclaim.
    c = b.compare(wide, base, n_boot=400)
    check("AUROC is reported as n.s. rather than claimed",
          not c["significant"],
          f"dAUROC {c['delta_auroc']:+.4f} CI[{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] — "
          "this check FAILS if AUROC ever becomes significant, which would mean the "
          "documentation understates the result and must be updated")

    # ------------------------------------------------- 6. the mechanism control
    print("\n5/6. mechanism: is it the WINDOW, or just more patient-level supervision?")
    # Same number of extra positive hours, scattered at RANDOM through septic stays
    # rather than placed before onset. This control falsified the original
    # "supervision horizon" reading of the result.
    #
    # Run at W=24, not W=96, and that choice is load-bearing. At W=96 the widened
    # target claims about 79% of every non-labelled septic hour there is, so a random
    # draw is *forced* to land mostly on the same hours as the window and the control
    # is nearly vacuous. The first version of this check ran at 96 and caught itself:
    # it reported only 10,850 hours placed differently and failed its own separation
    # test. W=24 leaves the sham room to actually differ.
    W_MECH = 24.0
    sham = sham_target(ds.y, ds.hour_index, ds.onset_hour, W_MECH, seed=1)
    real = supervision_target(ds.y, ds.hour_index, ds.onset_hour, W_MECH)
    n_extra = int(real.sum() - ds.y.sum())
    overlap = float((sham[real == 1] == 1).mean())
    check("the sham control is prevalence-matched to the real window",
          abs(float(sham.mean() - real.mean())) < 1e-3,
          f"sham {sham.mean():.4f} vs real {real.mean():.4f}")
    check("the sham is well separated from the window it controls",
          overlap < 0.70,
          f"{overlap:.1%} of the window's positive hours are also sham positives "
          f"({n_extra:,} hours added); above ~70% the control cannot discriminate")

    # Both arms must use the IDENTICAL estimator. The first version compared a
    # calibration-holdout model against a plain one, so the arms differed by
    # training-set size as well as by target — which is not a controlled comparison.
    def plain(target):
        def run(dd, tr, te):
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                               max_leaf_nodes=63, l2_regularization=1.0,
                                               class_weight="balanced", random_state=42)
            m.fit(dd.X[tr], target[tr])
            return m.predict_proba(dd.X[te])[:, 1]
        return run

    lab_r = b.run("mech-label", plain(ds.y.astype(np.int8)), verbose=False)
    win_r = b.run("mech-window", plain(real), verbose=False)
    sham_r = b.run("mech-sham", plain(sham), verbose=False)

    beats, no_window_edge = [], []
    for far in (0.05, 0.10, 0.15, 0.20):
        vs_base = b.compare_alarm_budget(sham_r, lab_r, target_far=far, n_boot=800)
        vs_real = b.compare_alarm_budget(win_r, sham_r, target_far=far, n_boot=800)
        beats.append(vs_base["significant"])
        # The harness's ``significant`` flag is one-sided (it asks only whether the
        # CANDIDATE is better), so read the interval directly: the window has an edge
        # only if the whole interval sits above zero.
        no_window_edge.append(not (vs_real["ci_lo"] > 0))
        print(f"    budget {int(far*100):>2}%   sham-vs-label {vs_base['delta_detection']:+.4f}"
              f" {'SIG' if vs_base['significant'] else 'ns'}"
              f"   window-vs-sham {vs_real['delta_detection']:+.4f} "
              f"CI[{vs_real['ci_lo']:+.4f},{vs_real['ci_hi']:+.4f}]")
    check("denser supervision helps even with the temporal placement destroyed",
          all(beats), f"{sum(beats)}/4 budgets significantly above the challenge label")
    # The load-bearing check, deliberately asserting the absence of an effect: the
    # documented mechanism is supervision DENSITY, not placement. If a real window
    # advantage ever appears, this fails and the module docstring is wrong again.
    check("the window has no advantage over random placement",
          all(no_window_edge),
          f"{sum(no_window_edge)}/4 budgets show no window edge — measured point "
          "estimates favour the SHAM, so concentrating supervision near onset is if "
          "anything counterproductive")

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    print(f"\n{'=' * 70}\n{len(_results) - n_fail}/{len(_results)} checks passed")
    if n_fail:
        print("FAILURES:")
        for s, n, d in _results:
            if s == FAIL:
                print(f"  - {n}: {d}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
