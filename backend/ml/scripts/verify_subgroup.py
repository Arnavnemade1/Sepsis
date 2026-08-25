"""Adversarial checks on the subgroup audit.

A fairness audit is unusually easy to write in a way that cannot embarrass the
model: re-tune the threshold inside each group and every group looks equally well
served; bootstrap rows instead of patients and every interval is narrow enough to
declare parity; skip the multiplicity correction and one of two dozen tests trips
by chance and gets written up as a finding. Each of those is a *silent* failure —
the output looks like a fairness audit either way.

So these checks plant the failures. They build synthetic cohorts where the right
answer is known by construction — one group the model genuinely scores worse, one
where the groups are exchangeable and no disparity exists, one where the same
patients are duplicated so a row bootstrap would betray itself — and assert the
audit says the true thing in each. The exchangeable case is the important one: an
audit that flags a disparity everywhere passes every "did it find the disparity"
test and is still worthless.

    python -m backend.ml.scripts.verify_subgroup
"""
from __future__ import annotations

import numpy as np

from backend.ml.data.features import FeatureMatrix
from backend.ml.scripts.subgroup_audit import (CI_METRICS, MIN_PATIENTS, PatientView,
                                               adjust, bootstrap_cell, build_cells,
                                               cell_metrics, gap)
from backend.ml.eval.utility import utility_terms

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# Synthetic cohorts with a known right answer
# --------------------------------------------------------------------------- #
FEATURES = ["Age", "Gender", "hr"]


def cohort(n=1200, seed=0, hours=(30, 70), septic_frac=0.12,
           old_male_noise=0.0, female_shift=0.0, dup=1):
    """A cohort plus scores, with disparities injected on purpose.

    ``old_male_noise`` replaces that cell's scores with pure noise (a genuinely
    worse model for those patients). ``female_shift`` multiplies women's scores
    (mis-calibration, which at a shared threshold becomes under- or over-alarming).
    ``dup`` repeats every patient's rows, which changes nothing a *patient*
    bootstrap should notice.
    """
    rng = np.random.default_rng(seed)
    X, y, g, h, o, p = [], [], [], [], [], []
    for i in range(n):
        T = int(rng.integers(*hours))
        age = float(rng.uniform(20, 89))
        sex = float(rng.integers(0, 2))
        septic = rng.random() < septic_frac
        onset = int(T * 0.7) if septic else -1
        base = rng.uniform(0.01, 0.03)
        s = base + rng.normal(0, 0.01, T)
        if septic:
            s = s + np.clip((np.arange(T) - onset + 24) / 24, 0, 1) * 0.14
        s = np.clip(s, 1e-4, 1.0)
        if old_male_noise and age >= 80 and sex == 1.0:
            s = np.clip(rng.uniform(0.005, 0.16, T), 1e-4, 1.0)
        if female_shift and sex == 0.0:
            s = np.clip(s * female_shift, 1e-4, 1.0)
        lab = np.zeros(T, dtype=np.int8)
        if septic:
            lab[onset:] = 1
        for _ in range(dup):
            X.append(np.stack([np.full(T, age), np.full(T, sex), s * 100], 1))
            y.append(lab); g.append(np.full(T, i)); h.append(np.arange(T, dtype=float))
            o.append(np.full(T, onset, dtype=float)); p.append(s)
    ds = FeatureMatrix(X=np.concatenate(X).astype(np.float32),
                       y=np.concatenate(y), groups=np.concatenate(g),
                       feature_names=list(FEATURES),
                       hour_index=np.concatenate(h), onset_hour=np.concatenate(o))
    return ds, np.concatenate(p)


def audit(ds, scores, thr=0.055, boot=200, seed=1):
    """Run the audit's machinery end to end and return (points, draws, cells)."""
    pv = PatientView(ds)
    pv.alarm_facts(scores, thr)
    terms = utility_terms(ds.y, ds.groups, ds.hour_index, ds.onset_hour)
    cells = build_cells(pv)
    points, draws = {}, {}
    for i, (name, _kind, mask) in enumerate(cells):
        idx = np.flatnonzero(mask)
        points[name] = cell_metrics(pv, scores, terms, idx, thr)
        draws[name] = bootstrap_cell(pv, scores, terms, idx, thr, boot, seed + i)
    return pv, points, draws, cells


print("=" * 78)
print("SUBGROUP-AUDIT VERIFICATION")
print("=" * 78)

# ------------------------------------------------------------------ structure
print("\n[1] the patient view refuses data it cannot group")
ds, sc = cohort(n=200, seed=3)
pv = PatientView(ds)
check("age and sex read per patient, not per row",
      pv.n == 200 and pv.age.shape == (200,) and pv.sex.shape == (200,))
bad = FeatureMatrix(X=ds.X.copy(), y=ds.y, groups=ds.groups,
                    feature_names=list(FEATURES), hour_index=ds.hour_index,
                    onset_hour=ds.onset_hour)
bad.X[np.flatnonzero(bad.groups == bad.groups[0])[0], 0] += 20.0   # one hour ages 20 years
raised = False
try:
    PatientView(bad)
except ValueError:
    raised = True
check("a patient whose Age changes mid-stay is rejected", raised,
      "otherwise one stay's hours land in two different cells")

# ------------------------------------------------------------------ threshold
print("\n[2] the alarm threshold is global, never re-tuned per subgroup")
# The shift has to leave the deflated group straddling the threshold. Too gentle a
# shift and both groups detect at 1.000, which passes nothing: the check would be
# vacuous rather than satisfied, so non-degeneracy is asserted alongside the gap.
ds, sc = cohort(n=1400, seed=5, female_shift=0.35)
pv, points, draws, cells = audit(ds, sc, boot=60)
fem, mal = points["female"], points["male"]
check("deflating one group's scores lowers ITS detection at the shared threshold",
      0.0 < fem["detection_rate"] < mal["detection_rate"],
      f"female {fem['detection_rate']:.3f} vs male {mal['detection_rate']:.3f}")
check("and shows up as under-confidence, the mechanism behind it",
      fem["calibration_ratio"] < mal["calibration_ratio"],
      f"cal ratio {fem['calibration_ratio']:.2f} vs {mal['calibration_ratio']:.2f}")
# The refutation of a re-tuned audit: per-group thresholds would erase the gap.
per_group_equal = []
for label, code in (("female", 0.0), ("male", 1.0)):
    idx = np.flatnonzero(pv.sex == code)
    rows = np.concatenate([pv.rows[i] for i in idx])
    per_group_equal.append(float(np.quantile(sc[rows], 0.955)))
check("a per-group threshold would have differed (so the choice is not cosmetic)",
      abs(per_group_equal[0] - per_group_equal[1]) > 1e-3,
      f"female thr {per_group_equal[0]:.4f} vs male thr {per_group_equal[1]:.4f}")

# ------------------------------------------------------------------ bootstrap
print("\n[3] the bootstrap resamples patients, not rows")
ds1, sc1 = cohort(n=600, seed=7)
ds4, sc4 = cohort(n=600, seed=7, dup=4)
pv1, pts1, dr1, _ = audit(ds1, sc1, boot=250, seed=11)
pv4, pts4, dr4, _ = audit(ds4, sc4, boot=250, seed=11)
w1 = np.nanpercentile(dr1["cohort"]["auroc"], 97.5) - np.nanpercentile(dr1["cohort"]["auroc"], 2.5)
w4 = np.nanpercentile(dr4["cohort"]["auroc"], 97.5) - np.nanpercentile(dr4["cohort"]["auroc"], 2.5)
check("duplicating every patient's rows 4x barely moves the interval",
      abs(w4 - w1) / max(w1, 1e-9) < 0.35, f"width {w1:.4f} -> {w4:.4f}")
# What a row bootstrap would have claimed on the same duplicated data.
rng = np.random.default_rng(0)
from backend.ml.eval import metrics as _m
row_vals = [_m.auroc(ds4.y[r], sc4[r]) for r in
            (rng.integers(0, len(ds4.y), size=len(ds4.y)) for _ in range(120))]
w_row = np.percentile(row_vals, 97.5) - np.percentile(row_vals, 2.5)
check("a row bootstrap on the same data would claim a far narrower interval",
      w_row < 0.5 * w4, f"row width {w_row:.4f} vs patient width {w4:.4f}")

# ------------------------------------------------------------------ detection
print("\n[4] a real disparity is found")
ds, sc = cohort(n=2600, seed=13, old_male_noise=1.0)
pv, points, draws, cells = audit(ds, sc, boot=300, seed=17)
worst = min((points[n]["auroc"], n) for n, k, _ in cells
            if k == "age×sex" and not np.isnan(points[n]["auroc"]))[1]
check("the planted cell is the worst-scoring one", "age 80+" in worst and "male" in worst,
      f"worst = {worst}")
wmask = dict((n, m) for n, _k, m in cells)[worst]
rest = np.flatnonzero(~wmask)
terms = utility_terms(ds.y, ds.groups, ds.hour_index, ds.onset_hour)
points["rest"] = cell_metrics(pv, sc, terms, rest, 0.055)
draws["rest"] = bootstrap_cell(pv, sc, terms, rest, 0.055, 300, 99)
gs = gap(draws[worst], draws["rest"], points[worst], points["rest"], worst, "rest")
adjust(gs)
au = [g for g in gs if g["metric"] == "auroc"][0]
check("its AUROC gap is negative and the interval excludes zero",
      au["delta"] < 0 and au["ci_hi"] < 0,
      f"delta {au['delta']:+.3f} [{au['ci_lo']:+.3f},{au['ci_hi']:+.3f}]")

# ------------------------------------------------------------- false positives
print("\n[5] no disparity is invented where none exists")
ds, sc = cohort(n=2600, seed=23)          # sexes exchangeable by construction
pv, points, draws, cells = audit(ds, sc, boot=300, seed=29)
gs = gap(draws["female"], draws["male"], points["female"], points["male"], "female", "male")
alpha = adjust(gs)
nominal = sum(1 for g in gs if g["significant"])
corrected = sum(1 for g in gs if g["significant_adj"])
check("no corrected-significant gap between exchangeable groups", corrected == 0,
      f"{nominal} nominal, {corrected} corrected of {len(gs)} tests")
check("the correction is what stops it, and it binds", alpha < 0.05,
      f"alpha {alpha:.4f} over {len(gs)} tests")
widths = [(g["ci_hi_adj"] - g["ci_lo_adj"]) >= (g["ci_hi"] - g["ci_lo"]) - 1e-12 for g in gs]
check("every adjusted interval is at least as wide as the plain one", all(widths))

# ------------------------------------------------------------------ discipline
print("\n[6] the audit refuses to over-claim")
ds, sc = cohort(n=2600, seed=31, old_male_noise=1.0)
pv, points, draws, cells = audit(ds, sc, boot=200, seed=37)
sel = gap(draws["female"], draws["male"], points["female"], points["male"],
          "planted", "rest", selected=True)
adjust(sel)
check("a post-hoc selected comparison is never marked significant",
      not any(g["significant_adj"] for g in sel),
      "its cell was chosen by looking at the answers")
tiny = np.zeros(pv.n, dtype=bool)
tiny[:MIN_PATIENTS - 1] = True
tm = cell_metrics(pv, sc, utility_terms(ds.y, ds.groups, ds.hour_index, ds.onset_hour),
                  np.flatnonzero(tiny), 0.055)
check("a cell below the size floor is flagged underpowered",
      tm["underpowered"] == 1.0, f"{int(tm['n_patients'])} patients < {MIN_PATIENTS}")
check("every CI metric is a rate, a score or an hour count — no raw counts get intervals",
      "n_patients" not in CI_METRICS and "n_hours" not in CI_METRICS)

print("\n" + "=" * 78)
ok = sum(results)
print(f"{ok}/{len(results)} checks passed")
print("=" * 78)
raise SystemExit(0 if ok == len(results) else 1)
