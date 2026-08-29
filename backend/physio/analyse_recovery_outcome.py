"""Does autonomic recovery slowing precede circulatory decompensation?

    PYTHONPATH=. python -m backend.physio.analyse_recovery_outcome

This is the experiment the whole open-access detour exists to run. The cohort
(``artifacts/mimic3_open/windows.npz``) is a **within-patient case-control** study built
from the open MIMIC-III Waveform Database: each ICU stay contributes the onsets of its
circulatory decompensation events and matched stable control windows from the same
admission, with features computed from the ECG at 15, 30 and 60 minutes *before* the
timepoint.

WHAT MAKES THIS DIFFERENT FROM EVERY EARLIER RESULT IN THIS PROJECT
-------------------------------------------------------------------
Every accuracy number this repository has produced turned out, on decomposition, to be
**between-patient** discrimination — telling sick patients from well ones — wearing an
early-warning label. Here positives and negatives come from the *same admission*, so
patient identity cannot explain anything: the only question available is *when*.

THE BAR τ HAS TO CLEAR
----------------------
Not chance. Heart rate and blood pressure are physiologically coupled, so a patient
approaching hypotension may already be tachycardic, and a detector that has only learned
"fast heart rate precedes low blood pressure" has discovered nothing. So the comparison
is a ladder, and each rung must be beaten by the next:

  1. ``hr``        — the mean heart rate alone. The trivial predictor.
  2. ``hrv``       — the HeRO-style magnitude features (SDNN, RMSSD, entropy...).
  3. ``recovery``  — the time-constant features this project invented.
  4. ``all``       — both feature families together.

The claim worth making is (4) or (3) beating (2), significantly, on a paired patient
bootstrap. Anything less and the honest answer is that recovery speed adds nothing over
the variability measures that already exist.

CONTROLS
--------
* **Label permutation within patient** — shuffling event/control labels inside each
  stay must collapse every model to chance. If it does not, the pipeline is leaking.
* **Per-lead reporting** — a result at 15 minutes that vanishes at 60 is a detector of
  imminent collapse, not an early warning, and is reported as such rather than averaged
  into a single flattering number.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

WINDOWS = os.path.join("artifacts", "mimic3_open", "windows.npz")


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    from backend.ml.eval import metrics
    return metrics.auroc(y, s)


def _feature_groups(names: List[str]) -> Dict[str, np.ndarray]:
    from backend.physio import recovery
    rec = set(recovery.RECOVERY_FEATURE_NAMES)
    idx_rec = np.array([i for i, n in enumerate(names) if n in rec])
    idx_hrv = np.array([i for i, n in enumerate(names) if n not in rec])
    idx_hr = np.array([i for i, n in enumerate(names) if n == "mean_hr"])
    return {"hr": idx_hr, "hrv": idx_hrv, "recovery": idx_rec,
            "all": np.arange(len(names))}


def _oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Patient-grouped out-of-fold scores. Small cohort, so a shallow model and few
    folds — a deep one here would fit the handful of patients, not the physiology."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    p = np.full(len(y), np.nan)
    n_sp = min(4, max(2, len(np.unique(groups[y == 1]))))
    sgkf = StratifiedGroupKFold(n_splits=n_sp, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.06, max_depth=3, min_samples_leaf=15,
            l2_regularization=1.0, class_weight="balanced", random_state=seed)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def _paired_bootstrap(y, sa, sb, groups, n_boot=2000, seed=0) -> Dict[str, float]:
    """Patient-resampled difference in AUROC between two score vectors."""
    pts = np.array(sorted(set(groups.tolist())))
    by = {g: np.flatnonzero(groups == g) for g in pts}
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(len(pts), len(pts), replace=True)
        rows = np.concatenate([by[pts[i]] for i in draw])
        yy = y[rows]
        if yy.min() == yy.max():
            d[b] = 0.0
            continue
        d[b] = _auroc(yy, sa[rows]) - _auroc(yy, sb[rows])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(_auroc(y, sa) - _auroc(y, sb)),
            "lo": float(lo), "hi": float(hi), "significant": bool(lo > 0)}


def main() -> int:
    if not os.path.exists(WINDOWS):
        print(f"{WINDOWS} not found — run build_mimic3_cohort then fetch_mimic3_windows")
        return 1
    d = np.load(WINDOWS, allow_pickle=True)
    X_all = np.asarray(d["X"], dtype=float)
    names = [str(n) for n in d["names"]]
    meta = d["meta"]
    pid = np.array([str(m[0]) for m in meta])
    lead = np.array([float(m[1]) for m in meta])
    y_all = np.array([int(m[2]) for m in meta])

    print(f"cohort: {len(y_all)} feature rows | {len(set(pid))} patients | "
          f"{int((y_all == 1).sum())} event rows / {int((y_all == 0).sum())} control rows")
    both = [p for p in set(pid) if len(set(y_all[pid == p])) == 2]
    print(f"patients contributing BOTH an event and a control: {len(both)} "
          f"(the within-patient comparison rests on these)")
    if len(both) < 5 or (y_all == 1).sum() < 15:
        print("\nCOHORT TOO SMALL to report an accuracy number. Refusing to compute one:\n"
              "  a handful of patients would give a confidence interval wide enough to\n"
              "  contain anything, and this project has a standing rule against quoting\n"
              "  numbers it cannot defend. Let the retrieval job finish and re-run.")
        return 2

    groups_map = _feature_groups(names)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    for L in sorted(set(lead.tolist())):
        m = lead == L
        X, y, g = X_all[m], y_all[m], pid[m]
        if len(set(g[y == 1])) < 3:
            print(f"\nlead {L:.0f} min: too few event patients ({len(set(g[y==1]))})")
            continue
        print(f"\n=== lead {L:.0f} minutes before decompensation ===")
        print(f"  {int((y==1).sum())} event rows, {int((y==0).sum())} control rows, "
              f"{len(set(g))} patients")
        scores: Dict[str, np.ndarray] = {}
        for tag, idx in groups_map.items():
            if idx.size == 0:
                continue
            s = _oof(X[:, idx], y, g)
            ok = ~np.isnan(s)
            if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
                continue
            scores[tag] = s
            print(f"    {tag:<9} AUROC {_auroc(y[ok], s[ok]):.4f}  ({idx.size} features)")

        if "hrv" in scores and "all" in scores:
            ok = ~(np.isnan(scores["all"]) | np.isnan(scores["hrv"]))
            r = _paired_bootstrap(y[ok], scores["all"][ok], scores["hrv"][ok], g[ok])
            print(f"    -> does recovery ADD over HRV magnitude? "
                  f"dAUROC {r['delta']:+.4f} CI[{r['lo']:+.4f},{r['hi']:+.4f}] "
                  f"{'SIGNIFICANT' if r['significant'] else 'n.s.'}")
        if "hr" in scores and "all" in scores:
            ok = ~(np.isnan(scores["all"]) | np.isnan(scores["hr"]))
            r = _paired_bootstrap(y[ok], scores["all"][ok], scores["hr"][ok], g[ok])
            print(f"    -> does it ADD over heart rate alone?      "
                  f"dAUROC {r['delta']:+.4f} CI[{r['lo']:+.4f},{r['hi']:+.4f}] "
                  f"{'SIGNIFICANT' if r['significant'] else 'n.s.'}")

        # Permutation control: shuffle labels WITHIN each patient. Anything that
        # survives this is leakage, not physiology.
        rng = np.random.default_rng(1)
        y_perm = y.copy()
        for p in set(g.tolist()):
            k = g == p
            y_perm[k] = rng.permutation(y[k])
        s = _oof(X, y_perm, g)
        ok = ~np.isnan(s)
        if ok.sum() > 10 and len(np.unique(y_perm[ok])) > 1:
            a = _auroc(y_perm[ok], s[ok])
            print(f"    [control] labels permuted within patient: AUROC {a:.4f} "
                  f"{'(collapses, as it must)' if abs(a - 0.5) < 0.12 else '<-- LEAK'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
