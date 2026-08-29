"""Adversarial audit of the coordination plane — including the audit of its own control.

    PYTHONPATH=. python -m backend.ml.scripts.verify_coordination

``backend/ml/models/coordination.py`` asks whether sepsis is visible in the *structure*
between vital-sign channels — Godin & Buchman's organ-uncoupling hypothesis, operating
on hourly EHR data instead of the beat-to-beat waveforms it is normally studied on. The
answer measured here is **no**, and a negative result is only worth having if the
machinery that produced it is demonstrably able to say yes. So this suite checks the
detector, then checks the *control*, then checks that the control can fail.

Every check plants a specific pathology and asserts the guard fires. An assertion that
has never been seen to fail is not evidence of anything.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.ml.eval.bench import Benchmark
from backend.ml.eval import metrics
from backend.ml.models import coordination as C
from backend.ml.models.coordination import (COORD_CHANNELS, NormativeStructure,
                                            build_state, coordination_features,
                                            organ_dysfunction, organ_synchrony_features,
                                            surrogate_uncouple)

PASS, FAIL = "PASS", "FAIL"
_results: List[Tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def _fit_auroc(X, y, folds, ds_y):
    p = np.full(len(ds_y), np.nan)
    for tr, te in folds:
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                           class_weight="balanced", random_state=0)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(p)
    return metrics.auroc(ds_y[ok], p[ok])


def main() -> int:
    b = Benchmark.load("dev", n_folds=4)
    ds = b.ds
    folds = b.folds[:2]
    print(f"cohort: {b.describe()}\n")

    # -------------------------------------------------------------- 1. causality
    print("1. the coordination state is causal")
    st = build_state(ds.X, ds.feature_names, ds.groups)
    codes, _ = pd.factorize(ds.groups)
    order = np.argsort(codes, kind="stable")
    g_s = codes[order]
    starts = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1]])
    bounds = np.r_[starts, len(g_s)]
    late = np.zeros(len(ds.X), bool)
    for a, bb in zip(bounds[:-1], bounds[1:]):
        late[order[a + (bb - a) // 2:bb]] = True
    rng = np.random.default_rng(0)
    Xc = ds.X.copy()
    cols = [ds.feature_names.index(c) for c in COORD_CHANNELS]
    Xc[np.ix_(late, cols)] = rng.normal(80, 30, size=(int(late.sum()), len(cols)))
    st2 = build_state(Xc, ds.feature_names, ds.groups)
    early = ~late
    drift = max(float(np.abs(getattr(st, k)[early] - getattr(st2, k)[early]).max())
                for k in ("r", "d", "S_fast", "S_slow", "D_fast"))
    check("rewriting the future moves nothing earlier", drift == 0.0,
          f"max drift {drift:.3e} over {int(early.sum()):,} untouched hours")
    check("the corruption was real (the control can fail)",
          not np.array_equal(ds.X, Xc),
          f"{int(late.sum()):,} hours overwritten with noise")

    # --------------------------------------------------------- 2. scale invariance
    print("\n2. the hard-coded channel scales cancel out, as documented")
    saved = dict(C.CHANNEL_SCALE)
    try:
        healthy = np.flatnonzero(ds.onset_hour < 0)
        rt = np.clip(C._normalise_plane(st.r, st.S_slow), -8, 8)
        nm = NormativeStructure.fit(st.r, healthy, COORD_CHANNELS, r_struct=rt)
        M1, _ = coordination_features(st, nm, plane="structure")
        for k in C.CHANNEL_SCALE:
            C.CHANNEL_SCALE[k] *= 3.7
        st_s = build_state(ds.X, ds.feature_names, ds.groups)
        rt_s = np.clip(C._normalise_plane(st_s.r, st_s.S_slow), -8, 8)
        nm_s = NormativeStructure.fit(st_s.r, healthy, COORD_CHANNELS, r_struct=rt_s)
        M2, _ = coordination_features(st_s, nm_s, plane="structure")
    finally:
        C.CHANNEL_SCALE.clear()
        C.CHANNEL_SCALE.update(saved)
    diff = float(np.abs(M1 - M2).max())
    check("rescaling every channel by 3.7x leaves the structure plane unchanged",
          diff < 1e-3, f"max |delta| = {diff:.2e}")

    # ------------------------------------------------- 3. the control that was WRONG
    print("\n3. the surrogate control is valid (a circular roll is not)")
    # A circular roll wraps a septic patient's late deteriorated hours to the FRONT of
    # the stay, so every hour inherits evidence of the outcome. The signature is
    # unmistakable and is asserted here so the mistake cannot be reintroduced: a
    # control that removes information must never RAISE a detector's score.
    def circular_roll(X):
        out = X.copy()
        r = np.random.default_rng(3)
        for ci, ch in enumerate(COORD_CHANNELS):
            col = ds.feature_names.index(ch)
            vals = out[order, col]
            for a, bb in zip(bounds[:-1], bounds[1:]):
                if bb - a < 3 or ci == 0:
                    continue
                vals[a:bb] = np.roll(vals[a:bb], int(r.integers(1, bb - a)))
            out[order, col] = vals
        return out

    a_real = _fit_auroc(ds.X, ds.y, folds, ds.y)
    a_roll = _fit_auroc(circular_roll(ds.X), ds.y, folds, ds.y)
    check("the circular-roll surrogate is detected as INVALID (it adds signal)",
          a_roll > a_real,
          f"incumbent {a_real:.4f} -> {a_roll:.4f} on the rolled cohort; a control that "
          "raises the score is leaking the outcome, not removing coupling")

    # The delay surrogate must instead leave a marginal-reading detector alone.
    a_common = _fit_auroc(surrogate_uncouple(ds.X, ds.feature_names, ds.groups,
                                             max_lag=6, seed=0, mode="common"),
                          ds.y, folds, ds.y)
    check("the delay surrogate is VALID (it does not manufacture signal)",
          a_common <= a_real + 0.005,
          f"incumbent {a_real:.4f} -> {a_common:.4f} under a common 6h delay")
    check("the identity arm is exactly the identity",
          np.array_equal(surrogate_uncouple(ds.X, ds.feature_names, ds.groups, max_lag=0),
                         ds.X))

    # ------------------------------------------------ 4. what alignment is worth
    print("\n4. matched-pair: independent delays vs a common delay of the same size")
    # Delaying a channel desynchronises it AND makes it older. Only the first is under
    # test, so the two arms are run at a matched lag and the staleness cancels.
    worth = []
    for lag in (6, 24):
        Xi = surrogate_uncouple(ds.X, ds.feature_names, ds.groups, max_lag=lag,
                                seed=0, mode="independent")
        Xc2 = surrogate_uncouple(ds.X, ds.feature_names, ds.groups, max_lag=lag,
                                 seed=0, mode="common")
        ai, ac = _fit_auroc(Xi, ds.y, folds, ds.y), _fit_auroc(Xc2, ds.y, folds, ds.y)
        worth.append(ac - ai)
        print(f"    lag {lag:>2}h  aligned {ac:.4f}  scrambled {ai:.4f}  "
              f"alignment worth {ac - ai:+.4f}")
    check("inter-channel alignment is worth ~nothing at hourly resolution",
          all(abs(w) < 0.02 for w in worth),
          "|delta| < 0.02 at every lag, and the sign is not stable — no dose-response")

    # ------------------------------------------------- 5. the organ-level hypothesis
    print("\n5. Buchman's hypothesis at the level it was stated (organ systems)")
    lab = np.load("artifacts/feature_cache/lab_limit8000.npz", allow_pickle=True)
    if not np.array_equal(lab["groups"], ds.groups):
        check("lab cache aligns with the dev cohort", False, "row mismatch; skipping")
    else:
        O, organs = organ_dysfunction(lab["X"], list(lab["feature_names"]), ds.groups)
        Ms, ns = organ_synchrony_features(O, organs, ds.groups)
        idx = ns.index("organS_synchrony")
        a_sync = metrics.auroc(ds.y, Ms[:, idx])
        print(f"    organ systems: {', '.join(organs)}")
        check("the organ-synchrony index is at chance",
              abs(a_sync - 0.5) < 0.05,
              f"AUROC {a_sync:.4f} — systems do not measurably decompensate together")
        # And the burden control: the MARGINAL reading of the same organ vector is not
        # at chance, which proves the organ scores are not simply broken.
        a_burden = metrics.auroc(ds.y, Ms[:, ns.index("organM_max")])
        check("the same organ scores DO carry marginal signal (they are not broken)",
              abs(a_burden - 0.5) > 0.05,
              f"organM_max AUROC {a_burden:.4f} — the null above is about synchrony, "
              "not about a dead feature")

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    print(f"\n{'=' * 70}\n{len(_results) - n_fail}/{len(_results)} checks passed")
    for s, n, d in _results:
        if s == FAIL:
            print(f"  FAILED - {n}: {d}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
