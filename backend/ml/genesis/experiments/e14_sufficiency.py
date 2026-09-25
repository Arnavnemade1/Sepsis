"""E14: is the tuned booster's score SUFFICIENT for its inputs?

If a score s(x) equals P(y | x), then y is independent of x given s — the features carry
nothing further. Test: give a second booster the 510 features AND the (out-of-fold) tuned
score, on a RANDOM sample (all positive hours + 10% of negative hours, never selected on
the score), and ask whether it beats the score alone. It does not — adding the features
back on top of the score makes the ranking slightly worse — so no rearrangement of these
inputs (stacking, cascades, new architectures) has headroom left; only new information can.

The first version of this test selected its sample ON the score (low-scored positives vs.
high-scored negatives) and "found" a residual AUROC of 0.886. That is Berkson's paradox:
inside a set chosen by the score, whatever the model reads as risky is automatically
associated with the other class, so a fresh model scores well by learning the main model
backwards. Selection must never depend on the score being tested.

    python -m backend.ml.genesis.experiments.e14_sufficiency
"""
import json
import os

import numpy as np

from backend.ml import config
from backend.ml.genesis import harness as H, learners as L


def main() -> None:
    c = H.Cohort.load()
    dev = c.dev_rows
    p = np.load(os.path.join(config.ARTIFACTS_DIR, "genesis", "heads", "direct_slow_oof.npy")).astype(float)
    y = c.y
    rng = np.random.default_rng(0)
    pos, neg = dev[y[dev] == 1], dev[y[dev] == 0]
    rows = np.sort(np.r_[pos, rng.choice(neg, len(neg) // 10, replace=False)])
    P = np.full((len(c.y), 1), np.nan, dtype=np.float32)
    P[rows, 0] = p[rows]
    data = L.HStack(c.F, P)
    folds = c.folds[c.seg[rows]]
    oof = np.full(len(rows), np.nan)
    for k in range(c.n_folds):
        tr, te = rows[folds != k], rows[folds == k]
        fit = L.fit_lgb(data, tr, y[tr].astype(float), c.seg[tr], strat=c.plabel[c.seg[tr]],
                        params={"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 400,
                                "feature_fraction": 0.3, "lambda_l2": 10.0})
        oof[folds == k] = fit.predict(data, te, raw=True)
    full = np.full(len(c.y), np.nan)
    full[rows] = oof
    cmp = H.compare(c, full, p, n_boot=500, rows=rows)
    out = {"n_rows": int(len(rows)), "n_pos": int(len(pos)),
           "auroc_score_alone": H.auroc(y[rows], p[rows]),
           "auroc_features_plus_score": H.auroc(y[rows], oof), "delta": cmp}
    with open(os.path.join(config.ARTIFACTS_DIR, "genesis", "sufficiency.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"score alone {out['auroc_score_alone']:.4f}  features+score {out['auroc_features_plus_score']:.4f}  "
          f"d {cmp['d_auroc']:+.4f} [{cmp['auroc_ci'][0]:+.4f}, {cmp['auroc_ci'][1]:+.4f}]")


if __name__ == "__main__":
    main()
