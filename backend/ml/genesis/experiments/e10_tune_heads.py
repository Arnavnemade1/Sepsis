"""E10: slower, more regularised boosting for the three GENESIS heads.

The direct head early-stops at ~160 trees (lr 0.05), which says it overfits patients
fast. Same heads, same folds, only the booster settings change; judged alone AND inside
the product (the product is what ships)."""
import os
from backend.ml import config
import sys, time, numpy as np
from backend.ml.genesis import harness as H, stack as S, targets as T

c = H.Cohort.load(); dev = c.dev_rows
SLOW = {"learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 800,
        "feature_fraction": 0.3, "bagging_fraction": 0.7, "lambda_l2": 10.0}
which = sys.argv[1] if len(sys.argv) > 1 else "slow"
params = {"slow": SLOW}[which]
ls = S.log_sigmoid
base = {k: S.head_oof(c, S.Head(k, t, neg_rate=nr)) for k, t, nr in
        [("direct", T.challenge, 1.0), ("who", T.patient, 0.5), ("when", T.timing, 1.0)]}
new = {}
for k, t, nr in [("direct", T.challenge, 1.0), ("who", T.patient, 0.5), ("when", T.timing, 1.0)]:
    t0 = time.time()
    new[k] = S.head_oof(c, S.Head(f"{k}_{which}", t, params=params, neg_rate=nr))
    print(f"{k}_{which}: {time.time()-t0:.0f}s", flush=True)

def rep(name, v, ref):
    s = H.score(c, v[dev], dev, with_utility=False)
    cmp = H.compare(c, v, ref, n_boot=1000, rows=dev)
    print(f"{name:<36} AUROC {s['auroc']:.4f} hm {s['auroc_hourmatched']:.4f} B {s['between']:.4f} "
          f"W {s['within_septic']:.4f} | d {cmp['d_auroc']:+.4f} [{cmp['auroc_ci'][0]:+.4f},{cmp['auroc_ci'][1]:+.4f}]", flush=True)

for k in new:
    rep(f"{k}_{which} vs {k}", new[k], base[k])
P0 = ls(base["who"]) + ls(base["when"]) + ls(base["direct"])
P1 = ls(new["who"]) + ls(new["when"]) + ls(new["direct"])
rep(f"GENESIS[{which}] vs GENESIS", P1, P0)
P2 = P0 + P1
rep(f"GENESIS + GENESIS[{which}] (bag of two)", P2, P0)
