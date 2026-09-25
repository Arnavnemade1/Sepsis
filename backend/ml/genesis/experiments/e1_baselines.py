import os
from backend.ml import config
import sys, time, numpy as np
from backend.ml.genesis import harness as H, learners as L
from backend.ml.genesis.rawstore import VITALS

c = H.Cohort.load()
print(c.F.shape, "folds", np.bincount(c.folds[c.folds >= 0]), "locked", int((c.folds < 0).sum()), flush=True)

def wearable_cols(names):
    keep = []
    for j, n in enumerate(names):
        base = n.split("__")[0]
        if base in VITALS or n in ("Age", "Gender", "unit_sicu", "HospAdmTime", "ICULOS", "hour", "icu_start"):
            keep.append(j)
    return np.array(keep)

def lgbm(cols=None, params=None, tag=""):
    def cand(c, tr, te):
        t0 = time.time()
        fit = L.fit_lgb(c.F, tr, c.y[tr], c.seg[tr], params=params, cols=cols,
                        strat=c.plabel[c.seg[tr]])
        p = fit.predict(c.F, te)
        print(f"      [{tag}] trees {fit.best_iter}  fit+pred {time.time()-t0:.0f}s", flush=True)
        return p
    return cand

which = sys.argv[1]
folds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
if which == "all":
    H.run(c, "base_lgb_all", lgbm(tag="all"), folds=folds)
elif which == "wear":
    cols = wearable_cols(c.names)
    print("wearable cols", len(cols))
    H.run(c, "base_lgb_wearable", lgbm(cols=cols, tag="wear"), folds=folds)
