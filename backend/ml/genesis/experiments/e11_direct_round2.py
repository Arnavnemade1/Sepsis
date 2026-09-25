"""E11: second regularisation round for the direct head only (it carried the E10 gain)."""
import os
from backend.ml import config
import sys, time, numpy as np
from backend.ml.genesis import harness as H, stack as S, targets as T
c = H.Cohort.load(); dev = c.dev_rows
ld = lambda n: np.load(os.path.join(config.ARTIFACTS_DIR, "genesis", "heads", f"{n}_oof.npy")).astype(float)
cfgs = {
    "slower": {"learning_rate": 0.01, "num_leaves": 31, "min_data_in_leaf": 1500,
               "feature_fraction": 0.2, "bagging_fraction": 0.6, "lambda_l2": 20.0},
    "wide_reg": {"learning_rate": 0.02, "num_leaves": 63, "min_data_in_leaf": 2000,
                 "feature_fraction": 0.25, "bagging_fraction": 0.7, "lambda_l2": 20.0},
}
ls = S.log_sigmoid
ds = ld("direct_slow"); w, h, gn = ld("who"), ld("when"), ld("gn2_direct")
for name in sys.argv[1:] or list(cfgs):
    t0 = time.time()
    v = S.head_oof(c, S.Head(f"direct_{name}", T.challenge, params=cfgs[name]))
    print(f"direct_{name}: {time.time()-t0:.0f}s", flush=True)
    for tag, a, b in [(f"direct_{name} vs direct_slow", v, ds),
                      (f"who*when*direct_{name}+net vs who*when*direct_slow+net",
                       ls(w)+ls(h)+ls(v)+ls(gn), ls(w)+ls(h)+ls(ds)+ls(gn))]:
        s = H.score(c, a[dev], dev, with_utility=False); cmp = H.compare(c, a, b, n_boot=800, rows=dev)
        print(f"{tag:<60} AUROC {s['auroc']:.4f} hm {s['auroc_hourmatched']:.4f} | d {cmp['d_auroc']:+.4f} [{cmp['auroc_ci'][0]:+.4f},{cmp['auroc_ci'][1]:+.4f}]", flush=True)
