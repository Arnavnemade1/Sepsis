"""E9: does the Genesis Network earn a place in the product of experts?

Pre-registered rule (lock_ledger.jsonl): an expert joins only if adding it raises dev-CV
pooled AUROC with a paired patient-bootstrap 95% CI that excludes zero."""
import os
from backend.ml import config
import sys, numpy as np
from backend.ml.genesis import harness as H, stack as S

prefix = sys.argv[1] if len(sys.argv) > 1 else "gn"
c = H.Cohort.load(); dev = c.dev_rows
ld = lambda n: np.load(os.path.join(config.ARTIFACTS_DIR, "genesis", "heads", f"{n}_oof.npy")).astype(float)
d, w, h = ld("direct"), ld("who"), ld("when")
g = {k: ld(f"{prefix}_{k}") for k in ("direct", "who", "when", "s3")}
ls = S.log_sigmoid
base = ls(w) + ls(h) + ls(d)

def rep(name, v, ref, n_boot=1000):
    s = H.score(c, v[dev], dev, with_utility=False)
    cmp = H.compare(c, v, ref, n_boot=n_boot, rows=dev)
    ok = cmp["auroc_ci"][0] > 0
    print(f"{name:<44} AUROC {s['auroc']:.4f} hm {s['auroc_hourmatched']:.4f} B {s['between']:.4f} "
          f"W {s['within_septic']:.4f} | d {cmp['d_auroc']:+.4f} [{cmp['auroc_ci'][0]:+.4f},{cmp['auroc_ci'][1]:+.4f}]"
          f"{'  PASSES RULE' if ok and ref is base else ''}", flush=True)

print(f"--- network heads alone ({prefix}), vs the tree direct head")
for k, v in g.items():
    rep(f"network {k}", v, d, n_boot=300)
rep("network who*when*direct", ls(g["who"]) + ls(g["when"]) + ls(g["direct"]), d, n_boot=300)
print("--- adding network experts to GENESIS (vs GENESIS)")
rep("GENESIS (reference)", base, base, n_boot=10)
rep("GENESIS + net direct", base + ls(g["direct"]), base)
rep("GENESIS + net s3 (AND gate)", base + ls(g["s3"]), base)
rep("GENESIS + net who*when*direct", base + ls(g["who"]) + ls(g["when"]) + ls(g["direct"]), base)
