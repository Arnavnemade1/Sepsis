"""E8: clinician-intent features — does reading orders as DECISIONS add signal?"""
import os
from backend.ml import config
import sys, json, numpy as np
from backend.ml.genesis import harness as H, stack as S, targets as T, learners as L, features as feat

c = H.Cohort.load()
dev = c.dev_rows
B, bnames = feat.intent_block(c.raw)
data = L.HStack(c.F, B)
print("intent block", B.shape, flush=True)
base = {k: S.head_oof(c, S.Head(k, t)) for k, t in
        [("direct", T.challenge), ("who", T.patient), ("when", T.timing)]}
new = {
    "direct_i": S.head_oof(c, S.Head("direct_i", T.challenge, data=data)),
    "when_i": S.head_oof(c, S.Head("when_i", T.timing, data=data)),
    "who_i": S.head_oof(c, S.Head("who_i", T.patient, neg_rate=0.5, data=data)),
}
ls = S.log_sigmoid
def rep(name, v, ref):
    s = H.score(c, v[dev], dev, with_utility=False)
    cmp = H.compare(c, v, ref, n_boot=500, rows=dev)
    print(f"{name:<28} AUROC {s['auroc']:.4f} hm {s['auroc_hourmatched']:.4f} B {s['between']:.4f} "
          f"W {s['within_septic']:.4f} | d {cmp['d_auroc']:+.4f} [{cmp['auroc_ci'][0]:+.4f},{cmp['auroc_ci'][1]:+.4f}] "
          f"dB {cmp['d_between']:+.4f} dW {cmp['d_within']:+.4f}", flush=True)
rep("direct+intent vs direct", new["direct_i"], base["direct"])
rep("when+intent vs when", new["when_i"], base["when"])
prod0 = ls(base["who"]) + ls(base["when"]) + ls(base["direct"])
prod1 = ls(new["who_i"]) + ls(new["when_i"]) + ls(new["direct_i"])
rep("product+intent vs product", prod1, prod0)
