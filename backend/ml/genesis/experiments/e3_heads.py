"""E3: calibrated heads (direct/who/when + horizons + label-genesis auxiliaries) and stage 2."""
import os
from backend.ml import config
import sys, time, json, numpy as np
from backend.ml.genesis import harness as H, stack as S, targets as T
from backend.ml.genesis.features import BLOOD_GAS

c = H.Cohort.load()
dev = c.dev_rows
raw = c.raw
obs = lambda ch: np.isfinite(raw.col(ch))
gas = np.any([obs(g) for g in BLOOD_GAS], axis=0)
temp = raw.col("Temp").astype(float)
mapv = raw.col("MAP").astype(float)
sofa = np.asarray(c.col("sofa_proxy"), dtype=float)
newlab = np.asarray(c.col("proc__new_now"), dtype=float) > 0
labs_now = np.asarray(c.col("proc__labs_now"), dtype=float)

heads = [
    S.Head("direct", T.challenge),
    S.Head("who", T.patient, neg_rate=0.5),
    S.Head("when", T.timing),
    S.Head("h12", lambda c: T.horizon(c, 12), neg_rate=0.5),
    S.Head("h24", lambda c: T.horizon(c, 24), neg_rate=0.5),
    S.Head("fut_lactate6", lambda c: T.future_any(c, obs("Lactate"), 6), neg_rate=0.5),
    S.Head("fut_gas6", lambda c: T.future_any(c, gas, 6), neg_rate=0.5),
    S.Head("fut_newlab6", lambda c: T.future_any(c, newlab, 6), neg_rate=0.5),
    S.Head("fut_burst6", lambda c: T.future_any(c, labs_now >= 8, 6), neg_rate=0.5),
    S.Head("fut_fever12", lambda c: T.future_any(c, np.nan_to_num(temp) >= 38.3, 12), neg_rate=0.5),
    S.Head("fut_hypot6", lambda c: T.future_any(c, np.nan_to_num(mapv, nan=99) < 65, 6), neg_rate=0.5),
    S.Head("fut_sofa2_24", lambda c: T.future_rise(c, sofa, 24, 2.0), neg_rate=0.5),
]
M = {}
for h in heads:
    t0 = time.time()
    M[h.name] = S.head_oof(c, h)
    print(f"{h.name}: {time.time()-t0:.0f}s", flush=True)
direct = M["direct"]

res = {}
def report(name, v):
    s = H.score(c, v[dev], dev)
    cmp = H.compare(c, v, direct, n_boot=500, rows=dev)
    res[name] = {"score": s, "vs_direct": cmp}
    print(f"{name:<34} AUROC {s['auroc']:.4f} hm {s['auroc_hourmatched']:.4f} B {s['between']:.4f} "
          f"W {s['within_septic']:.4f} P19 {s['patient_auroc']:.4f} | d {cmp['d_auroc']:+.4f} "
          f"[{cmp['auroc_ci'][0]:+.4f},{cmp['auroc_ci'][1]:+.4f}] dB {cmp['d_between']:+.4f} dW {cmp['d_within']:+.4f}",
          flush=True)

for k in M:
    report(f"single:{k}", M[k])
ls = S.log_sigmoid
report("prod:who*when", ls(M["who"]) + ls(M["when"]))
report("prod:who*when*direct", ls(M["who"]) + ls(M["when"]) + ls(direct))

sets = {
    "core": ["direct", "who", "when"],
    "core+horizon": ["direct", "who", "when", "h12", "h24"],
    "core+genesis": ["direct", "who", "when"] + [h.name for h in heads if h.name.startswith("fut_")],
    "all": list(M),
}
for tag, keys in sets.items():
    Mx, names = S.meta_matrix(c, {k: M[k] for k in keys}, temporal=False)
    r = H.run(c, f"e3_lr_{tag}", S.lr_meta_candidate(Mx), verbose=False)
    report(f"lr:{tag}", np.where(np.isfinite(r.oof), r.oof, 0.0))
for tag in ("core", "all"):
    keys = sets[tag]
    for temporal in (False, True):
        Mx, names = S.meta_matrix(c, {k: M[k] for k in keys}, temporal=temporal)
        nm = f"gbm:{tag}{'+temporal' if temporal else ''}"
        r = H.run(c, "e3_" + nm.replace(":", "_").replace("+", "_"), S.meta_candidate(Mx), verbose=False)
        report(nm, np.where(np.isfinite(r.oof), r.oof, 0.0))
json.dump(res, open(os.path.join(config.ARTIFACTS_DIR, "genesis", "e3_results.json"), "w"), indent=1, default=float)
