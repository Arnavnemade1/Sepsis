"""E6b: calibrated Genesis Network, plateau LR decay, patience 4, 3 seeds averaged."""
import os
from backend.ml import config
import sys, time, os, numpy as np, torch
from backend.ml.genesis import harness as H, targets as T, network as N
from backend.ml.genesis.features import BLOOD_GAS
torch.set_num_threads(6)
c = H.Cohort.load()
X, _ = N.input_matrix(c)
raw = c.raw
obs = lambda ch: np.isfinite(raw.col(ch))
gas = np.any([obs(g) for g in BLOOD_GAS], axis=0)
sofa = np.asarray(c.col("sofa_proxy"), dtype=float)
newlab = np.asarray(c.col("proc__new_now"), dtype=float) > 0
labs_now = np.asarray(c.col("proc__labs_now"), dtype=float)
Tg = {"direct": T.challenge(c), "who": T.patient(c), "when": T.timing(c),
      "fut_lactate6": T.future_any(c, obs("Lactate"), 6), "fut_gas6": T.future_any(c, gas, 6),
      "fut_newlab6": T.future_any(c, newlab, 6), "fut_burst6": T.future_any(c, labs_now >= 8, 6),
      "fut_sofa2_24": T.future_rise(c, sofa, 24, 2.0),
      "fut_hypot6": T.future_any(c, np.nan_to_num(raw.col("MAP").astype(float), nan=99) < 65, 6),
      "fut_fever12": T.future_any(c, np.nan_to_num(raw.col("Temp").astype(float)) >= 38.3, 12)}
W = N.world_targets(c, 6)
seeds = [0, 1, 2]
heads = ("direct", "who", "when", "s3")
out_dir = os.path.join(config.ARTIFACTS_DIR, "genesis", "heads")
oof = {h: np.full(len(c.y), np.nan) for h in heads}
pts = np.arange(c.raw.n_patients)
for k in range(c.n_folds):
    t0 = time.time()
    tr_pts, te_pts = pts[(c.folds >= 0) & (c.folds != k)], pts[c.folds == k]
    rows = c.rows_of(te_pts)
    acc = {h: np.zeros(len(rows)) for h in heads}
    for sd in seeds:
        cfg = N.NetConfig(device="cpu", epochs=25, patience=4, seed=sd)
        gn = N.GenesisNetwork(cfg).fit(c, X, Tg, tr_pts, world=W, verbose=False)
        pr = gn.predict(c, X, te_pts, heads)
        for h in heads:
            acc[h] += pr[h] / len(seeds)
        print(f"    fold {k} seed {sd}: best epoch {gn.best_epoch_} calib {({h: tuple(round(v,3) for v in gn.calib_[h]) for h in ('direct','s3')})}", flush=True)
    for h in heads:
        oof[h][rows] = acc[h]
    s = H.score(c, acc["direct"], rows, with_utility=False)
    print(f"  fold {k}: direct AUROC {s['auroc']:.4f} (B {s['between']:.4f} W {s['within_septic']:.4f}) [{time.time()-t0:.0f}s]", flush=True)
    for h in heads:
        np.save(os.path.join(out_dir, f"gn2_{h}_oof.npy"), oof[h].astype(np.float32))
print("done")
