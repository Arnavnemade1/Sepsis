"""E12: self-supervised pretraining of the Genesis Network (Stage A: GRU encoder).

Pre-registered in artifacts/genesis/lock_ledger.jsonl before it ran. Per dev fold: pretrain
the encoder on the fold's TRAINING patients (inputs only — no label), then fine-tune three
seeds exactly as the from-scratch network was trained (experiments/e6_network.py), starting
from the pretrained weights. The from-scratch arm is the existing gn2 OOF, not recomputed.

    python -m backend.ml.genesis.experiments.e12_ssl
"""
import json
import os
import time

import numpy as np
import torch

from backend.ml import config
from backend.ml.genesis import harness as H, network as N, stack as S, targets as T
from backend.ml.genesis.features import BLOOD_GAS

HEAD_DIR = os.path.join(config.ARTIFACTS_DIR, "genesis", "heads")
SEEDS = (0, 1, 2)
HEADS = ("direct", "who", "when", "s3")


def main() -> None:
    torch.set_num_threads(6)
    t_all = time.time()
    c = H.Cohort.load()
    dev = c.dev_rows
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
    oof = {h: np.full(len(c.y), np.nan) for h in HEADS}
    log = {"folds": []}
    pts = np.arange(c.raw.n_patients)
    for k in range(c.n_folds):
        t0 = time.time()
        tr_pts, te_pts = pts[(c.folds >= 0) & (c.folds != k)], pts[c.folds == k]
        pre = N.GenesisNetwork(N.NetConfig(device="cpu", mask_channels=True, seed=0)).pretrain(
            c, X, tr_pts, world=W)
        t_pre = time.time() - t0
        rows = c.rows_of(te_pts)
        acc = {h: np.zeros(len(rows)) for h in HEADS}
        best = []
        for sd in SEEDS:
            cfg = N.NetConfig(device="cpu", epochs=25, patience=4, seed=sd, mask_channels=True)
            gn = N.GenesisNetwork(cfg).fit(c, X, Tg, tr_pts, world=W, verbose=False, init=pre)
            pr = gn.predict(c, X, te_pts, HEADS)
            for h in HEADS:
                acc[h] += pr[h] / len(SEEDS)
            best.append(gn.best_epoch_)
        for h in HEADS:
            oof[h][rows] = acc[h]
            np.save(os.path.join(HEAD_DIR, f"gnssl_{h}_oof.npy"), oof[h].astype(np.float32))
        scratch = np.load(os.path.join(HEAD_DIR, "gn2_direct_oof.npy")).astype(float)[rows]
        a_ssl, a_scr = H.auroc(c.y[rows], acc["direct"]), H.auroc(c.y[rows], scratch)
        last = pre.pretext_[-1]
        log["folds"].append({"k": k, "pretext": pre.pretext_, "best_epochs": best,
                             "pretrain_sec": t_pre, "sec": time.time() - t0,
                             "auroc_ssl": a_ssl, "auroc_scratch": a_scr})
        print(f"  fold {k}: network AUROC pretrained {a_ssl:.4f} vs scratch {a_scr:.4f}  "
              f"recon {last['model']:.4f} (median {last['median']:.4f}, carry {last['carry']:.4f})  "
              f"best epochs {best}  [{time.time()-t0:.0f}s, pretrain {t_pre:.0f}s]", flush=True)

    # --------------------------------------------------------------- the verdict
    ld = lambda n: np.load(os.path.join(HEAD_DIR, f"{n}_oof.npy")).astype(float)
    ls = S.log_sigmoid
    trees = ls(ld("who")) + ls(ld("when")) + ls(ld("direct_slow"))
    scratch, ssl = ld("gn2_direct"), oof["direct"]
    res = {}
    for name, a, b in [("network_alone", ssl, scratch),
                       ("product", trees + ls(ssl), trees + ls(scratch))]:
        sa, sb = H.score(c, a[dev], dev, with_utility=False), H.score(c, b[dev], dev, with_utility=False)
        cmp = H.compare(c, a, b, n_boot=1000, rows=dev)
        res[name] = {"pretrained": sa, "scratch": sb, "delta": cmp}
        print(f"{name:<14} pretrained {sa['auroc']:.4f} (hm {sa['auroc_hourmatched']:.4f})  scratch "
              f"{sb['auroc']:.4f} (hm {sb['auroc_hourmatched']:.4f})  d {cmp['d_auroc']:+.4f} "
              f"[{cmp['auroc_ci'][0]:+.4f}, {cmp['auroc_ci'][1]:+.4f}]", flush=True)
    learned = all(f["pretext"][-1]["model"] < min(f["pretext"][-1]["median"], f["pretext"][-1]["carry"])
                  for f in log["folds"])
    primary = res["product"]["delta"]["auroc_ci"][0] > 0
    stage_b = res["network_alone"]["delta"]["auroc_ci"][0] > 0
    verdict = ("INCONCLUSIVE (pretext not learned)" if not learned else
               "PASS" if primary else "NULL (pretext learned, no gain)")
    log.update({"results": res, "pretext_learned": learned, "primary_pass": primary,
                "stage_b_allowed": stage_b, "verdict": verdict, "seconds": time.time() - t_all})
    with open(os.path.join(config.ARTIFACTS_DIR, "genesis", "ssl_stageA.json"), "w") as fh:
        json.dump(log, fh, indent=1, default=float)
    print(f"VERDICT: {verdict}; pretext learned in every fold: {learned}; "
          f"Stage B allowed: {stage_b}; total {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
