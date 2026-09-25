"""E13: physiological-twin features for the booster.

A tree cannot easily compute "observed minus what this patient's history predicts" — it
is a regression residual over many inputs. The self-supervised twin (E12's pretraining,
no labels) computes exactly that, so its causal readouts are handed to the tuned booster
as new features: the twin's state (PCA of the GRU state) and its forecast SURPRISE — the
observed value at hour t minus the forecast it made at t-1 — for every channel.

Pre-registered in artifacts/genesis/lock_ledger.jsonl. Per dev fold the twin sees only
that fold's training patients.

    python -m backend.ml.genesis.experiments.e13_twin_features
"""
import json
import os
import time

import numpy as np
import torch
from sklearn.decomposition import PCA

from backend.ml import config
from backend.ml.genesis import harness as H, learners as L, network as N, stack as S, targets as T
from backend.ml.genesis.features import _Seg

HEAD_DIR = os.path.join(config.ARTIFACTS_DIR, "genesis", "heads")
SLOW = {"learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 800,
        "feature_fraction": 0.3, "bagging_fraction": 0.7, "lambda_l2": 10.0}
N_PCA = 32


def twin_block(c: H.Cohort, X: np.ndarray, twin: N.GenesisNetwork, pts: np.ndarray,
               pca_rows_from: np.ndarray, seg: _Seg, seed: int) -> np.ndarray:
    """Full-length (N, 32 + 34 + 3) block; rows of patients outside ``pts`` stay NaN."""
    rows, hid, fc = twin.encode(c, X, pts)
    B = np.full((len(c.y), N_PCA + N.N_CH + 3), np.nan, dtype=np.float32)
    # PCA of the twin state, fitted on training rows only
    rng = np.random.default_rng(seed)
    fit_mask = np.isin(c.seg[rows], pca_rows_from)
    fit_idx = np.flatnonzero(fit_mask)
    sub = fit_idx if len(fit_idx) <= 200_000 else rng.choice(fit_idx, 200_000, replace=False)
    pca = PCA(n_components=N_PCA, random_state=seed).fit(hid[sub])
    B[rows, :N_PCA] = pca.transform(hid).astype(np.float32)
    # forecast surprise: observed (standardised) value at t minus the forecast made at t-1
    sur = np.full((len(rows), N.N_CH), np.nan, dtype=np.float32)
    step = 200_000
    for a in range(0, len(rows), step):
        r = rows[a:a + step]
        Xr = X[r]
        z = twin.std_(Xr)[:, N.VALUE_COLS]
        now = Xr[:, N.NOW_COLS] > 0.5
        prev_ok = np.r_[a > 0, np.ones(len(r) - 1, dtype=bool)]
        i = np.arange(a, a + len(r))
        prev_same = (i > 0) & (rows[np.maximum(i - 1, 0)] == r - 1) & \
                    (c.seg[rows[np.maximum(i - 1, 0)]] == c.seg[r])
        fprev = np.where(prev_same[:, None], fc[np.maximum(i - 1, 0)], np.nan)
        sur[a:a + len(r)] = np.where(now, z - fprev, np.nan)
    B[rows, N_PCA:N_PCA + N.N_CH] = sur
    absu = np.abs(sur)
    cnt = np.isfinite(absu).sum(1).astype(np.float64)
    tot = np.nansum(absu, 1).astype(np.float64)
    full_cnt, full_tot = np.zeros(len(c.y)), np.zeros(len(c.y))
    full_cnt[rows], full_tot[rows] = cnt, tot
    B[rows, N_PCA + N.N_CH] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    Pc, Pt = seg.prefix(full_cnt), seg.prefix(full_tot)
    for j, w in enumerate((6, 24), start=1):
        wc, wt = seg.window_sum(Pc, w)[rows], seg.window_sum(Pt, w)[rows]
        B[rows, N_PCA + N.N_CH + j] = np.where(wc > 0, wt / np.maximum(wc, 1), np.nan)
    return B


def main() -> None:
    torch.set_num_threads(6)
    t_all = time.time()
    c = H.Cohort.load()
    dev = c.dev_rows
    X, _ = N.input_matrix(c)
    W = N.world_targets(c, 6)
    seg = _Seg(c.raw)
    tgt, w = T.challenge(c)
    pts = np.arange(c.raw.n_patients)
    dev_pts = pts[c.folds >= 0]
    oof = np.full(len(c.y), np.nan)
    log = {"folds": []}
    for k in range(c.n_folds):
        t0 = time.time()
        tr_pts = pts[(c.folds >= 0) & (c.folds != k)]
        twin = N.GenesisNetwork(N.NetConfig(device="cpu", mask_channels=True, seed=0)).pretrain(
            c, X, tr_pts, world=W, verbose=False)
        B = twin_block(c, X, twin, dev_pts, tr_pts, seg, seed=k)
        data = L.HStack(c.F, B)
        head = S.Head("direct_twin", T.challenge, params=SLOW, data=data)
        tr, te = c.fold_rows(k)
        fit = S.fit_head(c, head, tr, tgt, w, seed=k)
        oof[te] = fit.predict(data, te, raw=True)
        gain = fit.booster.feature_importance(importance_type="gain")
        twin_share = float(gain[c.F.shape[1]:].sum() / max(gain.sum(), 1e-9))
        base = np.load(os.path.join(HEAD_DIR, "direct_slow_oof.npy")).astype(float)
        a_new, a_old = H.auroc(c.y[te], oof[te]), H.auroc(c.y[te], base[te])
        log["folds"].append({"k": k, "auroc_twin": a_new, "auroc_base": a_old, "trees": fit.best_iter,
                             "twin_gain_share": twin_share, "pretext": twin.pretext_[-1],
                             "sec": time.time() - t0})
        np.save(os.path.join(HEAD_DIR, "direct_twin_oof.npy"), oof.astype(np.float32))
        print(f"  fold {k}: AUROC booster+twin {a_new:.4f} vs booster {a_old:.4f}  "
              f"(twin features carry {twin_share:.1%} of split gain; trees {fit.best_iter})  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del B, data

    ld = lambda n: np.load(os.path.join(HEAD_DIR, f"{n}_oof.npy")).astype(float)
    ls = S.log_sigmoid
    base = ld("direct_slow")
    who, when = ld("who"), ld("when")
    res = {}
    for name, a, b in [("booster", oof, base),
                       ("product", ls(who) + ls(when) + ls(oof), ls(who) + ls(when) + ls(base))]:
        sa, sb = H.score(c, a[dev], dev, with_utility=False), H.score(c, b[dev], dev, with_utility=False)
        cmp = H.compare(c, a, b, n_boot=1000, rows=dev)
        res[name] = {"with_twin": sa, "without": sb, "delta": cmp}
        print(f"{name:<8} with twin {sa['auroc']:.4f} (hm {sa['auroc_hourmatched']:.4f}, B {sa['between']:.4f}, "
              f"W {sa['within_septic']:.4f})  without {sb['auroc']:.4f} (hm {sb['auroc_hourmatched']:.4f})  "
              f"d {cmp['d_auroc']:+.4f} [{cmp['auroc_ci'][0]:+.4f}, {cmp['auroc_ci'][1]:+.4f}]  "
              f"dB {cmp['d_between']:+.4f} dW {cmp['d_within']:+.4f}", flush=True)
    d = res["booster"]["delta"]
    passed = d["auroc_ci"][0] > 0
    meaningful = passed and d["d_auroc"] >= 0.005
    verdict = "MEANINGFUL" if meaningful else ("PASS (below +0.005)" if passed else "NULL")
    log.update({"results": res, "verdict": verdict, "seconds": time.time() - t_all})
    with open(os.path.join(config.ARTIFACTS_DIR, "genesis", "twin_features.json"), "w") as fh:
        json.dump(log, fh, indent=1, default=float)
    print(f"VERDICT: {verdict}  total {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
