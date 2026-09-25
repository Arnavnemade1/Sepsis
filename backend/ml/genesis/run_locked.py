"""The single locked evaluation of GENESIS.

    python -m backend.ml.genesis.run_locked

Fits every detector once on all dev patients, scores it once on the locked patients,
and writes ``artifacts/genesis/locked_results.json``. The architecture evaluated here
was fixed by the pre-registration entry in ``lock_ledger.jsonl`` before this ran.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from backend.ml.genesis import final as FN
from backend.ml.genesis import harness as H
from backend.ml.genesis import stack as S
from backend.ml.genesis import targets as T
from backend.ml.genesis.rawstore import VITALS

STATIC = ("Age", "Gender", "unit_sicu", "HospAdmTime", "ICULOS", "hour", "icu_start")
NET_SEEDS = (0, 1, 2)


def wearable_cols(names):
    return np.array([j for j, n in enumerate(names)
                     if n.split("__")[0] in VITALS or n in STATIC])


def network_locked(c: H.Cohort, lock: np.ndarray) -> dict:
    """Genesis Network fitted on all dev patients, margins on the locked rows."""
    path = os.path.join(FN.OUT, "network.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    import torch
    from backend.ml.genesis import network as N
    from backend.ml.genesis.features import BLOOD_GAS
    torch.set_num_threads(6)
    X, _ = N.input_matrix(c)
    raw = c.raw
    obs = lambda ch: np.isfinite(raw.col(ch))
    gas = np.any([obs(g) for g in BLOOD_GAS], axis=0)
    sofa = np.asarray(c.col("sofa_proxy"), dtype=float)
    newlab = np.asarray(c.col("proc__new_now"), dtype=float) > 0
    labs_now = np.asarray(c.col("proc__labs_now"), dtype=float)
    Tg = {"direct": T.challenge(c), "who": T.patient(c), "when": T.timing(c),
          "fut_lactate6": T.future_any(c, obs("Lactate"), 6),
          "fut_gas6": T.future_any(c, gas, 6),
          "fut_newlab6": T.future_any(c, newlab, 6),
          "fut_burst6": T.future_any(c, labs_now >= 8, 6),
          "fut_sofa2_24": T.future_rise(c, sofa, 24, 2.0),
          "fut_hypot6": T.future_any(c, np.nan_to_num(raw.col("MAP").astype(float), nan=99) < 65, 6),
          "fut_fever12": T.future_any(c, np.nan_to_num(raw.col("Temp").astype(float)) >= 38.3, 12)}
    W = N.world_targets(c, 6)
    pts = np.arange(c.raw.n_patients)
    locked_pts = pts[c.folds < 0]
    if not np.array_equal(c.rows_of(locked_pts), lock):
        raise RuntimeError("network row order does not match the locked rows")
    heads = ("direct", "who", "when", "s3")
    pr = {h: np.zeros(len(lock)) for h in heads}
    # EXACTLY the configuration that passed the dev-CV rule: 3 seeds, 25 epochs,
    # patience 4, plateau decay, per-head Platt maps, margins averaged over seeds
    for sd in NET_SEEDS:
        cfg = N.NetConfig(device="cpu", epochs=25, patience=4, seed=sd)
        gn = N.GenesisNetwork(cfg).fit(c, X, Tg, pts[c.folds >= 0], world=W, verbose=False)
        one = gn.predict(c, X, locked_pts, heads)
        for h in heads:
            pr[h] += one[h] / len(NET_SEEDS)
        print(f"    locked network seed {sd}: best epoch {gn.best_epoch_}", flush=True)
    np.savez(path, **pr)
    return pr


SPEC = os.path.join(H.RESULTS_DIR, "genesis_spec.json")
TARGETS = {"challenge": T.challenge, "patient": T.patient, "timing": T.timing}


def spec_heads(spec: dict) -> dict:
    return {k: S.Head(v["name"], TARGETS[v["target"]], params=dict(v.get("params", {})),
                      neg_rate=float(v.get("neg_rate", 1.0)))
            for k, v in spec["heads"].items()}


def p19_locked(c: H.Cohort) -> dict:
    """P19-style patient-level task (one label per record, scored from the whole
    record): a booster on the last-hour causal features, fitted on dev patients.
    Reported beside the literature's P19 numbers WITH its caveat — the whole-record task
    leaks record length, which in CinC 2019 is itself strongly associated with outcome."""
    from backend.ml.genesis import learners as L
    last = c.raw.starts + c.raw.lens - 1
    pts = np.arange(c.raw.n_patients)
    Fp = np.asarray(c.F[last], dtype=np.float32)
    yp = c.plabel.astype(float)
    dev, lk = pts[c.folds >= 0], pts[c.folds < 0]
    fit = L.fit_lgb(Fp, dev, yp[dev], dev, strat=yp[dev].astype(int), early_stop=200,
                    params={"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 50,
                            "feature_fraction": 0.4, "lambda_l2": 5.0})
    p = fit.predict(Fp, lk)
    # clock control for the patient-level task: the same model without any feature that
    # encodes how long the record is (hour, ICULOS, icu_start, counts-so-far)
    clockish = [j for j, n in enumerate(c.names)
                if n in ("hour", "ICULOS", "icu_start") or n.endswith("__n")
                or n in ("proc__draws_all", "proc__lab_types")]
    keep = np.setdiff1d(np.arange(Fp.shape[1]), clockish)
    fit2 = L.fit_lgb(Fp, dev, yp[dev], dev, strat=yp[dev].astype(int), early_stop=200, cols=keep,
                     params={"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 50,
                             "feature_fraction": 0.4, "lambda_l2": 5.0})
    p2 = fit2.predict(Fp, lk)
    return {"p19_auroc": H.auroc(yp[lk], p), "p19_auroc_without_length_features": H.auroc(yp[lk], p2),
            "n_patients": int(len(lk)), "n_septic": int(yp[lk].sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    t0 = time.time()
    spec = json.load(open(SPEC))           # the design frozen on the dev folds
    c = H.Cohort.load()
    lock = c.lock_rows(f"FINAL locked evaluation of spec {spec['version']} "
                       "(GENESIS vs tuned single LightGBM vs wearable vs shipped stack)")
    heads = spec_heads(spec)
    heads["wearable"] = S.Head("wearable", T.challenge, cols=wearable_cols(c.names))
    Z = {k: FN.head_locked(c, h, lock) for k, h in heads.items()}
    ls = S.log_sigmoid
    trees = ls(Z["who"]) + ls(Z["when"]) + ls(Z["direct"])
    preds = {"genesis_trees": trees, "direct_lgbm": Z["direct"], "wearable_lgbm": Z["wearable"]}
    if spec.get("network"):
        nz = network_locked(c, lock)
        preds["genesis"] = trees + ls(nz[spec["network"]["expert"]])
        preds["network_alone"] = nz[spec["network"]["expert"]]
    else:
        preds["genesis"] = trees
    shipped = FN.shipped_stack_scores(c, lock)
    preds["shipped_stack"] = shipped["meta"]
    preds["shipped_stack_served"] = shipped["served"]
    res = FN.score_all(c, lock, preds, reference="shipped_stack")
    full = {k: np.full(len(c.y), np.nan) for k in preds}
    for k, v in preds.items():
        full[k][lock] = v
    res["genesis"]["vs_direct_lgbm"] = H.compare(c, full["genesis"], full["direct_lgbm"],
                                                  n_boot=2000, rows=lock)
    if "network_alone" in preds:
        res["genesis"]["vs_genesis_trees"] = H.compare(c, full["genesis"], full["genesis_trees"],
                                                        n_boot=2000, rows=lock)
    p19 = p19_locked(c)
    out = {"spec": spec, "p19": p19, "lock_fingerprint": H.lock_fingerprint(c),
           "n_locked_patients": int((c.folds < 0).sum()),
           "n_locked_rows": int(len(lock)), "seconds": time.time() - t0, "results": res}
    os.makedirs(H.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(H.RESULTS_DIR, "locked_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    for k, r in res.items():
        s = r["score"]
        print(f"{k:<22} AUROC {s['auroc']:.4f}  hour-matched {s['auroc_hourmatched']:.4f}  "
              f"<60h {s['auroc_lt60']:.4f}  between {s['between']:.4f}  within {s['within_septic']:.4f}  "
              f"AUPRC {s['auprc']:.4f}  P19 {s['patient_auroc']:.4f}  util* {s['utility_best']:+.4f}  "
              f"det@FA15 {s['detect_far15']:.3f}")
    print(f"P19-style patient task: AUROC {p19['p19_auroc']:.4f}  "
          f"(without record-length features {p19['p19_auroc_without_length_features']:.4f})")
    for k, r in res.items():
        for key, cmp in r.items():
            if key.startswith("vs_") and cmp:
                print(f"  {k} {key}: dAUROC {cmp['d_auroc']:+.4f} CI [{cmp['auroc_ci'][0]:+.4f}, "
                      f"{cmp['auroc_ci'][1]:+.4f}]  dBetween {cmp['d_between']:+.4f}  dWithin {cmp['d_within']:+.4f}")


if __name__ == "__main__":
    main()
