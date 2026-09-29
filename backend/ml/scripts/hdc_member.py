"""Does HDC case retrieval add to the detector as a stack member? — the registered run.

    python -m backend.ml.scripts.hdc_member            # the pre-registered analysis
    python -m backend.ml.scripts.hdc_member --reuse     # re-score saved member OOF

The design, variants, baselines, endpoint, correction and controls are fixed in
``artifacts/hdc_member_prereg.json``, committed before this ran; its hash is written into
the result so the bar can be seen to predate the numbers.

The combination is NESTED: for fold k, a logistic combiner of [logit(baseline),
z(member)] is fitted on the other folds' out-of-fold rows and applied to fold k, so no
row's score is ever shaped by its own label. The bootstrap resamples PATIENTS (multinomial
counts as row weights, one draw shared by every comparison, so each is paired).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from backend.ml import config
from backend.ml.eval.bench import Benchmark, BenchResult
from backend.ml.genesis.harness import Ranked
from backend.ml.models.hdc_knn import HDCKNN

PREREG = os.path.join(config.ARTIFACTS_DIR, "hdc_member_prereg.json")
OOF = os.path.join(config.ARTIFACTS_DIR, "frontier", "hdc_dev_oof.npz")
OUT = os.path.join(config.ARTIFACTS_DIR, "hdc_member.json")
N_BOOT = 1000


def member_oof(bench: Benchmark, p: Dict) -> Dict[str, np.ndarray]:
    ds = bench.ds
    out = {f"{e}-{t}": np.full(len(ds.y), np.nan) for e in ("flat", "segmented") for t in ("hour", "patient")}
    for k, (tr, te) in enumerate(bench.folds):
        for enc in ("flat", "segmented"):
            t0 = time.time()
            m = HDCKNN(ds.feature_names, dim=p["dim"], levels=p["levels"], k=p["k"],
                       per_patient=p["per_patient"], encoding=enc, seed=p["seed"])
            m.fit(ds.X[tr], ds.y[tr], ds.groups[tr])
            for t, v in m.predict_votes(ds.X[te]).items():
                out[f"{enc}-{t}"][te] = v
            print(f"  fold {k} {enc:<9} bank {len(m.bank_rows):,}  [{time.time() - t0:.0f}s]", flush=True)
    return out


def nested_combine(bench: Benchmark, base: np.ndarray, member: np.ndarray) -> np.ndarray:
    y = bench.ds.y
    z = np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6)))
    out = np.full(len(y), np.nan)
    for tr_fold, te in bench.folds:
        others = np.setdiff1d(np.arange(len(y)), te)          # rows of the OTHER folds (all OOF)
        mu, sd = member[others].mean(), member[others].std() or 1.0
        X = np.c_[z, (member - mu) / sd]
        lr = LogisticRegression(C=100.0, max_iter=1000).fit(X[others], y[others])
        out[te] = lr.decision_function(X[te])
    return out


def exploratory(bench: Benchmark, base: Dict[str, np.ndarray], members: Dict[str, np.ndarray]) -> Dict:
    """NOT REGISTERED. The registered noise control came out significantly NEGATIVE: the
    nested combiner re-scales each fold differently, and pooled AUROC punishes that even
    when every fold's ranking is untouched — a penalty on ANY member. So each combination is
    also compared with the baseline passed through the same combiner on its own, and with
    per-fold deltas (immune to cross-fold scale), next to detection at the ward's budget."""
    ds = bench.ds
    scores: Dict[str, np.ndarray] = {}
    for b in base:
        scores[f"{b}~refit"] = nested_combine(bench, base[b], np.zeros(len(ds.y)))
        for m, v in members.items():
            scores[f"{b}+{m}"] = nested_combine(bench, base[b], v)
    ranked = {k: Ranked(v) for k, v in scores.items()}
    pts, inv = np.unique(ds.groups, return_inverse=True)
    br = np.random.default_rng(0)
    boot = {k: np.empty(N_BOOT) for k in scores}
    for i in range(N_BOOT):
        w = br.multinomial(len(pts), np.full(len(pts), 1.0 / len(pts)))[inv].astype(np.float64)
        for k, r in ranked.items():
            boot[k][i] = r.auc(ds.y, w)
    out: Dict = {}
    for b in base:
        ref = f"{b}~refit"
        out[ref] = {"auroc": ranked[ref].auc(ds.y), "raw_auroc": Ranked(base[b]).auc(ds.y)}
        for m in members:
            key = f"{b}+{m}"
            a_ = 0.05 / 4 if not m.startswith("control") else 0.05
            lo, hi = np.percentile(boot[key] - boot[ref], [100 * a_ / 2, 100 * (1 - a_ / 2)])
            per_fold = [Ranked(scores[key][te]).auc(ds.y[te]) - Ranked(scores[ref][te]).auc(ds.y[te])
                        for _, te in bench.folds]
            row = {"delta_vs_refit": ranked[key].auc(ds.y) - ranked[ref].auc(ds.y), "ci": [float(lo), float(hi)],
                   "significant": bool(lo > 0), "per_fold_delta": per_fold}
            if not m.startswith("control"):
                cb = bench.compare_alarm_budget(BenchResult(key, [], scores[key], 0.0),
                                                BenchResult(ref, [], scores[ref], 0.0), target_far=0.15, n_boot=1000)
                row["delta_detect_at_far15"] = cb["delta_detection"]
                row["detect_significant"] = cb["significant"]
            out[key] = row
            print(f"  [exploratory] {key:<30} d vs refit {row['delta_vs_refit']:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                  f"per-fold {[round(x, 4) for x in per_fold]}"
                  + (f"  d det@FA15 {row['delta_detect_at_far15']:+.3f}" if "delta_detect_at_far15" in row else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--exploratory", action="store_true",
                    help="also run the (unregistered) refit-matched comparison")
    a = ap.parse_args()
    reg = json.load(open(PREREG))
    reg_hash = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()[:16]
    bench = Benchmark.load("dev", n_folds=3, seed=42)
    ds = bench.ds
    base = {"primary": np.load(os.path.join(config.ARTIFACTS_DIR, "frontier", "mixed_dev_oof.npz"))["stack+xgb@utility"],
            "shipped": np.load(os.path.join(config.ARTIFACTS_DIR, "subgroup_oof_dev_stack-patient.npz"))["stack-patient"]}
    if a.reuse and os.path.exists(OOF):
        members = dict(np.load(OOF))
    else:
        members = member_oof(bench, reg["member"])
        os.makedirs(os.path.dirname(OOF), exist_ok=True)
        np.savez(OOF, **members)
    rng = np.random.default_rng(7)
    members["control-oracle"] = ds.y + rng.normal(size=len(ds.y))
    members["control-noise"] = rng.normal(size=len(ds.y))

    # every score to compare, all out of fold
    scores: Dict[str, np.ndarray] = {f"{b}": base[b] for b in base}
    for b in base:
        for m, v in members.items():
            scores[f"{b}+{m}"] = nested_combine(bench, base[b], v)
    ranked = {k: Ranked(v) for k, v in scores.items()}

    # paired patient bootstrap: multinomial patient counts as row weights, shared by all
    pts, inv = np.unique(ds.groups, return_inverse=True)
    brng = np.random.default_rng(0)
    boot = {k: np.empty(N_BOOT) for k in scores}
    for i in range(N_BOOT):
        w = brng.multinomial(len(pts), np.full(len(pts), 1.0 / len(pts)))[inv].astype(np.float64)
        for k, r in ranked.items():
            boot[k][i] = r.auc(ds.y, w)
    alpha = 0.05 / 4                                                    # Bonferroni, 4 variants
    point = {k: r.auc(ds.y) for k, r in ranked.items()}

    rows = {}
    for b in base:
        for m in members:
            key = f"{b}+{m}"
            d = boot[key] - boot[b]
            a_ = alpha if not m.startswith("control") else 0.05
            lo, hi = np.percentile(d, [100 * a_ / 2, 100 * (1 - a_ / 2)])
            rows[key] = {"baseline": b, "member": m, "auroc": point[key], "delta_auroc": point[key] - point[b],
                         "ci_lo": float(lo), "ci_hi": float(hi), "interval": f"{100 * (1 - a_):.2f}%",
                         "significant": bool(lo > 0),
                         "delta_auprc": float(average_precision_score(ds.y, scores[key]) - average_precision_score(ds.y, scores[b]))}
    for key, r in rows.items():                                         # detection at the ward's budget
        if r["member"].startswith("control"):
            continue
        cb = bench.compare_alarm_budget(BenchResult(key, [], scores[key], 0.0),
                                        BenchResult(r["baseline"], [], scores[r["baseline"]], 0.0),
                                        target_far=0.15, n_boot=1000)
        r["delta_detect_at_far15"] = {k: cb[k] for k in cb if k in ("delta_detection", "ci_lo", "ci_hi", "significant")}
    standalone = {m: {"auroc": Ranked(v).auc(ds.y),
                      "spearman_vs_primary": float(spearmanr(v, base["primary"]).correlation)}
                  for m, v in members.items() if not m.startswith("control")}
    verdict = {
        "success": any(r["significant"] for r in rows.values()
                       if r["baseline"] == "primary" and not r["member"].startswith("control")),
        "controls_ok": rows["primary+control-oracle"]["significant"] and not rows["primary+control-noise"]["significant"],
    }
    rep = {"preregistration": reg, "prereg_sha256_16": reg_hash, "baselines_auroc": {b: point[b] for b in base},
           "standalone": standalone, "comparisons": rows, "verdict": verdict, "n_boot": N_BOOT}
    json.dump(rep, open(OUT, "w"), indent=1, default=float)

    print(f"\nbaselines: primary {point['primary']:.4f}  shipped {point['shipped']:.4f}")
    for m, s in standalone.items():
        print(f"  member {m:<18} standalone AUROC {s['auroc']:.4f}  Spearman vs primary {s['spearman_vs_primary']:.3f}")
    for key, r in rows.items():
        det = r.get("delta_detect_at_far15", {})
        print(f"  {key:<34} AUROC {r['auroc']:.4f}  d {r['delta_auroc']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
              f"{r['interval']} {'SIG' if r['significant'] else 'n.s.'}  dAUPRC {r['delta_auprc']:+.4f}"
              + (f"  d det@FA15 {det.get('delta_detection', float('nan')):+.3f}" if det else ""))
    print("VERDICT:", "SUCCESS" if verdict["success"] else "NULL", "| controls", "OK" if verdict["controls_ok"] else "FAILED")
    if a.exploratory:
        noise = {"control-noise": members["control-noise"]}
        rep["exploratory"] = exploratory(bench, base, {k: v for k, v in members.items()
                                                      if not k.startswith("control")} | noise)
        json.dump(rep, open(OUT, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
