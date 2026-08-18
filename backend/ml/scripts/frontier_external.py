"""Cross-hospital confirmation of the frontier candidates — the deployment number.

Set A is Beth Israel Deaconess, set B is Emory. Within-site cross-validation says a
detector works on patients it has not seen; training on one hospital and testing on
the other says it works at a hospital it has not seen, which is the number that
survives contact with deployment. Phase 7 measured the incumbent at ~0.69 there
against ~0.77 within-site, and concluded the bottleneck was domain shift rather than
model capacity — so any candidate that only wins within-site has not actually won.

    python -m backend.ml.scripts.frontier_external --only gbm,hgb-utility,lgbm-utility

Both directions are always reported: a single direction can flatter a model whose
inductive bias happens to suit one site's monitoring practice.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from backend.ml import config
from backend.ml.eval import metrics
from backend.ml.eval.bench import _load_cached
from backend.ml.eval.utility import best_threshold_utility, utility_terms
from backend.ml.scripts.frontier import REGISTRY


def _score(ds, p) -> Dict[str, float]:
    rep = metrics.evaluate(ds.y, p, ds.groups, ds.hour_index, ds.onset_hour)
    ew = rep.early_warning or {}
    far = metrics.detection_at_false_alarm_rate(
        ds.y, p, ds.groups, ds.onset_hour, ds.hour_index, 0.15)
    u_best, _ = best_threshold_utility(
        ds.y, p, utility_terms(ds.y, ds.groups, ds.hour_index, ds.onset_hour))
    return {"auroc": rep.auroc, "auprc": rep.auprc, "ece": rep.ece,
            "utility": rep.normalized_utility, "utility_best": u_best,
            "detect_at_far15": far["detection_rate"],
            "lead_at_far15": far["median_lead_time_hours"],
            "detection": ew.get("detection_rate", float("nan")),
            "false_alarm": ew.get("false_alarm_rate", float("nan"))}


class _Joined:
    """Train and test sets glued into one FeatureMatrix-shaped object.

    Candidates in the registry take ``(dataset, train_idx, test_idx)`` because that
    is what cross-validation hands them. Concatenating the two hospitals and passing
    the index ranges lets exactly the same candidate functions run here — no
    candidate gets a second, subtly different implementation for this experiment.
    """

    def __init__(self, a, b):
        cat = lambda f: np.concatenate([getattr(a, f), getattr(b, f)])
        self.X, self.y = cat("X"), cat("y")
        self.groups, self.hour_index = cat("groups"), cat("hour_index")
        self.onset_hour = cat("onset_hour")
        self.feature_names = list(a.feature_names)
        self.n_a = len(a.y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="gbm,hgb-balanced,hgb-utility,lgbm-utility,stack,stack-utility")
    ap.add_argument("--out", default="artifacts/frontier/external.json")
    args = ap.parse_args()

    A = _load_cached("wearable_training_setA_limitall")
    B = _load_cached("wearable_training_setB_limitall")
    print(f"set A (BIDMC): {len(A.y):,} rows / {len(set(A.groups.tolist())):,} patients "
          f"/ prevalence {A.y.mean():.4f}")
    print(f"set B (Emory): {len(B.y):,} rows / {len(set(B.groups.tolist())):,} patients "
          f"/ prevalence {B.y.mean():.4f}\n")

    names = [n.strip() for n in args.only.split(",") if n.strip()]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name in names:
        fn = REGISTRY[name]
        results[name] = {}
        for direction, (tr_ds, te_ds) in {"A->B": (A, B), "B->A": (B, A)}.items():
            joined = _Joined(tr_ds, te_ds)
            tr = np.arange(joined.n_a)
            te = np.arange(joined.n_a, len(joined.y))
            p = np.asarray(fn(joined, tr, te), dtype=float)
            s = _score(te_ds, p)
            results[name][direction] = s
            print(f"  {name:<20} {direction}  AUROC {s['auroc']:.4f}  util {s['utility']:+.4f}  "
                  f"ECE {s['ece']:.4f}  detect@FA15 {s['detect_at_far15']:.3f}  "
                  f"lead@FA15 {s['lead_at_far15']:.1f}h", flush=True)
        m = {k: float(np.mean([results[name][d][k] for d in ("A->B", "B->A")]))
             for k in results[name]["A->B"]}
        results[name]["mean"] = m
        print(f"  {name:<20} MEAN  AUROC {m['auroc']:.4f}  util {m['utility']:+.4f}  "
              f"ECE {m['ece']:.4f}  detect@FA15 {m['detect_at_far15']:.3f}\n", flush=True)

    print("=" * 100)
    print(f"{'candidate':<22}{'A->B':>9}{'B->A':>9}{'mean AUROC':>13}{'mean util':>11}"
          f"{'detect@FA15':>13}{'lead@FA15':>11}")
    for n in names:
        r = results[n]
        print(f"{n:<22}{r['A->B']['auroc']:>9.4f}{r['B->A']['auroc']:>9.4f}"
              f"{r['mean']['auroc']:>13.4f}{r['mean']['utility']:>11.4f}"
              f"{r['mean']['detect_at_far15']:>13.3f}{r['mean']['lead_at_far15']:>11.1f}")
    print("=" * 100)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
