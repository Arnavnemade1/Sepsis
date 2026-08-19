"""Freeze the frontier comparison into one auditable artifact for the website.

The research page's rule is that every number it prints came out of a committed
artifact through `site/scripts/sync-results.mjs`, so a stale figure is a failing CI
check rather than a quiet lie. The frontier comparison was computed interactively from
saved out-of-fold predictions, which is reproducible but not *recorded* — this script
recomputes it from those same `*_oof.npz` vectors and writes it down.

    python -m backend.ml.scripts.frontier_summary

No model is refitted: the out-of-fold predictions already exist, so this is arithmetic
over vectors on disk. It writes artifacts/frontier_summary.json.

PROVENANCE CAVEAT: its inputs, artifacts/frontier/*_oof.npz, are gitignored (they are
large and regenerable), so this script cannot be re-run from a fresh clone — only after
re-running the benchmark that produces them. The OUTPUT is committed, which is what the
site's sync step and its dirty-diff CI check actually consume, so the published numbers
stay auditable even though the intermediate vectors do not ship.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

from backend.ml import config
from backend.ml.eval.bench import Benchmark, BenchResult

#: Candidate -> the run whose OOF file holds it. Both runs used the same spec,
#: fold count and seed, which is what makes the vectors comparable at all.
SOURCES = {
    "stack": "production_dev_oof.npz",
    "stack-utility": "production_dev_oof.npz",
    "xgb-utility": "production_dev_oof.npz",
    "stack+xgb@utility": "mixed_dev_oof.npz",
}
BASELINE = "stack"
PRODUCTION = "stack+xgb@utility"


def main() -> None:
    fdir = os.path.join(config.ARTIFACTS_DIR, "frontier")
    bench = Benchmark.load("dev", n_folds=3)
    oof: Dict[str, np.ndarray] = {}
    for name, fn in SOURCES.items():
        path = os.path.join(fdir, fn)
        if not os.path.exists(path):
            print(f"  missing {path}; skipping {name}")
            continue
        d = np.load(path)
        if name in d.files:
            oof[name] = d[name]
    if BASELINE not in oof:
        raise SystemExit(f"cannot summarise without the {BASELINE!r} baseline")

    res = {n: BenchResult(n, [], p, 0.0) for n, p in oof.items()}
    for r in res.values():
        r.per_fold = [bench._score(te, r.oof[te]) for _, te in bench.folds]

    keys = ("auroc", "auprc", "utility", "utility_best", "ece",
            "detect_at_far15", "lead_at_far15", "detection", "false_alarm")
    candidates = {
        n: {**{k: round(r.mean(k), 5) for k in keys},
            "aurocSd": round(r.std("auroc"), 5),
            "perFoldAuroc": [round(f["auroc"], 5) for f in r.per_fold]}
        for n, r in res.items()
    }

    base = res[BASELINE]
    comparisons = {}
    for n, r in res.items():
        if n == BASELINE:
            continue
        c = bench.compare(r, base, n_boot=2000)
        # Detection at a fixed alarm budget is swept across neighbouring budgets on
        # purpose: a gain that shows up at one budget and vanishes at both of its
        # neighbours is an artifact, and the page says so out loud.
        # NOTE bool is a subclass of int in Python, so a blanket round() over the
        # result dict turns `significant: True` into `1`. JSON consumers then type
        # it as a number and every `boolean` annotation downstream is a lie.
        def clean(d):
            return {k: (bool(v) if isinstance(v, bool) else round(v, 5))
                    for k, v in d.items() if isinstance(v, (int, float))}

        budgets = {
            f"{int(far*100)}": clean(
                bench.compare_alarm_budget(r, base, far, n_boot=2000))
            for far in (0.10, 0.15, 0.20)
        }
        comparisons[n] = {
            "vs": BASELINE,
            "deltaAuroc": round(c["delta_auroc"], 5),
            "ci": [round(c["ci_lo"], 5), round(c["ci_hi"], 5)],
            "significant": bool(c["significant"]),
            "rankCorr": round(float(np.corrcoef(rankdata(r.oof), rankdata(base.oof))[0, 1]), 4),
            "detectionAtBudget": budgets,
        }

    manifest_path = os.path.join(config.ARTIFACTS_DIR, "detector_manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    out = {
        "protocol": "within-site patient-grouped cross-validation (identical folds for "
                    "every candidate); NOT cross-hospital",
        "spec": "dev", "folds": len(bench.folds),
        "rows": int(len(bench.ds.y)),
        "patients": int(len(set(bench.ds.groups.tolist()))),
        "prevalence": round(float(bench.ds.y.mean()), 5),
        "bootstrap": 2000,
        "baseline": BASELINE,
        "production": PRODUCTION,
        "candidates": candidates,
        "comparisons": comparisons,
        "promotion": {
            "created": manifest.get("created"),
            "gates": manifest.get("gates"),
            "promoted": manifest.get("promoted"),
            "holdoutAurocDelta": round(manifest.get("auroc_delta", float("nan")), 5),
            "holdoutAuprcDelta": round(manifest.get("auprc_delta", float("nan")), 5),
            "operatingPoint": manifest.get("operating_point"),
            "configs": manifest.get("configs"),
        },
        "crossHospitalConfirmed": False,
    }
    path = os.path.join(config.ARTIFACTS_DIR, "frontier_summary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")
    for n, c in comparisons.items():
        print(f"  {n:<20} ΔAUROC {c['deltaAuroc']:+.4f} CI {c['ci']} "
              f"rankcorr {c['rankCorr']:.3f} "
              f"{'SIGNIFICANT' if c['significant'] else 'n.s.'}")


if __name__ == "__main__":
    main()
