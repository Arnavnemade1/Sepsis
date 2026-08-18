"""Rescore saved out-of-fold predictions and write the frontier results table.

    python -m backend.ml.scripts.frontier_report artifacts/frontier/*_oof.npz \
        --baseline gbm --md docs/FRONTIER_REPORT.md

Every ``frontier`` run stores each candidate's pooled out-of-fold prediction vector.
Because all runs share the same dataset spec, fold count and seed, those vectors are
directly comparable — so the full table can be rebuilt, and new metrics applied to old
candidates, without retraining anything. Anything reported here is therefore scored by
the same code on the same rows for every candidate, which is the only way a table of
this kind means anything.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

from backend.ml.eval.bench import Benchmark, BenchResult

ROW = ("| {name} | {auroc:.4f} ± {sd:.4f} | {auprc:.4f} | {util:+.4f} | {ece:.4f} | "
       "{d15:.3f} | {l15:.1f} | {fa:.3f} |")


def load(paths: List[str]) -> Dict[str, np.ndarray]:
    oof: Dict[str, np.ndarray] = {}
    for path in paths:
        d = np.load(path)
        for k in d.files:
            if k in oof and not np.allclose(np.nan_to_num(oof[k]), np.nan_to_num(d[k])):
                print(f"  note: {k!r} appears in more than one file with different values; "
                      f"keeping the first")
                continue
            oof[k] = d[k]
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--spec", default="dev")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--baseline", default="gbm")
    ap.add_argument("--boot", type=int, default=600)
    ap.add_argument("--order", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()

    bench = Benchmark.load(args.spec, n_folds=args.folds)
    oof = load(args.paths)
    print(f"Benchmark[{args.spec}]: {bench.describe()}")
    print(f"rescoring {len(oof)} candidates: {', '.join(sorted(oof))}\n")

    for name, p in oof.items():
        if np.isnan(p).any():
            print(f"  skipping {name}: {int(np.isnan(p).sum())} rows never scored")
            continue
        per_fold = [bench._score(te, p[te]) for _, te in bench.folds]
        bench.results[name] = BenchResult(name, per_fold, p, 0.0)

    order = [n.strip() for n in args.order.split(",") if n.strip()] or sorted(bench.results)
    order = [n for n in order if n in bench.results]

    hdr = ("| candidate | AUROC | AUPRC | utility | ECE | detect@FA15 | lead@FA15 (h) | "
           "FA@spec90 |")
    sep = "|" + "|".join(["---"] * 8) + "|"
    lines = [hdr, sep]
    for n in order:
        r = bench.results[n]
        lines.append(ROW.format(name=n, auroc=r.mean("auroc"), sd=r.std("auroc"),
                                auprc=r.mean("auprc"), util=r.mean("utility"),
                                ece=r.mean("ece"), d15=r.mean("detect_at_far15"),
                                l15=r.mean("lead_at_far15"), fa=r.mean("false_alarm")))
    table = "\n".join(lines)
    print(table)

    boot_lines = []
    base = bench.results.get(args.baseline)
    if base is not None:
        boot_lines.append(f"\nPaired patient-level bootstrap of pooled-OOF AUROC vs "
                          f"`{args.baseline}` ({args.boot} draws):\n")
        boot_lines.append("| candidate | ΔAUROC | 95% CI | verdict |")
        boot_lines.append("|---|---|---|---|")
        for n in order:
            if n == args.baseline:
                continue
            c = bench.compare(bench.results[n], base, n_boot=args.boot)
            boot_lines.append(f"| {n} | {c['delta_auroc']:+.4f} | "
                              f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
                              f"{'**significant**' if c['significant'] else 'n.s.'} |")
    print("\n".join(boot_lines))

    if args.md:
        os.makedirs(os.path.dirname(args.md) or ".", exist_ok=True)
        with open(args.md, "w") as fh:
            fh.write(f"# Detection accuracy frontier — `{args.spec}` benchmark\n\n")
            fh.write(f"{bench.describe()}; identical patient-grouped folds for every row.\n\n")
            fh.write(table + "\n")
            fh.write("\n".join(boot_lines) + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
