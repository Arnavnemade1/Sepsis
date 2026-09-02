"""Audit a detector and write ``artifacts/model_audit.json``.

    PYTHONPATH=. python -m backend.ml.scripts.run_audit                # the shipped detector
    PYTHONPATH=. python -m backend.ml.scripts.run_audit --oof path.npy:name

Separated from the serving path deliberately. The suspicion-echo probe fits two models,
which is far too slow for a request handler, so the audit is computed here and the REST
route serves the artifact. That also means the number a judge sees is the number that was
committed, not one recomputed on a warm cache.

The default subject is **our own shipped detector**, because an instrument for finding
soft numbers in other people's work has no standing until it has been pointed at ours.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np

from backend.ml import audit as audit_mod
from backend.ml import config
from backend.ml.eval.bench import Benchmark
from backend.ml.operating_point import load as load_operating_point

OUT = os.path.join(config.ARTIFACTS_DIR, "model_audit.json")


def _shipped_oof(bench: Benchmark, spec: str) -> tuple[np.ndarray, str]:
    """Out-of-fold scores for the shipped roster, reused from disk when available."""
    cached = os.path.join(config.ARTIFACTS_DIR, f"subgroup_oof_{spec}_stack-patient.npz")
    if os.path.exists(cached):
        d = np.load(cached, allow_pickle=True)
        key = [k for k in d.files if k != "y"][0]
        p = np.asarray(d[key], dtype=float)
        if len(p) == len(bench.ds.y):
            return p, "stack-patient (cached OOF)"
    from backend.ml.scripts.frontier import REGISTRY
    print("  no cached OOF — running stack-patient on the bench folds", flush=True)
    return bench.run("stack-patient", REGISTRY["stack-patient"]).oof, "stack-patient"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="dev")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--oof", default="", help="path.npy:name — audit any detector's scores")
    ap.add_argument("--threshold", type=float, default=None,
                    help="alarm threshold; defaults to the shipped operating point")
    ap.add_argument("--draws", type=int, default=25)
    ap.add_argument("--no-suspicion", action="store_true",
                    help="skip the two model fits the suspicion probe needs")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    t0 = time.time()
    bench = Benchmark.load(args.spec, n_folds=args.folds)
    ds = bench.ds
    print(f"cohort: {bench.describe()}", flush=True)

    if args.oof:
        path, _, name = args.oof.partition(":")
        p = np.load(path)
        name = name or os.path.basename(path)
        if len(p) != len(ds.y):
            raise SystemExit(f"{path} has {len(p)} scores for {len(ds.y)} rows")
    else:
        p, name = _shipped_oof(bench, args.spec)

    thr = args.threshold
    if thr is None:
        op = load_operating_point()
        thr = float(op.high) if op is not None else float(np.quantile(p[np.isfinite(p)], 0.95))
        print(f"  threshold {thr:.4f} "
              f"({'shipped operating point' if op is not None else 'p95 fallback'})")

    rep = audit_mod.audit(
        name=name, y=ds.y, p=p, groups=ds.groups,
        hour_index=ds.hour_index, onset_hour=ds.onset_hour, threshold=thr,
        X=None if args.no_suspicion else ds.X,
        feature_names=None if args.no_suspicion else ds.feature_names,
        folds=None if args.no_suspicion else bench.folds[:2],
        n_draws=args.draws,
    )

    print(f"\n=== AUDIT: {rep.detector} ===")
    print(f"  pooled AUROC                      {rep.pooled_auroc:.4f}")
    print(f"  BETWEEN  (which patient)          {rep.between_auroc:.4f}")
    print(f"  WITHIN   (when, per-stay)         {rep.within_auroc_strict:.4f}"
          f"   [pooled form: {rep.within_auroc_pooled:.4f}, over {rep.n_within_stays} stays]")
    print(f"  lead time observed                {rep.lead_observed_h:.1f}h "
          f"at detection {rep.lead_detection_rate:.3f}")
    print(f"  lead time, RANDOM null            {rep.lead_null_h:.1f}h "
          f"[{rep.lead_null_lo:.1f},{rep.lead_null_hi:.1f}]")
    print(f"  EXCESS over null                  {rep.lead_excess_h:+.1f}h")
    if np.isfinite(rep.suspicion_drop):
        print(f"  suspicion echo: {rep.suspicion_auroc_full:.4f} -> "
              f"{rep.suspicion_auroc_stripped:.4f} "
              f"(drop {rep.suspicion_drop:+.4f} removing {rep.suspicion_n_features} "
              f"measurement-pattern features)")
    print(f"\n  VERDICT: {rep.verdict()}")

    payload = rep.as_dict()
    payload["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["spec"] = args.spec
    payload["threshold"] = float(thr)
    payload["verdict"] = rep.verdict()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  wrote {args.out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
