"""Accuracy frontier — run detector technologies against the same fixed folds.

    python -m backend.ml.scripts.frontier --spec dev --only gbm,lgbm-utility

Each candidate is registered here as ``(dataset, train_idx, test_idx) -> p_test`` so
:class:`~backend.ml.eval.bench.Benchmark` can score them all on identical
patient-grouped folds and bootstrap the differences pairwise. Adding a technology
means adding one function; the comparison stays honest by construction.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.eval.bench import Benchmark

REGISTRY: Dict[str, Callable] = {}


def candidate(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------- #
# Incumbents — what the project ships today
# --------------------------------------------------------------------------- #
@candidate("gbm")
def _gbm(ds, tr, te):
    """The single calibrated sklearn HGB: the long-standing benchmark."""
    from backend.ml.models.gbm_baseline import GBMDetector
    m = GBMDetector(feature_names=ds.feature_names).fit(ds.X[tr], ds.y[tr], groups=ds.groups[tr])
    return m.predict_proba(ds.X[te])


@candidate("stack")
def _stack(ds, tr, te):
    """The production stacked ensemble (hgb+hgb2+et+lr)."""
    from backend.ml.models.stacked import StackedEnsembleDetector
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    m = StackedEnsembleDetector(channels=list(config.WEARABLE_CHANNELS), use_hdc=False)
    m.fit(feats.iloc[tr], ds.y[tr], groups=ds.groups[tr])
    return m.predict_proba(feats.iloc[te])


# --------------------------------------------------------------------------- #
# Booster × objective ablation
# --------------------------------------------------------------------------- #
def _booster(kind: str, objective: str):
    def run(ds, tr, te):
        from backend.ml.models.boosters import UtilityAlignedBooster
        m = UtilityAlignedBooster(kind=kind, objective=objective,
                                  feature_names=ds.feature_names)
        m.fit(ds.X[tr], ds.y[tr], groups=ds.groups[tr],
              hour_index=ds.hour_index[tr], onset_hour=ds.onset_hour[tr])
        return m.predict_proba(ds.X[te])
    return run


for _kind in ("hgb", "lgbm", "xgb"):
    for _obj in ("balanced", "utility", "utility-z", "utility-w"):
        REGISTRY[f"{_kind}-{_obj}"] = _booster(_kind, _obj)


def _booster_p(kind: str, objective: str, params: dict):
    def run(ds, tr, te):
        from backend.ml.models.boosters import UtilityAlignedBooster
        m = UtilityAlignedBooster(kind=kind, objective=objective, params=params,
                                  feature_names=ds.feature_names)
        m.fit(ds.X[tr], ds.y[tr], groups=ds.groups[tr],
              hour_index=ds.hour_index[tr], onset_hour=ds.onset_hour[tr])
        return m.predict_proba(ds.X[te])
    return run


# Fairness control for the library comparison. sklearn's HistGradientBoosting stops
# early against an internal validation split drawn at RANDOM ROW level, so the same
# patient sits on both sides of it and the stopping round is chosen against data the
# model has partly memorised. XGBoost and LightGBM here simply run every round. These
# two cells run the sklearn booster without that mechanism, so a library gap cannot be
# quietly credited to a stopping rule.
REGISTRY["hgb-utility-noes"] = _booster_p(
    "hgb", "utility", {"early_stopping": False, "max_iter": 600, "learning_rate": 0.05})
REGISTRY["hgb-balanced-noes"] = _booster_p(
    "hgb", "balanced", {"early_stopping": False, "max_iter": 600, "learning_rate": 0.05})


# --------------------------------------------------------------------------- #
# Tabular foundation model — in-context learning instead of fitting
# --------------------------------------------------------------------------- #
def _tabpfn(n_train: int = 10000, n_estimators: int = 4):
    """TabPFN: a transformer pre-trained on synthetic tabular tasks that *infers* a
    predictor from the training rows in one forward pass rather than fitting one.

    It is the newest thing in tabular ML and the obvious "is a foundation model
    better than our GBM?" question, but its context is bounded: it takes on the order
    of 10k rows, and this cohort has ~230k per fold. The subsample is patient-grouped
    and prevalence-stratified, and the handicap is stated rather than hidden — a loss
    here is a loss for TabPFN-at-this-budget, not for foundation models in general.
    """
    def run(ds, tr, te):
        from tabpfn import TabPFNClassifier
        rng = np.random.default_rng(0)
        pts = np.array(sorted(set(ds.groups[tr].tolist())))
        sep = np.array([ds.y[tr][ds.groups[tr] == g].max() for g in pts])
        # Keep every septic patient, sample the rest, then trim rows to the budget.
        keep = list(pts[sep == 1])
        others = pts[sep == 0]
        keep += list(rng.choice(others, size=min(len(others), 4 * max(len(keep), 1)),
                                replace=False))
        rows = tr[np.isin(ds.groups[tr], np.array(keep))]
        if len(rows) > n_train:
            rows = rng.choice(rows, size=n_train, replace=False)
        clf = TabPFNClassifier(n_estimators=n_estimators, ignore_pretraining_limits=True)
        clf.fit(ds.X[rows], ds.y[rows])
        out = np.empty(len(te))
        for a in range(0, len(te), 20000):            # bounded memory per forward pass
            b = min(a + 20000, len(te))
            out[a:b] = clf.predict_proba(ds.X[te[a:b]])[:, 1]
        return out
    return run


REGISTRY["tabpfn"] = _tabpfn()


# --------------------------------------------------------------------------- #
# Temporal decision layer — does a causal filter on the booster's own score
# recover what the sequence model gains?
# --------------------------------------------------------------------------- #
def _booster_filtered(kind: str, objective: str, mode: str, **fkw):
    def run(ds, tr, te):
        from backend.ml.models.boosters import UtilityAlignedBooster
        from backend.ml.models.temporal import causal_filter
        m = UtilityAlignedBooster(kind=kind, objective=objective,
                                  feature_names=ds.feature_names)
        m.fit(ds.X[tr], ds.y[tr], groups=ds.groups[tr],
              hour_index=ds.hour_index[tr], onset_hour=ds.onset_hour[tr])
        raw = m.predict_proba(ds.X[te])
        return causal_filter(raw, ds.groups[te], ds.hour_index[te], mode, **fkw)
    return run


for _mode in ("ema", "cummax", "hazard", "blend"):
    REGISTRY[f"hgb-utility-{_mode}"] = _booster_filtered("hgb", "utility", _mode)


# --------------------------------------------------------------------------- #
# The stacked ensemble under each objective
# --------------------------------------------------------------------------- #
def _stack(objective: str, roster=None, use_hdc: bool = False):
    def run(ds, tr, te):
        from backend.ml.models.stacked import StackedEnsembleDetector, DEFAULT_ROSTER
        feats = pd.DataFrame(ds.X, columns=ds.feature_names)
        m = StackedEnsembleDetector(channels=list(config.WEARABLE_CHANNELS),
                                    use_hdc=use_hdc, objective=objective,
                                    roster=list(roster or DEFAULT_ROSTER),
                                    params={k: {"n_jobs": int(os.environ.get("FRONTIER_JOBS", -1))}
                                            for k in ("et", "lgbm", "xgb")})
        m.fit(feats.iloc[tr], ds.y[tr], groups=ds.groups[tr],
              hour_index=ds.hour_index[tr], onset_hour=ds.onset_hour[tr])
        return m.predict_proba(feats.iloc[te])
    return run


REGISTRY["stack-utility"] = _stack("utility")
REGISTRY["stack-utility-lgbm"] = _stack("utility", ["hgb", "hgb2", "et", "lr", "lgbm"])
# The frontier sweep's strongest single learner, added to the roster for its error
# diversity rather than its solo score — that is the whole thesis of stacking.
REGISTRY["stack-utility-xgb"] = _stack("utility", ["hgb", "hgb2", "et", "lr", "xgb"])
REGISTRY["stack-xgb"] = _stack("balanced", ["hgb", "hgb2", "et", "lr", "xgb"])
# The minimal principled change to production: the four incumbent members are left
# exactly as they are, and one decorrelated member is added under the objective that
# makes it strong. Everything measured says this is where the headroom is — the
# objective applied uniformly is redundant (rank corr 0.965 with the incumbent), a
# different library is not (0.68).
REGISTRY["stack+xgb@utility"] = _stack("balanced", ["hgb", "hgb2", "et", "lr", "xgb@utility"])


# --------------------------------------------------------------------------- #
# Causal sequence models — a different inductive bias, not more features
# --------------------------------------------------------------------------- #
def _seq(arch: str, objective: str = "utility", **kw):
    def run(ds, tr, te):
        from backend.ml.models.sequence import DeepSequenceDetector
        m = DeepSequenceDetector(arch=arch, objective=objective, **kw)
        m.fit(ds.X[tr], ds.y[tr], ds.groups[tr], ds.hour_index[tr], ds.onset_hour[tr])
        # Test patients are scored on their OWN full stay: at the bedside a
        # patient's history is available, and none of these patients trained the net.
        return m.predict_proba(ds.X, ds.groups, ds.hour_index, te)
    return run


for _arch in ("tcn", "ssm", "xfmr"):
    for _obj in ("balanced", "utility"):
        REGISTRY[f"{_arch}-{_obj}"] = _seq(_arch, _obj)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="dev", help="dev | full | setA | setB")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated candidate names")
    ap.add_argument("--baseline", default="gbm")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=0,
                    help="cap total CPU threads (joblib + OpenMP); 0 = all cores")
    args = ap.parse_args()

    limits = None
    if args.jobs:
        # n_jobs alone leaves the OpenMP pools under HistGradientBoosting unbounded,
        # which is how a 'capped' run still takes every core.
        from threadpoolctl import threadpool_limits
        limits = threadpool_limits(limits=args.jobs)
        os.environ["FRONTIER_JOBS"] = str(args.jobs)
        print(f"capped to {args.jobs} threads")
    bench = Benchmark.load(args.spec, n_folds=args.folds)
    print(f"Benchmark[{args.spec}]: {bench.describe()}\n")

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(REGISTRY)
    for n in names:
        if n not in REGISTRY:
            raise SystemExit(f"unknown candidate {n!r}; have {sorted(REGISTRY)}")
        print(f"[{n}]", flush=True)
        bench.run(n, REGISTRY[n])
        print(flush=True)

    print("=" * 118)
    print(bench.table(names))
    print("=" * 118)
    print("util  = normalized CinC utility at the shared spec>=0.90 alarm policy")
    print("util* = ORACLE best-threshold utility (a ceiling, not deployable)\n")

    base = bench.results.get(args.baseline)
    if base is not None:
        print(f"Paired patient-bootstrap AUROC vs {args.baseline} (pooled out-of-fold):")
        for n in names:
            if n == args.baseline:
                continue
            c = bench.compare(bench.results[n], base, n_boot=args.boot)
            flag = "SIGNIFICANT" if c["significant"] else "n.s."
            print(f"  {n:<22} {c['delta_auroc']:+.4f}  "
                  f"95% CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  {flag}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {n: {"per_fold": bench.results[n].per_fold,
                       "pooled": bench.pooled(bench.results[n]),
                       "seconds": bench.results[n].seconds}
                   for n in names if n in bench.results}
        with open(args.out, "w") as fh:
            json.dump({"spec": args.spec, "folds": args.folds, "results": payload}, fh, indent=2)
        np.savez_compressed(args.out.replace(".json", "_oof.npz"),
                            **{n: bench.results[n].oof for n in names if n in bench.results})
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
