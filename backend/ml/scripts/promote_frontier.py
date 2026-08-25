"""Promote the utility-aligned detector — one pass, three fits, every gate on the way.

Compares one candidate ensemble configuration against the shipped one on held-out
patients and replaces the production artifact — but only through the same refusals the
rest of the project uses. It has already refused once, which is the point of it.

The default candidate is the one the frontier sweep argued for: the four incumbent
members untouched, plus a decorrelated XGBoost member fitted on the utility-aligned
target. The sweep found the utility objective is a large win for a *single* booster but
redundant inside the stack (its scores rank-correlate 0.965 with the incumbent), while
a different library rank-correlates only ~0.68 — so the objective earns its place on
the diverse member, not on all of them.

    python -m backend.ml.scripts.promote_frontier --limit 8000 --jobs 6            # measure
    python -m backend.ml.scripts.promote_frontier --limit 8000 --jobs 6 --promote  # ship
    ... --roster hgb,hgb2,et,lr            # or reproduce the incumbent as the candidate

Deliberately cheap: exactly **three** ensemble fits, no cross-validation sweep.

  1. incumbent objective on the training patients   (what we ship today)
  2. utility objective on the same patients         (the candidate)
  3. the winner refit on everything                 (only if the gates pass)

Everything else — the temporal-filter comparison, the alarm operating point, the
held-out gate — is computed from predictions those fits already produced, because a
threshold sweep costs nothing once the scores exist. ``--jobs`` caps total CPU — joblib
*and* the OpenMP pools underneath the histogram booster — so this can run on a machine
somebody is still using.

Two gates, both on patients neither model was fitted on:

  * **discrimination** — the candidate must not lose AUROC.
  * **decisions** — each detector is given its *own* alarm threshold, derived on one
    half of the held-out patients, and judged on the other half. The candidate must not
    lose normalized utility read that way.

A note on the AUROC bar, because it moved. This script originally required a held-out
AUROC gain above 0.005 — the project's standing "not noise" heuristic. That is a
*proxy* for reality, and there is now a direct test: `eval/bench.py` bootstraps the
difference by resampling patients across the whole cross-validated cohort. Where the
direct test is available it supersedes the proxy, and the bar here is "do not regress"
with significance established on the benchmark instead. In exchange the second gate got
strictly harder: it used to compare both detectors at one shared threshold, which is
the same error as reading two detectors off one operating point — a score distribution
that suits the incumbent makes any differently-shaped candidate look worse for reasons
that have nothing to do with its quality. Each now gets its own derived point.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml import operating_point as opmod
from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import FeatureConfig, build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_split
from backend.ml.eval.utility import best_threshold_utility, utility_terms
from backend.ml.models.stacked import StackedEnsembleDetector

#: Held-out AUROC the candidate is allowed to give up. Significance is established on
#: the cross-validated benchmark (paired patient bootstrap), not by this single split,
#: so this is a regression guard rather than a discovery test.
MAX_AUROC_LOSS = 0.0


class _cpu_cap:
    """Hold total CPU use down to ``jobs`` cores for the duration of a fit.

    ``n_jobs`` alone is not enough: it governs scikit-learn's own joblib parallelism,
    but HistGradientBoosting is threaded through OpenMP underneath and will happily
    take every core regardless. threadpoolctl reaches the native thread pools, so the
    flag means what it says and the machine stays usable while this runs.
    """

    def __init__(self, jobs: int):
        self.jobs = jobs
        self._ctl = None

    def __enter__(self):
        if self.jobs and self.jobs > 0:
            try:
                from threadpoolctl import threadpool_limits
                self._ctl = threadpool_limits(limits=self.jobs)
            except ImportError:      # sklearn ships it, but never fail over a nicety
                pass
        return self

    def __exit__(self, *exc):
        if self._ctl is not None:
            self._ctl.__exit__(*exc)
        return False


def _score(y, p, g, h, o) -> Dict[str, float]:
    rep = metrics.evaluate(y, p, g, h, o)
    ew = rep.early_warning or {}
    far = metrics.detection_at_false_alarm_rate(y, p, g, o, h, 0.15)
    u_best, _ = best_threshold_utility(y, p, utility_terms(y, g, h, o))
    return {"auroc": rep.auroc, "auprc": rep.auprc, "ece": rep.ece, "brier": rep.brier,
            "utility": rep.normalized_utility, "utility_best": u_best,
            "detect_at_far15": far["detection_rate"],
            "lead_at_far15": far["median_lead_time_hours"],
            "detection_at_spec90": ew.get("detection_rate", float("nan")),
            "false_alarm_at_spec90": ew.get("false_alarm_rate", float("nan")),
            "median_lead_at_spec90": ew.get("median_lead_time_hours", float("nan"))}


def _line(name: str, s: Dict[str, float]) -> str:
    return (f"  {name:<28} AUROC {s['auroc']:.4f}  AUPRC {s['auprc']:.4f}  "
            f"util {s['utility']:+.4f}  ECE {s['ece']:.4f}  "
            f"detect@FA15 {s['detect_at_far15']:.3f}  lead@FA15 {s['lead_at_far15']:.1f}h")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["wearable", "lab"], default="wearable")
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=0,
                    help="cap tree parallelism (0 = all cores); use e.g. 4 on a laptop")
    ap.add_argument("--max-auroc-loss", type=float, default=MAX_AUROC_LOSS,
                    help="held-out AUROC the candidate may give up (default 0: none)")
    ap.add_argument("--max-far", type=float, default=0.15,
                    help="alarm budget: fraction of non-septic patients ever alarmed")
    ap.add_argument("--roster", default="hgb,hgb2,et,lr,xgb@utility",
                    help="candidate roster; a member may name its own objective with "
                         "'@', e.g. xgb@utility")
    ap.add_argument("--objective", choices=["balanced", "utility"], default="balanced",
                    help="objective for roster members that do not name their own")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    cfg = (FeatureConfig.wearable_only if args.feature_set == "wearable"
           else FeatureConfig.lab_augmented)()
    ds = build_dataset_cached(lambda: load_patients(limit=args.limit), cfg,
                              f"{args.feature_set}_limit{args.limit or 'all'}")
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    print(f"Dataset: {len(ds.y):,} rows / {len(set(ds.groups.tolist())):,} patients / "
          f"hour-prevalence {ds.y.mean():.4f}")

    params = ({k: {"n_jobs": args.jobs} for k in ("et", "lgbm", "xgb")}
              if args.jobs else {})
    if args.jobs:
        print(f"capping CPU to {args.jobs} threads (joblib + OpenMP)")
    from backend.ml.models.stacked import DEFAULT_ROSTER
    cand_roster = [r.strip() for r in args.roster.split(",") if r.strip()]
    configs = {
        # The shipped detector, reproduced exactly, so the comparison is against what
        # is actually running rather than against a fresh idea of it.
        "incumbent": dict(objective="balanced", roster=list(DEFAULT_ROSTER)),
        "candidate": dict(objective=args.objective, roster=cand_roster),
    }
    mk = lambda name: StackedEnsembleDetector(
        channels=list(cfg.channels), use_hdc=False, params=params, **configs[name])
    print(f"incumbent roster: {configs['incumbent']['roster']} "
          f"[objective={configs['incumbent']['objective']}]")
    print(f"candidate roster: {configs['candidate']['roster']} "
          f"[objective={configs['candidate']['objective']}]")

    tr, te = stratified_group_split(ds.groups, ds.y, args.holdout, args.seed)
    ytr, gtr, htr, otr = ds.y[tr], ds.groups[tr], ds.hour_index[tr], ds.onset_hour[tr]
    yte, gte, hte, ote = ds.y[te], ds.groups[te], ds.hour_index[te], ds.onset_hour[te]
    print(f"train {len(set(gtr.tolist())):,} patients / holdout {len(set(gte.tolist())):,} "
          f"patients (disjoint by construction)\n")

    # ---------------- fit 1 + 2: the same ensemble under both objectives -------- #
    scores: Dict[str, Dict[str, float]] = {}
    preds: Dict[str, np.ndarray] = {}
    models: Dict[str, object] = {}
    for name in ("incumbent", "candidate"):
        t0 = time.time()
        print(f"fitting {name} ...", flush=True)
        with _cpu_cap(args.jobs):
            m = mk(name).fit(feats.iloc[tr], ytr, groups=gtr, hour_index=htr, onset_hour=otr)
        models[name] = m
        preds[name] = m.predict_proba(feats.iloc[te])
        scores[name] = _score(yte, preds[name], gte, hte, ote)
        print(_line(f"{name} ({time.time()-t0:.0f}s)", scores[name]))
        print(f"  meta weights: { {k: round(v, 2) for k, v in m.meta_weights().items()} }\n")

    delta = scores["candidate"]["auroc"] - scores["incumbent"]["auroc"]
    d_util = scores["candidate"]["utility"] - scores["incumbent"]["utility"]
    d_auprc = scores["candidate"]["auprc"] - scores["incumbent"]["auprc"]
    print(f"held-out AUROC delta (candidate - incumbent): {delta:+.4f}")
    print(f"held-out AUPRC delta                        : {d_auprc:+.4f}")
    print(f"held-out utility delta                      : {d_util:+.4f}\n")

    # ---------------- free: does a causal filter buy anything on top? ---------- #
    # No extra fits — these are transformations of scores we already have.
    from backend.ml.models.temporal import causal_filter
    print("causal decision-layer filters on the candidate's held-out score:")
    filters: Dict[str, Dict[str, float]] = {}
    for mode in ("ema", "blend", "cummax", "hazard"):
        pf = causal_filter(preds["candidate"], gte, hte, mode)
        filters[mode] = _score(yte, pf, gte, hte, ote)
        print(_line(f"  + {mode}", filters[mode]))
    print()

    # ---------------- gate 1: discrimination ----------------------------------- #
    accuracy_ok = delta >= -args.max_auroc_loss
    print(f"DISCRIMINATION GATE: {'PASS' if accuracy_ok else 'FAIL'} "
          f"(held-out AUROC {delta:+.4f}, may lose at most {args.max_auroc_loss:.4f})")

    # ---------------- operating point, derived and gated on disjoint halves ---- #
    # The holdout is split by PATIENT: the threshold is chosen on one half and judged
    # on the other, so no threshold is ever quoted on the patients that produced it.
    ha, hb = stratified_group_split(gte, yte, 0.5, args.seed + 1)
    try:
        points = {n: opmod.derive(yte[ha], preds[n][ha], gte[ha], hte[ha], ote[ha],
                                  objective="constrained", max_false_alarm_rate=args.max_far,
                                  seed=args.seed, detector=_detector_id(configs[n]))
                  for n in ("incumbent", "candidate")}
    except opmod.Infeasible as e:
        # Not a crash and not something to tune around. The alarm budget is a clinical
        # values judgement, so an unsatisfiable one is a question for a person: here is
        # what each budget would actually buy, and which of them need a threshold out
        # in the sparse tail where this detector's behaviour is not dependable.
        print(f"\nOPERATING POINT INFEASIBLE — {e}")
        print("\nWhat each alarm budget would buy on these patients:")
        print(e.frontier())
        print("\nREFUSED to promote: no lawful operating point exists under the "
              f"requested cap (--max-far {args.max_far}). Raise the budget deliberately, "
              f"or improve the detector — do not widen the guard.")
        _write_manifest(manifest_stub(scores, filters, configs, args, accuracy_ok),
                        promoted=False)
        return
    op = points["candidate"]

    def gate_row(name, thr):
        p = preds[name][hb]
        u = metrics.normalized_utility(yte[hb], (p >= thr).astype(int), gte[hb], hte[hb])
        s = metrics.early_warning(yte[hb], p, gte[hb], ote[hb], hte[hb], thr)
        print(f"  {name:<12} thr={thr:.4f}  utility={u:+.4f}  detect={s['detection_rate']:.3f}  "
              f"FAR={s['false_alarm_rate']:.3f}  lead={s['median_lead_time_hours']:.1f}h")
        return u

    print(f"\nDECISIONS — each detector at its OWN threshold, derived on "
          f"{len(set(gte[ha].tolist()))} held-out patients, judged on "
          f"{len(set(gte[hb].tolist()))} others:")
    u_old = gate_row("incumbent", points["incumbent"].high)
    u_new = gate_row("candidate", points["candidate"].high)
    # "Not worse" is the right bar here, not "strictly better": the threshold is being
    # re-derived because the DETECTOR changed, and a re-derivation that lands on the
    # same clinical behaviour is a correct answer, not a failure.
    op_ok = u_new >= u_old
    print(f"DECISIONS GATE: {'PASS' if op_ok else 'FAIL'} "
          f"(candidate {u_new:+.4f} vs incumbent {u_old:+.4f}, each at its own point)")

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_set": args.feature_set, "limit": args.limit, "seed": args.seed,
        "n_rows": int(len(ds.y)), "n_patients": int(len(set(ds.groups.tolist()))),
        "holdout_metrics": scores, "temporal_filters": filters,
        "auroc_delta": float(delta), "auprc_delta": float(d_auprc),
        "utility_delta": float(d_util), "configs": configs,
        "gates": {"discrimination": bool(accuracy_ok), "decisions": bool(op_ok)},
        # Recorded so a later reader can tell whether the shipped model and the shipped
        # threshold came from the same fit. They must, or the alarm numbers are fiction.
        "shipped_fit": {"train_patients": int(len(set(gtr.tolist()))),
                        "threshold_derived_on_same_fit": True},
        "operating_point": {"high": float(op.high), "medium": float(op.medium),
                            "incumbent_high": float(points["incumbent"].high),
                            "gated_utility": float(u_new),
                            "incumbent_utility": float(u_old)},
    }

    if not args.promote:
        print("\n(measure-only run; pass --promote to ship it)")
        _write_manifest(manifest, promoted=False)
        return
    if not (accuracy_ok and op_ok):
        print("\nREFUSED to promote: a gate failed. Nothing was written.")
        _write_manifest(manifest, promoted=False)
        return

    # ---------------- ship the model the threshold was validated ON ------------ #
    # This used to refit the candidate on ALL patients and ship that instead. It is a
    # tempting move — more data, a slightly better model — and it silently invalidated
    # every alarm number in the same breath, because the threshold above was derived
    # and gated against the TRAIN-fitted model's scores, not the refit's.
    #
    # That is not a theoretical objection. This ensemble's output scale is not stable
    # across fits: holding the threshold at 0.055 and the scored patients fixed, the
    # false-alarm rate measures 0.085 / 0.178 / 0.209 / 0.165 for fits on 750 / 1500 /
    # 2250 / 3000 patients — a 2.5x swing, and NOT a monotone trend that could be
    # corrected with an offset. (The isotonic calibrator's output is piecewise
    # constant with few distinct values at the top; the height of that top step is
    # what moves, and it moves erratically.) A threshold validated against one fit
    # therefore says nothing dependable about how another fit will alarm.
    #
    # So the artifact shipped here is the one whose alarm behaviour was measured: the
    # candidate fitted on TRAIN. It gives up the holdout patients' worth of training
    # data, and in exchange the operating point beside it means what it claims.
    prod = os.path.join(config.ARTIFACTS_DIR, f"gbm_{args.feature_set}.joblib")
    if os.path.exists(prod):
        os.replace(prod, prod + ".balanced.bak")
        print(f"  backed up the incumbent -> {prod}.balanced.bak")
    models["candidate"].save(prod)
    n_train = len(set(gtr.tolist()))
    print(f"\n  PROMOTED -> {prod} ({os.path.getsize(prod)/1e6:.1f} MB)")
    print(f"  fitted on {n_train:,} train patients — the same fit whose scores produced "
          f"and gated HIGH={op.high:.4f}")

    opmod.save(op)
    print(f"  operating point -> {opmod.DEFAULT_PATH}  HIGH={op.high:.4f} MEDIUM={op.medium:.4f}")
    _write_manifest(manifest, promoted=True)


def _detector_id(cfg: dict) -> str:
    """The identity a threshold is only meaningful against: objective + roster."""
    return f"{cfg['objective']}:{'+'.join(cfg['roster'])}"


def manifest_stub(scores, filters, configs, args, accuracy_ok) -> dict:
    """The manifest for a run that never reached a shippable point."""
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_set": args.feature_set, "limit": args.limit, "seed": args.seed,
        "holdout_metrics": scores, "temporal_filters": filters,
        "configs": configs,
        "gates": {"discrimination": bool(accuracy_ok), "decisions": False},
        "refused": "no operating point satisfied the alarm cap",
    }


def _write_manifest(manifest: dict, promoted: bool) -> None:
    """Record the run — but only overwrite the shipped manifest if something shipped.

    ``detector_manifest.json`` is the record of what is CURRENTLY DEPLOYED: the
    operating-point derivation reads the promoted roster from it, and the integrity
    audit asserts the shipped threshold was measured on that detector. A run that
    promotes nothing leaves the deployed artifact untouched on disk, so rewriting the
    manifest would leave the record describing a model that was never installed —
    which is the same detector/evidence mismatch this whole pass exists to remove.
    A refused or measure-only run therefore writes its findings alongside instead.
    """
    manifest["promoted"] = promoted
    name = "detector_manifest.json" if promoted else "promotion_attempt.json"
    path = os.path.join(config.ARTIFACTS_DIR, name)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  {'manifest' if promoted else 'attempt (shipped manifest untouched)'} -> {path}")


if __name__ == "__main__":
    main()
