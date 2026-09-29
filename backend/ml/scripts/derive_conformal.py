"""Calibrate the conformal abstention layer on real CinC data and gate it.

    python -m backend.ml.scripts.derive_conformal --oof artifacts/subgroup_oof_dev_stack+xgb@utility.npz:stack+xgb@utility
    python -m backend.ml.scripts.derive_conformal --candidate stack+xgb@utility --jobs 6
    python -m backend.ml.scripts.derive_conformal --oof ... --save
    python -m backend.ml.scripts.derive_conformal --external --jobs 6

Three splits are in play and confusing them is the whole risk:

1. the detector's own folds, which make ``p`` out-of-fold — a score no model that
   saw the patient produced;
2. the **calibration** patients, whose scores set the conformal quantiles;
3. the **evaluation** patients, who appear in neither of the above and are the only
   rows the reported coverage is measured on.

Bin edges for the strata are fitted on (2) alone and stored, never recomputed later.
The detector roster comes from ``artifacts/detector_manifest.json`` rather than a
default in this file, because a band is a reading of one particular score
distribution — the identical trap ``derive_operating_point.py`` documents.

``--save`` refuses to write bands whose held-out coverage misses nominal, since a
guarantee that has already failed on patients we can see is not one to install.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.ml import config, conformal as cf
from backend.ml import operating_point as opmod
from backend.ml.eval import metrics
from backend.ml.eval.bench import Benchmark, concat, _load_cached
from backend.ml.eval.splits import stratified_group_split
from backend.ml.scripts.subgroup_audit import AGE_BANDS, SEX_LABELS

MANIFEST = os.path.join(config.ARTIFACTS_DIR, "detector_manifest.json")
OP_PATH = os.path.join(config.ARTIFACTS_DIR, "operating_point.json")
#: How much MISCOVERAGE the layer may deliver relative to what it requested, as a
#: ratio. A coverage claim has to be judged on the error it promised, not on the
#: distance between two numbers near 1: an absolute 0.02 tolerance waves through
#: five times the requested error at alpha=0.005 and rejects a 1.2x overrun at
#: alpha=0.10, which inverts the actual quality ordering. 1.5 means "you may miss
#: up to half again as often as you said you would".
MISCOVERAGE_OVERRUN = 1.5


def holds(rep: Dict, alpha: float) -> Tuple[bool, float]:
    """(does the level hold, observed miscoverage / requested).

    Judged on the WORST of the four coverage readings — both classes, micro and
    macro. The macro figures are included because a guarantee that only survives
    when long stays are allowed to vote repeatedly is not one to install.
    """
    o = rep["overall"]
    worst = min(o["micro_pos"], o["micro_neg"], o["macro_pos"], o["macro_neg"])
    ratio = (1.0 - worst) / alpha
    return ratio <= MISCOVERAGE_OVERRUN, ratio


def promoted() -> Tuple[str, str]:
    """(detector identity, feature set) as actually shipped — the identity includes the
    training set, so bands cut on another fit of the same roster are refused at load."""
    with open(MANIFEST) as fh:
        man = json.load(fh)
    return opmod.manifest_key(man), man.get("feature_set", "wearable")


def shipped_scores():
    """The DEPLOYED artifact's scores on the patients it was promoted without.

    Bands are cut points on one fit's score distribution, so they belong on the fit
    that serves. Cross-validated scores of the same configuration come from other fits
    (on 5,333 patients each, for the dev benchmark) — close enough while the shipped
    model was trained on 5,996, not once it was trained on 24,206."""
    import joblib
    import pandas as pd
    from backend.ml.data.features import FeatureMatrix
    from backend.ml.scripts.promote_frontier import promotion_split
    with open(MANIFEST) as fh:
        man = json.load(fh)
    model = joblib.load(os.path.join(config.ARTIFACTS_DIR,
                                     f"gbm_{man.get('feature_set', 'wearable')}.joblib"))
    fit = getattr(model, "fit_digest_", None)
    if fit != man.get("fit_digest"):
        raise SystemExit(f"the artifact on disk is fit {fit} but the manifest records "
                         f"{man.get('fit_digest')}: these are not the same detector")
    ds, _, te = promotion_split(man)
    sub = FeatureMatrix(X=ds.X[te], y=ds.y[te], groups=ds.groups[te],
                        feature_names=list(ds.feature_names),
                        hour_index=ds.hour_index[te], onset_hour=ds.onset_hour[te])
    p = np.asarray(model.predict_proba(pd.DataFrame(sub.X, columns=sub.feature_names)), float)
    n = len(set(sub.groups.tolist()))
    return sub, p, f"deployed artifact [fit {fit}] on its {n:,} promotion-holdout patients"


def channels_for(feature_set: str) -> List[str]:
    chans = list(config.WEARABLE_CHANNELS)
    if feature_set == "lab":
        chans += list(config.LAB_CHANNELS)
    return chans


def load_oof(spec_str: str, n_rows: int) -> Tuple[np.ndarray, str]:
    path, _, key = spec_str.partition(":")
    d = np.load(path, allow_pickle=True)
    if not key:
        raise SystemExit(f"--oof needs a key; {path} holds {list(d.files)}")
    p = np.asarray(d[key], dtype=float)
    if p.shape != (n_rows,):
        raise SystemExit(f"{path}:{key} has {p.shape} scores for {n_rows} rows — wrong spec?")
    return p, f"{os.path.basename(path)}:{key}"


def run_candidate(bench: Benchmark, name: str) -> Tuple[np.ndarray, str]:
    from backend.ml.scripts.frontier import REGISTRY
    if name not in REGISTRY:
        raise SystemExit(f"unknown candidate {name!r}; registry has {sorted(REGISTRY)}")
    return bench.run(name, REGISTRY[name]).oof, f"fresh:{name}"


def demographic_groupings(ds, idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Age band and sex per row — REPORTED on, never part of the taxonomy.

    The subgroup audit's constant is imported rather than re-typed so the two
    reports cannot drift into describing different bands under the same names.
    """
    names = list(ds.feature_names)
    age = ds.X[idx, names.index("Age")].astype(float)
    sex = ds.X[idx, names.index("Gender")].astype(float)
    band = np.empty(len(idx), dtype=object)
    for lo, hi, label in AGE_BANDS:
        band[(age >= lo) & (age < hi)] = label
    return {"age_band": band,
            "sex": np.array([SEX_LABELS.get(s, "unknown") for s in sex], dtype=object)}


# --------------------------------------------------------------------------- #
# The conversion table: what abstention does to the alarm that exists today
# --------------------------------------------------------------------------- #
def policy_row(name: str, fire: np.ndarray, y, groups, hour_index, onset_hour) -> Dict[str, float]:
    f = fire.astype(float)
    st = metrics.early_warning(y, f, groups, onset_hour, hour_index, 0.5)
    return {"policy": name,
            "utility": float(metrics.normalized_utility(y, fire.astype(int), groups, hour_index)),
            "detection": float(st["detection_rate"]),
            "false_alarm": float(st["false_alarm_rate"]),
            "lead_h": float(st["median_lead_time_hours"]),
            "alarm_hours": float(fire.mean())}


def conversion_table(bands, y, p, groups, hour_index, onset_hour, strata,
                     thr: float) -> Tuple[str, List[Dict[str, float]]]:
    """What the abstention state does to the shipped alarm, bracketed both ways.

    UNCERTAIN is not a decision — it is a refusal to decide, and what a ward does
    with a refusal is a clinical values judgement, not something this script can
    settle. So both readings are reported and the truth is somewhere between them:
    silence it (a strictly quieter system that can only lose detection) or escalate
    it (a strictly louder one that can only lose specificity).
    """
    states = cf.predict_states(bands, p, strata)
    alarm = p >= thr
    rows = [
        policy_row("shipped (p >= thr)", alarm, y, groups, hour_index, onset_hour),
        policy_row("UNCERTAIN silenced", alarm & (states == "septic"),
                   y, groups, hour_index, onset_hour),
        policy_row("UNCERTAIN escalated", alarm | (states == "uncertain"),
                   y, groups, hour_index, onset_hour),
    ]
    lines = [f"{'policy':<22}{'utility':>9}{'detect':>8}{'FA':>7}{'lead(h)':>9}{'alarm-hrs':>11}"]
    for r in rows:
        lines.append(f"{r['policy']:<22}{r['utility']:>+9.4f}{r['detection']:>8.3f}"
                     f"{r['false_alarm']:>7.3f}{r['lead_h']:>9.1f}{r['alarm_hours']:>11.4f}")
    cross = [f"\n  of the {int(alarm.sum()):,} alarming hours: "
             + ", ".join(f"{s} {int(((states == s) & alarm).sum()):,}"
                         for s in ("septic", "uncertain", "not_septic", "conflicting")),
             f"  of the {int((~alarm).sum()):,} silent hours:  "
             + ", ".join(f"{s} {int(((states == s) & ~alarm).sum()):,}"
                         for s in ("not_septic", "uncertain", "septic", "conflicting"))]
    return "\n".join(lines) + "\n" + "\n".join(cross), rows


def alpha_sweep(ds, p, chans, seed, min_hours, min_patients, detector,
                cal_idx, eval_idx, alphas) -> str:
    """Measured coverage at each requested level — the table alpha is chosen from.

    alpha is a request, not a result. On exchangeable data the two coincide and this
    table is redundant; on ICU hours they do not, and picking alpha without looking
    at what it actually delivers is the same mistake as the hand-set alarm threshold
    this repo already fixed once. Every row is measured on the SAME held-out
    patients, so the column is comparable down its length.
    """
    lines = [f"  {'alpha':>7}{'nominal':>9}{'cov+':>9}{'cov-':>9}{'macro+':>9}"
             f"{'abstain':>9}{'miss/req':>10}{'holds?':>8}"]
    holding = []
    for a in alphas:
        _, _, rep, _, _ = derive(ds, p, a, chans, seed, min_hours, min_patients,
                                 detector, cal_idx, eval_idx)
        o = rep["overall"]
        ok, ratio = holds(rep, a)
        if ok:
            holding.append(a)
        lines.append(f"  {a:>7.3f}{1 - a:>9.3f}{o['micro_pos']:>9.4f}{o['micro_neg']:>9.4f}"
                     f"{o['macro_pos']:>9.4f}{rep['abstain_rate']:>9.3f}{ratio:>10.2f}"
                     f"{'yes' if ok else 'NO':>8}")
    # The SMALLEST holding alpha — the strongest claim the data actually supports.
    # Reaching for the largest instead would be optimising the wrong direction: a
    # weaker guarantee is easier to keep, and at the limit alpha=0.5 holds trivially
    # while saying nothing. The abstention rate that comes with the chosen level is
    # a finding about the detector, not a dial to turn.
    return "\n".join(lines), (min(holding) if holding else None)


def coverage_lines(rep: Dict) -> str:
    o = rep["overall"]
    out = [f"  nominal {rep['nominal']:.2f}   n={rep['n_hours']:,} hours / "
           f"{rep['n_patients']:,} patients   abstain rate {rep['abstain_rate']:.3f}",
           f"  {'':<14}{'micro':>9}{'macro':>9}   (macro = one vote per patient)",
           f"  {'septic hours':<14}{o['micro_pos']:>9.4f}{o['macro_pos']:>9.4f}",
           f"  {'non-septic':<14}{o['micro_neg']:>9.4f}{o['macro_neg']:>9.4f}",
           f"  {'all hours':<14}{o['micro']:>9.4f}{o['macro']:>9.4f}",
           "", f"  {'stratum':>10}{'hours':>10}{'cov+':>9}{'cov-':>9}{'abstain':>9}"]
    for s, d in sorted(rep["per_stratum"].items()):
        flag = "  (unseen at calibration)" if d.get("unseen") else ""
        out.append(f"  {s:>10}{d['n_hours']:>10,}{d['micro_pos']:>9.4f}"
                   f"{d['micro_neg']:>9.4f}{d['abstain_rate']:>9.3f}{flag}")
    for name, cells in rep.get("reported_only", {}).items():
        out.append(f"\n  reported only (never in the taxonomy) — {name}")
        for v, d in sorted(cells.items()):
            out.append(f"  {v:>14}{d['n_hours']:>10,}{d['micro_pos']:>9.4f}"
                       f"{d['micro_neg']:>9.4f}{d['abstain_rate']:>9.3f}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def three_way(groups, y, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patients split into calibrate / select / locked, none straddling.

    Three sets, because there are three distinct jobs and merging any two of them
    launders a choice into the evidence for it. The quantiles are FIT on
    ``calibrate``. ``alpha`` — which is a choice, since measured coverage does not
    track the requested level on this data — is CHOSEN on ``select``. The number
    that gets published, and that the save gate reads, comes from ``locked``, which
    nothing has been tuned against. ``search_alarm_policy.py`` splits the same way
    for the same reason.
    """
    cal, rest = stratified_group_split(groups, y, 0.5, seed)
    sel_rel, lock_rel = stratified_group_split(groups[rest], y[rest], 0.5, seed + 1)
    return cal, rest[sel_rel], rest[lock_rel]     # 50 / 25 / 25 by patient


def derive(ds, p: np.ndarray, alpha: float, channels: List[str], seed: int,
           min_hours: int, min_patients: int, detector: str,
           cal_idx: Optional[np.ndarray] = None, eval_idx: Optional[np.ndarray] = None):
    """Fit edges + quantiles on calibration patients, measure on evaluation ones."""
    if cal_idx is None or eval_idx is None:
        cal_idx, eval_idx = stratified_group_split(ds.groups, ds.y, 0.5, seed)
    n_meas, stale = cf.StratumSpec.features_from_matrix(ds.X, ds.feature_names, channels)
    spec = cf.StratumSpec.fit(n_meas[cal_idx], stale[cal_idx], channels)
    strata = spec.assign(n_meas, stale)
    bands = cf.calibrate(ds.y[cal_idx], p[cal_idx], ds.groups[cal_idx], strata[cal_idx],
                         alpha=alpha, min_hours=min_hours, min_patients=min_patients,
                         spec=spec, detector=detector)
    rep = cf.coverage_report(bands, ds.y[eval_idx], p[eval_idx], ds.groups[eval_idx],
                             strata[eval_idx], demographic_groupings(ds, eval_idx))
    return bands, strata, rep, cal_idx, eval_idx


def external(args) -> None:
    """Calibrate on hospital A, serve hospital B. The guarantee is expected to fail.

    Exchangeability is what conformal prediction trades on, and a different hospital
    is the textbook violation of it: different population, equipment and charting.
    The number worth having is therefore not the nominal 1-α but how far coverage
    actually falls when the assumption breaks — which is the same validation gap
    every other cross-site result in this repo is measured against.
    """
    from backend.ml.scripts.frontier import REGISTRY
    detector, feature_set = promoted()
    detector = opmod.config_of(detector)          # its own fit, never the served one
    chans = channels_for(feature_set)
    A = _load_cached("wearable_training_setA_limitall")
    B = _load_cached("wearable_training_setB_limitall")
    A, B = _subsample(A, args.external_limit, args.seed), _subsample(B, args.external_limit, args.seed)
    ds = concat([A, B])
    is_a = np.r_[np.ones(len(A.y), bool), np.zeros(len(B.y), bool)]
    a_idx, b_idx = np.flatnonzero(is_a), np.flatnonzero(~is_a)
    print(f"setA {len(A.y):,} rows / {len(set(A.groups.tolist())):,} patients")
    print(f"setB {len(B.y):,} rows / {len(set(B.groups.tolist())):,} patients")

    cand = REGISTRY[args.candidate or "stack+xgb@utility"]
    # Calibration scores must be out-of-sample WITHIN A, so A is split again.
    a_fit, a_cal = stratified_group_split(ds.groups[a_idx], ds.y[a_idx], 0.4, args.seed)
    a_fit, a_cal = a_idx[a_fit], a_idx[a_cal]
    # ONE fit, scoring both the calibration rows and the target. The previous version
    # fitted twice — A-train -> A-calibration, then all-of-A -> B — so the bands were
    # calibrated on one model's score distribution and applied to a DIFFERENT model's
    # scores. That is the threshold/fit-coupling failure this repo has now hit three
    # times, and it is what produced the reported 0.929 -> 0.739 cross-hospital
    # "coverage collapse": an artifact of the mismatch, not a property of the transfer.
    # Scoring B from a model fitted on 60% of A rather than 100% is the cheaper error.
    print("fit 1/1: A-train -> A-calibration and -> B (same model) ...", flush=True)
    p = np.full(len(ds.y), np.nan)
    both = np.concatenate([a_cal, b_idx])
    p[both] = cand(ds, a_fit, both)

    n_meas, stale = cf.StratumSpec.features_from_matrix(ds.X, ds.feature_names, chans)
    spec = cf.StratumSpec.fit(n_meas[a_cal], stale[a_cal], chans)
    strata = spec.assign(n_meas, stale)
    bands = cf.calibrate(ds.y[a_cal], p[a_cal], ds.groups[a_cal], strata[a_cal],
                         alpha=args.alpha, min_hours=args.min_hours,
                         min_patients=args.min_patients, spec=spec, detector=detector)
    within = cf.coverage_report(bands, ds.y[a_cal], p[a_cal], ds.groups[a_cal], strata[a_cal])
    across = cf.coverage_report(bands, ds.y[b_idx], p[b_idx], ds.groups[b_idx],
                                strata[b_idx], demographic_groupings(ds, b_idx))
    print("\n" + cf.summary_table(bands))
    print("\nWITHIN setA (calibration patients — optimistic by construction)")
    print(coverage_lines(within))
    print("\nACROSS to setB (a hospital the bands never saw)")
    print(coverage_lines(across))
    d_pos = across["overall"]["micro_pos"] - within["overall"]["micro_pos"]
    d_neg = across["overall"]["micro_neg"] - within["overall"]["micro_neg"]
    print(f"\ncoverage change A -> B:  septic {d_pos:+.4f}   non-septic {d_neg:+.4f}")
    out = os.path.join(config.ARTIFACTS_DIR, "conformal_external.json")
    with open(out, "w") as fh:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                   "detector": detector, "alpha": args.alpha,
                   "external_limit": args.external_limit,
                   "within_setA": within, "across_setB": across,
                   "delta_micro_pos": d_pos, "delta_micro_neg": d_neg}, fh, indent=2)
    print(f"\nwrote {out}")


def _subsample(fm, limit: Optional[int], seed: int):
    """Keep at most ``limit`` whole patients. Two full-set stack fits are expensive;
    the subsample is by PATIENT so no stay is ever cut in half."""
    if not limit:
        return fm
    pts = np.array(sorted(set(fm.groups.tolist())))
    if len(pts) <= limit:
        return fm
    keep = set(np.random.default_rng(seed).choice(pts, size=limit, replace=False).tolist())
    m = np.array([g in keep for g in fm.groups])
    from backend.ml.data.features import FeatureMatrix
    return FeatureMatrix(X=fm.X[m], y=fm.y[m], groups=fm.groups[m],
                         feature_names=list(fm.feature_names),
                         hour_index=fm.hour_index[m], onset_hour=fm.onset_hour[m])


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="dev")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--oof", default=None, help="reuse saved scores: path.npz:key")
    ap.add_argument("--min-hours", type=int, default=cf.MIN_CELL_HOURS)
    ap.add_argument("--min-patients", type=int, default=cf.MIN_CELL_PATIENTS)
    ap.add_argument("--auto-alpha", action="store_true",
                    help="choose alpha on the select split as the largest level whose "
                         "coverage actually holds, then confirm it on the locked split")
    ap.add_argument("--sweep", default="",
                    help="comma-separated alphas to measure coverage at, e.g. 0.01,0.02,0.05")
    ap.add_argument("--external", action="store_true", help="calibrate on setA, serve setB")
    ap.add_argument("--external-limit", type=int, default=6000, help="patients per set")
    ap.add_argument("--jobs", type=int, default=0,
                    help="cap worker threads; n_jobs alone leaves the OpenMP pools "
                         "under HistGradientBoosting unbounded, so this caps both")
    ap.add_argument("--shipped", action="store_true",
                    help="calibrate on the deployed artifact's own scores for the patients "
                         "it was promoted without, instead of cross-validated scores")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default=cf.DEFAULT_PATH)
    args = ap.parse_args()

    if args.jobs:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=args.jobs)
        os.environ["FRONTIER_JOBS"] = str(args.jobs)
        print(f"capped to {args.jobs} threads")

    if args.external:
        external(args)
        return
    if not (args.candidate or args.oof or args.shipped):
        ap.error("give --shipped to calibrate the deployed artifact, --candidate to fit "
                 "fresh, or --oof to reuse saved scores")

    detector, feature_set = promoted()
    if not args.shipped:
        # Cross-validated scores come from fits of the configuration, not from the fit
        # that serves, so the bands carry the configuration's key and nothing more:
        # stamping them with the served fit's key would launder exactly the cross-fit
        # read that key exists to refuse.
        detector = opmod.config_of(detector)
    print(f"detector: {detector}  (artifacts/detector_manifest.json)")
    if args.shipped:
        ds, p, source = shipped_scores()
        print(f"dataset : {len(ds.y):,} rows · {len(set(ds.groups.tolist())):,} patients · "
              f"hour-prevalence {ds.y.mean():.4f}")
    else:
        bench = Benchmark.load(args.spec, n_folds=args.folds, seed=args.seed)
        ds = bench.ds
        print(f"dataset : {bench.describe()}")
        p, source = (load_oof(args.oof, len(ds.y)) if args.oof
                     else run_candidate(bench, args.candidate))
    if np.isnan(p).any():
        raise SystemExit("scores contain NaN — every row must be scored out-of-fold")
    print(f"scores  : {source}")

    chans = channels_for(feature_set)
    cal_idx, sel_idx, lock_idx = three_way(ds.groups, ds.y, args.seed)
    for label, idx in (("calibrate", cal_idx), ("select", sel_idx), ("locked", lock_idx)):
        print(f"{label:>10} : {len(idx):>7,} hours / "
              f"{len(set(ds.groups[idx].tolist())):>5,} patients")

    alpha = args.alpha
    sweep_txt = None
    if args.sweep or args.auto_alpha:
        grid = [float(x) for x in (args.sweep or "0.005,0.01,0.02,0.03,0.05,0.10").split(",")]
        sweep_txt, best = alpha_sweep(ds, p, chans, args.seed, args.min_hours,
                                      args.min_patients, detector, cal_idx, sel_idx, grid)
        print("\nALPHA SWEEP on the SELECT patients — what each level actually delivers")
        print(sweep_txt)
        if args.auto_alpha:
            if best is None:
                raise SystemExit("no alpha on the grid delivers its own coverage; "
                                 "this detector cannot support an abstention layer "
                                 "at any of the requested levels")
            alpha = best
            print(f"\nchosen alpha = {alpha:g} — the STRONGEST level whose miscoverage "
                  f"stays within {MISCOVERAGE_OVERRUN}x of what it asked for on SELECT.")
            if alpha == min(grid):
                print("  note: that is the smallest alpha on the grid, so the grid may "
                      "not reach the strongest level this data supports — widen it.")

    bands, strata, rep, _, _ = derive(ds, p, alpha, chans, args.seed, args.min_hours,
                                      args.min_patients, detector, cal_idx, lock_idx)
    bands.coverage = {"locked": rep, "alpha_sweep": sweep_txt,
                      "alpha_selected_on": "select split" if args.auto_alpha else "--alpha"}

    print("\n" + cf.summary_table(bands))
    print("\nCOVERAGE on the LOCKED patients (nothing was tuned against them)")
    print(coverage_lines(rep))
    eval_idx = lock_idx

    with open(OP_PATH) as fh:
        thr = float(json.load(fh)["high"])
    print(f"\nCONVERSION at the shipped threshold {thr:g} "
          f"(artifacts/operating_point.json:high)")
    table, rows = conversion_table(bands, ds.y[eval_idx], p[eval_idx], ds.groups[eval_idx],
                                   ds.hour_index[eval_idx], ds.onset_hour[eval_idx],
                                   strata[eval_idx], thr)
    print(table)
    bands.coverage["conversion"] = {"threshold": thr, "policies": rows}

    ok, ratio = holds(rep, alpha)
    o = rep["overall"]
    worst = min(o["micro_pos"], o["micro_neg"], o["macro_pos"], o["macro_neg"])
    bands.coverage["gate"] = {"worst_coverage": worst, "miscoverage_ratio": ratio,
                              "overrun_limit": MISCOVERAGE_OVERRUN, "passed": bool(ok)}
    print(f"\ngate: worst LOCKED coverage {worst:.4f} vs nominal {1 - alpha:.3f} — "
          f"missed {1 - worst:.4f} against a requested {alpha:g} "
          f"({ratio:.2f}x, limit {MISCOVERAGE_OVERRUN}x)")
    if not args.save:
        print("not saved (pass --save to install)")
        return
    if detector != promoted()[0]:
        raise SystemExit(f"refused to save: these bands are cut on '{detector}', but the "
                         f"served detector is '{promoted()[0]}'. Serving would refuse them; "
                         f"calibrate the deployed fit with --shipped")
    if not ok:
        raise SystemExit("refused to save: coverage already misses nominal on patients "
                         "we can see; installing it would ship a guarantee we have "
                         "measured to be false")
    cf.save(bands, args.out)
    print(f"saved   : {args.out}")


if __name__ == "__main__":
    main()
