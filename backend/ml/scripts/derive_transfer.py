"""Label-free deployment transfer, measured on two real hospitals.

    python -m backend.ml.scripts.derive_transfer --limit 6000 --jobs 4

Fits the promoted detector on the SOURCE hospital, then asks what a target hospital
could do with it given **no labels of its own**: diagnose the shift, apply only the
remedy the diagnosis licenses, and re-derive the alarm threshold.

Three patient-disjoint roles, and confusing them is the whole risk:

    source        the detector is fitted here, and only here
    target/adapt  the ONLY target rows any estimator may see, and it sees no labels
    target/locked nothing is tuned against these; every published number comes from here

Run in both directions. A method that works A->B and not B->A has fitted one
hospital's idiosyncrasy rather than found a mechanism.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.ml import config, conformal as cf, shift
from backend.ml.eval import metrics
from backend.ml.eval.bench import _load_cached
from backend.ml.jsonio import json_safe
from backend.ml.eval.splits import stratified_group_split
from backend.ml.models.boosters import UtilityAlignedBooster
from backend.ml.eval.utility import utility_terms, normalized_utility_fast

SETS = {"A": "wearable_training_setA_limitall", "B": "wearable_training_setB_limitall"}


def subsample(fm, limit: Optional[int], seed: int):
    """Keep at most ``limit`` whole patients — never half a stay."""
    if not limit:
        return fm
    pts = np.array(sorted(set(fm.groups.tolist())))
    if len(pts) <= limit:
        return fm
    keep = set(np.random.default_rng(seed).choice(pts, limit, replace=False).tolist())
    m = np.array([g in keep for g in fm.groups])
    from backend.ml.data.features import FeatureMatrix
    return FeatureMatrix(X=fm.X[m], y=fm.y[m], groups=fm.groups[m],
                         feature_names=list(fm.feature_names),
                         hour_index=fm.hour_index[m], onset_hour=fm.onset_hour[m])


def sweep_threshold(y, p, terms, grid) -> Tuple[float, float]:
    """Best (threshold, utility) on a grid — used ONLY on rows we are allowed to see.

    Uses the precomputed per-row utility terms. The obvious version calls the full
    ``normalized_utility`` once per grid point, which re-derives those terms 72 times
    over a 200k-row site and made this script time out at ten minutes.
    """
    best_t, best_u = float(grid[0]), -np.inf
    for t in grid:
        u = normalized_utility_fast(y, p >= t, terms)
        if u > best_u:
            best_t, best_u = float(t), float(u)
    return best_t, best_u


def utility_at(y, p, groups, hour_index, onset_hour, thr, terms) -> Dict[str, float]:
    st = metrics.early_warning(y, p, groups, onset_hour, hour_index, thr)
    return {"utility": float(normalized_utility_fast(y, p >= thr, terms)),
            "detection": float(st["detection_rate"]),
            "false_alarm": float(st["false_alarm_rate"]),
            "lead_h": float(st["median_lead_time_hours"])}


def boot_delta(y, pa, pb, terms, groups, thr_a, thr_b, n=300, seed=0):
    """Paired patient bootstrap of the utility difference (adapted - unadapted).

    Resamples PATIENTS, not rows — hours within a stay are far too correlated for a
    row bootstrap, which would report an interval several times too narrow. Both
    policies are scored on the SAME draw, which is what makes the interval paired and
    therefore much tighter than two independent ones.
    """
    rng = np.random.default_rng(seed)
    pts = np.array(sorted(set(groups.tolist())))
    idx_by = {g: np.flatnonzero(groups == g) for g in pts}
    fa, fb = pa >= thr_a, pb >= thr_b
    out = np.empty(n)
    for i in range(n):
        rows = np.concatenate([idx_by[pts[j]] for j in rng.integers(0, len(pts), len(pts))])
        t = {"u1": terms["u1"][rows], "u0": terms["u0"][rows]}
        out[i] = (normalized_utility_fast(y[rows], fb[rows], t)
                  - normalized_utility_fast(y[rows], fa[rows], t))
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run_direction(src: str, tgt: str, args) -> Dict[str, object]:
    S = subsample(_load_cached(SETS[src]), args.limit, args.seed)
    T = subsample(_load_cached(SETS[tgt]), args.limit, args.seed + 1)
    print(f"\n{'=' * 78}\n{src} -> {tgt}   source {len(S.y):,} hrs / "
          f"{len(set(S.groups.tolist())):,} pts | target {len(T.y):,} hrs / "
          f"{len(set(T.groups.tolist())):,} pts")

    # ---- fit on source, holding out source rows for the confusion matrix ------ #
    s_fit, s_hold = stratified_group_split(S.groups, S.y, 0.25, args.seed)
    if args.detector == "stack":
        # The PROMOTED detector. Worth switching to because the conformal coverage
        # numbers this script produces disagreed with artifacts/conformal_external.json,
        # and the detector is the main thing that differed between the two runs.
        import pandas as pd
        from backend.ml.models.stacked import StackedEnsembleDetector
        fs_ = pd.DataFrame(S.X, columns=S.feature_names)
        ft_ = pd.DataFrame(T.X, columns=T.feature_names)
        m = StackedEnsembleDetector(channels=list(config.WEARABLE_CHANNELS), use_hdc=False,
                                    objective="balanced",
                                    roster=["hgb", "hgb2", "et", "lr", "xgb@utility"],
                                    params={k: {"n_jobs": int(os.environ.get("FRONTIER_JOBS", -1))}
                                            for k in ("et", "lgbm", "xgb")})
        m.fit(fs_.iloc[s_fit], S.y[s_fit], groups=S.groups[s_fit],
              hour_index=S.hour_index[s_fit], onset_hour=S.onset_hour[s_fit])
        p_hold = m.predict_proba(fs_.iloc[s_hold])
        p_tgt = m.predict_proba(ft_)
    else:
        m = UtilityAlignedBooster(kind="xgb", objective="utility", feature_names=S.feature_names)
        m.fit(S.X[s_fit], S.y[s_fit], groups=S.groups[s_fit],
              hour_index=S.hour_index[s_fit], onset_hour=S.onset_hour[s_fit])
        p_hold = m.predict_proba(S.X[s_hold])
        p_tgt = m.predict_proba(T.X)
    print(f"  source held-out AUROC {metrics.auroc(S.y[s_hold], p_hold):.4f} | "
          f"target AUROC {metrics.auroc(T.y, p_tgt):.4f}")

    # ---- target split: adapt (unlabeled) vs locked (never tuned against) ------ #
    t_adapt, t_lock = stratified_group_split(T.groups, T.y, 0.5, args.seed)
    grid = np.unique(np.round(np.r_[np.linspace(0.005, 0.15, 59),
                                    np.linspace(0.16, 0.40, 13)], 4))

    # ---- the shipped baseline: threshold chosen on the SOURCE ---------------- #
    terms_src = utility_terms(S.y[s_hold], S.groups[s_hold], S.hour_index[s_hold],
                              S.onset_hour[s_hold])
    thr_src, _ = sweep_threshold(S.y[s_hold], p_hold, terms_src, grid)

    # ---- diagnose using target ADAPT covariates only, no labels --------------- #
    rep = shift.diagnose(S.y[s_hold], p_hold, S.X[s_hold],
                         p_tgt[t_adapt], T.X[t_adapt], thr_src,
                         y_tgt_for_report_only=T.y[t_adapt], seed=args.seed)
    print(f"  prior: source {rep.pi_source:.4f} | EM {rep.pi_target_em:.4f} | "
          f"BBSE {rep.pi_target_bbse:.4f} | true {rep.pi_target_true:.4f} "
          f"(ratio true {rep.prior_ratio_true():.2f}x, EM {rep.prior_ratio_em():.2f}x)")
    print(f"  covariate: domain AUC {rep.domain_auc:.3f} -> "
          f"{'SHIFT DETECTED' if rep.covariate_shift_detected() else 'no detectable shift'}")

    # ---- apply only what the diagnosis licenses ------------------------------ #
    # The source threshold is a threshold on a CALIBRATED POSTERIOR. Prior correction
    # puts the target probabilities back onto that same posterior scale, so the
    # threshold must stay exactly where it is. Moving it too — which the first version
    # of this script did — applies the same monotone map to both sides of `p >= thr`
    # and yields a bit-identical decision set: a guaranteed +0.0000 with a CI of
    # [0, 0], which is what the run reported until the no-op was found.
    ref = float(np.mean(p_hold))
    p_adj = shift.adjust_for_prior(p_tgt, ref, rep.pi_target_em)
    thr_adj = thr_src

    # A second, assumption-free arm. Rate matching asks only that the ward's alarm
    # BUDGET transfer: pick the target threshold whose alarm-hour rate equals the one
    # the source threshold produced at home. It needs no label-shift assumption, no
    # calibration, and no density ratio — just a quantile of the target's own scores —
    # so it is the arm to beat before any of the machinery above earns its place.
    rate_src = float(np.mean(p_hold >= thr_src))
    thr_rate = float(np.quantile(p_tgt[t_adapt], 1.0 - rate_src))

    lk = dict(y=T.y[t_lock], groups=T.groups[t_lock],
              hour_index=T.hour_index[t_lock], onset_hour=T.onset_hour[t_lock])
    terms_lk = utility_terms(lk["y"], lk["groups"], lk["hour_index"], lk["onset_hour"])
    base = utility_at(lk["y"], p_tgt[t_lock], lk["groups"], lk["hour_index"],
                      lk["onset_hour"], thr_src, terms_lk)
    adapt = utility_at(lk["y"], p_adj[t_lock], lk["groups"], lk["hour_index"],
                       lk["onset_hour"], thr_adj, terms_lk)
    rate = utility_at(lk["y"], p_tgt[t_lock], lk["groups"], lk["hour_index"],
                      lk["onset_hour"], thr_rate, terms_lk)
    # The ceiling: what a hospital WITH labels could have achieved on the locked half.
    thr_or, _ = sweep_threshold(lk["y"], p_tgt[t_lock], terms_lk, grid)
    oracle = utility_at(lk["y"], p_tgt[t_lock], lk["groups"], lk["hour_index"],
                        lk["onset_hour"], thr_or, terms_lk)

    d, lo, hi = boot_delta(lk["y"], p_tgt[t_lock], p_adj[t_lock], terms_lk,
                           lk["groups"], thr_src, thr_adj, n=args.boot, seed=args.seed)
    dr, lor, hir = boot_delta(lk["y"], p_tgt[t_lock], p_tgt[t_lock], terms_lk,
                              lk["groups"], thr_src, thr_rate, n=args.boot, seed=args.seed)

    print(f"\n  {'policy':<34}{'thr':>8}{'utility':>10}{'detect':>8}{'FA':>7}{'lead':>7}")
    for nm, t, r in (("unadapted (source threshold)", thr_src, base),
                     ("prior-corrected (EM)", thr_adj, adapt),
                     ("rate-matched (assumption-free)", thr_rate, rate),
                     ("oracle (needs target labels)", thr_or, oracle)):
        print(f"  {nm:<34}{t:>8.4f}{r['utility']:>+10.4f}{r['detection']:>8.3f}"
              f"{r['false_alarm']:>7.3f}{r['lead_h']:>7.1f}")
    gap = oracle["utility"] - base["utility"]
    rec = (adapt["utility"] - base["utility"]) / gap if abs(gap) > 1e-9 else float("nan")
    recr = (rate["utility"] - base["utility"]) / gap if abs(gap) > 1e-9 else float("nan")
    print(f"\n  prior-corrected - unadapted: {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"  {'SIGNIFICANT' if (lo > 0 or hi < 0) else 'n.s.'}   "
          f"recovers {100 * rec:5.1f}% of the reachable gap")
    print(f"  rate-matched   - unadapted: {dr:+.4f}  95% CI [{lor:+.4f}, {hir:+.4f}]"
          f"  {'SIGNIFICANT' if (lor > 0 or hir < 0) else 'n.s.'}   "
          f"recovers {100 * recr:5.1f}% of the reachable gap")
    print(f"  reachable gap = oracle {oracle['utility']:+.4f} - unadapted "
          f"{base['utility']:+.4f} = {gap:+.4f}")

    # ---- secondary pre-registered criterion: conformal coverage transfer ----- #
    # This is the remedy the diagnosis actually licenses. Prior correction needs LABEL
    # shift, which these hospitals do not have; weighted conformal needs COVARIATE
    # shift, which at domain AUC ~0.90 is exactly what they do have.
    chans = list(config.WEARABLE_CHANNELS)
    nm_s, st_s = cf.StratumSpec.features_from_matrix(S.X[s_hold], S.feature_names, chans)
    spec = cf.StratumSpec.fit(nm_s, st_s, chans)
    strata_s = spec.assign(nm_s, st_s)
    nm_t, st_t = cf.StratumSpec.features_from_matrix(T.X, T.feature_names, chans)
    strata_t = spec.assign(nm_t, st_t)

    ALPHA = 0.06
    w_src, _ = shift.covariate_weights(S.X[s_hold], T.X[t_adapt], seed=args.seed)
    plain = cf.calibrate(S.y[s_hold], p_hold, S.groups[s_hold], strata_s, alpha=ALPHA,
                         spec=spec, min_hours=0, min_patients=0)
    wtd = cf.calibrate(S.y[s_hold], p_hold, S.groups[s_hold], strata_s, alpha=ALPHA,
                       spec=spec, min_hours=0, min_patients=0, weights=w_src)
    # Ceiling: a hospital that HAD labels would calibrate on its own adapt half.
    orc = cf.calibrate(T.y[t_adapt], p_tgt[t_adapt], T.groups[t_adapt], strata_t[t_adapt],
                       alpha=ALPHA, spec=spec, min_hours=0, min_patients=0)

    cov = {}
    for nmz, b in (("plain (transferred)", plain), ("weighted", wtd), ("oracle", orc)):
        r = cf.coverage_report(b, T.y[t_lock], p_tgt[t_lock], T.groups[t_lock],
                               strata_t[t_lock])
        cov[nmz] = {"cov_pos": r["overall"]["micro_pos"], "cov_neg": r["overall"]["micro_neg"],
                    "abstain": r["abstain_rate"]}
    print(f"\n  CONFORMAL COVERAGE on locked target (alpha {ALPHA}, nominal "
          f"{1 - ALPHA:.2f})")
    print(f"  {'bands':<22}{'cov+':>9}{'cov-':>9}{'abstain':>9}{'|cov+ - nominal|':>18}")
    for nmz, c in cov.items():
        print(f"  {nmz:<22}{c['cov_pos']:>9.4f}{c['cov_neg']:>9.4f}{c['abstain']:>9.3f}"
              f"{abs(c['cov_pos'] - (1 - ALPHA)):>18.4f}")

    return {"direction": f"{src}->{tgt}", "shift": rep.to_json(), "conformal": cov,
            "thresholds": {"source": thr_src, "prior_corrected": float(thr_adj),
                           "rate_matched": thr_rate, "oracle": thr_or},
            "locked": {"unadapted": base, "prior_corrected": adapt,
                       "rate_matched": rate, "oracle": oracle},
            "prior_corrected": {"delta": d, "ci": [lo, hi],
                                "significant": bool(lo > 0 or hi < 0), "recovered": rec},
            "rate_matched": {"delta": dr, "ci": [lor, hir],
                             "significant": bool(lor > 0 or hir < 0), "recovered": recr}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=6000, help="patients per hospital")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--detector", default="xgb", choices=["xgb", "stack"],
                    help="xgb@utility (fast) or the promoted stack+xgb@utility")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(config.ARTIFACTS_DIR, "transfer.json"))
    args = ap.parse_args()
    if args.jobs:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=args.jobs)
        os.environ["FRONTIER_JOBS"] = str(args.jobs)
        print(f"capped to {args.jobs} threads")

    out = [run_direction("A", "B", args), run_direction("B", "A", args)]
    with open(args.out, "w") as fh:
        json.dump(json_safe({"generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                             "limit_per_site": args.limit, "detector": args.detector,
                             "directions": out}), fh, indent=2, allow_nan=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
