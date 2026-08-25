"""Subgroup fairness + calibration audit — where the detector serves people worse.

Every accuracy number this project has published is a **cohort average**, and a
cohort average is exactly the shape of number that hides a disparity: a detector
can post AUROC 0.78 overall while being near-chance on women over 80, and nothing
in the existing evaluation would say so. ``Age`` and ``Gender`` have been in the
feature matrix since the beginning (``backend/ml/data/features.py``); this is the
first pass that reports *per subgroup* what the ward would actually experience.

Three methodological commitments, because a fairness audit is easy to do in a way
that flatters the model:

1. **The alarm threshold is global and deployed.** It comes from
   ``artifacts/operating_point.json`` — the threshold the OS actually alarms on —
   and is *never* re-tuned per subgroup. Re-tuning per subgroup measures a system
   that does not exist and launders the disparity into the threshold: the whole
   point is that one shared threshold lands differently on different people.

2. **Patients are the unit, everywhere.** Subgroups are defined per patient (age
   and sex are constant within a stay, verified below), and every confidence
   interval resamples *patients*, not rows. A row bootstrap on 40 hours per patient
   reports an interval several times too narrow and turns noise into a finding.

3. **Gaps get intervals too, and small cells are disqualified from claims.** A
   point difference between two subgroups is meaningless without knowing whether it
   survives resampling, so each gap is computed inside the same draw and reported
   with a CI. Cells under ``MIN_PATIENTS`` patients or ``MIN_SEPTIC`` septic
   patients are still printed — suppressing them would hide who we cannot speak
   about — but are flagged ``underpowered`` and excluded from gap claims.

A disparity found here is a **finding, not a failure**. The honest failure mode is
not reporting one.

    python -m backend.ml.scripts.subgroup_audit --spec dev --candidate stack+xgb@utility
    python -m backend.ml.scripts.subgroup_audit --oof artifacts/frontier/production_dev_oof.npz:stack-utility
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.ml import config
from backend.ml.eval import metrics
from backend.ml.eval.bench import Benchmark
from backend.ml.eval.utility import normalized_utility_fast, utility_terms

# A cell smaller than this cannot support a claim. Reported, but not compared.
MIN_PATIENTS = 100
MIN_SEPTIC = 10

#: Age bands. CinC de-identification caps age at 89, so the top band is open-ended
#: in name only. Boundaries are the conventional geriatric-risk cuts, fixed in
#: advance rather than chosen after seeing which split looks worst.
AGE_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 45.0, "age <45"),
    (45.0, 65.0, "age 45-64"),
    (65.0, 80.0, "age 65-79"),
    (80.0, 200.0, "age 80+"),
)
#: CinC 2019 codes Gender as 0 = female, 1 = male.
SEX_LABELS = {0.0: "female", 1.0: "male"}


# --------------------------------------------------------------------------- #
# Patient view of the dataset
# --------------------------------------------------------------------------- #
class PatientView:
    """Rows regrouped by patient, plus the per-patient facts an audit needs.

    Built once. Everything downstream (cell selection, bootstrap) indexes into
    ``rows`` rather than re-scanning the 310k-row matrix.
    """

    def __init__(self, ds):
        order = np.lexsort((ds.hour_index, ds.groups))
        g = ds.groups[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        bounds = np.r_[starts, len(g)]
        self.rows: List[np.ndarray] = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        self.gids = g[starts]
        self.n = len(self.gids)

        names = list(ds.feature_names)
        ai, gi = names.index("Age"), names.index("Gender")
        first = np.array([r[0] for r in self.rows])
        self.age = ds.X[first, ai].astype(float)
        self.sex = ds.X[first, gi].astype(float)
        self.onset = ds.onset_hour[first].astype(float)
        self.septic = self.onset >= 0

        # Age/sex must be constant within a stay for a per-patient subgroup to mean
        # anything. Verified rather than assumed — a varying value would silently
        # assign a patient's hours to two different cells.
        for idx, col in ((ai, "Age"), (gi, "Gender")):
            for i in range(0, self.n, max(self.n // 200, 1)):
                v = ds.X[self.rows[i], idx]
                if v.max() != v.min():
                    raise ValueError(f"{col} varies within patient {self.gids[i]}")

        self.ds = ds

    def alarm_facts(self, p: np.ndarray, thr: float) -> None:
        """Per-patient alarm outcomes at the fixed deployed threshold.

        Computed once because the threshold never moves: a bootstrap draw is then
        pure aggregation over these arrays instead of a re-scan of every hour.
        """
        ds = self.ds
        caught = np.zeros(self.n, dtype=bool)
        lead = np.full(self.n, np.nan)
        alarmed_any = np.zeros(self.n, dtype=bool)
        for i, r in enumerate(self.rows):
            pr, hr = p[r], ds.hour_index[r]
            fired = pr >= thr
            alarmed_any[i] = bool(fired.any())
            if self.septic[i]:
                pre = fired & (hr <= self.onset[i])
                if pre.any():
                    caught[i] = True
                    lead[i] = float(self.onset[i] - hr[pre].min())
        self.caught, self.lead, self.alarmed_any = caught, lead, alarmed_any


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
def build_cells(pv: PatientView) -> List[Tuple[str, str, np.ndarray]]:
    """``(name, kind, patient_mask)`` for the cohort, each margin, and the
    intersection. The intersection matters: a model can look even on sex and even
    on age while failing the specific corner where the two coincide."""
    cells: List[Tuple[str, str, np.ndarray]] = [("cohort", "all", np.ones(pv.n, dtype=bool))]
    sex_masks: Dict[str, np.ndarray] = {}
    for code, label in SEX_LABELS.items():
        m = pv.sex == code
        sex_masks[label] = m
        cells.append((label, "sex", m))
    age_masks: Dict[str, np.ndarray] = {}
    for lo, hi, label in AGE_BANDS:
        m = (pv.age >= lo) & (pv.age < hi)
        age_masks[label] = m
        cells.append((label, "age", m))
    for alabel, am in age_masks.items():
        for slabel, sm in sex_masks.items():
            cells.append((f"{alabel} · {slabel}", "age×sex", am & sm))
    return cells


# --------------------------------------------------------------------------- #
# Metrics for one cell
# --------------------------------------------------------------------------- #
def cell_metrics(pv: PatientView, p: np.ndarray, terms: Dict[str, np.ndarray],
                 pidx: np.ndarray, thr: float) -> Dict[str, float]:
    """Every reported number for one set of patients, at the global threshold."""
    if pidx.size == 0:
        return {}
    rows = np.concatenate([pv.rows[i] for i in pidx])
    y, pr = pv.ds.y[rows], p[rows]
    sub_terms = {k: v[rows] for k, v in terms.items() if k in ("u1", "u0")}

    septic = pv.septic[pidx]
    n_sep, n_non = int(septic.sum()), int((~septic).sum())
    leads = pv.lead[pidx][pv.caught[pidx]]

    out = {
        "n_patients": float(pidx.size),
        "n_septic_patients": float(n_sep),
        "n_hours": float(rows.size),
        "hour_prevalence": float(y.mean()),
        "patient_prevalence": float(n_sep / max(pidx.size, 1)),
        "auroc": metrics.auroc(y, pr) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": metrics.auprc(y, pr) if len(np.unique(y)) > 1 else float("nan"),
        "brier": metrics.brier(y, pr),
        "ece": metrics.expected_calibration_error(y, pr),
        # Calibration in the direction that decides who gets alarmed: if mean
        # predicted risk sits below the observed rate for a group, that group is
        # systematically under-alarmed at any shared threshold.
        "mean_predicted": float(pr.mean()),
        "observed_rate": float(y.mean()),
        "calibration_ratio": float(pr.mean() / y.mean()) if y.mean() > 0 else float("nan"),
        "utility": normalized_utility_fast(y, pr >= thr, sub_terms),
        "alarm_hour_rate": float((pr >= thr).mean()),
        "detection_rate": float(pv.caught[pidx].sum() / n_sep) if n_sep else float("nan"),
        "false_alarm_rate": float(pv.alarmed_any[pidx][~septic].sum() / n_non) if n_non else float("nan"),
        "median_lead_hours": float(np.median(leads)) if leads.size else float("nan"),
    }
    out["underpowered"] = float(pidx.size < MIN_PATIENTS or n_sep < MIN_SEPTIC)
    return out


#: The metrics worth an interval. Counts and prevalences are exact, not estimates.
CI_METRICS = ("auroc", "auprc", "ece", "calibration_ratio", "utility",
              "detection_rate", "false_alarm_rate", "median_lead_hours")


def bootstrap_cell(pv: PatientView, p: np.ndarray, terms: Dict[str, np.ndarray],
                   pidx: np.ndarray, thr: float, n_boot: int, seed: int) -> Dict[str, np.ndarray]:
    """Patient-resampled draws of every CI metric for one cell.

    Returns the raw draws (not just the interval) so that a *gap* between two cells
    can be formed draw-by-draw: the cells hold disjoint patients, so independent
    resampling within each is the correct sampling distribution for the difference.
    """
    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {k: [] for k in CI_METRICS}
    if pidx.size == 0:
        return {k: np.array([]) for k in CI_METRICS}
    rows_by = [pv.rows[i] for i in pidx]
    septic = pv.septic[pidx]
    caught, lead, alarmed = pv.caught[pidx], pv.lead[pidx], pv.alarmed_any[pidx]

    for _ in range(n_boot):
        take = rng.integers(0, pidx.size, size=pidx.size)
        rows = np.concatenate([rows_by[i] for i in take])
        y, pr = pv.ds.y[rows], p[rows]
        two_class = len(np.unique(y)) > 1
        draws["auroc"].append(metrics.auroc(y, pr) if two_class else np.nan)
        draws["auprc"].append(metrics.auprc(y, pr) if two_class else np.nan)
        draws["ece"].append(metrics.expected_calibration_error(y, pr))
        draws["calibration_ratio"].append(float(pr.mean() / y.mean()) if y.mean() > 0 else np.nan)
        draws["utility"].append(normalized_utility_fast(
            y, pr >= thr, {"u1": terms["u1"][rows], "u0": terms["u0"][rows]}))
        sd, nd = septic[take], ~septic[take]
        draws["detection_rate"].append(float(caught[take][sd].mean()) if sd.any() else np.nan)
        draws["false_alarm_rate"].append(float(alarmed[take][nd].mean()) if nd.any() else np.nan)
        ld = lead[take][caught[take]]
        draws["median_lead_hours"].append(float(np.median(ld)) if ld.size else np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in draws.items()}


def _ci(v: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(v, 100 * alpha / 2)),
            float(np.percentile(v, 100 * (1 - alpha / 2))))


def gap(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
        pa: Dict[str, float], pb: Dict[str, float],
        name_a: str, name_b: str, selected: bool = False) -> List[Dict[str, object]]:
    """Signed ``a - b`` differences with the bootstrap draws attached.

    The two cells must hold **disjoint** patients: independent resampling within
    each is only the correct sampling distribution for their difference when no
    patient sits on both sides. That is why the worst cell is compared against its
    complement and not against the cohort, which contains it.

    ``selected=True`` marks a comparison whose left-hand cell was *chosen by
    looking at the results* (the worst-scoring one). Nothing here corrects for that
    choice, so such a gap is exploratory — it says "this is the corner worth
    measuring properly", not "this disparity is established".
    """
    out: List[Dict[str, object]] = []
    for m in CI_METRICS:
        da, db = a.get(m), b.get(m)
        if da is None or db is None or da.size == 0 or db.size == 0:
            continue
        d = da - db
        lo, hi = _ci(d)
        out.append({
            "metric": m, "a": name_a, "b": name_b, "selected": selected,
            "value_a": pa.get(m, float("nan")), "value_b": pb.get(m, float("nan")),
            "delta": float(pa.get(m, np.nan) - pb.get(m, np.nan)),
            "ci_lo": lo, "ci_hi": hi,
            "significant": bool(not np.isnan(lo) and not np.isnan(hi) and (lo > 0 or hi < 0)),
            "_draws": d,
        })
    return out


def adjust(gaps: List[Dict[str, object]]) -> float:
    """Bonferroni-correct the family of gap tests, in place.

    Every metric × every comparison is a hypothesis test, and across two dozen of
    them one interval excluding zero by chance is the *expected* outcome rather
    than a finding. The plain 95% interval stays in the record because it is the
    effect size a reader wants; the significance verdict is taken at ``alpha/n``.
    Exploratory ``selected`` comparisons are held out of the family and never
    marked significant — including them would imply the cherry-pick had been
    accounted for, and it has not.
    """
    family = [g for g in gaps if not g["selected"]]
    n = max(len(family), 1)
    alpha = 0.05 / n
    for g in gaps:
        d = g.pop("_draws")
        lo, hi = _ci(d, alpha=alpha)
        g["ci_lo_adj"], g["ci_hi_adj"], g["alpha_adj"] = lo, hi, alpha
        g["significant_adj"] = bool(
            not g["selected"] and not np.isnan(lo) and not np.isnan(hi) and (lo > 0 or hi < 0))
    return alpha


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def load_oof(spec_str: str, n_rows: int) -> Tuple[np.ndarray, str]:
    """``path.npz:key`` — reuse out-of-fold scores already on disk."""
    path, _, key = spec_str.partition(":")
    d = np.load(path, allow_pickle=True)
    if not key:
        keys = list(d.files)
        raise SystemExit(f"--oof needs a key; {path} holds {keys}")
    p = np.asarray(d[key], dtype=float)
    if p.shape != (n_rows,):
        raise SystemExit(f"{path}:{key} has {p.shape} scores for {n_rows} rows — wrong spec?")
    return p, f"{os.path.basename(path)}:{key}"


def run_candidate(bench: Benchmark, name: str) -> Tuple[np.ndarray, str]:
    """Fit the named registry candidate on the fixed folds and pool its OOF."""
    from backend.ml.scripts.frontier import REGISTRY
    if name not in REGISTRY:
        raise SystemExit(f"unknown candidate {name!r}; registry has {sorted(REGISTRY)}")
    res = bench.run(name, REGISTRY[name])
    return res.oof, f"fresh:{name}"


def deployed_threshold(explicit: Optional[float]) -> Tuple[float, str]:
    if explicit is not None:
        return float(explicit), "--threshold"
    path = os.path.join(config.ARTIFACTS_DIR, "operating_point.json")
    with open(path) as fh:
        op = json.load(fh)
    return float(op["high"]), "artifacts/operating_point.json:high"


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="dev", help="bench dataset spec (dev/setA/setB/full)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--candidate", default=None,
                    help="frontier registry name to fit fresh (e.g. stack+xgb@utility)")
    ap.add_argument("--oof", default=None, help="reuse saved scores: path.npz:key")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the deployed alarm threshold (discouraged)")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=0,
                    help="cap worker threads; n_jobs alone leaves the OpenMP pools "
                         "under HistGradientBoosting unbounded, so this caps both")
    ap.add_argument("--out", default=os.path.join(config.ARTIFACTS_DIR, "subgroup_audit.json"))
    args = ap.parse_args()

    if not args.candidate and not args.oof:
        ap.error("give --candidate to fit fresh, or --oof to reuse saved scores")

    if args.jobs:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=args.jobs)
        os.environ["FRONTIER_JOBS"] = str(args.jobs)
        print(f"capped to {args.jobs} threads")

    t0 = time.time()
    bench = Benchmark.load(args.spec, n_folds=args.folds, seed=args.seed)
    print(f"dataset : {bench.describe()}")
    pv = PatientView(bench.ds)

    if args.oof:
        p, source = load_oof(args.oof, len(bench.ds.y))
    else:
        p, source = run_candidate(bench, args.candidate)
    if np.isnan(p).any():
        raise SystemExit("scores contain NaN — every row must be scored out-of-fold")
    if args.candidate:
        # A fresh ensemble fit costs minutes; keep the out-of-fold scores so the
        # next question about this detector is a second, not another refit.
        oof_path = os.path.join(config.ARTIFACTS_DIR,
                                f"subgroup_oof_{args.spec}_{args.candidate.replace('/', '_')}.npz")
        np.savez_compressed(oof_path, **{args.candidate: p})
        print(f"saved   : {oof_path}")

    thr, thr_src = deployed_threshold(args.threshold)
    print(f"scores  : {source}")
    print(f"threshold: {thr:g}  ({thr_src})  — global, never re-tuned per subgroup\n")

    pv.alarm_facts(p, thr)
    terms = utility_terms(bench.ds.y, bench.ds.groups, bench.ds.hour_index, bench.ds.onset_hour)

    cells = build_cells(pv)
    points: Dict[str, Dict[str, float]] = {}
    draws: Dict[str, Dict[str, np.ndarray]] = {}
    records: List[Dict[str, object]] = []
    for i, (name, kind, mask) in enumerate(cells):
        pidx = np.flatnonzero(mask)
        pt = cell_metrics(pv, p, terms, pidx, thr)
        points[name] = pt
        draws[name] = bootstrap_cell(pv, p, terms, pidx, thr, args.boot, args.seed + i)
        ci = {m: dict(zip(("lo", "hi"), _ci(draws[name][m]))) for m in CI_METRICS
              if draws[name][m].size}
        records.append({"name": name, "kind": kind, "metrics": pt, "ci": ci})

    # ---- readable table ---------------------------------------------------- #
    hdr = (f"{'subgroup':<22}{'pts':>6}{'sep':>5}{'AUROC':>18}"
           f"{'ECE':>8}{'cal':>7}{'detect':>9}{'FA':>8}{'lead':>7}{'util':>8}")
    print(hdr)
    print("-" * len(hdr))
    for rec in records:
        m, ci = rec["metrics"], rec["ci"]
        au = ci.get("auroc", {})
        flag = "  *underpowered" if m.get("underpowered") else ""
        print(f"{rec['name']:<22}{int(m['n_patients']):>6}{int(m['n_septic_patients']):>5}"
              f"{m['auroc']:>8.3f} [{au.get('lo', float('nan')):.3f},{au.get('hi', float('nan')):.3f}]"
              f"{m['ece']:>8.4f}{m['calibration_ratio']:>7.2f}"
              f"{m['detection_rate']:>9.3f}{m['false_alarm_rate']:>8.3f}"
              f"{m['median_lead_hours']:>7.1f}{m['utility']:>8.3f}{flag}")

    # ---- gaps -------------------------------------------------------------- #
    def ok(n: str) -> bool:
        return not points[n].get("underpowered", 1.0)

    gaps: List[Dict[str, object]] = []
    if ok("female") and ok("male"):
        gaps += gap(draws["female"], draws["male"], points["female"], points["male"],
                    "female", "male")
    age_named = [lbl for _, _, lbl in AGE_BANDS if lbl in points and ok(lbl)]
    if len(age_named) >= 2:
        gaps += gap(draws[age_named[-1]], draws[age_named[0]],
                    points[age_named[-1]], points[age_named[0]],
                    age_named[-1], age_named[0])

    # The worst-served cell against EVERYONE ELSE. Not against the cohort: the
    # cohort contains the cell, and the difference of a subset from a set that
    # includes it is not what independent resampling estimates. The complement is
    # disjoint, and "this group vs everyone else" is also the comparison a ward
    # would actually make.
    intersections = [n for n, k, _ in cells if k == "age×sex"]
    scored = [(points[n]["auroc"], n) for n in intersections
              if ok(n) and not np.isnan(points[n]["auroc"])]
    worst_name = None
    if scored:
        worst_name = min(scored)[1]
        wmask = dict((n, m) for n, _, m in cells)[worst_name]
        rest = np.flatnonzero(~wmask)
        points["everyone else"] = cell_metrics(pv, p, terms, rest, thr)
        draws["everyone else"] = bootstrap_cell(pv, p, terms, rest, thr,
                                                args.boot, args.seed + 999)
        gaps += gap(draws[worst_name], draws["everyone else"],
                    points[worst_name], points["everyone else"],
                    worst_name, "everyone else", selected=True)

    alpha = adjust(gaps)
    n_family = sum(1 for g in gaps if not g["selected"])
    print(f"\ngaps (a - b) · 95% patient-bootstrap CI · Bonferroni over "
          f"{n_family} pre-specified tests (alpha={alpha:.4f})")
    print("-" * 96)
    for g in gaps:
        if g["selected"]:
            mark = "  exploratory (cell chosen post-hoc)"
        elif g["significant_adj"]:
            mark = "  SIGNIFICANT (corrected)"
        elif g["significant"]:
            mark = "  nominal only"
        else:
            mark = ""
        print(f"  {g['metric']:<18} {g['a']:<18} - {g['b']:<14} "
              f"{g['delta']:+.4f}  [{g['ci_lo']:+.4f},{g['ci_hi']:+.4f}]{mark}")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "spec": args.spec, "folds": args.folds, "seed": args.seed,
        "scores": source, "n_boot": args.boot,
        "threshold": {"value": thr, "source": thr_src,
                      "policy": "global; never re-tuned per subgroup"},
        "min_patients": MIN_PATIENTS, "min_septic_patients": MIN_SEPTIC,
        "cohort": {"rows": int(len(bench.ds.y)), "patients": int(pv.n),
                   "hour_prevalence": float(bench.ds.y.mean())},
        "cells": records,
        "worst_cell": worst_name,
        "gaps": gaps,
        "multiplicity": {"family_size": n_family, "alpha_adjusted": alpha,
                         "method": "Bonferroni; post-hoc worst-cell gap excluded"},
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {args.out}  [{payload['seconds']}s]")


if __name__ == "__main__":
    main()
