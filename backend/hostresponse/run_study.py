"""Run the host-response study: a discovery phase, then ONE locked run.

TWO PHASES, AND WHAT EACH IS ALLOWED TO TOUCH
---------------------------------------------
``--phase discovery`` may be run as often as you like. It touches the DISCOVERY
cohort's labels only. The locked cohort contributes nothing but its GENE LIST — the
symbol intersection is a property of the two platforms, not of any patient — and its
labels are never read. It reports in-cohort numbers that measure memorisation at
p >> n and are explicitly not a result, and it records which pre-registration digest
the study is being held to.

``--phase locked`` runs once. Before anything is loaded it checks that the
pre-registration is byte-for-byte the one discovery recorded, and that no locked
result already exists. Then, in ONE invocation, it computes the primary endpoint and
every guard — the controls are never available on request, separately, where they
could be consulted selectively. It always writes its artifact: a locked run that did
not save would allow an untraced look at the one number this study exists to produce.

TWO VERDICTS, SIDE BY SIDE
--------------------------
The artifact reports the verdict under the bar IN FORCE (amendment 4) and under the
FROZEN acceptance conditions, evaluated on the same measurements. Amendment 4 retired
one guard (cohort identity as a void condition) and added a stricter one (each
direction >= 0.70). Printing both means the effect of that decision is visible in the
result itself, where a reader cannot miss it, rather than only in a document's history.
``published`` follows the bar in force.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

from backend.hostresponse import prereg
from backend.hostresponse.expression import (L2_C, GeoCohort, _fit_predict, align,
                                             cohort_identity_auroc, loco,
                                             mean_direction_ci, permutation_loco)
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_folds
from backend.ml.jsonio import json_safe

ARTIFACT = os.path.join("artifacts", "hostresponse.json")


# --------------------------------------------------------------------------- #
# pure pieces — verify_hostresponse plants numbers into these
# --------------------------------------------------------------------------- #
def evaluate_verdicts(m: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the bar in force AND the frozen conditions to one set of measurements.

    Every comparison is written so a NaN FAILS it: ``nan >= x`` is False, so a
    measurement that could not be computed can never pass a condition by accident.
    Permutation failure is checked first because it means leakage, and a leaking
    study has no primary to confound or to miss.
    """
    b = prereg.bar_in_force(doc)
    f = prereg.bar_as_frozen(doc)

    lo_b, hi_b = b["label_permutation_null_mean_range"]
    in_force = {
        "mean_auroc": bool(m["primary_mean"] >= b["mean_loco_auroc_min"]),
        "ci_lower_bound": bool(m["ci_low"] > b["ci95_lower_bound_min"]),
        "each_direction": bool(m["per_direction"]) and all(
            bool(v >= b["each_direction_auroc_min"]) for v in m["per_direction"].values()),
        "permutation_null": bool(lo_b <= m["perm_mean"] <= hi_b
                                 and m["perm_draws"] >= b["label_permutation_min_draws"]),
    }
    lo_f, hi_f = f["label_permutation_auroc_range"]
    frozen = {
        "mean_auroc": bool(m["primary_mean"] >= f["mean_loco_auroc_min"]),
        "ci_lower_bound": bool(m["ci_low"] > f["ci95_lower_bound_min"]),
        "cohort_identity_below_void": bool(m["cohort_identity"] < f["cohort_identity_auroc_max"]),
        "permutation_null": bool(lo_f <= m["perm_mean"] <= hi_f),
    }

    def label(checks: Dict[str, bool], confound: str = "") -> str:
        if not checks["permutation_null"]:
            return "VOID"
        if confound and not checks[confound]:
            return "CONFOUNDED"
        return "PASSED" if all(checks.values()) else "BAR MISSED"

    v_force = label(in_force)
    v_frozen = label(frozen, confound="cohort_identity_below_void")
    return {
        "in_force": {"verdict": v_force, "checks": in_force, "bar": b},
        "under_frozen_conditions": {"verdict": v_frozen, "checks": frozen, "bar": f},
        "amendment_changed_the_outcome": v_force != v_frozen,
        "published": v_force == "PASSED",
    }


def reading(verdicts: Dict[str, Any], m: Dict[str, Any]) -> str:
    """One paragraph that says what the result does and does not license."""
    v = verdicts["in_force"]["verdict"]
    dirs = ", ".join("held-out %s %.3f" % (k, val) for k, val in sorted(m["per_direction"].items()))
    if v == "VOID":
        return ("VOID. The within-cohort label-permutation null did not centre on chance "
                "(mean %.3f), which means the pipeline can score labels it has no "
                "information about. No transfer claim of any kind is available." % m["perm_mean"])
    if v == "PASSED":
        return ("Transfer detected. A blood expression signature trained on one cohort "
                "separated infectious from sterile patients in the other, on a different "
                "array platform, in both directions (%s; mean %.3f, 95%% patient-cluster "
                "interval %.3f-%.3f). This is Link A only: it does not show that the "
                "separation would reduce this product's false-alarm rate, because no bed "
                "in this system carries transcriptomics." % (dirs, m["primary_mean"], m["ci_low"], m["ci_high"]))
    return ("Bar missed (%s; mean %.3f, 95%% patient-cluster interval %.3f-%.3f). Read "
            "this with the power statement recorded before the run: the study can detect "
            "transfer at or above a true AUROC of about 0.85 and not below it. So this is "
            "'transfer at or above ~0.85 not detected' — not 'no transfer'." %
            (dirs, m["primary_mean"], m["ci_low"], m["ci_high"]))


def aggregate_summary(al) -> Dict[str, Any]:
    """Counts only, shaped so the no-individual guard can pass on its own merits.

    ``Aligned.summary()`` nests counts under "samples"/"patients" — key names the guard
    rightly forbids, because in an artifact they would read as lists of individuals.
    The payload is reshaped; the guard is not weakened.
    """
    s = al.summary()
    return {"n_samples": s["n_samples"], "n_patients": s["n_patients"],
            "n_symbols": s["n_symbols"], "n_infectious_samples": s["n_infectious"],
            "n_sterile_samples": s["n_sterile"],
            "per_cohort": {c: {"n_samples": v["samples"], "n_patients": v["patients"]}
                           for c, v in s["per_cohort"].items()}}


def in_cohort_cv_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       k: int = 5, seed: int = 42) -> float:
    """Patient-grouped k-fold AUROC inside ONE cohort. Descriptive; never an endpoint."""
    vals: List[float] = []
    for tr, te in stratified_group_folds(groups, y, k, seed):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        vals.append(metrics.auroc(y[te], _fit_predict(X[tr], y[tr], X[te], [])))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def _cohort_specs(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = prereg.admissible(doc)
    roles = sorted(c["role"] for c in specs)
    if roles != ["discovery", "locked"]:
        raise SystemExit("this runner expects exactly one discovery and one locked cohort; "
                         "the pre-registration names %s" % roles)
    return specs


def run_discovery(doc: Dict[str, Any], args: argparse.Namespace) -> None:
    specs = _cohort_specs(doc)
    d_spec = next(c for c in specs if c["role"] == "discovery")
    l_spec = next(c for c in specs if c["role"] == "locked")

    disc = GeoCohort(d_spec["accession"], d_spec["platform"]).load()
    # The locked cohort is loaded ONLY to read which gene symbols its platform measures.
    # Its labels, patients and values are never touched in this phase.
    locked_symbols = set(GeoCohort(l_spec["accession"], l_spec["platform"]).load().X.index)
    shared = sorted(set(disc.X.index) & locked_symbols)
    X = disc.X.loc[shared].to_numpy(dtype=float).T
    y, g = disc.y, np.array(disc.patients)

    cv = in_cohort_cv_auroc(X, y, g, seed=args.seed)
    null = []
    for d in range(args.perm_draws):
        yp = y.copy()
        np.random.default_rng(10_000 + d).shuffle(yp)
        null.append(in_cohort_cv_auroc(X, yp, g, seed=20_000 + d))
    null = np.array([v for v in null if not np.isnan(v)])

    print("DISCOVERY — %s only; the locked cohort contributed its gene list and nothing else"
          % d_spec["accession"])
    print("  patients %d (infectious %d)  shared symbols %d  learner L2 C=%g"
          % (len(set(g.tolist())), int(y.sum()), len(shared), L2_C))
    print("  in-cohort CV AUROC %.3f  — memorisation-scale at p >> n; NOT a result" % cv)
    print("  in-cohort permutation null over %d draws: mean %.3f  sd %.3f"
          % (len(null), null.mean(), null.std()))

    if args.save:
        digest = prereg.record_discovery(doc, extra={
            "discovery_cohort": d_spec["accession"],
            "n_shared_symbols": len(shared),
            "learner": "LogisticRegression(L2, no scaling)", "C": L2_C,
            "in_cohort_cv_auroc": round(cv, 4),
            "in_cohort_permutation_null_mean": round(float(null.mean()), 4),
            "note": "in-cohort figures measure memorisation at p >> n and are not a result",
        })
        print("\nrecorded pre-registration digest %s -> %s" % (digest, prereg.DISCOVERY_RECORD))


def run_locked(doc: Dict[str, Any], args: argparse.Namespace) -> None:
    if not args.save:
        raise SystemExit(
            "the locked phase always records its result. A locked run that did not save "
            "would be an untraced look at the one number this study exists to produce. "
            "Re-run with --save (make hostresponse PHASE=locked does).")
    digest = prereg.require_unchanged(prereg.recorded_digest())
    if os.path.exists(ARTIFACT):
        try:
            prior = json.load(open(ARTIFACT))
        except ValueError:
            prior = {}
        if prior.get("phase") == "locked":
            raise SystemExit(
                "a locked result already exists at %s (digest %s). The locked cohort is "
                "scored once. Deleting that file to re-run is possible and deliberate — "
                "and the changed result will then be visible in version control."
                % (ARTIFACT, prior.get("prereg_digest")))

    specs = _cohort_specs(doc)
    cohorts = [GeoCohort(c["accession"], c["platform"]).load() for c in specs]
    al = align(cohorts)

    print("LOCKED — scoring %s once, against pre-registration %s"
          % ([c["accession"] for c in specs], digest))
    lo = loco(al, seed=args.seed)
    prim = mean_direction_ci(al.y, lo["pooled"], al.cohort, al.patient_ids,
                             n_boot=args.boot, seed=args.seed)
    perm = permutation_loco(al, seed=7, n_draws=args.perm_draws)
    ident = cohort_identity_auroc(al, seed=args.seed)

    # easiness calibrator: the trivial question, asked on purpose, inside the one cohort
    # that has a healthy arm. Reported to show the gap; never quotable as the result.
    calib = {}
    healthy_specs = [c for c in specs if c["accession"] == "GSE40012"]
    if healthy_specs:
        hc = GeoCohort("GSE40012", healthy_specs[0]["platform"], contrast="healthy").load()
        Xh = hc.X.to_numpy(dtype=float).T
        calib = {"cohort": "GSE40012",
                 "question": "infectious vs HEALTHY (the trivial task)",
                 "in_cohort_patient_grouped_cv_auroc":
                     in_cohort_cv_auroc(Xh, hc.y, np.array(hc.patients), seed=args.seed),
                 "note": "reported only to show how much easier the trivial task is; "
                         "never the result"}

    m = {"primary_mean": prim["point_mean"], "ci_low": prim["ci_low"], "ci_high": prim["ci_high"],
         "per_direction": prim["per_direction"], "perm_mean": perm["mean"],
         "perm_draws": perm["n_draws"], "cohort_identity": ident}
    verdicts = evaluate_verdicts(m, doc)
    text = reading(verdicts, m)

    payload = {
        "study": "host-response gene expression: infectious vs sterile systemic inflammation",
        "protocol": doc.get("protocol"),
        "phase": "locked",
        "prereg_digest": digest,
        "link_tested": "A — expression separates infection from sterile inflammation across cohort and platform",
        "link_not_tested": "B — any effect on this product's false-alarm rate; no bed carries transcriptomics",
        "fit": {"learner": "LogisticRegression (L2, no feature scaling)", "C": L2_C,
                "features": "per-sample rank-z of gene symbols, probes collapsed on ranks",
                "cohorts": [c["accession"] for c in specs]},
        "data": aggregate_summary(al),
        "primary": {"metric": "mean of per-direction leave-one-cohort-out AUROCs",
                    "mean_auroc": prim["point_mean"],
                    "per_direction_auroc_by_held_out_cohort": prim["per_direction"],
                    "ci95": [prim["ci_low"], prim["ci_high"]],
                    "ci_unit": "patient, resampled within each held-out cohort",
                    "n_boot_used": prim["n_boot_used"]},
        "controls": {"label_permutation_null": {k: perm[k] for k in ("mean", "sd", "min", "max", "n_draws")},
                     "cohort_identity_auroc": ident,
                     "easiness_calibrator": calib},
        "verdict": verdicts,
        "published": verdicts["published"],
        "reading": text,
        "power_stated_before_run": next(
            (a.get("power_stated_before_any_outcome") for a in doc.get("amendments", [])
             if a.get("n") == 4), None),
    }
    prereg.assert_no_patient_level(payload)
    os.makedirs(os.path.dirname(ARTIFACT) or ".", exist_ok=True)
    with open(ARTIFACT, "w") as fh:
        json.dump(json_safe(payload), fh, indent=2)

    print("\n  per direction:  %s" % ", ".join(
        "held-out %s %.3f" % kv for kv in sorted(prim["per_direction"].items())))
    print("  PRIMARY mean %.3f   95%% CI %.3f-%.3f (patients)"
          % (prim["point_mean"], prim["ci_low"], prim["ci_high"]))
    print("  permutation null mean %.3f (sd %.3f, %d draws)"
          % (perm["mean"], perm["sd"], perm["n_draws"]))
    print("  cohort identity AUROC %.3f (diagnostic)" % ident)
    if calib:
        print("  easiness calibrator (infectious vs HEALTHY, in-cohort) %.3f"
              % calib["in_cohort_patient_grouped_cv_auroc"])
    print("\n  verdict IN FORCE:           %s" % verdicts["in_force"]["verdict"])
    print("  verdict under FROZEN rules: %s" % verdicts["under_frozen_conditions"]["verdict"])
    print("  published: %s\n\n%s\n\nwrote %s" % (verdicts["published"], text, ARTIFACT))


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--phase", choices=("discovery", "locked"), required=True)
    a.add_argument("--save", action="store_true")
    a.add_argument("--boot", type=int, default=2000)
    a.add_argument("--perm-draws", type=int, default=20)
    a.add_argument("--seed", type=int, default=42)
    args = a.parse_args()
    doc = prereg.load()
    (run_discovery if args.phase == "discovery" else run_locked)(doc, args)


if __name__ == "__main__":
    main()
