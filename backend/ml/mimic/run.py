"""The MIMIC-IV pipeline, end to end.

    python -m backend.ml.mimic.run --source demo            # the open 100-patient demo
    python -m backend.ml.mimic.run --source bigquery --project <gcp-project> --scale full
    python -m backend.ml.mimic.run --source lake --lake <dir> --scale full   # see backend.ml.physionet

Stages: store -> (design subsample) -> Ledger -> INFORMATION-FLOW AUDIT (covert channels;
the strict tier) -> tripwires -> candidates (tier-P / T / S boosters and their GENESIS
products, a strict-P booster, the clinician-action model, the Process Twin v2 in both
regimes: records as they were -> the full score, hazard and organ outlook; transplanted
records -> the physiology score and the per-alert process attribution) -> ADMISSION
AUDIT (every model attacked on edited patients; a failure is refused) -> CROSS-EXAMINATION
of the headline models -> CARE-PROCESS TRANSPLANT -> information ledger -> label
sensitivity -> temporal validation -> trust summary (hash chain, code fingerprint, the
power plan for the primary claim). Writes ``artifacts/mimic/<tag>_report.json``.

The demo run exists to exercise every stage on real MIMIC data; with 99 subjects its
numbers are noise and the report says so. Nothing here touches the locked subjects: that
takes a pre-registration in the hash chain (``trust``), which the harness refuses when the
lock could not answer it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np

from backend.ml.genesis import learners as L
from backend.ml.genesis.harness import auroc
from backend.ml.mimic import checkpoint as CK
from backend.ml.mimic import counterfactual as CF
from backend.ml.mimic import crossexam as CX
from backend.ml.mimic import extract as E
from backend.ml.mimic import infoflow as IF
from backend.ml.mimic import ledger as LG
from backend.ml.mimic import plan as PL
from backend.ml.mimic import progress as PR
from backend.ml.mimic import trust as TR
from backend.ml.mimic import tune as TN
from backend.ml.mimic import twin as TW
from backend.ml.mimic.ledger import Candidate, FoldModels
from backend.ml.mimic.sources import BigQuerySource, CsvSource


@dataclass
class Scale:
    lgbm: Dict = field(default_factory=dict)
    early_stop: int = 150
    twin_epochs: int = 25
    twin_seeds: tuple = (0, 1, 2)
    k_donors: int = 4                   # donor schedules behind the twin's physiology score
    phys_seeds: tuple = (0,)            # seeds of the physiology twin (trained on transplanted records)
    twin_d_model: int = 128
    twin_min_steps: int = 400           # optimiser steps at least, whatever the cohort size
    keep_rate: float = 1.0
    chunk_stays: int = 0
    dtype: str = "float32"
    infoflow_rows: int = 250_000
    cpt_stays: Optional[int] = None     # stays re-scored by the transplant (None = every dev stay)
    learn_stays: Optional[int] = None   # stays the transplant RETRAINS on (None = every dev stay)
    n_boot: int = 200
    # what the run fits (the planner turns stages off to fit a machine)
    tiers: tuple = ("P", "T", "S")
    products: bool = True               # who / when boosters and the GENESIS products
    soft: bool = True                   # the soft-label booster (all 9 onset definitions)
    distill: bool = True                # the privileged-tier distilled booster (tier P student)
    twin: bool = True
    info_ledger: bool = True
    trim_septic: bool = False           # drop septic stays' hours after every onset + 3 h (causal: exact)
    tune_trials: int = 0                # hyperparameter search (0 = use ``lgbm`` as given)
    tune_rows: int = 600_000
    tune_val_folds: tuple = (0,)
    #: when the machine is short, give up negatives (keep_rate) before stages: ``plan.choose``
    prefer_full_stages: bool = False


SCALES = {
    "demo": Scale(lgbm={"learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
                        "feature_fraction": 0.5, "lambda_l2": 5.0},
                  early_stop=50, twin_epochs=8, twin_seeds=(0,), keep_rate=1.0, twin_d_model=64,
                  twin_min_steps=200, tune_trials=4),
    # GENESIS's tuned booster (docs/GENESIS.md §4); never-septic stays kept at 25% and
    # re-weighted; features built 2,000 stays at a time into a float16 map on disk.
    "full": Scale(lgbm={"learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 800,
                        "feature_fraction": 0.3, "bagging_fraction": 0.7, "lambda_l2": 10.0},
                  early_stop=150, twin_epochs=25, twin_seeds=(0, 1), phys_seeds=(0,), keep_rate=0.25,
                  chunk_stays=2000, dtype="float16", cpt_stays=3000, learn_stays=8000, n_boot=300,
                  tune_trials=8, tune_val_folds=(0, 1), info_ledger=False, trim_septic=True),
}
# A dedicated cloud machine (deploy/physionet): every never-septic stay if the planner can afford
# it, and the full stage set kept before any negatives are given up. Everything stays on dev
# folds; the lock is untouched.
#
# Tuning is 8 trials on both scales. On the first full run, 32 trials moved the dev AUROC from
# 0.8131 to 0.8163, every setting tried landed within 0.004 of the others, and the best of the
# first 8 (0.8153) already had 70% of that gain: the hours belong to data, not search.
SCALES["cloud"] = replace(SCALES["full"], keep_rate=1.0, prefer_full_stages=True)
#: a credentialed run reports beside the lake; only the egress scan copies a report into the repo
REPORT_DIR = os.environ.get("MIMIC_REPORT_DIR") or LG.OUT_DIR
PHYSIO_FAMILIES = ["vitals", "gcs", "exam", "labs", "inflammation", "urine", "sofa", "history"]
HEADLINE = ["direct_P", "stack_P", "genesis_P", "twin_s3_P", "twin_phys_P", "twin_hazard6_P", "genesis_twin_P",
            "direct_Pstrict", "stack_S", "genesis_S"]
TRANSPLANTED = ["direct_P", "stack_P", "twin_s3_P", "twin_phys_P", "genesis_twin_P", "direct_Pstrict"]
#: what each headline number means — reported beside it, never apart from it
TIER_MEANING = {"P": "physiology only (no treatments, no orders)",
                "T": "physiology and treatments",
                "S": "every input, including the culture / antibiotic orders that make up half of the label"}
SETTING_MEANING = {
    "auroc": "hourly AUROC, early-warning cohort (onset >= 4 h) — the pre-registered primary setting",
    "auroc_hourmatched": "hourly AUROC with the ICU clock matched out",
    "auroc_all_stays": "hourly AUROC over every stay, sepsis present on arrival included",
    "patient_auroc": "patient level: each stay's highest score vs whether it became septic (early-warning cohort)",
    "patient_auroc_all_stays": "patient level, every stay",
}
#: the twin's outputs as harness candidates: name -> (regime of the twin, head, mode)
TWIN_OUTPUTS = {"s3": ("record", "s3", "full"), "direct": ("record", "direct", "full"),
                "susp6": ("record", "susp6", "full"), "organ": ("record", "organ", "full"),
                "hazard6": ("record", "hazard6", "full"), "outlook": ("record", "outlook", "full"),
                "phys": ("transplanted", "s3", "physiology")}


def booster(scale: Scale, target: str = "direct", teacher: Optional[np.ndarray] = None):
    """A LightGBM candidate: ``direct`` (the label), ``who`` (ever septic), ``when`` (the
    label, inside septic stays only), ``soft`` (the share of the 9 onset definitions that
    call the hour positive — the label's own uncertainty, cross-entropy on soft targets), or
    ``distill`` — PRIVILEGED-TIER DISTILLATION: the target is half the label and half the
    out-of-fold probability of a ``teacher`` that read every tier (the orders and treatments a
    physiology model may never see), so the student learns the physiology that PRECEDES what
    the clinicians' actions reveal — through its training targets only; its inputs stay tier P
    (Vapnik's learning using privileged information, as generalised distillation, Lopez-Paz
    et al. 2016). ``teacher`` is a :class:`Teachers`: every target the fold-k student trains on
    comes from a teacher that never saw fold k (see ``pairwise_teachers`` for why ordinary
    cross-fitting is NOT enough here)."""
    def fit(led: LG.Ledger, cols, tr, fold):
        s = led.store
        septic = np.isfinite(led.onset[s.seg])
        params, metric = scale.lgbm, "auc"
        if target == "direct":
            y, rows = led.y.astype(float), tr
        elif target == "who":
            y, rows = septic.astype(float), tr
        elif target == "soft":
            y, rows = CX.soft_labels(led).astype(float), tr
            params, metric = dict(scale.lgbm) | {"objective": "cross_entropy"}, "cross_entropy"
        elif target == "distill":
            z = teacher.for_fold(fold)
            pt = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            y = np.where(np.isfinite(pt), 0.5 * led.y + 0.5 * pt, led.y).astype(float)
            rows = tr
            params, metric = dict(scale.lgbm) | {"objective": "cross_entropy"}, "cross_entropy"
        else:
            y, rows = led.y.astype(float), tr[septic[tr]]
        f = L.fit_lgb(led.F, rows, y[rows], s.seg[rows], cols=cols, params=params,
                      weight=led.design_w[rows], strat=septic[rows].astype(int),
                      early_stop=scale.early_stop, feval_metric=metric)
        return lambda view, r: f.predict(view.F, r, raw=True)
    return fit


@dataclass
class Teachers:
    """Teacher margins for a distilled student: ``by_fold[k]`` covers the rows of every dev
    fold but k, each scored by a teacher that trained on neither fold k nor the row's own fold;
    ``lock`` (the teacher's ordinary out-of-fold margins) serves the student fitted on every dev
    row for the locked subjects, whom no teacher ever saw."""
    by_fold: Dict[int, np.ndarray]
    lock: np.ndarray
    fits: int = 0

    def for_fold(self, fold: int) -> np.ndarray:
        return self.by_fold[fold] if fold >= 0 else self.lock


def pairwise_teachers(led: LG.Ledger, scale: Scale, teacher_oof: np.ndarray, tier: str = "S") -> Teachers:
    """Leak-free teacher targets for distillation, with one teacher per PAIR of dev folds.

    Ordinary cross-fitting leaks here. A fold-j row's out-of-fold teacher score comes from a
    teacher trained on every other fold — fold k included — and a high-capacity student
    trained on fold j's targets then carries fold k's labels into its fold-k predictions. On
    the demo that second-order leak was worth +0.048 AUROC of apparent gain (0.859 vs 0.811
    with nested teachers; label-only 0.770). The fix: the teacher for folds {a, b} trains on
    the OTHER folds and scores a and b — it serves the fold-a student (on b's rows) and the
    fold-b student (on a's rows). C(5, 2) = 10 teachers instead of 5 x 4 = 20."""
    from itertools import combinations
    s = led.store
    f = led.folds[s.seg]
    cols = led.columns(tier)
    fit = booster(scale, "direct")
    pairs = {}
    for a, b in combinations(range(led.n_folds), 2):
        tr = np.flatnonzero((f >= 0) & (f != a) & (f != b) & led.train_mask)
        if len(tr):
            pairs[(a, b)] = fit(led, cols, tr, -1)
    by_fold = {}
    for k in range(led.n_folds):
        z = np.full(s.n_rows, np.nan)
        for j in range(led.n_folds):
            if j != k and (min(j, k), max(j, k)) in pairs:
                rows = np.flatnonzero((f == j) & led.train_mask)
                z[rows] = pairs[(min(j, k), max(j, k))](led, rows)
        by_fold[k] = z
    return Teachers(by_fold, teacher_oof, fits=len(pairs))


def twin_predictor(tws, head: str, mode: str = "full"):
    """Re-appliable Process Twin output (seed-averaged) for any ledger view."""
    def predict(view: LG.Ledger, rows: np.ndarray) -> np.ndarray:
        stays = np.unique(view.store.seg[rows])
        allrows = TW._rows(view.store, stays)
        acc = sum(tw.predict(view, stays, (head,), mode=mode)[head] for tw in tws) / len(tws)
        pos = np.full(view.store.n_rows, -1, dtype=np.int64)
        pos[allrows] = np.arange(len(allrows))
        return acc[pos[rows]]
    return predict


def fit_twin(led: LG.Ledger, scale: Scale):
    """The Process Twin v2, fold by fold, in both regimes (records as they were -> the full
    score and every head; transplanted records only -> the physiology score): every output as
    a re-appliable FoldModels, plus the twin's own diagnostics (process gate, organ outlook)."""
    T = TW.process_targets(led)
    s = led.store
    stays = np.arange(s.n_patients)
    oof = {n: np.full(s.n_rows, np.nan) for n in TWIN_OUTPUTS}
    organ_p = np.full((s.n_rows, len(TW.ORGANS)), np.nan)
    preds = {n: {} for n in TWIN_OUTPUTS}
    gates, epochs, explain = [], [], {}
    for k in range(led.n_folds):
        tr, te = stays[(led.folds >= 0) & (led.folds != k)], stays[led.folds == k]
        if not len(te):
            continue
        tws = {regime: [TW.ProcessTwin(TW.TwinConfig(
            epochs=scale.twin_epochs, seed=sd, k_donors=scale.k_donors, d_model=scale.twin_d_model,
            min_steps=scale.twin_min_steps, regime=regime)).fit(led, T, tr) for sd in seeds]
            for regime, seeds in (("record", scale.twin_seeds), ("transplanted", scale.phys_seeds))}
        rows = TW._rows(s, te)
        full = [tw.predict(led, te, TW.HEADS) for tw in tws["record"]]
        phys = [tw.predict(led, te, ("s3",), mode="physiology")["s3"] for tw in tws["transplanted"]]
        for n, (regime, head, mode) in TWIN_OUTPUTS.items():
            oof[n][rows] = (sum(phys) / len(phys)) if mode == "physiology" else sum(f[head] for f in full) / len(full)
            preds[n][k] = twin_predictor(tws[regime], head, mode)
        rec = tws["record"]
        Xv, Xp = rec[0]._inputs(led)
        organ_p[rows] = sum(tw._forward(s, Xv, Xp, te, (), organs=True)["organ_p1plus"] for tw in rec) / len(rec)
        gates += [tw.process_gate() for tw in rec]
        epochs += [tw.best_epoch_ for tw in rec]
        ex = rec[0].explain(led, te, physiology=tws["transplanted"][0])
        for key, v in ex.items():
            explain.setdefault(key, np.full((s.n_rows,) + v.shape[1:], np.nan))[rows] = v
    fms = {n: FoldModels(f"twin_{n}_P", "P", oof[n], preds[n]) for n in TWIN_OUTPUTS}
    return fms, T, {"organ_p": organ_p, "gate": float(np.mean(gates)) if gates else float("nan"),
                    "best_epochs": epochs, "explain": explain}


def twin_diagnostics(led: LG.Ledger, fms: Dict[str, FoldModels], T, extra) -> Dict:
    """What only the twin can report: its gate factors and hazard / outlook against their own
    ground truth, the calibration of its 6-h risk, and the per-alert process attribution."""
    s = led.store
    dev = np.flatnonzero(led.folds[s.seg] >= 0)
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def vs_truth(score, key, cond=None):
        y, m = T[key]
        r = dev[(m[dev] > 0) & np.isfinite(score[dev])]
        yy = (y[r] if cond is None else cond(y[r])).astype(float)
        return auroc(yy, score[r]) if 0 < yy.sum() < len(yy) else float("nan")
    out = {"process_gate": extra["gate"], "best_epochs": extra["best_epochs"],
           "factors": {"susp6": vs_truth(fms["susp6"].oof, "susp6"), "organ": vs_truth(fms["organ"].oof, "organ")},
           "hazard_onset6_auroc": vs_truth(fms["hazard6"].oof, "onset6"),
           "outlook_total_auroc": vs_truth(fms["outlook"].oof, "outlook_total")}
    cat, om = T["outlook"]
    op = extra["organ_p"]
    out["organ_outlook_auroc"] = {}
    for o, name in enumerate(TW.ORGANS):
        r = dev[(om[dev] > 0) & np.isfinite(op[dev, o])]
        yy = (cat[r, o] >= 1).astype(float)
        out["organ_outlook_auroc"][name] = auroc(yy, op[r, o]) if 0 < yy.sum() < len(yy) else float("nan")
    y6, m6 = T["onset6"]
    r = dev[(m6[dev] > 0) & np.isfinite(fms["hazard6"].oof[dev])]
    if len(r) >= 20:
        p = sig(fms["hazard6"].oof[r])
        bins = np.array_split(np.argsort(p), 5)
        out["hazard6_calibration"] = [{"predicted": float(p[b].mean()), "observed": float(y6[r][b].mean()),
                                       "n": int(len(b))} for b in bins]
    ev = np.flatnonzero(led.eval_mask & (led.folds[s.seg] >= 0) & np.isfinite(fms["s3"].oof)
                        & np.isfinite(fms["phys"].oof))
    if len(ev):
        full, phys = fms["s3"].oof[ev], fms["phys"].oof[ev]
        alert = full >= np.quantile(full, 0.9)
        att = (full - phys)[alert]
        out["alert_attribution"] = {"alerts": int(alert.sum()), "median_logit": float(np.median(att)),
                                    "share_driven_by_process": float(np.mean(att > 0))}
    # one alert, read in the definition's own language: the riskiest pre-onset hour of an
    # early-warning septic stay (dev folds, out-of-fold twins)
    ex = extra.get("explain", {})
    if ex:
        orow = np.floor(led.onset[s.seg])
        pre = np.flatnonzero((led.folds[s.seg] >= 0) & (led.onset[s.seg] >= LG.ONSET_MIN) & (s.hour < orow)
                             & np.isfinite(ex["p_sepsis3"]))
        if len(pre):
            r = int(pre[np.argmax(ex["p_sepsis3"][pre])])
            out["example_alert"] = {
                "hours_before_onset": int(orow[r] - s.hour[r]), "icu_hour": int(s.hour[r]),
                "p_sepsis3": float(ex["p_sepsis3"][r]), "p_sepsis3_physiology": float(ex["p_sepsis3_physiology"][r]),
                "process_attribution_logit": float(ex["process_attribution"][r]),
                "p_suspicion_within_6h": float(ex["p_suspicion_6h"][r]),
                "p_organ_rise_within_6h": {o: float(ex["p_organ_rise_6h"][r, i]) for i, o in enumerate(TW.ORGANS)},
                "p_total_sofa_rise_ge2_at_6h": float(ex["p_total_rise_ge2_6h"][r]),
                "p_onset_within_6h": float(ex["p_onset_6h"][r])}
    return out


def _source(source: str, project: str, lake: str):
    if source == "demo":
        return CsvSource(E.DEMO_ROOT)
    if source == "lake":
        from backend.ml.mimic.lake import LakeSource
        return LakeSource(lake)
    return BigQuerySource(project)


def run(source: str, tag: str, scale_name: str, project: str = "", hours: float = 12.0,
        plan_only: bool = False, lake: str = "") -> Dict:
    t0 = time.time()
    scale = SCALES[scale_name]
    rep: Dict = {"source": source, "scale": scale_name,
                 "caveat": ("demo: 99 subjects — every number below is noise; this run checks that "
                            "every stage executes on real MIMIC-IV data") if source == "demo" else ""}
    try:
        store = E.load(tag)
    except FileNotFoundError:
        store = E.build_store(_source(source, project, lake))
        E.save(store, tag)
    if scale_name != "demo" or plan_only:
        # will it fit this machine? the richest plan that does, before any feature is built
        n_feat = len(LG.build_features(E.subset(store, np.arange(min(8, store.n_patients))))[1])
        os.makedirs(E.CACHE_DIR, exist_ok=True)
        memo = CK.current()                       # earlier runs on this machine: how fast it really fits
        calib = PR.calibration(os.path.join(memo.dir, PR.JOURNAL), **PR.machine()) if memo else None
        scale, rep["plan"] = PL.choose(store, n_feat, scale, PL.Budget.detect(hours=hours, path=E.CACHE_DIR),
                                       time_factor=calib["time"] if calib else 1.0)
        rep["plan"]["calibration"] = calib
        est = rep["plan"]["estimate"]
        if calib:
            print(f"calibration: this machine fitted {calib['time']:.2f}x the planner's time, over {calib['fits']} "
                  f"earlier fits", flush=True)
        print(f"plan: {rep['plan']['chosen']} | keep_rate {scale.keep_rate} | {est['rows_kept']:,} rows x "
              f"{est['features']} features | ~{est['feature_disk_gb']:.1f} GB disk, ~{est['hot_ram_gb']:.1f} GB hot + "
              f"{est['cold_ram_gb']:.1f} GB cold RAM | ~{est['hours']:.1f} h ({est['booster_fits']} booster fits, twin {est['twin_hours']:.1f} h)",
              flush=True)
        if plan_only:
            os.makedirs(REPORT_DIR, exist_ok=True)
            with open(os.path.join(REPORT_DIR, f"{tag}_plan.json"), "w") as fh:
                json.dump(rep["plan"], fh, indent=1, default=_plain)
            return rep
        if memo:
            PR.record(memo.dir, event="start", planned_fits=int(est["booster_fits"] + est["tune_fits"]),
                      planned_hours=round(est["hours"], 2), twin_hours=round(est["twin_hours"], 2),
                      time_factor=round(est["time_factor"], 3), **PR.machine())
    design_w = None
    if scale.keep_rate < 1.0:
        store, design_w = E.sample_design(store, scale.keep_rate)
    if scale.trim_septic:
        stay_w = design_w[store.starts] if design_w is not None else None
        store = E.trim_septic(store)
        design_w = stay_w[store.seg] if stay_w is not None else None
    mm = os.path.join(E.CACHE_DIR, f"{tag}_features.npy") if scale.chunk_stays else None
    led = LG.Ledger.build(store, chunk_stays=scale.chunk_stays or None, memmap_path=mm,
                          dtype=getattr(np, scale.dtype), design_w=design_w)
    rep["cohort"] = {"stays": store.n_patients, "subjects": int(len(np.unique(store.subject))),
                     "hours": store.n_rows, "features": len(led.names),
                     "features_by_tier": {t: int((led.ftier == k).sum()) for t, k in (("P", 0), ("T", 1), ("S", 2))},
                     "lock_fingerprint": LG.lock_fingerprint(store, led.folds)}

    # the firewall, measured: covert channels define the strict tier
    flow = IF.audit(led, max_rows=scale.infoflow_rows)
    led.covert = frozenset(flow["flagged"])
    rep["infoflow"] = flow
    rep["cohort"]["features_P_strict"] = int(len(led.columns("P", strict=True)))
    print(f"information flow: covert channels {flow['flagged']} -> strict P keeps "
          f"{rep['cohort']['features_P_strict']} of {len(led.columns('P'))} columns", flush=True)
    rep["tripwires"] = led.tripwires()
    print("tripwires:", {k: round(v, 3) for k, v in rep["tripwires"].items()}, flush=True)

    if scale.tune_trials:                         # dev subjects only; the incumbent is trial 0
        tn = TN.tune(led, "P", n_trials=scale.tune_trials, val_folds=scale.tune_val_folds,
                     max_train_rows=scale.tune_rows, early_stop=scale.early_stop, base=scale.lgbm)
        rep["tuning"] = tn
        scale = replace(scale, lgbm=dict(tn["best"]))
        print(f"tuning: dev AUROC {tn['incumbent_auroc']:.4f} -> {tn['best_auroc']:.4f} with {tn['best']}", flush=True)

    fms: Dict[str, FoldModels] = {}
    targets = TW.MAIN if scale.products else ("direct",)

    def fit_tier(tier: str, extra=()) -> None:
        """A tier's heads, fitted FOLD by fold with the ones that bin the same rows adjacent
        (direct, who, then the P tier's soft and distilled students; ``when`` trains inside
        septic stays only, so it goes last): one LightGBM dataset per fold serves them all."""
        heads = [(f"{t}_{tier}", Candidate(f"booster {t} {tier}", tier, booster(scale, t)))
                 for t in targets if t != "when"]
        heads += list(extra)
        heads += [(f"when_{tier}", Candidate(f"booster when {tier}", tier, booster(scale, "when")))
                  for t in targets if t == "when"]
        out = led.fit_oof_many([c for _, c in heads],
                               report=[k.startswith("direct_") or k in ("soft_P", "distill_P") for k, _ in heads])
        fms.update({k: fm for (k, _), fm in zip(heads, out)})
        if scale.products:
            fms[f"genesis_{tier}"] = LG.product(f"genesis_{tier}", [fms[f"{t}_{tier}"] for t in TW.MAIN])

    # tiers T and S first: the distilled P student's teachers are their out-of-fold scores
    for tier in [t for t in scale.tiers if t != "P"]:
        fit_tier(tier)
    extra = []
    if scale.soft:
        extra.append(("soft_P", Candidate("booster soft-label P", "P", booster(scale, "soft"))))
    tier_t = next((t for t in ("S", "T") if f"direct_{t}" in fms), None)
    if tier_t is not None and scale.distill:
        teachers = pairwise_teachers(led, scale, fms[f"direct_{tier_t}"].oof, tier=tier_t)
        extra.append(("distill_P", Candidate(f"booster distilled from tier {tier_t} P", "P",
                                             booster(scale, "distill", teacher=teachers))))
        rep["distillation"] = {"teacher": f"direct_{tier_t}", "teacher_fits": teachers.fits,
                               "design": "pairwise-held-out teachers (no second-order leak)"}
    if "P" in scale.tiers:
        fit_tier("P", extra)
    else:                                          # no P tier: the students still get fitted
        out = led.fit_oof_many([c for _, c in extra], report=[True] * len(extra))
        fms.update({k: fm for (k, _), fm in zip(extra, out)})
    fms["direct_Pstrict"] = led.fit_oof(Candidate("booster direct P-strict", "P", booster(scale), strict=True))
    fms["clinician"] = led.fit_oof(Candidate("clinician actions", "S", booster(scale),
                                             columns=lambda l: l.clinician_columns()))
    twin = T = twin_extra = None
    if scale.twin:
        twin, T, twin_extra = fit_twin(led, scale)
        for n in ("s3", "direct", "phys", "hazard6"):
            fms[f"twin_{n}_P"] = twin[n]
        if scale.products:
            fms["genesis_twin_P"] = LG.product("genesis_twin_P", [fms[f"{t}_P"] for t in TW.MAIN] + [twin["s3"]])
    # stacks: a cross-fitted logistic combination of every model a tier may use
    p_parts = [k for k in ("direct_P", "who_P", "when_P", "soft_P", "distill_P", "twin_s3_P", "twin_direct_P",
                           "twin_hazard6_P") if k in fms]
    fms["stack_P"] = LG.stack(led, "stack_P", [fms[k] for k in p_parts])
    if "S" in scale.tiers:
        s_parts = p_parts + [k for k in ("direct_S", "who_S", "when_S") if k in fms]
        fms["stack_S"] = LG.stack(led, "stack_S", [fms[k] for k in s_parts])
    rep["stacks"] = {"stack_P": p_parts, "stack_S": s_parts if "S" in scale.tiers else []}
    for k, fm in fms.items():                     # every result is reported under its key
        fm.name = k

    # admission: every model attacked on edited patients before any number is believed
    audited = [fm for k, fm in fms.items() if not k.startswith(("who_", "when_"))]
    rep["admission"] = CF.audit(led, audited)
    refused = sorted(k for k, a in rep["admission"].items() if not a["admissible"])
    print(f"admission audit: {len(audited) - len(refused)}/{len(audited)} admitted"
          + (f"; REFUSED {refused}" if refused else ""), flush=True)

    rep["models"] = {k: led.score(fm.oof) for k, fm in fms.items() if not k.startswith(("who_", "when_"))}
    for k in [k for k in rep["models"] if k != "clinician"] + ["clinician"]:
        sc = rep["models"][k]
        print(f"  {k:<16} AUROC {sc['auroc']:.4f}  hour-matched {sc['auroc_hourmatched']:.4f}  "
              f"all stays {sc.get('auroc_all_stays', float('nan')):.4f}  patient {sc.get('patient_auroc', float('nan')):.4f}",
              flush=True)
    pairs = {"stack_over_direct_P": ("stack_P", "direct_P"), "twin_adds_to_trees_P": ("genesis_twin_P", "genesis_P"),
             "tier_T_over_P": ("genesis_T", "genesis_P"), "tier_S_over_P": ("stack_S", "stack_P"),
             "declared_P_over_strict_P": ("direct_P", "direct_Pstrict"), "soft_over_direct_P": ("soft_P", "direct_P"),
             "distilled_over_direct_P": ("distill_P", "direct_P"),
             "twin_full_over_physiology": ("twin_s3_P", "twin_phys_P")}
    rep["comparisons"] = {name: led.compare(fms[a].oof, fms[b].oof, n_boot=500)
                          for name, (a, b) in pairs.items() if a in fms and b in fms}
    rep["headline"] = headline(led, fms, rep, n_boot=scale.n_boot)
    for tier, row in rep["headline"].items():
        if tier in ("P", "T", "S"):
            print(f"  best tier {tier}: {row['model']:<14} early-warning {row['auroc']:.4f}  all stays "
                  f"{row['auroc_all_stays']:.4f}  patient {row['patient_auroc']:.4f}", flush=True)
    ideal = rep["headline"]["ideal_circumstances"]
    print(f"IDEAL CIRCUMSTANCES: AUROC {ideal['value']:.4f} — {ideal['model']}, {ideal['meaning']}", flush=True)

    if twin is not None:
        # the twin's own diagnostics: factors, hazard and outlook against their ground truth
        rep["twin"] = twin_diagnostics(led, twin, T, twin_extra)
        print("twin: process gate {:.2f}; hazard 6-h AUROC {:.3f}; outlook AUROC {:.3f}; alerts driven by the care "
              "process {}".format(rep["twin"]["process_gate"], rep["twin"]["hazard_onset6_auroc"],
                                  rep["twin"]["outlook_total_auroc"],
                                  rep["twin"].get("alert_attribution", {}).get("share_driven_by_process")), flush=True)

    # the cross-examination and the transplant
    clin = fms["clinician"].oof
    heads = [k for k in HEADLINE if k in fms]
    rep["crossexam"] = {k: CX.examine(led, fms[k].oof, fms[k].tier, clin, n_boot=scale.n_boot) for k in heads}
    rep["process_physiology"] = CF.process_physiology(led, [fms[k] for k in TRANSPLANTED if k in fms],
                                                      n_stays=scale.cpt_stays, n_boot=scale.n_boot)
    rep["process_physiology_learnable"] = CF.learnable(
        led, Candidate("booster direct P", "P", booster(scale)), n_stays=scale.learn_stays, n_boot=scale.n_boot)
    lr = rep["process_physiology_learnable"]
    if "auroc_retrained" in lr:
        print("retrained in each world: " + "  ".join(f"{k} {v:.3f}" for k, v in lr["auroc_retrained"].items())
              + f"  -> learnable process information {lr['learnable_process_information']['value']:+.3f}",
              flush=True)
    rep["verdicts"] = verdicts(rep, fms)
    for k, v in rep["verdicts"].items():
        print(f"  {k:<16} " + "  ".join(f"{kk} {vv:.3f}" if isinstance(vv, float) else f"{kk} {vv}"
                                        for kk, vv in v.items()), flush=True)

    # information ledger: each physiology family's marginal value (tier P)
    if scale.info_ledger:
        base = led.score(fms["direct_P"].oof)["auroc"]
        rep["information_ledger"] = {}
        for fam in PHYSIO_FAMILIES:
            if fam in LG.STATIC_FAMILIES:
                drop_names = set(LG.STATIC_FAMILIES[fam])
                keep = np.array([j for j in led.columns("P") if led.names[j] not in drop_names])
            else:
                drop = set(LG.FAMILIES[fam])
                keep = np.array([j for j in led.columns("P") if (LG.channel_of(led.names[j]) or "") not in drop])
            fm = led.fit_oof(Candidate(f"P without {fam}", "P", booster(scale), columns=lambda l, keep=keep: keep),
                             verbose=False)
            a = led.score(fm.oof)["auroc"]
            rep["information_ledger"][fam] = {"auroc_without": a, "auroc_lost": base - a}
    rep["label_sensitivity_P"] = led.label_sensitivity(fms["stack_P"].oof)
    rep["temporal_validation"] = temporal_validation(led, scale)
    rep["trust"] = {"chain": TR.summary(), "code": TR.code_fingerprint(), "power_plan": {
        "absolute_auroc_gt_0.80": TR.power_plan(led, fms["stack_P"].oof, bar=0.80),
        "stack_over_direct_by_0.01": TR.power_plan(led, fms["stack_P"].oof, fms["direct_P"].oof, bar=0.01)}}
    rep["seconds"] = time.time() - t0
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{tag}_report.json")
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1, default=_plain)
    print(f"report: {path}  [{rep['seconds']:.0f}s]")
    return rep


def _boot_ci(led: LG.Ledger, p: np.ndarray, setting: str, n_boot: int, seed: int = 0):
    """Subject-bootstrap 95% interval of a pooled AUROC setting (design-weighted)."""
    from backend.ml.genesis.harness import Ranked
    s = led.store
    base = led.eval_mask if setting == "auroc" else led.train_mask
    rows = np.flatnonzero(base & np.isfinite(p) & (led.folds[s.seg] >= 0))
    if not len(rows) or led.y[rows].min() == led.y[rows].max():
        return (float("nan"), float("nan"))
    y, w, R = led.y[rows], led.design_w[rows], Ranked(p[rows])
    codes, inv = np.unique(s.subject[s.seg[rows]], return_inverse=True)
    rng = np.random.default_rng(seed)
    d = [R.auc(y, w * np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[inv])
         for _ in range(n_boot)]
    return tuple(float(x) for x in np.nanpercentile(d, [2.5, 97.5]))


def headline(led: LG.Ledger, fms: Dict[str, FoldModels], rep: Dict, n_boot: int = 200) -> Dict:
    """Where the number stands: the best model of each tier (by the primary, early-warning
    dev AUROC) under every setting a paper might report — and the single highest number,
    always printed with what it means (tier and setting) and never without the primary."""
    settings = list(SETTING_MEANING)
    out: Dict = {}
    for tier in ("P", "T", "S"):
        cands = [k for k, fm in fms.items() if fm.tier == tier and k in rep["models"] and k != "clinician"]
        if not cands:
            continue
        k = max(cands, key=lambda c: np.nan_to_num(rep["models"][c]["auroc"], nan=-1.0))
        sc = rep["models"][k]
        out[tier] = {"model": k, **{st: sc.get(st, float("nan")) for st in settings},
                     "ci95": {st: _boot_ci(led, fms[k].oof, st, n_boot) for st in ("auroc", "auroc_all_stays")},
                     "admitted": rep["admission"].get(k, {}).get("admissible", "n/a")}
    best = max(((out[t][st], t, st) for t in out for st in settings if np.isfinite(out[t][st])),
               default=(float("nan"), "", ""))
    out["ideal_circumstances"] = {
        "value": best[0], "tier": best[1], "setting": best[2], "model": out.get(best[1], {}).get("model", ""),
        "meaning": f"{TIER_MEANING.get(best[1], '')}; {SETTING_MEANING.get(best[2], '')}",
        "primary": {"value": out.get("P", {}).get("auroc", float("nan")),
                    "meaning": f"{TIER_MEANING['P']}; {SETTING_MEANING['auroc']}"}}
    return out


def verdicts(rep: Dict, fms: Dict[str, FoldModels]) -> Dict[str, Dict]:
    """One row per headline model: the answer to each cross-examination question."""
    out = {}
    for k in [k for k in HEADLINE if k in rep["crossexam"]]:
        cx = rep["crossexam"][k]
        cpt = rep["process_physiology"].get(k, {})
        hz = cx["horizon"]
        at6 = hz["auroc_clock_matched"][hz["horizon_h"].index(6)] if 6 in hz["horizon_h"] else float("nan")
        one = cx["alert_frontier"]["budgets"].get("1.0") or {}
        out[k] = {
            "admitted": rep["admission"].get(k, {}).get("admissible", "n/a"),
            "auroc": rep["models"].get(k, {}).get("auroc", float("nan")),
            "prediction_auroc": cx["definitional"]["model"]["prediction"],
            "detection_margin_over_oracle": cx["definitional"]["margin"]["detection"],
            "specificity_index": cx["contrasts"]["specificity_index"],
            "beyond_clinicians": cx.get("clinician_conditional", {}).get("conditional_auroc", float("nan")),
            "clock_matched_6h": at6,
            "soft_label_auroc": cx["ambiguity"]["auroc_soft"],
            "process_reliance": cpt.get("process_reliance", {}).get("value", float("nan")),
            "episodes_caught_at_1_alert_per_day": one.get("episodes_caught", float("nan")),
        }
    return out


def temporal_validation(led: LG.Ledger, scale: Scale) -> Dict:
    """Train on the earliest year groups present, test on the latest (dev subjects only)."""
    s = led.store
    groups = sorted(g for g in np.unique(s.site) if g >= 0)
    if len(groups) < 2:
        return {"skipped": "fewer than two anchor_year_groups"}
    cut = groups[len(groups) // 2 - 1] if len(groups) > 2 else groups[0]
    dev = led.folds[s.seg] >= 0
    tr = np.flatnonzero(dev & led.train_mask & (s.site[s.seg] <= cut))
    te = np.flatnonzero(dev & led.train_mask & (s.site[s.seg] > cut))
    if not len(tr) or not len(te):
        return {"skipped": "empty side"}
    p = np.full(s.n_rows, np.nan)
    p[te] = booster(scale)(led, led.columns("P"), tr, -1)(led, te)
    sc = led.score(p, rows=te[led.eval_mask[te]])
    return {"train_groups": [E.YEAR_GROUPS[g] for g in groups if g <= cut],
            "test_groups": [E.YEAR_GROUPS[g] for g in groups if g > cut], **sc}


def _plain(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return float(o)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["demo", "bigquery", "lake"], default="demo")
    ap.add_argument("--project", default="")
    ap.add_argument("--lake", default="", help="the Parquet lake (backend.ml.physionet)")
    ap.add_argument("--checkpoint", default="", help="keep every fit here, so a re-run resumes where this one stopped")
    ap.add_argument("--scale", choices=list(SCALES), default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--hours", type=float, default=12.0, help="time budget the planner fits the run into")
    ap.add_argument("--plan-only", action="store_true", help="extract, print the plan, stop")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(L.THREADS)              # GENESIS_THREADS; 6 on the laptop
    with CK.active(a.checkpoint or None, TR.code_fingerprint()["sha256"] + "|" + (a.tag or a.source)):
        run(a.source, a.tag or a.source, a.scale or ("demo" if a.source == "demo" else "full"), a.project,
            a.hours, a.plan_only, a.lake)
