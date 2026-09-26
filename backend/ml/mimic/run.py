"""The MIMIC-IV pipeline, end to end.

    python -m backend.ml.mimic.run --source demo            # the open 100-patient demo
    python -m backend.ml.mimic.run --source bigquery --project <gcp-project> --scale full

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
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from backend.ml.genesis import learners as L
from backend.ml.genesis.harness import auroc
from backend.ml.mimic import counterfactual as CF
from backend.ml.mimic import crossexam as CX
from backend.ml.mimic import extract as E
from backend.ml.mimic import infoflow as IF
from backend.ml.mimic import ledger as LG
from backend.ml.mimic import trust as TR
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


SCALES = {
    "demo": Scale(lgbm={"learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
                        "feature_fraction": 0.5, "lambda_l2": 5.0},
                  early_stop=50, twin_epochs=8, twin_seeds=(0,), keep_rate=1.0, twin_d_model=64,
                  twin_min_steps=200),
    # GENESIS's tuned booster (docs/GENESIS.md §4); never-septic stays kept at 25% and
    # re-weighted; features built 2,000 stays at a time into a float16 map on disk.
    "full": Scale(lgbm={"learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 800,
                        "feature_fraction": 0.3, "bagging_fraction": 0.7, "lambda_l2": 10.0},
                  early_stop=150, twin_epochs=25, twin_seeds=(0, 1), phys_seeds=(0,), keep_rate=0.25,
                  chunk_stays=2000, dtype="float16", cpt_stays=3000, learn_stays=8000, n_boot=300),
}
PHYSIO_FAMILIES = ["vitals", "gcs", "labs", "urine", "sofa"]
HEADLINE = ["direct_P", "genesis_P", "twin_s3_P", "twin_phys_P", "twin_hazard6_P", "genesis_twin_P",
            "direct_Pstrict", "genesis_S"]
TRANSPLANTED = ["direct_P", "genesis_P", "twin_s3_P", "twin_phys_P", "genesis_twin_P", "direct_Pstrict"]
#: the twin's outputs as harness candidates: name -> (regime of the twin, head, mode)
TWIN_OUTPUTS = {"s3": ("record", "s3", "full"), "direct": ("record", "direct", "full"),
                "susp6": ("record", "susp6", "full"), "organ": ("record", "organ", "full"),
                "hazard6": ("record", "hazard6", "full"), "outlook": ("record", "outlook", "full"),
                "phys": ("transplanted", "s3", "physiology")}


def booster(scale: Scale, target: str = "direct"):
    """A LightGBM candidate: ``direct`` (the label), ``who`` (ever septic) or ``when`` (the
    label, inside septic stays only)."""
    def fit(led: LG.Ledger, cols, tr, fold):
        s = led.store
        septic = np.isfinite(led.onset[s.seg])
        if target == "direct":
            y, rows = led.y.astype(float), tr
        elif target == "who":
            y, rows = septic.astype(float), tr
        else:
            y, rows = led.y.astype(float), tr[septic[tr]]
        f = L.fit_lgb(led.F, rows, y[rows], s.seg[rows], cols=cols, params=scale.lgbm,
                      weight=led.design_w[rows], strat=septic[rows].astype(int),
                      early_stop=scale.early_stop)
        return lambda view, r: f.predict(view.F, r, raw=True)
    return fit


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


def run(source: str, tag: str, scale_name: str, project: str = "") -> Dict:
    t0 = time.time()
    scale = SCALES[scale_name]
    rep: Dict = {"source": source, "scale": scale_name,
                 "caveat": ("demo: 99 subjects — every number below is noise; this run checks that "
                            "every stage executes on real MIMIC-IV data") if source == "demo" else ""}
    try:
        store = E.load(tag)
    except FileNotFoundError:
        src = CsvSource(E.DEMO_ROOT) if source == "demo" else BigQuerySource(project)
        store = E.build_store(src)
        E.save(store, tag)
    design_w = None
    if scale.keep_rate < 1.0:
        store, design_w = E.sample_design(store, scale.keep_rate)
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

    fms: Dict[str, FoldModels] = {}
    for tier in ("P", "T", "S"):
        for tgt in TW.MAIN:
            fms[f"{tgt}_{tier}"] = led.fit_oof(Candidate(f"booster {tgt} {tier}", tier, booster(scale, tgt)),
                                               verbose=tgt == "direct")
        fms[f"genesis_{tier}"] = LG.product(f"genesis_{tier}", [fms[f"{t}_{tier}"] for t in TW.MAIN])
    fms["direct_Pstrict"] = led.fit_oof(Candidate("booster direct P-strict", "P", booster(scale), strict=True))
    fms["clinician"] = led.fit_oof(Candidate("clinician actions", "S", booster(scale),
                                             columns=lambda l: l.clinician_columns()))
    twin, T, twin_extra = fit_twin(led, scale)
    for n in ("s3", "direct", "phys", "hazard6"):
        fms[f"twin_{n}_P"] = twin[n]
    fms["genesis_twin_P"] = LG.product("genesis_twin_P", [fms[f"{t}_P"] for t in TW.MAIN] + [twin["s3"]])
    for k, fm in fms.items():                     # every result is reported under its key
        fm.name = k

    # admission: every model attacked on edited patients before any number is believed
    audited = [fm for k, fm in fms.items() if not k.startswith(("who_", "when_"))]
    rep["admission"] = CF.audit(led, audited)
    refused = sorted(k for k, a in rep["admission"].items() if not a["admissible"])
    print(f"admission audit: {len(audited) - len(refused)}/{len(audited)} admitted"
          + (f"; REFUSED {refused}" if refused else ""), flush=True)

    rep["models"] = {k: led.score(fm.oof) for k, fm in fms.items() if not k.startswith(("who_", "when_"))}
    for k in ("twin_s3_P", "twin_phys_P", "twin_hazard6_P", "genesis_P", "genesis_twin_P", "direct_Pstrict",
              "clinician", "genesis_S"):
        sc = led.score(fms[k].oof)
        print(f"  {k:<16} AUROC {sc['auroc']:.4f}  hour-matched {sc['auroc_hourmatched']:.4f}  "
              f"pre-suspicion {sc['auroc_presusp']:.4f}", flush=True)
    rep["comparisons"] = {
        "twin_adds_to_trees_P": led.compare(fms["genesis_twin_P"].oof, fms["genesis_P"].oof, n_boot=500),
        "tier_T_over_P": led.compare(fms["genesis_T"].oof, fms["genesis_P"].oof, n_boot=500),
        "tier_S_over_P": led.compare(fms["genesis_S"].oof, fms["genesis_P"].oof, n_boot=500),
        "declared_P_over_strict_P": led.compare(fms["direct_P"].oof, fms["direct_Pstrict"].oof, n_boot=500),
    }

    # the twin's own diagnostics: factors, hazard and outlook against their ground truth
    rep["twin"] = twin_diagnostics(led, twin, T, twin_extra)
    rep["comparisons"]["twin_full_over_physiology"] = led.compare(fms["twin_s3_P"].oof, fms["twin_phys_P"].oof,
                                                                  n_boot=500)
    print("twin: process gate {:.2f}; hazard 6-h AUROC {:.3f}; outlook AUROC {:.3f}; alerts driven by the care "
          "process {}".format(rep["twin"]["process_gate"], rep["twin"]["hazard_onset6_auroc"],
                              rep["twin"]["outlook_total_auroc"],
                              rep["twin"].get("alert_attribution", {}).get("share_driven_by_process")), flush=True)

    # the cross-examination and the transplant
    clin = fms["clinician"].oof
    rep["crossexam"] = {k: CX.examine(led, fms[k].oof, fms[k].tier, clin, n_boot=scale.n_boot) for k in HEADLINE}
    rep["process_physiology"] = CF.process_physiology(led, [fms[k] for k in TRANSPLANTED],
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
    base = led.score(fms["direct_P"].oof)["auroc"]
    rep["information_ledger"] = {}
    for fam in PHYSIO_FAMILIES:
        drop = set(LG.FAMILIES[fam])
        keep = np.array([j for j in led.columns("P") if (LG.channel_of(led.names[j]) or "") not in drop])
        fm = led.fit_oof(Candidate(f"P without {fam}", "P", booster(scale), columns=lambda l, keep=keep: keep),
                         verbose=False)
        a = led.score(fm.oof)["auroc"]
        rep["information_ledger"][fam] = {"auroc_without": a, "auroc_lost": base - a}
    rep["label_sensitivity_P"] = led.label_sensitivity(fms["genesis_twin_P"].oof)
    rep["temporal_validation"] = temporal_validation(led, scale)
    rep["trust"] = {"chain": TR.summary(), "code": TR.code_fingerprint(), "power_plan": {
        "absolute_auroc_gt_0.80": TR.power_plan(led, fms["genesis_twin_P"].oof, bar=0.80),
        "twin_over_trees_by_0.01": TR.power_plan(led, fms["genesis_twin_P"].oof, fms["genesis_P"].oof, bar=0.01)}}
    rep["seconds"] = time.time() - t0
    os.makedirs(LG.OUT_DIR, exist_ok=True)
    path = os.path.join(LG.OUT_DIR, f"{tag}_report.json")
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1, default=_plain)
    print(f"report: {path}  [{rep['seconds']:.0f}s]")
    return rep


def verdicts(rep: Dict, fms: Dict[str, FoldModels]) -> Dict[str, Dict]:
    """One row per headline model: the answer to each cross-examination question."""
    out = {}
    for k in HEADLINE:
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
    ap.add_argument("--source", choices=["demo", "bigquery"], default="demo")
    ap.add_argument("--project", default="")
    ap.add_argument("--scale", choices=list(SCALES), default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(6)
    run(a.source, a.tag or a.source, a.scale or ("demo" if a.source == "demo" else "full"), a.project)
