"""The MIMIC-IV pipeline, end to end.

    python -m backend.ml.mimic.run --source demo            # the open 100-patient demo
    python -m backend.ml.mimic.run --source bigquery --project <gcp-project> --scale full

Stages: store -> (design subsample) -> Ledger -> tripwires -> tier-P / T / S boosters ->
GENESIS product per tier -> Process Twin (tier P) -> twin x trees -> information ledger
(drop one variable family at a time; sufficiency) -> label sensitivity -> temporal
validation. Writes ``artifacts/mimic/<tag>_report.json``.

The demo run exists to exercise every stage on real MIMIC data; with 99 subjects its numbers
are noise and the report says so. Nothing here touches the locked subjects — that is
``--locked``, which is refused unless a pre-registration entry exists in the ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from backend.ml.genesis import learners as L
from backend.ml.genesis.stack import log_sigmoid
from backend.ml.mimic import extract as E
from backend.ml.mimic import ledger as LG
from backend.ml.mimic import twin as TW
from backend.ml.mimic.sources import BigQuerySource, CsvSource


@dataclass
class Scale:
    lgbm: Dict = field(default_factory=dict)
    early_stop: int = 150
    twin_epochs: int = 25
    twin_seeds: tuple = (0, 1, 2)
    keep_rate: float = 1.0
    chunk_stays: int = 0
    dtype: str = "float32"


SCALES = {
    "demo": Scale(lgbm={"learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
                        "feature_fraction": 0.5, "lambda_l2": 5.0},
                  early_stop=50, twin_epochs=8, twin_seeds=(0,), keep_rate=1.0),
    # GENESIS's tuned booster (docs/GENESIS.md §4); never-septic stays kept at 25% and
    # re-weighted; features built 2,000 stays at a time into a float16 map on disk.
    "full": Scale(lgbm={"learning_rate": 0.02, "num_leaves": 31, "min_data_in_leaf": 800,
                        "feature_fraction": 0.3, "bagging_fraction": 0.7, "lambda_l2": 10.0},
                  early_stop=150, twin_epochs=25, twin_seeds=(0, 1, 2), keep_rate=0.25,
                  chunk_stays=2000, dtype="float16"),
}
PHYSIO_FAMILIES = ["vitals", "gcs", "labs", "urine", "sofa"]


def _booster(scale: Scale, target: str = "direct"):
    def fit_predict(led: LG.Ledger, cols, tr, te):
        s = led.store
        septic = np.isfinite(led.onset[s.seg])
        if target == "direct":
            y, rows = led.y.astype(float), tr
        elif target == "who":
            y, rows = septic.astype(float), tr
        else:                                            # when: inside septic stays only
            y, rows = led.y.astype(float), tr[septic[tr]]
        fit = L.fit_lgb(led.F, rows, y[rows], s.seg[rows], cols=cols, params=scale.lgbm,
                        weight=led.design_w[rows], strat=septic[rows].astype(int),
                        early_stop=scale.early_stop)
        return fit.predict(led.F, te, raw=True)
    return fit_predict


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
    rep["tripwires"] = led.tripwires()
    print("tripwires:", {k: round(v, 3) for k, v in rep["tripwires"].items()}, flush=True)

    oof: Dict[str, np.ndarray] = {}
    for tier in ("P", "T", "S"):
        for tgt in ("direct", "who", "when"):
            oof[f"{tgt}_{tier}"] = led.run(f"booster {tgt}", _booster(scale, tgt), tier,
                                           verbose=tgt == "direct")
        oof[f"genesis_{tier}"] = sum(log_sigmoid(oof[f"{t}_{tier}"]) for t in ("direct", "who", "when"))

    # Process Twin, tier P
    T = TW.process_targets(led)
    X, _ = TW.twin_inputs(led, "P")
    stays = np.arange(store.n_patients)
    twin = {h: np.full(store.n_rows, np.nan) for h in ("direct", "s3", "susp6", "organ")}
    for k in range(led.n_folds):
        tr, te = stays[(led.folds >= 0) & (led.folds != k)], stays[led.folds == k]
        acc = {h: 0.0 for h in twin}
        for sd in scale.twin_seeds:
            tw = TW.ProcessTwin(TW.TwinConfig(epochs=scale.twin_epochs, seed=sd)).fit(led, X, T, tr)
            pr = tw.predict(led, X, te, tuple(twin))
            for h in twin:
                acc[h] = acc[h] + pr[h] / len(scale.twin_seeds)
        rows = TW.ProcessTwin._rows(led, te)
        for h in twin:
            twin[h][rows] = acc[h]
    oof["twin_s3_P"] = twin["s3"]
    oof["twin_direct_P"] = twin["direct"]
    oof["genesis_twin_P"] = oof["genesis_P"] + log_sigmoid(twin["s3"])

    rep["models"] = {k: led.score(v) for k, v in oof.items() if not k.startswith(("who_", "when_"))}
    for k in ("twin_s3_P", "genesis_P", "genesis_twin_P", "genesis_S"):
        s = rep["models"][k]
        print(f"  {k:<16} AUROC {s['auroc']:.4f}  hour-matched {s['auroc_hourmatched']:.4f}  "
              f"pre-suspicion {s['auroc_presusp']:.4f}", flush=True)
    rep["comparisons"] = {
        "twin_adds_to_trees_P": led.compare(oof["genesis_twin_P"], oof["genesis_P"], n_boot=500),
        "tier_T_over_P": led.compare(oof["genesis_T"], oof["genesis_P"], n_boot=500),
        "tier_S_over_P": led.compare(oof["genesis_S"], oof["genesis_P"], n_boot=500),
    }
    # the gate's factors against their own ground truth
    from backend.ml.genesis.harness import auroc
    dev = np.flatnonzero(led.folds[store.seg] >= 0)
    rep["twin_factors"] = {}
    for h in ("susp6", "organ"):
        y, m = T[h]
        r = dev[(m[dev] > 0) & np.isfinite(twin[h][dev])]
        rep["twin_factors"][h] = auroc(y[r], twin[h][r]) if 0 < y[r].sum() < len(r) else float("nan")

    # information ledger: each physiology family's marginal value (tier P)
    base = led.score(oof["direct_P"])["auroc"]
    rep["information_ledger"] = {}
    for fam in PHYSIO_FAMILIES:
        drop = set(LG.FAMILIES[fam])
        keep = np.array([j for j in led.columns("P") if (LG.channel_of(led.names[j]) or "") not in drop])

        def fp(led_, cols, tr, te, keep=keep):
            return _booster(scale)(led_, keep, tr, te)
        p = led.run(f"P without {fam}", fp, "P", verbose=False)
        rep["information_ledger"][fam] = {"auroc_without": led.score(p)["auroc"],
                                          "auroc_lost": base - led.score(p)["auroc"]}
    rep["label_sensitivity_P"] = led.label_sensitivity(oof["genesis_twin_P"])
    rep["temporal_validation"] = temporal_validation(led, scale)
    rep["seconds"] = time.time() - t0
    os.makedirs(LG.OUT_DIR, exist_ok=True)
    path = os.path.join(LG.OUT_DIR, f"{tag}_report.json")
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print(f"report: {path}  [{rep['seconds']:.0f}s]")
    return rep


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
    p[te] = _booster(scale)(led, led.columns("P"), tr, te)
    sc = led.score(p, rows=te[led.eval_mask[te]])
    return {"train_groups": [E.YEAR_GROUPS[g] for g in groups if g <= cut],
            "test_groups": [E.YEAR_GROUPS[g] for g in groups if g > cut], **sc}


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
