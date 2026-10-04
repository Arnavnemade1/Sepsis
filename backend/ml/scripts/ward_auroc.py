"""What does the WARD's risk discriminate at - before and after it read the detector?

The ward (ICU rooms 301-312) used to be scored by ``kernel.fusion`` (a hand-weighted
qSOFA/SIRS logistic) passed through the own-baseline learner; the validated detector
served only the live session. This scores, on the SAME rows:

    rules       the old ward path: fusion -> own-baseline personalisation
    rules_pop   fusion alone (the population input to it)
    detector    the shipped detector's calibrated probability
    index       the same on the ward index (``backend.aegis.ward_risk``): must equal it
    index_pers  the index pushed through the own-baseline learner (should it stay?)

on the shipped detector's promotion holdout - patients the serving fit never trained
on, proven by its fit digest - every hour, against the challenge label, with a paired
patient bootstrap for the difference. Writes ``artifacts/ward_risk.json``.

    python -m backend.ml.scripts.ward_auroc            # ~1 min, needs the feature cache
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.data import cinc2019
from backend.ml.eval.metrics import auroc
from backend.ml.genesis.harness import hour_matched_auroc
from backend.ml.scripts.promote_frontier import MANIFEST, promotion_split

OUT = os.path.join(config.ARTIFACTS_DIR, "ward_risk.json")


def _record(pid: str):
    for s in config.CINC_SETS:
        f = os.path.join(config.DATA_DIR, s, f"{pid}.psv")
        if os.path.exists(f):
            return cinc2019._parse_psv(f, pid, s)
    raise FileNotFoundError(pid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()

    from backend.aegis.cohort import CHANNELS, RealPatient
    from backend.aegis.kernel.fusion import FusionEngine
    from backend.aegis.learner import PatientBaselineLearner, personalized_risk
    from backend.aegis.ward_risk import IndexMap
    from backend.ml.models.stacked import StackedEnsembleDetector

    with open(MANIFEST) as fh:
        man = json.load(fh)
    ds, _, te = promotion_split(man)
    y, g, h = ds.y[te].astype(int), ds.groups[te], ds.hour_index[te]
    print(f"holdout: {len(te):,} rows, {len(np.unique(g)):,} patients (fit {man['fit_digest']})")

    det = StackedEnsembleDetector.load(os.path.join(config.ARTIFACTS_DIR, "gbm_wearable.joblib"))
    if getattr(det, "fit_digest_", None) != man["fit_digest"]:
        raise SystemExit("the deployed artifact is not the fit its manifest records")
    p = det.predict_proba(pd.DataFrame(ds.X[te], columns=ds.feature_names))
    p = p[:, 1] if p.ndim == 2 else p
    op = man["operating_point"]
    imap = IndexMap.pinned(op["medium"], op["high"], man["detector_id"])
    idx = imap.index_array(p)

    pos = {(pid, int(hr)): i for i, (pid, hr) in enumerate(zip(g, h))}
    rules = np.full(len(te), np.nan)
    rules_pop = np.full(len(te), np.nan)
    idx_pers = np.full(len(te), np.nan)
    ent_keys = ("temp_c", "resp_rate", "spo2", "map", "wbc", "lactate")
    t0 = time.time()
    for k, pid in enumerate(np.unique(g)):
        rp = RealPatient(record=_record(pid), name=pid, bed_id="b", weight=1.0)
        fe, lz = FusionEngine(), PatientBaselineLearner()
        for hr in range(rp.los_total):
            vals = {ch: float(rp._ff[hr, j]) for j, ch in enumerate(CHANNELS)}
            edge = {"hr": vals["hr"],
                    "perfusion": max(0.4, min(1.8, 1.0 + (vals["spo2"] - 95.0) / 18.0))}
            pr = fe.update("b", pid, edge, {kk: vals[kk] for kk in ent_keys}).fused_risk
            lz.observe(vals)
            snap = lz.snapshot(vals)
            i = pos.get((pid, hr))
            if i is not None:
                rules_pop[i] = pr
                rules[i] = personalized_risk(pr, snap["deviation"], snap["maturity"])
                idx_pers[i] = personalized_risk(idx[i], snap["deviation"], snap["maturity"])
        if k % 2000 == 0:
            print(f"  {k:,} patients replayed  {time.time() - t0:.0f}s", flush=True)
    assert not (np.isnan(rules).any() or np.isnan(idx_pers).any())

    scores = {"rules": rules, "rules_pop": rules_pop, "detector": p, "index": idx,
              "index_pers": idx_pers}
    out = {name: {"auroc": round(auroc(y, s), 4),
                  "hour_matched": round(hour_matched_auroc(y, s, h), 4)}
           for name, s in scores.items()}
    for name, r in out.items():
        print(f"  {name:11} AUROC {r['auroc']:.4f}  hour-matched {r['hour_matched']:.4f}")

    rng = np.random.default_rng(0)
    uniq, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    rows_of = [order[bounds[j]:bounds[j + 1]] for j in range(len(uniq))]
    d = []
    for _ in range(args.boot):
        rows = np.concatenate([rows_of[j] for j in rng.integers(0, len(uniq), len(uniq))])
        d.append(auroc(y[rows], p[rows]) - auroc(y[rows], rules[rows]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    delta = out["detector"]["auroc"] - out["rules"]["auroc"]
    print(f"  detector - rules: {delta:+.4f} [{lo:+.4f}, {hi:+.4f}] ({args.boot} patient draws)")

    res = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detector": man["detector_id"], "fit_digest": man["fit_digest"],
        "rows": int(len(te)), "patients": int(len(uniq)),
        "split": "the shipped detector's promotion holdout (patients it never trained on), "
                 "every hour, challenge label",
        "scores": out,
        "delta_detector_minus_rules": {"delta": round(delta, 4), "lo": round(float(lo), 4),
                                       "hi": round(float(hi), 4), "draws": args.boot},
        "index_map": imap.to_dict(),
        "index_preserves_auroc": abs(out["index"]["auroc"] - out["detector"]["auroc"]) < 1e-3,
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
