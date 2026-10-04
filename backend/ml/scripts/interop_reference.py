"""Reference distributions for the integration plane, and an audit of the data they come from.

The evidence engine (``backend/aegis/interop/evidence.py``) compares a feed against a
reference distribution per concept. Those references come from here:

  * the patients the shipped detector TRAINED on (the promotion split, proven by its
    fit digest), never its holdout and never the GENESIS locked test set;
  * per concept, the values inside the concept's physiological window (fixed a priori
    in ``vocab``), so a column that is itself contaminated (see below) cannot widen it;
  * concepts CinC does not record get an a-priori reference (stated as such).

It also measures what the a-priori checks say about the data itself, per hospital
(set A: Beth Israel Deaconess; set B: Emory), on the training patients and again on
the holdout as a replication:

  * the fraction of each column outside its concept's physiological window;
  * the violation rate of every physiological identity on same-hour rows.

Writes ``backend/aegis/interop/reference.json`` (served) and
``artifacts/interop/cinc_audit.json``.

    python -m backend.ml.scripts.interop_reference
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List

import numpy as np

from backend.ml import config
from backend.aegis.interop import evidence as E
from backend.aegis.interop import vocab as Vc

QUANTILES = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999]
AUDIT_OUT = os.path.join(config.ARTIFACTS_DIR, "interop", "cinc_audit.json")

# Concepts CinC does not record: an a-priori reference (log-normal median, log sd), stated as such.
A_PRIORI = {
    "sodium": (139.0, 0.030, "adult serum sodium reference interval 135–145 mmol/L"),
    "anc": (4500.0, 0.75, "adult absolute neutrophil count, broad hospital population"),
    "inr": (1.2, 0.30, "INR in a hospital population, therapeutic anticoagulation included"),
}


def _ref_from_values(c: Vc.Concept, v: np.ndarray, source: str) -> Dict[str, object]:
    q = np.quantile(v, QUANTILES)
    if c.scale == "log":
        t = np.log(v[v > 0])
    else:
        t = v
    q25, q50, q75 = np.quantile(t, [0.25, 0.5, 0.75])
    return {"n": int(len(v)), "source": source, "quantiles": {str(p): float(x) for p, x in zip(QUANTILES, q)},
            "loc": float(q50), "spread": float(max((q75 - q25) / 1.349, 1e-6)),
            "tail": [float(np.quantile(v, 0.005)), float(np.quantile(v, 0.995))]}


def _ref_a_priori(c: Vc.Concept, median: float, lsd: float, note: str) -> Dict[str, object]:
    from statistics import NormalDist
    nd = NormalDist(math.log(median), lsd)
    qs = {str(p): float(math.exp(nd.inv_cdf(p))) for p in QUANTILES}
    return {"n": 0, "source": f"a priori: {note}", "quantiles": qs, "loc": math.log(median), "spread": lsd,
            "tail": [float(math.exp(nd.inv_cdf(0.005))), float(math.exp(nd.inv_cdf(0.995)))]}


def _invariant_rates(cols: List[str], X: np.ndarray) -> Dict[str, Dict[str, object]]:
    out = {}
    for inv in E.INVARIANTS:
        idx = []
        for k in inv.concepts:
            c = Vc.BY_KEY[k]
            if not c.cinc or c.cinc not in cols:
                idx = None
                break
            idx.append(cols.index(c.cinc))
        if idx is None:
            continue
        sub = X[:, idx]
        ok = ~np.isnan(sub).any(axis=1)
        rows = sub[ok]
        # the identities read canonical units; within each concept's window only (the audit
        # counts out-of-window values separately)
        inwin = np.ones(len(rows), bool)
        for j, k in enumerate(inv.concepts):
            lo, hi = Vc.BY_KEY[k].window
            inwin &= (rows[:, j] >= lo) & (rows[:, j] <= hi)
        rows = rows[inwin]
        viol = 0
        for r in rows:
            s = dict(zip(inv.concepts, (float(x) for x in r)))
            try:
                viol += bool(inv.violated(s))
            except (ZeroDivisionError, ValueError):
                viol += 1
        n = int(len(rows))
        out[inv.name] = {"n": n, "violations": int(viol), "rate": (viol / n) if n else None, "p0": inv.p0}
    return out


def _audit(cols: List[str], X: np.ndarray, site_row: np.ndarray) -> Dict[str, object]:
    res: Dict[str, object] = {"columns": {}, "invariants": {}}
    for s, name in ((0, "A"), (1, "B")):
        Xs = X[site_row == s]
        for c in Vc.CONCEPTS:
            if not c.cinc or c.cinc not in cols:
                continue
            v = Xs[:, cols.index(c.cinc)]
            v = v[~np.isnan(v)]
            if not len(v):
                continue
            lo, hi = c.window
            out = (v < lo) | (v > hi)
            d = res["columns"].setdefault(c.key, {"cinc": c.cinc, "window": [lo, hi]})
            d[name] = {"n": int(len(v)), "outside_window": float(out.mean()), "outside_n": int(out.sum()),
                       "median": float(np.median(v)),
                       "median_outside": (float(np.median(v[out])) if out.any() else None)}
        for k, r in _invariant_rates(cols, Xs).items():
            res["invariants"].setdefault(k, {})[name] = r
    return res


def main() -> None:
    t0 = time.time()
    from backend.ml.genesis import rawstore
    from backend.ml.scripts.promote_frontier import MANIFEST, promotion_split

    with open(MANIFEST) as fh:
        man = json.load(fh)
    ds, tr, te = promotion_split(man)           # proves the split against the fit digest
    train_ids, hold_ids = np.unique(ds.groups[tr]), np.unique(ds.groups[te])
    del ds
    raw = rawstore.load()
    seg = raw.seg
    pid_row, site_row = raw.pid[seg], raw.site[seg]
    cols = list(raw.columns)
    trm = np.isin(pid_row, train_ids)
    hom = np.isin(pid_row, hold_ids)
    Xtr, str_ = raw.X[trm], site_row[trm]
    print(f"training patients {len(train_ids):,} ({trm.sum():,} rows); holdout {len(hold_ids):,} "
          f"({hom.sum():,} rows); fit {man['fit_digest']}")

    concepts: Dict[str, object] = {}
    for c in Vc.CONCEPTS:
        if c.cinc and c.cinc in cols:
            v = Xtr[:, cols.index(c.cinc)]
            v = v[~np.isnan(v)]
            lo, hi = c.window
            v = v[(v >= lo) & (v <= hi)]
            if len(v) >= 200:
                concepts[c.key] = _ref_from_values(c, v.astype(np.float64),
                                                   f"CinC 2019 training patients of fit {man['fit_digest']}, "
                                                   f"values inside the physiological window")
        elif c.key == "ionized_calcium":
            # Emory records ionized calcium (mmol/L) under "Calcium" next to total calcium
            # (mg/dL); the values below the total-calcium window are the ionized ones.
            v = Xtr[str_ == 1][:, cols.index("Calcium")]
            v = v[~np.isnan(v)]
            v = v[(v >= c.window[0]) & (v <= c.window[1])]
            concepts[c.key] = _ref_from_values(c, v.astype(np.float64),
                                               "CinC hospital B training patients: values recorded under 'Calcium' "
                                               "below the total-calcium window (ionized calcium, mmol/L)")
        elif c.key in A_PRIORI:
            med, lsd, note = A_PRIORI[c.key]
            concepts[c.key] = _ref_a_priori(c, med, lsd, note)
    for k, r in concepts.items():
        print(f"  {k:18s} n={r['n']:8d} loc={r['loc']:.3f} spread={r['spread']:.3f} tail={r['tail'][0]:.3g}–{r['tail'][1]:.3g}")

    audit_train = _audit(cols, Xtr, str_)
    audit_hold = _audit(cols, raw.X[hom], site_row[hom])
    findings = []
    for key, d in audit_train["columns"].items():
        for site in ("A", "B"):
            if site in d and d[site]["outside_window"] > E.P_IMPLAUSIBLE:
                findings.append({"kind": "window", "concept": key, "site": site, **d[site],
                                 "holdout": audit_hold["columns"].get(key, {}).get(site)})
    for name, d in audit_train["invariants"].items():
        p0 = E.INVARIANT_BY_NAME[name].p0
        for site in ("A", "B"):
            r = d.get(site)
            if r and r["rate"] is not None and r["n"] >= 50 and r["rate"] > p0:
                findings.append({"kind": "invariant", "invariant": name, "site": site, **r,
                                 "holdout": audit_hold["invariants"].get(name, {}).get(site)})
    print("findings (a-priori checks the data itself fails):")
    for f in findings:
        what = f.get("concept") or f.get("invariant")
        rate = f.get("outside_window", f.get("rate"))
        hold = f.get("holdout") or {}
        hrate = hold.get("outside_window", hold.get("rate"))
        print(f"  {f['kind']:9s} {what:22s} site {f['site']}: {rate:.4f}  (holdout {hrate if hrate is None else round(hrate, 4)})")

    meta = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "fit_digest": man["fit_digest"],
            "training_patients": int(len(train_ids)), "quantiles": QUANTILES,
            "note": "Reference distributions for the integration plane's evidence engine. Training patients of the "
                    "shipped detector only; values inside each concept's a-priori physiological window."}
    with open(Vc.REFERENCE_PATH, "w") as fh:
        json.dump({"meta": meta, "concepts": concepts}, fh, indent=1, sort_keys=True)
    os.makedirs(os.path.dirname(AUDIT_OUT), exist_ok=True)
    with open(AUDIT_OUT, "w") as fh:
        json.dump({"meta": {**meta, "holdout_patients": int(len(hold_ids)),
                            "sites": {"A": "Beth Israel Deaconess (training_setA)", "B": "Emory (training_setB)"}},
                   "findings": findings, "training": audit_train, "holdout": audit_hold}, fh, indent=1)
    print(f"wrote {Vc.REFERENCE_PATH} and {AUDIT_OUT} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
