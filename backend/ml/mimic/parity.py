"""Day one with credentials: does BigQuery bin the demo's patients exactly as the CSVs do?

    python -m backend.ml.mimic.parity --project <gcp-project>

The full run reads MIMIC-IV through ``BigQuerySource``, whose SQL mirrors ``CsvSource`` —
the reference semantics every check in ``verify_mimic`` runs on. The SQL cannot be tested
without credentials. But the open demo is 100 real MIMIC-IV patients: once BigQuery works,
every binned table is fetched for exactly those stays (``BigQuerySource.stay_filter``) and
compared with the CSV binning cell by cell — (stay, hour, variable) keys present on one
side only, and values that differ. The SQL is trusted only where it reproduces the
reference; ``--project`` runs it, and the verdict fails below ``AGREEMENT``.

Expect small, explainable differences if the cloud copy is a later MIMIC-IV release than the
demo (3.1 vs 2.2 — some rows were corrected between them); the report lists them per table.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict

import numpy as np
import pandas as pd

from backend.ml.mimic import extract as E
from backend.ml.mimic import ledger as LG

AGREEMENT = 0.99
TABLES: Dict[str, Callable] = {
    "chart": lambda src, st: src.binned_chart(st),
    "labs": lambda src, st: src.binned_labs(st),
    "urine": lambda src, st: src.binned_uo(st),
    "vasopressors": lambda src, st: src.interval_hours(st, "vaso"),
    "ventilation": lambda src, st: src.interval_hours(st, "vent"),
    "sedation": lambda src, st: src.interval_hours(st, "sedation"),
    "dialysis": lambda src, st: src.interval_hours(st, "dialysis"),
    "fluids_blood": lambda src, st: src.hourly_amounts(st),
}
EVENTS = ("culture", "abx", "abx_admin")


def _frame(result) -> pd.DataFrame:
    parts = list(E._pages(result))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["stay_id", "hr", "name", "value"])


def compare_binned(ref: pd.DataFrame, cand: pd.DataFrame, tol: float = 1e-3) -> Dict:
    key = ["stay_id", "hr", "name"]
    a = ref[key + ["value"]].astype({"stay_id": np.int64, "hr": np.int64})
    b = cand[key + ["value"]].astype({"stay_id": np.int64, "hr": np.int64})
    m = a.merge(b, on=key, how="outer", suffixes=("_ref", "_cand"), indicator=True)
    both = m[m["_merge"] == "both"]
    equal = np.isclose(both.value_ref.astype(float), both.value_cand.astype(float), rtol=tol, atol=tol)
    bad = both[~equal]
    return {"reference_rows": int(len(a)), "candidate_rows": int(len(b)), "matched_keys": int(len(both)),
            "values_equal": int(equal.sum()), "only_reference": int((m["_merge"] == "left_only").sum()),
            "only_candidate": int((m["_merge"] == "right_only").sum()),
            "agreement": float(equal.sum() / max(len(m), 1)),
            "worst_variables": bad.name.value_counts().head(5).to_dict()}


def compare_events(ref: pd.DataFrame, cand: pd.DataFrame, tol: float = 1e-3) -> Dict:
    a = ref.assign(t=ref.t.round(3)).groupby("stay_id").t.apply(lambda v: sorted(v)).to_dict()
    b = cand.assign(t=cand.t.round(3)).groupby("stay_id").t.apply(lambda v: sorted(v)).to_dict()
    stays = set(a) | set(b)
    same = sum(1 for k in stays if len(a.get(k, [])) == len(b.get(k, []))
               and np.allclose(a.get(k, []), b.get(k, []), atol=tol))
    return {"reference_events": int(len(ref)), "candidate_events": int(len(cand)), "stays": len(stays),
            "stays_identical": same, "agreement": float(same / max(len(stays), 1))}


def parity(reference, candidate, stays: pd.DataFrame, tol: float = 1e-3) -> Dict:
    out: Dict = {}
    for name, fn in TABLES.items():
        out[name] = compare_binned(_frame(fn(reference, stays)), _frame(fn(candidate, stays)), tol)
    for kind in EVENTS:
        out[kind] = compare_events(reference.event_times(stays, kind), candidate.event_times(stays, kind), tol)
    out["verdict"] = {"agreement_required": AGREEMENT,
                      "failing": sorted(k for k, v in out.items() if v["agreement"] < AGREEMENT)}
    out["verdict"]["passed"] = not out["verdict"]["failing"]
    return out


def main() -> int:  # pragma: no cover - needs credentials
    from backend.ml.mimic.sources import BigQuerySource, CsvSource
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    a = ap.parse_args()
    ref = CsvSource(E.DEMO_ROOT)
    stays = E.cohort(ref)[["stay_id", "hadm_id", "intime", "outtime"]]
    rep = parity(ref, BigQuerySource(a.project), stays)
    os.makedirs(LG.OUT_DIR, exist_ok=True)
    with open(os.path.join(LG.OUT_DIR, "parity.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    for k, v in rep.items():
        if k != "verdict":
            print(f"  {k:<14} agreement {v['agreement']:.4f}")
    print("PARITY", "PASSED" if rep["verdict"]["passed"] else f"FAILED: {rep['verdict']['failing']}")
    return 0 if rep["verdict"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
