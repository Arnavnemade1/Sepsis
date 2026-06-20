"""Honest validation of the Critical Slowing Down early-warning radar.

Measures whether the resilience-loss score actually PRECEDES sepsis on real
CinC-2019 data — leakage-free, causal — before any clinical claim is made:

  * hour-level AUROC of the composite EWS (and each sub-indicator separately, so
    we see which one carries the signal) vs the real SepsisLabel;
  * patient-level separation (peak pre-onset EWS, septic vs non-septic);
  * early-warning LEAD TIME: how many hours before labelled onset the EWS first
    rises above the patient's own baseline.

Reports the numbers the radar truly earns. CSD on hourly/sparse ICU data is
expected to be a MODEST, complementary research signal, not a primary detector.

    .venv/bin/python -m backend.aegis.verify_resilience
"""
from __future__ import annotations

import csv
import glob
import os
import random

import numpy as np

from backend.aegis.resilience import CSD_CHANNELS, WIN, BASE_W, ews_series
from backend.ml import config as _mlcfg

_CINC = {"hr": "HR", "temp_c": "Temp", "resp_rate": "Resp", "spo2": "O2Sat", "map": "MAP"}


def _load(path):
    with open(path) as f:
        r = csv.reader(f, delimiter="|"); hdr = next(r); idx = {h: i for i, h in enumerate(hdr)}
        rows = [row for row in r]
    n = len(rows)
    def col(name):
        j = idx.get(name); out = []; last = float("nan")
        for row in rows:
            v = row[j] if j is not None else "NaN"
            try: x = float(v)
            except Exception: x = float("nan")
            if x == x: last = x
            out.append(last)
        return out
    chans = {ch: col(_CINC[ch]) for ch in CSD_CHANNELS}
    lab = [int(float(v)) if v not in ("", "NaN") and float(v) == float(v) else 0
           for v in (col("SepsisLabel") if "SepsisLabel" in idx else [0] * n)]
    onset = next((i for i, v in enumerate(lab) if v == 1), None)
    return chans, lab, onset, n


def _files(limit):
    fs = (glob.glob(os.path.join(_mlcfg.DATA_DIR, "training_setA", "*.psv")) +
          glob.glob(os.path.join(_mlcfg.DATA_DIR, "training_setB", "*.psv")))
    if not fs:
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "cohort_sample")
        fs = glob.glob(os.path.join(d, "*.psv"))
    random.Random(13).shuffle(fs)
    return fs[:limit]


def _auroc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main(n_patients: int = 600) -> None:
    print("=" * 68)
    print("AegisOS — Critical Slowing Down early-warning radar · honest validation")
    print("=" * 68)
    warm = WIN + BASE_W
    H_ews, H_ar1, H_var, H_y = [], [], [], []
    peak_sep, peak_non = [], []
    leads = []
    n_sep = n_used = 0
    for path in _files(n_patients):
        chans, lab, onset, n = _load(path)
        if n < warm + 4:
            continue
        n_used += 1
        S = ews_series(chans, n)
        ews, ar1, var = S["ews"], S["ar1"], S["var"]
        for t in range(warm, n):                          # hour-level (post-warmup)
            H_ews.append(ews[t]); H_ar1.append(ar1[t]); H_var.append(var[t]); H_y.append(lab[t])
        if onset is not None and onset > warm:
            n_sep += 1
            peak_sep.append(float(np.max(ews[warm:onset + 1])))
            base = float(np.median(ews[warm:min(n, warm + 8)])) if n > warm else 0.0
            thr = max(0.15, base + 0.12)
            cross = next((t for t in range(warm, onset + 1) if ews[t] >= thr), None)
            if cross is not None:
                leads.append(onset - cross)
        elif onset is None:
            peak_non.append(float(np.max(ews[warm:])) if n > warm else 0.0)

    print(f"\npatients used {n_used}  ·  septic {n_sep}  ·  hours scored {len(H_y)}"
          f"  (prevalence {100*np.mean(H_y):.1f}% septic-hours)")
    print("\nHOUR-LEVEL AUROC (does the indicator rank septic hours above the rest?)")
    print(f"  composite EWS        {_auroc(H_y, H_ews):.3f}")
    print(f"  variance-rise only   {_auroc(H_y, H_var):.3f}")
    print(f"  autocorr-rise only   {_auroc(H_y, H_ar1):.3f}")
    if peak_sep and peak_non:
        y = [1] * len(peak_sep) + [0] * len(peak_non)
        print(f"\nPATIENT-LEVEL AUROC (peak pre-onset EWS, septic vs non-septic)")
        print(f"  {_auroc(y, peak_sep + peak_non):.3f}"
              f"   (septic peak {np.mean(peak_sep):.2f} vs non-septic {np.mean(peak_non):.2f})")
    if leads:
        leads = np.array(leads)
        print(f"\nEARLY-WARNING LEAD TIME (EWS rises above own baseline before onset)")
        print(f"  fired before onset in {len(leads)}/{n_sep} septic patients"
              f"  ·  median lead {np.median(leads):.0f} h  ·  mean {np.mean(leads):.1f} h")
    print("\n" + "-" * 68)
    print("Reported as MEASURED. CSD is a complementary research-grade resilience")
    print("signal on hourly ICU data — not a standalone primary detector.")


if __name__ == "__main__":
    main()
