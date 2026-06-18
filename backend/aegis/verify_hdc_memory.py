"""Honest validation of the HDC associative bio-memory.

Measures the STANDALONE discriminative power of the hyperdimensional case-based
retrieval, patient-level and leakage-free:

  * build the memory bank from TRAIN patients only;
  * encode every hour of held-out TEST patients, retrieve the k nearest
    historical states, and score the similarity-weighted "septic-soon" vote;
  * report AUROC of that resemblance score vs the real CinC SepsisLabel, plus
    its early-warning lead time.

No claim is made that HDC beats the gradient-boosted detector — this reports the
number HDC actually earns as a complementary, interpretable evidence channel.

    .venv/bin/python -m backend.aegis.verify_hdc_memory
"""
from __future__ import annotations

import csv
import glob
import os
import random

import numpy as np

from backend.aegis.hdc_memory import CHANNELS, SOON_H, HDCBioMemory, enrich_series
from backend.ml import config as _mlcfg

_CINC_COL = {"hr": "HR", "temp_c": "Temp", "resp_rate": "Resp", "spo2": "O2Sat",
             "map": "MAP", "wbc": "WBC", "lactate": "Lactate"}
_DEFAULT = {"hr": 80.0, "temp_c": 37.0, "resp_rate": 16.0, "spo2": 97.0,
            "map": 85.0, "wbc": 8.0, "lactate": 1.2}


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
    chans = {ch: col(_CINC_COL[ch]) for ch in CHANNELS}
    lab = [int(float(v)) if v not in ("", "NaN") and float(v) == float(v) else 0
           for v in (col("SepsisLabel") if "SepsisLabel" in idx else [0] * n)]
    onset = next((i for i, v in enumerate(lab) if v == 1), None)
    states = []
    for hr in range(n):
        states.append({ch: (chans[ch][hr] if chans[ch][hr] == chans[ch][hr] else _DEFAULT[ch]) for ch in CHANNELS})
    return enrich_series(states), lab, onset


def _patient_files(limit):
    files = (glob.glob(os.path.join(_mlcfg.DATA_DIR, "training_setA", "*.psv")) +
             glob.glob(os.path.join(_mlcfg.DATA_DIR, "training_setB", "*.psv")))
    if not files:
        files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "data", "cohort_sample", "*.psv"))
    random.Random(13).shuffle(files)
    return files[:limit]


def _auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(n_patients: int = 260, k: int = 16, mem_cap: int = 5000) -> None:
    print("=" * 64); print("AegisOS — HDC associative bio-memory verification"); print("=" * 64)
    files = _patient_files(n_patients)
    pats = []
    for p in files:
        try:
            states, lab, onset = _load(p)
        except Exception:
            continue
        if len(states) >= 8:
            pats.append((states, lab, onset))
    rng = random.Random(7); rng.shuffle(pats)
    cut = int(len(pats) * 0.55)
    train, test = pats[:cut], pats[cut:]
    print(f"[data] {len(pats)} real CinC patients  ·  {len(train)} memory / {len(test)} query (patient-level split)")

    mem = HDCBioMemory()
    # build the reference bank from TRAIN patients only (subsample for speed)
    mrows = []
    for states, lab, onset in train:
        for hr, st in enumerate(states):
            tto = (onset - hr) if onset is not None else None
            soon = (lab[hr] == 1 or (tto is not None and 0 <= tto <= SOON_H))
            mrows.append((st, soon))
    rng.shuffle(mrows); mrows = mrows[:mem_cap]
    M = np.stack([mem.encode(s) for s, _ in mrows]).astype(np.float32)
    msoon = np.array([1 if soon else 0 for _, soon in mrows])
    print(f"[memory] {len(mrows)} encoded historical states  ·  {int(msoon.sum())} septic-soon")

    # query test hours (vectorised)
    qstates, ylab = [], []
    for states, lab, onset in test:
        for hr, st in enumerate(states):
            qstates.append(st); ylab.append(lab[hr])
    Q = np.stack([mem.encode(s) for s in qstates]).astype(np.float32)
    sims = (Q @ M.T) / float(mem.eng.D)                         # (Mq, Nmem)
    res = np.zeros(len(qstates))
    for i in range(len(qstates)):
        top = np.argpartition(-sims[i], k)[:k]
        w = np.clip(sims[i, top], 0, None) + 1e-6
        res[i] = (w * msoon[top]).sum() / w.sum()
    auroc = _auroc(res, ylab)
    base = float(np.mean(ylab))
    print(f"[result] HDC retrieval resemblance — AUROC {auroc:.3f}  "
          f"(prevalence {base*100:.1f}% septic-hours, n={len(ylab)})")

    # per-patient early warning: first hour resemblance sustained >= thr (2h) vs onset
    thr = float(np.quantile(res, 0.85))
    leads, detected, sept = [], 0, 0
    qi = 0
    for states, lab, onset in test:
        L = len(states); seg = res[qi:qi + L]; qi += L
        if onset is None:
            continue
        sept += 1
        first = None; run = 0
        for hr in range(L):
            run = run + 1 if seg[hr] >= thr else 0
            if run >= 2:
                first = hr - 1; break
        if first is not None and first <= onset:
            detected += 1; leads.append(onset - first)
    if sept:
        print(f"[lead]   at the 85th-pct operating point: detected {detected}/{sept} septic "
              f"({100*detected/sept:.0f}%), median lead {np.median(leads) if leads else 0:.0f} h")
    print("-" * 64)
    print("HDC bio-memory is a COMPLEMENTARY, interpretable case-based channel — "
          "number reported as measured, not asserted to beat the GBM.")


if __name__ == "__main__":
    main()
