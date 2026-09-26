"""Information-flow audit — the tier firewall, measured instead of declared.

The Ledger's firewall trusts a declaration: a column computed from a physiology channel is
tier P, so a tier-P model "cannot" read treatment. Security engineering learned long ago
that declared flows are not actual flows (Lampson's confinement problem, 1973; Denning's
information-flow lattice, 1976): data carries information through channels nobody labelled.
ICU records are full of them. A verbal Glasgow score is charted "No Response-ETT" because
the patient has a breathing tube; end-tidal CO2 exists because a ventilator circuit is
attached; an FiO2 of 0.6 is a setting a clinician chose. Each is "physiology" by source and
ventilator state by content.

So the audit measures it. For every physiology channel, a small booster reads ONLY that
channel's feature columns and predicts each treatment state — invasive or non-invasive
ventilation, any vasopressor — at that hour, out of fold over subjects. A channel that
recovers a treatment state at AUROC >= ``threshold`` is a covert channel. It is reported
and flagged, and the ledger's STRICT tier (``Ledger.columns(tier, strict=True)``) drops
every column computed from a flagged channel, including composites built on it. The
declared tier stays the primary one; the strict tier is what a claim of "physiology alone"
must survive.

Suspicion (culture / antibiotic orders) is reported alongside but never flagged:
physiology that anticipates clinicians' suspicion is what an early-warning model is FOR.
Whether a model is reading the care process is the Care-Process Transplant's question
(``counterfactual``), not this one.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from backend.ml.genesis import learners as L
from backend.ml.genesis.harness import auroc
from backend.ml.mimic import extract as E
from backend.ml.mimic.ledger import Ledger, channel_of

THRESHOLD = 0.90
SMALL = {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 50, "feature_fraction": 1.0,
         "bagging_fraction": 1.0, "bagging_freq": 0, "lambda_l2": 1.0}


def treatment_states(led: Ledger) -> Dict[str, np.ndarray]:
    s = led.store
    vent = np.zeros(s.n_rows, dtype=bool)
    for c in E.VENT_CH:
        vent |= np.nan_to_num(s.col(c)) > 0
    vaso = np.zeros(s.n_rows, dtype=bool)
    for c in E.VASO_CH:
        vaso |= np.nan_to_num(s.col(c)) > 0
    return {"ventilated": vent, "vasopressor": vaso}


def suspicion_state(led: Ledger) -> np.ndarray:
    s = led.store
    first = s.onsets.first_order.to_numpy()[s.seg]
    return np.isfinite(first) & (first <= s.hour + 1)


def physiology_channels(led: Ledger) -> List[str]:
    return [c for c in E.DYNAMIC if led.store.tiers[c] == "P"]


def audit(led: Ledger, threshold: float = THRESHOLD, max_rows: int = 250_000, n_folds: int = 3,
          rounds: int = 60, seed: int = 0, channels: Optional[List[str]] = None) -> Dict:
    s = led.store
    rng = np.random.default_rng(seed)
    dev = np.flatnonzero((led.folds[s.seg] >= 0) & led.train_mask)
    if len(dev) > max_rows:
        dev = np.sort(rng.choice(dev, max_rows, replace=False))
    fold = led.folds[s.seg[dev]] % n_folds          # subject-grouped: dev folds are by subject
    states = treatment_states(led)
    states_all = {**states, "orders_begun": suspicion_state(led)}
    by_ch: Dict[str, List[int]] = {}
    for j, n in enumerate(led.names):
        ch = channel_of(n)
        if ch is not None:
            by_ch.setdefault(ch, []).append(j)
    table: Dict[str, Dict[str, float]] = {}
    for ch in channels or physiology_channels(led):
        cols = np.array(by_ch.get(ch, []))
        if not len(cols):
            continue
        row = {}
        for name, st in states_all.items():
            y = st[dev].astype(float)
            if y.min() == y.max():
                row[name] = float("nan")
                continue
            oof = np.full(len(dev), np.nan)
            for k in range(n_folds):
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                if not len(te) or y[tr].min() == y[tr].max():
                    continue
                fit = L.fit_lgb(led.F, dev[tr], y[tr], s.seg[dev[tr]], params=SMALL, cols=cols,
                                fixed_rounds=rounds, seed=seed)
                oof[te] = fit.predict(led.F, dev[te], raw=True)
            ok = np.isfinite(oof)
            row[name] = auroc(y[ok], oof[ok]) if ok.any() and 0 < y[ok].sum() < ok.sum() else float("nan")
        table[ch] = row
    flagged = sorted(ch for ch, r in table.items()
                     if max((r[k] for k in states if np.isfinite(r[k])), default=0.0) >= threshold)
    return {"threshold": threshold, "rows": int(len(dev)),
            "prevalence": {k: float(v[dev].mean()) for k, v in states_all.items()},
            "table": table, "flagged": flagged,
            "note": "flagged = recovers a treatment state (ventilation / vasopressor) from one physiology "
                    "channel alone; orders_begun is reported, never flagged"}
