"""Run the alarm-policy search on real CinC data and record the verdict.

Patients are split in two: the search sees one half and never the other. The
winner it proposes is then compared against the SHIPPED policy on the locked
half, under a paired patient-level bootstrap. Whatever the gate decides is
written to artifacts/alarm_policy.json — including, and especially, a refusal.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backend.ml import alarm_policy as ap
from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import build_dataset_cached
from backend.ml.eval.splits import stratified_group_folds, stratified_group_split
from backend.ml.models.stacked import StackedEnsembleDetector
from backend.ml.scripts.derive_operating_point import _feature_config

SHIPPED_HIGH = 0.0725


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--limit", type=int, default=4000)
    a.add_argument("--folds", type=int, default=3)
    a.add_argument("--max-far", type=float, default=0.106,
                   help="ceiling: the shipped policy's own out-of-fold false-alarm rate")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--save", action="store_true")
    args = a.parse_args()

    cfg = _feature_config("wearable")
    ds = build_dataset_cached(lambda: load_patients(limit=args.limit), cfg,
                              f"wearable_limit{args.limit}")
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    tr, _ = stratified_group_split(ds.groups, ds.y, 0.25, args.seed)
    y, g, h, o = ds.y[tr], ds.groups[tr], ds.hour_index[tr], ds.onset_hour[tr]
    f = feats.iloc[tr].reset_index(drop=True)

    print(f"building {args.folds}-fold out-of-fold scores...")
    oof = np.zeros(len(y))
    for k, (i_tr, i_te) in enumerate(stratified_group_folds(g, y, args.folds, args.seed)):
        m = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=False).fit(
            f.iloc[i_tr], y[i_tr], groups=g[i_tr])
        oof[i_te] = m.predict_proba(f.iloc[i_te])
        print(f"  fold {k + 1}/{args.folds}")

    series = ap.to_series(oof, g, h, o)

    # Split PATIENTS, not hours. The search never sees the locked half.
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(series))
    cut = len(series) // 2
    search_set = [series[i] for i in idx[:cut]]
    locked = [series[i] for i in idx[cut:]]
    print(f"\n{len(search_set)} patients for the search, {len(locked)} locked away")

    shipped = ap.Policy("plain", {"threshold": SHIPPED_HIGH})
    base = ap.evaluate(shipped, locked)
    print(f"\nSHIPPED on locked: {shipped.describe()}")
    print(f"  false alarm {base.false_alarm_rate:.1%}   detection {base.detection_rate:.1%}"
          f"   lead {base.median_lead_hours:.0f}h")

    print(f"\nsearching (ceiling: false alarm <= {args.max_far:.1%})...")
    best, res, field = ap.search(search_set, args.max_far)
    if best is None:
        print("  no candidate met the ceiling")
        return
    print(f"  {len(field)} candidates evaluated")
    print(f"  winner on the SEARCH half: {best.describe()}")
    print(f"    false alarm {res.false_alarm_rate:.1%}   detection {res.detection_rate:.1%}")

    # best of each family, for the record
    print("\n  best of each family (search half, within ceiling):")
    for fam in ("plain", "persistent", "integrator", "per_patient"):
        ok = [(p, r) for p, r in field
              if p.family == fam and r.false_alarm_rate <= args.max_far + 1e-12]
        if not ok:
            print(f"    {fam:12} — nothing within the ceiling")
            continue
        p, r = max(ok, key=lambda pr: pr[1].detection_rate)
        print(f"    {fam:12} {r.false_alarm_rate:>6.1%} FA  {r.detection_rate:>6.1%} det   {p.describe()}")

    print("\napplying the gate on the locked half...")
    v = ap.gate(shipped, best, locked, n_boot=2000, seed=args.seed)
    print(f"  champion   {v.champion}: FA {v.champion_far:.1%}  det {v.champion_detection:.1%}")
    print(f"  challenger {v.challenger}: FA {v.challenger_far:.1%}  det {v.challenger_detection:.1%}")
    print(f"  detection delta {v.delta_detection:+.3f}  95% CI [{v.ci_low:+.3f}, {v.ci_high:+.3f}]")
    print(f"\n  {'PROMOTED' if v.promoted else 'REFUSED'} — {v.reason}")

    if args.save:
        ap.save(v, shipped)
        print(f"\nwrote {ap.ARTIFACT}")


if __name__ == "__main__":
    main()
