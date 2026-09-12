"""Run the alarm-policy search AT THE SHIPPED OPERATING POINT and record the verdict.

Patients are split in two: the search sees one half and never the other. The winner
it proposes is then compared against the SHIPPED policy on the locked half, under a
paired patient-level bootstrap. Whatever the gate decides is written to
artifacts/alarm_policy.json — including, and especially, a refusal.

WHY THIS WAS REWRITTEN (2026-09-12)
-----------------------------------
This script hardcoded ``SHIPPED_HIGH = 0.0725`` and built a StackedEnsembleDetector
with the class's *constructor defaults*. The system ships ``high = 0.03`` on
``patient:hgb+hgb2+et+lr+xgb@utility``. So the conclusion published in
``backend/ml/alarm_policy.py`` — "four families searched, none beats a plain
threshold" — was measured at an operating point and on a model that are no longer
deployed, and it is quoted on the live console and the public site.

Both are now READ from the artifacts, and the identity is ASSERTED before anything is
measured (``operating_point.require_shipped_detector``). A mismatch is fatal.

TWO STUDIES RUN HERE
--------------------
1. FIRST-ALARM families — the historic study: when does the first alarm fire, and
   can any policy family beat a plain threshold at or below its false-alarm rate.
2. EPISODE ECONOMICS — the operational layer the roadmap asks for: how often a person
   is interrupted after that first alarm. Its candidates cannot move the first
   crossing, so detection and false-alarm rate are invariant by construction; the
   burden gate resamples NON-SEPTIC patients, because that is where the burden lives.

NO LEAD-TIME QUANTITY IS REPORTED. A random score reproduces this detector's median
lead at every matched detection rate (docs/LEAD_TIME_NULL.md).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from backend.ml import alarm_policy as ap
from backend.ml import fitid
from backend.ml import operating_point as opmod
from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import build_dataset_cached
from backend.ml.eval.splits import stratified_group_folds, stratified_group_split
from backend.ml.models.stacked import StackedEnsembleDetector
from backend.ml.scripts.derive_operating_point import MANIFEST, _feature_config, promoted_config


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--limit", type=int, default=4000)
    a.add_argument("--folds", type=int, default=3)
    a.add_argument("--max-far", type=float, default=None,
                   help="ceiling; default is the shipped policy's OWN realised "
                        "false-alarm rate on the search half")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--boot", type=int, default=2000)
    a.add_argument("--save", action="store_true")
    args = a.parse_args()

    # ---- the shipped configuration, read and asserted, never re-declared ----
    op = opmod.load()
    roster, objective = promoted_config()
    key = opmod.require_shipped_detector(op, roster, objective)
    print(f"shipped operating point: high={op.high} medium={op.medium}  detector={key}")

    cfg = _feature_config("wearable")
    ds = build_dataset_cached(lambda: load_patients(limit=args.limit), cfg,
                              f"wearable_limit{args.limit}")
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)
    tr, _ = stratified_group_split(ds.groups, ds.y, 0.25, args.seed)
    y, g, h, o = ds.y[tr], ds.groups[tr], ds.hour_index[tr], ds.onset_hour[tr]
    f = feats.iloc[tr].reset_index(drop=True)

    print(f"building {args.folds}-fold out-of-fold scores on the PROMOTED roster...")
    oof = np.zeros(len(y))
    for k, (i_tr, i_te) in enumerate(stratified_group_folds(g, y, args.folds, args.seed)):
        # objective="patient"/"utility" need hour_index + onset_hour at fit time.
        m = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=False,
                                    roster=list(roster), objective=objective).fit(
            f.iloc[i_tr], y[i_tr], groups=g[i_tr],
            hour_index=h[i_tr], onset_hour=o[i_tr])
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

    shipped = ap.Policy("plain", {"threshold": op.high})
    base = ap.evaluate(shipped, locked)
    print(f"\nSHIPPED on locked: {shipped.describe()}")
    print(f"  false alarm {base.false_alarm_rate:.1%}   detection {base.detection_rate:.1%}")

    # The ceiling is the shipped policy's own rate ON THE SET IT CONSTRAINS. The old
    # constant 0.106 was measured on the other half, so the ceiling and the candidates
    # it filtered were never measured on the same patients.
    ceiling = (args.max_far if args.max_far is not None
               else ap.evaluate(shipped, search_set).false_alarm_rate)
    print(f"\nsearching (ceiling: false alarm <= {ceiling:.1%})...")
    best, res, field = ap.search(search_set, ceiling)
    if best is None:
        print("  no candidate met the ceiling")
        return
    print(f"  {len(field)} candidates evaluated")
    print(f"  winner on the SEARCH half: {best.describe()}")
    print(f"    false alarm {res.false_alarm_rate:.1%}   detection {res.detection_rate:.1%}")

    print("\n  best of each family (search half, within ceiling):")
    for fam in ("plain", "persistent", "integrator", "per_patient"):
        ok = [(p, r) for p, r in field
              if p.family == fam and r.false_alarm_rate <= ceiling + 1e-12]
        if not ok:
            print(f"    {fam:12} — nothing within the ceiling")
            continue
        p, r = max(ok, key=lambda pr: pr[1].detection_rate)
        print(f"    {fam:12} {r.false_alarm_rate:>6.1%} FA  {r.detection_rate:>6.1%} det   {p.describe()}")

    print("\napplying the gate on the locked half...")
    v = ap.gate(shipped, best, locked, n_boot=args.boot, seed=args.seed)
    print(f"  champion   {v.champion}: FA {v.champion_far:.1%}  det {v.champion_detection:.1%}")
    print(f"  challenger {v.challenger}: FA {v.challenger_far:.1%}  det {v.challenger_detection:.1%}")
    print(f"  detection delta {v.delta_detection:+.3f}  95% CI [{v.ci_low:+.3f}, {v.ci_high:+.3f}]")
    print(f"\n  {'PROMOTED' if v.promoted else 'REFUSED'} — {v.reason}")

    # ---------------- episode economics -------------------------------------
    print("\n" + "-" * 72)
    print("EPISODE ECONOMICS (replayed through the LIVE AlertEpisodeController)")
    shipped_ep = ap.EpisodePolicy()                      # 4h re-notify: what runs today
    ship_ep_res = ap.evaluate_episodes(shipped_ep, locked, op.high, op.medium)
    print(f"  shipped {shipped_ep.describe()} on locked:")
    print(f"    non-septic episodes / patient-day      {ship_ep_res.nonseptic_episodes_per_patient_day:.3f}")
    print(f"    repeat notifications / septic episode  {ship_ep_res.repeat_notifications_per_septic_episode:.3f}")
    print(f"    escalation recall (conformance)        {ship_ep_res.escalation_recall}")
    print(f"    median time to clear (h)               {ship_ep_res.median_time_to_clear_h}")

    print("\n  schedules on the search half (burden = notifications / patient-day):")
    ep_field = []
    for cand in ap.episode_grid():
        r = ap.evaluate_episodes(cand, search_set, op.high, op.medium)
        burden = float(r.burden_per_patient.mean()) if r.burden_per_patient.size else float("nan")
        ep_field.append({"policy": cand.describe(),
                         "high_renotify_seconds": cand.high_renotify_seconds,
                         "burden_per_patient_day": burden,
                         "nonseptic_episodes_per_patient_day":
                             r.nonseptic_episodes_per_patient_day,
                         "repeat_notifications_per_septic_episode":
                             r.repeat_notifications_per_septic_episode,
                         "median_time_to_clear_h": r.median_time_to_clear_h})
        print(f"    {cand.describe():18} burden {burden:.3f}")
    finite = [e for e in ep_field if np.isfinite(e["burden_per_patient_day"])]
    best_ep = ap.EpisodePolicy(min(finite, key=lambda e: e["burden_per_patient_day"])
                               ["high_renotify_seconds"]) if finite else shipped_ep

    bv = ap.gate_burden(shipped_ep, best_ep, locked, op.high, op.medium,
                        n_boot=args.boot, seed=args.seed)
    print(f"\n  burden gate on the locked half: {bv.champion} vs {bv.challenger}")
    print(f"    burden {bv.champion_burden:.3f} -> {bv.challenger_burden:.3f}  "
          f"delta {bv.delta_burden:+.3f}  95% CI [{bv.ci_low:+.3f}, {bv.ci_high:+.3f}]")
    print(f"    first-alarm detection delta {bv.detection_delta:+.4f}  "
          f"false-alarm delta {bv.false_alarm_delta:+.4f}  (both must be 0 by construction)")
    print(f"\n  {'PROMOTED' if bv.promoted else 'REFUSED'} — {bv.reason}")

    if args.save:
        with open(MANIFEST) as fh:
            manifest = json.load(fh)
        fit = fitid.from_manifest(manifest, spec=f"wearable_limit{args.limit}",
                                  n_rows=int(len(y)),
                                  n_patients=int(len(set(g.tolist()))))
        families = [ap.FAMILY_LABELS[f] for f in
                    ("plain", "persistent", "integrator", "per_patient", "renotify")]
        note = ("No first-alarm policy beat a plain threshold. At a fixed detection "
                "rate the false-alarm rate is set by how well the score separates, and "
                "no post-processing recovers separation the score does not have."
                if not v.promoted else
                "A first-alarm policy beat a plain threshold at the shipped operating "
                "point; see the verdict block.")
        ap.save(v, shipped,
                families=families,
                verdict_note=note,
                operating_point={"high": op.high, "medium": op.medium,
                                 "detector": key,
                                 "source": "artifacts/operating_point.json"},
                fit=fit.to_json(),
                episode_semantics={
                    "controller": "backend.aegis.alert_episodes.AlertEpisodeController",
                    "high_renotify_seconds": shipped_ep.high_renotify_seconds,
                    "tier_thresholds": {"high": op.high, "medium": op.medium}},
                episode_metrics={
                    "shipped": {"policy": shipped_ep.describe(),
                                "nonseptic_episodes_per_patient_day":
                                    ship_ep_res.nonseptic_episodes_per_patient_day,
                                "repeat_notifications_per_septic_episode":
                                    ship_ep_res.repeat_notifications_per_septic_episode,
                                "escalation_recall": ship_ep_res.escalation_recall,
                                "median_time_to_clear_h": ship_ep_res.median_time_to_clear_h},
                    "field": ep_field},
                burden_verdict=bv.__dict__,
                candidate_field=[{"family": p.family,
                                  "params": {k2: round(float(v2), 6)
                                             for k2, v2 in p.params.items()},
                                  "false_alarm_rate": round(r.false_alarm_rate, 6),
                                  "detection_rate": round(r.detection_rate, 6)}
                                 for p, r in field])
        print(f"\nwrote {ap.ARTIFACT}")


if __name__ == "__main__":
    main()
