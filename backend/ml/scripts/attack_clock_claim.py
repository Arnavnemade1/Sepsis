"""Adversarial audit of this project's own clock-identifiability claim.

    PYTHONPATH=. python -m backend.ml.scripts.attack_clock_claim

THE CLAIM UNDER ATTACK
----------------------
Stated after the supervision-density work, and proposed as the seed of a field-level
audit instrument:

    "Within a septic stay the CinC label is a monotone function of time. The bare hour
     index scores a within-patient AUROC of 1.0000, so any model that can recover
     elapsed time gets timing skill for free — and a large share of published sepsis
     early-warning performance may be patient classification plus a clock."

Before anything is built on that, it gets attacked. Five attacks, each of which could
falsify or narrow it, run against the real cohort. The strong version is only worth
keeping if it survives all five, and the honest expectation going in is that it will
not survive intact.

THE ATTACKS
-----------
1. **Tautology.** Is "hour index gives within-patient AUROC 1.0" an empirical finding at
   all, or is it definitionally true and therefore vacuous as stated?
2. **Pair budget.** Pooled AUROC averages over pairs. What FRACTION of those pairs are
   within-septic-patient comparisons — the only ones a clock can touch? If that share is
   small, the clock cannot explain much pooled AUROC no matter how perfectly it ranks.
3. **Reliance.** Do real models actually lean on the clock? Delete every clock-bearing
   feature and measure what the incumbent loses.
4. **Legitimacy.** Is elapsed time *artefactually* predictive, or is it real clinical
   information? A longer stay genuinely means a sicker patient. If ICULOS predicts sepsis
   BETWEEN patients too, then using it is not cheating and calling it a clock artifact is
   unfair.
5. **Lead time.** The narrower claim: does a timing-blind detector manufacture an
   impressive-looking lead time purely by alarming somewhere in a flagged stay?
"""
from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np

from backend.ml.eval import metrics
from backend.ml.eval.bench import Benchmark

VERDICTS: List[Tuple[str, str]] = []


def verdict(name: str, text: str) -> None:
    VERDICTS.append((name, text))
    print(f"  >>> {name}: {text}", flush=True)


def main() -> int:
    b = Benchmark.load("dev", n_folds=4)
    ds = b.ds
    y, g, h, o = ds.y, ds.groups, ds.hour_index, ds.onset_hour
    septic = np.isin(g, np.array(sorted({x for x in set(g[y == 1].tolist())})))
    print(f"cohort: {len(y):,} hours | {len(set(g.tolist())):,} patients | "
          f"{septic.sum():,} hours in septic stays | prevalence {y.mean():.4f}\n")

    # ---------------------------------------------------------------- ATTACK 1
    print("ATTACK 1 — is the headline number a tautology?")
    # For a septic stay the label is 1 exactly when t >= onset-6. So it is a STEP in t,
    # and any strictly increasing function of t separates the two classes perfectly.
    # Verify that this is what is happening rather than something empirical.
    per_pt, viol = [], 0
    for pid in sorted({x for x in set(g[septic].tolist())}):
        m = g == pid
        yy, hh = y[m], h[m]
        if len(set(yy.tolist())) < 2:
            continue
        per_pt.append(metrics.auroc(yy, hh.astype(float)))
        # is the label really a monotone step in hour?
        order = np.argsort(hh)
        ys = yy[order]
        if np.any(np.diff(ys.astype(int)) < 0):
            viol += 1
    per_pt = np.array(per_pt)
    print(f"    within-patient AUROC of the bare hour index: "
          f"mean {per_pt.mean():.4f}, min {per_pt.min():.4f} over {len(per_pt)} stays")
    print(f"    stays where the label is NOT a monotone step in time: {viol}")
    if viol == 0 and per_pt.min() > 0.9999:
        verdict("ATTACK 1", "LANDS. The label IS a monotone step in time within every "
                            "septic stay, so AUROC 1.0 is definitional, not empirical. "
                            "Quoting it as a discovery is wrong.")
    else:
        verdict("ATTACK 1", f"survives — {viol} stays violate monotonicity, so the "
                            f"number is not purely definitional.")

    # ---------------------------------------------------------------- ATTACK 2
    print("\nATTACK 2 — how much of pooled AUROC can a clock even touch?")
    n_pos = int((y == 1).sum())
    neg_septic = int(((y == 0) & septic).sum())
    neg_nonseptic = int(((y == 0) & ~septic).sum())
    total_pairs = n_pos * (neg_septic + neg_nonseptic)
    within_pairs = n_pos * neg_septic
    frac = within_pairs / total_pairs if total_pairs else float("nan")
    print(f"    positive hours              {n_pos:,}")
    print(f"    negative hours, septic stay {neg_septic:,}   <- the only clock-touchable pairs")
    print(f"    negative hours, other stay  {neg_nonseptic:,}")
    print(f"    within-patient share of all discordant pairs: {frac:.1%}")
    if frac < 0.15:
        verdict("ATTACK 2", f"LANDS HARD. Only {frac:.1%} of the pairs pooled AUROC "
                            f"averages over are within-patient. A perfect clock can move "
                            f"pooled AUROC by at most about that much, so 'published "
                            f"performance is mostly a clock' is FALSE as stated.")
    else:
        verdict("ATTACK 2", f"partially survives — {frac:.1%} of pairs are clock-touchable.")

    # ---------------------------------------------------------------- ATTACK 3
    print("\nATTACK 3 — do real models actually lean on the clock?")
    names = ds.feature_names
    clockish = [i for i, n in enumerate(names)
                if n in ("ICULOS", "HospAdmTime")
                or n.startswith("burden_")            # cumulative counters = time proxies
                or n.endswith(("__emin", "__emax"))]  # expanding extremes = time proxies
    print(f"    clock-bearing features: {len(clockish)} of {len(names)} "
          f"({', '.join(names[i] for i in clockish[:5])}...)")
    keep = np.array([i for i in range(len(names)) if i not in set(clockish)])
    from sklearn.ensemble import HistGradientBoostingClassifier

    def oof(cols: np.ndarray) -> np.ndarray:
        p = np.full(len(y), np.nan)
        for tr, te in b.folds[:2]:
            m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                               class_weight="balanced", random_state=0)
            m.fit(ds.X[np.ix_(tr, cols)], y[tr])
            p[te] = m.predict_proba(ds.X[np.ix_(te, cols)])[:, 1]
        return p

    p_full = oof(np.arange(len(names)))
    p_noclk = oof(keep)
    ok = ~np.isnan(p_full)
    a_full = metrics.auroc(y[ok], p_full[ok])
    a_nock = metrics.auroc(y[ok], p_noclk[ok])
    print(f"    pooled AUROC with clock features    {a_full:.4f}")
    print(f"    pooled AUROC with them deleted      {a_nock:.4f}   delta {a_nock-a_full:+.4f}")
    if abs(a_nock - a_full) < 0.02:
        verdict("ATTACK 3", f"LANDS. Deleting every clock-bearing feature costs "
                            f"{a_full-a_nock:+.4f} AUROC. The incumbent is not "
                            f"meaningfully relying on the clock.")
    else:
        verdict("ATTACK 3", f"survives — removing the clock costs {a_full-a_nock:.4f}.")

    # ---------------------------------------------------------------- ATTACK 4
    print("\nATTACK 4 — is elapsed time a cheat, or real clinical information?")
    # If longer stays are genuinely septic more often, ICULOS is legitimate evidence and
    # calling its use an artifact is unfair.
    pts = np.array(sorted(set(g.tolist())))
    stay_len = np.array([h[g == p].max() for p in pts], dtype=float)
    is_sep = np.array([int(y[g == p].max()) for p in pts])
    a_between = metrics.auroc(is_sep, stay_len)
    print(f"    BETWEEN patients, stay length predicts 'ever septic': AUROC {a_between:.4f}")
    print(f"      median stay: septic {np.median(stay_len[is_sep==1]):.0f}h vs "
          f"non-septic {np.median(stay_len[is_sep==0]):.0f}h")
    if a_between > 0.6:
        verdict("ATTACK 4", f"LANDS. Stay length predicts sepsis between patients at "
                            f"AUROC {a_between:.3f}. Elapsed time carries real clinical "
                            f"signal, so a model using it is not simply cheating.")
    else:
        verdict("ATTACK 4", f"survives — stay length is near chance between patients "
                            f"({a_between:.3f}), so its within-patient power is artifactual.")

    # ---------------------------------------------------------------- ATTACK 5
    print("\nATTACK 5 — the narrow claim: is reported LEAD TIME a clock artifact?")
    # Simulate a detector that has perfect patient-level discrimination and ZERO timing
    # skill: it identifies septic patients, then alarms at a uniformly random hour of the
    # pre-onset window. If that manufactures a long lead time, lead time as usually
    # reported does not evidence early warning.
    rng = np.random.default_rng(0)
    leads = []
    for pid in sorted({x for x in set(g[septic].tolist())}):
        m = g == pid
        onset = int(o[m][0])
        if onset < 0:
            continue
        pre = h[m][h[m] <= onset]
        if pre.size < 2:
            continue
        leads.append(onset - float(rng.choice(pre)))
    leads = np.array(leads)
    print(f"    timing-BLIND detector, alarm placed uniformly at random before onset:")
    print(f"      median apparent lead time {np.median(leads):.1f}h "
          f"over {len(leads)} septic stays")
    print(f"    for comparison, this project has reported lead times of 26-40h")
    if np.median(leads) > 10:
        verdict("ATTACK 5", f"claim SURVIVES. A detector with zero timing skill still "
                            f"shows a median lead of {np.median(leads):.1f}h purely from "
                            f"stay geometry, so a long lead time is not by itself "
                            f"evidence of early warning.")
    else:
        verdict("ATTACK 5", f"LANDS — random alarm placement yields only "
                            f"{np.median(leads):.1f}h, so long leads do mean something.")

    # -------------------------------------------------------------- ATTACK 5b
    print("\nATTACK 5b — the null that matches how detectors ACTUALLY alarm")
    # Attack 5 placed the alarm uniformly at random inside the pre-onset window, and a
    # real detector does not do that: it fires at the FIRST crossing of a threshold,
    # which for any low threshold lands near the start of a stay. That is a far harsher
    # null, and comparisons must be at MATCHED DETECTION RATE — a detector that alarms
    # on more stays reaches further back on average for reasons unrelated to skill.
    sep_ids = sorted({x for x in set(g[septic].tolist())})
    rng2 = np.random.default_rng(0)

    def lead_curve(score: np.ndarray) -> Dict[float, float]:
        out: Dict[float, float] = {}
        for q in np.linspace(0.80, 0.999, 60):
            thr = float(np.quantile(score, q))
            leads = []
            for pid in sep_ids:
                m = g == pid
                onset = int(o[m][0])
                if onset < 0:
                    continue
                hh, ss = h[m], score[m]
                pre = hh <= onset
                if pre.sum() < 2:
                    continue
                fired = np.flatnonzero(ss[pre] >= thr)
                if fired.size:
                    leads.append(onset - hh[pre][fired[0]])
            if leads:
                out[round(len(leads) / len(sep_ids), 2)] = float(np.median(leads))
        return out

    c_rand = lead_curve(rng2.random(len(y)))
    c_real = lead_curve(p_full)
    print(f"    {'detection':>10} {'RANDOM score':>14} {'real model':>12}")
    gaps = []
    for det in (0.30, 0.40, 0.50, 0.60):
        def near(c):
            ks = [k for k in c if abs(k - det) < 0.06]
            return float(np.median([c[k] for k in ks])) if ks else float("nan")
        r, m_ = near(c_rand), near(c_real)
        if np.isfinite(r) and np.isfinite(m_):
            gaps.append(m_ - r)
            print(f"    {det:>10.0%} {r:>13.1f}h {m_:>11.1f}h")
    if gaps and abs(float(np.median(gaps))) < 5.0:
        verdict("ATTACK 5b", f"the STRONG form holds. At matched detection a pure "
                             f"RANDOM score produces the same median lead as the real "
                             f"model (median gap {np.median(gaps):+.1f}h). Lead time as "
                             f"universally reported carries no early-warning information "
                             f"— including every lead time THIS project has published.")
    else:
        verdict("ATTACK 5b", f"real model beats the random null by "
                             f"{np.median(gaps):+.1f}h — lead time is informative.")

    print("\n" + "=" * 74)
    print("VERDICTS")
    for n, t in VERDICTS:
        print(f"  {n}: {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
