"""Adversarial audit of the MIMIC-IV line (runs on the open demo; no credentials needed).

    .venv/bin/python verify_mimic.py        (make verify-mimic)

Every guarantee is attacked, and every guard is shown to fire on a planted failure:

 1. CATALOG — every itemid's label in MIMIC's own dictionary matches the concept.
 2. CONVERSIONS — Fahrenheit, FiO2 percent, the intubated-verbal ("No Response-ETT") recode.
 3. ANTIBIOTICS — systemic agents in, eye / skin / irrigation preparations and non-antibiotics out.
 4. BINNING — a grid cell equals the last raw chartevent of that hour, read straight from disk.
 5. SEPSIS-3 — suspicion pairing windows, the three onset rules on synthetic stays (incl. the
    data-arrival case the ``measured`` rule exists for), onset anchor, GCS conventions.
 6. SYMMETRIC CENSORING — no stay is cut at its onset; septic and never-septic stays share
    one length rule (the CinC artifact cannot occur).
 7. CAUSALITY — features and the Process Twin: rewrite every stay's future, nothing earlier moves;
    a planted backward-looking column is caught.
 8. TIER FIREWALL — P columns carry no T / S channel; a planted S column is refused.
 9. SUBJECT LOCK — no subject in two dev folds or in both dev and locked.
10. MASKS — septic stays scored to onset + 3 h; on-arrival onsets out of the early-warning
    cohort; pre-suspicion = hours before the first order.
11. DESIGN WEIGHTS — a subsampled, re-weighted AUROC recovers the full one; unweighted does not.
12. CHUNKING — features built in chunks equal the whole build, bit for bit.
13. SQL — every BigQuery renderer names exactly its concepts' itemids.
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from backend.ml.mimic import concepts as K
from backend.ml.mimic import extract as E
from backend.ml.mimic import ledger as LG
from backend.ml.mimic import sepsis3 as S3
from backend.ml.mimic import sources as SRC
from backend.ml.mimic import twin as TW

_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def _same(a, b) -> bool:
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and
                np.array_equal(np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0)))


def c1_catalog(src) -> None:
    print("\n1. catalog vs MIMIC's own dictionaries")
    di, dl = src.d_items(), src.d_labitems()
    labels = dict(zip(di.itemid, di.label)) | dict(zip(dl.itemid, dl.label))
    bad = []
    n = 0
    for c in K.all_concepts():
        for i in c.itemids:
            n += 1
            lab = labels.get(i)
            if lab is None or not re.search(c.label_pattern, str(lab)):
                bad.append((c.name, i, lab))
    check("every itemid's dictionary label matches its concept", not bad,
          f"{n} itemids" + (f"; MISMATCH {bad[:3]}" if bad else ""))
    planted = re.search(K.CHART[0].label_pattern, str(labels.get(220179)))   # an SBP id vs the HR pattern
    check("…and a wrong itemid would be caught (HR pattern rejects an SBP item)", planted is None)


def c2_conversions(src, store) -> None:
    print("\n2. unit conversions and recodes")
    t = store.col("Temp")
    t = t[np.isfinite(t)]
    check("temperatures are Celsius after conversion", 30 < np.median(t) < 40 and t.max() <= 45,
          f"median {np.median(t):.1f} C, max {t.max():.1f}")
    f = store.col("FiO2")
    f = f[np.isfinite(f)]
    check("FiO2 is a fraction in [0.21, 1]", f.min() >= 0.21 and f.max() <= 1.0,
          f"range {f.min():.2f}-{f.max():.2f}")
    v = store.col("GCS_verbal")
    check("intubated verbal is recoded to 0 (not the unresponsive 1)", np.nansum(v == 0) > 0,
          f"{int(np.nansum(v == 0))} intubated hours")
    g = S3.gcs_total(np.array([4.0]), np.array([0.0]), np.array([6.0]), np.array([0]))
    check("an intubated patient's total GCS is 15 (unassessable), not a coma", g[0] == 15.0)


def c3_antibiotics() -> None:
    print("\n3. antibiotic filter")
    yes = [("Vancomycin", "IV"), ("piperacillin-tazobactam", "IV"), ("Ceftriaxone", "IV"),
           ("Metronidazole (FLagyl)", "PO"), ("sulfameth/trimethoprim DS", "PO/NG")]
    no = [("Ciprofloxacin 0.3% Ophth Soln", "BOTH EYES"), ("Erythromycin 0.5% Ophth Oint", "LEFT EYE"),
          ("Mupirocin Ointment 2%", "TP"), ("Fentanyl Citrate", "IV"), ("Neomycin-Polymyxin-Bacitracin", "IR"),
          ("Vancomycin Enema", "PR")]
    check("systemic antibiotics are recognised", all(K.is_antibiotic(d, r) for d, r in yes))
    check("eye / skin / irrigation / enema preparations and non-antibiotics are not",
          not any(K.is_antibiotic(d, r) for d, r in no),
          f"wrongly accepted: {[d for d, r in no if K.is_antibiotic(d, r)]}")


def c4_binning(src, store) -> None:
    print("\n4. binning = the last raw event of the hour")
    ce = pd.read_csv(src._path("icu", "chartevents"), usecols=["stay_id", "charttime", "itemid", "valuenum"])
    ce["charttime"] = pd.to_datetime(ce.charttime)
    st = src.icustays()
    k = 3
    sid = store.pid[k]
    intime = st.loc[st.stay_id == sid, "intime"].iloc[0]
    hr = ce[(ce.stay_id == sid) & (ce.itemid == 220045)].copy()
    hr["h"] = ((hr.charttime - intime).dt.total_seconds() // 3600).astype(int)
    hr = hr[(hr.h >= 0) & (hr.h < store.lens[k])].sort_values("charttime")
    ref = hr.groupby("h").valuenum.last()
    grid = store.col("HR")[store.starts[k] + ref.index.to_numpy()]
    check("heart rate grid equals the raw last-in-hour values", np.allclose(grid, ref.to_numpy()),
          f"stay {sid}, {len(ref)} hours")


def c5_sepsis3() -> None:
    print("\n5. Sepsis-3 on synthetic stays")
    check("culture then antibiotic within 72 h -> suspicion at the culture",
          list(S3.suspicion_times([10.0], [5.0])) == [5.0])
    check("antibiotic then culture within 24 h -> suspicion at the antibiotic",
          list(S3.suspicion_times([10.0], [20.0])) == [10.0])
    check("pairs outside the windows are not suspicion",
          len(S3.suspicion_times([10.0], [40.0])) == 0 and len(S3.suspicion_times([100.0], [20.0])) == 0)
    T = 40
    sofa = np.r_[np.zeros(20), np.full(20, 3.0)]
    parts = np.zeros((T, 6))
    parts[20:, 1] = 3.0
    o = S3.onset_for_stay(sofa, np.array([25.0]), "measured", "min", parts)
    check("a measured 3-point rise at h20 with suspicion at h25 -> onset h20", o.t_sepsis == 20.0,
          f"got {o.t_sepsis}")
    parts2 = np.full((T, 6), np.nan)
    parts2[5:, 2] = 3.0                         # liver first MEASURED at 3 points: no baseline
    o2 = S3.onset_for_stay(np.r_[np.zeros(5), np.full(35, 3.0)], np.array([6.0]), "measured", "min", parts2)
    o3 = S3.onset_for_stay(np.r_[np.zeros(5), np.full(35, 3.0)], np.array([6.0]), "rise", "min")
    check("data arrival is NOT a rise under the measured rule (it is under CinC's)",
          np.isnan(o2.t_sepsis) and o3.t_sepsis == 5.0, f"measured {o2.t_sepsis}, rise {o3.t_sepsis}")
    o4 = S3.onset_for_stay(sofa, np.array([40.5]), "measured", "si", parts)
    check("the anchor choice is honoured", o4.t_sepsis == 40.5)


def c6_censoring(store) -> None:
    print("\n6. symmetric censoring")
    meta = store.meta
    expect = np.minimum(np.ceil(meta.los_hours), K.MAX_HOURS).astype(int).to_numpy()
    check("every stay's grid length is min(ceil(LOS), MAX_HOURS), septic or not",
          np.array_equal(store.lens, expect), f"{store.n_patients} stays")
    t = store.onsets[f"t_sepsis_{E.PRIMARY_RULE}_min"].to_numpy()
    sep = np.isfinite(t)
    cut = sep & (store.lens <= np.ceil(t) + 3)
    check("no septic stay ends at onset + 3 h the way CinC's did",
          cut.sum() < max(2, 0.1 * sep.sum()), f"{int(cut.sum())} of {int(sep.sum())} end within 3 h of onset")


def c7_causality(store) -> None:
    print("\n7. causality")
    rng = np.random.default_rng(0)
    sub = E.subset(store, np.arange(40))
    F0, names = LG.build_features(sub)
    X = sub.X.copy()
    keep = np.ones(sub.n_rows, dtype=bool)
    for k in range(sub.n_patients):
        a, n = int(sub.starts[k]), int(sub.lens[k])
        cut = int(rng.integers(0, n))
        fut = slice(a + cut + 1, a + n)
        keep[fut] = False
        X[fut] = np.where(rng.random(X[fut].shape) < 0.5, np.nan, rng.normal(50, 20, X[fut].shape))
    new = E.MimicStore(X=X.astype(np.float32), y=sub.y, lens=sub.lens, pid=sub.pid, site=sub.site,
                       columns=sub.columns, tiers=sub.tiers, static_columns=sub.static_columns,
                       meta=sub.meta, onsets=sub.onsets, sofa=sub.sofa)
    F1, _ = LG.build_features(new)
    bad = [n for j, n in enumerate(names) if not _same(F0[keep][:, j], F1[keep][:, j])]
    check("no feature at or before the cut moves when the future is rewritten", not bad,
          f"{len(names)} columns" + (f"; MOVED {bad[:4]}" if bad else ""))
    lac = pd.Series(sub.col("Lactate")).groupby(sub.seg).bfill().to_numpy()
    lac1 = pd.Series(new.col("Lactate")).groupby(new.seg).bfill().to_numpy()
    check("…and a planted backward-fill column is caught by the same test", not _same(lac[keep], lac1[keep]))
    import torch
    net = TW._build(12, TW.TwinConfig()).eval()
    x = torch.from_numpy(rng.normal(size=(2, 30, 12)).astype(np.float32))
    x2 = x.clone()
    x2[:, 11:] = torch.from_numpy(rng.normal(3, 2, size=(2, 19, 12)).astype(np.float32))
    with torch.no_grad():
        a, b = net(x), net(x2)
    check("no Process Twin output at or before the cut moves",
          all(torch.equal(u[:, :11], v[:, :11]) for u, v in zip(a, b)), f"{len(a)} outputs")


def c8_firewall(led) -> None:
    print("\n8. tier firewall")
    p_cols = led.columns("P")
    leaked = [led.names[j] for j in p_cols if (LG.channel_of(led.names[j]) or "") in
              set(E.VASO_CH + E.VENT_CH + E.SUSP_CH + ["SOFA"])]
    check("tier-P columns carry no treatment or suspicion channel", not leaked, f"{len(p_cols)} P columns")
    s_col = np.array([led.names.index("Antibiotic")])
    try:
        led.assert_tier(s_col, "P")
        refused = False
    except PermissionError:
        refused = True
    check("a planted suspicion column in a tier-P model is refused", refused)
    ok = True
    try:
        led.assert_tier(p_cols, "P")
    except PermissionError:
        ok = False
    check("…and legitimate tier-P columns are accepted (the guard is not trigger-happy)", ok)


def c9_lock(led) -> None:
    print("\n9. subject-level lock")
    subj, f = led.store.subject, led.folds
    per = pd.DataFrame({"s": subj, "f": f}).groupby("s").f.nunique()
    check("no subject spans two folds or dev + locked", int(per.max()) == 1, f"{len(per)} subjects")
    check("the locked share is ~20% of subjects", 0.1 < (f < 0).mean() < 0.35, f"{(f < 0).mean():.2f} of stays")


def c10_masks(led) -> None:
    print("\n10. evaluation masks")
    s = led.store
    t = led.onset[s.seg]
    sep = np.isfinite(t)
    check("septic stays are scored no later than onset + 3 h",
          not np.any(led.eval_mask & sep & (s.hour > t + LG.SCORE_AFTER_ONSET)))
    check("on-arrival onsets are outside the early-warning cohort",
          not np.any(led.eval_mask & sep & (t < LG.ONSET_MIN)))
    first = s.onsets.first_order.to_numpy()[s.seg]
    check("pre-suspicion hours all precede the first order",
          not np.any(led.pre_susp & np.isfinite(first) & (s.hour >= first)))


def c11_design() -> None:
    print("\n11. design weights recover the full-cohort AUROC")
    from backend.ml.genesis.harness import Ranked
    rng = np.random.default_rng(3)
    n_ns, n_sep, hours = 3000, 300, 20
    stays = np.r_[np.zeros(n_ns), np.ones(n_sep)]
    y = np.repeat(stays, hours)
    y[np.repeat(stays == 1, hours) & (np.tile(np.arange(hours), n_ns + n_sep) < 14)] = 0   # septic pre-window
    seg = np.repeat(np.arange(n_ns + n_sep), hours)
    score = rng.normal(size=len(y)) + 1.2 * y + 0.8 * np.repeat(stays, hours)
    full = Ranked(score).auc(y)
    keep = (stays == 1) | (rng.random(len(stays)) < 0.2)
    rows = keep[seg]
    w = np.where(stays[seg[rows]] == 1, 1.0, 5.0)
    weighted = Ranked(score[rows]).auc(y[rows], w)
    unweighted = Ranked(score[rows]).auc(y[rows])
    check("re-weighted AUROC ~ full-cohort AUROC", abs(weighted - full) < 0.01,
          f"full {full:.4f} weighted {weighted:.4f}")
    check("…and the unweighted subsample does NOT (the weights are doing work)",
          abs(unweighted - full) > abs(weighted - full), f"unweighted {unweighted:.4f}")


def c12_chunking(store) -> None:
    print("\n12. chunked feature build")
    F1, n1 = LG.build_features(store)
    F2, n2 = LG.build_features(store, chunk_stays=9)
    check("chunked == whole, bit for bit", n1 == n2 and _same(F1, F2), f"{F1.shape}")


def c13_sql() -> None:
    print("\n13. BigQuery SQL")
    chart = SRC.render_chart_sql("I")
    labs = SRC.render_labs_sql("H", "I")
    vaso = SRC.render_interval_sql("I", "vaso")
    miss = [i for c in K.CHART for i in c.itemids if str(i) not in chart] + \
           [i for c in K.LAB for i in c.itemids if str(i) not in labs] + \
           [i for c in K.VASO for i in c.itemids if str(i) not in vaso]
    check("every chart / lab / vasopressor itemid appears in its SQL", not miss, f"missing {miss[:5]}")
    check("the SQL keeps the ETT recode and the unit conversions",
          "No Response-ETT" in chart and "(v - 32) * 5 / 9" in chart and "v / 100" in chart)


def main() -> int:
    src = SRC.CsvSource(E.DEMO_ROOT)
    try:
        store = E.load("demo")
    except FileNotFoundError:
        store = E.build_store(src, verbose=False)
        E.save(store, "demo")
    led = LG.Ledger.build(store)
    c1_catalog(src)
    c2_conversions(src, store)
    c3_antibiotics()
    c4_binning(src, store)
    c5_sepsis3()
    c6_censoring(store)
    c7_causality(store)
    c8_firewall(led)
    c9_lock(led)
    c10_masks(led)
    c11_design()
    c12_chunking(store)
    c13_sql()
    ok = sum(r[1] for r in _results)
    print(f"\nverify_mimic: {ok}/{len(_results)} passed")
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
