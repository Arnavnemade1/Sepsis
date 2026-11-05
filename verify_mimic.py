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

The harness layers (docs/MIMIC.md §4-7), each attacked with a planted failure:

14. VIEWS — an edited patient keeps its outcome; derived channels follow their inputs (a
    stale SOFA is caught); views reproduce the ledger bit for bit.
15. CARE-PROCESS TRANSPLANT — in a world whose label lives only in the measurement
    schedule, the transplant attributes it to process; in a world whose label lives only in
    values under identical schedules, to physiology — exactly; retrained inside each world,
    the same split of what is learnable; urine moves as a rate.
16. ADMISSION AUDIT — a clean model is admitted; a look-ahead model, a stay-length model and
    a test-set-normalising model are each refused by the attack aimed at them.
17. INFORMATION FLOW — the ETT recode is found as a covert ventilation channel; a planted
    vasopressor copy is flagged, planted noise is not; the strict tier drops exactly the
    flagged channel and every composite built on it.
18. CROSS-EXAMINATION — a suspicion reader fails against infection-only stays and a sickness
    detector against organ-only stays; the clock scores exactly 0.5 at every clock-matched
    horizon; a clinician-score copy scores exactly 0.5 beyond the clinicians; soft-label AUROC
    equals its brute-force definition; a planted failing subgroup is flagged; alerts respect
    their budget.
19. TRUST — tampering with, deleting or reordering a chain entry is caught; the code
    fingerprint moves with the code; the power gate refuses an unanswerable pre-registration;
    a confirmation is spent once and code drift needs a recorded deviation; the Ladder resists
    the boosting attack that climbs a naive holdout, and never writes a raw lock score.
21. INFORMATION v3 — planted source tables: urinalysis text coding (an unknown code is
    dropped), sedation / dialysis intervals, fluid and blood volumes, bedside antibiotic
    administrations ("Rewritten" rows ignored); comorbidities read only admissions discharged
    BEFORE the stay (a sepsis code in the current or a later admission is ignored); one source
    and one service per stay; every renderer carries the new items.
22. LABEL STRUCTURE — block tiers follow their source channels; the suspicion pair completes
    exactly where Sepsis-3's windows say; urine per kg and norepinephrine equivalents on planted
    values; trimming septic stays after onset changes no feature on a kept row.
23. AUROC MACHINERY — patient-level AUROC equals its brute force; "all stays" includes sepsis on
    arrival; the stacker never sees the labels it scores and at infinite shrinkage IS the
    product; tuning reads dev rows only; the headline's ideal number is its settings' maximum
    and never comes without the primary; the planner's estimate matches a real build and
    refuses an impossible budget; the parity check passes identical sources and flags a
    planted one-hour shift.
24. RESEARCH LEVERS — the bedside exam's worst-of-hour coding matches the raw charting; the
    hour's lowest MAP matches the raw readings and SOFA reads it (a transient dip scores);
    culture specimen groups and antibiotic spectrum (oral vancomycin is not anti-MRSA);
    signature terms equal their brute-force iterated sums; the distilled student reads tier P
    only and no teacher that wrote its targets ever trained on the fold it is scored on.
20. TEMPORAL MODEL (Process Twin v2) — hazard targets follow discrete-time survival
    semantics (an event, survival, censoring, not-at-risk) and the per-bin loss IS the
    survival likelihood; the risk curve and the composed organ outlook equal their brute-force
    definitions; the twin reads a 30-h lag a windowed booster cannot; the physiology twin
    keeps a physiological signal and drops a care-process one; its randomness is a pure
    function of (stay, hour, channel), and a sequential generator in its place is refused by
    the admission audit.
"""
from __future__ import annotations

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from backend.ml.mimic import concepts as K
from backend.ml.mimic import counterfactual as CF
from backend.ml.mimic import crossexam as CX
from backend.ml.mimic import extract as E
from backend.ml.mimic import infoflow as IF
from backend.ml.mimic import ledger as LG
from backend.ml.mimic import sepsis3 as S3
from backend.ml.mimic import sources as SRC
from backend.ml.mimic import trust as TR
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
    net = TW._build(12, 9, TW.TwinConfig()).eval()
    xv = torch.from_numpy(rng.normal(size=(2, 30, 12)).astype(np.float32))
    xp = torch.from_numpy(rng.normal(size=(2, 30, 9)).astype(np.float32))
    xv2, xp2 = xv.clone(), xp.clone()
    xv2[:, 11:] = torch.from_numpy(rng.normal(3, 2, size=(2, 19, 12)).astype(np.float32))
    xp2[:, 11:] = torch.from_numpy(rng.normal(3, 2, size=(2, 19, 9)).astype(np.float32))
    with torch.no_grad():
        a, b = net(xv, xp), net(xv2, xp2)
    check("no Process Twin output at or before the cut moves (every head, both streams)",
          all(torch.equal(a[k][:, :11], b[k][:, :11]) for k in a), f"{len(a)} outputs")


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
    # full MIMIC-IV builds float16, whose top is 65,504: CK, ferritin and NT-proBNP pass it.
    # Plant a CK a million times too large; neither path may write an inf.
    import copy
    big = copy.copy(store)
    big.X = np.array(store.X, copy=True)
    j = store.columns.index("CK")
    big.X[:, j] = big.X[:, j] * 1e6
    H1, m1 = LG.build_features(big, dtype=np.float16)
    H2, _ = LG.build_features(big, chunk_stays=9, dtype=np.float16)
    ck = H1[:, m1.index("CK")].astype(np.float64)
    # every fit forces column-wise histograms to save memory; that must not change a model
    import lightgbm as lgb
    from backend.ml.genesis import learners as GL
    rows = np.arange(store.n_rows)
    yb = np.nan_to_num(store.y.astype(float))
    seg = store.seg
    a = GL.fit_lgb(F1, rows, yb, seg, params={"num_leaves": 15}, rounds=40, early_stop=10)
    real = lgb.train
    lgb.train = lambda p, *x, **k: real({**p, "force_col_wise": False}, *x, **k)
    try:
        b = GL.fit_lgb(F1, rows, yb, seg, params={"num_leaves": 15}, rounds=40, early_stop=10)
    finally:
        lgb.train = real
    check("column-wise histograms change no prediction (only memory)",
          np.array_equal(a.predict(F1, rows), b.predict(F1, rows)) and a.best_iter == b.best_iter,
          f"{len(rows):,} rows, {a.best_iter} trees")
    # a design-weighted fit whose validation slice is all weight 1 (e.g. all septic stays): LightGBM
    # dropped those weights and read the slice as weight 0, so AUC said 1.0 every round and a
    # cross-entropy metric aborted. All 1s and all 2s are the same metric, so the models must match
    fit_pos, val_pos = GL.grouped_holdout(seg, 0.1, 0, None)
    w1 = np.where(np.arange(len(rows)) % 3 == 0, 4.0, 1.0)
    w1[val_pos] = 1.0
    w2 = w1.copy()
    w2[val_pos] = 2.0
    kw = dict(params={"num_leaves": 15}, rounds=200, early_stop=10)
    f1 = GL.fit_lgb(F1, rows, yb, seg, weight=w1, **kw)
    f2 = GL.fit_lgb(F1, rows, yb, seg, weight=w2, **kw)
    try:
        GL.fit_lgb(F1, rows, yb, seg, weight=w1, params={"num_leaves": 15, "objective": "cross_entropy"},
                   rounds=20, early_stop=5, feval_metric="cross_entropy")
        xent = "ran"
    except Exception as e:
        xent = str(e)[:50]
    check("a validation slice of all-1 weights early-stops like any constant weight, and cross-entropy runs",
          f1.best_iter == f2.best_iter and xent == "ran"
          and np.array_equal(f1.predict(F1, rows, raw=True, calibrated=False), f2.predict(F1, rows, raw=True, calibrated=False)),
          f"best iteration {f1.best_iter} vs {f2.best_iter}; cross-entropy {xent}")

    # the full run's path: sampled, trimmed, chunked into a float16 map on disk. It must equal the
    # exact saturated cast of a whole float32 build of the same store, cell for cell
    import tempfile
    sub, _ = E.sample_design(store, 0.5)
    sub = E.trim_septic(sub)
    W32, wn = LG.build_features(sub)
    d = tempfile.mkdtemp()
    try:
        M16, mn = LG.build_features(sub, chunk_stays=7, memmap_path=os.path.join(d, "f.npy"), dtype=np.float16)
        ref = np.clip(W32, -65504.0, 65504.0).astype(np.float16)
        check("the full run's path (sampled, trimmed, chunked float16 on disk) == the saturated cast of a whole float32 build",
              wn == mn and isinstance(M16, np.memmap) and np.array_equal(np.asarray(M16), ref, equal_nan=True),
              f"{sub.n_rows:,} of {store.n_rows:,} rows, {len(mn)} features")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    check("a float16 build saturates at 65,504 instead of writing inf, whole or chunked",
          not np.isinf(H1).any() and not np.isinf(H2).any() and np.nanmax(ck) == 65504.0
          and np.array_equal(H1, H2, equal_nan=True), f"CK max {np.nanmax(ck):g}")


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


# --------------------------------------------------------------------------- #
# The harness layers
# --------------------------------------------------------------------------- #
def _pool(store, c):
    v = store.col(c)
    return v[np.isfinite(v)]


def _window(store):
    t = store.onsets[f"t_sepsis_{E.PRIMARY_RULE}_min"].to_numpy()[store.seg]
    return np.isfinite(t) & (store.hour >= t - 6) & (store.hour <= t + 3)


def _booster():
    from backend.ml.mimic.run import SCALES, booster
    return booster(SCALES["demo"])


def c14_views(store, led) -> None:
    print("\n14. views and derived channels")
    r = E.rederive(store, store.X)
    check("re-deriving an unedited grid reproduces every channel bit for bit", _same(r.X, store.X))
    stays = np.arange(0, store.n_patients, 7)
    v = led.view(stays)
    check("a view's features equal the ledger's rows bit for bit",
          _same(v.F, np.asarray(led.F[v.parent_rows])), f"{len(stays)} stays, {v.store.n_rows} rows")
    j = store.columns.index("Creatinine")
    k = int(np.argmax([np.isfinite(store.X[store.starts[i]:store.starts[i] + store.lens[i], j]).sum()
                       for i in range(store.n_patients)]))
    sub = E.subset(store, np.array([k]))
    Xe = sub.X.copy()
    Xe[np.isfinite(Xe[:, j]), j] = 6.0                        # renal failure, 4 SOFA points
    fresh = E.rederive(sub, Xe)
    stale = E.MimicStore(X=Xe, y=sub.y, lens=sub.lens, pid=sub.pid, site=sub.site, columns=sub.columns,
                         tiers=sub.tiers, static_columns=sub.static_columns, meta=sub.meta,
                         onsets=sub.onsets, sofa=sub.sofa)
    rise = fresh.col("SOFA_P") - sub.col("SOFA_P")
    check("an edited input reaches its derived channel (creatinine 6.0 -> SOFA_P rises)",
          rise.max() >= 1 and rise.min() >= 0, f"max +{rise.max():.0f}")
    check("…and a grid edited WITHOUT re-deriving is caught holding a stale SOFA_P",
          _same(stale.col("SOFA_P"), sub.col("SOFA_P")) and not _same(fresh.col("SOFA_P"), stale.col("SOFA_P")))
    bank = CF.DonorBank.from_store(store, (stays + 3) % store.n_patients)
    cf = led.view(stays, CF.transplant(store, stays, np.arange(len(stays)), bank, False, True, CF.reference(led)))
    check("a counterfactual patient keeps its outcome (labels, masks, onsets untouched)",
          np.array_equal(cf.y, v.y) and np.array_equal(cf.eval_mask, v.eval_mask)
          and _same(cf.onset, v.onset) and not _same(cf.store.X, v.store.X))


def c15_transplant(store) -> None:
    print("\n15. Care-Process Transplant on planted worlds")
    from backend.ml.mimic.ledger import Candidate
    rng = np.random.default_rng(11)
    j = {c: i for i, c in enumerate(store.columns)}
    win = _window(store)
    # world A: every value is label-independent noise; only the SCHEDULE carries the label
    XA = store.X.copy()
    for c in CF.STATE_CH:
        m = np.isfinite(XA[:, j[c]])
        pl = _pool(store, c)
        if len(pl):
            XA[m, j[c]] = rng.choice(pl, int(m.sum()))
    XA[win, j["Lactate"]] = rng.choice(_pool(store, "Lactate"), int(win.sum()))   # hourly draws near onset
    ledA = LG.Ledger.build(E.rederive(store, XA))
    fmA = ledA.fit_oof(Candidate("A", "P", _booster()), verbose=False)
    a = CF.process_physiology(ledA, [fmA], n_boot=50)["A"]
    check("schedule-only world: the transplant attributes the signal to PROCESS",
          a["phi_process"]["value"] > 0.3 and a["phi_physiology"]["value"] < 0.05
          and a["process_reliance"]["value"] > 0.3,
          f"phi_process {a['phi_process']['value']:.3f}, phi_physiology {a['phi_physiology']['value']:.3f}, "
          f"reliance {a['process_reliance']['value']:.3f}")
    # world B: everyone measured every hour; only lactate's VALUE carries the label
    XB = store.X.copy()
    for c in CF.STATE_CH:
        pl = _pool(store, c)
        XB[:, j[c]] = rng.choice(pl, store.n_rows) if len(pl) else 0.0
    XB[:, j["Lactate"]] = rng.choice(_pool(store, "Lactate"), store.n_rows) + 2.5 * win
    XB[:, j["UrineOutput"]] = 80.0
    ledB = LG.Ledger.build(E.rederive(store, XB))
    fmB = ledB.fit_oof(Candidate("B", "P", _booster()), verbose=False)
    b = CF.process_physiology(ledB, [fmB], n_boot=50)["B"]
    check("value-only world: the transplant attributes the signal to PHYSIOLOGY, process exactly 0",
          abs(b["phi_process"]["value"]) < 1e-12 and abs(b["process_reliance"]["value"]) < 1e-12
          and b["phi_physiology"]["value"] > 0.2,
          f"phi_physiology {b['phi_physiology']['value']:.3f}, reliance {b['process_reliance']['value']:.1e}")
    la = CF.learnable(ledA, Candidate("A", "P", _booster()), n_boot=50)
    lb = CF.learnable(ledB, Candidate("B", "P", _booster()), n_boot=50)
    ra, rb = la["auroc_retrained"], lb["auroc_retrained"]
    check("retrained in each world: the schedule world has nothing to learn from values, the value "
          "world nothing from the schedule",
          ra["physiology_only"] < 0.65 and ra["process_only"] > 0.85
          and abs(lb["learnable_process_information"]["value"]) < 1e-12 and rb["process_only"] < 0.65,
          f"schedule world: physiology {ra['physiology_only']:.3f} / process {ra['process_only']:.3f}; "
          f"value world: physiology {rb['physiology_only']:.3f} / process {rb['process_only']:.3f}")
    A = a["auroc"]
    check("the two Shapley shares sum to factual - baseline",
          abs(a["phi_process"]["value"] + a["phi_physiology"]["value"] - (A["factual"] - A["neither"])) < 1e-12)
    own = np.full(48, np.nan)
    own[2::3] = 300.0                                  # 100 mL/h, charted as 300 mL every 3 h
    moved = CF._uo_rate(own, 50.0) * CF._interval(np.ones(48, dtype=bool))    # an hourly-charting donor
    naive = pd.Series(own).ffill().fillna(50.0).to_numpy()                    # repeat the last chart
    check("urine is transplanted as a RATE: an hourly donor schedule keeps the 24-h volume "
          "(repeating the last chart would triple it)",
          np.allclose(moved[24:], 100.0) and abs(moved[24:].sum() - 2400.0) < 1e-6,
          f"24-h volume {moved[24:].sum():.0f} mL vs {naive[24:].sum():.0f} naive, true 2400")


def c16_admission(led) -> None:
    print("\n16. admission audit")
    from backend.ml.mimic.ledger import Candidate, FoldModels
    base = led.fit_oof(Candidate("clean", "P", _booster()), verbose=False)

    def plant(name, fn):
        preds = {k: (lambda view, rows, pr=pr: fn(view, rows, pr(view, rows))) for k, pr in base.predictors.items()}
        oof = np.full(led.store.n_rows, np.nan)
        for k, pr in preds.items():
            _, te = led.fold_rows(k)
            oof[te] = pr(led, te)
        return FoldModels(name, "P", oof, preds)

    def stay_max_lactate(view):
        v = pd.Series(view.store.col("Lactate")).groupby(view.store.seg).transform("max").to_numpy()
        return np.nan_to_num(v)
    look = plant("lookahead", lambda view, rows, p: p + 0.5 * stay_max_lactate(view)[rows])
    length = plant("length", lambda view, rows, p: p + 0.01 * view.store.lens[view.store.seg[rows]])
    norm = plant("normalised", lambda view, rows, p: p - p.mean())
    res = CF.audit(led, [base, look, length, norm])
    check("a causal, patient-local model is admitted", res["clean"]["admissible"],
          f"max change {max(res['clean']['max_abs_change'].values()):.1e}")
    check("a look-ahead model (whole-stay max lactate) is refused by the future rewrite",
          not res["lookahead"]["passed"]["future_rewritten"] and not res["lookahead"]["admissible"])
    check("a stay-length model is refused by truncation (and ONLY truncation catches it)",
          not res["length"]["passed"]["stay_truncated"] and res["length"]["passed"]["future_rewritten"]
          and res["length"]["passed"]["neighbours_scrambled"])
    check("a test-set-normalising model is refused by scrambling its neighbours",
          not res["normalised"]["passed"]["neighbours_scrambled"] and not res["normalised"]["admissible"])


def c17_infoflow(store, led) -> None:
    print("\n17. information flow")
    real = IF.audit(led, channels=["GCS_verbal", "HR"])
    check("the ETT recode is found: verbal GCS carries the ventilator (covert channel)",
          "GCS_verbal" in real["flagged"], f"AUROC {real['table']['GCS_verbal']['ventilated']:.3f}")
    rng = np.random.default_rng(2)
    X = store.X.copy()
    j = {c: i for i, c in enumerate(store.columns)}
    vaso = IF.treatment_states(led)["vasopressor"]
    m = np.isfinite(X[:, j["HR"]])
    X[m, j["HR"]] = 80 + 30 * vaso[m] + rng.normal(0, 5, int(m.sum()))
    m = np.isfinite(X[:, j["Phosphate"]])
    X[m, j["Phosphate"]] = rng.choice(_pool(store, "Phosphate"), int(m.sum()))
    ledp = LG.Ledger.build(E.rederive(store, X))
    pl = IF.audit(ledp, channels=["HR", "Phosphate"])
    check("a planted vasopressor copy in heart rate is flagged",
          "HR" in pl["flagged"], f"AUROC {pl['table']['HR']['vasopressor']:.3f}")
    check("…and planted noise is not (the audit is not trigger-happy)",
          "Phosphate" not in pl["flagged"], f"max AUROC {max(pl['table']['Phosphate'][k] for k in ('ventilated', 'vasopressor')):.3f}")
    strict = LG.Ledger(led.store, led.F, led.names, led.ftier, led.folds, covert=frozenset({"FiO2"}))
    dropped = {led.names[k] for k in set(led.columns("P")) - set(strict.columns("P", strict=True))}
    # FiO2's own columns, the composites built on it, and everything DERIVED from it — the
    # physiology SOFA (P/F respiration) with its label-structure block
    expect = {n for n in led.names if LG.channel_of(n) == "FiO2"} | \
             {n for n, src in LG.COMPOSITE_SOURCES.items() if "FiO2" in src} | \
             {n for n in led.names if LG.channel_of(n) == "SOFA_P"} | \
             {n for n in led.names if "__levy_" in n and n.split("__levy_")[1].rsplit("_", 1)[0] in ("FiO2", "SOFA_P")}
    check("the strict tier drops exactly the covert channel, every composite built on it, every channel "
          "derived from it (SOFA_P) and every signature cross term that reads one of those",
          dropped == expect, f"{len(dropped)} columns: FiO2 {sum(n.startswith('FiO2') for n in dropped)}, "
          f"composites {len(dropped & set(LG.COMPOSITE_SOURCES))}, SOFA_P {sum(n.startswith('SOFA_P') for n in dropped)}, "
          f"cross terms {sum('__levy_' in n and not n.startswith('SOFA_P') for n in dropped)}")


def c18_crossexam(led) -> None:
    print("\n18. cross-examination")
    s = led.store
    rng = np.random.default_rng(5)
    y = led.y.astype(float)
    orc = CX.oracle(led, "P")
    d_orc = CX.definitional(led, 2 * orc + 3, "P")
    d_inf = CX.definitional(led, y + rng.normal(0, 1, s.n_rows), "P")
    rows = CX._rows(led, orc)
    check("prediction + detection partition the positive hours",
          d_orc["n_pos"]["prediction"] + d_orc["n_pos"]["detection"] == int(led.y[rows].sum()))
    check("re-deriving the label earns exactly zero margin over the definitional oracle",
          all(abs(v) < 1e-12 for v in d_orc["margin"].values() if np.isfinite(v)),
          f"informative score's detection margin {d_inf['margin']['detection']:+.3f}")
    fsi = s.onsets.first_si.to_numpy()[s.seg]
    susp = (np.isfinite(fsi) & (fsi <= s.hour + 1)) + 1e-3 * rng.random(s.n_rows)
    sick = CX.oracle(led, "T") + 1e-3 * rng.random(s.n_rows)
    cs, ck = CX.contrasts(led, susp)["event_aligned"], CX.contrasts(led, sick)["event_aligned"]
    check("a suspicion reader separates sepsis from organ failure but NOT from infection",
          cs["infection_only"] < 0.65 and cs["organ_only"] > 0.8,
          f"vs infection-only {cs['infection_only']:.3f}, vs organ-only {cs['organ_only']:.3f}")
    check("a sickness detector fails against organ-dysfunction-only stays",
          ck["organ_only"] < 0.5 and ck["organ_only"] < ck["infection_only"],
          f"vs organ-only {ck['organ_only']:.3f}, vs infection-only {ck['infection_only']:.3f}")
    hc = CX.horizon(led, s.hour.astype(float))["auroc_clock_matched"]
    he = CX.horizon(led, np.isfinite(led.onset[s.seg]).astype(float))["auroc_clock_matched"]
    check("the ICU clock scores exactly 0.5 at every clock-matched horizon; a future oracle 1.0",
          all(abs(v - 0.5) < 1e-12 for v in hc if np.isfinite(v)) and all(v == 1.0 for v in he if np.isfinite(v)),
          f"{sum(np.isfinite(hc))} horizons")
    c = y + rng.normal(size=s.n_rows)
    cr = CX._rows(led, c)
    edges = np.unique(np.quantile(c[cr], np.linspace(0, 1, 11)[1:-1]))
    copy = np.searchsorted(edges, c, side="right").astype(float)
    cc = CX.clinician_conditional(led, copy, c)
    ci = CX.clinician_conditional(led, y + rng.normal(size=s.n_rows), c)
    check("a copy of the clinician score scores exactly 0.5 beyond the clinicians; an independent "
          "signal keeps its AUROC", abs(cc["conditional_auroc"] - 0.5) < 1e-12
          and abs(ci["conditional_auroc"] - ci["model_auroc"]) < 0.05,
          f"independent {ci['conditional_auroc']:.3f} vs unconditional {ci['model_auroc']:.3f}")
    rr = np.random.default_rng(0)
    sc, q, w = rr.normal(size=40), rr.random(40), rr.random(40) + 0.5
    q[:8] = rr.integers(0, 2, 8)
    num = den = 0.0
    for i in range(40):
        for k in range(40):
            if i != k:
                pw = w[i] * q[i] * w[k] * (1 - q[k])
                den += pw
                num += pw * (1.0 if sc[i] > sc[k] else 0.5 if sc[i] == sc[k] else 0.0)
    hard = (rr.random(40) < 0.4).astype(float)
    from backend.ml.genesis.harness import Ranked
    check("soft-label AUROC equals its brute-force definition (self-pairs excluded); hard labels "
          "reduce to the ordinary AUROC", abs(CX.soft_auc(sc, q, w) - num / den) < 1e-12
          and abs(CX.soft_auc(sc, hard, w) - Ranked(sc).auc(hard.astype(int), w)) < 1e-12)
    agree = s.onsets.copy()
    for col in [c for c in agree.columns if c.startswith("t_sepsis_")]:
        agree[col] = agree[f"t_sepsis_{E.PRIMARY_RULE}_min"]
    st2 = E.MimicStore(X=s.X, y=s.y, lens=s.lens, pid=s.pid, site=s.site, columns=s.columns, tiers=s.tiers,
                       static_columns=s.static_columns, meta=s.meta, onsets=agree, sofa=s.sofa)
    led2 = LG.Ledger(st2, led.F, led.names, led.ftier, led.folds)
    amb = CX.ambiguity(led2, y + rng.normal(size=s.n_rows))
    check("when every definition agrees, nothing is ambiguous and the soft AUROC is the AUROC",
          amb["ambiguous_share_pos"] == 0 and amb["ambiguous_share_neg"] == 0
          and abs(amb["auroc_soft"] - amb["auroc_primary"]) < 1e-12)
    fem = (s.meta.sex.to_numpy() == "Female")[s.seg]
    sg = CX.subgroups(led, np.where(fem, rng.normal(size=s.n_rows), y + rng.normal(0, 0.7, s.n_rows)))
    check("a subgroup the score fails on is flagged (Holm-corrected); the other is not",
          "sex=Female" in sg["flagged"] and "sex=Male" not in sg["flagged"],
          f"Female {sg['groups']['sex']['Female']['auroc']:.3f}, Male {sg['groups']['sex']['Male']['auroc']:.3f}")
    check("audit-only attributes can never reach a model (not in the grid, not a feature)",
          not set(E.AUDIT_ATTRS) & (set(s.columns) | set(led.names)))
    af = CX.alert_frontier(led, y + rng.normal(0, 0.8, s.n_rows))
    rates = [v["alerts_per_pt_day"] for v in af["budgets"].values()]
    caught = [v["episodes_caught"] for v in af["budgets"].values()]
    perfect = CX.alert_frontier(led, y)["budgets"]["0.25"]
    check("alerts respect every budget and catch more episodes as the budget grows; a perfect "
          "score catches every episode with no false alert",
          all(r <= b + 1e-9 for r, b in zip(rates, CX.BUDGETS)) and caught == sorted(caught)
          and perfect["episodes_caught"] == 1.0 and perfect["ppv"] == 1.0,
          f"caught {[round(x, 2) for x in caught]}")


def c19_trust(led) -> None:
    print("\n19. trust layer")
    import os
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="verify_trust_")
    try:
        ch = TR.Chain(os.path.join(tmp, "chain.jsonl"))
        for i in range(5):
            ch.append("note", {"i": i, "x": 0.1 * i})
        ok0 = ch.verify()
        lines = open(ch.path).read().split("\n")[:-1]

        def attempt(new_lines):
            with open(ch.path, "w") as fh:
                fh.write("\n".join(new_lines) + "\n")
            return ch.verify()
        edited = list(lines)
        edited[2] = edited[2].replace('"x":0.2', '"x":0.25')
        t1 = attempt(edited)
        t2 = attempt(lines[:1] + lines[2:])
        t3 = attempt(lines[:3] + [lines[4], lines[3]])
        try:
            ch.append("note", {"i": 99})
            refused = False
        except RuntimeError:
            refused = True
        check("an intact chain verifies; an edited, a deleted and a reordered entry are each caught "
              "at the first bad entry", ok0 == (True, 5, None) and t1[2] == 2 and t2[2] == 1 and t3[2] == 3,
              f"edit@{t1[2]} delete@{t2[2]} reorder@{t3[2]}")
        check("…and a broken chain refuses to grow", refused)

        root = os.path.join(tmp, "repo")
        os.makedirs(os.path.join(root, "backend", "ml", "mimic"))
        f = os.path.join(root, "backend", "ml", "mimic", "a.py")
        open(f, "w").write("x = 1\n")
        f1 = TR.code_fingerprint(root)["sha256"]
        open(f, "w").write("x = 2\n")
        f2 = TR.code_fingerprint(root)["sha256"]
        open(f, "w").write("x = 1\n")
        check("the code fingerprint moves with one changed byte of harness code, and only then",
              f1 != f2 and TR.code_fingerprint(root)["sha256"] == f1)

        rng = np.random.default_rng(9)
        y = led.y.astype(float)
        p = y + rng.normal(0, 1.0, len(y))
        q = y + rng.normal(0, 1.2, len(y))
        pc = TR.Chain(os.path.join(tmp, "prereg.jsonl"))
        try:
            TR.preregister(led, "tiny", "p beats q by 0.01", p, q, bar=0.01, chain=pc)
            gate = False
        except PermissionError as e:
            gate = "underpowered" in str(e)
        plan = TR.power_plan(led, p, q, bar=0.01)
        check("the power gate refuses a claim the lock cannot resolve",
              gate and len(pc.entries()) == 0, f"MDE {plan['mde']:.3f} with {plan['n_lock_subjects']} locked subjects")
        reg = TR.preregister(led, "coarse", "p above chance", p, bar=0.5, chain=pc)
        lock = np.flatnonzero(led.folds[led.store.seg] < 0)
        pl = np.full(len(y), np.nan)
        pl[lock] = p[lock]
        res = TR.confirm(led, reg["hash"], pl, chain=pc)
        try:
            TR.confirm(led, reg["hash"], pl, chain=pc)
            again = True
        except PermissionError:
            again = False
        check("an answerable claim is registered, confirmed once, and cannot be confirmed twice",
              res["passed"] and not again, f"lock AUROC {res['value']:.3f}, lower bound {res['lower']:.3f}")
        real = TR.code_fingerprint
        TR.code_fingerprint = lambda root=None: {"sha256": "0" * 64, "files": 0, "git": None, "dirty": None}
        reg2 = TR.preregister(led, "drift", "p above chance", p, bar=0.5, chain=pc)
        TR.code_fingerprint = real
        try:
            TR.confirm(led, reg2["hash"], pl, chain=pc)
            drift_refused = False
        except PermissionError:
            drift_refused = True
        TR.confirm(led, reg2["hash"], pl, chain=pc, deviation="verify: planted drift")
        check("code that changed after registration is refused unless the deviation is chained",
              drift_refused and len(pc.of_kind("deviation")) == 1 and pc.verify()[0])

        from backend.ml.genesis.harness import auroc
        n, k = 4000, 400
        yh, yf = rng.integers(0, 2, n), rng.integers(0, 2, n)
        eta = float(np.sqrt((n + 1) / (12 * (yh == 1).sum() * (yh == 0).sum())))
        lad = TR.LadderCore(eta=eta, budget=k + 1)
        naive, kept = [], []
        for _ in range(k):
            u = rng.normal(size=n)
            a = auroc(yh, u)
            if a > 0.5:
                naive.append(u)
            out = lad.submit(a)                    # keep what the mechanism SAYS beat chance
            if out["improved"] and out["released"] > 0.5:
                kept.append(u)
        nv = np.sum(naive, axis=0)
        lv = np.sum(kept, axis=0)
        naive_err = auroc(yh, nv) - auroc(yf, nv)
        released = lad.submit(auroc(yh, lv))["released"]
        ladder_err = released - auroc(yf, lv)
        check("boosting attack: a naive holdout is climbed on pure noise, the Ladder is not",
              naive_err > 0.06 and ladder_err < naive_err / 2,
              f"naive reports +{naive_err:.3f} over the truth, Ladder +{ladder_err:.3f}")
        check("the Ladder only ever releases multiples of its step",
              abs(released / eta - round(released / eta)) < 1e-6)
        lc = TR.Chain(os.path.join(tmp, "ladder.jsonl"))
        out = TR.Ladder(lc, budget=2).submit(led, "demo", pl)
        TR.Ladder(lc, budget=2).submit(led, "demo again", pl)
        try:
            TR.Ladder(lc, budget=2).submit(led, "one too many", pl)
            spent = False
        except PermissionError:
            spent = True
        raw = TR.estimate(led, pl, locked=True, n_boot=0)["value"]
        text = open(lc.path).read()
        check("a Ladder submission chains the released number, never the raw lock score; the "
              "budget is enforced", repr(raw) not in text and f"{raw:.6f}" not in text and spent
              and lc.verify()[0], f"released {out['released']} (step {out['eta']:.3f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 20. The temporal model
# --------------------------------------------------------------------------- #
def _lag_world(n=400, L=90, lag=30, seed=3):
    """Every stay has one 3-h heart-rate spike; in stays that become septic it comes exactly
    ``lag`` hours before onset, elsewhere at a random time. Only the lag carries the label."""
    rng = np.random.default_rng(seed)
    cols = E.DYNAMIC + E.STATIC
    j = {c: i for i, c in enumerate(cols)}
    lens = np.full(n, L, np.int64)
    N = n * L
    X = np.full((N, len(cols)), np.nan, np.float32)
    for c in E.VASO_CH + E.VENT_CH:
        X[:, j[c]] = 0.0
    septic = rng.random(n) < 0.4
    onset = np.where(septic, rng.integers(42, 80, n), -1).astype(float)
    spike = np.where(septic, onset - lag, rng.integers(8, 70, n))
    hour = np.tile(np.arange(L), n)
    seg = np.repeat(np.arange(n), L)
    X[:, j["HR"]] = rng.normal(85, 6, N) + 45 * ((hour >= spike[seg]) & (hour < spike[seg] + 3))
    X[:, j["MAP"]] = rng.normal(80, 8, N)
    for c in E.STATIC:
        X[:, j[c]] = 0.0 if c.startswith("Unit_") else 1.0
    X[:, j["ICULOS"]] = hour + 1
    E.derive_channels(X, cols, lens)
    t = np.where(septic, onset, np.nan)
    on = {"stay_id": np.arange(n) + 1, "first_si": t, "first_order": t}
    for r in E.RULES:
        for a in ("min", "si", "sofa"):
            on[f"t_sepsis_{r}_{a}"] = t
        on[f"t_si_{r}"], on[f"t_sofa_{r}"] = t, t
    meta = pd.DataFrame({"subject_id": np.arange(n) + 1, "hadm_id": np.arange(n) + 1, "stay_id": np.arange(n) + 1,
                         "first_careunit": "MICU", "anchor_year_group": "2014 - 2016", "age": 60.0,
                         "los_hours": float(L), "n_hours": L, "hospital_expire_flag": 0,
                         **{a: "x" for a in E.AUDIT_ATTRS if a != "anchor_year_group"}})
    y = (np.isfinite(t[seg]) & (hour >= t[seg] - 6)).astype(np.int8)
    return E.MimicStore(X=X, y=y, lens=lens, pid=np.arange(n) + 1, site=np.zeros(n, np.int8), columns=cols,
                        tiers={c: E.tier_of(c) for c in E.DYNAMIC} | {c: "P" for c in E.STATIC},
                        static_columns=list(E.STATIC), meta=meta, onsets=pd.DataFrame(on), sofa={})


def _twin_world(store, kind, seed=11):
    """Planted worlds on the demo's real schedules and labels. D: a care-process shortcut
    (hourly lactate near onset) AND a physiological one (heart rate +30 near onset, a channel
    everyone's schedule measures). A: the shortcut only; every value, urine included, noise."""
    rng = np.random.default_rng(seed)
    j = {c: i for i, c in enumerate(store.columns)}
    win = _window(store)
    X = store.X.copy()
    for c in CF.STATE_CH + ([CF.RATE_CH] if kind == "A" else []):
        m = np.isfinite(X[:, j[c]])
        pl = _pool(store, c)
        if len(pl):
            X[m, j[c]] = rng.choice(pl, int(m.sum()))
    X[win, j["Lactate"]] = rng.choice(_pool(store, "Lactate"), int(win.sum()))
    if kind == "D":
        m = np.isfinite(X[:, j["HR"]])
        X[m & win, j["HR"]] += 30.0
    return LG.Ledger.build(E.rederive(store, X))


def _twin_oof(led, regime, modes=("full",), d_model=64, **kw):
    T = TW.process_targets(led)
    s = led.store
    stays = np.arange(s.n_patients)
    out = {m: np.full(s.n_rows, np.nan) for m in modes}
    for k in range(led.n_folds):
        tr, te = stays[(led.folds >= 0) & (led.folds != k)], stays[led.folds == k]
        tw = TW.ProcessTwin(TW.TwinConfig(d_model=d_model, seed=0, regime=regime, **kw)).fit(led, T, tr)
        r = TW._rows(s, te)
        for m in modes:
            out[m][r] = tw.predict(led, te, ("direct",), mode=m)["direct"]
    return {m: led.score(v) for m, v in out.items()}


def c20_temporal(store, led) -> None:
    print("\n20. temporal model (Process Twin v2)")
    lagled = LG.Ledger.build(_lag_world())
    c20a_math(lagled)
    c20b_capacity(lagled)
    c20c_physiology(store)
    c20d_locality(store, led)


def c20a_math(lagled) -> None:
    import torch
    from backend.ml.genesis.network import _bce
    T = TW.process_targets(lagled)
    hy, hm = T["hazard"]
    ls = lagled.store
    sep = int(np.flatnonzero(np.isfinite(lagled.onset))[0])
    non = int(np.flatnonzero(~np.isfinite(lagled.onset))[0])
    o = int(lagled.onset[sep])
    a, b = ls.starts[sep], ls.starts[non]
    L = int(ls.lens[non])
    r_event, r_far, r_after = a + o - 5, a + max(o - 30, 0), a + o + 1
    r_cens = b + L - 1 - 3                                  # 3 h before discharge, never septic
    ok = (hy[r_event].tolist() == [0, 0, 0, 0, 1, 0, 0, 0] and hm[r_event].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
          and hy[r_far].sum() == 0 and hm[r_far].sum() == TW.N_BINS
          and hy[r_cens].sum() == 0 and hm[r_cens].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
          and hm[r_after].sum() == 0)
    check("hazard targets follow survival semantics: onset in 5 h is an event in (4,6]; 30 h away "
          "survives every bin; 3 h before discharge only 3 bins are observed; after onset, not at risk", ok)
    rng = np.random.default_rng(1)
    rows = np.array([r_event, r_far, r_cens])
    logits = rng.normal(0, 1.5, (len(rows), TW.N_BINS))
    h = 1 / (1 + np.exp(-logits))
    brute = 0.0
    for i, r in enumerate(rows):
        for bb in range(TW.N_BINS):
            if hm[r, bb]:
                brute -= np.log(h[i, bb]) if hy[r, bb] else np.log(1 - h[i, bb])
    mm = torch.from_numpy(hm[rows])
    ours = float(_bce(torch.from_numpy(logits), torch.from_numpy(hy[rows]).double(), mm.double())) * float(mm.sum())
    F = TW.cumulative_incidence(h, 24)
    Fs = [TW.cumulative_incidence(h, H) for H in TW.HAZ_EDGES[1:]]
    tm = TW._t_hazard_margin(torch.from_numpy(logits), 6).numpy()
    F6 = TW.cumulative_incidence(h, 6)
    check("the per-bin loss IS the discrete-time survival likelihood; the risk curve is monotone "
          "and its 6-h margin is exactly logit F(6)",
          abs(ours - brute) < 1e-9 and all(np.all(Fs[i + 1] >= Fs[i] - 1e-15) for i in range(len(Fs) - 1))
          and np.allclose(F, 1 - np.prod(1 - h, axis=1)) and np.allclose(tm, np.log(F6 / (1 - F6))),
          f"NLL {ours:.4f} vs brute force {brute:.4f}")

    p = rng.dirichlet(np.ones(3), size=(5, len(TW.ORGANS)))
    import itertools
    brute = np.zeros(5)
    for combo in itertools.product(range(3), repeat=len(TW.ORGANS)):
        pr = np.prod([p[:, o, c] for o, c in enumerate(combo)], axis=0)
        brute += pr * (sum(combo) >= 2)
    tl = TW._t_outlook_log_le1(torch.from_numpy(np.log(p))).numpy()
    check("the composed organ outlook equals brute-force enumeration of all 729 organ outcomes",
          np.allclose(TW.outlook_p_ge2(p), brute, atol=1e-12) and np.allclose(1 - np.exp(tl), brute, atol=1e-9))



def c20b_capacity(lagled) -> None:
    from backend.ml.mimic.ledger import Candidate
    # only columns that VARY in this world: LightGBM samples columns at random, and every
    # all-empty channel a later catalog adds would otherwise change which informative ones
    # each tree sees (the windowed booster read 0.781 on the v2 catalog, 0.818 on v4, for no
    # reason but that) — the comparison must not drift with irrelevant columns
    F = np.asarray(lagled.F, dtype=np.float64)
    live = [j for j in lagled.columns("P") if np.nanstd(F[:, j]) > 0]
    windowed = np.array([j for j in live if "__qv" not in lagled.names[j] and "__levy_" not in lagled.names[j]])
    with_sig = np.array(live)
    bst = lagled.fit_oof(Candidate("booster", "P", _booster(), columns=lambda l: windowed), verbose=False)
    sig = lagled.fit_oof(Candidate("booster+signatures", "P", _booster(), columns=lambda l: with_sig), verbose=False)
    tw = _twin_oof(lagled, "record")["full"]
    bs, bg = lagled.score(bst.oof), lagled.score(sig.oof)
    # the claim is about WINDOWED features (6 / 24 h trends, extremes, deltas); the signature
    # block reaches part of the lag itself — reported beside it, not hidden
    check("the twin reads a 30-h lag that a windowed booster cannot (clock-matched)",
          tw["auroc_hourmatched"] - bs["auroc_hourmatched"] >= 0.08,
          f"hour-matched AUROC twin {tw['auroc_hourmatched']:.3f} vs windowed booster {bs['auroc_hourmatched']:.3f} "
          f"(with signature features {bg['auroc_hourmatched']:.3f})")



def c20c_physiology(store) -> None:
    wd = _twin_world(store, "D")
    rec = _twin_oof(wd, "record", modes=("full", "physiology"))
    phy = _twin_oof(wd, "transplanted", modes=("physiology",))
    # the bar is the MECHANISM — a clear margin over a record-trained twin scored on the same
    # transplanted patients, well clear of chance — not an absolute level: the planted world's
    # level fell 0.740 -> 0.698 when v3 added 17 (randomised, noise) channels to every stay
    check("the physiology twin keeps a physiological signal (trained on transplanted patients; a "
          "record-trained twin scored on them does not)",
          phy["physiology"]["auroc"] >= 0.65 and phy["physiology"]["auroc"] >= rec["physiology"]["auroc"] + 0.08
          and rec["full"]["auroc"] >= 0.75,
          f"physiology twin {phy['physiology']['auroc']:.3f}, record twin on transplanted "
          f"{rec['physiology']['auroc']:.3f}, record twin's full score {rec['full']['auroc']:.3f}")
    wa = _twin_world(store, "A")
    rec_a = _twin_oof(wa, "record")
    phy_a = _twin_oof(wa, "transplanted", modes=("physiology",))
    check("…and drops a care-process one: in a schedule-only world the physiology score sits near "
          "chance (clock-matched) while the full score does not",
          phy_a["physiology"]["auroc_hourmatched"] <= 0.62 and rec_a["full"]["auroc_hourmatched"] >= 0.7,
          f"physiology {phy_a['physiology']['auroc_hourmatched']:.3f} vs full "
          f"{rec_a['full']['auroc_hourmatched']:.3f} (hour-matched)")



def c20d_locality(store, led) -> None:
    u1 = CF.hash_uniform(7, np.arange(100_000), 3, 5)
    u2 = CF.hash_uniform(7, np.arange(100_000), 3, 5)
    u3 = CF.hash_uniform(8, np.arange(100_000), 3, 5)
    ks = np.max(np.abs(np.sort(u1) - (np.arange(1, len(u1) + 1) / len(u1))))
    check("counterfactual draws are a pure function of their keys (deterministic, key-sensitive, uniform)",
          np.array_equal(u1, u2) and np.mean(u1 == u3) < 1e-3 and ks < 0.01, f"KS {ks:.4f}")

    from backend.ml.mimic.run import twin_predictor
    from backend.ml.mimic.ledger import FoldModels
    Tl = TW.process_targets(led)
    stays = np.arange(store.n_patients)
    k = 0
    tr = stays[(led.folds >= 0) & (led.folds != k)]
    twin = TW.ProcessTwin(TW.TwinConfig(d_model=32, seed=0, regime="transplanted", min_steps=30, epochs=1)).fit(led, Tl, tr)
    fm = FoldModels("phys", "P", np.full(store.n_rows, np.nan), {k: twin_predictor([twin], "s3", "physiology")})
    te = TW._rows(store, stays[led.folds == k])
    fm.oof[te] = fm.predictors[k](led, te)
    clean = CF.audit(led, [fm])["phys"]
    real = CF.hash_uniform
    gen = np.random.default_rng(0)
    CF.hash_uniform = lambda *keys: gen.random(np.broadcast_arrays(*[np.asarray(x) for x in keys])[0].shape)
    try:
        fm2 = FoldModels("sequential", "P", fm.oof.copy(), fm.predictors)
        dirty = CF.audit(led, [fm2])["sequential"]
    finally:
        CF.hash_uniform = real
    check("the physiology score passes the admission audit; with a sequential random generator in "
          "place of the keyed hash it is refused (draws coupled across patients)",
          clean["admissible"] and not dirty["admissible"],
          f"clean max change {max(clean['max_abs_change'].values()):.1e}; sequential refused by "
          f"{[kk for kk, v in dirty['passed'].items() if not v]}")


# --------------------------------------------------------------------------- #
# 21-23. Information v3, label structure, AUROC machinery
# --------------------------------------------------------------------------- #
def _planted_source(tmp):
    """A CsvSource over hand-written tables: one admission, one ICU stay (intime T)."""
    import os
    T = pd.Timestamp("2150-01-01 00:00")
    h = lambda x: (T + pd.Timedelta(hours=x)).strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(tmp, "hosp"))
    os.makedirs(os.path.join(tmp, "icu"))
    pd.DataFrame([(1, h(1.2), 51486, np.nan, "MOD"), (1, h(2.1), 51486, np.nan, "XYZ"),
                  (1, h(3.5), 51487, np.nan, "POS"), (1, h(4.0), 50889, 55.0, "55")],
                 columns=["hadm_id", "charttime", "itemid", "valuenum", "value"]).to_csv(
        os.path.join(tmp, "hosp", "labevents.csv"), index=False)
    pd.DataFrame([(10, h(2), h(5), 222168, np.nan, "FinishedRunning", "01-Drips"),
                  (10, h(3.2), h(3.3), 225158, 500.0, "FinishedRunning", "03-IV Fluid Bolus"),
                  (10, h(3.6), h(3.7), 225158, 250.0, "FinishedRunning", "03-IV Fluid Bolus"),
                  (10, h(3.1), h(3.2), 225158, 900.0, "Rewritten", "03-IV Fluid Bolus"),
                  (10, h(6.5), h(7.5), 225168, 300.0, "FinishedRunning", "07-Blood Products"),
                  (10, h(4.0), h(4.5), 225798, 1.0, "FinishedRunning", "08-Antibiotics (IV)"),
                  (10, h(9.0), h(9.5), 225798, 1.0, "Rewritten", "08-Antibiotics (IV)")],
                 columns=["stay_id", "starttime", "endtime", "itemid", "amount", "statusdescription",
                          "ordercategoryname"]).to_csv(os.path.join(tmp, "icu", "inputevents.csv"), index=False)
    pd.DataFrame([(10, h(10), h(13), 225802)], columns=["stay_id", "starttime", "endtime", "itemid"]).to_csv(
        os.path.join(tmp, "icu", "procedureevents.csv"), index=False)
    stays = pd.DataFrame({"stay_id": [10], "hadm_id": [1], "intime": [T], "outtime": [T + pd.Timedelta(hours=20)]})
    return SRC.CsvSource(tmp), stays


def c21_information() -> None:
    print("\n21. information v3")
    import shutil
    import tempfile
    from backend.ml.mimic import history as H
    tmp = tempfile.mkdtemp(prefix="verify_v3_")
    try:
        src, stays = _planted_source(tmp)
        labs = src.binned_labs(stays).set_index(["hr", "name"]).value.to_dict()
        check("urinalysis text is coded (MOD -> 2, POS -> 1) and an unknown code is dropped",
              labs.get((1, "UrineLeuk")) == 2.0 and labs.get((3, "UrineNitrite")) == 1.0
              and (2, "UrineLeuk") not in labs and labs.get((4, "CRP")) == 55.0, f"{labs}")
        sed = src.interval_hours(stays, "sedation")
        dia = src.interval_hours(stays, "dialysis")
        amt = src.hourly_amounts(stays).set_index(["hr", "name"]).value.to_dict()
        abx = src.event_times(stays, "abx_admin")
        check("sedation and dialysis cover their hours; fluids and blood sum by the hour they start; "
              "rewritten rows are ignored",
              sorted(sed.hr) == [2, 3, 4] and sorted(dia.hr) == [10, 11, 12]
              and amt == {(3, "Crystalloid"): 750.0, (6, "BloodProducts"): 300.0},
              f"sedation {sorted(sed.hr)}, dialysis {sorted(dia.hr)}, volumes {amt}")
        check("bedside IV antibiotics come from their category, timed from ICU admission",
              list(abx.t.round(3)) == [4.0], f"{list(abx.t)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    T = pd.Timestamp("2150-06-01")
    stays = pd.DataFrame({"stay_id": [5], "subject_id": [7], "hadm_id": [3], "admittime": [T]})
    adm = pd.DataFrame({"subject_id": [7, 7, 7], "hadm_id": [1, 3, 4],
                        "admittime": [T - pd.Timedelta(days=100), T, T + pd.Timedelta(days=30)],
                        "dischtime": [T - pd.Timedelta(days=90), T + pd.Timedelta(days=10), T + pd.Timedelta(days=40)]})
    dx = pd.DataFrame({"hadm_id": [1, 1, 3, 4], "icd_code": ["A419", "4254", "C77", "B20"],
                       "icd_version": [10, 9, 10, 10]})
    hx = H.prior_history(stays, adm, dx).iloc[0]
    check("comorbidities read only admissions discharged BEFORE the stay (current and later codes ignored)",
          hx.Hx_sepsis == 1 and hx.Hx_chf == 1 and hx.Hx_metastatic == 0 and hx.Hx_hiv == 0
          and hx.Hx_prior_adm == 1 and abs(hx.Hx_days_since_discharge - 90) < 1e-9,
          f"sepsis {hx.Hx_sepsis}, chf {hx.Hx_chf}, metastatic(current) {hx.Hx_metastatic}, hiv(later) {hx.Hx_hiv}")
    check("code ranges expand exactly (I425-I429, 196-199)",
          H.expand("I425-I429 I43") == ("I425", "I426", "I427", "I428", "I429", "I43")
          and H.expand("196-199") == ("196", "197", "198", "199"))
    st = pd.DataFrame({"stay_id": [1, 2], "hadm_id": [11, 12], "intime": [T, T],
                       "admission_location": ["EMERGENCY ROOM", "SOMEWHERE NEW"]})
    sv = pd.DataFrame({"hadm_id": [11, 11, 12], "transfertime": [T - pd.Timedelta(hours=5), T + pd.Timedelta(hours=5),
                                                               T + pd.Timedelta(hours=1)],
                       "curr_service": ["MED", "SURG", "CSURG"]})
    ctx = H.admission_context(st, sv).set_index("stay_id")
    one = (ctx[H.SRC_COLS].sum(axis=1) == 1).all() and (ctx[H.SVC_COLS].sum(axis=1) == 1).all()
    check("one admission source and one admitting service per stay (the service at ICU admission)",
          one and ctx.loc[1, "Src_ED"] == 1 and ctx.loc[1, "Svc_MED"] == 1 and ctx.loc[2, "Src_Other"] == 1
          and ctx.loc[2, "Svc_CSURG"] == 1)
    labs_sql, amt_sql = SRC.render_labs_sql("H", "I"), SRC.render_amount_sql("I")
    sed_sql, abx_sql = SRC.render_interval_sql("I", "sedation"), SRC.render_events_sql("H", "I", "abx_admin")
    miss = [i for c in K.LAB_V3 + K.LAB_TEXT for i in c.itemids if str(i) not in labs_sql] + \
           [i for c in K.TREAT_AMOUNT for i in c.itemids if str(i) not in amt_sql] + \
           [i for c in K.TREAT_INTERVAL if c.kind == "sedation" for i in c.itemids if str(i) not in sed_sql]
    check("every v3 item reaches its SQL, with the urinalysis coding and the antibiotic category",
          not miss and "WHEN 'MOD' THEN 2.0" in labs_sql and K.ABX_ADMIN_CATEGORY in abx_sql, f"missing {miss}")


def c22_structure(store, led) -> None:
    print("\n22. the label's structure as features")
    from backend.ml.mimic import mimicfeat as MF
    blk = [n for n in led.names if n.split("__")[0] in ("SOFA_P", "SOFA", "Antibiotic", "UrineOutput",
                                                         "Neutrophils", "Norepinephrine")
           and "__" in n and not n.endswith(("__age", "__n", "__n24", "__prev", "__dlast", "__first", "__dfirst",
                                             "__emax", "__emin", "__dmean", "__dtlast"))
           and any(n.endswith(x) for x in ("_pts", "_rise", "rise_total", "since_rise2", "si_known", "since_si",
                                           "mlkgh6", "mlkgh12", "mlkgh24", "nlr", "nee", "nee_max24"))]
    tiers = {n: int(led.ftier[led.names.index(n)]) for n in blk}
    expect = {n: K.TIER_ORDER[store.tiers[n.split("__")[0]]] for n in blk}
    check("every label-structure feature carries its source channel's tier (SOFA -> T, suspicion -> S)",
          tiers == expect and tiers["Antibiotic__si_known"] == 2 and tiers["SOFA__cardio_pts"] == 1
          and tiers["SOFA_P__cardio_pts"] == 0, f"{len(blk)} columns")
    seg = np.zeros(80, np.int64)
    hour = np.arange(80)
    ev = lambda hs: np.isin(hour, hs)
    k1 = MF.suspicion_known(ev([5]), ev([60]), seg, hour)[0]
    k2 = MF.suspicion_known(ev([40]), ev([10]), seg, hour)[0]
    k3 = MF.suspicion_known(ev([20]), ev([10]), seg, hour)[0]
    k4 = MF.suspicion_known(ev([5]), ev([78]), seg, hour)[0]
    check("the suspicion pair completes where Sepsis-3's windows say (culture->abx <= 72 h, abx->culture <= 24 h)",
          k1 == 60 and np.isnan(k2) and k3 == 20 and np.isnan(k4), f"{k1}, {k2}, {k3}, {k4}")
    sub = E.subset(store, np.array([0]))
    X = sub.X.copy()
    j = {c: i for i, c in enumerate(sub.columns)}
    X[:, j["UrineOutput"]] = 100.0
    X[:, j["Weight"]] = 50.0
    for c in E.VASO_CH:
        X[:, j[c]] = 0.0
    X[:, j["Norepinephrine"]] = 0.1
    X[:, j["Vasopressin"]] = 2.4                          # U/h = 0.04 U/min = 0.1 NE-equivalent
    B, names = MF.block(E.rederive(sub, X))
    b = {n: B[:, i] for i, n in enumerate(names)}
    check("urine per kg per hour and norepinephrine equivalents on planted values",
          np.allclose(b["UrineOutput__mlkgh6"][6:], 2.0) and np.allclose(b["Norepinephrine__nee"], 0.2),
          f"UO {b['UrineOutput__mlkgh6'][6]:.2f} mL/kg/h, NEE {b['Norepinephrine__nee'][0]:.3f}")
    trimmed = E.trim_septic(store)
    F0, _ = LG.build_features(store)
    F1, _ = LG.build_features(trimmed)
    keep = np.concatenate([np.arange(store.starts[k], store.starts[k] + trimmed.lens[k]) for k in range(store.n_patients)])
    check("trimming septic stays after their latest onset changes no feature on a kept row",
          _same(F0[keep], F1) and trimmed.n_rows < store.n_rows, f"{store.n_rows:,} -> {trimmed.n_rows:,} rows")


def c23_auroc(store, led) -> None:
    print("\n23. the AUROC machinery")
    from backend.ml.genesis.harness import Ranked
    from backend.ml.mimic import parity as PY
    from backend.ml.mimic import plan as PL
    from backend.ml.mimic import tune as TN
    from backend.ml.mimic.ledger import Candidate
    rng = np.random.default_rng(4)
    s = led.store
    p = led.y + rng.normal(0, 1.0, s.n_rows)
    sc = led.score(p)
    rows = np.flatnonzero(led.eval_mask & (led.folds[s.seg] >= 0))
    per = pd.DataFrame({"seg": s.seg[rows], "p": p[rows]}).groupby("seg").p.max()
    lab = np.isfinite(led.onset[per.index.to_numpy()]).astype(int)
    brute = Ranked(per.to_numpy()).auc(lab, led.design_w[s.starts[per.index.to_numpy()]])
    every = np.flatnonzero(led.train_mask & (led.folds[s.seg] >= 0))
    arrival = np.isfinite(led.onset) & (led.onset < LG.ONSET_MIN)
    check("patient-level AUROC equals its brute force; 'all stays' includes sepsis present on arrival",
          abs(sc["patient_auroc"] - brute) < 1e-12 and arrival[s.seg[every]].any() and not arrival[s.seg[rows]].any(),
          f"patient {sc['patient_auroc']:.3f}, all-stays {sc['auroc_all_stays']:.3f} vs early-warning {sc['auroc']:.3f}")

    fms = [led.fit_oof(Candidate(t, "P", _booster_target(t)), verbose=False) for t in ("direct", "who", "when")]
    st = LG.stack(led, "stack", fms)
    k = 0
    fk = led.folds[s.seg] == k
    flipped = LG.Ledger(led.store, led.F, led.names, led.ftier, led.folds, led.rule, led.anchor, led.design_w)
    flipped.y = np.where(fk, 1 - led.y, led.y).astype(np.int8)
    st2 = LG.stack(flipped, "stack", fms)
    other = ~fk & np.isfinite(st.oof)
    check("the stacker never sees the labels it scores (flipping fold-0 labels leaves fold-0 scores unchanged)",
          _same(st.oof[fk], st2.oof[fk]) and not _same(st.oof[other], st2.oof[other]))
    big = LG.stack(led, "inf", fms, shrink=1e12)
    prod = LG.product("prod", fms)
    ok = np.isfinite(big.oof) & np.isfinite(prod.oof)
    spread = max(float(np.std((big.oof - prod.oof)[ok & (led.folds[s.seg] == f)]))
                 for f in range(led.n_folds) if (ok & (led.folds[s.seg] == f)).any())
    check("…and at infinite shrinkage the stack IS the product plus its fold's intercept",
          spread < 1e-6, f"largest within-fold spread of (stack - product): {spread:.1e}")

    seen, scored = [], []                   # rows trained on AND rows scored for selection
    real = TN.L.fit_lgb
    folds_before = led.folds.copy()

    def spy(F, rows, *a, **kw):
        seen.append(np.asarray(rows))
        fit = real(F, rows, *a, **kw)
        pred = fit.predict
        fit.predict = lambda F2, r, *aa, **kk: (scored.append(np.asarray(r)), pred(F2, r, *aa, **kk))[1]
        return fit
    TN.L.fit_lgb = spy
    try:
        tn = TN.tune(led, "P", n_trials=3, verbose=False, base={"learning_rate": 0.05, "num_leaves": 15})
    finally:
        TN.L.fit_lgb = real
    used = np.concatenate(seen + scored)
    check("tuning reads development rows only — to train AND to select — and the incumbent is trial 0",
          (folds_before[s.seg[used]] >= 0).all() and len(scored) > 0
          and tn["trials"][0]["params"] == {"learning_rate": 0.05, "num_leaves": 15}
          and tn["best_auroc"] >= tn["incumbent_auroc"], f"{len(seen)} fits, {len(scored)} scorings, gain {tn['gain']:+.4f}")

    from backend.ml.mimic.run import headline, SETTING_MEANING
    rep = {"models": {f.name: led.score(f.oof) for f in fms}, "admission": {}}
    for f in fms:
        f.tier = "P"
    hl = headline(led, {f.name: f for f in fms}, rep, n_boot=20)
    vals = [hl["P"][st_] for st_ in SETTING_MEANING if np.isfinite(hl["P"][st_])]
    check("the headline's ideal number is the maximum of its settings, reported with the primary beside it",
          abs(hl["ideal_circumstances"]["value"] - max(vals)) < 1e-12
          and abs(hl["ideal_circumstances"]["primary"]["value"] - hl["P"]["auroc"]) < 1e-12)

    from backend.ml.mimic.run import SCALES
    n_feat = len(led.names)
    est = PL.estimate(store, n_feat, SCALES["demo"])
    try:
        PL.choose(store, n_feat, SCALES["demo"], PL.Budget(disk_gb=1e-6, ram_gb=1e-6, hours=1e-6))
        refused = False
    except RuntimeError:
        refused = True
    check("the planner's estimate matches a real build, and an impossible budget is refused before any work",
          est["rows_kept"] == store.n_rows and est["features"] == n_feat and refused,
          f"{est['rows_kept']:,} rows x {est['features']} features")

    # A budget the full stage set misses at keep 1.0 but makes at 0.5, while a leaner stage set
    # would still fit at 1.0: the cloud scale gives up negatives first, the laptop scale is unchanged.
    from dataclasses import replace
    cl = SCALES["cloud"]
    h1 = PL.estimate(store, n_feat, replace(cl, keep_rate=1.0))["hours"]
    h5 = PL.estimate(store, n_feat, replace(cl, keep_rate=0.5))["hours"]
    lean1 = PL.estimate(store, n_feat, replace(cl, keep_rate=1.0, **PL.STAGE_SETS[1][1]))["hours"]
    tight = PL.Budget(disk_gb=1e9, ram_gb=1e9, hours=(h1 + h5) / 2)
    sc_c, pl_c = PL.choose(store, n_feat, cl, tight)
    full_order = [(t["plan"], t["keep_rate"]) for t in
                  PL.choose(store, n_feat, SCALES["full"], PL.Budget(disk_gb=1e9, ram_gb=1e9, hours=1e9))[1]["tried"]]
    check("the cloud planner keeps every stage and gives up negatives first; the laptop plan starts where it did",
          h5 < tight.hours < h1 and lean1 <= tight.hours and pl_c["chosen"] == "full" and sc_c.keep_rate == 0.5
          and [(t["plan"], t["keep_rate"]) for t in pl_c["tried"]] == [("full", 1.0), ("full", 0.5)]
          and full_order == [("full", 0.25)] and SCALES["full"].keep_rate == 0.25,
          f"budget {tight.hours:.3f} h between {h5:.3f} and {h1:.3f}; chose {pl_c['chosen']} at {sc_c.keep_rate}")

    # The planner runs on an extracted store, so a grid that took more memory to build than the
    # machine budgets must not refuse the modelling that fits; and fewer threads take longer.
    class Built:                                  # the demo, as if its extraction had been huge
        def __init__(self, s):
            self._s = s
        n_rows = property(lambda self: self._s.n_rows * 10_000)
        def __getattr__(self, k):
            return getattr(self._s, k)
    big = PL.estimate(Built(store), n_feat, SCALES["demo"])
    sc_b, pl_b = PL.choose(Built(store), n_feat, SCALES["demo"],
                           PL.Budget(disk_gb=1e9, ram_gb=big["model_ram_gb"], hours=1e9))
    e6, e2 = (PL.estimate(store, n_feat, SCALES["cloud"], n_threads=t) for t in (6, 2))
    check("the planner judges only the modelling still ahead of it, and a 2-thread machine is planned 3x slower",
          big["extract_ram_gb"] > 10 * big["model_ram_gb"] and pl_b["chosen"] == "full"
          and abs(e2["booster_hours"] - 3 * e6["booster_hours"]) < 1e-9
          and abs(e2["twin_hours"] - 3 * e6["twin_hours"]) < 1e-9,
          f"extraction {big['extract_ram_gb']:.0f} GB vs modelling {big['model_ram_gb']:.1f} GB; "
          f"boosters {e6['booster_hours']:.3f} h at 6 threads, {e2['booster_hours']:.3f} h at 2")

    # memory in two parts, as measured on the first full run: the hot set (grid + LightGBM) must
    # be RAM; cold memory (every model's scores, the heap) may sit in swap where swap exists
    e = PL.estimate(store, n_feat, SCALES["demo"])
    ok = lambda ram, swap: PL.fits_memory(e, PL.Budget(disk_gb=1e9, ram_gb=ram, hours=1e9, swap_gb=swap))
    check("no swap: hot and cold must both fit in RAM; with swap, cold may use it; hot never may",
          not ok(e["hot_ram_gb"], 0.0) and ok(e["model_ram_gb"], 0.0) and ok(e["hot_ram_gb"], e["cold_ram_gb"])
          and not ok(e["hot_ram_gb"] - 0.01, 1e9) and abs(e["model_ram_gb"] - e["hot_ram_gb"] - e["cold_ram_gb"]) < 1e-9,
          f"hot {e['hot_ram_gb']:.2f} GB, cold {e['cold_ram_gb']:.2f} GB")
    lo, hi = (PL.estimate(store, n_feat, replace(SCALES["cloud"], keep_rate=k)) for k in (0.25, 1.0))
    check("cold memory grows with the rows kept, at the measured rate",
          hi["rows_kept"] > lo["rows_kept"] and abs((hi["cold_ram_gb"] - lo["cold_ram_gb"])
                                                    - (hi["rows_kept"] - lo["rows_kept"]) * PL.COLD_BYTES_PER_ROW / 1e9) < 1e-5,
          f"{lo['rows_kept']:,} -> {hi['rows_kept']:,} rows: cold {lo['cold_ram_gb']:.3f} -> {hi['cold_ram_gb']:.3f} GB")

    class Shifted(SRC.CsvSource):
        def binned_uo(self, stays):
            d = super().binned_uo(stays)
            return d.assign(hr=d.hr + 1)
    ref = SRC.CsvSource(E.DEMO_ROOT)
    cohort = E.cohort(ref)[["stay_id", "hadm_id", "intime", "outtime"]].head(25)
    same = PY.parity(ref, SRC.CsvSource(E.DEMO_ROOT), cohort)
    bad = PY.parity(ref, Shifted(E.DEMO_ROOT), cohort)
    check("the parity check passes identical sources and flags a planted one-hour shift",
          same["verdict"]["passed"] and bad["verdict"]["failing"] == ["urine"],
          f"shifted urine agreement {bad['urine']['agreement']:.3f}")


def _booster_target(target):
    from backend.ml.mimic.run import SCALES, booster
    return booster(SCALES["demo"], target)


# --------------------------------------------------------------------------- #
# 24. The research levers
# --------------------------------------------------------------------------- #
def c24_levers(src, store, led) -> None:
    print("\n24. research levers")
    from backend.ml.mimic import mimicfeat as MF
    ce = pd.read_csv(src._path("icu", "chartevents"), usecols=["stay_id", "charttime", "itemid", "value", "valuenum"])
    ce["charttime"] = pd.to_datetime(ce.charttime)
    st = src.icustays().set_index("stay_id")
    pos = {sid: k for k, sid in enumerate(store.pid)}

    def raw_hourly(itemids, how, text_map=None):
        r = ce[ce.itemid.isin(itemids) & ce.stay_id.isin(pos)].copy()
        r["h"] = ((r.charttime - r.stay_id.map(st.intime)).dt.total_seconds() // 3600).astype(int)
        r = r[(r.h >= 0) & (r.h < store.lens[r.stay_id.map(pos).to_numpy()])]
        if text_map is not None:
            r["v"] = r.value.fillna("").astype(str).str.strip().str.upper().map(dict(text_map)).astype(float)
        else:
            r["v"] = r.valuenum.astype(float)
        r = r.dropna(subset=["v"])
        g = r.groupby(["stay_id", "h"]).v
        return (g.max() if how == "max" else g.min()).reset_index()
    cap = next(c for c in K.EXAM if c.name == "CapRefill")
    ref = raw_hourly(cap.itemids, "max", cap.text_map)
    rows = store.starts[ref.stay_id.map(pos).to_numpy()] + ref.h.to_numpy()
    grid = store.col("CapRefill")[rows]
    both = ce[ce.itemid.isin(cap.itemids)].groupby(["stay_id", "charttime"]).itemid.nunique().gt(1).sum()
    check("the bedside exam is the WORST of the hour's readings (capillary refill, both sides), read from disk",
          np.allclose(grid, ref.v.to_numpy()) and ref.v.max() == 1.0, f"{len(ref)} stay-hours, {both} with L and R charted together")
    ref = raw_hourly([220052, 220181], "min")
    ref = ref[ref.v.between(0, 300)]
    rows = store.starts[ref.stay_id.map(pos).to_numpy()] + ref.h.to_numpy()
    last = store.col("MAP")[rows]
    check("MAP_min is the hour's lowest raw reading (never above its last)",
          np.allclose(store.col("MAP_min")[rows], ref.v.to_numpy()) and bool(np.all(ref.v.to_numpy() <= last + 1e-6)),
          f"{len(ref)} stay-hours, {int((ref.v.to_numpy() < last).sum())} where the minimum is below the last")
    one = np.array([0])
    cols = {"MAP": np.array([80.0]), "MAP_min": np.array([60.0]), "VentInvasive": np.zeros(1)}
    col = lambda nm: cols.get(nm, np.full(1, np.nan))
    pts = S3.component_points_now(col, one, one)
    check("SOFA's cardiovascular point reads the hour's LOWEST MAP (a dip to 60 inside an hour that ends at 80)",
          pts[0, 3] == 1.0, f"cardiovascular points {pts[0, 3]}")

    groups = [K.specimen_group(x) for x in ("BLOOD CULTURE", "MRSA SCREEN", "SPUTUM", "URINE", "TISSUE")]
    ev = pd.DataFrame({"stay_id": [1] * 4, "t": [1.0, 2.0, 3.0, 4.0],
                       "drug": ["Vancomycin", "Vancomycin Oral Liquid", "Piperacillin-Tazobactam", "CeFAZolin"],
                       "route": ["IV", "PO", "IV", "IV"]})
    typed = SRC.typed_events(ev, "abx")
    check("culture specimen groups (a routine MRSA screen is not an infection workup) and antibiotic spectrum "
          "(oral vancomycin treats C. difficile, not MRSA)",
          groups == ["CultureBlood", "CultureRoutine", "CultureResp", "CultureUrine", "CultureOther"]
          and typed.mrsa.tolist() == [True, False, False, False] and typed.broad.tolist() == [False, False, True, False])

    k = int(np.argmax(store.lens))
    sub = E.subset(store, np.array([k]))
    cols_, names = MF.signature_block(sub, ["HR", "MAP_min", "Resp"], (6, 24))
    sig = dict(zip(names, cols_))
    x = {c: MF._locf(sub.col(c).astype(float), sub.row_start) for c in ("HR", "MAP_min", "Resp")}
    d = {c: np.r_[0.0, np.where(np.isfinite(v[1:]) & np.isfinite(v[:-1]), v[1:] - v[:-1], 0.0)] for c, v in x.items()}
    X = {c: np.cumsum(v) for c, v in d.items()}
    worst = 0.0
    for t in range(0, sub.n_rows, 7):
        for w in (6, 24):
            lo = t - w
            base = {c: (X[c][lo] if lo >= 0 else 0.0) for c in X}
            rng_ = range(max(lo + 1, 1), t + 1)
            qv = sum(d["HR"][r] ** 2 for r in range(max(lo + 1, 0), t + 1))
            area = 0.5 * sum((X["HR"][r - 1] - base["HR"]) * d["Resp"][r] - (X["Resp"][r - 1] - base["Resp"]) * d["HR"][r]
                             for r in rng_)
            worst = max(worst, abs(sig[f"HR__qv{w}"][t] - qv), abs(sig[f"HR__levy_Resp_{w}"][t] - area))
    check("signature terms (quadratic variation, Levy area) equal their brute-force iterated sums",
          worst < 1e-6, f"largest difference {worst:.1e} over {sub.n_rows} hours, windows 6 and 24")

    from backend.ml.genesis import learners as L
    from backend.ml.mimic.ledger import Candidate
    from backend.ml.mimic.run import SCALES, booster, pairwise_teachers
    trained = []
    real = L.fit_lgb
    L.fit_lgb = lambda F, rows, *a, **kw: (trained.append(np.asarray(rows)), real(F, rows, *a, **kw))[1]
    try:
        teachers = pairwise_teachers(led, SCALES["demo"], np.zeros(led.store.n_rows))
    finally:
        L.fit_lgb = real
    from itertools import combinations
    f = led.folds[led.store.seg]
    clean = all(not np.isin(f[rows], pair).any() for rows, pair in zip(trained, combinations(range(led.n_folds), 2)))
    covers = all(np.array_equal(np.isfinite(teachers.by_fold[kk]), (f >= 0) & (f != kk) & led.train_mask)
                 for kk in range(led.n_folds))
    student = Candidate("distill", "P", booster(SCALES["demo"], "distill", teacher=teachers))
    cols_p = student.cols(led)
    check("the distilled student reads tier P only; each fold's targets come from teachers that trained on "
          "neither that fold nor the row's own (10 pairwise teachers, not 20)",
          clean and covers and teachers.fits == 10 and (led.ftier[cols_p] == 0).all(),
          f"{teachers.fits} teachers, {len(cols_p)} tier-P columns")


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)       # all-NaN channels of planted worlds
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
    c14_views(store, led)
    c15_transplant(store)
    c16_admission(led)
    c17_infoflow(store, led)
    c18_crossexam(led)
    c19_trust(led)
    c21_information()
    c22_structure(store, led)
    c23_auroc(store, led)
    c24_levers(src, store, led)
    c20_temporal(store, led)
    ok = sum(r[1] for r in _results)
    print(f"\nverify_mimic: {ok}/{len(_results)} passed")
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
