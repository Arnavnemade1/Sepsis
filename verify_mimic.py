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
"""
from __future__ import annotations

import re
import sys

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
    donors = (stays + 3) % store.n_patients
    cf = led.view(stays, CF.transplant(store, stays, donors, False, True, CF.reference(led)))
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
    expect = {n for n in led.names if LG.channel_of(n) == "FiO2"} | \
             {n for n, src in LG.COMPOSITE_SOURCES.items() if "FiO2" in src}
    check("the strict tier drops exactly the covert channel and every composite built on it",
          dropped == expect, f"{len(dropped)} columns ({sorted(dropped - {n for n in dropped if n.startswith('FiO2')})})")


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
    c14_views(store, led)
    c15_transplant(store)
    c16_admission(led)
    c17_infoflow(store, led)
    c18_crossexam(led)
    c19_trust(led)
    ok = sum(r[1] for r in _results)
    print(f"\nverify_mimic: {ok}/{len(_results)} passed")
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
