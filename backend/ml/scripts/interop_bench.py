"""Silent interface faults vs the evidence gate, on patients the detector never saw.

Protocol: ``artifacts/interop/interop_prereg.json`` (committed in 9b03bae before this
file existed). In short:

  * the shipped detectors' promotion holdout (8,062 patients, fingerprint checked);
  * H1: each of 19 realistic faults applied to every holdout observation of its
    column(s), the 510 features rebuilt, the shipped ICU detector scored;
  * H2/H3: the holdout replayed through the DEPLOYED ``Integration`` engine as HL7
    ORU^R01 (one message per patient-hour, LOINC codes, canonical units written,
    patients interleaved by hour since admission); rows 0-1,999 clean, the fault from
    row 2,000 on; what the engine delivered, withheld or quarantined is then scored;
  * H4: clean, per-hospital and septic-ward streams through the same engine, counting
    quarantines;
  * H5: onboarding decisions on each site's first 60 observations per concept.

Writes ``artifacts/interop/interop_bench.json``.

Two operationalisation fixes were made after a smoke run on 400 holdout patients and
before the registered run, both in this script and neither in the gate: in H5(c) an
equivalent unit (10^9/L for 10^3/uL, mEq/L for a monovalent ion) is not counted as a
wrong label; in H5(d) the feed writes the values' true unit (the smoke version wrote
none when that unit could not convert to the wrong concept).

    python -m backend.ml.scripts.interop_bench            # ~15 min, one job at a time
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.ml import config
from backend.aegis.interop import evidence as E
from backend.aegis.interop import hl7
from backend.aegis.interop import vocab as V
from backend.aegis.interop.engine import Integration, Interface
from backend.aegis.interop.units import IDENTITY, Conversion, norm

PREREG = os.path.join(config.ARTIFACTS_DIR, "interop", "interop_prereg.json")
OUT = os.path.join(config.ARTIFACTS_DIR, "interop", "interop_bench.json")
FAULT_START = 2000
UNIT_WRITTEN = {"/min": "/min", "%": "%", "Cel": "Cel", "mm[Hg]": "mm[Hg]", "1": "1", "[pH]": "[pH]",
                "10*3/uL": "10*3/uL", "/uL": "/uL", "g/dL": "g/dL", "mg/dL": "mg/dL", "mmol/L": "mmol/L",
                "U/L": "U/L", "ng/mL": "ng/mL", "s": "s", "L/min": "L/min"}


class NullRooms:
    """The replay has no hospital: every room is free and every effect is dropped."""

    def interface_room_free(self, room: str) -> bool:
        return True

    def __getattr__(self, name: str):
        if name.startswith("interface_"):
            return lambda *a, **k: None
        raise AttributeError(name)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(prereg: Dict[str, Any]):
    from backend.ml.genesis import rawstore
    from backend.ml.scripts.promote_frontier import MANIFEST, promotion_split
    with open(MANIFEST) as fh:
        man = json.load(fh)
    ds, tr, te = promotion_split(man)
    hold = sorted(np.unique(ds.groups[te]).tolist())
    fp = hashlib.sha256("|".join(map(str, hold)).encode()).hexdigest()[:16]
    assert fp == prereg["data"]["holdout_fingerprint"], f"holdout fingerprint {fp} is not the registered one"
    del ds
    raw = rawstore.load()
    pidx = np.flatnonzero(np.isin(raw.pid, np.array(hold)))
    sub = raw.subset(pidx)
    return sub, fp


def stream_order(sub, rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Patient-hours interleaved round-robin by hour since admission (a site-level feed)."""
    seg, hour = sub.seg, sub.hour
    idx = np.arange(sub.n_rows) if rows is None else np.asarray(rows)
    o = np.lexsort((seg[idx], hour[idx]))
    return idx[o]


def apply_fault(X: np.ndarray, cols: List[str], spec: List[Any], columns: List[str], rows: np.ndarray) -> np.ndarray:
    Y = X.copy()
    ci = [columns.index(c) for c in cols]
    if spec[0] == "mul":
        Y[rows, ci[0]] = X[rows, ci[0]] * spec[1]
    elif spec[0] == "affine":
        Y[rows, ci[0]] = X[rows, ci[0]] * spec[1] + spec[2]
    elif spec[0] == "swap":
        a, b = ci
        Y[rows, a], Y[rows, b] = X[rows, b], X[rows, a]
    return Y


# --------------------------------------------------------------------------- #
# the replay: holdout patient-hours through the deployed engine, as HL7
# --------------------------------------------------------------------------- #
def replay(sub, X: np.ndarray, order: np.ndarray, cols: List[str], watch: Optional[List[str]] = None,
           fault_from: Optional[int] = None, restore_unexcluded: bool = False,
           excluded: Optional[callable] = None, progress: str = "") -> Dict[str, Any]:
    """Replay rows (in ``order``) of columns ``cols`` as ORU^R01 through a fresh engine.

    Returns per-observation outcomes for the ``watch`` columns, quarantine events, and
    onboarding results. ``fault_from`` is the stream position where a fault begins (for
    latency). With ``restore_unexcluded`` a quarantine is counted and the mapping put
    back (a person restoring it) so the stream keeps measuring."""
    t = [1_800_000_000.0]
    eng = Integration(rooms=NullRooms(), clock=lambda: t[0], hospital_rooms=None)
    eng.register(Interface("cinc", "CinC replay", "HL7 v2.5.1", "inbound", "HTTP", "CINC", "SITE", "mixed"))
    columns = list(sub.columns)
    ci = {c: columns.index(c) for c in cols}
    concept = {c: V.BY_CINC[c] for c in cols}
    seg, hour = sub.seg, sub.hour
    pid = sub.pid
    admitted = set()
    watch = watch or []
    out_status = {c: np.full(sub.n_rows, -1, np.int8) for c in watch}       # 1 delivered, 0 withheld, 2 other
    out_mapping = {c: None for c in watch}
    q_events: List[Dict[str, Any]] = []
    seen_q = set()
    n_obs = 0
    n_faulty_seen: Dict[str, int] = {c: 0 for c in watch}
    t0 = time.time()
    msh_sender = ("CINC", "SITE")
    for pos, r in enumerate(order):
        p = int(seg[r])
        if p not in admitted:
            admitted.add(p)
            ack, _ = eng.ingest_hl7(hl7.build([
                hl7.msh(msh_sender[0], msh_sender[1], "AEGISOS", "SITE", ("ADT", "A01", "ADT_A01"), f"A{p}", t[0]),
                hl7.segment("PID", "1", "", (str(pid[p]), "", "", "SITE", "MR")),
                hl7.segment("PV1", "1", "I", ("ICU", str(pid[p]), "1", "SITE"))]))
        obx = []
        present = []
        for c in cols:
            v = X[r, ci[c]]
            if v != v:
                continue
            k = concept[c]
            obx.append(hl7.segment("OBX", str(len(obx) + 1), "NM", (k.loinc, k.loinc_display[:40], "LN"), "",
                                   float(v), UNIT_WRITTEN.get(k.unit, k.unit), "", "", "", "", "F"))
            present.append(c)
        if not obx:
            continue
        t[0] += 1.0
        ack, rec = eng.ingest_hl7(hl7.build([hl7.msh(msh_sender[0], msh_sender[1], "AEGISOS", "SITE",
                                                     ("ORU", "R01", "ORU_R01"), f"R{pos}", t[0]),
                                             hl7.segment("PID", "1", "", (str(pid[p]), "", "", "SITE", "MR"))] + obx))
        n_obs += len(present)
        for c, o in zip(present, rec["outcomes"]):
            if c in out_status:
                st = o["status"]
                out_status[c][r] = 1 if st == "delivered" else 0 if st.startswith("withheld") else 2
                out_mapping[c] = o["mapping"]
                if fault_from is not None and pos >= fault_from:
                    n_faulty_seen[c] += 1
        for m in eng.by_id.values():
            if m.status == "quarantined" and m.id not in seen_q:
                seen_q.add(m.id)
                c = next((cc for cc in cols if V.BY_CINC[cc].key == m.concept), m.code)
                ev = {"pos": pos, "obs": n_obs, "column": c, "concept": m.concept,
                      "cause": (m.fault or {}).get("cause"), "suggested": (m.fault or {}).get("suggested"),
                      "by_identity": (m.fault or {}).get("by_identity"),
                      "faulty_seen": dict(n_faulty_seen), "excluded": bool(excluded and excluded(m, rec))}
                q_events.append(ev)
                if restore_unexcluded:
                    eng.restore(m.id, "bench")
                    seen_q.discard(m.id)
        if progress and pos and pos % 100000 == 0:
            print(f"    {progress}: {pos:,} rows  [{time.time() - t0:.0f}s]", flush=True)
    maps = {}
    for m in eng.by_id.values():
        c = next((cc for cc in cols if V.BY_CINC[cc].loinc == m.code), m.code)
        maps[c] = {"status": m.status, "concept": m.concept, "conversion": m.conversion.label if m.conversion else None,
                   "reason": m.reason, "fault": m.fault, "n": len(m.samples), "counts": dict(m.counts),
                   "mapping": m.id}
    return {"status": out_status, "mapping_of": out_mapping, "quarantines": q_events, "mappings": maps,
            "observations": n_obs, "seconds": round(time.time() - t0, 1), "engine": eng}


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(det, sub, X: np.ndarray) -> np.ndarray:
    from backend.ml.genesis import features as feat
    from backend.ml.genesis.rawstore import RawStore
    rs = RawStore(X=np.ascontiguousarray(X, np.float32), y=sub.y, lens=sub.lens, pid=sub.pid, site=sub.site,
                  columns=list(sub.columns))
    F, names = feat.build(rs, det.feature_spec())
    assert names == det.feature_names
    p = det.predict_matrix(F)
    del F
    return p


def alarm_rates(p: np.ndarray, y: np.ndarray, seg: np.ndarray, thr: float) -> Tuple[float, float]:
    P = seg.max() + 1
    pmax = np.full(P, -1.0)
    np.maximum.at(pmax, seg, p)
    ylab = np.zeros(P, np.int8)
    np.maximum.at(ylab, seg, y.astype(np.int8))
    alarmed = pmax >= thr
    return float(alarmed[ylab == 0].mean()), float(alarmed[ylab == 1].mean())


def paired_boot(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray, n: int = 1000,
                seed: int = 0) -> Dict[str, float]:
    from backend.ml.genesis import harness as H
    Ra, Rb = H.Ranked(a), H.Ranked(b)
    _, inv = np.unique(groups, return_inverse=True)
    P = inv.max() + 1
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for j in range(n):
        w = np.bincount(rng.integers(0, P, P), minlength=P)[inv].astype(np.float64)
        d[j] = Ra.auc(y, w) - Rb.auc(y, w)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(Ra.auc(y) - Rb.auc(y)), "lo": float(lo), "hi": float(hi)}


def identity_partners(cols: List[str]) -> List[str]:
    keys = {V.BY_CINC[c].key for c in cols}
    extra = set()
    for inv in E.INVARIANTS:
        if keys & set(inv.concepts):
            for k in inv.concepts:
                c = V.BY_KEY[k].cinc
                if c:
                    extra.add(c)
    return sorted(set(cols) | extra)


# --------------------------------------------------------------------------- #
# H5: onboarding decisions
# --------------------------------------------------------------------------- #
def auto_adopts(c: V.Concept, values: List[float], written: str) -> Tuple[bool, str]:
    """The engine's auto-adopt rule for one proposal (engine._propose)."""
    tok = norm(written)
    declared = c.conversion(tok) if tok else None
    inf = E.infer(c, values, declared)
    if inf.decision == "declared":
        return True, "declared"
    if declared is None and inf.decision == "inferred" and inf.best is not None and inf.best.conversion.identity:
        return True, "identity (no unit)"
    return False, inf.decision


def alt_units(c: V.Concept) -> List[Tuple[str, float, float]]:
    """(token, factor, offset) for each unit this analyte is reported in, besides canonical."""
    out = [(tok, f, o) for tok, f, o in c.molar]
    extra = {"hemoglobin": ["g/L"], "wbc": ["/uL", "10*9/L"], "platelets": ["/uL", "10*9/L"], "temp": ["[degF]"],
             "fio2": ["%"], "hematocrit": ["1"], "spo2": ["1"], "sao2": ["1"], "troponin": ["ng/L"],
             "paco2": ["kPa"], "etco2": ["kPa"], "sbp": ["kPa"], "dbp": ["kPa"], "map": ["kPa"],
             "fibrinogen": ["g/L"], "ast": ["ukat/L"], "alp": ["ukat/L"]}
    for tok in extra.get(c.key, []):
        cv = c.conversion(tok)
        if cv is not None and all(t != tok for t, _f, _o in out):
            out.append((tok, cv.factor, cv.offset))
    return out


def h5(sub, order: np.ndarray, blind: List[List[str]]) -> Dict[str, Any]:
    columns = list(sub.columns)
    site_row = sub.site[sub.seg]
    res = {"a": [0, 0], "b": [0, 0], "c": [0, 0], "d": [0, 0], "d_blind": [0, 0], "unsafe_c": [], "unsafe_d": [],
           "unsafe_d_blind": [], "not_adopted_a": []}
    blind_set = {tuple(p) for p in blind}
    for col in columns[:34]:
        c = V.BY_CINC.get(col)
        if c is None or c.key not in V.references():
            continue
        v = sub.X[:, columns.index(col)]
        for s in (0, 1):
            rows = [r for r in order if site_row[r] == s and v[r] == v[r]][:60]
            if len(rows) < 30:
                continue
            vals = [float(v[r]) for r in rows]
            if (col == "Calcium" and s == 1) or (col == "SaO2" and s == 0):
                continue                                  # audited contaminated: reported in H4
            ok, why = auto_adopts(c, vals, c.unit)
            res["a"][0] += ok
            res["a"][1] += 1
            if not ok:
                res["not_adopted_a"].append(f"{c.key}@{'AB'[s]}: {why}")
            for tok, f, o in alt_units(c):
                raw = [(x - o) / f for x in vals]           # the same values, in that unit
                ok_b, _ = auto_adopts(c, raw, tok)
                res["b"][0] += ok_b
                res["b"][1] += 1
                if abs(f - 1.0) < 1e-9 and o == 0.0:
                    continue        # an equivalent unit (10^9/L = 10^3/uL, mEq/L = mmol/L) is not a wrong label
                bad, _ = auto_adopts(c, vals, tok)          # canonical values under that label
                res["c"][0] += bad
                res["c"][1] += 1
                if bad:
                    res["unsafe_c"].append(f"{c.key}@{'AB'[s]} labelled {tok}")
            for c2 in V.CONCEPTS:
                if c2.key == c.key or c2.key not in V.references():
                    continue
                # the feed writes the values' TRUE unit; only the proposal is wrong
                bad, _ = auto_adopts(c2, vals, c.unit)
                key = "d_blind" if (c.key, c2.key) in blind_set else "d"
                res[key][0] += bad
                res[key][1] += 1
                if bad:
                    res["unsafe_" + key].append(f"{c.key}@{'AB'[s]} as {c2.key}")
    return res


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# --------------------------------------------------------------------------- #
# descriptive: proposers on a fixed catalogue of local code names
# --------------------------------------------------------------------------- #
CATALOGUE = [
    # (dialect, code, display text, unit, true concept or None)
    ("US", "CREAT", "Creatinine, Serum", "mg/dL", "creatinine"), ("US", "BUN", "Urea Nitrogen, Blood", "mg/dL", "bun"),
    ("US", "K", "Potassium", "mmol/L", "potassium"), ("US", "NA", "Sodium", "mmol/L", "sodium"),
    ("US", "CL", "Chloride", "mmol/L", "chloride"), ("US", "CO2", "Carbon Dioxide", "mmol/L", "hco3"),
    ("US", "GLU", "Glucose, Random", "mg/dL", "glucose"), ("US", "WBC", "WBC Count", "K/uL", "wbc"),
    ("US", "HGB", "Hemoglobin", "g/dL", "hemoglobin"), ("US", "HCT", "Hematocrit", "%", "hematocrit"),
    ("US", "PLT", "Platelet Count", "K/uL", "platelets"), ("US", "LACTA", "Lactic Acid, Plasma", "mmol/L", "lactate"),
    ("US", "TBIL", "Bilirubin, Total", "mg/dL", "bilirubin"), ("US", "DBIL", "Bilirubin, Direct", "mg/dL", "bilirubin_direct"),
    ("US", "AST", "AST (SGOT)", "U/L", "ast"), ("US", "ALKP", "Alkaline Phosphatase", "U/L", "alp"),
    ("US", "TROPI", "Troponin I", "ng/mL", "troponin"), ("US", "PTT", "PTT", "sec", "ptt"),
    ("US", "FIB", "Fibrinogen", "mg/dL", "fibrinogen"), ("US", "MG", "Magnesium", "mg/dL", "magnesium"),
    ("US", "PHOS", "Phosphorus", "mg/dL", "phosphate"), ("US", "CA", "Calcium", "mg/dL", "calcium"),
    ("US", "PCT", "Procalcitonin", "ng/mL", None), ("US", "CRP", "C-Reactive Protein", "mg/L", None),
    ("VENDOR", "2160-0", "CREATININE (ENZYMATIC)", "mg/dL", "creatinine"),
    ("VENDOR", "LAB1032", "WHITE BLOOD CELL COUNT", "x10E3/uL", "wbc"),
    ("VENDOR", "LAB2210", "LACTATE, WHOLE BLOOD (ABG)", "mmol/L", "lactate"),
    ("VENDOR", "LAB7781", "POTASSIUM, WHOLE BLOOD (ISE)", "mmol/L", "potassium"),
    ("VENDOR", "LAB5521", "PH, ARTERIAL", "", "ph"), ("VENDOR", "LAB5522", "PCO2, ARTERIAL", "mmHg", "paco2"),
    ("VENDOR", "LAB5523", "O2 SATURATION, ARTERIAL (CALC)", "%", "sao2"),
    ("VENDOR", "LAB9001", "HEMOGLOBIN A1C", "%", None), ("VENDOR", "LAB3333", "INR", "", "inr"),
    ("NPU", "NPU18016", "P-Kreatinin", "umol/L", "creatinine"), ("NPU", "NPU03429", "P-Natrium", "mmol/L", "sodium"),
    ("NPU", "NPU03230", "P-Kalium", "mmol/L", "potassium"), ("NPU", "NPU02319", "B-Hb", "mmol/L", "hemoglobin"),
    ("NPU", "NPU02593", "B-Leukocytter", "10*9/L", "wbc"), ("NPU", "NPU03568", "B-Trombocytter", "10*9/L", "platelets"),
    ("NPU", "NPU19748", "P-Laktat", "mmol/L", "lactate"), ("NPU", "NPU02192", "P-Glukose", "mmol/L", "glucose"),
    ("NPU", "NPU01370", "P-Bilirubiner", "umol/L", "bilirubin"), ("NPU", "NPU19651", "P-Albumin", "g/L", None),
]


def proposer_accuracy(llm: Any) -> Dict[str, Any]:
    from backend.aegis.interop.proposer import ask_model, coded, lexical
    rows = []
    items = [{"code": code, "text": text, "unit": unit, "values": []} for _d, code, text, unit, _k in CATALOGUE]
    model, model_id = ({}, None)
    if llm is not None and getattr(llm, "available", False):
        model, model_id = ask_model(llm, items)
    for d, code, text, unit, truth in CATALOGUE:
        cp = coded(code, text, "LN" if code[:1].isdigit() and "-" in code else "L")
        lx = lexical(code, text, unit)
        lex = cp.concept if cp else (lx[0].concept if lx and lx[0].score >= 0.75 else None)
        mdl = model[code].concept if code in model else None
        rows.append({"dialect": d, "code": code, "text": text, "truth": truth, "lexical": lex, "model": mdl})

    def acc(key: str, dialect: Optional[str] = None) -> Dict[str, Any]:
        rr = [r for r in rows if dialect in (None, r["dialect"])]
        right = sum(1 for r in rr if r[key] == r["truth"])
        wrong = sum(1 for r in rr if r[key] is not None and r[key] != r["truth"])
        return {"n": len(rr), "right": right, "wrong_concept": wrong, "abstained": len(rr) - right - wrong}
    return {"model": model_id, "rows": rows,
            "lexical": {d: acc("lexical", d) for d in (None, "US", "VENDOR", "NPU")},
            "model_acc": ({d: acc("model", d) for d in (None, "US", "VENDOR", "NPU")} if model_id else None)}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--smoke", type=int, default=0,
                    help="bug check only: the first N patients, two faults, output to a scratch file")
    args = ap.parse_args()
    t_all = time.time()
    with open(PREREG) as fh:
        prereg = json.load(fh)
    import joblib
    sub, fp = load(prereg)
    out_path = OUT
    if args.smoke:
        sub = sub.subset(np.arange(min(args.smoke, sub.n_patients)))
        prereg = {**prereg, "faults": [f for f in prereg["faults"] if f["id"] in ("creatinine_umol", "swap_sbp_map")]}
        out_path = os.path.join(config.ARTIFACTS_DIR, "interop", "interop_bench_smoke.json")
        args.boot = 50
    det = joblib.load(os.path.join(config.ARTIFACTS_DIR, "gbm_icu.joblib"))
    assert det.detector_id == prereg["data"]["detector"].split(" ")[0], det.detector_id
    thr = float(prereg["data"]["threshold"])
    columns = list(sub.columns)
    y, seg = sub.y.astype(int), sub.seg
    groups = seg
    print(f"holdout {sub.n_patients:,} patients / {sub.n_rows:,} rows (fingerprint {fp}); detector {det.detector_id}")
    X0 = np.array(sub.X, dtype=np.float32)
    p_clean = score(det, sub, X0)
    from backend.ml.eval.metrics import auroc
    from backend.ml.genesis import harness as H
    base = {"auroc": auroc(y, p_clean), "hour_matched": H.hour_matched_auroc(y, p_clean, sub.hour)}
    base["far"], base["detection"] = alarm_rates(p_clean, y, seg, thr)
    print(f"clean: AUROC {base['auroc']:.4f}  FAR {base['far']:.3f}  detection {base['detection']:.3f}")
    order = stream_order(sub)
    pos_of = np.empty(sub.n_rows, np.int64)
    pos_of[order] = np.arange(len(order))
    late = order[FAULT_START:]                   # rows the fault affects in the stream arms

    results: Dict[str, Any] = {"prereg": "artifacts/interop/interop_prereg.json", "fingerprint": fp,
                               "detector": det.detector_id, "threshold": thr, "clean": base, "faults": {}}
    for f in prereg["faults"]:
        fid, cols, spec = f["id"], f["columns"].split("|"), f["transform"]
        t0 = time.time()
        # H1: silent, every observation
        Xs = apply_fault(X0, cols, spec, columns, np.arange(sub.n_rows))
        p_s = score(det, sub, Xs)
        far_s, det_s = alarm_rates(p_s, y, seg, thr)
        silent = {"auroc": auroc(y, p_s), "far": far_s, "detection": det_s,
                  "vs_clean": paired_boot(y, p_s, p_clean, groups, args.boot, 1)}
        del Xs
        # H2/H3: the stream through the deployed engine, fault from row FAULT_START
        Xf = apply_fault(X0, cols, spec, columns, late)
        watch = cols
        rp = replay(sub, Xf, order, identity_partners(cols), watch=watch, fault_from=FAULT_START)
        q = [e for e in rp["quarantines"] if e["column"] in cols]
        detected = bool(q)
        first = min(q, key=lambda e: e["pos"]) if q else None
        faulty_to_q = (first["faulty_seen"].get(first["column"], 0) if first else None)
        late_mask = np.zeros(sub.n_rows, bool)
        late_mask[late] = True
        exposure = {c: int(((rp["status"][c] == 1) & late_mask & ~np.isnan(Xf[:, columns.index(c)])).sum())
                    for c in cols}
        n_faulty = {c: int((late_mask & ~np.isnan(Xf[:, columns.index(c)])).sum()) for c in cols}
        # arms
        Xq = X0.copy()
        Xfix = X0.copy()
        Xmiss = X0.copy()
        fixes = {}
        for c in cols:
            j = columns.index(c)
            st = rp["status"][c]
            vals = Xf[:, j]
            obs = ~np.isnan(vals)
            m = late_mask & obs
            Xmiss[late_mask, j] = np.nan
            deliver = m & (st == 1)
            Xq[m, j] = np.where(deliver[m], vals[m], np.nan)
            mp = rp["mappings"].get(c) or {}
            sug = ((mp.get("fault") or {}).get("suggested"))
            fixes[c] = sug
            if sug:
                fixed = vals * sug["factor"] + sug["offset"]
                Xfix[m, j] = np.where(deliver[m], vals[m], fixed[m])
            else:
                Xfix[m, j] = Xq[m, j]
        p_q, p_fix, p_m = score(det, sub, Xq), score(det, sub, Xfix), score(det, sub, Xmiss)
        arms = {}
        for name, p in (("gate_quarantine", p_q), ("gate_fix", p_fix), ("variable_missing", p_m)):
            fa, de = alarm_rates(p, y, seg, thr)
            arms[name] = {"auroc": auroc(y, p), "far": fa, "detection": de,
                          "vs_clean": paired_boot(y, p, p_clean, groups, args.boot, 2)}
        arms["gate_quarantine"]["vs_missing"] = paired_boot(y, p_q, p_m, groups, args.boot, 3)
        results["faults"][fid] = {
            "what": f["what"], "columns": cols, "silent": silent, "detected": detected,
            "quarantine": first, "faulty_to_quarantine": faulty_to_q, "faulty_observations": n_faulty,
            "delivered_faulty": exposure, "fixes": fixes, "arms": arms,
            "onboarding": {c: rp["mappings"].get(c, {}).get("status") for c in cols},
            "replay_seconds": rp["seconds"], "seconds": round(time.time() - t0, 1)}
        r = results["faults"][fid]
        print(f"  {fid:18s} silent ΔAUROC {silent['vs_clean']['delta']:+.4f} [{silent['vs_clean']['lo']:+.4f},"
              f"{silent['vs_clean']['hi']:+.4f}] ΔFAR {far_s - base['far']:+.3f} | "
              f"{'Q@' + str(faulty_to_q) if detected else 'NOT detected'} delivered {sum(exposure.values())} | "
              f"quar {arms['gate_quarantine']['vs_clean']['delta']:+.4f} fix {arms['gate_fix']['vs_clean']['delta']:+.4f} "
              f"miss {arms['variable_missing']['vs_clean']['delta']:+.4f}  [{r['seconds']:.0f}s]", flush=True)
        del Xf, Xq, Xfix, Xmiss

    # H4: false quarantines
    site_row = sub.site[seg]
    allcols = [c for c in columns[:34] if c in V.BY_CINC]

    def excluded_for(site_set):
        def f(m, rec):
            cause = (m.fault or {}).get("cause") or ""
            if m.concept == "calcium" and 1 in site_set:
                return True
            if (m.concept in ("sao2", "spo2") and 0 in site_set
                    and ("arterial saturation" in cause or "SaO" in cause or m.concept == "sao2")):
                return True
            return False
        return f

    h4 = {}
    streams = {"clean": (order, {0, 1}), "site_A": (stream_order(sub, np.flatnonzero(site_row == 0)), {0}),
               "site_B": (stream_order(sub, np.flatnonzero(site_row == 1)), {1})}
    ylab = np.zeros(sub.n_patients, np.int8)
    np.maximum.at(ylab, seg, sub.y.astype(np.int8))
    onset = sub.onset_hour()
    sept_rows = np.flatnonzero((ylab[seg] == 1) & (sub.hour >= np.maximum(onset[seg] - 24, 0)))
    ordinary = order[:min(20000, len(order) // 3)]
    streams["septic_ward"] = (np.concatenate([ordinary, stream_order(sub, sept_rows)]), {0, 1})
    for name, (o, sites) in streams.items():
        rp = replay(sub, X0, o, allcols, watch=[], restore_unexcluded=True, excluded=excluded_for(sites),
                    progress=f"H4 {name}")
        qs = rp["quarantines"]
        if name == "septic_ward":
            n_ord = sum(int((~np.isnan(X0[ordinary][:, [columns.index(c) for c in allcols]])).sum()) for _ in [0])
            qs = [e for e in qs if e["obs"] > n_ord]
            obs = rp["observations"] - n_ord
        else:
            obs = rp["observations"]
        false_q = [e for e in qs if not e["excluded"]]
        h4[name] = {"observations": obs, "quarantines": len(qs), "false": len(false_q),
                    "per_100k": 1e5 * len(false_q) / max(1, obs), "excluded": [e for e in qs if e["excluded"]][:6],
                    "false_events": false_q[:12],
                    "refused_at_onboarding": {c: v["status"] for c, v in rp["mappings"].items()
                                              if v["status"] not in ("adopted", "quarantined")},
                    "seconds": rp["seconds"]}
        print(f"  H4 {name:12s} {obs:,} obs: {len(false_q)} false quarantines ({h4[name]['per_100k']:.2f}/100k), "
              f"{len(qs) - len(false_q)} excluded; not adopted: {h4[name]['refused_at_onboarding']}", flush=True)
    results["h4"] = h4

    # H5: onboarding
    r5 = h5(sub, order, prereg["hypotheses"]["H5_onboarding"]["blind_pairs"])
    results["h5"] = {k: v for k, v in r5.items()}
    for k in ("a", "b", "c", "d", "d_blind"):
        kk, nn = r5[k]
        lo, hi = wilson(kk, nn)
        results["h5"][k + "_rate"] = {"k": kk, "n": nn, "rate": kk / max(1, nn), "wilson": [lo, hi]}
    print(f"  H5 adopted correct a {r5['a']}, b {r5['b']}; unsafe c {r5['c']}, d {r5['d']}, d_blind {r5['d_blind']}")

    # descriptive: proposers
    llm = None
    if not args.no_llm:
        try:
            from backend.llm.provider import LLMProvider
            llm = LLMProvider()
        except Exception:
            llm = None
    results["proposers"] = proposer_accuracy(llm)
    print(f"  proposers: lexical {results['proposers']['lexical'][None]}, model {results['proposers']['model']} "
          f"{(results['proposers']['model_acc'] or {}).get(None)}")

    # verdicts against the registered bars
    F = results["faults"]
    harmful = [k for k, v in F.items() if (v["silent"]["vs_clean"]["delta"] < -0.01 and v["silent"]["vs_clean"]["hi"] < 0)
               or abs(v["silent"]["far"] - base["far"]) > 0.05]
    det_list = [k for k, v in F.items() if v["detected"]]
    lat = [v["faulty_to_quarantine"] for v in F.values() if v["detected"]]
    h3_rows = []
    for k in harmful:
        v = F[k]
        if not v["detected"]:
            continue
        gq = v["arms"]["gate_quarantine"]["vs_missing"]["delta"]
        has_fix = any(v["fixes"].values())
        gf = v["arms"]["gate_fix"]["vs_clean"]["delta"] if has_fix else None
        h3_rows.append({"fault": k, "quarantine_vs_missing": gq, "fix_vs_clean": gf,
                        "pass": abs(gq) <= 0.005 and (gf is None or abs(gf) <= 0.002)})
    a_rate = (r5["a"][0] + r5["b"][0]) / max(1, r5["a"][1] + r5["b"][1])
    results["verdicts"] = {
        "H1_harm": {"harmful": harmful, "pass": len(harmful) >= 3},
        "H2_detection": {"detected": det_list, "missed": [k for k in F if k not in det_list],
                         "median_faulty_to_quarantine": (float(np.median(lat)) if lat else None),
                         "pass": len(det_list) >= 15 and bool(lat) and float(np.median(lat)) <= 40},
        "H3_recovery": {"rows": h3_rows, "pass": all(r["pass"] for r in h3_rows) and bool(h3_rows)},
        "H4_false_quarantine": {k: {"per_100k": v["per_100k"], "pass": v["per_100k"] <= 1.0} for k, v in h4.items()},
        "H5_onboarding": {"correct_auto_adopted": a_rate, "unsafe_wrong_unit": results["h5"]["c_rate"],
                          "unsafe_wrong_concept": results["h5"]["d_rate"],
                          "unsafe_wrong_concept_blind": results["h5"]["d_blind_rate"],
                          "pass": a_rate >= 0.80 and results["h5"]["c_rate"]["rate"] <= 0.01
                          and results["h5"]["d_rate"]["rate"] <= 0.01},
    }
    results["seconds"] = round(time.time() - t_all)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(results["verdicts"], indent=1, default=str))
    print(f"wrote {out_path} [{results['seconds']}s]")


if __name__ == "__main__":
    main()
