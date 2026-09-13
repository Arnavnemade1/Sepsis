"""Planted-failure suite for the host-response study.

Every guard in this study exists because a specific confident-wrong-answer was
possible, and several of them were found the hard way. So each check PLANTS that
failure and asserts the guard fires — and where a guard could pass for the wrong
reason, it is paired with a comparator proving the check can fail at all. A green
assertion is not evidence until it has been seen to go red.

All data here is synthetic and small, so the suite is fast and never touches a
real cohort. In particular the run-once check replaces cohort loading with a
sentinel that raises: if that guard ever broke, this suite would fail loudly
instead of quietly scoring the locked cohort.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import List, Optional

import numpy as np
import pandas as pd

from backend.hostresponse import prereg
from backend.hostresponse import run_study as rs
from backend.hostresponse.data import build_geo_manifest as gm
from backend.hostresponse.data import geo_series as gs
from backend.hostresponse.expression import (Cohort, GeoCohort, align,
                                             cohort_identity_auroc,
                                             collapse_to_symbols, loco,
                                             mean_direction_ci, permutation_loco,
                                             rank_within_sample)
from backend.ml.eval import metrics
from backend.ml.jsonio import json_safe

results: List[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" — " + detail) if detail else ""))


def synth(tag: str, n_pos: int = 40, n_neg: int = 25, reps: int = 1, n_genes: int = 300,
          k_sig: int = 25, effect: float = 1.4, batch: Optional[pd.Series] = None,
          seed: int = 0) -> Cohort:
    """A synthetic cohort: patient-level expression, optional repeats, optional batch."""
    rng = np.random.default_rng(seed)
    genes = ["G%d" % j for j in range(n_genes)]
    cols, y, patients, data = [], [], [], []
    for i in range(n_pos + n_neg):
        lab = 1 if i < n_pos else 0
        profile = rng.normal(8.0, 2.0, n_genes)
        profile[:k_sig] += effect * 2.0 * lab
        for r in range(reps):
            data.append(profile + rng.normal(0.0, 0.3, n_genes))
            cols.append("%s_s%d_%d" % (tag, i, r))
            y.append(lab)
            patients.append("%s:%d" % (tag, i))
    raw = pd.DataFrame(np.array(data).T, index=genes, columns=cols)
    if batch is not None:                       # gene-specific: changes within-sample RANKS
        raw = raw.add(batch.reindex(genes).to_numpy(), axis=0)
    return Cohort(accession=tag, platform="SYN", X=rank_within_sample(raw),
                  y=np.array(y), samples=cols, patients=patients)


# --------------------------------------------------------------------------- #
print("\n[1] the normalisation computes nothing across samples")
rng = np.random.default_rng(0)
raw = pd.DataFrame(rng.normal(8, 2, (200, 12)), index=["G%d" % i for i in range(200)])
base = rank_within_sample(raw)
t = raw.copy()
t[0] = np.exp(t[0] / 3.0)
t[1] = t[1] * 7.0 - 4.0
check("invariant to a monotone per-sample transform", np.allclose(base.values, rank_within_sample(t).values))
check("deleting other samples leaves a sample's features unchanged",
      np.allclose(base.iloc[:, :5].values, rank_within_sample(raw.iloc[:, :5]).values))
cross = lambda d: d.sub(d.mean(axis=1), axis=0).div(d.std(axis=1), axis=0)
check("...and that test CAN fail: a cross-sample per-gene z-score does not survive deletion",
      not np.allclose(cross(raw).iloc[:, :5].values, cross(raw.iloc[:, :5]).values))
sym = {"G%d" % i: "S%d" % (i // 2) for i in range(200)}
check("collapsing probes ON THE RANKS keeps the invariance",
      np.allclose(collapse_to_symbols(base, sym).values,
                  collapse_to_symbols(rank_within_sample(t), sym).values))

# --------------------------------------------------------------------------- #
print("\n[2] the primary interval cannot be moved by a cohort offset")
A, B = synth("A", seed=1), synth("B", seed=2)
al = align([A, B])
lo_ = loco(al)
ci0 = mean_direction_ci(al.y, lo_["pooled"], al.cohort, al.patient_ids, n_boot=300, seed=1)
shifted = lo_["pooled"].copy()
shifted[al.cohort == "B"] += 3.0
ci1 = mean_direction_ci(al.y, shifted, al.cohort, al.patient_ids, n_boot=300, seed=1)
check("synthetic transfer is real, so the controls below have something to beat",
      ci0["point_mean"] > 0.85, "%.3f" % ci0["point_mean"])
check("mean and interval identical under a +3.0 offset on one cohort",
      abs(ci0["point_mean"] - ci1["point_mean"]) < 1e-12 and ci0["ci_low"] == ci1["ci_low"])
pb, ps = metrics.auroc(al.y, lo_["pooled"]), metrics.auroc(al.y, shifted)
check("...while a POOLED AUROC on the same scores moves", abs(pb - ps) > 0.01, "%.3f -> %.3f" % (pb, ps))

# --------------------------------------------------------------------------- #
print("\n[3] cohort identity: detected when planted, reported rather than voiding")
batch = pd.Series(np.random.default_rng(9).normal(0, 1.5, 300), index=["G%d" % i for i in range(300)])
ident_batch = cohort_identity_auroc(align([A, synth("B", seed=2, batch=batch)]))
ident_clean = cohort_identity_auroc(align([A, synth("B", seed=3)]))
check("a planted gene-specific batch effect is learnable as cohort identity",
      ident_batch >= 0.90, "%.3f" % ident_batch)
check("...and two identically generated cohorts are not", ident_clean < 0.75, "%.3f" % ident_clean)
doc = prereg.load()
v = rs.evaluate_verdicts(dict(primary_mean=0.9, ci_low=0.8, ci_high=0.95,
                              per_direction={"A": 0.9, "B": 0.9}, perm_mean=0.5,
                              perm_draws=20, cohort_identity=ident_batch), doc)
check("that planted batch reads CONFOUNDED under the frozen rules, and is flagged as the "
      "outcome amendment 4 changed",
      v["under_frozen_conditions"]["verdict"] == "CONFOUNDED"
      and v["in_force"]["verdict"] == "PASSED" and v["amendment_changed_the_outcome"])

# --------------------------------------------------------------------------- #
print("\n[4] the permutation null centres on chance, and genuinely permutes")
null = permutation_loco(al, seed=7, n_draws=20)
check("null mean over 20 draws is inside [0.42, 0.58]", 0.42 <= null["mean"] <= 0.58,
      "mean %.3f, sd %.3f" % (null["mean"], null["sd"]))
check("...and sits far below the real transfer, so a no-op shuffle would be caught",
      ci0["point_mean"] - null["mean"] > 0.25, "%.3f vs %.3f" % (ci0["point_mean"], null["mean"]))

# --------------------------------------------------------------------------- #
print("\n[5] a patient is the unit, not a sample")


class _Series:
    pass


fs = _Series()
fs.samples = ["s1", "s2", "s3", "s4"]
fs.characteristics = {"id": ["7", "7", "9", "9"]}
key, pats = GeoCohort("ACC", "SYN")._patients(fs, fs.samples)
check("a repeated id collapses samples into patients",
      key == "id" and pats == ["ACC:7", "ACC:7", "ACC:9", "ACC:9"])
fs.characteristics = {"id": ["1", "2", "3", "4"]}
key2, pats2 = GeoCohort("ACC", "SYN")._patients(fs, fs.samples)
check("an id unique per row names the SAMPLE, and is refused as a patient key",
      key2 is None and len(set(pats2)) == 4)

rng = np.random.default_rng(4)
y5, p5, g5 = [], [], []
for i in range(40):
    lab = int(i < 26)
    s = rng.normal(1.0 * lab, 1.0)
    for _ in range(5):
        y5.append(lab); p5.append(s + rng.normal(0, 0.1)); g5.append("C:%d" % i)
y5, p5, g5 = np.array(y5), np.array(p5), np.array(g5)
pat = mean_direction_ci(y5, p5, np.array(["C"] * len(y5)), g5, n_boot=600, seed=2)
row_lo, row_hi = metrics.bootstrap_ci(y5, p5, np.arange(len(y5)), n_boot=600, seed=2)
w_pat, w_row = pat["ci_high"] - pat["ci_low"], row_hi - row_lo
check("five repeats per patient do not narrow the interval as if they were independent",
      w_row < 0.75 * w_pat, "patient-clustered width %.3f vs row-level %.3f" % (w_pat, w_row))

# --------------------------------------------------------------------------- #
print("\n[6] the manifest gate admits sterile controls and never healthy ones")


def _series(labels: List[str]) -> gs.GeoSeries:
    n = len(labels)
    return gs.GeoSeries(accession="GSEX", title="t", platforms=["GPL1"],
                        samples=["a%d" % i for i in range(n)],
                        characteristics={"diagnosis": labels},
                        raw_characteristics=[["diagnosis: %s" % x] for x in labels])


ch = gm.classify(_series(["sepsis"] * 30 + ["healthy control"] * 30))
check("a healthy-volunteer control arm is never admissible",
      ch["control_type"] == "healthy" and not ch["inclusion_ok"])
cs = gm.classify(_series(["sepsis"] * 30 + ["POST_SURGICAL"] * 20))
check("a genuine sterile arm is — including when written POST_SURGICAL",
      cs["inclusion_ok"] and cs["control_type"] in gm.STERILE_TYPES)
one = gm.gate([dict(accession="X", inclusion_ok=True, platforms=["GPL1"])])
same = gm.gate([dict(accession="X", inclusion_ok=True, platforms=["GPL1"]),
                dict(accession="Y", inclusion_ok=True, platforms=["GPL1"])])
two = gm.gate([dict(accession="X", inclusion_ok=True, platforms=["GPL1"]),
               dict(accession="Y", inclusion_ok=True, platforms=["GPL2"])])
check("the gate refuses one cohort and one platform, and passes two platforms",
      (not one["passed"]) and (not same["passed"]) and two["passed"])

# --------------------------------------------------------------------------- #
print("\n[7] the pre-registration binds the locked phase")
check("withdrawn cohorts take no part",
      [c["accession"] for c in prereg.admissible(doc)] == ["GSE74224", "GSE40012"])
refused = []
for fn in (lambda: prereg.require_unchanged(None),
           lambda: prereg.require_unchanged("0" * 16)):
    try:
        fn()
        refused.append(False)
    except SystemExit:
        refused.append(True)
check("a missing discovery record and a moved digest both refuse", all(refused))
check("the unchanged document is accepted",
      prereg.require_unchanged(prereg.digest(doc)) == prereg.digest(doc))

saved = (rs.ARTIFACT, rs._cohort_specs, prereg.recorded_digest)
ok = False
with tempfile.TemporaryDirectory() as td:
    fake = os.path.join(td, "hostresponse.json")
    with open(fake, "w") as fh:
        json.dump({"phase": "locked", "prereg_digest": "x"}, fh)

    def _sentinel(_doc):
        raise AssertionError("run-once guard FAILED: cohorts would have been loaded and scored")

    rs.ARTIFACT, rs._cohort_specs = fake, _sentinel
    prereg.recorded_digest = lambda path=prereg.DISCOVERY_RECORD: prereg.digest(doc)
    try:
        rs.run_locked(doc, argparse.Namespace(save=True, boot=10, perm_draws=20, seed=42))
    except SystemExit:
        ok = True
    except AssertionError as e:
        print("   ", e)
    finally:
        rs.ARTIFACT, rs._cohort_specs, prereg.recorded_digest = saved
check("an existing locked result refuses a second scoring before any cohort loads", ok)
try:
    rs.run_locked(doc, argparse.Namespace(save=False, boot=10, perm_draws=20, seed=42))
    unsaved = False
except SystemExit:
    unsaved = True
check("a locked run that would not record its result is refused", unsaved)

# --------------------------------------------------------------------------- #
print("\n[8] the artifact is strict JSON and names no individual")
with tempfile.TemporaryDirectory() as td:
    pth = os.path.join(td, "a.json")
    with open(pth, "w") as fh:
        json.dump(json_safe({"ci": [float("nan"), 0.7]}), fh)
    raw_text = open(pth).read()
parsed = json.loads(raw_text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
check("a NaN serialises as null and parses strictly", "NaN" not in raw_text and parsed["ci"][0] is None)


def _refuses(payload) -> bool:
    try:
        prereg.assert_no_patient_level(payload)
        return False
    except ValueError:
        return True


check("a GSM accession anywhere in a value is refused", _refuses({"notes": ["see GSM983403"]}))
check("a namespaced patient id is refused", _refuses({"worst": "GSE40012:23"}))
check("a per-patient key is refused", _refuses({"data": {"patient_ids": []}}))
check("cohort aggregates pass",
      not _refuses({"data": {"n_patients": 145,
                             "per_cohort": {"GSE40012": {"n_samples": 154, "n_patients": 40}}},
                    "primary": {"per_direction_auroc_by_held_out_cohort": {"GSE40012": 0.8}}}))

print("\n" + "=" * 72)
print("HOST-RESPONSE VERIFICATION: %d/%d checks passed" % (sum(results), len(results)))
print("=" * 72)
raise SystemExit(0 if all(results) else 1)
