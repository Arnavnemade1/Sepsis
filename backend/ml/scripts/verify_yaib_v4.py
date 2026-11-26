"""Synthetic checks for what Forge's YAIB lane v3 asks of the bench; no credentialed data.

    screening   --train-fraction fits on a stratified, deterministic share of training stays
    targets     --target trains on early6 / stay labels; validation stays YAIB's label
    families    causal blocks beside the feature set's matrix, built once, never merged
    validation  --save-validation writes each run's validation predictions
    gain        each column group's share of split gain
    ensembles   yaib_ensemble select / combine against a by-hand reference

python -m backend.ml.scripts.verify_yaib_v4
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.ml.scripts import yaib_ensemble as E
from backend.ml.scripts import yaib_final as F
from backend.ml.scripts import yaib_sweep as S
from backend.ml.scripts.verify_yaib import synthetic, write
from backend.ml.yaib import bench as B
from backend.ml.yaib import checkpoint as K
from backend.ml.yaib import cohort as C
from backend.ml.yaib import families as FAM
from backend.ml.yaib import features as FE

PAIRS = [(0, 0), (0, 1)]
CFG = B.TrainConfig(max_rounds=30, early_stopping=10, feature_fraction=0.3, bagging_fraction=0.7,
                    min_child_samples=20, num_leaves=7, learning_rate=0.1)


def _cohort(n=120, seed=7):
    return C.load(write(tempfile.mkdtemp(), *synthetic(n=n, seed=seed)))


class ScreeningChecks(unittest.TestCase):
    def test_a_fraction_keeps_a_stratified_deterministic_share_of_training_stays(self):
        labels = pd.Series([1.0] * 20 + [0.0] * 80, index=np.arange(100) * 3)
        train = labels.index.to_numpy()
        a = B.train_stays(train, labels, 0.25, "split-x")
        self.assertEqual(len(a), 25)
        self.assertEqual(int(labels.reindex(a).sum()), 5)                  # 25% of each class
        np.testing.assert_array_equal(a, B.train_stays(train[::-1], labels, 0.25, "split-x"))
        self.assertFalse(np.array_equal(a, B.train_stays(train, labels, 0.25, "split-y")))
        np.testing.assert_array_equal(B.train_stays(train, labels, 1.0, "s"), np.sort(train))
        for bad in (0.0, 1.5):
            with self.assertRaises(ValueError):
                B.train_stays(train, labels, bad, "s")

    def test_a_screened_run_fits_fewer_stays_on_the_same_validation_rows(self):
        c = _cohort()
        full = B.run(c, CFG, runs=PAIRS, memory_check=False, validation_only=True)
        c = _cohort()
        cut = B.run(c, replace(CFG, train_fraction=0.5), runs=PAIRS, memory_check=False, validation_only=True)
        for f, s in zip(full["runs"], cut["runs"]):
            self.assertEqual(f["split"], s["split"])
            self.assertLess(s["train_stays"], f["train_stays"])
            self.assertAlmostEqual(s["train_stays"] / f["train_stays"], 0.5, delta=0.05)

    def test_the_default_fraction_is_the_fit_as_before(self):
        c = _cohort()
        a = B.run(c, CFG, runs=PAIRS, memory_check=False, validation_only=True)
        c = _cohort()
        b = B.run(c, replace(CFG, train_fraction=1.0, target="yaib"), runs=PAIRS, memory_check=False,
                  validation_only=True)
        self.assertEqual([r["val_auroc"] for r in a["runs"]], [r["val_auroc"] for r in b["runs"]])


class TargetChecks(unittest.TestCase):
    def test_targets_from_yaibs_label(self):
        stay = np.array([1] * 10 + [2] * 5)
        hour = np.r_[np.arange(10), np.arange(5)].astype(float)
        y = np.r_[[0] * 7 + [1] * 3, [0, np.nan, 0, 0, 0]].astype(float)
        np.testing.assert_array_equal(B.training_target(y, stay, hour, "yaib"), y)
        early = B.training_target(y, stay, hour, "early6")
        np.testing.assert_array_equal(early[:10], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1])   # first positive hour 7
        np.testing.assert_array_equal(early[10:], [0, np.nan, 0, 0, 0])
        st = B.training_target(y, stay, hour, "stay")
        np.testing.assert_array_equal(st, np.r_[[1] * 10, [0, np.nan, 0, 0, 0]])
        with self.assertRaises(ValueError):
            B.training_target(y, stay, hour, "future")

    def test_a_target_changes_training_only(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(3000, 6)).astype(np.float32)
        y = (rng.random(3000) < 1 / (1 + np.exp(-(X[:, 0] * 2 - 2)))).astype(float)
        idx = rng.permutation(3000)
        tr, va, te = idx[:1800], idx[1800:2400], idx[2400:]
        other = (X[:, 1] > 0.8).astype(float)                    # a training label driven by x1, not x0
        a = B._fit_one(X, y, tr, va, te, CFG, validation_predictions=True)
        b = B._fit_one(X, y, tr, va, te, CFG, validation_predictions=True, y_train=other)
        # each booster's risk follows the column its TRAINING label follows ...
        self.assertGreater(np.corrcoef(a["p"], X[te, 0])[0, 1], 0.5)
        self.assertGreater(np.corrcoef(b["p"], X[te, 1])[0, 1], 0.5)
        # ... while both are stopped on, and scored with, YAIB's label on the validation rows
        self.assertGreater(a["val_auroc"], b["val_auroc"] + 0.1)
        self.assertAlmostEqual(b["val_auroc"], roc_auc_score(y[va], b["pv"]), places=6)


class FamilyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = _cohort(n=80)

    def test_every_family_is_causal_and_label_free(self):
        c = self.c
        sid = c.dyn.stay_id.to_numpy()
        rows = np.flatnonzero(sid == sid[0])
        nxt = np.flatnonzero(sid == np.unique(sid)[1])
        k = rows[len(rows) // 2]
        d2 = c.dyn.copy()
        for v in ("hr", "map", "sbp", "lact", "crea", "temp", "urine", "o2sat"):
            d2.loc[k, v] = 400.0
        for f in FAM.FAMILIES:
            with self.subTest(family=f):
                X, names = FAM.build_family(c, f)
                self.assertEqual(names, FAM.family_names(f))
                X2, _ = FAM.build_family(C.Cohort(d2, c.sta, c.label), f)
                earlier = rows[rows < k]
                self.assertTrue(np.array_equal(X[earlier], X2[earlier], equal_nan=True), "an earlier hour moved")
                self.assertFalse(np.array_equal(X[k], X2[k], equal_nan=True), "the check cannot fail")
                self.assertTrue(np.array_equal(X[nxt], X2[nxt], equal_nan=True), "the next stay moved")
                X3, _ = FAM.build_family(C.Cohort(c.dyn, c.sta, 1 - np.nan_to_num(c.label)), f)
                self.assertTrue(np.array_equal(X, X3, equal_nan=True), "a label moved a feature")

    def test_hand_checked_values_and_a_stay_after_huge_values(self):
        dyn = pd.DataFrame({"stay_id": [1, 1, 1, 1, 2, 2, 3, 3, 3], "hour": [0.0, 1, 2, 3, 0, 1, 0, 1, 2]})
        for v in C.DYNAMIC:
            dyn[v] = np.nan
        dyn["hr"] = [80, np.nan, 120, 90, 3e7, 3.1e7, 80, 80.5, 81]
        dyn["temp"] = [37.0, 37, 39, 38, np.nan, np.nan, 36, 36, 36]
        dyn["crea"] = [1.0, np.nan, 1.6, 1.4, np.nan, np.nan, np.nan, np.nan, np.nan]
        dyn["urine"] = [100.0, 50, np.nan, 25, np.nan, np.nan, np.nan, np.nan, np.nan]
        sta = pd.DataFrame({"stay_id": [1, 2, 3], "age": [60.0, 70, 50], "sex": [1.0, 0, 1],
                            "height": [170.0, 160, 180], "weight": [50.0, 60, 80]})
        c = C.Cohort(dyn, sta, np.zeros(9))
        Xb, nb = FAM.build_family(c, "baseline")
        col = lambda X, n, names: X[:, names.index(n)]
        np.testing.assert_allclose(col(Xb, "hr__mean48", nb)[:4], [80, 80, 100, 290 / 3])
        np.testing.assert_allclose(col(Xb, "hr__max48", nb)[:4], [80, 80, 120, 120])
        z = col(Xb, "hr__z_own", nb)
        m, sd = 290 / 3, np.std([80, 120, 90])
        self.assertTrue(np.isnan(z[:2]).all())                             # under two measurements
        np.testing.assert_allclose(z[3], (90 - m) / sd, rtol=1e-6)
        np.testing.assert_allclose(col(Xb, "hr__mean48", nb)[6:9], [80, 80.25, 80.5], rtol=1e-6)
        Xe, ne = FAM.build_family(c, "ewm")
        a = 1 - 0.5 ** (1 / 3)
        e3 = [80.0, 80.0]
        e3.append(a * 120 + (1 - a) * e3[-1])
        e3.append(a * 90 + (1 - a) * e3[-1])
        np.testing.assert_allclose(col(Xe, "hr__ewm3", ne)[:4], e3, rtol=1e-6)
        np.testing.assert_allclose(col(Xe, "hr__ewm3", ne)[6], 80.0)          # restarts at the stay
        Xr, nr = FAM.build_family(c, "ratios")
        np.testing.assert_allclose(col(Xr, "crea_rise48", nr)[:4], [0, 0, 0.6, 0.4], rtol=1e-6)
        np.testing.assert_allclose(col(Xr, "urine6_per_kg", nr)[:4], [2, 3, 3, 3.5], rtol=1e-6)
        np.testing.assert_allclose(col(Xr, "hr_temp_corrected", nr)[:4], [80, 80, 100, 80], rtol=1e-6)

    def test_a_memmapped_block_is_the_in_memory_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            for f in FAM.FAMILIES:
                X, _ = FAM.build_family(self.c, f)
                Xm, _ = FAM.build_family(self.c, f, out=str(Path(tmp) / f"{f}.npy"))
                self.assertEqual(X.tobytes(), np.asarray(Xm).tobytes())
                self.assertTrue(np.load(Path(tmp) / f"{f}.npy", mmap_mode="r").flags.c_contiguous)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), sorted(f"{f}.npy" for f in FAM.FAMILIES))

    def test_columns_read_blocks_side_by_side(self):
        a = np.arange(12, dtype=np.float32).reshape(4, 3)
        b = np.arange(8, dtype=np.float32).reshape(4, 2) + 100
        cols = FAM.Columns([a, b])
        self.assertEqual(cols.shape, (4, 5))
        np.testing.assert_array_equal(cols[np.array([2, 0])], np.hstack([a, b])[[2, 0]])
        np.testing.assert_array_equal(cols[1], np.hstack([a, b])[1])
        np.testing.assert_array_equal(cols[1:3], np.hstack([a, b])[1:3])
        with self.assertRaises(TypeError):
            cols[:, :2]
        with self.assertRaises(ValueError):
            FAM.Columns([a, b[:3]])

    def test_families_are_canonical(self):
        self.assertEqual(FAM.canonical(["ratios", "baseline"]), ["baseline", "ratios"])
        for bad in (["baseline", "baseline"], ["everything"]):
            with self.assertRaises(ValueError):
                FAM.canonical(bad)


class BenchFamilyChecks(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.input = write(str(self.root / "input"), *synthetic(n=120))

    def bench(self, name, store="store", **kw):
        return B.run(C.load(self.input), CFG, runs=PAIRS, resume_dir=str(self.root / name),
                     feature_store=str(self.root / store), memory_check=False, validation_only=True, **kw)

    def test_a_family_run_equals_a_fit_on_the_hstacked_matrix(self):
        res = self.bench("fam", feature_set="trend", families=["baseline", "ratios"])
        self.assertEqual(res["model"]["n_features"],
                         len(FE.feature_names("trend")) + len(FAM.family_names("baseline")) + len(FAM.RATIOS))
        c = C.load(self.input)
        X = np.hstack([FE.build(c, feature_set="trend")[0], FAM.build_family(c, "baseline")[0],
                       FAM.build_family(c, "ratios")[0]])
        res_dense = B.run(C.load(self.input), CFG, runs=PAIRS, memory_check=False, validation_only=True,
                          feature_set="trend", families=["baseline", "ratios"])
        self.assertEqual([r["val_auroc"] for r in res["runs"]], [r["val_auroc"] for r in res_dense["runs"]])
        self.assertEqual(X.shape[1], res["model"]["n_features"])

    def test_blocks_are_built_once_and_the_trend_matrix_is_untouched(self):
        self.bench("one", feature_set="trend", families=["baseline"])
        (trend,) = [p for p in (self.root / "store").glob("*/feature_spec.json")
                    if json.loads(p.read_text())["feature_set"] == "trend"]
        before = trend.parent.joinpath("features.json").read_text()
        with patch.object(FAM, "build_family", side_effect=AssertionError("block rebuilt")), \
                patch.object(FE, "build", side_effect=AssertionError("matrix rebuilt")):
            res = self.bench("two", feature_set="rich", matrix_from="trend", families=["baseline"])
        self.assertEqual(len(res["runs"]), 2)
        self.assertEqual(trend.parent.joinpath("features.json").read_text(), before)
        specs = [json.loads(p.read_text()) for p in (self.root / "store").glob("*/feature_spec.json")]
        self.assertEqual(sorted(s["feature_set"] for s in specs), ["family:baseline", "trend"])

    def test_the_feature_set_identity_does_not_read_the_family_code(self):
        c = C.load(self.input)
        spec = K.experiment_spec(c, {}, {}, "trend")
        self.assertIn("families.py", spec["implementation"])
        self.assertNotIn("families.py", K.feature_spec(spec)["implementation"])
        self.assertNotIn("families.py", K.FEATURE_SOURCES)

    def test_validation_predictions_and_gain_shares(self):
        vdir = self.root / "val"
        res = self.bench("v", feature_set="trend", families=["ewm"], validation_dir=str(vdir))
        for r in res["runs"]:
            frame = pd.read_parquet(vdir / f"val_r{r['repetition']}_f{r['fold']}.parquet")
            self.assertEqual(list(frame.columns), ["stay_id", "hour", "label", "p"])
            self.assertAlmostEqual(roc_auc_score(frame.label, frame.p), r["val_auroc"], places=6)
            shares = r["gain_share"]
            self.assertEqual(set(shares), {"base", "rich", "trend", "ewm"})
            self.assertAlmostEqual(sum(shares.values()), 1.0, places=3)
        # a resumed run that lacks them refits deterministically, checks the checkpoint, and writes them
        for p in vdir.iterdir():
            p.unlink()
        again = self.bench("v", feature_set="trend", families=["ewm"], validation_dir=str(vdir))
        self.assertEqual([r["val_auroc"] for r in res["runs"]], [r["val_auroc"] for r in again["runs"]])
        self.assertEqual(len(list(vdir.glob("val_r*_f*.parquet"))), 2)

    def test_a_bag_saves_its_averaged_validation_predictions(self):
        vdir = self.root / "bag"
        B.run(C.load(self.input), replace(CFG, seeds=2), runs=PAIRS, memory_check=False, validation_only=True,
              feature_set="rich", validation_dir=str(vdir))
        frame = pd.read_parquet(vdir / "val_r0_f0.parquet")
        self.assertTrue(((frame.p >= 0) & (frame.p <= 1)).all())



def _val_dir(root, name, runs, y, scores):
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    for r, sc in zip(runs, scores):
        rep, fold = r.split(":")
        n = len(y)
        pd.DataFrame({"stay_id": np.arange(n) // 5, "hour": np.arange(n) % 5, "label": y, "p": sc}).to_parquet(
            d / f"val_r{rep}_f{fold}.parquet", index=False)
    return str(d)


class EnsembleChecks(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        rng = np.random.default_rng(5)
        n = 4000
        self.y = (rng.random(n) < 0.2).astype(float)
        # two complementary noisy members and a useless one
        self.a = [E.sigmoid(self.y * 1.0 + rng.normal(0, 1, n)) for _ in range(2)]
        self.b = [E.sigmoid(self.y * 1.0 + rng.normal(0, 1, n)) for _ in range(2)]
        self.c = [E.sigmoid(rng.normal(0, 1, n)) for _ in range(2)]
        self.runs = ["0:0", "0:1"]
        self.members = {n_: _val_dir(self.root, n_, self.runs, self.y, s_)
                        for n_, s_ in (("a", self.a), ("b", self.b), ("c", self.c))}

    def test_caruana_matches_a_by_hand_greedy_and_caps_members(self):
        L = np.column_stack([E.logit(self.a[0]), E.logit(self.b[0]), E.logit(self.c[0])])
        w = E.caruana(L, self.y, max_members=3, steps=6)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertEqual(w[2], 0.0)                                     # the useless member is left out
        self.assertGreater(roc_auc_score(self.y, L @ w), max(roc_auc_score(self.y, L[:, j]) for j in range(3)))
        # by hand: two steps = the best single, then the best pair
        single = [roc_auc_score(self.y, L[:, j]) for j in range(3)]
        first = int(np.argmax(single))
        pair = max(range(3), key=lambda j: roc_auc_score(self.y, (L[:, first] + L[:, j]) / 2))
        expect = np.zeros(3)
        expect[first] += 0.5
        expect[pair] += 0.5
        if roc_auc_score(self.y, (L[:, first] + L[:, pair]) / 2) <= single[first]:
            expect = np.eye(3)[first]
        np.testing.assert_allclose(E.caruana(L, self.y, steps=2), expect)
        self.assertEqual((E.caruana(L, self.y, max_members=1, steps=6) > 0).sum(), 1)

    def test_select_cross_fits_each_runs_weights_on_the_other_run(self):
        sel = E.select(self.members, self.runs)
        self.assertEqual(set(sel["cross_fitted"]), set(self.runs))
        self.assertNotIn("c", sel["weights"])
        best_single = {r: max(sel["singles"][r].values()) for r in self.runs}
        self.assertTrue(all(sel["cross_fitted"][r] > best_single[r] for r in self.runs))
        # spoiling run 0:1 changes the weights run 0:0 is scored with, never run 0:1's own
        _val_dir(self.root, "a", self.runs, self.y, [self.a[0], E.sigmoid(np.zeros(len(self.y)))])
        spoiled = E.select(self.members, self.runs)
        self.assertEqual(spoiled["weights_by_run"]["0:1"], sel["weights_by_run"]["0:1"])
        self.assertNotEqual(spoiled["weights_by_run"]["0:0"], sel["weights_by_run"]["0:0"])
        with self.assertRaisesRegex(ValueError, "two development runs"):
            E.select(self.members, ["0:0"])

    def test_misaligned_members_are_refused(self):
        bad = _val_dir(self.root, "shuffled", self.runs, self.y[::-1].copy(), self.a)
        with self.assertRaisesRegex(ValueError, "same rows"):
            E.select({"a": self.members["a"], "s": bad}, self.runs)

    def test_combine_is_the_weighted_mean_logit_and_checks_runs(self):
        out = self.root / "combined"
        for n_, s_ in (("a", self.a), ("b", self.b)):
            d = self.root / f"t-{n_}"
            d.mkdir()
            for r, sc in zip(self.runs, s_):
                rep, fold = r.split(":")
                pd.DataFrame({"stay_id": np.arange(len(sc)), "hour": 0.0, "label": self.y, "p": sc}).to_parquet(
                    d / f"ours_r{rep}_f{fold}.parquet", index=False)
        members = {"a": str(self.root / "t-a"), "b": str(self.root / "t-b")}
        files = E.combine(members, {"a": 0.75, "b": 0.25}, str(out))
        self.assertEqual(len(files), 2)
        got = pd.read_parquet(out / "ours_r0_f1.parquet").p.to_numpy()
        np.testing.assert_allclose(got, E.sigmoid(0.75 * E.logit(self.a[1]) + 0.25 * E.logit(self.b[1])))
        (self.root / "t-b" / "ours_r0_f1.parquet").unlink()
        with self.assertRaisesRegex(ValueError, "different runs"):
            E.combine(members, {"a": 0.75, "b": 0.25}, str(out))
        for bad in ({"a": 0.5}, {"a": 0.5, "z": 0.5}, {"a": 1.2, "b": -0.2}):
            with self.assertRaises(ValueError):
                E.combine(members, bad, str(out))

    def test_profile_gives_band_aurocs_and_correlations_only(self):
        prof = E.profile(self.members, self.runs)
        self.assertEqual(set(prof["by_hour"]), {"a", "b", "c"})
        self.assertIn("a|b", prof["correlation"])
        self.assertNotIn("p", json.dumps(prof["by_hour"]))


class SweepAndFinalChecks(unittest.TestCase):
    def plan(self, name, settings, feature_set="rich", **extra):
        return {"feature_set": feature_set, "seed": 1234, "reference": {"name": "-"},
                "configs": {name: settings}, **extra}

    def test_new_settings_reach_the_bench_call(self):
        cmd = S.command("/c", self.plan("x", {"families": ["baseline", "ratios"], "target": "early6",
                                             "train_fraction": 0.3}), "x", "0:0", "/w", "/s", "/o", 2,
                        validation="/v")
        self.assertEqual(cmd[cmd.index("--families") + 1], "baseline,ratios")
        self.assertEqual(cmd[cmd.index("--target") + 1], "early6")
        self.assertEqual(cmd[cmd.index("--train-fraction") + 1], "0.3")
        self.assertEqual(cmd[cmd.index("--save-validation") + 1], "/v")
        empty = S.command("/c", self.plan("x", {"families": []}), "x", "0:0", "/w", "/s", "/o", 2)
        self.assertNotIn("--families", empty)

    def test_an_ensemble_final_scores_combined_predictions_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cohort = write(str(tmp / "cohort"), *synthetic(n=120))
            small = {"max_rounds": 20, "early_stopping": 5, "num_leaves": 7, "min_child_samples": 20,
                     "learning_rate": 0.1}
            plans = {"m_rich": self.plan("m_rich", small),
                     "m_trend": self.plan("m_trend", {**small, "families": ["ratios"]}, feature_set="trend")}
            selection = {"chosen": "ensemble", "note": "chosen on validation AUROC only",
                         "plan": {"stage2_runs": "0:0,0:1", "min_gain": 0.001,
                                  "reference": {"name": "alloy", "val_auroc": {"0:0": 0.5, "0:1": 0.5}}},
                         "ensemble": {"weights": {"m_rich": 0.5, "m_trend": 0.5}, "plans": plans,
                                      "cross_fitted": {"0:0": 0.9, "0:1": 0.9}}}
            work, store, pred = tmp / "work", tmp / "store", tmp / "pred"
            store.mkdir(mode=0o700)
            jobs = F.build_ensemble(selection, cohort, str(work), str(store), str(pred), str(tmp / "final.json"))
            self.assertEqual([j["name"] for j in jobs], ["m_rich", "m_trend"])
            self.assertIn("--families", jobs[1]["cmd"])
            self.assertNotIn("--runs", jobs[0]["cmd"])
            res = F.run_ensemble(selection, jobs, str(pred), str(tmp / "final.json"))
            self.assertEqual(res["yaib_metric"]["n_runs"], 25)
            self.assertEqual(len(list(pred.glob("ours_r*_f*.parquet"))), 25)
            frame = pd.read_parquet(pred / "ours_r2_f3.parquet")
            ra = pd.read_parquet(pred / "members" / "m_rich" / "ours_r2_f3.parquet")
            tb = pd.read_parquet(pred / "members" / "m_trend" / "ours_r2_f3.parquet")
            np.testing.assert_allclose(frame.p, E.sigmoid(0.5 * E.logit(ra.p) + 0.5 * E.logit(tb.p)))
            self.assertAlmostEqual(res["runs"][0]["auroc"], roc_auc_score(
                pd.read_parquet(pred / "ours_r0_f0.parquet").label, pd.read_parquet(pred / "ours_r0_f0.parquet").p))
        for broken in ({"note": "chosen on test"}, {"ensemble": {**selection["ensemble"], "weights": {"zz": 1.0}}},
                       {"ensemble": {**selection["ensemble"], "cross_fitted": {"0:0": 0.5, "0:1": 0.5}}}):
            with self.assertRaises(ValueError):
                F.build_ensemble({**selection, **broken}, "/c", "/w", "/s", "/p", "/o.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
