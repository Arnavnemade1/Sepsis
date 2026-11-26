"""Synthetic checks for seed bagging, the model seed and reading one feature set from a
wider set's matrix; no credentialed data.

python -m backend.ml.scripts.verify_yaib_seeds
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.ml.scripts import yaib_sweep as S
from backend.ml.scripts.verify_yaib import synthetic, write
from backend.ml.yaib import bench as B
from backend.ml.yaib import cohort as C
from backend.ml.yaib import features as FE

PAIRS = [(0, 0), (1, 1)]
# subsampled rows and columns, as Alloy is, so a seed changes the booster
CFG = B.TrainConfig(max_rounds=30, early_stopping=10, feature_fraction=0.3, bagging_fraction=0.7,
                    min_child_samples=20, num_leaves=7, learning_rate=0.1)


class MatrixFromChecks(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.input = write(str(self.root / "input"), *synthetic(n=120))

    def bench(self, name, store, **kw):
        return B.run(C.load(self.input), CFG, runs=PAIRS, resume_dir=str(self.root / name),
                     feature_store=str(self.root / store), predictions_dir=str(self.root / f"{name}-p"),
                     memory_check=False, **kw)

    def test_rich_read_from_the_trend_matrix_is_the_rich_model_exactly(self):
        direct = self.bench("direct", "store-rich", feature_set="rich")
        viewed = self.bench("viewed", "store-trend", feature_set="rich", matrix_from="trend")
        self.assertEqual(direct["runs"], viewed["runs"])          # same rounds, same validation AUROC
        self.assertEqual(direct["yaib_metric"], viewed["yaib_metric"])
        self.assertEqual(viewed["model"]["n_features"], len(FE.feature_names("rich")))
        self.assertEqual(viewed["model"]["matrix_from"], "trend")
        for rep, fold in PAIRS:
            name = f"ours_r{rep}_f{fold}.parquet"
            pd.testing.assert_frame_equal(pd.read_parquet(self.root / "direct-p" / name),
                                          pd.read_parquet(self.root / "viewed-p" / name))
        (spec,) = (self.root / "store-trend").glob("*/feature_spec.json")
        self.assertEqual(json.loads(spec.read_text())["feature_set"], "trend")
        (matrix,) = (self.root / "store-trend").glob("*/features.npy")
        self.assertEqual(np.load(matrix, mmap_mode="r").shape[1], len(FE.feature_names("trend")))

    def test_a_trend_run_and_a_rich_run_share_one_matrix(self):
        self.bench("trend", "store", feature_set="trend", validation_only=True)
        with patch.object(FE, "build", side_effect=AssertionError("matrix rebuilt")):
            res = self.bench("rich", "store", feature_set="rich", matrix_from="trend", validation_only=True)
        self.assertEqual(len(res["runs"]), 2)
        self.assertEqual(len(list((self.root / "store").glob("*/features.npy"))), 1)

    def test_only_a_leading_prefix_can_be_read_from_another_matrix(self):
        for feature_set, wider in (("trend", "rich"), ("rich", "rich")):
            with self.subTest(feature_set=feature_set, wider=wider), \
                    self.assertRaisesRegex(ValueError, "leading columns"):
                self.bench("x", "store", feature_set=feature_set, matrix_from=wider)

    def test_reading_from_another_matrix_is_another_experiment(self):
        self.bench("same", "store", feature_set="rich", validation_only=True)
        with self.assertRaisesRegex(ValueError, "Experiment changed"):
            self.bench("same", "store", feature_set="rich", matrix_from="trend", validation_only=True)


class SeedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        n = 6000
        cls.X = rng.normal(size=(n, 12)).astype(np.float32)
        logit = cls.X[:, 0] + 0.5 * cls.X[:, 1] * cls.X[:, 2] - 2.5
        cls.y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
        idx = rng.permutation(n)
        cls.tr, cls.va, cls.te = idx[:3600], idx[3600:4800], idx[4800:]

    def fit(self, **kw):
        from dataclasses import replace
        return B._fit_predict(self.X, self.y, self.tr, self.va, self.te, replace(CFG, **kw))

    def test_one_seed_is_the_single_booster_as_before(self):
        one = self.fit(seeds=1)
        single = B._fit_one(self.X, self.y, self.tr, self.va, self.te, CFG)
        np.testing.assert_array_equal(one["p"], single["p"])
        self.assertEqual((one["rounds"], one["val_auroc"]), (single["rounds"], single["val_auroc"]))
        self.assertNotIn("members", one)

    def test_the_model_seed_changes_a_subsampled_booster(self):
        a, b = self.fit(seed=0), self.fit(seed=1000)
        self.assertFalse(np.array_equal(a["p"], b["p"]))
        self.assertNotEqual(a["val_auroc"], b["val_auroc"])

    def test_a_bag_averages_its_members_and_scores_their_mean(self):
        bag = self.fit(seeds=3, seed=10)
        members = [B._fit_one(self.X, self.y, self.tr, self.va, self.te,
                              B.TrainConfig(**{**CFG.__dict__, "seed": 10 + i}), validation_predictions=True)
                   for i in range(3)]
        np.testing.assert_allclose(bag["p"], np.mean([m["p"] for m in members], axis=0), rtol=1e-12)
        pv = np.mean([m["pv"] for m in members], axis=0)
        self.assertAlmostEqual(bag["val_auroc"], roc_auc_score(self.y[self.va], pv), places=12)
        self.assertEqual(bag["members"]["val_auroc"], [m["val_auroc"] for m in members])
        self.assertEqual(bag["members"]["rounds"], [m["rounds"] for m in members])
        self.assertIs(type(bag["rounds"]), int)

    def test_a_member_validation_score_is_lightgbms_own(self):
        # what lets one seed's score be compared with a bag's: the same AUROC on the same rows
        m = B._fit_one(self.X, self.y, self.tr, self.va, self.te, CFG, validation_predictions=True)
        self.assertAlmostEqual(m["val_auroc"], roc_auc_score(self.y[self.va], m["pv"]), places=6)

    def test_zero_seeds_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            self.fit(seeds=0)


class SweepCommandChecks(unittest.TestCase):
    def plan(self, **extra):
        return {"feature_set": "rich", "seed": 1234, "reference": {"name": "-"},
                "configs": {"c": {"learning_rate": 0.03, "seeds": 3, "model_seed": 1000, "unbalanced": True}},
                **extra}

    def test_seeds_and_model_seed_reach_the_bench_call(self):
        cmd = S.command("/cohort", self.plan(), "c", "0:0,0:1", "/work", "/store", "/out.json", 2)
        self.assertEqual(cmd[cmd.index("--seeds") + 1], "3")
        self.assertEqual(cmd[cmd.index("--model-seed") + 1], "1000")
        self.assertNotIn("--matrix-from", cmd)

    def test_a_plan_can_read_its_feature_set_from_a_wider_matrix(self):
        cmd = S.command("/cohort", self.plan(matrix_from="trend"), "c", "0:0", "/w", "/s", "/o", 2)
        self.assertEqual(cmd[cmd.index("--matrix-from") + 1], "trend")
        self.assertEqual(cmd[cmd.index("--feature-set") + 1], "rich")

    def test_an_old_plan_makes_the_same_call_as_before(self):
        plan = self.plan()
        plan["configs"]["c"] = {"learning_rate": 0.03, "unbalanced": True}
        cmd = S.command("/cohort", plan, "c", "0:0", "/w", "/s", "/o", 2)
        self.assertFalse({"--seeds", "--model-seed", "--matrix-from"} & set(cmd))

    def test_the_bench_cli_accepts_every_flag_a_sweep_can_pass(self):
        flags = set(S.FLAGS.values()) | set(S.SWITCHES.values()) | {"--matrix-from"}
        import argparse
        seen = set()
        real = argparse.ArgumentParser.add_argument

        def spy(parser, *names, **kw):
            seen.update(n for n in names if n.startswith("--"))
            return real(parser, *names, **kw)

        with patch.object(argparse.ArgumentParser, "add_argument", spy), \
                patch.object(argparse.ArgumentParser, "parse_args", side_effect=SystemExit(0)), \
                self.assertRaises(SystemExit):
            B.main([])
        self.assertLessEqual(flags, seen)


if __name__ == "__main__":
    unittest.main(verbosity=2)
