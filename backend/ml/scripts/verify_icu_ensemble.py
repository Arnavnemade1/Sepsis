"""Adversarial checks for the research ICU architecture; no clinical cache required.

    .venv/bin/python -m backend.ml.scripts.verify_icu_ensemble
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.ml.genesis.features import build
from backend.ml.genesis.harness import Cohort, patient_folds
from backend.ml.genesis.rawstore import DYNAMIC, STATIC, RawStore
from backend.ml.models.icu_ensemble import ICUEnsembleDetector, blend, trajectory_block
from backend.ml.scripts.improve_icu import split_roles, weight_grid, choose_blend


def fixture(patients=12, hours=32):
    rng = np.random.default_rng(123)
    X = rng.normal(10, 2, (patients * hours, 40)).astype(np.float32)
    X[rng.random(X.shape) < 0.4] = np.nan
    X[:, (DYNAMIC + STATIC).index("FiO2")] = 0.3
    X[:, (DYNAMIC + STATIC).index("ICULOS")] = np.tile(np.arange(hours)+1, patients)
    y = np.zeros((patients, hours), dtype=np.int8)
    y[::3, hours//2:] = 1
    return RawStore(X, y.ravel(), np.full(patients, hours),
                    np.array([f"p{i}" for i in range(patients)]),
                    np.arange(patients) % 2, DYNAMIC + STATIC)


class CausalDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.raw = fixture()
        self.F, self.names = trajectory_block(self.raw)

    def test_future_rewrite_and_truncation(self):
        changed = copy.deepcopy(self.raw)
        keep = self.raw.hour <= 12
        changed.X[~keep] = 10000
        after, _ = trajectory_block(changed)
        np.testing.assert_array_equal(self.F[keep], after[keep])
        changed.X = changed.X[keep]
        changed.y = changed.y[keep]
        changed.lens[:] = 13
        short, _ = trajectory_block(changed)
        np.testing.assert_array_equal(self.F[keep], short)
        # Positive control: a future-reading feature must fail this very assertion.
        leaked = self.F.copy()
        leaked[keep, 0] = np.nanmean(self.raw.X[~keep, 0])
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(leaked[keep], short)

    def test_patient_isolation_and_reordering(self):
        for patient in (0, 4, 11):
            isolated, _ = trajectory_block(self.raw.subset(np.array([patient])))
            np.testing.assert_array_equal(isolated, self.F[self.raw.seg == patient])
        reordered, _ = trajectory_block(self.raw.subset(np.array([11, 0, 4])))
        np.testing.assert_array_equal(reordered, np.vstack([self.F[self.raw.seg == p] for p in (11, 0, 4)]))

    def test_label_and_site_independence(self):
        changed = copy.deepcopy(self.raw)
        changed.y = 1-changed.y
        changed.site = 1-changed.site
        result, names = trajectory_block(changed)
        np.testing.assert_array_equal(result, self.F)
        self.assertEqual(names, self.names)

    def test_observed_history_math(self):
        raw = fixture(1, 8)
        raw.X[:, 0] = [1, np.nan, 5, np.nan, 9, np.nan, 13, 100]
        F, names = trajectory_block(raw)
        # At hour 6, preceding actual observations are (0,1), (2,5), (4,9).
        self.assertAlmostEqual(F[6, names.index("HR__obs_slope6")], 2)
        self.assertAlmostEqual(F[6, names.index("HR__obs_std6")], np.std([1, 5, 9]), places=6)
        self.assertAlmostEqual(F[6, names.index("HR__innovation6")],
                               (13-5)/(np.std([1, 5, 9])+0.15), places=6)
        # Current outlier cannot change its own reference slope.
        self.assertAlmostEqual(F[7, names.index("HR__obs_slope6")], 2)

    def test_missing_and_patient_start(self):
        raw = fixture()
        raw.X[:, 0] = np.nan
        F, names = trajectory_block(raw)
        self.assertTrue(np.isnan(F[:, [i for i, n in enumerate(names) if n.startswith("HR__")]]).all())
        self.assertTrue(np.isnan(F[raw.starts]).all())

    def test_serving_feature_parity_and_empty_record(self):
        base, names = build(self.raw)
        detector = ICUEnsembleDetector([{"kind": "lgbm", "columns": len(names)}], [1], names+self.names)
        expected = np.hstack((base, self.F))
        for patient in (0, 7):
            mask = self.raw.seg == patient
            frame = pd.DataFrame(self.raw.X[mask], columns=self.raw.columns)
            # An outcome column in the frame cannot become an input.
            frame["SepsisLabel"] = self.raw.y[mask]
            np.testing.assert_array_equal(detector.record_features(frame), expected[mask])
        empty = detector.record_features(pd.DataFrame())
        self.assertEqual(empty.shape, (0, len(detector.feature_names)))
        self.assertEqual(len(detector.predict_matrix(empty)), 0)


class BlendAndPartitionTests(unittest.TestCase):
    def test_invalid_weights_and_probabilities_fail_closed(self):
        for w in ([1, -0.1], [0.2, 0.2], [np.nan, 1], [1]):
            with self.assertRaises(ValueError):
                blend(np.zeros((5, 2)), np.array(w))
        for invalid in (np.nan, np.inf, -0.1, 1.1):
            with self.assertRaises(ValueError):
                blend(np.full((3, 1), invalid), np.ones(1))

    def test_weight_grid_contains_baseline_and_all_vertices(self):
        grid = weight_grid(4)
        self.assertEqual(len(grid), 35)
        for v in np.eye(4):
            self.assertTrue(any(np.array_equal(v, w) for w in grid))
        self.assertTrue(all((w >= 0).all() and w.sum() == 1 for w in grid))

    def test_blend_selection_can_reject_a_bad_member(self):
        y = np.tile([0, 1], 100)
        p = np.column_stack([0.1+0.8*y, 0.9-0.8*y])
        w, _ = choose_blend(y, np.zeros(len(y), dtype=int), p)
        np.testing.assert_array_equal(w, [1, 0])

    def test_roles_exclude_lock_and_each_other(self):
        raw = fixture(300, 8)
        c = Cohort(raw, np.zeros((raw.n_rows, 1)), ["dummy"], patient_folds(raw, 5, 20260923))
        roles = split_roles(c)
        sets = [set(c.seg[r]) for r in roles.values()]
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
        self.assertFalse(set.union(*sets) & set(np.flatnonzero(c.folds < 0)))
        self.assertEqual(set.union(*sets), set(np.flatnonzero(c.folds >= 0)))
        original_lock = c.folds < 0
        other = patient_folds(raw, 4, 123)
        np.testing.assert_array_equal(original_lock, other < 0)

    def test_calibration_must_preserve_ranking(self):
        with self.assertRaises(ValueError):
            ICUEnsembleDetector([{"kind": "lgbm", "columns": 1}], [1], ["x"], (-1, 0))

    def test_native_serialization_and_batch_invariance(self):
        import lightgbm as lgb
        import xgboost as xgb
        rng = np.random.default_rng(4)
        X = rng.normal(size=(100, 4)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        lb = lgb.train({"objective": "binary", "num_threads": 1, "verbosity": -1,
                        "min_data_in_leaf": 5}, lgb.Dataset(X, label=y), num_boost_round=8)
        xb = xgb.train({"objective": "binary:logistic", "nthread": 1, "max_depth": 2},
                       xgb.DMatrix(X, label=y), num_boost_round=8)
        members = [{"kind": "lgbm", "columns": 4, "model": lb.model_to_string(), "trees": 8, "calib": (1, 0)},
                   {"kind": "xgb", "columns": 4, "model": bytes(xb.save_raw(raw_format="ubj")), "trees": 8, "calib": (1, 0)}]
        detector = ICUEnsembleDetector(members, [0.5, 0.5], [f"x{i}" for i in range(4)])
        p = detector.predict_matrix(X)
        expected = (lb.predict(X, num_threads=1)+xb.predict(xgb.DMatrix(X)))/2
        np.testing.assert_allclose(p, expected, atol=1e-7)
        np.testing.assert_allclose(detector.predict_matrix(X, batch=7), p, atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.joblib"
            joblib.dump(detector, path)
            restored = joblib.load(path)
            self.assertFalse(restored._models)
            np.testing.assert_array_equal(restored.predict_matrix(X), p)
        with self.assertRaises(ValueError):
            detector.predict_matrix(X[:, :3])
        with self.assertRaises(ValueError):
            detector.predict_matrix(X, batch=0)
        # A zero-weight member's invalid payload must not be loaded.
        detector.weights = [1, 0]
        detector._models.clear()
        detector.members[1]["model"] = b"invalid"
        np.testing.assert_allclose(detector.predict_matrix(X), lb.predict(X, num_threads=1), atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
