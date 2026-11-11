"""Catch whole-matrix fallback, incorrect native parameters and unbounded batches.

    python -m backend.ml.scripts.verify_yaib_streaming

Only invented arrays are used. No patient data or AWS calls.
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np

from backend.ml.yaib import bench as B
from backend.ml.yaib import features as FE


class BoundedMatrix:
    def __init__(self, values, maximum):
        self.values, self.maximum, self.reads = values, maximum, []

    def __getitem__(self, rows):
        size = 1 if np.isscalar(rows) else len(rows)
        if size > self.maximum:
            raise AssertionError("whole training matrix read")
        self.reads.append(size)
        return self.values[rows]

    def __array__(self, *args, **kwargs):
        raise AssertionError("whole training matrix converted")


def fixture(size=1500, columns=12):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(size, columns)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=size) > 0).astype(float)
    X[rng.random(X.shape) < 0.03] = np.nan
    X[rng.random(X.shape) < 0.03] = 0
    tr_end, va_end = int(size * 0.6), int(size * 0.8)
    return X, y, np.arange(tr_end), np.arange(tr_end, va_end), np.arange(va_end, size)


class StreamingChecks(unittest.TestCase):
    def test_adapter_is_native_sequence_and_has_safe_sample_and_batch_dtypes(self):
        X, _, train, _, _ = fixture()
        sequence = B._row_sequence(X, train)
        self.assertIsInstance(sequence, lgb.Sequence)
        self.assertEqual(sequence[0].dtype, np.float64)
        self.assertEqual(sequence[:10].dtype, np.float32)
        np.testing.assert_equal(sequence[0], X[train[0]].astype(np.float64))
        with self.assertRaisesRegex(TypeError, "whole-matrix"):
            np.asarray(sequence)
        self.assertLessEqual(sequence.batch_size, FE.DATASET_BATCH_ROWS)

    def test_real_fit_uses_sequences_never_fallback_and_all_reads_are_bounded(self):
        X, y, tr, va, te = fixture()
        cfg = B.TrainConfig(max_rounds=20, early_stopping=5)
        expected = B._fit_predict(X, y, tr, va, te, cfg)
        bounded = BoundedMatrix(X, 37)
        original = lgb.Dataset.__init__
        calls = []

        def observe(dataset, data, *args, **kwargs):
            self.assertIsInstance(data, lgb.Sequence)
            original(dataset, data, *args, **kwargs)
            self.assertEqual(dataset.get_params()["num_threads"], cfg.threads)
            self.assertEqual(
                dataset.get_params()["min_child_samples"],
                min(cfg.min_child_samples, max(5, len(tr) // 50)),
            )
            calls.append(data)

        # This catches the old fake Sequence path even on tiny data where it fitted.
        with (
            patch.object(FE, "STREAM_ROWS", 37),
            patch.object(lgb.Dataset, "__init__", observe),
        ):
            actual = B._fit_predict(bounded, y, tr, va, te, cfg, stream=True)
        self.assertEqual(len(calls), 2)
        self.assertTrue(bounded.reads and max(bounded.reads) <= 37)
        np.testing.assert_array_equal(actual["p"], expected["p"])
        self.assertEqual(actual["rounds"], expected["rounds"])
        self.assertEqual(actual["val_auroc"], expected["val_auroc"])

    def test_native_bin_sampling_parity_above_default_sample_ceiling(self):
        X, y, _, _, _ = fixture(210100, 6)
        tr, va, te = (
            np.arange(200050),
            np.arange(200050, 205075),
            np.arange(205075, 210100),
        )
        cfg = B.TrainConfig(max_rounds=8, early_stopping=3)
        expected = B._fit_predict(X, y, tr, va, te, cfg)
        actual = B._fit_predict(X, y, tr, va, te, cfg, stream=True)
        np.testing.assert_array_equal(actual["p"], expected["p"])
        self.assertEqual(actual["rounds"], expected["rounds"])
        self.assertEqual(actual["val_auroc"], expected["val_auroc"])

    def test_native_datasets_are_freed_even_if_prediction_fails(self):
        X, y, tr, va, te = fixture()

        class Booster:
            best_iteration = 1

            def __init__(self):
                self.best_score = {"valid_0": {"auc": 0.8}}

            freed = False

            def predict(self, *a, **kw):
                raise RuntimeError("planted prediction failure")

            def free_dataset(self):
                self.freed = True

        booster = Booster()
        with (
            patch.object(lgb, "train", return_value=booster),
            self.assertRaisesRegex(RuntimeError, "planted"),
        ):
            B._fit_predict(X, y, tr, va, te, B.TrainConfig(), stream=True)
        self.assertTrue(booster.freed)

    def test_memory_plan_includes_sampling_and_respects_container_cap(self):
        with patch.object(FE, "_cgroup_limit", return_value=7 * 1024**3):
            cold = FE.memory_plan(
                6400000, available_bytes=4_000_000_000, feature_set="rich"
            )
            loaded = FE.memory_plan(
                6400000,
                available_bytes=4_000_000_000,
                feature_set="rich",
                cohort_loaded=True,
            )
        self.assertFalse(cold["fits"])
        self.assertTrue(loaded["fits"])
        self.assertLess(loaded["additional_heap_gb"], loaded["peak_heap_gb"])
        self.assertGreater(loaded["sampled_columns_gb"], 1)
        self.assertLessEqual(loaded["dataset_batch_rows"], 8192)
        with patch.object(FE, "_cgroup_limit", return_value=2_000_000_000):
            small = FE.memory_plan(
                6400000, available_bytes=100_000_000_000, cohort_loaded=True
            )
        self.assertFalse(small["fits"])

    def test_cgroup_v2_v1_and_unlimited_values(self):
        for v2, v1, expected in [
            ("7516192768", None, 7516192768),
            ("max", "1000000000", 1000000000),
            ("max", str(1 << 63), None),
        ]:

            def read(path, v2=v2, v1=v1):
                value = v2 if path.endswith("memory.max") else v1
                if value is None:
                    raise FileNotFoundError(path)
                return io.StringIO(value)

            with self.subTest(v2=v2, v1=v1), patch("builtins.open", side_effect=read):
                self.assertEqual(FE._cgroup_limit(), expected)

    def test_private_profile_emits_resources_but_never_scores_or_rows(self):
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        from backend.ml.scripts.verify_yaib import synthetic, write
        from backend.ml.yaib import cohort as C

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = write(str(root / "cohort"), *synthetic(n=100))
            features = root / "matrix.npy"
            matrix, _ = FE.build(C.load(directory), out=str(features))
            del matrix
            output = root / "diagnostics" / "resources.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "backend.ml.scripts.profile_yaib_fit",
                    "--cohort",
                    directory,
                    "--features",
                    str(features),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=60,
                env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            report = json.loads(result.stdout)
            self.assertEqual(report, json.loads(output.read_text()))
            self.assertEqual(report["status"], "completed")
            stages = [item["stage"] for item in report["stages"]]
            for name in ("train_dataset_built", "validation_dataset_built", "booster_fit_completed"):
                self.assertEqual(stages.count(name), 1)
            self.assertGreater(report["stages"][-1]["peak_rss_bytes"], 0)
            self.assertEqual(set(report), {"purpose", "status", "stages"})
            self.assertEqual(output.parent.stat().st_mode & 0o777, 0o700)

    def test_cohort_projection_and_normalization_preserve_all_values_and_order(self):
        import tempfile
        from pathlib import Path

        import pandas as pd
        import pyarrow.parquet as pq

        from backend.ml.scripts.verify_yaib import synthetic, write
        from backend.ml.yaib import cohort as C

        dyn, sta, outc = synthetic(n=100)
        # Irrelevant columns should never be materialized, even if they are large.
        dyn["unused_blob"] = "unneeded-" * 100
        with tempfile.TemporaryDirectory() as root:
            directory = write(str(Path(root) / "cohort"), dyn, sta, outc)
            original = pq.ParquetFile.read
            reads = []

            def read(parquet, **kwargs):
                self.assertIsNotNone(kwargs.get("columns"))
                self.assertFalse(kwargs["use_threads"])
                reads.append(kwargs["columns"])
                return original(parquet, **kwargs)

            with patch.object(pq.ParquetFile, "read", read):
                got = C.load(directory)
        expected = dyn.assign(hour=dyn[C.SEQUENCE] / pd.Timedelta(hours=1))
        expected = expected.sort_values(
            [C.GROUP, "hour"], kind="mergesort"
        ).reset_index(drop=True)
        expected = expected[[C.GROUP, "hour"] + C.DYNAMIC]
        expected["hour"] = expected["hour"].astype(float)
        pd.testing.assert_frame_equal(got.dyn, expected)
        labels = outc.assign(hour=outc[C.SEQUENCE] / pd.Timedelta(hours=1)).set_index(
            [C.GROUP, "hour"]
        )[C.LABEL]
        truth = labels.reindex(
            pd.MultiIndex.from_frame(expected[[C.GROUP, "hour"]])
        ).to_numpy(dtype=float)
        np.testing.assert_array_equal(got.label, truth)
        self.assertNotIn(
            "unused_blob", [column for columns in reads for column in columns]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
