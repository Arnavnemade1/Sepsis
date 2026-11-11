"""Synthetic interruption, parity and invalidation checks; no credentialed data.

python -m backend.ml.scripts.verify_yaib_resume
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.ml.scripts.verify_yaib import synthetic, write
from backend.ml.yaib import bench as B
from backend.ml.yaib import checkpoint as CP
from backend.ml.yaib import cohort as C
from backend.ml.yaib import features as FE


class ResumeChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.input = write(str(self.root / "input"), *synthetic(n=100))
        self.cfg = B.TrainConfig(max_rounds=15, early_stopping=5)
        self.pairs = [(0, 0), (1, 1)]
        self.resume = self.root / "resume"

    def run_bench(self, **overrides):
        arguments = {
            "cfg": self.cfg,
            "runs": self.pairs,
            "resume_dir": str(self.resume),
            "memory_check": False,
        }
        arguments.update(overrides)
        return B.run(C.load(self.input), **arguments)

    @staticmethod
    def fake_fit(X, y, tr, va, te, cfg, stream=False):
        column = FE.feature_names().index("hr")

        def probability(rows):
            v = np.nan_to_num(X[rows, column], nan=80)
            return 1 / (1 + np.exp(-(v.astype(float) - 80) / 10))

        return {
            "p": probability(te),
            "rounds": 5,
            "val_auroc": float(roc_auc_score(y[va], probability(va))),
        }

    def seed_checkpoint(self):
        with patch.object(B, "_fit_predict", self.fake_fit):
            return self.run_bench()

    def test_interruption_reuses_matrix_and_only_refits_missing_split(self):
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("planted interruption")
            return self.fake_fit(*args, **kwargs)

        with (
            patch.object(B, "_fit_predict", interrupted),
            self.assertRaisesRegex(RuntimeError, "planted"),
        ):
            self.run_bench()
        self.assertEqual(len(list(self.resume.glob("fit-*.json"))), 1)
        with (
            patch.object(FE, "build", side_effect=AssertionError("matrix rebuilt")),
            patch.object(B, "_fit_predict", side_effect=self.fake_fit) as fit,
        ):
            result = self.run_bench()
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(len(result["runs"]), 2)

    def test_completed_replay_performs_zero_fits_and_repairs_prediction_export(self):
        expected = self.seed_checkpoint()
        predictions = self.root / "export"
        with (
            patch.object(FE, "build", side_effect=AssertionError("matrix rebuilt")),
            patch.object(B, "_fit_predict", side_effect=AssertionError("fit repeated")),
        ):
            actual = self.run_bench(predictions_dir=str(predictions))
        for key in ("runs", "yaib_metric", "validation_metric", "model", "protocol"):
            self.assertEqual(expected[key], actual[key])
        self.assertEqual(len(list(predictions.glob("*.parquet"))), 2)

    def test_real_streaming_model_results_are_identical_with_checkpoints(self):
        original = self.run_bench(
            resume_dir=None,
            features_path=str(self.root / "original.npy"),
            predictions_dir=str(self.root / "original-p"),
        )
        cached = self.run_bench(predictions_dir=str(self.root / "cached-p"))
        self.assertEqual(original["runs"], cached["runs"])
        self.assertEqual(original["yaib_metric"], cached["yaib_metric"])
        for rep, fold in self.pairs:
            name = f"ours_r{rep}_f{fold}.parquet"
            pd.testing.assert_frame_equal(
                pd.read_parquet(self.root / "original-p" / name),
                pd.read_parquet(self.root / "cached-p" / name),
            )

    def test_changes_to_config_protocol_environment_or_inputs_are_rejected(self):
        self.seed_checkpoint()
        for override in (
            {"cfg": B.TrainConfig(num_leaves=7)},
            {"seed": 999},
            {"cv_folds": 4},
            {"feature_set": "rich"},
        ):
            with (
                self.subTest(override=override),
                self.assertRaisesRegex(ValueError, "Experiment changed"),
            ):
                self.run_bench(**override)
        with (
            patch.object(CP.importlib.metadata, "version", return_value="changed"),
            self.assertRaisesRegex(ValueError, "Experiment changed"),
        ):
            self.run_bench()
        for name in ("dynamic", "static", "labels"):
            cohort = C.load(self.input)
            if name == "dynamic":
                cohort.dyn.loc[0, "hr"] += 1
            elif name == "static":
                cohort.sta.loc[0, "age"] += 1
            else:
                cohort.label[0] = 1 - cohort.label[0]
            with (
                self.subTest(input=name),
                self.assertRaisesRegex(ValueError, "Experiment changed"),
            ):
                B.run(
                    cohort,
                    self.cfg,
                    runs=self.pairs,
                    resume_dir=str(self.resume),
                    memory_check=False,
                )

    def test_source_change_is_not_reused(self):
        self.seed_checkpoint()
        old = CP.experiment_spec

        def changed(*args):
            spec = old(*args)
            spec["implementation"]["bench.py"] = "changed"
            return spec

        with (
            patch.object(CP, "experiment_spec", changed),
            self.assertRaisesRegex(ValueError, "Experiment changed"),
        ):
            self.run_bench()

    def test_corrupt_matrix_and_prediction_files_are_rejected(self):
        self.seed_checkpoint()
        for name in ("features.npy", "fit-r0-f0.npy"):
            path = self.resume / name
            original = path.read_bytes()
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([value[0] ^ 1]))
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "integrity failure"),
            ):
                self.run_bench()
            path.write_bytes(original)

    def test_changed_validation_receipt_is_rejected(self):
        self.seed_checkpoint()
        path = self.resume / "fit-r0-f0.json"
        receipt = json.loads(path.read_text())
        receipt["val_auroc"] = 0.123
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "receipt integrity"):
            self.run_bench()

    def test_requested_run_subset_can_grow_without_repeating_old_fit(self):
        with patch.object(B, "_fit_predict", self.fake_fit):
            first = self.run_bench(runs=[self.pairs[0]])
        with patch.object(B, "_fit_predict", side_effect=self.fake_fit) as fit:
            both = self.run_bench()
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(first["runs"][0], both["runs"][0])

    def test_incomplete_feature_build_is_removed_and_recoverable(self):
        builder = FE.build

        def broken(cohort, out=None, **kwargs):
            Path(out).write_text("incomplete matrix")
            Path(str(out) + ".columns.npy").write_text("incomplete columns")
            raise RuntimeError("planted copy failure")

        with (
            patch.object(FE, "build", broken),
            self.assertRaisesRegex(RuntimeError, "planted copy"),
        ):
            self.run_bench()
        self.assertFalse(list(self.resume.glob("*.build*")))
        with (
            patch.object(FE, "build", builder),
            patch.object(B, "_fit_predict", self.fake_fit),
        ):
            self.assertEqual(len(self.run_bench()["runs"]), 2)

    def test_disk_preflight_happens_before_build(self):
        from collections import namedtuple

        Disk = namedtuple("Disk", "total used free")
        with (
            patch.object(CP.shutil, "disk_usage", return_value=Disk(1, 1, 0)),
            patch.object(FE, "build", side_effect=AssertionError("unexpected build")),
            self.assertRaisesRegex(OSError, "Insufficient disk"),
        ):
            self.run_bench()

    def test_file_lock_prevents_two_coordinators(self):
        spec = CP.experiment_spec(C.load(self.input), {}, {}, "base")
        (one, two) = (
            CP.Checkpoints(self.resume, spec),
            CP.Checkpoints(self.resume, spec),
        )
        with (
            one.locked(),
            self.assertRaisesRegex(RuntimeError, "already in use"),
            two.locked(),
        ):
            self.fail("second coordinator acquired lock")

    def test_unowned_artifacts_are_not_adopted(self):
        self.resume.mkdir(mode=0o700)
        (self.resume / "features.npy").write_text("unowned")
        with self.assertRaisesRegex(ValueError, "unowned artifacts"):
            self.run_bench()

    def test_invalid_fit_is_not_committed(self):
        for field, value in [
            ("p", np.array([np.nan])),
            ("rounds", 0),
            ("val_auroc", float("nan")),
        ]:
            with self.subTest(field=field):
                fit = {"p": np.array([0.2]), "rounds": 1, "val_auroc": 0.8}
                fit[field] = value
                with self.assertRaises(ValueError):
                    CP.Checkpoints.validate_fit(fit, 1)

    def test_json_manifest_has_hashes_not_rows_or_note_values(self):
        spec = CP.experiment_spec(C.load(self.input), {}, {}, "base")
        self.assertEqual(set(spec["inputs"]), {"dynamic", "static", "labels"})
        self.assertTrue(all(len(value) == 64 for value in spec["inputs"].values()))
        self.assertNotIn("stay_id", json.dumps(spec))
        changed = copy.deepcopy(spec)
        changed["inputs"]["dynamic"] = "changed"
        self.assertNotEqual(
            CP.Checkpoints(self.resume, spec).identity,
            CP.Checkpoints(self.resume, changed).identity,
        )

    def test_matrix_without_commit_receipt_is_rebuilt(self):
        original = CP.atomic_json

        def fail_receipt(path, value):
            if path.name == "features.json":
                raise OSError("planted receipt failure")
            return original(path, value)

        with (
            patch.object(CP, "atomic_json", fail_receipt),
            self.assertRaisesRegex(OSError, "planted receipt"),
        ):
            self.run_bench()
        self.assertTrue((self.resume / "features.npy").is_file())
        self.assertFalse((self.resume / "features.json").exists())
        with (
            patch.object(FE, "build", wraps=FE.build) as build,
            patch.object(B, "_fit_predict", self.fake_fit),
        ):
            self.run_bench()
        self.assertEqual(build.call_count, 1)

    def test_public_checkpoint_directory_is_refused_without_chmodding_it(self):
        self.resume.mkdir(mode=0o755)
        self.resume.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "must be private"):
            self.run_bench()
        self.assertEqual(self.resume.stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main(verbosity=2)
