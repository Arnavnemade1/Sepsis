"""Private, content-validated checkpoints for a fixed YAIB experiment.

No row values are written to JSON. The matrix and test probabilities remain local
binary artifacts and must stay inside the authorized lake for credentialed cohorts.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HASH_ROWS = 65536
#: the sources that decide a feature matrix; a learner or CLI change does not rebuild it
FEATURE_SOURCES = ("cohort.py", "features.py")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint_frame(frame):
    h = hashlib.sha256(
        json.dumps(
            [(str(c), str(frame[c].dtype)) for c in frame], separators=(",", ":")
        ).encode()
    )
    for start in range(0, len(frame), HASH_ROWS):
        values = pd.util.hash_pandas_object(
            frame.iloc[start : start + HASH_ROWS], index=False
        ).to_numpy()
        h.update(values.tobytes())
    return h.hexdigest()


def experiment_spec(cohort, configuration, protocol, feature_set):
    root = Path(__file__).parent
    return {
        "version": 1,
        "inputs": {
            "dynamic": fingerprint_frame(cohort.dyn),
            "static": fingerprint_frame(cohort.sta),
            "labels": fingerprint_frame(pd.DataFrame({"label": cohort.label})),
        },
        "configuration": configuration,
        "protocol": protocol,
        "feature_set": feature_set,
        "implementation": {p.name: sha_file(p) for p in sorted(root.glob("*.py"))},
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "lightgbm",
                    "pyarrow",
                )
            },
        },
    }


def feature_spec(spec):
    """The part of an experiment that determines its feature matrix: the inputs, the
    feature set, the feature code and the environment. The learner's configuration and
    the protocol are deliberately absent, so configurations can share one matrix."""
    return {
        "version": 1,
        "inputs": spec["inputs"],
        "feature_set": spec["feature_set"],
        "implementation": {name: spec["implementation"][name] for name in FEATURE_SOURCES},
        "environment": spec["environment"],
    }


def identity_of(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".receipt-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def committed_matrix(root, identity, shape, builder):
    """The float32 matrix committed in ``root`` under ``identity``: validated by content
    hash when a receipt exists, otherwise built (after a disk preflight) and committed."""
    path, receipt = root / "features.npy", root / "features.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if (
            saved.get("identity") != identity
            or not path.is_file()
            or sha_file(path) != saved.get("sha256")
        ):
            raise ValueError("Feature checkpoint integrity failure")
    else:
        # The builder holds a full column-major scratch plus the row-major output.
        # Refuse before partially filling the disk, including header/receipt headroom.
        path.unlink(
            missing_ok=True
        )  # owned output without a commit receipt is incomplete
        needed = (
            2 * int(np.prod(shape)) * np.dtype(np.float32).itemsize
            + 16 * 1024 * 1024
        )
        temporary = root / "features.build.npy"
        for incomplete in (temporary, Path(str(temporary) + ".columns.npy")):
            incomplete.unlink(missing_ok=True)
        if shutil.disk_usage(root).free < needed:
            raise OSError(
                "Insufficient disk for the feature matrix and conversion scratch"
            )
        try:
            matrix = builder(str(temporary))
            if (
                matrix.shape != shape
                or matrix.dtype != np.float32
                or not matrix.flags.c_contiguous
            ):
                raise ValueError("Feature builder returned an unexpected matrix")
            if hasattr(matrix, "flush"):
                matrix.flush()
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            del matrix
            os.replace(temporary, path)
            sync_directory(root)
            atomic_json(receipt, {"identity": identity, "sha256": sha_file(path)})
        finally:
            for incomplete in (temporary, Path(str(temporary) + ".columns.npy")):
                incomplete.unlink(missing_ok=True)
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        matrix.shape != shape
        or matrix.dtype != np.float32
        or not matrix.flags.c_contiguous
    ):
        raise ValueError("Invalid feature checkpoint layout")
    return matrix


class FeatureStore:
    """One committed feature matrix per feature identity, shared by every experiment
    whose inputs, feature set, feature code and environment are the same.

    A learner setting does not change a feature, so a sweep of configurations over a
    full cohort needs one 18 GB matrix, not one per configuration. The matrix is
    validated by content hash on every open, exactly as an experiment's own is."""

    def __init__(self, root, spec):
        self.spec = feature_spec(spec)
        self.identity = identity_of(self.spec)
        base = Path(root).resolve()
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        if base.stat().st_mode & 0o077:
            raise ValueError("Feature store must be private (mode 0700)")
        self.root = base / self.identity[:16]
        self.root.mkdir(exist_ok=True, mode=0o700)

    @contextlib.contextmanager
    def _locked(self):
        with (self.root / "store.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("This feature store is being written") from None
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _own_spec(self):
        path = self.root / "feature_spec.json"
        if path.exists():
            if json.loads(path.read_text()) != self.spec:
                raise ValueError("Feature store identity collision")
        else:
            atomic_json(path, self.spec)

    def matrix(self, shape, builder):
        with self._locked():
            self._own_spec()
            return committed_matrix(self.root, self.identity, shape, builder)

    def adopt(self, checkpoint_dir):
        """Share the matrix an experiment checkpoint already committed, without copying
        or rebuilding it. Accepted only when that experiment's receipt is intact, its
        matrix still hashes to the receipt, and its feature identity (inputs, feature
        set, feature code, environment) is this store's. The file is hard-linked, so it
        costs no disk and the experiment checkpoint stays valid."""
        source = Path(checkpoint_dir).resolve()
        experiment = json.loads((source / "experiment.json").read_text())
        receipt = json.loads((source / "features.json").read_text())
        if receipt.get("identity") != identity_of(experiment):
            raise ValueError("That checkpoint's feature receipt is not its experiment's")
        if feature_spec(experiment) != self.spec:
            raise ValueError("That checkpoint's features come from different inputs, code or environment")
        with self._locked():
            self._own_spec()
            target, mine = self.root / "features.npy", self.root / "features.json"
            if mine.exists():
                saved = json.loads(mine.read_text())
                if saved.get("identity") != self.identity or saved.get("sha256") != receipt.get("sha256"):
                    raise ValueError("The store already holds a different matrix")
                return saved
            if sha_file(source / "features.npy") != receipt.get("sha256"):
                raise ValueError("That checkpoint's matrix no longer matches its receipt")
            target.unlink(missing_ok=True)       # an uncommitted leftover is incomplete
            os.link(source / "features.npy", target)
            sync_directory(self.root)
            saved = {"identity": self.identity, "sha256": receipt["sha256"],
                     "adopted_from": receipt["identity"]}
            atomic_json(mine, saved)
            return saved


class Checkpoints:
    def __init__(self, root, spec):
        self.root, self.spec = Path(root).resolve(), spec
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.stat().st_mode & 0o077:
            raise ValueError("Checkpoint directory must be private (mode 0700)")
        self.identity = identity_of(spec)

    @contextlib.contextmanager
    def locked(self):
        with (self.root / "checkpoint.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("This checkpoint is already in use") from None
            path = self.root / "experiment.json"
            if path.exists():
                if json.loads(path.read_text()) != self.spec:
                    raise ValueError(
                        "Experiment changed; choose a new checkpoint directory"
                    )
            else:
                # Never adopt somebody else's unreceipted matrix or probabilities.
                if any(p.name != "checkpoint.lock" for p in self.root.iterdir()):
                    raise ValueError("Checkpoint directory contains unowned artifacts")
                atomic_json(path, self.spec)
            try:
                yield self
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def features(self, shape, builder, store=None):
        """This experiment's matrix. With ``store`` (a :class:`FeatureStore`) the matrix
        lives there and is shared by every configuration with the same feature identity;
        a directory that already committed its own matrix keeps using it."""
        if store is not None and not (self.root / "features.json").exists():
            return store.matrix(shape, builder)
        return committed_matrix(self.root, self.identity, shape, builder)

    def _paths(self, repetition, fold):
        stem = self.root / f"fit-r{int(repetition)}-f{int(fold)}"
        return stem.with_suffix(".npy"), stem.with_suffix(".json")

    @staticmethod
    def validate_fit(fit, size):
        p = np.asarray(fit["p"])
        if p.shape != (size,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid checkpoint probabilities")
        if type(fit["rounds"]) is not int or fit["rounds"] < 1:
            raise ValueError("Invalid checkpoint training length")
        value = fit["val_auroc"]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Invalid checkpoint validation metric")

    def load_fit(self, repetition, fold, split, size):
        path, receipt = self._paths(repetition, fold)
        if not receipt.exists():
            return None
        saved = json.loads(receipt.read_text())
        receipt_sha = saved.pop("receipt_sha256", None)
        if (
            receipt_sha
            != hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
        ):
            raise ValueError("Fit receipt integrity failure")
        if (
            saved.get("identity") != self.identity
            or saved.get("split") != split
            or not path.is_file()
            or sha_file(path) != saved.get("sha256")
        ):
            raise ValueError("Fit checkpoint integrity failure")
        fit = {
            "p": np.load(path, allow_pickle=False),
            "rounds": saved["rounds"],
            "val_auroc": saved["val_auroc"],
        }
        self.validate_fit(fit, size)
        return fit

    def save_fit(self, repetition, fold, split, fit, size):
        self.validate_fit(fit, size)
        path, receipt = self._paths(repetition, fold)
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".fit-")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.save(stream, fit["p"], allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_directory(self.root)
            saved = {
                "identity": self.identity,
                "split": split,
                "sha256": sha_file(path),
                "rounds": fit["rounds"],
                "val_auroc": fit["val_auroc"],
            }
            saved["receipt_sha256"] = hashlib.sha256(
                json.dumps(saved, sort_keys=True).encode()
            ).hexdigest()
            atomic_json(receipt, saved)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
