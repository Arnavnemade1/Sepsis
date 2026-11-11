"""Private one-round memory diagnostic, never a scored clinical benchmark.

Keeps logs beside the matrix in a mode-0700 directory; emits numeric resource usage
and fixed stage names only. Model scores and patient rows are never emitted.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from backend.ml.yaib import bench as B
from backend.ml.yaib import cohort as C
from backend.ml.yaib import features as FE
from backend.ml.yaib import splits as SP
from backend.ml.yaib.checkpoint import atomic_json


@contextmanager
def private_logs(path):
    saved = os.dup(1), os.dup(2)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for item in (*saved, fd):
            os.close(item)


def profile(directory, features, output):
    features, output = Path(features).resolve(), Path(output).resolve()
    if not output.is_relative_to(features.parent):
        raise ValueError("Keep diagnostics in the matrix private directory")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.parent.stat().st_mode & 0o077:
        raise ValueError("Diagnostic directory must be private")
    if output.exists():
        raise ValueError("Do not overwrite an existing diagnostic")
    report = {"purpose": "memory_diagnostic_only", "status": "running", "stages": []}
    start = time.monotonic()

    def record(stage, status=None, error_type=None):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["stages"].append(
            {
                "stage": stage,
                "seconds": round(time.monotonic() - start, 3),
                "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            }
        )
        if status:
            report["status"] = status
        if error_type:
            report["error_type"] = error_type
        atomic_json(output, report)

    with private_logs(output.parent / "private-fit.log"):
        try:
            record("loading_cohort")
            cohort = C.load(str(directory), stage=record)
            record("cohort_loaded")
            X = np.load(features, mmap_mode="r", allow_pickle=False)
            if (
                X.shape != (len(cohort.dyn), len(FE.feature_names("base")))
                or X.dtype != np.float32
            ):
                raise ValueError("Diagnostic matrix does not match base features")
            split = SP.split(cohort.stay_labels(), 0, 0, seed=1234)
            stay = cohort.dyn[C.GROUP].to_numpy().copy()
            valid = np.isfinite(cohort.label)
            y = np.nan_to_num(cohort.label, nan=0.0)
            tr = np.flatnonzero(np.isin(stay, split.train) & valid)
            va = np.flatnonzero(np.isin(stay, split.val) & valid)
            te = np.flatnonzero(np.isin(stay, split.test) & valid)
            del cohort, stay, valid
            gc.collect()
            record("fit_prepared")
            record("native_fit_started")
            result = B._fit_predict(
                X,
                y,
                tr,
                va,
                te,
                B.TrainConfig(max_rounds=1, early_stopping=1),
                stream=True,
                stage=record,
            )
            del result
            gc.collect()
            record("native_fit_completed", status="completed")
        except Exception as error:  # noqa: BLE001 - privacy boundary: emit only the exception type
            record("failed", status="failed", error_type=type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        report = profile(args.cohort, args.features, args.output)
    except Exception as error:  # noqa: BLE001 - privacy boundary: emit only the exception type
        report = {
            "purpose": "memory_diagnostic_only",
            "status": "failed",
            "error_type": type(error).__name__,
        }
    print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
