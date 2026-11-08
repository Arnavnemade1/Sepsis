"""YAIB's repeated cross-validation split, reproduced.

From ``icu_benchmarks.data.split_process_data.make_single_split_polars`` (YAIB commit
af9a5b3, the default polars path), for a classification task with no fixed train size:

    stays   = unique stay ids, sorted
    labels  = each stay's max hourly label, in that order
    outer   = StratifiedKFold(cv_repetitions, shuffle=True, random_state=seed)
    dev, test   = outer.split(stays, labels)[repetition]
    inner   = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
    train, val  = inner.split(stays[dev], labels[dev])[fold]

The seed is the run's ``--seed``. YAIB's command line defaults to 1234; the function's
own default (42) is never reached from the CLI, so 1234 is the default here too. Pass
whatever the baseline runs were given.

``conformance`` compares these ids with YAIB's own function when ``icu_benchmarks`` is
importable (on the worker that runs the baselines), and the split fingerprint below is
written into every result so two runs can be checked against each other later.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

YAIB_CLI_SEED = 1234


@dataclass(frozen=True)
class Split:
    repetition: int
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for part in (self.train, self.val, self.test):
            h.update(np.sort(np.asarray(part, dtype=np.int64)).tobytes())
            h.update(b"|")
        return h.hexdigest()[:16]


def split(stay_labels: pd.Series, repetition: int, fold: int, cv_repetitions: int = 5,
          cv_folds: int = 5, seed: int = YAIB_CLI_SEED) -> Split:
    """``stay_labels``: per-stay label indexed by stay id (``Cohort.stay_labels``)."""
    s = stay_labels.sort_index()
    stays = s.index.to_numpy()
    labels = s.to_numpy()
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2 or counts.min() < cv_folds:
        raise ValueError(f"the smallest class has {int(counts.min()) if len(counts) else 0} stays, "
                         f"fewer than the {cv_folds} folds requested (YAIB refuses this too)")
    outer = StratifiedKFold(cv_repetitions, shuffle=True, random_state=seed)
    dev, test = list(outer.split(stays, labels))[repetition]
    inner = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
    tr, va = list(inner.split(stays[dev], labels[dev]))[fold]
    return Split(repetition, fold, stays[dev][tr], stays[dev][va], stays[test])


def all_splits(stay_labels: pd.Series, cv_repetitions: int = 5, cv_folds: int = 5,
               seed: int = YAIB_CLI_SEED) -> Iterator[Split]:
    for r in range(cv_repetitions):
        for f in range(cv_folds):
            yield split(stay_labels, r, f, cv_repetitions, cv_folds, seed)


def conformance(stay_labels: pd.Series, cv_repetitions: int = 5, cv_folds: int = 5,
                seed: int = YAIB_CLI_SEED) -> Optional[Dict[str, bool]]:
    """Our split vs YAIB's own, for every (repetition, fold). None if YAIB is absent."""
    try:
        import polars as pl
        from icu_benchmarks.data.split_process_data import make_single_split_polars
    except ImportError:
        return None
    s = stay_labels.sort_index()
    outc = pl.DataFrame({"stay_id": s.index.to_numpy(), "label": s.to_numpy().astype(bool)})
    vars_ = {"GROUP": "stay_id", "LABEL": "label", "SEQUENCE": "time"}
    out: Dict[str, bool] = {}
    for r in range(cv_repetitions):
        for f in range(cv_folds):
            theirs = make_single_split_polars({"OUTCOME": outc}, vars_, cv_repetitions, r, cv_folds, f, seed=seed)
            ours = split(stay_labels, r, f, cv_repetitions, cv_folds, seed)
            same = all(np.array_equal(np.sort(theirs[k]["OUTCOME"]["stay_id"].to_numpy()), np.sort(getattr(ours, k)))
                       for k in ("train", "val", "test"))
            out[f"r{r}f{f}"] = bool(same)
    return out
