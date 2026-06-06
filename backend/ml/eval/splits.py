"""Patient-level (group) splits. Rows from one patient must never straddle the
train/test boundary, or the model effectively memorises the patient and every
metric becomes a fantasy. We split by patient and stratify by each patient's
septic status so prevalence is stable across folds.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def patient_labels(groups: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse row labels to one label per patient (septic if any hour is)."""
    uniq = np.array(sorted(set(groups.tolist())))
    plabel = np.array([int(y[groups == g].max()) for g in uniq])
    return uniq, plabel


def stratified_group_split(
    groups: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single train/test split of *row indices*, grouped by patient and
    stratified by patient-level sepsis status."""
    n_splits = max(2, round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


def stratified_group_folds(
    groups: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) row-index folds for cross-validation."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from sgkf.split(np.zeros(len(y)), y, groups)
