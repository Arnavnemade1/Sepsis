"""Run a YAIB machine-learning baseline in less memory, with the same model inputs.

YAIB's LightGBM baseline cannot run on the 7.6 GB worker against full eICU: two one-fold
probes were killed (exit 137) after preprocessing finished, while ``train_common`` built
the train dataset. Three things in YAIB af9a5b3 hold copies that its ML models never read:

1. ``configs/tasks/BinaryClassification.gin`` sets ``train_common.ram_cache = True``.
   That caches every stay padded to the longest stay, as tensors, for the deep-learning
   ``__getitem__`` path. ML models read their data through ``get_data_and_labels`` (fit,
   and ``to_tensor`` for the test split), never through ``__getitem__``.
2. ``PredictionPolarsDataset.__init__`` always partitions the split by stay and keeps a
   float32 array per stay (``_feat_arrays``). Only ``__getitem__`` reads those arrays.
3. The preprocessed features are float64 and boolean, and ``get_data_and_labels`` copies
   them three times (``to_numpy``, ``astype(float32)`` and the ``valid`` row filter),
   even though the model is handed float32.

This driver removes those copies and nothing else:

1. It binds ``train_common.ram_cache=False``, through YAIB's own ``-hp`` bindings.
2. It replaces the dataset class that ``train_common`` uses with a subclass that skips
   the per-stay partition and builds a stay's arrays only when an item is asked for.
   ML models ask for one shuffled batch, which ``train_common`` reads for its shape
   (``data_shape = next(iter(train_loader))[0].shape``). Those items come from YAIB's own
   ``_build_item``, and the shuffle draws from the same random generator as stock YAIB.
3. Right after ``preprocess_data`` returns, it casts the float64 and boolean feature
   columns to float32, and ``get_data_and_labels`` fills one preallocated float32 array.
4. Before each fold's ``preprocess_data``, it empties the previous fold's data dict.
   ``execute_repeated_cv`` still holds that dict while it preprocesses the next fold, so
   the first full-eICU run held two folds at once and was killed starting fold 1 (both
   LightGBM and LR: fold 0 scored, exit 137 with swap full). Nothing reads a fold's data
   after its ``train_common`` returns. The folds stay in one process, so the random
   state each fold starts from is the same as in stock YAIB.

Why the model inputs are the same: YAIB hands the ML model ``to_numpy().astype(float32)``
of the same frame. A float64 value rounded to float32 once is the same value whether
the rounding happens after preprocessing or after ``to_numpy``; a boolean becomes 0.0
or 1.0 either way. Nothing does arithmetic on the features between those two points
(the dataset only sorts rows, drops the hour column and reorders columns). The integer
columns (the hour index, encoded static categories) are left as they are and converted
column by column. They are far below 2**24, so they are exact in float32.
That argument is checked, not trusted: ``deploy/physionet/yaib_lean_check.sh`` runs stock
YAIB and this driver on YAIB's eICU demo cohort and requires, for every run, the label
and probability columns of ``pred_indicators.csv`` byte-identical row by row and the same
set of (stay, hour) ids (the order of YAIB's id columns varies between two stock runs).
Use the driver only after that check passes.

Run it like ``backend.ml.yaib.run_baseline`` (YAIB's checkout as working directory)::

    python -m backend.ml.scripts.yaib_lean_baseline train -d <cohort> -n <name> \\
        -t BinaryClassification -tn Sepsis -m LGBMClassifier --seed 1234 -l <logs>
"""
from __future__ import annotations

import gc
import logging
import sys
from typing import List, Optional, Sequence

RAM_CACHE_OFF = "train_common.ram_cache=False"


def with_ram_cache_off(argv: Sequence[str]) -> List[str]:
    """Add the binding to YAIB's ``-hp`` list. A second ``-hp`` would replace the first."""
    args = list(argv)
    if any(a.startswith("train_common.ram_cache") for a in args):
        raise ValueError("the lean driver sets train_common.ram_cache itself")
    for flag in ("-hp", "--hyperparams"):
        if flag in args:
            i = args.index(flag)
            return args[: i + 1] + [RAM_CACHE_OFF] + args[i + 1:]
    return args + ["-hp", RAM_CACHE_OFF]


def release(data: Optional[dict]) -> None:
    """Empty a fold's {split: {segment: frame}} dict in place, so the caller's reference frees it."""
    if not data:
        return
    for split in data.values():
        if isinstance(split, dict):
            split.clear()
    data.clear()


def float32_features(data: dict, keep: Sequence[str]) -> dict:
    """Cast float64 and boolean feature columns to float32, in place of the originals."""
    import polars as pl
    from icu_benchmarks.data.constants import DataSegment

    for split in data.values():
        frame = split.get(DataSegment.features)
        if frame is None:
            continue
        cast = [c for c, t in frame.schema.items()
                if c not in keep and (t == pl.Float64 or t == pl.Boolean)]
        split[DataSegment.features] = frame.with_columns(pl.col(cast).cast(pl.Float32))
    return data


def install() -> None:
    import gin
    import numpy as np
    import polars as pl
    import icu_benchmarks.cross_validation as cross_validation
    import icu_benchmarks.models.train as train
    from icu_benchmarks.data.loader import CommonPolarsDataset, PredictionPolarsDataset

    stock_preprocess = cross_validation.preprocess_data

    previous: List[dict] = []

    def preprocess_data(*args, **kwargs):
        while previous:
            release(previous.pop())
        gc.collect()
        data, target_transform = stock_preprocess(*args, **kwargs)
        previous.append(data)
        vars = gin.query_parameter("vars/gin.macro.value")
        float32_features(data, keep=[vars["GROUP"], vars["SEQUENCE"]])
        return data, target_transform

    class PerStay:
        """One stay's float32 array, made when it is looked up (read-only, as in the stock class)."""

        def __init__(self, frame, group, pick):
            self.frame, self.group, self.pick = frame, group, pick

        def __getitem__(self, stay):
            array = self.pick(self.frame.filter(pl.col(self.group) == stay)).to_numpy().astype(np.float32)
            array.setflags(write=False)
            return array

    class LeanPredictionPolarsDataset(PredictionPolarsDataset):
        """``PredictionPolarsDataset`` for ML models: no per-stay copy, no item cache."""

        def __init__(self, data, *args, ram_cache: bool = False, vars: Optional[dict] = None, **kwargs):
            if ram_cache:
                raise ValueError("the lean dataset does not cache deep-learning items")
            # Gin injects PredictionPolarsDataset.vars into the stock class only; it is the %vars macro
            vars = vars if vars is not None else gin.query_parameter("vars/gin.macro.value")
            CommonPolarsDataset.__init__(self, data, *args, vars=vars, **kwargs)
            self.outcome_df = self.grouping_df
            group, label = self.vars["GROUP"], self.vars["LABEL"]
            # The stock class keys these by partition_by(GROUP, maintain_order=True): stays in order of
            # first appearance, rows in frame order. A filter keeps the same rows in the same order.
            self._stay_order = self.outcome_df[group].unique(maintain_order=True).to_list()
            self._feat_arrays = PerStay(self.features_df, group, lambda part: part.select(pl.exclude(group)))
            self._label_arrays = PerStay(self.outcome_df, group, lambda part: part[label])
            self._cached_dataset = None

        def get_data_and_labels(self):
            labels = self.outcome_df[self.vars["LABEL"]].to_numpy().astype(np.float32)
            rep = self.features_df
            group = self.vars["GROUP"]
            if len(labels) == self.num_stays:
                rep = self.outcome_df.select(group).join(rep.group_by(group).last(), on=group, how="left")
                rep = rep.drop("counter", strict=False)
                row_indicators = self.outcome_df.select(group)
            else:
                row_indicators = self.row_indicators
            rep = rep.drop(group)
            valid = np.isfinite(labels.reshape(len(labels), -1)).all(axis=1)
            keep = None if valid.all() else np.flatnonzero(valid)
            rows = len(rep) if keep is None else len(keep)
            out = np.empty((rows, rep.width), dtype=np.float32)
            for j, column in enumerate(rep.iter_columns()):
                values = column.to_numpy()
                out[:, j] = values if keep is None else values[keep]
            del rep
            return out, labels[valid], row_indicators.to_numpy()[valid]

    cross_validation.preprocess_data = preprocess_data
    train.PredictionPolarsDataset = LeanPredictionPolarsDataset
    logging.info("lean ML dataset installed (no item cache, no per-stay copy, float32 features)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import icu_benchmarks.models.wrappers as W
    from icu_benchmarks.run import main as yaib_main

    install()
    W.BaseModule.debug = True          # save per-hour test predictions, as run_baseline does
    yaib_main(tuple(with_ram_cache_off(sys.argv[1:] if argv is None else argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
