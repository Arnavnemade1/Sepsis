"""Train and score our model under YAIB's protocol; compare it with YAIB's baselines.

THE METRIC, AS YAIB COMPUTES IT
-------------------------------
For each of the cv_repetitions x cv_folds runs, a model is fitted on that run's train
stays (validation stays only for early stopping), and ``roc_auc_score`` /
``average_precision_score`` are computed over every test-stay hour whose label is finite
(``MLWrapper.test_step`` -> ``log_metrics``; ``get_data_and_labels`` drops NaN labels).
The reported number is the mean and standard deviation of those per-run values. That
is what ``run`` returns under ``"yaib_metric"``.

PAIRED COMPARISON
-----------------
YAIB saves each run's test predictions as ``pred_indicators.csv``
(stay id, hour, label, p0, p1). ``compare`` joins ours to theirs on (run, stay, hour),
refuses if the rows or labels differ, and bootstraps the AUROC difference over test
STAYS within each repetition (the inner folds of a repetition share one test set, so
their difference is averaged before resampling). A claim of "better than the baseline"
needs that interval to exclude zero; the mean of 25 numbers with overlapping error bars
is not one.

OUTPUT
------
Aggregates only: per-run AUROC / AUPRC, their mean and std, the split fingerprints, the
comparison. Never a row.
"""
from __future__ import annotations

import contextlib
import glob
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from backend.ml.yaib import families as FAM
from backend.ml.yaib import features as FE
from backend.ml.yaib import splits as SP
from backend.ml.yaib.cohort import GROUP, Cohort


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 200
    feature_fraction: float = 0.5
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    max_rounds: int = 3000
    early_stopping: int = 100
    threads: int = 2
    seed: int = 0
    balanced: bool = True
    force_col_wise: bool = True
    extra_trees: bool = False      # random split thresholds: less variance on wide, noisy columns
    path_smooth: float = 0.0       # shrink small leaves' outputs toward their parents'
    seeds: int = 1                 # average this many boosters, LightGBM seeds seed, seed + 1, ...
    train_fraction: float = 1.0    # screening: fit on this share of the run's training stays only
    target: str = "yaib"           # the label trees are TRAINED on; validation and test are always YAIB's


TARGETS = ("yaib", "early6", "stay")


class _Rows:
    """Some rows of a (memmapped) matrix as a LightGBM ``Sequence``: LightGBM samples rows
    to place its bins, then reads ``batch_size`` rows at a time, so the training rows are
    never copied whole onto the heap."""

    def __init__(self, X: np.ndarray, rows: np.ndarray):
        self.X, self.rows = X, np.asarray(rows)
        self.batch_size = min(FE.STREAM_ROWS, FE.DATASET_BATCH_ROWS)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        # Native bin construction requires double sample rows. Pushed batches can
        # retain the matrix's float32 storage; every float32 sample converts exactly.
        dtype = np.float64 if np.isscalar(idx) else np.float32
        return np.asarray(self.X[self.rows[idx]], dtype=dtype)

    def __array__(self, *args, **kwargs):
        raise TypeError("A row adapter must use LightGBM Sequence, never whole-matrix conversion")


def _row_sequence(X, rows):
    # Lazy import preserves the optional dependency and avoids loading OpenMP at import.
    import lightgbm as lgb

    class Rows(_Rows, lgb.Sequence):
        pass

    return Rows(X, rows)


def _predict(booster, X: np.ndarray, rows: np.ndarray, stream: bool) -> np.ndarray:
    if not stream:
        return booster.predict(X[rows], num_iteration=booster.best_iteration)
    out = np.empty(len(rows))
    batch_size = min(FE.STREAM_ROWS, FE.DATASET_BATCH_ROWS)
    for i in range(0, len(rows), batch_size):
        j = rows[i:i + batch_size]
        out[i:i + len(j)] = booster.predict(np.asarray(X[j]), num_iteration=booster.best_iteration)
    return out


def _fit_predict(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray,
                 cfg: TrainConfig, stream: bool = False, stage=None, validation_predictions: bool = False,
                 y_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Fit on ``tr``, stop early on ``va``, score ``te``. ``stream`` feeds LightGBM from the
    matrix in batches (for a memmap bigger than RAM) instead of copying the rows.

    With ``cfg.seeds`` > 1 it fits that many boosters, LightGBM seeds ``seed``, ``seed + 1``,
    ..., one after another (so memory is one booster's), each stopped early on ``va``, and
    averages their probabilities. The validation AUROC is then the averaged validation
    predictions' (sklearn), the number a selection compares. One seed is the single booster
    exactly as before, its validation AUROC LightGBM's own at the best iteration."""
    if cfg.seeds < 1:
        raise ValueError("seeds must be at least 1")
    if cfg.seeds == 1:
        return _fit_one(X, y, tr, va, te, cfg, stream, stage, validation_predictions, y_train)
    from dataclasses import replace
    p, pv, gain, rounds, member_val = 0.0, 0.0, 0.0, [], []
    for i in range(cfg.seeds):
        fit = _fit_one(X, y, tr, va, te, replace(cfg, seed=cfg.seed + i, seeds=1), stream, stage,
                       validation_predictions=True, y_train=y_train)
        p, pv = p + fit["p"] / cfg.seeds, pv + fit["pv"] / cfg.seeds
        gain = gain + fit["gain"] / cfg.seeds
        rounds.append(fit["rounds"])
        member_val.append(fit["val_auroc"])
    bag = {"p": np.clip(p, 0.0, 1.0), "rounds": int(round(sum(rounds) / len(rounds))),
           "val_auroc": float(roc_auc_score(y[va], pv)),
           "members": {"rounds": rounds, "val_auroc": member_val}, "gain": gain}
    if validation_predictions:
        bag["pv"] = np.clip(pv, 0.0, 1.0)
    return bag


def _fit_one(X: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray,
             cfg: TrainConfig, stream: bool = False, stage=None,
             validation_predictions: bool = False, y_train: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """One booster. ``y_train`` (default ``y``) is the label it is trained on; it stops early
    on, and is scored with, ``y`` on the validation rows, so a target variant is still
    chosen and judged by YAIB's own label."""
    import lightgbm as lgb
    yt = y if y_train is None else y_train
    pos = max(float(yt[tr].sum()), 1.0)
    neg = max(float(len(tr) - yt[tr].sum()), 1.0)
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        # never ask for more rows per leaf than a small cohort can give
        "min_child_samples": int(min(cfg.min_child_samples, max(5, len(tr) // 50))),
        "feature_fraction": cfg.feature_fraction, "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": 1, "lambda_l2": cfg.lambda_l2,
        "num_threads": cfg.threads, "seed": cfg.seed, "deterministic": True, "verbose": -1,
        "force_col_wise": cfg.force_col_wise,
    }
    # set only when asked, so every earlier configuration passes LightGBM the same parameters
    if cfg.extra_trees:
        params["extra_trees"] = True
    if cfg.path_smooth:
        params["path_smooth"] = cfg.path_smooth
    if cfg.balanced:
        # YAIB trains with weight="balanced"; the same class balance here
        params["scale_pos_weight"] = neg / pos
    if stream:
        # no weights on a Sequence: LightGBM 4.x silently zeroes a weighted Sequence's
        # validation weights (AUC 1.0); class balance goes through scale_pos_weight only
        class SequenceDataset(lgb.Dataset):
            def _init_from_sample(self, sample_data, sample_indices, sample_cnt, total_nrow):
                result = super()._init_from_sample(sample_data, sample_indices, sample_cnt, total_nrow)
                # LightGBM 4.6 retains these lists while pushing the entire dataset,
                # although native bin creation has already consumed their values.
                # Releasing them now changes no bin or row and avoids ~1 GB overlap.
                sample_data.clear()
                sample_indices.clear()
                if stage is not None:
                    stage("native_bins_constructed")
                return result

            def construct(self):
                first = self._handle is None
                result = super().construct()
                if first and stage is not None:
                    stage("train_dataset_built" if self.reference is None else "validation_dataset_built")
                return result

            def get_params(self):
                # Sequence construction calls this method for its native sample path.
                # LightGBM's default filter omits min_child_samples and num_threads,
                # unlike dense/CSR construction. Preserve the exact supplied settings.
                return dict(self.params or {})

        dtr = SequenceDataset(_row_sequence(X, tr), label=yt[tr], params=params, free_raw_data=True)
        dva = SequenceDataset(_row_sequence(X, va), label=y[va], params=params, reference=dtr, free_raw_data=True)
    else:
        dtr = lgb.Dataset(X[tr], yt[tr], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr, free_raw_data=True)
    booster = lgb.train(params, dtr, cfg.max_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])
    try:
        if stage is not None:
            stage("booster_fit_completed")
        fit = {"p": _predict(booster, X, te, stream), "rounds": int(booster.best_iteration),
               "val_auroc": float(booster.best_score["valid_0"]["auc"]),
               "gain": booster.feature_importance("gain", iteration=booster.best_iteration)}
        if validation_predictions:
            fit["pv"] = _predict(booster, X, va, stream)
        return fit
    finally:
        booster.free_dataset()


def training_target(label: np.ndarray, stay: np.ndarray, hour: np.ndarray, target: str) -> np.ndarray:
    """The label a member is trained on, from YAIB's hourly label (NaN = unlabelled, kept NaN).

    yaib    YAIB's label as it is
    early6  also positive in the 6 hours before the stay's first positive hour: a member that
            learns earlier signs, which a blend can use (the CinC lane's early targets)
    stay    every labelled hour of a stay that is ever positive (who will become septic)

    It reads future labels of the same stay, which a TRAINING target may: no feature does,
    and validation and test are scored on YAIB's label only."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    y = np.asarray(label, float)
    if target == "yaib":
        return y.copy()
    s = pd.Series(np.nan_to_num(y, nan=0.0) > 0)
    if target == "stay":
        ever = s.groupby(stay, sort=False).transform("max").to_numpy()
        return np.where(np.isfinite(y), ever.astype(float), np.nan)
    first = pd.Series(np.where(s, hour, np.inf)).groupby(stay, sort=False).transform("min").to_numpy()
    early = np.isfinite(first) & (hour >= first - 6)
    return np.where(np.isfinite(y), np.maximum(np.nan_to_num(y, nan=0.0), early.astype(float)), np.nan)


def train_stays(train: np.ndarray, stay_labels: pd.Series, fraction: float, split: str) -> np.ndarray:
    """A deterministic share of a run's training stays, stratified by stay label, for cheap
    screening fits. ``fraction`` 1 is every training stay. The draw depends only on the
    split's fingerprint and the fraction, so a candidate and the reference screened on the
    same run see the same stays; validation and test stays are never touched."""
    train = np.sort(np.asarray(train))
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if fraction == 1:
        return train
    import hashlib
    seed = int(hashlib.sha256(f"{split}:{fraction!r}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    septic = stay_labels.reindex(train).to_numpy() > 0
    keep = []
    for group in (train[septic], train[~septic]):
        k = max(1, int(round(fraction * len(group)))) if len(group) else 0
        keep.append(rng.choice(group, size=k, replace=False) if k else group[:0])
    return np.sort(np.concatenate(keep))


def yaib_summary(name: str, v: np.ndarray) -> Dict[str, Any]:
    """Summarise per-run values as ``run_utils.aggregate_results`` does: the mean, the
    field YAIB labels "std" (population SD / sqrt(n): a standard error, not an SD), and a
    t interval. The plain SD over runs is kept too, under its own name."""
    from scipy import stats
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return {f"{name}_mean": float(v.mean()) if n else math.nan}
    m, sd = float(v.mean()), float(v.std())
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=stats.sem(v))
    return {f"{name}_mean": m, f"{name}_yaib_std_field": sd / math.sqrt(n), f"{name}_sd_over_runs": sd,
            f"{name}_ci95": [float(lo), float(hi)]}


def _metric(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auroc": math.nan, "auprc": math.nan}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


def column_groups(feature_set: str, width: int, families: Sequence[str]) -> List[tuple]:
    """(group, first column, end) for the gain shares: the feature set's own layers (base,
    then what rich and trend append), then each family's block."""
    groups, start = [], 0
    for layer in ("base", "rich", "trend"):
        end = min(len(FE.feature_names(layer)), width)
        if end > start:
            groups.append((layer, start, end))
            start = end
        if layer == feature_set:
            break
    for f in families:
        end = start + len(FAM.family_names(f))
        groups.append((f, start, end))
        start = end
    return groups


def gain_shares(gain: np.ndarray, groups: List[tuple]) -> Dict[str, float]:
    """Each column group's share of the booster's total split gain (an aggregate, no rows)."""
    total = float(np.sum(gain))
    if not total > 0:
        return {}
    return {g: round(float(np.sum(gain[a:b])) / total, 4) for g, a, b in groups}


def _write_validation(path: str, stay, hour, y, pv) -> None:
    """One run's validation predictions, beside the checkpoints (rows: inside the lake)."""
    tmp = path + ".tmp"
    pd.DataFrame({"stay_id": stay, "hour": hour, "label": y, "p": pv}).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run(cohort: Cohort, cfg: TrainConfig = TrainConfig(), cv_repetitions: int = 5, cv_folds: int = 5,
        seed: int = SP.YAIB_CLI_SEED, runs: Optional[Sequence[tuple]] = None,
        predictions_dir: Optional[str] = None, features_path: Optional[str] = None,
        memory_check: bool = True, feature_set: str = "base",
        resume_dir: Optional[str] = None, feature_store: Optional[str] = None,
        validation_only: bool = False, matrix_from: Optional[str] = None,
        families: Optional[Sequence[str]] = None, validation_dir: Optional[str] = None) -> Dict[str, Any]:
    """Every (repetition, fold) of YAIB's protocol. ``runs`` limits it to some pairs.
    ``predictions_dir`` writes each run's test predictions (rows: keep them in the lake).
    ``features_path`` keeps the feature matrix in a disk-backed memmap (also in the lake)
    and then streams it to LightGBM in batches, so no copy of the rows is held in RAM.
    Each run also records its validation AUROC: choose between configurations on that,
    never on the test score.
    ``feature_store`` (with ``resume_dir``) keeps the matrix in a store shared by every
    configuration with the same inputs, feature set and feature code, so a sweep builds
    it once. ``validation_only`` leaves every test metric out of the result: a
    development run then cannot show the number it must not be selected on.
    A run whose planned peak exceeds this host's available memory is refused up front,
    instead of swapping for hours or being killed mid-protocol.
    ``matrix_from`` reads ``feature_set`` as the leading columns of a wider set's matrix
    (``rich`` is the first 695 columns of ``trend``, unchanged and in order), so a sweep
    holding the wider matrix never builds the narrower one. The columns are a view of the
    memmap; LightGBM still reads it in batches, so nothing is copied whole.
    ``families`` appends feature-family blocks (``families.py``), each built once into the
    feature store and read beside the matrix, never merged into a new one.
    ``validation_dir`` writes each run's validation predictions there (rows: inside the
    lake), for ensembles and evidence chosen on validation without refitting."""
    if resume_dir is not None and features_path is not None:
        raise ValueError("Use --resume-dir or --features, not both")
    if feature_store is not None and resume_dir is None:
        raise ValueError("--feature-store needs --resume-dir")
    families = FAM.canonical(families)
    source = matrix_from or feature_set
    width = len(FE.feature_names(feature_set))
    if matrix_from is not None and (matrix_from == feature_set
                                    or FE.feature_names(matrix_from)[:width] != FE.feature_names(feature_set)):
        raise ValueError(f"{feature_set} is not the leading columns of {matrix_from}")
    shared = None
    if resume_dir is not None:
        from backend.ml.yaib.checkpoint import Checkpoints, FeatureStore, experiment_spec
        protocol = {"cv_repetitions": cv_repetitions, "cv_folds": cv_folds, "seed": seed}
        spec = experiment_spec(cohort, asdict(cfg), protocol, feature_set)
        if matrix_from is not None:
            spec["matrix_from"] = matrix_from
        if families:
            spec["families"] = families
        store = Checkpoints(resume_dir, spec)
        if feature_store is not None:
            # the store is the wider set's: the same matrix a run of that set shares
            shared = FeatureStore(feature_store, experiment_spec(cohort, asdict(cfg), protocol, source))
        context = store.locked()
    else:
        context = contextlib.nullcontext(None)
    with context as checkpoints:
        t0 = time.time()
        if memory_check:
            plan = FE.memory_plan(len(cohort.dyn), on_disk=features_path is not None or resume_dir is not None, feature_set=source, cohort_loaded=True)
            extra = sum(len(FAM.family_names(f)) for f in families)
            if extra and plan["available_gb"] is not None:
                # family blocks are binned and sampled like the matrix's own columns
                rows = len(cohort.dyn)
                per_col = int(rows * 0.64 * 1.25) + min(int(rows * 0.64), 200000) * 20
                plan["additional_heap_gb"] += extra * per_col / 1e9
                if plan["additional_heap_gb"] >= 0.85 * plan["available_gb"]:
                    plan["fits"] = False
            if plan["fits"] is False:
                raise MemoryError(f"planned peak {plan['peak_heap_gb']:.1f} GB, additional "
                                  f"{plan['additional_heap_gb']:.1f} GB, free {plan['available_gb']:.1f} GB, "
                                  f"cgroup cap {plan['cgroup_limit_gb']}; memory safety margin exceeded")
        if checkpoints:
            X = checkpoints.features((len(cohort.dyn), len(FE.feature_names(source))),
                                     lambda path: FE.build(cohort, out=path, feature_set=source)[0],
                                     store=shared)
        else:
            X, _ = FE.build(cohort, out=features_path, feature_set=source)
        names = FE.feature_names(feature_set)
        if X.shape[1] != width:
            X = X[:, :width]          # a view: the memmap is not read here
        if families:
            blocks = [X]
            for f in families:
                fnames = FAM.family_names(f)
                if checkpoints and feature_store is not None:
                    from backend.ml.yaib.checkpoint import FamilyStore
                    blocks.append(FamilyStore(feature_store, spec, f).matrix(
                        (len(cohort.dyn), len(fnames)), lambda path, f=f: FAM.build_family(cohort, f, out=path)[0]))
                else:
                    out = f"{features_path}.{f}.npy" if features_path else None
                    blocks.append(FAM.build_family(cohort, f, out=out)[0])
                names = names + fnames
            X = FAM.Columns(blocks)
        groups = column_groups(feature_set, width, families)
        stream = features_path is not None or checkpoints is not None
        y_all = cohort.label
        stay = cohort.dyn[GROUP].to_numpy().copy()
        hour = cohort.dyn["hour"].to_numpy().copy()
        labels = cohort.stay_labels()
        if stream:
            # the features are on disk now; the cohort's variables are not needed again, and on
            # a full cohort they are gigabytes. The caller's cohort keeps its ids, hours, labels.
            cohort.dyn = cohort.dyn[[GROUP, "hour"]]
            import gc
            gc.collect()
        y_train = None
        if cfg.target != "yaib":
            y_train = np.nan_to_num(training_target(y_all, stay, hour, cfg.target), nan=0.0)
        per_run: List[Dict[str, Any]] = []
        for sp in SP.all_splits(labels, cv_repetitions, cv_folds, seed):
            if runs is not None and (sp.repetition, sp.fold) not in runs:
                continue
            lab = np.isfinite(y_all)
            fitted = train_stays(sp.train, labels, cfg.train_fraction, sp.fingerprint())
            tr = np.flatnonzero(np.isin(stay, fitted) & lab)
            va = np.flatnonzero(np.isin(stay, sp.val) & lab)
            te = np.flatnonzero(np.isin(stay, sp.test) & lab)
            y = np.nan_to_num(y_all, nan=0.0)
            vpath = (os.path.join(validation_dir, f"val_r{sp.repetition}_f{sp.fold}.parquet")
                     if validation_dir else None)
            fit = checkpoints.load_fit(sp.repetition, sp.fold, sp.fingerprint(), len(te)) if checkpoints else None
            if fit is None or (vpath and not os.path.exists(vpath)):
                extra = {}
                if y_train is not None:
                    extra["y_train"] = y_train
                if vpath is not None:
                    extra["validation_predictions"] = True
                fresh = _fit_predict(X, y, tr, va, te, cfg, stream=stream, **extra)
                if fit is None:
                    fit = fresh
                    if checkpoints:
                        checkpoints.save_fit(sp.repetition, sp.fold, sp.fingerprint(), fit, len(te))
                elif not np.array_equal(fresh["p"], fit["p"]):
                    # a resumed run asked for its validation predictions: the deterministic
                    # refit must be the checkpointed model, or the two would not be one model
                    raise ValueError("the refit for validation predictions differs from its checkpoint")
                else:
                    fit = {**fit, "pv": fresh["pv"], **({"gain": fresh["gain"]} if "gain" in fresh else {})}
            if vpath:
                os.makedirs(validation_dir, mode=0o700, exist_ok=True)
                _write_validation(vpath, stay[va], hour[va], y[va], fit["pv"])
            m = {} if validation_only else _metric(y[te], fit["p"])
            per_run.append({"repetition": sp.repetition, "fold": sp.fold, "split": sp.fingerprint(),
                            "rounds": fit["rounds"], "val_auroc": fit["val_auroc"], "test_rows": int(len(te)),
                            "test_stays": int(len(sp.test)), "train_stays": int(len(fitted)), **m,
                            **({"members": fit["members"]} if "members" in fit else {}),
                            **({"gain_share": gain_shares(fit["gain"], groups)} if "gain" in fit else {})})
            if predictions_dir:
                os.makedirs(predictions_dir, exist_ok=True)
                pd.DataFrame({"stay_id": stay[te], "hour": hour[te], "label": y[te], "p": fit["p"]}).to_parquet(
                    os.path.join(predictions_dir, f"ours_r{sp.repetition}_f{sp.fold}.parquet"), index=False)
        va_au = np.array([r["val_auroc"] for r in per_run], float)
        tested = {}
        if not validation_only:
            au = np.array([r["auroc"] for r in per_run], float)
            pr = np.array([r["auprc"] for r in per_run], float)
            tested = {"yaib_metric": {**yaib_summary("auroc", au), **yaib_summary("auprc", pr),
                                      "n_runs": int(np.isfinite(au).sum())}}
        return {
            "schema": "yaib_bench/1",
            "cohort": {"name": cohort.name, "stays": int(len(labels)), "septic_stays": int(labels.sum()),
                       "rows": int(len(stay)), "labelled_rows": int(np.isfinite(y_all).sum())},
            "protocol": {"cv_repetitions": cv_repetitions, "cv_folds": cv_folds, "seed": seed,
                         "split_source": "YAIB make_single_split_polars, reproduced (backend/ml/yaib/splits.py)"},
            "model": {"learner": "LightGBM", **asdict(cfg), "feature_set": feature_set, "n_features": len(names),
                      "streamed_from_memmap": stream, **({"matrix_from": matrix_from} if matrix_from else {}),
                      **({"families": families} if families else {})},
            "validation_only": validation_only,
            "runs": per_run,
            **tested,
            "validation_metric": yaib_summary("val_auroc", va_au),
            "seconds": round(time.time() - t0, 1),
        }


# --------------------------------------------------------------------------- #
# paired comparison with YAIB's own saved predictions
# --------------------------------------------------------------------------- #
def read_yaib_predictions(path: str) -> pd.DataFrame:
    """One run's ``pred_indicators.csv``: stay id, hour, label, p(class 0), p(class 1)."""
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    if a.shape[1] != 5:
        raise ValueError(f"{path}: expected 5 columns (id, hour, label, p0, p1), got {a.shape[1]}")
    return pd.DataFrame({"stay_id": a[:, 0].astype(np.int64), "hour": a[:, 1], "label": a[:, 2], "p": a[:, 4]})


#: YAIB's polars loader writes the time column as it is held ("already in hours?" in its
#: source), so the saved unit depends on the cohort's dtype. Seconds through nanoseconds
#: are tried; a unit is accepted only if it makes the two row sets identical.
_UNITS = (1.0, 3600.0, 3.6e6, 3.6e9, 3.6e12)


def _hours_like(theirs: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    want = set(zip(ours["stay_id"].to_numpy(), np.round(ours["hour"].to_numpy(), 6)))
    for u in _UNITS:
        h = np.round(theirs["hour"].to_numpy() / u, 6)
        if set(zip(theirs["stay_id"].to_numpy(), h)) == want:
            return theirs.assign(hour=h)
    return theirs


def _join(ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
    """Pair our rows with the baseline's.

    YAIB's saved labels and predictions are in (stay, hour) order: its loader sorts the
    feature rows, and the labels come out in the same order. The stay/hour ids written
    beside them are NOT in that order (the polars loader takes them before the sort;
    seen on YAIB af9a5b3, eICU demo: 324 of 7,970 rows carried another hour's label when
    joined on the written ids). So rows are paired by sorted position, and that is
    accepted only if the two id sets are identical AND every label agrees position by
    position. Anything else is refused.
    """
    ours = ours.assign(hour=np.round(ours["hour"].to_numpy(), 6))
    theirs = _hours_like(theirs, ours)
    a = ours.sort_values(["stay_id", "hour"], kind="mergesort").reset_index(drop=True)
    ka = pd.MultiIndex.from_arrays([a["stay_id"], a["hour"]])
    kb = pd.MultiIndex.from_arrays([theirs["stay_id"], theirs["hour"]])
    if len(a) != len(theirs) or not ka.sort_values().equals(kb.sort_values()):
        only_a, only_b = len(ka.difference(kb)), len(kb.difference(ka))
        raise ValueError(f"rows differ: {only_a} only ours, {only_b} only the baseline's "
                         f"({len(a)} vs {len(theirs)} rows)")
    if not np.array_equal(a["label"].to_numpy(), theirs["label"].to_numpy()):
        raise ValueError("labels differ position by position: the baseline's rows are not in "
                         "(stay, hour) order, so they cannot be paired")
    return pd.DataFrame({"stay_id": a["stay_id"], "hour": a["hour"], "label_ours": a["label"],
                         "p_ours": a["p"], "p_base": theirs["p"].to_numpy()})


def compare(ours: Dict[tuple, pd.DataFrame], theirs: Dict[tuple, pd.DataFrame], n_boot: int = 1000,
            seed: int = 0) -> Dict[str, Any]:
    """Paired AUROC difference (ours - baseline), bootstrapping test stays per repetition."""
    keys = sorted(set(ours) & set(theirs))
    if not keys or set(ours) != set(theirs):
        raise ValueError(f"runs differ: ours {sorted(ours)}, baseline {sorted(theirs)}")
    by_rep: Dict[int, List[pd.DataFrame]] = {}
    for k in keys:
        by_rep.setdefault(k[0], []).append(_join(ours[k], theirs[k]))
    rng = np.random.default_rng(seed)
    point, draws = [], np.zeros(n_boot)
    for r, frames in sorted(by_rep.items()):
        stays = np.unique(frames[0]["stay_id"].to_numpy())
        codes = [np.searchsorted(stays, f["stay_id"].to_numpy()) for f in frames]
        point.append(np.mean([roc_auc_score(f["label_ours"], f["p_ours"]) - roc_auc_score(f["label_ours"], f["p_base"])
                              for f in frames]))
        for b in range(n_boot):
            m = np.bincount(rng.integers(0, len(stays), len(stays)), minlength=len(stays)).astype(float)
            d = []
            for f, c in zip(frames, codes):
                w = m[c]
                yy = f["label_ours"].to_numpy()
                if w.sum() == 0 or len(np.unique(yy[w > 0])) < 2:
                    continue
                d.append(roc_auc_score(yy, f["p_ours"], sample_weight=w)
                         - roc_auc_score(yy, f["p_base"], sample_weight=w))
            draws[b] += np.mean(d) if d else np.nan
    draws /= len(by_rep)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    delta = float(np.mean(point))
    return {"auroc_delta": delta, "ci_low": float(lo), "ci_high": float(hi), "repetitions": len(by_rep),
            "runs": len(keys), "better": bool(lo > 0), "worse": bool(hi < 0)}


def load_runs(pattern_ours: str, yaib_root: str) -> tuple:
    """Our ``ours_r{r}_f{f}.parquet`` files and YAIB's ``repetition_{r}/fold_{f}/pred_indicators.csv``."""
    ours, theirs = {}, {}
    for f in glob.glob(pattern_ours):
        b = os.path.basename(f)[len("ours_"):-len(".parquet")]
        r, k = b.split("_")
        ours[(int(r[1:]), int(k[1:]))] = pd.read_parquet(f)
    for f in glob.glob(os.path.join(yaib_root, "**", "pred_indicators.csv"), recursive=True):
        parts = os.path.normpath(f).split(os.sep)
        rep = next((p for p in parts if p.startswith("repetition_")), None)
        fold = next((p for p in parts if p.startswith("fold_")), None)
        if rep and fold:
            theirs[(int(rep.split("_")[1]), int(fold.split("_")[1]))] = read_yaib_predictions(f)
    return ours, theirs


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    from backend.ml.jsonio import json_safe
    from backend.ml.yaib.cohort import load

    ap = argparse.ArgumentParser(description="Our model under YAIB's protocol (aggregates only).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="5x5 runs on a YAIB cohort directory")
    r.add_argument("cohort")
    r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=SP.YAIB_CLI_SEED, help="the seed the YAIB baselines ran with")
    r.add_argument("--repetitions", type=int, default=5)
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--predictions", help="directory for per-run test predictions (inside the lake)")
    r.add_argument("--features", help="memmap the feature matrix to this .npy path (inside the lake)")
    r.add_argument("--resume-dir", help="private immutable feature/fit checkpoints; rerun with the same settings")
    r.add_argument("--feature-set", choices=FE.FEATURE_SETS, default="base")
    r.add_argument("--runs", help="only these runs, e.g. 0:0,0:1 (development; select on validation AUROC)")
    r.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    r.add_argument("--num-leaves", type=int, default=TrainConfig.num_leaves)
    r.add_argument("--min-child-samples", type=int, default=TrainConfig.min_child_samples)
    r.add_argument("--feature-fraction", type=float, default=TrainConfig.feature_fraction)
    r.add_argument("--unbalanced", action="store_true", help="no scale_pos_weight")
    r.add_argument("--bagging-fraction", type=float, default=TrainConfig.bagging_fraction)
    r.add_argument("--lambda-l2", type=float, default=TrainConfig.lambda_l2)
    r.add_argument("--max-rounds", type=int, default=TrainConfig.max_rounds)
    r.add_argument("--early-stopping", type=int, default=TrainConfig.early_stopping)
    r.add_argument("--feature-store", help="private store shared by configurations (with --resume-dir)")
    r.add_argument("--extra-trees", action="store_true", help="LightGBM extra_trees")
    r.add_argument("--path-smooth", type=float, default=TrainConfig.path_smooth, help="LightGBM path_smooth")
    r.add_argument("--validation-only", action="store_true",
                   help="development: report validation AUROC only, no test metric")
    r.add_argument("--seeds", type=int, default=TrainConfig.seeds, help="average this many LightGBM seeds")
    r.add_argument("--model-seed", type=int, default=TrainConfig.seed, help="LightGBM's (first) seed")
    r.add_argument("--matrix-from", choices=FE.FEATURE_SETS,
                   help="read the feature set as the leading columns of this wider set's matrix")
    r.add_argument("--families", help="comma-separated feature families appended as blocks: "
                   + ",".join(FAM.FAMILIES))
    r.add_argument("--save-validation", help="directory for per-run validation predictions (inside the lake)")
    r.add_argument("--target", choices=TARGETS, default=TrainConfig.target,
                   help="the label trees are trained on (validation and test stay YAIB's)")
    r.add_argument("--train-fraction", type=float, default=TrainConfig.train_fraction,
                   help="screening: fit on this share of each run's training stays (stratified)")
    f = sub.add_parser("adopt-features", help="share a checkpoint's committed matrix through a feature store")
    f.add_argument("cohort")
    f.add_argument("--from", dest="source", required=True, help="an experiment checkpoint directory")
    f.add_argument("--feature-store", required=True)
    f.add_argument("--feature-set", choices=FE.FEATURE_SETS, default="base")
    p = sub.add_parser("plan", help="planned peak memory for a cohort, without building anything")
    p.add_argument("cohort")
    p.add_argument("--features", action="store_true", help="plan with the matrix memmapped to disk")
    p.add_argument("--feature-set", choices=FE.FEATURE_SETS, default="base")
    c = sub.add_parser("compare", help="paired AUROC vs YAIB's saved baseline predictions")
    c.add_argument("--ours", required=True, help="directory of ours_r*_f*.parquet")
    c.add_argument("--yaib", required=True, help="YAIB log directory of one baseline model")
    c.add_argument("--out", required=True)
    c.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)

    if a.cmd == "plan":
        import pyarrow.parquet as pq
        rows = pq.ParquetFile(os.path.join(a.cohort, "dyn.parquet")).metadata.num_rows
        print(json.dumps(FE.memory_plan(rows, on_disk=a.features, feature_set=a.feature_set)))
        return 0
    if a.cmd == "adopt-features":
        from backend.ml.yaib.checkpoint import FeatureStore, experiment_spec
        spec = experiment_spec(load(a.cohort), {}, {}, a.feature_set)
        store = FeatureStore(a.feature_store, spec)
        saved = store.adopt(a.source)
        print(json.dumps({"store": store.identity[:16], "sha256": saved["sha256"]}))
        return 0
    if a.cmd == "run":
        cfg = TrainConfig(learning_rate=a.learning_rate, num_leaves=a.num_leaves,
                          min_child_samples=a.min_child_samples, feature_fraction=a.feature_fraction,
                          bagging_fraction=a.bagging_fraction, lambda_l2=a.lambda_l2,
                          max_rounds=a.max_rounds, early_stopping=a.early_stopping,
                          threads=a.threads, balanced=not a.unbalanced,
                          extra_trees=a.extra_trees, path_smooth=a.path_smooth,
                          seeds=a.seeds, seed=a.model_seed, train_fraction=a.train_fraction,
                          target=a.target)
        runs = None if not a.runs else [tuple(int(x) for x in r.split(":")) for r in a.runs.split(",")]
        cohort = load(a.cohort)
        res = run(cohort, cfg, a.repetitions, a.folds, a.seed, runs=runs,
                  predictions_dir=a.predictions, features_path=a.features, feature_set=a.feature_set,
                  resume_dir=a.resume_dir, feature_store=a.feature_store, validation_only=a.validation_only,
                  matrix_from=a.matrix_from, families=a.families.split(",") if a.families else None,
                  validation_dir=a.save_validation)
        m, v = res.get("yaib_metric"), res["validation_metric"]
        print(f"validation AUROC {v.get('val_auroc_mean', float('nan')):.4f} over {len(res['runs'])} runs")
        if runs is None and m is not None:
            print(f"AUROC {m['auroc_mean']:.4f} ± {m['auroc_yaib_std_field']:.4f}  AUPRC {m['auprc_mean']:.4f} "
                  f"± {m['auprc_yaib_std_field']:.4f}  over {m['n_runs']} runs (± as YAIB reports it)")
    else:
        ours, theirs = load_runs(os.path.join(a.ours, "ours_r*_f*.parquet"), a.yaib)
        res = compare(ours, theirs, a.n_boot)
        print(f"AUROC ours - baseline {res['auroc_delta']:+.4f} [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]")
    with open(a.out, "w") as fh:
        json.dump(json_safe(res), fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
