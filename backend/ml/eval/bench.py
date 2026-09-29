"""Fixed-fold benchmark harness — the referee for every detector technology.

Every accuracy claim in this project has to survive the same test: patient-grouped
cross-validation on real CinC 2019 patients, with the *identical* folds for every
candidate so a comparison is paired rather than a race between different splits.
This module is that referee.

Why it exists as its own module: previous accuracy pushes each rolled their own
split, which makes two results incomparable and makes fold luck look like progress
(see the +79% AUPRC that evaporated under CV — ``models/augmented.py``). Here a
candidate is a *function* ``(train_idx, test_idx) -> p_test``; the harness owns the
data, the folds, the metrics, and the significance test.

    bench = Benchmark.load("dev")             # 8k patients, fast iteration
    res   = bench.run("gbm", fit_predict_gbm) # scored on the fixed folds
    bench.compare(res, baseline)              # paired, patient-bootstrapped

The significance test resamples *patients* (not rows): rows within a patient are
massively correlated, so a row bootstrap would report a confidence interval several
times too narrow and turn noise into a discovery.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.ml.data.features import FeatureMatrix
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_folds
from backend.ml.eval.utility import best_threshold_utility, utility_terms

#: Named dataset specs. ``dev`` is the screening set (fast); ``full`` is the whole
#: challenge cohort (both hospitals) used to confirm anything that screens well.
SPECS: Dict[str, List[str]] = {
    "dev": ["wearable_limit8000"],
    "setA": ["wearable_training_setA_limitall"],
    "setB": ["wearable_training_setB_limitall"],
    "full": ["wearable_training_setA_limitall", "wearable_training_setB_limitall"],
}

#: A candidate: given row indices to train on and row indices to score, return
#: P(sepsis) for the scored rows. Receives the whole dataset so it can use raw
#: features, sequences, groups, or hour indices as it likes.
Candidate = Callable[[FeatureMatrix, np.ndarray, np.ndarray], np.ndarray]


#: Every CinC patient except the 20% GENESIS locked before any architecture work (its
#: one untouched test set). The shipped detector is trained on these.
GENESIS_DEV = "wearable_genesis_dev"


def genesis_dev() -> FeatureMatrix:
    """The ``full`` cohort minus GENESIS's locked patients, built once and cached.

    Filtered from the ``full`` shards rather than re-featurised, so its rows are the
    ones every other result here was measured on. The lock is drawn with its own seed
    before the dev folds, so it does not depend on the fold arguments below."""
    import os
    from backend.ml import config
    path = os.path.join(config.ARTIFACTS_DIR, "feature_cache", f"{GENESIS_DEV}.npz")
    if not os.path.exists(path):
        from backend.ml.genesis import harness, rawstore
        raw = rawstore.load()
        locked = raw.pid[harness.patient_folds(raw, 5, 20260923) < 0]
        full = concat([_load_cached(k) for k in SPECS["full"]])
        keep = ~np.isin(full.groups, locked)
        # If the two id formats ever diverge, isin matches nothing and every locked
        # patient walks into the training set; so every one of them must be found.
        if len(np.unique(full.groups[~keep])) != len(locked):
            raise AssertionError("not every locked GENESIS patient was found in the cohort")
        np.savez(path, X=full.X[keep], y=full.y[keep], groups=full.groups[keep],
                 hour_index=full.hour_index[keep], onset_hour=full.onset_hour[keep],
                 feature_names=np.array(full.feature_names))
    return _load_cached(GENESIS_DEV)


def _load_cached(key: str) -> FeatureMatrix:
    import os
    from backend.ml import config
    path = os.path.join(config.ARTIFACTS_DIR, "feature_cache", f"{key}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing feature cache {path}; build it with build_dataset_cached first")
    d = np.load(path, allow_pickle=True)
    return FeatureMatrix(
        X=d["X"], y=d["y"], groups=d["groups"], feature_names=list(d["feature_names"]),
        hour_index=d["hour_index"], onset_hour=d["onset_hour"])


def concat(parts: Sequence[FeatureMatrix]) -> FeatureMatrix:
    """Join cached shards. Patient ids are globally unique in CinC, so grouping
    (and therefore the leakage guarantee) survives concatenation untouched."""
    if len(parts) == 1:
        return parts[0]
    names = parts[0].feature_names
    for p in parts[1:]:
        if p.feature_names != names:
            raise ValueError("cannot concatenate shards with different feature names")
    return FeatureMatrix(
        X=np.concatenate([p.X for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        feature_names=list(names),
        hour_index=np.concatenate([p.hour_index for p in parts]),
        onset_hour=np.concatenate([p.onset_hour for p in parts]),
    )


@dataclass
class BenchResult:
    """One candidate's scores. ``oof`` is the pooled out-of-fold prediction for
    every row — each row scored by a model that never saw that patient — which is
    what the paired significance test consumes."""
    name: str
    per_fold: List[Dict[str, float]]
    oof: np.ndarray
    seconds: float

    def mean(self, key: str) -> float:
        return float(np.mean([f[key] for f in self.per_fold]))

    def std(self, key: str) -> float:
        return float(np.std([f[key] for f in self.per_fold]))

    def line(self) -> str:
        return (f"{self.name:<22} AUROC {self.mean('auroc'):.4f}±{self.std('auroc'):.4f}  "
                f"AUPRC {self.mean('auprc'):.4f}  util {self.mean('utility'):+.4f}  "
                f"util* {self.mean('utility_best'):+.4f}  ECE {self.mean('ece'):.4f}  "
                f"detect@FA15 {self.mean('detect_at_far15'):.3f}  "
                f"lead@FA15 {self.mean('lead_at_far15'):.1f}h  "
                f"[detect {self.mean('detection'):.3f} FA {self.mean('false_alarm'):.3f}]  "
                f"[{self.seconds:.0f}s]")


@dataclass
class Benchmark:
    ds: FeatureMatrix
    folds: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    results: Dict[str, BenchResult] = field(default_factory=dict)
    _terms_cache: Dict[object, Dict[str, np.ndarray]] = field(default_factory=dict, repr=False)
    _summary: Optional[object] = field(default=None, repr=False)

    @staticmethod
    def load(spec: str = "dev", n_folds: int = 4, seed: int = 42) -> "Benchmark":
        ds = concat([_load_cached(k) for k in SPECS[spec]])
        folds = list(stratified_group_folds(ds.groups, ds.y, n_folds, seed))
        return Benchmark(ds=ds, folds=folds)

    def describe(self) -> str:
        pts = len(set(self.ds.groups.tolist()))
        return (f"{len(self.ds.y):,} rows · {pts:,} patients · "
                f"hour-prevalence {self.ds.y.mean():.4f} · {len(self.folds)} folds")

    # ------------------------------------------------------------------ #
    def run(self, name: str, candidate: Candidate, verbose: bool = True) -> BenchResult:
        t0 = time.time()
        oof = np.full(len(self.ds.y), np.nan)
        per_fold: List[Dict[str, float]] = []
        for k, (tr, te) in enumerate(self.folds):
            p = np.asarray(candidate(self.ds, tr, te), dtype=float)
            if p.shape != te.shape:
                raise ValueError(f"{name}: returned {p.shape} predictions for {te.shape} rows")
            oof[te] = p
            per_fold.append(self._score(te, p))
            if verbose:
                print(f"    fold {k}: AUROC {per_fold[-1]['auroc']:.4f} "
                      f"util {per_fold[-1]['utility']:+.4f}", flush=True)
        res = BenchResult(name=name, per_fold=per_fold, oof=oof, seconds=time.time() - t0)
        self.results[name] = res
        if verbose:
            print("  " + res.line(), flush=True)
        return res

    def _terms(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        # Hash the whole index array: keying on (first, last, size) would silently
        # serve one fold's utility terms to a different fold that happened to share
        # its endpoints, and the resulting numbers would look perfectly plausible.
        key = hash(np.ascontiguousarray(idx).tobytes())
        cached = self._terms_cache.get(key)
        if cached is None:
            d = self.ds
            cached = utility_terms(d.y[idx], d.groups[idx], d.hour_index[idx], d.onset_hour[idx])
            self._terms_cache[key] = cached
        return cached

    def _score(self, idx: np.ndarray, p: np.ndarray) -> Dict[str, float]:
        d = self.ds
        rep = metrics.evaluate(d.y[idx], p, d.groups[idx], d.hour_index[idx], d.onset_hour[idx])
        ew = rep.early_warning or {}
        # Best-threshold utility is ORACLE-thresholded (chosen on the rows it scores),
        # so it is a ceiling, not a deployable number. It is here only so a candidate
        # is not penalised for suiting the shared spec>=0.90 alarm policy badly; the
        # shipped operating point is always derived out-of-fold elsewhere.
        u_best, t_best = best_threshold_utility(d.y[idx], p, self._terms(idx))
        # The comparison that matters clinically: at a FIXED alarm budget, how many
        # septic patients does this detector catch and how early?
        far = metrics.detection_at_false_alarm_rate(
            d.y[idx], p, d.groups[idx], d.onset_hour[idx], d.hour_index[idx], 0.15)
        return {
            "detect_at_far15": far["detection_rate"],
            "lead_at_far15": far["median_lead_time_hours"],
            "auroc": rep.auroc, "auprc": rep.auprc, "ece": rep.ece,
            "brier": rep.brier, "utility": rep.normalized_utility,
            "utility_best": u_best, "thr_best": t_best,
            "sens_at_spec90": rep.operating_points["sens@spec>=0.90"]["sensitivity"],
            "detection": ew.get("detection_rate", float("nan")),
            "false_alarm": ew.get("false_alarm_rate", float("nan")),
            "lead_h": ew.get("median_lead_time_hours", float("nan")),
        }

    # ------------------------------------------------------------------ #
    def compare(self, cand: BenchResult, base: BenchResult, n_boot: int = 400,
                seed: int = 0) -> Dict[str, float]:
        """Paired patient-level bootstrap of the pooled-OOF AUROC difference.

        Patients — not rows — are the resampling unit: hours within one stay are
        so correlated that a row bootstrap understates the interval several-fold
        and would let fold luck pass as a finding. Both candidates are scored on
        the *same* resampled patients each draw, so the shared difficulty of a
        draw cancels and only the model difference is measured.
        """
        d = self.ds
        pts = np.array(sorted(set(d.groups.tolist())))
        idx_by_pt = metrics._rows_by_patient(d.groups, pts)
        rng = np.random.default_rng(seed)
        deltas = np.empty(n_boot)
        for b in range(n_boot):
            draw = rng.choice(len(pts), size=len(pts), replace=True)
            rows = np.concatenate([idx_by_pt[pts[i]] for i in draw])
            yb = d.y[rows]
            if yb.min() == yb.max():
                deltas[b] = 0.0
                continue
            deltas[b] = metrics.auroc(yb, cand.oof[rows]) - metrics.auroc(yb, base.oof[rows])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        point = metrics.auroc(d.y, cand.oof) - metrics.auroc(d.y, base.oof)
        return {"delta_auroc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                "p_gt_0": float((deltas > 0).mean()), "significant": bool(lo > 0)}

    # ------------------------------------------------------------------ #
    def _patient_summary(self):
        """Per-patient peaks, computed once, so an alarm-budget bootstrap is cheap.

        Detection at a fixed budget only ever needs two numbers per patient: a
        non-septic patient's highest score anywhere in the stay (does it alarm at
        all?) and a septic patient's highest score at or before onset (is it caught
        in time?). Reducing to those up front turns each bootstrap draw from a sort
        over 310k rows into a sort over 8k numbers.
        """
        if getattr(self, "_summary", None) is not None:
            return self._summary
        d = self.ds
        order = np.lexsort((d.hour_index, d.groups))
        g, h, o = d.groups[order], d.hour_index[order], d.onset_hour[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        bounds = np.r_[starts, len(g)]
        spans, septic = [], []
        for a, b in zip(bounds[:-1], bounds[1:]):
            onset = int(o[a])
            spans.append((a, b, onset, np.flatnonzero(h[a:b] <= onset) if onset >= 0 else None))
            septic.append(onset >= 0)
        self._summary = (order, np.array(spans, dtype=object), np.array(septic))
        return self._summary

    def alarm_budget_curve(self, p: np.ndarray, target_far: float = 0.15):
        """(threshold, detection) at a fixed non-septic alarm budget, from peaks."""
        order, spans, septic = self._patient_summary()
        ps = p[order]
        peaks_ns, peaks_sep = [], []
        for (a, b, onset, pre), is_sep in zip(spans, septic):
            if not is_sep:
                peaks_ns.append(ps[a:b].max())
            elif pre is not None and pre.size:
                peaks_sep.append(ps[a:b][pre].max())
        return np.asarray(peaks_ns), np.asarray(peaks_sep)

    def compare_alarm_budget(self, cand: BenchResult, base: BenchResult,
                             target_far: float = 0.15, n_boot: int = 2000,
                             seed: int = 0) -> Dict[str, float]:
        """Paired patient bootstrap of detection at a fixed alarm budget.

        AUROC asks whether a detector ranks hours well overall. A ward asks a
        narrower question: inside the alarm load we are willing to carry, how many
        septic patients do we catch? Those come apart — a change can leave ranking
        untouched and still move the answer at the operating point — so the
        difference that matters gets its own significance test rather than being
        inferred from the AUROC one.
        """
        ns_c, sep_c = self.alarm_budget_curve(cand.oof, target_far)
        ns_b, sep_b = self.alarm_budget_curve(base.oof, target_far)
        rng = np.random.default_rng(seed)

        def detect(ns, sep, i_ns, i_sep):
            return float((sep[i_sep] >= metrics.budget_threshold(ns[i_ns], target_far)).mean())

        point = detect(ns_c, sep_c, np.arange(len(ns_c)), np.arange(len(sep_c))) - \
                detect(ns_b, sep_b, np.arange(len(ns_b)), np.arange(len(sep_b)))
        deltas = np.empty(n_boot)
        for j in range(n_boot):
            i_ns = rng.integers(0, len(ns_c), len(ns_c))
            i_sep = rng.integers(0, len(sep_c), len(sep_c))
            deltas[j] = detect(ns_c, sep_c, i_ns, i_sep) - detect(ns_b, sep_b, i_ns, i_sep)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        return {"delta_detection": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                "p_gt_0": float((deltas > 0).mean()), "significant": bool(lo > 0),
                "target_far": target_far}

    def pooled(self, res: BenchResult) -> Dict[str, float]:
        """Metrics over every row's out-of-fold prediction (one big honest test set)."""
        return self._score(np.arange(len(self.ds.y)), res.oof)

    def table(self, order: Optional[List[str]] = None) -> str:
        names = order or list(self.results)
        return "\n".join(self.results[n].line() for n in names if n in self.results)
