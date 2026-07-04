"""ML control plane — train and SPECIALIZE sepsis detectors from the OS itself.

This turns model development into a first-class OS operation: from the AegisOS
terminal an operator can spin up a real training job, watch it run, and register
the result — no notebook, no separate script. Every number it reports is measured,
never fabricated:

  * data is the real PhysioNet/CinC-2019 cohort (patient-grouped, leakage-safe),
    read through the same feature pipeline the production detector uses;
  * a job trains a real ``GBMDetector`` under **patient-grouped cross-validation**
    and reports out-of-fold AUROC / AUPRC / ECE / sensitivity@spec — the honest
    generalisation numbers, computed on held-out patients;
  * **specialisation** is a genuine relabelling: an *early-warning* model is trained
    to fire only within an ``horizon``-hour window before the labelled onset, so it
    is optimised for lead time rather than concurrent detection.

Jobs run on background threads (real sklearn training is CPU-bound, so this is an
honest worker pool, not a cooperative coroutine). The registry lives for the life
of the process and is exposed read-only through ``/proc/models``. Nothing here
hot-swaps the live clinical detector — a trained model is registered and can be
marked ACTIVE, but promoting it into the certified pipeline still goes through the
audited ``train_stacked --promote`` path. That boundary is deliberate.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import FeatureConfig, build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_folds
from backend.ml.models.gbm_baseline import GBMDetector

FEATURE_SETS = {
    "wearable": FeatureConfig.wearable_only,
    "lab": FeatureConfig.lab_augmented,
}


@dataclass
class TrainJob:
    id: str
    name: str
    spec: Dict[str, Any]
    status: str = "QUEUED"            # QUEUED|LOADING|TRAINING|EVAL|DONE|FAILED
    progress: float = 0.0
    log: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "spec": self.spec,
            "status": self.status, "progress": round(self.progress, 3),
            "log": self.log[-40:], "metrics": self.metrics,
            "elapsed_s": round((self.finished or time.time()) - self.started, 1),
            "error": self.error,
        }


@dataclass
class ModelRecord:
    name: str
    spec: Dict[str, Any]
    metrics: Dict[str, Any]
    n_rows: int
    n_patients: int
    n_features: int
    created: float = field(default_factory=time.time)
    active: bool = False
    model: Optional[Any] = None       # the fitted detector (in-memory)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "spec": self.spec, "metrics": self.metrics,
            "n_rows": self.n_rows, "n_patients": self.n_patients,
            "n_features": self.n_features, "active": self.active,
            "created": self.created,
        }


class ModelControlPlane:
    """The OS-level trainer + model registry."""

    def __init__(self) -> None:
        self.jobs: Dict[str, TrainJob] = {}
        self.models: Dict[str, ModelRecord] = {}
        self._lock = threading.Lock()

    # -- launch a training / specialisation job -------------------------- #
    def train(self, name: str, feature_set: str = "wearable",
              horizon: Optional[int] = None, limit: int = 1000,
              folds: int = 3, seed: int = 42) -> TrainJob:
        feature_set = feature_set if feature_set in FEATURE_SETS else "wearable"
        limit = max(200, min(int(limit), 20000))
        folds = max(2, min(int(folds), 5))
        spec = {"feature_set": feature_set, "horizon": horizon, "limit": limit,
                "folds": folds, "seed": seed,
                "kind": "early-warning-specialised" if horizon else "detector"}
        job = TrainJob(id=uuid.uuid4().hex[:8], name=name, spec=spec)
        with self._lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), name=f"train-{job.id}",
                         daemon=True).start()
        return job

    def specialize(self, name: str, horizon: int, feature_set: str = "wearable",
                   limit: int = 1000) -> TrainJob:
        """Train an early-warning-specialised model (fires within ``horizon`` h of onset)."""
        return self.train(name, feature_set=feature_set, horizon=int(horizon), limit=limit)

    # -- the worker ------------------------------------------------------ #
    def _run(self, job: TrainJob) -> None:
        try:
            fs, limit = job.spec["feature_set"], job.spec["limit"]
            horizon, folds, seed = job.spec["horizon"], job.spec["folds"], job.spec["seed"]
            job.status = "LOADING"
            job.log.append(f"loading {fs} features (limit {limit}, patient-grouped)…")
            cfg = FEATURE_SETS[fs]()
            ds = build_dataset_cached(lambda: load_patients(limit=limit), cfg,
                                      f"{fs}_limit{limit}")
            X, groups = ds.X, ds.groups
            y = self._relabel(ds, horizon)
            n_pat = int(len(np.unique(groups)))
            prev = float(y.mean())
            job.log.append(f"{len(y)} rows · {n_pat} patients · {X.shape[1]} features · "
                           f"prevalence {prev:.3f}"
                           + (f" · specialised: onset−{horizon}h window" if horizon else ""))
            if len(np.unique(y)) < 2:
                raise ValueError("only one class after labelling — widen the horizon/limit")

            job.status = "TRAINING"
            oof = np.full(len(y), np.nan, dtype=np.float64)
            for i, (tr, te) in enumerate(stratified_group_folds(groups, y, folds, seed)):
                det = GBMDetector(feature_names=ds.feature_names, seed=seed)
                det.fit(X[tr], y[tr], groups[tr])
                oof[te] = det.predict_proba(X[te])
                job.progress = (i + 1) / (folds + 1)
                job.log.append(f"fold {i + 1}/{folds} — out-of-fold AUROC "
                               f"{metrics.auroc(y[te], oof[te]):.3f}")

            job.status = "EVAL"
            m = oof == oof                      # non-NaN mask
            yv, pv = y[m], oof[m]
            sens, _ = metrics.sensitivity_at_specificity(yv, pv, 0.85)
            job.metrics = {
                "auroc": round(float(metrics.auroc(yv, pv)), 4),
                "auprc": round(float(metrics.auprc(yv, pv)), 4),
                "ece": round(float(metrics.expected_calibration_error(yv, pv)), 4),
                "sens_at_spec85": round(float(sens), 4),
                "prevalence": round(prev, 4),
                "cv": f"{folds}-fold patient-grouped",
            }
            job.log.append(f"CV complete — AUROC {job.metrics['auroc']} · "
                           f"AUPRC {job.metrics['auprc']} · sens@85%spec {job.metrics['sens_at_spec85']}")

            # refit on ALL rows for a usable registered model
            final = GBMDetector(feature_names=ds.feature_names, seed=seed)
            final.fit(X, y, groups)
            job.progress = 1.0
            with self._lock:
                self.models[job.name] = ModelRecord(
                    name=job.name, spec=job.spec, metrics=job.metrics,
                    n_rows=int(len(y)), n_patients=n_pat, n_features=int(X.shape[1]),
                    model=final)
            job.status = "DONE"
            job.finished = time.time()
            job.log.append(f"registered model '{job.name}' ✓")
        except Exception as exc:  # pragma: no cover — a bad job must not crash the server
            job.status = "FAILED"
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished = time.time()
            job.log.append(f"FAILED: {job.error}")

    @staticmethod
    def _relabel(ds, horizon: Optional[int]) -> np.ndarray:
        """Standard labels, OR an early-warning relabel: positive iff the row is
        within ``horizon`` hours BEFORE the labelled onset (0..horizon)."""
        if not horizon:
            return ds.y.astype(np.int8)
        lead = ds.onset_hour - ds.hour_index
        pos = (ds.onset_hour >= 0) & (lead >= 0) & (lead <= int(horizon))
        return pos.astype(np.int8)

    # -- registry ops ---------------------------------------------------- #
    def use(self, name: str) -> Dict[str, Any]:
        with self._lock:
            if name not in self.models:
                return {"ok": False, "error": f"no model '{name}'"}
            for r in self.models.values():
                r.active = (r.name == name)
        return {"ok": True, "active": name,
                "note": "marked ACTIVE in the experimental registry; certified "
                        "promotion still goes through train_stacked --promote"}

    def job(self, job_id: str) -> Optional[Dict[str, Any]]:
        j = self.jobs.get(job_id)
        return j.to_dict() if j else None

    def list_models(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in sorted(self.models.values(),
                                            key=lambda r: r.created, reverse=True)]

    def status(self) -> Dict[str, Any]:
        running = [j.to_dict() for j in self.jobs.values()
                   if j.status not in ("DONE", "FAILED")]
        return {
            "models": self.list_models(),
            "jobs": [j.to_dict() for j in sorted(self.jobs.values(),
                                                 key=lambda j: j.started, reverse=True)][:12],
            "running": len(running),
            "feature_sets": list(FEATURE_SETS),
            "active": next((r.name for r in self.models.values() if r.active), None),
        }
