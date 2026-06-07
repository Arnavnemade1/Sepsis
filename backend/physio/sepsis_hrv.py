"""HRV -> sepsis modeling pipeline (credential-ready).

This is the Link-B pipeline: slice a continuous RR-interval series into windows,
compute the full HRV feature set per window, and train a calibrated detector with
the same leakage-free, patient-grouped evaluation harness used for the clinical
model. It is written to consume ANY source of RR intervals + labels via the
``RRLabeledSource`` protocol, so the day credentialed data (MIMIC-IV Waveform /
HiRID) is available, it runs unchanged.

Validation status (honest):
  * HRV FEATURE extraction is validated on REAL physiology (BIDMC PPG vs ECG —
    see backend.physio.validate_ppg_hrv).
  * The HRV->sepsis MODEL is validated here against a synthetic cohort whose
    septic physiology follows the published HeRO "heart rate characteristics"
    signature (reduced variability + reduced sample entropy + transient
    decelerations). Validation on real sepsis LABELS awaits credentialed data;
    the gap is explicit, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Tuple

import numpy as np

from backend.physio import hrv
from backend.physio.hrv import HRVFeatures
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_split
from backend.ml.models.gbm_baseline import GBMDetector

# The ordered HRV features fed to the model (drop n_beats/quality bookkeeping).
HRV_FEATURE_NAMES = [
    "mean_hr", "mean_nn", "sdnn", "rmssd", "sdsd", "pnn50", "cvnn",
    "lf", "hf", "lf_hf", "total_power", "lf_nu", "hf_nu",
    "sd1", "sd2", "sd1_sd2", "sample_entropy", "sample_asymmetry",
]


class RRLabeledSource(Protocol):
    """Interface a labelled RR dataset must implement to plug into this pipeline.

    Yield one tuple per patient:
        (patient_id, rr_ms, labels) where
        - rr_ms  : 1-D np.ndarray of consecutive RR intervals in milliseconds
        - labels : 1-D np.ndarray (same length as rr_ms) of 0/1 sepsis labels,
                   OR a single int 0/1 for a patient-level label.
    A MIMIC-IV-Waveform / HiRID adapter implementing this is all that's needed.
    """
    def __iter__(self) -> "Iterable[Tuple[str, np.ndarray, np.ndarray]]": ...


@dataclass
class WindowedHRVDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[str]


def _features_vector(rr_window: np.ndarray) -> Optional[np.ndarray]:
    f: HRVFeatures = hrv.hrv_features(rr_window)
    if f.quality <= 0.0:
        return None
    d = f.as_dict()
    return np.array([d[k] for k in HRV_FEATURE_NAMES], dtype=np.float64)


def build_hrv_dataset(
    source: Iterable[Tuple[str, np.ndarray, object]],
    window_beats: int = 128,
    step_beats: int = 32,
) -> WindowedHRVDataset:
    """Slice each patient's RR series into overlapping windows -> HRV features.

    Window length follows the HeRV/HeRO convention of a few-hundred-beat epoch.
    Labelling per window: a window is positive if any beat in it is labelled 1
    (patient-level int labels are broadcast to all windows).
    """
    Xs, ys, gs = [], [], []
    for pid, rr_ms, labels in source:
        rr_ms = np.asarray(rr_ms, dtype=np.float64)
        n = len(rr_ms)
        if n < window_beats:
            continue
        per_beat = (np.full(n, int(labels)) if np.ndim(labels) == 0
                    else np.asarray(labels, dtype=int))
        for start in range(0, n - window_beats + 1, step_beats):
            win = rr_ms[start:start + window_beats]
            vec = _features_vector(win)
            if vec is None:
                continue
            lab = int(per_beat[start:start + window_beats].max()) if per_beat.ndim else int(per_beat)
            Xs.append(vec)
            ys.append(lab)
            gs.append(pid)
    if not Xs:
        raise ValueError("No usable HRV windows produced from the source.")
    return WindowedHRVDataset(np.vstack(Xs), np.array(ys, dtype=int),
                              np.array(gs), list(HRV_FEATURE_NAMES))


def train_and_evaluate(ds: WindowedHRVDataset, seed: int = 42):
    """Train a calibrated GBM on HRV windows with patient-grouped split."""
    tr, te = stratified_group_split(ds.groups, ds.y, test_size=0.25, seed=seed)
    model = GBMDetector(feature_names=ds.feature_names)
    model.fit(ds.X[tr], ds.y[tr], groups=ds.groups[tr])
    p = model.predict_proba(ds.X[te])
    report = metrics.evaluate(ds.y[te], p, ds.groups[te])
    return model, report


# --------------------------------------------------------------------------- #
# HeRO-grounded synthetic cohort (proxy validation of the MODEL machinery)
# --------------------------------------------------------------------------- #
def simulate_rr_series(septic: bool, n_beats: int = 600, seed: int = 0) -> np.ndarray:
    """Generate a realistic RR series.

    Healthy: ~normal variability with respiratory sinus arrhythmia (HF).
    Septic : the published HeRO signature — reduced overall variability, reduced
             sample entropy (more monotonous baseline), plus intermittent
             transient DECELERATIONS (sudden RR increases), and a higher base HR.
    """
    rng = np.random.default_rng(seed)
    if not septic:
        mean_rr = rng.uniform(720, 1000)           # ~60-83 bpm
        rsa_amp = rng.uniform(20, 45)
        rsa = rsa_amp * np.sin(2 * np.pi * 0.25 * np.arange(n_beats) / 4.0)
        rr = mean_rr + rsa + rng.normal(0, rng.uniform(22, 42), n_beats)
    else:
        # Severity varies per patient so distributions OVERLAP with healthy —
        # mild cases look near-normal, only severe ones are obvious.
        sev = rng.uniform(0.25, 1.0)
        mean_rr = rng.uniform(600, 860) - 60 * sev   # partial overlap w/ healthy
        rsa_amp = rng.uniform(8, 26) * (1.1 - 0.5 * sev)
        rsa = rsa_amp * np.sin(2 * np.pi * 0.25 * np.arange(n_beats) / 4.0)
        rr = mean_rr + rsa + rng.normal(0, rng.uniform(10, 26) * (1.1 - 0.4 * sev), n_beats)
        # Transient decelerations (HeRO hallmark) — frequency scales with severity.
        n_decel = int(rng.integers(0, 2) + round(5 * sev))
        for _ in range(n_decel):
            s = rng.integers(0, n_beats - 6)
            rr[s:s + rng.integers(3, 6)] += rng.uniform(80, 220) * sev
    return np.clip(rr, 300, 2000)


def proxy_validation(n_per_class: int = 60, seed: int = 0):
    """Build a synthetic HeRO-grounded cohort and run the full pipeline."""
    source = []
    rng = np.random.default_rng(seed)
    for i in range(n_per_class):
        source.append((f"healthy_{i}", simulate_rr_series(False, 600, int(rng.integers(1e9))), 0))
        source.append((f"septic_{i}", simulate_rr_series(True, 600, int(rng.integers(1e9))), 1))
    ds = build_hrv_dataset(source, window_beats=128, step_beats=64)
    model, report = train_and_evaluate(ds)
    return ds, report


if __name__ == "__main__":  # pragma: no cover
    print("HRV->sepsis pipeline — proxy validation on HeRO-grounded synthetic cohort")
    ds, rep = proxy_validation()
    print(f"  windows={len(ds.y)}  positive={int(ds.y.sum())}  features={len(ds.feature_names)}")
    print(rep.pretty())
    print("\nNOTE: synthetic septic physiology follows the published HeRO signature.")
    print("Validation on REAL sepsis labels awaits credentialed MIMIC-IV/HiRID data;")
    print("HRV feature extraction itself is validated on real BIDMC PPG vs ECG.")
