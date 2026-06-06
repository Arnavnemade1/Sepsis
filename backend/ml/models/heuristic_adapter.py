"""Run the EXISTING (hand-tuned) detector over real CinC 2019 patients.

This is the most important honesty check in the whole project: it takes the
current HDC + digital-twin + NeurosymbolicEngine pipeline exactly as shipped and
scores it against real labelled data, producing the baseline AUROC/AUPRC/utility
that every future improvement must beat. No retuning, no peeking — just the
heuristic as it stands today.

Known, deliberately-unhidden limitations of this baseline:
  * CinC has no HRV or motion channel, so the twin runs them at defaults. The
    real detector leans on HRV, so this understates what a wearable could do —
    but it is the honest number for *this* dataset.
  * CinC ``Temp`` is core temperature; the engine treats it as skin temp.
  * CinC ``Lactate`` is blood lactate, used in place of the project's sweat
    lactate proxy.
These are reported, not papered over.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.core.hdc import HDCEngine
from backend.core.twin import PatientDigitalTwin
from backend.core.neurosymbolic import NeurosymbolicEngine
from backend.ml.data.cinc2019 import PatientRecord
from backend.ml.data.features import NORMAL_DEFAULTS
from backend.ml import config


# CinC column -> twin vital name, for channels the twin understands.
_VITAL_MAP = {
    "HR": "heart_rate",
    "O2Sat": "spo2",
    "Temp": "skin_temp",
    "SBP": "systolic_bp",
    "MAP": "map",
    "DBP": "diastolic_bp",
    "Resp": "respiratory_rate",
    "Lactate": "sweat_lactate",
}


def _ffill_with_default(values: np.ndarray, default: float) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float64)
    last = default
    for i, v in enumerate(values):
        if not np.isnan(v):
            last = float(v)
        out[i] = last
    return out


def _patient_vital_streams(rec: PatientRecord) -> Dict[str, np.ndarray]:
    streams: Dict[str, np.ndarray] = {}
    for cinc_col in _VITAL_MAP:
        raw = rec.frame.get(cinc_col)
        vals = raw.to_numpy(dtype=np.float64) if raw is not None else np.full(rec.n_hours, np.nan)
        streams[cinc_col] = _ffill_with_default(vals, NORMAL_DEFAULTS.get(cinc_col, 0.0))
    # WBC (sparse) feeds the SIRS clause through clinical_history.
    wbc_raw = rec.frame.get("WBC")
    streams["WBC"] = _ffill_with_default(
        wbc_raw.to_numpy(dtype=np.float64) if wbc_raw is not None else np.full(rec.n_hours, np.nan),
        NORMAL_DEFAULTS["WBC"],
    )
    return streams


def score_patient(
    rec: PatientRecord,
    hdc_engine: HDCEngine,
    neuro: NeurosymbolicEngine,
) -> np.ndarray:
    """Return the heuristic sepsis_probability for every hour of one patient."""
    streams = _patient_vital_streams(rec)
    twin = PatientDigitalTwin(rec.patient_id, hdc_engine)
    probs = np.zeros(rec.n_hours, dtype=np.float64)

    for t in range(rec.n_hours):
        vitals = {twin_name: streams[cinc][t] for cinc, twin_name in _VITAL_MAP.items()}
        twin_update = twin.update_telemetry(vitals)
        twin_state = twin.get_current_twin_state()
        history = {"wbc": float(streams["WBC"][t]), "gcs": 15}

        _, _, _, _, details = neuro.evaluate_patient(
            vitals,
            twin_update["sepsis_similarity"],
            history,
            twin_update["healthy_similarity"],
            twin_state["trends"],
            twin_state["features"],
        )
        probs[t] = details["sepsis_probability"]

    return probs


def score_cohort(
    records: List[PatientRecord],
    hdc_engine: Optional[HDCEngine] = None,
    neuro: Optional[NeurosymbolicEngine] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score every patient. Returns (p, y, groups, hour_index, onset_hour)."""
    hdc_engine = hdc_engine or HDCEngine()
    neuro = neuro or NeurosymbolicEngine()

    p_all, y_all, g_all, h_all, o_all = [], [], [], [], []
    for i, rec in enumerate(records, 1):
        p_all.append(score_patient(rec, hdc_engine, neuro))
        y_all.append(rec.labels.astype(np.int8))
        g_all.append(np.array([rec.patient_id] * rec.n_hours))
        h_all.append(np.arange(rec.n_hours))
        onset = rec.label_onset_hour
        o_all.append(np.full(rec.n_hours, -1 if onset is None else onset))
        if verbose and (i % 250 == 0 or i == len(records)):
            print(f"  scored {i}/{len(records)} patients")

    return (
        np.concatenate(p_all),
        np.concatenate(y_all),
        np.concatenate(g_all),
        np.concatenate(h_all),
        np.concatenate(o_all),
    )
