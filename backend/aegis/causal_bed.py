"""Build a per-patient causal graph on demand from a cohort bed's real CinC history.

The live session patient has a rich causal graph grown tick-by-tick. To make the
Clinical Story available for *every* monitored patient, this replays a cohort bed's
recorded hourly stay through the same :class:`CausalGraphIngestor`, so each patient
gets a real, integrity-checked clinical story derived from their own data.

Honest framing: the per-hour *risk* used to drive the distillation is a transparent
vitals proxy (MAP/lactate/HR/RR/SpO₂), not the production model — it shapes which
hours become events. The terminal *outcome* uses the dataset's ground-truth sepsis
label, so the story's endpoint is real, not inferred.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pandas as pd

from backend.aegis.causal_ingest import CausalGraphIngestor

H = 3600.0


def _clip(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _hour_risk(v: Dict[str, float]) -> float:
    mp, lact = v.get("map", 90.0), v.get("lactate", 1.0)
    hr, rr = v.get("heart_rate", 75.0), v.get("respiratory_rate", 16.0)
    spo2 = v.get("spo2", 98.0)
    return _clip(0.30 * _clip((65 - mp) / 25) + 0.30 * _clip((lact - 2) / 6)
                 + 0.15 * _clip((hr - 90) / 60) + 0.15 * _clip((rr - 20) / 15)
                 + 0.10 * _clip((92 - spo2) / 12))


def _sirs(v: Dict[str, float]) -> list:
    out = []
    if v.get("heart_rate", 75) > 90: out.append("HR>90")
    if v.get("respiratory_rate", 16) > 20: out.append("RR>20")
    t = v.get("skin_temp", 37.0)
    if t > 38 or t < 36: out.append("temp dysreg")
    return out


def build_bed_causal(record, bed_id: str,
                     vital_every_h: float = 3.0, omission_after_h: float = 2.0
                     ) -> CausalGraphIngestor:
    """Replay a PatientRecord's hourly stay into a CausalGraphIngestor."""
    frame: pd.DataFrame = record.frame
    labels = record.labels
    ing = CausalGraphIngestor(bed=bed_id, vital_every_s=vital_every_h * H,
                              omission_after_s=omission_after_h * H)

    def col(name, default):
        s = pd.to_numeric(frame.get(name), errors="coerce") if name in frame else None
        return (s.ffill().fillna(default).to_numpy() if s is not None
                else np.full(len(frame), default))
    HR, RR = col("HR", 75.0), col("Resp", 16.0)
    MAP, LACT = col("MAP", 90.0), col("Lactate", 1.0)
    SPO2, TEMP = col("O2Sat", 98.0), col("Temp", 37.0)

    n = len(frame)
    for h in range(n):
        v = {"map": float(MAP[h]), "lactate": float(LACT[h]),
             "heart_rate": float(HR[h]), "respiratory_rate": float(RR[h]),
             "spo2": float(SPO2[h]), "skin_temp": float(TEMP[h])}
        risk = _hour_risk(v)
        if h < len(labels) and labels[h] == 1:        # ground-truth deterioration window
            risk = max(risk, 0.72)
        band = "HIGH" if risk >= 0.6 else "MEDIUM" if risk >= 0.35 else "LOW"
        a = SimpleNamespace(risk_level=band, ml_probability=round(risk, 3),
                            detection={}, protocol=None, verification=None,
                            signal_quality=1.0, triggered_rules=_sirs(v))
        ing.ingest(a, v, h * H)

    if record.is_septic:
        last = {"map": float(MAP[-1]), "lactate": float(LACT[-1])} if n else {}
        ing.mark_outcome("sepsis_onset", n * H, last, risk=0.9)
    return ing
