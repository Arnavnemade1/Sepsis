"""Streaming feature builder — guarantees train/serve consistency.

The detector was trained on causal features computed by
``backend.ml.data.features`` over CinC hourly frames. To use that exact model on
a live stream we must compute the *identical* features online. The safest way to
avoid train/serve skew is to reuse the very same feature code: we accumulate
incoming readings into a CinC-column frame and call the same
``build_patient_features`` function, then take the latest row.

Incoming telemetry may use the system's internal vital names (heart_rate, spo2,
...) or raw CinC names (HR, O2Sat, ...); both are accepted and normalised here.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.ml import config
from backend.ml.data.cinc2019 import PatientRecord
from backend.ml.data.features import FeatureConfig, build_patient_features

# internal vital name -> CinC column name (reverse of config.CINC_TO_INTERNAL).
_INTERNAL_TO_CINC = {v: k for k, v in config.CINC_TO_INTERNAL.items()}


#: The ICU clock the live stream reports stops here. Every reading advances the clock
#: one ICU hour (the demo compresses time), so a session left running was at "ICU hour
#: 1,000" by the evening - and CinC 2019's never-septic records end at ~60 h while
#: septic ones run to 336 h, so past ~60 h the clock alone reads as sepsis (GENESIS:
#: the ICU clock by itself scores AUROC 0.636). Measured on the shipped detector with
#: constant, normal vitals: p = 0.007 at hour 48, 0.043 at hour 72 (just under the
#: paging line, which noise then crosses), so a healthy live patient paged on elapsed
#: time alone. Held at 48 h - inside the range where both classes are represented in
#: training - a stable course stays at p ~ 0.007 indefinitely, and a deteriorating one
#: crosses the line on exactly the same reading as before.
ICULOS_CAP = 48.0


class StreamingFeatureBuilder:
    def __init__(self, feature_cfg: FeatureConfig, max_history: int = 256):
        self.cfg = feature_cfg
        self._rows: Deque[Dict[str, float]] = deque(maxlen=max_history)
        self._iculos = 0.0

    def add(self, telemetry: Dict[str, float]) -> None:
        """Append one reading. Keys may be internal or CinC names."""
        row = {c: np.nan for c in config.CINC_COLUMNS}
        for key, val in telemetry.items():
            if val is None:
                continue
            cinc = _INTERNAL_TO_CINC.get(key, key)   # accept either naming
            if cinc in row:
                row[cinc] = float(val)
        self._iculos += 1.0
        row["ICULOS"] = min(self._iculos, ICULOS_CAP)
        # Carry static demographics if supplied via internal keys.
        for k in ("Age", "Gender", "HospAdmTime"):
            if k in telemetry and telemetry[k] is not None:
                row[k] = float(telemetry[k])
        self._rows.append(row)

    def ready(self) -> bool:
        return len(self._rows) >= 1

    def latest_features(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Build the causal feature row for the most recent reading.

        Returns a single-row DataFrame; if ``feature_names`` is given it is
        reindexed to exactly match the model's expected columns/order.
        """
        if not self._rows:
            raise ValueError("No telemetry ingested yet.")
        frame = pd.DataFrame(list(self._rows), columns=config.CINC_COLUMNS)
        rec = PatientRecord(
            patient_id="stream", set_name="live",
            frame=frame.reset_index(drop=True),
            labels=np.zeros(len(frame), dtype=np.int8),
        )
        feats = build_patient_features(rec, self.cfg)
        last = feats.iloc[[-1]].reset_index(drop=True)
        if feature_names is not None:
            last = last.reindex(columns=feature_names, fill_value=0.0)
        return last
