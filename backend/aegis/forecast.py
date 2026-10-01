"""Trajectory forecasting — predicting the patient's near-future response.

Detection answers "is this patient septic now?"; forecasting answers "where is
this patient heading, and when?". Given the recent fused trajectory (edge optical
tier + enterprise labs/telemetry tier, temporally aligned by the fusion clock),
:class:`TrajectoryForecaster` projects each scored physiological channel forward
with Holt's linear exponential smoothing (level + trend), propagates an
uncertainty band that widens with the horizon, runs those forecasts back through
the same fusion ontology to get a **forecast risk envelope**, and reads off a
**projected onset** — the hour the risk is expected to breach the critical
threshold.

Honesty (the rule that matters most here): this is a transparent statistical
forecaster, not an oracle. Sepsis is partly stochastic; the forecast carries an
explicit, widening confidence band and the projected onset is a best estimate,
not a guarantee. Its real accuracy is measured against the ground-truth labels
(see verify_forecast.py / the Forecast validation), and reported as-is.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from backend.aegis.kernel.fusion import FusionEngine

SCORED = ["hr", "temp_c", "resp_rate", "spo2", "map", "wbc", "lactate"]
ENT_KEYS = ("temp_c", "resp_rate", "spo2", "map", "wbc", "lactate")
RANGES = {"hr": (30, 200), "temp_c": (33, 42), "resp_rate": (5, 60),
          "spo2": (60, 100), "map": (30, 140), "wbc": (0.3, 60), "lactate": (0.2, 25)}
# direction (per channel) that increases sepsis risk
_WORSE = {"hr": 1, "resp_rate": 1, "wbc": 1, "lactate": 1, "map": -1, "spo2": -1}
# channels that are abnormal on BOTH sides (SIRS: temp > 38 or < 36, WBC > 12 or < 4):
# worse is away from the centre of normal, better is toward it and stops there, so a
# "better" WBC of 4.4 is never pushed into leukopenia
_TWO_SIDED = {"temp_c": 37.0, "wbc": 8.0}
ALARM = 0.70


def _clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v


class TrajectoryForecaster:
    def __init__(self, horizon: int = 8):
        self.horizon = horizon
        self._fe = FusionEngine()

    # -- Holt's linear method: returns (level, trend, residual std) ------ #
    @staticmethod
    def _holt(y: Sequence[float], alpha: float = 0.4, beta: float = 0.25):
        n = len(y)
        if n < 3:
            return float(y[-1]), 0.0, 0.0
        level = float(y[0])
        trend = float(y[1] - y[0])
        resid: List[float] = []
        for t in range(1, n):
            fitted = level + trend
            resid.append(y[t] - fitted)
            new_level = alpha * y[t] + (1 - alpha) * (level + trend)
            trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
        m = sum(resid) / len(resid)
        sigma = math.sqrt(sum((r - m) ** 2 for r in resid) / len(resid)) if resid else 0.0
        return level, trend, sigma

    def _worse_dir(self, ch: str, val: float) -> int:
        if ch in _TWO_SIDED:
            return 1 if val >= _TWO_SIDED[ch] else -1     # e.g. fever or hypothermia both worsen
        return _WORSE.get(ch, 0)

    def _scenario(self, ch: str, base: float, band: float, sign: int) -> float:
        """``base`` moved by ``band`` in the worse (+1) or better (-1) direction."""
        d = self._worse_dir(ch, base)
        v = base + sign * d * band
        if sign < 0 and ch in _TWO_SIDED:                 # better stops at the centre
            c = _TWO_SIDED[ch]
            v = max(v, c) if base >= c else min(v, c)
        return _clamp(v, *RANGES[ch])

    def _fuse_once(self, vals: Dict[str, float]) -> float:
        edge = {"hr": vals["hr"],
                "perfusion": max(0.4, min(1.8, 1.0 + (vals["spo2"] - 95.0) / 18.0))}
        ent = {k: vals[k] for k in ENT_KEYS}
        return FusionEngine().update("_fc", "_fc", edge, ent).fused_risk

    # -- the forecast --------------------------------------------------- #
    def forecast(self, windows: Dict[str, Sequence[float]],
                 start_hour: int = 0) -> Dict[str, Any]:
        H = self.horizon
        ch_fc: Dict[str, Dict[str, List[float]]] = {}
        for ch in SCORED:
            y = list(windows.get(ch, []))
            level, trend, sigma = self._holt(y)
            mean, band = [], []
            for h in range(1, H + 1):
                mean.append(_clamp(level + h * trend, *RANGES[ch]))
                band.append(sigma * math.sqrt(h))
            ch_fc[ch] = {"mean": mean, "band": band, "trend": round(trend, 3)}

        risk_mean, risk_hi, risk_lo = [], [], []
        for h in range(H):
            base = {ch: ch_fc[ch]["mean"][h] for ch in SCORED}
            worse = {ch: self._scenario(ch, base[ch], ch_fc[ch]["band"][h], +1) for ch in SCORED}
            better = {ch: self._scenario(ch, base[ch], ch_fc[ch]["band"][h], -1) for ch in SCORED}
            risk_mean.append(round(self._fuse_once(base), 3))
            risk_hi.append(round(self._fuse_once(worse), 3))
            risk_lo.append(round(self._fuse_once(better), 3))

        # current risk = fuse the last observed sample; recent risk slope from the window
        cur = {ch: float(windows[ch][-1]) for ch in SCORED if windows.get(ch)}
        cur_risk = round(self._fuse_once(cur), 3) if len(cur) == len(SCORED) else risk_mean[0]
        L = len(windows.get(SCORED[0], []))
        wr = [self._fuse_once({ch: float(windows[ch][t]) for ch in SCORED})
              for t in range(max(0, L - 5), L)] if L >= 2 else [cur_risk]
        risk_slope = (wr[-1] - wr[0]) / max(len(wr) - 1, 1) if len(wr) > 1 else 0.0

        # projected onset: first horizon hour the mean risk breaches the alarm
        proj = None
        for h in range(H):
            if risk_mean[h] >= ALARM:
                proj = start_hour + h + 1
                break
        # earliest plausible (upper band) breach
        proj_worst = None
        for h in range(H):
            if risk_hi[h] >= ALARM:
                proj_worst = start_hour + h + 1
                break

        end = risk_mean[-1]
        if risk_slope > 0.015 or end > cur_risk + 0.06:
            traj = "deteriorating"
        elif risk_slope < -0.015 and end < cur_risk - 0.02:
            traj = "improving"
        else:
            traj = "stable"
        avg_band = sum(risk_hi[h] - risk_lo[h] for h in range(H)) / H
        confidence = round(_clamp(1.0 - avg_band, 0.3, 0.95), 2)

        return {
            "horizon": H, "current_risk": cur_risk,
            "risk_mean": risk_mean, "risk_hi": risk_hi, "risk_lo": risk_lo,
            "channels": ch_fc, "trajectory": traj, "risk_slope": round(risk_slope, 4),
            "projected_onset_in": (proj - start_hour) if proj is not None else None,
            "projected_onset_hour": proj, "earliest_breach_in": (proj_worst - start_hour) if proj_worst else None,
            "confidence": confidence, "alarm": ALARM,
        }
