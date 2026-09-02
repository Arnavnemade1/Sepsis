"""The risk timeline, drawn from the model that actually decides.

WHAT WAS WRONG WITH THE OLD CHART
---------------------------------
The FUSED SEPSIS-RISK TIMELINE plotted ``forecast.TrajectoryForecaster``, which forecasts
each physiological channel with Holt smoothing and pushes the result back through the
``FusionEngine``. Nothing on that chart was the model's output, and four separate defects
followed from it:

1. **Wrong scale, wrong threshold.** It drew a fusion risk in [0,1] against a hardcoded
   ``ALARM = 0.70``. The shipped detector alarms at **0.03** on its calibrated
   probability. The line a viewer read as "the alarm" was not the alarm.
2. **A discontinuity at *now*.** ``current_risk`` fused the *last observation* while
   ``risk_mean[h]`` fused ``level + h*trend`` from Holt's *smoothed level*. Those are
   different numbers, so the forecast did not start where the history ended — the
   observed 0.62 -> 0.95 jump in a single step.
3. **A confidence that was not one.** ``confidence = clamp(1 - avg_band, 0.3, 0.95)``.
   That is one minus the mean width of the envelope, rendered as "forecast confidence
   82%". It has no probabilistic meaning; nothing is 82% likely.
4. **A badge fighting its own projection.** ``trajectory`` came from the recent *history*
   slope while ``projected_onset`` came from the *forecast* crossing the alarm, so
   "IMPROVING - projected critical in ~1h" was two different series disagreeing in one
   sentence.

WHAT THIS DRAWS INSTEAD
-----------------------
The **calibrated detector probability** — the quantity that drives the alarm, the
operating point and the conformal decision. Every element on the chart resolves to a
shipped artifact, and ``provenance`` names which:

    history          the detector's own calibrated output per hour
    alarm line       artifacts/operating_point.json  (HIGH, currently 0.03)
    shaded band      artifacts/conformal.json        (the UNCERTAIN region)
    forecast         a stated extrapolation of the model's own score, anchored at now

Forecasting the score rather than re-deriving it through a second engine is the point.
It cannot drift onto a different scale, and the anchoring bug cannot recur because the
first forecast point is *defined* as the current score.

WHAT THE FORECAST IS, STATED PLAINLY
------------------------------------
Holt linear extrapolation of the model's recent probability, with a band from the
residual spread of that same fit. It is a trend line, not a physiological simulation and
not a claim about the patient's future. It carries no calibrated probability of breach,
because none has been measured — ``breach_in_hours`` is "where this trend crosses the
line if it continues", which is a different and much weaker statement than a forecast
probability, and it is labelled as such.

THE UNCERTAIN BAND IS THE INTERESTING PART
-------------------------------------------
Because the conformal region is drawn, the chart shows something a bare risk trace
cannot: the span where the model's own prediction set contains BOTH labels, i.e. where it
cannot separate septic from not-septic at the requested error rate. On the shipped
bands that region is roughly p in [0.006, 0.078] — and the alarm threshold sits *inside*
it. A viewer can therefore see directly that the operating point lives in the band where
the model abstains, which is the honest picture and was invisible before.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

#: Minimum observed points before a trend is extrapolated at all. Below this the
#: forecast is refused rather than drawn from noise.
MIN_HISTORY = 4


@dataclass
class TimelinePoint:
    hour: float
    probability: float
    source: str                      # "observed" | "forecast"
    lo: Optional[float] = None
    hi: Optional[float] = None


@dataclass
class RiskTimeline:
    history: List[TimelinePoint] = field(default_factory=list)
    forecast: List[TimelinePoint] = field(default_factory=list)
    alarm_threshold: Optional[float] = None
    medium_threshold: Optional[float] = None
    uncertain_lo: Optional[float] = None
    uncertain_hi: Optional[float] = None
    trajectory: str = "unknown"
    trend_per_hour: float = 0.0
    breach_in_hours: Optional[float] = None
    onset_hour: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    limits: List[str] = field(default_factory=list)
    refused: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _holt(y: Sequence[float], alpha: float = 0.4, beta: float = 0.25):
    """Level, trend and residual sigma of Holt's linear smoothing."""
    y = [float(v) for v in y]
    if len(y) < 2:
        return (y[-1] if y else 0.0), 0.0, 0.0
    level, trend = y[0], y[1] - y[0]
    resid: List[float] = []
    for v in y[1:]:
        pred = level + trend
        resid.append(v - pred)
        new_level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
    sigma = (sum(r * r for r in resid) / max(len(resid) - 1, 1)) ** 0.5 if resid else 0.0
    return level, trend, sigma


def build(probabilities: Sequence[float],
          hours: Optional[Sequence[float]] = None,
          horizon: int = 8,
          onset_hour: Optional[float] = None) -> RiskTimeline:
    """Build the timeline from the model's own calibrated probability history."""
    tl = RiskTimeline()
    p = [float(v) for v in (probabilities or []) if v is not None]
    hrs = list(hours) if hours is not None else list(range(len(p)))
    tl.onset_hour = onset_hour

    # --- provenance: every drawn element names the artifact it came from -------
    op = bands = None
    try:
        from backend.ml.operating_point import load as _load_op
        from backend.ml import conformal as _conf
        op = _load_op()
        if op is not None:
            tl.alarm_threshold = float(op.high)
            tl.medium_threshold = float(op.medium)
            bands = _conf.load(expect_detector=getattr(op, "detector", None))
    except Exception:
        pass
    if bands is not None:
        tl.uncertain_lo = float(bands.marginal.lo)
        tl.uncertain_hi = float(bands.marginal.hi)
    tl.provenance = {
        "series": "calibrated detector probability (the quantity that drives the alarm)",
        "detector": getattr(op, "detector", None),
        "alarm_source": "artifacts/operating_point.json:high",
        "band_source": ("artifacts/conformal.json:marginal" if bands is not None
                        else "unavailable"),
        "band_alpha": getattr(bands, "alpha", None),
        "forecast_method": "Holt linear extrapolation of the model's own score, "
                           "anchored at the current value",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tl.limits = [
        "The forecast is a trend line on the model's score, not a physiological "
        "simulation and not a claim about the patient's future.",
        "breach_in_hours is 'where this trend crosses the line if it continues'. It is "
        "NOT a probability of breach — no such probability has been measured for this "
        "forecaster, so none is shown.",
        "The forecast band is the residual spread of the trend fit widening with the "
        "horizon. It is not a calibrated prediction interval.",
        "The alarm threshold sits INSIDE the conformal uncertain band, so at the "
        "operating point the model's own prediction set often contains both labels.",
    ]

    for h, v in zip(hrs, p):
        tl.history.append(TimelinePoint(hour=float(h), probability=float(v),
                                        source="observed"))

    if len(p) < MIN_HISTORY:
        tl.refused = (f"only {len(p)} observed point(s); a trend needs at least "
                      f"{MIN_HISTORY}. Refusing to extrapolate from noise.")
        tl.trajectory = "insufficient history"
        return tl

    # --- the forecast, ANCHORED at the present --------------------------------
    # The old chart used Holt's smoothed level as the origin, which differs from the
    # last observation and produced a visible jump at `now`. The trend is taken from
    # the fit; the ORIGIN is the current score, by construction.
    _, trend, sigma = _holt(p)
    now_h, now_p = float(hrs[-1]), float(p[-1])
    tl.trend_per_hour = round(float(trend), 5)
    for k in range(1, horizon + 1):
        mean = max(0.0, min(1.0, now_p + k * trend))
        width = sigma * math.sqrt(k)
        tl.forecast.append(TimelinePoint(
            hour=now_h + k, probability=round(mean, 5), source="forecast",
            lo=round(max(0.0, mean - width), 5), hi=round(min(1.0, mean + width), 5)))

    # --- trajectory and projection from the SAME series -----------------------
    # The old chart derived the badge from the history slope and the projection from the
    # forecast, so they could contradict each other in one sentence. Both now read the
    # forecast that is actually drawn.
    end = tl.forecast[-1].probability
    delta = end - now_p
    scale = max(tl.alarm_threshold or 0.03, 1e-6)
    if delta > 0.10 * scale:
        tl.trajectory = "deteriorating"
    elif delta < -0.10 * scale:
        tl.trajectory = "improving"
    else:
        tl.trajectory = "stable"

    if tl.alarm_threshold is not None:
        if now_p >= tl.alarm_threshold:
            tl.breach_in_hours = 0.0
        elif trend > 0:
            for pt in tl.forecast:
                if pt.probability >= tl.alarm_threshold:
                    tl.breach_in_hours = pt.hour - now_h
                    break
    # A rising projection while the badge says improving is now impossible: both come
    # from `trend`. Assert it rather than trust it.
    if tl.trajectory == "improving" and tl.breach_in_hours not in (None, 0.0):
        raise AssertionError("improving trajectory cannot project a future breach — "
                             "the badge and the projection have desynchronised")
    return tl
