"""The ward's risk: the shipped detector, scoring every replayed ICU record.

WHAT WAS WRONG
--------------
The twelve ICU beds (rooms 301-312) page through the paging OS, fill the census, light
the floor plan and gate the bundle engine. Their risk came from ``kernel.fusion`` - a
hand-weighted qSOFA/SIRS logistic, ``sigmoid(1.5 * sum - 1.3)`` - passed through the
own-baseline learner. The validated detector served exactly one bed: the live session
in room 402. Measured on the 8,062 patients the shipped detector never trained on (its
promotion holdout), every hour, against the challenge label:

    ward path (fusion -> own-baseline personalisation)   AUROC 0.632   hour-matched 0.651
    shipped detector, calibrated                          AUROC 0.800   hour-matched 0.733
    difference, paired patient bootstrap (200 draws)      +0.168 [+0.145, +0.191]

The Evidence page described the 0.800 detector as "the sepsis detector that raises
pages". For the ward that was not true. It is now.

Own-baseline personalisation on top of the detector LOSES discrimination (0.776, -0.024
on the same patients): a calibrated model already conditions on the patient's own
trajectory (``__bz``, expanding extremes), and the learner's multiplicative dampening
reorders patients it should not. It stays where it earns its place - the record's
explanation of how this patient differs from their own baseline - and is out of the
alarm path.

THE INDEX, AND WHY THERE IS ONE
-------------------------------
Everything downstream of the ward reads a 0-1 risk against lines drawn on it: the
paging line (0.70), the watch band (0.60-0.70), the census tiers, the bundle's and the
write syscalls' severities, the scheduler. Re-expressing a dozen thresholds in
probability units would move every one of them for no gain in discrimination, so the
detector's calibrated probability ``p`` is mapped onto that scale instead, by a strictly
monotone power-odds transform pinned at the detector's OWN validated operating point
(``artifacts/operating_point.json``, derived on this exact fit):

    index = sigmoid(a * (logit(p) - c))      index(MEDIUM) = 0.35,  index(HIGH) = 0.70

Monotone, so the ranking - and AUROC - is the detector's exactly (0.79999 both, on the
holdout). The paging line IS the detector's validated HIGH point; the medium line IS
its MEDIUM point. The index is a position on the alarm scale, not a probability, so the
calibrated probability travels beside it everywhere risk is the headline.

CAUSALITY AND DATA
------------------
The detector scores hour ``h`` from features of hours ``<= h`` only (forward fill,
rolling and expanding windows; every wearable channel has a fixed population default,
so no leading gap is filled from the future). The replay looks up the score of the
CURRENT hour and never interpolates toward the next one: a value between hours that
leans on hour ``h+1`` would show the future. ``verify_ward_risk`` rewrites a record's
future and checks no past score moves.

The demo ward is drawn from patients the serving fit never trained on
(``data/cohort_sample/heldout.json``), so every number on it is out of sample.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: The ward's lines on the index scale. HIGH and MEDIUM are the detector's operating
#: point by construction; WATCH is the band just below the paging line.
INDEX_MEDIUM = 0.35
INDEX_WATCH = 0.60
INDEX_HIGH = 0.70

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HELDOUT_PATH = os.path.join(_REPO, "data", "cohort_sample", "heldout.json")
#: written by ``python -m backend.ml.scripts.ward_auroc``: the numbers above, measured
MEASURED_PATH = os.path.join(_REPO, "artifacts", "ward_risk.json")


def measured() -> Optional[Dict[str, Any]]:
    """The measurement behind this module's claims, or None if it was never run."""
    try:
        with open(MEASURED_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class IndexMap:
    """``index = sigmoid(a * (logit(p) - c))``, pinned at the operating point."""
    a: float
    c: float
    p_medium: float
    p_high: float
    detector: str

    @staticmethod
    def pinned(p_medium: float, p_high: float, detector: str = "unknown") -> "IndexMap":
        if not (0.0 < p_medium < p_high < 1.0):
            raise ValueError(f"operating point must satisfy 0 < medium < high < 1, got "
                             f"{p_medium}, {p_high}")
        a = (_logit(INDEX_HIGH) - _logit(INDEX_MEDIUM)) / (_logit(p_high) - _logit(p_medium))
        c = _logit(p_medium) - _logit(INDEX_MEDIUM) / a
        return IndexMap(a, c, float(p_medium), float(p_high), detector)

    def index(self, p: float) -> float:
        q = min(max(float(p), 1e-9), 1.0 - 1e-9)
        z = self.a * (_logit(q) - self.c)
        return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0

    def index_array(self, p: np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
        return 1.0 / (1.0 + np.exp(-self.a * (np.log(q / (1.0 - q)) - self.c)))

    def probability(self, index: float) -> float:
        """The calibrated probability an index stands for (the inverse map)."""
        r = min(max(float(index), 1e-12), 1.0 - 1e-12)
        z = _logit(r) / self.a + self.c
        return 1.0 / (1.0 + math.exp(-z))

    def to_dict(self) -> Dict[str, Any]:
        return {"a": round(self.a, 6), "c": round(self.c, 6),
                "p_medium": self.p_medium, "p_high": self.p_high,
                "index_medium": INDEX_MEDIUM, "index_watch": INDEX_WATCH,
                "index_high": INDEX_HIGH, "detector": self.detector,
                "formula": "index = sigmoid(a * (logit(p) - c))"}


def shift_logodds(p: float, uplift: float) -> float:
    """A prior shift (an imported chart's susceptibility) applied to a probability."""
    if not uplift:
        return float(p)
    q = min(max(float(p), 1e-9), 1.0 - 1e-9)
    return 1.0 / (1.0 + math.exp(-(_logit(q) + float(uplift))))


# --------------------------------------------------------------------------- #
# The detector, loaded once per process
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()
_SHARED: Dict[str, Any] = {}


def _artifact_path() -> str:
    from backend.ml import config as mlcfg
    return os.path.join(mlcfg.ARTIFACTS_DIR, "gbm_wearable.joblib")


def shared_model(model: Any = None) -> Any:
    """The serving detector, loaded at most once per process.

    The live session's core loads the same artifact; the server hands that object in
    here so the process holds one copy (the stack is ~60 MB unpickled, and the free
    tier has 512 MB). Returns None when the artifact cannot be loaded."""
    with _LOCK:
        if model is not None:
            _SHARED["model"] = model
        if "model" not in _SHARED:
            try:
                import joblib
                _SHARED["model"] = joblib.load(_artifact_path())
            except Exception as exc:  # pragma: no cover - reported, not raised
                print(f"[ward] detector unavailable ({exc}); the ward falls back to fusion")
                _SHARED["model"] = None
        return _SHARED["model"]


class WardDetector:
    """Scores a replayed CinC record, every hour, with the shipped detector."""

    def __init__(self, model: Any = None, op: Any = None):
        from backend.ml.data.features import FeatureConfig
        from backend.ml.operating_point import load as load_op
        self.model = shared_model(model)
        self.op = op if op is not None else load_op()
        self.cfg = FeatureConfig.wearable_only()
        self.map: Optional[IndexMap] = None
        self.fit_mismatch: Optional[str] = None
        self.conformal = None
        if self.model is not None and self.op is not None:
            key = getattr(self.op, "detector", "") or ""
            want = key.split("#", 1)[1] if "#" in key else None
            have = getattr(self.model, "fit_digest_", None)
            if want != have:
                # The lines are readings of ONE fit's score distribution. A model from
                # another fit read at these cut points pages at an unvalidated rate.
                self.fit_mismatch = f"operating point is fit {want}, model is fit {have}"
            self.map = IndexMap.pinned(self.op.medium, self.op.high, key)
            if not self.fit_mismatch:
                # The conformal bands are cut points on this same fit's output, so a
                # ward bed - now scored by that output - can carry the claim too.
                from backend.ml import conformal
                self.conformal = conformal.load(expect_detector=key)

    @property
    def ready(self) -> bool:
        return self.model is not None and self.map is not None

    def score(self, record: Any) -> Optional[Dict[str, np.ndarray]]:
        """Every hour of ``record``: calibrated P(sepsis label) and, when the conformal
        bands are loaded, the stratum each hour's prediction set is read from."""
        if not self.ready:
            return None
        from backend.ml.data.features import build_patient_features
        feats = build_patient_features(record, self.cfg)
        # numpy 2 on Apple's Accelerate BLAS raises spurious divide/overflow flags in
        # the logistic member's matmul with finite inputs and outputs. Silenced here,
        # and replaced by the check that matters: every score is a finite probability.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            p = self.model.predict_proba(feats)
        p = np.asarray(p[:, 1] if getattr(p, "ndim", 1) == 2 else p, dtype=np.float64)
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise ValueError(f"detector produced a non-probability for {record.patient_id}")
        out: Dict[str, np.ndarray] = {"p": p}
        if self.conformal is not None:
            spec = self.conformal.spec
            try:
                n_measured = feats["n_measured_this_hour"].to_numpy(dtype=float)
                stale = np.max(np.stack([feats[f"{ch}__age"].to_numpy(dtype=float)
                                         for ch in spec.channels], axis=1), axis=1)
                out["strata"] = spec.assign(n_measured, stale)
            except KeyError:
                pass                # no claim beats a claim read from the wrong cell
        return out

    def series(self, record: Any) -> Optional[np.ndarray]:
        """Calibrated P(sepsis label) for every hour of ``record``; None if unavailable."""
        s = self.score(record)
        return None if s is None else s["p"]

    def certainty(self, p: Optional[float], stratum: Optional[str]) -> Optional[Dict[str, Any]]:
        """The conformal prediction set for one scored hour, as the live session reports it."""
        if self.conformal is None or p is None or stratum is None:
            return None
        from backend.ml import conformal
        out = conformal.predict_set(self.conformal, float(p), str(stratum))
        cov = (self.conformal.coverage or {}).get("locked", {}).get("overall", {})
        out["measured_coverage"] = {"septic": cov.get("micro_pos"),
                                    "non_septic": cov.get("micro_neg")}
        return out

    def describe(self) -> Dict[str, Any]:
        return {
            "source": "detector" if self.ready else "fusion (detector unavailable)",
            "detector": getattr(self.op, "detector", None),
            "fit_mismatch": self.fit_mismatch,
            "map": self.map.to_dict() if self.map else None,
            "measured": measured(),
        }


class ICUDetector:
    """The ICU replay beds' detector: one LightGBM over all 40 CinC variables.

    The beds replay full ICU records, labs included, which the 7-vital stack never
    reads. Fitted on the same 24,206 patients as the stack and judged on the same 8,062
    (``artifacts/icu_detector.json``, registered in ``icu_detector_prereg.json``): AUROC
    0.846 vs 0.800, hour-matched 0.799 vs 0.733, +0.046 [+0.035, +0.057]. Inside a fixed
    15% alarm budget its detection is a tie, so the gain is in ranking, not alarm count.

    Same interface as :class:`WardDetector`. Its lines are its OWN operating point
    (``operating_point_icu.json``), and the identity check is the full detector id, not
    just the training digest: both detectors share the digest (same patients), so a
    digest check could not tell the stack's thresholds from this model's."""

    MODEL = "gbm_icu.joblib"
    OP = "operating_point_icu.json"

    def __init__(self, model: Any = None, op: Any = None):
        from backend.ml import config as mlcfg
        from backend.ml.operating_point import load as load_op
        self.model = model
        if self.model is None:
            try:
                import joblib
                self.model = joblib.load(os.path.join(mlcfg.ARTIFACTS_DIR, self.MODEL))
            except Exception as exc:  # pragma: no cover - reported, not raised
                print(f"[ward] ICU detector unavailable ({exc})")
        self.op = op if op is not None else load_op(os.path.join(mlcfg.ARTIFACTS_DIR, self.OP))
        self.conformal = None            # no bands were derived for this fit: no claim
        self.map: Optional[IndexMap] = None
        self.fit_mismatch: Optional[str] = None
        if self.model is not None and self.op is not None:
            have = getattr(self.model, "detector_id", None)
            if self.op.detector != have:
                self.fit_mismatch = f"operating point is {self.op.detector}, model is {have}"
            else:
                self.map = IndexMap.pinned(self.op.medium, self.op.high, have)

    @property
    def ready(self) -> bool:
        return self.model is not None and self.map is not None

    @property
    def detector_id(self) -> Optional[str]:
        return getattr(self.model, "detector_id", None)

    def score(self, record: Any) -> Optional[Dict[str, np.ndarray]]:
        if not self.ready:
            return None
        p = np.asarray(self.model.predict_record(record), dtype=np.float64)
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise ValueError(f"ICU detector produced a non-probability for {record.patient_id}")
        return {"p": p}

    def series(self, record: Any) -> Optional[np.ndarray]:
        s = self.score(record)
        return None if s is None else s["p"]

    def certainty(self, p: Optional[float], stratum: Optional[str]) -> Optional[Dict[str, Any]]:
        return None

    def describe(self) -> Dict[str, Any]:
        return {"source": "icu detector" if self.ready else "unavailable",
                "detector": self.detector_id, "fit_mismatch": self.fit_mismatch,
                "map": self.map.to_dict() if self.map else None, "measured": measured()}


def ward_detector(stack_model: Any = None) -> Any:
    """What scores the ICU replay beds: the all-variable detector when its artifact and
    its own operating point load and agree, else the 7-vital stack."""
    icu = ICUDetector()
    if icu.ready:
        return icu
    if icu.fit_mismatch:
        print(f"[ward] ICU detector refused: {icu.fit_mismatch}; the stack scores the ward")
    return WardDetector(model=stack_model)


# --------------------------------------------------------------------------- #
# Held-out patients
# --------------------------------------------------------------------------- #
def heldout() -> Dict[str, Any]:
    """The patients the serving fit never trained on, with the fit they belong to."""
    try:
        with open(HELDOUT_PATH) as fh:
            d = json.load(fh)
        d["ids"] = frozenset(d.get("patients", []))
        return d
    except (OSError, ValueError):
        return {"ids": frozenset(), "fit_digest": None}


# --------------------------------------------------------------------------- #
# The forecast, on the same scale as the history it continues
# --------------------------------------------------------------------------- #
#: hours of history the trend is fitted on, and the per-hour damping of that trend
TREND_HOURS = 6
DAMPING = 0.8


def _trend(y: Sequence[float]):
    """Least-squares slope of the last ``TREND_HOURS`` points and the fit's residual sd."""
    y = [float(v) for v in y][-TREND_HOURS:]
    n = len(y)
    if n < 3:
        return 0.0, 0.0
    xm, ym = (n - 1) / 2.0, sum(y) / n
    sxx = sum((i - xm) ** 2 for i in range(n))
    slope = sum((i - xm) * (v - ym) for i, v in enumerate(y)) / sxx
    resid = [v - (ym + slope * (i - xm)) for i, v in enumerate(y)]
    return slope, (sum(r * r for r in resid) / max(n - 2, 1)) ** 0.5


def index_forecast(history: Sequence[float], start_hour: int, horizon: int = 8,
                   channels: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A damped trend of the ward index itself, anchored at the current value.

    The old projection fused forecast VITALS through the hand-weighted engine: a second
    model on a second scale, so the projection did not start where the history ended.
    This continues the drawn series from its last point along the least-squares slope
    of the last six hours, damped by 0.8 per hour (so it levels off rather than running
    to 0 or 1 on one noisy hour). It is a trend line on the score, not a physiological
    simulation; the band is the fit's residual spread widening with the horizon, not a
    calibrated interval."""
    h = [float(v) for v in history if v is not None]
    now = h[-1] if h else 0.0
    trend, sigma = _trend(h)
    mean, band, acc = [], [], 0.0
    for k in range(1, horizon + 1):
        acc += DAMPING ** k
        mean.append(min(1.0, max(0.0, now + trend * acc)))
        band.append(sigma * math.sqrt(k))
    hi = [round(min(1.0, m + b), 3) for m, b in zip(mean, band)]
    lo = [round(max(0.0, m - b), 3) for m, b in zip(mean, band)]
    cross = next((k + 1 for k, m in enumerate(mean) if m >= INDEX_HIGH), None)
    cross_hi = next((k + 1 for k, m in enumerate(hi) if m >= INDEX_HIGH), None)
    end = mean[-1] if mean else now
    traj = ("deteriorating" if end > now + 0.04 else "improving" if end < now - 0.04 else "stable")
    if now >= INDEX_HIGH:
        cross = cross_hi = 0
    return {
        "horizon": horizon, "current_risk": round(now, 3),
        "risk_mean": [round(m, 3) for m in mean], "risk_hi": hi, "risk_lo": lo,
        "channels": channels or {}, "trajectory": traj, "risk_slope": round(trend, 4),
        "projected_onset_in": cross, "projected_onset_hour": (start_hour + cross) if cross is not None else None,
        "earliest_breach_in": cross_hi, "confidence": None, "alarm": INDEX_HIGH,
        "method": "least-squares slope of the last 6 h of the ward risk index, damped 0.8/h, anchored at the current value",
    }
