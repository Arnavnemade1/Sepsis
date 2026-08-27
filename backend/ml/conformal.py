"""Mondrian split-conformal abstention — the third state the alarm never had.

WHY THIS EXISTS
---------------
The OS reads one scalar and sorts it into LOW / MEDIUM / HIGH. There is no way for
the detector to say *I cannot separate these two classes for this hour*, so an hour
whose evidence is genuinely ambiguous is rounded down to LOW and disappears. Two
measured findings in this repo are the same failure seen from different sides:

* ``artifacts/alarm_policy.json`` records a **refusal**: four alarm-policy families
  (plain, persistent, integrator, per-patient) and none beats a plain threshold.
  The false-alarm rate is not fixable by a better *firing rule*.
* ``docs/SUBGROUP_AUDIT.md`` measured that the alarm burden **is not flat** — 80+
  absorb +6.3pp more false alarms than under-45s — and deliberately refused to
  launder that into per-group thresholds.

Both are a single threshold being asked to carry a decision it has no vocabulary to
qualify. This module adds the vocabulary: a per-hour prediction SET over {0,1}.

    {0}     the model can rule sepsis out at this confidence level
    {1}     the model can rule it in
    {0,1}   UNCERTAIN — it can do neither, and says so
    {}      conflicting (both labels excluded); reported, never silently dropped

METHOD
------
Split conformal prediction. For a candidate label ``y`` the nonconformity score is
``s_y = 1 - p̂(y)``; label ``y`` enters the set when ``s_y <= q̂_y``, where ``q̂_y``
is the ``⌈(n+1)(1-α)⌉``-th smallest calibration score. Because
``s_1 = 1-p`` and ``s_0 = p``, every cell collapses to two thresholds:

    1 ∈ set  ⟺  p >= lo   (lo = 1 - q̂_1)
    0 ∈ set  ⟺  p <= hi   (hi = q̂_0)

**Class-conditional quantiles are mandatory here, not a refinement.** Hourly
prevalence is ~2% and this detector's ``prob_max`` is 0.167
(``artifacts/operating_point.json``): a marginal quantile is set almost entirely by
negatives, so the label ``1`` would never enter any set and the whole construction
would degenerate to "always {0}". Conditioning on the class instead yields a claim
per class — ``1-α`` of septic hours contain ``1`` (an hour-level sensitivity floor),
``1-α`` of non-septic hours contain ``0``.

**The Mondrian taxonomy is data quality, never the patient.** Cells are formed from
``n_measured_this_hour`` (how many channels were actually observed) crossed with
channel staleness (``max_ch {ch}__age``, hours since the last real measurement).
Both already exist in ``backend/ml/data/features.py`` and — because
``backend.aegis.streaming_features.StreamingFeatureBuilder`` calls the very same
``build_patient_features`` — are identical at train and serve time, so there is no
skew to manage. Age and sex are deliberately EXCLUDED from the taxonomy and only
*reported* on, preserving the subgroup audit's commitment that the bar is never
re-tuned according to who the patient is.

WHAT THIS DOES NOT BUY (state it, do not discover it later)
-----------------------------------------------------------
* It does **not** reduce false alarms for free. It re-labels some of them
  UNCERTAIN. That is a win only if the ward's response to UNCERTAIN is cheaper than
  its response to an alarm — a clinical judgement, not a modelling result. Every
  derivation therefore reports the alarm→UNCERTAIN conversion under BOTH readings
  (UNCERTAIN as alarm, UNCERTAIN as silence), because those bracket the truth.
* The finite-sample guarantee holds under **exchangeability**, which ICU data
  violates twice over: hours within a stay are heavily correlated, and a different
  hospital is a different distribution outright. So ``1-α`` is a hypothesis to be
  measured, not a number to be printed. This module computes empirical coverage at
  BOTH the hour level (micro) and the patient level (macro, each patient weighted
  once) precisely because the gap between them is the correlation the nominal
  guarantee ignores.
* It is not a regulatory artifact. Nothing here satisfies any FDA SaMD requirement.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Below this many calibration hours (or patients) a cell cannot support its own
#: quantile and falls back to the marginal one. The formal minimum for α=0.05 is
#: ⌈1/α⌉-1 = 19 hours, at which the quantile IS the maximum observed score and the
#: interval is the widest the data can express. These floors sit far above that,
#: because a quantile estimated from 19 correlated hours of two patients is a number
#: with a guarantee attached and no information in it.
MIN_CELL_HOURS = 200
MIN_CELL_PATIENTS = 25


# --------------------------------------------------------------------------- #
# Strata
# --------------------------------------------------------------------------- #
@dataclass
class StratumSpec:
    """How a feature row becomes a stratum label.

    The bin edges are FIXED at calibration time and stored in the artifact. Deriving
    them again at serve time from whatever data happens to be in front of the system
    would silently redefine every cell — the quantiles would then belong to cells
    that no longer exist, and nothing would error.
    """
    measured_edges: List[float]      # interior edges on n_measured_this_hour
    staleness_edges: List[float]     # interior edges on max channel age (hours)
    channels: List[str]              # channels the staleness max was taken over

    # -- construction ------------------------------------------------------- #
    @staticmethod
    def fit(n_measured: np.ndarray, staleness: np.ndarray, channels: Sequence[str],
            n_bins: int = 3) -> "StratumSpec":
        """Quantile edges, de-duplicated.

        Both inputs are discrete and lumpy (``n_measured_this_hour`` is an integer
        count over 7 channels; staleness piles up at 0). Requesting terciles of such
        a variable can return repeated edges, which would create empty cells that
        look real in the artifact. De-duplicating collapses those to however many
        bins the data can actually distinguish, and the cell census in the result
        shows how many that turned out to be.
        """
        return StratumSpec(
            measured_edges=_edges(n_measured, n_bins),
            staleness_edges=_edges(staleness, n_bins),
            channels=list(channels),
        )

    # -- application -------------------------------------------------------- #
    def assign(self, n_measured: np.ndarray, staleness: np.ndarray) -> np.ndarray:
        m = np.digitize(np.asarray(n_measured, dtype=float), self.measured_edges)
        s = np.digitize(np.asarray(staleness, dtype=float), self.staleness_edges)
        return np.array([f"m{int(a)}/s{int(b)}" for a, b in zip(m, s)], dtype=object)

    def assign_one(self, n_measured: float, staleness: float) -> str:
        m = int(np.digitize([float(n_measured)], self.measured_edges)[0])
        s = int(np.digitize([float(staleness)], self.staleness_edges)[0])
        return f"m{m}/s{s}"

    # -- reading a feature matrix ------------------------------------------- #
    @staticmethod
    def columns(feature_names: Sequence[str], channels: Sequence[str]) -> Tuple[int, List[int]]:
        """Locate ``n_measured_this_hour`` and the ``{ch}__age`` columns.

        Raises rather than defaulting: a missing column here would silently send
        every row into one stratum and the artifact would still look well-formed.
        """
        names = list(feature_names)
        try:
            n_idx = names.index("n_measured_this_hour")
        except ValueError as exc:
            raise KeyError("feature matrix has no 'n_measured_this_hour' column — "
                           "the stratum taxonomy cannot be built from it") from exc
        age_idx = []
        for ch in channels:
            col = f"{ch}__age"
            if col not in names:
                raise KeyError(f"feature matrix has no {col!r} column")
            age_idx.append(names.index(col))
        return n_idx, age_idx

    @staticmethod
    def features_from_matrix(X: np.ndarray, feature_names: Sequence[str],
                             channels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """(n_measured, staleness) for every row of a feature matrix."""
        n_idx, age_idx = StratumSpec.columns(feature_names, channels)
        n_measured = np.asarray(X[:, n_idx], dtype=float)
        staleness = np.asarray(X[:, age_idx], dtype=float).max(axis=1)
        return n_measured, staleness

    def channels_needed(self) -> List[str]:
        """Every feature column an assignment reads.

        Exposed so a serving path can check availability ONCE, up front, instead of
        discovering a missing column per tick — or, worse, not discovering it,
        because a reindex fills an absent column with a perfectly plausible 0.0.
        """
        return ["n_measured_this_hour"] + [f"{ch}__age" for ch in self.channels]

    def to_json(self) -> Dict[str, Any]:
        return {"measured_edges": [float(e) for e in self.measured_edges],
                "staleness_edges": [float(e) for e in self.staleness_edges],
                "channels": list(self.channels)}


def _edges(v: np.ndarray, n_bins: int) -> List[float]:
    v = np.asarray(v, dtype=float)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return sorted(set(round(float(q), 6) for q in np.quantile(v, qs)))


# --------------------------------------------------------------------------- #
# Bands
# --------------------------------------------------------------------------- #
@dataclass
class CellBand:
    """The two thresholds one stratum's prediction sets reduce to, plus its census.

    ``lo``/``hi`` are stored rather than the raw nonconformity quantiles because the
    serve path has only ``p``; keeping the derived form means serving cannot
    re-derive it differently. Both quantiles are kept alongside so the artifact is
    auditable.
    """
    stratum: str
    lo: float                 # p >= lo  →  label 1 admitted
    hi: float                 # p <= hi  →  label 0 admitted
    q_pos: float              # ⌈(n+1)(1-α)⌉-th smallest of (1-p) over septic hours
    q_neg: float              # ... of p over non-septic hours
    n_hours_pos: int
    n_hours_neg: int
    n_patients_pos: int
    n_patients_neg: int
    fallback_pos: bool        # cell too small for its own positive-class quantile
    fallback_neg: bool

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConformalBands:
    """A calibrated abstention layer plus the evidence that produced it."""
    alpha: float
    spec: StratumSpec
    cells: Dict[str, CellBand]
    marginal: CellBand
    detector: str = "unknown"
    n_calib_hours: int = 0
    n_calib_patients: int = 0
    min_cell_hours: int = MIN_CELL_HOURS
    min_cell_patients: int = MIN_CELL_PATIENTS
    #: Filled by :func:`coverage_report` on patients held out of the calibration.
    #: An artifact whose ``coverage`` is empty has a nominal guarantee and no
    #: measured one, which is exactly the state this module exists to avoid.
    coverage: Dict[str, Any] = field(default_factory=dict)

    # -- serving ------------------------------------------------------------ #
    def band(self, stratum: str) -> CellBand:
        """The cell for ``stratum``, or the marginal fallback for an unseen one.

        A stratum absent from the artifact is not an error: bin edges are fixed, so
        a live patient can land in a combination the calibration cohort never
        produced. Serving it from the marginal band is the honest answer, and
        :func:`predict_set` reports that it happened.
        """
        return self.cells.get(stratum, self.marginal)

    def to_json(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "spec": self.spec.to_json(),
            "cells": {k: v.to_json() for k, v in self.cells.items()},
            "marginal": self.marginal.to_json(),
            "detector": self.detector,
            "n_calib_hours": self.n_calib_hours,
            "n_calib_patients": self.n_calib_patients,
            "min_cell_hours": self.min_cell_hours,
            "min_cell_patients": self.min_cell_patients,
            "coverage": self.coverage,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "ConformalBands":
        spec = StratumSpec(**d["spec"])
        cells = {k: CellBand(**v) for k, v in d["cells"].items()}
        return ConformalBands(
            alpha=float(d["alpha"]), spec=spec, cells=cells,
            marginal=CellBand(**d["marginal"]), detector=d.get("detector", "unknown"),
            n_calib_hours=int(d.get("n_calib_hours", 0)),
            n_calib_patients=int(d.get("n_calib_patients", 0)),
            min_cell_hours=int(d.get("min_cell_hours", MIN_CELL_HOURS)),
            min_cell_patients=int(d.get("min_cell_patients", MIN_CELL_PATIENTS)),
            coverage=d.get("coverage", {}),
        )


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ``⌈(n+1)(1-α)⌉``-th smallest score — ``inf`` when n is too small.

    The ``+1`` is the whole finite-sample guarantee: it accounts for the test point
    itself being exchangeable with the calibration set. Dropping to the plain
    empirical ``(1-α)`` quantile — the version that shows up in most tutorials —
    under-covers by roughly ``1/n``, which at these cell sizes is the difference
    between a guarantee and a slogan.

    When ``⌈(n+1)(1-α)⌉ > n`` the data cannot express the requested level at all;
    returning ``inf`` (admit the label always) is the only sound answer, and the
    cell-size floors exist so that this never happens quietly in a shipped cell.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def conformal_quantile_weighted(scores: np.ndarray, weights: np.ndarray,
                                alpha: float) -> float:
    """Weighted conformal quantile (Tibshirani/Barber/Candes) under covariate shift.

    Ordinary split conformal assumes calibration and test points are exchangeable. Under
    covariate shift they are not, but weighting each calibration point by the density
    ratio ``w(x) = dP_target/dP_source`` restores it: the quantile is taken over the
    normalised weights rather than over counts.

    The test point contributes its own mass, which is what the ``+1`` does in the
    unweighted form. Its weight is unknown at calibration time, so the standard
    conservative substitute is the mean calibration weight.

    Returns ``inf`` when the cumulative weight never reaches ``1 - alpha`` — that is,
    when the reweighted calibration set cannot express the requested level at all.
    Admitting the label always is the only sound answer there, and it is visible rather
    than silently approximated by the largest score on hand.
    """
    s = np.asarray(scores, dtype=float)
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if s.size == 0 or w.sum() <= 0:
        return float("inf")
    order = np.argsort(s)
    s, w = s[order], w[order]
    cw = np.cumsum(w) / (w.sum() + float(w.mean()))
    idx = int(np.searchsorted(cw, 1.0 - alpha, side="left"))
    if idx >= s.size:
        return float("inf")
    return float(s[idx])


def _cell_band(stratum: str, y: np.ndarray, p: np.ndarray, groups: np.ndarray,
               alpha: float, fallback: Optional[CellBand],
               min_hours: int, min_patients: int,
               w: Optional[np.ndarray] = None) -> CellBand:
    pos, neg = y == 1, y == 0
    n_hp, n_hn = int(pos.sum()), int(neg.sum())
    n_pp = int(len(set(groups[pos].tolist()))) if n_hp else 0
    n_pn = int(len(set(groups[neg].tolist()))) if n_hn else 0

    small_pos = n_hp < min_hours or n_pp < min_patients
    small_neg = n_hn < min_hours or n_pn < min_patients
    use_pos = bool(small_pos and fallback is not None)
    use_neg = bool(small_neg and fallback is not None)

    # Uniform weights reduce this to the unweighted estimator exactly, which
    # verify_conformal asserts rather than assumes — so adding the weighted path
    # cannot silently change the shipped, unweighted bands.
    def q(scores, mask):
        return (conformal_quantile(scores, alpha) if w is None
                else conformal_quantile_weighted(scores, w[mask], alpha))

    q_pos = fallback.q_pos if use_pos else q(1.0 - p[pos], pos)
    q_neg = fallback.q_neg if use_neg else q(p[neg], neg)
    return CellBand(
        stratum=stratum,
        lo=1.0 - q_pos, hi=q_neg, q_pos=q_pos, q_neg=q_neg,
        n_hours_pos=n_hp, n_hours_neg=n_hn,
        n_patients_pos=n_pp, n_patients_neg=n_pn,
        fallback_pos=use_pos, fallback_neg=use_neg,
    )


def calibrate(y: np.ndarray, p: np.ndarray, groups: np.ndarray, strata: np.ndarray,
              *, alpha: float = 0.05, min_hours: int = MIN_CELL_HOURS,
              min_patients: int = MIN_CELL_PATIENTS, spec: Optional[StratumSpec] = None,
              detector: str = "unknown",
              weights: Optional[np.ndarray] = None) -> ConformalBands:
    """Fit class-conditional conformal quantiles per stratum.

    ``p`` **must** come from models that never saw these patients — out-of-fold or a
    held-out fit. In-sample scores are far sharper than anything the model emits at
    serve time, so calibrating on them produces bands that are tight against a
    distribution which will never recur; the coverage they promise is then wrong in
    the one direction that matters, and nothing errors. This is the identical
    discipline ``backend/ml/operating_point.py`` enforces for the threshold, for the
    identical reason.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    strata = np.asarray(strata, dtype=object)
    if not (len(y) == len(p) == len(groups) == len(strata)):
        raise ValueError("y, p, groups and strata must be the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    if np.isnan(p).any():
        raise ValueError("scores contain NaN — every calibration row must be scored")

    w = None if weights is None else np.asarray(weights, dtype=float)
    marginal = _cell_band("marginal", y, p, groups, alpha, None, 0, 0, w)
    cells: Dict[str, CellBand] = {}
    for s in sorted(set(strata.tolist())):
        m = strata == s
        cells[s] = _cell_band(s, y[m], p[m], groups[m], alpha, marginal,
                              min_hours, min_patients,
                              None if w is None else w[m])
    return ConformalBands(
        alpha=alpha,
        spec=spec if spec is not None else StratumSpec([], [], []),
        cells=cells, marginal=marginal, detector=detector,
        n_calib_hours=int(len(y)), n_calib_patients=int(len(set(groups.tolist()))),
        min_cell_hours=min_hours, min_cell_patients=min_patients,
    )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_set(bands: ConformalBands, p: float, stratum: str) -> Dict[str, Any]:
    """The prediction set for one scored hour, with the cell it was read from."""
    b = bands.band(stratum)
    labels = []
    if p >= b.lo:
        labels.append(1)
    if p <= b.hi:
        labels.append(0)
    st = ("uncertain" if len(labels) == 2 else
          "conflicting" if not labels else
          "septic" if labels == [1] else "not_septic")
    return {
        "set": sorted(labels),
        "state": st,
        "abstain": st in ("uncertain", "conflicting"),
        "stratum": stratum,
        "cell": b.stratum,
        "unseen_stratum": stratum not in bands.cells,
        "fallback_used": bool(b.fallback_pos or b.fallback_neg),
        "lo": round(b.lo, 6), "hi": round(b.hi, 6),
        "alpha": bands.alpha,
    }


def predict_states(bands: ConformalBands, p: np.ndarray,
                   strata: np.ndarray) -> np.ndarray:
    """Vectorised ``state`` for a whole cohort (evaluation path)."""
    p = np.asarray(p, dtype=float)
    strata = np.asarray(strata, dtype=object)
    lo = np.array([bands.band(s).lo for s in strata], dtype=float)
    hi = np.array([bands.band(s).hi for s in strata], dtype=float)
    has1, has0 = p >= lo, p <= hi
    out = np.empty(len(p), dtype=object)
    out[has1 & has0] = "uncertain"
    out[has1 & ~has0] = "septic"
    out[~has1 & has0] = "not_septic"
    out[~has1 & ~has0] = "conflicting"
    return out


# --------------------------------------------------------------------------- #
# Coverage — the measured claim, as opposed to the nominal one
# --------------------------------------------------------------------------- #
def coverage_report(bands: ConformalBands, y: np.ndarray, p: np.ndarray,
                    groups: np.ndarray, strata: np.ndarray,
                    extra_groupings: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """Empirical coverage on rows the calibration never saw.

    Reported twice, always:

    ``micro`` pools hours. It is what the conformal theorem speaks about, and it is
    also the number that a handful of very long ICU stays can carry on their own.

    ``macro`` computes each patient's own coverage rate and averages those, one vote
    per patient. Hours within a stay are strongly correlated, so a gap between micro
    and macro is a direct readout of how far the exchangeability assumption is from
    holding on this data. Reporting only micro would hide exactly that.

    ``extra_groupings`` maps a name to a per-row label array (age band, sex, hospital)
    and is reported on WITHOUT ever entering the taxonomy — the audit, not the knob.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    strata = np.asarray(strata, dtype=object)
    states = predict_states(bands, p, strata)
    covered = np.where(y == 1, np.isin(states, ["septic", "uncertain"]),
                       np.isin(states, ["not_septic", "uncertain"]))

    rep: Dict[str, Any] = {
        "alpha": bands.alpha,
        "nominal": round(1.0 - bands.alpha, 4),
        "n_hours": int(len(y)),
        "n_patients": int(len(set(groups.tolist()))),
        "overall": _cov(covered, y, groups),
        "abstain_rate": round(float(np.mean(states == "uncertain")), 4),
        "conflicting_rate": round(float(np.mean(states == "conflicting")), 6),
        "state_counts": {s: int((states == s).sum())
                         for s in ("not_septic", "uncertain", "septic", "conflicting")},
        "per_stratum": {},
        "reported_only": {},
    }
    for s in sorted(set(strata.tolist())):
        m = strata == s
        rep["per_stratum"][str(s)] = dict(
            _cov(covered[m], y[m], groups[m]),
            n_hours=int(m.sum()),
            abstain_rate=round(float(np.mean(states[m] == "uncertain")), 4),
            unseen=bool(s not in bands.cells),
        )
    for name, labels in (extra_groupings or {}).items():
        labels = np.asarray(labels, dtype=object)
        rep["reported_only"][name] = {}
        for v in sorted(set(labels.tolist())):
            m = labels == v
            rep["reported_only"][name][str(v)] = dict(
                _cov(covered[m], y[m], groups[m]),
                n_hours=int(m.sum()),
                abstain_rate=round(float(np.mean(states[m] == "uncertain")), 4),
            )
    return rep


def _cov(covered: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    def micro(mask):
        return round(float(covered[mask].mean()), 4) if mask.any() else float("nan")

    def macro(mask):
        if not mask.any():
            return float("nan")
        g, c = groups[mask], covered[mask]
        per = [c[g == u].mean() for u in set(g.tolist())]
        return round(float(np.mean(per)), 4)

    pos, neg = y == 1, y == 0
    return {
        "micro": micro(np.ones(len(y), dtype=bool)),
        "macro": macro(np.ones(len(y), dtype=bool)),
        "micro_pos": micro(pos), "macro_pos": macro(pos),
        "micro_neg": micro(neg), "macro_neg": macro(neg),
    }


# --------------------------------------------------------------------------- #
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "artifacts", "conformal.json")


def save(bands: ConformalBands, path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(bands.to_json(), fh, indent=2)


def load(path: str = DEFAULT_PATH) -> Optional[ConformalBands]:
    """The calibrated bands, or ``None`` when they have not been derived.

    ``None`` rather than an exception or an invented default, matching
    ``operating_point.load()``: a machine without the artifact must serve exactly as
    it did before this module existed and say the layer is absent, not fail closed
    on a file nobody generated.
    """
    try:
        with open(path) as fh:
            return ConformalBands.from_json(json.load(fh))
    except (OSError, TypeError, ValueError, KeyError):
        return None


def summary_table(bands: ConformalBands) -> str:
    """Per-cell bands and census — the evidence behind the abstention region."""
    lines = [f"{'stratum':>10}  {'lo':>8}  {'hi':>8}  {'+hrs':>7}  {'+pts':>6}  "
             f"{'-hrs':>8}  {'-pts':>6}  fallback"]
    for name in sorted(bands.cells):
        b = bands.cells[name]
        fb = ",".join([x for x, on in (("pos", b.fallback_pos), ("neg", b.fallback_neg)) if on])
        lines.append(f"{name:>10}  {b.lo:8.4f}  {b.hi:8.4f}  {b.n_hours_pos:7d}  "
                     f"{b.n_patients_pos:6d}  {b.n_hours_neg:8d}  {b.n_patients_neg:6d}  "
                     f"{fb or '-'}")
    m = bands.marginal
    lines.append(f"{'marginal':>10}  {m.lo:8.4f}  {m.hi:8.4f}  {m.n_hours_pos:7d}  "
                 f"{m.n_patients_pos:6d}  {m.n_hours_neg:8d}  {m.n_patients_neg:6d}  -")
    return "\n".join(lines)
