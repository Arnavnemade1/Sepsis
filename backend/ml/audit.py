"""What is a detector's performance actually made of?

A pooled AUROC is a single number averaged over every pair of a positive and a negative
patient-hour, and it silently mixes together answers to two completely different
questions:

    "is this the kind of patient who becomes septic?"     — patient selection
    "is this the hour it is about to happen?"             — event timing

An early-warning system is sold on the second and, on this project's own detector, is
almost entirely delivering the first. Nothing in the standard metric suite reveals that.
This module is the instrument that does, and it is applied to our own model first and
hardest.

THE THREE MEASUREMENTS, AND WHY THESE THREE
--------------------------------------------
Each has survived a deliberate attempt to break it (``attack_clock_claim.py``). Two
earlier candidates did not survive and are deliberately absent — see the last section.

**1. Between / within decomposition.** Split the pooled AUROC into the two questions.
   ``between`` compares a septic patient's positive hours against hours belonging to
   patients who never became septic. ``within`` stays inside septic admissions and asks
   whether the pre-onset hours rank above the rest — the actual early-warning question.
   Both a pooled and a strictly per-patient version of ``within`` are reported, because
   they disagree, and the disagreement is Simpson's paradox rather than a bug: pooled,
   the comparison is polluted by between-patient variance among septic stays; stratified,
   each admission is its own control.

**2. Lead time against a random null.** ``metrics.lead_time_vs_null``. A detector alarms
   at the *first* crossing of its threshold, which for any low threshold lands near the
   start of a stay, so a long lead is manufactured by how long patients happen to lie in
   the ICU. The comparison is held at matched **detection rate**. On this project's own
   shipped detector the excess over the null is **negative**.

**3. Suspicion echo.** The published critique of the Epic Sepsis Model — deployed across
   hundreds of hospitals, later found to miss roughly two thirds of cases — was that
   because antibiotics are ordered when a clinician already suspects sepsis, the model
   was partly detecting *the clinician's own suspicion* rather than providing independent
   warning. That critique was argued qualitatively. Here it is measured: refit with every
   measurement-pattern feature deleted (``__measured``, ``__age``,
   ``n_measured_this_hour`` — which encode what the care team chose to order and when)
   and report what the detector loses.

WHAT THIS INSTRUMENT DOES NOT CLAIM
-----------------------------------
Honesty about the tool matters more here than anywhere, because its whole purpose is to
catch other people's numbers being softer than they look.

* **Suspicion echo is an upper bound, not an attribution.** The deleted features encode
  clinician behaviour *and* genuine information — a channel measured recently is a
  channel someone thought worth measuring, and its recency is also a real data-quality
  signal. The drop is what the detector loses when both go, so it bounds the suspicion
  component from above rather than isolating it.
* **A clock-amplification measure was tried and removed.** The claim that published
  performance is "patient classification plus a clock" was attacked and failed: the
  headline number was definitionally true rather than empirical, and only ~11% of the
  pairs pooled AUROC averages over are within-patient, so a *perfect* clock could move it
  by at most about that. It is absent from this instrument on purpose.
* **A low within-patient score is not automatically a flaw.** A model may be honestly
  aimed at triage. The point of separating the two numbers is that a system sold as early
  warning should be able to show the second one.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from backend.ml.eval import metrics

#: Feature-name patterns that encode the care team's *ordering behaviour* rather than the
#: patient's physiology: whether a channel was measured this hour, and how stale the last
#: measurement is.
SUSPICION_PATTERNS = ("__measured", "__age")
SUSPICION_EXACT = ("n_measured_this_hour",)


@dataclass
class AuditReport:
    detector: str
    n_rows: int
    n_patients: int
    n_septic_patients: int
    prevalence: float

    pooled_auroc: float = float("nan")
    between_auroc: float = float("nan")
    within_auroc_pooled: float = float("nan")
    within_auroc_strict: float = float("nan")
    n_within_stays: int = 0

    lead_observed_h: float = float("nan")
    lead_null_h: float = float("nan")
    lead_null_lo: float = float("nan")
    lead_null_hi: float = float("nan")
    lead_excess_h: float = float("nan")
    lead_detection_rate: float = float("nan")

    suspicion_auroc_full: float = float("nan")
    suspicion_auroc_stripped: float = float("nan")
    suspicion_drop: float = float("nan")
    suspicion_n_features: int = 0

    caveats: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return asdict(self)

    def verdict(self) -> str:
        """One sentence a human can act on."""
        bits = []
        if np.isfinite(self.between_auroc) and np.isfinite(self.within_auroc_strict):
            bits.append(
                f"selects patients at {self.between_auroc:.3f} and times the event at "
                f"{self.within_auroc_strict:.3f}")
        if np.isfinite(self.lead_excess_h):
            bits.append(
                f"lead time is {self.lead_excess_h:+.1f}h against its random null"
                + (" (no early-warning excess)" if self.lead_excess_h <= 0 else ""))
        if np.isfinite(self.suspicion_drop):
            bits.append(
                f"loses {self.suspicion_drop:.3f} AUROC when measurement-pattern "
                f"features are removed")
        return "; ".join(bits) if bits else "insufficient data to audit"


# --------------------------------------------------------------------------- #
def decompose(y: np.ndarray, p: np.ndarray, groups: np.ndarray
              ) -> Dict[str, float]:
    """Split pooled AUROC into patient selection and event timing.

    ``within_strict`` averages a per-stay AUROC, so between-patient variance is removed
    by construction rather than merely diluted. It routinely disagrees with the pooled
    version by a wide margin and the pooled one is the misleading of the two.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(p)
    septic_ids = {x for x in set(groups[(y == 1) & ok].tolist())}
    septic = np.isin(groups, np.array(sorted(septic_ids))) if septic_ids else np.zeros(len(y), bool)

    out: Dict[str, float] = {}
    out["pooled"] = metrics.auroc(y[ok], p[ok]) if len(set(y[ok].tolist())) > 1 else float("nan")

    m_btw = ok & ((y == 1) | (~septic))
    out["between"] = (metrics.auroc(y[m_btw], p[m_btw])
                      if m_btw.any() and len(set(y[m_btw].tolist())) > 1 else float("nan"))

    m_wth = ok & septic
    out["within_pooled"] = (metrics.auroc(y[m_wth], p[m_wth])
                            if m_wth.any() and len(set(y[m_wth].tolist())) > 1 else float("nan"))

    per: List[float] = []
    for pid in sorted(septic_ids):
        m = m_wth & (groups == pid)
        yy = y[m]
        if yy.size < 2 or len(set(yy.tolist())) < 2:
            continue
        per.append(metrics.auroc(yy, p[m]))
    out["within_strict"] = float(np.mean(per)) if per else float("nan")
    out["n_within_stays"] = float(len(per))
    return out


def suspicion_echo(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   feature_names: Sequence[str], folds, seed: int = 0
                   ) -> Dict[str, float]:
    """What the detector loses when the care team's ordering behaviour is removed.

    This is the Epic Sepsis Model critique, measured. Both models are fitted on identical
    folds; only the feature set differs.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    names = list(feature_names)
    drop = {i for i, n in enumerate(names)
            if n in SUSPICION_EXACT or any(pat in n for pat in SUSPICION_PATTERNS)}
    keep = np.array([i for i in range(len(names)) if i not in drop])
    if keep.size == len(names) or keep.size == 0:
        return {"full": float("nan"), "stripped": float("nan"),
                "drop": float("nan"), "n_features": 0}

    def oof(cols: np.ndarray) -> np.ndarray:
        out = np.full(len(y), np.nan)
        for tr, te in folds:
            m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                               class_weight="balanced", random_state=seed)
            m.fit(X[np.ix_(tr, cols)], y[tr])
            out[te] = m.predict_proba(X[np.ix_(te, cols)])[:, 1]
        return out

    p_full = oof(np.arange(len(names)))
    p_strip = oof(keep)
    ok = np.isfinite(p_full) & np.isfinite(p_strip)
    a_full = metrics.auroc(y[ok], p_full[ok])
    a_strip = metrics.auroc(y[ok], p_strip[ok])
    return {"full": float(a_full), "stripped": float(a_strip),
            "drop": float(a_full - a_strip), "n_features": int(len(drop))}


def audit(name: str, y, p, groups, hour_index, onset_hour,
          threshold: float,
          X: Optional[np.ndarray] = None,
          feature_names: Optional[Sequence[str]] = None,
          folds=None, n_draws: int = 25, seed: int = 0) -> AuditReport:
    """Run every measurement that survived attack against one detector's scores."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    septic_ids = {x for x in set(groups[y == 1].tolist())}
    rep = AuditReport(
        detector=name,
        n_rows=int(len(y)),
        n_patients=int(len(set(groups.tolist()))),
        n_septic_patients=int(len(septic_ids)),
        prevalence=float(np.mean(y)),
    )

    d = decompose(y, p, groups)
    rep.pooled_auroc = d["pooled"]
    rep.between_auroc = d["between"]
    rep.within_auroc_pooled = d["within_pooled"]
    rep.within_auroc_strict = d["within_strict"]
    rep.n_within_stays = int(d["n_within_stays"])

    lt = metrics.lead_time_vs_null(y, p, groups, onset_hour, hour_index,
                                   threshold, n_draws=n_draws, seed=seed)
    rep.lead_observed_h = lt["observed_lead_hours"]
    rep.lead_null_h = lt["null_lead_hours"]
    rep.lead_null_lo = lt["null_lo"]
    rep.lead_null_hi = lt["null_hi"]
    rep.lead_excess_h = lt["excess_lead_hours"]
    rep.lead_detection_rate = lt["detection_rate"]

    if X is not None and feature_names is not None and folds is not None:
        se = suspicion_echo(X, y, groups, feature_names, folds, seed=seed)
        rep.suspicion_auroc_full = se["full"]
        rep.suspicion_auroc_stripped = se["stripped"]
        rep.suspicion_drop = se["drop"]
        rep.suspicion_n_features = se["n_features"]

    rep.caveats = [
        "Suspicion echo is an UPPER BOUND, not an attribution: the deleted features "
        "encode clinician ordering behaviour and genuine information (a recently "
        "measured channel is also fresher data), so the drop bounds the suspicion "
        "component from above.",
        "A low within-patient score is not automatically a flaw — a model may be "
        "honestly aimed at triage. It is a flaw only when the system is sold as early "
        "warning.",
        "No clock-amplification measure is reported. That claim was attacked and failed: "
        "the headline was definitionally true, and only ~11% of the pairs pooled AUROC "
        "averages over are within-patient.",
        "Pooled within-patient AUROC and the strictly per-stay version disagree by "
        "construction (Simpson's paradox). The stratified one is the honest answer.",
    ]
    return rep
