"""What the system DOES with a score — one auditable clinical disposition.

Detection accuracy on this cohort is data-limited and has been measured to its floor
eleven times. The decision layer is not, and it is where the remaining clinical value
lives: the same detector, read three different ways, produces false-alarm rates of 14%,
27% and 94%. That is a larger spread than any modelling change this project has ever
found.

THE THIRD OPTION, AND WHY IT IS THE POINT
------------------------------------------
A binary alarm forces every hour into *fire* or *stay silent*, and a 2%-prevalence
detector with an AUROC near 0.76 is wrong often in both directions. The conformal layer
(``backend/ml/conformal.py``) gives a genuine third answer: a prediction *set*. When that
set contains both labels the model is not hedging rhetorically, it is reporting that at
the requested error rate it cannot separate the two — and that is a different clinical
object from either an alarm or silence.

The measured consequences at the shipped threshold, from
``artifacts/conformal.json:coverage.conversion``:

    policy                detection   false alarm   utility
    shipped (p >= thr)      0.688        0.265       +0.418
    UNCERTAIN silenced      0.538        0.141       +0.397
    UNCERTAIN escalated     0.963        0.943       -0.092

Silencing the uncertain hours cuts false alarms by **47%** for 15 points of detection.
Escalating them catches nearly everything and alarms on 94% of non-septic patients, which
is not a system anyone would keep switched on. Neither column is a good answer, and that
is the argument for the third one: **route the uncertain hours to a human instead of
resolving them automatically in either direction.** ``REVIEW`` is a real disposition with
a real cost — someone's attention — and it is priced here rather than hidden.

ORDERING, WHICH IS A SAFETY PROPERTY
------------------------------------
Abstention must never be able to *downgrade* an escalation. ``alert_episodes.py`` already
states this for notifications ("uncertainty is reported but can never suppress" a HIGH
escalation) and the same rule holds here: a hard clinical rule (qSOFA >= 2, shock) or a
HIGH risk tier produces ALARM regardless of what the conformal set says, because those
escalations are not model predictions and no conformal claim covers them. Abstention only
ever acts on the band where the model alone was going to decide.

HONEST LIMITS
-------------
* The conformal guarantee is **marginal and cohort-conditional**. It was calibrated on
  one hospital and this project has already measured it collapsing across sites — septic
  coverage falls 0.929 -> 0.739 while non-septic coverage holds. A disposition carries
  the coverage that was actually measured, never a nominal 1-alpha.
* The conversion numbers above are from the calibration cohort at one threshold. They
  describe what the policies did there, not what they will do elsewhere.
* ``REVIEW`` spends clinician attention. Calling it "free" because it is not a false
  alarm would be the same accounting error as reporting detection without an alarm rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

#: The four dispositions, in increasing order of what they demand from a person.
NO_ACTION = "NO_ACTION"
MONITOR = "MONITOR"
REVIEW = "REVIEW"
ALARM = "ALARM"

_DEMAND = {NO_ACTION: 0, MONITOR: 1, REVIEW: 2, ALARM: 3}


@dataclass
class Disposition:
    """One decision, with everything needed to argue with it."""
    action: str
    route: str                              # who this lands on
    urgency: str                            # none | routine | prompt | immediate
    because: List[str] = field(default_factory=list)
    abstained: bool = False
    certainty_state: Optional[str] = None   # septic | not_septic | uncertain | conflicting
    coverage: Optional[Dict[str, Any]] = None
    tradeoff: Optional[Dict[str, Any]] = None
    gate: Optional[Dict[str, Any]] = None
    limits: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _conversion() -> Optional[Dict[str, Any]]:
    """The measured cost of each way of handling the uncertain band."""
    from backend.ml import conformal
    from backend.ml.operating_point import load as load_op
    op = load_op()
    bands = conformal.load(expect_detector=getattr(op, "detector", None))
    if bands is None:
        return None
    conv = (bands.coverage or {}).get("conversion")
    if not conv:
        return None
    out = {"threshold": conv.get("threshold"), "policies": {}}
    for row in conv.get("policies", []):
        out["policies"][row["policy"]] = {
            "detection": row.get("detection"),
            "false_alarm": row.get("false_alarm"),
            "utility": row.get("utility"),
        }
    silenced = out["policies"].get("UNCERTAIN silenced")
    shipped = out["policies"].get("shipped (p >= thr)")
    if silenced and shipped and shipped.get("false_alarm"):
        out["false_alarm_cut_if_silenced"] = round(
            1.0 - silenced["false_alarm"] / shipped["false_alarm"], 4)
        out["detection_lost_if_silenced"] = round(
            shipped["detection"] - silenced["detection"], 4)
    return out


def decide(risk_level: str,
           ml_probability: Optional[float],
           certainty: Optional[Dict[str, Any]] = None,
           override: Optional[str] = None,
           episode: Optional[Dict[str, Any]] = None) -> Disposition:
    """Turn a scored hour into an auditable clinical disposition.

    ``certainty`` is the conformal prediction set from ``AegisCore`` — ``None`` whenever
    no conformal claim covers this score, which is a normal state and is reported as
    such rather than defaulted to "confident".
    """
    because: List[str] = []
    state = (certainty or {}).get("state")
    abstain = bool((certainty or {}).get("abstain"))
    unavailable = bool((certainty or {}).get("unavailable")) or certainty is None

    # --- hard escalations first. These are not model predictions, so no conformal
    # --- claim covers them and abstention must not be able to soften them.
    if override:
        because.append(f"hard clinical rule fired: {override} — not a model prediction, "
                       f"so no conformal claim applies and abstention cannot soften it")
        return _finish(Disposition(
            action=ALARM, route="bedside clinician", urgency="immediate",
            because=because, abstained=False, certainty_state=state))

    if risk_level == "HIGH":
        because.append("risk tier HIGH")
        if abstain:
            because.append("model abstained, but abstention may never downgrade a HIGH "
                           "escalation — same rule alert_episodes.py applies to notifications")
        return _finish(Disposition(
            action=ALARM, route="bedside clinician", urgency="immediate",
            because=because, abstained=False, certainty_state=state))

    # --- the band where the model alone would have decided ---------------------
    if unavailable:
        reason = (certainty or {}).get("reason") or "conformal layer not available"
        because.append(f"no conformal claim covers this score ({reason}); "
                       f"falling back to the risk tier alone")
        act = MONITOR if risk_level == "MEDIUM" else NO_ACTION
        return _finish(Disposition(
            action=act, route="chart" if act == MONITOR else "none",
            urgency="routine" if act == MONITOR else "none",
            because=because, abstained=False, certainty_state=None))

    if state == "uncertain":
        because.append("conformal set contains BOTH labels — at the requested error rate "
                       "the model cannot separate septic from not-septic here")
        because.append("routed to a person rather than resolved automatically: silencing "
                       "these hours cuts false alarms 47% but loses 15 points of "
                       "detection, and escalating them alarms on 94% of non-septic "
                       "patients (artifacts/conformal.json)")
        return _finish(Disposition(
            action=REVIEW, route="charge nurse — review queue", urgency="prompt",
            because=because, abstained=True, certainty_state=state))

    if state == "conflicting":
        because.append("conformal set is EMPTY — the score sits outside both bands, "
                       "which is a data-quality or out-of-distribution signal rather "
                       "than a clinical one")
        return _finish(Disposition(
            action=REVIEW, route="data quality — check the feed", urgency="prompt",
            because=because, abstained=True, certainty_state=state))

    if state == "septic":
        because.append("conformal set is {septic} alone at the requested error rate")
        return _finish(Disposition(
            action=ALARM, route="bedside clinician", urgency="prompt",
            because=because, abstained=False, certainty_state=state))

    # state == "not_septic"
    because.append("conformal set is {not septic} alone")
    if risk_level == "MEDIUM":
        because.append("risk tier MEDIUM — kept under observation rather than dismissed")
        return _finish(Disposition(
            action=MONITOR, route="chart", urgency="routine",
            because=because, abstained=False, certainty_state=state))
    return _finish(Disposition(
        action=NO_ACTION, route="none", urgency="none",
        because=because, abstained=False, certainty_state=state))


def _finish(d: Disposition) -> Disposition:
    """Attach the measured cost of the policy and the limits that qualify it."""
    d.tradeoff = _conversion()
    d.limits = [
        "The conformal guarantee is marginal and cohort-conditional. Calibrated on one "
        "hospital, it has been measured collapsing across sites (septic coverage "
        "0.929 -> 0.739 while non-septic coverage held). Coverage shown is what was "
        "measured, never a nominal 1-alpha.",
        "REVIEW spends clinician attention. It is not a free middle option, and calling "
        "it one would be the same accounting error as reporting detection without an "
        "alarm rate.",
        "The conversion numbers describe the calibration cohort at one threshold. They "
        "say what these policies did there, not what they will do elsewhere.",
    ]
    return d


def escalates_over(a: Disposition, b: Disposition) -> bool:
    """Does ``a`` demand more of a person than ``b``? Used to prove ordering."""
    return _DEMAND[a.action] > _DEMAND[b.action]
