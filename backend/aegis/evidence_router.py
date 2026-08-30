"""AERIS — an advisory, deterministic active-evidence router.

The calibrated detector answers ``what is the current risk?``.  AERIS answers a
strictly narrower operational question: ``is one *available* additional signal
worth requesting or computing now?``  It never changes probability, risk tier,
conformal prediction set, treatment, or notification state.

This first slice is deliberately a transparent baseline rather than a learned
policy.  It uses only current/past feature freshness, model certainty, trajectory
context, source capabilities, and alert-episode state.  A learned value model is
not allowed until logged action data supports it; its absent value is represented
explicitly as ``estimated_value=None`` rather than invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence


WAIT = "WAIT"
REACQUIRE = "REACQUIRE"
COMPUTE_WINDOW = "COMPUTE_WINDOW"
ESCALATE_FOR_REVIEW = "ESCALATE_FOR_REVIEW"

_ACTIONS = {WAIT, REACQUIRE, COMPUTE_WINDOW, ESCALATE_FOR_REVIEW}


@dataclass(frozen=True)
class EvidenceDecision:
    """One advisory decision, serializable into the live snapshot."""

    action: str
    channel: Optional[str]
    request: bool
    reason: str
    episode_id: Optional[int]
    selection_mode: str = "deterministic_baseline"
    estimated_value: Optional[float] = None
    cost_vector: Dict[str, float] = field(default_factory=dict)
    eligibility_reasons: Sequence[str] = field(default_factory=tuple)
    unavailable_sources: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "channel": self.channel,
            "request": self.request,
            "reason": self.reason,
            "episode_id": self.episode_id,
            "selection_mode": self.selection_mode,
            # Null is intentional: no outcome/value model has been fitted yet.
            "estimated_value": self.estimated_value,
            "cost_vector": dict(self.cost_vector),
            "eligibility_reasons": list(self.eligibility_reasons),
            "unavailable_sources": list(self.unavailable_sources),
            "advisory_only": True,
        }


@dataclass(frozen=True)
class _Candidate:
    action: str
    channel: str
    feature_age: Optional[str]
    stale_after_hours: Optional[float]
    requires: str
    cost_vector: Dict[str, float]


# The order expresses an auditable baseline priority in an unresolved elevated
# episode. It is *not* a learned estimate of clinical benefit. MAP/SpO2 are
# refreshed only when stale; derived waveform signals are computed only when a
# caller says the raw source is actually available.
_CANDIDATES = (
    _Candidate(REACQUIRE, "map", "MAP__age", 2.0, "map",
               {"request": 1.0, "compute": 0.05}),
    _Candidate(REACQUIRE, "spo2", "O2Sat__age", 2.0, "spo2",
               {"request": 1.0, "compute": 0.05}),
    _Candidate(COMPUTE_WINDOW, "hrv", None, None, "hrv_window",
               {"request": 0.0, "compute": 1.0}),
    _Candidate(COMPUTE_WINDOW, "respiration", "Resp__age", 2.0, "respiration_window",
               {"request": 0.0, "compute": 0.7}),
)


class EvidenceRouter:
    """Bounded, deterministic AERIS policy.

    Each alert episode may issue at most one non-escalation evidence request. A
    hard clinical override always produces ``ESCALATE_FOR_REVIEW`` and bypasses
    that budget, because routing must never delay an existing safety path.
    ``capabilities`` are a caller-declared contract; omitted means unavailable,
    which causes WAIT rather than an imagined sensor request.
    """

    def __init__(self, max_requests_per_episode: int = 1):
        if max_requests_per_episode < 1:
            raise ValueError("max_requests_per_episode must be at least one")
        self.max_requests_per_episode = max_requests_per_episode
        self._requests_by_episode: Dict[int, int] = {}
        self._last_timestamp: Optional[float] = None

    @staticmethod
    def _unresolved(certainty: Optional[Dict[str, Any]]) -> bool:
        return bool(certainty and (certainty.get("abstain")
                                   or certainty.get("state") in {"uncertain", "conflicting"}))

    @staticmethod
    def _row_value(feature_row: Optional[Any], key: str) -> Optional[float]:
        """Read a scalar from a pandas one-row frame or a plain feature mapping."""
        if feature_row is None:
            return None
        try:
            if hasattr(feature_row, "iloc"):
                value = feature_row.iloc[0][key]
            else:
                value = feature_row[key]
            value = float(value)
            return value if value == value else None  # NaN without numpy dependency
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def decide(
        self,
        *,
        timestamp: float,
        risk_level: str,
        certainty: Optional[Dict[str, Any]],
        alert_episode: Optional[Dict[str, Any]],
        feature_row: Optional[Any],
        capabilities: Optional[Dict[str, bool]] = None,
        trajectory_deviation: float = 0.0,
        hard_override: bool = False,
    ) -> EvidenceDecision:
        if risk_level not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError(f"unknown risk level: {risk_level!r}")
        caps = capabilities or {}
        episode_id = (alert_episode or {}).get("episode_id")
        episode_id = int(episode_id) if episode_id is not None else None
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            return EvidenceDecision(
                WAIT, None, False, "stale assessment ignored; router state was not changed", episode_id,
                eligibility_reasons=("stale_timestamp",),
            )
        self._last_timestamp = timestamp

        if hard_override:
            return EvidenceDecision(
                ESCALATE_FOR_REVIEW, None, True,
                "hard clinical override is already active; do not delay review for acquisition",
                episode_id, cost_vector={"request": 1.0, "compute": 0.0},
                eligibility_reasons=("hard_override",),
            )

        elevated = risk_level in {"MEDIUM", "HIGH"}
        deviating = float(trajectory_deviation) >= 1.5
        unresolved = self._unresolved(certainty)
        if not (elevated or deviating):
            return EvidenceDecision(
                WAIT, None, False, "risk is low without sustained trajectory concern", episode_id,
                eligibility_reasons=("not_elevated", "no_trajectory_trigger"),
            )

        stale = []
        available = []
        unavailable = []
        for candidate in _CANDIDATES:
            if not caps.get(candidate.requires, False):
                unavailable.append(candidate.channel)
                continue
            if candidate.feature_age is None:
                available.append(candidate)
                continue
            age = self._row_value(feature_row, candidate.feature_age)
            if age is not None and age >= float(candidate.stale_after_hours):
                stale.append(candidate)

        # A fresh, resolved assessment should not create work merely because an
        # optional waveform source happens to be connected.
        if not unresolved and not stale:
            return EvidenceDecision(
                WAIT, None, False, "prediction is resolved and eligible channels are fresh", episode_id,
                eligibility_reasons=("resolved", "channels_fresh"),
                unavailable_sources=tuple(unavailable),
            )

        eligible = stale + [c for c in available if c not in stale]
        if not eligible:
            return EvidenceDecision(
                WAIT, None, False, "no eligible evidence source is currently available", episode_id,
                eligibility_reasons=(("unresolved" if unresolved else "stale"), "no_available_source"),
                unavailable_sources=tuple(unavailable),
            )

        used = self._requests_by_episode.get(episode_id, 0) if episode_id is not None else 0
        if episode_id is not None and used >= self.max_requests_per_episode:
            return EvidenceDecision(
                WAIT, None, False, "episode evidence-request budget is exhausted", episode_id,
                eligibility_reasons=("eligible_source", "episode_budget_exhausted"),
                unavailable_sources=tuple(unavailable),
            )

        chosen = eligible[0]
        if episode_id is not None:
            self._requests_by_episode[episode_id] = used + 1
        why = ["unresolved" if unresolved else "stale_channel"]
        if deviating:
            why.append("trajectory_deviation")
        if chosen in stale:
            why.append("channel_stale")
        else:
            why.append("source_available")
        return EvidenceDecision(
            chosen.action, chosen.channel, True,
            f"advisory {chosen.action.lower()} for eligible {chosen.channel} evidence",
            episode_id, cost_vector=chosen.cost_vector,
            eligibility_reasons=tuple(why), unavailable_sources=tuple(unavailable),
        )
