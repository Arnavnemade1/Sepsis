"""Stateful, study-ready alert episodes for the streaming runtime.

The detector answers a *clinical-state* question every tick.  A notification
system answers a different question: whether a person should be interrupted
again.  Treating the former as the latter makes a sustained elevated score look
like an unlimited stream of new alarms, hides the operational burden, and makes
it impossible to evaluate alarm fatigue honestly.

This module deliberately does not change a risk tier, score, or protocol.  It
adds a small, explicit notification state machine alongside them.  Its defaults
are an advisory study policy, not a validated paging policy: callers receive the
event and decide how to present or route it.  In particular, a LOW -> MEDIUM
entry and every escalation to HIGH notify immediately; uncertainty is reported
but can never suppress either one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@dataclass(frozen=True)
class AlertEpisode:
    """One decision about an ongoing alert episode.

    ``notify`` means the episode layer recommends a fresh human interruption.
    ``state`` remains useful even when it is false: a UI can show that an alert
    is active without repeatedly presenting it as new.
    """

    state: str                    # IDLE | OPENED | ACTIVE | ESCALATED | RENOTIFY | RESOLVED | CLOCK_ERROR
    notify: bool
    risk_level: str
    episode_id: Optional[int]
    opened_at: Optional[float]
    last_notified_at: Optional[float]
    elapsed_seconds: float
    reason: str
    certainty_state: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "notify": self.notify,
            "risk_level": self.risk_level,
            "episode_id": self.episode_id,
            "opened_at": self.opened_at,
            "last_notified_at": self.last_notified_at,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "reason": self.reason,
            "certainty_state": self.certainty_state,
            "advisory_only": True,
        }


class AlertEpisodeController:
    """Translate risk-tier transitions into bounded notification episodes.

    A HIGH tier that persists can re-notify after ``high_renotify_seconds`` so a
    genuine unresolved deterioration does not disappear behind a cooldown.  A
    MEDIUM tier intentionally has no periodic re-notification by default; it
    stays visibly active while avoiding a stream of advisory pages.  Both values
    are explicit study parameters and must be evaluated on held-out patient
    series before any external notification integration is enabled.
    """

    def __init__(self, high_renotify_seconds: Optional[float] = 4 * 3600.0):
        if high_renotify_seconds is not None and high_renotify_seconds <= 0:
            raise ValueError("high_renotify_seconds must be positive or None")
        self.high_renotify_seconds = high_renotify_seconds
        self._next_episode_id = 1
        self._active = False
        self._current_level = "LOW"
        self._episode_id: Optional[int] = None
        self._opened_at: Optional[float] = None
        self._last_notified_at: Optional[float] = None
        self._last_timestamp: Optional[float] = None

    @staticmethod
    def _certainty_state(certainty: Optional[Dict[str, Any]]) -> str:
        if not certainty:
            return "unavailable"
        if certainty.get("unavailable"):
            return "unavailable"
        if certainty.get("override"):
            return "rule_override"
        return str(certainty.get("state", "resolved"))

    def update(
        self,
        risk_level: str,
        timestamp: float,
        certainty: Optional[Dict[str, Any]] = None,
    ) -> AlertEpisode:
        """Advance the state machine for one assessment.

        Stale observations are not allowed to mutate state: processing an old
        event after a newer one must not reopen or re-notify an already-resolved
        episode.  This is especially important when telemetry sources arrive
        out of order.
        """
        if risk_level not in _ORDER:
            raise ValueError(f"unknown risk level: {risk_level!r}")
        certainty_state = self._certainty_state(certainty)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            return AlertEpisode(
                "CLOCK_ERROR", False, risk_level, self._episode_id,
                self._opened_at, self._last_notified_at, 0.0,
                "stale assessment ignored; episode state was not changed", certainty_state,
            )
        self._last_timestamp = timestamp

        elevated = _ORDER[risk_level] >= _ORDER["MEDIUM"]
        if not self._active:
            if not elevated:
                return AlertEpisode("IDLE", False, risk_level, None, None, None, 0.0,
                                    "risk is low; no active episode", certainty_state)
            self._active = True
            self._current_level = risk_level
            self._episode_id = self._next_episode_id
            self._next_episode_id += 1
            self._opened_at = timestamp
            self._last_notified_at = timestamp
            return AlertEpisode("OPENED", True, risk_level, self._episode_id,
                                self._opened_at, self._last_notified_at, 0.0,
                                "entered elevated risk; fresh advisory notification", certainty_state)

        assert self._opened_at is not None
        elapsed = max(0.0, timestamp - self._opened_at)
        if not elevated:
            event = AlertEpisode("RESOLVED", False, risk_level, self._episode_id,
                                 self._opened_at, self._last_notified_at, elapsed,
                                 "risk returned to low; episode resolved", certainty_state)
            self._active = False
            self._current_level = "LOW"
            self._episode_id = None
            self._opened_at = None
            self._last_notified_at = None
            return event

        previous = self._current_level
        self._current_level = risk_level
        if _ORDER[risk_level] > _ORDER[previous]:
            self._last_notified_at = timestamp
            return AlertEpisode("ESCALATED", True, risk_level, self._episode_id,
                                self._opened_at, self._last_notified_at, elapsed,
                                "risk tier escalated; cooldown cannot suppress escalation", certainty_state)

        if (risk_level == "HIGH" and self.high_renotify_seconds is not None
                and self._last_notified_at is not None
                and timestamp - self._last_notified_at >= self.high_renotify_seconds):
            self._last_notified_at = timestamp
            return AlertEpisode("RENOTIFY", True, risk_level, self._episode_id,
                                self._opened_at, self._last_notified_at, elapsed,
                                "persistent high risk reached the study re-notification interval", certainty_state)

        return AlertEpisode("ACTIVE", False, risk_level, self._episode_id,
                            self._opened_at, self._last_notified_at, elapsed,
                            "elevated episode remains active; no duplicate notification", certainty_state)
