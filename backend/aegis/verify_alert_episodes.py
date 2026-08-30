"""Adversarial checks for the alert-episode state machine.

Run with ``python -m backend.aegis.verify_alert_episodes``.  These are behavior
checks, not a claim that the default timing is clinically validated.
"""
from __future__ import annotations

from backend.aegis.alert_episodes import AlertEpisodeController


def _ok(message: str) -> None:
    print(f"      [PASS] {message}")


def test_lifecycle_and_duplicate_suppression() -> None:
    print("[1] one elevated interval is one episode, not a page per tick")
    c = AlertEpisodeController(high_renotify_seconds=100.0)
    assert c.update("LOW", 0).state == "IDLE"
    opened = c.update("MEDIUM", 10)
    active = c.update("MEDIUM", 20)
    resolved = c.update("LOW", 30)
    assert opened.state == "OPENED" and opened.notify and opened.episode_id == 1
    assert active.state == "ACTIVE" and not active.notify and active.episode_id == 1
    assert resolved.state == "RESOLVED" and not resolved.notify and resolved.episode_id == 1
    _ok("MEDIUM opens once, remains active without duplicate notifications, then resolves")


def test_escalation_and_uncertainty_are_never_suppressed() -> None:
    print("[2] escalation cannot be hidden by cooldown or uncertainty")
    c = AlertEpisodeController(high_renotify_seconds=1000.0)
    c.update("MEDIUM", 0)
    ev = c.update("HIGH", 1, {"state": "uncertain", "abstain": True})
    assert ev.state == "ESCALATED" and ev.notify
    assert ev.certainty_state == "uncertain"
    _ok("MEDIUM -> HIGH emits immediately even when the model is uncertain")


def test_high_renotify_and_stale_event() -> None:
    print("[3] persistent high re-notifies on schedule; stale data cannot reopen state")
    c = AlertEpisodeController(high_renotify_seconds=10.0)
    c.update("HIGH", 100)
    assert c.update("HIGH", 109).state == "ACTIVE"
    again = c.update("HIGH", 110)
    assert again.state == "RENOTIFY" and again.notify
    c.update("LOW", 120)
    stale = c.update("HIGH", 119)
    assert stale.state == "CLOCK_ERROR" and not stale.notify
    assert c.update("LOW", 121).state == "IDLE"
    _ok("only a due persistent HIGH re-notifies; late packets leave resolved state untouched")


def main() -> None:
    print("=" * 64)
    print("AegisOS — alert episode verification")
    print("=" * 64)
    test_lifecycle_and_duplicate_suppression()
    test_escalation_and_uncertainty_are_never_suppressed()
    test_high_renotify_and_stale_event()
    print("-" * 64)
    print("ALL ALERT-EPISODE CHECKS PASSED")


if __name__ == "__main__":
    main()
