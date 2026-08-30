"""Adversarial checks for AERIS's deterministic first-slice policy."""
from __future__ import annotations

from backend.aegis.evidence_router import (
    COMPUTE_WINDOW, ESCALATE_FOR_REVIEW, REACQUIRE, WAIT, EvidenceRouter,
)


def _ok(message: str) -> None:
    print(f"      [PASS] {message}")


def _episode(i=1):
    return {"episode_id": i, "state": "OPENED"}


def _uncertain():
    return {"state": "uncertain", "abstain": True}


def test_unavailable_sources_are_not_imagined() -> None:
    print("[1] unavailable hardware is never routed as evidence")
    r = EvidenceRouter()
    out = r.decide(timestamp=1, risk_level="HIGH", certainty=_uncertain(),
                   alert_episode=_episode(), feature_row={"MAP__age": 8.0}, capabilities={})
    assert out.action == WAIT and not out.request and "map" in out.unavailable_sources
    _ok("an unresolved HIGH assessment waits when no declared source is available")


def test_stale_channel_beats_optional_window() -> None:
    print("[2] a stale available MAP is selected before optional waveform compute")
    r = EvidenceRouter()
    out = r.decide(timestamp=1, risk_level="MEDIUM", certainty=_uncertain(),
                   alert_episode=_episode(), feature_row={"MAP__age": 2.0, "O2Sat__age": 0.0},
                   capabilities={"map": True, "hrv_window": True})
    assert (out.action, out.channel, out.request) == (REACQUIRE, "map", True), out
    assert out.estimated_value is None and out.selection_mode == "deterministic_baseline"
    _ok("stale MAP is a transparent REACQUIRE request; no made-up learned value")


def test_episode_budget_prevents_duplicate_requests() -> None:
    print("[3] one episode gets one ordinary evidence request")
    r = EvidenceRouter(max_requests_per_episode=1)
    first = r.decide(timestamp=1, risk_level="HIGH", certainty=_uncertain(),
                     alert_episode=_episode(7), feature_row={}, capabilities={"hrv_window": True})
    duplicate = r.decide(timestamp=2, risk_level="HIGH", certainty=_uncertain(),
                         alert_episode=_episode(7), feature_row={}, capabilities={"hrv_window": True})
    assert (first.action, first.channel) == (COMPUTE_WINDOW, "hrv")
    assert duplicate.action == WAIT and not duplicate.request
    _ok("a persistent uncertain episode cannot spam repeated HRV-window requests")


def test_hard_override_bypasses_budget() -> None:
    print("[4] hard clinical escalation is never suppressed by routing budget")
    r = EvidenceRouter()
    r.decide(timestamp=1, risk_level="HIGH", certainty=_uncertain(), alert_episode=_episode(),
             feature_row={}, capabilities={"hrv_window": True})
    out = r.decide(timestamp=2, risk_level="HIGH", certainty=_uncertain(), alert_episode=_episode(),
                   feature_row={}, capabilities={}, hard_override=True)
    assert out.action == ESCALATE_FOR_REVIEW and out.request
    _ok("hard override emits review escalation even after the ordinary request budget is used")


def test_stale_time_never_mutates_router() -> None:
    print("[5] an out-of-order assessment cannot spend a request budget")
    r = EvidenceRouter()
    r.decide(timestamp=10, risk_level="HIGH", certainty=_uncertain(), alert_episode=_episode(3),
             feature_row={}, capabilities={"hrv_window": True})
    late = r.decide(timestamp=9, risk_level="HIGH", certainty=_uncertain(), alert_episode=_episode(4),
                    feature_row={}, capabilities={"hrv_window": True})
    assert late.action == WAIT and "stale_timestamp" in late.eligibility_reasons
    fresh = r.decide(timestamp=11, risk_level="HIGH", certainty=_uncertain(), alert_episode=_episode(4),
                     feature_row={}, capabilities={"hrv_window": True})
    assert fresh.action == COMPUTE_WINDOW
    _ok("late data is ignored and cannot consume a later episode's request budget")


def main() -> None:
    print("=" * 64)
    print("AegisOS — AERIS evidence-router verification")
    print("=" * 64)
    test_unavailable_sources_are_not_imagined()
    test_stale_channel_beats_optional_window()
    test_episode_budget_prevents_duplicate_requests()
    test_hard_override_bypasses_budget()
    test_stale_time_never_mutates_router()
    print("-" * 64)
    print("ALL EVIDENCE-ROUTER CHECKS PASSED")


if __name__ == "__main__":
    main()
