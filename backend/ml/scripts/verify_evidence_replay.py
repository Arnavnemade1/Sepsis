"""Verify that AERIS replay metrics stay patient-level and descriptive."""
from __future__ import annotations

from backend.ml.eval.evidence_replay import EvidenceReplayRow, summarize


def main() -> None:
    rows = [
        EvidenceReplayRow("short", 0, {"action": "WAIT", "request": False}),
        EvidenceReplayRow("short", 1, {"action": "COMPUTE_WINDOW", "request": True,
                                        "cost_vector": {"compute": 1.0}}, onset_hour=2),
        EvidenceReplayRow("long", 0, {"action": "WAIT", "request": False}),
        EvidenceReplayRow("long", 23, {"action": "REACQUIRE", "request": True,
                                        "cost_vector": {"request": 1.0}}, onset_hour=12),
        EvidenceReplayRow("long", 47, {"action": "WAIT", "request": False}),
    ]
    report = summarize(rows)
    assert report["n_patients"] == 2
    assert report["n_requests"] == 2
    # Per-patient daily rates: short=24/2=12; long=1/2=0.5; mean=6.25.
    assert report["requests_per_patient_day"] == 6.25, report
    assert report["pre_onset_request_rate"] == 0.5, report
    assert report["causal_claim"].startswith("none")
    print("[PASS] patient-level request burden, onset timing and descriptive-only guard")


if __name__ == "__main__":
    main()
