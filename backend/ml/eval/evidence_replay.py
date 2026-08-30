"""Patient-level operational metrics for AERIS replay studies.

This module intentionally does not estimate causal benefit from retrospective
orders. It only measures what a *logged* router did: action count, compute cost,
availability/refusal mix, and timing relative to a supplied onset. Causal value
requires the shadow/randomized phase described in ``docs/AERIS_RESEARCH.md``.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class EvidenceReplayRow:
    patient_id: str
    hour: float
    decision: Dict[str, Any]
    onset_hour: Optional[float] = None


def summarize(rows: Iterable[EvidenceReplayRow]) -> Dict[str, Any]:
    """Summarize router operations at the patient, never row, level.

    A long admission must not manufacture a more precise estimate merely because
    it has more hourly records. ``requests_per_patient_day`` is calculated per
    patient first and then averaged.
    """
    ordered = sorted(rows, key=lambda r: (r.patient_id, r.hour))
    by_patient: Dict[str, List[EvidenceReplayRow]] = defaultdict(list)
    for row in ordered:
        by_patient[row.patient_id].append(row)
    action_counts: Counter[str] = Counter()
    per_day: List[float] = []
    compute = 0.0
    request = 0.0
    pre_onset_request: List[float] = []
    for patient_rows in by_patient.values():
        sent = [r for r in patient_rows if bool(r.decision.get("request"))]
        action_counts.update(str(r.decision.get("action", "UNKNOWN")) for r in patient_rows)
        for r in sent:
            costs = r.decision.get("cost_vector") or {}
            compute += float(costs.get("compute", 0.0))
            request += float(costs.get("request", 0.0))
        span = max(1.0 / 24.0, (patient_rows[-1].hour - patient_rows[0].hour + 1.0) / 24.0)
        per_day.append(len(sent) / span)
        onset = next((r.onset_hour for r in patient_rows if r.onset_hour is not None), None)
        if onset is not None:
            pre_onset_request.append(float(any(r.hour <= onset for r in sent)))
    n = len(ordered)
    return {
        "n_rows": n,
        "n_patients": len(by_patient),
        "action_counts": dict(sorted(action_counts.items())),
        "n_requests": int(sum(1 for r in ordered if bool(r.decision.get("request")))),
        "requests_per_patient_day": round(sum(per_day) / len(per_day), 4) if per_day else 0.0,
        "mean_compute_units_per_tick": round(compute / n, 6) if n else 0.0,
        "mean_request_units_per_tick": round(request / n, 6) if n else 0.0,
        "pre_onset_request_rate": round(sum(pre_onset_request) / len(pre_onset_request), 4)
        if pre_onset_request else None,
        "causal_claim": "none — descriptive replay metrics only",
    }
