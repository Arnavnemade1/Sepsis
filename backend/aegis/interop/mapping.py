"""A mapping: one local code on one interface, and everything known about it.

Status is a small state machine, and every transition is recorded with who made it
and why:

    learning ──(enough values)──▶ proposing ──▶ adopted      evidence agreed, declared unit
                                      │    ──▶ review       a person must decide (inferred
                                      │                     unit, ambiguous, or refused)
                                      │    ──▶ unmapped     nothing in the vocabulary fits
    adopted ──(live evidence fails)──▶ quarantined ──(a person)──▶ adopted (restored / fixed)
    review  ──(a person)──▶ adopted | rejected

Only ``adopted`` mappings deliver values to rooms. A quarantined mapping keeps
counting what it withholds, so the room can say "creatinine withheld" instead of
silently showing nothing.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from backend.aegis.interop.units import Conversion

DELIVERING = ("adopted",)
STATUSES = ("learning", "proposing", "review", "adopted", "quarantined", "rejected", "unmapped")
SAMPLE_N = 240                      # raw values kept per mapping (evidence + drift windows)


@dataclass
class Sample:
    t: float                        # observation time
    raw: float                      # value as received
    unit: str                       # unit as written in that message
    group: str                      # message control id: values in one message are co-observed
    room: Optional[str]
    obs_id: Optional[str] = None    # set when delivered to a room (for retraction)
    delivered: bool = False
    seq: int = 0                    # arrival order across the engine
    withheld: bool = False


@dataclass
class Mapping:
    id: str
    interface: str
    code: str
    system: str
    text: str
    unit_written: str = ""
    unit_written_adopted: str = ""  # the unit as written when it was adopted
    concept: Optional[str] = None
    conversion: Optional[Conversion] = None
    unit_source: str = ""           # declared | inferred | loinc | human
    status: str = "learning"
    proposer: Optional[str] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Optional[Dict[str, Any]] = None
    fault: Optional[Dict[str, Any]] = None
    reason: str = ""
    version: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    adopted_at: Optional[float] = None
    adopted_by: Optional[str] = None
    counts: Dict[str, int] = field(default_factory=lambda: {"seen": 0, "delivered": 0, "withheld": 0,
                                                            "backfilled": 0, "retracted": 0, "non_numeric": 0})
    samples: Deque[Sample] = field(default_factory=lambda: deque(maxlen=SAMPLE_N))
    since_check: int = 0            # values since the last drift evaluation
    checks_run: int = 0
    model_asked: bool = False
    drift_from: int = 0             # live evidence reads samples arriving after this seq
    tentative: Optional[Tuple[str, Conversion]] = None   # best viable reading while not adopted
    model_props: List[Any] = field(default_factory=list)  # what the language model proposed, kept
    baseline: Optional[Dict[str, float]] = None           # the feed's own distribution at adoption

    def transition(self, status: str, who: str, why: str, now: Optional[float] = None, **extra: Any) -> None:
        now = time.time() if now is None else now
        self.history.append({"at": now, "from": self.status, "to": status, "by": who, "why": why, **extra})
        del self.history[:-40]
        self.status, self.reason, self.updated_at = status, why, now

    def values(self, last: Optional[int] = None) -> List[float]:
        xs = [s.raw for s in self.samples]
        return xs[-last:] if last else xs

    def to_dict(self, detail: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id, "interface": self.interface, "code": self.code, "system": self.system, "text": self.text,
            "unit_written": self.unit_written, "concept": self.concept,
            "conversion": self.conversion.to_dict() if self.conversion else None, "unit_source": self.unit_source,
            "status": self.status, "proposer": self.proposer, "reason": self.reason, "version": self.version,
            "created_at": self.created_at, "updated_at": self.updated_at, "adopted_at": self.adopted_at,
            "adopted_by": self.adopted_by, "counts": dict(self.counts), "n": len(self.samples),
            "fault": self.fault, "checks_run": self.checks_run,
        }
        if detail:
            d["candidates"] = self.candidates
            d["evidence"] = self.evidence
            d["history"] = list(self.history)
            d["samples"] = [{"t": s.t, "raw": s.raw, "unit": s.unit, "room": s.room, "delivered": s.delivered}
                            for s in list(self.samples)[-120:]]
        return d


def key_of(interface: str, system: str, code: str) -> Tuple[str, str, str]:
    return (interface, (system or "").upper(), code)
