"""The pre-registration, and the guard that makes freezing it mean anything.

A pre-registration that nothing enforces is a promise. This module is what turns it
into a constraint:

* :func:`digest` hashes the frozen document with the same canonical-JSON idiom the Lab
  registry uses to decide whether two proposals are the same hypothesis
  (``backend.aegis.lab_registry.change_key``) — imported, not reimplemented, so there
  is exactly one definition of "the same document" in this repository.
* The discovery phase WRITES that digest to :data:`DISCOVERY_RECORD`. The locked phase
  READS it and calls :func:`require_unchanged`, which is fatal if the document moved in
  between. Moving the bar after seeing a discovery number is then a diff with a
  timestamp, not a quiet decision.
* :func:`assert_no_patient_level` is the honesty boundary made executable. This study
  can say something about cohorts; it cannot say anything about a patient in this
  product, because no bed carries transcriptomics. So the artifact may carry counts and
  aggregates, and never a per-sample or per-patient field.

Amendments are recorded INSIDE the document (``amendments``), and each amendment
changes the digest. That is intended: the digest identifies the exact rules a locked
run was held to, and the amendment trail explains every change to them.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from backend.aegis.lab_registry import change_key

PREREG_PATH = os.path.join("backend", "hostresponse", "preregistration.json")
DISCOVERY_RECORD = os.path.join("artifacts", "hostresponse", "discovery_record.json")

#: A cohort whose role starts with this is kept in the document — so the cohort set
#: that was frozen stays visible — and contributes nothing.
WITHDRAWN = "WITHDRAWN"

#: Keys that would put an individual in the artifact. Aggregates (``n_patients``,
#: ``per_cohort`` counts) are fine; identities and per-row scores are not.
FORBIDDEN_KEYS = frozenset({
    "patient_ids", "patients", "sample_ids", "samples", "per_patient", "per_sample",
    "predictions", "scores", "held_out_scores", "pooled",
})
#: A GEO sample accession, or a patient id namespaced the way ``expression.py`` builds
#: them ("GSE40012:23"). Either one in a string value is an individual.
_INDIVIDUAL = re.compile(r"\bGSM\d{3,}\b|\bGSE\d+:[^\s,;]+")


def load(path: str = PREREG_PATH) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def digest(doc: Dict[str, Any]) -> str:
    """The canonical identity of a pre-registration document."""
    return change_key(doc)


def admissible(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The cohorts that actually take part — withdrawn ones excluded, never deleted."""
    return [c for c in doc["cohorts"]["admissible"]
            if not str(c.get("role", "")).upper().startswith(WITHDRAWN)]


def role_of(doc: Dict[str, Any], accession: str) -> Optional[str]:
    for c in admissible(doc):
        if c["accession"] == accession:
            return c["role"]
    return None


def bar_as_frozen(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The bar exactly as it was frozen, before any amendment."""
    return dict(doc["acceptance_bar"])


def bar_in_force(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The bar a run is actually held to.

    The highest-numbered amendment that restates the bar wins; with none, the frozen
    bar stands. There are deliberately TWO functions rather than one ``bar()``: a run
    reports its verdict under both, so an amendment that changed a guard is visible in
    the result rather than only in the document's history.
    """
    restated = [a for a in doc.get("amendments", []) if "acceptance_bar_in_force" in a]
    if restated:
        return dict(max(restated, key=lambda a: a["n"])["acceptance_bar_in_force"])
    return bar_as_frozen(doc)


def record_discovery(doc: Dict[str, Any], extra: Optional[Dict[str, Any]] = None,
                     path: str = DISCOVERY_RECORD) -> str:
    """Write the digest the locked phase will be held to. Returns it."""
    d = digest(doc)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = {"prereg_digest": d, "protocol": doc.get("protocol")}
    body.update(extra or {})
    with open(path, "w") as fh:
        json.dump(body, fh, indent=2)
    return d


def recorded_digest(path: str = DISCOVERY_RECORD) -> Optional[str]:
    try:
        with open(path) as fh:
            return json.load(fh).get("prereg_digest")
    except (OSError, ValueError):
        return None


def require_unchanged(recorded: Optional[str], path: str = PREREG_PATH) -> str:
    """Fatal unless the pre-registration is byte-for-byte the one discovery recorded.

    ``SystemExit`` rather than an exception a caller could catch and continue past: the
    locked cohort is scored once, and a run held to rules other than the recorded ones
    is not a degraded result — it is not this study at all.
    """
    if not recorded:
        raise SystemExit(
            "no discovery record (%s). The locked phase runs only after the discovery "
            "phase has recorded which pre-registration it was held to — run "
            "`make hostresponse PHASE=discovery` first." % DISCOVERY_RECORD)
    now = digest(load(path))
    if now != recorded:
        raise SystemExit(
            "PRE-REGISTRATION CHANGED after discovery: recorded %s, now %s. The locked "
            "cohort will not be scored against rules that moved after a discovery "
            "number existed. If the change is legitimate, record it as an amendment "
            "and re-run discovery so the new digest is the one on record."
            % (recorded, now))
    return now


def assert_no_patient_level(payload: Any, _path: str = "$") -> None:
    """Raise if the artifact carries any individual — by key or by value."""
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise ValueError("per-individual field %r at %s: this study may publish "
                                 "cohort aggregates only" % (k, _path))
            assert_no_patient_level(v, "%s.%s" % (_path, k))
    elif isinstance(payload, (list, tuple)):
        for i, v in enumerate(payload):
            assert_no_patient_level(v, "%s[%d]" % (_path, i))
    elif isinstance(payload, str) and _INDIVIDUAL.search(payload):
        raise ValueError("an individual identifier appears in a value at %s: %r"
                         % (_path, payload[:60]))
