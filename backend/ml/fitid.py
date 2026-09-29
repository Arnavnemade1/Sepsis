"""Fit fingerprints — a quantity is valid only against the fit it came from.

WHAT THIS COST BEFORE IT EXISTED
--------------------------------
Three separate "findings" in one investigation were arithmetic, not science, and all
three were the same mistake — a number read off one model's score distribution and
applied to another's:

  1. the shipped operating point was derived against a different fit than served;
  2. `external_validation_wearable.json` scored a target hospital with that constant,
     which manufactured an apparent **54% cross-hospital utility collapse**;
  3. `derive_conformal.py --external` calibrated conformal bands on a model fitted to
     60% of setA and applied them to scores from a model fitted to 100% of it, which
     manufactured an apparent **0.929 -> 0.739 coverage collapse**.

Each read as a dramatic result about deployment. Each was a mismatch nobody could see,
because the artifacts did not record which fit produced them: at the time only 4 of 16
did.

`operating_point.py` already carries a `detector` string for exactly this reason and
its docstring explains what it cost to learn. This module generalises that field into
something every derived artifact can carry and every consumer can check.

WHY BOTH A HASH AND A TUPLE
---------------------------
The hash answers "are these the same fit?" in one comparison. The tuple answers "which
part differs?", which is the question anyone actually has once the answer to the first
is no. A fingerprint that only hashes turns a five-second diagnosis into an afternoon.

WHY THE CHECK LIVES AT THE POINT OF USE
---------------------------------------
Stamping is not the safeguard; comparing is. All three failures above happened at the
moment two artifacts were read together, not at the moment either was written. So
`require_same` is called by the consumer, and a MISSING fingerprint reads as
**unknown, not compatible** — the precise failure mode being prevented is a
confident-looking artifact with no provenance at all.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence

#: Fields that define a fit. Order is fixed so the hash is stable across runs.
FIELDS = ("roster", "objective", "feature_set", "spec", "seed", "n_rows", "n_patients")


@dataclass
class FitId:
    roster: List[str] = field(default_factory=list)
    objective: str = "unknown"
    feature_set: str = "unknown"
    spec: str = "unknown"
    seed: Optional[int] = None
    n_rows: Optional[int] = None
    n_patients: Optional[int] = None

    # ------------------------------------------------------------------ #
    def key(self) -> str:
        """Human-readable identity, e.g. ``balanced:hgb+xgb@utility/wearable/dev``."""
        return (f"{self.objective}:{'+'.join(self.roster) or 'unknown'}"
                f"/{self.feature_set}/{self.spec}")

    def digest(self) -> str:
        """Short stable hash over the defining fields."""
        payload = json.dumps({f: getattr(self, f) for f in FIELDS}, sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["key"] = self.key()
        d["digest"] = self.digest()
        return d

    @staticmethod
    def from_json(d: Optional[Dict[str, Any]]) -> Optional["FitId"]:
        if not isinstance(d, dict):
            return None
        return FitId(**{f: d.get(f, FitId.__dataclass_fields__[f].default
                                 if f != "roster" else []) for f in FIELDS})


def fingerprint(roster: Sequence[str] = (), objective: str = "unknown",
                feature_set: str = "unknown", spec: str = "unknown",
                seed: Optional[int] = None, n_rows: Optional[int] = None,
                n_patients: Optional[int] = None) -> FitId:
    return FitId(list(roster), objective, feature_set, spec, seed, n_rows, n_patients)


def from_manifest(manifest: Dict[str, Any], spec: str = "unknown",
                  n_rows: Optional[int] = None,
                  n_patients: Optional[int] = None) -> FitId:
    """Read the promoted configuration out of ``artifacts/detector_manifest.json``.

    Read rather than re-declared: a roster written down a second time is a roster that
    can disagree with the one that shipped, which is the original bug in a new place.
    """
    cand = (manifest.get("configs") or {}).get("candidate") or {}
    return fingerprint(
        roster=cand.get("roster", []), objective=cand.get("objective", "unknown"),
        feature_set=manifest.get("feature_set", "unknown"), spec=spec,
        seed=manifest.get("seed"), n_rows=n_rows or manifest.get("n_rows"),
        n_patients=n_patients or manifest.get("n_patients"))


def training_digest(train_groups: Sequence[Any], source: str, seed: int) -> str:
    """Which patients a fit saw: 12 hex of sha256 over the sorted training patient ids,
    the data source and the split seed.

    ``FitId`` describes a fit by its configuration and dataset size, which cannot tell
    two fits apart when only the patients changed. This can: it is the suffix the
    promoter puts on the served detector's key, and anything that recomputes the split
    can prove it reproduced the shipped training set by recomputing this."""
    ids = sorted({str(g) for g in train_groups})
    payload = json.dumps({"source": source, "seed": int(seed), "patients": ids},
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
def stamp(artifact: Dict[str, Any], fit: FitId) -> Dict[str, Any]:
    """Write the ``fit`` block into an artifact dict, in place."""
    artifact["fit"] = fit.to_json()
    return artifact


class FitMismatch(ValueError):
    """Two quantities that were never measured against the same model."""


def differing(a: FitId, b: FitId) -> List[str]:
    return [f for f in FIELDS if getattr(a, f) != getattr(b, f)]


def require_same(a: Optional[FitId], b: Optional[FitId], what: str = "these artifacts",
                 strict: bool = True) -> None:
    """Raise unless ``a`` and ``b`` describe the same fit.

    An absent fingerprint is UNKNOWN, never compatible. Being permissive here would
    reproduce the original failure exactly: the artifacts that caused all three false
    findings had no fingerprint, so a "missing means fine" rule would have waved every
    one of them through.

    ``strict=False`` downgrades to a printed warning, for reading historical artifacts
    written before this module existed. It is never the default.
    """
    if a is None or b is None:
        msg = (f"{what}: fit identity is UNKNOWN on "
               f"{'the first' if a is None else 'the second'} artifact — it was written "
               f"before fingerprints existed, or by a script that does not stamp. "
               f"Unknown is not the same as compatible.")
        if strict:
            raise FitMismatch(msg)
        print(f"[fitid] WARNING — {msg}")
        return
    if a.digest() == b.digest():
        return
    diff = differing(a, b)
    msg = (f"{what}: these were measured against DIFFERENT fits "
           f"({a.key()} [{a.digest()}] vs {b.key()} [{b.digest()}]); "
           f"differing fields: {', '.join(diff) or 'unknown'}")
    if strict:
        raise FitMismatch(msg)
    print(f"[fitid] WARNING — {msg}")


def read(artifact: Dict[str, Any]) -> Optional[FitId]:
    """The fit an artifact records, or None when it records none."""
    return FitId.from_json(artifact.get("fit"))
