"""Adversarial checks on fit fingerprints.

The regression test that matters is the last one: reproducing this session's three
false findings must now be impossible, because the artifacts involved refuse to be
compared. Everything else is scaffolding for that.

    python -m backend.ml.scripts.verify_fitid
"""
from __future__ import annotations

import json
import os
import tempfile

from backend.ml import fitid
from backend.ml.fitid import FitMismatch, fingerprint, require_same, stamp, read

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


BASE = dict(roster=["hgb", "hgb2", "et", "lr", "xgb@utility"], objective="balanced",
            feature_set="wearable", spec="dev", seed=42, n_rows=310061, n_patients=8000)


def check_digest_is_stable_and_sensitive():
    """1. The same fit hashes the same; any defining change does not."""
    a, b = fingerprint(**BASE), fingerprint(**BASE)
    check("identical fits produce identical digests", a.digest() == b.digest(), a.digest())
    for f, v in (("objective", "utility"), ("spec", "setA"), ("seed", 7),
                 ("feature_set", "lab"), ("n_patients", 4000)):
        d = dict(BASE); d[f] = v
        if fingerprint(**d).digest() == a.digest():
            check(f"changing {f} changes the digest", False)
            return
    check("changing any defining field changes the digest", True, "5 fields probed")
    d = dict(BASE); d["roster"] = ["hgb", "hgb2", "et", "lr"]
    check("dropping a roster member changes it too",
          fingerprint(**d).digest() != a.digest())


def check_mismatch_is_refused():
    """2. Different fits refuse to be compared, and say which field differs."""
    a = fingerprint(**BASE)
    d = dict(BASE); d["spec"] = "setB"; d["n_patients"] = 19999
    b = fingerprint(**d)
    try:
        require_same(a, b, "op vs external")
        check("a cross-fit comparison raises", False)
        return
    except FitMismatch as e:
        msg = str(e)
    check("a cross-fit comparison raises", True)
    check("...and names the differing fields", "spec" in msg and "n_patients" in msg,
          msg.split("differing fields:")[-1].strip()[:60])


def check_unknown_is_not_compatible():
    """3. PLANTED: a missing fingerprint must NOT read as 'fine'.

    Every artifact that caused a false finding had no fingerprint at all. A permissive
    'missing means compatible' rule would have waved all three through, which makes
    this the single most important check in the file.
    """
    a = fingerprint(**BASE)
    raised = 0
    for pair in ((a, None), (None, a), (None, None)):
        try:
            require_same(pair[0], pair[1], "legacy read")
        except FitMismatch:
            raised += 1
    check("an absent fingerprint is refused, not assumed compatible", raised == 3,
          f"{raised}/3 combinations raised")
    # ...and the escape hatch must be explicit, never the default.
    try:
        require_same(a, None, "legacy read", strict=False)
        ok = True
    except FitMismatch:
        ok = False
    check("strict=False downgrades to a warning for historical artifacts", ok)


def check_stamp_roundtrip():
    """4. A stamped artifact survives JSON and reads back identical."""
    art = {"high": 0.0525, "medium": 0.0262}
    fp = fingerprint(**BASE)
    stamp(art, fp)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.json")
        json.dump(art, open(p, "w"), allow_nan=False)
        back = read(json.load(open(p)))
    check("a stamped artifact round-trips through JSON",
          back is not None and back.digest() == fp.digest(),
          f"{fp.digest()} -> {back.digest() if back else None}")
    check("an unstamped artifact reads as None, not as a default fit",
          read({"high": 0.05}) is None)


def check_manifest_is_read_not_retyped():
    """5. The roster comes from the manifest the promotion gate wrote."""
    man = {"feature_set": "wearable", "seed": 42, "n_rows": 310061, "n_patients": 8000,
           "configs": {"candidate": {"objective": "balanced",
                                     "roster": ["hgb", "hgb2", "et", "lr", "xgb@utility"]}}}
    fp = fitid.from_manifest(man, spec="dev")
    check("from_manifest reproduces the shipped roster",
          fp.roster == man["configs"]["candidate"]["roster"], fp.key())
    # A roster written down a second time is a roster that can disagree with the one
    # that shipped — the original bug wearing a different hat.
    man2 = json.loads(json.dumps(man))
    man2["configs"]["candidate"]["roster"] = ["hgb", "hgb2", "et", "lr"]
    check("...and a manifest with a different roster yields a different digest",
          fitid.from_manifest(man2, spec="dev").digest() != fp.digest())


def check_the_three_false_findings_are_now_impossible():
    """6. THE regression test. Replay each of the three mismatches; each must refuse.

    These are the actual configurations that produced the three published-then-retracted
    results. If any of them compares cleanly, this module has failed at the only job it
    was built for.
    """
    # (1) operating point derived on one fit, served against another
    op_fit = fingerprint(roster=["hgb", "hgb2", "et", "lr"], objective="utility",
                         feature_set="wearable", spec="dev", seed=42)
    served = fingerprint(roster=["hgb", "hgb2", "et", "lr", "xgb@utility"],
                         objective="balanced", feature_set="wearable", spec="dev", seed=42)
    # (2) external validation scoring setB with a threshold measured on dev
    ext = fingerprint(roster=["hgb", "hgb2", "et", "lr", "xgb@utility"],
                      objective="balanced", feature_set="wearable", spec="setB", seed=42)
    # (3) conformal bands calibrated on a 60%-of-setA fit, applied to a 100%-of-setA fit
    cal = fingerprint(roster=["stack+xgb@utility"], objective="balanced",
                      feature_set="wearable", spec="setA", seed=42, n_patients=2400)
    scored = fingerprint(roster=["stack+xgb@utility"], objective="balanced",
                         feature_set="wearable", spec="setA", seed=42, n_patients=4000)

    refused = 0
    for a, b, what in ((op_fit, served, "operating point vs served detector"),
                       (served, ext, "dev threshold vs setB scoring"),
                       (cal, scored, "conformal bands vs target scores")):
        try:
            require_same(a, b, what)
        except FitMismatch:
            refused += 1
    check("all three of this session's false findings are now refused at the point of use",
          refused == 3, f"{refused}/3 comparisons raised")
    # Vacuity guard: a genuinely matched pair must still pass, or the check above is
    # satisfied by a function that refuses everything.
    try:
        require_same(served, served, "same fit")
        ok = True
    except FitMismatch:
        ok = False
    check("...while a genuinely matched pair still compares cleanly", ok)


def main() -> None:
    print("Fit fingerprints — adversarial checks\n")
    for fn in (check_digest_is_stable_and_sensitive, check_mismatch_is_refused,
               check_unknown_is_not_compatible, check_stamp_roundtrip,
               check_manifest_is_read_not_retyped,
               check_the_three_false_findings_are_now_impossible):
        fn()
    ok, n = sum(results), len(results)
    print(f"\n{ok}/{n} checks passed")
    raise SystemExit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
