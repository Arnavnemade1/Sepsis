"""Run the falsification audit and record what survives.

    python -m backend.ml.scripts.falsify_audit
    python -m backend.ml.scripts.falsify_audit --save
    python -m backend.ml.scripts.falsify_audit --gate      # fail on regression only

The number to read is not the coverage percentage. It is the SURVIVOR LIST: mutations
that no named check catches, i.e. claims this project publishes that nothing is
actually defending.
"""
from __future__ import annotations

import argparse
import sys

from backend.ml import falsify
import backend.ml.mutations  # noqa: F401  — registers the catalogue on import


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="", help="comma-separated suite substrings")
    ap.add_argument("--family", default="", help="comma-separated mutation families")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--slow", action="store_true",
                    help="include suites whose single run costs more than the rest of "
                         "the audit combined (verify_subgroup ~130s, verify_alarm_policy ~35s)")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 if coverage fell or a killed mutation now survives")
    args = ap.parse_args()

    from backend.ml.mutations import SLOW_SUITES
    suites = None
    if not args.slow:
        suites = sorted({m.suite for m in falsify.MUTATIONS} - SLOW_SUITES)
    if args.suite:
        want = [s.strip() for s in args.suite.split(",") if s.strip()]
        suites = [m.suite for m in falsify.MUTATIONS
                  if any(w in m.suite for w in want)]
    families = [f.strip() for f in args.family.split(",") if f.strip()] or None
    if not args.slow:
        print("(slow suites excluded — pass --slow to include them)\n")

    print("Falsification audit — proving the checks can fail\n")
    print("baseline (a suite must be green before a mutation means anything):")
    rep = falsify.audit(suites=suites, families=families)

    print("\ncoverage by suite")
    print(falsify.table(rep))

    print("\nby mutation family")
    for f, d in sorted(rep["families"].items()):
        cov = f"{d['killed'] / d['n']:.0%}" if d["n"] else "—"
        print(f"  {f:<22}{d['killed']:>3}/{d['n']:<3}{cov:>7}")

    na = rep.get("not_applicable", [])
    if na:
        print(f"\nNOT APPLICABLE — {len(na)} mutation(s) the audit could not test")
        for r in na:
            print(f"  [{r['family']}] {r['suite'].split('.')[-1]}::{r['name']} — {r['error']}")

    surv = rep["survivors"]
    print(f"\nSURVIVORS — {len(surv)} mutation(s) no named check catches")
    if not surv:
        print("  (none)")
    for r in surv:
        why = r["error"] or ("caught nothing" if not r["caught_by"]
                             else "caught the wrong check")
        print(f"  [{r['family']}] {r['suite'].split('.')[-1]}::{r['name']} — {why}")
        for m in r["missed"]:
            print(f"        expected to fail: {m!r}")

    if args.save:
        falsify.save(rep)
        print(f"\nwrote {falsify.DEFAULT_PATH}")

    if args.gate:
        prev = falsify.load()
        if prev is None:
            print("\ngate: no previous record — nothing to regress against")
        else:
            probs = falsify.regression(prev, rep)
            if probs:
                print("\nGATE FAILED:")
                for p in probs:
                    print("  " + p)
                sys.exit(1)
            print("\ngate: no regression")


if __name__ == "__main__":
    main()
