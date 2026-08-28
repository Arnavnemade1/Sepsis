"""Falsification coverage — proving this project's own checks can fail.

WHY THIS EXISTS
---------------
There are 452 named checks across 63 verification suites in this repository, and until
this module nothing proved that a single one of them was *capable of failing*. A check
that cannot fail is not a weak check; it is a lie with a green tick, and it is worse
than no check at all because it is counted as evidence.

That is not a hypothetical. Four vacuous checks were found by hand in a single session:

    verify_conformal   a `>=` comparison that equality satisfied, so deleting the
                       (n+1) conformal correction left the check passing
    verify_conformal   a state count that swept in the EMPTY prediction set, so the
                       degeneracy the check existed to expose read as success
    verify_minirocket  a harness expectation that never matched the check's own text
    verify_shift       an assertion written for the wrong direction of an effect

Each was found with a throwaway mutation harness in /tmp that was then deleted. This
module is that harness, kept.

NOT CODE-COVERAGE MUTATION TESTING
----------------------------------
Classical mutation testing (Jia & Harman) perturbs *syntax* — flip a `<` to `<=`, drop
a statement — and asks whether any test fails. It measures how thoroughly the tests
pin the code.

This perturbs the **scientific invariant** and asks whether a *specific named check*
fails. The difference is the whole point. A surviving syntactic mutant means weak test
coverage. A surviving semantic mutant here means something worse and more specific:

    a claim this project publishes that nobody is actually defending.

So the deliverable is not the coverage percentage. It is the **survivor list**.

THE MUTATIONS ARE REAL FAILURE MODES
------------------------------------
Every family below is something this repository has actually suffered and paid for, not
a mutation operator chosen because it was easy to write. `fit-identity` alone produced
three separate false findings in one investigation.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import pathlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class Mutation:
    """One semantic break, and the checks that are supposed to notice it."""
    family: str
    suite: str                       # module path under backend/
    name: str
    must_fail: List[str]             # substrings of check names that MUST report [FAIL]
    apply: Callable                  # apply(suite_module) -> None
    note: str = ""


MUTATIONS: List[Mutation] = []


def mutation(family: str, suite: str, must_fail: Sequence[str], note: str = ""):
    """Register a semantic mutation.

    ``must_fail`` names the checks that are supposed to catch it. Naming them — rather
    than accepting "something failed" — is what turns this from a smoke test into a
    map from claims to the mutations that defend them. A mutation that fails the wrong
    check is reported as a survivor, because the check that was supposed to be load-
    bearing was not.
    """
    def deco(fn):
        MUTATIONS.append(Mutation(family=family, suite=suite, name=fn.__name__,
                                  must_fail=list(must_fail), apply=fn, note=note))
        return fn
    return deco


# --------------------------------------------------------------------------- #
# Running a suite, mutated or clean
# --------------------------------------------------------------------------- #
@dataclass
class SuiteRun:
    suite: str
    ok: int
    total: int
    failed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    seconds: float = 0.0


#: Modules a mutation may patch. Reloaded before EVERY run so one mutation cannot
#: leak into the next — without this the audit silently drifts, and a later mutation
#: would be scored against damage an earlier one left behind.
RESET_MODULES = [
    "backend.ml.conformal",
    "backend.ml.shift",
    "backend.ml.operating_point",
    "backend.ml.models.minirocket",
    "backend.ml.fitid",
]


def _reset() -> None:
    for name in RESET_MODULES:
        m = sys.modules.get(name)
        if m is not None:
            importlib.reload(m)


def run_suite(suite: str, apply: Optional[Callable] = None) -> SuiteRun:
    """Execute a suite (optionally mutated) and report which named checks failed.

    TWO SUITE SHAPES, ONE RUNNER
    ----------------------------
    Some suites run their checks inside ``main()``; others (``verify_operating_point``,
    ``verify_subgroup``) run them at module level and end with a bare
    ``raise SystemExit``. Neither ``import_module`` nor ``reload`` can serve the second
    kind: the module-level exit fires during import, Python purges the half-built
    module from ``sys.modules``, and there is no module object left to inspect. Worse,
    ``SystemExit`` derives from ``BaseException``, so it sails through ``except
    Exception``, unwinds the whole audit and terminates the process with that suite's
    own exit code — silently, and with a 0 if the suite happened to be green.

    So the source is compiled into a module object this function owns and executed
    under a caught ``SystemExit``. That works for both shapes and puts the mutation
    where it has to be: applied to the DEPENDENCY modules *before* the suite body runs,
    which is the only moment an import-time suite can be influenced at all.

    The second ``apply`` is not redundant. Executing the body rebinds the suite's own
    module-level names (``cf``, ``shift``, ``opmod``), which would undo the first pass
    for any suite that reaches through its own reference rather than calling into the
    dependency directly.
    """
    import types

    t0 = time.time()
    _reset()
    spec = importlib.util.find_spec(suite)
    if spec is None or not spec.origin:
        return SuiteRun(suite, 0, 0, error="module not found")
    src = pathlib.Path(spec.origin).read_text()

    mod = types.ModuleType(suite)
    mod.__file__ = spec.origin
    mod.__package__ = suite.rpartition(".")[0]
    prev = sys.modules.get(suite)
    sys.modules[suite] = mod            # so the body's own imports resolve normally

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            if apply is not None:
                # Phase 1 is best-effort. A mutation that targets the SUITE's own
                # helpers (rather than a dependency) cannot be applied yet — the
                # namespace is still empty — and raising here would abort the run and
                # report the mutation as a survivor when it was simply applied at the
                # wrong moment. Those land in phase 2. A mutation that fails BOTH
                # phases still surfaces, because nothing will have changed.
                try:
                    apply(mod)                   # (1) mutate the dependencies
                except Exception:
                    pass
            try:
                exec(compile(src, spec.origin, "exec"), mod.__dict__)
            except SystemExit:
                pass                             # module-level suites exit here
            if apply is not None:
                apply(mod)                       # (2) re-point the suite's own refs
            if not hasattr(mod, "results"):
                return SuiteRun(suite, 0, 0, error="no module-level `results` list")
            entry = getattr(mod, "main", None)
            if entry is not None:
                mod.results.clear()              # main() re-runs everything itself
                try:
                    entry()
                except SystemExit:
                    pass
        out = buf.getvalue()
        failed = [ln.strip()[len("[FAIL]"):].strip()
                  for ln in out.splitlines() if ln.strip().startswith("[FAIL]")]
        res = list(mod.results)
        return SuiteRun(suite, sum(1 for r in res if r), len(res), failed,
                        seconds=round(time.time() - t0, 2))
    except Exception as exc:                     # a mutation may break the body
        return SuiteRun(suite, 0, 0, error=f"{type(exc).__name__}: {exc}",
                        seconds=round(time.time() - t0, 2))
    finally:
        if prev is not None:
            sys.modules[suite] = prev
        else:
            sys.modules.pop(suite, None)
        _reset()                                 # never leave a mutation installed


def _matched(failed: Sequence[str], want: str) -> bool:
    return any(want.lower() in f.lower() for f in failed)


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #
@dataclass
class MutationResult:
    family: str
    suite: str
    name: str
    killed: bool                     # a check that was supposed to catch it did
    caught_by: List[str]
    missed: List[str]                # named checks that should have failed and did not
    collateral: List[str]            # other checks that failed too (informational)
    error: Optional[str] = None

    def to_json(self) -> Dict[str, object]:
        return asdict(self)


def audit(suites: Optional[Sequence[str]] = None,
          families: Optional[Sequence[str]] = None,
          verbose: bool = True) -> Dict[str, object]:
    """Run every registered mutation and report falsification coverage."""
    muts = [m for m in MUTATIONS
            if (suites is None or m.suite in suites)
            and (families is None or m.family in families)]
    by_suite = sorted({m.suite for m in muts})

    # Clean baseline first. A suite that is already failing cannot tell us anything
    # about whether a mutation caused a failure, so it is reported and skipped rather
    # than silently producing 'killed' results from pre-existing breakage.
    baseline: Dict[str, SuiteRun] = {}
    for s in by_suite:
        baseline[s] = run_suite(s)
        if verbose:
            b = baseline[s]
            state = (f"{b.ok}/{b.total}" if not b.error else f"ERROR {b.error}")
            print(f"  baseline {s.split('.')[-1]:<22} {state:>12}  [{b.seconds:.1f}s]")

    out: List[MutationResult] = []
    for m in muts:
        b = baseline.get(m.suite)
        if b is None or b.error or b.ok != b.total:
            out.append(MutationResult(m.family, m.suite, m.name, False, [], m.must_fail, [],
                                      error="baseline not green"))
            continue
        r = run_suite(m.suite, m.apply)
        if r.error:
            # An import-time explosion is NOT a kill. The mutation has to be caught by
            # a check, not by the interpreter — otherwise "it crashed" would count as
            # evidence that a claim is defended.
            out.append(MutationResult(m.family, m.suite, m.name, False, [], m.must_fail,
                                      [], error=r.error))
            continue
        caught = [w for w in m.must_fail if _matched(r.failed, w)]
        missed = [w for w in m.must_fail if not _matched(r.failed, w)]
        collateral = [f for f in r.failed
                      if not any(w.lower() in f.lower() for w in m.must_fail)]
        out.append(MutationResult(m.family, m.suite, m.name, len(missed) == 0,
                                  caught, missed, collateral))

    killed = [r for r in out if r.killed]
    survivors = [r for r in out if not r.killed]
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "n_mutations": len(out),
        "n_killed": len(killed),
        "coverage": round(len(killed) / len(out), 4) if out else 0.0,
        "suites": {s: {"checks": baseline[s].total, "green": baseline[s].ok,
                       "error": baseline[s].error,
                       "mutations": len([r for r in out if r.suite == s]),
                       "killed": len([r for r in out if r.suite == s and r.killed])}
                   for s in by_suite},
        "families": {f: {"n": len([r for r in out if r.family == f]),
                         "killed": len([r for r in out if r.family == f and r.killed])}
                     for f in sorted({m.family for m in muts})},
        # The actual deliverable: claims nobody is defending.
        "survivors": [r.to_json() for r in survivors],
        "killed_detail": [r.to_json() for r in killed],
    }
    return report


def table(report: Dict[str, object]) -> str:
    lines = [f"  {'suite':<24}{'checks':>7}{'mutations':>11}{'killed':>8}{'coverage':>10}"]
    for s, d in sorted(report["suites"].items()):
        n, k = d["mutations"], d["killed"]
        cov = f"{k / n:.0%}" if n else "—"
        lines.append(f"  {s.split('.')[-1]:<24}{d['checks']:>7}{n:>11}{k:>8}{cov:>10}")
    lines.append(f"  {'TOTAL':<24}{'':>7}{report['n_mutations']:>11}"
                 f"{report['n_killed']:>8}{report['coverage']:>10.0%}")
    return "\n".join(lines)


DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "artifacts", "falsification.json")


def save(report: Dict[str, object], path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)


def load(path: str = DEFAULT_PATH) -> Optional[Dict[str, object]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def regression(prev: Dict[str, object], now: Dict[str, object]) -> List[str]:
    """What got WORSE since the recorded run.

    A fixed coverage floor was considered and rejected: with this many checks it fails
    on day one and forces a backfill before any other work can proceed. Regression
    gating catches a newly-added vacuous check the moment it lands, which is the actual
    failure mode, while leaving the existing survivor list as published work to do.
    """
    problems = []
    if now["coverage"] < prev["coverage"] - 1e-9:
        problems.append(f"coverage fell {prev['coverage']:.0%} -> {now['coverage']:.0%}")
    was_killed = {(r["suite"], r["name"]) for r in prev.get("killed_detail", [])}
    now_killed = {(r["suite"], r["name"]) for r in now.get("killed_detail", [])}
    for s, n in sorted(was_killed - now_killed):
        problems.append(f"mutation now SURVIVES: {s.split('.')[-1]}::{n}")
    return problems
