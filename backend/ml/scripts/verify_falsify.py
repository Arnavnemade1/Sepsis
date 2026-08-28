"""Adversarial checks on the falsification engine itself.

A mutation harness that cannot fail is precisely the thing this engine exists to
prevent, so it is held to its own rule. Each check builds a throwaway suite with a
known property and asserts the engine reaches the right verdict about it.

The two verdicts that must never be confused:

    KILLED     a named check reported [FAIL] because of the mutation
    SURVIVOR   nothing caught it — the claim is undefended

and the two ways a naive harness gets that wrong: counting an interpreter crash as a
kill, and counting "some other check failed" as a kill.

    python -m backend.ml.scripts.verify_falsify
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap

from backend.ml import falsify

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
_TMP = tempfile.mkdtemp(prefix="falsify_selftest_")
sys.path.insert(0, _TMP)


def write_suite(name: str, body: str) -> str:
    """Write a throwaway suite that follows the house contract."""
    src = textwrap.dedent('''
        results = []
        def check(name, ok, detail=""):
            results.append(bool(ok))
            print(("  [PASS] " if ok else "  [FAIL] ") + name + (" — " + detail if detail else ""))
        import selftest_dep as dep
    ''') + textwrap.dedent(body)
    with open(os.path.join(_TMP, name + ".py"), "w") as fh:
        fh.write(src)
    return name


def write_dep(body: str) -> None:
    with open(os.path.join(_TMP, "selftest_dep.py"), "w") as fh:
        fh.write(textwrap.dedent(body))


write_dep('''
    def guarded(x):
        """The behaviour under test: returns True only for positive x."""
        return x > 0
''')

# Register the throwaway dependency for reset. Without this the self-audit suffers
# EXACTLY the failure it is auditing for: a mutation from one check (`guarded = boom`)
# survives into the next, whose baseline then explodes and whose mutation is scored
# against damage it did not cause. Two checks below failed that way before this line
# existed — the engine's own leak guard is the fix for its own test suite.
falsify.RESET_MODULES.append("selftest_dep")


# --------------------------------------------------------------------------- #
def check_real_check_is_killed():
    """1. A genuinely load-bearing check is reported KILLED when broken."""
    s = write_suite("selftest_real", '''
        def main():
            results.clear()
            check("guarded rejects a negative", dep.guarded(-1) is False)
            check("guarded accepts a positive", dep.guarded(1) is True)
            raise SystemExit(0 if all(results) else 1)
    ''')

    def mutate(mod):
        import selftest_dep as d
        d.guarded = lambda x: True                 # always-true: the guard is gone
        mod.dep = d

    base = falsify.run_suite(s)
    r = falsify.run_suite(s, mutate)
    check("baseline of the throwaway suite is green", base.ok == base.total == 2,
          f"{base.ok}/{base.total}")
    check("breaking the guard makes the named check FAIL",
          any("rejects a negative" in f for f in r.failed),
          f"failed: {r.failed}")


def check_vacuous_check_survives():
    """2. PLANTED: a check that cannot fail is reported as a SURVIVOR.

    This is the whole purpose. The suite is green, the mutation is real, and nothing
    notices — which is exactly the state four checks in this repository were in.
    """
    s = write_suite("selftest_vacuous", '''
        def main():
            results.clear()
            # Reads like a guard, tests nothing: True is True whatever dep does.
            check("guarded is self-consistent", dep.guarded(1) == dep.guarded(1))
            raise SystemExit(0)
    ''')

    def mutate(mod):
        import selftest_dep as d
        d.guarded = lambda x: True
        mod.dep = d

    r = falsify.run_suite(s, mutate)
    check("a vacuous check does not fail under a real mutation", r.failed == [],
          f"failed: {r.failed}")
    check("...so the suite still reports green, which is the danger", r.ok == r.total == 1)


def check_crash_is_not_a_kill():
    """3. An interpreter crash must NOT count as evidence a claim is defended."""
    s = write_suite("selftest_crash", '''
        def main():
            results.clear()
            check("guarded rejects a negative", dep.guarded(-1) is False)
            raise SystemExit(0 if all(results) else 1)
    ''')

    def mutate(mod):
        import selftest_dep as d
        def boom(x):
            raise RuntimeError("mutation exploded")
        d.guarded = boom
        mod.dep = d

    r = falsify.run_suite(s, mutate)
    check("a mutation that raises is reported as an error, not a pass",
          r.error is not None or r.failed != [], f"error={r.error} failed={r.failed}")
    # And the audit must not score it as killed.
    m = falsify.Mutation("selftest", s, "boom", ["guarded rejects a negative"], mutate)
    saved, falsify.MUTATIONS[:] = list(falsify.MUTATIONS), [m]
    try:
        rep = falsify.audit(verbose=False)
    finally:
        falsify.MUTATIONS[:] = saved
    killed = rep["n_killed"]
    check("...and the audit does not count a crash as a kill", killed == 0,
          f"n_killed={killed}")


def check_wrong_check_is_not_a_kill():
    """4. 'Something failed' is not the same as 'the right thing failed'."""
    s = write_suite("selftest_wrong", '''
        def main():
            results.clear()
            check("the load-bearing claim", True)          # never fails
            check("an unrelated bookkeeping check", dep.guarded(1) is True)
            raise SystemExit(0 if all(results) else 1)
    ''')

    def mutate(mod):
        import selftest_dep as d
        d.guarded = lambda x: False                # breaks only the unrelated check
        mod.dep = d

    m = falsify.Mutation("selftest", s, "wrongcheck", ["the load-bearing claim"], mutate)
    saved, falsify.MUTATIONS[:] = list(falsify.MUTATIONS), [m]
    try:
        rep = falsify.audit(verbose=False)
    finally:
        falsify.MUTATIONS[:] = saved
    surv = rep["survivors"]
    check("a mutation caught by the WRONG check is a survivor, not a kill",
          rep["n_killed"] == 0 and len(surv) == 1,
          f"killed={rep['n_killed']} survivors={len(surv)}")
    check("...and the collateral failure is recorded, not silently dropped",
          bool(surv and surv[0]["collateral"]),
          f"collateral={surv[0]['collateral'] if surv else None}")


def check_module_level_suite_is_handled():
    """5. A suite that runs at import and exits at module level still works.

    verify_operating_point is shaped this way. A runner built on import/reload cannot
    read it at all, and the module-level SystemExit escapes `except Exception` and
    terminates the whole audit — silently, with that suite's own exit code.
    """
    s = write_suite("selftest_modlevel", '''
        check("guarded rejects a negative", dep.guarded(-1) is False)
        raise SystemExit(0 if all(results) else 1)
    ''')
    base = falsify.run_suite(s)
    check("a module-level suite is read without terminating the audit",
          base.error is None and base.total == 1, f"{base.ok}/{base.total} err={base.error}")

    def mutate(mod):
        import selftest_dep as d
        d.guarded = lambda x: True
        mod.dep = d

    r = falsify.run_suite(s, mutate)
    check("...and a mutation applied BEFORE its body still reaches it",
          any("rejects a negative" in f for f in r.failed), f"failed={r.failed}")


def check_no_leak_between_runs():
    """6. One mutation must not survive into the next run."""
    s = write_suite("selftest_leak", '''
        def main():
            results.clear()
            check("guarded rejects a negative", dep.guarded(-1) is False)
            raise SystemExit(0 if all(results) else 1)
    ''')

    def mutate(mod):
        import selftest_dep as d
        d.guarded = lambda x: True
        mod.dep = d

    falsify.run_suite(s, mutate)                   # install the damage
    after = falsify.run_suite(s)                   # clean run immediately after
    check("the next clean run is unaffected by the previous mutation",
          after.failed == [] and after.ok == after.total,
          f"{after.ok}/{after.total} failed={after.failed}")


def check_regression_gate():
    """7. The gate fires on a coverage drop and on a newly-surviving mutation."""
    prev = {"coverage": 0.95,
            "killed_detail": [{"suite": "s", "name": "m1"}, {"suite": "s", "name": "m2"}]}
    now_drop = {"coverage": 0.80, "killed_detail": [{"suite": "s", "name": "m1"}]}
    probs = falsify.regression(prev, now_drop)
    check("a coverage drop is caught", any("coverage fell" in p for p in probs), str(probs))
    check("a mutation that now survives is caught",
          any("now SURVIVES" in p for p in probs), str(probs))
    same = falsify.regression(prev, dict(prev))
    check("an unchanged report is not a regression", same == [], str(same))


# --------------------------------------------------------------------------- #
def main() -> None:
    print("Falsification engine — adversarial checks on itself\n")
    for fn in (check_real_check_is_killed, check_vacuous_check_survives,
               check_crash_is_not_a_kill, check_wrong_check_is_not_a_kill,
               check_module_level_suite_is_handled, check_no_leak_between_runs,
               check_regression_gate):
        fn()
    ok, n = sum(results), len(results)
    print(f"\n{ok}/{n} checks passed")
    raise SystemExit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
