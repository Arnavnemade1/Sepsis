"""Concurrency safety of the Z3 layer — the check that was missing.

Production crashed with this, on Render:

    z3.z3types.Z3Exception: b'ast is not an expression'      (inside a finalizer)
    z3.z3types.Z3Exception: b'unreachable'
    ASSERTION VIOLATION … UNEXPECTED CODE WAS REACHED

The z3-solver bindings share ONE global context, and that context is not
thread-safe. AegisOS touches Z3 from several threads at once: the simulator drive
loop, the cognitive scheduler's per-tick verification, and FastAPI's threadpool
running sync request handlers. A process-wide lock existed, but it was applied by
CONSUMERS — the neurosymbolic cage guarded its own ``certify`` — so any path that
reached ``ProtocolVerifier`` directly ran unguarded. Two did: the what-if
scheduler and AegisCore's protocol synthesis, the second of those on the drive
loop. A background thread building Z3 terms while a request thread solved on the
same context corrupted it, and the corruption surfaced later in a *finalizer*,
where there is no stack to blame.

The lock now lives on the resource rather than on each consumer who might forget.
These checks prove it, by doing the thing that used to break it:

  1. STRUCTURAL — every public Z3 entry point is guarded, found by walking the AST
     rather than by grepping, so a new unguarded builder fails the suite.
  2. CONCURRENT — the previously-unguarded paths are hammered from many threads at
     once, together with the guarded cage, and must complete with no exception and
     no corruption.
  3. DETERMINISM UNDER LOAD — a protocol's verdict must not depend on whether
     something else was verifying at the same time.

    python -m backend.aegis.verify_z3_safety
"""
from __future__ import annotations

import ast
import queue
import threading
from typing import Any, Dict, List, Tuple

PASS, FAIL = "PASS", "FAIL"
_results: List[Tuple[str, bool, str]] = []

# Modules that build Z3 terms, and the Z3 constructors that count as "building".
Z3_MODULES = [
    "backend/core/verifier.py",
    "backend/aegis/kernel/neurosym.py",
    "backend/aegis/kernel/scheduler.py",
    "backend/aegis/clinical_dsl.py",
    "backend/aegis/identity.py",
    "backend/aegis/rbac.py",
    "backend/aegis/pharmacopoeia.py",
    "backend/aegis/proofexport.py",
]
BUILDERS = {"Solver", "Optimize", "Real", "Bool", "Int", "Const", "Function"}


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def _unguarded_public(path: str) -> List[str]:
    """Public functions that build Z3 terms without holding the lock.

    Private helpers (leading underscore) are exempt: they are reached through a
    guarded public entry point, and guarding them again would only add re-entrant
    lock traffic. A PUBLIC builder is the dangerous case — anything can call it.
    """
    src = open(path).read()
    tree = ast.parse(src)
    bad: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        builds = any(isinstance(c, ast.Call)
                     and getattr(c.func, "id", "") in BUILDERS
                     for c in ast.walk(node))
        if not builds:
            continue
        decs = {getattr(d, "id", None) or getattr(d, "attr", None)
                for d in node.decorator_list}
        guarded = any(d and "guard" in str(d) for d in decs)
        body = ast.get_source_segment(src, node) or ""
        if not guarded and "Z3_LOCK" not in body:
            bad.append(f"{path.split('/')[-1]}:{node.name}")
    return bad


def main() -> int:
    print("\n=== 1. EVERY PUBLIC Z3 ENTRY POINT IS GUARDED ===")
    print("      (walked from the AST, so a newly added unguarded builder fails)")
    all_bad: List[str] = []
    for m in Z3_MODULES:
        try:
            all_bad += _unguarded_public(m)
        except FileNotFoundError:
            continue
    check("no public function builds Z3 terms without the lock", not all_bad,
          f"{len(Z3_MODULES)} modules audited"
          if not all_bad else "UNGUARDED: " + ", ".join(all_bad))

    from backend.core.verifier import ProtocolVerifier
    check("ProtocolVerifier.verify_protocol carries the guard",
          hasattr(ProtocolVerifier.verify_protocol, "__wrapped__"),
          "the resource owns the lock, not the caller")
    check("ProtocolVerifier.safety_model carries the guard",
          hasattr(ProtocolVerifier.safety_model, "__wrapped__"),
          "exported obligations build terms too")

    # ---------------------------------------------------------------- #
    print("\n=== 2. CONCURRENT ACCESS DOES NOT CORRUPT THE CONTEXT ===")
    print("      (this is the crash: the drive loop synthesising a protocol while")
    print("       a request thread verifies a bundle, on one global context)")
    from backend.aegis.kernel.neurosym import PROTOCOL, NeuroSymbolicRuntime

    errors: "queue.Queue[str]" = queue.Queue()
    verdicts: "queue.Queue[Tuple[str, bool]]" = queue.Queue()
    N_THREADS, N_ITERS = 8, 12

    hist = {"chf": True, "renal_failure": False, "penicillin_allergy": True}
    vitals = {"map": 58.0, "systolic_bp": 84.0, "diastolic_bp": 45.0}
    proto = {"fluid_rate": 4.0, "vasopressor_dose": 0.12, "antibiotic_code": 3}

    def direct_verifier(i: int) -> None:
        """The path AegisCore uses on the drive loop — previously unguarded."""
        try:
            pv = ProtocolVerifier()
            for _ in range(N_ITERS):
                ok, _msg, _pf = pv.verify_protocol(hist, vitals, "HIGH", proto)
                verdicts.put(("direct", ok))
        except Exception as exc:                       # noqa: BLE001
            errors.put(f"direct/{i}: {type(exc).__name__}: {exc}")

    def via_cage(i: int) -> None:
        """The path the bundle engine and the syscall gate use."""
        try:
            cage = NeuroSymbolicRuntime()
            for _ in range(N_ITERS):
                ok, _cert = cage.dispatch(PROTOCOL, proto, {
                    "risk_level": "HIGH", "vitals": vitals,
                    "clinical_history": hist})
                verdicts.put(("cage", ok))
        except Exception as exc:                       # noqa: BLE001
            errors.put(f"cage/{i}: {type(exc).__name__}: {exc}")

    def exporter(i: int) -> None:
        """Proof export builds its own terms off the same safety model."""
        try:
            from backend.aegis.proofexport import certify_core_protocol
            for _ in range(max(2, N_ITERS // 4)):
                certify_core_protocol(hist, vitals, "HIGH", proto)
        except Exception as exc:                       # noqa: BLE001
            errors.put(f"export/{i}: {type(exc).__name__}: {exc}")

    threads: List[threading.Thread] = []
    for i in range(N_THREADS):
        fn = (direct_verifier, via_cage, exporter)[i % 3]
        t = threading.Thread(target=fn, args=(i,), name=f"z3-{i}")
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    errs = []
    while not errors.empty():
        errs.append(errors.get())
    check(f"{N_THREADS} threads × {N_ITERS} verifications raise nothing", not errs,
          f"{verdicts.qsize()} verdicts returned"
          if not errs else f"{len(errs)} failures — first: {errs[0][:110]}")
    check("no thread hung", all(not t.is_alive() for t in threads),
          f"{sum(1 for t in threads if not t.is_alive())}/{len(threads)} finished")

    # ---------------------------------------------------------------- #
    print("\n=== 3. THE VERDICT DOES NOT DEPEND ON WHO ELSE IS SOLVING ===")
    seen: Dict[str, set] = {}
    while not verdicts.empty():
        src, ok = verdicts.get()
        seen.setdefault(src, set()).add(ok)
    check("every path agreed with itself across all threads",
          all(len(v) == 1 for v in seen.values()),
          " · ".join(f"{k}={list(v)}" for k, v in seen.items()) or "no verdicts")
    check("all paths agreed with each other",
          len({tuple(v)[0] for v in seen.values() if len(v) == 1}) <= 1,
          "the cage and the direct verifier reach the same conclusion")

    # A protocol this patient must NOT receive — verified under load, since a
    # corrupted context previously returned garbage rather than raising.
    pv = ProtocolVerifier()
    bad_ok, bad_msg, _ = pv.verify_protocol(
        {"penicillin_allergy": True}, {"map": 75.0}, "HIGH",
        {"fluid_rate": 2.0, "vasopressor_dose": 0.0, "antibiotic_code": 1})
    check("a penicillin-allergic pip-tazo order is still refused after the storm",
          not bad_ok and "Allergy" in bad_msg,
          bad_msg.split(":")[-1].strip()[:76])

    n_pass = sum(1 for _, ok_, _ in _results if ok_)
    n_tot = len(_results)
    print("\n" + "=" * 72)
    print(f"  Z3 CONCURRENCY VERIFICATION: {n_pass}/{n_tot} checks passed")
    if n_pass != n_tot:
        print("  FAILED:")
        for name, ok_, detail in _results:
            if not ok_:
                print(f"    · {name} — {detail}")
    print("=" * 72 + "\n")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
