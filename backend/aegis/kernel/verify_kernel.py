"""Run every Aegis bio-kernel layer verification in one shot.

    .venv/bin/python -m backend.aegis.kernel.verify_kernel

WHY THE BANNER CHANGED: this runner used to print "BIO-KERNEL L1-L4: ALL LAYERS
VERIFIED" while never running five of the suites in this package — ebpf, ebpfdrv,
supervisor, causal_graph and scheduler_invariants. The eBPF verifier is the load
gate for the drivers on three real beds, so "all layers" excluded the one that
admits untrusted bytecode. It now runs every suite in the package and the banner
names what actually ran.

Two calling conventions live here, which is why each entry declares its own:
the older suites raise AssertionError on failure and return None, the newer ones
return a process exit code. A runner that ignored the return value would report
success for a suite that had just failed.
"""
from __future__ import annotations

import sys
import time
import traceback
from typing import Callable, List, Tuple

from backend.aegis.kernel import (verify_bundle, verify_causal_graph, verify_devbio,
                                  verify_ebpf, verify_ebpfdrv, verify_executor,
                                  verify_fusion, verify_hdkernel, verify_hdsched,
                                  verify_memfabric, verify_neurosym,
                                  verify_scheduler, verify_scheduler_invariants,
                                  verify_supervisor)

# (label, callable, returns_exit_code)
SUITES: List[Tuple[str, Callable[[], object], bool]] = [
    ("L1  /dev/bio device-driver bus",        verify_devbio.main, False),
    ("L2  zero-copy memory fabric",           verify_memfabric.main, False),
    ("L2w single-writer kernel executor",     verify_executor.main, True),
    ("L3  multi-tenant cognitive scheduler",  verify_scheduler.main, False),
    ("L3  scheduler invariants (Z3)",         verify_scheduler_invariants.run_invariant_proofs, True),
    ("L3b HDSched CPU-scheduler benchmark",   verify_hdsched.main, False),
    ("L3c HDKernel cooperative kernel",       verify_hdkernel.main, False),
    ("L4  neurosymbolic egress cage",         verify_neurosym.main, False),
    ("eBPF static driver verifier",           verify_ebpf.main, True),
    ("eBPF driver ring binding",              verify_ebpfdrv.main, True),
    ("driver supervisor / fault isolation",   verify_supervisor.main, True),
    ("temporal causal graph",                 verify_causal_graph.main, True),
    ("core data fusion (sepsis ontology)",    verify_fusion.main, False),
    ("Hour-1 sepsis bundle engine",           verify_bundle.main, False),
]


#: Suites too slow for a per-push CI run. AEGIS_KERNEL_FAST=1 skips them; they
#: are never skipped silently — the banner lists what was skipped and why.
SLOW = {"L3b HDSched CPU-scheduler benchmark"}      # GBM surrogate search, minutes


def main() -> int:
    import os

    fast = os.getenv("AEGIS_KERNEL_FAST") == "1"
    print("#" * 64)
    print("#  AEGIS BIO-KERNEL — full package verification"
          + ("  [FAST: slow benchmarks skipped]" if fast else ""))
    print("#" * 64)
    passed: List[str] = []
    failed: List[Tuple[str, str]] = []
    skipped: List[str] = []
    for label, fn, returns_code in SUITES:
        if fast and label in SLOW:
            skipped.append(label)
            continue
        print("\n" + "-" * 64)
        print(f"--  {label}")
        print("-" * 64)
        t0 = time.perf_counter()
        try:
            out = fn()
            bad = returns_code and int(out or 0) != 0
            if bad:
                failed.append((label, f"exit code {out}"))
            else:
                passed.append(label)
        except BaseException as exc:                  # noqa: BLE001 - reported below
            traceback.print_exc()
            failed.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"    ({time.perf_counter() - t0:.1f}s)")

    print("\n" + "#" * 64)
    for label in passed:
        print(f"#  PASS  {label}")
    for label in skipped:
        print(f"#  SKIP  {label} (AEGIS_KERNEL_FAST=1)")
    for label, why in failed:
        print(f"#  FAIL  {label} — {why}")
    print("#" + "-" * 62)
    print(f"#  {len(passed)}/{len(SUITES) - len(skipped)} kernel suites passed"
          + (f", {len(skipped)} skipped" if skipped else ""))
    print("#" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
