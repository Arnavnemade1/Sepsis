"""Verification harness for Aegis-eBPF (the untrusted-driver sandbox).

Runs the static verifier against four real driver programs — one well-formed and
three hostile — and then *executes* the verified one against a real memory window.
Every verdict below is computed by :class:`Verifier`, not hardcoded: the script
declares the outcome it expects for each driver and FAILS loudly if the verifier
disagrees, so this doubles as the module's correctness test.

    python -m backend.aegis.kernel.verify_ebpf
"""
from __future__ import annotations

import numpy as np

from backend.aegis.kernel.ebpf import VM, Verifier, assemble

WIN = 100                                  # this driver's own ring: cells [0,100)
HELPERS = ("bio_input", "bio_time")        # the only kernel calls a driver may make

# --------------------------------------------------------------------------- #
# Driver #1 — a well-formed ECG ingest driver. Pulls a sample + a target index
# from the device FIFO, bound-checks the (untrusted) index, and writes the sample
# into its own window. Note r8 (callee-saved) carries the sample across the second
# CALL — using a caller-saved reg here would be caught as a clobbered/uninit read.
# --------------------------------------------------------------------------- #
VALID_ECG = """
        CALL bio_input        # r0 = ECG sample from the hardware FIFO
        MOV  r8, r0           # stash in callee-saved r8 (survives the next CALL)
        CALL bio_input        # r0 = target slot index (UNTRUSTED, unbounded)
        JLTI r0, 0, drop      # reject negative index
        JGTI r0, 99, drop     # reject index past the window  (WIN-1 = 99)
        STX  r0, 0, r8        # mem[idx] = sample — idx now proven in [0,99]
        LDI  r0, 0            # return OK
        EXIT
 drop:  LDI  r0, 1            # return DROP
        EXIT
"""

# Driver #2 — denial of service: an infinite loop via a backward jump.
LOOP_DOS = """
        LDI  r0, 0
 spin:  ADDI r0, 1
        JMP  spin             # back-edge → unbounded loop → would starve the scheduler
        EXIT
"""

# Driver #3 — cross-patient memory corruption: a store far outside the window,
# landing in a neighbouring bed's ring inside the shared arena.
CROSS_PATIENT = """
        CALL bio_input
        MOV  r8, r0
        LDI  r2, 100000       # index 100000 — well past this window
        STX  r2, 0, r8        # mem[100000] = sample → corrupts another patient
        LDI  r0, 0
        EXIT
"""

# Driver #4 — wild pointer: dereferences an untrusted input as an address with no
# bound check (the classic "use before you check" memory-safety bug).
WILD_POINTER = """
        CALL bio_input        # r0 = value
        MOV  r8, r0
        CALL bio_input        # r0 = index (UNTRUSTED, never bound-checked)
        STX  r0, 0, r8        # mem[idx] = value → idx is unbounded
        LDI  r0, 0
        EXIT
"""

CASES = [
    ("valid_ecg",     VALID_ECG,     True,  None),
    ("loop_dos",      LOOP_DOS,      False, "BackEdge"),
    ("cross_patient", CROSS_PATIENT, False, "OutOfBounds"),
    ("wild_pointer",  WILD_POINTER,  False, "UnboundedAddr"),
]


def _hr(title: str) -> None:
    print(f"\n{title}\n" + "─" * len(title))


def main() -> int:
    vrf = Verifier(HELPER_WHITELIST := HELPERS)
    failures = 0

    _hr(f"AEGIS-eBPF STATIC VERIFIER  ·  window = [0,{WIN})  ·  helpers = {{{', '.join(HELPERS)}}}")
    for name, src, expect_ok, expect_rule in CASES:
        prog = assemble(src)
        rep = vrf.verify(prog, WIN)
        ok_match = (rep.ok == expect_ok)
        rule_match = expect_ok or (rep.violations and rep.violations[0].rule == expect_rule)
        good = ok_match and rule_match
        failures += 0 if good else 1

        verdict = "✓ ACCEPTED" if rep.ok else "✗ REJECTED"
        tag = "OK" if good else "UNEXPECTED!"
        print(f"\n[{verdict}] driver {name!r}  ({len(prog)} instr)   <{tag}>")
        if rep.ok:
            print(f"    ⊢ {rep.proof}")
        else:
            v = rep.violations[0]
            print(f"    ✗ instr #{v.index}  {prog[v.index]!r}")
            print(f"      {v.rule}: {v.detail}")

    # ---- execute the verified driver: prove it with a real write ----------- #
    _hr("EXECUTING THE VERIFIED DRIVER (real numpy window)")
    fifo = iter([72.5, 7])                       # device feeds: sample 72.5, index 7
    vm = VM({"bio_input": lambda: next(fifo), "bio_time": lambda: 0.0})
    mem = np.zeros(WIN, dtype=np.float64)
    prog = assemble(VALID_ECG)
    res = vm.run(prog, mem)
    print(f"  ran {res.steps} steps (static bound was {vrf.verify(prog, WIN).max_steps}), "
          f"returned {res.returned}, trapped={res.trapped}")
    print(f"  writes: {res.writes}   →   mem[7] = {mem[7]}")
    exec_ok = res.trapped is None and res.writes == [(7, 72.5)] and mem[7] == 72.5
    failures += 0 if exec_ok else 1

    # ---- defense in depth: the VM traps a loop even WITHOUT the verifier ---- #
    _hr("DEFENSE IN DEPTH (VM runtime guard, verifier bypassed)")
    res2 = vm.run(assemble(LOOP_DOS), np.zeros(WIN), step_budget=64)
    print(f"  unverified loop driver → trapped after {res2.steps} steps: {res2.trapped}")
    guard_ok = res2.trapped is not None
    failures += 0 if guard_ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓" if failures == 0
          else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
