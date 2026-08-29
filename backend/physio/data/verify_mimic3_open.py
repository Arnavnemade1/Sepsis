"""Adversarial audit of the open-access MIMIC-III waveform outcome pipeline.

    PYTHONPATH=. python -m backend.physio.data.verify_mimic3_open

The entire workaround rests on one claim: that a circulatory decompensation event
derived from an arterial line is a *patient* event and not a *plumbing* event. Arterial
lines disconnect, zero, flush, damp and flatline, and every one of those reads as
profound sustained hypotension to a naive MAP < 65 rule. The very first MIMIC segment
opened while building this had a median ABP of -1.9 mmHg.

So each artifact is planted here in turn and the gate is required to catch it, and the
event detector is required to produce nothing from it. A gate nobody has watched fail is
not a gate.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np

from backend.physio.data import mimic3_open as M

PASS, FAIL = "PASS", "FAIL"
_results: List[Tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def synth_abp(minutes: float, map_mmhg: float, fs: float = 125.0,
              pulse: float = 40.0, hr: float = 75.0) -> np.ndarray:
    """A plausible arterial waveform at a given mean pressure."""
    n = int(minutes * 60 * fs)
    t = np.arange(n) / fs
    wave = np.sin(2 * np.pi * (hr / 60.0) * t)
    wave = wave + 0.3 * np.sin(4 * np.pi * (hr / 60.0) * t)      # dicrotic-ish
    wave = wave / (np.max(np.abs(wave)) or 1.0)
    return map_mmhg + pulse * 0.5 * wave


def main() -> int:
    fs = 125.0

    # ------------------------------------------------- 1. the gate accepts real signal
    print("1. a clean arterial trace survives the gate")
    good = synth_abp(20, 85.0)
    v = M.abp_valid_mask(good, fs)
    check("a clean 85 mmHg trace is accepted", v.mean() > 0.95, f"{v.mean():.1%} kept")

    # --------------------------------------------------- 2. each artifact is caught
    print("\n2. planted artifacts — each must be rejected")
    cases = {
        "disconnection (zeroed line, ~0 mmHg)": np.zeros(int(10 * 60 * fs)) + 0.5,
        "negative offset (transducer open to air)": synth_abp(10, -2.0),
        "flush (square spike to 400 mmHg)": np.full(int(10 * 60 * fs), 400.0),
        "damped line (mean ok, pulse collapsed)": np.full(int(10 * 60 * fs), 70.0)
                                                  + 0.5 * np.sin(np.linspace(0, 500, int(10 * 60 * fs))),
        "flatline hold at 60 mmHg": np.full(int(10 * 60 * fs), 60.0),
    }
    for name, sig in cases.items():
        kept = M.abp_valid_mask(sig, fs).mean()
        check(f"rejected: {name}", kept < 0.05, f"{kept:.1%} of samples kept")

    # ------------------------------ 3. artifacts must not manufacture outcome events
    print("\n3. artifacts must produce NO decompensation events")
    for name, sig in cases.items():
        valid = M.abp_valid_mask(sig, fs)
        t, m = M.beat_map_series(sig, fs, valid)
        ev = M.find_decompensation_events(t, m)
        check(f"no event from: {name}", len(ev) == 0,
              f"{len(ev)} events, {len(t)} valid pressure bins")

    # ------------------------------------------- 4. a REAL event is still detected
    print("\n4. a real decompensation is still found (the gate is not just strict)")
    stable = synth_abp(15, 85.0)
    crash = synth_abp(15, 55.0)
    real = np.concatenate([stable, crash])
    v = M.abp_valid_mask(real, fs)
    t, m = M.beat_map_series(real, fs, v)
    ev = M.find_decompensation_events(t, m)
    check("a sustained fall 85 -> 55 mmHg after stability is detected",
          len(ev) == 1, f"{len(ev)} event(s) at {[round(e/60,1) for e in ev]} min "
                        f"(true onset 15.0 min)")
    if ev:
        check("...and it is timed to within a minute",
              abs(ev[0] - 15 * 60) < 60, f"detected {ev[0]/60:.2f} min vs true 15.00")

    # ---------------------------------- 5. the definition rejects CHRONIC hypotension
    print("\n5. chronic hypotension is not an event (this must detect ONSET)")
    chronic = synth_abp(40, 58.0)
    v = M.abp_valid_mask(chronic, fs)
    t, m = M.beat_map_series(chronic, fs, v)
    check("a patient hypotensive throughout yields no onset event",
          len(M.find_decompensation_events(t, m)) == 0,
          "no preceding stable run, so there is no onset to detect — otherwise this "
          "would be another patient-classifier, which the project already has")

    # ------------------------------------------- 6. brief dips are not events either
    print("\n6. a transient dip shorter than the sustain requirement is not an event")
    brief = np.concatenate([synth_abp(15, 85.0), synth_abp(2, 55.0), synth_abp(10, 85.0)])
    v = M.abp_valid_mask(brief, fs)
    t, m = M.beat_map_series(brief, fs, v)
    check("a 2-minute dip does not qualify (needs 5)",
          len(M.find_decompensation_events(t, m)) == 0)

    # ----------------------------------- 7. non-circularity, stated as a code property
    print("\n7. the label and the predictor come from different transducers")
    import inspect
    from backend.physio import recovery
    src = inspect.getsource(recovery.recovery_features) + inspect.getsource(
        recovery.find_perturbations)
    check("recovery features never touch a pressure channel",
          all(tok not in src for tok in ("abp", "ABP", "map_mmhg", "pressure")),
          "tau is computed from RR intervals only; the outcome is computed from the "
          "arterial line — a separate transducer, so the label is not circular")

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    print(f"\n{'=' * 72}\n{len(_results) - n_fail}/{len(_results)} checks passed")
    for s, nm, d in _results:
        if s == FAIL:
            print(f"  FAILED — {nm}: {d}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
