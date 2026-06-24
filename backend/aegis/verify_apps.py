"""Phase-A validation: the app sandbox actually confines apps.

Proves the capability boundary is real, on a real fitted resonance gate + real
CinC-2019 frames:
  1. A well-behaved app reads its declared channels and emits its declared ALERT
     — and the ALERT is certified by the Z3 cage.
  2. An app that reads an UNDECLARED channel is blocked (CapabilityViolation),
     and never sees the value.
  3. An app that emits an UNDECLARED egress class is blocked before the cage.
  4. A deliberately-crashing app is CONTAINED — reported, never propagated.

    python -m backend.aegis.verify_apps
"""
from __future__ import annotations

from typing import Any, Dict

from backend.aegis.apps.sdk import AegisApp, AppContext, AppManifest, tick_app
from backend.aegis.apps.resonance_app import ResonanceApp
from backend.aegis.cohort import build_cohort
from backend.aegis.hdc_memory import HDCBioMemory, SOON_H
from backend.aegis.kernel.neurosym import ALERT, NOTE, PROTOCOL, NeuroSymbolicRuntime
from backend.aegis.resonance import CrossModalResonance


class RogueReadApp(AegisApp):
    """Declares it reads only HR, then tries to peek at lactate."""
    manifest = AppManifest(name="rogue_read", reads=("hr",), emits=())

    def on_tick(self, ctx: AppContext) -> Dict[str, Any]:
        ctx.read("hr")                     # allowed
        return {"stolen_lactate": ctx.read("lactate")}     # DENIED


class RogueEmitApp(AegisApp):
    """Declares it may emit only ALERT, then tries to write a treatment PROTOCOL."""
    manifest = AppManifest(name="rogue_emit", reads=("map",), emits=(ALERT,))

    def on_tick(self, ctx: AppContext) -> Dict[str, Any]:
        ctx.emit(PROTOCOL, {"fluid_rate": 30.0, "vasopressor_dose": 0.2,
                            "antibiotic_code": 1}, {"risk_level": "HIGH"})   # DENIED
        return {}


class CrashApp(AegisApp):
    manifest = AppManifest(name="crasher", reads=("hr",), emits=())

    def on_tick(self, ctx: AppContext) -> Dict[str, Any]:
        raise ValueError("intentional fault inside an app")


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def main() -> int:
    patients = build_cohort(n=12, seed=7)
    if not patients:
        print("CinC-2019 corpus unavailable — cannot run the real-data validation.")
        return 1
    rows = [r for p in patients for r in HDCBioMemory.rows_from_patient(p)]
    hdc = HDCBioMemory().build(rows)
    gate = CrossModalResonance.from_patients(hdc, patients)
    cage = NeuroSymbolicRuntime()
    # a real septic-soon frame to drive the resonance app
    frame = next(s for s, lab, tto, _ in rows
                 if lab == 1 or (tto is not None and 0 <= tto <= SOON_H))

    failures = 0

    _hr("1 · WELL-BEHAVED APP (resonance) — reads declared channels, emits ALERT")
    res = tick_app(ResonanceApp(gate), cage, frame)
    print(f"  ok={res.ok} · risk={res.output['gated_risk']:.2f} · "
          f"emitted={[e['egress'] for e in res.emitted]} · "
          f"ALERT certified={res.output['alert_certified']} · {res.ms:.2f} ms")
    ok1 = res.ok and res.emitted and res.emitted[0]["egress"] == ALERT
    failures += 0 if ok1 else 1

    _hr("2 · UNDECLARED READ — app declares {hr}, tries to read 'lactate'")
    res = tick_app(RogueReadApp(), cage, frame)
    print(f"  ok={res.ok} · violations={res.violations}")
    ok2 = (not res.ok) and any("lactate" in v for v in res.violations) and res.output is None
    failures += 0 if ok2 else 1

    _hr("3 · UNDECLARED EMIT — app may emit ALERT, tries to emit PROTOCOL")
    res = tick_app(RogueEmitApp(), cage, frame)
    print(f"  ok={res.ok} · violations={res.violations} · "
          f"reached_cage={len(res.emitted) > 0}")
    ok3 = (not res.ok) and any("PROTOCOL" in v for v in res.violations) and not res.emitted
    failures += 0 if ok3 else 1

    _hr("4 · FAULT CONTAINMENT — app raises mid-tick")
    res = tick_app(CrashApp(), cage, frame)
    print(f"  ok={res.ok} · error={res.error}  (kernel did not crash — we got here)")
    ok4 = (not res.ok) and res.error is not None and "intentional fault" in res.error
    failures += 0 if ok4 else 1

    _hr("VERDICT")
    for label, ok in [("well-behaved app runs + ALERT certified", ok1),
                      ("undeclared channel read blocked", ok2),
                      ("undeclared egress emit blocked", ok3),
                      ("in-app fault contained", ok4)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("\n  APP SANDBOX VALIDATED ✓" if failures == 0
          else f"\n  {failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
