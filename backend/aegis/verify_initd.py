"""Phase-B validation: aegisd starts, orders, supervises and connects services.

On real CinC-2019 frames, proves:
  1. DEPENDENCY ORDER — aegisd starts resonance + radar before triage (which
     depends on them).
  2. INTER-SERVICE IPC — triage reads resonance's published /proc output and emits
     a Z3-certified NOTE built from it.
  3. SUPERVISED FAULT ISOLATION — a service that throws is contained,
     auto-restarted, and every OTHER service keeps ticking (gap = 0, measured).
  4. PROCFS — the namespace reflects live service state.

    python -m backend.aegis.verify_initd
"""
from __future__ import annotations

from typing import Any, Dict

from backend.aegis.apps.radar_app import RadarApp
from backend.aegis.apps.resonance_app import ResonanceApp
from backend.aegis.apps.sdk import AegisApp, AppContext, AppManifest
from backend.aegis.apps.triage_app import TriageApp
from backend.aegis.cohort import build_cohort
from backend.aegis.hdc_memory import HDCBioMemory
from backend.aegis.initd import RUNNING, ServiceManager
from backend.aegis.kernel.neurosym import NeuroSymbolicRuntime
from backend.aegis.procfs import ProcFS
from backend.aegis.resonance import CrossModalResonance

T = 40
FAULT_AT = 25


class FlakyApp(AegisApp):
    """A bystander service that faults once, on command."""
    manifest = AppManifest(name="flaky", reads=("hr",), emits=())

    def __init__(self) -> None:
        self.fault_now = False

    def on_tick(self, ctx: AppContext) -> Dict[str, Any]:
        ctx.read("hr")
        if self.fault_now:
            self.fault_now = False
            raise RuntimeError("injected service fault")
        return {"ok": True}


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
    # stream one septic patient's hours as the focus frames
    septic = next(p for p in patients if p.is_septic)
    frames = [s for (s, *_r) in HDCBioMemory.rows_from_patient(septic)]

    cage = NeuroSymbolicRuntime()
    procfs = ProcFS()
    aegisd = ServiceManager(cage, procfs)
    aegisd.register(ResonanceApp(gate))
    aegisd.register(RadarApp())
    flaky = FlakyApp()
    aegisd.register(flaky)
    aegisd.register(TriageApp())               # registered last, but depends on res+radar

    failures = 0

    _hr("1 · DEPENDENCY-ORDERED START")
    order = aegisd.start_all()
    print(f"  start order: {order}")
    ord_ok = (order.index("resonance") < order.index("triage")
              and order.index("radar") < order.index("triage"))
    failures += 0 if ord_ok else 1

    _hr(f"2 · RUN {T} TICKS (inject a fault on 'flaky' at tick {FAULT_AT})")
    triage_certs = 0
    ipc_ok = True
    for t in range(T):
        if t == FAULT_AT:
            flaky.fault_now = True
        frame = frames[t % len(frames)]
        res = aegisd.tick(frame)
        tri = res.get("triage")
        if tri and tri.ok:
            triage_certs += int(tri.output.get("note_certified", False))
            rs = res.get("resonance")
            if rs and rs.ok and abs(tri.output["fused_risk"] - rs.output["gated_risk"]) > 2e-3:
                ipc_ok = False                     # triage rounds to 3dp, resonance to 4dp
    svc = {s["name"]: s for s in aegisd.status()["services"]}
    for n in ("resonance", "radar", "triage", "flaky"):
        s = svc[n]
        print(f"  {n:10} state={s['state']:9} ticks={s['ticks']:2} "
              f"faults={s['faults']} restarts={s['restarts']}")

    _hr("3 · SUPERVISED FAULT ISOLATION (no other service dropped a tick)")
    bystanders_full = all(svc[n]["ticks"] == T for n in ("resonance", "radar", "triage"))
    flaky_recovered = svc["flaky"]["faults"] >= 1 and svc["flaky"]["state"] == RUNNING
    print(f"  flaky faulted+auto-restarted: {flaky_recovered} · "
          f"others ran all {T} ticks: {bystanders_full}")
    for e in [e for e in aegisd.status()["events"] if e["svc"] == "flaky"][-3:]:
        print(f"    [{e['kind']:9}] flaky · {e['detail']}")

    _hr("4 · PROCFS reflects live state")
    proc_services = procfs.read("/proc/services")
    proc_triage = procfs.read("/proc/triage")
    print(f"  /proc/services → {proc_services['running']} running, "
          f"{proc_services['n']} total")
    print(f"  /proc/triage   → level={proc_triage.get('level')} "
          f"risk={proc_triage.get('fused_risk')} note_certified={proc_triage.get('note_certified')}")
    proc_ok = (proc_services["n"] == 4 and "level" in proc_triage)

    _hr("VERDICT")
    checks = [
        ("aegisd starts dependencies before dependents", ord_ok),
        ("triage consumes resonance via /proc + certified NOTE", ipc_ok and triage_certs > 0),
        ("faulting service contained + auto-restarted", flaky_recovered),
        ("every other service kept ticking (gap = 0)", bystanders_full),
        ("/proc namespace reflects live state", proc_ok),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1
    print("\n  SERVICE MANAGER VALIDATED ✓" if failures == 0
          else f"\n  {failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
