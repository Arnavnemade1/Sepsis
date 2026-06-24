"""Phase-C validation: the package registry gates every install (verify-then-run).

Proves nothing unverified is ever registered:
  1. A valid third-party package INSTALLS (manifest → policy → audit → integrity →
     register) and then runs, emitting a Z3-certified ALERT.
  2. An OVER-CAPABLE package (lane budget beyond policy) is rejected at the policy gate.
  3. A package whose source imports a forbidden module (os/subprocess) is rejected
     by the static AST audit.
  4. A TAMPERED package (content changed after signing) is rejected at integrity.
  5. A bytecode (eBPF) package that loops forever is rejected by the real verifier.

    python -m backend.aegis.verify_pkg
"""
from __future__ import annotations

import inspect

from backend.aegis.apps.examples import tachy_app
from backend.aegis.apps.examples.tachy_app import TachyApp
from backend.aegis.apps.sdk import AppManifest, tick_app
from backend.aegis.initd import ServiceManager
from backend.aegis.kernel.ebpf import Verifier
from backend.aegis.kernel.neurosym import ALERT, NeuroSymbolicRuntime
from backend.aegis.pkg import AppPackage, PackageRegistry
from backend.aegis.procfs import ProcFS

HELPERS = ("bio_input", "bio_time")
TACHY_SRC = inspect.getsource(tachy_app)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def main() -> int:
    cage = NeuroSymbolicRuntime()
    aegisd = ServiceManager(cage, ProcFS())
    reg = PackageRegistry(aegisd, verifier=Verifier(HELPERS))
    failures = 0

    _hr("1 · VALID third-party package installs + runs")
    pkg = AppPackage(TachyApp.manifest, TACHY_SRC, kind="python", app=TachyApp()).sign()
    r = reg.install(pkg)
    print(f"  install → ok={r.ok} stage={r.stage} ({r.reason})")
    ran = tick_app(aegisd.services["tachy"].app, cage, {"hr": 130.0, "map": 55.0}) \
        if r.ok and "tachy" in aegisd.services else None
    if ran:
        print(f"  run → shock={ran.output['shock_index']} severity={ran.output['severity']} "
              f"ALERT certified={ran.output['alert_certified']}")
    ok1 = r.ok and ran is not None and ran.ok and ran.emitted[0]["egress"] == ALERT
    failures += 0 if ok1 else 1

    _hr("2 · OVER-CAPABLE package rejected by policy")
    greedy_m = AppManifest(name="greedy", reads=("hr",), emits=(ALERT,), lane_budget=99)
    gp = AppPackage(greedy_m, "x=1\n", kind="python", app=TachyApp()).sign()
    r = reg.install(gp)
    print(f"  install → ok={r.ok} stage={r.stage} · {r.detail}")
    ok2 = (not r.ok) and r.stage == "policy"
    failures += 0 if ok2 else 1

    _hr("3 · FORBIDDEN-import source rejected by AST audit")
    evil_m = AppManifest(name="evil", reads=("hr",), emits=())
    evil_src = "import os\nclass E:\n    def on_tick(self, ctx):\n        os.system('id')\n"
    ep = AppPackage(evil_m, evil_src, kind="python", app=TachyApp()).sign()
    r = reg.install(ep)
    print(f"  install → ok={r.ok} stage={r.stage} · {r.detail}")
    ok3 = (not r.ok) and r.stage == "verify" and any("os" in d for d in r.detail)
    failures += 0 if ok3 else 1

    _hr("4 · TAMPERED package rejected at integrity")
    tp = AppPackage(AppManifest(name="tamper", reads=("hr",), emits=()),
                    "y=1\n", kind="python", app=TachyApp()).sign()
    tp.source += "\n# slipped in after signing"           # mutate post-sign
    r = reg.install(tp)
    print(f"  install → ok={r.ok} stage={r.stage}")
    ok4 = (not r.ok) and r.stage == "integrity"
    failures += 0 if ok4 else 1

    _hr("5 · LOOPING bytecode package rejected by the eBPF verifier")
    loop_m = AppManifest(name="looppkg", reads=(), emits=())
    loop_src = "LDI r0, 0\nspin: ADDI r0, 1\nJMP spin\nEXIT"
    lp = AppPackage(loop_m, loop_src, kind="ebpf").sign()
    r = reg.install(lp)
    print(f"  install → ok={r.ok} stage={r.stage} · {r.detail}")
    ok5 = (not r.ok) and r.stage == "verify" and any("BackEdge" in d for d in r.detail)
    failures += 0 if ok5 else 1

    _hr("VERDICT")
    for label, ok in [("valid package installs + runs (certified)", ok1),
                      ("over-capable package rejected (policy)", ok2),
                      ("forbidden-import source rejected (audit)", ok3),
                      ("tampered package rejected (integrity)", ok4),
                      ("looping bytecode rejected (verifier)", ok5)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("\n  PACKAGE REGISTRY VALIDATED ✓" if failures == 0
          else f"\n  {failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
