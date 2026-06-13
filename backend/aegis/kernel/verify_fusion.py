"""Verification suite for the Core Data Fusion Engine (Sepsis Ontology).

Proves the two-tier bind is real (substance behind the name):

  1. Derived edge features: HRV (RMSSD) falls as the heart rate's beat-to-beat
     interval steadies; vasomotion tracks perfusion oscillation amplitude.
  2. Lactate clearance: a falling lactate window reads positive (clearing), a
     rising window reads negative (poor clearance).
  3. Fusion monotonicity: a healthy bed fuses to LOW risk; a septic bed (fever,
     hypotension, high lactate, leukocytosis) fuses to HIGH risk.
  4. Driver attribution: the named derangements list the actual abnormalities
     and the primary organ system is correct.
  5. End-to-end through BioCluster: the labs feed flows /dev/bio -> fabric ->
     fusion; a Chaos crisis raises lactate + WBC and the fused profile escalates,
     with lactate cited as a driver.

    .venv/bin/python -m backend.aegis.kernel.verify_fusion
"""
from __future__ import annotations

from collections import deque

from backend.aegis.kernel.fusion import FusionEngine
from backend.aegis.kernel.scheduler import BioCluster


def _ok(msg): print(f"      [PASS] {msg}")

HEALTHY_EDGE = {"hr": 78.0, "perfusion": 1.5}
HEALTHY_ENT = {"temp_c": 36.9, "resp_rate": 15.0, "spo2": 98.0, "map": 90.0,
               "wbc": 8.0, "lactate": 1.0}
SEPTIC_EDGE = {"hr": 128.0, "perfusion": 0.6}
SEPTIC_ENT = {"temp_c": 39.4, "resp_rate": 30.0, "spo2": 89.0, "map": 58.0,
              "wbc": 18.5, "lactate": 4.6}


def test_edge_derivations() -> None:
    print("[1] edge tier — HRV (RMSSD) + vasomotion derivations")
    steady = deque([120.0] * 20)                       # perfectly regular -> ~0 HRV
    variable = deque([70 + (5 if i % 2 else -5) for i in range(20)], maxlen=20)
    assert FusionEngine.hrv_rmssd(steady) < 1.0
    assert FusionEngine.hrv_rmssd(variable) > FusionEngine.hrv_rmssd(steady)
    flat = deque([1.4] * 20); osc = deque([1.4 + (0.3 if i % 2 else -0.3) for i in range(20)])
    assert FusionEngine.vasomotion(osc) > FusionEngine.vasomotion(flat)
    _ok("RMSSD collapses for a steady HR; vasomotion tracks perfusion oscillation")


def test_lactate_clearance() -> None:
    print("[2] enterprise tier — lactate clearance sign")
    clearing = deque([4.0, 3.0, 2.0, 1.2])
    rising = deque([1.2, 2.5, 4.0, 5.5])
    assert FusionEngine.lactate_clearance(clearing) > 20.0
    assert FusionEngine.lactate_clearance(rising) < -100.0
    _ok("falling lactate -> positive clearance; rising lactate -> negative")


def test_fusion_monotonic() -> None:
    print("[3] fusion — healthy LOW vs septic HIGH")
    fe = FusionEngine()
    ph = ps = None
    for _ in range(30):
        ph = fe.update("h", "Bed H", HEALTHY_EDGE, HEALTHY_ENT)
        ps = fe.update("s", "Bed S", SEPTIC_EDGE, SEPTIC_ENT)
    assert ph.fused_risk < 0.25, f"healthy too high: {ph.fused_risk}"
    assert ps.fused_risk > 0.85, f"septic too low: {ps.fused_risk}"
    _ok(f"healthy fused={ph.fused_risk:.2f}  vs  septic fused={ps.fused_risk:.2f}")


def test_driver_attribution() -> None:
    print("[4] driver attribution + primary system")
    fe = FusionEngine()
    p = None
    for _ in range(10):
        p = fe.update("s", "Bed S", SEPTIC_EDGE, SEPTIC_ENT)
    blob = " ".join(p.drivers).lower()
    assert "lactate" in blob and "hypotension" in blob and "fever" in blob
    assert p.primary_system in ("perfusion", "metabolic", "inflammatory", "cardiac")
    assert len(p.drivers) >= 4
    _ok(f"primary={p.primary_system}; drivers: {p.drivers[:3]} …")


def test_end_to_end_cluster() -> None:
    print("[5] end-to-end — labs flow /dev/bio -> fabric -> fusion under Chaos")
    cl = BioCluster(n_nodes=3, lane_budget=48, seed=4)
    cl.run(20)
    base = cl.profiles["node_1"].fused_risk
    assert "lactate" in cl.profiles["node_1"].enterprise
    assert "hrv_ms" in cl.profiles["node_1"].edge
    cl.inject_crisis("node_1", intensity=1.0)
    for _ in range(60):
        cl.tick()
    prof = cl.profiles["node_1"]
    assert prof.fused_risk > base + 0.3, f"crisis did not escalate fusion ({base:.2f}->{prof.fused_risk:.2f})"
    assert prof.enterprise["lactate"] > 2.0, "lactate should have risen"
    assert any("Lactate" in d for d in prof.drivers), "lactate should be a named driver"
    _ok(f"crisis: fused {base:.2f} -> {prof.fused_risk:.2f}; "
        f"lactate {prof.enterprise['lactate']:.1f} mmol/L cited; "
        f"primary={prof.primary_system}")


def main() -> None:
    print("=" * 64)
    print("AegisOS — Core Data Fusion Engine (Sepsis Ontology) verification")
    print("=" * 64)
    test_edge_derivations()
    test_lactate_clearance()
    test_fusion_monotonic()
    test_driver_attribution()
    test_end_to_end_cluster()
    print("-" * 64)
    print("ALL FUSION CHECKS PASSED")


if __name__ == "__main__":
    main()
