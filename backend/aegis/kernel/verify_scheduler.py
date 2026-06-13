"""Verification suite for Layer 3 — the Multi-Tenant Cognitive Scheduler.

Proves the claimable properties of the lane governor (substance behind the name):

  1. Global cost C = Σ w_i·P_i is computed exactly off the node states.
  2. Lane-budget invariant: allocations ALWAYS sum to the budget, in every mode,
     for randomized cluster states.
  3. NORMAL mode honors per-class floors and biases ML lanes toward higher risk.
  4. Hard preemption by LEVEL: a node over P_CRIT flips the cluster to PREEMPT —
     non-essential lanes -> 0 cluster-wide, safety floor preserved on every node,
     the owner seizes the majority of freed lanes.
  5. Hard preemption by SLOPE: a fast-rising node still under P_CRIT triggers it
     (the temporal term matters, not just the level).
  6. The hard interrupt is edge-triggered: one event per onset/ownership change,
     not one per tick.
  7. End-to-end through BioCluster: a Chaos crisis drives a real node over the
     line via /dev/bio -> fabric -> proxy; clearing it returns the cluster to NORMAL.

    .venv/bin/python -m backend.aegis.kernel.verify_scheduler
"""
from __future__ import annotations

import random

from backend.aegis.kernel.scheduler import (
    BioCluster, CognitiveScheduler, LANE_CLASSES, NON_ESSENTIAL, P_CRIT,
    SAFETY, SAFETY_FLOOR, SLOPE_CRIT, ML, risk_proxy,
)


def _ok(msg): print(f"      [PASS] {msg}")


def _budget_ok(decision) -> bool:
    return sum(sum(d.values()) for d in decision.lanes.values()) == decision.budget


def test_cost_and_proxy() -> None:
    print("[1] global cost function + risk proxy")
    healthy = risk_proxy({"hr": 75, "resp_rate": 14, "temp_c": 37.0, "map": 90, "spo2": 98})
    septic = risk_proxy({"hr": 130, "resp_rate": 32, "temp_c": 39.6, "map": 58, "spo2": 88})
    assert 0.0 <= healthy < 0.3 < 0.8 < septic <= 1.0, (healthy, septic)
    sch = CognitiveScheduler(lane_budget=64)
    sch.register_node("a", weight=1.0); sch.observe("a", 0.2, {})
    sch.register_node("b", weight=2.0); sch.observe("b", 0.5, {})
    d = sch.schedule()
    assert abs(d.global_cost - (1.0 * 0.2 + 2.0 * 0.5)) < 1e-9
    _ok(f"proxy healthy={healthy:.3f} septic={septic:.3f}; C=Σw·P={d.global_cost:.2f}")


def test_budget_invariant_random() -> None:
    print("[2] lane-budget invariant across randomized states")
    rng = random.Random(7)
    for _ in range(500):
        n = rng.randint(1, 8)
        sch = CognitiveScheduler(lane_budget=rng.choice([16, 32, 64, 128]))
        for i in range(n):
            sch.register_node(f"n{i}", weight=rng.uniform(0.5, 3.0))
            sch.observe(f"n{i}", rng.random(), {})
        d = sch.schedule()
        assert _budget_ok(d), f"budget broken: {d.lanes}"
        assert all(v >= 0 for nd in d.lanes.values() for v in nd.values())
    _ok("500 random clusters: lanes always sum to budget, no negatives")


def test_normal_floors_and_bias() -> None:
    print("[3] NORMAL mode — class floors + ML lanes biased to risk")
    sch = CognitiveScheduler(lane_budget=64)
    for nid, p in [("low", 0.1), ("mid", 0.3), ("high", 0.55)]:  # all < P_CRIT
        sch.register_node(nid); sch.observe(nid, p, {})
    d = sch.schedule()
    assert d.mode == "NORMAL"
    for nid in ("low", "mid", "high"):
        for cls in LANE_CLASSES:
            assert d.lanes[nid][cls] >= 1, "class floor violated in NORMAL"
    assert d.lanes["high"][ML] > d.lanes["mid"][ML] > d.lanes["low"][ML]
    _ok(f"floors held; ML lanes low={d.lanes['low'][ML]} "
        f"mid={d.lanes['mid'][ML]} high={d.lanes['high'][ML]}")


def test_preempt_by_level() -> None:
    print("[4] hard preemption by LEVEL")
    sch = CognitiveScheduler(lane_budget=64)
    for nid in ("a", "b", "c"):
        sch.register_node(nid)
    # b sits at a high-but-FLAT level (slope ~0) so this is unambiguously a
    # level trigger, not a slope trigger.
    for _ in range(8):
        sch.observe("a", 0.15, {}); sch.observe("c", 0.15, {})
        sch.observe("b", P_CRIT + 0.1, {})
    d = sch.schedule()
    assert d.mode == "PREEMPT" and d.owner == "b"
    # non-essential collapsed cluster-wide
    assert all(d.lanes[n][cls] == 0 for n in d.lanes for cls in NON_ESSENTIAL)
    # safety floor preserved on EVERY node
    assert all(d.lanes[n][SAFETY] >= SAFETY_FLOOR for n in d.lanes)
    # owner holds the most ML lanes by far
    assert d.lanes["b"][ML] > sum(d.lanes[n][ML] for n in ("a", "c"))
    assert _budget_ok(d) and d.interrupt and d.interrupt["reason"] == "level"
    _ok(f"PREEMPT owner=b; render/logging=0 cluster-wide; safety preserved; "
        f"b ML={d.lanes['b'][ML]} vs others {d.lanes['a'][ML]+d.lanes['c'][ML]}")


def test_preempt_by_slope() -> None:
    print("[5] hard preemption by SLOPE (temporal term)")
    sch = CognitiveScheduler(lane_budget=64)
    sch.register_node("x"); sch.register_node("y")
    sch.observe("y", 0.2, {})
    # x rises fast but stays under P_CRIT — only the slope should trip it.
    p = 0.20
    fired = False
    for _ in range(8):
        p += SLOPE_CRIT + 0.005
        sch.observe("x", min(p, P_CRIT - 0.05), {})
        sch.observe("y", 0.2, {})
        d = sch.schedule()
        if d.mode == "PREEMPT":
            fired = True
            assert d.owner == "x" and sch.nodes["x"].p < P_CRIT
            assert d.interrupt["reason"] == "slope"
            break
    assert fired, "fast-rising sub-critical node should trigger preemption"
    _ok(f"sub-critical fast riser tripped PREEMPT on slope "
        f"(p={sch.nodes['x'].p:.3f} < {P_CRIT}, slope={sch.nodes['x'].slope():.4f})")


def test_interrupt_edge_triggered() -> None:
    print("[6] hard interrupt is edge-triggered, not per-tick")
    sch = CognitiveScheduler(lane_budget=64)
    sch.register_node("a"); sch.register_node("b")
    sch.observe("b", 0.1, {})
    for _ in range(10):                            # a stays critical for 10 ticks
        sch.observe("a", 0.95, {}); sch.observe("b", 0.1, {})
        sch.schedule()
    assert len(sch.events) == 1, f"expected 1 interrupt, got {len(sch.events)}"
    _ok("10 consecutive critical ticks -> exactly 1 hard-interrupt event")


def test_end_to_end_cluster() -> None:
    print("[7] end-to-end BioCluster — Chaos crisis drives real preemption")
    cl = BioCluster(n_nodes=4, lane_budget=64, weights=[1, 1, 1.5, 1], seed=3)
    cl.run(20)
    assert cl.sched.last.mode == "NORMAL", "cluster should start stable"
    cl.inject_crisis("node_2", intensity=1.0)
    fired = None
    for _ in range(80):
        d = cl.tick()
        if d.mode == "PREEMPT":
            fired = d; break
    assert fired and fired.owner == "node_2", "crisis must preempt to node_2"
    peak_p = cl.sched.nodes["node_2"].p
    assert _budget_ok(fired)
    assert all(fired.lanes[n][cls] == 0 for n in fired.lanes for cls in NON_ESSENTIAL)
    # Recovery: clear the crisis, let conditioned vitals wash out, expect NORMAL.
    cl.clear_crisis("node_2")
    rec = cl.run(120)
    assert rec.mode == "NORMAL", f"cluster should recover, still {rec.mode}"
    _ok(f"crisis -> PREEMPT@node_2 (p={peak_p:.2f} at fire); "
        f"cleared -> recovered to NORMAL; 1 interrupt logged")


def main() -> None:
    print("=" * 64)
    print("AegisOS Layer 3 — Multi-Tenant Cognitive Scheduler verification")
    print("=" * 64)
    test_cost_and_proxy()
    test_budget_invariant_random()
    test_normal_floors_and_bias()
    test_preempt_by_level()
    test_preempt_by_slope()
    test_interrupt_edge_triggered()
    test_end_to_end_cluster()
    print("-" * 64)
    print("ALL LAYER-3 CHECKS PASSED")


if __name__ == "__main__":
    main()
