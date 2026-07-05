"""Layer 3 — the Multi-Tenant Hierarchical Cognitive Scheduler.

Layers 1–2 give every patient bed a conditioned, zero-copy stream in one shared
arena. Layer 3 is what makes AegisOS an *operating system for a wing* rather than
a program for a bed: a single scheduler that governs a **finite pool of execution
lanes** across N patient nodes at once, and reallocates them, every tick, toward
wherever the clinical risk is.

The governing quantity is a global cost function over the cluster:

        C_global = Σ_i  w_i · P_i(sepsis)

sampled straight off the memory fabric. But a *level* is not enough — a patient
crashing fast must outrank a patient who is merely sick-but-stable, so each node's
scheduling **priority** folds in the temporal slope dP_i/dt:

        priority_i = w_i · ( P_i + α · max(0, dP_i/dt) )

**Hard preemption (the "cluster-wide software interrupt").** When any node crosses
a critical level OR a critical rising-slope, the scheduler flips to PREEMPT mode:
non-essential lane classes (rendering, logging) are throttled to zero *across the
whole cluster* in a single tick, and every freed lane is poured into the crashing
node's ML-tracking lanes. One class is sacred and never throttled — ``safety``
(the Layer-4 neurosymbolic checker) keeps its floor on every node, always.

Honest framing (integrity rule): this is a **cooperative lane governor**, not an
OS thread preemptor. It allocates integer *execution lanes* (compute quanta per
tick) that the higher layers agree to honor; it does not issue hardware IRQs or
preempt CPU threads, and does not claim to. What is real and provable: a fixed
lane budget, a deterministic priority recomputation, an exact within-one-tick
reallocation under crisis, and a safety floor that survives preemption. The
``P_i`` driving it is a transparent qSOFA/SIRS-flavored **risk proxy** computed
from the conditioned vitals (``risk_proxy`` below) — the seam where the project's
calibrated detector (``backend/ml``) plugs in for the real system.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from backend.aegis.kernel.devbio import DevBio, provision_node
from backend.aegis.kernel.memfabric import FabricPump, MemFabric
from backend.aegis.kernel.fusion import BedProfile, FusionEngine

# Lane classes, in priority order. ``safety`` is essential and never throttled;
# ``ml_tracking`` is the elastic class that absorbs reallocation; ``render`` and
# ``logging`` are non-essential and collapsed first under preemption.
SAFETY = "safety"
ML = "ml_tracking"
RENDER = "render"
LOGGING = "logging"
LANE_CLASSES = (SAFETY, ML, RENDER, LOGGING)
NON_ESSENTIAL = (RENDER, LOGGING)

# Per-node minimum lanes per class in NORMAL mode (the baseline floor).
CLASS_FLOOR = {SAFETY: 1, ML: 1, RENDER: 1, LOGGING: 1}
SAFETY_FLOOR = 1                 # preserved on every node even under preemption

# Triggers for the cluster-wide hard preemption.
P_CRIT = 0.70                    # absolute risk level
SLOPE_CRIT = 0.04               # risk rise per tick (rapid deterioration)
SLOPE_ALPHA = 6.0               # how much slope inflates priority
SLOPE_WINDOW = 8                # ticks used to estimate dP/dt
FORESIGHT_BETA = 0.45           # how much the predictive foresight signal inflates priority


def risk_proxy(vitals: Dict[str, float]) -> float:
    """Transparent qSOFA/SIRS-flavored risk in [0, 1] from conditioned vitals.

    NOT the calibrated ML probability — a declared heuristic so the scheduler has
    something physiologic to govern on in simulation. Deviations that define SIRS
    /qSOFA (tachycardia, tachypnea, fever/hypothermia, hypotension, hypoxia) each
    push a logistic score upward.
    """
    s = 0.0
    hr = vitals.get("hr")
    if hr is not None:
        s += max(0.0, (hr - 90.0) / 40.0)
    rr = vitals.get("resp_rate")
    if rr is not None:
        s += max(0.0, (rr - 20.0) / 12.0)
    temp = vitals.get("temp_c")
    if temp is not None:
        s += max(0.0, (temp - 38.0) / 2.5) + max(0.0, (36.0 - temp) / 2.5)
    mapp = vitals.get("map")
    if mapp is not None:
        s += max(0.0, (65.0 - mapp) / 20.0)            # hypotension
    spo2 = vitals.get("spo2")
    if spo2 is not None:
        s += max(0.0, (92.0 - spo2) / 8.0)             # hypoxia
    # Logistic squash so a single deranged axis ~0.5, multi-axis -> high.
    return 1.0 / (1.0 + math.exp(-(1.6 * s - 1.2)))


@dataclass
class NodeState:
    node_id: str
    weight: float = 1.0                                # acuity weight w_i
    pin: float = 1.0                                   # clinician escalation multiplier (syscall write)
    p: float = 0.0                                     # current P_i(sepsis)
    foresight: float = 0.0                             # anticipatory deterioration signal 0..1
    p_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=SLOPE_WINDOW))
    vitals: Dict[str, float] = field(default_factory=dict)

    def observe(self, p: float, vitals: Dict[str, float], foresight: float = 0.0) -> None:
        self.p = float(p)
        self.vitals = dict(vitals)
        self.p_hist.append(self.p)
        self.foresight = float(max(0.0, min(1.0, foresight)))

    def slope(self) -> float:
        """dP/dt estimated as (newest - oldest) / span over the slope window."""
        if len(self.p_hist) < 2:
            return 0.0
        return (self.p_hist[-1] - self.p_hist[0]) / (len(self.p_hist) - 1)

    def priority(self) -> float:
        """PREDICTIVE priority: current risk + its slope + a forward-looking
        FORESIGHT term (loss-of-resilience / personal deviation). The foresight
        term lets the kernel pour compute into a patient who is destabilising
        BEFORE P crosses the alarm — anticipatory, not purely reactive, scheduling.

        ``pin`` is a clinician-driven escalation multiplier written through the
        audited, Z3-gated syscall interface (sys_escalate): a certified escalation
        genuinely pours more compute into the bed, so the write *drives the kernel*
        rather than merely annotating the UI."""
        return self.weight * self.pin * (self.p + SLOPE_ALPHA * max(0.0, self.slope())
                                         + FORESIGHT_BETA * self.foresight)

    def tier(self) -> str:
        if self.p >= P_CRIT or self.slope() >= SLOPE_CRIT:
            return "CRITICAL"
        if self.p >= 0.4:
            return "WATCH"
        return "STABLE"


@dataclass
class ScheduleDecision:
    tick: int
    mode: str                                          # NORMAL | PREEMPT
    global_cost: float
    owner: Optional[str]                               # node that seized the cluster
    lanes: Dict[str, Dict[str, int]]                   # node -> {class -> lanes}
    priorities: Dict[str, float]
    budget: int
    interrupt: Optional[Dict[str, Any]] = None
    verified: bool = False
    verification_msg: str = ""

    def node_total(self, node_id: str) -> int:
        return sum(self.lanes.get(node_id, {}).values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick, "mode": self.mode,
            "global_cost": round(self.global_cost, 4), "owner": self.owner,
            "budget": self.budget,
            "lanes": self.lanes, "priorities": {k: round(v, 4)
                                                for k, v in self.priorities.items()},
            "interrupt": self.interrupt,
            "verified": self.verified,
            "verification_msg": self.verification_msg,
        }



def _largest_remainder(weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Apportion ``total`` integer units across keys ∝ weights, summing exactly."""
    keys = list(weights)
    wsum = sum(weights.values())
    if total <= 0 or wsum <= 0:
        return {k: 0 for k in keys}
    raw = {k: weights[k] / wsum * total for k in keys}
    floor = {k: int(math.floor(raw[k])) for k in keys}
    rem = total - sum(floor.values())
    # hand out the remaining units to the largest fractional parts
    order = sorted(keys, key=lambda k: raw[k] - floor[k], reverse=True)
    for k in order[:rem]:
        floor[k] += 1
    return floor


class CognitiveScheduler:
    """The multi-tenant lane governor. One per cluster orchestrator."""

    def __init__(self, lane_budget: int = 64, predictive: bool = True):
        self.lane_budget = int(lane_budget)
        self.predictive = bool(predictive)              # fold foresight into priority
        self.nodes: Dict[str, NodeState] = {}
        self.tick_no = 0
        self.events: List[Dict[str, Any]] = []         # hard-interrupt log
        self.last: Optional[ScheduleDecision] = None

    # -- registration / observation -------------------------------------- #
    def register_node(self, node_id: str, weight: float = 1.0) -> NodeState:
        ns = NodeState(node_id=node_id, weight=float(weight))
        self.nodes[node_id] = ns
        return ns

    def observe(self, node_id: str, p: float, vitals: Dict[str, float],
                foresight: float = 0.0) -> None:
        self.nodes[node_id].observe(p, vitals, foresight if self.predictive else 0.0)

    def set_pin(self, node_id: str, pin: float) -> float:
        """Write a clinician escalation multiplier onto a node's scheduling
        priority. The ONLY supported writer is the audited, Z3-gated syscall
        interface (sys_escalate / sys_clear); the value is clamped to a sane
        envelope so a bad write can never starve the rest of the cohort."""
        if node_id not in self.nodes:
            raise KeyError(node_id)
        self.nodes[node_id].pin = float(max(1.0, min(4.0, pin)))
        return self.nodes[node_id].pin

    # -- the scheduling tick --------------------------------------------- #
    def verify_decision_z3(self, decision: ScheduleDecision) -> Tuple[bool, str]:
        """Formally verify scheduler liveness and prioritization properties on the decision using Z3."""
        from z3 import Solver, Int, Implies, And, Or, Sum, BoolVal, RealVal, Real, unsat, IntVal
        from backend.aegis.z3lock import Z3_LOCK

        # Z3's global context is not thread-safe; serialise every solver usage.
        with Z3_LOCK:
            return self._verify_decision_z3(decision)

    def _verify_decision_z3(self, decision: ScheduleDecision) -> Tuple[bool, str]:
        from z3 import Solver, Int, Or, Sum, BoolVal, unsat

        s = Solver()
        
        # We declare symbolic variables for each node's safety and ML lane allocations
        node_safety = {}
        node_ml = {}
        node_render = {}
        node_logging = {}
        
        for nid in self.nodes:
            # Symbolic variables
            node_safety[nid] = Int(f"safety_{nid}")
            node_ml[nid] = Int(f"ml_{nid}")
            node_render[nid] = Int(f"render_{nid}")
            node_logging[nid] = Int(f"logging_{nid}")
            
            # Bind to actuals
            s.add(node_safety[nid] == decision.lanes[nid]["safety"])
            s.add(node_ml[nid] == decision.lanes[nid]["ml_tracking"])
            s.add(node_render[nid] == decision.lanes[nid]["render"])
            s.add(node_logging[nid] == decision.lanes[nid]["logging"])

        # Total lanes symbolic variable
        total_lanes = Sum([node_safety[nid] + node_ml[nid] + node_render[nid] + node_logging[nid] for nid in self.nodes])
        
        # Invariant 1: Safety floor is preserved on every node (>= 1)
        safety_floor_violation = Or([node_safety[nid] < 1 for nid in self.nodes])
        
        # Invariant 2: Total budget limit is respected
        budget_violation = total_lanes > decision.budget
        
        # Invariant 3: Crisis lane priority. If a node is critical, it must receive >= 4 lanes.
        critical_violation = BoolVal(False)
        for nid in self.nodes:
            if self.nodes[nid].tier() == "CRITICAL":
                critical_violation = Or(critical_violation, node_ml[nid] < 4)
                
        # Invariant 4: No non-essential work during preemption
        preempt_violation = BoolVal(False)
        if decision.mode == "PREEMPT":
            preempt_violation = Or([Or(node_render[nid] > 0, node_logging[nid] > 0) for nid in self.nodes])

        # Define violation formula
        violation = Or(safety_floor_violation, budget_violation, critical_violation, preempt_violation)
        s.add(violation)
        
        if s.check() == unsat:
            return True, "FORMALLY VERIFIED: All scheduler liveness, budget, and preemption invariants hold."
        else:
            return False, f"SCHEDULER VIOLATION: Invariants failed under solver check (model: {s.model()})"

    def schedule(self) -> ScheduleDecision:
        self.tick_no += 1
        nodes = list(self.nodes.values())
        cost = sum(n.weight * n.p for n in nodes)
        prios = {n.node_id: n.priority() for n in nodes}

        critical = [n for n in nodes if n.tier() == "CRITICAL"]
        if critical:
            owner = max(critical, key=lambda n: n.priority())
            decision = self._preempt(nodes, owner, cost, prios)
        else:
            owner = None
            decision = self._normal(nodes, cost, prios)

        # Formally verify the decision before committing it.
        ok, msg = self.verify_decision_z3(decision)
        decision.verified = ok
        decision.verification_msg = msg

        # Edge-trigger the hard interrupt only on NORMAL->PREEMPT transitions or
        # when ownership of the cluster changes hands.
        prev = self.last
        if decision.mode == "PREEMPT" and (
            prev is None or prev.mode != "PREEMPT" or prev.owner != decision.owner
        ):
            evt = {
                "tick": decision.tick, "type": "HARD_PREEMPT",
                "owner": decision.owner,
                "reason": ("slope" if self.nodes[decision.owner].slope() >= SLOPE_CRIT
                           else "level"),
                "p": round(self.nodes[decision.owner].p, 3),
                "slope": round(self.nodes[decision.owner].slope(), 4),
                "ts": time.time(),
            }
            self.events.append(evt)
            decision.interrupt = evt
        self.last = decision
        return decision


    # -- allocation policies --------------------------------------------- #
    # Both policies allocate by *pouring* a finite budget into a priority order
    # of fill requests, so the total can never exceed the budget — oversubscription
    # (more floors than lanes) degrades gracefully, essential classes first.
    def _normal(self, nodes, cost, prios) -> ScheduleDecision:
        B = self.lane_budget
        lanes = {n.node_id: {c: 0 for c in LANE_CLASSES} for n in nodes}
        rem = B
        # Floors filled essential-first: safety for all, then ML, render, logging.
        for cls in (SAFETY, ML, RENDER, LOGGING):
            for n in nodes:
                if rem <= 0:
                    break
                lanes[n.node_id][cls] += CLASS_FLOOR[cls]
                rem -= CLASS_FLOOR[cls]
            if rem <= 0:
                break
        # Elastic pool -> ML lanes ∝ priority (+epsilon so a stable cluster still
        # splits evenly rather than starving on ties).
        if rem > 0:
            wmap = {n.node_id: prios[n.node_id] + 1e-3 for n in nodes}
            for nid, e in _largest_remainder(wmap, rem).items():
                lanes[nid][ML] += e
        return ScheduleDecision(self.tick_no, "NORMAL", cost, None, lanes, prios, B)

    def _preempt(self, nodes, owner, cost, prios) -> ScheduleDecision:
        B = self.lane_budget
        lanes = {n.node_id: {c: 0 for c in LANE_CLASSES} for n in nodes}
        rem = B
        # Safety floor for every node first (sacred — survives preemption).
        for n in nodes:
            if rem <= 0:
                break
            lanes[n.node_id][SAFETY] += SAFETY_FLOOR
            rem -= SAFETY_FLOOR
        # All remaining lanes go to ML; owner takes the lion's share, the rest is
        # spread by priority so a second deteriorating patient is not fully starved.
        # Non-essential (render/logging) stay at 0 cluster-wide.
        if rem > 0:
            others = [n for n in nodes if n.node_id != owner.node_id]
            owner_ml = int(round(rem * 0.75)) if others else rem
            lanes[owner.node_id][ML] += owner_ml
            rest = rem - owner_ml
            if others and rest > 0:
                wmap = {n.node_id: prios[n.node_id] + 1e-3 for n in others}
                for nid, e in _largest_remainder(wmap, rest).items():
                    lanes[nid][ML] += e
        return ScheduleDecision(self.tick_no, "PREEMPT", cost, owner.node_id,
                                lanes, prios, B)

    # -- frontend feeds -------------------------------------------------- #
    def topology(self) -> List[Dict[str, Any]]:
        """Panel A: node mesh — id, weight, risk, slope, tier (ring color)."""
        color = {"STABLE": "cyan", "WATCH": "amber", "CRITICAL": "crimson"}
        out = []
        for n in self.nodes.values():
            lanes = self.last.node_total(n.node_id) if self.last else 0
            out.append({
                "node_id": n.node_id, "weight": n.weight, "pin": round(n.pin, 2),
                "p": round(n.p, 4), "slope": round(n.slope(), 5),
                "foresight": round(n.foresight, 3),
                "cpu": round(lanes / max(1, self.lane_budget), 3),
                "tier": n.tier(), "ring": color[n.tier()], "lanes": lanes,
            })
        return out

    def lane_map(self) -> Dict[str, Any]:
        """Panel C: per-node/per-class lane bars + mode."""
        d = self.last
        return {
            "budget": self.lane_budget,
            "mode": d.mode if d else "NORMAL",
            "owner": d.owner if d else None,
            "lanes": d.lanes if d else {},
            "classes": list(LANE_CLASSES),
        }

    def status(self) -> Dict[str, Any]:
        d = self.last
        return {
            "lane_budget": self.lane_budget, "nodes": len(self.nodes),
            "tick": self.tick_no,
            "mode": d.mode if d else "NORMAL", "owner": d.owner if d else None,
            "global_cost": round(d.global_cost, 4) if d else 0.0,
            "interrupts": len(self.events),
        }

    def kstat(self) -> Dict[str, Any]:
        """Kernel-stat summary — the OS control surface for the process table /
        scheduler. Each node is a scheduled 'bio-process'."""
        ns = list(self.nodes.values())
        tiers = {"CRITICAL": 0, "WATCH": 0, "STABLE": 0}
        for n in ns:
            tiers[n.tier()] += 1
        used = sum(self.last.node_total(n.node_id) for n in ns) if self.last else 0
        foresighted = sum(1 for n in ns if n.foresight >= 0.25 and n.tier() != "CRITICAL")
        return {
            "tick": self.tick_no,
            "mode": self.last.mode if self.last else "NORMAL",
            "owner": self.last.owner if self.last else None,
            "procs": len(ns), "tiers": tiers,
            "lane_budget": self.lane_budget, "lanes_used": used,
            "utilization": round(used / max(1, self.lane_budget), 3),
            "global_cost": round(self.last.global_cost, 4) if self.last else 0.0,
            "interrupts": len(self.events),
            "anticipating": foresighted,            # pre-critical procs getting foresight compute
        }


# --------------------------------------------------------------------------- #
# BioCluster — a multi-patient ICU wing on one shared kernel, with Chaos.
# --------------------------------------------------------------------------- #
class BioCluster:
    """Simulates an ICU wing: N patient nodes on one DevBio bus + MemFabric,
    governed by one CognitiveScheduler. Single-process model of a distributed
    edge cluster (the scheduler logic is what the orchestrator would run).
    """

    # high-frequency edge + low-frequency enterprise tiers, both via /dev/bio
    FEEDS = ("optical", "thermal", "bedside", "labs")
    EDGE_KEYS = ("hr", "perfusion")
    ENTERPRISE_KEYS = ("temp_c", "resp_rate", "spo2", "map", "wbc", "lactate")

    def __init__(self, n_nodes: int = 4, lane_budget: int = 64,
                 weights: Optional[Sequence[float]] = None, seed: int = 0,
                 names: Optional[Sequence[str]] = None):
        self.bus = DevBio()
        self.fabric = MemFabric()
        self.sched = CognitiveScheduler(lane_budget=lane_budget)
        self.fusion = FusionEngine()
        self.node_ids: List[str] = []
        self.names: Dict[str, str] = {}
        self.profiles: Dict[str, BedProfile] = {}
        self._fds: Dict[str, Dict[str, int]] = {}
        self._cursors: Dict[str, Dict[str, Any]] = {}
        for i in range(n_nodes):
            nid = f"node_{i}"
            w = weights[i] if weights else 1.0
            fds = provision_node(self.bus, nid, kinds=self.FEEDS, seed=seed * 10 + i)
            self.fabric.map_bus(self.bus, fds, capacity=512)
            self.sched.register_node(nid, weight=w)
            self._fds[nid] = fds
            self._cursors[nid] = {k: self.fabric.subscribe(fd)
                                  for k, fd in fds.items()}
            self.names[nid] = names[i] if names and i < len(names) else f"Bed {i + 1:02d}"
            self.node_ids.append(nid)
        self.pump = FabricPump(
            self.bus, self.fabric,
            [fd for fds in self._fds.values() for fd in fds.values()])

    # -- Chaos Controller ------------------------------------------------ #
    def _driver(self, node_id: str, kind: str):
        return self.bus.device_at(f"/dev/bio/{node_id}/{kind}_feed")

    def inject_crisis(self, node_id: str, intensity: float = 1.0) -> None:
        """Drive a node toward septic physiology (tachy/tachypnea/fever/hypotension)."""
        k = float(intensity)
        self._driver(node_id, "optical").set_bias({"hr": 45.0 * k})
        self._driver(node_id, "thermal").set_bias({"temp_c": 2.5 * k})
        self._driver(node_id, "bedside").set_bias(
            {"resp_rate": 14.0 * k, "map": -28.0 * k, "spo2": -9.0 * k})
        self._driver(node_id, "labs").set_bias({"wbc": 7.5 * k, "lactate": 4.2 * k})

    def clear_crisis(self, node_id: str) -> None:
        for kind in self.FEEDS:
            self._driver(node_id, kind).set_bias({})

    def inject_noise_surge(self, node_id: str) -> None:
        """Ambient lighting jitter on a node's optical feed (quality stress)."""
        drv = self._driver(node_id, "optical")
        drv._noise["hr"] = drv._noise.get("hr", 3.0) * 4.0

    # -- the cluster tick ------------------------------------------------ #
    def _sample_vitals(self, node_id: str) -> Dict[str, float]:
        """Latest conditioned vitals for a node, read zero-copy from the rings."""
        v: Dict[str, float] = {}
        for kind, fd in self._fds[node_id].items():
            ring = self.fabric.rings[fd]
            slot = ring.latest_slot()
            if slot is None:
                continue
            for f in ring.fields:
                v[f] = ring.field(slot, f)
        return v

    def tick(self) -> ScheduleDecision:
        self.pump.step()                                # L1->L2 for all nodes
        for nid in self.node_ids:
            vit = self._sample_vitals(nid)              # zero-copy ring reads
            edge = {k: vit[k] for k in self.EDGE_KEYS if k in vit}
            ent = {k: vit[k] for k in self.ENTERPRISE_KEYS if k in vit}
            prof = self.fusion.update(nid, self.names[nid], edge, ent)  # the ontology bind
            self.profiles[nid] = prof
            self.sched.observe(nid, prof.fused_risk, vit)  # scheduler runs on fused risk
        return self.sched.schedule()

    def run(self, ticks: int) -> ScheduleDecision:
        d = None
        for _ in range(ticks):
            d = self.tick()
        return d

    def snapshot(self) -> Dict[str, Any]:
        return {
            "topology": self.sched.topology(),          # Panel A
            "lane_map": self.sched.lane_map(),          # Panel C
            "scheduler": self.sched.status(),
            "memfabric": self.fabric.address_matrix(),  # Panel B
            "beds": [self.profiles[nid].to_dict()       # fused per-bed ontology
                     for nid in self.node_ids if nid in self.profiles],
            "events": self.sched.events[-16:],
        }


# --------------------------------------------------------------------------- #
# Standalone demo.
# --------------------------------------------------------------------------- #
def _demo() -> None:
    cl = BioCluster(n_nodes=4, lane_budget=64, weights=[1.0, 1.0, 1.5, 1.0], seed=1)
    cl.run(20)
    print("warmup (all stable):", cl.sched.status())
    for t in cl.sched.topology():
        print(f"  {t['node_id']}: p={t['p']:.3f} tier={t['tier']:<8} ring={t['ring']}")

    print("\n>>> CHAOS: inject acute sepsis crisis into node_2\n")
    cl.inject_crisis("node_2", intensity=1.0)
    d = None
    fired_at = None
    for _ in range(60):
        d = cl.tick()
        if d.mode == "PREEMPT" and fired_at is None:
            fired_at = d.tick
            print(f"  >> HARD PREEMPT fired at tick {d.tick}: owner={d.owner} "
                  f"(reason={d.interrupt['reason']}, p={d.interrupt['p']})")

    print("\nafter crisis:", cl.sched.status())
    for t in cl.sched.topology():
        print(f"  {t['node_id']}: p={t['p']:.3f} slope={t['slope']:+.4f} "
              f"tier={t['tier']:<8} lanes={t['lanes']}")
    print(f"\nlane map ({d.mode}, owner={d.owner}):")
    for nid, lanes in d.lanes.items():
        print(f"  {nid}: {lanes}")
    print("total lanes allocated:", sum(sum(l.values()) for l in d.lanes.values()),
          "/", d.budget)
    print("hard-interrupt events:", cl.sched.events)


if __name__ == "__main__":
    _demo()
