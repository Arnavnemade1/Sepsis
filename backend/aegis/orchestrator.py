"""The Autonomous ICU Orchestrator — AegisOS as a self-governing clinical environment.

MEGA-CONCEPT 5 is the *integration* concept: it does not add new subsystems, it
**composes the ones that already exist** into one hierarchy governed by a single,
named safety invariant. The layers:

    Sensor   eBPF driver sandbox + Byzantine sensor/specialist consensus
    Bed      physiologic microkernel + causal graph + Z3 protocol synthesizer
    Wing     multi-patient cognitive scheduler (BioCluster)
    Global   epidemiological pattern brain + (redacted) multi-site consensus

The one contribution that ties it together is the **Non-Maleficence Governor**.

  THE HONESTY BOUNDARY (this is the whole concept, stated precisely):
  We do NOT claim the OS "proves no action can clinically harm a patient" — patient
  outcomes depend on biology no solver models, and that claim is unfalsifiable. What
  IS provable, and what this enforces, is an *architectural* invariant:

      ── No action reaches a patient without a safety certificate. ──
      Uncertified egress is structurally impossible: every action at every layer is
      routed through the invariant, which either returns a Z3/quorum-backed
      certificate or hard-blocks the action. Every governed action — allowed or
      blocked — is appended to a tamper-evident attestation ledger.

  So "Primum Non Nocere, enforced continuously" means: the system cannot emit an
  uncertified action, and that property is asserted by measurement (see
  verify_orchestrator.py), not asserted by adjective.

Deliberately scoped OUT (overclaim risk, same as the rest of the project): federated
*learning*, real cross-hospital networking, zero-knowledge proofs, HIPAA compliance.
"Multi-site" appears only as the existing need-to-know redaction projection.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.aegis.kernel.neurosym import (ACTUATOR, ALERT, PROTOCOL,
                                           NeuroSymbolicRuntime)
from backend.aegis.kernel.bundle import SepsisBundleEngine
from backend.aegis.causal_ingest import CausalGraphIngestor
from backend.aegis.causal_mine import BASE_CONDITIONS
from backend.aegis.consensus import ClinicalBFT, default_panel

# Layer names — the orchestration hierarchy.
SENSOR, BED, WING, GLOBAL = "sensor", "bed", "wing", "global"

INVARIANT_NAME = ("Primum Non Nocere (non-maleficence): no action reaches a patient "
                  "without a safety certificate")

# The wing/cage protocol synthesizer and the Clinical-Safety DSL number antibiotics
# in DIFFERENT codebooks. We translate by drug NAME so the exported proof certifies
# the *same drug* the cage approved — not whatever drug that integer means in the DSL.
# wing/cage:  0 None, 1 Pip-Tazo, 2 Vancomycin, 3 Cefepime, 4 Meropenem, 5 Levofloxacin
# dsl ENUMS:  NONE 0, Cefepime 1, PipTazo 2, Vancomycin 3, Meropenem 4, …, Ciprofloxacin 6
_WING_ABX_NAME = {0: "None", 1: "Piperacillin-Tazobactam", 2: "Vancomycin",
                  3: "Cefepime", 4: "Meropenem", 5: "Levofloxacin"}
# Levofloxacin has no exact DSL enum; map to Ciprofloxacin (same class, also
# penicillin-free) so the allergy invariant is evaluated faithfully. Noted, not hidden.
_WING_TO_DSL_ABX = {0: 0, 1: 2, 2: 3, 3: 1, 4: 4, 5: 6}


def _canon(o: Any) -> bytes:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str).encode()


def _risk_level(fused: float) -> str:
    return "HIGH" if fused >= 0.6 else "MEDIUM" if fused >= 0.35 else "LOW"


def _norm_vitals(prof) -> Dict[str, float]:
    """Normalize a BedProfile's tiers into the key names the predicates expect."""
    e = dict(getattr(prof, "enterprise", {}) or {})
    edge = dict(getattr(prof, "edge", {}) or {})
    return {
        "map": e.get("map", 90.0),
        "lactate": e.get("lactate", 1.0),
        "heart_rate": edge.get("hr", e.get("hr", 75.0)),
        "respiratory_rate": e.get("resp_rate", 14.0),
        "spo2": e.get("spo2", 98.0),
    }


# --------------------------------------------------------------------------- #
# The per-action certificate + the append-only attestation ledger.
# --------------------------------------------------------------------------- #
@dataclass
class NonMaleficenceCertificate:
    ok: bool
    layer: str
    action: str
    target: Optional[str]
    z3_status: str
    message: str
    violations: List[str]
    seq: int
    prev_hash: str
    hash: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "layer": self.layer, "action": self.action,
                "target": self.target, "z3_status": self.z3_status,
                "message": self.message, "violations": self.violations,
                "seq": self.seq, "prev_hash": self.prev_hash[:12],
                "hash": self.hash[:12], "ts": round(self.ts, 3)}


class AttestationLedger:
    """Append-only, hash-chained record of every governed action (allowed or blocked)."""
    GENESIS = "genesis"

    def __init__(self):
        self.entries: List[NonMaleficenceCertificate] = []
        self._head = self.GENESIS

    def append(self, ok, layer, action, target, z3_status, message, violations
               ) -> NonMaleficenceCertificate:
        seq = len(self.entries)
        body = {"ok": ok, "layer": layer, "action": action, "target": target,
                "z3_status": z3_status, "message": message,
                "violations": violations, "seq": seq}
        h = hashlib.sha256(self._head.encode() + _canon(body)).hexdigest()
        cert = NonMaleficenceCertificate(ok, layer, action, target, z3_status,
                                         message, violations, seq, self._head, h)
        self.entries.append(cert)
        self._head = h
        from backend.aegis.kernel.journal import KJOURNAL, NOTICE, WARN
        KJOURNAL.append("clinical", "govern", severity=NOTICE if ok else WARN,
                        layer=layer, action=action, target=target, ok=bool(ok),
                        z3=z3_status, attestation=h[:12])
        return cert

    def verify(self) -> bool:
        prev = self.GENESIS
        for c in self.entries:
            body = {"ok": c.ok, "layer": c.layer, "action": c.action,
                    "target": c.target, "z3_status": c.z3_status,
                    "message": c.message, "violations": c.violations, "seq": c.seq}
            if c.prev_hash != prev or hashlib.sha256(
                    prev.encode() + _canon(body)).hexdigest() != c.hash:
                return False
            prev = c.hash
        return prev == self._head

    def head(self) -> str:
        return self._head


# --------------------------------------------------------------------------- #
# The named invariant — the single safety gate every action routes through.
# --------------------------------------------------------------------------- #
class NonMaleficenceInvariant:
    """Routes an action through the Z3 egress cage plus cross-layer hard guards.

    Returns (ok, z3_status, message, violations). Default-deny: an action it does
    not recognize is rejected, so nothing can slip past unrecognized."""

    name = INVARIANT_NAME

    def __init__(self, cage: NeuroSymbolicRuntime):
        self.cage = cage

    def check(self, layer: str, action: str, payload: Dict[str, Any],
              state: Dict[str, Any]):
        risk = state.get("risk_level", "MEDIUM")
        vit = state.get("vitals", {})
        hist = state.get("clinical_history", {})

        if action in ("treat", "protocol"):
            ok, cert = self.cage.dispatch(PROTOCOL, payload, {
                "risk_level": risk, "vitals": {"map": vit.get("map", 75.0)},
                "clinical_history": hist})
            return ok, cert.z3_status, cert.message, cert.violations

        if action in ("escalate", "alert"):
            target_sev = int(payload.get("severity", 0))
            # Cross-layer hard guard: never DOWNGRADE alarm on a critical patient.
            critical = (risk == "HIGH") or (vit.get("map", 90) < 65 and vit.get("lactate", 0) >= 2)
            if critical and target_sev < 2:
                return (False, "sat",
                        "non-maleficence: refusing to downgrade alarm on a critical patient",
                        ["alarm_downgrade_on_critical"])
            ok, cert = self.cage.dispatch(ALERT, {"severity": target_sev}, {
                "probability": float(payload.get("probability", 0.5)),
                "calibrated": True, "evidence_count": int(payload.get("evidence", 1))})
            return ok, cert.z3_status, cert.message, cert.violations

        if action in ("schedule", "pin"):
            # Cross-layer hard guard: a critical bed must keep a monitoring priority
            # floor — the scheduler may not de-prioritize a crashing patient.
            if state.get("tier") == "CRITICAL" and float(payload.get("pin", 1.0)) < 1.0:
                return (False, "sat",
                        "non-maleficence: critical bed cannot be de-prioritized below floor",
                        ["critical_priority_floor"])
            ok, cert = self.cage.dispatch(ACTUATOR, {
                "channel": "schedule", "value": float(payload.get("pin", 1.0)),
                "min": 1.0, "max": 4.0}, {})
            return ok, cert.z3_status, cert.message, cert.violations

        if action == "actuator":
            ok, cert = self.cage.dispatch(ACTUATOR, payload, {})
            return ok, cert.z3_status, cert.message, cert.violations

        # default-deny
        return False, "n/a", f"unknown action {action!r} — default-deny", ["unknown_action"]


# --------------------------------------------------------------------------- #
# The orchestrator.
# --------------------------------------------------------------------------- #
class Orchestrator:
    """Composes the four layers under one governed invariant + one attestation ledger."""

    def __init__(self, cluster, cage: Optional[NeuroSymbolicRuntime] = None,
                 bundle: Optional[SepsisBundleEngine] = None,
                 histories: Optional[Dict[str, Dict[str, Any]]] = None,
                 enable_proof_export: bool = True):
        self.cluster = cluster
        self.cage = cage or NeuroSymbolicRuntime()
        self.bundle = bundle or SepsisBundleEngine()
        self.invariant = NonMaleficenceInvariant(self.cage)
        self.ledger = AttestationLedger()
        self.histories = histories or {}
        self.graphs: Dict[str, CausalGraphIngestor] = {
            nid: CausalGraphIngestor(bed=nid) for nid in cluster.node_ids}
        self.panel = default_panel()

        # Portable proof-export: the cage ENFORCES the order; this emits the
        # third-party-checkable, Ed25519-signed evidence for the same invariants.
        self.enable_proof_export = enable_proof_export
        self._proof_ca = None
        self.proofs: Dict[int, Dict[str, Any]] = {}   # ledger seq -> portable cert
        self.last_proof: Optional[Dict[str, Any]] = None

    # -- bed state helper ------------------------------------------------ #
    def _state(self, bed: str) -> Dict[str, Any]:
        prof = self.cluster.profiles.get(bed)
        fused = float(getattr(prof, "fused_risk", 0.0)) if prof else 0.0
        vit = _norm_vitals(prof) if prof else {}
        tier = None
        node = getattr(self.cluster, "sched", None)
        if node is not None and bed in node.nodes:
            tier = node.nodes[bed].tier()
        return {"bed": bed, "risk_level": _risk_level(fused), "fused_risk": fused,
                "vitals": vit, "tier": tier,
                "clinical_history": self.histories.get(bed, {})}

    # -- portable, third-party-checkable proof export -------------------- #
    def _ca(self):
        """Lazy Ed25519 signing authority; disables itself (never breaks govern) if
        the crypto/keystore is unavailable."""
        if not self.enable_proof_export:
            return None
        if self._proof_ca is None:
            try:
                from backend.aegis.proofexport import CertificateAuthority
                self._proof_ca = CertificateAuthority()
            except Exception as exc:  # pragma: no cover
                print(f"[orchestrator] proof-export disabled ({exc})")
                self.enable_proof_export = False
                return None
        return self._proof_ca

    def export_protocol_proof(self, plan: Dict[str, Any], state: Dict[str, Any],
                              bind: Optional[Dict[str, Any]] = None):
        """Discharge the hospital DSL policy against this plan+context and return an
        Ed25519-signed, portable :class:`proofexport.ProofCertificate` — the evidence
        a third party re-checks with their own solver. ``None`` if export is off."""
        ca = self._ca()
        if ca is None:
            return None
        from backend.aegis.proofexport import certify_dsl_protocol
        hist = state.get("clinical_history", {}) or {}
        wing_abx = int(plan.get("antibiotic_code", 0))
        protocol = {
            "antibiotic_code": _WING_TO_DSL_ABX.get(wing_abx, 0),    # translated codebook
            "antibiotic_name": _WING_ABX_NAME.get(wing_abx, "Unknown"),
            "fluid_rate": float(plan.get("fluid_rate", 0.0)),
            "vasopressor_dose": float(plan.get("vasopressor_dose", 0.0)),
            # Hour-1 bundle: planned administration time (cap is the WITHIN-1hr rule).
            "antibiotic_time_hr": float(plan.get("antibiotic_time_hr", 0.5)),
        }
        context = {
            "risk_level": state.get("risk_level", "MEDIUM"),
            "chf": bool(hist.get("chf", False)),
            "penicillin_allergy": bool(hist.get("penicillin_allergy", False)),
            "renal_failure": bool(hist.get("renal_failure", False)),
        }
        try:
            return certify_dsl_protocol(protocol, context, ca=ca,
                                        policy_id="default-hospital-policy",
                                        context_salt=str(state.get("bed", "")),
                                        extra_subject=bind)
        except Exception as exc:  # pragma: no cover
            print(f"[orchestrator] proof export failed ({exc})")
            return None

    def proof_for(self, seq: int) -> Optional[Dict[str, Any]]:
        """The portable proof certificate bound to a ledger entry, if any."""
        return self.proofs.get(seq)

    # -- THE GATE: every action goes through here ------------------------ #
    def govern(self, layer: str, action: str, payload: Dict[str, Any],
               target: Optional[str] = None) -> NonMaleficenceCertificate:
        """Route an action through the non-maleficence invariant, record it on the
        attestation ledger, and apply the mutation only if certified.

        For treatment actions, also emit a portable, signed proof certificate bound
        to the ledger entry (the cage *enforces*; the proof is *exportable* evidence)."""
        from backend.aegis.z3lock import Z3_LOCK
        with Z3_LOCK:                       # cage + proof export both build Z3 solvers
            return self._govern(layer, action, payload, target)

    def _govern(self, layer: str, action: str, payload: Dict[str, Any],
                target: Optional[str] = None) -> NonMaleficenceCertificate:
        state = self._state(target) if target else {"risk_level": "MEDIUM", "vitals": {}}
        ok, z3, msg, viol = self.invariant.check(layer, action, payload, state)
        cert = self.ledger.append(ok, layer, action, target, z3, msg, viol)
        if action in ("treat", "protocol"):
            proof = self.export_protocol_proof(
                payload, state,
                bind={"ledger_seq": cert.seq, "ledger_hash": cert.hash[:12]})
            if proof is not None:
                pd = proof.to_dict()       # pristine: ledger link is in the SIGNED body
                self.proofs[cert.seq] = pd
                self.last_proof = pd
        if ok and target:
            # apply the certified mutation (best-effort, layer-appropriate)
            if action in ("treat", "protocol") and target in self.graphs:
                self.graphs[target].graph.add_event(
                    "protocol_decision", time.time(),
                    {"protocol": payload, "z3_certified": True, "z3_status": z3},
                    risk=state.get("fused_risk", 0.0))
            elif action in ("schedule", "pin") and hasattr(self.cluster, "sched"):
                try:
                    self.cluster.sched.set_pin(target, float(payload.get("pin", 1.0)))
                except Exception:
                    pass
        return cert

    # -- GLOBAL brain: recognize known signatures in a bed --------------- #
    def recognize_signatures(self, bed: str) -> List[str]:
        """Which known high-risk signatures the bed currently exhibits (the
        epidemiological brain's pattern match — an observed signature, not a
        diagnosis)."""
        vit = self._state(bed)["vitals"]
        return [c.label for c in BASE_CONDITIONS if c.test(vit)]

    # -- hierarchy status ----------------------------------------------- #
    def status(self) -> Dict[str, Any]:
        beds = self.cluster.node_ids
        sched = getattr(self.cluster, "sched", None)
        tiers = {}
        if sched is not None:
            for nid in beds:
                if nid in sched.nodes:
                    tiers[nid] = sched.nodes[nid].tier()
        crit = [b for b, t in tiers.items() if t == "CRITICAL"]
        return {
            "invariant": self.invariant.name,
            "attestation": {"actions": len(self.ledger.entries),
                            "head": self.ledger.head()[:12],
                            "integrity_ok": self.ledger.verify()},
            "hierarchy": {
                SENSOR: {
                    "ebpf_sandbox": "static driver verifier (termination/memory/capability)",
                    "byzantine_consensus": f"specialist panel N={len(self.panel)}",
                    "beds_instrumented": len(beds),
                },
                BED: {
                    "microkernel": "/dev/bio + zero-copy fabric + Z3 protocol synth",
                    "causal_graphs": {nid: g.graph.stats()["nodes"] for nid, g in self.graphs.items()},
                },
                WING: {
                    "scheduler": sched.status() if sched is not None else None,
                    "beds": len(beds), "critical": crit,
                },
                GLOBAL: {
                    "epidemiological_brain": f"pattern miner over {len(self.graphs)} bed graphs",
                    "multi_site_consensus": "redacted need-to-know projection (not ZK/HIPAA)",
                },
            },
        }

    # -- continuous attestation: the invariant holds across the wing ----- #
    def attest(self) -> Dict[str, Any]:
        """Prove the invariant currently holds across all beds: the ledger is
        intact, and every CRITICAL bed retains its priority floor (none silently
        de-prioritized). Returns a global attestation."""
        ledger_ok = self.ledger.verify()
        sched = getattr(self.cluster, "sched", None)
        breaches: List[Dict[str, Any]] = []
        per_bed = {}
        for nid in self.cluster.node_ids:
            tier = sched.nodes[nid].tier() if (sched and nid in sched.nodes) else None
            pin = float(sched.nodes[nid].pin) if (sched and nid in sched.nodes) else 1.0
            ok = not (tier == "CRITICAL" and pin < 1.0)
            per_bed[nid] = {"tier": tier, "pin": round(pin, 2), "ok": ok}
            if not ok:
                breaches.append({"bed": nid, "reason": "critical bed below priority floor"})
        ok = ledger_ok and not breaches
        return {"ok": ok, "ledger_ok": ledger_ok, "breaches": breaches,
                "per_bed": per_bed, "head": self.ledger.head()[:12],
                "invariant": self.invariant.name}

    # -- THE DEMO MOMENT: a governed cross-layer cascade ----------------- #
    def run_scenario(self, bed: str, warmup: int = 20, crisis_ticks: int = 60
                     ) -> Dict[str, Any]:
        """Induce a sepsis cascade in ``bed`` and walk the governed response down
        the hierarchy, ending in a signed, certified, immutably-logged order."""
        from backend.aegis.kernel.executor import EXEC
        from backend.aegis.z3lock import Z3_LOCK

        def _guarded() -> Dict[str, Any]:
            with Z3_LOCK:                   # cluster ticks + consensus + govern + proof: all Z3
                return self._run_scenario(bed, warmup, crisis_ticks)

        # This used to tick the cluster straight from the threadpool worker while
        # the live cluster loop ticked it too — two writers into one ring. It now
        # runs ON the writer thread, which also serialises it against the loop.
        # Lock order is executor -> Z3 (never the reverse): the writer thread only
        # ever takes Z3 non-blockingly, so nothing it holds can block a thread that
        # is itself waiting for the writer.
        #
        # Only a cluster whose arena the executor OWNS goes through it. A private
        # cluster (the boot warm-up's throwaway wing) has no other writer to be
        # serialised against — and routing it through the executor would deadlock
        # a caller that already holds Z3: it would wait on the writer while the
        # writer waits on Z3.
        arena = getattr(getattr(self.cluster, "fabric", None), "arena", None)
        if EXEC.owns(arena):
            return EXEC.call(_guarded)
        return _guarded()

    def _run_scenario(self, bed: str, warmup: int, crisis_ticks: int) -> Dict[str, Any]:
        timeline: List[Dict[str, Any]] = []
        self.cluster.run(warmup)

        # SENSOR → crisis enters through the (verified) driver plane.
        self.cluster.inject_crisis(bed, intensity=1.0)
        preempt = None
        for _ in range(crisis_ticks):
            d = self.cluster.tick()
            if getattr(d, "mode", None) == "PREEMPT" and preempt is None:
                preempt = {"tick": d.tick, "owner": d.owner}
        timeline.append({"layer": SENSOR, "event": "crisis injected via verified driver plane",
                         "bed": bed})

        st = self._state(bed)
        timeline.append({"layer": BED, "event": "microkernel detects deterioration",
                         "fused_risk": round(st["fused_risk"], 3), "tier": st["tier"]})

        # WING → cognitive scheduler reallocates compute lanes.
        sched_status = self.cluster.sched.status() if hasattr(self.cluster, "sched") else {}
        timeline.append({"layer": WING, "event": "cognitive scheduler reallocates lanes",
                         "preempt": preempt, "critical": [b for b in self.cluster.node_ids
                         if hasattr(self.cluster, "sched") and b in self.cluster.sched.nodes
                         and self.cluster.sched.nodes[b].tier() == "CRITICAL"]})

        # GLOBAL → epidemiological brain recognizes the signature + panel convenes.
        sigs = self.recognize_signatures(bed)
        det = 0.85 if st["risk_level"] == "HIGH" else 0.4
        cons = ClinicalBFT(self.panel).run_round(
            {"map": st["vitals"].get("map", 70), "risk_level": st["risk_level"],
             "clinical_history": st["clinical_history"],
             "infection_suspected": True}, deterioration=det)
        timeline.append({"layer": GLOBAL, "event": "epidemiological brain recognizes signature",
                         "signatures": sigs, "consensus_path": cons.path})

        # BED → formal protocol synthesizer compiles a plan (timed).
        prof = self.cluster.profiles.get(bed)
        t0 = time.perf_counter()
        plan = self.bundle._derive_protocol(prof, st["clinical_history"]) if prof else \
            {"fluid_rate": 5.0, "vasopressor_dose": 0.0, "antibiotic_code": 3}
        synth_ms = (time.perf_counter() - t0) * 1000.0
        timeline.append({"layer": BED, "event": "formal protocol synthesizer compiles plan",
                         "plan": plan, "synth_ms": round(synth_ms, 2)})

        # GOVERNANCE → the order must pass the non-maleficence invariant, and emits a
        # portable signed proof certificate any third party can independently re-check.
        cert = self.govern("bed", "treat", plan, target=bed)
        timeline.append({"layer": "governance", "event": "non-maleficence invariant gate",
                         "certified": cert.ok, "z3_status": cert.z3_status,
                         "cert_hash": cert.hash[:12], "violations": cert.violations})

        proof = self.proofs.get(cert.seq)
        proof_audit = None
        if proof is not None:
            # NOTE: we deliberately do NOT auto re-verify here. The independent
            # re-check re-solves every obligation with fresh Z3 solvers — cheap on a
            # real CPU but seconds on the throttled free tier, and it's redundant with
            # the cert we just produced. The frontend's "re-verify independently"
            # button drives it on demand via /aegis/proof/verify (the honest audit
            # path — a third party re-checking without trusting us).
            sig = proof.get("signature", {})
            timeline.append({
                "layer": "governance",
                "event": "portable proof certificate issued (independently re-checkable on demand)",
                "obligations": proof.get("verdict", {}).get("n_obligations"),
                "all_satisfied": proof.get("verdict", {}).get("all_satisfied"),
                "signed_by": sig.get("key_id"),
            })

        return {
            "bed": bed,
            "invariant": self.invariant.name,
            "timeline": timeline,
            "committed_plan": plan if cert.ok else None,
            "certificate": cert.to_dict(),
            "proof_certificate": proof,           # the (decision, signed_cert) pair
            "proof_audit": proof_audit,
            "attestation": self.attest(),
            "causal_events": self.graphs[bed].graph.stats()["nodes"] if bed in self.graphs else 0,
        }


# --------------------------------------------------------------------------- #
def _demo() -> None:
    from backend.aegis.kernel.scheduler import BioCluster
    cl = BioCluster(n_nodes=4, lane_budget=64, seed=1)
    orch = Orchestrator(cl, histories={"node_2": {"chf": False}})
    res = orch.run_scenario("node_2")
    print("invariant:", res["invariant"])
    for s in res["timeline"]:
        print(f"  [{s['layer']:10}] {s['event']}")
    c = res["certificate"]
    print(f"final cert: ok={c['ok']} z3={c['z3_status']} hash={c['hash']}")
    print("attestation:", res["attestation"]["ok"], "ledger head", res["attestation"]["head"])


if __name__ == "__main__":
    _demo()
