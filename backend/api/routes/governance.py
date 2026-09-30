"""Orchestrator, consensus brain, portable proofs, federation, RBAC and the audited syscall gate.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

import os as _os
from typing import Any, Dict, Optional

from fastapi import APIRouter

from backend.api.schemas import (
    CoreProofIn,
    GovernIn,
    ProofVerifyIn,
    RbacAuthIn,
    RbacLoginIn,
    SyscallIn,
)
from backend.aegis.server import (
    CAP_MIN_RING,
    GATE,
    GRANTS,
    RING,
    SYSCALL_CAP,
    USERS,
    _AuthError,
    _FED_CACHE,
    _PermissionDenied,
    _broadcast_cluster,
    _cluster_payload,
    _cred,
    _proof_ca,
    _verify_cert,
    cluster,
    export_bundle,
    orchestrator,
    prove_policy_sound,
    rbac,
    session,
    syscalls,
)

router = APIRouter()


@router.post("/aegis/rbac/login")
def rbac_login(p: RbacLoginIn) -> Dict[str, Any]:
    """Mint a signed, expiring session token for a role + return that role's shell
    capability contract (what the role's UI may render and do)."""
    # The role picks a DEMO ACCOUNT; the account carries the role. This route used
    # to mint a token for any role the request named, with no credential at all.
    try:
        token, state, cred = USERS.login_role(p.role, client="rbac-overlay")
    except _AuthError as e:
        return {"status": "error", "errno": "EAUTH", "detail": str(e)}
    s = rbac.verify_token(token)
    return {"session": s.to_dict() if s else {}, "token": token,
            "shell": rbac.shell_capabilities(cred.role), "account": cred.to_dict()}


@router.get("/aegis/rbac/policy")
def rbac_policy() -> Dict[str, Any]:
    """The full RBAC matrix + the Z3 proof that the policy is sound (no grant above
    its capability's ring)."""
    return {"rings": RING, "cap_min_ring": CAP_MIN_RING,
            "grants": {r: sorted(c) for r, c in GRANTS.items()},
            "soundness": prove_policy_sound()}


@router.post("/aegis/rbac/authorize")
def rbac_authorize(p: RbacAuthIn) -> Dict[str, Any]:
    """Check whether a session may perform a capability (audited, no mutation)."""
    ok, reason, s = rbac.authorize(p.token, p.capability)
    return {"ok": ok, "reason": reason, "capability": p.capability,
            "role": s.role if s else None, "audit_head": rbac.audit.head()[:12]}


@router.get("/aegis/rbac/audit")
def rbac_audit(limit: int = 24) -> Dict[str, Any]:
    return {"entries": [e.to_dict() for e in rbac.audit.entries[-limit:]],
            "count": len(rbac.audit.entries), "head": rbac.audit.head()[:12],
            "integrity_ok": rbac.audit.verify()}


@router.get("/aegis/fedlearn")
def fedlearn_result(limit: int = 400, refresh: bool = False) -> Dict[str, Any]:
    """Federated-learning experiment on the real CinC two-hospital split: does
    cross-site AUROC recover without sharing data, and does robust aggregation
    survive a poisoned hospital? Cached (first call computes ~5-15s)."""
    key = f"l{limit}"
    if refresh or key not in _FED_CACHE:
        from backend.aegis.fedlearn import run_experiment, run_poisoning
        _FED_CACHE[key] = {
            "experiment": run_experiment(limit=limit, n_nodes=4, rounds=10, epochs=50),
            "poisoning": run_poisoning(limit=limit, n_nodes=5, rounds=10, epochs=50),
            "caveats": [
                "Federated model is logistic regression (averageable weights); the "
                "production GBM is not FL-compatible.",
                "Only signed weight-deltas cross the wire — no patient records. "
                "Data-minimization, not differential privacy.",
                "Byzantine tolerance = signed updates + robust aggregation "
                "(Krum/median), not a proof that a gradient is benign.",
            ],
        }
    return _FED_CACHE[key]


@router.get("/aegis/orchestrator")
def orchestrator_status() -> Dict[str, Any]:
    """The orchestration hierarchy (sensor/bed/wing/global) under the one
    Non-Maleficence invariant, plus the attestation-ledger head + integrity."""
    return orchestrator.status()


@router.get("/aegis/orchestrator/attest")
def orchestrator_attest() -> Dict[str, Any]:
    """Continuous attestation: prove the invariant currently holds across the wing
    (ledger intact + no critical bed dropped below its priority floor)."""
    return orchestrator.attest()


@router.post("/aegis/orchestrator/govern")
def orchestrator_govern(payload: GovernIn) -> Dict[str, Any]:
    """Route a single action through the non-maleficence invariant. Returns its
    certificate (allowed → applied; blocked → nothing mutates). Every call is
    appended to the attestation ledger."""
    cert = orchestrator.govern(payload.layer, payload.action,
                               {k: v for k, v in payload.payload.items()},
                               target=payload.target)
    return cert.to_dict()


@router.post("/aegis/orchestrator/scenario")
def orchestrator_scenario(bed: Optional[str] = None) -> Dict[str, Any]:
    """The demo cascade: induce a sepsis crisis in a wing bed and walk the governed
    response sensor→bed→wing→global→governance, ending in a signed, Z3-certified,
    immutably-logged order. A deliberate drill on the live wing (clear via syscalls)."""
    beds = getattr(cluster, "node_ids", [])
    if not beds:
        return {"status": "error", "detail": "no wing beds available"}
    if bed is None:
        # prefer a known-septic bed, else the first.
        bed = next((b for b in beds if getattr(cluster, "_by", {}).get(b)
                    and getattr(cluster._by[b], "is_septic", False)), beds[0])
    if bed not in beds:
        return {"status": "error", "detail": f"no such bed '{bed}'"}
    # The live wing is already warm (the stream loop ticks it); keep the drill short.
    # Fewer ticks: enough to drive the crisis + trip PREEMPT, but light on a
    # throttled CPU so the synthesis returns well inside the gateway timeout.
    return orchestrator.run_scenario(bed, warmup=0, crisis_ticks=12)


@router.get("/aegis/orchestrator/proof")
def orchestrator_proof_latest() -> Dict[str, Any]:
    """The most recent portable proof certificate the orchestrator emitted for a
    treatment order, plus a fresh independent re-check of it."""
    pc = orchestrator.last_proof
    if pc is None:
        return {"status": "empty",
                "detail": "no treatment order has been governed yet (run a scenario)"}
    return {"status": "ok", "certificate": pc,
            "audit": _verify_cert(pc).to_dict()}


@router.get("/aegis/orchestrator/proof/{seq}")
def orchestrator_proof_by_seq(seq: int) -> Dict[str, Any]:
    """The portable proof certificate bound to attestation-ledger entry ``seq``."""
    pc = orchestrator.proof_for(seq)
    if pc is None:
        return {"status": "not_found", "detail": f"no proof bound to ledger seq {seq}"}
    return {"status": "ok", "certificate": pc, "audit": _verify_cert(pc).to_dict()}


@router.post("/aegis/proof/core")
def proof_core_certify(p: CoreProofIn) -> Dict[str, Any]:
    """Export a portable, signed proof certificate from the CORE Z3 cage — the exact
    prover the live runtime uses on every HIGH-risk decision — one re-checkable
    SMT-LIB2 obligation per clinical invariant, then independently re-audit it."""
    from backend.aegis.proofexport import certify_core_protocol
    cert = certify_core_protocol(p.clinical_history, p.vitals, p.risk_level,
                                 p.protocol, ca=_proof_ca())
    cd = cert.to_dict()
    return {"status": "ok", "certificate": cd, "audit": _verify_cert(cd).to_dict()}


@router.get("/aegis/proof/core/latest")
def proof_core_latest() -> Dict[str, Any]:
    """The portable proof certificate for the live patient's most recent HIGH-risk
    protocol decision, plus a fresh independent re-check. Emitted by AegisCore."""
    pc = getattr(session.core, "last_proof", None)
    if pc is None:
        return {"status": "empty",
                "detail": "no HIGH-risk protocol has been certified for this patient yet"}
    return {"status": "ok", "certificate": pc, "audit": _verify_cert(pc).to_dict()}


@router.get("/aegis/proof/core/latest/bundle")
def proof_core_latest_bundle():
    """Download a self-contained .zip audit bundle for the live patient's most recent
    certified order — verifiable by a third party with only z3 + cryptography."""
    import io
    import shutil
    import tempfile
    import zipfile
    from fastapi import Response
    from backend.aegis.proofexport import ProofCertificate

    pc = getattr(session.core, "last_proof", None)
    if pc is None:
        return {"status": "empty", "detail": "no certified order yet"}
    tmp = tempfile.mkdtemp(prefix="aegis-core-bundle-")
    try:
        export_bundle(ProofCertificate.from_dict(pc), tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in _os.walk(tmp):
                for fn in files:
                    full = _os.path.join(root, fn)
                    zf.write(full, _os.path.relpath(full, tmp))
        buf.seek(0)
        return Response(
            content=buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition":
                     'attachment; filename="aegis-core-proof-bundle.zip"'})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@router.post("/aegis/proof/verify")
def proof_verify(p: ProofVerifyIn) -> Dict[str, Any]:
    """Independently audit ANY proof certificate the caller supplies — re-runs every
    SMT-LIB2 obligation in a fresh solver, recomputes the content digest, and checks
    the Ed25519 signature. This is the third-party audit path: a signed-but-false
    certificate fails here because the math is re-run, not trusted."""
    return _verify_cert(p.certificate, require_signature=p.require_signature).to_dict()


@router.get("/aegis/orchestrator/proof/{seq}/bundle")
def orchestrator_proof_bundle(seq: int):
    """Download a self-contained audit bundle (.zip): certificate.json, one .smt2 per
    obligation, the public key, AUDIT.md, and a standalone verifier that needs only
    z3 + cryptography. A third party verifies the order without any AegisOS code."""
    import io
    import shutil
    import tempfile
    import zipfile
    from fastapi import Response
    from backend.aegis.proofexport import ProofCertificate

    pc = orchestrator.proof_for(seq)
    if pc is None:
        return {"status": "not_found", "detail": f"no proof bound to ledger seq {seq}"}
    tmp = tempfile.mkdtemp(prefix=f"aegis-bundle-{seq}-")
    try:
        export_bundle(ProofCertificate.from_dict(pc), tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in _os.walk(tmp):
                for fn in files:
                    full = _os.path.join(root, fn)
                    zf.write(full, _os.path.relpath(full, tmp))
        buf.seek(0)
        return Response(
            content=buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition":
                     f'attachment; filename="aegis-proof-bundle-seq{seq}.zip"'})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@router.get("/aegis/consensus")
def consensus_round(deterioration: Optional[float] = None,
                    faulty: Optional[str] = None,
                    remote: Optional[str] = None,
                    bed: Optional[str] = None) -> Dict[str, Any]:
    """Run one Byzantine-fault-tolerant specialist consensus round over a patient's
    state (live session, or a cohort bed via ``bed``). ``faulty`` marks a replica id
    as Byzantine; ``remote`` marks a remote consultant; leave ``deterioration``
    unset to derive it from the patient's risk band."""
    from backend.aegis.consensus import ClinicalBFT, default_panel
    if bed and bed != session.patient_id and getattr(cluster, "_by", {}).get(bed):
        prof = getattr(cluster, "profiles", {}).get(bed)
        raw = cluster._sample_vitals(bed) if hasattr(cluster, "_sample_vitals") else {}
        fused = float(getattr(prof, "fused_risk", 0.0)) if prof else 0.0
        vit = {"map": float(raw.get("map", 80.0))}
        risk = "HIGH" if fused >= 0.6 else "MEDIUM" if fused >= 0.35 else "LOW"
        hist = getattr(cluster, "histories", {}).get(bed, {})
    else:
        vit = session.core.twin.latest_vitals or {}
        risk = session.latest.risk_level if session.latest else "MEDIUM"
        hist = session.core.clinical_history or {}
    if deterioration is None:
        deterioration = {"HIGH": 0.7, "MEDIUM": 0.35, "LOW": 0.1}.get(risk, 0.3)
        if float(vit.get("map", 90)) < 60:        # very low MAP ⇒ crashing faster
            deterioration = max(deterioration, 0.85)
    state = {"map": float(vit.get("map", 75.0)), "risk_level": risk,
             "clinical_history": hist,
             "infection_suspected": risk in ("HIGH", "MEDIUM")}
    panel = default_panel(remote_id=remote)
    if faulty:
        for a in panel:
            if a.agent_id == faulty:
                a.byzantine = True
    bft = ClinicalBFT(panel)
    out = bft.run_round(state, deterioration=float(deterioration))
    res = out.to_dict()
    res["state"] = {"map": state["map"], "risk_level": risk,
                    "clinical_history": state["clinical_history"]}
    res["panel"] = [{"id": a.agent_id, "specialty": a.specialty,
                     "byzantine": a.byzantine, "remote": a.remote} for a in panel]
    return res


@router.post("/aegis/syscall")
async def aegis_syscall(payload: SyscallIn) -> Dict[str, Any]:
    """The write-side trap gate. The console invokes treat/escalate/crisis/clear
    here; the mutation is applied to the running kernel ONLY if the Z3 egress cage
    certifies it. Every call — allowed or denied — is appended to the audit log.
    A denied write returns EPERM and changes nothing; the certificate explains why."""
    cred = _cred()
    name = (payload.syscall or "").lower().strip()
    cap = SYSCALL_CAP.get(name)
    if cap is not None:
        # The route let anyone holding order:escalate in; THIS trap needs its own
        # capability (treat needs order:treat — the chief holds escalate, not treat).
        try:
            GATE.require(cap, cred, what=f"syscall {name}")
        except _PermissionDenied as exc:
            return {"sid": None, "syscall": name, "bed": payload.bed, "allowed": False,
                    "errno": 1, "errno_name": "EPERM", "z3_status": "n/a",
                    "message": f"EPERM — {exc}", "capability": cap, "uid": cred.username}
    # The caller is WHO THE SESSION IS. The body's free-text `caller` is ignored:
    # an audit record whose author is whatever the request typed is not a record.
    res = syscalls.invoke(payload.syscall, payload.bed, payload.args, caller=cred.username)
    # push the post-write state to all cockpits immediately so the grid reacts.
    if res.allowed:
        await _broadcast_cluster(_cluster_payload())
    out = res.to_dict()
    if res.allowed and res.syscall == "treat":
        # A certified order now has somewhere to go: the treatment twin can show
        # where this drug travels and what it risks. The trap gate deliberately
        # does NOT compute it here — a PBPK solve plus a Monte Carlo does not
        # belong on the write path, and a projection is a read. The console opens
        # the twin against this href.
        out["projection"] = {"available": True,
                             "href": f"/aegis/pk/bed/{payload.bed}",
                             "label": "treatment projection"}
    return out


@router.get("/aegis/syscall/audit")
def aegis_syscall_audit(limit: int = 64) -> Dict[str, Any]:
    """The append-only syscall audit log — the system of record for every
    operator write and the prover's verdict on it."""
    return {"status": syscalls.status(), "log": syscalls.audit_log(limit)}

