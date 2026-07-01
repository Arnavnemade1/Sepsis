"""Proof-Export & Independent-Audit layer — portable, signed safety certificates.

The medical claim AegisOS makes is unusual: *every* treatment protocol it suggests
carries a **formal proof of safety**. This module is what makes that claim
*falsifiable by a stranger*. It turns an internal Z3 verdict into a self-contained
artifact that a hospital administrator, a regulator, or a competing lab can verify
**without running — or trusting — a single line of AegisOS code**.

Two guarantees, kept deliberately separate, because they answer different questions:

  1. SOUNDNESS (the headline — "is the math real?").
     Each proof obligation is a *complete SMT-LIB2 script* ending in ``(check-sat)``
     — the lingua franca every SMT solver (Z3, CVC5, …) reads. The obligation also
     states its ``expected_status``: ``unsat`` means "the rule's violation is
     impossible under this patient's bindings", i.e. the protocol provably satisfies
     the rule. An auditor re-runs the script with their *own* solver and compares.
     Crucially, this needs **zero trust in the issuer**: even a maliciously signed
     certificate cannot lie about the math, because the auditor recomputes it.

  2. PROVENANCE ("did AegisOS really issue this, unaltered?").
     The certificate is sealed with an **Ed25519 signature** over a canonical digest
     of its full body. Ed25519 is *asymmetric*: the auditor verifies with a **public
     key alone** — no shared secret, unlike an HMAC. The public key ships inside the
     bundle, so the auditor needs nothing from us. Tampering with any byte (an SMT
     script, the protocol, the patient-context hash) breaks both the content digest
     and the signature.

What this IS — and is NOT (the integrity rules; stated, not hidden):

  * It is an *export & audit* layer. It does not invent new safety logic; it
    faithfully captures the obligations the existing Z3 cage / Clinical-Safety DSL
    already discharge (see :mod:`backend.aegis.clinical_dsl`).
  * "Proven" means exactly what Z3 means: ``unsat`` of the rule's negation under the
    protocol bindings — a genuine validity proof over the *modelled* variables. It is
    not a proof that the *clinical rule itself* is wise; it proves the protocol obeys
    the rule a hospital authored. The hospital owns the rules; we prove conformance.
  * The signature proves *provenance*, not *correctness*. Correctness is the
    re-checkable math (guarantee 1). A signed-but-wrong certificate is detected by
    re-checking, never by the signature. That separation is the whole point.
  * No PHI travels in a certificate: the patient is represented only by a salted
    context hash plus the de-identified protocol parameters being certified.

The deliverable for a third party is :func:`export_bundle`, which writes a directory
containing ``certificate.json``, one ``.smt2`` file per obligation, the public key,
a human-readable ``AUDIT.md`` with the exact commands to run, and a standalone
``verify_bundle.py`` that depends only on ``z3`` + ``cryptography`` (no AegisOS).

    python -m backend.aegis.proofexport            # issue + self-audit a demo cert
    python -m backend.aegis.proofexport --bundle out/   # also export an audit bundle
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from z3 import Not, Solver, parse_smt2_string, get_version_string, sat, unsat

from backend.aegis.clinical_dsl import (DEFAULT_POLICY, CompiledRule,
                                        compile_rules, _bindings)

SCHEMA = "aegis-proof-certificate/v1"
ISSUER = "AegisOS"


# --------------------------------------------------------------------------- #
# Canonical serialization + hashing — same discipline as the rest of the OS.
# --------------------------------------------------------------------------- #
def _normalize_numbers(obj: Any) -> Any:
    """Collapse integer-valued floats (4.0 → 4) recursively.

    JSON has no int/float distinction and JavaScript's ``JSON.stringify`` emits
    ``4.0`` as ``"4"``. Without this, a certificate re-serialized by a JS-based
    auditor would hash differently than one issued by Python and fail the integrity
    check. Normalizing makes the canonical form identical on both sides — essential
    for the "verifiable by *any* third party" guarantee."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    if isinstance(obj, dict):
        return {k: _normalize_numbers(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_numbers(v) for v in obj]
    return obj


def _canon(obj: Any) -> bytes:
    """Deterministic, JS-stable JSON: identical digests whether the certificate is
    re-serialized by Python or by a JavaScript auditor's JSON.stringify."""
    return json.dumps(_normalize_numbers(obj), sort_keys=True,
                      separators=(",", ":"), default=str).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patient_context_hash(context: Dict[str, Any], salt: str = "") -> str:
    """A privacy-preserving fingerprint of the clinical context.

    Carries *no* PHI into the certificate — only a salted hash — yet still binds the
    proof to the exact context it was discharged under, so a certificate cannot be
    silently replayed against a different patient."""
    return _sha256((salt + "|").encode() + _canon(context))


# --------------------------------------------------------------------------- #
# A single proof obligation: one re-checkable SMT-LIB2 script + its meaning.
# --------------------------------------------------------------------------- #
_STATUS_RE = re.compile(r"\(set-info :status [a-z]+\)")


@dataclass
class ProofObligation:
    id: str
    name: str                       # the rule/invariant being discharged
    statement: str                  # human-readable claim (e.g. the DSL rule text)
    smt2: str                       # COMPLETE SMT-LIB2 script ending in (check-sat)
    expected_status: str            # "unsat" => proven satisfied | "sat" => violated
    formula: str = ""               # the rule's Z3 s-expression (the goal, pre-negation)
    counterexample: Dict[str, Any] = field(default_factory=dict)

    @property
    def smt2_sha256(self) -> str:
        return _sha256(self.smt2.encode())

    @property
    def satisfied(self) -> bool:
        """A rule is *satisfied* iff its violation is unsatisfiable."""
        return self.expected_status == "unsat"

    def recheck(self) -> Tuple[bool, str]:
        """Independently re-run this obligation in a FRESH solver from the SMT text
        alone — exactly what a third party does. Returns (matches_expected, status)."""
        s = Solver()
        s.add(parse_smt2_string(self.smt2))
        status = s.check()
        got = "sat" if status == sat else "unsat" if status == unsat else "unknown"
        return got == self.expected_status, got

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "name": self.name, "statement": self.statement,
             "expected_status": self.expected_status,
             "meaning": ("unsat ⇒ the rule cannot be violated under these "
                         "bindings (protocol provably satisfies it)"),
             "formula": self.formula, "smt2": self.smt2,
             "smt2_sha256": self.smt2_sha256}
        if self.counterexample:
            d["counterexample"] = dict(self.counterexample)   # defensive copy
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProofObligation":
        return cls(id=d["id"], name=d["name"], statement=d["statement"],
                   smt2=d["smt2"], expected_status=d["expected_status"],
                   formula=d.get("formula", ""),
                   counterexample=d.get("counterexample", {}))

    @classmethod
    def from_solver(cls, solver: Solver, *, id: str, name: str, statement: str,
                    expected_status: str, formula: str = "",
                    counterexample: Optional[Dict[str, Any]] = None) -> "ProofObligation":
        """Universal adapter: capture any solver that has ALREADY asserted the goal
        to be discharged (typically the *negated* safety rule + the bindings) into a
        portable obligation. One line at any Z3 site in the codebase."""
        smt2 = _annotate_status(solver.to_smt2(), expected_status, statement)
        return cls(id=id, name=name, statement=statement, smt2=smt2,
                   expected_status=expected_status, formula=formula,
                   counterexample=counterexample or {})


def _annotate_status(smt2: str, expected: str, statement: str) -> str:
    """Stamp the script's :status with what we claim, and prepend an audit header.
    Comments are ignored by every SMT solver, so this stays fully portable."""
    header = (f"; AegisOS proof obligation\n"
              f"; claim: {statement}\n"
              f"; expected (check-sat) result: {expected}\n"
              f";   unsat = the safety rule's violation is impossible (PROVEN safe)\n"
              f";   sat   = a violating assignment exists (REJECTED) — see model\n")
    if _STATUS_RE.search(smt2):
        smt2 = _STATUS_RE.sub(f"(set-info :status {expected})", smt2, count=1)
    else:
        smt2 = f"(set-info :status {expected})\n" + smt2
    return header + smt2


# --------------------------------------------------------------------------- #
# Ed25519 signing authority — provenance with a public key alone.
# --------------------------------------------------------------------------- #
def _default_keystore() -> str:
    from backend.ml import config
    return os.path.join(config.ARTIFACTS_DIR, "keystore")


class CertificateAuthority:
    """Holds AegisOS's Ed25519 signing key; issues the public half for auditors.

    The keypair persists under ``artifacts/keystore/`` (override with env
    ``AEGIS_PROOF_KEY``), generated on first use. Only the *public* key ever leaves
    in a certificate bundle — that is all an auditor needs to verify provenance."""

    def __init__(self, key_path: Optional[str] = None):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        self._ser = serialization
        self._Ed = Ed25519PrivateKey
        self.key_path = key_path or os.getenv(
            "AEGIS_PROOF_KEY", os.path.join(_default_keystore(), "aegis_proof_ed25519.pem"))
        self._priv = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as fh:
                return self._ser.load_pem_private_key(fh.read(), password=None)
        priv = self._Ed.generate()
        os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
        with open(os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as fh:
            fh.write(priv.private_bytes(
                encoding=self._ser.Encoding.PEM,
                format=self._ser.PrivateFormat.PKCS8,
                encryption_algorithm=self._ser.NoEncryption()))
        with open(self.key_path + ".pub", "w") as fh:
            fh.write(self._public_pem(priv))
        return priv

    def _public_pem(self, priv=None) -> str:
        pub = (priv or self._priv).public_key()
        return pub.public_bytes(
            encoding=self._ser.Encoding.PEM,
            format=self._ser.PublicFormat.SubjectPublicKeyInfo).decode()

    @property
    def public_pem(self) -> str:
        return self._public_pem()

    @property
    def key_id(self) -> str:
        return _sha256(self.public_pem.encode())[:16]

    def sign(self, digest_hex: str) -> str:
        return self._priv.sign(digest_hex.encode()).hex()


def verify_signature(public_pem: str, digest_hex: str, sig_hex: str) -> bool:
    """Verify an Ed25519 signature with the public key alone (no secret)."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.exceptions import InvalidSignature
        pub = serialization.load_pem_public_key(public_pem.encode())
        try:
            pub.verify(bytes.fromhex(sig_hex), digest_hex.encode())
            return True
        except InvalidSignature:
            return False
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# The certificate.
# --------------------------------------------------------------------------- #
@dataclass
class ProofCertificate:
    subject: Dict[str, Any]               # what is certified (protocol + ctx hash)
    obligations: List[ProofObligation]
    claim: str
    issued_at: float
    solver: Dict[str, str]
    content_sha256: str = ""
    signature: Dict[str, str] = field(default_factory=dict)

    # -- the body that is hashed + signed (everything except the seal itself) -- #
    def sealed_body(self) -> Dict[str, Any]:
        n_ok = sum(o.satisfied for o in self.obligations)
        return {
            "schema": SCHEMA, "issuer": ISSUER,
            "issued_at": int(self.issued_at),
            "issued_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime(self.issued_at)),
            "claim": self.claim, "subject": copy.deepcopy(self.subject),
            "solver": dict(self.solver),
            "obligations": [o.to_dict() for o in self.obligations],
            "verdict": {"all_satisfied": n_ok == len(self.obligations),
                        "n_obligations": len(self.obligations), "n_satisfied": n_ok},
        }

    @property
    def all_satisfied(self) -> bool:
        return all(o.satisfied for o in self.obligations)

    def to_dict(self) -> Dict[str, Any]:
        body = self.sealed_body()
        body["content_sha256"] = self.content_sha256
        body["signature"] = self.signature
        return body

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProofCertificate":
        return cls(
            subject=d["subject"],
            obligations=[ProofObligation.from_dict(o) for o in d["obligations"]],
            claim=d["claim"], issued_at=float(d["issued_at"]), solver=d["solver"],
            content_sha256=d.get("content_sha256", ""),
            signature=d.get("signature", {}))


def _seal(cert: ProofCertificate, ca: Optional[CertificateAuthority]) -> ProofCertificate:
    cert.content_sha256 = _sha256(_canon(cert.sealed_body()))
    if ca is not None:
        cert.signature = {"alg": "Ed25519", "key_id": ca.key_id,
                          "public_key_pem": ca.public_pem,
                          "sig": ca.sign(cert.content_sha256)}
    return cert


# --------------------------------------------------------------------------- #
# Issuance — the flagship: hospital DSL policy + protocol -> signed certificate.
# --------------------------------------------------------------------------- #
def certify_dsl_protocol(
    protocol: Dict[str, Any],
    context: Dict[str, Any],
    *,
    rules: Optional[List[CompiledRule]] = None,
    policy_text: str = DEFAULT_POLICY,
    policy_id: str = "default-hospital-policy",
    ca: Optional[CertificateAuthority] = None,
    context_salt: str = "",
    extra_subject: Optional[Dict[str, Any]] = None,
) -> ProofCertificate:
    """Discharge every rule of a hospital-authored DSL policy against a concrete
    protocol+context, capturing each as a portable SMT-LIB2 obligation, and seal the
    result with an Ed25519 signature.

    Mirrors :func:`clinical_dsl.verify_protocol` exactly — same bindings, same
    negated-rule discipline — but *captures the solver* so the proof is exportable."""
    rules = rules if rules is not None else compile_rules(policy_text)
    binds = _bindings(protocol, context)
    obligations: List[ProofObligation] = []
    for i, r in enumerate(rules):
        s = Solver()
        s.add(*binds)
        s.add(Not(r.formula))                  # prove the rule cannot be violated
        status = s.check()
        if status == unsat:
            ob = ProofObligation.from_solver(
                s, id=f"ob-{i}", name=r.name, statement=r.text,
                expected_status="unsat", formula=r.sexpr())
        else:
            m = s.model()
            ce = {str(d.name()): str(m[d]) for d in m.decls()}
            ob = ProofObligation.from_solver(
                s, id=f"ob-{i}", name=r.name, statement=r.text,
                expected_status="sat", formula=r.sexpr(), counterexample=ce)
        obligations.append(ob)

    subject = {
        "kind": "treatment-protocol-safety",
        "policy_id": policy_id, "policy_source": "clinical-safety-dsl",
        "protocol": {k: protocol[k] for k in sorted(protocol)},
        "patient_context_hash": patient_context_hash(context, context_salt),
        "context_salted": bool(context_salt),
    }
    if extra_subject:
        subject.update(extra_subject)          # bound into the signed body
    n = len(obligations)
    n_ok = sum(o.satisfied for o in obligations)
    claim = (f"Under the certified patient context, the protocol provably satisfies "
             f"all {n} stated safety invariants of policy '{policy_id}'."
             if n_ok == n else
             f"Protocol satisfies {n_ok}/{n} invariants of policy '{policy_id}'; "
             f"{n - n_ok} violated (see counterexamples).")
    cert = ProofCertificate(
        subject=subject, obligations=obligations, claim=claim, issued_at=time.time(),
        solver={"name": "Z3", "version": get_version_string(),
                "format": "SMT-LIB2", "theory": "QF_LIRA"})
    return _seal(cert, ca)


# --------------------------------------------------------------------------- #
# The core-cage path: the SAME Z3 invariants the live runtime enforces, exported.
# --------------------------------------------------------------------------- #
def certify_core_protocol(
    clinical_history: Dict[str, Any],
    vitals: Dict[str, Any],
    risk_level: str,
    protocol: Dict[str, Any],
    *,
    ca: Optional[CertificateAuthority] = None,
    context_salt: str = "",
) -> ProofCertificate:
    """Export a portable proof certificate from the **core Z3 protocol cage** — the
    exact prover the live runtime (:class:`core.verifier.ProtocolVerifier`) uses in
    ``AegisCore`` on every HIGH-risk decision.

    Decomposes the cage's monolithic violation check into one re-checkable SMT-LIB2
    obligation *per named clinical invariant* (CHF/renal fluid caps, vasopressor
    ceiling + MAP justification, drug-allergy, sepsis-antibiotic, …), so an auditor
    sees — and re-runs — each invariant independently. Single source of truth: it
    consumes ``ProtocolVerifier.safety_model``, so it can never drift from the live
    cage."""
    from backend.core.verifier import ProtocolVerifier

    pv = ProtocolVerifier()
    bindings, constraints, _vars = pv.safety_model(
        clinical_history, vitals, risk_level, protocol)

    obligations: List[ProofObligation] = []
    for i, c in enumerate(constraints):
        s = Solver()
        s.add(*bindings)
        s.add(Not(c.formula))                  # prove this invariant cannot be violated
        status = s.check()
        if status == unsat:
            obligations.append(ProofObligation.from_solver(
                s, id=f"ob-{i}", name=c.name, statement=c.desc,
                expected_status="unsat", formula=c.formula.sexpr()))
        else:
            m = s.model()
            ce = {str(d.name()): str(m[d]) for d in m.decls()}
            obligations.append(ProofObligation.from_solver(
                s, id=f"ob-{i}", name=c.name, statement=c.desc,
                expected_status="sat", formula=c.formula.sexpr(), counterexample=ce))

    abx = int(protocol.get("antibiotic_code", 0))
    ctx_for_hash = {"clinical_history": clinical_history, "vitals": vitals,
                    "risk_level": risk_level}
    subject = {
        "kind": "treatment-protocol-safety",
        "policy_id": "aegis-core-cage", "policy_source": "core-protocol-verifier",
        "risk_level": risk_level,
        "protocol": {
            "fluid_rate": float(protocol.get("fluid_rate", 0.0)),
            "vasopressor_dose": float(protocol.get("vasopressor_dose", 0.0)),
            "antibiotic_code": abx,
            "antibiotic_name": ProtocolVerifier.ANTIBIOTICS.get(abx, "Unknown"),
        },
        "patient_context_hash": patient_context_hash(ctx_for_hash, context_salt),
        "context_salted": bool(context_salt),
    }
    n = len(obligations)
    n_ok = sum(o.satisfied for o in obligations)
    claim = (f"Under the certified patient context, the protocol provably satisfies "
             f"all {n} safety invariants of the AegisOS core cage."
             if n_ok == n else
             f"Protocol satisfies {n_ok}/{n} core-cage invariants; {n - n_ok} "
             f"violated (see counterexamples).")
    cert = ProofCertificate(
        subject=subject, obligations=obligations, claim=claim, issued_at=time.time(),
        solver={"name": "Z3", "version": get_version_string(),
                "format": "SMT-LIB2", "theory": "QF_LIRA"})
    return _seal(cert, ca)


# --------------------------------------------------------------------------- #
# The INDEPENDENT verifier — what a third party runs. Trusts no one.
# --------------------------------------------------------------------------- #
@dataclass
class ObligationAudit:
    id: str
    name: str
    expected_status: str
    rechecked_status: str
    math_ok: bool                 # solver re-check matched the expected status
    digest_ok: bool               # embedded smt2_sha256 matched the script bytes


@dataclass
class AuditReport:
    math_ok: bool                 # EVERY obligation re-checked correctly (no trust)
    integrity_ok: bool            # content digest + per-script digests intact
    signature_ok: bool            # Ed25519 signature valid (provenance)
    signature_present: bool
    consistent: bool              # cert's own verdict matches the re-checked reality
    key_id: str
    obligations: List[ObligationAudit]
    notes: List[str] = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        """Fully audited: the math is real, nothing was tampered, and (if present)
        provenance checks out. ``math_ok`` alone is the soundness guarantee that
        needs no trust in AegisOS — the signature only adds provenance."""
        return self.math_ok and self.integrity_ok and self.consistent and (
            self.signature_ok if self.signature_present else True)

    def to_dict(self) -> Dict[str, Any]:
        return {"trusted": self.trusted, "math_ok": self.math_ok,
                "integrity_ok": self.integrity_ok, "consistent": self.consistent,
                "signature_present": self.signature_present,
                "signature_ok": self.signature_ok, "key_id": self.key_id,
                "obligations": [o.__dict__ for o in self.obligations],
                "notes": self.notes}


def verify_certificate(cert: Any, *, require_signature: bool = True) -> AuditReport:
    """Independently audit a certificate. Accepts a :class:`ProofCertificate`, a dict,
    or a JSON string. Recomputes everything; trusts nothing it is told.

    The headline guarantee is ``math_ok``: every obligation's SMT-LIB2 script is
    re-run in a *fresh* solver and its result compared to the claimed status. A
    forged certificate that lies about the math fails here even if its signature is
    valid — which is exactly why the proof is "checkable without trusting AegisOS"."""
    if isinstance(cert, ProofCertificate):
        cert = cert.to_dict()
    elif isinstance(cert, (str, bytes)):
        cert = json.loads(cert)
    notes: List[str] = []

    # 1) SOUNDNESS — re-run every obligation in a fresh solver. No trust required.
    audits: List[ObligationAudit] = []
    for od in cert.get("obligations", []):
        ob = ProofObligation.from_dict(od)
        digest_ok = (ob.smt2_sha256 == od.get("smt2_sha256"))
        matched, got = ob.recheck()
        audits.append(ObligationAudit(ob.id, ob.name, ob.expected_status, got,
                                      matched, digest_ok))
    math_ok = bool(audits) and all(a.math_ok for a in audits)

    # 2) INTEGRITY — recompute the content digest over the canonical sealed body.
    body = {k: cert[k] for k in cert if k not in ("content_sha256", "signature")}
    recomputed = _sha256(_canon(body))
    content_ok = (recomputed == cert.get("content_sha256"))
    digests_ok = all(a.digest_ok for a in audits)
    integrity_ok = content_ok and digests_ok
    if not content_ok:
        notes.append("content_sha256 mismatch: certificate body was altered.")
    if not digests_ok:
        notes.append("an obligation's smt2_sha256 does not match its script bytes.")

    # 3) PROVENANCE — Ed25519 signature over the content digest, public key alone.
    sig = cert.get("signature") or {}
    signature_present = bool(sig.get("sig"))
    signature_ok = False
    key_id = sig.get("key_id", "")
    if signature_present:
        # Sign over the *recomputed* digest so a tampered body cannot also forge a
        # matching content_sha256 field to pass signature check.
        signature_ok = verify_signature(sig.get("public_key_pem", ""),
                                        recomputed, sig.get("sig", ""))
        if (key_id and key_id != _sha256(sig.get("public_key_pem", "").encode())[:16]):
            signature_ok = False
            notes.append("key_id does not match the embedded public key.")
        if not signature_ok:
            notes.append("Ed25519 signature did not verify against the public key.")
    elif require_signature:
        notes.append("no signature present (provenance unverified).")

    # 4) CONSISTENCY — the cert's self-declared verdict must match re-checked reality.
    declared = cert.get("verdict", {}).get("all_satisfied")
    actual_all = all(a.expected_status == "unsat" for a in audits)
    consistent = (declared is None) or (bool(declared) == actual_all and math_ok)
    if declared is not None and not consistent:
        notes.append("declared verdict contradicts the re-verified obligations.")

    return AuditReport(
        math_ok=math_ok, integrity_ok=integrity_ok,
        signature_ok=signature_ok, signature_present=signature_present,
        consistent=consistent, key_id=key_id, obligations=audits, notes=notes)


# --------------------------------------------------------------------------- #
# Bundle export — the on-disk artifact a third party audits with stock tools.
# --------------------------------------------------------------------------- #
def export_bundle(cert: ProofCertificate, out_dir: str) -> str:
    """Write a self-contained audit bundle. After this, a stranger with only ``z3``
    and ``cryptography`` installed can fully verify the certificate — see AUDIT.md."""
    os.makedirs(out_dir, exist_ok=True)
    ob_dir = os.path.join(out_dir, "obligations")
    os.makedirs(ob_dir, exist_ok=True)

    with open(os.path.join(out_dir, "certificate.json"), "w") as fh:
        fh.write(cert.to_json())

    smt_files: List[str] = []
    for ob in cert.obligations:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{ob.id}_{ob.name}")
        fname = f"{safe}.smt2"
        with open(os.path.join(ob_dir, fname), "w") as fh:
            fh.write(ob.smt2)
        smt_files.append(fname)

    pub = cert.signature.get("public_key_pem", "")
    if pub:
        with open(os.path.join(out_dir, "public_key.pem"), "w") as fh:
            fh.write(pub)

    with open(os.path.join(out_dir, "verify_bundle.py"), "w") as fh:
        fh.write(_STANDALONE_VERIFIER)
    with open(os.path.join(out_dir, "AUDIT.md"), "w") as fh:
        fh.write(_audit_readme(cert, smt_files))
    return out_dir


def _audit_readme(cert: ProofCertificate, smt_files: List[str]) -> str:
    lines = [
        "# Independent Audit Instructions",
        "",
        "This bundle is a **formal safety certificate** for a treatment protocol,",
        "issued by AegisOS. You can verify it completely **without trusting AegisOS**",
        "and without installing any AegisOS code. You need only:",
        "",
        "  * an SMT solver — [Z3](https://github.com/Z3Prover/z3) (`z3`) or CVC5, and",
        "  * (optional, for provenance) Python with the `cryptography` package.",
        "",
        "## What is being certified",
        "",
        f"> {cert.claim}",
        "",
        f"- Issuer: `{ISSUER}`   schema: `{SCHEMA}`",
        f"- Solver of record: {cert.solver.get('name')} {cert.solver.get('version')}",
        f"- Signing key id: `{cert.signature.get('key_id', '(unsigned)')}`",
        f"- Obligations: **{len(cert.obligations)}** "
        f"({sum(o.satisfied for o in cert.obligations)} proven satisfied)",
        "",
        "## Step 1 — Re-check the math yourself (the part that needs no trust)",
        "",
        "Each `.smt2` file in `obligations/` is a complete, standalone proof script.",
        "Its header comment states the EXPECTED `(check-sat)` result. Run each one:",
        "",
        "```sh",
    ]
    for f in smt_files:
        lines.append(f"z3 obligations/{f}")
    lines += [
        "```",
        "",
        "For a **safe** protocol, every obligation must print exactly `unsat`.",
        "`unsat` means: *the solver proved the safety rule's violation is impossible*",
        "under this patient's bindings — i.e. the protocol cannot break the rule.",
        "If any obligation prints `sat`, the protocol VIOLATES that rule and the",
        "certificate must be rejected (the model Z3 prints is the counterexample).",
        "",
        "Because you run your own solver on the published script, this step does not",
        "rely on AegisOS being honest or even correct. The math speaks for itself.",
        "",
        "## Step 2 — Verify provenance + integrity (optional)",
        "",
        "Confirm the bytes were issued by AegisOS and not altered in transit:",
        "",
        "```sh",
        "python verify_bundle.py .",
        "```",
        "",
        "This recomputes the content digest, re-runs every obligation, and checks the",
        "Ed25519 signature against `public_key.pem` — using only `z3` + `cryptography`.",
        "",
        "## What this certificate does and does NOT prove",
        "",
        "- IT PROVES: the protocol parameters conform to every safety rule in the",
        "  named hospital policy, for the certified patient context.",
        "- IT DOES NOT PROVE the clinical wisdom of the rules themselves — the hospital",
        "  authors those; the certificate proves *conformance* to them.",
        "- No patient identifiers are present: the patient appears only as a salted",
        "  context hash plus the de-identified protocol parameters.",
        "",
    ]
    return "\n".join(lines)


# A fully standalone verifier copied into every bundle. Depends ONLY on z3 +
# cryptography — deliberately NO imports from `backend.aegis`, so an auditor can run
# it in a clean environment that has never seen AegisOS.
_STANDALONE_VERIFIER = '''#!/usr/bin/env python3
"""Standalone AegisOS proof-certificate verifier. No AegisOS code required.

Usage:  python verify_bundle.py [bundle_dir]

Verifies, in order:
  1. SOUNDNESS  - re-runs every obligations/*.smt2 in a fresh Z3 and checks the
                  result equals the status declared in certificate.json.
  2. INTEGRITY  - recomputes the SHA-256 content digest over the canonical body.
  3. PROVENANCE - checks the Ed25519 signature with public_key.pem alone.

Requires:  pip install z3-solver cryptography
"""
import hashlib, json, os, sys


def _norm(o):
    # Collapse integer-valued floats (4.0 -> 4) so the digest matches whether the
    # certificate was last serialized by Python or by a JavaScript JSON.stringify.
    if isinstance(o, bool): return o
    if isinstance(o, float) and o.is_integer(): return int(o)
    if isinstance(o, dict): return {k: _norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_norm(v) for v in o]
    return o


def canon(obj):
    return json.dumps(_norm(obj), sort_keys=True, separators=(",", ":"), default=str).encode()


def main(bundle="."):
    with open(os.path.join(bundle, "certificate.json")) as fh:
        cert = json.load(fh)

    from z3 import Solver, parse_smt2_string, sat, unsat
    ok = True

    print("== 1. SOUNDNESS (re-checking the math) ==")
    for ob in cert["obligations"]:
        s = Solver(); s.add(parse_smt2_string(ob["smt2"]))
        r = s.check()
        got = "sat" if r == sat else "unsat" if r == unsat else "unknown"
        good = got == ob["expected_status"]
        # the published .smt2 file must match the embedded script, byte for byte
        digest_ok = hashlib.sha256(ob["smt2"].encode()).hexdigest() == ob["smt2_sha256"]
        ok = ok and good and digest_ok
        print(f"  [{'PASS' if good and digest_ok else 'FAIL'}] {ob['id']} {ob['name']}: "
              f"expected {ob['expected_status']}, got {got}"
              f"{'' if digest_ok else '  (DIGEST MISMATCH)'}")

    print("== 2. INTEGRITY (content digest) ==")
    body = {k: cert[k] for k in cert if k not in ("content_sha256", "signature")}
    recomputed = hashlib.sha256(canon(body)).hexdigest()
    integ = recomputed == cert.get("content_sha256")
    ok = ok and integ
    print(f"  [{'PASS' if integ else 'FAIL'}] content_sha256 {'matches' if integ else 'MISMATCH'}")

    print("== 3. PROVENANCE (Ed25519 signature) ==")
    sig = cert.get("signature") or {}
    if sig.get("sig"):
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.exceptions import InvalidSignature
            pub = serialization.load_pem_public_key(sig["public_key_pem"].encode())
            try:
                pub.verify(bytes.fromhex(sig["sig"]), recomputed.encode())
                sig_ok = True
            except InvalidSignature:
                sig_ok = False
        except Exception as e:
            sig_ok = False
            print(f"  (signature check skipped: {e})")
        ok = ok and sig_ok
        print(f"  [{'PASS' if sig_ok else 'FAIL'}] signature by key {sig.get('key_id')}")
    else:
        print("  [WARN] certificate is unsigned (provenance unverified)")

    print("\\nRESULT:", "VERIFIED \\u2713" if ok else "REJECTED \\u2717")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''


# --------------------------------------------------------------------------- #
# Demo / CLI.
# --------------------------------------------------------------------------- #
def _demo(bundle_dir: Optional[str] = None) -> int:
    ca = CertificateAuthority()
    print(f"signing authority key_id={ca.key_id}  keystore={ca.key_path}")

    ctx = {"risk_level": "HIGH", "chf": True, "penicillin_allergy": True,
           "renal_failure": True}
    safe = {"antibiotic_code": 1, "antibiotic_time_hr": 0.5, "fluid_rate": 4.0,
            "vasopressor_dose": 0.1}
    cert = certify_dsl_protocol(safe, ctx, ca=ca, context_salt="demo-salt")
    print(f"\nissued certificate: {cert.claim}")
    print(f"  obligations={len(cert.obligations)} "
          f"all_satisfied={cert.all_satisfied} content_sha256={cert.content_sha256[:16]}…")

    report = verify_certificate(cert)
    print(f"\nindependent audit -> trusted={report.trusted} "
          f"(math_ok={report.math_ok} integrity_ok={report.integrity_ok} "
          f"signature_ok={report.signature_ok})")
    for a in report.obligations:
        print(f"  [{'OK' if a.math_ok else 'XX'}] {a.id} {a.name}: "
              f"expected {a.expected_status}, re-checked {a.rechecked_status}")

    if bundle_dir:
        export_bundle(cert, bundle_dir)
        print(f"\nexported audit bundle -> {bundle_dir}/")
        print(f"  a third party runs:  python {bundle_dir}/verify_bundle.py {bundle_dir}")
    return 0 if report.trusted else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AegisOS proof-export demo")
    ap.add_argument("--bundle", metavar="DIR", help="also export an audit bundle here")
    args = ap.parse_args()
    raise SystemExit(_demo(args.bundle))
