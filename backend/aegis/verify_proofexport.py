"""Verify harness for the Proof-Export & Independent-Audit layer — claims asserted.

We prove the export layer delivers what the medical claim requires: a certificate a
*stranger* can check without trusting AegisOS, and that no adversary can forge.

  1. ISSUE        — a safe protocol yields a signed certificate; every obligation is
                    a complete SMT-LIB2 script with expected status ``unsat``.
  2. SELF-AUDIT   — the issuer's own re-check passes (math + integrity + signature).
  3. INDEPENDENCE — every obligation re-verifies in a FRESH solver from the SMT text
                    alone — exactly what a third party with only the .smt2 sees.
  4. CATCH        — a violating protocol produces a ``sat`` obligation with a
                    counterexample, and the audit reports it (not "trusted").
  5. TAMPER       — flipping a byte in an SMT script, or in the protocol body, breaks
                    the content digest AND the signature.
  6. FORGE        — re-signing a tampered body with a DIFFERENT key still fails: the
                    embedded public key cannot validate AegisOS's intent... and more
                    importantly, a signed-but-LYING certificate (claims unsat, math is
                    sat) is caught by the independent re-check. Signature ≠ truth.
  7. PROVENANCE   — the Ed25519 signature verifies with the PUBLIC KEY ALONE; the
                    wrong public key rejects it.
  8. BUNDLE       — export writes certificate.json, per-obligation .smt2 files, the
                    public key, AUDIT.md, and a standalone verify_bundle.py; and the
                    standalone verifier (no AegisOS imports) verifies the bundle.
  9. CORE PATH    — the live runtime's core Z3 cage exports one obligation per named
                    clinical invariant, agreeing with the cage's own verdict (single
                    source of truth), self-documenting the bound patient context.

    python -m backend.aegis.verify_proofexport
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

from z3 import Solver, parse_smt2_string, sat, unsat

from backend.aegis.proofexport import (CertificateAuthority, ProofCertificate,
                                       certify_dsl_protocol, certify_core_protocol,
                                       export_bundle, verify_certificate,
                                       verify_signature)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


SAFE_CTX = {"risk_level": "HIGH", "chf": True, "penicillin_allergy": True,
            "renal_failure": True}
SAFE = {"antibiotic_code": 1, "antibiotic_time_hr": 0.5, "fluid_rate": 4.0,
        "vasopressor_dose": 0.1}
BAD = {"antibiotic_code": 2, "antibiotic_time_hr": 3.0, "fluid_rate": 30.0,
       "vasopressor_dose": 0.1}   # PipTazo w/ allergy, late, CHF fluid overload


def main() -> int:
    checks = []
    ca = CertificateAuthority()

    # ---- 1. ISSUE -------------------------------------------------------- #
    _hr("1. ISSUE (safe protocol -> signed certificate)")
    cert = certify_dsl_protocol(SAFE, SAFE_CTX, ca=ca, context_salt="t")
    print(f"    claim: {cert.claim}")
    print(f"    obligations={len(cert.obligations)} all_satisfied={cert.all_satisfied}")
    checks.append(("certificate issued with one obligation per rule",
                   len(cert.obligations) == 4))
    checks.append(("every obligation is a complete SMT-LIB2 script ending in (check-sat)",
                   all("(check-sat)" in o.smt2 for o in cert.obligations)))
    checks.append(("every obligation expects unsat (proven safe)",
                   all(o.expected_status == "unsat" for o in cert.obligations)))
    checks.append(("certificate carries a content digest + Ed25519 signature",
                   bool(cert.content_sha256) and cert.signature.get("alg") == "Ed25519"))
    checks.append(("no PHI in subject — only a context hash + protocol params",
                   "patient_context_hash" in cert.subject
                   and "risk_level" not in json.dumps(cert.subject)))

    # ---- 2. SELF-AUDIT --------------------------------------------------- #
    _hr("2. SELF-AUDIT (issuer re-checks its own certificate)")
    rep = verify_certificate(cert)
    print(f"    trusted={rep.trusted} math_ok={rep.math_ok} "
          f"integrity_ok={rep.integrity_ok} signature_ok={rep.signature_ok}")
    checks.append(("a freshly issued certificate is fully trusted", rep.trusted))
    checks.append(("audit reports the signing key id",
                   rep.key_id == ca.key_id and len(rep.key_id) == 16))

    # ---- 3. INDEPENDENCE ------------------------------------------------- #
    _hr("3. INDEPENDENCE (re-verify from the SMT text alone, fresh solver)")
    indep_ok = True
    for o in cert.obligations:
        s = Solver()
        s.add(parse_smt2_string(o.smt2))      # only the published bytes; no AegisOS
        got = s.check()
        status = "sat" if got == sat else "unsat" if got == unsat else "unknown"
        indep_ok = indep_ok and status == o.expected_status
        print(f"    {o.id} {o.name}: fresh solver -> {status}")
    checks.append(("every obligation re-checks correctly in a fresh solver", indep_ok))

    # ---- 4. CATCH -------------------------------------------------------- #
    _hr("4. CATCH (violating protocol -> sat obligation + counterexample)")
    bad_cert = certify_dsl_protocol(BAD, SAFE_CTX, ca=ca)
    violated = [o for o in bad_cert.obligations if not o.satisfied]
    print(f"    violated rules: {[o.name for o in violated]}")
    bad_rep = verify_certificate(bad_cert)
    print(f"    audit trusted={bad_rep.trusted} (math_ok={bad_rep.math_ok} "
          f"consistent={bad_rep.consistent})")
    checks.append(("a violating protocol yields >=1 sat obligation", len(violated) >= 1))
    checks.append(("violations carry a Z3 counterexample model",
                   all(o.counterexample for o in violated)))
    checks.append(("the audit's math re-check still passes (sat WAS expected)",
                   bad_rep.math_ok and bad_rep.consistent))
    checks.append(("a certificate with violations is NOT 'all satisfied'",
                   not bad_cert.all_satisfied))

    # ---- 5. TAMPER ------------------------------------------------------- #
    _hr("5. TAMPER (mutating bytes breaks digest + signature)")
    # (a) tamper an SMT script
    d = cert.to_dict()
    d["obligations"][0]["smt2"] = d["obligations"][0]["smt2"].replace("4.0", "40.0", 1)
    rep_a = verify_certificate(d)
    print(f"    smt-script tamper -> trusted={rep_a.trusted} "
          f"integrity_ok={rep_a.integrity_ok} math_ok={rep_a.math_ok}")
    checks.append(("tampering an SMT script is detected (not trusted)", not rep_a.trusted))
    # (b) tamper the protocol body
    d2 = cert.to_dict()
    d2["subject"]["protocol"]["fluid_rate"] = 999.0
    rep_b = verify_certificate(d2)
    print(f"    protocol-body tamper -> trusted={rep_b.trusted} "
          f"integrity_ok={rep_b.integrity_ok} signature_ok={rep_b.signature_ok}")
    checks.append(("tampering the protocol body breaks the content digest",
                   not rep_b.integrity_ok))
    checks.append(("tampering the protocol body breaks the signature",
                   not rep_b.signature_ok))

    # ---- 6. FORGE -------------------------------------------------------- #
    _hr("6. FORGE (signature ≠ truth; a signed lie is caught by re-checking)")
    # Forge a certificate that CLAIMS all-satisfied but whose SMT is actually sat,
    # then sign it with a brand-new valid key. The signature is valid — the MATH is not.
    forge_dir = tempfile.mkdtemp(prefix="aegis-forge-")
    rogue = CertificateAuthority(key_path=os.path.join(forge_dir, "rogue.pem"))
    forged = certify_dsl_protocol(BAD, SAFE_CTX, ca=rogue)   # genuinely-sat obligations
    fd = forged.to_dict()
    fd["verdict"]["all_satisfied"] = True                    # the LIE
    for o in fd["obligations"]:
        o["expected_status"] = "unsat"                       # claim each is proven
    # re-seal with the rogue key so signature is internally valid over the lie
    from backend.aegis.proofexport import _canon, _sha256
    body = {k: fd[k] for k in fd if k not in ("content_sha256", "signature")}
    fd["content_sha256"] = _sha256(_canon(body))
    fd["signature"] = {"alg": "Ed25519", "key_id": rogue.key_id,
                       "public_key_pem": rogue.public_pem,
                       "sig": rogue.sign(fd["content_sha256"])}
    forged_rep = verify_certificate(fd)
    print(f"    forged cert: signature_ok={forged_rep.signature_ok} "
          f"math_ok={forged_rep.math_ok} trusted={forged_rep.trusted}")
    checks.append(("a forged cert can have a VALID signature", forged_rep.signature_ok))
    checks.append(("...but the independent math re-check exposes the lie",
                   not forged_rep.math_ok))
    checks.append(("...so a signed lie is NEVER trusted", not forged_rep.trusted))

    # ---- 7. PROVENANCE --------------------------------------------------- #
    _hr("7. PROVENANCE (public key alone verifies; wrong key rejects)")
    digest = cert.content_sha256
    sig_hex = cert.signature["sig"]
    good_pub = cert.signature["public_key_pem"]
    ok_pub = verify_signature(good_pub, digest, sig_hex)
    bad_pub = verify_signature(rogue.public_pem, digest, sig_hex)
    print(f"    correct public key -> {ok_pub}; rogue public key -> {bad_pub}")
    checks.append(("the genuine public key verifies the signature", ok_pub))
    checks.append(("a different public key rejects the signature", not bad_pub))

    # ---- 8. BUNDLE ------------------------------------------------------- #
    _hr("8. BUNDLE (on-disk artifact; standalone verifier with no AegisOS)")
    out = tempfile.mkdtemp(prefix="aegis-bundle-")
    export_bundle(cert, out)
    have = os.listdir(out)
    ob_files = os.listdir(os.path.join(out, "obligations"))
    print(f"    bundle files: {sorted(have)}")
    print(f"    obligation scripts: {sorted(ob_files)}")
    checks.append(("bundle has certificate.json + AUDIT.md + public_key.pem + verifier",
                   {"certificate.json", "AUDIT.md", "public_key.pem",
                    "verify_bundle.py"} <= set(have)))
    checks.append(("bundle writes one .smt2 per obligation",
                   len([f for f in ob_files if f.endswith(".smt2")]) == 4))
    # Run the STANDALONE verifier as a subprocess — no aegis on its import path.
    proc = subprocess.run([sys.executable, os.path.join(out, "verify_bundle.py"), out],
                          capture_output=True, text=True)
    print("    --- standalone verify_bundle.py output ---")
    for ln in proc.stdout.strip().splitlines():
        print(f"      {ln}")
    checks.append(("standalone verifier (no AegisOS imports) returns VERIFIED",
                   proc.returncode == 0 and "VERIFIED" in proc.stdout))

    # ---- 9. CORE-CAGE PATH ---------------------------------------------- #
    _hr("9. CORE-CAGE PATH (export the live runtime's Z3 invariants)")
    # The SAME prover AegisCore uses live. Safe order → all unsat; and it must agree
    # with the cage's own verify_protocol verdict (single source of truth).
    from backend.core.verifier import ProtocolVerifier
    pv = ProtocolVerifier()
    chist = {"chf": True, "renal_failure": False, "penicillin_allergy": True}
    cvit = {"map": 62.0}
    safe_core = {"fluid_rate": 4.0, "vasopressor_dose": 0.2, "antibiotic_code": 3}
    core_cert = certify_core_protocol(chist, cvit, "HIGH", safe_core, ca=ca,
                                      context_salt="patientX")
    core_rep = verify_certificate(core_cert)
    cage_ok = pv.verify_protocol(chist, cvit, "HIGH", safe_core)[0]
    print(f"    obligations={len(core_cert.obligations)} all_satisfied={core_cert.all_satisfied}")
    print(f"    independent audit trusted={core_rep.trusted}; cage verdict ok={cage_ok}")
    print(f"    drug certified: {core_cert.subject['protocol']['antibiotic_name']}")
    checks.append(("core path emits one obligation per cage invariant",
                   len(core_cert.obligations) == 10))
    checks.append(("core path: safe order is fully trusted on independent re-check",
                   core_rep.trusted))
    checks.append(("core path agrees with the live cage's verify_protocol verdict",
                   cage_ok and core_cert.all_satisfied))
    # A genuinely unsafe order (penicillin allergy + Pip-Tazo code 1) must be caught.
    bad_core = {"fluid_rate": 4.0, "vasopressor_dose": 0.2, "antibiotic_code": 1}
    bad_core_cert = certify_core_protocol(chist, cvit, "HIGH", bad_core, ca=ca)
    bad_names = [o.name for o in bad_core_cert.obligations if not o.satisfied]
    cage_bad = pv.verify_protocol(chist, cvit, "HIGH", bad_core)[0]
    print(f"    unsafe order violated: {bad_names}; cage ok={cage_bad}")
    checks.append(("core path catches the allergy violation, agreeing with the cage",
                   "allergy_guard" in bad_names and not cage_bad))
    checks.append(("core path obligations carry self-documenting bound context",
                   "has_penicillin_allergy" in core_cert.obligations[0].smt2))
    # Cross-language robustness: a JS auditor's JSON.stringify collapses 4.0 -> 4;
    # the digest + signature must survive that round-trip.
    from backend.aegis.proofexport import _normalize_numbers
    import json as _json
    js_wire = _json.loads(_json.dumps(_normalize_numbers(core_cert.to_dict())))
    checks.append(("certificate survives a JS-style number round-trip (4.0→4)",
                   verify_certificate(js_wire).trusted))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    failures = 0
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — every protocol ships a portable proof certificate "
          "any third party can verify with their own solver, and no forgery survives "
          "an independent re-check."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
