"""/etc/policy.dsl — the hospital's safety policy, as a file the kernel enforces.

WHY THIS EXISTS
---------------
The Clinical Safety DSL (``clinical_dsl.py``) compiles hospital-authored rules to
Z3 and proves a protocol against them — and until now nothing in the running
system used it: the Safety Policy overlay proved hypothetical orders against a
policy typed into a text box, while the orders the kernel actually certified were
checked only against the cage's hardcoded invariants. A policy file that does not
constrain anything is decoration.

Now the policy is a file. Every Hour-1 bundle the kernel certifies is also proven
against the ACTIVE policy (``SepsisBundleEngine.policy``), and the bundle is only
``all_verified`` if the policy holds. Writing the file needs ``policy:write``
(the chief — who, by separation of duties, cannot prescribe) and is REFUSED if:

  * it does not parse, or
  * it is self-contradictory for any patient profile — no protocol could satisfy
    it (checked across all 24 combinations of CHF × penicillin allergy × renal
    failure × LOW/MEDIUM/HIGH risk, with the first contradictory profile named).

A refused write leaves the active policy untouched. The file persists through the
kernel's file store, so it survives a restart.

CODEBOOKS. The cage numbers antibiotics one way and the DSL another; an order is
translated drug-for-drug before it is proven, so the policy reasons about the drug
the patient would actually get. (The DSL gained Levofloxacin for this — the
browser's translation had been mapping it to Ciprofloxacin.)
"""
from __future__ import annotations

import itertools
import threading
import time
from typing import Any, Dict, List, Optional

from backend.aegis import clinical_dsl as dsl
from backend.aegis.kernel.journal import NOTICE, WARN, Journal, KJOURNAL

#: cage / bundle antibiotic code → DSL enum value, by drug
CAGE_TO_DSL = {
    0: dsl.ENUMS["antibiotic"]["NONE"],
    1: dsl.ENUMS["antibiotic"]["PipTazo"],
    2: dsl.ENUMS["antibiotic"]["Vancomycin"],
    3: dsl.ENUMS["antibiotic"]["Cefepime"],
    4: dsl.ENUMS["antibiotic"]["Meropenem"],
    5: dsl.ENUMS["antibiotic"]["Levofloxacin"],
}

#: The Hour-1 bundle's planned time-to-antibiotic: the bundle IS the first hour.
BUNDLE_ANTIBIOTIC_TIME_HR = 1.0

POLICY_PATH = "/etc/policy.dsl"


class PolicyRejected(ValueError):
    def __init__(self, reason: str, detail: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.detail = detail or {}


class PolicyStore:
    def __init__(self, journal: Optional[Journal] = None, store: Any = None,
                 initial: Optional[str] = None) -> None:
        self.journal = journal if journal is not None else KJOURNAL
        self.store = store                       # vfs.FileStore (owner "root") or None
        self._lock = threading.Lock()
        text = initial if initial is not None else self._load_persisted()
        if text is None:
            text = dsl.DEFAULT_POLICY
        self.text = text
        self.rules = dsl.compile_rules(text)
        self.version = 1
        self.author = "root"
        self.updated = time.time()
        self.certified = 0
        self.violations = 0

    def _load_persisted(self) -> Optional[str]:
        if self.store is None:
            return None
        try:
            rec = self.store.files("root").get(POLICY_PATH.lstrip("/"))
            return (rec or {}).get("content")
        except Exception:
            return None

    # -- writing ------------------------------------------------------------ #
    def consistency(self, rules: List[Any]) -> Dict[str, Any]:
        for chf, pen, ren, risk in itertools.product((False, True), (False, True),
                                                      (False, True), ("LOW", "MEDIUM", "HIGH")):
            ctx = {"chf": chf, "penicillin_allergy": pen, "renal_failure": ren, "risk_level": risk}
            res = dsl.check_consistency(rules, ctx)
            if not res.get("consistent"):
                return {"consistent": False, "profile": ctx}
        return {"consistent": True, "profiles_checked": 24}

    def write(self, cred: Any, text: str) -> Dict[str, Any]:
        """Replace the active policy (the VFS checks policy:write before calling)."""
        try:
            rules = dsl.compile_rules(text)
        except SyntaxError as exc:
            self.journal.append("policy", "policy.rejected", severity=WARN,
                                uid=cred.username, reason=f"parse: {exc}")
            raise PolicyRejected(f"the policy does not parse: {exc}")
        if not rules:
            raise PolicyRejected("a policy with no rules constrains nothing")
        cons = self.consistency(rules)
        if not cons["consistent"]:
            self.journal.append("policy", "policy.rejected", severity=WARN, uid=cred.username,
                                reason="contradictory", profile=cons["profile"])
            raise PolicyRejected("the policy contradicts itself for a patient profile — no "
                                 "protocol could satisfy it", cons)
        with self._lock:
            self.text, self.rules = text, rules
            self.version += 1
            self.author, self.updated = cred.username, time.time()
        if self.store is not None:
            self.store.put("root", POLICY_PATH.lstrip("/"), text)
        self.journal.append("policy", "policy.activated", severity=NOTICE, uid=cred.username,
                            version=self.version, rules=[r.name for r in rules])
        return self.status()

    # -- enforcing ------------------------------------------------------------ #
    def certify(self, protocol: Dict[str, Any], risk_level: str,
                history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prove a cage-codebook protocol against the active policy for a patient."""
        history = history or {}
        with self._lock:
            rules, version = self.rules, self.version
        dsl_proto = {
            "antibiotic_code": CAGE_TO_DSL.get(int(protocol.get("antibiotic_code", 0)), 0),
            "fluid_rate": float(protocol.get("fluid_rate", 0.0)),
            "vasopressor_dose": float(protocol.get("vasopressor_dose", 0.0)),
            "antibiotic_time_hr": float(protocol.get("antibiotic_time_hr",
                                                     BUNDLE_ANTIBIOTIC_TIME_HR)),
        }
        ctx = {"risk_level": risk_level, "chf": bool(history.get("chf")),
               "penicillin_allergy": bool(history.get("penicillin_allergy")),
               "renal_failure": bool(history.get("renal_failure"))}
        cert = dsl.verify_protocol(rules, dsl_proto, ctx)
        if cert.ok:
            self.certified += 1
        else:
            self.violations += 1
        return {"ok": cert.ok, "version": version, "n_rules": cert.n_rules,
                "n_satisfied": cert.n_satisfied,
                "violated": [{"rule": r.rule, "text": r.text, "counterexample": r.counterexample}
                             for r in cert.rules if not r.satisfied],
                "path": POLICY_PATH}

    def status(self) -> Dict[str, Any]:
        return {"path": POLICY_PATH, "version": self.version, "author": self.author,
                "updated": round(self.updated, 1), "rules": [r.name for r in self.rules],
                "certified": self.certified, "violations": self.violations}
