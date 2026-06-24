"""aegis-pkg — the application package registry (verify-then-install).

An app does not just "run" on AegisOS; it is **installed**, and installation is a
gate, not a copy. Every package is checked before it can ever be scheduled:

  1. **Manifest validity** — well-formed name, known egress classes, sane budget.
  2. **Capability policy** — the package may not exceed the platform's limits
     (lane budget, allowed egress classes); an over-reaching manifest is refused.
  3. **Code verification** —
       * a *bytecode* (eBPF) package is run through the real `kernel.ebpf.Verifier`
         (termination + memory + capability) — the formal guarantee, for genuinely
         untrusted code;
       * a *Python* package's source is statically **AST-audited** for the obvious
         escape hatches (os/subprocess/socket imports, eval/exec, dunder-escape
         attribute access). Honest scope: this is a static capability audit plus the
         runtime AppContext confinement — Python is not exec-sandboxable, so the
         *formal* path for untrusted third parties is the bytecode one.
  4. **Integrity** — the package's content hash must match its signature.

Only a package that clears all four is handed to `aegisd` and started. Default-deny:
nothing unverified is ever registered.
"""
from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.aegis.apps.sdk import AegisApp, AppManifest, EGRESS_CLASSES

# Python modules an app source may NEVER import (ambient-authority escapes).
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "importlib", "ctypes",
    "pickle", "marshal", "builtins", "multiprocessing", "threading", "pty",
    "signal", "resource", "mmap", "fcntl", "requests", "urllib",
}
# call names that are escape vectors
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "globals",
                   "locals", "vars", "input", "breakpoint", "memoryview"}
# attribute names that break out of the object graph
FORBIDDEN_ATTRS = {"__globals__", "__builtins__", "__class__", "__bases__",
                   "__subclasses__", "__mro__", "__dict__", "__code__",
                   "__closure__", "__getattribute__", "__reduce__"}


def audit_python_source(src: str) -> List[str]:
    """Static AST audit — returns a list of capability violations (empty == clean)."""
    out: List[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"source does not parse: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in FORBIDDEN_MODULES:
                    out.append(f"forbidden import {a.name!r}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in FORBIDDEN_MODULES:
                out.append(f"forbidden import from {node.module!r}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                out.append(f"forbidden call {node.func.id}()")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            out.append(f"forbidden attribute access {node.attr!r}")
    return out


@dataclass
class InstallPolicy:
    max_lane_budget: int = 4
    allowed_egress: tuple = EGRESS_CLASSES

    def check(self, m: AppManifest) -> List[str]:
        out: List[str] = []
        if m.lane_budget > self.max_lane_budget:
            out.append(f"lane_budget {m.lane_budget} exceeds policy max "
                       f"{self.max_lane_budget}")
        for e in m.emits:
            if e not in self.allowed_egress:
                out.append(f"egress class {e!r} not permitted by policy")
        return out


@dataclass
class AppPackage:
    manifest: AppManifest
    source: str                            # python source OR eBPF assembly (audited)
    kind: str = "python"                   # "python" | "ebpf"
    app: Optional[AegisApp] = None         # the loadable instance (python apps)
    signature: str = ""                    # integrity hash over manifest+kind+source

    def content_hash(self) -> str:
        blob = (self.kind + "|" + self.manifest.integrity_hash() + "|"
                + self.source).encode()
        return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]

    def sign(self) -> "AppPackage":
        self.signature = self.content_hash()
        return self


@dataclass
class InstallResult:
    ok: bool
    name: str
    stage: str                             # manifest | policy | verify | integrity | register
    reason: str = ""
    detail: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "name": self.name, "stage": self.stage,
                "reason": self.reason, "detail": self.detail}


class PackageRegistry:
    """The verify-then-install gate + the registry of installed packages."""

    def __init__(self, aegisd, policy: Optional[InstallPolicy] = None,
                 verifier=None, win_len: int = 2):
        self.aegisd = aegisd               # ServiceManager
        self.policy = policy or InstallPolicy()
        self.verifier = verifier           # kernel.ebpf.Verifier (for bytecode pkgs)
        self.win_len = win_len
        self.installed: Dict[str, Dict[str, Any]] = {}
        self.audit: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def install(self, pkg: AppPackage, start: bool = True) -> InstallResult:
        name = pkg.manifest.name
        res = self._gate(pkg)
        if not res.ok:
            self._record(res)
            return res
        # all gates passed → hand to aegisd
        try:
            self.aegisd.register(pkg.app)
            if start:
                self.aegisd._start(name)
        except Exception as exc:
            res = InstallResult(False, name, "register", f"{type(exc).__name__}: {exc}")
            self._record(res)
            return res
        self.installed[name] = {"manifest": pkg.manifest.to_dict(), "kind": pkg.kind,
                                "hash": pkg.signature, "ts": time.time()}
        ok = InstallResult(True, name, "register", "installed + started")
        self._record(ok)
        return ok

    def _gate(self, pkg: AppPackage) -> InstallResult:
        name = pkg.manifest.name
        # 1. manifest
        merr = pkg.manifest.validate()
        if merr:
            return InstallResult(False, name, "manifest", "invalid manifest", merr)
        if name in self.installed:
            return InstallResult(False, name, "manifest", "already installed")
        # 2. capability policy
        perr = self.policy.check(pkg.manifest)
        if perr:
            return InstallResult(False, name, "policy", "capability policy violation", perr)
        # 3. code verification
        if pkg.kind == "python":
            verr = audit_python_source(pkg.source)
            if verr:
                return InstallResult(False, name, "verify",
                                     "static code audit failed", verr)
            if pkg.app is None:
                return InstallResult(False, name, "verify", "no loadable app instance")
        elif pkg.kind == "ebpf":
            if self.verifier is None:
                return InstallResult(False, name, "verify", "no bytecode verifier")
            from backend.aegis.kernel.ebpf import assemble
            rep = self.verifier.verify(assemble(pkg.source), self.win_len)
            if not rep.ok:
                v = rep.violations[0]
                return InstallResult(False, name, "verify", "bytecode verifier rejected",
                                     [f"{v.rule} @#{v.index}: {v.detail}"])
        else:
            return InstallResult(False, name, "verify", f"unknown package kind {pkg.kind!r}")
        # 4. integrity
        if pkg.signature != pkg.content_hash():
            return InstallResult(False, name, "integrity",
                                 "signature does not match content hash",
                                 [f"declared {pkg.signature} != computed {pkg.content_hash()}"])
        return InstallResult(True, name, "verify", "all gates passed")

    def uninstall(self, name: str) -> bool:
        if name not in self.installed:
            return False
        self.aegisd.stop(name)
        self.aegisd.services.pop(name, None)
        self.installed.pop(name, None)
        self._record(InstallResult(True, name, "uninstall", "removed"))
        return True

    def _record(self, res: InstallResult) -> None:
        self.audit = (self.audit + [{**res.to_dict(), "ts": round(time.time(), 3)}])[-128:]

    # ------------------------------------------------------------------ #
    def list(self) -> Dict[str, Any]:
        return {"installed": list(self.installed.keys()),
                "packages": [{"name": n, **v} for n, v in self.installed.items()],
                "policy": {"max_lane_budget": self.policy.max_lane_budget,
                           "allowed_egress": list(self.policy.allowed_egress)},
                "audit": self.audit[-12:]}

    def info(self, name: str) -> Dict[str, Any]:
        return self.installed.get(name, {"error": f"not installed: {name}"})
