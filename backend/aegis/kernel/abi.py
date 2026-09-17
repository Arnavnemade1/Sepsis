"""The syscall ABI — every route the kernel serves, numbered, classed and capability-bound.

WHY THIS EXISTS
---------------
The gate cannot decide anything about a call it has no entry for, and a route
added later without one would be a door nobody guards. So the table of what the
kernel exposes is DATA, committed to the repository (``abi_table.json``), and
``verify_abi`` fails the build when a live route has no entry or an entry names a
route that no longer exists. Adding an endpoint now means stating, in review, what
class of call it is and which capability it needs.

Each entry:

    nr      stable syscall number (never reused)
    name    readable name derived from the path, e.g. ``cohort[bed_id].bundle``
    method  GET | POST | WS
    path    the route template, exactly as FastAPI declares it
    class   read | compute | write | egress | stream | auth
    cap     the capability the caller must hold (rbac.CAP_MIN_RING)
    refine  optional: the handler narrows further (``syscall`` checks the per-trap
            capability; ``signal`` lets owners signal their own processes)

CLASSES
    read     observes state                      never journalled
    compute  analyses without changing state     never journalled
    stream   a WebSocket subscription            journalled on connect only
    auth     sign-in / out / lock                journalled by kernel/users.py
    write    changes kernel or clinical state    journalled
    egress   an order that reaches a patient     journalled (and Z3-gated inside)

Regenerate after adding routes:  ``python -m backend.aegis.kernel.abi --sync``
(existing numbers, classes and capabilities are preserved; new routes get the
next numbers and a rule-based proposal that must be reviewed before commit).
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abi_table.json")
CLASSES = ("read", "compute", "write", "egress", "stream", "auth")
MUTATING = ("write", "egress")


@dataclass
class ABIEntry:
    nr: int
    name: str
    method: str
    path: str
    klass: str
    cap: str
    refine: str = ""
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["class"] = d.pop("klass")
        if not d["refine"]:
            d.pop("refine")
        if not d["note"]:
            d.pop("note")
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "ABIEntry":
        return ABIEntry(nr=int(d["nr"]), name=d["name"], method=d["method"], path=d["path"],
                        klass=d["class"], cap=d["cap"], refine=d.get("refine", ""),
                        note=d.get("note", ""))


def _template_regex(path: str) -> "re.Pattern":
    out, i = "^", 0
    for m in re.finditer(r"\{([^}:]+)(?::([a-z]+))?\}", path):
        out += re.escape(path[i:m.start()])
        out += "(?P<%s>.+)" % m.group(1) if m.group(2) == "path" else "(?P<%s>[^/]+)" % m.group(1)
        i = m.end()
    return re.compile(out + re.escape(path[i:]) + "$")


class ABI:
    """The loaded table, with O(1) exact lookups and a short regex list for templates."""

    def __init__(self, entries: Iterable[ABIEntry]) -> None:
        self.entries: List[ABIEntry] = sorted(entries, key=lambda e: e.nr)
        self.by_nr = {e.nr: e for e in self.entries}
        self._exact: Dict[Tuple[str, str], ABIEntry] = {}
        self._templ: Dict[str, List[Tuple["re.Pattern", ABIEntry]]] = {}
        for e in self.entries:
            if "{" in e.path:
                self._templ.setdefault(e.method, []).append((_template_regex(e.path), e))
            else:
                self._exact[(e.method, e.path)] = e

    @classmethod
    def load(cls, path: str = TABLE_PATH) -> "ABI":
        with open(path) as fh:
            data = json.load(fh)
        return cls(ABIEntry.from_json(d) for d in data["syscalls"])

    def resolve(self, method: str, path: str) -> Optional[ABIEntry]:
        e = self._exact.get((method, path))
        if e is not None:
            return e
        for rx, e in self._templ.get(method, ()):
            if rx.match(path):
                return e
        return None

    def coverage(self, routes: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
        live = {(m, p) for m, p in routes if p.startswith("/aegis")}
        table = {(e.method, e.path) for e in self.entries}
        missing = sorted(live - table)
        stale = sorted(table - live)
        return {"live": len(live), "entries": len(table), "missing": missing,
                "stale": stale, "complete": not missing and not stale}

    def summary(self) -> Dict[str, Any]:
        by_class: Dict[str, int] = {}
        for e in self.entries:
            by_class[e.klass] = by_class.get(e.klass, 0) + 1
        return {"syscalls": len(self.entries), "by_class": by_class}


# --------------------------------------------------------------------------- #
# Proposing entries for new routes (reviewed by a human before commit)
# --------------------------------------------------------------------------- #
#: (method, path regex, class, capability, refine). First match wins. These are a
#: PROPOSAL for a route nobody has classified yet; the committed table is the law.
RULES: List[Tuple[str, str, str, str, str]] = [
    # --- identity: sign-in and the policy it enforces
    ("POST", r"^/aegis/sys/(login|logout|lock|unlock)$", "auth", "view:public", ""),
    ("POST", r"^/aegis/rbac/login$", "auth", "view:public", ""),
    ("POST", r"^/aegis/rbac/authorize$", "compute", "view:public", ""),
    ("GET", r"^/aegis/rbac/audit$", "read", "view:audit", ""),
    ("POST", r"^/aegis/sys/breakglass$", "write", "breakglass:invoke", ""),
    ("POST", r"^/aegis/sys/breakglass/review$", "write", "breakglass:review", ""),
    ("GET", r"^/aegis/sys/breakglass$", "read", "view:audit", ""),
    ("POST", r"^/aegis/sys/revoke$", "write", "session:admin", ""),
    ("GET", r"^/aegis/sys/who$", "read", "view:sessions", ""),
    ("POST", r"^/aegis/sys/kill$", "write", "view:public", "signal"),
    ("POST", r"^/aegis/sys/gate/mode$", "write", "session:admin", ""),
    ("GET", r"^/aegis/sys/journal$", "read", "view:audit", ""),
    # --- the clinical egress path
    ("POST", r"^/aegis/syscall$", "egress", "order:escalate", "syscall"),
    ("POST", r"^/aegis/orchestrator/govern$", "egress", "order:escalate", ""),
    ("POST", r"^/aegis/orchestrator/scenario$", "write", "drill:crisis", ""),
    ("POST", r"^/aegis/cluster/chaos$", "write", "drill:crisis", ""),
    ("POST", r"^/aegis/identity/swap$", "write", "drill:crisis", ""),
    ("POST", r"^/aegis/kernel/(fault|recover)$", "write", "drill:device", ""),
    # --- administration
    ("POST", r"^/aegis/pkg/(install|uninstall)$", "write", "pkg:install", ""),
    ("POST", r"^/aegis/services/", "write", "svc:control", ""),
    ("POST", r"^/aegis/ml/(train|use)$", "write", "ml:train", ""),
    # --- records and bedside
    ("POST", r"^/aegis/patient/[^/]+/history(/link)?$", "write", "record:import", ""),
    ("POST", r"^/aegis/patient/[^/]+/(track|synthesis)$", "write", "annotate", ""),
    ("POST", r"^/aegis/devices/", "write", "device:manage", ""),
    ("POST", r"^/aegis/ingest$", "write", "telemetry:ingest", ""),
    ("POST", r"^/aegis/(simulate/|ppg/vasodilation|hdkernel/run|scenarios/)", "write",
     "sim:control", ""),
    # --- analyses that change nothing
    ("POST", r"^/aegis/(llm/|doctor/|agent/invoke|speech)", "compute", "compute:ask", ""),
    ("POST", r"^/aegis/(whatif|pk/project|pharmacopoeia/screen|dsl/verify|proof/|"
             r"causal/replay|kernel/verify|identity/self_check|hyperscan)", "compute",
     "compute:analyze", ""),
    # --- sensitive reads
    ("GET", r"^/aegis/(syscall/audit|devices/audit|agent/activity|audit)$", "read",
     "view:audit", ""),
    # --- streams
    ("WS", r"^/aegis/doctor/live$", "stream", "compute:ask", ""),
    ("WS", r"^/aegis/", "stream", "view:public", ""),
    # --- everything else readable
    ("GET", r"^/aegis/", "read", "view:public", ""),
]


def classify(method: str, path: str) -> Tuple[str, str, str]:
    probe = re.sub(r"\{[^}]+\}", "x", path)
    for m, rx, klass, cap, refine in RULES:
        if m == method and re.search(rx, probe):
            return klass, cap, refine
    return ("write" if method == "POST" else "read"), "session:admin", ""


def name_for(method: str, path: str, taken: set) -> str:
    parts = [p for p in path[len("/aegis/"):].split("/") if p]
    name = ""
    for p in parts:
        m = re.fullmatch(r"\{([^}:]+)(?::[a-z]+)?\}", p)
        name += ("[%s]" % m.group(1)) if m else ("." + p if name else p)
    name = name or "root"
    if method == "WS":
        name += ".stream" if not name.endswith("stream") else ""
    if name in taken:
        name += ".set" if method == "POST" else ".get"
    return name


def live_routes(app: Any) -> List[Tuple[str, str]]:
    from starlette.routing import Route, WebSocketRoute
    out = []
    for r in app.routes:
        path = getattr(r, "path", "")
        if not path.startswith("/aegis"):
            continue
        if isinstance(r, WebSocketRoute):
            out.append(("WS", path))
        elif isinstance(r, Route):
            for m in sorted(r.methods or ()):
                if m in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                    out.append((m, path))
    return out


def sync_table(routes: List[Tuple[str, str]], existing: Optional[ABI] = None
               ) -> Tuple[List[ABIEntry], List[ABIEntry]]:
    """(table, newly_proposed). Existing entries are kept verbatim."""
    keep = {} if existing is None else {(e.method, e.path): e for e in existing.entries}
    live = set(routes)
    table = [e for k, e in keep.items() if k in live]
    taken = {e.name for e in table}
    nr = max([e.nr for e in (existing.entries if existing else [])] or [0])
    new: List[ABIEntry] = []
    for method, path in sorted(live - set(keep)):
        nr += 1
        klass, cap, refine = classify(method, path)
        e = ABIEntry(nr=nr, name=name_for(method, path, taken), method=method, path=path,
                     klass=klass, cap=cap, refine=refine)
        taken.add(e.name)
        table.append(e)
        new.append(e)
    return sorted(table, key=lambda e: e.nr), new


def write_table(entries: List[ABIEntry], path: str = TABLE_PATH) -> None:
    body = {"schema": 1,
            "about": "AegisOS syscall ABI. Every /aegis route, its class and required "
                     "capability. Enforced by kernel/gate.py; completeness checked by "
                     "verify_abi.py. Regenerate with: python -m backend.aegis.kernel.abi --sync",
            "syscalls": [e.to_json() for e in entries]}
    with open(path, "w") as fh:
        json.dump(body, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":                        # pragma: no cover
    if "--sync" in sys.argv or "--check" in sys.argv:
        from backend.aegis.server import app
        routes = live_routes(app)
        existing = ABI.load() if os.path.exists(TABLE_PATH) else None
        if "--check" in sys.argv:
            cov = (existing or ABI([])).coverage(routes)
            print(json.dumps(cov, indent=1))
            sys.exit(0 if cov["complete"] else 1)
        table, new = sync_table(routes, existing)
        write_table(table)
        print(f"{len(table)} syscalls; {len(new)} new (REVIEW their class and capability):")
        for e in new:
            print(f"  {e.nr:4d} {e.method:4s} {e.path:48s} {e.klass:8s} {e.cap}")
        sys.stdout.flush()
        os._exit(0)                               # the server started background threads
    print(__doc__)
