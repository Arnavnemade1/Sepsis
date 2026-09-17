"""/proc — the AegisOS introspection namespace.

A real OS exposes its live state through one read-only filesystem (``/proc``,
``/sys``) instead of a hundred bespoke status calls. AegisOS does the same: the
kernel and the service manager register *providers* at paths, and every consumer —
the console (`ps`, `kstat`, `dmesg`), the UI, and one service reading another's
output — goes through this single source of truth. Providers are pull-based
(evaluated on read) so the namespace never holds stale state.

Reads are side-effect free. (Until the kernel process table existed,
``/proc/kthreads`` seeded and ran an HDKernel demo workload on first read, so
reading the process table created the processes it reported. It now reads the
real table; the demo workload runs only when asked, via ``/aegis/hdkernel/run``.)

Besides fixed paths, a *dynamic* mount resolves a family of paths on demand —
``/proc/<pid>`` for every live process — without registering one provider per
process and leaving stale ones behind when processes exit.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

Provider = Callable[[], Any]


class ProcFS:
    def __init__(self) -> None:
        self._providers: Dict[str, Provider] = {}
        # (compiled pattern, resolve(match) -> value, list() -> paths)
        self._dynamic: List[Tuple[Any, Callable[[Any], Any], Callable[[], List[str]]]] = []

    def register_dynamic(self, pattern: str, resolve: Callable[[Any], Any],
                         listing: Callable[[], List[str]]) -> None:
        """Mount a family of paths, e.g. ``r"/proc/(\\d+)"`` → one process."""
        self._dynamic.append((re.compile("^" + pattern + "$"), resolve, listing))

    def _resolve_dynamic(self, path: str) -> Optional[Tuple[Callable[[Any], Any], Any]]:
        for pat, resolve, _ in self._dynamic:
            m = pat.match(path)
            if m:
                return resolve, m
        return None

    def register(self, path: str, provider: Provider) -> None:
        """Mount a live state provider at ``path`` (e.g. ``/proc/services``)."""
        if not path.startswith("/"):
            path = "/" + path
        self._providers[path] = provider

    def unregister(self, path: str) -> None:
        self._providers.pop(path, None)

    def read(self, path: str) -> Any:
        if not path.startswith("/"):
            path = "/" + path
        prov = self._providers.get(path)
        if prov is not None:
            return prov()
        dyn = self._resolve_dynamic(path)
        if dyn is None:
            raise KeyError(f"no such proc path: {path}")
        resolve, m = dyn
        return resolve(m)

    def exists(self, path: str) -> bool:
        path = path if path.startswith("/") else "/" + path
        if path in self._providers:
            return True
        dyn = self._resolve_dynamic(path)
        if dyn is None:
            return False
        try:
            dyn[0](dyn[1])
            return True
        except KeyError:
            return False

    def list(self, prefix: str = "/") -> List[str]:
        paths = set(self._providers)
        for _, _, listing in self._dynamic:
            try:
                paths.update(listing())
            except Exception:
                pass
        return sorted(p for p in paths if p.startswith(prefix))

    def tree(self) -> Dict[str, Any]:
        """Snapshot every path (used by the UI / `proc` console command)."""
        out: Dict[str, Any] = {}
        for p in sorted(self._providers):
            try:
                out[p] = self._providers[p]()
            except Exception as exc:                 # a bad provider can't break the read
                out[p] = {"error": f"{type(exc).__name__}: {exc}"}
        return out
