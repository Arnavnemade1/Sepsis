"""/proc — the AegisOS introspection namespace.

A real OS exposes its live state through one read-only filesystem (``/proc``,
``/sys``) instead of a hundred bespoke status calls. AegisOS does the same: the
kernel and the service manager register *providers* at paths, and every consumer —
the console (`ps`, `kstat`, `dmesg`), the UI, and one service reading another's
output — goes through this single source of truth. Providers are pull-based
(evaluated on read) so the namespace never holds stale state, and reads are
side-effect free.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

Provider = Callable[[], Any]


class ProcFS:
    def __init__(self) -> None:
        self._providers: Dict[str, Provider] = {}

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
        if prov is None:
            raise KeyError(f"no such proc path: {path}")
        return prov()

    def exists(self, path: str) -> bool:
        return (path if path.startswith("/") else "/" + path) in self._providers

    def list(self, prefix: str = "/") -> List[str]:
        return sorted(p for p in self._providers if p.startswith(prefix))

    def tree(self) -> Dict[str, Any]:
        """Snapshot every path (used by the UI / `proc` console command)."""
        out: Dict[str, Any] = {}
        for p in sorted(self._providers):
            try:
                out[p] = self._providers[p]()
            except Exception as exc:                 # a bad provider can't break the read
                out[p] = {"error": f"{type(exc).__name__}: {exc}"}
        return out
