"""Where a PhysioNet password comes from.

In order: the environment (a container, fed from a secret store), the macOS Keychain, a
``~/.netrc`` that only its owner can read. Nothing here writes a password anywhere or accepts
one as an argument, because argv is world-readable through ``ps`` and ends up in shell history.

``login_command`` prints the Keychain command for the user to run themselves: ``security``
prompts on the terminal, so the password never passes through this process, a shell history
or an AI assistant's transcript.
"""
from __future__ import annotations

import base64
import netrc
import os
import platform
import re
import stat
import subprocess
from dataclasses import dataclass
from typing import Mapping, Optional

SERVICE = "physionet.org"


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str
    source: str

    def __repr__(self) -> str:                 # a traceback or a log line must never carry it
        return f"Credentials(username={self.username!r}, password=<hidden>, source={self.source!r})"

    __str__ = __repr__

    def header(self) -> str:
        return "Basic " + base64.b64encode(f"{self.username}:{self.password}".encode()).decode()


def _keychain(account: Optional[str]) -> Optional[Credentials]:
    if platform.system() != "Darwin":
        return None
    who = ["-a", account] if account else []
    try:
        meta = subprocess.run(["security", "find-generic-password", "-s", SERVICE, *who],
                              capture_output=True, text=True, timeout=10)
        if meta.returncode != 0:
            return None
        found = re.search(r'"acct"<blob>="([^"]*)"', meta.stdout)
        secret = subprocess.run(["security", "find-generic-password", "-s", SERVICE, *who, "-w"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if secret.returncode != 0 or not found:
        return None
    return Credentials(found.group(1), secret.stdout.rstrip("\n"), "keychain")


def _netrc(path: str) -> Optional[Credentials]:
    if not os.path.exists(path):
        return None
    if stat.S_IMODE(os.stat(path).st_mode) & 0o077:
        raise PermissionError(f"{path} is readable by others; chmod 600 it before it can be used")
    entry = netrc.netrc(path).authenticators(SERVICE)
    return Credentials(entry[0], entry[2], "netrc") if entry and entry[2] else None


def resolve(env: Optional[Mapping[str, str]] = None, netrc_path: Optional[str] = None,
            keychain: bool = True) -> Optional[Credentials]:
    env = os.environ if env is None else env
    user, password = env.get("PHYSIONET_USERNAME"), env.get("PHYSIONET_PASSWORD")
    if user and password:
        return Credentials(user, password, "environment")
    found = _keychain(user) if keychain else None
    return found or _netrc(netrc_path or os.path.expanduser("~/.netrc"))


def login_command(username: str = "<your-physionet-username>") -> str:
    return f"security add-generic-password -a {username} -s {SERVICE} -U -w"
