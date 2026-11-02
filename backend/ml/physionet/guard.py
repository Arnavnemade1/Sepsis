"""The Data Use Agreement, as code.

The agreement asks for four things this machine can check: don't share access (item 3), protect
the data's security (item 4), keep identities out of communications (items 1-2), and stay within
research use. PhysioNet has also said in plain words that sending credentialed data to third-party
LLM APIs is prohibited (physionet.org/news/post/gpt-responsible-use). So this project's OpenRouter
and consumer Gemini paths may never see a row, and neither may a person's copy-paste.

**Posture** (``doctor``): is this machine fit to hold the data? Encrypted disk, a lake that only
its owner can read, outside the repository, outside any folder that syncs to someone's cloud,
excluded from backups and from search indexing.

**Egress** (``scan``): can a file leave? Three independent signals:

* *Forbidden files*: Parquet, the grid caches, anything inside the lake.
* *Identifier canary*: the novel one. When the lake is built, the salted hash of every real
  ``subject_id``, ``hadm_id`` and ``stay_id`` goes into a local file. A scanner then recognises a
  real patient's identifier anywhere, in any format, while the canary file itself reveals nothing
  and needs no copy of the data. The demo's identifiers are open data, so they are excluded.
* *Shape*: shifted dates (21xx, a calendar only MIMIC uses), and row-level identifier columns.

Small cells (a count under 10 in a report) are a warning: PhysioNet sets no number, and 10 is
this project's own policy, a common convention for re-identification risk.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import stat
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from backend.ml import config

SYNCED = ("Library/Mobile Documents", "Dropbox", "Google Drive", "GoogleDrive", "OneDrive", "iCloud", "Box Sync")
#: on macOS, iCloud's "Desktop & Documents" sync moves these without changing their path
SYNC_PRONE = ("Desktop", "Documents")
FORBIDDEN_SUFFIX = (".parquet", ".csv.gz", "_store.npz", "_meta.pkl", "_onsets.pkl", "_features.npy")
ID_TOKEN = re.compile(r"(?<![\d.\-])\d{8}(?![\d.])")
SHIFTED_DATE = re.compile(r"(?<!\d)2[12]\d\d-[01]\d-[0-3]\d")
ID_COLUMN = re.compile(r"""["'\b](subject_id|hadm_id|stay_id|charttime|storetime|intime|outtime|admittime)["'\b]\s*[:,]""")
SMALL_N = re.compile(r"""["'](n|n_[a-z_]+|count|stays|subjects|patients|deaths)["']\s*:\s*([1-9])\s*[,}\n]""")
SOURCE_CODE = (".py", ".js", ".mjs", ".ts", ".tsx", ".css", ".html", ".sql", ".sh")
TEXT_MAX_BYTES = 32 << 20
SMALL_CELL = 10


def default_lake() -> str:
    return os.environ.get("PHYSIONET_LAKE") or os.path.expanduser("~/physionet-lake")


def _inside(path: str, parent: str) -> bool:
    path, parent = os.path.realpath(path), os.path.realpath(parent)
    return path == parent or path.startswith(parent + os.sep)


# --------------------------------------------------------------------------- #
# Posture
# --------------------------------------------------------------------------- #
def init_lake(root: str) -> None:
    """Create the lake private from its first moment, and out of the reach of backups and search."""
    os.makedirs(root, mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    open(os.path.join(root, ".metadata_never_index"), "a").close()
    if platform.system() == "Darwin":
        subprocess.run(["tmutil", "addexclusion", root], capture_output=True, timeout=20)


def _disk_encrypted() -> Optional[bool]:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["fdesetup", "status"], capture_output=True, text=True, timeout=10).stdout
            return "FileVault is On" in out
        return None
    except (OSError, subprocess.SubprocessError):
        return None


def doctor(root: str, repo: str = config.REPO_ROOT, creds_needed: bool = True) -> Dict:
    from backend.ml.physionet import auth
    checks: List[Dict] = []

    def add(name: str, ok: Optional[bool], detail: str, fix: str = "", severity: str = "fail") -> None:
        checks.append({"check": name, "ok": ok, "detail": detail, "fix": fix, "severity": severity})

    enc = _disk_encrypted()
    add("disk encryption", enc, {True: "FileVault is on", False: "FileVault is OFF", None: "not measurable here"}[enc],
        "turn on FileVault (or use an encrypted volume)",
        severity="fail" if enc is False else "warn")
    add("lake outside the repository", not _inside(root, repo), root, "set PHYSIONET_LAKE to a path outside the repo")
    real = os.path.realpath(root)
    synced = [s for s in SYNCED if s.lower() in real.lower()]
    add("lake outside synced folders", not synced, f"matched {synced}" if synced else real,
        "move it off the synced folder: a sync provider is a third party")
    home = os.path.expanduser("~")
    prone = [s for s in SYNC_PRONE if _inside(real, os.path.join(home, s))]
    add("lake outside Desktop and Documents", not prone, f"inside ~/{prone[0]}" if prone else "ok",
        "iCloud's Desktop & Documents sync would upload it")
    if os.path.exists(root):
        mode = stat.S_IMODE(os.stat(root).st_mode)
        add("lake readable only by its owner", not mode & 0o077, oct(mode), f"chmod 700 {root}")
        if platform.system() == "Darwin":
            tm = subprocess.run(["tmutil", "isexcluded", root], capture_output=True, text=True, timeout=20).stdout
            add("lake excluded from Time Machine", "[Excluded]" in tm, tm.strip().split("\n")[0][:80] or "unknown",
                f"tmutil addexclusion {root}", severity="warn")
    else:
        add("lake readable only by its owner", None, "not created yet; created private on first use", severity="warn")
    hook = git_hook_path(repo)
    installed = bool(hook) and os.path.exists(hook) and HOOK_MARK in open(hook).read()
    add("egress hook on git commit", installed, hook or "no git hooks directory",
        "python -m backend.ml.physionet hook", severity="warn")
    try:
        creds = auth.resolve()
        add("PhysioNet credentials available", creds is not None,
            f"from {creds.source}" if creds else "none found", auth.login_command(),
            severity="fail" if creds_needed else "warn")
    except PermissionError as err:
        add("PhysioNet credentials available", False, str(err), severity="fail")
    failing = [c["check"] for c in checks if c["ok"] is False and c["severity"] == "fail"]
    return {"ok": not failing, "failing": failing, "checks": checks}


# --------------------------------------------------------------------------- #
# The canary
# --------------------------------------------------------------------------- #
@dataclass
class Canary:
    salt: str = ""
    hashes: Set[str] = field(default_factory=set)

    @staticmethod
    def _h(salt: str, token: str) -> str:
        return hashlib.sha256((salt + token).encode()).hexdigest()[:20]

    @classmethod
    def build(cls, ids: Iterable[int], open_ids: Iterable[int] = ()) -> "Canary":
        salt = secrets.token_hex(16)
        skip = {str(int(i)) for i in open_ids}
        return cls(salt, {cls._h(salt, str(int(i))) for i in ids if str(int(i)) not in skip})

    def hit(self, token: str) -> bool:
        return self._h(self.salt, token) in self.hashes

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"salt": self.salt, "hashes": sorted(self.hashes)}, fh)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> Optional["Canary"]:
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["salt"], set(d["hashes"]))


# --------------------------------------------------------------------------- #
# Egress
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    path: str
    rule: str
    severity: str     # block | warn
    count: int = 1

    def text(self) -> str:                # names the rule and the count, never the matched value
        return f"{self.severity.upper():<5} {self.path}: {self.rule}" + (f" ({self.count}x)" if self.count > 1 else "")


def scan_text(path: str, text: str, canary: Optional[Canary]) -> List[Finding]:
    out: List[Finding] = []
    if canary is not None:
        hits = sum(1 for tok in ID_TOKEN.findall(text) if canary.hit(tok))
        if hits:
            out.append(Finding(path, "a real patient identifier from the lake", "block", hits))
    # test fixtures in source code use the shifted calendar on purpose; the canary still reads them
    dates = 0 if path.endswith(SOURCE_CODE) else len(SHIFTED_DATE.findall(text))
    if dates:
        out.append(Finding(path, "dates in MIMIC's shifted calendar", "block", dates))
    is_data = path.endswith((".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".md"))
    if is_data:
        cols = len(ID_COLUMN.findall(text))
        if cols:
            out.append(Finding(path, "row-level identifier or time columns", "block", cols))
        small = len(SMALL_N.findall(text))
        if small:
            out.append(Finding(path, f"counts under {SMALL_CELL} (project policy)", "warn", small))
    return out


def scan_bytes(path: str, data: bytes, canary: Optional[Canary], lake: Optional[str] = None) -> List[Finding]:
    low = path.lower()
    if low.endswith(FORBIDDEN_SUFFIX):
        return [Finding(path, "patient-level data format", "block")]
    if data[:4] == b"PAR1":
        return [Finding(path, "a Parquet file under another name", "block")]
    if lake and _inside(path, lake) and not _inside(path, os.path.join(lake, "reports")):
        return [Finding(path, "a file inside the lake", "block")]
    if len(data) > TEXT_MAX_BYTES:
        return [Finding(path, "too large to be an aggregate report", "block")]
    if b"\0" in data[:8192]:
        return []                         # binary: the format rules above are what apply
    return scan_text(path, data.decode("utf-8", "replace"), canary)


def scan_paths(paths: Sequence[str], canary: Optional[Canary], lake: Optional[str] = None) -> List[Finding]:
    out: List[Finding] = []
    for root in paths:
        files = [root] if os.path.isfile(root) else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
        for f in files:
            with open(f, "rb") as fh:
                out += scan_bytes(f, fh.read(TEXT_MAX_BYTES + 1), canary, lake)
    return out


def scan_staged(repo: str, canary: Optional[Canary]) -> List[Finding]:
    """What this commit would publish: the staged bytes, not the working copy."""
    names = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], cwd=repo,
                           capture_output=True, timeout=60).stdout.decode().split("\0")
    out: List[Finding] = []
    for name in filter(None, names):
        data = subprocess.run(["git", "show", f":{name}"], cwd=repo, capture_output=True, timeout=60).stdout
        out += scan_bytes(name, data, canary)
    return out


# --------------------------------------------------------------------------- #
# The hook
# --------------------------------------------------------------------------- #
HOOK_MARK = "# aegis-physionet-egress"


def git_hook_path(repo: str) -> Optional[str]:
    try:
        d = subprocess.run(["git", "rev-parse", "--git-path", "hooks"], cwd=repo, capture_output=True, text=True,
                           timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return os.path.join(repo, d, "pre-commit") if d else None


def install_hook(repo: str = config.REPO_ROOT) -> str:
    """Add the egress scan to pre-commit, after whatever is already there."""
    path = git_hook_path(repo)
    if path is None:
        raise RuntimeError("not a git repository")
    body = open(path).read() if os.path.exists(path) else "#!/bin/sh\n"
    if HOOK_MARK in body:
        return path
    block = (f"\n{HOOK_MARK}\nPY=.venv/bin/python; [ -x \"$PY\" ] || PY=python3\n"
             "\"$PY\" -m backend.ml.physionet scan --staged || exit 1\n")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body.rstrip("\n") + "\n" + block)
    os.chmod(path, 0o755)
    return path
