"""Verified transport. Every byte that reaches the lake is checked against PhysioNet's own hash.

The failures this exists for, each planted by ``verify_physionet``:

* **A login page is not a file.** A web server can answer 200 with HTML for a resource you
  may not read; saved as ``chartevents.csv.gz``, it looks like a download that worked.
* **A short read is silent.** ``http.client`` returns fewer bytes and then EOF when a
  connection drops mid-body. Only comparing against Content-Length notices.
* **A range request can be ignored.** A server may answer 200 and the whole file to a Range
  request; appending that corrupts the file. And a file that changed between two connections
  must not be stitched together, so resumes carry ``If-Range`` with the ETag.
* **Credentials must not follow a redirect** to another host, and never travel in clear text.
* **A manifest is untrusted input.** A ``SHA256SUMS.txt`` line of ``../../.ssh/authorized_keys``
  must not choose where a file lands.

A resume restarts the hash over the bytes already on disk, because hashlib cannot save state.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import urllib3

from backend.ml.physionet.auth import Credentials

CHUNK = 1 << 20
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
SHA_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+)$")


class FetchError(RuntimeError):
    pass


class AuthRejected(FetchError):
    """401: the username or password is wrong."""


class NoAccess(FetchError):
    """403, or a login page: authenticated but not approved for this dataset."""


class IntegrityError(FetchError):
    """The bytes do not hash to what PhysioNet published."""


class SourceChanged(FetchError):
    """The file changed between two connections; its parts must not be joined."""


class DiskFull(FetchError):
    pass


class UnsafePath(FetchError):
    pass


def safe_relpath(path: str) -> str:
    """A manifest path, or a refusal: nothing absolute, nothing climbing, nothing strange."""
    p = path.strip().replace("\\", "/")
    parts = p.split("/")
    if not p or p.startswith("/") or re.match(r"^[A-Za-z]:", p) or any(x in ("", ".", "..") for x in parts) \
            or any(ord(c) < 32 for c in p):
        raise UnsafePath(f"refusing manifest path {path!r}")
    return p


def parse_sums(text: str) -> Dict[str, str]:
    """``SHA256SUMS.txt`` as {relative path: sha256}. A malformed line refuses the whole file."""
    out: Dict[str, str] = {}
    for ln in text.splitlines():
        if not ln.strip():
            continue
        m = SHA_LINE.match(ln.strip())
        if not m:
            raise IntegrityError("SHA256SUMS.txt has a malformed line")
        out[safe_relpath(m.group(2))] = m.group(1).lower()
    if not out:
        raise IntegrityError("SHA256SUMS.txt is empty")
    return out


def require_free(path: str, needed: int, reserve: int = 2 << 30) -> None:
    probe = path
    while not os.path.exists(probe):
        probe = os.path.dirname(probe) or "."
    free = shutil.disk_usage(probe).free
    if free < needed + reserve:
        raise DiskFull(f"{needed / 1e9:.1f} GB needed plus a {reserve / 1e9:.0f} GB reserve; "
                       f"{free / 1e9:.1f} GB free on the volume of {probe}")


def _session(origin: str, creds: Optional[Credentials]):
    """A session from PhysioNet's own client (the ``physionet`` package). PhysioNet reads a login
    only from clients it recognises (wget, the WFDB tools, this one: ``http_auth_allowed`` in
    physionet-build) and serves anyone else as anonymous, which for a credentialed file is a 403
    however right the password. So requests go through its client rather than claiming to be one."""
    from physionet.api.client import PhysioNetClient
    who = (creds.username, creds.password) if creds else (None, None)
    return PhysioNetClient(origin, *who).session


class _Response:
    """The urllib response surface this module reads, over a streamed ``requests`` response."""

    def __init__(self, resp):
        self._resp, self.headers = resp, resp.headers
        resp.raw.enforce_content_length = False    # a short body is RemoteStream's to notice and resume

    def getcode(self) -> int:
        return self._resp.status_code

    def read(self, n: int = -1) -> bytes:
        try:
            return self._resp.raw.read(None if n is None or n < 0 else n, decode_content=True)
        except urllib3.exceptions.HTTPError as err:     # a dropped connection: the callers reconnect on OSError
            raise OSError(str(err)) from err

    def close(self) -> None:
        self._resp.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class Client:
    creds: Optional[Credentials] = None
    timeout: float = 60.0
    retries: int = 8
    backoff: float = 2.0
    sleep: Callable[[float], None] = time.sleep
    max_redirects: int = 5
    _sessions: Dict[Tuple[str, bool], object] = field(default_factory=dict, repr=False)

    def _open_one(self, url: str, headers: Dict[str, str], auth: bool):
        parts = urllib.parse.urlsplit(url)
        creds = self.creds if auth else None
        if creds is not None:
            if parts.scheme != "https" and parts.hostname not in LOOPBACK:
                raise FetchError(f"refusing to send credentials in clear text to {parts.hostname}")
        origin = f"{parts.scheme}://{parts.netloc}"
        key = (origin, creds is not None)
        if key not in self._sessions:             # the login is scoped to this origin by the session itself
            self._sessions[key] = _session(origin, creds)
        resp = self._sessions[key].get(url, headers=headers, allow_redirects=False, stream=True,
                                       timeout=self.timeout)     # redirects are followed by open(), which decides on auth
        return _Response(resp)

    def open(self, url: str, headers: Optional[Dict[str, str]] = None):
        """A response object, after redirects. Credentials stay on the host they were given
        for, and 401 / 403 / a login page raise; retryable statuses are returned to the caller."""
        headers = dict(headers or {})
        origin = urllib.parse.urlsplit(url).hostname
        cur = url
        for _ in range(self.max_redirects + 1):
            resp = self._open_one(cur, headers, urllib.parse.urlsplit(cur).hostname == origin)
            status = resp.getcode()
            if status in (301, 302, 303, 307, 308):
                target = resp.headers.get("Location")
                resp.close()
                if not target:
                    raise FetchError(f"redirect without a location from {cur}")
                cur = urllib.parse.urljoin(cur, target)
                continue
            if status == 401:
                resp.close()
                raise AuthRejected(f"401 from {urllib.parse.urlsplit(cur).hostname}: username or password rejected")
            if status == 403:
                resp.close()
                # PhysioNet answers 403, not 401, to a request it treats as anonymous: it can't tell us why
                raise NoAccess(f"403 for {cur}: no access. Check the stored username and password, then the "
                               f"credentialing status and this project's Data Use Agreement on physionet.org")
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if status in (200, 206) and "text/html" in ctype and not cur.lower().endswith((".html", ".htm")):
                resp.close()
                raise NoAccess(f"got an HTML page instead of {cur}: most likely a login form")
            return resp
        raise FetchError(f"too many redirects from {url}")

    def _delay(self, attempt: int, retry_after: Optional[str] = None) -> None:
        if retry_after and retry_after.isdigit():
            self.sleep(min(float(retry_after), 300.0))
        else:
            self.sleep(min(self.backoff * (2 ** attempt), 120.0) * (0.5 + random.random() / 2))

    def get_text(self, url: str) -> str:
        for attempt in range(self.retries + 1):
            try:
                resp = self.open(url)
                if resp.getcode() == 200:
                    with resp:
                        return resp.read().decode("utf-8")
                status, wait = resp.getcode(), resp.headers.get("Retry-After")
                resp.close()
                if status not in RETRYABLE_STATUS:
                    raise FetchError(f"HTTP {status} for {url}")
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                wait = None
                if attempt == self.retries:
                    raise
            self._delay(attempt, wait)
        raise FetchError(f"gave up on {url}")


class RemoteStream:
    """A readable, hashed view of one remote file that survives dropped connections.

    ``read`` reconnects with ``Range`` + ``If-Range`` when a connection fails or ends short,
    so a consumer (a gunzip, a CSV parser) never sees the interruption and the hash covers
    exactly the bytes it was given. ``offset`` and ``hasher`` let a download resume from disk."""

    def __init__(self, client: Client, url: str, offset: int = 0, hasher=None, etag: Optional[str] = None,
                 on_retry: Optional[Callable[[str], None]] = None):
        self.client, self.url = client, url
        self.offset = offset
        self.hasher = hasher or hashlib.sha256()
        self.etag = etag
        self.length: Optional[int] = None
        self.on_retry = on_retry
        self.reconnects = 0
        self._resp = None
        self._failures = 0

    def readable(self) -> bool:
        return True

    def _connect(self) -> None:
        while True:
            headers = {"Accept-Encoding": "identity"}
            if self.offset:
                headers["Range"] = f"bytes={self.offset}-"
                if self.etag:
                    headers["If-Range"] = self.etag
            try:
                resp = self.client.open(self.url, headers)
            except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
                self._retry(str(err))
                continue
            status = resp.getcode()
            if status == 416:                 # nothing past our offset: the file ended exactly here
                resp.close()
                self._resp, self.length = None, self.offset
                return
            if status in RETRYABLE_STATUS:
                wait = resp.headers.get("Retry-After")
                resp.close()
                self._retry(f"HTTP {status}", wait)
                continue
            if status not in (200, 206):
                resp.close()
                raise FetchError(f"HTTP {status} for {self.url}")
            etag = resp.headers.get("ETag")
            if self.etag and etag and etag != self.etag:
                resp.close()
                raise SourceChanged(f"{self.url} changed during the transfer ({self.etag} -> {etag})")
            self.etag = etag or self.etag
            if status == 206:
                start = re.match(r"bytes (\d+)-\d*/(\d+|\*)", resp.headers.get("Content-Range", ""))
                if not start or int(start.group(1)) != self.offset:
                    resp.close()
                    raise FetchError(f"server answered the wrong range for {self.url}")
                if start.group(2) != "*":
                    self.length = int(start.group(2))
            else:
                size = resp.headers.get("Content-Length")
                self.length = int(size) if size else None
                skipped = 0                   # Range ignored: the bytes we already hold come again
                while skipped < self.offset:
                    got = resp.read(min(CHUNK, self.offset - skipped))
                    if not got:
                        resp.close()
                        self._retry("connection ended while skipping an ignored range")
                        break
                    skipped += len(got)
                else:
                    self._resp = resp
                    return
                continue
            self._resp = resp
            return

    def _retry(self, why: str, retry_after: Optional[str] = None) -> None:
        self._failures += 1
        self.reconnects += 1
        if self._failures > self.client.retries:
            raise FetchError(f"gave up on {self.url} at byte {self.offset} after "
                             f"{self.client.retries} retries: {why}")
        if self.on_retry:
            self.on_retry(why)
        self.client._delay(self._failures - 1, retry_after)

    def read(self, n: int = -1) -> bytes:
        n = n if n and n > 0 else CHUNK
        while True:
            if self._resp is None:
                if self.length is not None and self.offset >= self.length:
                    return b""
                self._connect()
                if self._resp is None:
                    return b""
            try:
                data = self._resp.read(n)
            except (OSError, http.client.HTTPException) as err:
                self._drop()
                self._retry(str(err))
                continue
            if data:
                self.offset += len(data)
                self.hasher.update(data)
                self._failures = 0            # progress: the retry budget is per stall, not per file
                return data
            self._drop()
            if self.length is not None and self.offset < self.length:
                self._retry(f"connection ended at byte {self.offset} of {self.length}")
                continue
            self.length = self.offset
            return b""

    def _drop(self) -> None:
        if self._resp is not None:
            try:
                self._resp.close()
            except OSError:
                pass
            self._resp = None

    def drain(self) -> None:
        while self.read(CHUNK):
            pass

    def hexdigest(self) -> str:
        return self.hasher.hexdigest()

    def close(self) -> None:
        self._drop()


class LocalStream:
    """The same reading surface over a file already on disk."""

    def __init__(self, path: str):
        self._fh = open(path, "rb")
        self.hasher = hashlib.sha256()
        self.offset = 0
        self.reconnects = 0

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        data = self._fh.read(n if n and n > 0 else CHUNK)
        self.offset += len(data)
        self.hasher.update(data)
        return data

    def drain(self) -> None:
        while self.read(CHUNK):
            pass

    def hexdigest(self) -> str:
        return self.hasher.hexdigest()

    def close(self) -> None:
        self._fh.close()


def _hash_prefix(path: str, n: int):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        left = n
        while left > 0:
            got = fh.read(min(CHUNK, left))
            if not got:
                break
            h.update(got)
            left -= len(got)
    return h


def fetch_file(client: Client, url: str, dest: str, sha256: Optional[str], reserve: int = 2 << 30,
               attempts: int = 2) -> Tuple[str, int]:
    """Download ``url`` to ``dest`` atomically: resumable across runs, verified before it
    appears under its real name. Returns (sha256, bytes). A hash mismatch discards the partial
    copy and tries once more, because the fault may be in what we resumed from."""
    if sha256 and os.path.exists(dest):
        if _hash_prefix(dest, os.path.getsize(dest)).hexdigest() == sha256:
            return sha256, os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part, side = dest + ".part", dest + ".part.json"
    for attempt in range(attempts):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        etag = None
        if have and os.path.exists(side):
            with open(side) as fh:
                etag = json.load(fh).get("etag")
        if have and not etag:                 # nothing proves those bytes belong to this version
            os.remove(part)
            have = 0
        stream = RemoteStream(client, url, offset=have, hasher=_hash_prefix(part, have) if have else None, etag=etag)
        try:
            try:
                with open(part, "ab") as out:
                    since_check = 0
                    while True:
                        data = stream.read(CHUNK)
                        if stream.etag and not os.path.exists(side):
                            with open(side, "w") as fh:
                                json.dump({"etag": stream.etag}, fh)
                        if not data:
                            break
                        if since_check == 0 and stream.length is not None:
                            require_free(part, stream.length - stream.offset, reserve)
                        out.write(data)
                        since_check = (since_check + len(data)) % (1 << 30)
                    out.flush()
                    os.fsync(out.fileno())
            except SourceChanged:
                for p in (part, side):        # a new version: what we hold is the old one
                    if os.path.exists(p):
                        os.remove(p)
                if attempt + 1 == attempts:
                    raise
                continue
            digest = stream.hexdigest()
        finally:
            stream.close()
        if sha256 is None or digest == sha256:
            size = os.path.getsize(part)
            os.replace(part, dest)
            if os.path.exists(side):
                os.remove(side)
            return digest, size
        for p in (part, side):
            if os.path.exists(p):
                os.remove(p)
        if attempt + 1 == attempts:
            raise IntegrityError(f"{url}: SHA-256 {digest[:16]}... where PhysioNet published {sha256[:16]}...")
    raise IntegrityError(f"{url}: could not be fetched intact")
