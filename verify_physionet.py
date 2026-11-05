"""Adversarial audit of the PhysioNet data platform. No credentials, no network.

    .venv/bin/python verify_physionet.py        (make verify-physionet, ~1 min)

The real service is impersonated by a local server that misbehaves on request, and everything
downstream runs on the open demo. Every guarantee is attacked:

 1. MANIFEST     a traversal path, an absolute path and a malformed line are refused.
 2. CREDENTIALS  not in repr, not sent in clear text, not sent across a redirect.
 3. ACCESS       401 and 403 stop at once with the right reason; a login page served as 200
                 is not a file.
 4. TRANSPORT    a dropped connection, a short body, 503s, an ignored Range, a corrupted byte
                 and a file that changes mid-transfer: resumed correctly or refused.
 5. STREAMING    a dropped connection in the middle of a gzip stream changes nothing.
 6. LAKE         nothing is published unverified; a stale catalog is refused, a missing column
                 is named.
 7. EXACTNESS    CSV, lake and SQL produce the same grid, byte for byte. The two bugs the SQL
                 check found stay fixed: same-timestamp ties, and no future at hour 335.
 8. CONTRACT     a repointed itemid, a unit change, an orphan stay, a truncated cohort.
 9. EGRESS       a real identifier in any format, shifted dates, identifier columns, a Parquet
                 file under a false name; a clean report passes.
10. PIPELINE     stage order, gates, resume, invalidation, one job at a time, and it stops at
                 pre-registration.
11. CHECKPOINTS  a cached fit predicts bit for bit; edited features never reuse a model.
12. MUTATIONS    each mechanism is broken on purpose and the check is seen to go red.
"""
from __future__ import annotations

import gzip
import hashlib
import http.server
import importlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import threading
import warnings
from contextlib import contextmanager
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from backend.ml.mimic import checkpoint as CK
from backend.ml.mimic import concepts as K
from backend.ml.mimic import contracts as CT
from backend.ml.mimic import extract as E
from backend.ml.mimic import lake as ML
from backend.ml.mimic import parity as PA
from backend.ml.mimic import plan as PLN_MOD
from backend.ml.mimic import progress as PR
from backend.ml.mimic import sources as SRC
from backend.ml.mimic import sqlcheck as SC
from backend.ml.physionet import auth, catalog, fetch, guard, pipeline
from backend.ml.physionet import lake as PL

_results: List = []
_quiet = False


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    if not _quiet:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def _raises(exc, fn) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


# --------------------------------------------------------------------------- #
# A server that misbehaves on request
# --------------------------------------------------------------------------- #
class World:
    """Files plus a behaviour per path. ``seen`` records every request's headers."""

    def __init__(self):
        self.files: Dict[str, bytes] = {}
        self.behaviour: Dict[str, Callable] = {}
        self.seen: List[Dict] = []
        self.etag = '"v1"'


class Handler(http.server.BaseHTTPRequestHandler):
    world: World = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        w = self.world
        w.seen.append({"path": self.path, **{k.lower(): v for k, v in self.headers.items()}})
        fn = w.behaviour.get(self.path)
        if fn is not None and fn(self) is True:
            return
        self.serve()

    def serve(self, ignore_range: bool = False, limit: int = None, ctype: str = "application/gzip", etag: str = None):
        body = self.world.files.get(self.path)
        if body is None:
            self.send_error(404)
            return
        start, end, rng = 0, len(body) - 1, self.headers.get("Range")
        if rng and not ignore_range:
            lo, hi = rng.split("=")[1].split("-")
            start = int(lo)
            end = min(int(hi), len(body) - 1) if hi else len(body) - 1      # "a-b" as well as "a-"
            if start >= len(body):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(body)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
        else:
            self.send_response(200)
        data = body[start:end + 1]
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", etag or self.world.etag)
        self.end_headers()
        self.wfile.write(data if limit is None else data[:limit])
        self.wfile.flush()
        if limit is not None:
            self.close_connection = True


@contextmanager
def server(world: World):
    cls = type("H", (Handler,), {"world": world})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), cls)
    srv.handle_error = lambda *a: None                    # a client hanging up is this test's whole point
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def client(creds=None, retries=6) -> fetch.Client:
    return fetch.Client(creds, retries=retries, timeout=5, sleep=lambda s: None)


CREDS = auth.Credentials("alice", "s3cret-pw", "test")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def once(behaviour: Callable, then=None) -> Callable:
    """Misbehave on the first request only."""
    state = {"n": 0}

    def fn(h):
        state["n"] += 1
        if state["n"] == 1:
            behaviour(h)
            return True
        if then:
            then(h)
            return True
        return False
    return fn


def status(code: int, ctype: str = "text/plain", body: bytes = b"no"):
    def fn(h):
        h.send_response(code)
        h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)
        return True
    return fn


# --------------------------------------------------------------------------- #
def c1_manifest() -> None:
    print("\n1. the manifest is untrusted input")
    good = f"{'a' * 64} hosp/patients.csv.gz\n{'b' * 64} *icu/d_items.csv.gz\n"
    p = fetch.parse_sums(good)
    check("a real manifest parses, with binary-mode stars", p == {"hosp/patients.csv.gz": "a" * 64, "icu/d_items.csv.gz": "b" * 64})
    for bad in ("../../.ssh/authorized_keys", "/etc/passwd", "a/../../b", "C:\\x", "a//b", "a/./b"):
        check(f"refuses {bad!r}", _raises(fetch.UnsafePath, lambda: fetch.parse_sums(f"{'a' * 64} {bad}\n")))
    check("a malformed line refuses the whole file", _raises(fetch.IntegrityError, lambda: fetch.parse_sums(good + "not a hash line\n")))
    check("an empty manifest is refused", _raises(fetch.IntegrityError, lambda: fetch.parse_sums("\n")))


def c2_credentials() -> None:
    print("\n2. credentials")
    fresh = auth.Credentials("alice", "s3cret-pw", "test")   # built here: a mutation replaces the class
    check("the password is not in repr or str", "s3cret" not in repr(fresh) and "s3cret" not in str(fresh) and "s3cret" not in f"{fresh}")
    check("the environment supplies them", auth.resolve({"PHYSIONET_USERNAME": "u", "PHYSIONET_PASSWORD": "p"}, keychain=False).username == "u")
    tmp = tempfile.mkdtemp()
    try:
        rc = os.path.join(tmp, "netrc")
        with open(rc, "w") as fh:
            fh.write("machine physionet.org login bob password pw\n")
        os.chmod(rc, 0o644)
        check("a world-readable netrc is refused", _raises(PermissionError, lambda: auth.resolve({}, rc, keychain=False)))
        os.chmod(rc, 0o600)
        check("a private netrc is read", auth.resolve({}, rc, keychain=False).username == "bob")
    finally:
        shutil.rmtree(tmp)
    check("no credentials sent to a plain-http host", _raises(fetch.FetchError, lambda: client(CREDS, 0)._open_one(
        "http://example.invalid/x", {}, True)))
    w = World()
    w.files["/f"] = b"payload"

    def to_other(port_url):
        def fn(h):
            h.send_response(302)
            h.send_header("Location", port_url + "/f")
            h.send_header("Content-Length", "0")
            h.end_headers()
            return True
        return fn
    with server(w) as url:
        other = url.replace("127.0.0.1", "localhost")      # the same server under another host name
        w.behaviour["/start"] = to_other(other)
        w.behaviour["/same"] = to_other(url)
        with client(CREDS).open(url + "/same") as r:
            r.read()
        auth_same = w.seen[-1].get("authorization")
        check("credentials follow a same-host redirect", bool(auth_same))
        w.seen.clear()
        with client(CREDS).open(url + "/start") as r:
            r.read()
        check("credentials do not follow a redirect to another host",
              w.seen[0].get("authorization") and not w.seen[-1].get("authorization"))
        check("requests go through PhysioNet's own client, the only one it reads a login from",
              all(s.get("user-agent", "").startswith("PhysioNet-Python-Client/") for s in w.seen))


def c3_access() -> None:
    print("\n3. access")
    w = World()
    w.files["/f.csv.gz"] = b"x" * 100
    with server(w) as url:
        w.behaviour["/401"] = status(401)
        w.behaviour["/403"] = status(403, "text/html", b"<html>")
        w.behaviour["/f.csv.gz"] = status(200, "text/html; charset=utf-8", b"<html>log in</html>")
        n = len(w.seen)
        check("401 is a bad password", _raises(fetch.AuthRejected, lambda: client().get_text(url + "/401")))
        check("...and is not retried", len(w.seen) - n == 1, f"{len(w.seen) - n} request")
        check("403 means no access to this dataset", _raises(fetch.NoAccess, lambda: client().get_text(url + "/403")))
        tmp = tempfile.mkdtemp()
        try:
            dest = os.path.join(tmp, "f.csv.gz")
            check("a login page served as 200 is refused",
                  _raises(fetch.NoAccess, lambda: fetch.fetch_file(client(), url + "/f.csv.gz", dest, None)))
            check("...and leaves no file behind", not os.path.exists(dest))
        finally:
            shutil.rmtree(tmp)


def c4b_parallel() -> None:
    print("\n4b. parallel ranges")
    data = os.urandom(3_000_000)
    good = sha(data)
    w = World()
    w.files["/f"] = data
    tmp = tempfile.mkdtemp()
    dest = os.path.join(tmp, "out", "f")
    keep_min = fetch.PART_MIN
    fetch.PART_MIN = 500_000                  # small parts, so a 3 MB file splits like a 3 GB one

    def fresh():
        shutil.rmtree(os.path.join(tmp, "out"), ignore_errors=True)
        w.seen.clear()

    def left():
        d = os.path.join(tmp, "out")
        return sorted(os.listdir(d)) if os.path.isdir(d) else []

    def parts_only(fn):
        """Misbehave on the first request for a part (never the one-byte probe)."""
        state = {"n": 0}

        def b(h):
            if h.headers.get("Range") != "bytes=0-0":
                state["n"] += 1
                if state["n"] == 1:
                    fn(h)
                    return True
            return None
        return b
    try:
        with server(w) as url:
            fresh()
            w.behaviour["/f"] = None
            d, n = fetch.fetch_parallel(client(), url + "/f", dest, good, parts=4)
            ranges = {s.get("range") for s in w.seen if s.get("range") not in (None, "bytes=0-0")}
            check("four ranges, joined into exactly the published bytes", d == good and n == len(data)
                  and open(dest, "rb").read() == data and len(ranges) == 4, f"{sorted(ranges)}")
            check("...every range asked with If-Range", all(s.get("if-range") for s in w.seen if s.get("range") in ranges))
            check("...and nothing left but the file", left() == ["f"])

            fresh()
            w.behaviour["/f"] = parts_only(lambda h: h.serve(limit=100_000))
            d, _ = fetch.fetch_parallel(client(), url + "/f", dest, good, parts=4)
            check("a part dropped mid-way resumes from where it stopped",
                  d == good and open(dest, "rb").read() == data
                  and any(s.get("range", "").split("=")[1].split("-")[0] not in {str(i * len(data) // 4) for i in range(4)}
                          for s in w.seen if s.get("range") not in (None, "bytes=0-0")))

            fresh()

            def shifted(h):                    # a 206 for a different range than the one asked
                if h.headers.get("Range") == "bytes=0-0":
                    return None
                lo, hi = h.headers["Range"].split("=")[1].split("-")
                h.headers.replace_header("Range", f"bytes={int(lo) + 1000}-{hi}")
                h.serve()
                return True
            w.behaviour["/f"] = shifted
            try:
                fetch.fetch_parallel(client(retries=1), url + "/f", dest, good, parts=4)
                why = "accepted"
            except fetch.IntegrityError:
                why = "only the hash noticed"
            except fetch.FetchError as e:
                why = "wrong range" if "wrong range" in str(e) else str(e)[:60]
            check("a part answered with the wrong range is refused for that, before any hashing", why == "wrong range"
                  and left() == [], why)

            fresh()
            w.behaviour["/f"] = lambda h: None if h.headers.get("Range") == "bytes=0-0" else (h.serve(ignore_range=True) or True)
            try:
                fetch.fetch_parallel(client(retries=1), url + "/f", dest, good, parts=4)
                ok = "accepted"
            except fetch.RangeIgnored:
                ok = "named"
            except fetch.FetchError as e:
                ok = str(e)[:60]
            check("a whole file sent for a part is named as ignoring ranges, not written as that part", ok == "named"
                  and left() == [], ok)

            fresh()
            w.behaviour["/f"] = lambda h: h.serve(ignore_range=True) or True
            try:
                fetch.fetch_parallel(client(), url + "/f", dest, good, parts=4)
                ok = False
            except fetch.RangeIgnored:
                ok = True
            check("a server that ignores ranges is named, so the caller can stream", ok and left() == [])

            fresh()
            w.behaviour["/f"] = parts_only(lambda h: h.serve(etag='"v2"'))
            try:
                fetch.fetch_parallel(client(), url + "/f", dest, good, parts=4)
                ok = False
            except fetch.SourceChanged:
                ok = True
            check("a file that changes mid-transfer is refused, and nothing is kept", ok and left() == [])

            fresh()
            w.behaviour["/f"] = None
            try:
                fetch.fetch_parallel(client(), url + "/f", dest, sha(b"another file"), parts=4)
                ok = False
            except fetch.IntegrityError:
                ok = True
            check("joined bytes that miss the published hash are refused, and nothing is kept", ok and left() == [])
    finally:
        fetch.PART_MIN = keep_min
        shutil.rmtree(tmp, ignore_errors=True)


def c4_transport() -> None:
    print("\n4. transport")
    data = os.urandom(5_000_000)
    good = sha(data)
    w = World()
    w.files["/f"] = data
    tmp = tempfile.mkdtemp()
    dest = os.path.join(tmp, "out", "f")

    def fresh():
        shutil.rmtree(os.path.join(tmp, "out"), ignore_errors=True)

    def got(digest, label):
        check(label, digest == good and open(dest, "rb").read() == data)

    try:
        with server(w) as url:
            fresh()
            w.behaviour["/f"] = None
            got(fetch.fetch_file(client(), url + "/f", dest, good)[0], "a clean download")
            n = len(w.seen)
            fetch.fetch_file(client(), url + "/f", dest, good)
            check("a verified file is not fetched again", len(w.seen) == n)

            fresh()
            w.behaviour["/f"] = once(lambda h: h.serve(limit=1_000_000))
            d = fetch.fetch_file(client(), url + "/f", dest, good)[0]
            got(d, "a connection dropped at 1 MB resumes")
            check("...with a range and an If-Range", any(s.get("range") == "bytes=1000000-" and s.get("if-range") for s in w.seen))

            fresh()
            w.behaviour["/f"] = once(status(503))
            got(fetch.fetch_file(client(), url + "/f", dest, good)[0], "a 503 is retried")

            fresh()
            w.behaviour["/f"] = once(lambda h: h.serve(limit=1_000_000), lambda h: h.serve(ignore_range=True))
            n = len(w.seen)
            got(fetch.fetch_file(client(), url + "/f", dest, good)[0], "a server that ignores Range cannot corrupt the file")
            # the hash would rescue a corrupted join on a second attempt, so assert the mechanism itself
            check("...because the repeated bytes are skipped, with no second attempt", len(w.seen) - n == 2, f"{len(w.seen) - n} requests")

            fresh()
            w.behaviour["/f"] = once(lambda h: h.serve(limit=1_000_000), lambda h: h.serve(etag='"v2"'))
            d = fetch.fetch_file(client(), url + "/f", dest, good)[0]
            got(d, "a file that changed mid-transfer restarts from nothing, never joins two versions")
            w.behaviour["/f"] = once(lambda h: h.serve(limit=1_000_000), lambda h: h.serve(etag='"v2"'))
            stream = fetch.RemoteStream(client(), url + "/f")

            def read_all():
                while stream.read(fetch.CHUNK):
                    pass
            check("...the stream itself refuses to continue across versions", _raises(fetch.SourceChanged, read_all))
            stream.close()

            fresh()
            os.makedirs(os.path.dirname(dest))
            with open(dest + ".part", "wb") as fh:
                fh.write(b"\0" * 1000)         # a partial with no record of its version
            w.behaviour["/f"] = None
            n = len(w.seen)
            got(fetch.fetch_file(client(), url + "/f", dest, good)[0], "a partial file of unknown origin is discarded")
            check("...before it is ever resumed from", not any("range" in r for r in w.seen[n:]))

            fresh()
            os.makedirs(os.path.dirname(dest))
            with open(dest + ".part", "wb") as fh:
                fh.write(b"\0" * 1000)
            with open(dest + ".part.json", "w") as fh:
                json.dump({"etag": w.etag}, fh)
            got(fetch.fetch_file(client(), url + "/f", dest, good)[0], "a corrupted partial is caught by the hash and refetched")

            fresh()
            w.files["/bad"] = data[:-1] + bytes([data[-1] ^ 1])
            check("one flipped bit is refused", _raises(fetch.IntegrityError, lambda: fetch.fetch_file(client(), url + "/bad", dest, good)))
            check("...and nothing appears under the real name", not os.path.exists(dest))

            fresh()
            w.behaviour["/f"] = status(503)
            check("a server that never recovers ends in an error, not a hang",
                  _raises(fetch.FetchError, lambda: fetch.fetch_file(client(retries=2), url + "/f", dest, good)))
            w.behaviour["/f"] = None
            check("a full disk stops before writing", _raises(fetch.DiskFull, lambda: fetch.require_free(tmp, 1 << 62)))
    finally:
        shutil.rmtree(tmp)


SPEC = PL.TableSpec("icu", "events", (("stay_id", "int64"), ("charttime", "timestamp"), ("itemid", "int64"),
                                      ("value", "string"), ("valuenum", "float64")),
                    key="stay_id", buckets=4, items=frozenset({1, 2}), text_items=frozenset({2}))


def _csv_gz(n: int = 40_000) -> bytes:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"stay_id": rng.integers(1, 50, n), "charttime": "2150-01-01 00:00:00", "itemid": rng.integers(1, 4, n),
                       "value": "abc", "valuenum": rng.random(n), "junk": "dropme"})
    return gzip.compress(df.to_csv(index=False).encode()), df


def c5_streaming() -> None:
    print("\n5. streaming through gunzip")
    body, df = _csv_gz()
    w = World()
    w.files["/t.csv.gz"] = body
    tmp = tempfile.mkdtemp()
    try:
        with server(w) as url:
            clean = PL.ingest(SPEC, fetch.RemoteStream(client(), url + "/t.csv.gz"), tmp, sha(body))
            import pyarrow.parquet as pq
            d = PL.table_dir(tmp, SPEC)
            ref = sorted(np.round(pq.read_table(os.path.join(d, f)).column("valuenum").to_numpy(), 9).tolist() for f in clean["files"])
            ref_all = sorted(x for r in ref for x in r)
            w.behaviour["/t.csv.gz"] = once(lambda h: h.serve(limit=len(body) // 2))
            stream = fetch.RemoteStream(client(), url + "/t.csv.gz")
            m = PL.ingest(SPEC, stream, tmp, sha(body))
            check("a connection dropped mid-gzip is invisible to the parser", m["rows_kept"] == clean["rows_kept"] and stream.reconnects == 1,
                  f"{stream.reconnects} reconnect, {m['rows_kept']:,} rows")
            again = sorted(x for f in m["files"] for x in np.round(pq.read_table(os.path.join(d, f)).column("valuenum").to_numpy(), 9).tolist())
            check("...and the rows are the same rows", again == ref_all)
            check("the item filter kept exactly the catalog's items", m["rows_kept"] == int(df.itemid.isin([1, 2]).sum()) and m["rows_in"] == len(df))
            t = pq.read_table(os.path.join(d, m["files"][0])).to_pandas()
            check("text survives only where it's needed", t[t.itemid == 2].value.eq("abc").all() and t[t.itemid == 1].value.isna().all())
            check("unused columns are gone", "junk" not in t.columns)
            check("each bucket holds only its own stays", all(
                (pq.read_table(os.path.join(d, f)).column("stay_id").to_numpy() % 4 == int(f[1:5])).all() for f in m["files"]))
    finally:
        shutil.rmtree(tmp)


def c6_lake(lake_root: str) -> None:
    print("\n6. the lake publishes nothing unverified")
    body, _ = _csv_gz(5000)
    tmp = tempfile.mkdtemp()
    w = World()
    w.files["/t.csv.gz"] = body
    try:
        with server(w) as url:
            check("a wrong hash publishes nothing", _raises(fetch.IntegrityError, lambda: PL.ingest(
                SPEC, fetch.RemoteStream(client(), url + "/t.csv.gz"), tmp, "0" * 64)) and PL.read_manifest(tmp, SPEC) is None)
            check("...and leaves no temporary directory", not any(".tmp-" in d for d, _, _ in os.walk(tmp)))
            trunc = body[:len(body) // 2]
            w.files["/t.csv.gz"] = trunc
            check("a truncated gzip with no hash to catch it is still refused", _raises(fetch.IntegrityError, lambda: PL.ingest(
                SPEC, fetch.RemoteStream(client(), url + "/t.csv.gz"), tmp, None)) and PL.read_manifest(tmp, SPEC) is None)
            w.files["/t.csv.gz"] = gzip.compress(b"stay_id,charttime\n1,2150-01-01 00:00:00\n")
            check("a table that lost a promised column is refused", _raises(fetch.IntegrityError, lambda: PL.ingest(
                SPEC, fetch.RemoteStream(client(), url + "/t.csv.gz"), tmp, None)) and PL.read_manifest(tmp, SPEC) is None)
            w.files["/t.csv.gz"] = body
            PL.ingest(SPEC, fetch.RemoteStream(client(), url + "/t.csv.gz"), tmp, sha(body))
            check("a table is current for its hash and spec", PL.is_current(tmp, SPEC, sha(body)))
            check("...and stale for another source", not PL.is_current(tmp, SPEC, "0" * 64))
            changed = PL.TableSpec(SPEC.module, SPEC.table, SPEC.columns, SPEC.key, SPEC.buckets, frozenset({1, 2, 3}), SPEC.text_items)
            check("a catalog change makes the old table stale", not PL.is_current(tmp, changed, sha(body)))
    finally:
        shutil.rmtree(tmp)
    src = ML.LakeSource(lake_root)
    check("asking for a column the lake never kept is an error, not a NaN",
          _raises(ML.LakeStale, lambda: src._read("icu", "chartevents", ["stay_id", "warning"])))
    other = tempfile.mkdtemp()
    try:
        shutil.copytree(lake_root, os.path.join(other, "l"))
        os.remove(os.path.join(other, "l", "icu", "chartevents", PL.MANIFEST))
        check("a lake missing a table refuses to serve", _raises(ML.LakeStale, lambda: ML.LakeSource(os.path.join(other, "l"))))
    finally:
        shutil.rmtree(other)
    covered = {s.table: set(s.names) for s in ML.project()}
    asked: Dict[str, set] = {}
    real = SRC.CsvSource._read

    def spy(self, module, table, usecols, keep=None):
        asked.setdefault(table, set()).update(usecols)
        return real(self, module, table, usecols, keep)
    SRC.CsvSource._read = spy
    try:
        E.build_store(SRC.CsvSource(E.DEMO_ROOT), verbose=False)
    finally:
        SRC.CsvSource._read = real
    gaps = {t: sorted(c - covered.get(t, set())) for t, c in asked.items() if c - covered.get(t, set())}
    check("the projection covers every table and column the reference reads", not gaps and len(asked) >= 10,
          f"{len(asked)} tables, {sum(len(v) for v in asked.values())} columns" + (f"; GAPS {gaps}" if gaps else ""))


def _same(a, b) -> bool:
    return a.X.tobytes() == b.X.tobytes() and (a.y == b.y).all() and a.onsets.equals(b.onsets) and a.meta.equals(b.meta) \
        and a.columns == b.columns and all(np.array_equal(a.sofa[k], b.sofa[k], equal_nan=True) for k in a.sofa)


def c7_exact(lake_root: str, ref_store) -> None:
    print("\n7. three paths, one grid")
    st = E.build_store(ML.LakeSource(lake_root), verbose=False)
    check("lake == CSV, byte for byte (grid, labels, onsets, SOFA, metadata)", _same(ref_store, st), f"{ref_store.X.shape}")
    q = E.build_store(SC.EmulatedBigQuery(E.DEMO_ROOT), verbose=False)
    check("BigQuery SQL (emulated) == CSV, byte for byte", _same(ref_store, q))
    keep = SRC.BigQuerySource.FILTER_MAX
    SRC.BigQuerySource.FILTER_MAX = 0                     # the full-cohort branch of every query
    try:
        q2 = E.build_store(SC.EmulatedBigQuery(E.DEMO_ROOT), verbose=False)
    finally:
        SRC.BigQuerySource.FILTER_MAX = keep
    check("...and through the full-cohort branch of the SQL", _same(ref_store, q2))
    stays = E.cohort(SRC.CsvSource(E.DEMO_ROOT))[["stay_id", "hadm_id", "intime", "outtime"]]
    rep = SC.check(E.DEMO_ROOT, stays)
    check("the SQL's every table agrees exactly, not merely at 0.99", rep["verdict"]["passed"] and all(
        v["agreement"] == 1.0 for k, v in rep.items() if k != "verdict"))

    t0 = pd.Timestamp("2150-01-01")
    stays1 = pd.DataFrame({"stay_id": [1], "intime": [t0]})
    ev = pd.DataFrame({"stay_id": [1, 1, 1], "name": ["SBP"] * 3, "itemid": [220179, 220050, 220179],
                       "charttime": [t0 + pd.Timedelta(minutes=30)] * 3, "value": [120.0, 95.0, 130.0]})
    out = SRC._bin_last(ev, stays1, "charttime", lo=0)
    check("a same-time tie takes the lowest itemid (the arterial line)", out.value.iloc[0] == 95.0)
    both = pd.concat([ev.iloc[[0]], ev.iloc[[2]]])
    check("...then the highest value, in any file order", SRC._bin_last(both, stays1, "charttime", lo=0).value.iloc[0] == 130.0
          and SRC._bin_last(both.iloc[::-1], stays1, "charttime", lo=0).value.iloc[0] == 130.0)
    iv = pd.DataFrame({"stay_id": [1, 1], "name": ["Norepinephrine"] * 2, "value": [0.1, 0.9],
                       "starttime": [t0, t0 + pd.Timedelta(hours=K.MAX_HOURS + 40)],
                       "endtime": [t0 + pd.Timedelta(minutes=90), t0 + pd.Timedelta(hours=K.MAX_HOURS + 50)]})
    out = SRC._expand_intervals(iv, stays1)
    check("an infusion past the grid writes nothing into its last hour", (out.hr.max() == 1) and (out.value.max() == 0.1), f"hours {out.hr.tolist()}")
    check("BigQuery's dataset names are the published ones", SRC.BigQuerySource("p").hosp.endswith("mimiciv_v3_1_hosp")
          and SRC.BigQuerySource("p").icu.endswith("mimiciv_v3_1_icu"))
    check("the catalog agrees", set(catalog.get("mimiciv").bigquery) == {SRC.BigQuerySource("p").hosp.split(".")[1], SRC.BigQuerySource("p").icu.split(".")[1]})


def _edit(lake_root: str, table: str, fn) -> str:
    """A copy of the lake with one table rewritten by ``fn(df) -> df``; manifests kept."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tempfile.mkdtemp()
    out = os.path.join(d, "l")
    shutil.copytree(lake_root, out)
    spec = {s.table: s for s in ML.project()}[table]
    td = PL.table_dir(out, spec)
    for f in PL.read_manifest(out, spec)["files"]:
        p = os.path.join(td, f)
        sch = pq.read_schema(p)
        tb = pa.Table.from_pandas(fn(pq.read_table(p).to_pandas(coerce_temporal_nanoseconds=True)), schema=sch, preserve_index=False)
        pq.write_table(tb, p, compression="zstd")
    return out


def c8_contract(lake_root: str) -> None:
    print("\n8. the data contract")
    base = ML.LakeSource(lake_root)
    rep = CT.check(base, base, min_stays=100)
    check("the demo passes its own contract", rep["passed"], f"{len(rep['checks'])} checks; failed {rep['failed']}")
    check("a clean contract reports no small cell as a number", all(
        not any(ch in c["detail"] for ch in ("subject_id=", "stay_id=")) for c in rep["checks"]))

    def attacked(table, fn, name, min_stays=100):
        path = _edit(lake_root, table, fn)
        try:
            r = CT.check(ML.LakeSource(path), base, min_stays=min_stays)
            check(f"catches {name}", (not r["passed"]) and bool(r["failed"]), f"failed {r['failed'][:2]}")
        finally:
            shutil.rmtree(os.path.dirname(path))
    attacked("d_items", lambda d: d.assign(label=d.label.where(d.itemid != 220045, "Something Else")), "a repointed itemid")
    attacked("chartevents", lambda d: d.assign(valuenum=d.valuenum.where(d.itemid != 220045, d.valuenum * 9)), "a unit change in heart rate")
    # Base excess sits near zero: the first full run's median moved -1 -> 0, which a ratio calls a
    # unit change. The demo has too few base-excess rows in the usual 8-bucket sample (70 < 100) to
    # compare at all, so these two read every bucket: they test the rule, not the demo's sample size.
    sample = CT.SAMPLE_BUCKETS
    CT.SAMPLE_BUCKETS = 64
    attacked("labevents", lambda d: d.assign(valuenum=d.valuenum.where(d.itemid != 50802, d.valuenum + 20)),
             "a 20 mEq/L shift in base excess (a signed lab)")
    path = _edit(lake_root, "labevents", lambda d: d.assign(valuenum=d.valuenum.where(
        d.itemid != 50802, d.valuenum - d.valuenum[d.itemid == 50802].median())))
    try:
        r = CT.check(ML.LakeSource(path), base, min_stays=100)
        drift = next(c for c in r["checks"] if c["check"] == "no item moved like a unit change")
        check("a signed lab re-centred near zero is sampling noise, not a unit change", drift["ok"], drift["detail"][:80])
    finally:
        shutil.rmtree(os.path.dirname(path))
        CT.SAMPLE_BUCKETS = sample
    attacked("icustays", lambda d: d.assign(subject_id=d.subject_id.where(d.index != 0, 1)), "a stay with no patient")
    attacked("icustays", lambda d: d.assign(hadm_id=d.hadm_id.where(d.index != 0, 1)), "a stay with no admission")
    attacked("icustays", lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True), "a duplicated stay")
    attacked("icustays", lambda d: d.assign(outtime=d.intime - pd.Timedelta(hours=1)), "stays that end before they begin")
    attacked("icustays", lambda d: d.assign(intime=d.intime - pd.Timedelta(days=365 * 200), outtime=d.outtime - pd.Timedelta(days=365 * 200)),
             "real calendar dates where shifted ones belong")
    attacked("chartevents", lambda d: d.assign(stay_id=d.stay_id + 100000000), "chart rows of unknown stays")
    attacked("chartevents", lambda d: d.iloc[:0], "an empty chart table")
    # Across releases: PhysioNet revised 2 of the demo's stays between 2.2 and 3.1. A revised stay
    # is excluded and reported; a changed value under an unchanged row count is not; too many fail.
    stays = E.cohort(SRC.CsvSource(E.DEMO_ROOT))[["stay_id", "hadm_id", "intime", "outtime"]]
    inp = base._read("icu", "inputevents", ["stay_id", "itemid", "rate"])
    vaso = inp[inp.itemid.isin([i for c in K.VASO for i in c.itemids]) & inp.rate.notna()]
    one = int(vaso.stay_id.value_counts().index[0])
    many = list(stays.stay_id.astype(np.int64)[:int(0.08 * len(stays)) + 1])

    def across(fn):
        path = _edit(lake_root, "inputevents", fn)
        try:
            return PA.parity(base, ML.LakeSource(path), stays, agreement=1.0, across_releases=True), \
                PA.parity(base, ML.LakeSource(path), stays, agreement=1.0)
        finally:
            shutil.rmtree(os.path.dirname(path))
    rel, same = across(lambda d: d[d.stay_id != one])
    check("a stay the release revised is left out and reported, and only across releases",
          rel["verdict"]["passed"] and rel["revised"]["stays"] == 1 and not same["verdict"]["passed"],
          f"revised {rel['revised']['stays']} of {rel['revised']['of']}; same-release failing {same['verdict']['failing']}")
    rel, _ = across(lambda d: d.assign(rate=d.rate.where(d.stay_id != one, d.rate * 2)))
    check("...but a changed value under the same row count still fails", not rel["verdict"]["passed"]
          and rel["revised"]["stays"] == 0 and "vasopressors" in rel["verdict"]["failing"], f"failing {rel['verdict']['failing']}")
    rel, _ = across(lambda d: d[~d.stay_id.isin(many)])
    check(f"...and more than {PA.MAX_REVISED:.0%} of stays revised is refused", not rel["verdict"]["passed"]
          and "revised" in rel["verdict"]["failing"], f"revised {rel['revised']['stays']} of {rel['revised']['of']}")

    r = CT.check(base, base, min_stays=90_000)
    check("catches a demo passed off as the full database", "cohort size" in r["failed"])
    r = CT.check(base, None, min_stays=0)
    check("no baseline is a warning, never a silent pass", any(c["severity"] == "warn" and not c["ok"] for c in r["checks"]))


def c9_egress(lake_root: str) -> None:
    print("\n9. egress")
    src = ML.LakeSource(lake_root)
    ids = list(src.icustays().stay_id)
    can = guard.Canary.build(ids + [12345678], open_ids=ids[:1])
    real, excl = "12345678", str(ids[0])
    tmp = tempfile.mkdtemp()
    try:
        cp = os.path.join(tmp, "canary.json")
        can.save(cp)
        check("the canary file is owner-only and holds no identifier", oct(os.stat(cp).st_mode & 0o777) == "0o600" and real not in open(cp).read())
        check("the canary reloads", guard.Canary.load(cp).hit(real))

        def blocked(name, text, path="artifacts/mimic/r.json", c=can):
            return any(f.severity == "block" for f in guard.scan_bytes(path, text.encode(), c))
        for form, text in (("a JSON value", f'{{"x": {real}}}'), ("a string", f'{{"who": "{real}"}}'), ("a CSV cell", f"a,b\n1,{real}\n"),
                           ("a log line", f"fit stay {real} failed"), ("a list", f"[1, {real}, 3]")):
            check(f"a real identifier as {form} is blocked", blocked(form, text, "x.log" if "log" in form else "artifacts/mimic/r.json"))
        check("the finding never repeats the identifier", real not in guard.scan_bytes("x.json", f'{{"x": {real}}}'.encode(), can)[0].text())
        check("an open (demo) identifier passes", not blocked("open", f'{{"x": {excl}}}'))
        check("a float with those digits passes", not blocked("float", '{"x": 0.12345678}'))
        check("shifted dates are blocked, canary or no canary", blocked("date", '{"t": "2153-04-11"}', c=None))
        check("a modern date passes", not blocked("date", '{"t": "2026-11-01"}', c=None))
        check("a synthetic fixture in source code passes", not blocked("date", 't0 = "2150-01-01"', "verify.py", None))
        check("...but a real identifier in source code does not", blocked("id", f"stay = {real}", "verify.py"))
        check("identifier columns are blocked", blocked("cols", '{"stay_id": 1}', c=None))
        for path in ("a/b.parquet", "x/hosp/labevents.csv.gz", "cache/lake_store.npz", "cache/lake_meta.pkl"):
            check(f"{path} is blocked by its format", blocked("fmt", "x", path, None))
        check("Parquet under a false name is blocked", any(f.severity == "block" for f in guard.scan_bytes("r.json", b"PAR1xxxx", None)))
        check("a huge 'report' is blocked", any(f.severity == "block" for f in guard.scan_bytes("r.json", b"1" * (guard.TEXT_MAX_BYTES + 1), None)))
        small = guard.scan_bytes("r.json", b'{"n": 4, "auroc": 0.8}', None)
        check("a small cell warns but does not block", [f.severity for f in small] == ["warn"])
        rep = os.path.join(E.config.ARTIFACTS_DIR, "mimic", "demo_report.json")
        if os.path.exists(rep):
            found = [f for f in guard.scan_paths([rep], can) if f.severity == "block"]
            check("the harness's real report format passes clean", not found, "; ".join(f.text() for f in found[:2]))
        inside = os.path.join(tmp, "lakey", "icu", "x.dat")
        os.makedirs(os.path.dirname(inside))
        open(inside, "w").write("hello")
        check("anything inside the lake is blocked", any(f.severity == "block" for f in guard.scan_paths([inside], None, os.path.join(tmp, "lakey"))))
        # the hook, in a throwaway repository
        import subprocess
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        hook = guard.install_hook(repo)
        first = open(hook).read()
        check("the hook installs", guard.HOOK_MARK in first and os.access(hook, os.X_OK))
        guard.install_hook(repo)
        check("...once", open(hook).read() == first)
        with open(hook, "w") as fh:
            fh.write("#!/bin/sh\necho mine\n")
        guard.install_hook(repo)
        body = open(hook).read()
        check("...and keeps a hook that was already there", "echo mine" in body and guard.HOOK_MARK in body)
        with open(os.path.join(repo, "r.json"), "w") as fh:
            fh.write(f'{{"x": {real}}}')
        subprocess.run(["git", "add", "r.json"], cwd=repo, check=True)
        found = guard.scan_staged(repo, can)
        check("the staged bytes are what gets scanned", any(f.severity == "block" for f in found))
        with open(os.path.join(repo, "r.json"), "w") as fh:
            fh.write('{"x": 1}')                 # clean in the working copy, dirty in the index
        check("...not the working copy", any(f.severity == "block" for f in guard.scan_staged(repo, can)))
        # posture
        check("a lake inside the repo fails the doctor", not guard.doctor(os.path.join(E.config.REPO_ROOT, "data", "x"), creds_needed=False)["ok"])
        d = guard.doctor(os.path.expanduser("~/Documents/physionet-lake"), creds_needed=False)
        check("a lake in ~/Documents fails the doctor", "lake outside Desktop and Documents" in d["failing"])
        d = guard.doctor(os.path.expanduser("~/Dropbox/lake"), creds_needed=False)
        check("a lake in Dropbox fails the doctor", "lake outside synced folders" in d["failing"])
        open_lake = os.path.join(tmp, "open")
        os.makedirs(open_lake)
        os.chmod(open_lake, 0o755)
        check("a world-readable lake fails the doctor", "lake readable only by its owner" in guard.doctor(open_lake, creds_needed=False)["failing"])
        guard.init_lake(open_lake)
        check("init_lake closes it", "lake readable only by its owner" not in guard.doctor(open_lake, creds_needed=False)["failing"])
    finally:
        shutil.rmtree(tmp)


def _pipeline_world() -> World:
    w = World()
    lines = []
    for spec in ML.project():
        with open(os.path.join(E.DEMO_ROOT, spec.source), "rb") as fh:
            b = fh.read()
        w.files["/" + spec.source] = b
        lines.append(f"{sha(b)} {spec.source}")
    w.files["/SHA256SUMS.txt"] = ("\n".join(lines) + "\n").encode()
    return w


def c10_pipeline() -> None:
    print("\n10. the pipeline")
    w = _pipeline_world()
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="physionet-verify"))
    logs: List[str] = []
    calls: List[str] = []
    ids_in_report = {"value": False}
    try:
        with server(w) as url:
            def planted_runner(ctx, name, plan_only):
                calls.append(name)
                os.makedirs(ctx.reports, exist_ok=True)
                t = pipeline.tag(ctx)
                if plan_only:
                    json.dump({"chosen": "test"}, open(os.path.join(ctx.reports, f"{t}_plan.json"), "w"))
                else:
                    body = {"auroc": 0.8}
                    if ids_in_report["value"]:
                        sid = int(ML.LakeSource(ctx.lake).icustays().stay_id.iloc[0])
                        body["example_stay"] = sid
                    json.dump(body, open(os.path.join(ctx.reports, f"{t}_report.json"), "w"))
            repo = os.path.join(tmp, "repo")
            os.makedirs(os.path.join(repo, "artifacts", "mimic"))
            import subprocess
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

            def ctx(**k):
                base = dict(dataset=catalog.get("mimic-iv-demo"), root=os.path.join(tmp, "lake"), base_url=url, client=client(),
                            runner=planted_runner, repo=repo, log=logs.append, demo_root=E.DEMO_ROOT)
                base.update(k)
                return pipeline.Ctx(**base)
            held = None
            try:
                pipeline.up(ctx())
            except pipeline.Hold as err:
                held = str(err)
            check("it runs end to end from an empty disk", held is not None and calls == ["plan", "dev"], f"runner saw {calls}")
            check("it stops at pre-registration, lock unread", held is not None and "lock has not been read" in held)
            released = [f for f in os.listdir(os.path.join(repo, "artifacts", "mimic"))]
            check("the clean report was released", len(released) == 1 and released[0].endswith("_report.json"))
            names = [s["stage"] for s in pipeline.status(ctx())]
            check("stages ran in order", names == ["doctor", "access", "capacity", "ingest", "contract", "canary", "parity", "plan", "dev_run", "release"])
            check("every stage is done", all(s["status"] == "done" for s in pipeline.status(ctx())))

            before = (len(w.seen), list(calls))
            logs.clear()
            try:
                pipeline.up(ctx())
            except pipeline.Hold:
                pass
            table_gets = [s for s in w.seen[before[0]:] if s["path"].endswith(".csv.gz")]
            check("a re-run repeats no work", not table_gets and calls == before[1] and all(l.startswith("=") for l in logs),
                  f"{len(table_gets)} table downloads")
            from backend.ml.mimic import trust as TR
            jr = TR.Chain(os.path.join(tmp, "lake", "journal.jsonl"))
            check("the journal is an intact hash chain", jr.verify()[0] and jr.verify()[1] >= 10, f"{jr.verify()[1]} entries")

            # a dropped table: only that table comes back
            state = os.path.join(tmp, "lake", "mimic-iv-demo-2.2", "lake", "hosp", "services")
            shutil.rmtree(state)
            n = len(w.seen)
            try:
                pipeline.up(ctx())
            except pipeline.Hold:
                pass
            got = {s["path"] for s in w.seen[n:] if s["path"].endswith(".csv.gz")}
            check("a lost table is restored, and only it", got == {"/hosp/services.csv.gz"}, str(sorted(got)))

            # the published hashes change: refused, nothing overwritten
            keep = w.files["/SHA256SUMS.txt"]
            # a new file only: every table the pipeline reads is still there, so only the pin can object
            w.files["/SHA256SUMS.txt"] = keep + f"{'c' * 64} hosp/new_table.csv.gz\n".encode()
            st = json.load(open(ctx().state_path))
            st.pop("access")
            json.dump(st, open(ctx().state_path, "w"))
            check("a changed release is refused until a person accepts it", _raises(pipeline.Refused, lambda: pipeline.up(ctx())))
            w.files["/SHA256SUMS.txt"] = keep

            # a broken table in the source: the hash gate holds
            root2 = os.path.join(tmp, "lake2")
            real = w.files["/icu/outputevents.csv.gz"]
            w.files["/icu/outputevents.csv.gz"] = real[:-1] + bytes([real[-1] ^ 1])
            check("a corrupted table stops the pipeline at ingest",
                  _raises((fetch.IntegrityError, pipeline.Refused), lambda: pipeline.up(ctx(root=root2))))
            statuses = {s["stage"]: s["status"] for s in pipeline.status(ctx(root=root2))}
            check("...and nothing after it ran", statuses["ingest"] == "refused" and statuses["contract"] == "")
            w.files["/icu/outputevents.csv.gz"] = real

            # the egress gate
            ids_in_report["value"] = True
            shutil.rmtree(os.path.join(repo, "artifacts", "mimic"))
            os.makedirs(os.path.join(repo, "artifacts", "mimic"))
            can_ctx = ctx(root=os.path.join(tmp, "lake3"), dataset=catalog.get("mimic-iv-demo"))
            check("a report carrying a real identifier does not leave the lake", _raises(pipeline.Refused, lambda: pipeline.up(can_ctx)))
            check("...and nothing was released", not os.listdir(os.path.join(repo, "artifacts", "mimic")))
            ids_in_report["value"] = False

            # capacity
            big = ctx(root=os.path.join(tmp, "lake4"), dataset=catalog.get("mimiciv"))
            real_free = pipeline._free_gb
            pipeline._free_gb = lambda p: 1.0
            try:
                check("a disk too small refuses before a byte is downloaded",
                      _raises(pipeline.Refused, lambda: pipeline._stage_capacity(big)))
            finally:
                pipeline._free_gb = real_free
            with pipeline._Lock(os.path.join(tmp, "lake")):
                check("one heavy job at a time", _raises(pipeline.Refused, lambda: pipeline.up(ctx())))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    src = inspect.getsource(pipeline)
    check("no stage touches the lock", "lock_scores" not in src and "confirm(" not in src.replace("trust.confirm", ""))


def c11_checkpoints(store) -> None:
    print("\n11. checkpoints")
    from backend.ml.genesis import learners as L
    from backend.ml.mimic import ledger as LG
    from backend.ml.mimic import twin as TW
    led = LG.Ledger.build(store)
    rows = np.flatnonzero(led.folds[store.seg] >= 0)
    cols = led.columns("P")[:80]
    y = led.y.astype(float)
    tmp = tempfile.mkdtemp()
    params = {"learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 20}
    kw = dict(cols=cols, params=params, early_stop=10, rounds=40)
    probe = rows[:500]
    try:
        with CK.active(tmp, "salt") as memo:
            a = L.fit_lgb(led.F, rows, y[rows], store.seg[rows], **kw)
            pa_ = a.predict(led.F, probe, raw=True)
            check("the first fit is a miss", memo.misses == 1 and memo.hits == 0)
            b = L.fit_lgb(led.F, rows, y[rows], store.seg[rows], **kw)
            check("the same fit is a hit", memo.hits == 1)
            check("a cached model predicts bit for bit", np.array_equal(pa_, b.predict(led.F, probe, raw=True)))
            L.fit_lgb(led.F, rows, y[rows], store.seg[rows], **{**kw, "seed": 1})
            check("another seed refits", memo.misses == 2)
            L.fit_lgb(led.F, rows, 1 - y[rows], store.seg[rows], **kw)
            check("another target refits", memo.misses == 3)
            L.fit_lgb(led.F, rows[10:], y[rows[10:]], store.seg[rows[10:]], **kw)
            check("other rows refit", memo.misses == 4)
            F2 = np.array(led.F, copy=True)
            F2[:, cols[0]] += 1.0
            L.fit_lgb(F2, rows, y[rows], store.seg[rows], **kw)
            check("edited features refit (a counterfactual world never reuses a model)", memo.misses == 5)
        with CK.active(tmp, "other-code") as memo:
            L.fit_lgb(led.F, rows, y[rows], store.seg[rows], **kw)
            check("changed harness code refits", memo.misses == 1 and memo.hits == 0)
        with CK.active(tmp, "salt") as memo:
            for f in os.listdir(tmp):            # a crash mid-write leaves garbage: a miss, not a crash
                open(os.path.join(tmp, f), "wb").write(b"torn")
            L.fit_lgb(led.F, rows, y[rows], store.seg[rows], **kw)
            check("a torn checkpoint refits instead of crashing", memo.misses == 1)
        check("the patch is undone afterwards", L.fit_lgb.__module__ == "backend.ml.genesis.learners")

        T = TW.process_targets(led)
        stays = np.arange(store.n_patients)
        tr, te = stays[(led.folds >= 0) & (led.folds != 0)], stays[led.folds == 0]
        cfg = dict(epochs=1, d_model=16, min_steps=4, seed=0)
        with CK.active(tmp, "tw") as memo:
            t1 = TW.ProcessTwin(TW.TwinConfig(**cfg)).fit(led, T, tr)
            p1 = t1.predict(led, te, ("s3",))["s3"]
            t2 = TW.ProcessTwin(TW.TwinConfig(**cfg)).fit(led, T, tr)
            check("a cached twin is a hit", memo.hits == 1 and memo.failures == 0, f"{memo.failures} not written")
            check("...and predicts bit for bit", np.array_equal(p1, t2.predict(led, te, ("s3",))["s3"], equal_nan=True))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def c12_progress(store) -> None:
    print("\n12. progress journal and calibration")
    import json as _json
    from backend.ml.genesis import learners as L
    from backend.ml.mimic import ledger as LG
    from backend.ml.mimic import plan as PLN
    from backend.ml.mimic import progress as PR
    from backend.ml.mimic.run import SCALES
    from backend.ml.physionet.__main__ import progress_line
    led = LG.Ledger.build(store)
    rows = np.flatnonzero(led.folds[store.seg] >= 0)
    cols = led.columns("P")[:60]
    y = led.y.astype(float)
    kw = dict(cols=cols, params={"learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 20}, early_stop=10, rounds=30)
    tmp = tempfile.mkdtemp()
    try:
        with CK.active(tmp, "salt") as memo:
            L.fit_lgb(led.F, rows, y[rows], store.seg[rows], **kw)
            L.fit_lgb(led.F, rows, y[rows], store.seg[rows], **kw)
            L.fit_lgb(led.F, rows[:2000], y[rows[:2000]], store.seg[rows[:2000]], **kw)
        lines = PR.read(os.path.join(tmp, PR.JOURNAL))
        check("every fit, fitted or resumed, is one journal line",
              len(lines) == 3 and [e["hit"] for e in lines] == [False, True, False] and all(e["kind"] == "lgb" for e in lines))
        check("...holding only counts, times and memory",
              all(set(e) <= PR.ALLOWED for e in lines) and lines[0]["rows"] == len(rows) and lines[0]["cols"] == len(cols)
              and lines[2]["rows"] == 2000 and lines[0]["seconds"] > 0 and 0 <= lines[1]["seconds"] < lines[0]["seconds"])
        for bad, why in ((dict(label=1), "a key outside the allow-list"), (dict(rows=np.arange(3)), "an array"),
                         (dict(kind="x" * 40), "a long string")):
            try:
                PR.record(tmp, **bad)
                refused = False
            except ValueError:
                refused = True
            check(f"the journal refuses {why}", refused)

        # calibration: an earlier run on this machine fitted at exactly 2x the planner's model
        j = os.path.join(tmp, "cal.jsonl")
        me = {"threads": 2, "ram_gb": 8.16}

        def write(entries):
            with open(j, "w") as fh:
                fh.write("\n".join(_json.dumps({"t": i, **e}) for i, e in enumerate(entries)) + "\n")
        pred = lambda r, c: PLN.SEC_PER_ROW_COL * r * c * PLN.CALIBRATION_THREADS / 2
        fitted = [{"kind": "lgb", "hit": False, "rows": 10_000 * (k + 1), "cols": 900, "seconds": 2 * pred(10_000 * (k + 1), 900)}
                  for k in range(6)]
        resumed = [{"kind": "lgb", "hit": True, "rows": 50_000, "cols": 900, "seconds": 0.4}] * 8   # loading takes time too
        write([{"event": "start", **me}] + fitted + resumed)
        c = PR.calibration(j, **me)
        check("calibration is the measured / predicted time of fitted boosters, resumed ones ignored",
              c is not None and abs(c["time"] - 2.0) < 1e-9 and c["fits"] == 6, f"{c}")
        check("...from this machine only", PR.calibration(j, threads=16, ram_gb=64.0) is None
              and PR.calibration(j, threads=2, ram_gb=16.0) is None)
        write([{"event": "start", **me}] + fitted[:4])
        check(f"...and only after {PR.MIN_FITS} of them", PR.calibration(j, **me) is None)
        e1 = PLN.estimate(store, len(led.names), SCALES["demo"], time_factor=1.0)
        e2 = PLN.estimate(store, len(led.names), SCALES["demo"], time_factor=2.0)
        check("the planner scales booster and twin time by the factor",
              abs(e2["booster_hours"] - 2 * e1["booster_hours"]) < 1e-9 and abs(e2["twin_hours"] - 2 * e1["twin_hours"]) < 1e-9)

        # progress and ETA: 10 planned, 7 done (4 resumed, which take seconds), three fitted at 60 s, the twin to come
        write([{"event": "start", "planned_fits": 10, "twin_hours": 1.0, **me}]
              + [{"kind": "lgb", "hit": False, "seconds": 60.0, "rows": 1, "cols": 1, "rss_gb": 5.4, "swap_gb": 4.7}] * 3
              + [{"kind": "lgb", "hit": True, "seconds": 0.4, "rows": 1, "cols": 1, "rss_gb": 5.4, "swap_gb": 4.7}] * 4)
        sm = PR.summary(j, now=100.0)
        check("progress counts fits, and the ETA is the fits left at the median FITTED time plus the twin",
              sm["done"] == 7 and sm["resumed"] == 4 and sm["planned"] == 10 and sm["median_fit_s"] == 60.0
              and abs(sm["eta_h"] - (3 * 60 / 3600 + 1.0)) < 1e-9, f"ETA {sm['eta_h']:.3f} h")
        line = progress_line(sm)
        check("physionet status prints it", "7 of ~10 booster fits (4 resumed)" in line and "5.4 GB RAM + 4.7 GB swap" in line
              and "median fit 1.0 min" in line, line.replace("\n", " | "))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Mutations: a green check is not evidence until it has been seen to fail
# --------------------------------------------------------------------------- #
@contextmanager
def _mutated(module, old: str, new: str):
    src = inspect.getsource(module)
    if old not in src:
        raise AssertionError(f"mutation target not found in {module.__name__}: {old!r}")
    saved = {k: v for k, v in vars(module).items() if inspect.isfunction(v) or inspect.isclass(v)}
    try:
        exec(compile(src.replace(old, new), module.__file__, "exec"), module.__dict__)
        yield
    finally:
        module.__dict__.update(saved)


MUTATIONS = [
    ("a safe-path check that lets .. through", fetch, 'any(x in ("", ".", "..") for x in parts)', 'any(x in ("", ".") for x in parts)', "1"),
    ("credentials shown in repr", auth, 'password=<hidden>, source=', 'password={self.password}, source=', "2"),
    ("credentials sent over clear text", fetch, 'if parts.scheme != "https" and parts.hostname not in LOOPBACK:', 'if False:', "2"),
    ("credentials follow any redirect", fetch, 'urllib.parse.urlsplit(cur).hostname == origin)', 'True)', "2"),
    ("a client PhysioNet serves as anonymous", fetch, 'return PhysioNetClient(origin, *who).session',
     'return __import__("requests").Session()', "2"),
    ("a login page accepted as a file", fetch, 'and "text/html" in ctype', 'and False', "3"),
    ("a short body taken as complete", fetch, 'if self.length is not None and self.offset < self.length:', 'if False:', "4"),
    ("an ignored Range appended", fetch, 'while skipped < self.offset:', 'while False:', "4"),
    ("two versions of a file joined", fetch, 'if self.etag and etag and etag != self.etag:', 'if False:', "4"),
    ("the download hash not compared", fetch, 'if sha256 is None or digest == sha256:', 'if True:', "4"),
    ("a part's range not checked", fetch, "if not m or int(m.group(1)) != pos or int(m.group(2)) > end:", "if False:", "4b"),
    ("a part's version not checked", fetch, "if etag and got and got != etag:", "if False:", "4b"),
    ("a whole-file answer taken as a part", fetch, "if status != 206:\n                    resp.close()\n                    raise RangeIgnored",
     "if False:\n                    resp.close()\n                    raise RangeIgnored", "4b"),
    ("joined parts not hashed", fetch, "if sha256 is not None and digest != sha256:\n            raise IntegrityError(f\"{url}: SHA-256",
     "if False:\n            raise IntegrityError(f\"{url}: SHA-256", "4b"),
    ("a part resumed from its start", fetch, 'headers = {"Range": f"bytes={pos}-{end}", "Accept-Encoding": "identity"}',
     'headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}', "4b"),
    ("a partial of unknown origin kept", fetch, 'if have and not etag:', 'if False:', "4"),
    ("text kept for every item", PL, 'if text is not None and tb.num_rows:', 'if False:', "5"),
    ("a bucket that mixes stays", PL, "bucket = tb[spec.key].to_numpy() % spec.buckets", "bucket = np.arange(tb.num_rows) % spec.buckets", "5"),
    ("the item filter removed", PL, 'if items is not None:', 'if False:', "5"),
    ("the source hash not compared at publish", PL, 'if sha256 is not None and digest != sha256:', 'if False:', "6"),
    ("a stale table served", PL, 'm["spec_hash"] == spec.spec_hash() and ', '', "6"),
    ("a missing column read as nothing", ML, "if missing:", "if False:", "6"),
    ("a lost tie rule", SRC, '"_t", "_tie", "value"]', '"_t"]', "7"),
    ("the future back in the last hour", SRC, "s = np.maximum(s, 0)", "s = np.clip(s, 0, K.MAX_HOURS - 1)", "7"),
    ("the SQL's tie rule gone", SRC, "ORDER BY charttime DESC, itemid ASC, value DESC LIMIT 1", "ORDER BY charttime DESC LIMIT 1", "7"),
    ("the old BigQuery names", SRC, 'hosp: str = "physionet-data.mimiciv_v3_1_hosp"', 'hosp: str = "physionet-data.mimiciv_3_1_hosp"', "7"),
    ("no label check", CT, 'or not re.search(c.label_pattern, str(labels[i]))', '', "8"),
    ("no drift check", CT, 'if not (1 / DRIFT_RATIO <= ratio <= DRIFT_RATIO):', 'if False:', "8"),
    ("revised stays never excluded", PA, 'stays = stays[~stays.stay_id.isin(rev)]', 'pass', "8"),
    ("no cap on revised stays", PA, '"agreement": 1.0 if share <= MAX_REVISED else 0.0}', '"agreement": 1.0}', "8"),
    ("no drift check for signed labs", CT, 'if abs(cur["median"] - ref["median"]) > DRIFT_SIGNED_SHARE * signed[item]:', 'if False:', "8"),
    ("no orphan check", CT, 'r.add("every stay has its patient", not orphan_p.any()', 'r.add("every stay has its patient", True', "8"),
    ("no cohort floor", CT, 'r.add("cohort size", len(st) >= min_stays', 'r.add("cohort size", True', "8"),
    ("a canary that ignores identifiers", guard, 'if canary.hit(tok))', 'if False)', "9"),
    ("demo identifiers not excluded", guard, 'if str(int(i)) not in skip', '', "9"),
    ("no shifted-date rule", guard, 'if dates:', 'if False:', "9"),
    ("a scan of the working copy", guard, '["git", "show", f":{name}"]', '["cat", name]', "9"),
    ("the hook overwrites what was there", guard, 'fh.write(body.rstrip("\\n") + "\\n" + block)', 'fh.write("#!/bin/sh\\n" + block)', "9"),
    ("a lake in Documents allowed", guard, 'add("lake outside Desktop and Documents", not prone', 'add("lake outside Desktop and Documents", True', "9"),
    ("no release scan", pipeline, 'if blocks:', 'if False:', "10"),
    ("a changed release accepted", pipeline, 'if pinned != text:', 'if False:', "10"),
    ("stages that always re-run", pipeline, 'if prev and prev.get("status") == "done" and prev.get("fp") == fp:', 'if False:', "10"),
    ("no job lock", pipeline, 'fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)', 'pass', "10"),
    ("no capacity gate", pipeline, 'if free + have < need:', 'if False:', "10"),
    ("a checkpoint keyed without its features", CK, 'key = memo.key("lgb", _sample(F, rows, cols),', 'key = memo.key("lgb", None,', "11"),
    ("a checkpoint keyed without its target", CK, 'rows, np.asarray(target),', 'rows, None,', "11"),
    ("the journal takes any key", PR, "if bad:", "if False:", "12"),
    ("calibration across machines", PR, 'if s.get("threads") != threads or abs(s.get("ram_gb", 0) - ram_gb) > SAME_RAM * ram_gb:',
     "if False:", "12"),
    ("resumed fits calibrate", PR, 'and not f.get("hit") and f.get("rows")', 'and f.get("rows")', "12"),
    ("resumed fits set the ETA", PR, 'fitted = [f["seconds"] for f in lgb if not f.get("hit")]', 'fitted = [f["seconds"] for f in lgb]', "12"),
    ("the time factor ignored", PLN_MOD, "slow = time_factor * CALIBRATION_THREADS", "slow = CALIBRATION_THREADS", "12"),
    ("a torn checkpoint crashes the run", CK, 'except Exception as err:                 # torn by a crash: a miss, not a failure', 'except ZeroDivisionError as err:', "11"),
]


def run_mutations(sections: Dict[str, Callable]) -> None:
    global _quiet
    print("\n13. mutations: every mechanism broken on purpose")
    for label, module, old, new, section in MUTATIONS:
        if old not in inspect.getsource(module):     # a stale target must not read as "caught"
            check(f"mutation target exists: {label}", False, old)
            continue
        start = len(_results)
        _quiet = True
        out = sys.stdout
        sys.stdout = open(os.devnull, "w")
        crashed = False
        try:
            with _mutated(module, old, new):
                sections[section]()
        except Exception:
            crashed = True
        finally:
            sys.stdout.close()
            sys.stdout = out
            _quiet = False
        red = crashed or any(not r[1] for r in _results[start:])
        del _results[start:]
        check(f"caught: {label}", red, f"section {section}")


def main(argv=None) -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = sys.argv[1:] if argv is None else argv
    work = tempfile.mkdtemp(prefix="physionet-lake-verify")
    lake_root = os.path.join(work, "lake")
    try:
        for spec in ML.project():
            stream = fetch.LocalStream(os.path.join(E.DEMO_ROOT, spec.source))
            PL.ingest(spec, stream, lake_root, None)
            stream.close()
        ref = E.build_store(SRC.CsvSource(E.DEMO_ROOT), verbose=False)
        sections = {"1": c1_manifest, "2": c2_credentials, "3": c3_access, "4": c4_transport, "4b": c4b_parallel,
                    "5": c5_streaming,
                    "6": lambda: c6_lake(lake_root), "7": lambda: c7_exact(lake_root, ref),
                    "8": lambda: c8_contract(lake_root), "9": lambda: c9_egress(lake_root), "10": c10_pipeline,
                    "11": lambda: c11_checkpoints(ref), "12": lambda: c12_progress(ref)}
        for fn in sections.values():
            fn()
        if "--no-mutations" not in args:
            run_mutations(sections)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ok = sum(r[1] for r in _results)
    failed = [r[0] for r in _results if not r[1]]
    print(f"\nverify_physionet: {ok}/{len(_results)} passed")
    for name in failed:
        print("  FAILED:", name)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
