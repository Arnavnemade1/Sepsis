"""csv.gz -> a bucketed Parquet lake, streamed. Raw files need not ever exist on this disk.

Full MIMIC-IV is roughly 10 GB compressed and ten times that raw, and most of ``chartevents``
is nursing documentation no model reads. So a table can flow straight from the socket:
HTTP -> SHA-256 -> gunzip -> CSV parser -> keep the catalog's items and columns -> Parquet.
Only what the pipeline reads lands on disk.

**Buckets, not a sort.** Rows go to ``key % n_buckets``. A global sort of hundreds of millions
of rows needs temporary disk this machine does not have; buckets need none. They are also the
unit of work downstream, because a stay's rows live in one bucket: binning one bucket at a time
is exact and bounds memory. File order is preserved within a bucket.

**Nothing is published unverified.** Rows are written to a temporary directory, and it is
renamed into place only after the source's SHA-256 has matched. A dropped run leaves no table
at all; a re-run skips what is complete and redoes what is not. A table's manifest binds it to
its source hash and to its ``spec_hash``, so a catalog change makes the old table visibly stale
instead of silently missing the new item.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import time
import zlib
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

FORMAT = 1
MANIFEST = "_table.json"
#: pandas' default NA strings: ``CsvSource`` is the reference, so a cell must read as it does
NA_VALUES = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
             "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"]
FLUSH_ROWS = 100_000
BLOCK_BYTES = 1 << 24


@dataclass(frozen=True)
class TableSpec:
    """One table as the lake keeps it."""
    module: str
    table: str
    columns: Tuple[Tuple[str, str], ...]         # (name, int64 | float64 | string | timestamp)
    key: Optional[str] = None                    # bucket column; rows with a null key cannot join and are dropped
    buckets: int = 1
    items: Optional[FrozenSet[int]] = None       # keep only these itemids
    text_items: Optional[FrozenSet[int]] = None  # keep ``value`` only for these; a null elsewhere saves most of the bytes

    @property
    def names(self) -> List[str]:
        return [c for c, _ in self.columns]

    @property
    def rel(self) -> str:
        return f"{self.module}/{self.table}" if self.module else self.table

    @property
    def source(self) -> str:
        return f"{self.rel}.csv.gz"

    def spec_hash(self) -> str:
        body = {"format": FORMAT, "rel": self.rel, "columns": list(self.columns), "key": self.key,
                "buckets": self.buckets, "items": sorted(self.items) if self.items is not None else None,
                "text_items": sorted(self.text_items) if self.text_items is not None else None}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.csv as pcsv
        import pyarrow.parquet as pq
    except ImportError as err:                   # pragma: no cover
        raise ImportError("pip install -r backend/ml/physionet/requirements.txt") from err
    return pa, pc, pcsv, pq


def _arrow_type(pa, t: str):
    return {"int64": pa.int64(), "float64": pa.float64(), "string": pa.string(),
            "timestamp": pa.timestamp("s")}[t]


def table_dir(root: str, spec: TableSpec) -> str:
    return os.path.join(root, spec.rel)


def read_manifest(root: str, spec: TableSpec) -> Optional[Dict]:
    path = os.path.join(table_dir(root, spec), MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def is_current(root: str, spec: TableSpec, sha256: Optional[str]) -> bool:
    m = read_manifest(root, spec)
    return bool(m and m["spec_hash"] == spec.spec_hash() and (sha256 is None or m["source_sha256"] == sha256))


class _BucketWriter:
    """Per-bucket buffers flushed as Parquet row groups; memory is bounded by the buffers."""

    def __init__(self, directory: str, schema, buckets: int, disk_check=None):
        self.pa, _, _, self.pq = _pa()
        self.dir, self.schema, self.n = directory, schema, buckets
        self.buffers: List[List] = [[] for _ in range(buckets)]
        self.counts = np.zeros(buckets, dtype=np.int64)
        self.written = np.zeros(buckets, dtype=np.int64)
        self.writers: Dict[int, object] = {}
        self.disk_check = disk_check

    def add(self, table, bucket: Optional[np.ndarray]) -> None:
        if not table.num_rows:
            return
        if bucket is None:
            self._push(0, table)
            return
        order = np.argsort(bucket, kind="stable")      # stable: file order survives inside a bucket
        table = table.take(self.pa.array(order))
        sorted_b = bucket[order]
        cuts = np.flatnonzero(np.diff(sorted_b)) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(sorted_b)]):
            self._push(int(sorted_b[lo]), table.slice(int(lo), int(hi - lo)))

    def _push(self, b: int, table) -> None:
        self.buffers[b].append(table)
        self.counts[b] += table.num_rows
        if self.counts[b] >= FLUSH_ROWS:
            self._flush(b)

    def _flush(self, b: int) -> None:
        if not self.buffers[b]:
            return
        if b not in self.writers:
            if self.disk_check:
                self.disk_check()
            self.writers[b] = self.pq.ParquetWriter(os.path.join(self.dir, f"b{b:04d}.parquet"), self.schema,
                                                    compression="zstd", compression_level=3)
        self.writers[b].write_table(self.pa.concat_tables(self.buffers[b]))
        self.written[b] += self.counts[b]
        self.buffers[b], self.counts[b] = [], 0
        if self.disk_check:
            self.disk_check()

    def close(self) -> None:
        for b in range(self.n):
            self._flush(b)
        for w in self.writers.values():
            w.close()


def ingest(spec: TableSpec, stream, root: str, sha256: Optional[str], compressed: bool = True,
           disk_check=None) -> Dict:
    """Stream one table from ``stream`` (``fetch.RemoteStream`` or ``LocalStream``) into the lake."""
    pa, pc, pcsv, _ = _pa()
    final = table_dir(root, spec)
    tmp = final + f".tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    t0 = time.time()
    types = {c: _arrow_type(pa, t) for c, t in spec.columns}
    schema = pa.schema([(c, types[c]) for c in spec.names])
    writer = _BucketWriter(tmp, schema, spec.buckets, disk_check)
    items = pa.array(sorted(spec.items), type=pa.int64()) if spec.items is not None else None
    text = pa.array(sorted(spec.text_items), type=pa.int64()) if spec.text_items is not None else None
    rows_in = 0
    try:
        reader = pcsv.open_csv(                  # reads the first block: already inside the guard
            gzip.GzipFile(fileobj=stream) if compressed else stream,
            read_options=pcsv.ReadOptions(block_size=BLOCK_BYTES, use_threads=False),
            parse_options=pcsv.ParseOptions(newlines_in_values=True),
            convert_options=pcsv.ConvertOptions(include_columns=spec.names, column_types=types, null_values=NA_VALUES,
                                                strings_can_be_null=True, quoted_strings_can_be_null=True))
        for batch in reader:
            rows_in += batch.num_rows
            tb = pa.Table.from_batches([batch]).select(spec.names)
            if items is not None:
                tb = tb.filter(pc.is_in(tb["itemid"], value_set=items))
            if spec.key is not None:
                tb = tb.filter(pc.is_valid(tb[spec.key]))
            if text is not None and tb.num_rows:
                kept = pc.if_else(pc.is_in(tb["itemid"], value_set=text), tb["value"], pa.scalar(None, pa.string()))
                tb = tb.set_column(tb.schema.get_field_index("value"), "value", kept)
            bucket = None
            if spec.buckets > 1 and tb.num_rows:
                bucket = tb[spec.key].to_numpy() % spec.buckets
            writer.add(tb, bucket)
        writer.close()
        stream.drain()                           # the hash covers every byte, trailing ones too
        digest = stream.hexdigest()
        if sha256 is not None and digest != sha256:
            from backend.ml.physionet.fetch import IntegrityError
            raise IntegrityError(f"{spec.source}: SHA-256 {digest[:16]}... where PhysioNet published {sha256[:16]}...")
        files = sorted(f for f in os.listdir(tmp) if f.endswith(".parquet"))
        manifest = {"format": FORMAT, "table": spec.rel, "spec_hash": spec.spec_hash(), "source_sha256": digest,
                    "source_bytes": int(stream.offset), "rows_in": int(rows_in), "rows_kept": int(writer.written.sum()),
                    "buckets": spec.buckets, "files": files,
                    "bytes": int(sum(os.path.getsize(os.path.join(tmp, f)) for f in files)),
                    "reconnects": int(getattr(stream, "reconnects", 0)), "seconds": round(time.time() - t0, 1)}
        with open(os.path.join(tmp, MANIFEST), "w") as fh:
            json.dump(manifest, fh, indent=1)
    except (EOFError, gzip.BadGzipFile, zlib.error, pa.ArrowInvalid, KeyError) as err:
        # truncated, corrupted, or no longer the columns we were promised: the source's fault
        from backend.ml.physionet.fetch import IntegrityError
        shutil.rmtree(tmp, ignore_errors=True)
        raise IntegrityError(f"{spec.source} could not be read: {type(err).__name__}: {str(err)[:200]}") from err
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(final, ignore_errors=True)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.replace(tmp, final)
    return manifest


def fingerprint(root: str, specs: List[TableSpec]) -> str:
    """One hash for the lake's contents: every table's source hash and projection."""
    h = hashlib.sha256()
    for spec in sorted(specs, key=lambda s: s.rel):
        m = read_manifest(root, spec)
        if m is None:
            raise FileNotFoundError(f"lake table {spec.rel} is missing under {root}")
        h.update(f"{spec.rel}|{m['spec_hash']}|{m['source_sha256']}\n".encode())
    return h.hexdigest()
