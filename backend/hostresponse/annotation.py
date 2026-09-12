"""Probe -> gene symbol, without a multi-gigabyte download.

WHY THIS FILE EXISTS
--------------------
The cohorts are measured on different platforms, so their feature spaces share no
identifier: GSE74224 (GPL5175) indexes numeric Affymetrix exon transcript clusters
like ``2315554``, GSE40012 (GPL6947) indexes Illumina probes like ``ILMN_1343291``.
A leave-one-cohort-out study needs one shared space, and the pre-registration says
that space is the gene-symbol intersection. Getting there is not free, and the two
platforms need different routes:

    GPL6947   curated GPL6947.annot.gz, 7.0 MB, explicit "Gene symbol" column.
    GPL5175   NO curated annot exists. Its platform-level annotation is a 4.5 GB
              family SOFT (identically sized in MINiML form) — a disproportionate
              download to map 15,989 ids. The SERIES-level GSE74224_family.soft.gz
              is 25.6 MB and embeds the GPL5175 platform table as that series uses
              it, with a `gene_assignment` column. That is the route used.

Both were measured before choosing, and the choice is recorded in
preregistration.json amendment 2 — a feature space that cannot be reproduced is
not a feature space.

THE AFFYMETRIX `gene_assignment` FORMAT
--------------------------------------
One cell holds every transcript assigned to the cluster, ``///``-separated, each
``//``-separated::

    NM_001005484 // OR4F5 // olfactory receptor, family 4 // 1p36.33 // 79501 /// ...

The symbol is the second field of the first assignment. ``---`` is Affymetrix's
null and yields no symbol — an unannotated probe is dropped rather than guessed at.

A DOWNSTREAM DECISION THIS FILE DOES NOT MAKE
---------------------------------------------
Several probes map to one symbol, so the study must collapse them. The order
matters and is NOT arbitrary: collapsing by mean on raw values would break the
pre-registered invariance, because GSE74224 ships log-scale values (2.07-14.08)
and GSE40012 ships linear intensities (41.7-34,356), and a mean is not invariant
to a monotone per-sample transform while a rank is. So the pipeline ranks probes
WITHIN a sample first, then averages ranks per symbol. See expression.py.
"""
from __future__ import annotations

import gzip
import os
import urllib.request
from typing import Dict, Optional

try:
    from backend.physio.data import netguard
except Exception:                                   # pragma: no cover
    netguard = None                                 # type: ignore

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo"

#: Which route each platform uses, with the measured cost of each. Recorded rather
#: than rediscovered: the sizes are the whole reason the routes differ.
PLATFORM_ROUTE = {
    "GPL6947": {"route": "annot", "size_mb": 7.0,
                "why": "curated annot with an explicit 'Gene symbol' column"},
    "GPL5175": {"route": "series_soft", "size_mb": 25.6,
                "why": "no curated annot exists; the platform SOFT is 4.5 GB, the "
                       "series SOFT embeds the same table for 25.6 MB"},
}


def _shard(acc: str) -> str:
    """GEO's FTP sharding: GSE74224 -> GSE74nnn, GPL5175 -> GPL5nnn, GPL570 -> GPLnnn.

    Worth stating because getting it wrong does not 404 loudly in a way that reads as
    a bug — it reads as "this platform has no annotation", which is a false fact about
    the archive. That mistake was made once here, about GPL570.
    """
    return acc[:-3] + "nnn" if len(acc) > 6 else acc[:3] + "nnn"


def annot_url(gpl: str) -> str:
    return "%s/platforms/%s/%s/annot/%s.annot.gz" % (GEO_FTP, _shard(gpl), gpl, gpl)


def series_soft_url(gse: str) -> str:
    return "%s/series/%s/%s/soft/%s_family.soft.gz" % (GEO_FTP, _shard(gse), gse, gse)


def _download(url: str, dest: str, timeout: int = 600) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _get() -> str:
        with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as fh:
            fh.write(r.read())
        return dest

    if netguard is not None:
        netguard.install_timeouts()
        with netguard.deadline(timeout):
            return netguard.retry(_get, attempts=2)
    return _get()


def _platform_table(path: str):
    """Yield the platform table's header then its rows, and stop at the end sentinel.

    Streamed on purpose: a series family SOFT also contains every sample's data table,
    so reading the whole file to reach one section would cost far more than the section.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        in_tbl = False
        for line in fh:
            if line.startswith("!platform_table_begin"):
                in_tbl = True
                continue
            if line.startswith("!platform_table_end"):
                return
            if in_tbl:
                yield line.rstrip("\n").split("\t")


def symbols_from_annot(path: str) -> Dict[str, str]:
    """ID -> symbol from a curated ``GPLxxxx.annot.gz``."""
    out: Dict[str, str] = {}
    hdr: Optional[list] = None
    col = None
    for parts in _platform_table(path):
        if hdr is None:
            hdr = parts
            for i, c in enumerate(hdr):
                if c.strip().lower() == "gene symbol":
                    col = i
            if col is None:
                raise ValueError("no 'Gene symbol' column in %s" % path)
            continue
        if len(parts) > col:
            pid, sym = parts[0].strip(), parts[col].strip()
            # a multi-symbol cell ("A///B") is ambiguous at the gene level; take the
            # first and let the collapse step handle duplicates
            sym = sym.split("///")[0].strip()
            if pid and sym and sym != "---":
                out[pid] = sym
    return out


def symbols_from_series_soft(path: str) -> Dict[str, str]:
    """ID -> symbol from a series family SOFT's embedded platform table."""
    out: Dict[str, str] = {}
    hdr: Optional[list] = None
    col = None
    for parts in _platform_table(path):
        if hdr is None:
            hdr = parts
            for i, c in enumerate(hdr):
                if c.strip().lower() == "gene_assignment":
                    col = i
            if col is None:
                raise ValueError("no 'gene_assignment' column in %s" % path)
            continue
        if len(parts) > col:
            pid, cell = parts[0].strip(), parts[col].strip()
            if not pid or not cell or cell == "---":
                continue
            first = cell.split("///")[0]
            fields = [f.strip() for f in first.split("//")]
            if len(fields) >= 2 and fields[1] and fields[1] != "---":
                out[pid] = fields[1]
    return out


def symbol_map(accession: str, platform: str,
               root: str = os.path.join("data", "geo")) -> Dict[str, str]:
    """Probe id -> gene symbol for one cohort, by whichever route that platform has."""
    route = (PLATFORM_ROUTE.get(platform) or {}).get("route")
    if route == "annot":
        path = _download(annot_url(platform), os.path.join(root, "%s.annot.gz" % platform))
        return symbols_from_annot(path)
    if route == "series_soft":
        path = _download(series_soft_url(accession),
                         os.path.join(root, accession, "%s_family.soft.gz" % accession))
        return symbols_from_series_soft(path)
    raise ValueError(
        "no annotation route recorded for platform %r. Measure the options before "
        "adding one: a curated .annot is usually a few MB, a family SOFT can be "
        "gigabytes, and the difference decides whether this study is worth running."
        % platform)
