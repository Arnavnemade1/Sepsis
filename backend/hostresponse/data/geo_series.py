"""Parse a GEO **series matrix** file — metadata and expression, no dependencies.

FORMAT
------
A series matrix is a single (usually gzipped) text file in two halves:

    !Series_title          "Discrimination of SIRS from Sepsis in Critically Ill Adults"
    !Series_geo_accession   "GSE74224"
    !Sample_geo_accession   "GSM1..."  "GSM2..."   ...
    !Sample_characteristics_ch1  "diagnosis: sepsis"  "diagnosis: SIRS"   ...
    !Sample_characteristics_ch1  "age: 55"            "age: 60"          ...
    !series_matrix_table_begin
    "ID_REF"   "GSM1..."   "GSM2..."
    "1007_s_at"   5.21       4.88
    !series_matrix_table_end

Metadata lines start with ``!``, are tab-separated, and quote every field. A
characteristics row repeats once per attribute, and each cell carries its own
``key: value`` — so the same attribute can be named differently across samples in
the same series, which is why the RAW strings are preserved alongside the parsed
ones (see below).

WHY NOT GEOparse
----------------
`GEOparse` is not installed here and is not needed: the format above is a sentinel
split and a TSV. Adding a dependency to parse two sentinels would be a dependency
this project then has to carry, install and audit forever. The repo's precedent for
a genuinely unavoidable one is the commented-out `wfdb` block in
`backend/requirements.txt` — declared, not installed.

WHY THE RAW CHARACTERISTIC STRINGS ARE KEPT
-------------------------------------------
The study's inclusion criterion is a judgement about what a cohort's CONTROL ARM
actually is — sterile inflammation, or merely healthy volunteers. That judgement
decides whether a cohort counts at all, so it must be auditable against the source
text rather than asserted by a parser's guess. Every sample's verbatim
characteristics are therefore carried through to the manifest as evidence.
"""
from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

TABLE_BEGIN = "!series_matrix_table_begin"
TABLE_END = "!series_matrix_table_end"


@dataclass
class GeoSeries:
    """One parsed GEO series. ``matrix`` is None when parsed metadata-only."""
    accession: str
    title: str
    platforms: List[str]
    samples: List[str]
    #: attribute -> per-sample value, aligned with ``samples``. Missing entries are
    #: None rather than dropped, so the columns stay aligned with the matrix.
    characteristics: Dict[str, List[Optional[str]]] = field(default_factory=dict)
    #: per-sample list of the VERBATIM ``!Sample_characteristics_ch1`` cells. This is
    #: the evidence the manifest quotes; never reconstruct it from ``characteristics``.
    raw_characteristics: List[List[str]] = field(default_factory=list)
    sample_titles: List[str] = field(default_factory=list)
    matrix: Optional[pd.DataFrame] = None

    @property
    def n_samples(self) -> int:
        return len(self.samples)


def _open(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _cells(line: str) -> List[str]:
    """Split a metadata line into its unquoted fields, dropping the ``!key``."""
    parts = line.rstrip("\n").split("\t")
    return [p.strip().strip('"') for p in parts[1:]]


def parse_series_matrix(path: str, with_matrix: bool = True) -> GeoSeries:
    """Parse one series-matrix file.

    ``with_matrix=False`` stops at the table sentinel and returns metadata only.
    The manifest gate uses that path: deciding whether a cohort is admissible needs
    the characteristics, not a 54,000-row expression matrix, and reading the matrix
    anyway would make the gate cost scale with data it may well reject.
    """
    accession, title = "", ""
    platforms: List[str] = []
    samples: List[str] = []
    sample_titles: List[str] = []
    char_rows: List[List[str]] = []
    table_lines: List[str] = []

    with _open(path) as fh:
        in_table = False
        for line in fh:
            if line.startswith(TABLE_BEGIN):
                if not with_matrix:
                    break
                in_table = True
                continue
            if line.startswith(TABLE_END):
                in_table = False
                continue
            if in_table:
                table_lines.append(line)
                continue
            if not line.startswith("!"):
                continue
            key = line.split("\t", 1)[0][1:].strip().lower()
            if key == "series_geo_accession":
                accession = (_cells(line) or [""])[0]
            elif key == "series_title":
                title = (_cells(line) or [""])[0]
            elif key == "series_platform_id":
                platforms.extend(c for c in _cells(line) if c)
            elif key == "sample_geo_accession":
                samples = _cells(line)
            elif key == "sample_title":
                sample_titles = _cells(line)
            elif key.startswith("sample_characteristics"):
                char_rows.append(_cells(line))
            elif key == "sample_platform_id":
                platforms.extend(sorted({c for c in _cells(line) if c}))

    n = len(samples)
    raw: List[List[str]] = [[] for _ in range(n)]
    characteristics: Dict[str, List[Optional[str]]] = {}
    for row in char_rows:
        for i in range(n):
            cell = row[i] if i < len(row) else ""
            if not cell:
                continue
            raw[i].append(cell)
            # "diagnosis: sepsis" -> ("diagnosis", "sepsis"). A cell with no colon is
            # kept in raw only: inventing a key for it would put a guess in the field
            # the inclusion decision is read from.
            if ":" in cell:
                k, _, v = cell.partition(":")
                slot = characteristics.setdefault(k.strip().lower(), [None] * n)
                if slot[i] is None:
                    slot[i] = v.strip()

    matrix = None
    if with_matrix and table_lines:
        matrix = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t",
                             index_col=0, low_memory=False)
        matrix.index = [str(i).strip().strip('"') for i in matrix.index]
        matrix.columns = [str(c).strip().strip('"') for c in matrix.columns]

    return GeoSeries(accession=accession, title=title,
                     platforms=sorted(set(platforms)), samples=samples,
                     characteristics=characteristics, raw_characteristics=raw,
                     sample_titles=sample_titles, matrix=matrix)


def series_matrix_url(accession: str) -> str:
    """The canonical public URL for a series-matrix file.

    GEO shards by accession prefix: GSE74224 lives under ``GSE74nnn``. No
    credentialing, no API key — these are open archives.
    """
    acc = accession.strip().upper()
    stub = acc[:-3] + "nnn" if len(acc) > 6 else acc
    return ("https://ftp.ncbi.nlm.nih.gov/geo/series/%s/%s/matrix/%s_series_matrix.txt.gz"
            % (stub, acc, acc))


def local_path(accession: str, root: str = os.path.join("data", "geo")) -> str:
    return os.path.join(root, accession.upper(),
                        "%s_series_matrix.txt.gz" % accession.upper())
