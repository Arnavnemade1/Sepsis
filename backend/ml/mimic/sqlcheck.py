"""The BigQuery SQL, run without BigQuery.

Until now the SQL could only be tested with credentials, on day one, against a 0.99 agreement
bar. That bar hid a real difference: ~0.9% of chart cells disagreed, because the two paths broke
same-timestamp ties differently, and the check would have passed at 0.9907. Here each query is
transpiled from BigQuery's dialect to DuckDB's and run on the open demo's CSVs, and must match
the reference EXACTLY. What this cannot prove is BigQuery's own behaviour, table names and
permissions; ``parity`` still runs against the real service once access is linked.
"""
from __future__ import annotations

import re
from typing import Dict

import pandas as pd

from backend.ml.mimic import parity as PA
from backend.ml.mimic.sources import TIME_COLS, BigQuerySource, CsvSource


class EmulatedBigQuery(BigQuerySource):
    """``BigQuerySource`` with its SQL sent to DuckDB over a CSV tree instead of Google."""

    def __init__(self, root: str):
        import os

        import duckdb
        super().__init__(project="emulated")
        self.con = duckdb.connect()
        for module in ("hosp", "icu"):
            for name in sorted(os.listdir(os.path.join(root, module))):
                table = name.split(".")[0]
                self.con.execute(f"CREATE VIEW {module}__{table} AS SELECT * FROM read_csv("
                                 f"'{root}/{module}/{name}', header=true, sample_size=-1)")

    def translate(self, sql: str) -> str:
        import logging

        import sqlglot
        s = re.sub(r"`[^`]*?_(hosp|icu)\.(\w+)`", lambda m: f"{m.group(1)}__{m.group(2)}", sql)
        logging.getLogger("sqlglot").setLevel(logging.ERROR)     # it warns about the rewrite below
        d = sqlglot.transpile(s, read="bigquery", write="duckdb")[0]
        return re.sub(r"ARRAY_AGG\((\w+) ORDER BY (.*?) LIMIT 1\)\[1\]", r"(LIST(\1 ORDER BY \2))[1]", d)

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(self.translate(sql)).fetchdf()

    def _pages(self, sql: str, page_rows: int = 2_000_000):
        df = self.query(sql)
        for c in set(df.columns) & TIME_COLS:
            df[c] = pd.to_datetime(df[c])
        yield df


def check(root: str, stays: pd.DataFrame) -> Dict:
    """Every binned table and event list, through the SQL, against the reference: exact."""
    return PA.parity(CsvSource(root), EmulatedBigQuery(root), stays, agreement=1.0)
