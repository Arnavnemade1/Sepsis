"""What PhysioNet holds that this project can use, and how each one is reached.

Access classes were measured, not assumed: on 2026-11-01 an unauthenticated range request for
each project's ``SHA256SUMS.txt`` returned 206 for the ``open`` entries and 403 for the
``credentialed`` ones. Approval is per dataset (each project has its own Data Use Agreement),
so ``python -m backend.ml.physionet access`` probes what an account can really reach instead
of trusting this table.

``expected_stays`` is a floor for the contract check: a partial table set must not pass for
the full database.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

BASE_URL = "https://physionet.org/files"


@dataclass(frozen=True)
class Dataset:
    slug: str
    version: str
    title: str
    access: str                          # "open" | "credentialed"
    use: str
    demo: Optional[str] = None           # the open dataset that verifies this one's code path
    bigquery: Tuple[str, ...] = ()       # datasets on the physionet-data project
    expected_stays: int = 0

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.slug}/{self.version}/"

    @property
    def key(self) -> str:
        return f"{self.slug}-{self.version}"


DATASETS: Dict[str, Dataset] = {d.slug: d for d in [
    Dataset("mimiciv", "3.1", "MIMIC-IV Clinical Database", "credentialed",
            "the full Sepsis-3 run: docs/MIMIC.md", demo="mimic-iv-demo",
            bigquery=("mimiciv_v3_1_hosp", "mimiciv_v3_1_icu"), expected_stays=90_000),
    Dataset("mimic-iv-demo", "2.2", "MIMIC-IV Clinical Database Demo", "open",
            "100 real patients: every code path is verified on these first"),
    Dataset("mimic-iv-note", "2.2", "MIMIC-IV-Note", "credentialed",
            "radiology and discharge text: the strongest unbuilt lever (docs/MIMIC.md section 7)"),
    Dataset("mimic-iv-ed", "2.2", "MIMIC-IV-ED", "credentialed",
            "the hours before the ICU", demo="mimic-iv-ed-demo"),
    Dataset("mimic-iv-ed-demo", "2.2", "MIMIC-IV-ED Demo", "open", "verifies MIMIC-IV-ED code paths"),
    Dataset("eicu-crd", "2.0", "eICU Collaborative Research Database", "credentialed",
            "external validation across 200+ hospitals", demo="eicu-crd-demo"),
    Dataset("eicu-crd-demo", "2.0.1", "eICU Collaborative Research Database Demo", "open",
            "verifies eICU code paths"),
    Dataset("mimic4wdb", "0.1.0", "MIMIC-IV Waveform Database", "open",
            "ECG and arterial waveforms: autonomic recovery, cardio-respiratory coupling"),
]}


def get(slug: str) -> Dataset:
    try:
        return DATASETS[slug]
    except KeyError:
        raise SystemExit(f"unknown dataset {slug!r}; known: {', '.join(sorted(DATASETS))}") from None
