"""Acquire candidate GEO cohorts, classify their CONTROL ARM, and gate the study.

THE GATE IS THE POINT OF THIS FILE
----------------------------------
The study asks whether expression separates infection from STERILE systemic
inflammation. That question only exists if a cohort's comparator arm is genuinely
sterile inflammation — post-surgical patients, trauma, pancreatitis, culture-negative
SIRS. A cohort whose controls are healthy volunteers answers a different and far
easier question ("is this person ill?"), and a classifier trained on it would score
well while being useless for the thing this system actually confuses.

So no cohort enters on its title. Each one is classified from its own
``!Sample_characteristics_ch1`` cells, the VERBATIM strings are recorded as evidence,
and a cohort whose controls cannot be positively identified as sterile is marked
``unknown`` — never optimistically as sterile. ``python -m
backend.hostresponse.data.build_geo_manifest --inspect`` prints the distinct label
values per cohort so the classification can be read against the source before any
model is fitted.

The gate then refuses the whole study unless at least two admissible cohorts exist
across at least two platforms. A failed gate is a legitimate terminal state, not an
error: it means the public archive does not support the question, which is a finding
worth writing down rather than a problem to route around.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from backend.hostresponse.data import geo_series as gs
from backend.ml.jsonio import json_safe

try:                                        # the repo's network discipline
    from backend.physio.data import netguard
except Exception:                           # pragma: no cover - optional at import
    netguard = None                         # type: ignore

MANIFEST = os.path.join("artifacts", "hostresponse", "cohort_manifest.json")

#: Candidates, with the reason each is on the list. Sample counts are NOT recorded
#: here — they are read from the series itself, because a number typed into a
#: constant is a number that can disagree with the data it claims to describe.
CANDIDATES: List[Tuple[str, str]] = [
    ("GSE74224", "titled 'Discrimination of SIRS from Sepsis in Critically Ill Adults'; "
                 "controls are post-surgical, infection-negative (SeptiCyte discovery)"),
    ("GSE63042", "RNA-seq; SIRS vs sepsis vs septic shock (CAPSOD)"),
    ("GSE40012", "bacterial vs influenza pneumonia vs non-infective SIRS vs healthy"),
    ("GSE28750", "sepsis vs post-surgical sterile inflammation vs healthy; small"),
    ("GSE65682", "MARS consortium; comparator arm UNVERIFIED from the summary alone"),
]

#: Characteristics keys that plausibly carry the diagnostic group, most specific first.
LABEL_KEYS: Tuple[str, ...] = (
    "diagnosis", "disease state", "disease", "group", "subject group", "patient group",
    "condition", "status", "sample group", "sample type", "phenotype", "description",
)

# Substring rules. Deliberately narrow: anything unmatched becomes "other" and, if it
# dominates a cohort's controls, the cohort becomes unknown rather than admissible.
INFECTIOUS = ("sepsis", "septic", "infection", "infected", "bacterial", "influenza",
              "pneumonia", "viral")
# matched against the separator-normalised value, so "post surgical" also catches
# "post-surgical" and "POST_SURGICAL"
STERILE = ("sirs", "systemic inflammatory response", "post surgical", "postsurgical",
           "postoperative", "post operative", "trauma", "surgery",
           "non infect", "noninfect", "sterile", "pancreatitis")
HEALTHY = ("healthy", "normal volunteer", "healthy control", "hc")

STERILE_TYPES = ("sterile_inflammation", "post_surgical_sirs")
MIN_INFECTIOUS, MIN_STERILE = 20, 15


def fetch(accession: str, root: str = os.path.join("data", "geo"),
          timeout: int = 180) -> str:
    """Download one series matrix if it is not already on disk. Returns the path.

    Wrapped in the repo's netguard because two adapters in this codebase have already
    hung for hours on a silent socket. GEO is an open archive: no key, no DUA.
    """
    path = gs.local_path(accession, root)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = gs.series_matrix_url(accession)

    def _get() -> str:
        with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as fh:
            fh.write(r.read())
        return path

    if netguard is not None:
        netguard.install_timeouts()
        with netguard.deadline(timeout):
            return netguard.retry(_get, attempts=2)
    return _get()


def _key_score(values: Sequence[str]) -> Tuple[int, int]:
    """How well a characteristics key partitions samples into the classes we care about.

    ``(distinct non-other classes, samples classified)``. A key naming the diagnostic
    group scores 2+ because it separates infectious from sterile or healthy; a key
    holding patient ids, gender or tissue scores 0 and loses.
    """
    classes = [_class_of(v) for v in values]
    kinds = {c for c in classes if c != "other"}
    return len(kinds), sum(1 for c in classes if c != "other")


def label_field(series: gs.GeoSeries) -> Tuple[Optional[str], List[str]]:
    """Let the cohort NAME ITS OWN label field, then prove the choice.

    An earlier version scanned a fixed list of key names and otherwise fell back to
    each sample's FIRST characteristics cell. On real cohorts that silently picked
    "patient id", "tissue: whole blood" and "gender: male" — so three cohorts that do
    carry a diagnosis were recorded as unlabelled. Excluding a cohort because a parser
    guessed the wrong column is exactly the kind of decision this study is not allowed
    to make quietly.

    Every key is therefore scored by how well its values partition into
    infectious/sterile/healthy, and the best partition wins. ``LABEL_KEYS`` survives
    only as a tie-break, so an explicitly-named "diagnosis" beats an equally
    discriminative "description". A cohort where no key separates anything returns
    None and is classified ``unknown`` — never optimistically.
    """
    best: Optional[Tuple[Tuple[int, int, int], str, List[str]]] = None
    for k, vals in series.characteristics.items():
        vs = [(v or "") for v in vals]
        kinds, classified = _key_score(vs)
        if kinds < 2:
            continue
        pref = len(LABEL_KEYS) - LABEL_KEYS.index(k) if k in LABEL_KEYS else 0
        rank = (kinds, classified, pref)
        if best is None or rank > best[0]:
            best = (rank, k, vs)
    if best is not None:
        return best[1], best[2]
    raw = [(cells[0] if cells else "") for cells in series.raw_characteristics]
    return (None, raw) if any(raw) else (None, [""] * series.n_samples)


def all_keys(series: gs.GeoSeries, max_values: int = 8) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """Every characteristics key with its distinct values — the audit surface."""
    out = []
    for k, vals in series.characteristics.items():
        c = Counter((v or "").strip() for v in vals if (v or "").strip())
        out.append((k, c.most_common(max_values)))
    return out


def _class_of(value: str) -> str:
    """Classify one label value, conservatively.

    Separators are normalised first: GSE28750 writes ``POST_SURGICAL`` where GSE74224
    writes ``post-surgical``, and without this the underscore form fell through to
    "other" — a sterile arm invisible to the gate because of a punctuation choice.
    """
    v = re.sub(r"[_\-/]+", " ", (value or "").strip().lower())
    v = re.sub(r"\s+", " ", v)
    if not v:
        return "other"
    # healthy first: "healthy control" contains neither an infection nor a SIRS term,
    # but "control" alone is ambiguous and must NOT be read as either. Short tokens
    # match whole words only, so "hc" cannot fire inside another word.
    tokens = set(v.split())
    if any((t in tokens) if len(t) <= 3 else (t in v) for t in HEALTHY):
        return "healthy"
    if any(t in v for t in STERILE):
        return "sterile"
    if any(t in v for t in INFECTIOUS):
        return "infectious"
    return "other"


def classify(series: gs.GeoSeries) -> Dict[str, object]:
    """Count the arms and decide the cohort's control type, conservatively."""
    key, values = label_field(series)
    classes = [_class_of(v) for v in values]
    n = Counter(classes)
    n_inf, n_ster, n_healthy, n_other = (n["infectious"], n["sterile"],
                                         n["healthy"], n["other"])

    if n_ster >= 1 and n_healthy >= 1:
        control_type = "mixed"
    elif n_ster >= 1:
        control_type = "post_surgical_sirs" if any(
            "surg" in (v or "").lower() for v in values) else "sterile_inflammation"
    elif n_healthy >= 1:
        control_type = "healthy"
    else:
        control_type = "unknown"

    # An admissible cohort needs a genuinely sterile arm, not a healthy one — and a
    # "mixed" cohort is admissible only on its sterile samples, which is why the
    # counts, not the label, drive the decision.
    inclusion_ok = bool(
        (control_type in STERILE_TYPES or control_type == "mixed")
        and n_inf >= MIN_INFECTIOUS and n_ster >= MIN_STERILE)
    reason = ""
    if not inclusion_ok:
        if control_type == "healthy":
            reason = ("controls are healthy volunteers, not sterile inflammation — this "
                      "cohort answers an easier question than the one being asked")
        elif control_type == "unknown":
            reason = "no arm could be positively identified from the characteristics"
        elif n_inf < MIN_INFECTIOUS:
            reason = "too few infectious samples (%d < %d)" % (n_inf, MIN_INFECTIOUS)
        else:
            reason = "too few sterile samples (%d < %d)" % (n_ster, MIN_STERILE)

    distinct = Counter(v.strip() for v in values if (v or "").strip())
    return {
        "label_key": key,
        "n_total": series.n_samples,
        "n_infectious": n_inf, "n_sterile": n_ster,
        "n_healthy": n_healthy, "n_other": n_other,
        "control_type": control_type,
        # verbatim, and capped only in count — this is the evidence the inclusion
        # decision is read against, so it is never paraphrased
        "control_evidence": [{"value": v, "n": c} for v, c in distinct.most_common(12)],
        "inclusion_ok": inclusion_ok,
        "exclusion_reason": reason,
    }


def build(accessions: Sequence[str], do_fetch: bool = True) -> Dict[str, object]:
    cohorts = []
    for acc, why in [(a, w) for a, w in CANDIDATES if a in accessions]:
        path = fetch(acc) if do_fetch else gs.local_path(acc)
        if not os.path.exists(path):
            cohorts.append({"accession": acc, "candidate_reason": why,
                            "error": "not downloaded", "inclusion_ok": False})
            continue
        s = gs.parse_series_matrix(path, with_matrix=False)
        row = {"accession": acc, "candidate_reason": why, "title": s.title,
               "platforms": s.platforms,
               # role is assigned by preregistration.json, BEFORE any locked matrix is
               # parsed — never here, where the counts are already visible
               "role": None}
        row.update(classify(s))
        cohorts.append(row)
    return {"candidates": cohorts, "gate": gate(cohorts)}


def gate(cohorts: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Refuse the study unless the archive actually supports the question."""
    ok = [c for c in cohorts if c.get("inclusion_ok")]
    platforms = sorted({p for c in ok for p in (c.get("platforms") or [])})
    passed = len(ok) >= 2 and len(platforms) >= 2
    return {
        "passed": passed,
        "n_admissible": len(ok),
        "admissible": [c["accession"] for c in ok],
        "platforms": platforms,
        "requirement": ("at least 2 cohorts with a sterile (non-infectious) control arm, "
                        "n_infectious >= %d and n_sterile >= %d each, spanning at least "
                        "2 platforms" % (MIN_INFECTIOUS, MIN_STERILE)),
        "verdict": ("gate passed — the question is answerable on public data" if passed
                    else "GATE FAILED — no admissible cohort pair; the honest outcome is "
                         "docs/HOST_RESPONSE_NULL.md, not a workaround"),
    }


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--accessions", nargs="*", default=[c[0] for c in CANDIDATES])
    a.add_argument("--inspect", action="store_true",
                   help="print each cohort's distinct label values and stop")
    a.add_argument("--no-fetch", action="store_true")
    a.add_argument("--save", action="store_true")
    args = a.parse_args()

    if args.inspect:
        for acc in args.accessions:
            path = gs.local_path(acc) if args.no_fetch else fetch(acc)
            s = gs.parse_series_matrix(path, with_matrix=False)
            key, values = label_field(s)
            print("\n%s  %s" % (acc, s.title[:70]))
            print("  platforms %s   n=%d   CHOSEN label key: %s"
                  % (", ".join(s.platforms) or "?", s.n_samples, key))
            for v, c in Counter(v.strip() for v in values if v.strip()).most_common(15):
                print("    %4d  %s  -> %s" % (c, v[:64], _class_of(v)))
            print("  all characteristics keys (audit):")
            for k, vals in all_keys(s)[:12]:
                kinds, classified = _key_score([v for v, _ in vals])
                print("    %-28s %s%s" % (
                    k[:28], ", ".join("%s(%d)" % (v[:26], c) for v, c in vals[:4]),
                    "   <- partitions" if kinds >= 2 else ""))
        return

    man = build(args.accessions, do_fetch=not args.no_fetch)
    for c in man["candidates"]:
        print("%-10s %-22s inf=%-4s ster=%-4s healthy=%-4s  %s"
              % (c["accession"], c.get("control_type", "?"), c.get("n_infectious", "-"),
                 c.get("n_sterile", "-"), c.get("n_healthy", "-"),
                 "ADMISSIBLE" if c.get("inclusion_ok") else
                 "excluded: " + str(c.get("exclusion_reason") or c.get("error", ""))))
    g = man["gate"]
    print("\n%s" % g["verdict"])
    print("  admissible: %s across %s" % (g["admissible"] or "none", g["platforms"] or "-"))

    if args.save:
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        with open(MANIFEST, "w") as fh:
            json.dump(json_safe(man), fh, indent=2)
        print("\nwrote %s" % MANIFEST)


if __name__ == "__main__":
    main()
