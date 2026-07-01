"""Epidemiological pattern mining over the cohort of causal graphs (MEGA-CONCEPT 1-B).

Across many patients, the per-patient causal graphs form a clinical knowledge
network. This module mines that network for statements of the form:

    "When MAP<65 AND lactate>4 AND no antibiotics within 2h, organ failure
     followed within 4h in 71% of cases (n=24, 95% CI 51–85%, lift 2.3×)."

The scientific honesty here is the entire point — and it is where the original
"discover clinical guidelines from formal causal evidence" framing is dangerous:

  * **These are ASSOCIATIONS, not causation.** Every rule is stamped
    ``observed_association`` (the same firewall class as the edges). The kernel
    does not — and this module must never — present a mined rule as a proven
    mechanism or a guideline. Confounding by indication is severe in sepsis
    (sicker patients are treated differently), so a high co-occurrence rate is
    *evidence to investigate*, not a causal claim. That caveat rides on output.

  * **A raw rate is meaningless without a baseline.** "94% had organ failure"
    says nothing if 94% of the whole cohort did. So every rule carries the cohort
    **baseline** rate and the **lift** (confidence / baseline). Lift≈1 ⇒ the
    antecedent carries no information; only lift≫1 is interesting.

  * **Small N must not overclaim.** Confidence is reported with a Wilson 95%
    interval and gated by a minimum support count; a rule seen in 3 patients is
    shown with a wide interval, not a confident percentage.

  * **Multiple comparisons are disclosed.** Mining K candidate antecedents and
    reporting the best inflates false positives; the result reports how many
    candidates were tested so the reader can judge nominal CIs accordingly.

The cohort is the set of persisted patient graphs (``CausalGraphStore.load_all``);
mining grows more powerful as real recorded histories accumulate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from backend.aegis.kernel.causal_graph import (
    DECISION, DRUG, OUTCOME, VITAL, TemporalCausalGraph)

H = 3600.0   # seconds per hour


# --------------------------------------------------------------------------- #
# Predicate vocabulary over a vital snapshot's payload.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    label: str
    test: Callable[[Dict[str, Any]], bool]


def _get(p: Dict[str, Any], *keys) -> Optional[float]:
    for k in keys:
        if p.get(k) is not None:
            return float(p[k])
    return None


def _lt(key, thr, *alts):
    return lambda p: (_get(p, key, *alts) is not None) and _get(p, key, *alts) < thr


def _gt(key, thr, *alts):
    return lambda p: (_get(p, key, *alts) is not None) and _get(p, key, *alts) > thr


# A curated base of clinically meaningful conditions (the feature space we mine).
BASE_CONDITIONS: List[Condition] = [
    Condition("MAP<65", _lt("map", 65.0)),
    Condition("lactate>4", _gt("lactate", 4.0, "sweat_lactate")),
    Condition("lactate>2", _gt("lactate", 2.0, "sweat_lactate")),
    Condition("HR>100", _gt("heart_rate", 100.0, "hr")),
    Condition("RR>22", _gt("respiratory_rate", 22.0, "rr")),
    Condition("SpO2<92", _lt("spo2", 92.0)),
]


@dataclass(frozen=True)
class Antecedent:
    """A condition set that must co-hold on one vital snapshot, optionally with the
    requirement that no antibiotic was given within ``untreated_h`` hours after."""
    conditions: List[Condition]
    untreated_h: Optional[float] = None

    @property
    def label(self) -> str:
        base = " ∧ ".join(c.label for c in self.conditions)
        return base + (f" ∧ no-abx<{self.untreated_h:g}h" if self.untreated_h else "")


@dataclass
class AssociationRule:
    antecedent: str
    consequent: str                 # e.g. "organ_failure within 4h"
    n_antecedent: int               # support: patients matching the antecedent
    n_both: int                     # of those, how many had the consequent
    confidence: float               # P(consequent | antecedent)
    ci_low: float
    ci_high: float
    baseline: float                 # P(consequent) across the cohort
    lift: float                     # confidence / baseline
    n_cohort: int
    evidence_type: str = "observed_association"
    caveat: str = ("Observed association across recorded cases — NOT causation. "
                   "Subject to confounding by indication; interpret as a signal to "
                   "investigate, never as a guideline.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "antecedent": self.antecedent, "consequent": self.consequent,
            "n_antecedent": self.n_antecedent, "n_both": self.n_both,
            "confidence": round(self.confidence, 4),
            "ci_low": round(self.ci_low, 4), "ci_high": round(self.ci_high, 4),
            "baseline": round(self.baseline, 4), "lift": round(self.lift, 3),
            "n_cohort": self.n_cohort, "evidence_type": self.evidence_type,
            "caveat": self.caveat,
        }


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- #
class PatternMiner:
    """Mines association rules across a cohort of patient causal graphs."""

    def __init__(self, graphs: List[TemporalCausalGraph],
                 outcome_state: str = "organ_failure",
                 outcome_window_h: float = 4.0):
        self.graphs = graphs
        self.outcome_state = outcome_state
        self.outcome_window_h = outcome_window_h

    # -- per-graph helpers ---------------------------------------------- #
    @staticmethod
    def _is_antibiotic(node) -> bool:
        if node.kind not in (DRUG, DECISION):
            return False
        p = node.payload or {}
        proto = p.get("protocol") or p
        code = proto.get("antibiotic_code", proto.get("antibiotic", 0))
        try:
            return int(code) > 0
        except (TypeError, ValueError):
            return bool(code)

    def _match_time(self, g: TemporalCausalGraph, ante: Antecedent) -> Optional[float]:
        """Time of the antecedent's first onset in this patient, else None.

        The ``untreated_h`` requirement is evaluated at that *onset* — if an
        antibiotic was given within the window of the first deranged state, the
        patient counts as treated and does NOT match (we do not hunt for a later
        uncovered window, which would spuriously re-include treated patients)."""
        vitals = sorted((n for n in g._nodes.values() if n.kind == VITAL),
                        key=lambda n: n.t)
        onset = next((v for v in vitals
                      if all(c.test(v.payload) for c in ante.conditions)), None)
        if onset is None:
            return None
        if ante.untreated_h is not None:
            w = ante.untreated_h * H
            abx_times = [n.t for n in g._nodes.values() if self._is_antibiotic(n)]
            if any(onset.t <= t <= onset.t + w for t in abx_times):
                return None     # treated within the window ⇒ not an "untreated" case
        return onset.t

    def _has_outcome(self, g: TemporalCausalGraph, t_a: float) -> bool:
        w = self.outcome_window_h * H
        return any(n.kind == OUTCOME
                   and (n.payload or {}).get("state") == self.outcome_state
                   and t_a < n.t <= t_a + w
                   for n in g._nodes.values())

    def _ever_outcome(self, g: TemporalCausalGraph) -> bool:
        return any(n.kind == OUTCOME and (n.payload or {}).get("state") == self.outcome_state
                   for n in g._nodes.values())

    # -- evaluation ------------------------------------------------------ #
    def evaluate(self, ante: Antecedent) -> Optional[AssociationRule]:
        n_cohort = len(self.graphs)
        if n_cohort == 0:
            return None
        baseline = sum(self._ever_outcome(g) for g in self.graphs) / n_cohort

        n_ante = n_both = 0
        for g in self.graphs:
            t_a = self._match_time(g, ante)
            if t_a is None:
                continue
            n_ante += 1
            if self._has_outcome(g, t_a):
                n_both += 1
        if n_ante == 0:
            return None
        conf = n_both / n_ante
        lo, hi = _wilson(n_both, n_ante)
        lift = (conf / baseline) if baseline > 0 else float("inf")
        return AssociationRule(
            antecedent=ante.label,
            consequent=f"{self.outcome_state} within {self.outcome_window_h:g}h",
            n_antecedent=n_ante, n_both=n_both, confidence=conf,
            ci_low=lo, ci_high=hi, baseline=baseline, lift=lift, n_cohort=n_cohort)

    # -- mining ---------------------------------------------------------- #
    def default_candidates(self) -> List[Antecedent]:
        """Singles, key pairs, and the headline untreated-perfusion-failure triple."""
        c = {x.label: x for x in BASE_CONDITIONS}
        cands = [Antecedent([x]) for x in BASE_CONDITIONS]
        pairs = [("MAP<65", "lactate>4"), ("HR>100", "RR>22"), ("MAP<65", "SpO2<92")]
        cands += [Antecedent([c[a], c[b]]) for a, b in pairs]
        # the clinically central rule: deranged perfusion left untreated
        cands.append(Antecedent([c["MAP<65"], c["lactate>4"]], untreated_h=2.0))
        return cands

    def mine(self, candidates: Optional[List[Antecedent]] = None,
             min_support: int = 5, min_lift: float = 1.2) -> Dict[str, Any]:
        cands = candidates or self.default_candidates()
        tested = 0
        rules: List[AssociationRule] = []
        for ante in cands:
            tested += 1
            r = self.evaluate(ante)
            if r is None:
                continue
            if r.n_antecedent >= min_support and r.lift >= min_lift:
                rules.append(r)
        rules.sort(key=lambda r: (r.lift, r.n_antecedent), reverse=True)
        return {
            "cohort_size": len(self.graphs),
            "outcome": f"{self.outcome_state} within {self.outcome_window_h:g}h",
            "candidates_tested": tested,
            "min_support": min_support, "min_lift": min_lift,
            "rules": [r.to_dict() for r in rules],
            "evidence_type": "observed_association",
            "disclaimer": ("Associations mined across recorded cases — NOT causal "
                           "guidelines. Confounded by indication; nominal CIs are not "
                           f"corrected for the {tested} candidates tested."),
        }


# --------------------------------------------------------------------------- #
# Standalone demo: a cohort where the untreated-perfusion-failure pattern is real.
# --------------------------------------------------------------------------- #
def _demo() -> None:
    import random
    from backend.aegis.causal_ingest import CausalGraphIngestor
    from types import SimpleNamespace as A

    rng = random.Random(7)
    graphs = []
    for i in range(60):
        ing = CausalGraphIngestor(bed=f"p{i}", vital_every_s=60, omission_after_s=180)
        deranged = rng.random() < 0.4
        treated = rng.random() < 0.5
        t0 = 0.0
        for j in range(40):
            t = t0 + j * 60
            if deranged and j >= 10:
                vit = {"map": 58, "lactate": 5.0, "heart_rate": 118, "respiratory_rate": 26, "spo2": 90}
                band, mp = "HIGH", 0.8
            else:
                vit = {"map": 82, "lactate": 1.5, "heart_rate": 78, "respiratory_rate": 16, "spo2": 98}
                band, mp = "LOW", 0.08
            proto = ({"antibiotic_code": 3} if (deranged and treated and j == 12) else None)
            ver = ({"safe": True, "z3_status": "unsat"} if proto else None)
            ing.ingest(A(risk_level=band, ml_probability=mp, detection={}, protocol=proto,
                         verification=ver, signal_quality=1.0, triggered_rules=[]), vit, t)
        # untreated deranged patients mostly fail; treated/healthy mostly don't
        fails = (deranged and not treated and rng.random() < 0.75) or (not deranged and rng.random() < 0.05)
        if fails:
            ing.mark_outcome("organ_failure", t0 + 41 * 60, {"map": 50})
        graphs.append(ing.graph)

    res = PatternMiner(graphs).mine(min_support=4)
    print(f"cohort={res['cohort_size']} outcome={res['outcome']} tested={res['candidates_tested']}")
    for r in res["rules"]:
        print(f"  {r['antecedent']:38} → {int(r['confidence']*100)}% "
              f"(n={r['n_antecedent']}, CI {int(r['ci_low']*100)}–{int(r['ci_high']*100)}%, "
              f"base {int(r['baseline']*100)}%, lift {r['lift']}×)")


if __name__ == "__main__":
    _demo()
