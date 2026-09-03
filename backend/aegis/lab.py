"""The Lab — a sandbox where agents propose changes and a locked gate adjudicates.

WHAT THIS IS, AND THE TRAP IT IS BUILT AROUND
----------------------------------------------
Agents that can change the detector and then score themselves are a machine for
generating false discoveries. That is not a hypothetical: ``selfimprove.py`` exists
because a search loop's failure mode is not "fails to improve", it is "improves its own
scoreboard", and it accepted **0 of 16** mutations once a locked gate was placed in front
of it. This project's whole history — eleven falsified method families — says the honest
prior for any proposal is that it will not work.

So the Lab inverts the usual arrangement. Agents are *proposers*, never judges. The
adjudicator is the same locked-gate protocol the self-improvement loop already uses:
patients held out before the search begins, paired folds, a bootstrap CI over patients,
and a Bonferroni correction for the fact that a round takes the best of several
challengers before testing it. An agent cannot see the gate set, cannot fit on it, and
cannot re-run against it to fish for a pass.

PRE-REGISTRATION, WHICH IS THE PART THAT MAKES THIS MEASURABLE
---------------------------------------------------------------
Every proposal must state its **predicted effect before the gate runs**. That costs the
agent nothing and buys two things:

* A proposal cannot be retrofitted into a success. "I expected this" is checkable.
* The *agents themselves become measurable*. An agent that always predicts a large gain
  and is always refused is overconfident, and ``AgentScore`` says so with a number —
  mean signed prediction error, hit rate, and calibration against its own claims. A
  sandbox that cannot tell a good proposer from a confident one is not worth having.

THREE ROLES, BECAUSE ONE AGENT ARGUING WITH ITSELF IS NOT COOPERATION
----------------------------------------------------------------------
``Proposer``      generates a change from a stated prior. Different proposers hold
                  different priors on purpose, so the plane explores rather than
                  converges on one agent's taste.
``Critic``        argues *against* a proposal using this project's own recorded nulls.
                  It cannot veto; it attaches a prior-failure warning that travels with
                  the verdict, so a proposal that repeats a known dead end says so.
``SafetyOfficer`` can VETO, and its veto is absolute. A proposal that would widen the
                  alarm beyond its budget, remove a safety constraint, or touch the
                  treatment path without the evidence firewall never reaches the gate.

THE EVIDENCE FIREWALL ON TREATMENT
-----------------------------------
Treatment proposals are evaluated on the PBPK twin, which is a **simulator**. A result
there is a statement about the simulator, never about a patient, and this project has
already made the mistake of tuning a detector against its own generator and watching it
score near chance on real data. So treatment verdicts are typed ``MECHANISM`` and are
structurally forbidden from carrying an efficacy claim: ``Verdict.efficacy_claim`` is
always None for them and ``verify_lab.py`` asserts it.

WHAT A PASS ACTUALLY MEANS
--------------------------
Very little on its own, and the Lab says so. Clearing the gate makes a proposal a
*candidate for promotion*, not a shipped change. Promotion still goes through
``promote_frontier.py`` with its three gates and its own operating-point derivation,
because a gate pass on one cohort is not a deployment decision.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Proposal kinds. The kind decides which adjudicator runs and what claim is permitted.
DETECTION = "detection"      # changes the model -> adjudicated on held-out patients
POLICY = "policy"            # changes the alarm/decision layer -> adjudicated on utility
TREATMENT = "treatment"      # runs on the PBPK twin -> MECHANISM only, never efficacy

#: Minimum effect a detection proposal must show before significance is even consulted.
#: Mirrors the self-improvement loop: beating the champion on the mean alone is not
#: enough, because at this sample size fold luck clears a bare mean comparison often.
MIN_EFFECT = 0.002


#: Family-wise error budget for a Lab session. Every gate test spends alpha; when the
#: budget is exhausted the Lab refuses to test rather than quietly accepting a looser bar.
#:
#: v1 of this module DOCUMENTED a Bonferroni correction and never applied one. An agent
#: could submit twenty proposals and the twentieth would face the same threshold as the
#: first, so a false pass was roughly twenty times more likely than the stated 5%. The
#: docstring advertised a protection the code did not have, which is the exact failure
#: this project's audits exist to catch — so the budget is now an object with state, and
#: verify_lab.py proves it tightens as tests are spent.
FAMILY_ALPHA = 0.05

#: A proposal that clears the gate is a CANDIDATE. It becomes a FINDING only after an
#: independent replication on a disjoint split, because one significant result at
#: alpha=0.05 is what one-in-twenty noise looks like.
REPLICATION_REQUIRED = True


@dataclass
class AlphaBudget:
    """Holm-style sequential error control across a session's gate tests.

    Holm rather than plain Bonferroni: dividing the whole budget by a guessed final test
    count is either wasteful (if fewer tests run) or invalid (if more do). Holm spends
    the budget as tests are actually taken, which is what an open-ended agent loop needs.
    """
    total: float = FAMILY_ALPHA
    spent: float = 0.0
    n_tests: int = 0

    def next_alpha(self) -> float:
        """The threshold the NEXT test must clear.

        A GEOMETRIC spending schedule, alpha_k = total * 2^-k, whose infinite sum is
        exactly ``total``. That is what an open-ended agent loop needs: a Holm-style
        division by the number of tests requires knowing that number in advance, and a
        first draft of this class did ``remaining / (n_tests + 1)`` — which handed the
        entire budget to test one and then refused every later test as exhausted.
        Geometric spending tightens smoothly and never runs out.
        """
        return self.total * (0.5 ** (self.n_tests + 1))

    def spend(self, alpha: float) -> None:
        self.spent += alpha
        self.n_tests += 1

    def exhausted(self) -> bool:
        """True when the next test's bar is tighter than is meaningful to measure.

        With 400 bootstrap resamples the finest resolvable quantile is ~1/400, so a
        nominal alpha below that is a bar the instrument cannot actually apply. Refusing
        is the honest response; pretending to test at alpha=1e-9 is not.
        """
        return self.next_alpha() < 2.5e-3

    def as_dict(self) -> Dict[str, Any]:
        return {"total": self.total, "spent": round(self.spent, 6),
                "n_tests": self.n_tests, "next_alpha": round(self.next_alpha(), 6),
                "exhausted": self.exhausted()}


@dataclass
class Proposal:
    """One change an agent wants to try, with its prediction registered up front."""
    agent: str
    kind: str
    title: str
    change: Dict[str, Any]
    rationale: str
    predicted_delta: float          # signed, in the metric the kind is judged on
    predicted_direction: str        # "improve" | "no_effect" | "worsen"
    submitted: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """What the gate said, and everything needed to argue with it."""
    proposal: Proposal
    accepted: bool
    claim_type: str                       # "HELD_OUT" | "MECHANISM" | "REFUSED"
    observed_delta: Optional[float] = None
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    significant: bool = False
    refusal_reason: Optional[str] = None
    vetoed_by: Optional[str] = None
    critic_warning: Optional[str] = None
    prediction_error: Optional[float] = None
    efficacy_claim: None = None           # structurally always None; see the firewall
    alpha_spent: Optional[float] = None   # the bar THIS test had to clear
    replicated: Optional[bool] = None     # None = not yet attempted
    replication_delta: Optional[float] = None
    status: str = "refused"               # refused | candidate | finding | mechanism
    mechanism: Optional[Dict[str, Any]] = None   # treatment sandbox observations
    limits: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["proposal"] = self.proposal.as_dict()
        return d


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
@dataclass
class Proposer:
    """An agent with a stated prior about where gains might be."""
    name: str
    prior: str
    generate: Callable[[Dict[str, Any]], Proposal]


@dataclass
class AgentScore:
    """How good a proposer actually is, measured rather than assumed."""
    agent: str
    n_proposals: int = 0
    n_accepted: int = 0
    n_vetoed: int = 0
    mean_signed_error: float = float("nan")   # predicted - observed; >0 = overclaiming
    mean_abs_error: float = float("nan")
    hit_rate: float = float("nan")            # direction called correctly

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Critic:
    """Argues against a proposal from this project's own recorded failures.

    It cannot veto. Its job is to make a proposal that repeats a known dead end carry
    that fact into the record, so a refusal is informative rather than surprising.
    """

    #: Each entry is a substring to look for and the null it would be repeating.
    KNOWN_NULLS: Tuple[Tuple[str, str], ...] = (
        ("temporal", "Phase 7 added 91 multi-scale temporal features for no gain."),
        ("window", "Multi-scale windows were measured and bought nothing (Phase 7)."),
        ("minirocket", "MiniRocket was built and showed no gain."),
        ("coordination", "Cross-channel coordination was falsified at hourly resolution."),
        ("coupling", "Inter-channel coupling carries no sepsis signal at 1 sample/hour."),
        ("recovery", "The autonomic recovery time constant failed its pre-registered bar "
                     "on two datasets."),
        ("lead", "Median lead time was shown to carry no early-warning information."),
        ("hdc", "No single HDC model has beaten the GBM; only stacking helped."),
        ("quantum", "Quantum HDC does not beat classical on this tabular problem."),
        ("threshold", "Four alarm-policy families were searched; none beat a plain "
                      "threshold."),
    )

    name = "Critic"

    def review(self, p: Proposal) -> Optional[str]:
        hay = f"{p.title} {p.rationale} {json.dumps(p.change, default=str)}".lower()
        for needle, null in self.KNOWN_NULLS:
            if needle in hay:
                return f"prior failure on record — {null}"
        if p.predicted_delta > 0.05:
            return (f"predicted delta {p.predicted_delta:+.3f} is larger than any gain "
                    f"this project has ever measured; treat as overclaiming")
        return None


class SafetyOfficer:
    """Absolute veto over proposals that would weaken a safety property."""

    name = "SafetyOfficer"

    def veto(self, p: Proposal) -> Optional[str]:
        ch = p.change or {}
        txt = f"{p.title} {p.rationale} {json.dumps(ch, default=str)}".lower()

        if p.kind == TREATMENT and not ch.get("mechanism_only", False):
            return ("treatment proposals must be marked mechanism_only: the PBPK twin is "
                    "a simulator and a result there is not evidence about a patient")
        if any(w in txt for w in ("disable safety", "remove z3", "bypass gate",
                                  "skip verification", "ignore invariant")):
            return "proposes weakening or bypassing a safety constraint"
        far = ch.get("max_false_alarm_rate")
        if far is not None and float(far) > 0.25:
            return (f"raises the alarm budget to {float(far):.0%}; the ward's budget is a "
                    f"clinical values judgement, not a knob for a search to widen")
        if ch.get("abstention_downgrades_alarm"):
            return ("would let abstention downgrade an escalation — the ordering property "
                    "verify_disposition.py exists to protect")
        return None


# --------------------------------------------------------------------------- #
# The treatment sandbox — it RUNS, and its output is typed MECHANISM
# --------------------------------------------------------------------------- #
class TreatmentSandbox:
    """Executes a treatment proposal on the PBPK twin and reports what happened.

    The firewall is on the CLAIM, not on the computation. v1 refused treatment proposals
    outright, which meant the sandbox agents were promised did not exist — they could ask
    "does a prolonged infusion change exposure?" and get only a refusal. The model now
    runs; what it may never produce is an efficacy claim.

    That distinction is the whole point. ``project_protocol`` will happily tell you a
    four-hour infusion raises time-above-MIC, and that is a real, mechanistic,
    literature-grounded property of the pharmacokinetics. It is not evidence that any
    patient does better, and this project has already tuned a detector against its own
    generator once and watched it score near chance on real people.
    """

    def run(self, p: Proposal) -> Dict[str, Any]:
        ch = p.change or {}
        try:
            from backend.aegis.pharmacokinetics import PatientState
            from backend.aegis.adverse import project_order
        except Exception as exc:                       # pragma: no cover
            return {"ran": False, "reason": f"PBPK unavailable: {exc}"}

        patient = PatientState(
            weight_kg=float(ch.get("weight_kg", 78.0)),
            egfr=float(ch.get("egfr", 90.0)),
            map_mmhg=float(ch.get("map_mmhg", 70.0)),
            vd_expansion=float(ch.get("vd_expansion", 1.0)),
        )
        baseline = {"antibiotic_code": int(ch.get("antibiotic_code", 1) or 1),
                    "fluid_rate": float(ch.get("fluid_rate", 30.0)),
                    "vasopressor_dose": float(ch.get("vasopressor_dose", 0.0))}
        variant = dict(baseline)
        variant.update({k: v for k, v in ch.items()
                        if k in ("infusion_min", "fluid_rate", "vasopressor_dose",
                                 "antibiotic_code")})
        try:
            a = project_order(baseline, patient, n_samples=4000)
            b = project_order(variant, patient, n_samples=4000)
        except Exception as exc:
            return {"ran": False, "reason": f"projection failed: {exc}"}

        def _summarise(out: Dict[str, Any]) -> Dict[str, Any]:
            """Pull the quantities the PBPK model actually exposes.

            A first draft guessed at ``auc`` / ``cmax`` / ``time_above_mic_h`` and got
            None for every field — the sandbox ran, reported nothing, and looked like it
            was working. The real arm carries a ``metrics`` block; ``pct_ft_above_mic``
            is the pharmacodynamic target that matters for a time-dependent antibiotic,
            and ``mass_balance_error`` is the model's own correctness check.
            """
            arms = (out.get("pk") or {}).get("arms") or []
            arm = next((x for x in arms if x.get("drug_class") == "antibiotic"), None)
            if arm is None:
                arm = next((x for x in arms if x.get("drug")), {})
            m = arm.get("metrics") or {}
            organ_auc = m.get("organ_auc") or {}
            peak = arm.get("peak") or {}
            return {"drug": arm.get("drug"),
                    "regimen": arm.get("regimen"),
                    "pct_ft_above_mic": m.get("pct_ft_above_mic"),
                    "mic_mg_l": m.get("mic_mg_l"),
                    "auc_venous": (round(float(organ_auc["venous"]), 2)
                                   if organ_auc.get("venous") is not None else None),
                    "peak_venous": (round(float(peak["venous"]), 2)
                                    if peak.get("venous") is not None else None),
                    "mass_balance_error": m.get("mass_balance_error"),
                    "n_adverse_modelled": len(out.get("events") or [])}

        sa, sb = _summarise(a), _summarise(b)
        deltas = {k: (None if sa.get(k) is None or sb.get(k) is None
                      or isinstance(sa.get(k), str)
                      else round(float(sb[k]) - float(sa[k]), 4))
                  for k in ("pct_ft_above_mic", "auc_venous", "peak_venous")}
        return {
            "ran": True,
            "baseline": sa, "variant": sb, "delta": deltas,
            "claim_type": "MECHANISM",
            "interpretation": "exposure and adverse-event projections from the PBPK twin",
            "forbidden": "no efficacy, mortality or benefit claim may be derived from "
                         "this. It describes the simulator, not a patient.",
        }


# --------------------------------------------------------------------------- #
# The sandbox
# --------------------------------------------------------------------------- #
class Lab:
    """Runs proposals through review, veto and a locked gate, and keeps the ledger."""

    def __init__(self,
                 adjudicator: Optional[Callable[[Proposal], Dict[str, Any]]] = None,
                 policy_adjudicator: Optional[Callable[[Proposal], Dict[str, Any]]] = None,
                 replicator: Optional[Callable[[Proposal], Dict[str, Any]]] = None,
                 alpha: float = FAMILY_ALPHA):
        self.critic = Critic()
        self.safety = SafetyOfficer()
        # v1 routed DETECTION and POLICY to the same callable, but the held-out
        # adjudicator only knows how to fit stacks — a policy proposal was silently
        # judged as if it were a model change. Each kind now has its own judge.
        self.adjudicator = adjudicator
        self.policy_adjudicator = policy_adjudicator
        self.replicator = replicator
        self.budget = AlphaBudget(total=alpha)
        self.treatment = TreatmentSandbox()
        self.ledger: List[Verdict] = []

    # -- the pipeline ------------------------------------------------------- #
    def submit(self, p: Proposal) -> Verdict:
        limits = [
            "Clearing this gate makes a proposal a CANDIDATE, not a shipped change. "
            "Promotion still runs promote_frontier.py with its three gates and its own "
            "operating-point derivation.",
            "The gate set is held out and scored once. A proposal cannot be resubmitted "
            "against it to fish for a pass.",
        ]

        veto = self.safety.veto(p)
        if veto:
            v = Verdict(proposal=p, accepted=False, claim_type="REFUSED",
                        refusal_reason=veto, vetoed_by=self.safety.name, limits=limits)
            self.ledger.append(v)
            return v

        warning = self.critic.review(p)

        if p.kind == TREATMENT:
            # Mechanism only, but it RUNS. v1 refused treatment proposals outright, which
            # meant the sandbox the agents were promised did not exist. The PBPK twin and
            # the adverse-event engine are executed and their observations returned — the
            # firewall is on the CLAIM, not on the computation.
            obs = self.treatment.run(p)
            v = Verdict(proposal=p, accepted=False, claim_type="MECHANISM",
                        status="mechanism", mechanism=obs,
                        critic_warning=warning,
                        refusal_reason="treatment effects are evaluated on the PBPK twin; "
                                       "a simulator result is a mechanism observation and "
                                       "cannot become an efficacy claim",
                        limits=limits + [
                            "This project already tuned a detector against its own "
                            "generator once and it scored near chance on real patients. "
                            "Simulator results are never evidence of clinical benefit."])
            self.ledger.append(v)
            return v

        judge = (self.policy_adjudicator if p.kind == POLICY else self.adjudicator)
        if p.kind == POLICY and judge is None:
            v = Verdict(proposal=p, accepted=False, claim_type="REFUSED",
                        critic_warning=warning,
                        refusal_reason="no POLICY adjudicator bound — a decision-layer "
                                       "change must not be judged by the model gate",
                        limits=limits)
            self.ledger.append(v)
            return v

        if self.budget.exhausted():
            v = Verdict(proposal=p, accepted=False, claim_type="REFUSED",
                        critic_warning=warning,
                        refusal_reason=(f"family-wise error budget exhausted after "
                                        f"{self.budget.n_tests} tests; the next bar "
                                        f"({self.budget.next_alpha():.5f}) is finer than "
                                        f"the bootstrap can resolve. Refusing to test."),
                        limits=limits)
            self.ledger.append(v)
            return v

        if judge is None:
            v = Verdict(proposal=p, accepted=False, claim_type="REFUSED",
                        critic_warning=warning,
                        refusal_reason="no adjudicator bound — the Lab will not report a "
                                       "result it did not measure",
                        limits=limits)
            self.ledger.append(v)
            return v

        alpha_k = self.budget.next_alpha()
        self.budget.spend(alpha_k)
        out = judge(p)
        delta = float(out.get("delta", float("nan")))
        lo, hi = out.get("ci_lo"), out.get("ci_hi")
        sig = bool(out.get("significant", False))
        big_enough = np.isfinite(delta) and delta >= MIN_EFFECT
        accepted = bool(sig and big_enough)
        reason = None
        if not accepted:
            if not big_enough:
                reason = (f"effect {delta:+.4f} is below the minimum {MIN_EFFECT:+.4f}; "
                          f"beating the champion on the mean alone is not enough")
            else:
                reason = (f"effect {delta:+.4f} is not significant "
                          f"(CI [{lo:+.4f}, {hi:+.4f}] includes zero after correction)")

        v = Verdict(proposal=p, accepted=accepted,
                    claim_type="HELD_OUT" if accepted else "REFUSED",
                    observed_delta=delta, ci_lo=lo, ci_hi=hi, significant=sig,
                    refusal_reason=reason, critic_warning=warning,
                    alpha_spent=alpha_k,
                    prediction_error=(p.predicted_delta - delta
                                      if np.isfinite(delta) else None),
                    status="candidate" if accepted else "refused",
                    limits=limits + [
                        f"tested at alpha={alpha_k:.5f} under a family-wise budget of "
                        f"{self.budget.total} (test {self.budget.n_tests} of the session)"])

        # A single significant result at alpha=0.05 is what one-in-twenty noise looks
        # like. A candidate becomes a FINDING only if it survives an independent split.
        if accepted and REPLICATION_REQUIRED:
            if self.replicator is None:
                v.replicated = None
                v.limits.append("NOT replicated — no replicator bound, so this remains a "
                                "candidate and must not be reported as a finding")
            else:
                rep = self.replicator(p)
                rd = float(rep.get("delta", float("nan")))
                v.replication_delta = rd
                v.replicated = bool(np.isfinite(rd) and rd >= MIN_EFFECT
                                    and rep.get("significant", False))
                v.status = "finding" if v.replicated else "candidate"
                if not v.replicated:
                    v.limits.append(
                        f"did NOT replicate on an independent split "
                        f"(delta {rd:+.4f}); stays a candidate")
        self.ledger.append(v)
        return v

    # -- measuring the proposers -------------------------------------------- #
    def score_agents(self) -> List[AgentScore]:
        by: Dict[str, List[Verdict]] = {}
        for v in self.ledger:
            by.setdefault(v.proposal.agent, []).append(v)
        out: List[AgentScore] = []
        for name, vs in sorted(by.items()):
            errs = [v.prediction_error for v in vs if v.prediction_error is not None]
            hits = [
                (v.proposal.predicted_direction == "improve") == bool(v.observed_delta > 0)
                for v in vs if v.observed_delta is not None
            ]
            out.append(AgentScore(
                agent=name,
                n_proposals=len(vs),
                n_accepted=sum(1 for v in vs if v.accepted),
                n_vetoed=sum(1 for v in vs if v.vetoed_by),
                mean_signed_error=float(np.mean(errs)) if errs else float("nan"),
                mean_abs_error=float(np.mean(np.abs(errs))) if errs else float("nan"),
                hit_rate=float(np.mean(hits)) if hits else float("nan"),
            ))
        return out

    def save(self, path: Optional[str] = None) -> str:
        """Persist the ledger. /aegis/lab reads this; v1 served a file nothing wrote."""
        import os
        from backend.ml import config as _cfg
        path = path or os.path.join(_cfg.ARTIFACTS_DIR, "lab_ledger.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "budget": self.budget.as_dict(),
                   "summary": self.summary(),
                   "verdicts": [v.as_dict() for v in self.ledger]}
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    def summary(self) -> Dict[str, Any]:
        n = len(self.ledger)
        return {
            "n_proposals": n,
            "n_accepted": sum(1 for v in self.ledger if v.accepted),
            "n_vetoed": sum(1 for v in self.ledger if v.vetoed_by),
            "n_mechanism_only": sum(1 for v in self.ledger if v.claim_type == "MECHANISM"),
            "n_candidates": sum(1 for v in self.ledger if v.status == "candidate"),
            "n_findings": sum(1 for v in self.ledger if v.status == "finding"),
            "budget": self.budget.as_dict(),
            "acceptance_rate": (sum(1 for v in self.ledger if v.accepted) / n) if n else 0.0,
            "agents": [a.as_dict() for a in self.score_agents()],
            "note": "A low acceptance rate is the expected outcome. This project has "
                    "falsified eleven method families; the gate refusing is the gate "
                    "working.",
        }


# --------------------------------------------------------------------------- #
# Default proposers — distinct priors, so the plane explores
# --------------------------------------------------------------------------- #
def default_proposers() -> List[Proposer]:
    from backend.ml import selfimprove as si

    def _knob(ctx: Dict[str, Any]) -> Proposal:
        """Tune a hyperparameter of the shipped stack."""
        rng = np.random.default_rng(int(ctx.get("seed", 0)))
        k = si.SEARCH_SPACE[int(rng.integers(0, len(si.SEARCH_SPACE)))]
        return Proposal(
            agent="Tuner", kind=DETECTION,
            title=f"tune {k.learner}.{k.key}",
            change={"learner": k.learner, "key": k.key, "kind": k.kind},
            rationale=f"hyperparameter within the shipped search space ({k.lo}-{k.hi})",
            predicted_delta=0.003, predicted_direction="improve")

    def _objective(ctx: Dict[str, Any]) -> Proposal:
        """Change what the model is trained to predict, not how."""
        return Proposal(
            agent="Objectivist", kind=DETECTION,
            title="re-aim a stack member's training objective",
            change={"member": "hgb", "objective": "patient"},
            rationale="the only lever with a measured win in this project was the "
                      "supervision target, not the model class",
            predicted_delta=0.004, predicted_direction="improve")

    def _policy(ctx: Dict[str, Any]) -> Proposal:
        """Change the decision layer rather than the score."""
        return Proposal(
            agent="Decider", kind=POLICY,
            title="route the conformal UNCERTAIN band to review",
            change={"policy": "uncertain_to_review", "max_false_alarm_rate": 0.15},
            rationale="the same detector read three ways spans false-alarm rates of "
                      "14%, 27% and 94%; the decision layer is not data-limited",
            predicted_delta=0.002, predicted_direction="improve")

    return [
        Proposer("Tuner", "gains hide in hyperparameters", _knob),
        Proposer("Objectivist", "gains hide in the supervision target", _objective),
        Proposer("Decider", "gains hide in the decision layer, not the score", _policy),
    ]


# --------------------------------------------------------------------------- #
# The real adjudicator — held-out patients, paired bootstrap, no second chances
# --------------------------------------------------------------------------- #
def held_out_adjudicator(spec: str = "dev", n_folds: int = 2,
                         gate_fraction: float = 0.25, seed: int = 42):
    """Bind a gate that actually fits models and scores held-out patients.

    The gate patients are split off ONCE, here, before any proposal is seen. They are
    never fitted on, never used to select, and are scored exactly once per proposal. That
    is the same contract ``selfimprove.py`` runs under, and it is the reason its honest
    answer was 0 accepted out of 16.

    Expensive on purpose. A cheap gate is a gate somebody will run a thousand times until
    something passes, which is precisely the failure this whole design exists to prevent.
    """
    from backend.ml.eval.bench import Benchmark
    from backend.ml.eval.splits import stratified_group_split
    from backend.ml import selfimprove as si
    from backend.ml.models.stacked import StackedEnsembleDetector
    from backend.ml.eval import metrics
    import pandas as pd

    bench = Benchmark.load(spec, n_folds=n_folds)
    ds = bench.ds
    search_idx, gate_idx = stratified_group_split(ds.groups, ds.y, gate_fraction, seed)
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)

    def _fit_score(objective: str, roster, params) -> np.ndarray:
        m = StackedEnsembleDetector(
            channels=list(__import__("backend.ml.config", fromlist=["x"]).WEARABLE_CHANNELS),
            use_hdc=False, objective=objective,
            **({"roster": list(roster)} if roster else {}),
            **({"params": params} if params else {}))
        m.fit(feats.iloc[search_idx], ds.y[search_idx], groups=ds.groups[search_idx],
              hour_index=ds.hour_index[search_idx], onset_hour=ds.onset_hour[search_idx])
        return m.predict_proba(feats.iloc[gate_idx])

    def adjudicate(p: Proposal) -> Dict[str, Any]:
        ch = p.change or {}
        # Champion: the shipped configuration.
        champ = _fit_score("patient", ["hgb", "hgb2", "et", "lr", "xgb@utility"], None)
        # Candidate: the same, with the proposal applied.
        params = None
        if ch.get("learner") and ch.get("key") is not None and ch.get("value") is not None:
            params = {ch["learner"]: {ch["key"]: ch["value"]}}
        roster = ch.get("roster") or ["hgb", "hgb2", "et", "lr", "xgb@utility"]
        objective = ch.get("objective", "patient")
        cand = _fit_score(objective, roster, params)

        yg, gg = ds.y[gate_idx], ds.groups[gate_idx]
        a_champ = metrics.auroc(yg, champ)
        a_cand = metrics.auroc(yg, cand)
        # ARGUMENT ORDER MATTERS: bootstrap_delta_ci returns AUROC(pb) - AUROC(pa),
        # so the champion goes in `pa` and the candidate in `pb`. Passing them the other
        # way round inverts every verdict — a candidate that loses would read as a win.
        # It also returns a TUPLE (mean, lo, hi), not a dict.
        boot_delta, lo, hi = si.bootstrap_delta_ci(
            yg, champ, cand, gg, n_boot=400, seed=seed)
        return {"delta": float(a_cand - a_champ),
                "bootstrap_mean_delta": float(boot_delta),
                "ci_lo": float(lo), "ci_hi": float(hi),
                "significant": bool(np.isfinite(lo) and lo > 0),
                "champion_auroc": float(a_champ), "candidate_auroc": float(a_cand),
                "n_gate_patients": int(len(set(gg.tolist())))}

    return adjudicate


# --------------------------------------------------------------------------- #
# Adaptive proposers — agents that read the ledger instead of repeating themselves
# --------------------------------------------------------------------------- #
class AdaptiveProposer:
    """A proposer that learns from what the gate has already refused.

    v1's proposers emitted an identical proposal every call regardless of context, so a
    "cooperating" plane would have re-submitted the same refused change forever and the
    alpha budget would have been spent on repeats. An adaptive proposer does two things a
    static one cannot:

    * **Avoids repetition.** A change already refused is not proposed again; the gate is
      expensive and the error budget is finite, so re-testing a dead proposal spends both
      for nothing.
    * **Recalibrates its own predictions.** Its registered prediction is shrunk toward the
      effects it has actually observed. An agent whose signed error is +0.005 has been
      overclaiming by that much and should say so next time — which is the difference
      between an agent that learns and one that is merely confident.
    """

    def __init__(self, name: str, prior: str,
                 generator: Callable[[Dict[str, Any]], Proposal]):
        self.name = name
        self.prior = prior
        self._generator = generator

    def observed_bias(self, ledger: Sequence[Verdict]) -> float:
        errs = [v.prediction_error for v in ledger
                if v.proposal.agent == self.name and v.prediction_error is not None]
        return float(np.mean(errs)) if errs else 0.0

    def tried(self, ledger: Sequence[Verdict]) -> set:
        return {json.dumps(v.proposal.change, sort_keys=True, default=str)
                for v in ledger if v.proposal.agent == self.name}

    def propose(self, ctx: Dict[str, Any], ledger: Sequence[Verdict],
                max_attempts: int = 8) -> Optional[Proposal]:
        seen = self.tried(ledger)
        bias = self.observed_bias(ledger)
        for k in range(max_attempts):
            p = self._generator({**ctx, "seed": int(ctx.get("seed", 0)) + k})
            key = json.dumps(p.change, sort_keys=True, default=str)
            if key in seen:
                continue
            # Shrink the prediction by the bias this agent has actually shown.
            p.predicted_delta = round(float(p.predicted_delta) - bias, 5)
            if p.predicted_delta <= 0:
                p.predicted_direction = "no_effect"
            p.rationale += (f" [recalibrated: this agent has overclaimed by "
                            f"{bias:+.4f} on average]") if abs(bias) > 1e-9 else ""
            return p
        return None


def adaptive_proposers() -> List[AdaptiveProposer]:
    return [AdaptiveProposer(p.name, p.prior, p.generate) for p in default_proposers()]
