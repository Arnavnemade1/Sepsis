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
    #: Tests this session declared it would run, BEFORE running any. None = open-ended.
    planned_tests: Optional[int] = None

    #: With 400 bootstrap resamples the finest resolvable quantile is ~1/400. A nominal
    #: alpha below that is a bar the instrument cannot actually apply, and pretending to
    #: test at alpha=1e-9 would be a claim about arithmetic, not about patients.
    RESOLUTION_FLOOR: float = 2.5e-3

    def plan(self, n: int) -> "AlphaBudget":
        """Pre-declare how many tests this session will run. Once, before test one.

        WHY THIS EXISTS
        ---------------
        Geometric spending is what an *open-ended* loop needs, and it is badly wasteful
        when the count is actually known. It hands half the whole budget to test one, and
        by test four the bar is 0.00156 — under the bootstrap's resolution. A real
        campaign measured it: 8 proposals, 4 tested, and 4 refused for "budget exhausted"
        without ever being adjudicated.

        A campaign knows its ceiling — ``rounds x proposers`` — so it can pre-register the
        count and spend a flat ``total / n``. That is plain Bonferroni, it is uniformly
        more powerful than geometric spending for a known count, and it is valid *only*
        because the count is fixed in advance and cannot be revised.

        Which is the property this method actually protects. Raising the planned count
        after seeing a result is how a family-wise budget becomes a knob for widening the
        bar, so a plan may be declared once and only before any alpha has been spent.
        """
        if self.n_tests > 0 or self.spent > 0:
            raise ValueError(
                "the test count must be declared BEFORE any test is run. Revising it "
                "after seeing a result is how a family-wise budget becomes a knob.")
        if self.planned_tests is not None:
            raise ValueError("the test count is already declared and cannot be revised")
        if int(n) < 1:
            raise ValueError("a session must plan at least one test")
        self.planned_tests = int(n)
        return self

    def next_alpha(self) -> float:
        """The threshold the NEXT test must clear.

        With a pre-declared count: a flat ``total / n``, so every planned test faces the
        same bar and the budget is spent evenly rather than front-loaded.

        Without one: a GEOMETRIC schedule, alpha_k = total * 2^-k, whose infinite sum is
        exactly ``total``. A first draft used ``remaining / (n_tests + 1)``, which handed
        the entire budget to test one and refused every later test as exhausted.
        """
        if self.planned_tests:
            return self.total / float(self.planned_tests)
        return self.total * (0.5 ** (self.n_tests + 1))

    def spend(self, alpha: float) -> None:
        self.spent += alpha
        self.n_tests += 1

    def exhausted(self) -> bool:
        """True when there is no honest test left to run.

        Two ways to get there: the declared plan is used up, or the next bar is finer
        than the bootstrap can resolve. Refusing is the honest response to both.
        """
        if self.planned_tests is not None and self.n_tests >= self.planned_tests:
            return True
        return self.next_alpha() < self.RESOLUTION_FLOOR

    def schedule(self) -> str:
        return ("flat (pre-declared count)" if self.planned_tests else "geometric")

    def as_dict(self) -> Dict[str, Any]:
        return {"total": self.total, "spent": round(self.spent, 6),
                "n_tests": self.n_tests, "planned_tests": self.planned_tests,
                "schedule": self.schedule(),
                "next_alpha": round(self.next_alpha(), 6),
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
    #: The smallest true effect this test could have called significant. A refusal
    #: without it is a shrug; with it, a refusal is a measurement — it says which
    #: effects were ruled out and which were simply too small to see.
    mde: Optional[float] = None
    power_note: Optional[str] = None
    #: Whether the adjudicator actually APPLIED the alpha it was charged, or merely had
    #: it recorded. See Lab._call_judge — this was silently False for every test.
    alpha_applied: bool = False
    arm: Optional[str] = None             # which held-out pool scored it
    patient_ids_hash: Optional[str] = None
    replication_arm: Optional[str] = None
    replication_pool_hash: Optional[str] = None
    duplicate_of: Optional[str] = None    # a past campaign already tested this
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

    def __init__(self, registry: Optional[Any] = None):
        #: The Critic shipped arguing only from the project's PUBLISHED nulls, which are
        #: the minority of nulls this project owns. Most of them were produced by the Lab
        #: itself and were forgotten the moment a campaign ended, so an agent could
        #: rediscover a dead end the Lab had killed last week and nothing would say so.
        self.registry = registry

    def review(self, p: Proposal) -> Optional[str]:
        hay = f"{p.title} {p.rationale} {json.dumps(p.change, default=str)}".lower()
        for needle, null in self.KNOWN_NULLS:
            if needle in hay:
                return f"prior failure on record — {null}"
        near = self._near_miss(p)
        if near:
            return near
        if p.predicted_delta > 0.05:
            return (f"predicted delta {p.predicted_delta:+.3f} is larger than any gain "
                    f"this project has ever measured; treat as overclaiming")
        return None

    def _near_miss(self, p: Proposal) -> Optional[str]:
        """Has this Lab already refused something of the same shape?

        Not an exact-change match — that is the registry's dedup and it refuses outright.
        This is the softer signal: the same LEVER has been tried and produced nothing, so
        the record should say how many times before the gate spends alpha on it again.
        """
        if self.registry is None:
            return None
        lever = _lever_of(p.change)
        if lever is None:
            return None
        prior = [n for n in self.registry.recorded_nulls(limit=200)
                 if _lever_of(n.get("change")) == lever]
        if len(prior) < 2:
            return None
        best = max((n["delta"] for n in prior if n.get("delta") is not None),
                   default=None)
        return (f"this Lab has already refused {len(prior)} change(s) to the same lever "
                f"({lever})"
                + (f"; the best of them measured {best:+.4f}" if best is not None else "")
                + ". Not a veto — but the prior on this lever is poor.")


def _lever_of(change: Dict[str, Any]) -> Optional[str]:
    """The knob a change turns, coarser than the change itself."""
    ch = change or {}
    if ch.get("learner") and ch.get("key"):
        return f"{ch['learner']}.{ch['key']}"
    if ch.get("roster"):
        return "stack roster"
    if ch.get("policy"):
        return f"policy:{ch['policy']}"
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


def _fmt(x: Optional[float]) -> str:
    """Signed 4dp, or an em dash. A refusal message must not print 'None'."""
    try:
        f = float(x)
        return f"{f:+.4f}" if np.isfinite(f) else "—"
    except (TypeError, ValueError):
        return "—"


def _call_judge(judge: Callable, p: "Proposal", alpha: float) -> Tuple[Dict[str, Any], bool]:
    """Call an adjudicator, passing the test's alpha when it can accept one.

    THE BUG THIS EXISTS TO CLOSE
    ----------------------------
    ``AlphaBudget`` computed a geometric bar, spent it, recorded it in the verdict, and
    printed "tested at alpha=0.00625" in the limits. The adjudicator was then called with
    no alpha at all and computed a fixed 95% interval — ``bootstrap_delta_ci`` takes an
    ``alpha`` argument defaulting to 0.05 and the Lab never passed one. So the correction
    the Lab advertised was, for the third time in this module's history, documented and
    not applied.

    ``alpha_applied`` travels with the verdict so the distinction is visible rather than
    assumed: a judge that cannot take an alpha reports False, and the limits say the test
    ran at the adjudicator's own default.
    """
    import inspect
    try:
        sig = inspect.signature(judge)
        if "alpha" in sig.parameters:
            return judge(p, alpha=alpha), True
    except (TypeError, ValueError):
        pass
    return judge(p), False


def minimum_detectable_effect(ci_lo: Optional[float], ci_hi: Optional[float]) -> Optional[float]:
    """The smallest true effect this test could have called significant.

    The gate accepts on ``ci_lo > 0``, so an effect must exceed the half-width of its own
    interval before that interval can clear zero — and it must also clear ``MIN_EFFECT``.
    The MDE is therefore the larger of the two, and it needs no extra computation: it
    falls straight out of the interval the adjudicator already returned.

    This is what makes a refusal informative. "Refused" alone does not distinguish *this
    change does nothing* from *this test could not have seen it*, and eight refusals with
    no power statement are not a scientific result.
    """
    if ci_lo is None or ci_hi is None:
        return None
    try:
        lo, hi = float(ci_lo), float(ci_hi)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
        return None
    return float(max(MIN_EFFECT, (hi - lo) / 2.0))


def _power_note(delta: Optional[float], mde: Optional[float], accepted: bool) -> Optional[str]:
    """State plainly what the test ruled out, and what it could not have seen."""
    if mde is None or delta is None or not np.isfinite(delta):
        return None
    if accepted:
        return (f"this test could resolve effects down to {mde:+.4f}; the observed "
                f"{delta:+.4f} cleared it")
    if delta >= mde:
        return (f"observed {delta:+.4f} exceeds the resolvable {mde:+.4f} but the "
                f"interval still covers zero — an unstable effect, not a small one")
    return (f"rules out true effects at or above {mde:+.4f}. Effects smaller than that "
            f"were NOT testable here — this refusal is not evidence that the change "
            f"does nothing, only that it does nothing that large")


# --------------------------------------------------------------------------- #
# The sandbox
# --------------------------------------------------------------------------- #
class Lab:
    """Runs proposals through review, veto and a locked gate, and keeps the ledger."""

    def __init__(self,
                 adjudicator: Optional[Callable[[Proposal], Dict[str, Any]]] = None,
                 policy_adjudicator: Optional[Callable[[Proposal], Dict[str, Any]]] = None,
                 replicator: Optional[Callable[[Proposal], Dict[str, Any]]] = None,
                 alpha: float = FAMILY_ALPHA,
                 registry: Optional[Any] = None):
        #: The Lab's memory across campaigns. Without it the Lab forgets every result
        #: the moment a campaign ends, which makes both re-testing a dead hypothesis and
        #: campaign-shopping free. See lab_registry.LabRegistry.
        self.registry = registry
        self.critic = Critic(registry=registry)
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

        # A hypothesis is tested ONCE, ever. Re-testing a change a past campaign already
        # put through the gate spends alpha and gate time for a result already on the
        # record, and re-asking the same question until it passes is exactly how a search
        # loop manufactures a false discovery. This refusal costs no alpha.
        if self.registry is not None:
            prior = self.registry.seen(p.change)
            if prior is not None:
                v = Verdict(proposal=p, accepted=False, claim_type="REFUSED",
                            critic_warning=warning, duplicate_of=prior.campaign,
                            observed_delta=prior.observed_delta,
                            ci_lo=prior.ci_lo, ci_hi=prior.ci_hi, mde=prior.mde,
                            refusal_reason=(
                                f"already tested in campaign {prior.campaign} "
                                f"({prior.status}, observed "
                                f"{_fmt(prior.observed_delta)}). A hypothesis gets one "
                                f"test; asking again until it passes is the failure "
                                f"this protocol exists to prevent."),
                            limits=limits + ["no alpha was spent — this was not a new "
                                             "test, it was a repeat of an old one"])
                self.ledger.append(v)
                return v

        alpha_k = self.budget.next_alpha()
        self.budget.spend(alpha_k)
        out, alpha_applied = _call_judge(judge, p, alpha_k)
        delta = float(out.get("delta", float("nan")))
        lo, hi = out.get("ci_lo"), out.get("ci_hi")
        sig = bool(out.get("significant", False))
        big_enough = np.isfinite(delta) and delta >= MIN_EFFECT
        accepted = bool(sig and big_enough)
        mde = minimum_detectable_effect(lo, hi)
        reason = None
        if not accepted:
            if not big_enough:
                reason = (f"effect {delta:+.4f} is below the minimum {MIN_EFFECT:+.4f}; "
                          f"beating the champion on the mean alone is not enough")
            else:
                reason = (f"effect {delta:+.4f} is not significant "
                          f"(CI [{lo:+.4f}, {hi:+.4f}] includes zero after correction)")

        alpha_limit = (
            f"tested at alpha={alpha_k:.5f} under a family-wise budget of "
            f"{self.budget.total} (test {self.budget.n_tests} of the session)"
            if alpha_applied else
            f"alpha={alpha_k:.5f} was CHARGED to the budget but this adjudicator does "
            f"not accept an alpha, so the interval is at its own default. The correction "
            f"is accounted for, not applied — do not read this test as corrected.")

        v = Verdict(proposal=p, accepted=accepted,
                    claim_type="HELD_OUT" if accepted else "REFUSED",
                    observed_delta=delta, ci_lo=lo, ci_hi=hi, significant=sig,
                    refusal_reason=reason, critic_warning=warning,
                    alpha_spent=alpha_k, alpha_applied=alpha_applied,
                    mde=mde, power_note=_power_note(delta, mde, accepted),
                    arm=out.get("arm"), patient_ids_hash=out.get("patient_ids_hash"),
                    prediction_error=(p.predicted_delta - delta
                                      if np.isfinite(delta) else None),
                    status="candidate" if accepted else "refused",
                    limits=limits + [alpha_limit])

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
                v.replication_arm = rep.get("arm")
                v.replication_pool_hash = rep.get("patient_ids_hash")
                v.replicated = bool(np.isfinite(rd) and rd >= MIN_EFFECT
                                    and rep.get("significant", False))
                # A replication scored on the SAME patients is not a replication. The
                # policy arm was exactly that — the same 8000 patients twice, raw delta
                # bit-identical at +0.025037 — so a candidate replicated with probability
                # ~1. The pool fingerprints are compared rather than trusted.
                same_pool = (v.patient_ids_hash is not None
                             and v.patient_ids_hash == v.replication_pool_hash)
                if same_pool:
                    v.replicated = False
                    v.status = "candidate"
                    v.limits.append(
                        "REPLICATION REJECTED — the replication arm scored the same "
                        "patients as the gate (identical pool fingerprint "
                        f"{v.patient_ids_hash}). That is the same test twice, not a "
                        "confirmation, and it cannot promote a candidate to a finding.")
                else:
                    v.status = "finding" if v.replicated else "candidate"
                    if not v.replicated:
                        v.limits.append(
                            f"did NOT replicate on a disjoint patient pool "
                            f"(delta {rd:+.4f}); stays a candidate")
                    else:
                        v.limits.append(
                            f"replicated on a disjoint pool (gate {v.patient_ids_hash} "
                            f"-> replication {v.replication_pool_hash}), which shares no "
                            f"patient with the gate by construction")
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

    def power_summary(self) -> Dict[str, Any]:
        """What this campaign could and could not have seen.

        A campaign that reports eight refusals and no power statement has not said
        whether the changes do nothing or whether the instrument could not resolve them.
        The median MDE answers that in one number: it is the effect size below which
        this campaign was blind.
        """
        mdes = [v.mde for v in self.ledger if v.mde is not None]
        tested = [v for v in self.ledger if v.alpha_spent is not None]
        n_uncorrected = sum(1 for v in tested if not v.alpha_applied)
        return {
            "n_tested": len(tested),
            "median_mde": (round(float(np.median(mdes)), 5) if mdes else None),
            "worst_mde": (round(float(np.max(mdes)), 5) if mdes else None),
            "n_alpha_uncorrected": n_uncorrected,
            "note": ("The campaign was blind to true effects below the median MDE. A "
                     "refusal is evidence against effects LARGER than a test's MDE and "
                     "says nothing about effects smaller than it."
                     + (f" {n_uncorrected} test(s) were charged alpha that their "
                        f"adjudicator could not apply." if n_uncorrected else "")),
        }

    @staticmethod
    def json_safe(obj: Any) -> Any:
        """Replace non-finite floats with null, recursively.

        NaN is meaningful INSIDE the Lab — an agent with no tested proposal has no
        measurable bias, and that is different from a bias of zero. It is not meaningful
        in JSON: ``json.dump`` writes a bare ``NaN`` token, which is valid Python and
        invalid JSON, so every browser's ``JSON.parse`` rejects the whole document. The
        Lab page died with "the string did not match the expected pattern" because three
        agents had never been tested and carried NaN scores.

        So NaN survives in memory and becomes ``null`` at the boundary, which is what
        JSON has for "no value" and what the UI already renders as an em dash.
        """
        if isinstance(obj, dict):
            return {k: Lab.json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Lab.json_safe(v) for v in obj]
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        if isinstance(obj, (np.floating, np.integer)):
            f = float(obj)
            return f if np.isfinite(f) else None
        return obj

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
            # allow_nan=False makes a NaN that slipped past json_safe an ERROR rather
            # than a silently unparseable file. A serialiser that cannot fail is how the
            # bad ledger reached the browser in the first place.
            json.dump(Lab.json_safe(payload), fh, indent=2, default=str, allow_nan=False)
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
            "n_duplicates": sum(1 for v in self.ledger if v.duplicate_of),
            "acceptance_rate": (sum(1 for v in self.ledger if v.accepted) / n) if n else 0.0,
            "agents": [a.as_dict() for a in self.score_agents()],
            "power": self.power_summary(),
            "programme": (self.registry.programme() if self.registry is not None else None),
            "note": "A low acceptance rate is the expected outcome. This project has "
                    "falsified eleven method families; the gate refusing is the gate "
                    "working.",
        }


# --------------------------------------------------------------------------- #
# Default proposers — distinct priors, so the plane explores
# --------------------------------------------------------------------------- #
def default_proposers() -> List[Proposer]:
    """Three priors, and generators that actually EXPLORE.

    v1's generators returned an identical proposal on every call, so an adaptive
    proposer — which refuses to repeat itself — would exhaust its whole space after one
    attempt and a campaign would stall in round one. Each generator now samples from a
    real space, seeded by the caller, so successive rounds ask different questions.
    """
    from backend.ml import selfimprove as si

    def _knob(ctx: Dict[str, Any]) -> Proposal:
        """Tune one hyperparameter of the shipped stack, sampled from SEARCH_SPACE."""
        rng = np.random.default_rng(int(ctx.get("seed", 0)))
        k = si.SEARCH_SPACE[int(rng.integers(0, len(si.SEARCH_SPACE)))]
        if k.kind == "int":
            val: Any = int(rng.integers(int(k.lo), int(k.hi) + 1))
        elif k.kind == "log":
            val = float(np.exp(rng.uniform(np.log(k.lo), np.log(k.hi))))
        else:
            val = float(rng.uniform(k.lo, k.hi))
        return Proposal(
            agent="Tuner", kind=DETECTION,
            title=f"set {k.learner}.{k.key} = {val if k.kind=='int' else round(float(val),4)}",
            change={"learner": k.learner, "key": k.key, "value": val},
            rationale=f"hyperparameter inside the shipped search space "
                      f"({k.lo}-{k.hi}, {k.kind}-scaled)",
            predicted_delta=0.003, predicted_direction="improve")

    _ROSTERS = [
        ["hgb", "hgb2", "et", "lr", "xgb@utility", "hgb@patient"],
        ["hgb@utility", "hgb2", "et", "lr", "xgb@utility"],
        ["hgb", "hgb2", "et", "lr", "xgb@utility", "lr@utility"],
        ["hgb", "hgb2", "et", "xgb@utility"],
    ]

    def _objective(ctx: Dict[str, Any]) -> Proposal:
        """Change WHAT a member is trained to predict — the only lever with a win."""
        rng = np.random.default_rng(int(ctx.get("seed", 0)) + 101)
        roster = _ROSTERS[int(rng.integers(0, len(_ROSTERS)))]
        return Proposal(
            agent="Objectivist", kind=DETECTION,
            title=f"roster {'+'.join(roster)}",
            change={"roster": roster, "objective": "patient"},
            rationale="stacking is paid for disagreement, not strength; the supervision "
                      "target is the only lever this project has measured a win from",
            predicted_delta=0.004, predicted_direction="improve")

    _POLICIES = [
        # Named for what the metric MEASURES (silencing), not for the clinical policy
        # it is a proxy for (routing to review). The detection metric charges nothing
        # for a review queue, so calling this "route to review" would claim a free
        # middle option that the disposition layer's own limits say does not exist.
        ({"policy": "uncertain_to_review", "max_false_alarm_rate": 0.15},
         "silence the conformal UNCERTAIN band (proxy for routing it to review)"),
        ({"policy": "uncertain_to_review", "max_false_alarm_rate": 0.10},
         "silence the UNCERTAIN band at a tighter alarm budget"),
        ({"policy": "ema", "halflife": 6},
         "smooth the score causally before thresholding"),
        ({"policy": "uncertain_to_review", "max_false_alarm_rate": 0.20},
         "silence the UNCERTAIN band at the shipped operating point's budget"),
    ]

    def _policy(ctx: Dict[str, Any]) -> Proposal:
        """Change the decision layer rather than the score."""
        rng = np.random.default_rng(int(ctx.get("seed", 0)) + 211)
        ch, why = _POLICIES[int(rng.integers(0, len(_POLICIES)))]
        return Proposal(
            agent="Decider", kind=POLICY, title=why, change=dict(ch),
            rationale="the same detector read three ways spans false-alarm rates of "
                      "14%, 27% and 94%; the decision layer is not data-limited",
            predicted_delta=0.002, predicted_direction="improve")

    def _pharm(ctx: Dict[str, Any]) -> Proposal:
        """Ask a mechanistic pharmacology question on the PBPK twin."""
        rng = np.random.default_rng(int(ctx.get("seed", 0)) + 313)
        inf = int(rng.choice([120, 180, 240]))
        egfr = float(rng.choice([30.0, 45.0, 90.0]))
        return Proposal(
            agent="Pharmacologist", kind=TREATMENT,
            title=f"{inf//60}h extended infusion at eGFR {egfr:.0f}",
            change={"mechanism_only": True, "antibiotic_code": 1,
                    "infusion_min": inf, "egfr": egfr},
            rationale="beta-lactam killing is time-dependent: the same dose over a "
                      "longer infusion trades peak concentration for time above MIC",
            predicted_delta=0.0, predicted_direction="no_effect")

    return [
        Proposer("Tuner", "gains hide in hyperparameters", _knob),
        Proposer("Objectivist", "gains hide in the supervision target", _objective),
        Proposer("Decider", "gains hide in the decision layer, not the score", _policy),
        Proposer("Pharmacologist", "mechanism questions belong on the twin", _pharm),
    ]


# --------------------------------------------------------------------------- #
# The partition — search / gate / replication, disjoint by construction
# --------------------------------------------------------------------------- #
def three_way_split(groups: np.ndarray, y: np.ndarray, seed: int = 42):
    """Partition patients ONCE into search, gate and replication pools.

    THE BUG THIS REPLACES
    ---------------------
    The gate drew its held-out patients with ``stratified_group_split(..., seed=42)`` and
    the replicator drew *its* patients with the same call at ``seed=977``. Two independent
    draws from the same cohort are not disjoint — they are two random 25% samples, and on
    this cohort they shared **503 of 2001 patients, 25.1% of the replication set**. So a
    quarter of the "patients the first answer never saw" had already seen it, and a
    finding was being confirmed partly on its own gate set.

    A k-fold partition cannot make that mistake. ``StratifiedGroupKFold`` test sets are
    disjoint by definition, so taking fold 0 as the gate and fold 1 as the replication
    set makes the disjointness a property of the construction rather than of a lucky
    seed. ``verify_lab.py`` asserts the intersection is empty.

    Patient-grouped and outcome-stratified throughout: an hour of a patient must never
    appear on both sides of any of these boundaries.
    """
    from backend.ml.eval.splits import stratified_group_folds
    folds = [te for _, te in stratified_group_folds(groups, y, n_splits=4, seed=seed)]
    gate_idx, rep_idx = folds[0], folds[1]
    search_idx = np.concatenate([folds[2], folds[3]])
    return np.sort(search_idx), np.sort(gate_idx), np.sort(rep_idx)


# --------------------------------------------------------------------------- #
# The real adjudicator — held-out patients, paired bootstrap, no second chances
# --------------------------------------------------------------------------- #
def held_out_adjudicator(spec: str = "dev", n_folds: int = 2,
                         gate_fraction: float = 0.25, seed: int = 42,
                         arm: str = "gate"):
    """Bind a gate that actually fits models and scores held-out patients.

    The pools are split off ONCE, by ``three_way_split``, before any proposal is seen.
    They are never fitted on, never used to select, and are scored exactly once per
    proposal. That is the contract ``selfimprove.py`` runs under, and it is why its
    honest answer was 0 accepted out of 16.

    ``arm`` chooses which held-out pool this instance scores. ``"gate"`` is the first
    test; ``"replication"`` is the confirmation, and because both arms come from one
    partition the replication patients are *provably* ones the gate never touched.
    Both arms fit on the same search pool, which is correct: replication asks whether the
    same change generalises to new patients, not whether a different fit does.

    Expensive on purpose. A cheap gate is a gate somebody will run a thousand times until
    something passes, which is precisely the failure this design exists to prevent.
    """
    from backend.ml.eval.bench import Benchmark
    from backend.ml import selfimprove as si
    from backend.ml.models.stacked import StackedEnsembleDetector
    from backend.ml.eval import metrics
    from backend.ml import config as _mlcfg
    import pandas as pd

    bench = Benchmark.load(spec, n_folds=n_folds)
    ds = bench.ds
    search_idx, gate_idx, rep_idx = three_way_split(ds.groups, ds.y, seed)
    score_idx = rep_idx if arm == "replication" else gate_idx
    feats = pd.DataFrame(ds.X, columns=ds.feature_names)

    CHAMPION_ROSTER = ["hgb", "hgb2", "et", "lr", "xgb@utility"]
    _champ_cache: Dict[str, np.ndarray] = {}

    def _fit_score(objective: str, roster, params) -> np.ndarray:
        m = StackedEnsembleDetector(
            channels=list(_mlcfg.WEARABLE_CHANNELS),
            use_hdc=False, objective=objective,
            **({"roster": list(roster)} if roster else {}),
            **({"params": params} if params else {}))
        m.fit(feats.iloc[search_idx], ds.y[search_idx], groups=ds.groups[search_idx],
              hour_index=ds.hour_index[search_idx], onset_hour=ds.onset_hour[search_idx])
        return m.predict_proba(feats.iloc[score_idx])

    def _champion() -> np.ndarray:
        """Fit the shipped configuration once per arm, not once per proposal.

        The champion is the same model every time — same search pool, same roster, same
        objective, same seed — and it was being refitted for every proposal in the
        campaign. On a full run that is one wasted stack fit per test, which roughly
        doubled the campaign's wall time for a result known to be identical.
        """
        if "p" not in _champ_cache:
            _champ_cache["p"] = _fit_score("patient", CHAMPION_ROSTER, None)
        return _champ_cache["p"]

    def adjudicate(p: Proposal, alpha: float = 0.05) -> Dict[str, Any]:
        """``alpha`` is the bar the Lab charged this test, and it is APPLIED here.

        The Lab computes a geometric bar, spends it and prints it; until this parameter
        existed it then called the adjudicator with no alpha at all, so every test used
        ``bootstrap_delta_ci``'s 95% default. The correction was accounted for and never
        applied — the same defect, in the same module, for the third time.
        """
        ch = p.change or {}
        champ = _champion()
        params = None
        if ch.get("learner") and ch.get("key") is not None and ch.get("value") is not None:
            params = {ch["learner"]: {ch["key"]: ch["value"]}}
        roster = ch.get("roster") or CHAMPION_ROSTER
        objective = ch.get("objective", "patient")
        cand = _fit_score(objective, roster, params)

        yg, gg = ds.y[score_idx], ds.groups[score_idx]
        a_champ = metrics.auroc(yg, champ)
        a_cand = metrics.auroc(yg, cand)
        # ARGUMENT ORDER MATTERS: bootstrap_delta_ci returns AUROC(pb) - AUROC(pa),
        # so the champion goes in `pa` and the candidate in `pb`. Passing them the other
        # way round inverts every verdict — a candidate that loses would read as a win.
        # It also returns a TUPLE (mean, lo, hi), not a dict.
        boot_delta, lo, hi = si.bootstrap_delta_ci(
            yg, champ, cand, gg, n_boot=400, seed=seed, alpha=alpha)
        return {"delta": float(a_cand - a_champ),
                "alpha": float(alpha),
                "bootstrap_mean_delta": float(boot_delta),
                "ci_lo": float(lo), "ci_hi": float(hi),
                "significant": bool(np.isfinite(lo) and lo > 0),
                "champion_auroc": float(a_champ), "candidate_auroc": float(a_cand),
                "arm": arm,
                "n_score_patients": int(len(set(gg.tolist()))),
                "patient_ids_hash": _pool_hash(gg),
                "metric": "AUROC on held-out patients, paired patient bootstrap"}

    return adjudicate


def _pool_hash(groups: np.ndarray) -> str:
    """A short fingerprint of exactly which patients an arm scored.

    Printed into every verdict so the gate and replication arms can be shown to be
    different sets of people without publishing patient identifiers. Two arms reporting
    the same hash would mean the replication was a re-run, which is the defect the
    policy arm actually had.
    """
    import hashlib
    ids = ",".join(str(g) for g in sorted(set(groups.tolist())))
    return hashlib.sha256(ids.encode()).hexdigest()[:12]


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
