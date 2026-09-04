# The Lab — an agent research plane with a locked gate

`backend/aegis/lab.py` · `lab_runner.py` · `lab_registry.py` · `webui/lab.html`
Audit: `make verify-lab` (90 checks) · Run: `make lab-run` / `make lab-run-full` ·
Record: `make lab-registry`

## What it is

Agents propose changes to detection, decision policy and treatment. A locked gate scores
them on patients no agent can see or fit on, and a second, **disjoint** pool decides
whether the result survives. Agents run the research; they never judge it.

That separation is the whole design. `selfimprove.py` exists because a search loop's
failure mode is not "fails to improve", it is "improves its own scoreboard" — and it
accepted **0 of 16** mutations once a locked gate was put in front of it.

## The protocol

| stage | who | can it accept? |
|---|---|---|
| Proposal | `Proposer` — generates a change from a stated prior, and registers its predicted effect **before** the gate runs | no |
| Review | `Critic` — argues from this project's published nulls *and* the Lab's own refusals | no, warning only |
| Veto | `SafetyOfficer` — alarm budget, safety constraints, the treatment firewall | it can only **refuse** |
| Gate | held-out patients, paired patient bootstrap, geometric α | → **candidate** |
| Replication | a second pool, disjoint from the gate by construction | → **finding** |

Treatment proposals run on the PBPK twin and are typed `MECHANISM`. The firewall is on
the **claim**, not the computation: the model runs and reports exposure, and
`Verdict.efficacy_claim` is structurally always `None`. This project has already tuned a
detector against its own generator and watched it score near chance on real patients.

## Three defects found by auditing our own gate (protocol v1 → v2)

All three made the gate **looser** than it advertised. Each is now a regression test.

**1. The replication pool overlapped the gate pool.** The gate drew held-out patients at
`seed=42`; the replicator drew its own at `seed=977`. Two independent 25% draws from one
cohort are not disjoint — measured, they shared **503 of 2 001 patients, 25.1% of the
replication set**. A finding was being confirmed partly on its own gate set.

**2. The policy arm did not split at all.** `policy_adjudicator` read cached out-of-fold
scores over the whole cohort, so calling it again at another seed re-scored *the same
8 000 patients*. The raw delta came back **bit-identical at +0.025037**; only the
suppression null's RNG moved the reported number. A policy candidate replicated with
probability ~1 — a rubber stamp. The conformal-band policy would have been published as
a finding at +0.0543 "replicated +0.0523".

**3. The alpha was charged but never applied.** `AlphaBudget` computed a geometric bar,
spent it, and printed "tested at alpha=0.00625" in the verdict's limits. The adjudicator
was then called with no alpha and used `bootstrap_delta_ci`'s 95% default. This is the
third time a correction has been documented in this module and not applied.

Defect 3 was load-bearing on its own. In the first v2 campaign the conformal-band policy
scored **+0.0497**:

```
at 95%            CI [+0.0100, +0.0873]   significant  -> accepted
at its real bar   CI [-0.0066, +0.1019]   not significant -> refused   (α = 0.00625)
```

Under v1 that proposal would have cleared the gate and then "replicated" on the identical
patient pool. Two independent bugs pointed the same way.

**The fix.** `three_way_split` partitions patients **once** into search / gate /
replication with a patient-grouped, outcome-stratified k-fold, whose test sets are
disjoint by definition rather than by a lucky seed:

```
search 3 993 | gate 2 004 | replication 2 003     pairwise intersection 0
septic rate   2.14%          1.89%         2.21%
```

Every verdict carries a fingerprint of the patients that scored it, and a replication
whose fingerprint matches its gate is **rejected outright** — so the defect cannot
silently return.

## Power: a refusal is a measurement

"Refused" alone does not distinguish *this change does nothing* from *this test could not
have seen it*, and eight refusals with no power statement are not a scientific result.
Every verdict now reports a **minimum detectable effect** — the smallest true effect the
test could have called significant, which is the half-width of its own interval floored
at `MIN_EFFECT`. A refusal reads:

> rules out true effects at or above +0.0300. Effects smaller than that were NOT
> testable here — this refusal is not evidence that the change does nothing, only that
> it does nothing that large.

## The registry: the Lab remembers

Each campaign used to overwrite the last one's ledger, which made two things free:
**campaign shopping** (run it ten times, publish the run with a hit) and **re-testing a
dead hypothesis**. `lab_registry.py` keeps every campaign ever run, hash-chained
(`sha256(canonical(entry) || prev_hash)`), and enforces:

- **A hypothesis is tested once, ever.** Dedup matches the canonical *change*, not the
  title, so rewording does not defeat it. A repeat costs no alpha. This is what actually
  prevents shopping — shopping only inflates error if you can re-ask the same question.
- **Cumulative alpha is disclosed.** The sum of alpha over every test in every campaign
  **is** the expected number of false findings for the programme. It is the number to
  compare the finding count against, and the console leads with it.
- **Verdicts cannot be quietly edited or deleted.** `verify_chain()` re-derives every
  link and names the first entry that breaks.

Per-campaign FWER stays at 0.05. Carrying one geometric counter across campaigns was
tried on paper and rejected: `exhausted()` trips after about four tests, so a permanent
counter would refuse everything forever after campaign one. That is a broken instrument,
not rigour.

### Retraction

Records carry the protocol they were produced under. The eight v1 verdicts are marked
retracted: their **refusals stand** (a change that failed a loose bar fails a tighter
one), their **acceptances do not**, and `seen()` lets a retracted acceptance be asked
again under v2 rather than treating a pass nobody should trust as a settled question.

## What a campaign is expected to produce

Mostly refusals. This project has falsified eleven method families. A campaign reporting
a string of wins is evidence its gate is broken, not that its agents are clever — so the
console leads with the refusal count, the error budget and the power, and the agent
scoreboard measures the proposers rather than flattering them.
