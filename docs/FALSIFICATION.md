# What we invented, and proving it can fail

Two questions, answered in order, because the second is only worth asking once the
first is answered honestly.

---

## 1. What did we actually invent?

Most of this system is assembled from standard parts, and saying so is what makes the
rest credible.

**Assembled, not invented.** Stacked ensembles. Conformal prediction (Vovk; Tibshirani
et al. for the weighted form). MiniRocket (Dempster et al.). Hyperdimensional
computing / VSA (Kanerva). Label-shift estimation — BBSE (Lipton et al.) and EM
(Saerens et al.). Z3. FedAvg. Transfer entropy (Schreiber). rPPG. None of these are
ours and none are claimed.

**Ours**, strongest first:

### The refusal architecture
Verification that tests **claims**, not code. `verify_integrity.py` plants known lies
and asserts the guards fire. `selfimprove.py` searched 16 champion mutations and
admitted **zero**. `artifacts/alarm_policy.json` records a *refusal* as its result.
Three derive scripts refuse to save an artifact that loses. Most repositories test that
code runs; this tests that claims are true, and that the machinery meant to refuse
actually refuses.

### Cross-modal HDC resonance gate
Hyperdimensional geometry used to arbitrate **agreement between physically independent
acquisition pathways** rather than to classify. Real deterioration moves every pathway;
a sensor glitch moves one. 91% artifact false-alarm removal, LOPO-validated. HDC is
old; this use of it is not.

### Portable proof certificates
Ed25519-signed SMT-LIB2 that a third party re-checks **without trusting AegisOS** — a
signed lie is caught by the re-check, not by taking our word for the signature.

### Physiologic-liveness BFT fallback
Consensus liveness established from physiology rather than from a timeout.

---

## 2. What can be improved in it?

Two gaps in the crown jewel, both measured, both found the expensive way.

### Gap 1 — 452 checks, and nothing proved any could fail

A check that cannot fail is not a weak check. It is **a lie with a green tick**, and it
is worse than no check because it is counted as evidence.

Four vacuous checks were found by hand in one session:

| suite | the defect |
|---|---|
| `verify_conformal` | a `>=` that equality satisfied — deleting the (n+1) conformal correction left it passing |
| `verify_conformal` | a state count that swept in the EMPTY prediction set, so the degeneracy it existed to expose read as success |
| `verify_minirocket` | a harness expectation that never matched the check's own text |
| `verify_shift` | an assertion written for the wrong direction of the effect |

Each was caught with a throwaway harness in `/tmp` that was then deleted. **The
technique worked and was not in the repository.** Now it is: `backend/ml/falsify.py`.

### Gap 2 — only 4 of 16 artifacts recorded which fit produced them

Three "findings" in one investigation were arithmetic, not science, and all three were
the same mistake — a number read off one model's score distribution and applied to
another's:

1. the shipped operating point derived against a different fit than served;
2. `external_validation_wearable.json` scoring a target hospital with that constant,
   manufacturing an apparent **54% cross-hospital utility collapse**;
3. conformal bands calibrated on a 60%-of-setA fit and applied to scores from a
   100%-of-setA fit, manufacturing an apparent **0.929 → 0.739 coverage collapse**.

Each read as a dramatic result about deployment. `backend/ml/fitid.py` refuses all
three at the point of use.

---

## 3. The Falsification Engine

**This is not code-coverage mutation testing.** Classical mutation testing (Jia &
Harman) perturbs *syntax* — flip a `<`, drop a statement — and asks whether any test
fails; it measures how tightly tests pin the code. This perturbs the **scientific
invariant** and asks whether a *specific named check* fails.

The difference is the whole point:

> A surviving **syntactic** mutant means weak test coverage.
> A surviving **semantic** mutant here means **a published claim nobody is defending.**

So the deliverable is not the percentage. It is the survivor list.

### Mutation families — every one a mistake actually made here

| family | what it breaks | what it cost |
|---|---|---|
| `fit-identity` | a quantity read against a different fit | **3 false findings in one investigation** |
| `causality` | a causal transform reads forward in time | would have posted a lead time made of hindsight |
| `exchangeability` | patient-grouped → row-level | an interval several times too narrow |
| `class-conditioning` | class-conditional → pooled marginal | the conformal construction going degenerate at 2% prevalence |
| `naive-statistic` | corrected estimator → uncorrected | an estimator that reads positive on pure noise |
| `label-leak` | a label-free estimator handed the answers | a deployment method silently becoming an oracle |
| `floor-removal` | a min-cell / memory guard disabled | a tail quantile from two patients, with a guarantee on it |

### Result

    make falsify

Extended from 5 suites to **9**, covering **480 named checks** in the readable set.

| suite | checks | mutations | killed | coverage |
|---|---|---|---|---|
| `verify_conformal` | 24 | 6 | 6 | 100% |
| `verify_shift` | 20 | 6 | 6 | 100% |
| `verify_minirocket` | 17 | 5 | 5 | 100% |
| `verify_fitid` | 13 | 3 | 3 | 100% |
| `verify_coupling` | 12 | 3 | 3 | 100% |
| `verify_agent` | 68 | 1 | 1 | 100% |
| `verify_z3_safety` | 14 | 2 | 2 | 100% |
| `verify_operating_point` | 14 | 3 | 2 | 67% |
| `verify_selfimprove` | 14 | 1 | 0 | 0% |
| **total** | | **30** | **28** | **93%** |

Two more mutations are reported **not applicable** rather than counted: a suite that
runs its checks at import time has already executed them by the time its own helpers
exist, so a mutation targeting those helpers can never reach it. Scoring that as a
survivor would name a claim undefended when the audit simply could not test it — a lie
in the exact direction this engine exists to prevent, so it is excluded from the
denominator and listed separately.

### The survivors, published rather than quietly fixed

**1 — `verify_operating_point` does not defend its own bootstrap margin.**
Setting `far_margin=False` in `derive()` ships a false-alarm cap that holds only on the
patients used to choose it, and every check in the suite still passes. The check that
*looks* like it defends this — *"setting it on the upper bound breaches strictly less
often"* — calls `far_upper_bound()` **directly** rather than through `derive()`. It
proves the margin mechanism works. Nothing proves `derive()` still uses it.

**2 — `verify_selfimprove` does not test the code it claims to.** This one matters
more. The suite defines its own `apply_rule()`, whose docstring reads *"The shipped
rule"*, and tests that. It never references `RecursiveSelfImprover` — not once. So the
shipped acceptance rule's own `min_gain` and `alpha` can be set to accept anything at
all and all 14 checks still pass.

> That rule's verdict — **0 of 16 candidates admitted** — is the headline evidence for
> the refusal architecture, the thing this project claims as its primary invention. It
> has been verified against a re-implementation in the test file rather than against
> the code that produced it.

The engine was built to find exactly this, and found it on the extension run.

### The engine is subject to its own rule

`make verify-falsify` — 17 checks. A crash is **not** counted as a kill (the mutation
must be caught by a check, not by the interpreter). A mutation caught by the *wrong*
check is a survivor, not a kill. And the audit corrected its own author twice: two
mutations were reported as survivors when the fault was the mutation spec naming the
wrong defender — which is the argument for naming a specific check instead of accepting
"something failed".

Its self-audit also suffered, and then fixed, the exact leak the engine guards against:
a mutation from one check surviving into the next because the throwaway dependency was
not in `RESET_MODULES`.

### The gate

`--gate` fails **only on regression** — coverage dropping, or a previously-killed
mutation now surviving. A fixed coverage floor was considered and rejected: it fails
loudly on day one and forces a backfill before any other work can proceed, whereas
regression gating catches a newly-added vacuous check the moment it lands and leaves
the existing survivor as published work to do.

---

## Honesty boundary

- Coverage is over **30 scored mutations across 9 suites**, spanning 480 named checks.
  It is emphatically **not** 30 mutations covering 480 checks — most checks in those
  suites have no mutation aimed at them. 93% is 93% *of what was attempted*, and
  pretending otherwise would be the same species of lie this apparatus exists to catch.
- Two suites are excluded from the default run on cost (`verify_subgroup` ~130s,
  `verify_alarm_policy` ~35s per mutation); `--slow` includes them.
- Three container shapes exist in the repository — `results: List[bool]`,
  `_results: List[(name, ok, detail)]`, and `_failures: List[str]`. Names are read from
  the container wherever one exists, because `verify_agent` and `verify_doctor` print
  `detail or name`: when a check supplies a detail the check's own name never reaches
  stdout, and matching on the printed line would silently never fire.
- The engine proves a check **can** fail. It does not prove the check is *the right
  check* — only that something is defending the claim.
- `fitid` refuses mismatches at the point of use. Artifacts written before it existed
  carry no fingerprint and read as **unknown**, which is refused rather than waved
  through; reading them requires an explicit `strict=False`.

## Reproduce

```bash
make verify-falsify
```

```bash
make falsify
```

```bash
make verify-fitid
```
