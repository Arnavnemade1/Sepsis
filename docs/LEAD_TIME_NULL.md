# Median lead time does not measure early warning

**Date:** 2026-08-30 · **Reproduce:** `PYTHONPATH=. python -m backend.ml.scripts.attack_clock_claim`
**Status:** this retracts a claim this project made, and replaces it with a stronger one.

## What was claimed, and what happened to it

After the supervision-density work this project proposed a field-level claim:

> Within a septic stay the CinC label is a monotone function of time. The bare hour index
> scores a within-patient AUROC of 1.0000, so any model that recovers elapsed time gets
> timing skill for free — and a large share of published sepsis early-warning performance
> may be patient classification plus a clock.

It was attacked before anything was built on it. **Two of five attacks landed and killed
the strong form.**

| attack | result |
|---|---|
| 1. Is the headline a tautology? | **LANDS.** The label *is* a monotone step in time in all 607 septic stays — 0 violations. AUROC 1.0 is definitional. Quoting it as a discovery was wrong. |
| 2. What share of pooled AUROC can a clock touch? | **LANDS HARD.** Only **10.9%** of discordant pairs are within-patient. A *perfect* clock moves pooled AUROC by at most ~that. "Mostly a clock" is FALSE. |
| 3. Do models lean on clock features? | Survives, weakly — deleting them costs 0.0876 AUROC, but they carry real physiology too (see caveat). |
| 4. Is elapsed time legitimate information? | Survives. Stay length predicts sepsis *between* patients at AUROC **0.5233** — chance (36h vs 37h). Perfect within, chance between: the signature of an artifact. |
| 5. Is reported lead time a clock artifact? | Survives, and strengthens under a harsher null. |

**Caveat on attack 3, stated because it flatters us otherwise:** the deleted features
(`__emin`, `__emax`, `burden_*`) are cumulative summaries that encode elapsed time *and*
genuine clinical information — worst lactate seen, hours spent hypotensive. The 0.0876
drop cannot be attributed to the clock alone, so attack 3 "surviving" is weak support.

## The finding that replaces it

Attack 5 originally used a weak null — an alarm placed uniformly at random inside the
pre-onset window — which gave 12.0h against our reported 26–40h and looked like a
defence of our numbers. That null is wrong. **A real detector fires at the FIRST
crossing of a threshold**, which for any low threshold lands near the start of a stay.

Re-run against that null, **at matched detection rate**:

| detection rate | random score | patient-ID only, zero timing skill | **shipped detector** |
|---|---|---|---|
| 30% | 35.2h | 34.5h | 40.0h |
| 40% | 39.0h | — | 40.0h |
| 50% | 38.0h | — | 38.5h |
| 60% | 39.0h | 42.0h | 38.5h |
| 70% | 38.0h | 37.0h | 34.8h |

**A pure random score produces the same median lead time as the shipped detector at every
matched detection rate, and beats it at 70%.**

Median lead time is therefore determined by stay geometry and alarm rate, not by
detector skill. It is not a measure of early warning.

## What this costs us

Every lead time this project has published — 26h, 29h, 34h, 37h, 39h, 40h — is
indistinguishable from what a random number generator produces at the same detection
rate. They are not evidence of early warning and must not be quoted as such. That
includes the 37h reported alongside the shipped patient-supervision detector.

## What it is worth

The claim that survives is narrower than the one we started with, and better founded:
it is empirical rather than definitional (random vs real is a measurement, not an
identity), it applies to any paper reporting median lead time, and it comes with a fix.

**Report lead time against a random-score null at matched detection rate.** A detector
whose lead does not exceed that null has not demonstrated early warning, whatever its
AUROC. The null costs nothing to compute — it needs no extra model, only a shuffled
score vector.

## Method note

The comparison must be at matched **detection rate**, not matched alarm rate. A detector
that alarms on more stays reaches further back on average for reasons unrelated to skill,
and the first version of this analysis compared 511 fired stays against 367 and would
have drawn the wrong conclusion.
