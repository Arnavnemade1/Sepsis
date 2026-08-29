# Sepsis detection is patient classification, not event timing

**The question this pass started with.** Detection had been stuck for a long time — the
HDC/GBM stack at AUROC ≈ 0.78 within site, ≈ 0.69 across hospitals — and six independent
attempts to move it (multi-scale temporal features, MiniRocket, Forest-VSA, a dilated
TCN, a Mamba-2 state space model, quantum HDC) had all returned nothing. The brief was
to invent something genuinely new rather than add a seventh model to the same family.

**What came back.** A new method, a falsified hypothesis, and a correction to the new
method's own explanation. In order of how much they matter:

1. **Supervision density is a real, cheap, transferable lever.** Labelling more of a
   septic patient's stay positive during *training* raises detection at a fixed alarm
   budget by 20–48% relative, holds at every budget from 5% to 30%, and transfers
   across hospitals in both directions. The best target is the **patient label** — it
   has no parameter to tune.
2. **The temporal window does not just fail to help — it slightly hurts.** The gain
   has nothing to do with capturing the approach to sepsis. Scattering the extra
   positive hours at *random* through the stay beats placing them before onset, at
   every alarm budget. This was caught by a negative control in the verification suite
   after the module had already been written the other way round, and the control then
   caught a flaw in *itself* before the claim could be stated properly.
3. **Cross-channel coordination carries no sepsis signal at hourly resolution.**
   A Buchman-style organ-uncoupling detector was built and then falsified with a
   matched-pair surrogate design.

Together these say one thing: **at the accuracy currently reachable on hourly EHR
vitals, this is a problem about *which patients*, not about *when*.**

---

## 1. The diagnostic that redirected everything

The pooled AUROC that this field reports silently averages two different questions:

| | question | production ensemble |
|---|---|---|
| **BETWEEN** | is this the kind of patient who becomes septic? | **0.792** |
| **WITHIN** | inside a septic stay, which hours are the pre-onset ones? | see below |
| pooled | (the number everyone quotes) | 0.763 |

The WITHIN number is where it gets interesting, and where it is easy to be fooled.
Measured *pooled over septic patients* it is **0.503** — chance. Measured **strictly
per patient**, each stay its own stratum, it is **0.716**. Both are computed from the
same predictions.

That gap is Simpson's paradox, in the data:

- **within** a septic stay the score genuinely rises toward onset (paired difference
  +0.0167, rises in 72.4% of stays, Wilcoxon *p* = 1×10⁻⁹, median within-patient
  Spearman +0.51);
- **across** patients the score is dominated by length of stay (Spearman +0.525;
  stays ≤ 24 h average 0.027, stays > 72 h average 0.070 — a 2.6× offset with nothing
  to do with deterioration).

So the patient-level offset is **larger than the within-stay rise that actually signals
sepsis**. A global alarm threshold therefore fires permanently on long-stay patients
and never on a short-stay patient who is genuinely crashing.

### The trap in reporting within-patient timing at all

Inside a septic stay the CinC label is a perfect monotone function of the clock — the
bare hour index scores a within-patient AUROC of **1.0000**. Any model that can recover
elapsed time therefore gets "timing skill" for free and for no clinical reason. Testing
this directly: with the clock features removed, a within-patient ranker scores 0.9198
on the label — but 0.9736 on the *clock*. Remove the expanding extremes as well and it
is 0.7668 on the label against 0.8612 on the clock.

**In both cases the model ranks time better than it ranks the label.** Within-stay
timing performance on this dataset is not identifiable from timekeeping, and any paper
reporting it without this control is reporting a clock.

---

## 2. The lever: supervision density

The challenge label marks an hour positive when `t ≥ t_sepsis − 6h`. On this cohort
that is 2.1% of hours — even though **12.8% of the patients become septic**. The
extreme class imbalance that shapes every other decision in this project is
substantially an artifact of how narrowly the label was drawn.

Same features, same model, same folds; only the training target changes. Evaluation and
calibration are always against the original challenge label.

| training target | train prevalence | AUROC | det@5% | det@10% | det@15% | det@20% |
|---|---|---|---|---|---|---|
| challenge label (`W=0`) | 0.021 | 0.7411 | 0.274 | 0.353 | 0.411 | 0.477 |
| 96 h window before onset | 0.105 | 0.7505 | 0.358 | 0.437 | 0.496 | 0.546 |
| **same count, RANDOM hours** | 0.105 | 0.7538 | 0.384 | 0.461 | 0.505 | 0.555 |
| **patient label (all hours)** | 0.128 | 0.7534 | **0.406** | **0.468** | **0.508** | 0.554 |

Detection at a 5% alarm budget: **0.274 → 0.406, +48% relative.** At 10%: +33%.

**Significance.** Paired patient bootstrap on identical patient-grouped folds, over five
supervision densities × five alarm budgets: **25 of 25 cells significant**, with the
gain rising monotonically in density. The standard this project set for telling a real
gain from metric shopping is that it must not appear at one alarm budget and vanish at
its neighbours; this appears at all of them.

**AUROC does not move** (`W=96`: +0.0067, 95% CI [−0.0066, +0.0202], n.s.) and is
reported as n.s. rather than omitted. That is not a weakness of the result, it is the
mechanism: denser supervision trades WITHIN for BETWEEN (0.629 → 0.579 and 0.788 →
0.799), pooled AUROC cancels, and detection-at-budget — a *patient-level* statistic —
inherits only the gain. **This lever was invisible to a project that had been reading
AUROC.**

### Cross-hospital, both directions

Set A is Beth Israel Deaconess (790k rows, 2.17% prevalence), set B is Emory (762k
rows, 1.41%). Patient label vs challenge label, 2000 patient bootstraps:

| budget | A → B | B → A |
|---|---|---|
| 5% | +0.019 **sig** | +0.118 **sig** |
| 10% | +0.028 **sig** | +0.103 **sig** |
| 15% | +0.015 n.s. | +0.093 **sig** |
| 20% | +0.010 n.s. | +0.073 **sig** |

Positive everywhere, significant in six of eight cells, and significant in both
directions at the tight budgets where wards actually operate. The asymmetry tracks
prevalence — the sparser site gains more from denser supervision — which is a
hypothesis consistent with the numbers, not a tested claim.

### On the production ensemble

Applied to the shipped stacked ensemble (2 folds, so roughly half the power of the
sweep above): positive in all eight cells, significant at the 5% budget for both
`W=96` (+0.032) and the patient label (+0.038), and at 20% for the patient label
(+0.031).

---

## 3. The correction: the window is not the mechanism

This module was first written as a **"supervision horizon"** — the hypothesis being
that a wider *pre-onset window* teaches the model the physiology of the approach to
sepsis. That reading produced the right numbers for the wrong reason.

The negative control in `verify_supervision.py` builds a **sham** target: exactly the
same number of extra positive hours, scattered at random through septic stays instead
of placed before onset. Same patients, same prevalence, same class balance; the
temporal structure is the only thing removed.

**A note on where this control is run, because the first attempt got it wrong.** At
W = 96 the widened target claims about 79% of every non-labelled septic hour that
exists, so a random draw is *forced* to land mostly on the same hours as the window and
the control cannot discriminate. The suite caught this itself — its separation check
failed, reporting only 10,850 hours placed differently. The comparison below is run at
W = 24, where the window takes 34% of the pool and the sham genuinely differs, with an
identical estimator in every arm (the first version compared a calibration-holdout
model against a plain one, so the arms differed by training-set size too).

| W | budget | window − sham (seed 1) | window − sham (seed 2) |
|---|---|---|---|
| 12 | 5% | −0.069 [−0.097, −0.035] | −0.077 [−0.105, −0.044] |
| 12 | 10% | −0.056 [−0.087, −0.013] | −0.056 [−0.081, −0.015] |
| 24 | 5% | −0.063 [−0.099, −0.032] | −0.062 [−0.088, −0.028] |
| 24 | 10% | −0.057 [−0.093, −0.024] | −0.031 [−0.063, +0.000] |

**Every point estimate is negative, and the tight-budget intervals sit entirely below
zero.** The window is not merely inert: concentrating the extra supervision just before
onset is *worse* than scattering it at random. Which fits the mechanism — the hours
immediately before onset are the ones most similar to hours the model already sees
labelled, so they add the least new information about who the patient is, while hours
drawn from across the stay teach it to recognise that patient from a wider range of
physiologic states.

That also explains why the best target is the one with the widest possible spread and
no temporal content whatsoever: the patient label.

The documentation was rewritten rather than quietly adjusted, because the difference
between "a wider early-warning window" and "more patient-level supervision" is the
difference between a claim about physiology and a claim about labelling — and only the
second one survives the control. The check that caught it is kept pointed the same way:
if a real window effect ever appears, `verify_supervision.py` fails and the docstring
is wrong again.

---

## 4. The falsified hypothesis: physiologic coordination

Every feature this project had ever built (`backend/ml/data/features.py`) is a
**marginal** statistic — one channel's value, slope, rolling spread, personal z-score.
The only cross-channel terms are a shock index and two threshold counts. But sepsis
presents *relationally*: fever without the tachycardia that should accompany it,
tachypnea without hypoxia, a heart rate that stops tracking blood pressure. Godin &
Buchman (Crit Care Med 1996;24:1107) proposed exactly this — organs as coupled
biological oscillators that sepsis **uncouples** — and it is normally studied on
beat-to-beat waveforms, out of reach of an hourly pipeline.

`backend/ml/models/coordination.py` operationalises it at hourly resolution, including
a decomposition that appears to be new to this problem: for a Gaussian normative model,
the pointwise joint surprisal minus the summed marginal surprisals,

    C(x) = ½[ rᵀΣ⁻¹r − Σᵢ rᵢ²/Σᵢᵢ ]

is the **pointwise total correlation** — the surprise living only in the dependency
structure, with every marginal term subtracted off, and therefore orthogonal by
construction to the entire incumbent feature space.

**It does not work, at three levels of organisation:**

| level | detector | result |
|---|---|---|
| channel coupling (per-patient correlation structure) | 43 features | 0.642 alone, **−0.005 concatenated** |
| conditional residuals (each vital vs what the others predict) | 33 features | 0.572 alone, +0.002 |
| **organ-system synchrony** (7 real organ systems) | 13 features | synchrony index **0.515 — chance** |

The decisive test was a **matched-pair surrogate**: delaying each channel by its own
random lag destroys alignment but also adds staleness, so it was paired against a
*common* delay that adds identical staleness with alignment preserved. The difference
is the value of alignment with staleness cancelled:

| lag | alignment worth (incumbent) | (coordination plane) |
|---|---|---|
| 3 h | +0.0017 | +0.0017 |
| 6 h | −0.0019 | +0.0001 |
| 12 h | −0.0031 | −0.0030 |
| 24 h | −0.0032 | −0.0175 |

**No dose-response, and the sign flips.** At hourly resolution, the temporal alignment
between the seven bedside vital-sign channels carries no sepsis information.

A first version of this control used a circular `np.roll`, the textbook surrogate for
stationary signals. It was wrong here: wrapping moves a septic patient's late,
deteriorated hours to the *front* of the stay, so every hour inherits evidence of the
outcome — and **both** detectors scored *higher* on that surrogate than on real data. A
control that manufactures signal instead of removing it is worse than none. It is now
an assertion in `verify_coordination.py`: the circular roll must still be *detected* as
invalid (incumbent 0.7514 → 0.7683 on the rolled cohort), so the mistake cannot be
reintroduced.

The same suite also caught a defect in the module's own advertised property. The
structure plane is documented as scale-invariant "by construction", and it was not:
absolute numerical floors (`1e-2` on a standard deviation, `1e-6` on a variance) sat
inside scale-dependent quantities, so rescaling every channel by 3.7× moved rows across
those thresholds and shifted the features by up to 14.8. Replacing the floors with
strict positivity tests — which are themselves scale-free — makes the invariance real
(max |Δ| now 1.5×10⁻⁸, float32 rounding) and leaves every result above unchanged.

---

## 5. What this explains, and what to do next

If most of the accessible signal is patient identity rather than temporal shape, then
this project's long run of null results is not eight unlucky experiments — it is **one
fact, measured eight times**. Multi-scale temporal features, MiniRocket, Forest-VSA,
the TCN, the SSM, critical slowing down, quantum HDC and now coordination all failed
for the same reason, and the frontier pass's utility-target win was most likely also a
relabelling effect rather than a timing one.

**Shipped** (see "What shipped" below): the production detector is now
`patient:hgb+hgb2+et+lr+xgb@utility`.

**Do not claim:** an AUROC improvement, or that this detects sepsis earlier. Median
lead time is essentially unchanged, and a long lead time from a timing-blind detector
is not evidence of early warning.

**Open, and worth attacking next:**
- Is within-stay timing answerable *at all* from hourly vitals, once the clock is
  properly controlled for? The evidence here is "not demonstrably", not "no".
- Does denser supervision compose with the utility-aligned objective, or are they the
  same lever twice? The `{kind}-win{W}` × objective cells are registered and unrun.
- The `WHO × WHEN` composition beat a bare-clock control on AUROC (+0.0159, CI
  [+0.0035, +0.0276]) while *costing* detection-at-budget. Two forms, two different
  metrics; neither dominates, and it needs a locked-gate adjudication rather than
  another sweep.

## What shipped

`StackedEnsembleDetector` gained `objective="patient"`, and the production artifact was
re-promoted through `promote_frontier.py` with the same roster it already had:

| | incumbent | shipped |
|---|---|---|
| detector | `balanced:hgb+hgb2+et+lr` | `patient:hgb+hgb2+et+lr+xgb@utility` |
| held-out AUROC | 0.7622 | **0.7673** (+0.0051) |
| held-out utility | +0.3333 | **+0.3426** |
| ECE | 0.0028 | **0.0010** |
| detect@FA15 | 0.477 | **0.728** |
| operating point | HIGH 0.0525 | HIGH 0.0300 / MEDIUM 0.0150 |

All three promotion gates passed on 1,004 judged patients that neither model was fitted
on. Detection inside a fixed alarm load, paired patient bootstrap:

| budget | incumbent | shipped | delta |
|---|---|---|---|
| 5% | 0.360 | 0.333 | −0.027 CI [−0.080, +0.533] — n.s. |
| 10% | 0.360 | **0.707** | +0.347 CI [+0.240, +0.460] |
| 15% | 0.427 | **0.707** | +0.280 CI [+0.187, +0.387] |
| 20% | 0.453 | **0.707** | +0.253 CI [+0.133, +0.360] |

### Three defects the promotion exposed

Shipping this found more than it changed, because a new detector is the first thing
that had ever made these three assumptions false.

1. **The alarm-budget gate did not exist.** The promoter had two gates — AUROC and
   utility-at-a-threshold — and neither asks the question a supervision-density change
   actually makes. A promoter whose gates are silent on a candidate's own claim cannot
   refuse it for the right reason. The new third gate bootstraps over *patients*: its
   first, point-estimate version refused this candidate on a −0.027 blip at the 5%
   budget whose confidence interval turned out to be [−0.080, **+0.533**].

2. **The conformal layer was silently cross-fit.** `conformal.json` is a set of cut
   points on one detector's score distribution, and nothing checked which. Promoting
   moved the shipped p95 from 0.0532 to 0.0479 while leaving bands on disk calibrated
   against the previous model, and `AegisCore` would have applied them without a word.
   `conformal.load()` now takes the identity of the model being served and refuses a
   mismatch — the layer reports absent, a state the caller already handles.

3. **The integrity audit was auditing a proxy.** `verify_integrity.py` built its own
   `StackedEnsembleDetector` from constructor defaults on 600 patients and evaluated
   the shipped threshold against *that*. It is the cross-fit threshold read the rest of
   this project forbids in writing — `promote_frontier.py` records the false-alarm rate
   swinging 0.085 / 0.178 / 0.209 / 0.165 across fits of 750 / 1500 / 2250 / 3000
   patients — and it passed for years only because the defaults happened to equal what
   was deployed. On the proxy it reported the shipped threshold at utility −0.88; on
   the artifact actually on disk, scored on 2,004 patients it never trained on, the
   same threshold measures **+0.3178** against +0.1290 for the old hand-set guess. The
   fixture now reads the promoted configuration, and the check scores the deployed
   artifact.

`verify_full_system.py` and `verify_integrity.py` (9/9) both pass against the new
detector.

## Reproducing

```bash
make verify-supervision
```

```bash
make verify-coordination
```

```bash
make frontier SPEC=dev
```

`verify-supervision` is 15 checks and `verify-coordination` is 9; both currently pass.
Two of them assert a **null** on purpose — that the temporal window has no advantage
over random placement, and that inter-channel alignment is worth nothing. If either
ever starts failing, a documented mechanism here is wrong and these files say so before
the README does.
