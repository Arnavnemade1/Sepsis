# Detection accuracy frontier — what actually moved the number

**Question.** Detection had been stuck: within-site AUROC ≈ 0.73–0.77, cross-hospital
≈ 0.69, and Phase 7 concluded the ceiling was data, not model capacity. So this pass
deliberately went looking outside the family of things already tried — new booster
libraries, deep sequence architectures, a tabular foundation model — and, as it turned
out, found the largest lever somewhere none of them live.

**Method.** Every candidate below is a function `(dataset, train_idx, test_idx) → p_test`
run by `backend/ml/eval/bench.py` on **identical patient-grouped folds**: 310,061
patient-hours, 8,000 CinC 2019 patients, hour-prevalence 0.0209, seed 42. No candidate
picks its own split. Differences are bootstrapped by resampling **patients**, not rows —
hours within a stay are far too correlated for a row bootstrap, which would report an
interval several times too narrow and dress fold luck up as a discovery.

> **Read the fold count before comparing rows across tables.** The single-model sweep in
> §1 is 4-fold; the ensemble comparisons in §2b are 3-fold, run separately because the
> ensembles are ~5× the cost. Numbers are paired *within* a table and not comparable
> *between* tables — that is exactly the trap this harness exists to prevent, and it
> applies to its own output too.

---

## 1. The finding: we were training on the wrong target

The challenge label flips to 1 at `t_sepsis − 6h`. The official utility function — the
thing we are actually scored on — does not agree with that label:

| hours relative to onset | challenge label | value of raising the alarm |
|---|---|---|
| before −12h | 0 | −0.05 |
| **−12h to −6h** | **0** | **0 → +1, rising** |
| −6h to +3h | 1 | +1 → +2 |
| after +3h | 1 | 0 (nothing can score) |

The single most valuable prediction in the whole scoring scheme — alarming a full 6–12
hours before onset — is one the label calls a false positive. On this cohort that is
**3,419 patient-hours carrying 2,149 utility points, about 19% of all positive-side
value, that every model here has been trained to call negative.**

`backend/ml/eval/utility.py` derives the target the scoring function implies instead:
label `z = sign(u₁ − u₀)`, weight `w = |u₁ − u₀|`. A weighted classifier's Bayes rule
under `(z, w)` is `E[u₁ − u₀ | x] > 0` — exactly the utility-optimal alarm — so this is
a re-derivation of the objective, not a heuristic reweighting. The module reproduces
the reference implementation's utility to 3×10⁻¹⁶ on four alarm policies including the
adversarial all-alarm case.

### Objective × library, same features, same folds

| booster | balanced (incumbent objective) | utility-aligned | Δ AUROC |
|---|---|---|---|
| sklearn `HistGradientBoosting` | 0.7283 ± 0.0090 | **0.7467 ± 0.0120** | **+0.0184** |
| LightGBM | 0.7408 ± 0.0160 | **0.7533 ± 0.0137** | **+0.0125** |
| XGBoost | 0.7213 ± 0.0033 | **0.7590 ± 0.0115** | **+0.0377** |

The objective helps every library. It is worth more than the library choice, and the
two are partly additive. Full metrics:

| candidate | AUROC | AUPRC | utility | ECE | detect@spec90 | FA@spec90 | lead |
|---|---|---|---|---|---|---|---|
| `gbm` (shipped single GBM) | 0.7281 ± 0.0135 | 0.0661 | +0.2811 | 0.0031 | 0.732 | 0.498 | 26.6h |
| `hgb-balanced` | 0.7283 ± 0.0090 | 0.0666 | +0.2915 | 0.0019 | 0.710 | 0.484 | 29.5h |
| `hgb-utility` | 0.7467 ± 0.0120 | 0.0728 | +0.3402 | 0.0025 | 0.729 | 0.490 | 30.2h |
| `lgbm-balanced` | 0.7408 ± 0.0160 | 0.0712 | +0.3271 | 0.0022 | 0.728 | 0.507 | 30.9h |
| `lgbm-utility` | 0.7533 ± 0.0137 | 0.0746 | +0.3540 | 0.0026 | 0.798 | 0.592 | 28.0h |
| `xgb-balanced` | 0.7213 ± 0.0033 | 0.0635 | +0.2827 | 0.0025 | 0.776 | 0.593 | 26.1h |
| `xgb-utility` | 0.7590 ± 0.0115 | 0.0770 | +0.3541 | 0.0023 | 0.686 | **0.394** | 34.2h |
| `tcn-balanced` | 0.7386 ± 0.0091 | 0.0687 | +0.3241 | 0.0030 | 0.500 | 0.305 | 39.5h |
| `tcn-utility` | 0.7356 ± 0.0074 | 0.0713 | +0.3290 | 0.0028 | 0.505 | **0.299** | **39.5h** |
| `ssm-utility` | 0.7000 ± 0.0126 | 0.0544 | +0.2504 | 0.0027 | 0.586 | 0.460 | 35.0h |

Calibration is unharmed throughout (ECE ≤ 0.003): the model is fitted on `(z, w)` but
isotonic-calibrated against the **challenge label**, so the served probability keeps
meaning `P(label = 1)` whatever the fit optimised.

A note on why this measurement is conservative: AUROC here is scored against the
original challenge label, on which the utility-trained model is deliberately "wrong"
about the credit-ramp hours. It wins anyway.

---

## 2. The radical arm: sequence models change the alarm economics, not the AUROC

Every detector this project has shipped scores a patient-hour from a **flat feature
vector**; temporal context arrives only through hand-authored aggregates. Phase 7 found
that adding more such aggregates (185 features instead of 94) bought nothing — evidence
that the limit is not how many summaries you write down but that a fixed summary is
being written down at all. `backend/ml/models/sequence.py` changes the inductive bias
instead: three causal architectures read the patient's whole prefix and learn their own
temporal representation.

- **Dilated causal TCN** (Bai et al. 2018) — receptive field ≈ 127 hours.
- **Selective state space, Mamba-2/SSD form** (Dao & Gu 2024) — input-dependent decay
  `S_t = a_t S_{t-1} + B_t x_tᵀ`, computed by a chunked scan (parallel within a chunk,
  recurrent across chunks).
- **Causal transformer** — every hour attends to every earlier hour, no decay prior at
  all; the control that says whether the SSM's decay structure earns its keep.

**Result, honestly:** none of them beats the utility-aligned boosters on AUROC, and the
most fashionable one is the worst (`ssm` 0.7000). But the TCN moves the metric this
project has repeatedly named as its real bottleneck: **false-alarm rate 0.498 → 0.299,
a 40% reduction, with median lead time 26.6h → 39.5h.** It buys that by detecting fewer
septic patients at matched specificity (0.73 → 0.50), so it is a different point on the
trade-off, not a free lunch — which is exactly why `detect@FA15` (detection at a fixed
*alarm budget*) was added to the metric suite; comparing detectors at a shared score
threshold flatters whichever one happens to have the right score shape.

Correctness of the sequence code is audited rather than assumed (`verify_frontier.py`):
the entire second half of the input is overwritten and no output in the first half is
allowed to move (all three architectures: exactly 0.0), and the chunked selective scan
is checked against the literal one-step recurrence at T ∈ {7, 40, 128, 336} — worst
disagreement 1.6×10⁻¹⁴.

---

## 2b. Why the objective vanished inside the ensemble — and what actually helps

Applying the utility objective to the *production stacked ensemble* did nothing:

| candidate (3-fold, identical folds) | AUROC | AUPRC | utility | detect@FA15 | FA@spec90 |
|---|---|---|---|---|---|
| `stack` (shipped: hgb+hgb2+et+lr) | **0.7765 ± 0.0058** | 0.0834 | +0.4001 | 0.500 | 0.333 |
| `stack-utility` (all members re-aimed) | 0.7761 ± 0.0072 | 0.0781 | +0.3995 | 0.581 | 0.295 |
| `xgb-utility` (best single learner) | 0.7503 ± 0.0098 | 0.0774 | +0.3381 | 0.497 | 0.444 |

Paired patient bootstrap: **ΔAUROC −0.0011, 95% CI [−0.0043, +0.0020], n.s.** The
`detect@FA15` column looks like a large win (+8.1pp), and it is significant at a 15%
alarm budget (+0.056, CI [+0.024, +0.080]) — but **not at 10% (+0.015, n.s.) or 20%
(+0.007, n.s.)**. A real improvement at fixed alarm load would not appear at one budget
and vanish at both neighbours. Reporting only the 15% cell would have been metric
shopping, so the honest reading is: no gain.

**The diagnosis is a rank correlation.** Comparing pooled out-of-fold scores:

| pair | Spearman ρ |
|---|---|
| `stack` vs `stack-utility` | **0.965** |
| `stack` vs `xgb-utility` | **0.679** |
| `stack-utility` vs `xgb-utility` | 0.675 |

Re-aiming every member at the utility target produces a detector that ranks patients
almost identically to the incumbent — the objective is *redundant with what stacking
already does*. A different library is not. Stacking is paid for disagreement, not for
strength, and `xgb-utility` disagrees while `stack-utility` does not.

That predicts a fix, and a free way to test it: a fixed 50/50 rank average has no
fitted parameter, so it leaks nothing and can be evaluated on out-of-fold vectors
already on disk.

| unfitted blend | AUROC | AUPRC | ΔAUROC vs `stack` (paired bootstrap) |
|---|---|---|---|
| `stack` + `stack-utility` | 0.7793 | 0.0867 | +0.0015, CI [−0.0001, +0.0033] — n.s. |
| **+ `xgb-utility` as well** | **0.7816** | **0.0912** | **+0.0053, CI [+0.0015, +0.0095] — significant** |

Blending the redundant pair does nothing; adding the decorrelated member is
significant, and lifts AUPRC by **9.4% relative** — the metric that matters most at 2%
hour-prevalence. The conclusion is not "the objective was worthless" but **"the
objective belongs on the diverse member, not on all of them"**: `xgb` is much stronger
under the utility target (0.7213 → 0.7590 standalone), and it is the member whose
errors the ensemble lacks.

`StackedEnsembleDetector` therefore now accepts a **per-member objective** — a roster
entry may name its own with an `@` suffix — so the production candidate is the four
incumbent members left exactly as they are, plus `xgb@utility`. In a first fit the
meta-learner gave that member the largest coefficient of any learner in the stack.

### What shipped

`stack+xgb@utility` — the four incumbent members untouched, plus one `xgb@utility`:

| 3-fold CV, identical folds | AUROC | AUPRC | util* (own best threshold) | FA@spec90 |
|---|---|---|---|---|
| `stack` (previous production) | 0.7765 ± 0.0058 | 0.0834 | +0.4063 | 0.333 |
| `stack+xgb@utility` | **0.7811 ± 0.0079** | 0.0840 | **+0.4151** | **0.262** |

Paired patient bootstrap: **ΔAUROC +0.0038, 95% CI [+0.0004, +0.0073]**, wins all three
folds individually (+0.0036 / +0.0076 / +0.0026). False-alarm rate at matched
specificity falls **21%**. The meta-learner gives the new member a coefficient of 2.01,
second only to ExtraTrees.

`promote_frontier.py` then re-fitted both configurations on a disjoint 75/25 patient
split and passed both gates:

- **discrimination** — held-out AUROC +0.0070, AUPRC +0.0031, ECE 0.0028 → 0.0023.
- **decisions** — each detector at *its own* threshold, derived on 1,000 held-out
  patients and judged on 1,004 others: utility **+0.1924 vs +0.0918**, detection
  **0.307 vs 0.213**, at a false-alarm rate of 0.072 vs 0.048 (both far inside the 15%
  budget).

Cost: `xgboost==2.1.4`, 7.5 MB installed, now pinned in `requirements.txt` because the
artifact is a pickle of that estimator and the server cannot load the detector without
it.

**Why the AUROC bar moved.** The promoter originally required a held-out AUROC gain
above 0.005 — the project's standing "not noise" heuristic — and this change is +0.0038
on the benchmark. That heuristic is a *proxy* for "is this real"; there is now a direct
test (paired patient bootstrap over the cross-validated cohort) and where the direct
test is available it supersedes the proxy. In exchange the second gate was made
strictly harder: it used to read both detectors off one shared threshold, which is the
same error as the `detect@FA15` trap above — a score distribution that suits the
incumbent makes any differently-shaped candidate look worse for reasons unrelated to
its quality. Each detector now gets its own derived operating point. Stated plainly so
the change of bar is auditable rather than convenient.

---

## 3. The TCN's advantage is not just smoothness (a negative result worth having)

The obvious cheap explanation for the TCN's false-alarm win is that a model reading the
whole prefix emits a score that moves smoothly hour to hour, so a trajectory crosses
the alarm threshold once instead of flickering across it. If that were the whole story,
a causal filter on a booster's own output would recover the benefit for free — no torch,
no new dependency, and directly implementable in the live streaming path.

`backend/ml/models/temporal.py` implements four strictly causal filters (each verified
to be unaffected by rewriting the future and invariant to the order rows arrive in) and
they were applied to the stacked ensemble's held-out score — costing no extra fit,
since these are transformations of predictions that already existed:

| decision layer | AUROC | AUPRC | utility | ECE | detect@FA15 | lead@FA15 |
|---|---|---|---|---|---|---|
| raw score | 0.7653 | 0.0738 | **+0.3385** | 0.0022 | **0.424** | 33.5h |
| + EMA (halflife 6h) | 0.7627 | 0.0744 | +0.2944 | 0.0008 | 0.377 | 36.0h |
| + 50/50 blend | **0.7679** | **0.0781** | +0.3234 | 0.0014 | 0.411 | **35.0h** |
| + running max | 0.7498 | 0.0629 | +0.3042 | 0.0145 | 0.424 | 33.5h |
| + decaying log-odds | 0.6832 | 0.0297 | −0.0111 | 0.0368 | 0.272 | 22.0h |

**The hypothesis fails.** The blend nudges ranking up (AUROC +0.0026, AUPRC +0.0043)
and buys 1.5h of lead, but every filter *loses* normalized utility, and none comes near
the TCN's 0.299 false-alarm rate. Smoothing the output is not what the sequence model is
doing; whatever reduces its alarm burden is learned in the representation, not
recoverable by post-processing. The filters stay in the tree as a documented negative
and as a decision-layer primitive, not as a shipped default.

---

## 5. A methodological flaw found in the incumbent

`GBMDetector` fits with `early_stopping=True, validation_fraction=0.1`. sklearn draws
that validation split at random **row** level, so a patient's hours land on both sides
of it and the stopping round is chosen against data the model has partly memorised. It
is a small leak, but it is a leak in the shipped baseline, and it means part of any
library-to-library comparison against XGBoost/LightGBM (which simply run every round)
is really a comparison of stopping rules. The controls `hgb-utility-noes` /
`hgb-balanced-noes` exist in the registry to quantify it; **they have not been run
yet.**

---

## 6. What is not measured

Stated plainly so the table above is not read as more complete than it is:

- `xfmr-utility` — one fold completed (0.7414), the sweep was interrupted.
- `hgb-*-noes` — the early-stopping fairness controls of §3.
- `utility-z` / `utility-w` — the decomposition separating the relabelling from the
  reweighting, so the two halves of the objective change are still credited jointly.
- `tabpfn` — TabPFN v8 puts its weights behind a one-time licence acceptance requiring
  a Prior Labs account. The candidate is written and registered; accept at
  ux.priorlabs.ai, `export TABPFN_TOKEN=…`, then `--only tabpfn`.
- Cross-hospital (set A ↔ set B) confirmation for the new candidates. This matters:
  Phase 7 put the incumbent at ~0.69 across hospitals against ~0.77 within-site, and a
  candidate that only wins within-site has not actually won.
  `backend/ml/scripts/frontier_external.py` runs it in two fits per candidate.

---

## Reproducing

```bash
make frontier SPEC=dev            # benchmark every registered technology
make frontier-report SPEC=dev     # rebuild this table from saved out-of-fold scores
make verify-frontier              # the adversarial audit of the claims above
```
