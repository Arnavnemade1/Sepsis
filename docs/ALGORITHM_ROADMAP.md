# Algorithm & Feature Roadmap — evidence before novelty

This repository has already ruled out the easy answer: more feature windows,
better post-processing, recursive hyperparameter search, HDC variants, random
convolutions, and unsupervised hospital adaptation do not materially improve the
current score.  The next work should therefore improve **signal**, **decision
quality**, or **operational safety** — and each proposal below has a falsifiable
success criterion.

## Ship now: alert episodes, not per-tick pseudo-alerts

The runtime now exposes `alert_episode` on every `AegisAssessment`.  It treats a
continuous MEDIUM/HIGH interval as one advisory episode, emits a fresh event on
escalation to HIGH, permits bounded re-notification only for unresolved HIGH
risk, and marks resolution.  It does not modify risk, treatment, or external
notifications.

Why this complements the current system:

* `alarm_policy.py` measures only the *first* alarm in a stay.  It cannot tell
  whether an interface will produce one usable alert or hundreds of duplicates.
* The calibrated detector and conformal layer remain the source of risk and
  uncertainty.  An uncertain score is visible in an episode but is never used
  to suppress an escalation.
* The state machine makes operational outcomes measurable before a pager or UI
  workflow is attached.

Study it on patient-held-out CinC trajectories before enabling notifications:

| Primary outcome | Candidate policy must beat | Safety guard |
|---|---|---|
| non-septic alert episodes / patient-day | per-tick notification baseline | first pre-onset alert detection may not fall (lead time is **retracted** — a random score reproduces it; see `docs/LEAD_TIME_NULL.md`) |
| repeat notifications / septic episode | unbounded duplicate baseline | every MEDIUM→HIGH escalation is immediately emitted |
| active-episode duration | report only | no retrospective "resolution" before the score actually returns LOW |

Use grouped train/search/locked patient splits and a paired patient bootstrap.
The full candidate field, including policies that lose, belongs in the artifact.

## Highest-value algorithm studies

1. Waveform-to-risk incremental value (highest expected yield)

   The documented ceiling is constrained by hourly ICU values that omit HRV,
   pulse morphology, respiratory effort, motion, and recovery dynamics.  The
   MIMIC waveform adapter and the BIDMC PPG↔ECG validation path already exist.
   Build a patient-disjoint, time-aligned study with three locked comparisons:
   tabular stack; waveform-only; tabular + waveform.  The endpoint is the
   incremental AUPRC, utility, and detection at a pre-declared alert burden —
   not a peak AUROC.  Hold a hospital out completely; do not create waveform
   windows using future outcome information.

2. Real wearable transport study

   The most consequential uncertainty is the mapping from ICU core temperature,
   cuff BP and blood lactate to skin temperature, cuffless BP and wearable
   proxies.  Build a channel-by-channel paired study that records the reference,
   wearable estimate, latency, availability and quality.  Feed the *measured*
   wearable values through the existing `StreamingFeatureBuilder`, then measure
   score and operating-point transport.  A model improvement is not a win if its
   required channel is unavailable too often to serve.

3. Personalisation as a gated companion signal

   `PatientBaselineLearner` already captures clinically useful within-patient
   deviations, but it has only cohort-simulator evidence and must not silently
   change the calibrated production probability.  Evaluate it as a companion:
   population alert versus "population alert OR sustained personal deviation".
   Require non-inferior first-alert detection at a fixed false-alarm ceiling,
   plus a bootstrap confidence interval for earlier alerts.  Train its warm-up,
   freeze and persistence settings only on a search split.  If it loses, retain
   its explanation panel without changing alerting.

4. Episode-aware alarm economics

   Extend `backend/ml/alarm_policy.py` with episode metrics matching the runtime
   controller: alerts per patient-day, duplicate events per active episode,
   escalation recall, and time-to-clear.  The existing first-alarm metric remains
   mandatory; episode metrics are additional operational outcomes, not a way to
   hide missed patients.  Evaluate cooldown and re-notify schedules on locked
   patients only.

5. Acquisition-quality and missingness stress test

   The model carries missingness features and conformal quality strata, while
   camera estimates carry quality scores.  What is missing is a single
   end-to-end stress test: replay held-out trajectories with empirically measured
   dropout, stale samples, clock skew, motion artefacts, and one corrupted channel
   at a time.  Report discrimination, calibration, abstention and overrides by
   failure mode.  A candidate is admissible only if it degrades visibly rather
   than failing open or inventing certainty.

6. Clinician actionability / next-best-measurement study

   When conformal output is uncertain, do not force a lower-risk label.  Use the
   existing feature schema to identify which *available* measurement would most
   reduce missingness/staleness for that hour, and label it as an evidence request
   rather than a diagnosis.  Validate recommendations by information gain on a
   locked set and prohibit recommendations whose value is unsupported by a source
   channel.  This makes uncertainty actionable without claiming it improves AUROC.

## Do not spend the next cycle on

* A new tabular architecture or HDC encoding alone: this has been tested across
  several independent representation families and the dominant evidence says
  signal, not capacity, is limiting.
* Threshold retuning or label-free prevalence correction per hospital: both have
  already been shown to move along the ROC curve or harm locked transfer.
* Using conformal abstention as an automatic silence/escalate rule: the locked
  analysis showed both readings lose to the base policy.

## Promotion contract for every new feature

1. Freeze the clinical question, primary endpoint, alarm budget, and no-harm
   constraint before looking at locked outcomes.
2. Split by patient; for transport, split by site before doing any feature or
   threshold selection.
3. Bind the model, calibration, conformal bands, and operating point with the
   existing fit identity.  No distribution-derived object transfers across fits.
4. Bootstrap patients, not hourly rows; show the full candidate field and all
   failures.
5. Keep an advisory path until the feature wins its gate.  A useful explanation
   is not permission to change a clinical risk tier.
