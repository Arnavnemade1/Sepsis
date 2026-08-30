.PHONY: help setup verify verify-core verify-alert-episodes verify-evidence-router verify-evidence-replay verify-integrity verify-frontier verify-supervision verify-coordination verify-recovery verify-mimic3 mimic3-cohort mimic3-windows recovery-outcome verify-scenarios scenarios frontier frontier-report verify-agent verify-doctor verify-llm verify-pk verify-ae verify-pk-api verify-opts verify-z3 verify-alarm verify-selfimprove verify-subgroup verify-op verify-conformal conformal conformal-external verify-minirocket verify-shift transfer verify-coupling falsify verify-falsify verify-fitid subgroup-audit alarm-policy serve train-gbm train-stack self-improve operating-point rigorous clean-cache

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SPEC ?= dev
HOST ?= 127.0.0.1
PORT ?= 8100

help:
	@echo "AegisOS developer commands"
	@echo ""
	@echo "  make setup        Create/update the local Python environment"
	@echo "  make verify       Run the full end-to-end verification suite"
	@echo "  make verify-core  Run the fast legacy core/HDC/rules checks"
	@echo "  make verify-alert-episodes  Alert lifecycle, escalation and stale-event checks"
	@echo "  make verify-evidence-router AERIS source, budget, override and stale-event checks"
	@echo "  make verify-evidence-replay AERIS patient-level replay metric checks"
	@echo "  make verify-integrity  Adversarial audit: inject known lies, prove the guards fire"
	@echo "  make verify-alarm      Alarm-policy search + promotion gate (16 checks)"
	@echo "  make verify-supervision Supervision density: the window-is-inert control"
	@echo "  make verify-coordination Coordination plane: matched-pair alignment surrogate"
	@echo "  make verify-recovery   Autonomic recovery: known-tau, amplitude invariance, causality"
	@echo "  make verify-mimic3     Open-access outcome pipeline: plants 5 arterial-line artifacts"
	@echo "  make mimic3-cohort     Stream the open MIMIC-III decompensation cohort (hours, resumable)"
	@echo "  make mimic3-windows    Fetch ECG windows for the discovered outcomes (resumable)"
	@echo "  make recovery-outcome  Does recovery slowing precede circulatory decompensation?"
	@echo "  make verify-frontier   Accuracy-frontier audit: causality, exact utility, leak tripwire"
	@echo "  make verify-scenarios  Scenario Lab audit: sandbox isolation + the synthetic-data firewall"
	@echo "  make scenarios    Run the standing scenario suite against the shipped detector"
	@echo "  make verify-selfimprove Self-improvement acceptance rule (14 checks)"
	@echo "  make verify-subgroup   Subgroup audit: global threshold, patient bootstrap, multiplicity"
	@echo "  make verify-op         Operating point: in-range rule, cap margin, detector identity"
	@echo "  make verify-conformal  Conformal abstention: plants leakage/shift/degeneracy, asserts each shows"
	@echo "  make verify-minirocket Causal MiniRocket: plants a future spike, asserts no earlier hour moves"
	@echo "  make verify-shift      Label-free transfer: label vs covariate vs concept shift, and which remedy fails"
	@echo "  make transfer          Two-hospital deployment transfer under a locked gate"
	@echo "  make verify-coupling   Cardio-respiratory coupling on real BIDMC (mechanism, not detection)"
	@echo "  make falsify           Falsification audit: prove the checks themselves can FAIL"
	@echo "  make verify-falsify    The engine audited by its own rule"
	@echo "  make verify-fitid      Fit fingerprints: a cross-fit read must be refused"
	@echo "  make verify-pk    PBPK treatment-projection checks (mass balance, Vd/CL/t½ recovery)"
	@echo "  make verify-ae    Adverse-event checks (anchoring to published rates, evidence firewall)"
	@echo "  make verify-pk-api  Treatment-projection REST contract (in-process, no server)"
	@echo "  make verify-opts  Treatment options: reasons traceable to numbers, cage refusals shown"
	@echo "  make verify-z3    Z3 concurrency: every entry point guarded, no context corruption"
	@echo "  make verify-doctor AI-doctor consult: grounding, provenance, scale firewall, failover"
	@echo "  make verify-agent  Agent plane: catalogue, read confinement, gated writes"
	@echo "  make verify-llm    LLM transport: fallback chain, cooldown, cache, budget"
	@echo "  make serve        Start the AegisOS API + web shell"
	@echo "  make train-gbm    Train the calibrated GBM/HDC comparison baseline"
	@echo "  make train-stack  Train the stacked detector candidate"
	@echo "  make frontier     Benchmark every detector technology on identical folds"
	@echo "  make frontier-report  Rebuild the frontier results table from saved OOF"
	@echo "  make self-improve Recursive self-improvement search (gated on locked patients)"
	@echo "  make operating-point  Derive the alarm thresholds from data (add --save to apply)"
	@echo "  make subgroup-audit   Per-subgroup fairness + calibration audit of the shipped detector"
	@echo "  make conformal        Derive the abstention layer (add --save to apply)"
	@echo "  make conformal-external  Calibrate on hospital A, measure coverage on hospital B"
	@echo "  make rigorous     Cross-hospital evaluation with reduced bootstrap"
	@echo "  make clean-cache  Remove regenerated feature caches"

setup:
	python3 -m venv .venv
	$(PIP) install -r backend/requirements.txt

verify:
	$(PY) verify_full_system.py

verify-frontier:
	$(PY) verify_frontier.py

# Denser supervision of septic stays raises detection at every alarm budget. The
# load-bearing check asserts a NULL: that the TEMPORAL PLACEMENT of the extra positive
# hours does not matter. An earlier version of the module claimed it did, and this
# control is what falsified it — so it stays pointed the same way.
verify-supervision:
	PYTHONPATH=. $(PY) -m backend.ml.scripts.verify_supervision

# Cross-channel coordination (Buchman's organ-uncoupling hypothesis) at hourly
# resolution. Matched-pair surrogate: independent per-channel delays destroy alignment,
# a common delay adds the same staleness with alignment intact, and the gap between
# them is what alignment is worth. Measured: nothing.
verify-coordination:
	PYTHONPATH=. $(PY) -m backend.ml.scripts.verify_coordination

# The new detection algorithm: recovery SPEED of the autonomic control loop rather than
# response SIZE, which is what beta-blockade and sedation destroy. 14 checks, and two of
# them assert limitations rather than capabilities: the measured ECG-vs-PPG concordance
# is poor, and the pipeline must refuse a pulse source while that is true.
verify-recovery:
	PYTHONPATH=. $(PY) -m backend.physio.verify_recovery

# The workaround for not having credentialed data. MIMIC-III Waveform is OPEN ACCESS
# (ODbL): ~30k ICU patients with ECG and an arterial line, but no clinical chart — so the
# outcome is derived from the arterial line while the predictor reads only the ECG, which
# makes the label non-circular. This suite plants five arterial-line artifacts
# (disconnection, zeroing, flush, damping, flatline) and requires each to be rejected AND
# to produce no event, because otherwise a detector here learns to predict line handling.
verify-mimic3:
	PYTHONPATH=. $(PY) -m backend.physio.data.verify_mimic3_open

mimic3-cohort:
	PYTHONPATH=. $(PY) -m backend.physio.data.build_mimic3_cohort --target-events 200

# Phase 2: fetch the ECG only for the minutes each discovered outcome points at, and
# reduce to features. Resumable; the monitor's own HR channel gates beat detection,
# because a missed beat is indistinguishable from an enormous physiologic perturbation.
mimic3-windows:
	PYTHONPATH=. $(PY) -m backend.physio.data.fetch_mimic3_windows --max-per-patient 3

# The experiment: does autonomic recovery slowing precede circulatory decompensation,
# and does it add anything over heart rate and classic HRV? Refuses to report a number
# on a cohort too small to defend one.
recovery-outcome:
	PYTHONPATH=. $(PY) -m backend.physio.analyse_recovery_outcome

verify-scenarios:
	PYTHONPATH=. $(PY) backend/aegis/verify_scenarios.py

scenarios:
	$(PY) -c "from backend.aegis.scenarios import LAB; LAB.load_suite(); \
		[print(r.summary(), end='\n\n') for r in LAB.run_all()]; \
		r=LAB.report(); print(f\"SUITE: {r['passed']}/{r['ran']} passed\"); print(r['caveat'])"

frontier:
	$(PY) -m backend.ml.scripts.frontier --spec $(SPEC) --out artifacts/frontier/$(SPEC).json

frontier-report:
	$(PY) -m backend.ml.scripts.frontier_report artifacts/frontier/*_oof.npz \
		--spec $(SPEC) --baseline gbm --md docs/FRONTIER_REPORT.md

verify-core:
	$(PY) verify_system.py

# Alert episodes are independent of the detector: an elevated interval opens one
# advisory notification, escalation is never cooldown-suppressed, and out-of-order
# telemetry cannot re-open an old episode.  This verifies those safety semantics;
# it does not claim the timing parameters are clinically validated.
verify-alert-episodes:
	$(PY) -m backend.aegis.verify_alert_episodes

# AERIS only requests an explicitly available source, bounds ordinary requests to
# one per episode, and cannot delay a hard clinical escalation.
verify-evidence-router:
	$(PY) -m backend.aegis.verify_evidence_router

verify-evidence-replay:
	$(PY) -m backend.ml.scripts.verify_evidence_replay

# Verifies the CLAIMS, not the code: plants a known pathology (shuffled labels,
# in-sample predictions, zero-mean noise deltas, identical predictors, an out-of-range
# alarm) and asserts the honesty machinery catches each one. Complements `verify`.
verify-integrity:
	$(PY) verify_integrity.py

# The treatment-projection engine: checks the pharmacology, not the code. Mass
# balance to machine precision, published Vd/CL/t½ recovered off the simulated
# curves, and an independent integrator that re-derives the ODEs longhand.
verify-pk:
	$(PY) -m backend.aegis.verify_pharmacokinetics

# Adverse-event projection: the anchoring check (standard dose in a typical
# patient must return the PUBLISHED incidence exactly) plus the evidence firewall
# that stops an invented dose-response passing as a measured one.
verify-ae:
	$(PY) -m backend.aegis.verify_adverse

# The wiring, not the pharmacology: administered-vs-hypothetical labelling, cache
# keys that cover every covariate, and the guarantee that projecting mutates
# nothing. Runs the FastAPI app in-process — no server required.
verify-pk-api:
	$(PY) -m backend.aegis.verify_pk_api

# Candidate treatment paths: that every upside/downside traces to a number in its
# own option (change the patient, the reasons change), and that an option the Z3
# cage refuses is labelled refused with the cage's own words rather than hidden.
verify-opts:
	$(PY) -m backend.aegis.verify_treatment_options

# The agent plane — Dr. Aegis's control surface over the OS. Checks that every
# advertised tool is callable, that the universal read primitive stays a READ
# confined to /aegis, that every write goes through the Z3 trap gate and a
# refused write leaves the kernel untouched, and that a UI command never claims
# to have read what it only displayed. In-process, no LLM, no spend.
verify-agent:
	$(PY) -m backend.aegis.verify_agent

# The AI-doctor consult layer. Checks the properties the safety argument rests on
# — that the dossier names the criteria that fired, that its performance figures
# are read from the committed artifacts, that the two probability scales are
# firewalled apart, and that every LLM failure still lands on a template that
# refuses to prescribe. Offline (injected transports); AEGIS_DOCTOR_LIVE=1 adds
# one real call.
verify-doctor:
	$(PY) -m backend.llm.verify_doctor

# The LLM transport underneath it: fallback chain, cooldown, cache, budget.
verify-llm:
	$(PY) -m backend.llm.verify_llm

# The z3-solver bindings share one non-thread-safe global context. This walks the
# AST for any PUBLIC function that builds Z3 terms without the lock, then hammers
# the previously-unguarded paths from 8 threads to prove the context survives.
verify-z3:
	$(PY) -m backend.aegis.verify_z3_safety

serve:
	$(PY) -m uvicorn backend.aegis.server:app --host $(HOST) --port $(PORT)

train-gbm:
	$(PY) -m backend.ml.scripts.train_and_compare --feature-set wearable --save

train-stack:
	$(PY) -m backend.ml.scripts.train_stacked --feature-set wearable

# Searches its own champion each round and gates the winner on patients the search
# never touched. Add --promote to let a CONFIRMED gain replace the production artifact;
# without a confirmed gain it deliberately changes nothing.
self-improve:
	$(PY) -m backend.ml.scripts.self_improve --limit 2000 --rounds 4 --population 4

# Derives the alarm thresholds from out-of-fold predictions and gates them on held-out
# patients. Reports only; add --save to write artifacts/operating_point.json (it refuses
# to save a point that loses to the incumbent).
operating-point:
	$(PY) -m backend.ml.scripts.derive_operating_point --limit 4000

# The alarm POLICY, not the threshold: searches persistence / integrator /
# per-patient families against a locked holdout and refuses to promote anything
# that does not beat the shipped policy with a CI excluding zero.
verify-alarm:
	$(PY) -m backend.ml.scripts.verify_alarm_policy

# The self-improvement loop's acceptance rule: refuses for the right reasons, and
# still lets a genuine winner through.
verify-selfimprove:
	$(PY) -m backend.ml.scripts.verify_selfimprove

# Who does the shipped detector serve worse? Reports every metric per age x sex
# cell at the DEPLOYED threshold — never re-tuned per group, because a re-tuned
# audit measures a system that does not exist and hides the real disparity.
verify-subgroup:
	$(PY) -m backend.ml.scripts.verify_subgroup

# Each of these three constraints was once enforced somewhere other than where it
# could be broken: the in-range rule lived in the integrity audit, the detector
# identity was not recorded at all, and the false-alarm cap was imposed only on the
# scores used to pick the threshold.
verify-op:
	$(PY) -m backend.ml.scripts.verify_operating_point

# Standard MiniRocket is centred and pools PPV over the whole stay, so it reads
# forward in time. That failure would show up as a superb lead time rather than as an
# error, which is why it is planted here rather than trusted.
verify-minirocket:
	$(PY) -m backend.ml.scripts.verify_minirocket

# Three of these checks pass by showing a correction FAIL where it should: prior
# correction actively breaks a covariate-shifted site, and nothing label-free touches
# concept shift. A suite where every remedy always works would be evidence of nothing.
verify-shift:
	$(PY) -m backend.ml.scripts.verify_shift

# A check that cannot fail is a lie with a green tick. This mutates the SCIENTIFIC
# invariant — causality, the exchangeability unit, class conditioning, fit identity —
# and asserts a NAMED check notices. The number to read is the survivor list: claims
# nothing is actually defending. --gate fails only on regression.
falsify:
	$(PY) -m backend.ml.scripts.falsify_audit --save

verify-falsify:
	$(PY) -m backend.ml.scripts.verify_falsify

# Three false findings in one investigation were one arithmetic mistake: a quantity
# read off one model's scores and applied to another's. This refuses that at the
# point of use, and treats a MISSING fingerprint as unknown rather than compatible.
verify-fitid:
	$(PY) -m backend.ml.scripts.verify_fitid

# Mechanism validation only: BIDMC has no sepsis labels, so no AUROC/lead-time claim is
# testable here. Proves the estimators measure coupling and fall when it is destroyed.
verify-coupling:
	$(PY) -m backend.physio.verify_coupling

# Fits on one hospital, adapts on the other's UNLABELLED half, reports on a locked half.
transfer:
	$(PY) -m backend.ml.scripts.derive_transfer --limit 4000 --boot 200 --jobs 4

# Three of these checks pass by demonstrating a LOSS of coverage — row-level
# calibration flattering itself, distribution shift breaking the guarantee, and the
# non-class-conditional taxonomy going degenerate at 2% prevalence. A run where
# everything passes by looking healthy would be evidence of nothing.
verify-conformal:
	$(PY) -m backend.ml.scripts.verify_conformal

# alpha is a REQUEST, not a result: on correlated ICU hours the delivered coverage
# does not track it, so the level is chosen on a select split and confirmed on a
# locked one. Reports only; add --save to write artifacts/conformal.json.
conformal:
	$(PY) -m backend.ml.scripts.derive_conformal \
		--oof artifacts/subgroup_oof_dev_stack-patient.npz:stack-patient \
		--auto-alpha --sweep 0.03,0.04,0.05,0.06,0.08,0.10

# Exchangeability is what the guarantee trades on, and a different hospital is the
# textbook violation of it. This measures how far coverage actually falls.
conformal-external:
	$(PY) -m backend.ml.scripts.derive_conformal --external --alpha 0.06 --jobs 6

subgroup-audit:
	$(PY) -m backend.ml.scripts.subgroup_audit --candidate 'stack+xgb@utility' \
		--spec $(SPEC) --folds 3 --boot 400 --jobs 6

alarm-policy:
	$(PY) -m backend.ml.scripts.search_alarm_policy --save

rigorous:
	$(PY) -m backend.ml.scripts.rigorous_eval --feature-set wearable --boot 100

clean-cache:
	rm -rf artifacts/feature_cache
