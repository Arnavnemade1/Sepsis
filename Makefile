.PHONY: help setup verify verify-core verify-integrity verify-pk verify-ae verify-pk-api verify-opts verify-z3 verify-alarm verify-selfimprove alarm-policy serve train-gbm train-stack self-improve operating-point rigorous clean-cache

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
HOST ?= 127.0.0.1
PORT ?= 8100

help:
	@echo "AegisOS developer commands"
	@echo ""
	@echo "  make setup        Create/update the local Python environment"
	@echo "  make verify       Run the full end-to-end verification suite"
	@echo "  make verify-core  Run the fast legacy core/HDC/rules checks"
	@echo "  make verify-integrity  Adversarial audit: inject known lies, prove the guards fire"
	@echo "  make verify-alarm      Alarm-policy search + promotion gate (16 checks)"
	@echo "  make verify-selfimprove Self-improvement acceptance rule (14 checks)"
	@echo "  make verify-pk    PBPK treatment-projection checks (mass balance, Vd/CL/t½ recovery)"
	@echo "  make verify-ae    Adverse-event checks (anchoring to published rates, evidence firewall)"
	@echo "  make verify-pk-api  Treatment-projection REST contract (in-process, no server)"
	@echo "  make verify-opts  Treatment options: reasons traceable to numbers, cage refusals shown"
	@echo "  make verify-z3    Z3 concurrency: every entry point guarded, no context corruption"
	@echo "  make serve        Start the AegisOS API + web shell"
	@echo "  make train-gbm    Train the calibrated GBM/HDC comparison baseline"
	@echo "  make train-stack  Train the stacked detector candidate"
	@echo "  make self-improve Recursive self-improvement search (gated on locked patients)"
	@echo "  make operating-point  Derive the alarm thresholds from data (add --save to apply)"
	@echo "  make rigorous     Cross-hospital evaluation with reduced bootstrap"
	@echo "  make clean-cache  Remove regenerated feature caches"

setup:
	python3 -m venv .venv
	$(PIP) install -r backend/requirements.txt

verify:
	$(PY) verify_full_system.py

verify-core:
	$(PY) verify_system.py

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

alarm-policy:
	$(PY) -m backend.ml.scripts.search_alarm_policy --save

rigorous:
	$(PY) -m backend.ml.scripts.rigorous_eval --feature-set wearable --boot 100

clean-cache:
	rm -rf artifacts/feature_cache
