# AegisOS — bundled real-patient sample

This directory contains **32 real, de-identified ICU stays** (hourly vitals +
labs + ground-truth sepsis labels) curated from the public

> **The PhysioNet/Computing in Cardiology Challenge 2019: Early Prediction of
> Sepsis from Clinical Data.** Reyna MA, Josef C, Jeter R, Shashikumar SP, Moody
> B, Westover MB, Sharma A, Nemati S, Clifford GD. *Computing in Cardiology* 2019.

Source: https://physionet.org/content/challenge-2019/1.0.0/
License: **Creative Commons Attribution 4.0 International (CC BY 4.0)** —
redistribution permitted with attribution (provided above).

These are the unmodified Challenge `.psv` files for the curated cohort: 8 severe
septic, 8 mild septic and 16 stable stays, from both hospitals. AegisOS replays them
through the bio-kernel so the demo runs on **real ICU data**, not synthetic, on any
deploy, with or without the full 40,000-patient corpus.

**Every patient here is one the serving detector never trained on.** The ward is
scored by the shipped sepsis detector (`backend/aegis/ward_risk.py`), so a bed that
replayed one of its training patients would show an in-sample prediction. The sample
is drawn from the detector's promotion holdout, and `heldout.json` lists all 8,062
patients in it with the fit they belong to (`7fbbdd0ad322`). Selection used the same
rules as `build_cohort` (record length, onset at or after hour 6, a severity score on
the vitals near onset) and never looked at the detector's output, so the demo is not
picked toward cases the detector happens to catch.

The sample is the ward's source locally and in production alike. Only a caller that
needs more beds than it holds (the scale harness) reads the full corpus, and then
only patients on the held-out list. To rebuild it for a new detector fit, re-run the
curation against that fit's holdout and replace `heldout.json`;
`make verify-ward` refuses a list stamped with a different fit.
