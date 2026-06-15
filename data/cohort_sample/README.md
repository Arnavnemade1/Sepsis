# AegisOS — bundled real-patient sample

This directory contains **32 real, de-identified ICU stays** (hourly vitals +
labs + ground-truth sepsis labels) curated from the public

> **The PhysioNet/Computing in Cardiology Challenge 2019: Early Prediction of
> Sepsis from Clinical Data.** Reyna MA, Josef C, Jeter R, Shashikumar SP, Moody
> B, Westover MB, Sharma A, Nemati S, Clifford GD. *Computing in Cardiology* 2019.

Source: https://physionet.org/content/challenge-2019/1.0.0/
License: **Creative Commons Attribution 4.0 International (CC BY 4.0)** —
redistribution permitted with attribution (provided above).

These are the unmodified Challenge `.psv` files for the curated cohort (a mix of
severe-septic, mild-septic, and stable stays). AegisOS replays them through the
bio-kernel so the live demo runs on **real ICU data**, not synthetic — even on a
fresh deploy that does not have the full 40,000-patient corpus. When the full
corpus is present locally (`data/cinc2019/`), it is preferred for richer
curation; otherwise this bundled sample is used.
