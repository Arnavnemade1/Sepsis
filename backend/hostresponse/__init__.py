"""Host-response gene expression — infection vs STERILE systemic inflammation.

WHY THIS SUBSYSTEM EXISTS
-------------------------
The production detector reads hourly vitals and labs. It is therefore structurally
blind to the distinction that drives much of its false-alarm burden: a patient whose
heart rate, respiratory rate and white count are deranged by trauma, pancreatitis or
major surgery looks, in those channels, like a patient deranged by infection. No
amount of post-processing recovers a separation the input channels do not carry —
that is the documented conclusion of `backend/ml/alarm_policy.py`, and it is what
redirects effort here, to a different measurement rather than a better reading of
the same one.

THE TWO LINKS, AND WHICH ONE THIS CAN TEST
------------------------------------------
  Link A — an expression signature separates infectious sepsis from sterile
           inflammation, and that separation survives a change of cohort and
           platform.
           **Testable here**, on public GEO cohorts with infection-negative
           comparators.

  Link B — that separation reduces THIS product's false-alarm floor.
           **Not testable here, and not claimed.** No bed in this system's cohort
           carries transcriptomics; there is no per-patient input to fuse. Nothing
           in this subsystem may render against a patient, and the study artifact
           is asserted to contain no per-patient field.

Stating the untestable link explicitly is the point. A module that quietly implied
Link B from a Link A result would be making exactly the leap this repository exists
to refuse.

WHAT WOULD MAKE THE RESULT FAKE
-------------------------------
Archival cohorts are measured on different platforms in different labs, so a model
can score well by learning WHICH COHORT a sample came from rather than anything
about the patient. Three defences are pre-registered, not bolted on afterwards:
within-sample normalisation (no statistic crosses samples), leave-one-cohort-out
validation, and a cohort-identity detector whose success VOIDS the primary result.
See `run_study.py` and `preregistration.json`.
"""
