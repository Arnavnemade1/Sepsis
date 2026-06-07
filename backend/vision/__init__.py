"""Contactless optical biosensing for HyperSepsisTwin-Ω.

Extracts physiological signals from a plain RGB camera (e.g. the MacBook webcam):

  * rPPG  -> heart rate, heart-rate variability, respiratory rate
  * micro-perfusion proxy (skin chrominance / spatial perfusion mapping)
  * sympathetic-distress proxies (micro-tremor, pupil variability)

Scientific honesty (read this):
  - rPPG-derived **heart rate** is a well-validated measurement and is checked
    here against synthetic ground truth.
  - rPPG **HRV** is feasible but webcam-noisy; always reported with a quality
    score, never as a clean clinical value.
  - The **perfusion** and **distress** channels are *proxies / research signals*.
    They are explicitly labelled experimental and are NOT validated surrogates
    for blood pressure, lactate, or a clinical sepsis diagnosis.

The camera does not "diagnose sepsis". It produces a subset of the same
physiological channels the data-grounded detector (backend/ml) already consumes,
plus experimental proxies, each with a confidence the detector can weight.
"""
