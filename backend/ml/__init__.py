"""HyperSepsisTwin-Ω machine-learning subsystem.

This package grounds the project's detection in *real* labelled clinical data
(PhysioNet/CinC 2019) instead of hand-tuned heuristics. It provides:

- ``data``    : dataset download + parsing + feature engineering
- ``eval``    : leakage-free patient-level splits and clinical metrics
- ``models``  : the data-grounded HDC detector and benchmark models

Nothing in here invents physiology. Every threshold, weight, and archetype is
either learned from data or reported with the metric that justifies it.
"""
