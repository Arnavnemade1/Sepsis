"""The GENESIS experiments, in the order they were run (results: ``docs/GENESIS.md``).

Run any of them from the repo root, e.g. ``python -m backend.ml.genesis.experiments.e3_heads``.
All use the dev folds only; none touches the locked set.

e1_baselines        single LightGBM, all 510 features vs the 7 wearable vitals
e3_heads            the calibrated heads (direct/who/when/horizons/label-genesis
                    auxiliaries) and every stage-2 combiner
                    (an earlier e2 used UNcalibrated heads and is superseded by e3)
e5_p19              the P19-style patient-level task (0.964 — record-length leakage)
e6_network          the Genesis Network, 3 seeds, calibrated
e7_learning_curve   the ceiling: AUROC vs training patients, power-law asymptote
e8_intent           clinician-intent features (null inside the product)
e9_network_in_product   the pre-registered rule for adding the network as an expert
e10_tune_heads      regularised boosting for the three heads (+0.0065 for direct)
e11_direct_round2   a second regularisation round (converged: +0.0001)
e12_ssl             self-supervised pretraining of the network (pre-registered NULL:
                    +0.0008 [-0.00002, +0.0017]; within-septic timing +0.012)
"""
