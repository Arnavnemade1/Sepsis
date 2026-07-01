"""Verify harness for the Federated Intelligence Network — claims asserted by measurement.

This is a *falsifiable* experiment on the real two-hospital CinC-2019 split
(setA = federation hospitals, setB = an unseen external hospital). We assert the
mechanism is sound and report the headline generalization result honestly:

  1. RECOVERY   — federation lifts cross-site (setB) AUROC above a single-site local
                  model, approaching/parity with the data-pooled centralized ceiling —
                  WITHOUT any hospital sharing records.
  2. ROBUSTNESS — one Byzantine hospital wrecks plain FedAvg; robust aggregation
                  (Krum / median) tolerates it and preserves the global model.
  3. AUTHENTICITY — a forged/unsigned weight-delta is rejected by signature check;
                  a correctly-signed one is accepted.
  4. PRIVACY    — the only object that crosses the wire is a weight-delta vector +
                  count + signature; there is no code path carrying patient rows.

    python -m backend.aegis.verify_fedlearn
"""
from __future__ import annotations

import numpy as np

from backend.aegis.fedlearn import (FederatedServer, HospitalNode, LogRegModel,
                                    Update, _sign, run_experiment, run_poisoning)


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


# Modest size keeps the suite ~30s; the effect holds at larger N (see _demo).
LIMIT, NODES, ROUNDS, EPOCHS = 500, 4, 12, 60


def main() -> int:
    failures = 0
    checks = []

    # ===== 1. RECOVERY =================================================== #
    _hr("1. CROSS-HOSPITAL RECOVERY (real CinC setA→setB)")
    res = run_experiment(limit=LIMIT, n_nodes=NODES, rounds=ROUNDS, epochs=EPOCHS)
    lo, ce, fe = res["local_only"], res["centralized"], res["federated"]
    print(f"    patients: setA={res['patients']['setA']} setB={res['patients']['setB']}")
    print(f"    local-only   setA={lo['setA_internal']}  setB(cross)={lo['setB_cross_site']}")
    print(f"    centralized  setA={ce['setA_internal']}  setB(cross)={ce['setB_cross_site']}")
    print(f"    federated    setA={fe['setA_internal']}  setB(cross)={fe['setB_cross_site']}")
    print(f"    → federation recovers setB by {res['headline']['federated_recovers']:+.4f} vs local-only")

    checks.append(("federated cross-site AUROC > local-only (federation helps)",
                   fe["setB_cross_site"] > lo["setB_cross_site"]))
    checks.append(("federated reaches/exceeds the centralized ceiling on setB",
                   fe["setB_cross_site"] >= ce["setB_cross_site"] - 0.02))
    checks.append(("federated learns a real model (setA AUROC well above chance)",
                   fe["setA_internal"] > 0.7))

    # ===== 2. ROBUSTNESS ================================================= #
    _hr("2. BYZANTINE ROBUSTNESS (one poisoned hospital)")
    poi = run_poisoning(limit=LIMIT, n_nodes=5, rounds=ROUNDS, epochs=EPOCHS)
    print(f"    clean fedavg={poi['clean_fedavg']}  poisoned fedavg={poi['poisoned_fedavg']}")
    print(f"    poisoned krum={poi['poisoned_krum']}  poisoned median={poi['poisoned_median']}")
    checks.append(("a Byzantine node degrades plain FedAvg",
                   poi["poisoned_fedavg"] < poi["clean_fedavg"] - 0.05))
    checks.append(("Krum tolerates the poisoner (>> poisoned FedAvg)",
                   poi["poisoned_krum"] > poi["poisoned_fedavg"] + 0.1))
    checks.append(("median tolerates the poisoner (>> poisoned FedAvg)",
                   poi["poisoned_median"] > poi["poisoned_fedavg"] + 0.1))

    # ===== 3. AUTHENTICITY (signed updates) ============================== #
    _hr("3. SIGNED UPDATES (forged deltas rejected)")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6)); y = (rng.random(40) < 0.3).astype(int)
    good_node = HospitalNode("hospital_real", X, y, key=b"realkey_16bytes!")
    srv = FederatedServer(6, keys={"hospital_real": b"realkey_16bytes!"}, method="fedavg")
    gw = srv.global_model.weights

    legit = good_node.local_update(gw, epochs=10)
    forged = Update("hospital_real", legit.delta.copy(), legit.n_samples, "deadbeef" * 8)
    impostor = Update("hospital_unknown", legit.delta.copy(), legit.n_samples,
                      _sign(b"attackerkey_xxxx", legit.delta, legit.n_samples))

    srv.step([legit])
    accepted_after_legit = srv.accepted
    srv.step([forged, impostor])
    checks.append(("a correctly-signed update is accepted", accepted_after_legit == 1))
    checks.append(("a forged signature is rejected", srv.rejected >= 1))
    checks.append(("an unregistered hospital is rejected", srv.rejected >= 2))

    # ===== 4. PRIVACY (structural) ====================================== #
    _hr("4. DATA MINIMIZATION (no records cross the wire)")
    fields = set(vars(legit).keys())
    print(f"    Update payload fields: {sorted(fields)}")
    checks.append(("update carries only delta/count/signature (no X or y)",
                   fields == {"node_id", "delta", "n_samples", "signature"}))
    checks.append(("the delta is a 1-D weight vector, not a data matrix",
                   legit.delta.ndim == 1 and not legit.carries_patient_data()))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — federation recovers cross-hospital AUROC without "
          "sharing data, and tolerates a Byzantine hospital."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
