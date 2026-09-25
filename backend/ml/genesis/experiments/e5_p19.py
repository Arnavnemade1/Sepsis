"""E5: P19-style patient-level classification (whole record -> did sepsis occur).

One row per patient: the causal feature vector at the LAST hour of the record, whose
expanding statistics summarise the whole stay. Same dev folds as everything else."""
import os
from backend.ml import config
import sys, numpy as np
from backend.ml.genesis import harness as H, learners as L

c = H.Cohort.load()
last = c.raw.starts + c.raw.lens - 1
pts = np.arange(c.raw.n_patients)
dev_pts = pts[c.folds >= 0]
Fp = np.asarray(c.F[last], dtype=np.float32)          # (P, D)
yp = c.plabel.astype(float)
segp = pts                                            # each "row" is its own patient
oof = np.full(len(pts), np.nan)
for k in range(c.n_folds):
    tr = dev_pts[c.folds[dev_pts] != k]
    te = dev_pts[c.folds[dev_pts] == k]
    fit = L.fit_lgb(Fp, tr, yp[tr], segp[tr], strat=yp[tr].astype(int),
                    params={"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 50,
                            "feature_fraction": 0.4, "lambda_l2": 5.0}, early_stop=200)
    oof[te] = fit.predict(Fp, te)
    print(f"fold {k}: P19 AUROC {H.auroc(yp[te], oof[te]):.4f}  trees {fit.best_iter}", flush=True)
a = H.auroc(yp[dev_pts], oof[dev_pts])
per = [H.auroc(yp[dev_pts[c.folds[dev_pts]==k]], oof[dev_pts[c.folds[dev_pts]==k]]) for k in range(c.n_folds)]
print(f"P19-style patient model: pooled AUROC {a:.4f}  folds {np.mean(per):.4f}±{np.std(per):.4f}")
np.save(os.path.join(config.ARTIFACTS_DIR, "genesis", "p19_patient_oof.npy"), oof.astype(np.float32))
