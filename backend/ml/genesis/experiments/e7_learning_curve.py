"""E7: learning curve of the direct booster — is the ceiling data or model?

Train on a random fraction of each dev fold's TRAINING patients (the test fold is
unchanged), for fractions 1/8, 1/4, 1/2, 1, and fit AUROC(n) = a - b * n^(-alpha).
``a`` is the AUROC this representation approaches with unlimited patients from the
same distribution: if it sits near today's number, more data or a better learner of
the same features cannot reach 0.90; if it sits far above, estimation (not
information) is the bottleneck."""
import os
from backend.ml import config
import sys, json, numpy as np
from scipy.optimize import curve_fit
from backend.ml.genesis import harness as H, learners as L

c = H.Cohort.load()
dev = c.dev_rows
fracs = [0.125, 0.25, 0.5, 1.0]
out = {}
for frac in fracs:
    oof = np.full(len(c.y), np.nan)
    n_pts = []
    for k in range(c.n_folds):
        tr, te = c.fold_rows(k)
        rng = np.random.default_rng(100 + k)
        pts = np.unique(c.seg[tr])
        keep = rng.choice(pts, size=int(round(frac * len(pts))), replace=False)
        m = np.zeros(c.raw.n_patients, bool); m[keep] = True
        trf = tr[m[c.seg[tr]]]
        n_pts.append(len(keep))
        fit = L.fit_lgb(c.F, trf, c.y[trf], c.seg[trf], strat=c.plabel[c.seg[trf]])
        oof[te] = fit.predict(c.F, te, raw=True)
    s = H.score(c, oof[dev], dev, with_utility=False)
    out[frac] = {"n_patients": float(np.mean(n_pts)), "auroc": s["auroc"],
                 "hourmatched": s["auroc_hourmatched"], "between": s["between"],
                 "within": s["within_septic"]}
    print(f"frac {frac:5.3f}  patients {np.mean(n_pts):7.0f}  AUROC {s['auroc']:.4f}  "
          f"hour-matched {s['auroc_hourmatched']:.4f}  B {s['between']:.4f}  W {s['within_septic']:.4f}", flush=True)

n = np.array([out[f]["n_patients"] for f in fracs])
for key in ("auroc", "hourmatched"):
    y = np.array([out[f][key] for f in fracs])
    f = lambda n, a, b, al: a - b * n ** (-al)
    try:
        (a, b, al), cov = curve_fit(f, n, y, p0=[y[-1] + 0.01, 1.0, 0.5], maxfev=20000,
                                    bounds=([0.5, 0, 0.01], [1.0, 1e3, 3.0]))
        err = np.sqrt(np.diag(cov))
        print(f"{key}: asymptote a = {a:.4f} ± {err[0]:.4f}  (alpha {al:.2f}); "
              f"at 2x data -> {f(2*n[-1], a, b, al):.4f}; at 10x -> {f(10*n[-1], a, b, al):.4f}")
        out[f"fit_{key}"] = {"a": a, "a_se": err[0], "b": b, "alpha": al,
                             "x2": f(2 * n[-1], a, b, al), "x10": f(10 * n[-1], a, b, al)}
    except Exception as exc:
        print(key, "fit failed", exc)
json.dump(out, open(os.path.join(config.ARTIFACTS_DIR, "genesis", "learning_curve.json"), "w"),
          indent=1, default=float)
