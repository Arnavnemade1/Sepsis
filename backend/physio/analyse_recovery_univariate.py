"""Model-free test of the recovery features against circulatory decompensation.

    PYTHONPATH=. python -m backend.physio.analyse_recovery_univariate

The gradient-boosted analysis in ``analyse_recovery_outcome.py`` returned AUROCs
*below* chance (0.376 at a 15-minute lead). Consistent sub-chance performance under
grouped cross-validation is the signature of overfitting, not of an inverted effect,
and the cause was obvious once looked at: 28 features fitted on ~121 rows across 29
patients, roughly 90 training rows per fold. That model would bury a real effect as
readily as a fake one.

So this is the fair test at this sample size. Univariate AUROC per feature involves no
fitting at all and therefore cannot overfit; the regularised logistic regression with
leave-one-patient-out is the multivariate companion. Confidence intervals are
patient-bootstrapped throughout, and the count of features clearing significance is
reported against the number expected by chance, because 28 features x 3 lead times is
84 tests and some of them will clear a 5% bar for nothing.
"""
import numpy as np
from backend.ml.eval import metrics
from backend.physio import recovery

d=np.load('artifacts/mimic3_open/windows.npz', allow_pickle=True)
X=np.nan_to_num(np.asarray(d['X'],float),nan=0.0,posinf=0.0,neginf=0.0)
names=[str(n) for n in d['names']]; meta=d['meta']
pid=np.array([str(m[0]) for m in meta]); lead=np.array([float(m[1]) for m in meta])
y=np.array([int(m[2]) for m in meta])
both=np.array(sorted([p for p in set(pid) if len(set(y[pid==p]))==2]))
dual=np.isin(pid,both)
REC=set(recovery.RECOVERY_FEATURE_NAMES)

def boot_auc(yy,ss,gg,nb=2000,seed=0):
    pts=np.array(sorted(set(gg.tolist()))); by={g:np.flatnonzero(gg==g) for g in pts}
    rng=np.random.default_rng(seed); out=np.empty(nb)
    for b in range(nb):
        dr=rng.choice(len(pts),len(pts),replace=True)
        rows=np.concatenate([by[pts[i]] for i in dr]); v=yy[rows]
        out[b]=0.5 if v.min()==v.max() else metrics.auroc(v,ss[rows])
    return metrics.auroc(yy,ss), np.percentile(out,2.5), np.percentile(out,97.5)

for L in (15.0,30.0,60.0):
    m=dual&(lead==L); yy=y[m]; gg=pid[m]
    print(f"\n=== lead {L:.0f} min — univariate, NO model ({int((yy==1).sum())}ev/"
          f"{int((yy==0).sum())}ctl, {len(set(gg))} patients) ===")
    rows=[]
    for i,n in enumerate(names):
        a,lo,hi=boot_auc(yy,X[m,i],gg)
        rows.append((abs(a-0.5),a,lo,hi,n))
    rows.sort(reverse=True)
    for s,a,lo,hi,n in rows[:8]:
        tag="REC" if n in REC else "hrv"
        sig="*" if (lo>0.5 or hi<0.5) else " "
        print(f"   {sig}{tag} {n:<22} AUROC {a:.3f} CI[{lo:.3f},{hi:.3f}]")
    nsig=sum(1 for s,a,lo,hi,n in rows if lo>0.5 or hi<0.5)
    print(f"   -> {nsig}/{len(rows)} features with CI excluding 0.5 "
          f"(expect ~{0.05*len(rows):.1f} by chance)")

print("\n=== regularised logistic regression (leave-one-patient-out) ===")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
for L in (15.0,30.0,60.0):
    m=dual&(lead==L); Xm,yy,gg=X[m],y[m],pid[m]
    for tag,idx in (("recovery",[i for i,n in enumerate(names) if n in REC]),
                    ("hrv",[i for i,n in enumerate(names) if n not in REC]),
                    ("all",list(range(len(names))))):
        p=np.full(len(yy),np.nan)
        for tr,te in LeaveOneGroupOut().split(Xm,yy,gg):
            if len(np.unique(yy[tr]))<2: continue
            mdl=make_pipeline(StandardScaler(),
                LogisticRegression(C=0.05,max_iter=3000,class_weight='balanced'))
            mdl.fit(Xm[tr][:,idx],yy[tr]); p[te]=mdl.predict_proba(Xm[te][:,idx])[:,1]
        ok=~np.isnan(p)
        if ok.sum()>10 and len(np.unique(yy[ok]))>1:
            a,lo,hi=boot_auc(yy[ok],p[ok],gg[ok])
            print(f"  lead {L:.0f} {tag:<9} AUROC {a:.3f} CI[{lo:.3f},{hi:.3f}]")
