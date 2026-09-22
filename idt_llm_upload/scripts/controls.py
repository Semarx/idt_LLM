import sys, numpy as np, pandas as pd, time
sys.path.insert(0,'/home/claude/trackB')
from core import *
NREP=100; NPERM=500; COS_MAX=0.50; BLOCK=0.15
df=pd.read_csv('/mnt/user-data/uploads/Data/turns_all_ts.csv')
g=df.groupby('conv_id').size()
TOP=list(g[g>=25].sort_values(ascending=False).index[:3])
print("cells from:",TOP,flush=True)
def kfor(n): return 2 if n<150 else (3 if n<400 else 4)

def q95(S,A,Sp,k,rng):
    v=[eps_from(S,np.random.default_rng(rng.integers(1<<30)).permutation(A),Sp,k)[0] for _ in range(NPERM)]
    v=np.array(v,float); v=v[np.isfinite(v)]; return np.quantile(v,0.95)

rows=[]
for cid in TOP:
    d=df[df.conv_id==cid].sort_values('Turn')
    P=d['Prompt'].fillna('').astype(str).tolist(); R=d['Response'].fillna('').astype(str).tolist()
    n=len(d); k=kfor(n)
    Ep=embed(P); Er=embed(R)
    cp=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Ep)
    cr=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Er)
    Lp=np.array([len(x) for x in P]); Lr=np.array([len(x) for x in R])
    for side in ('model','user'):
        if side=='model':
            S,A,Sp=cp[:-1],cr[:-1],cp[1:]; EA=Er[:-1]; LA=Lr[:-1]
        else:
            S,A,Sp=cr[:-1],cp[1:],cr[1:];  EA=Ep[1:];  LA=Lp[1:]
        m=len(S); blk=max(3,int(BLOCK*m))
        base=eps_from(S,A,Sp,k)[0]
        ref=q95(S,A,Sp,k,np.random.default_rng(seed_of(cid,side,'ref')))
        rows.append(dict(record=cid,side=side,k=k,N=m,cell='C0_observed',
                         eps_median=base,eps_lo=base,eps_hi=base,clears=bool(base>ref),ref=ref,
                         rep_clear_rate=np.nan))
        cos=EA@EA.T
        for cell in ('C1_compat_nonadj','C2_lengthA','C3_dissimB'):
            rng=np.random.default_rng(seed_of(cid,side,cell))
            vals=[];clears=0
            for rep in range(NREP):
                A2=A.copy()
                for t in range(m):
                    if cell=='C1_compat_nonadj':
                        cand=np.where((S==S[t])&(np.abs(np.arange(m)-t)>=2)&(np.abs(np.arange(m)-t)<=blk))[0]
                        if len(cand)==0:
                            cand=np.where((S==S[t])&(np.abs(np.arange(m)-t)>=2))[0]
                    elif cell=='C2_lengthA':
                        ok=np.abs(np.arange(m)-t)>=2
                        rel=np.abs(LA-LA[t])/np.maximum(LA[t],1)
                        cand=np.where(ok&(rel<=0.05))[0]
                        if len(cand)==0: cand=np.where(ok)[0][np.argsort(rel[ok])[:10]]
                    else:
                        ok=np.abs(np.arange(m)-t)>=2
                        for band in (0.10,0.20,0.50,1.00,1e9):
                            rel=np.abs(LA-LA[t])/np.maximum(LA[t],1)
                            cand=np.where(ok&(rel<=band)&(cos[t]<COS_MAX))[0]
                            if len(cand)>0: break
                        if len(cand)==0:
                            cand=np.where(ok)[0][np.argsort(cos[t][ok])[:10]]
                    if len(cand): A2[t]=A[rng.choice(cand)]
                e=eps_from(S,A2,Sp,k)[0]
                vals.append(e); clears+= int(np.isfinite(e) and e>ref)
            v=np.array(vals,float); v=v[np.isfinite(v)]
            rows.append(dict(record=cid,side=side,k=k,N=m,cell=cell,
                eps_median=float(np.median(v)),eps_lo=float(np.percentile(v,2.5)),
                eps_hi=float(np.percentile(v,97.5)),clears=np.nan,ref=ref,
                rep_clear_rate=clears/NREP))
            print(f"  {cid[:16]} {side:5} {cell:18} med={np.median(v):.4f} obs={base:.4f} clr={clears}/100 {time.time():.0f}",flush=True)
pd.DataFrame(rows).to_csv('/home/claude/trackB/c1_controls.csv',index=False)
print("saved")
