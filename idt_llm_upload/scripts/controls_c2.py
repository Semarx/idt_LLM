import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/trackB')
from core import *
NREP=100; NPERM=500; COS_MAX=0.50; BLOCK=0.15
d=pd.read_csv('/mnt/user-data/uploads/advancement/omega_advancement_test_v1_turns.csv')
d=d[d.turn>=2].reset_index(drop=True)
runs=sorted(d.run_id.unique())
P=d.prompt.fillna('').astype(str).tolist(); R=d.response.fillna('').astype(str).tolist()
Ep=embed(P); Er=embed(R); Lp=np.array([len(x) for x in P]); Lr=np.array([len(x) for x in R])
rows=[]
for k in (3,2):
    cp=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Ep)
    cr=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Er)
    for side in ('model','user'):
        S=[];A=[];Sp=[];EAi=[];LAi=[];RUN=[]
        for r in runs:
            idx=d.index[d.run_id==r].tolist()
            a=cp[idx]; b=cr[idx]
            if side=='model': S+=list(a[:-1]);A+=list(b[:-1]);Sp+=list(a[1:]);EAi+=idx[:-1];RUN+=[r]*(len(idx)-1); LAi+=list(Lr[idx[:-1]])
            else:             S+=list(b[:-1]);A+=list(a[1:]); Sp+=list(b[1:]);EAi+=idx[1:]; RUN+=[r]*(len(idx)-1); LAi+=list(Lp[idx[1:]])
        S=np.array(S);A=np.array(A);Sp=np.array(Sp);RUN=np.array(RUN);LA=np.array(LAi)
        EA=(Er if side=='model' else Ep)[np.array(EAi)]
        m=len(S); pos=np.arange(m); blk=max(3,int(BLOCK*m/len(runs)))
        base=eps_from(S,A,Sp,k)[0]
        rng0=np.random.default_rng(seed_of('C2',k,side,'ref'))
        ref=np.quantile([eps_from(S,rng0.permutation(A),Sp,k)[0] for _ in range(NPERM)],0.95)
        rows.append(dict(record=f'C2_pooled_k{k}',side=side,k=k,N=m,cell='C0_observed',
                         eps_median=base,eps_lo=base,eps_hi=base,ref=ref,rep_clear_rate=np.nan,
                         cos_median=np.nan))
        cos=EA@EA.T
        for cell in ('C1_compat_nonadj','C2_lengthA','C3_dissimB'):
            rng=np.random.default_rng(seed_of('C2',k,side,cell)); vals=[];clears=0;coss=[]
            for rep in range(NREP):
                A2=A.copy()
                for t in range(m):
                    same=(RUN==RUN[t])&(np.abs(pos-t)>=2)     # never cross a run boundary
                    if cell=='C1_compat_nonadj':
                        cand=np.where(same&(S==S[t])&(np.abs(pos-t)<=blk))[0]
                        if len(cand)==0: cand=np.where(same&(S==S[t]))[0]
                    elif cell=='C2_lengthA':
                        rel=np.abs(LA-LA[t])/np.maximum(LA[t],1)
                        cand=np.where(same&(rel<=0.05))[0]
                        if len(cand)==0:
                            f=np.where(same)[0]; cand=f[np.argsort(rel[f])[:10]] if len(f) else np.array([],int)
                    else:
                        cand=np.array([],int)
                        for band in (0.10,0.20,0.50,1.00,1e9):
                            rel=np.abs(LA-LA[t])/np.maximum(LA[t],1)
                            cand=np.where(same&(rel<=band)&(cos[t]<COS_MAX))[0]
                            if len(cand)>0: break
                        if len(cand)==0:
                            f=np.where(same)[0]; cand=f[np.argsort(cos[t][f])[:10]] if len(f) else np.array([],int)
                    if len(cand):
                        j=rng.choice(cand); A2[t]=A[j]
                        if rep==0: coss.append(cos[t][j])
                e=eps_from(S,A2,Sp,k)[0]; vals.append(e); clears+=int(np.isfinite(e) and e>ref)
            v=np.array(vals,float); v=v[np.isfinite(v)]
            rows.append(dict(record=f'C2_pooled_k{k}',side=side,k=k,N=m,cell=cell,
                eps_median=float(np.median(v)),eps_lo=float(np.percentile(v,2.5)),
                eps_hi=float(np.percentile(v,97.5)),ref=ref,rep_clear_rate=clears/NREP,
                cos_median=float(np.median(coss)) if coss else np.nan))
            print(f"  k={k} {side:5} {cell:18} med={np.median(v):.4f} obs={base:.4f} red={100*(1-np.median(v)/base):.1f}% clr={clears}/100",flush=True)
pd.DataFrame(rows).to_csv('/home/claude/trackB/c2_controls.csv',index=False); print("saved")
