import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/trackB')
from core import *
COS_MAX=0.50; BLOCK=0.15
df=pd.read_csv('/mnt/user-data/uploads/Data/turns_all_ts.csv')
g=df.groupby('conv_id').size(); TOP=list(g[g>=25].sort_values(ascending=False).index[:3])
rows=[]
for cid in TOP:
    d=df[df.conv_id==cid].sort_values('Turn')
    P=d['Prompt'].fillna('').astype(str).tolist(); R=d['Response'].fillna('').astype(str).tolist()
    n=len(d); k=4
    Ep=embed(P); Er=embed(R)
    cp=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Ep)
    cr=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Er)
    Lp=np.array([len(x) for x in P]); Lr=np.array([len(x) for x in R])
    for side in ('model','user'):
        if side=='model': S,A=cp[:-1],cr[:-1]; EA=Er[:-1]; LA=Lr[:-1]
        else:             S,A=cr[:-1],cp[1:];  EA=Ep[1:];  LA=Lp[1:]
        m=len(S); pos=np.arange(m); blk=max(3,int(BLOCK*m)); cos=EA@EA.T
        for cell in ('C1_compat_nonadj','C2_lengthA','C3_dissimB'):
            rng=np.random.default_rng(seed_of(cid,side,cell)); cs=[]
            for t in range(m):
                ok=np.abs(pos-t)>=2
                if cell=='C1_compat_nonadj':
                    cand=np.where(ok&(S==S[t])&(np.abs(pos-t)<=blk))[0]
                    if len(cand)==0: cand=np.where(ok&(S==S[t]))[0]
                elif cell=='C2_lengthA':
                    rel=np.abs(LA-LA[t])/np.maximum(LA[t],1)
                    cand=np.where(ok&(rel<=0.05))[0]
                    if len(cand)==0: f=np.where(ok)[0]; cand=f[np.argsort(rel[f])[:10]]
                else:
                    cand=np.array([],int)
                    for band in (0.10,0.20,0.50,1.00,1e9):
                        rel=np.abs(LA-LA[t])/np.maximum(LA[t],1)
                        cand=np.where(ok&(rel<=band)&(cos[t]<COS_MAX))[0]
                        if len(cand)>0: break
                    if len(cand)==0: f=np.where(ok)[0]; cand=f[np.argsort(cos[t][f])[:10]]
                if len(cand): cs.append(cos[t][rng.choice(cand)])
            rows.append(dict(record=cid,side=side,cell=cell,cos_median=float(np.median(cs))))
            print(cid[:16],side,cell,round(np.median(cs),3),flush=True)
pd.DataFrame(rows).to_csv('/home/claude/trackB/c1_cos.csv',index=False)
