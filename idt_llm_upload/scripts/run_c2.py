import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/trackB')
from core import *
NPERM=2000; KAPPA=10; MINN=5; COVER=0.70
d=pd.read_csv('/mnt/user-data/uploads/advancement/omega_advancement_test_v1_turns.csv')
d=d[d.turn>=2].copy()                      # turn 1 excluded, per the omega definition
runs=sorted(d.run_id.unique()); print("runs:",len(runs),"turns/run:",d.groupby('run_id').size().unique())
P=d.prompt.fillna('').astype(str).tolist(); R=d.response.fillna('').astype(str).tolist()
Ep=embed(P); Er=embed(R)                   # pooled embedding, declared departure
rows=[]
for k in (3,2):
    cp=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Ep)   # one shared alphabet
    cr=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Er)
    d['_cp']=cp; d['_cr']=cr
    for side in ('model','user'):
        S=[];A=[];Sp=[];ES=[]
        for r in runs:                      # triples within run only
            m=d[d.run_id==r]
            a=m['_cp'].values; b=m['_cr'].values
            ea=Ep[m.index.map(lambda i: d.index.get_loc(i))] if False else None
            idx=[d.index.get_loc(i) for i in m.index]
            if side=='model': S+=list(a[:-1]); A+=list(b[:-1]); Sp+=list(a[1:]); ES+=[Ep[i] for i in idx[:-1]]
            else:             S+=list(b[:-1]); A+=list(a[1:]);  Sp+=list(b[1:]); ES+=[Er[i] for i in idx[:-1]]
        S=np.array(S);A=np.array(A);Sp=np.array(Sp);ES=np.vstack(ES)
        eps,I,room,resid=eps_from(S,A,Sp,k)
        occ=np.bincount((S*k+A)*k+Sp,minlength=k**3); cells=int((occ>0).sum()); npc=len(S)/cells
        rec=dict(corpus='C2',record=f'pooled_k{k}',side=side,N=len(S),k=k,eps=eps,I=I,room=room,resid=resid,
                 cells=cells,npc=round(npc,2),
                 symbol_ok=len(np.unique(S))==k and len(np.unique(A))==k and len(np.unique(Sp))==k,
                 cell_ok=npc>=3, room_ok=room>0.05, saturated=bool(eps==eps and eps>=0.99))
        rec['screen']='PASS' if (rec['symbol_ok'] and rec['cell_ok'] and rec['room_ok'] and eps==eps) else 'FAIL'
        if rec['screen']=='PASS':
            sets=knn_sets(ES,KAPPA)
            for name in ('global','strat','local','circ'):
                rng=np.random.default_rng(seed_of('C2',k,side,name)); vals=[]
                if name=='strat':
                    A0,mask=null_strat(A,S,rng,MINN); cov=mask.mean(); rec['strat_cover']=round(cov,3)
                    if cov<COVER: rec['clear_strat']='NOT_ADMISSIBLE'; continue
                    Sm,Am,Spm=S[mask],A[mask],Sp[mask]; obs=eps_from(Sm,Am,Spm,k)[0]
                    for _ in range(NPERM): vals.append(eps_from(Sm,null_strat(Am,Sm,rng,MINN)[0],Spm,k)[0])
                else:
                    obs=eps
                    for _ in range(NPERM):
                        Ax = null_global(A,rng) if name=='global' else (null_circ(A,rng) if name=='circ' else null_local(A,sets,rng))
                        vals.append(eps_from(S,Ax,Sp,k)[0])
                v=np.array(vals,float); v=v[np.isfinite(v)]
                rec[f'q95_{name}']=round(float(np.quantile(v,0.95)),5)
                rec[f'pct_{name}']=round(float((v<obs).mean()*100),2)
                rec[f'clear_{name}']=bool(obs>np.quantile(v,0.95))
        rows.append(rec); print(f"  k={k} {side:5} eps={eps:.4f} N={len(S)} screen={rec['screen']}",flush=True)
out=pd.DataFrame(rows); out.to_csv('/home/claude/trackB/c2_nulls.csv',index=False)
print(out[['record','side','N','k','eps','screen']+[c for c in out.columns if c.startswith('clear_')]].to_string(index=False))
