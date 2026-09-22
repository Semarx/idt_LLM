import sys, numpy as np, pandas as pd, time
sys.path.insert(0,'/home/claude/trackB')
from core import *

NPERM=2000; KAPPA=10; MINN=5; COVER=0.70
df=pd.read_csv('/mnt/user-data/uploads/Data/turns_all_ts.csv')
g=df.groupby('conv_id').size(); convs=sorted(g[g>=25].index)
def kfor(n): return 2 if n<150 else (3 if n<400 else 4)

rows=[]; t0=time.time()
for ci,cid in enumerate(convs):
    d=df[df.conv_id==cid].sort_values('Turn')
    P=d['Prompt'].fillna('').tolist(); R=d['Response'].fillna('').tolist()
    n=len(d); k=kfor(n)
    Ep=embed(P); Er=embed(R)
    cp=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Ep)
    cr=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Er)
    for side in ('model','user'):
        if side=='model':   # S=p_t, A=r_t, S'=p_{t+1}
            S,A,Sp = cp[:-1], cr[:-1], cp[1:]
            ES = Ep[:-1]
        else:               # S=r_t, A=p_{t+1}, S'=r_{t+1}
            S,A,Sp = cr[:-1], cp[1:], cr[1:]
            ES = Er[:-1]
        eps,I,room,resid=eps_from(S,A,Sp,k)
        # screens
        sym_ok = (len(np.unique(S))==k) and (len(np.unique(A))==k) and (len(np.unique(Sp))==k)
        occ=np.bincount((S*k+A)*k+Sp,minlength=k**3); cells=int((occ>0).sum())
        npc = len(S)/cells if cells else 0
        cell_ok = npc>=3
        room_ok = room>0.05
        sat = (eps==eps) and eps>=0.99
        rec=dict(corpus='C1',record=cid,side=side,N=len(S),k=k,
                 eps=eps,I=I,room=room,resid=resid,cells=cells,npc=round(npc,2),
                 symbol_ok=sym_ok,cell_ok=cell_ok,room_ok=room_ok,saturated=bool(sat))
        if not (sym_ok and cell_ok and room_ok) or not (eps==eps):
            rec.update(screen='FAIL'); rows.append(rec); continue
        rec['screen']='PASS'
        sets=knn_sets(ES,KAPPA) if len(S)<=3000 else None
        for name in ('global','strat','local','circ'):
            rng=np.random.default_rng(seed_of(cid,side,name))
            vals=[]; cover=1.0
            if name=='strat':
                A0,mask=null_strat(A,S,rng,MINN); cover=mask.mean()
                if cover<COVER:
                    rec[f'q95_{name}']=np.nan; rec[f'clear_{name}']='NOT_ADMISSIBLE'
                    rec['strat_cover']=round(cover,3); continue
                Sm,Am,Spm=S[mask],A[mask],Sp[mask]
                base=eps_from(Sm,Am,Spm,k)[0]
                for _ in range(NPERM):
                    Ax,_m=null_strat(Am,Sm,rng,MINN)
                    vals.append(eps_from(Sm,Ax,Spm,k)[0])
                obs=base; rec['strat_cover']=round(cover,3); rec['eps_strat_sub']=base
            else:
                obs=eps
                for _ in range(NPERM):
                    if name=='global': Ax=null_global(A,rng)
                    elif name=='circ': Ax=null_circ(A,rng)
                    else:
                        if sets is None: Ax=null_global(A,rng)
                        else: Ax=null_local(A,sets,rng)
                    vals.append(eps_from(S,Ax,Sp,k)[0])
            v=np.array(vals,dtype=float); v=v[~np.isnan(v)]
            q=np.quantile(v,0.95); pct=(v<obs).mean()*100
            rec[f'q95_{name}']=round(float(q),5); rec[f'pct_{name}']=round(float(pct),2)
            rec[f'clear_{name}']=bool(obs>q)
        rows.append(rec)
    print(f"[{ci+1}/{len(convs)}] {cid} n={n} k={k}  {time.time()-t0:.0f}s",flush=True)
out=pd.DataFrame(rows)
out.to_csv('/home/claude/trackB/c1_nulls.csv',index=False)
print(out[['record','side','N','k','eps','screen']+[c for c in out.columns if c.startswith('clear_')]].to_string())
