import sys, numpy as np, pandas as pd, time
sys.path.insert(0,'/home/claude/trackB')
from core import *
NPERM=2000; KAPPA=10; MINN=5; COVER=0.70; K=2
pairs=pd.read_csv('/home/claude/trackB/c3_pairs.csv')
games=sorted(pairs.game.unique()); print("games:",len(games),flush=True)
rows=[]; t0=time.time()
for gi,g in enumerate(games):
    d=pairs[pairs.game==g].sort_values('turn')
    MA=d.msgA.fillna('').astype(str).tolist(); MB=d.msgB.fillna('').astype(str).tolist()
    EA=embed(MA); EB=embed(MB)
    ca=KMeans(n_clusters=K,random_state=0,n_init=10).fit_predict(EA)
    cb=KMeans(n_clusters=K,random_state=0,n_init=10).fit_predict(EB)
    for side in ('model','user'):
        if side=='model':  S,A,Sp,ES = ca[:-1],cb[:-1],ca[1:],EA[:-1]   # action = second speaker
        else:              S,A,Sp,ES = cb[:-1],ca[1:], cb[1:],EB[:-1]   # action = first speaker
        eps,I,room,resid=eps_from(S,A,Sp,K)
        occ=np.bincount((S*K+A)*K+Sp,minlength=K**3); cells=int((occ>0).sum()); npc=len(S)/cells if cells else 0
        rec=dict(corpus='C3',record=g,side=side,N=len(S),k=K,eps=eps,I=I,room=room,resid=resid,
                 cells=cells,npc=round(npc,2),
                 symbol_ok=len(np.unique(S))==K and len(np.unique(A))==K and len(np.unique(Sp))==K,
                 cell_ok=npc>=3, room_ok=room>0.05, saturated=bool(eps==eps and eps>=0.99))
        rec['screen']='PASS' if (rec['symbol_ok'] and rec['cell_ok'] and rec['room_ok'] and eps==eps) else 'FAIL'
        if rec['screen']=='PASS':
            sets=knn_sets(ES,KAPPA)
            for name in ('global','strat','local','circ'):
                rng=np.random.default_rng(seed_of(g,side,name)); vals=[]
                if name=='strat':
                    A0,mask=null_strat(A,S,rng,MINN); cov=mask.mean(); rec['strat_cover']=round(cov,3)
                    if cov<COVER: rec['clear_strat']='NOT_ADMISSIBLE'; continue
                    Sm,Am,Spm=S[mask],A[mask],Sp[mask]; obs=eps_from(Sm,Am,Spm,K)[0]
                    for _ in range(NPERM): vals.append(eps_from(Sm,null_strat(Am,Sm,rng,MINN)[0],Spm,K)[0])
                else:
                    obs=eps
                    for _ in range(NPERM):
                        Ax = null_global(A,rng) if name=='global' else (null_circ(A,rng) if name=='circ' else null_local(A,sets,rng))
                        vals.append(eps_from(S,Ax,Sp,K)[0])
                v=np.array(vals,float); v=v[np.isfinite(v)]
                if len(v)<100: rec[f'clear_{name}']='NOT_ADMISSIBLE'; continue
                rec[f'q95_{name}']=round(float(np.quantile(v,0.95)),5)
                rec[f'clear_{name}']=bool(obs>np.quantile(v,0.95))
        rows.append(rec)
    if (gi+1)%50==0:
        print(f"[{gi+1}/{len(games)}] {time.time()-t0:.0f}s",flush=True)
        pd.DataFrame(rows).to_csv('/home/claude/trackB/c3_nulls.csv',index=False)
pd.DataFrame(rows).to_csv('/home/claude/trackB/c3_nulls.csv',index=False)
print("done",time.time()-t0)
