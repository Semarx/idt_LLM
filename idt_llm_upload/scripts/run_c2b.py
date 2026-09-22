import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude/trackB')
from core import *
NPERM=2000; KAPPA=10; MINN=5; COVER=0.70

def strat_within(A,S,G,rng,minn=MINN):
    """Permute A within (run, S-symbol) cells only — never across runs."""
    A2=A.copy(); mask=np.zeros(len(A),bool)
    for g in np.unique(G):
        for s in np.unique(S[G==g]):
            idx=np.where((G==g)&(S==s))[0]
            if len(idx)>=minn:
                A2[idx]=rng.permutation(A[idx]); mask[idx]=True
    return A2,mask

def local_within(A,sets,G,rng):
    n=len(A); A2=np.empty(n,dtype=A.dtype); used=np.zeros(n,bool)
    for i in rng.permutation(n):
        c=np.array([j for j in sets[i] if G[j]==G[i]],dtype=int)
        free=c[~used[c]] if len(c) else c
        j=rng.choice(free) if len(free) else (rng.choice(c) if len(c) else i)
        A2[i]=A[j]; used[j]=True
    return A2

def circ_within(A,G,rng,B=5):
    A2=A.copy()
    for g in np.unique(G):
        idx=np.where(G==g)[0]; m=len(idx)
        if m>2*B: A2[idx]=np.roll(A[idx],rng.integers(B,m-B))
        else:     A2[idx]=rng.permutation(A[idx])
    return A2

d=pd.read_csv('/mnt/user-data/uploads/advancement/omega_advancement_test_v1_turns.csv')
d=d[d.turn>=2].reset_index(drop=True)
runs=sorted(d.run_id.unique())
P=d.prompt.fillna('').astype(str).tolist(); R=d.response.fillna('').astype(str).tolist()
Ep=embed(P); Er=embed(R)
rows=[]
for k in (3,2):
    cp=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Ep)
    cr=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(Er)
    for side in ('model','user'):
        S=[];A=[];Sp=[];ES=[];G=[]
        for r in runs:
            idx=d.index[d.run_id==r].tolist()
            a=cp[idx]; b=cr[idx]
            if side=='model': S+=list(a[:-1]);A+=list(b[:-1]);Sp+=list(a[1:]);ES+=[Ep[i] for i in idx[:-1]]
            else:             S+=list(b[:-1]);A+=list(a[1:]); Sp+=list(b[1:]);ES+=[Er[i] for i in idx[:-1]]
            G+=[r]*(len(idx)-1)
        S=np.array(S);A=np.array(A);Sp=np.array(Sp);G=np.array(G);ES=np.vstack(ES)
        eps=eps_from(S,A,Sp,k)[0]
        sets=knn_sets(ES,KAPPA)
        rec=dict(corpus='C2',record=f'pooled_k{k}',side=side,N=len(S),k=k,eps=eps,
                 triples_per_run=int((G==runs[0]).sum()))
        for name in ('strat_within','local_within','circ_within','strat_pooled'):
            rng=np.random.default_rng(seed_of('C2b',k,side,name)); vals=[]
            if name=='strat_within':
                A0,mask=strat_within(A,S,G,rng); cov=mask.mean(); rec['cover_within']=round(cov,3)
                if cov<COVER: rec['clear_'+name]='NOT_ADMISSIBLE'; continue
                Sm,Am,Spm,Gm=S[mask],A[mask],Sp[mask],G[mask]
                obs=eps_from(Sm,Am,Spm,k)[0]; rec['eps_sub_within']=round(obs,4)
                for _ in range(NPERM): vals.append(eps_from(Sm,strat_within(Am,Sm,Gm,rng)[0],Spm,k)[0])
            elif name=='strat_pooled':
                A0,mask=null_strat(A,S,rng,MINN); cov=mask.mean(); rec['cover_pooled']=round(cov,3)
                Sm,Am,Spm=S[mask],A[mask],Sp[mask]; obs=eps_from(Sm,Am,Spm,k)[0]
                for _ in range(NPERM): vals.append(eps_from(Sm,null_strat(Am,Sm,rng,MINN)[0],Spm,k)[0])
            else:
                obs=eps
                for _ in range(NPERM):
                    Ax = local_within(A,sets,G,rng) if name=='local_within' else circ_within(A,G,rng)
                    vals.append(eps_from(S,Ax,Sp,k)[0])
            v=np.array(vals,float); v=v[np.isfinite(v)]
            q=np.quantile(v,0.95)
            rec[f'q95_{name}']=round(float(q),5)
            rec[f'pct_{name}']=round(float((v<obs).mean()*100),2)
            rec[f'clear_{name}']=bool(obs>q)
        rows.append(rec)
        print(f"  k={k} {side:5} eps={eps:.4f}  within={rec.get('clear_strat_within')} "
              f"local={rec.get('clear_local_within')} circ={rec.get('clear_circ_within')} "
              f"pooled={rec.get('clear_strat_pooled')}  cover_within={rec.get('cover_within')}",flush=True)
pd.DataFrame(rows).to_csv('/home/claude/trackB/c2_nulls_runpreserving.csv',index=False)
print("saved")
