"""Calibration: type-I error and power for each null design.
DGP: S_t Markov chain (serial correlation rho), A_t ~ p(A|S) with dependence strength beta,
     S'_t ~ p(S'|S) [null] or p(S'|S,A) [alternative, strength gamma].
Factors: dependence, serial correlation, nonstationarity, occupancy imbalance, (N,k).
"""
import sys, numpy as np, pandas as pd, itertools, time
sys.path.insert(0,'/home/claude/trackB')
from core import eps_from, null_global, null_strat, null_local, null_circ, knn_sets, seed_of

NPERM=200; ALPHA=0.05

def softmax(x):
    e=np.exp(x-x.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)

def simulate(N,k,beta,rho,gamma,nonstat,imbal,rng):
    # S: Markov chain with self-transition prob rho, stationary skew imbal
    base=np.ones(k); base[0]=imbal; base=base/base.sum()
    fresh=rng.choice(k,size=N,p=base); keep=rng.random(N)<rho; keep[0]=False
    idx=np.where(~keep,np.arange(N),0); np.maximum.accumulate(idx,out=idx)
    S=fresh[idx]
    # A | S  with dependence beta
    WA=rng.normal(0,1,(k,k))*beta
    if nonstat: WA2=rng.normal(0,1,(k,k))*beta
    PA=softmax(WA)[S]
    if nonstat:
        PA2=softmax(WA2)[S]; half=np.arange(N)>N//2; PA=np.where(half[:,None],PA2,PA)
    u=rng.random((N,1)); A=(PA.cumsum(1)<u).sum(1).clip(0,k-1)
    # S' | S (+A if gamma>0)
    WS=rng.normal(0,1,(k,k)); WSA=rng.normal(0,1,(k,k))*gamma
    lo=WS[S]+(WSA[A] if gamma>0 else 0)
    PS=softmax(lo); u2=rng.random((N,1)); Sp=(PS.cumsum(1)<u2).sum(1).clip(0,k-1)
    # pseudo-embedding of S for the local null: symbol + noise
    E=np.eye(k)[S]+rng.normal(0,0.15,(N,k)); E/=np.linalg.norm(E,axis=1,keepdims=True)
    return S,A,Sp,E

def one(N,k,beta,rho,gamma,nonstat,imbal,seed):
    rng=np.random.default_rng(seed)
    S,A,Sp,E=simulate(N,k,beta,rho,gamma,nonstat,imbal,rng)
    obs=eps_from(S,A,Sp,k)[0]
    if not np.isfinite(obs): return None
    res={}
    sets=knn_sets(E,10)
    for name in ('global','strat','local','circ'):
        if name=='strat':
            A0,mask=null_strat(A,S,rng,5)
            if mask.mean()<0.70: res[name]=np.nan; continue
            Sm,Am,Spm=S[mask],A[mask],Sp[mask]
            o=eps_from(Sm,Am,Spm,k)[0]
            v=[eps_from(Sm,null_strat(Am,Sm,rng,5)[0],Spm,k)[0] for _ in range(NPERM)]
        else:
            o=obs
            if name=='global': v=[eps_from(S,null_global(A,rng),Sp,k)[0] for _ in range(NPERM)]
            elif name=='circ': v=[eps_from(S,null_circ(A,rng),Sp,k)[0] for _ in range(NPERM)]
            else: v=[eps_from(S,null_local(A,sets,rng),Sp,k)[0] for _ in range(NPERM)]
        v=np.array(v,float); v=v[np.isfinite(v)]
        res[name]= bool(o>np.quantile(v,0.95)) if len(v)>50 else np.nan
    return res

def wilson(x,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=x/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h),min(1,c+h))

if __name__=="__main__":
    grid=[]
    for N,k in [(40,2),(100,2),(250,3),(800,4)]:
        for beta in (0.0,2.0):
            for rho in (0.0,0.5):
                for nonstat in (False,True):
                    for imbal in (1.0,4.0):
                        for gamma in (0.0,0.8):
                            grid.append((N,k,beta,rho,gamma,nonstat,imbal))
    print("cells:",len(grid),flush=True)
    rows=[];t0=time.time()
    nsim=int(sys.argv[1]) if len(sys.argv)>1 else 200
    for gi,(N,k,beta,rho,gamma,nonstat,imbal) in enumerate(grid):
        acc={n:[0,0] for n in ('global','strat','local','circ')}
        for s in range(nsim):
            r=one(N,k,beta,rho,gamma,nonstat,imbal,seed_of(gi,s))
            if r is None: continue
            for n,v in r.items():
                if isinstance(v,bool): acc[n][0]+=int(v); acc[n][1]+=1
        row=dict(N=N,k=k,beta=beta,rho=rho,gamma=gamma,nonstat=nonstat,imbal=imbal,
                 kind='type_I' if gamma==0 else 'power')
        for n,(x,m) in acc.items():
            rate=x/m if m else np.nan; lo,hi=wilson(x,m)
            row[f'{n}_rate']=round(rate,4) if m else np.nan
            row[f'{n}_lo']=round(lo,4) if m else np.nan
            row[f'{n}_hi']=round(hi,4) if m else np.nan
            row[f'{n}_n']=m
        rows.append(row)
        print(f"[{gi+1}/{len(grid)}] N={N} k={k} g={gamma} {time.time()-t0:.0f}s "
              +" ".join(f"{n}={row[f'{n}_rate']}" for n in acc),flush=True)
        pd.DataFrame(rows).to_csv('/home/claude/trackB/calibration.csv',index=False)
    print("done")
