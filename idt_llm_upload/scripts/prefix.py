import sys, numpy as np, pandas as pd, time
sys.path.insert(0,'/home/claude/trackB')
from core import *
NPERM=1000; MINN=5; COVER=0.70
PREF_C1=[25,35,45,55,65,90,120,200,400]; PREF_C3=[25,35]
def kfor(n): return 2 if n<150 else (3 if n<400 else 4)
def clears(S,A,Sp,k,seed):
    if len(np.unique(S))<k or len(np.unique(A))<k or len(np.unique(Sp))<k: return None
    occ=np.bincount((S*k+A)*k+Sp,minlength=k**3); cells=int((occ>0).sum())
    if cells==0 or len(S)/cells<3: return None
    e,I,room,res=eps_from(S,A,Sp,k)
    if not np.isfinite(e) or room<=0.05: return None
    rng=np.random.default_rng(seed)
    A0,mask=null_strat(A,S,rng,MINN)
    if mask.mean()<COVER: return None
    Sm,Am,Spm=S[mask],A[mask],Sp[mask]; obs=eps_from(Sm,Am,Spm,k)[0]
    v=np.array([eps_from(Sm,null_strat(Am,Sm,rng,MINN)[0],Spm,k)[0] for _ in range(NPERM)],float)
    v=v[np.isfinite(v)]
    if len(v)<100: return None
    return bool(obs>np.quantile(v,0.95))
rows=[];t0=time.time()
# ---- C1 ----
df=pd.read_csv('/mnt/user-data/uploads/Data/turns_all_ts.csv')
g=df.groupby('conv_id').size(); convs=sorted(g[g>=25].index)
for ci,cid in enumerate(convs):
    d=df[df.conv_id==cid].sort_values('Turn')
    P=d['Prompt'].fillna('').astype(str).tolist(); R=d['Response'].fillna('').astype(str).tolist()
    Ep=embed(P); Er=embed(R); n=len(d)
    for L in PREF_C1:
        if n<L: continue
        k=kfor(L); ep,er=Ep[:L],Er[:L]
        cp=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(ep)
        cr=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(er)
        cm=clears(cp[:-1],cr[:-1],cp[1:],k,seed_of(cid,L,'m'))
        cu=clears(cr[:-1],cp[1:],cr[1:],k,seed_of(cid,L,'u'))
        rows.append(dict(corpus='C1',record=cid,prefix=L,k=k,
                         model=cm,user=cu,both=bool(cm and cu) if (cm is not None and cu is not None) else None))
    if (ci+1)%10==0: print(f"C1 {ci+1}/{len(convs)} {time.time()-t0:.0f}s",flush=True)
pd.DataFrame(rows).to_csv('/home/claude/trackB/prefix.csv',index=False)
# ---- C3 ----
pairs=pd.read_csv('/home/claude/trackB/c3_pairs.csv'); games=sorted(pairs.game.unique())
for gi,gm in enumerate(games):
    d=pairs[pairs.game==gm].sort_values('turn')
    MA=d.msgA.fillna('').astype(str).tolist(); MB=d.msgB.fillna('').astype(str).tolist()
    EA=embed(MA); EB=embed(MB); n=len(d)
    for L in PREF_C3:
        if n<L: continue
        ca=KMeans(n_clusters=2,random_state=0,n_init=10).fit_predict(EA[:L])
        cb=KMeans(n_clusters=2,random_state=0,n_init=10).fit_predict(EB[:L])
        cm=clears(ca[:-1],cb[:-1],ca[1:],2,seed_of(gm,L,'m'))
        cu=clears(cb[:-1],ca[1:],cb[1:],2,seed_of(gm,L,'u'))
        rows.append(dict(corpus='C3',record=gm,prefix=L,k=2,
                         model=cm,user=cu,both=bool(cm and cu) if (cm is not None and cu is not None) else None))
    if (gi+1)%100==0:
        print(f"C3 {gi+1}/{len(games)} {time.time()-t0:.0f}s",flush=True)
        pd.DataFrame(rows).to_csv('/home/claude/trackB/prefix.csv',index=False)
pd.DataFrame(rows).to_csv('/home/claude/trackB/prefix.csv',index=False)
print("done")
