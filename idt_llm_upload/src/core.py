import numpy as np, pandas as pd, torch, hashlib
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
MODEL="/home/claude/models/minilm"
_tok=None;_mod=None
def embed(texts,bs=64):
    global _tok,_mod
    if _tok is None:
        _tok=AutoTokenizer.from_pretrained(MODEL);_mod=AutoModel.from_pretrained(MODEL).eval()
    out=[]
    for i in range(0,len(texts),bs):
        b=_tok([str(t) for t in texts[i:i+bs]],padding=True,truncation=True,max_length=256,return_tensors='pt')
        with torch.no_grad(): h=_mod(**b).last_hidden_state
        m=b['attention_mask'].unsqueeze(-1).float()
        v=(h*m).sum(1)/m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(v,dim=1).numpy())
    return np.vstack(out)

def mm_entropy(counts):
    """Miller-Madow corrected entropy (bits) from a count vector."""
    c=counts[counts>0]; N=c.sum()
    if N<=0: return 0.0
    p=c/N
    H=-(p*np.log2(p)).sum()
    return H+(len(c)-1)/(2*N*np.log(2))

def eps_from(S,A,Sp,k):
    """S,A,Sp integer arrays in [0,k). Returns (eps, I, room, resid)."""
    N=len(S)
    hS  = mm_entropy(np.bincount(S,minlength=k))
    hSS = mm_entropy(np.bincount(S*k+Sp,minlength=k*k))
    hSA = mm_entropy(np.bincount(S*k+A,minlength=k*k))
    hSAS= mm_entropy(np.bincount((S*k+A)*k+Sp,minlength=k*k*k))
    room = hSS-hS                 # H(S'|S)
    resid= hSAS-hSA               # H(S'|S,A)
    I    = room-resid
    eps  = I/room if room>0 else np.nan
    return eps,I,room,resid

# ---------- null generators ----------
def null_global(A,rng,**kw):
    return rng.permutation(A)

def null_strat(A,S,rng,minn=5,**kw):
    """Permute A only within S strata of size >= minn. Returns (A', mask)."""
    A2=A.copy(); mask=np.zeros(len(A),bool)
    for s in np.unique(S):
        idx=np.where(S==s)[0]
        if len(idx)>=minn:
            A2[idx]=rng.permutation(A[idx]); mask[idx]=True
    return A2,mask

def knn_sets(E,kappa=10,excl=2):
    """E: (n,d) embeddings of the S message. Returns list of neighbour index arrays."""
    n=len(E); sim=E@E.T
    order=np.argsort(-sim,axis=1)
    sets=[]
    for i in range(n):
        cand=[j for j in order[i] if abs(j-i)>=excl]
        sets.append(np.array(cand[:kappa],dtype=int))
    return sets

def null_local(A,sets,rng,**kw):
    n=len(A); A2=np.empty(n,dtype=A.dtype); used=np.zeros(n,bool)
    for i in rng.permutation(n):
        c=sets[i]
        free=c[~used[c]] if len(c) else c
        j=rng.choice(free) if len(free) else (rng.choice(c) if len(c) else i)
        A2[i]=A[j]; used[j]=True
    return A2

def null_circ(A,rng,B=10,**kw):
    n=len(A)
    if n<=2*B: return rng.permutation(A)
    off=rng.integers(B,n-B)
    return np.roll(A,off)

def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str,parts)).encode()).hexdigest()[:8],16)
