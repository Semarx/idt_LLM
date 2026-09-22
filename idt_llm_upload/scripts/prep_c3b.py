import json,glob,os,pandas as pd,numpy as np
rows=[];meta=[]
for f in sorted(glob.glob('/tmp/pb/logs_x/logs/*.json')):
    d=json.load(open(f)); gname=os.path.basename(f)[:-5]
    sc=[r.get('score',0) for r in d['rounds']]
    vals=[]
    for s in sc:
        vals+= list(s.values()) if isinstance(s,dict) else [s]
    score_anom=any(v>3 for v in vals if isinstance(v,(int,float)))
    utt=[]
    for r in d['rounds']:
        for m in r['messages']:
            tx=str(m['message'])
            if '<selection>' in tx: continue          # 2. drop selections FIRST
            sp=m['speaker']
            if utt and utt[-1][0]==sp: utt[-1][1]+=' '+tx   # 3. then collapse
            else: utt.append([sp,tx])
    if not utt: meta.append((gname,0,0,'no_messages')); continue
    A=utt[0][0]                                        # 4. side A = first surviving speaker
    pairs=[];i=0
    while i<len(utt)-1:                                # 5. non-overlapping A->B walk
        if utt[i][0]==A and utt[i+1][0]!=A: pairs.append((utt[i][1],utt[i+1][1])); i+=2
        else: i+=1
    n=len(pairs)
    reason='score_anomaly' if score_anom else ('below_floor' if n<25 else '')
    meta.append((gname,len(utt),n,reason))
    if reason=='':
        for t,(a,b) in enumerate(pairs): rows.append((gname,t,a,b))
md=pd.DataFrame(meta,columns=['game','n_utt','n_turns','drop_reason'])
md.to_csv('c3_manifest.csv',index=False)
pd.DataFrame(rows,columns=['game','turn','msgA','msgB']).to_csv('c3_pairs.csv',index=False)
print("games:",len(md),"| kept:",(md.drop_reason=='').sum(),
      "| score_anomaly:",(md.drop_reason=='score_anomaly').sum(),
      "| below_floor:",(md.drop_reason=='below_floor').sum())
pb=pd.read_csv('/mnt/user-data/uploads/Claude outputs/pb_full.csv').set_index('game')
j=md[md.drop_reason==''].set_index('game').join(pb[['n_turns']],rsuffix='_theirs',how='inner')
print("joined:",len(j),"| n_turns exact match:",int((j.n_turns==j.n_turns_theirs).sum()))
print("game-set identical:",set(md[md.drop_reason==''].game)==set(pb.index.astype(str)))
print("utterance range:",md[md.drop_reason==''].n_utt.min(),"-",md[md.drop_reason==''].n_utt.max(),
      "| turns:",j.n_turns.min(),"-",j.n_turns.max())
