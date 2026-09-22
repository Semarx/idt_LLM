import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BLUE='#2A78D6'; ORANGE='#EB6834'; PBLUE='#A8C7EC'; PORANGE='#F5C0AB'; BG='#FCFCFB'
INK='#1a1a1a'; MUT='#6b6b6b'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.facecolor':BG,
    'axes.facecolor':BG,'axes.edgecolor':'#cfcfcf','axes.labelcolor':INK,
    'xtick.color':MUT,'ytick.color':MUT,'axes.spines.top':False,'axes.spines.right':False})

a=pd.read_csv('c1_controls.csv'); a['corpus']='C1'
b=pd.read_csv('c2_controls.csv'); b['corpus']='C2'
d=pd.concat([a,b],ignore_index=True)
cos=pd.concat([pd.read_csv('c1_cos.csv'),
    b[['record','side','cell','cos_median']].dropna()],ignore_index=True)
obs=d[d.cell=='C0_observed'].set_index(['record','side'])['eps_median']
d['obs']=d.set_index(['record','side']).index.map(obs)
order=[('claude_90b6161d','model'),('claude_90b6161d','user'),
       ('claude_3df4ba21','model'),('claude_3df4ba21','user'),
       ('claude_9c24efbd','model'),('claude_9c24efbd','user'),
       ('C2_pooled_k3','model'),('C2_pooled_k3','user'),
       ('C2_pooled_k2','model'),('C2_pooled_k2','user')]
def lab(r,s):
    who='model acting' if s=='model' else 'human acting'
    if r.startswith('C2'): return f"C2  pooled  k = {r[-1]}  ·  {'answerer' if s=='model' else 'asker'} acting"
    return f"C1  {r[7:15]}  ·  {who}"
CELLS=[('C1_compat_nonadj','Control C   state-matched',  '^'),
       ('C2_lengthA','Control A   length-matched','s'),
       ('C3_dissimB','Control B   length + dissimilar','D')]

fig=plt.figure(figsize=(12.9,7.4))
gs=fig.add_gridspec(2,2,width_ratios=[1.38,1],height_ratios=[1,1],hspace=.55,wspace=.30,
                    left=.255,right=.965,top=.88,bottom=.085)
ax=fig.add_subplot(gs[:,0])
for i,(r,s) in enumerate(order):
    y=len(order)-1-i; col=ORANGE if s=='model' else BLUE
    row=d[(d.record==r)&(d.side==s)]
    o=float(row[row.cell=='C0_observed'].eps_median.iloc[0])
    ax.plot([0,o],[y,y],color='#d8d8d8',lw=1,zorder=1,solid_capstyle='round')
    for (cell,_,mk) in CELLS:
        rr=row[row.cell==cell]
        if not len(rr): continue
        m=float(rr.eps_median.iloc[0]); lo=float(rr.eps_lo.iloc[0]); hi=float(rr.eps_hi.iloc[0])
        ax.plot([lo,hi],[y,y],color=col,lw=2.6,alpha=.30,solid_capstyle='round',zorder=2)
        ax.plot(m,y,marker=mk,ms=7,mfc=BG,mec=col,mew=1.7,zorder=4)
    ax.plot(o,y,'o',ms=9,color=col,zorder=5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([lab(r,s) for r,s in order][::-1],fontsize=8.6,color=INK)
ax.axvline(0,color='#9a9a9a',lw=.9,zorder=0)
ax.set_xlabel('ε̂',fontsize=11)
ax.grid(axis='x',color='#ececec',lw=.8); ax.set_axisbelow(True)
ax.set_title('a   ε̂ when the adjacency is broken',loc='left',fontsize=11.5,
             fontweight='bold',color=INK,pad=16)
ax.text(0,1.018,'bars: 2.5–97.5 % over 100 replicates   ·   filled circle: observed',
        transform=ax.transAxes,fontsize=8.4,color=MUT)
ax.legend(handles=[Line2D([],[],marker='o',ls='',ms=8,color='#555',label='observed')]+
  [Line2D([],[],marker=mk,ls='',ms=7,mfc=BG,mec='#555',mew=1.6,label=t) for _,t,mk in CELLS],
  loc='upper right',frameon=False,fontsize=8.3,handletextpad=.5,labelspacing=.45)

ax2=fig.add_subplot(gs[0,1])
w=.26
for j,(cell,t,_) in enumerate(CELLS):
    xs=[];ys=[];cs=[]
    for i,(r,s) in enumerate(order):
        m=cos[(cos.record==r)&(cos.side==s)&(cos.cell==cell)]
        xs.append(i+(j-1)*w); ys.append(float(m.cos_median.iloc[0]) if len(m) else np.nan)
        cs.append((ORANGE if s=='model' else BLUE) if j==2 else (PORANGE if s=='model' else PBLUE))
    ax2.bar(xs,ys,width=w*.92,color=cs,edgecolor=BG,linewidth=1)
ax2.axhline(.5,ls=(0,(4,3)),color='#7a7a7a',lw=1)
ax2.text(len(order)-.4,.518,'predeclared ceiling 0.50',ha='right',fontsize=8.2,color=MUT)
ax2.set_xticks([]); ax2.set_ylabel('median cosine,\ntrue action vs substitute',fontsize=8.8)
ax2.set_ylim(0,.70); ax2.grid(axis='y',color='#ececec',lw=.8); ax2.set_axisbelow(True)
ax2.set_title('b   Similarity of the substitute',loc='left',fontsize=11.5,fontweight='bold',color=INK,pad=14)
ax2.text(0,1.035,'pale → solid: Control C, A, B',transform=ax2.transAxes,fontsize=8.4,color=MUT)

ax3=fig.add_subplot(gs[1,1])
for j,(cell,t,_) in enumerate(CELLS):
    xs=[];ys=[];cs=[]
    for i,(r,s) in enumerate(order):
        m=d[(d.record==r)&(d.side==s)&(d.cell==cell)]
        xs.append(i+(j-1)*w); ys.append(100*float(m.rep_clear_rate.iloc[0]) if len(m) else np.nan)
        cs.append((ORANGE if s=='model' else BLUE) if j==2 else (PORANGE if s=='model' else PBLUE))
    ax3.bar(xs,ys,width=w*.92,color=cs,edgecolor=BG,linewidth=1)
ax3.set_xticks(range(len(order)))
ax3.set_xticklabels([('90b\n'+('mod' if s=='model' else 'hum')) if r=='claude_90b6161d' else
                     ('3df\n'+('mod' if s=='model' else 'hum')) if r=='claude_3df4ba21' else
                     ('9c2\n'+('mod' if s=='model' else 'hum')) if r=='claude_9c24efbd' else
                     (('k=3\n' if 'k3' in r else 'k=2\n')+('mod' if s=='model' else 'hum'))
                     for r,s in order],fontsize=7.6)
ax3.set_ylabel('replicates still clearing\nout of 100',fontsize=8.8); ax3.set_ylim(0,105)
ax3.grid(axis='y',color='#ececec',lw=.8); ax3.set_axisbelow(True)
ax3.set_title('c   Residual that still clears',loc='left',fontsize=11.5,fontweight='bold',color=INK,pad=14)
ax3.text(0,1.035,'pale → solid: Control C, A, B',transform=ax3.transAxes,fontsize=8.4,color=MUT)
fig.savefig('/mnt/user-data/outputs/fig5_controls_v2.png',dpi=200,facecolor=BG)
print('saved')
