import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BLUE='#2A78D6'; ORANGE='#EB6834'; BG='#FCFCFB'; INK='#1a1a1a'; MUT='#6b6b6b'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.facecolor':BG,
 'axes.facecolor':BG,'axes.edgecolor':'#cfcfcf','axes.labelcolor':INK,
 'xtick.color':MUT,'ytick.color':MUT,'axes.spines.top':False,'axes.spines.right':False})
c1=pd.read_csv('c1_nulls.csv'); c3=pd.read_csv('c3_nulls.csv'); pre=pd.read_csv('prefix.csv')
c1=c1[c1.screen=='PASS']; c3=c3[c3.screen=='PASS']
fig=plt.figure(figsize=(12.9,8.2))
gs=fig.add_gridspec(2,2,hspace=.60,wspace=.22,left=.065,right=.98,top=.90,bottom=.085)

def scat(ax,d,title,sub,logx=True):
    for side,col,lbl in (('user',BLUE,"first side's message acting"),('model',ORANGE,"second side's message acting")):
        s=d[d.side==side]; cl=s.clear_strat.astype(str)=='True'
        ax.scatter(s.N[cl]+1,s.eps[cl],s=34,color=col,alpha=.85,linewidths=0,zorder=3)
        ax.scatter(s.N[~cl]+1,s.eps[~cl],s=30,facecolors='none',edgecolors=col,linewidths=1.1,alpha=.6,zorder=2)
    ax.axhline(0,color='#9a9a9a',lw=.9,zorder=1)
    if logx:
        ax.set_xscale('log'); ax.set_xticks([25,50,100,200,400,800,1400])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('paired turns'); ax.set_ylabel('ε̂')
    ax.grid(axis='y',color='#ececec',lw=.8); ax.set_axisbelow(True)
    ax.set_title(title,loc='left',fontsize=11.5,fontweight='bold',color=INK,pad=16)
    ax.text(0,1.02,sub,transform=ax.transAxes,fontsize=8.4,color=MUT)

ax=fig.add_subplot(gs[0,0])
scat(ax,c1,'a   C1   human–LLM','58 conversations · 25 model-side and 36 human-side readings clear')
ax.legend(handles=[Line2D([],[],marker='o',ls='',ms=7,color=BLUE,label="human's message acting"),
                   Line2D([],[],marker='o',ls='',ms=7,color=ORANGE,label="model's message acting"),
                   Line2D([],[],marker='o',ls='',ms=7,mfc='none',mec='#666',label='inconclusive')],
          loc='upper left',frameon=False,fontsize=8.2,handletextpad=.4,labelspacing=.35)
ax2=fig.add_subplot(gs[0,1])
scat(ax2,c3,'b   C3   human–human','2,171 games · 620 second-speaker and 784 first-speaker readings clear',logx=False)
ax2.set_xticks([25,35,45,55,65])

ax3=fig.add_subplot(gs[1,0])
for c,mk,mfc,lab,yy in (('C1','o',None,'C1  human–LLM',1.045),('C3','s','none','C3  human–human',1.115)):
    s=pre[(pre.corpus==c)&pre.both.notna()].groupby('prefix')['both'].agg(['count','mean'])
    ax3.plot(s.index,s['mean'],marker=mk,ms=7,color='#333',mfc=(mfc or '#333'),lw=1.3,label=lab)
    for x,(n,r) in s.iterrows(): ax3.annotate(str(int(n)),(x,yy),ha='center',fontsize=7.2,color=MUT,
                                              annotation_clip=False,xycoords=('data','axes fraction'))
    ax3.text(-0.008,yy,'n',transform=ax3.transAxes,fontsize=7.4,color=MUT,ha='right')
ax3.set_xscale('log'); ax3.set_xticks([25,35,45,55,65,90,120,200,400])
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax3.set_xlabel('prefix length  (first N paired turns)'); ax3.set_ylabel('share of records\nwhere both sides clear')
ax3.set_ylim(-.04,1.05); ax3.grid(axis='y',color='#ececec',lw=.8); ax3.set_axisbelow(True)
ax3.legend(loc='upper left',frameon=False,fontsize=8.4)
ax3.set_title('c   Clearing against record length',loc='left',fontsize=11.5,fontweight='bold',color=INK,pad=44)
ax3.text(0,1.175,'prefixes with fewer than 10 records are not plotted; n printed above',
         transform=ax3.transAxes,fontsize=8.4,color=MUT)

ax4=fig.add_subplot(gs[1,1])
eps=c1.pivot_table(index='record',columns='side',values='eps',aggfunc='first')
w=c1.pivot_table(index='record',columns='side',values='clear_strat',aggfunc='first')
both=w.index[(w['model'].astype(str)=='True')&(w['user'].astype(str)=='True')]
dd=np.sort((eps.loc[both,'model']-eps.loc[both,'user']).values)
ax4.bar(range(len(dd)),dd,color=[BLUE if v<0 else ORANGE for v in dd],width=.78)
ax4.axhline(0,color='#555',lw=1)
ax4.set_xticks([]); ax4.set_ylabel('Δε̂'); ax4.grid(axis='y',color='#ececec',lw=.8); ax4.set_axisbelow(True)
ax4.set_xlabel(f'the {len(dd)} C1 conversations where both sides cleared, ordered')
ax4.set_ylim(min(dd)*1.12, max(dd)*2.35)
ax4.text(.03,.965,f'{int((dd<0).sum())} of {len(dd)} negative   ·   median \u0394\u03b5\u0302 = {np.median(dd):+.3f}\n95 % CI [\u22120.084, \u22120.029]   ·   exploratory',
         transform=ax4.transAxes,fontsize=8.8,color=INK,va='top',ha='left',
         bbox=dict(boxstyle='round,pad=0.35',fc=BG,ec='none',alpha=0.9))
ax4.set_title('d   ε̂(model) − ε̂(human), both sides clear (n = 24)',loc='left',fontsize=11.5,fontweight='bold',color=INK,pad=16)
ax4.text(0,1.02,'C1 only  ·  one author  ·  the 34 records without a bilateral reading are not shown',transform=ax4.transAxes,fontsize=8.4,color=MUT)
fig.savefig('/mnt/user-data/outputs/fig4_epsilon_v5.png',dpi=200,facecolor=BG)
print('saved', len(dd))
