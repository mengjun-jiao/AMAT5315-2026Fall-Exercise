#!/usr/bin/env python3
"""Plot saved exact counts and the 6-queens sampled boards."""
import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root=Path(__file__).resolve().parent.parent
def read(name):return json.loads((root/'runs'/name/'run.json').read_text())['evidence']
ns=list(range(4,11));counts=[read(f'count_nq{n}')['count'] for n in ns]
known=[2,10,4,40,92,352,724]
sample=read('sample_nq6');freq=Counter(tuple(b) for b in sample['samples']);boards=sorted(freq)
assert len(boards)==4
fig=plt.figure(figsize=(11.5,5.7),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[4.0,1.05])
ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1]);boardax=fig.add_subplot(gs[1,1]);blank=fig.add_subplot(gs[1,0]);blank.axis('off')
ax.plot(ns,counts,'o-',color='#28638c',label='contraction')
ax.plot(ns,known,'x',color='#bd5d20',ms=7,label='known (OEIS A000170)')
for n,v in zip(ns,counts):ax.annotate(str(v),(n,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8,color='#28638c')
ax.set_yscale('log');ax.set_xticks(ns);ax.set_xlabel('Board size N');ax.set_ylabel('Valid boards (log scale)')
ax.set_title('(a) Exact N-queens counts');ax.legend(frameon=False,fontsize=9)
bars=bx.bar(range(4),[freq[b] for b in boards],color='#28638c',width=.65)
for bar in bars:bx.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,str(int(bar.get_height())),ha='center',fontsize=9)
bx.axhline(50,color='0.4',ls='--',lw=1);bx.text(-.3,51,'uniform',ha='left',fontsize=9,color='0.4')
bx.set_ylim(0,max(70,max(freq.values())+10));bx.set_xticks([]);bx.set_ylabel('Times drawn');bx.set_title('(b) 200 sampled 6-queens boards')
boardax.set_xlim(-.5,3.5);boardax.set_ylim(-.3,1);boardax.axis('off')
for i,board in enumerate(boards):
    for r in range(6):
        for c in range(6):
            x=i-.18+c*.06;y=.6-r*.1
            boardax.add_patch(Rectangle((x,y),.06,.1,facecolor='#e0e9f1' if (r+c)%2==0 else 'white',edgecolor='none'))
        boardax.plot(i-.15+board[r]*.06,.65-r*.1,'o',color='#28638c',ms=2.7)
    boardax.add_patch(Rectangle((i-.18,.1),.36,.6,fill=False,ec='.5',lw=.6))
boardax.text(1.5,-.04,'The 4 solutions',ha='center',va='top',fontsize=9)
for a in (ax,bx):a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15)
fig.savefig(root/'artifacts/nqueens.png',dpi=180)
