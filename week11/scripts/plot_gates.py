#!/usr/bin/env python3
"""Plot every row from the saved Rust gadget run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent.parent
evidence=json.loads((root/'runs/gadgets/run.json').read_text())['evidence']
fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
names=['AND','OR','NOT','COPY','SET0','SET1','XOR']
for i,name in enumerate(names):
    rows=[r for r in evidence['gates'] if r['gate']==name]
    for j,row in enumerate(rows):
        x=i+(j-(len(rows)-1)/2)*.09
        ax.scatter(x,row['energy'],marker='o' if row['valid'] else 'x',s=22,
                   c='#24658f' if row['valid'] else '#bd5d20',zorder=3)
ax.scatter([],[],marker='o',c='#24658f',s=22,label='valid row')
ax.scatter([],[],marker='x',c='#bd5d20',s=22,label='invalid row')
ax.set_xticks(range(len(names)),names,rotation=38)
ax.set_ylabel('Energy of a truth-table row')
ax.set_title('(a) Gate energies'); ax.legend(frameon=False,fontsize=9)
for i,row in enumerate(evidence['circuit']):
    bx.scatter(i,row['energy'],marker='o' if row['valid'] else 'x',s=25,
               c='#24658f' if row['valid'] else '#bd5d20')
    if row['valid']:
        bx.annotate(''.join(map(str,row['bits'][:2])),(i,row['energy']),xytext=(0,-14),
                    textcoords='offset points',ha='center',fontsize=8,color='#24658f')
bx.set_xticks([]);bx.set_xlabel('The 16 states of a, b, t, out')
bx.set_ylabel('Energy of a state');bx.set_title('(b) Three gates joined')
bx.set_ylim(-6.2,9.7)
for a in (ax,bx): a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.18)
fig.savefig(root/'artifacts/gates.png',dpi=180)
