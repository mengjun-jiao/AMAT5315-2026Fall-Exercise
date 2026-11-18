#!/usr/bin/env python3
"""Plot contraction plans and actual factor-15 samples from saved run JSON."""
import json,math
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent.parent
def read(name): return json.loads((root/'runs'/name/'run.json').read_text())['evidence']
ks=range(2,9)
greedy=[read(f'plan_k{k}_greedy')['sc'] for k in ks]
treesa=[read(f'plan_k{k}_treesa')['sc'] for k in ks]
s=read('sample_f15'); hist=Counter(x['energy'] for x in s['samples'])
predicted=len(s['samples'])*s['ground_degeneracy']*math.exp(-2*s['e_min']-s['log_z'])
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
ax.plot(list(ks),greedy,'o--',color='#bc5d1e',label='greedy')
ax.plot(list(ks),treesa,'o-',color='#28638c',label='treesa')
ax.axhline(30,color='0.6',ls=':',lw=1);ax.text(2.05,30.7,'sc = 30 (~8 GiB)',fontsize=9,color='0.4')
ax.set_xlabel('Bits per factor k');ax.set_ylabel('Space complexity sc (log₂ entries)')
ax.set_title('(a) Contraction memory cost');ax.set_xticks(list(ks));ax.legend(frameon=False)
es=sorted(hist);ys=[hist[e] for e in es]
bars=bx.bar(range(len(es)),ys,color=['#28638c']+['#bc5d1e']*(len(es)-1))
for bar,count in zip(bars,ys):bx.text(bar.get_x()+bar.get_width()/2,count+max(ys)*.015,str(count),ha='center',fontsize=9)
bx.axhline(predicted,color='0.4',ls=':',lw=1)
bx.text(.98,predicted+max(ys)*.03,f'predicted from Z: {predicted:.0f}',transform=bx.get_yaxis_transform(),ha='right',fontsize=9,color='0.35')
bx.set_xticks(range(len(es)),[str(x) for x in es]);bx.set_xlabel('Spin-glass energy E');bx.set_ylabel('Samples (out of 2000)')
bx.set_title('(b) N = 15 sampled states (β = 2)');bx.set_ylim(0,max(ys)*1.24)
bx.annotate('ground state\n(decodes to 3 × 5)',(0,ys[0]*.65),xytext=(.28,.59),textcoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#28638c'},color='#28638c',fontsize=9)
for a in (ax,bx):a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.16)
fig.savefig(root/'artifacts/factoring.png',dpi=180)
