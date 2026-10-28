from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
root=Path(__file__).resolve().parent.parent; result=json.loads((root/'artifacts'/'part3_memory.json').read_text(encoding='utf8')); ratios=result['ratios']; sizes=2**np.arange(3,17); names=['outer','least_squares','relu','relu2','relu3']; labels=['Outer product','Least squares','ReLU','ReLU²','ReLU³']; colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
predictions=[np.sqrt(sizes/64),np.sqrt(np.maximum(sizes/64-1,0)),np.sqrt(sizes/(2*64)),np.sqrt(3*sizes/(2*64**2)),np.sqrt(15*sizes/(2*64**3))]
fig,ax=plt.subplots(figsize=(9,6))
for name,label,color,pred in zip(names,labels,colors,predictions):
 y=np.array([ratios['64'][str(int(m))][name] for m in sizes]);ax.loglog(sizes,y,'o-',ms=3,color=color,label=label);ax.loglog(sizes,np.where(pred>0,pred,np.nan),'--',color=color,alpha=.5,lw=1)
ax.axhline(1,color='black',ls=':',lw=1);ax.set(xlabel='Number of memories M',ylabel='Interference / signal, d=64',ylim=(.001,100),title='Solid: measured; dashed: theoretical estimate');ax.legend(loc='upper left',fontsize=8)
ins=inset_axes(ax,width='36%',height='28%',loc='lower right',borderpad=1.7);xs=np.array([8,16,32,64]);ys=np.array([ratios['64'][str(int(m))]['least_squares'] for m in xs]); ins.loglog(xs,ys,'o-',color=colors[1],ms=3);ins.set_title('Least squares, M ≤ d',fontsize=8);ins.set_xticks(xs,[str(v) for v in xs]);ins.xaxis.set_minor_formatter(NullFormatter());ins.tick_params(axis='both',which='both',labelsize=7);ins.grid(alpha=.2)
fig.savefig(root/'evidence'/'part3_interference.png',dpi=170,bbox_inches='tight')
fig,ax=plt.subplots(figsize=(8,5));dims=np.array([16,32,64]);xs=np.geomspace(16,64,100)
for name,label,color in zip(names,labels,colors):
 caps=np.array(result['capacities'][name]); slope,intercept=np.polyfit(np.log(dims),np.log(caps),1);ax.loglog(dims,caps,'o',color=color,label=f'{label}: {slope:.3f}');ax.loglog(xs,np.exp(intercept)*xs**slope,'--',color=color)
ax.set_xticks(dims,[str(v) for v in dims]);ax.xaxis.set_minor_formatter(NullFormatter());ax.set(xlabel='Dimension d',ylabel='Capacity at R = 1',title='Points: measured capacity; dashed: fitted power law');ax.legend();fig.tight_layout();fig.savefig(root/'evidence'/'part3_capacity.png',dpi=170)
