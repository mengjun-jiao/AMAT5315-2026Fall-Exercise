from pathlib import Path
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
root=Path(__file__).resolve().parent.parent; ev=root/'evidence';art=root/'artifacts';ev.mkdir(exist_ok=True)
def get(p):return json.loads(p.read_text(encoding='utf8'))
labels={0:'A',1:'T',2:'S',3:'L',4:'B',5:'E',6:'X',7:'D'}
# Tensor-network convention as in the learning sheet: tensors are boxes,
# shared indices are wires with junction dots, selectors are caps, and T is open.
fig,ax=plt.subplots(figsize=(10.8,6.4));ax.set_xlim(-.1,10.1);ax.set_ylim(.2,6.65);ax.axis('off')
factor_boxes=[(2.,5.65,r'$p(a)$'),(2.,4.4,r'$p(t\mid a)$'),(7.5,5.65,r'$p(s)$'),
              (6.1,4.4,r'$p(l\mid s)$'),(8.85,4.4,r'$p(b\mid s)$'),
              (4.15,2.85,r'$p(e\mid t,l)$'),(2.,1.3,r'$p(x\mid e)$'),
              (6.95,1.3,r'$p(d\mid e,b)$')]
box_w,box_h=1.35,.54
def wire(points):
 xs,ys=zip(*points);ax.plot(xs,ys,color='#263c4b',lw=1.45,zorder=1,solid_capstyle='round')
def junction(x,y):
 ax.plot(x,y,'o',color='#263c4b',ms=3.3,zorder=4)
def cap(x,y,label,color):
 ax.add_patch(FancyBboxPatch((x-.48,y-.25),.96,.5,boxstyle='round,pad=.035',fc='white',ec=color,lw=1.5,zorder=3))
 ax.text(x,y,label,ha='center',va='center',fontsize=11,color=color,zorder=4)
# A is shared by p(a), p(t|a), and the A=0 selector.
wire([(2,5.38),(2,4.67)]);wire([(1.24,5.03),(2,5.03)]);junction(2,5.03);cap(.72,5.03,'(1, 0)','#bb6221')
# S is the same index in p(s), p(l|s), and p(b|s).
wire([(6.1,4.67),(6.1,5.03),(8.85,5.03),(8.85,4.67)]);wire([(7.5,5.38),(7.5,5.03)]);junction(7.5,5.03)
# T joins two tensors and stays open at the left; L joins its two tensors.
wire([(2,4.13),(2,2.85),(3.475,2.85)]);wire([(.92,2.85),(2,2.85)]);junction(2,2.85)
ax.plot(.92,2.85,'o',mfc='white',mec='#bf542c',ms=5,zorder=5)
wire([(6.1,4.13),(6.1,2.85),(4.825,2.85)])
# E is shared by three tables; B joins the bronchitis and dyspnoea tables.
wire([(2.675,1.3),(6.275,1.3)]);wire([(4.15,2.58),(4.15,1.3)]);junction(4.15,1.3)
wire([(8.85,4.13),(6.95,1.57)])
# Unknown X is summed; D=0 selects the first dyspnoea state.
wire([(1.02,1.3),(1.325,1.3)]);cap(.50,1.3,'(1, 1)','#2672a2')
wire([(7.625,1.3),(8.5,1.3)]);cap(9.02,1.3,'(1, 0)','#bb6221')
for x,y,label in factor_boxes:
 ax.add_patch(FancyBboxPatch((x-box_w/2,y-box_h/2),box_w,box_h,boxstyle='round,pad=.035',fc='white',ec='#3d5267',lw=1.3,zorder=3))
 ax.text(x,y,label,ha='center',va='center',fontsize=12,color='#263c4b',zorder=4)
for x,y,label in [(2.18,4.88,'a'),(7.75,4.78,'s'),(1.55,3.11,'t'),(5.45,3.11,'l'),(8.16,3.05,'b'),(4.34,1.94,'e'),(1.15,1.56,'x'),(8.1,1.56,'d')]:
 ax.text(x,y,rf'$\mathit{{{label}}}$',fontsize=11,color='#386f9c')
ax.text(.49,2.85,r'$T$',ha='center',va='center',fontsize=13,color='#bd432b',fontweight='bold')
ax.text(.72,5.43,r'$A=0$',ha='center',fontsize=10,color='#a9541e')
ax.text(9.02,1.71,r'$D=0$',ha='center',fontsize=10,color='#a9541e')
ax.text(5.,.58,r'Output: $p(T,A=0,D=0)$  •  only $T$ remains open',ha='center',fontsize=11,color='#263c4b')
ax.set_title('ASIA AD tuberculosis query: tensor network',fontsize=14)
fig.savefig(ev/'graphs.svg',bbox_inches='tight');fig.savefig(ev/'graphs.png',dpi=180,bbox_inches='tight');plt.close(fig)

# Match the learning-sheet comparison: panels are observation cases,
# horizontal bars are enumeration, and black marks are contractions.
cases=['prior','AD','ADX'];runs={c:get(art/c/'run.json') for c in cases}
names=['Tuberculosis','Lung cancer','Bronchitis'];vids=[1,3,4]
case_titles=['Prior','Visit + dyspnoea','Visit + dyspnoea\n+ positive X-ray']
fig,axs=plt.subplots(1,3,figsize=(11,4.2),sharey=True,layout='constrained')
for ax,c,title in zip(axs,cases,case_titles):
 ref=[runs[c]['comparison_with_enumeration']['enumeration_marginals'][v][0] for v in vids]
 contracted=[runs[c]['marginals'][str(v)][0] for v in vids]
 y=np.arange(len(vids));ax.barh(y,ref,height=.52,color='#296798',label='Enumeration')
 ax.plot(contracted,y,'|',color='black',markersize=18,markeredgewidth=1.5,label='Tensor contraction')
 for i,z in enumerate(contracted):ax.text(z+.025,i,f'{z:.3f}',va='center',fontsize=9)
 ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xticks([0,.5,1])
 ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
axs[1].set_xlabel('Probability of disease being present')
fig.suptitle('ASIA disease probabilities: enumeration and contraction',fontsize=13)
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
fig.savefig(ev/'posteriors.png',dpi=180);plt.close(fig)

fig,axs=plt.subplots(3,3,figsize=(10,10),layout='constrained')
common_lim=max(float(np.max(np.abs(np.array(runs[c]['joint_TL'])-np.outer(runs[c]['marginals']['1'],runs[c]['marginals']['3'])))) for c in cases)
for i,c in enumerate(cases):
 run=runs[c];j=np.array(run['joint_TL']);p=np.outer(run['marginals']['1'],run['marginals']['3']);diff=j-p
 for col,(data,title,cmap) in enumerate([(j,'Joint p(T,L)','Blues'),(p,'p(T) p(L)','Blues'),(diff,'Joint − product','RdBu_r')]):
  ax=axs[i,col];ax.imshow(data,cmap=cmap,**({'vmin':-common_lim,'vmax':common_lim} if col==2 else {'vmin':0,'vmax':1}));ax.set_xticks([0,1],['L yes','L no']);ax.set_yticks([0,1],['T yes','T no']);ax.set_title(f'{c}: {title}');
  for yy in range(2):
   for xx in range(2):ax.text(xx,yy,f'{data[yy,xx]:.4f}',ha='center',va='center',color='black',fontsize=10,bbox=dict(fc='white',alpha=.65,ec='none',pad=1))
fig.suptitle('Dependence between tuberculosis and lung cancer (shared color scales)');fig.savefig(ev/'joint.png',dpi=180);plt.close(fig)

# Draw exact saved binary trees; leaves refer to factors after evidence conditioning.
def tree_coordinates(plan):
 children={s['output']:s['inputs'] for s in plan['steps']};rootid=plan['steps'][-1]['output'];pos={};edges=[];leaves=[]
 def walk(node,depth):
  if node not in children:x=len(leaves);leaves.append(node)
  else:
   xs=[walk(child,depth+1) for child in children[node]];x=sum(xs)/len(xs)
   for child in children[node]:edges.append((node,child))
  pos[node]=(x,-depth);return x
 walk(rootid,0);return pos,edges,leaves
fig=plt.figure(figsize=(13,10),layout='constrained')
grid=fig.add_gridspec(3,2,height_ratios=[2.2,2.2,1.15])
tree_axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,:])]
order_plans={}
factor_names={0:'p(T|A=0)',1:'p(S)',2:'p(L|S)',3:'p(B|S)',4:'p(E|L,T)',5:'p(X|E)',6:'p(D=0|E,B)'}
for ax,name in zip(tree_axes,['random','greedy']):
 p=get(root/'runs'/name/'order.json');pos,edges,leaves=tree_coordinates(p);steps={s['output']:s for s in p['steps']}
 assert p['input_scopes']==[[1],[2],[3,2],[4,2],[5,3,1],[6,5],[5,4]],'Unexpected conditioned ASIA factors'
 order_plans[name]=p
 for a,b in edges: ax.plot([pos[a][0],pos[b][0]],[pos[a][1],pos[b][1]],color='#a3a3a3',lw=1.2)
 for node,(x,y) in pos.items():
  if node in steps:
   st=steps[node];vs=''.join(labels[v] for v in st['retained_labels']) or '∅';size=st['output_entries'];txt=f"{vs}\n{size} {'entry' if size==1 else 'entries'}";fc='#f5a77d' if size==p['largest_generated'] else '#d7ecf4'
  else:vs=''.join(labels[v] for v in p['input_scopes'][node]);size=2**len(p['input_scopes'][node]);txt=f'f{node}: {factor_names[node]}\n{vs} · {size} entries';max_input=max(2**len(s) for s in p['input_scopes']);fc='#ffe292' if size==max_input else '#f1f1f1'
  ax.text(x,y,txt,ha='center',va='center',fontsize=8,bbox=dict(boxstyle='round,pad=.25',fc=fc,ec='#5a7a89'))
 ax.set_title(f"{name.capitalize()} saved tree | work {p['total_work']:,} | largest tensor {p['largest_tensor']:,} | largest intermediate {p['largest_generated']:,}",loc='left',fontsize=11)
 ax.axis('off');ax.margins(x=.1,y=.16)
for ax,key,title in [(fig.add_subplot(grid[2,0]),'total_work','Work: multiply-accumulate terms'),
                     (fig.add_subplot(grid[2,1]),'largest_tensor','Largest tensor: entries (including inputs)')]:
 vals=[order_plans[n][key] for n in ['random','greedy']]
 ax.barh(['Random','Greedy'],vals,color=['#b96728','#2f6f9e']);ax.invert_yaxis()
 ax.set_xlim(0,max(vals)*1.22);ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 for i,val in enumerate(vals):ax.text(val+max(vals)*.025,i,f'{val:,}',va='center',fontsize=9)
z=get(root/'runs'/'greedy'/'run.json')['evidence_mass']
fig.suptitle(f"Same ASIA query: P(A=0, D=0) = {z:.9f}\nOrange nodes: largest intermediate; yellow: largest input. Scalar factor p(A=0) tracked separately.",fontsize=12)
fig.savefig(ev/'orders.png',dpi=180);plt.close(fig)

fig,axs=plt.subplots(3,2,figsize=(10,9),layout='constrained')
for row,c in enumerate(cases):
 bp=get(art/('bp-'+c)/'run.json');ex=np.array(runs[c]['comparison_with_enumeration']['enumeration_marginals']);steps=np.array([q['sweep'] for q in bp['trace']]);res=np.array([q['residual'] for q in bp['trace']]);errors=np.array([np.max(abs(np.array(q['beliefs'])-ex)) for q in bp['trace']]);
 for ax,values,title in [(axs[row,0],res,'Undamped message residual'),(axs[row,1],errors,'Maximum marginal error')]:
  ax.semilogy(steps,np.maximum(values,1e-16));ax.set_title(f'{c}: {title}');ax.set_xlabel('Sweep');ax.grid(alpha=.2)
 axs[row,0].axhline(1e-10,color='gray',ls='--',lw=.8,label='tolerance');axs[row,0].legend(fontsize=8)
fig.suptitle('ASIA loopy BP: convergence and accuracy');fig.savefig(ev/'bp_asia.png',dpi=180);plt.close(fig)

# Additional view matching the ADX example in the learning sheet; keep the
# three-case residual/error figure above for the full hands-on requirement.
bp=get(art/'bp-ADX'/'run.json');exact=runs['ADX'];vids_adx=[1,3,4,2]
names_adx=['Tuberculosis','Lung cancer','Bronchitis','Smoking']
exact_vals=[exact['marginals'][str(v)][0] for v in vids_adx]
bp_vals=[bp['marginals'][v][0] for v in vids_adx]
truth=np.array(exact['comparison_with_enumeration']['enumeration_marginals'])
steps=np.array([row['sweep'] for row in bp['trace']])
res=np.array([row['residual'] for row in bp['trace']])
errors=np.array([np.max(abs(np.array(row['beliefs'])-truth)) for row in bp['trace']])
fig,axs=plt.subplots(1,2,figsize=(10.8,4.6),layout='constrained')
y=np.arange(len(vids_adx));h=.32
axs[0].barh(y-h/2,exact_vals,height=h,color='#296eaa',label='Exact')
axs[0].barh(y+h/2,bp_vals,height=h,color='#b76522',label='Loopy BP')
for i,(e,b) in enumerate(zip(exact_vals,bp_vals)):
 axs[0].text(e+.012,i-h/2,f'{e:.3f}',ha='left',va='center',fontsize=8,color='#296eaa')
 axs[0].text(b+.012,i+h/2,f'{b:.3f}',ha='left',va='center',fontsize=8,color='#a95616')
axs[0].set_yticks(y,names_adx);axs[0].invert_yaxis();axs[0].set_xlim(0,1)
axs[0].set(title='(a) Final probabilities',xlabel='Probability of being present');axs[0].legend(loc='lower right',fontsize=8)
axs[0].grid(axis='x',alpha=.2);axs[0].set_axisbelow(True)
axs[1].plot(steps,res,color='#296eaa',label='Undamped message residual')
axs[1].plot(steps,errors,'--',color='#b76522',label='Largest marginal error')
axs[1].set(title='(b) Convergence and error',xlabel='Sweep',ylabel='Absolute difference')
axs[1].grid(alpha=.2);axs[1].legend(fontsize=8)
fig.suptitle('ASIA: visit + dyspnoea + positive X-ray — loopy BP is close, not exact',fontsize=12)
fig.text(.5,-.015,f"After {bp['sweeps']} sweeps: residual {bp['residual']:.2e}; maximum marginal error {errors[-1]:.5f}.",ha='center',fontsize=9)
fig.savefig(ev/'bp_adx_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)

check=get(art/'promedus_checks.json');bp=get(art/'promedus-bp/run.json');ex=np.array([check['greedy_marginals'][str(v)] for v in [59,102,139]]);steps=np.array([t['sweep'] for t in bp['trace']]);res=np.array([t['residual'] for t in bp['trace']]);errs=np.array([np.max(abs(np.array(t['beliefs'])[[59,102,139]]-ex)) for t in bp['trace']]);g=check['plans_independently_recounted']['promedus-greedy/order.json'];t=check['plans_independently_recounted']['promedus-treesa/order.json'];fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for ax,field,title in [(axs[0,0],'work','Scalar plan: operation count'),(axs[0,1],'largest_generated','Largest generated tensor (entries)')]:
 vals=[g[field],t[field]];ax.bar(['Greedy','TreeSA'],vals,color=['#d58257','#4389b5']);ax.set_yscale('log');ax.set_title(title)
 for i,x in enumerate(vals):ax.text(i,x*1.15,f'{x:,}',ha='center',fontsize=8)
 ax.set_ylim(min(vals)*.5,max(vals)*5)
axs[1,0].semilogy(steps,np.maximum(res,1e-18));axs[1,0].axhline(1e-10,color='gray',ls='--',label='tolerance');axs[1,0].legend();axs[1,0].set(title='BP undamped residual',xlabel='Sweep');axs[1,1].semilogy(steps,np.maximum(errs,1e-18),color='#b75042');axs[1,1].set(title='BP max error for IDs 59, 102, 139',xlabel='Sweep');fig.suptitle('Promedus_14: exact-order cost and loopy BP');fig.savefig(ev/'promedus.png',dpi=180);plt.close(fig)
print('Saved tensor diagram and six evidence plots in',ev)
