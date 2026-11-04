from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parent.parent;data=root/'promedus';art=root/'artifacts';words=iter((data/'Promedus_14.uai').read_text().split());assert next(words)=='MARKOV';n=int(next(words));dims=[int(next(words)) for _ in range(n)];assert n==414 and all(x==2 for x in dims);nf=int(next(words));scopes=[[int(next(words)) for _ in range(int(next(words)))] for _ in range(nf)];tables=[]
for scope in scopes:
 size=int(next(words)); assert size==2**len(scope); tables.append(np.array([float(next(words)) for _ in range(size)]).reshape((2,)*len(scope)))
evs=list(map(int,(data/'Promedus_14.uai.evid').read_text().split()));assert evs[0]==9;obs=dict(zip(evs[1::2],evs[2::2]));assert len(obs)==9
pr=float((data/'Promedus_14.uai.PR').read_text().split()[1]);tokens=iter((data/'Promedus_14.uai.MAR').read_text().split());assert next(tokens)=='MAR' and int(next(tokens))==414;mar=[]
for _ in range(n):
 assert int(next(tokens))==2;mar.append([float(next(tokens)),float(next(tokens))])
ids=[59,102,139];greedy=json.loads((art/'promedus-greedy/run.json').read_text());tree=json.loads((art/'promedus-treesa/run.json').read_text());bp=json.loads((art/'promedus-bp/run.json').read_text());
assert abs(greedy['log10_partition']-tree['log10_partition'])<1e-10
assert abs(greedy['log10_partition']-pr)<1e-4
for v in ids:
 a=np.array(greedy['marginals'][str(v)]);b=np.array(tree['marginals'][str(v)]);c=np.array(mar[v]);assert np.max(abs(a-b))<1e-10 and np.max(abs(a-c))<1e-5 and np.max(abs(b-c))<1e-5;assert abs(a.sum()-1)<1e-12 and abs(b.sum()-1)<1e-12
# Independently recount each saved scalar and marginal contraction plan.
counts={}
for name in ['promedus-greedy','promedus-treesa']:
 for file in ['order.json']+[f'query-{v}-order.json' for v in ids]:
  plan=json.loads((art/name/file).read_text());live={i:set(scope) for i,scope in enumerate(plan['input_scopes'])};largest=max([2**len(x) for x in live.values()]+[1]);generated=1;work=0
  for st in plan['steps']:
   operands=[live.pop(i) for i in st['inputs']];union=set().union(*operands);outside=set(plan['query']).union(*live.values());keep=union&outside;cost=2**len(union);size=2**len(keep);assert keep==set(st['retained_labels']) and cost==st['work'] and size==st['output_entries'];work+=cost;largest=max(largest,size);generated=max(generated,size);live[st['output']]=keep
  assert len(live)==1 and work==plan['total_work'] and largest==plan['largest_tensor'] and generated==plan['largest_generated'];counts[name+'/'+file]={'work':work,'largest':largest,'largest_generated':generated}
# Recompute an undamped BP proposal at the exact saved final message state.
factors=[]
for scope,t in zip(scopes,tables):
 t=t.copy()
 for axis,v in enumerate(scope):
  if v in obs:
   slicer=[slice(None)]*len(scope);slicer[axis]=1-obs[v];t[tuple(slicer)]=0
 factors.append(t)
adj=[[] for _ in range(n)]
for f,scope in enumerate(scopes):
 for pos,v in enumerate(scope):adj[v].append((f,pos))
v2f=bp['variable_to_factor'];f2v=bp['factor_to_variable'];residual=0.; beliefs=[]
for v,edges in enumerate(adj):
 b=np.ones(2)
 for f,pos in edges:b*=f2v[f][pos]
 b/=b.sum();beliefs.append(b)
 for f,pos in edges:
  proposal=np.ones(2)
  for other,op in edges:
   if other!=f:proposal*=f2v[other][op]
  proposal/=proposal.sum();residual=max(residual,float(np.max(abs(proposal-np.array(v2f[f][pos])))))
for fi,(scope,t) in enumerate(zip(scopes,factors)):
 for pos,v in enumerate(scope):
  output=np.zeros(2)
  for index in np.ndindex(t.shape):
   weight=t[index]
   for other in range(len(scope)):
    if other!=pos:weight*=v2f[fi][other][index[other]]
   output[index[pos]]+=weight
  output/=output.sum();residual=max(residual,float(np.max(abs(output-np.array(f2v[fi][pos])))))
assert abs(residual-bp['residual'])<1e-14 and np.max(abs(np.array(beliefs)-np.array(bp['marginals'])))<1e-14
err=max(abs(bp['marginals'][v][s]-greedy['marginals'][str(v)][s]) for v in ids for s in [0,1]);assert bp['sweeps']==1000 and not bp['converged'] and residual>1e-10 and err>.3
report={'published_log10_z':pr,'greedy_log10_z':greedy['log10_partition'],'treesa_log10_z':tree['log10_partition'],'log10_z_abs_difference':abs(greedy['log10_partition']-tree['log10_partition']),'query_ids':ids,'published_marginals':{str(v):mar[v] for v in ids},'greedy_marginals':greedy['marginals'],'treesa_marginals':tree['marginals'],'bp_marginals':{str(v):bp['marginals'][v] for v in ids},'bp_selected_max_abs_error':err,'bp_residual_independently_recomputed':residual,'bp_sweeps':bp['sweeps'],'bp_converged':bp['converged'],'plans_independently_recounted':counts,'runtime_seconds':{'greedy_all_queries':greedy['runtime_seconds'],'treesa_all_queries':tree['runtime_seconds'],'bp_all_variables':bp['seconds']},'bp_update_work':bp['cumulative_work']}
(art/'promedus_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print('Promedus independent checks passed')
