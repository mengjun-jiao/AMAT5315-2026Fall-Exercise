from pathlib import Path
import itertools,json,numpy as np
root=Path(__file__).resolve().parent.parent
words=iter((root/'asia.uai').read_text().split());assert next(words)=='MARKOV';n=int(next(words));dims=[int(next(words)) for _ in range(n)];assert n==8 and dims==[2]*8
nf=int(next(words));scopes=[[int(next(words)) for _ in range(int(next(words)))] for _ in range(nf)];tables=[]
for scope in scopes:
 count=int(next(words));values=np.array([float(next(words)) for _ in range(count)]);tables.append(values.reshape((2,)*len(scope)))
assert scopes[1]==[1,0] and np.array_equal(tables[1],np.array([[.05,.01],[.95,.99]]))
print('T,A factor:',tables[1].tolist())
def enumerate_case(obs):
 z=0.;m=np.zeros((8,2));j=np.zeros((2,2))
 for bits in itertools.product(range(2),repeat=8):
  if any(bits[v]!=state for v,state in obs.items()):continue
  w=np.prod([table[tuple(bits[v] for v in scope)] for scope,table in zip(scopes,tables)]);z+=w
  for i,b in enumerate(bits):m[i,b]+=w
  j[bits[1],bits[3]]+=w
 return z,m/z,j/z
summary={}
for name,obs in [('prior',{}),('AD',{0:0,7:0}),('ADX',{0:0,7:0,6:0})]:
 run=json.loads((root/'artifacts'/name/'run.json').read_text());z,m,j=enumerate_case(obs);got=np.array([run['marginals'][str(i)] for i in range(8)]);err=float(np.max(abs(got-m)));je=float(np.max(abs(np.array(run['joint_TL'])-j)));rel=float(abs(z-run['evidence_mass'])/z);assert max(err,je,rel)<1e-10;summary[name]={'evidence':z,'marginal_abs_error':err,'joint_abs_error':je,'evidence_rel_error':rel}; print(name,summary[name])
 joint=np.array(run['joint_TL']);assert np.max(abs(got.sum(axis=1)-1))<1e-12 and abs(joint.sum()-1)<1e-12
 assert np.max(abs(joint.sum(axis=1)-got[1]))<1e-12 and np.max(abs(joint.sum(axis=0)-got[3]))<1e-12
assert abs(summary['AD']['evidence']-.004501375)<1e-14 and abs(summary['ADX']['evidence']-.00098822675)<1e-14
assert abs(enumerate_case({0:0,7:0,6:0})[2][0,0]-.0222135253877716)<1e-14
A=np.array([[1.,2.],[3.,4.]]);B=np.array([[2.,0.],[1.,2.]]);numpy_ab=A@B;numpy_trace=float(np.trace(numpy_ab));rust=json.loads((root/'artifacts'/'AD'/'run.json').read_text())['self_checks'];rust_ab=np.array(rust['matrix_col_major']).reshape((2,2),order='F');matrix_err=float(np.max(abs(rust_ab-numpy_ab)));trace_err=abs(rust['trace']-numpy_trace);assert matrix_err<1e-12 and trace_err<1e-12
spins=np.array([-1.,1.]);K=np.exp(.5*np.outer(spins,spins));z=np.einsum('ij,jk,ki->',K,K,K);direct=sum(np.exp(.5*(a*b+b*c+c*a)) for a,b,c in itertools.product(spins,repeat=3));ising_err=abs(rust['ising_contraction']-direct);assert abs(z-direct)<1e-12 and ising_err<1e-12
print('NumPy AB:',numpy_ab.tolist(),'tenferro AB:',rust_ab.tolist(),'absolute max error:',matrix_err)
print('NumPy trace:',numpy_trace,'tenferro trace:',rust['trace'],'absolute error:',trace_err)
print('Enumerated Ising Z:',direct,'tenferro Ising Z:',rust['ising_contraction'],'absolute error:',ising_err)
for name in ['random','greedy','analysis','replay']:
 plan=json.loads((root/'runs'/name/'order.json').read_text());live={i:set(s) for i,s in enumerate(plan['input_scopes'])};work=0;largest=max([2**len(s) for s in live.values()]+[1])
 for st in plan['steps']:
  inputs=[live.pop(i) for i in st['inputs']];union=set().union(*inputs);outside=set(plan['query']).union(*live.values());keep=union&outside;work+=2**len(union);largest=max(largest,2**len(keep));assert keep==set(st['retained_labels']);assert 2**len(union)==st['work'];live[st['output']]=keep
 assert len(live)==1 and work==plan['total_work'] and largest==plan['largest_tensor'];print(name,'independent work',work,'largest',largest)
assert json.loads((root/'runs/random/order.json').read_text())['total_work']==json.loads((root/'runs/analysis/order.json').read_text())['total_work']
for name in ['random','greedy','replay']:
 r=json.loads((root/'runs'/name/'run.json').read_text());assert abs(r['evidence_mass']-.004501375)/.004501375<1e-10
# Recompute each undamped proposal and belief from the saved final messages.
bp_checks={}
for name,obs in [('prior',{}),('AD',{0:0,7:0}),('ADX',{0:0,7:0,6:0})]:
 bp=json.loads((root/'artifacts'/('bp-'+name)/'run.json').read_text());factors=[];adj=[[] for _ in range(n)]
 for fi,(scope,table) in enumerate(zip(scopes,tables)):
  t=table.copy()
  for pos,v in enumerate(scope):
   adj[v].append((fi,pos))
   if v in obs:
    idx=[slice(None)]*len(scope);idx[pos]=1-obs[v];t[tuple(idx)]=0
  factors.append(t)
 v2f=bp['variable_to_factor'];f2v=bp['factor_to_variable'];residual=0.;beliefs=[]
 for v,edges in enumerate(adj):
  b=np.ones(2)
  for fi,pos in edges:b*=f2v[fi][pos]
  b/=b.sum();beliefs.append(b)
  for fi,pos in edges:
   proposal=np.ones(2)
   for other,op in edges:
    if other!=fi:proposal*=f2v[other][op]
   proposal/=proposal.sum();residual=max(residual,float(np.max(abs(proposal-np.array(v2f[fi][pos])))))
 for fi,(scope,t) in enumerate(zip(scopes,factors)):
  for pos,v in enumerate(scope):
   proposal=np.zeros(2)
   for index in np.ndindex(t.shape):
    weight=t[index]
    for other in range(len(scope)):
     if other!=pos:weight*=v2f[fi][other][index[other]]
    proposal[index[pos]]+=weight
   proposal/=proposal.sum();residual=max(residual,float(np.max(abs(proposal-np.array(f2v[fi][pos])))))
 assert abs(residual-bp['residual'])<1e-14 and np.max(abs(np.array(beliefs)-np.array(bp['marginals'])))<1e-14
 _,exact,_=enumerate_case(obs);err=float(np.max(abs(np.array(beliefs)-exact)))
 assert bp['converged'] and residual<=1e-10
 bp_checks[name]={'sweeps':bp['sweeps'],'residual':residual,'max_abs_error_against_independent_enumeration':err}
 print('BP independently recomputed:',name,bp_checks[name])
(root/'artifacts'/'independent_checks.json').write_text(json.dumps({'cases':summary,'bp_checks':bp_checks,'matrix_max_error_against_numpy':matrix_err,'trace_abs_error_against_numpy':trace_err,'ising_abs_error_against_enumeration':ising_err,'plans':'independently recounted and replayed'},indent=2))
print('All independent ASIA checks passed')
