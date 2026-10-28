from pathlib import Path
import json,time
import numpy as np
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent.parent; rng=np.random.default_rng(1848); sizes=2**np.arange(3,17); names=['outer','least_squares','relu','relu2','relu3']; result={}; trials={}; started=time.time()
for d in [16,32,64]:
 result[str(d)]={}; trials[str(d)]={}
 for M in sizes:
  draws=[]
  for trial in range(3):
   K=rng.normal(size=(M,d)).astype(np.float32);K/=np.linalg.norm(K,axis=1,keepdims=True); V=rng.normal(size=(M,d)).astype(np.float32);V/=np.linalg.norm(V,axis=1,keepdims=True)
   cross=K.T@V; inv=np.linalg.pinv(K.astype(np.float64)); accum={n:0. for n in names}
   for start in range(0,M,256):
    stop=min(start+256,M); q=K[start:stop]; vals=V[start:stop]
    out=q@cross; accum['outer']+=float(np.linalg.norm(out-vals)**2)
    weights=q.astype(np.float64)@inv; diag=weights[np.arange(stop-start),np.arange(start,stop)]; out=weights@V; accum['least_squares']+=float(np.sum(((out-diag[:,None]*vals)/diag[:,None])**2))
    sims=q@K.T; w=np.maximum(sims,0)
    for name in ['relu','relu2','relu3']:
     out=w@V; accum[name]+=float(np.linalg.norm(out-vals)**2);w=w*np.maximum(sims,0)
   draws.append({name:np.sqrt(accum[name]/M) for name in names})
  trials[str(d)][str(int(M))]=[{name:float(draw[name]) for name in names} for draw in draws]
  result[str(d)][str(int(M))]={n:float(np.median([q[n] for q in draws])) for n in names};print('d',d,'M',M,'seconds',round(time.time()-started,1),flush=True)
  (root/'artifacts'/'part3_memory_trials.json').write_text(json.dumps({'seed':1848,'metric':'root mean square of per-key interference/signal ratios','draws':trials},indent=2),encoding='utf8')
  (root/'artifacts'/'part3_memory_partial.json').write_text(json.dumps(result,indent=2),encoding='utf8')
capacities={n:[] for n in names}
for d in [16,32,64]:
 for name in names:
  vals=np.array([result[str(d)][str(int(m))][name] for m in sizes]); i=np.where(vals>=1)[0]
  if not len(i): capacities[name].append(None); continue
  j=i[0]; z=np.interp(0.,np.log(vals[j-1:j+1]),np.log(sizes[j-1:j+1]));capacities[name].append(float(np.exp(z)))
slopes={name:float(np.polyfit(np.log([16,32,64]),np.log(v),1)[0]) if None not in v else None for name,v in capacities.items()}
(root/'artifacts'/'part3_memory.json').write_text(json.dumps({'seed':1848,'ratios':result,'capacities':capacities,'slopes':slopes,'seconds':time.time()-started},indent=2),encoding='utf8')
import runpy
runpy.run_path(str(root/'scripts'/'part3_plot.py'),run_name='__main__')
