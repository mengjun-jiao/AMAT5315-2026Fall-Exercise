from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parent.parent; rng=np.random.default_rng(4819); sizes=2**np.arange(3,10);runs={}
for d in [16,32,64]:
 runs[str(d)]={}
 for M in sizes:
  draws=[]
  for _ in range(3):
   K=rng.normal(size=(M,d));K/=np.linalg.norm(K,axis=1,keepdims=True);V=rng.normal(size=(M,d));V/=np.linalg.norm(V,axis=1,keepdims=True);S=np.maximum(K@K.T,0);np.fill_diagonal(S,0);leak=S@V;draws.append(float(np.sqrt(np.mean(np.sum(leak**2,axis=1)))))
  runs[str(d)][str(int(M))]={'draws':draws,'median':float(np.median(draws))}
caps=[]
for d in [16,32,64]:
 vals=np.array([runs[str(d)][str(int(m))]['median'] for m in sizes]);j=np.flatnonzero(vals>=1)[0]; caps.append(float(np.exp(np.interp(0,np.log(vals[j-1:j+1]),np.log(sizes[j-1:j+1])))))
slope=float(np.polyfit(np.log([16,32,64]),np.log(caps),1)[0]);out={'seed':4819,'runs':runs,'capacities':caps,'slope':slope};(root/'artifacts'/'part3_relu_repeat.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('independent repeat capacities',caps,'slope',slope)
