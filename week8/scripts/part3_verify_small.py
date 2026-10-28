from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parent.parent; rng=np.random.default_rng(4818); results={}
for d in [16,32,64]:
 results[str(d)]={}
 for M in 2**np.arange(3,int(np.log2(d))+1):
  draws=[]
  for _ in range(3):
   K=rng.normal(size=(M,d));K/=np.linalg.norm(K,axis=1,keepdims=True);V=rng.normal(size=(M,d));V/=np.linalg.norm(V,axis=1,keepdims=True);P=K@np.linalg.pinv(K);diag=np.diag(P);leak=(P@V-diag[:,None]*V)/diag[:,None];draws.append(float(np.sqrt(np.mean(np.sum(leak**2,axis=1)))))
  results[str(d)][str(int(M))]={'draws':draws,'median':float(np.median(draws))}
max_error=max(v['median'] for entries in results.values() for v in entries.values());out={'seed':4818,'dimensions':[16,32,64],'draws':results,'max_median_error':max_error};(root/'artifacts'/'part3_small_validation.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('Max small least-squares error:',max_error);assert max_error<1e-10
