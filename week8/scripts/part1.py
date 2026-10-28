from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from safetensors.numpy import load_file
from tokenizers import Tokenizer
root = Path(__file__).resolve().parent.parent
out = root / 'evidence'; out.mkdir(exist_ok=True)
data = root / 'data'
tables = load_file(str(data / 'model.safetensors'))
print('table keys:', [(k, v.shape) for k, v in tables.items()])
E = next(v for v in tables.values() if v.ndim == 2)
E = E / np.linalg.norm(E, axis=1, keepdims=True)
control = E[np.random.default_rng(1730).permutation(len(E))]
tok = Tokenizer.from_file(str(data/'tokenizer.json'))
relations = {
 'country to capital': 'iran tehran, turkey ankara, russia moscow, china beijing, france paris, sweden stockholm, japan tokyo, peru lima, kenya nairobi',
 'male to female': 'king queen, man woman, he she, brother sister, father mother, son daughter, uncle aunt, boy girl, husband wife',
 'adjective to comparative': 'big bigger, small smaller, fast faster, dark darker, cold colder, old older, strong stronger, long longer, tall taller'}
results={}; fig,axs=plt.subplots(1,3,figsize=(14,5))
for ax,(label,s) in zip(axs,relations.items()):
    pairs=[x.strip().split() for x in s.split(',')]
    ids=[]
    for a,b in pairs:
        aa,bb=tok.encode(a,add_special_tokens=False).ids,tok.encode(b,add_special_tokens=False).ids
        assert len(aa)==len(bb)==1,(a,b,aa,bb)
        ids.append((aa[0],bb[0]))
    vals=[]
    for mat in [E,control]:
        offsets=np.array([mat[j]-mat[i] for i,j in ids]); other=(offsets.sum(axis=0)-offsets)/8
        vals.append(np.sum(offsets*other,axis=1)/np.linalg.norm(offsets,axis=1)/np.linalg.norm(other,axis=1))
    results[label]={'pairs':pairs,'table':vals[0].tolist(),'shuffled':vals[1].tolist(),'means':[float(x.mean()) for x in vals]}
    for v,sign,col in [(vals[0],1,'#2878b5'),(vals[1],-1,'#999999')]:
        for c in v: ax.arrow(0,0,float(c),float(sign*np.sqrt(max(0,1-c*c))),width=.006,head_width=.04,length_includes_head=True,color=col,alpha=.65)
    ax.axhline(0,color='black',lw=.6); ax.axvline(0,color='black',ls=':',lw=.7); ax.set_xlim(-.5,1); ax.set_ylim(-1.1,1.1); ax.set_aspect('equal'); ax.set_title(label+'\nmeans %.3f / %.3f'%tuple(results[label]['means']))
fig.tight_layout(); fig.savefig(out/'part1_relations.png',dpi=170)
(root/'artifacts'/'part1.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({k:v['means'] for k,v in results.items()},indent=2))
