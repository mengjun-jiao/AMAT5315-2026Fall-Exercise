from pathlib import Path
import json
import numpy as np
import sentencepiece as spm
import matplotlib.pyplot as plt
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax
root=Path(__file__).resolve().parent.parent; data=np.load(root/'artifacts'/'part3_mlp_weights.npz'); sp=spm.SentencePieceProcessor(model_file=str(root/'data'/'tok512.model')); hidden=np.tanh(data['emb']@data['w1']+data['b1']); W=data['w2']; sides=[]
for j in range(512):
 for sign in [1,-1]:
  activ=np.maximum(sign*hidden[:,j],0); top=np.argsort(-activ)[:5]; output=np.argsort(-sign*W[j])[:5]; share=float(np.sum(activ[top]**2)/max(np.sum(activ**2),1e-30));sides.append({'unit':j,'side':sign,'previous':[sp.id_to_piece(int(i)) for i in top],'next':[sp.id_to_piece(int(i)) for i in output],'top5_activation_share':share})
keys=jax.random.split(jax.random.key(1848),4); emb=np.asarray(jax.random.normal(keys[0],(512,64)))*np.sqrt(2/(512+64)); w1=np.asarray(jax.random.normal(keys[1],(64,512)))*np.sqrt(2/(64+512)); initial=np.tanh(emb@w1); initial_shares=[]
for j in range(512):
 for sign in [1,-1]:
  a=np.maximum(sign*initial[:,j],0); top=np.argsort(-a)[:5]; initial_shares.append(float(np.sum(a[top]**2)/max(np.sum(a**2),1e-30)))
(root/'artifacts'/'part3_unit_analysis.json').write_text(json.dumps({'median_share':float(np.median([x['top5_activation_share'] for x in sides])),'initial_median_share':float(np.median(initial_shares)),'sides':sides},indent=2,ensure_ascii=False),encoding='utf8')
fig,ax=plt.subplots(figsize=(8,4));ax.hist(initial_shares,bins=35,alpha=.5,label='untrained');ax.hist([x['top5_activation_share'] for x in sides],bins=35,alpha=.6,label='trained');ax.legend();ax.set(xlabel='Share of squared activation in top 5 input tokens (each signed side)',ylabel='Unit sides');fig.tight_layout();fig.savefig(root/'evidence'/'part3_unit_concentration.png',dpi=160);print('trained median',np.median([x['top5_activation_share'] for x in sides]),'untrained median',np.median(initial_shares))
