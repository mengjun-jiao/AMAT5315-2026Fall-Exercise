from pathlib import Path
import pickle,os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
import jax.numpy as jnp
import jax
import sentencepiece as spm
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent.parent
with open(root/'artifacts'/'part5_weights.pkl','rb') as f: p=jax.tree.map(jnp.asarray,pickle.load(f))
sp=spm.SentencePieceProcessor(model_file=str(root/'data'/'tok512.model')); text=(root/'data'/'TinyStoriesV2-GPT4-valid.txt').read_text(encoding='utf8'); stories=[s.strip() for s in text.split('<|endoftext|>') if s.strip()]; story=stories[int(.9*len(stories))]; ids=sp.encode(story)[:32]; t=jnp.asarray(ids)[None,:]; T=len(ids);x=p['tok'][t]+p['pos'][None,:T,:];mask=jnp.tril(jnp.ones((T,T),dtype=bool));maps=[]
def norm(x,p): return (x-jnp.mean(x,axis=-1,keepdims=True))*jax.lax.rsqrt(jnp.var(x,axis=-1,keepdims=True)+1e-5)*p['s']+p['b']
def dense(x,p): return x@p['w']+p['b']
for b in p['blocks']:
 y=norm(x,b['ln1']);q,k,v=jnp.split(dense(y,b['qkv']),3,axis=-1);q,k,v=[z.reshape(1,T,4,24).transpose(0,2,1,3) for z in (q,k,v)];a=jax.nn.softmax(jnp.where(mask[None,None,:,:],q@k.transpose(0,1,3,2)/jnp.sqrt(24.),-1e30),axis=-1);maps.append(np.asarray(a)[0]); x=x+dense((a@v).transpose(0,2,1,3).reshape(1,T,96),b['proj']);x=x+dense(jax.nn.gelu(dense(norm(x,b['ln2']),b['fc'])),b['out'])
labels=[sp.id_to_piece(i).replace('▁','·') for i in ids]; fig,axs=plt.subplots(2,4,figsize=(20,11))
for layer in range(2):
 for h in range(4):
  ax=axs[layer,h];ax.imshow(maps[layer][h],vmin=0,vmax=1,cmap='Blues',interpolation='nearest');ax.set_title(f'Layer {layer+1}, head {h+1}');ax.set_xticks(range(T),labels,rotation=90,fontsize=5);ax.set_yticks(range(T),labels,fontsize=5);ax.set_xlabel('Key token');ax.set_ylabel('Query token')
fig.suptitle('Attention on the opening of the first held-out story');fig.tight_layout();fig.savefig(root/'evidence'/'part5_attention.png',dpi=180);np.savez(root/'artifacts'/'part5_attention.npz',weights=np.stack(maps),tokens=np.asarray(ids));print('saved',len(ids),'tokens; maximum row-sum error',max(np.max(np.abs(a.sum(-1)-1)) for a in maps))
