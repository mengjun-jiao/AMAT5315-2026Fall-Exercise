from pathlib import Path
import json,time
import numpy as np
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax,jax.numpy as jnp
import optax
root=Path(__file__).resolve().parent.parent; tr=np.load(root/'artifacts'/'train_tokens.npy'); va=np.load(root/'artifacts'/'valid_tokens.npy'); rng=np.random.default_rng(1848)
key=jax.random.key(1848)
def init(shape):
 global key
 key,k=jax.random.split(key); return jax.random.normal(k,shape)*.02
def linear(a,b): return {'w':init((a,b)),'b':jnp.zeros((b,))}
def ln(): return {'s':jnp.ones(96),'b':jnp.zeros(96)}
p={'tok':init((512,96)),'pos':init((64,96)),'blocks':[{'ln1':ln(),'qkv':linear(96,288),'proj':linear(96,96),'ln2':ln(),'fc':linear(96,384),'out':linear(384,96)} for _ in range(2)],'ln':ln(),'readout':linear(96,512)}
def norm(x,p): return (x-jnp.mean(x,axis=-1,keepdims=True))*jax.lax.rsqrt(jnp.var(x,axis=-1,keepdims=True)+1e-5)*p['s']+p['b']
def dense(x,p): return x@p['w']+p['b']
def forward(p,t):
 B,T=t.shape; x=p['tok'][t]+p['pos'][None,:T,:]; mask=jnp.tril(jnp.ones((T,T),dtype=bool))
 for block in p['blocks']:
  y=norm(x,block['ln1']); q,k,v=jnp.split(dense(y,block['qkv']),3,axis=-1)
  q,k,v=[z.reshape(B,T,4,24).transpose(0,2,1,3) for z in (q,k,v)]
  att=jax.nn.softmax(jnp.where(mask[None,None,:,:],q@k.transpose(0,1,3,2)/jnp.sqrt(24.),-1e30),axis=-1)
  y=(att@v).transpose(0,2,1,3).reshape(B,T,96);x=x+dense(y,block['proj']); y=norm(x,block['ln2']); x=x+dense(jax.nn.gelu(dense(y,block['fc'])),block['out'])
 return dense(norm(x,p['ln']),p['readout'])
schedule=optax.warmup_cosine_decay_schedule(0.,.003,300,6000,end_value=.0003); opt=optax.adamw(schedule,weight_decay=.01); state=opt.init(p)
@jax.jit
def update(p,state,x,y):
 def loss(p): return optax.softmax_cross_entropy_with_integer_labels(forward(p,x),y).mean()
 l,g=jax.value_and_grad(loss)(p); u,state=opt.update(g,state,p);return optax.apply_updates(p,u),state,l
@jax.jit
def losses(p,x,y): return optax.softmax_cross_entropy_with_integer_labels(forward(p,x),y)
start=time.time(); offsets=np.arange(65)
for step in range(6000):
 idx=rng.integers(0,len(tr)-65,32); seq=tr[idx[:,None]+offsets];p,state,loss=update(p,state,jnp.asarray(seq[:,:-1],dtype=jnp.int32),jnp.asarray(seq[:,1:],dtype=jnp.int32))
 if step%100==0: print('step',step,'loss',float(loss),'seconds',round(time.time()-start,1),flush=True)
np.savez(root/'artifacts'/'part5_weights.npz',*jax.tree.leaves(p)); import pickle
with open(root/'artifacts'/'part5_weights.pkl','wb') as f: pickle.dump(jax.tree.map(np.asarray,p),f)
# Consecutive windows: each position starts with its own available context.
windows=(len(va)-1)//64; arr=va[:windows*64+1]; accum=[]
for i in range(0,windows,32):
 inds=np.arange(i,min(i+32,windows))*64; x=arr[inds[:,None]+np.arange(64)]; y=arr[inds[:,None]+np.arange(1,65)]; accum.append(np.asarray(losses(p,jnp.asarray(x,dtype=jnp.int32),jnp.asarray(y,dtype=jnp.int32))))
allloss=np.concatenate(accum).astype(np.float64); result={'steps':6000,'batch_sequences':32,'learning_rate':.003,'warmup_steps':300,'seed':1848,'seconds':time.time()-start,'mean_nats':float(allloss.mean()),'nats_by_position':allloss.mean(axis=0).tolist(),'standard_errors':(allloss.std(axis=0,ddof=1)/np.sqrt(windows)).tolist(),'windows':windows}
result['bpc']=result['mean_nats']*len(va)/json.loads((root/'artifacts'/'part2.json').read_text())['validation_characters']/np.log(2);(root/'artifacts'/'part5_training.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print('validation',result['mean_nats'],result['bpc'],flush=True)
