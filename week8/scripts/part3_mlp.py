from pathlib import Path
import json,time
import numpy as np
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax,jax.numpy as jnp
import optax
root=Path(__file__).resolve().parent.parent; tr=np.load(root/'artifacts'/'train_tokens.npy'); va=np.load(root/'artifacts'/'valid_tokens.npy'); rng=np.random.default_rng(1848)
key=jax.random.key(1848); keys=jax.random.split(key,4)
def weight(k,a,b): return jax.random.normal(k,(a,b))*np.sqrt(2/(a+b))
p={'emb':weight(keys[0],512,64),'w1':weight(keys[1],64,512),'b1':jnp.zeros(512),'w2':weight(keys[2],512,512),'b2':jnp.zeros(512)}
schedule=optax.warmup_cosine_decay_schedule(0.,.006,200,4000,end_value=.0006); opt=optax.adamw(schedule,weight_decay=.01); state=opt.init(p)
@jax.jit
def update(p,state,counts):
 def loss(p):
  logits=jnp.tanh(p['emb']@p['w1']+p['b1'])@p['w2']+p['b2']; return -(counts*jax.nn.log_softmax(logits)).sum()/counts.sum()
 l,g=jax.value_and_grad(loss)(p); changes,state=opt.update(g,state,p);return optax.apply_updates(p,changes),state,l
started=time.time()
for step in range(4000):
 ix=rng.integers(0,len(tr)-1,size=16384); counts=np.bincount(tr[ix].astype(np.int32)*512+tr[ix+1],minlength=512*512).reshape(512,512).astype(np.float32)
 p,state,loss=update(p,state,jnp.asarray(counts))
 if step%100==0: print('step',step,'loss',float(loss),'seconds',round(time.time()-started,1),flush=True)
logits=jnp.tanh(p['emb']@p['w1']+p['b1'])@p['w2']+p['b2']; logp=np.asarray(jax.nn.log_softmax(logits)); loss=float(-logp[va[:-1],va[1:]].mean()); print('validation nats',loss,flush=True)
np.savez(root/'artifacts'/'part3_mlp_weights.npz',**{k:np.asarray(v) for k,v in p.items()}); (root/'artifacts'/'part3_mlp.json').write_text(json.dumps({'heldout_nats':loss,'steps':4000,'batch_examples':16384,'lr':.006,'warmup':200,'end_lr':.0006,'weight_decay':.01,'seed':1848,'seconds':time.time()-started},indent=2),encoding='utf8')
