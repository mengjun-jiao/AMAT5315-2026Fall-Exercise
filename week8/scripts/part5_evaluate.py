from pathlib import Path
import json,pickle
import numpy as np
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax,jax.numpy as jnp
from scipy.stats import chi2
import sentencepiece as spm
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent.parent
with open(root/'artifacts'/'part5_weights.pkl','rb') as f: p=pickle.load(f)
p=jax.tree.map(jnp.asarray,p); tr=np.load(root/'artifacts'/'train_tokens.npy'); result=json.loads((root/'artifacts'/'part5_training.json').read_text()); counts=json.loads((root/'artifacts'/'part2.json').read_text()); sp=spm.SentencePieceProcessor(model_file=str(root/'data'/'tok512.model'))
def norm(x,p): return (x-jnp.mean(x,axis=-1,keepdims=True))*jax.lax.rsqrt(jnp.var(x,axis=-1,keepdims=True)+1e-5)*p['s']+p['b']
def dense(x,p): return x@p['w']+p['b']
@jax.jit
def forward(p,t):
 B,T=t.shape;x=p['tok'][t]+p['pos'][None,:T,:];mask=jnp.tril(jnp.ones((T,T),dtype=bool))
 for b in p['blocks']:
  y=norm(x,b['ln1']);q,k,v=jnp.split(dense(y,b['qkv']),3,axis=-1);q,k,v=[z.reshape(B,T,4,24).transpose(0,2,1,3) for z in (q,k,v)];att=jax.nn.softmax(jnp.where(mask[None,None,:,:],q@k.transpose(0,1,3,2)/jnp.sqrt(24.),-1e30),axis=-1); y=(att@v).transpose(0,2,1,3).reshape(B,T,96);x=x+dense(y,b['proj']);x=x+dense(jax.nn.gelu(dense(norm(x,b['ln2']),b['fc'])),b['out'])
 return dense(norm(x,p['ln']),p['readout'])
def logits_for(seq):
 a=np.zeros((len(seq),64),dtype=np.int32); length=min(seq.shape[1],64);a[:,:length]=seq[:,-length:];return np.asarray(forward(p,jnp.asarray(a)))[:,length-1,:]
def distribution(logits,temp=1.,kind=None,threshold=None,exclude_eos=False):
 z=np.asarray(logits,dtype=np.float64)/temp
 if exclude_eos: z[...,sp.eos_id()]=-np.inf
 z-=z.max(axis=-1,keepdims=True);prob=np.exp(z);prob/=prob.sum(axis=-1,keepdims=True)
 if kind=='topk':
  order=np.argsort(-prob,axis=-1); mask=np.zeros_like(prob,bool); np.put_along_axis(mask,order[...,:threshold],True,axis=-1);prob=np.where(mask,prob,0)
 if kind=='topp':
  order=np.argsort(-prob,axis=-1); sortedp=np.take_along_axis(prob,order,axis=-1); keep=np.cumsum(sortedp,axis=-1)-sortedp<threshold;mask=np.zeros_like(prob,bool);np.put_along_axis(mask,order,keep,axis=-1);prob=np.where(mask,prob,0)
 if kind=='minp': prob=np.where(prob>=prob.max(axis=-1,keepdims=True)*threshold,prob,0)
 return prob/prob.sum(axis=-1,keepdims=True)
def draw_probabilities(probabilities, rng):
 # The single drawing implementation used both here and for generated stories.
 cdf=np.cumsum(probabilities,axis=-1)
 return np.minimum((cdf<rng.random((len(probabilities),1))).sum(axis=1),511)
def reference_distribution(logits,temp,kind,threshold):
 # Independently construct the target, retaining the item that crosses top-p.
 logits=np.asarray(logits,dtype=np.float64)/temp
 probs=np.exp(logits-logits.max()); probs/=probs.sum()
 order=np.argsort(-probs)
 if kind=='topk': selected=order[:threshold]
 elif kind=='topp':
  count=np.searchsorted(np.cumsum(probs[order]),threshold,side='left')+1
  selected=order[:count]
 elif kind=='minp': selected=np.flatnonzero(probs>=threshold*probs.max())
 else: selected=np.arange(len(probs))
 result=np.zeros_like(probs); result[selected]=probs[selected]; return result/result.sum()
def sample_many(probabilities,rng,count):
 draws=[]
 for start in range(0,count,2000):
  draws.append(draw_probabilities(np.broadcast_to(probabilities,(min(2000,count-start),len(probabilities))),rng))
 return np.concatenate(draws)
def chisquare_pooled(observed,expected):
 large=expected>=5
 o=list(observed[large].astype(float));e=list(expected[large])
 low_obs=float(observed[~large].sum());low_exp=float(expected[~large].sum())
 if low_exp>0:
  if low_exp<5:
   idx=int(np.argmin(e));o[idx]+=low_obs;e[idx]+=low_exp
  else: o.append(low_obs);e.append(low_exp)
 o=np.array(o);e=np.array(e); assert np.all(e>=5)
 stat=float(np.sum((o-e)**2/e));df=len(e)-1
 return stat,df,float(chi2.sf(stat,df)),float(e.min())
def generate(batch,n,temp=1.,kind=None,threshold=None,seed=1848,exclude_eos=False,measure=False):
 rng=np.random.default_rng(seed);seq=np.full((batch,1),sp.eos_id(),dtype=np.int32);ds=[]
 for i in range(n):
  logits=logits_for(seq);base=distribution(logits);q=distribution(logits,temp,kind,threshold,exclude_eos); draws=draw_probabilities(q,rng); entropy=-np.sum(base*np.log(np.maximum(base,1e-300)),axis=1); surprise=-np.log(np.maximum(base[np.arange(batch),draws],1e-300));ds.append(surprise-entropy);seq=np.concatenate([seq,draws[:,None]],axis=1)
 return seq[:,1:],np.asarray(ds).mean(axis=0)
# Sampling goodness of fit, exact probabilities and pooled small expected counts.
valid=np.load(root/'artifacts'/'valid_tokens.npy'); contexts=np.stack([valid[i:i+32] for i in range(0,128*128,128)]).astype(np.int32); candidates=logits_for(contexts); probs=distribution(candidates); entropy=-np.sum(probs*np.log(np.maximum(probs,1e-300)),axis=1); ordered=np.sort(candidates,axis=1); gap=ordered[:,-1]-ordered[:,-2]; chosen=int(np.argmax(np.where(gap>.005,entropy,-np.inf))); assert gap[chosen]>.005; context=contexts[chosen]; z=candidates[chosen]; print('chosen context',sp.decode(context.tolist()),'entropy',float(entropy[chosen]),flush=True); settings=[('temperature',.8,None,None),('top-k',.8,'topk',20),('top-p',.8,'topp',.9),('min-p',.7,'minp',.1)]; tests={};rng=np.random.default_rng(1848)
for name,temp,kind,threshold in settings:
 q=distribution(z,temp,kind,threshold); expected_q=reference_distribution(z,temp,kind,threshold); np.testing.assert_allclose(q,expected_q,rtol=1e-12,atol=1e-15); draw=sample_many(q,rng,100000);obs=np.bincount(draw,minlength=512);stat,df,pvalue,min_exp=chisquare_pooled(obs,expected_q*100000);tests[name]={'chi2':stat,'df':df,'p':pvalue,'min_expected':min_exp,'observed':obs.tolist(),'expected_probabilities':expected_q.tolist()};print(name,{k:tests[name][k] for k in ['chi2','df','p','min_expected']},flush=True)
q=distribution(z,.0001); tests['greedy_probability']=float(q[np.argmax(z)]); tests['greedy_draws_match']=bool(np.all(sample_many(q,rng,100000)==np.argmax(z)))
samples={};sample_token_ids={}
for name,temp,kind,threshold in settings:
 seq,_=generate(1,60,temp,kind,threshold,exclude_eos=True);sample_token_ids[name]=seq[0].tolist();samples[name]=sp.decode(sample_token_ids[name])
self_consistency_runs={}
for temp in [1.,.7]:
 seq,diff=generate(128,128,temp,seed=1848);self_consistency_runs['tokens_'+str(temp)]=seq;self_consistency_runs['difference_'+str(temp)]=diff;tests['self_consistency_'+str(temp)]={'mean_difference_nats':float(diff.mean()),'z':float(diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))))};print('consistency',temp,tests['self_consistency_'+str(temp)],flush=True)
np.savez_compressed(root/'artifacts'/'part5_self_consistency_runs.npz',**self_consistency_runs,context_logits=z)
# Generate a consistent comparison from the count models (sample same seed, no EOS).
uni=np.bincount(tr,minlength=512).astype(float)+counts['models']['1']['k'];uni[sp.eos_id()]=0;uni/=uni.sum(); bi=np.load(root/'artifacts'/'bigram_probabilities.npy'); tri=(tr[:-2].astype(np.uint64)<<18)|(tr[1:-1].astype(np.uint64)<<9)|tr[2:].astype(np.uint64); tri_keys,tri_counts=np.unique(tri,return_counts=True)
def count_sample(n):
 rng=np.random.default_rng(1848); seq=[sp.eos_id()]
 for i in range(60):
  if n==0: prob=np.ones(512)
  elif n==1: prob=uni.copy()
  elif n==2 or len(seq)<2: prob=bi[seq[-1]].copy()
  else:
   base=(int(seq[-2])<<18)|(int(seq[-1])<<9);a=np.searchsorted(tri_keys,base);b=np.searchsorted(tri_keys,base+512);prob=np.full(512,counts['models']['3']['k']);prob[(tri_keys[a:b]&511).astype(int)]+=tri_counts[a:b]
  prob[sp.eos_id()]=0;prob/=prob.sum();seq.append(int(rng.choice(512,p=prob)))
 return seq[1:]
for n,name in enumerate(['uniform','unigram','bigram','trigram']):
 sample_token_ids[name]=count_sample(n); samples[name]=sp.decode(sample_token_ids[name])
sample_token_ids['transformer']=generate(1,60,1.,exclude_eos=True)[0][0].tolist();samples['transformer']=sp.decode(sample_token_ids['transformer'])
(root/'artifacts'/'part5_sampling.json').write_text(json.dumps({'seed':1848,'context_tokens':context.tolist(),'context_text':sp.decode(context.tolist()),'context_entropy':float(entropy[chosen]),'tests':tests,'samples':samples,'sample_token_ids':sample_token_ids},indent=2,ensure_ascii=False),encoding='utf8')
fig,ax=plt.subplots(figsize=(8,4));names=['uniform','unigram','bigram','trigram','4-gram','transformer'];bpc=[counts['models'][k]['bpc'] for k in ['uniform','1','2','3','4']]+[result['bpc']];bars=ax.bar(names,bpc,color=['#758693']*5+['#256ca9']);ax.bar_label(bars,fmt='%.3f',padding=2,fontsize=8);ax.set_ylim(0,max(bpc)*1.12);ax.set_ylabel('Held-out bits / character');fig.tight_layout();fig.savefig(root/'evidence'/'part5_comparison.png',dpi=160)
fig,ax=plt.subplots(figsize=(8,4));x=np.arange(1,65);y=np.array(result['nats_by_position']);se=np.array(result['standard_errors']);ax.plot(x,y,label='transformer');ax.fill_between(x,y-2*se,y+2*se,alpha=.2);ax.axhline(counts['models']['2']['nats'],ls='--',label='bigram');ax.axhline(counts['models']['3']['nats'],ls='--',label='trigram');ax.set(xlabel='Tokens read',ylabel='Nats / token');ax.legend();fig.tight_layout();fig.savefig(root/'evidence'/'part5_context.png',dpi=160)
print('Sampling tests and generated samples saved to artifacts/part5_sampling.json',flush=True)
