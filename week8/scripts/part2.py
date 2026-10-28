from pathlib import Path
import json
import numpy as np
import sentencepiece as spm
from scipy.stats import chi2
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent.parent; data=root/'data'; artifacts=root/'artifacts'
sp=spm.SentencePieceProcessor(model_file=str(data/'tok512.model')); V=sp.vocab_size(); assert V==512
stories=[s.strip() for s in (data/'TinyStoriesV2-GPT4-valid.txt').read_text(encoding='utf8').split('<|endoftext|>') if s.strip()]; cut=int(.9*len(stories)); train,valid=stories[:cut],stories[cut:]
def encode(ss):
 a=[]
 for s in ss: a.extend(sp.encode(s)); a.append(sp.eos_id())
 return np.array(a,dtype=np.uint16)
tr=encode(train); va=encode(valid); np.save(artifacts/'train_tokens.npy',tr); np.save(artifacts/'valid_tokens.npy',va)
chars=sum(map(len,valid)); print('stories, tokens, characters',len(train),len(valid),len(tr),len(va),chars,flush=True)
def keys(tokens,n):
 key=tokens[n-1:].astype(np.uint64)
 for i in range(1,n): key |= tokens[n-1-i:len(tokens)-i].astype(np.uint64) << (9*i)
 return key
def lookup(sorted_keys,counts,query):
 ix=np.searchsorted(sorted_keys,query); good=ix<len(sorted_keys); ix=np.minimum(ix,len(sorted_keys)-1); return np.where(good & (sorted_keys[ix]==query),counts[ix],0)
def train_model(tokens,n):
 ng,c=np.unique(keys(tokens,n),return_counts=True)
 if n==1: return ng,c,None,None
 cg,cc=np.unique(keys(tokens[:-1],n-1),return_counts=True); return ng,c,cg,cc
def score(model,tokens,n,k):
 ng,c,cg,cc=model; num=lookup(ng,c,keys(tokens,n)); den=len(tr) if n==1 else lookup(cg,cc,keys(tokens[:-1],n-1)); return -np.log((num+k)/(den+V*k)),float(np.mean(num==0))
ks=[.0001,.001,.01,.1,1.]; results={'stories':[len(train),len(valid)],'tokens':[len(tr),len(va)],'validation_characters':chars,'characters_per_token':chars/len(va),'decode_mismatches':sum(sp.decode(sp.encode(s))!=s for s in valid),'models':{}}
results['models']['uniform']={'nats':float(np.log(V)),'perplexity':V,'bpc':float(np.log(V)*len(va)/chars/np.log(2))}
split=int(len(tr)*.9)
for n in range(1,7):
 print('n',n,'tuning',flush=True); model=train_model(tr[:split],n)
 if n==1:
  ng,c,_,_=model; den=split
  nums=lookup(ng,c,keys(tr[split:],1)); ls=[float(np.mean(-np.log((nums+k)/(den+V*k)))) for k in ks]
 else:
  nums=lookup(model[0],model[1],keys(tr[split:],n)); den=lookup(model[2],model[3],keys(tr[split:-1],n-1)); ls=[float(np.mean(-np.log((nums+k)/(den+V*k)))) for k in ks]
 k=ks[int(np.argmin(ls))]; del model,nums,den
 print('n',n,'k',k,'full train',flush=True); model=train_model(tr,n)
 losses,unseen=score(model,va,n,k); loss=float(losses.mean()); bpc=loss*len(va)/chars/np.log(2); results['models'][str(n)]={'k':k,'tuning_losses':ls,'nats':loss,'perplexity':float(np.exp(loss)),'bpc':float(bpc),'unseen':unseen}; print(results['models'][str(n)],flush=True)
 if n==2:
  counts=np.zeros((V,V)); keys2,cc=model[:2]; counts[(keys2>>9).astype(int),(keys2%512).astype(int)]=cc; P=(counts+k)/(counts.sum(axis=1,keepdims=True)+V*k); p0=np.full(V,1/V)
  # Explicit enumeration of all length-1, -2 and -3 paths in chunks.
  sums=[p0.sum(),sum((p0[i]*P[i]).sum() for i in range(V)),sum((p0[i]*P[i,:,None]*P).sum() for i in range(V))]; results['normalization_sums']=[float(z) for z in sums]
  rng=np.random.default_rng(1848); cdf=np.cumsum(P,axis=1); stream=np.empty(250001,dtype=int); stream[0]=sp.eos_id()
  for i in range(250000): stream[i+1]=np.searchsorted(cdf[stream[i]],rng.random())
  top=np.argsort(np.bincount(tr,minlength=V))[-20:]; stat=0.; df=0; min_expected=float('inf'); context_runs=[]
  for a in top:
   observed=np.bincount(stream[1:][stream[:-1]==a],minlength=V); expected=P[a]*observed.sum(); large=expected>=5; obs=np.r_[observed[large],observed[~large].sum()].astype(float); exp=np.r_[expected[large],expected[~large].sum()];
   if 0<exp[-1]<5:
    idx=int(np.argmin(exp[:-1]));obs[idx]+=obs[-1];exp[idx]+=exp[-1];obs=obs[:-1];exp=exp[:-1]
   good=exp>0;assert np.all(exp[good]>=5);min_expected=min(min_expected,float(exp[good].min()));stat+=np.sum((obs[good]-exp[good])**2/exp[good]);df+=good.sum()-1;context_runs.append({'token':int(a),'observed_counts':observed.tolist(),'total':int(observed.sum())})
  results['bigram_sampling']={'draws':250000,'seed':1848,'chi2':float(stat),'df':int(df),'p':float(chi2.sf(stat,df)),'minimum_pooled_expected':min_expected,'contexts':context_runs}; np.save(artifacts/'bigram_probabilities.npy',P)
 del model
(artifacts/'part2.json').write_text(json.dumps(results,indent=2),encoding='utf8')
fig,ax=plt.subplots(1,2,figsize=(10,4)); ns=np.arange(1,7); ax[0].plot(ns,[results['models'][str(n)]['bpc'] for n in ns],'-o');ax[0].set(xlabel='n',ylabel='Bits per character');ax[1].plot(ns,[100*results['models'][str(n)]['unseen'] for n in ns],'-o');ax[1].set(xlabel='n',ylabel='Unseen held-out n-grams (%)');fig.tight_layout();fig.savefig(root/'evidence'/'part2_ngram.png',dpi=160)
print('Results and context counts saved to artifacts/part2.json',flush=True)
