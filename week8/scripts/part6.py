from pathlib import Path
import csv,json
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent.parent
rows=list(csv.DictReader(open(root/'data'/'chinchilla-runs.csv',encoding='utf8')))
N=np.array([float(r['Model Size']) for r in rows]); C=np.array([float(r['Training FLOP']) for r in rows]); L=np.array([float(r['loss']) for r in rows]); keep=np.argsort(L)[:-5]; N,C,L=N[keep],C[keep],L[keep]; D=C/(6*N)
x=np.log(N); y=np.log(D); target=np.log(L); delta=.001
# q=(log E, log A, log B, alpha, beta); analytic gradients.
def fun(q,ids):
 e,a,b=np.exp(q[:3]); alpha,beta=q[3:]; u=a*np.exp(-alpha*x[ids]); v=b*np.exp(-beta*y[ids]); pred=e+u+v; r=np.log(pred)-target[ids]; h=np.where(abs(r)<=delta,.5*r*r,delta*(abs(r)-delta/2)); g=np.clip(r,-delta,delta)/pred; deriv=np.stack([np.full(len(ids),e),u,v,-u*x[ids],-v*y[ids]],axis=1); return h.sum(),(g[:,None]*deriv).sum(axis=0)
ids=np.arange(len(N)); bounds=[(-3,3),(-5,20),(-5,20),(.05,1),(.05,1)]
fits=[]
for a in [.25,.35,.45]:
 for b in [.25,.35,.45]:
  q=np.array([np.log(1.8),np.log(500),np.log(2000),a,b]); res=minimize(fun,q,args=(ids,),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':4000,'ftol':1e-15,'gtol':1e-11,'maxls':40}); fits.append(res)
best=min(fits,key=lambda z:z.fun); q=best.x; print('best',best.fun,q,flush=True)
rng=np.random.default_rng(2026); boots=[]; successes=[]
for i in range(400):
 sample=rng.integers(0,len(N),len(N)); z=minimize(fun,q,args=(sample,),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':5000,'ftol':1e-15,'gtol':1e-10,'maxls':60}); boots.append(z.x); successes.append(bool(z.success))
 if i%50==0: print('bootstrap',i,flush=True)
boots=np.array(boots)
def params(z): return np.r_[np.exp(z[:3]),z[3:]]
def predict(z,n,d):
 e,a,b,alpha,beta=params(z); return e+a/n**alpha+b/d**beta
def optimal(z,c):
 e,a,b,alpha,beta=params(z); g=(alpha*a/(beta*b))**(1/(alpha+beta)); return g*(c/6)**(beta/(alpha+beta)),(c/6)**(alpha/(alpha+beta))/g
printed=np.r_[np.log([1.69,406.4,410.7]),.34,.28]; pred=predict(q,N,D); pp=predict(printed,N,D); ratio=np.array([optimal(z,5.76e23)[1]/optimal(z,5.76e23)[0] for z in boots]); ns,ds=optimal(q,5.76e23)
result={'parameters':dict(zip(['E','A','B','alpha','beta'],params(q).tolist())),'bootstrap_se':dict(zip(['E','A','B','alpha','beta'],np.std(np.array([params(z) for z in boots]),axis=0,ddof=1).tolist())),'ratio':float(ds/ns),'ratio_95_interval':np.quantile(ratio,[.025,.975]).tolist(),'mean_log_residual_fit':float(np.mean(np.log(pred)-target)),'mean_log_residual_printed':float(np.mean(np.log(pp)-target)),'runs':len(N),'bootstrap_replicates':400,'seed':2026,'objective_reduction':'sum','optimizer_ftol':1e-15,'optimizer_gtol':1e-10,'finite_bootstrap':int(sum(np.isfinite(boots[:,0]))),'optimizer_successes':sum(successes)}
result['optimal_by_compute']={str(10**i):{'N':float(optimal(q,10**i)[0]),'D':float(optimal(q,10**i)[1])} for i in range(18,26)}
(root/'artifacts'/'part6.json').write_text(json.dumps(result,indent=2),encoding='utf8'); np.save(root/'artifacts'/'part6_bootstrap.npy',boots); print(json.dumps(result,indent=2),flush=True)
fig,ax=plt.subplots(1,2,figsize=(13,5)); ax[0].scatter(N,L,color='lightgray',s=10,alpha=.5,label='all fitted runs'); nsweep=np.logspace(7,11,300)
for c,color in [(1e19,'#1f77b4'),(1e20,'#ff7f0e'),(1e21,'#2ca02c')]:
 close=np.abs(C/c-1)<=.15; ax[0].scatter(N[close],L[close],s=16,c=color,alpha=.8); ax[0].plot(nsweep,predict(q,nsweep,c/(6*nsweep)),color=color,label=f'{c:.0e} FLOP ±15%')
ax[0].set_xscale('log');ax[0].set_ylim(1.8,5.3);ax[0].set_xlabel('Parameters');ax[0].set_ylabel('Loss (nats/token)');ax[0].legend(); sc=ax[1].scatter(pred,L,c=np.log10(C),s=13,alpha=.6); fig.colorbar(sc,ax=ax[1],label='log10 training FLOP'); ax[1].plot([2,5.5],[2,5.5],color='k');ax[1].set_xlabel('Predicted loss');ax[1].set_ylabel('Measured loss');fig.tight_layout();fig.savefig(root/'evidence'/'part6_fit.png',dpi=160)
fig,ax=plt.subplots(figsize=(8,5));ax.scatter(N,np.log(pred)-target,s=12,alpha=.6,label='refit');ax.scatter(N,np.log(pp)-target,s=12,alpha=.5,label='printed');ax.axhline(0,color='k',lw=.7);ax.set_xscale('log');ax.set_xlabel('Parameters');ax.set_ylabel('Predicted − measured log loss');ax.legend();fig.tight_layout();fig.savefig(root/'evidence'/'part6_residuals.png',dpi=160)
fig,ax=plt.subplots(figsize=(8,5)); cs=np.logspace(18,25,100)
for z,style,name in [(q,'-','refit'),(printed,'--','printed')]:
 n,d=optimal(z,cs);ax.loglog(cs,n,style,label=name+' parameters');ax.loglog(cs,d,style,label=name+' tokens')
ax.scatter([5.76e23,5.76e23],[7e10,1.4e12],marker='*',s=130,c='k');ax.set_xlabel('Compute (FLOP)');ax.set_ylabel('Optimal parameters / tokens');ax.legend();fig.tight_layout();fig.savefig(root/'evidence'/'part6_optimal.png',dpi=160)
