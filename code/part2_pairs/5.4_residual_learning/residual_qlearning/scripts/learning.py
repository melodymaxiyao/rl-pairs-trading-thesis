"""Single-series conservative training, independently seeded; test data never enter."""
from common import *
from kernel import fit,policy,cube_values
from scipy.special import ndtr

def holm(p):
 p=np.asarray(p,float);order=np.argsort(p,kind='stable');out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return out

def nw(a,lag=21):
 a=np.asarray(a,float);a=a[np.isfinite(a)];n=len(a)
 if n<2:return dict(mean=float(np.mean(a)) if n else np.nan,se=np.nan,n=n,p=1.)
 u=a-a.mean();meat=u@u
 for k in range(1,min(lag,n-1)+1):meat+=2*(1-k/(lag+1))*(u[k:]@u[:-k])
 se=float(np.sqrt(max(0,meat))/n);mean=float(a.mean());return dict(mean=mean,se=se,n=n,p=float(ndtr(-mean/se)) if se>0 else (0. if mean>0 else 1.))

def train_ensemble(z,cube,kind,year,gate,cost,double):
 fits=[];label='DBL' if double else 'POW'
 for j in range(10):
  checkpoint();stem=f'runtime/checkpoints/{kind}_{year}_{gate}_c{cost}_{label}_seed{j}';v=cached(stem)
  if v is None:
   v=fit(z,cube,td_seed(j),cost,double=double);save(stem,v);print('Saved seed',kind,year,gate,cost,label,j+1,'/10','epochs',v['epochs'],flush=True);count('real_TD_seed_fits');count('real_learning_steps',v['learning_steps'])
  fits.append(v)
 return dict(policy=policy(fits,double=double,lcb=double),fits=[dict(q=f['q'],visits=f['visits'],diagnostics=f['diagnostics'],epochs=f['epochs'],converged=f['converged'],convergence_epoch=f['convergence_epoch'],learning_steps=f['learning_steps'],seed=f['seed']) for f in fits])

def band(z,cube,cost):
 candidates=[]
 for entry in [1.25,1.5,2,2.5]:
  for exit in [.5,0,-.5]:
   p=dict(kind=1,enter=entry,exit=exit);value=float(cube_values(z,cube,p,cost).sum(axis=1).mean());candidates.append(dict(policy=p,training_net_bps=value))
 best=max(range(len(candidates)),key=lambda i:(candidates[i]['training_net_bps'],-i));return dict(**candidates[best]['policy'],training_grid=candidates)

def spi(z,cube,cost,qbest,lengths):
 base=dict(kind=1,enter=1.25,exit=.5);baseline=cube_values(z,cube,base,cost,realized=True);tests=[]
 # Average series within each calendar-year training block, then concatenate
 # daily observations. Padded sessions are removed before HAC.
 for st in range(8):
  override=dict(kind=3,bits=int(qbest['bits'])|(1<<(8+st)),enter=1.25,exit=.5)
  change=cube_values(z,cube,override,cost,realized=True)-baseline;arr=[];offset=0
  for n,L in lengths:arr.extend(change[offset:offset+n,-L:].mean(axis=0));offset+=n
  stats=nw(arr,21);tests.append(dict(state=st,**stats))
 adjusted=holm([t['p'] for t in tests]);allowed=0
 for t,p in zip(tests,adjusted):
  t['holm_p']=float(p);t['accepted']=bool(p<.05 and t['mean']>0)
  if t['accepted']:allowed|=1<<t['state']
 return dict(kind=3,bits=int(qbest['bits'])|(allowed<<8),enter=1.25,exit=.5,tests=tests,accepted_states=allowed,base='AL',candidate='Q_best',training_accounting='PP_FULL')
