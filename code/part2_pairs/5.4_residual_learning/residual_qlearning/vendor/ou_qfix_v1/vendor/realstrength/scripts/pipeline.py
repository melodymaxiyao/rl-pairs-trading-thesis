"""Causal60-session imported fits; one observation/filter shared by all learners."""
from common import *
from residual_models import ou_fit,sign_series
from numpy.lib.stride_tricks import sliding_window_view
from source_stats import ic_records

def features(paths,block=64,metrics=False):
 paths=np.asarray(paths,float);assert paths.ndim==2 and paths.shape[1]>=62
 returns=np.diff(paths,axis=1);T=returns.shape[1]-60;n=len(paths);s=np.empty((n,T+1));half=s.copy();sigma=s.copy() if metrics else None;r2=s.copy() if metrics else None
 for start in range(0,n,block):
  fit=ou_fit(sliding_window_view(returns[start:start+block],60,axis=1));s[start:start+block]=fit['s'];half[start:start+block]=fit['half_life']
  if metrics:sigma[start:start+block]=fit['sigma_epsilon'];r2[start:start+block]=fit['increment_predictive_r2']
 eligible=np.isfinite(s)&np.isfinite(half)&(half>0)&(half<30)
 out=dict(s=s,half=half,valid=eligible,z=np.where(eligible,s,np.inf),returns=returns[:,60:],levels=paths[:,60:])
 if metrics:out.update(sigma=sigma,r2=r2)
 return out

def calibration_panel(paths,year,identity='calibration',seed=20261031):
 """One500-series panel-year; the original N2 code flips daily increments."""
 n=paths.shape[0];T=paths.shape[1]-61;raw=np.diff(paths,axis=1);dates=(np.arange(raw.shape[1])+year*1000).astype('datetime64[D]');worlds=[paths]
 for rep in [0,1]:
  signs=np.stack([sign_series(seed,f'{identity}:{i}',rep,dates) for i in range(n)]);e=raw*signs;worlds.append(np.column_stack([np.zeros(n),np.cumsum(e,axis=1)]))
 fs=[features(x,metrics=True) for x in worlds];scores=np.stack([f['s'][:,:T] for f in fs]);halves=np.stack([f['half'][:,:T] for f in fs]);future=np.full((3,n,T,2),np.nan)
 for wi,f in enumerate(fs):
  e=f['returns']
  for hi,h in enumerate([1,5]):
   sums=np.cumsum(e,axis=1);sums=np.column_stack([np.zeros(n),sums]);future[wi,:,:T-h+1,hi]=sums[:,h:]-sums[:,:-h]
 rows=[]
 for t in range(T):rows+=ic_records('synthetic',f'panel{year}_day{t}',scores[:,:,t],halves[:,:,t],np.full((3,n),np.nan),future[:,:,t,:],np.ones(n))
 # The real calibration masks by reversion, not by the trading half-life<30 gate.
 fields={}
 for name,key in [('sigma','sigma'),('r2','r2'),('half','half')]:
  fields[name]=np.stack([np.where(np.isfinite(f['half'][:,:T]),f[key][:,:T],np.nan) for f in fs])
 ic=np.array([[[row['ic'] for row in rows if row['world']==world and row['replicate']==rep and row['h']==h] for h in [1,5]] for world,rep in [('real',0),('N2',0),('N2',1)]])
 return dict(fields=fields,ic=ic,days=T,series=n,year=year)

def pooled_panel_statistics(panels):
 import warnings
 names=['sigma','r2','half'];result={};stats=[]
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  for name in names:
   values=np.concatenate([p['fields'][name] for p in panels],axis=2);med=np.nanmedian(values,axis=2)
   result[name]=dict(real=float(np.nanmedian(med[0])),N2=float(np.nanmedian(med[1:].ravel())),replica_medians=[float(np.nanmedian(a)) for a in med])
 ic=np.concatenate([p['ic'] for p in panels],axis=2);ic_means=np.nanmean(ic,axis=2)
 result['IC_h1']=dict(real=float(ic_means[0,0]),N2=float(ic_means[1:,0].mean()));result['IC_h5']=dict(real=float(ic_means[0,1]),N2=float(ic_means[1:,1].mean()))
 result['excess']={f'IC_excess_h{h}':result[f'IC_h{h}']['real']-result[f'IC_h{h}']['N2'] for h in [1,5]};result['excess']['increment_R2_excess']=result['r2']['real']-result['r2']['N2'];return result

# Exact specialization for synthetic equal-weight cross-sections. The imported
# IC and Spearman bodies stay unchanged; arbitrary weights use the original.
import source_stats as _stats
_sealed_weighted_ranks=_stats.weighted_ranks

def uniform_ranks(x,w):
 x=np.asarray(x,float);w=np.asarray(w,float)
 if len(x) and np.isfinite(x).all() and np.all(w==1):return (_stats.rankdata(x,method='average')-.5)/len(x)
 return _sealed_weighted_ranks(x,w)
_stats.weighted_ranks=uniform_ranks
