"""IPW Spearman, daily HAC and unchanged sealed year-block inference."""
from common import *
import types,importlib.util
from scipy.stats import rankdata

def weighted_quantile(x,w,q):
 x=np.asarray(x,float);w=np.asarray(w,float);ok=np.isfinite(x)&np.isfinite(w)&(w>0);x=x[ok];w=w[ok]
 if not len(x):return np.nan
 order=np.argsort(x,kind='stable');x=x[order];w=w[order];return float(x[min(np.searchsorted(w.cumsum(),q*w.sum(),side='left'),len(x)-1)])
def weighted_ranks(x,w):
 x=np.asarray(x,float);w=np.asarray(w,float);order=np.argsort(x,kind='stable');a=x[order];v=w[order];start=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1];stop=np.r_[start[1:],len(a)];group=np.add.reduceat(v,start);mid=group.cumsum()-group/2;out=np.empty(len(x))
 for lo,hi,value in zip(start,stop,mid):out[order[lo:hi]]=value/w.sum()
 return out
def spearman(x,y,w=None):
 x=np.asarray(x,float);y=np.asarray(y,float);w=np.ones(len(x)) if w is None else np.asarray(w,float);ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0);x=x[ok];y=y[ok];w=w[ok]
 if len(x)<3:return np.nan
 rx=weighted_ranks(x,w);ry=weighted_ranks(y,w);rx-=np.average(rx,weights=w);ry-=np.average(ry,weights=w);den=np.sqrt(np.sum(w*rx**2)*np.sum(w*ry**2));return float(np.sum(w*rx*ry)/den) if den>0 else np.nan

def nw_mean(values,lag):
 a=np.asarray(values,float);n=int(np.isfinite(a).sum())
 if n<2:return dict(mean=np.nan,se=np.nan,n=n)
 mean=np.nanmean(a);u=np.where(np.isfinite(a),a-mean,0.);meat=u@u
 for k in range(1,min(lag,len(a)-1)+1):meat+=2*(1-k/(lag+1))*(u[k:]@u[:-k])
 return dict(mean=float(mean),se=float(np.sqrt(max(0,meat))/n),n=n)

def pooled_regression(x,y,w=None):
 x=np.asarray(x,float);y=np.asarray(y,float);w=np.ones_like(x) if w is None else np.asarray(w,float);ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0);x=x[ok];y=y[ok];w=w[ok]
 if len(x)<3:return dict(n=len(x),slope=np.nan,intercept=np.nan)
 xm=np.average(x,weights=w);ym=np.average(y,weights=w);den=np.sum(w*(x-xm)**2);slope=np.sum(w*(x-xm)*(y-ym))/den if den>0 else np.nan
 return dict(n=len(x),slope=float(slope),intercept=float(ym-slope*xm))

def sealed_module(rel,alias,aliases):
 registration=json.loads((RUN/'audit/input_registration.json').read_text())['source_snapshots'];r=registration[rel];p=RUN/'vendor'/rel;assert sha(p)==r['sha256'] and p.stat().st_size==r['size'];module=types.ModuleType(alias);module.__file__=str(p)
 old={k:sys.modules.get(k) for k in aliases}
 try:
  sys.modules.update(aliases);exec(compile(p.read_bytes(),str(p),'exec'),module.__dict__)
 finally:
  for k,v in old.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v
 return module

def bootstrap_module():
 if not hasattr(bootstrap_module,'value'):
  shim=types.ModuleType('aio');shim.count=count
  bootstrap_module.value=sealed_module('experiments/15_pair_trading/trend_anchor_trading_v1/scripts/inference.py','_residual_sealed_bootstrap',{'aio':shim})
 return bootstrap_module.value

def block_estimate(values,dates,draws=None):
 import pandas as pd
 values=np.asarray(values,float);dates=np.asarray(dates,dtype='datetime64[D]');years=dates.astype('datetime64[Y]').astype(int)+1970
 f=pd.DataFrame(dict(value=values,test_year=years,weight=np.ones(len(values))))
 return bootstrap_module().estimate(f,draws=spec()['bootstrap_draws'] if draws is None else draws)

def market_hac(y,spy,lag=5):
 from collections import Counter
 shim=types.ModuleType('common');shim.COUNTS=Counter();module=sealed_module('experiments/15_pair_trading/thesis_tables_v1/scripts/methods.py','_residual_sealed_market',{'common':shim})
 return module.ols_hac(y,spy,lag,252)
