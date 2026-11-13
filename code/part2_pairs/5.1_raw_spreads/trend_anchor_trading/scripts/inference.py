"""IPW paired year-block inference and fixed grouped shrinkage."""
from aio import *
import numpy as np,pandas as pd

def holm(p):
    p=np.asarray(p,float);order=np.argsort(p,kind='stable');out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def shrink(group,pool,visits):
    visits=np.asarray(visits,float);n0=float(np.median(visits));w=np.divide(visits,visits+n0,out=np.zeros_like(visits),where=visits+n0>0)
    return w[:,:,None]*group+(1-w[:,:,None])*pool[None,:,:],w,n0

def blocks(labels,draws=5000):
    names=sorted(set(labels));random=np.random.default_rng(20261026)
    ix=random.integers(len(names),size=(draws,len(names))) if names else np.zeros((draws,0),int)
    return names,np.array([np.bincount(row,minlength=len(names)) for row in ix],float)

def estimate(frame,column='value',scheme='year',draws=5000):
    f=frame[np.isfinite(frame[column])].copy();label='test_year' if scheme=='year' else 'cluster'
    names,W=blocks(f[label],draws);observed=float(np.average(f[column],weights=f.weight)) if len(f) else np.nan
    sums=np.array([np.sum(f.loc[f[label]==k,column]*f.loc[f[label]==k,'weight']) for k in names]);den=np.array([f.loc[f[label]==k,'weight'].sum() for k in names])
    num=W@sums;den=W@den;b=np.divide(num,den,out=np.full(draws,np.nan),where=den>0);finite=np.isfinite(b)
    p=float((1+np.sum(~finite)+np.sum(b[finite]-observed>=observed))/(draws+1)) if np.isfinite(observed) else 1.
    result=dict(estimate=observed,ci95_low=float(np.quantile(b[finite],.025)) if finite.any() and len(names)>=2 else np.nan,ci95_high=float(np.quantile(b[finite],.975)) if finite.any() and len(names)>=2 else np.nan,p_one_sided=p if len(names)>=2 else 1.,clusters=len(names),pair_folds=len(f),undefined_draws=int((~finite).sum()),scheme=scheme)
    count('bootstrap_draws',draws);return result

def paired(a,b):
    keys=['pair','test_year'];left=a[keys+['net_bps','weight']];right=b[keys+['net_bps']]
    out=left.merge(right,on=keys,validate='one_to_one',suffixes=('_a','_b'));assert len(out)==len(left),('Incomplete paired comparator',len(out),len(left))
    out['value']=out.net_bps_a-out.net_bps_b;return out
