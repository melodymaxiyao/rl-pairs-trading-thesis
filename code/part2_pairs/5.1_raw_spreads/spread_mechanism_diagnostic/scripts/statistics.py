"""IPW statistics, exact fold multiplicities, weighted medians and product limit."""
from mio import SEED,count
import numpy as np

def matrix(years,draws=2000,seed=SEED):
    names=sorted(set(int(y) for y in years));assert names
    ix=np.random.default_rng(seed).integers(len(names),size=(draws,len(names)))
    small=np.stack([np.bincount(r,minlength=len(names)) for r in ix]).astype(float)
    result=np.zeros((draws,22))
    for i,y in enumerate(names):result[:,y-2004]=small[:,i]
    count('bootstrap_matrices');count('bootstrap_draws',draws)
    return result

def quantile(a,p):
    a=np.sort(np.asarray(a,float));a=a[~np.isnan(a)]
    if not len(a):return np.nan
    k=(len(a)-1)*p;i=int(np.floor(k));j=int(np.ceil(k))
    if i==j or a[i]==a[j]:return float(a[i])
    if np.isinf(a[i]) or np.isinf(a[j]):
        return float(a[i]) if np.isneginf(a[i]) and not np.isposinf(a[j]) else float(a[j]) if not np.isneginf(a[i]) else np.nan
    return float(a[i]+(k-i)*(a[j]-a[i]))

def summarise(point,draws,n=0,folds=0):
    count('bootstrap_statistics')
    draws=np.asarray(draws,float);unknown=np.isnan(draws)
    lo=quantile(draws,.025);hi=quantile(draws,.975)
    defined=np.isfinite(point) and folds>=2
    with np.errstate(invalid='ignore'):
        positive=(1+int(np.sum(draws-point>=point))+int(unknown.sum()))/(len(draws)+1) if defined else np.nan
        negative=(1+int(np.sum(draws-point<=point))+int(unknown.sum()))/(len(draws)+1) if defined else np.nan
    return dict(estimate=float(point),ci95_low=lo if folds>=2 else np.nan,ci95_high=hi if folds>=2 else np.nan,
                p_positive=positive,p_negative=negative,n=int(n),folds=int(folds),undefined_draws=int(unknown.sum()))

def mean_draws(values,weights,years,M):
    x=np.asarray(values,float);w=np.asarray(weights,float);f=np.asarray(years,int)-2004
    keep=np.isfinite(x)&np.isfinite(w)&(w>0);x=x[keep];w=w[keep];f=f[keep]
    sums=np.bincount(f,weights=x*w,minlength=22);mass=np.bincount(f,weights=w,minlength=22)
    denom=M@mass
    draws=np.divide(M@sums,denom,out=np.full(len(M),np.nan),where=denom>0)
    point=sums.sum()/mass.sum() if mass.sum()>0 else np.nan
    return point,draws,int(keep.sum()),len(np.unique(f))

def median_draws(values,weights,years,M):
    x=np.asarray(values,float);w=np.asarray(weights,float);f=np.asarray(years,int)-2004
    keep=~np.isnan(x)&np.isfinite(w)&(w>0);x=x[keep];w=w[keep];f=f[keep]
    if not len(x):return np.nan,np.full(len(M),np.nan),0,0
    order=np.argsort(x,kind='stable');x=x[order];w=w[order];f=f[order]
    prefix=np.zeros((22,len(x)+1))
    for k in np.unique(f):prefix[k,1:]=np.cumsum(np.where(f==k,w,0.))
    def med(B):
        totals=B@prefix[:,-1];lo=np.zeros(len(B),int);hi=np.full(len(B),len(x)-1,int)
        while (lo<hi).any():
            mid=(lo+hi)//2
            cdf=np.einsum('ij,ji->i',B,prefix[:,mid+1])
            hit=cdf>=totals*.5-1e-12*np.maximum(1.,totals)
            hi=np.where(hit,mid,hi);lo=np.where(hit,lo,mid+1)
        return np.where(totals>0,x[lo],np.nan)
    point=med(np.ones((1,22)))[0]
    return float(point),med(M),len(x),len(np.unique(f))

def ratio_draws(numerator,denominator,M):
    numerator=np.asarray(numerator,float);denominator=np.asarray(denominator,float)
    den=M@denominator
    draws=np.divide(M@numerator,den,out=np.full(len(M),np.nan),where=den>0)
    point=numerator.sum()/denominator.sum() if denominator.sum()>0 else np.nan
    return point,draws

def product_limit(deaths,censors,horizon=63):
    d=np.atleast_2d(np.asarray(deaths,float));c=np.atleast_2d(np.asarray(censors,float))
    assert d.shape==c.shape and (d>=0).all() and (c>=0).all()
    risk=np.cumsum((d+c)[:,::-1],axis=1)[:,::-1]
    hazards=np.divide(d,risk,out=np.zeros_like(d),where=risk>0)
    survival=np.cumprod(1-hazards,axis=1);total=(d+c).sum(axis=1)
    med=np.full(len(d),np.inf)
    for i in range(len(d)):
        hit=np.flatnonzero(survival[i]<=.5+1e-12)
        if len(hit):med[i]=float(hit[0])
    h=min(horizon,d.shape[1]-1)
    identifiable=(risk[:,h]>0)|(survival[:,h]<=1e-12)
    probability=np.where((total>0)&identifiable,1-survival[:,h],np.nan)
    return probability,med

def km_draws(deaths,censors,M,horizon=63):
    point,median=product_limit(deaths.sum(axis=0),censors.sum(axis=0),horizon)
    draws,medians=product_limit(M@deaths,M@censors,horizon)
    return float(point[0]),draws,float(median[0]),medians

def holm(ps):
    ps=np.array([p if np.isfinite(p) else 1. for p in ps],float);order=np.argsort(ps,kind='stable')
    result=np.empty(len(ps));result[order]=np.minimum(1.,np.maximum.accumulate((len(ps)-np.arange(len(ps)))*ps[order]))
    return result
