"""Fold-conditioned ranks, FDR and year-block ratio inference."""
from common import *
import numpy as np,pandas as pd
from scipy.stats import rankdata

def empirical(values,reference,tail='lower'):
 reference=np.sort(np.asarray(reference,float));reference=reference[np.isfinite(reference)];x=np.asarray(values,float)
 if not len(reference):return np.full(len(x),np.nan)
 n=np.searchsorted(reference,x,side='right') if tail=='lower' else len(reference)-np.searchsorted(reference,x,side='left')
 return np.where(np.isfinite(x),(1+n)/(len(reference)+1),np.nan)
def bh(p):
 p=np.asarray(p,float);q=np.full(len(p),np.nan);ix=np.flatnonzero(np.isfinite(p));order=ix[np.argsort(p[ix],kind='stable')]
 if len(order):q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
 return q
def pi0(p,weights=None):
 p=np.asarray(p,float);w=np.ones(len(p)) if weights is None else np.asarray(weights,float);ok=np.isfinite(p)&np.isfinite(w)&(w>0)
 return min(1.,float(np.sum(w[ok]*(p[ok]>.5))/(.5*w[ok].sum()))) if ok.any() else np.nan

def calibrate(frame):
 f=frame.copy();f['empirical_p']=np.nan;f['empirical_q']=np.nan
 for (win,stat),g in f.groupby(['window','statistic_name'],sort=True):
  tail='upper' if stat.startswith('theta') or stat=='Johansen' else 'lower'
  for (world,rep),v in g.groupby(['world','replicate'],sort=True):
   ref=g[g.world.eq('N2') & (~g.replicate.eq(rep) if world=='N2' else True)]
   p=empirical(v.statistic.to_numpy(),ref.statistic.to_numpy(),tail)
   f.loc[v.index,'empirical_p']=p;f.loc[v.index,'empirical_q']=bh(p)
 return f

def blocks(labels,draws=5000):return _blocks(tuple(sorted(set(labels))),draws)
@lru_cache(maxsize=128)
def _blocks(labels,draws):
 names=list(labels);random=np.random.default_rng(SEED)
 ix=random.integers(len(names),size=(draws,len(names))) if names else np.zeros((draws,0),int)
 return names,np.array([np.bincount(row,minlength=len(names)) for row in ix],float)
def describe(obs,b,n_years,n_pairs):
 b=np.asarray(b,float);ok=np.isfinite(b);finite=b[ok]
 return dict(estimate=float(obs),ci95_low=float(np.quantile(finite,.025)) if ok.any() and n_years>=2 else np.nan,ci95_high=float(np.quantile(finite,.975)) if ok.any() and n_years>=2 else np.nan,
  p_one_sided=float((1+sum(~ok)+sum(finite-obs>=obs))/(len(b)+1)) if np.isfinite(obs) and n_years>=2 else 1.,clusters=n_years,pair_folds=n_pairs,undefined_draws=int(sum(~ok)))
def mean(frame,value='value',draws=5000):
 f=frame[np.isfinite(frame[value])].copy();names,W=blocks(f.test_year,draws);count('bootstrap_draws',draws)
 if not len(f):return describe(np.nan,np.full(draws,np.nan),0,0)
 num=np.array([sum(g[value]*g.weight) for k in names for g in [f[f.test_year.eq(k)]]]);den=np.array([f.loc[f.test_year.eq(k),'weight'].sum() for k in names]);b=np.divide(W@num,W@den,out=np.full(draws,np.nan),where=W@den>0)
 return describe(np.average(f[value],weights=f.weight),b,len(names),len(f))
def contrast(frame,terms,draws=5000):
 """terms=[(coefficient, boolean-mask, value-column)] with ratios refit in draws."""
 names,W=blocks(frame.test_year,draws);observed=0.;b=np.zeros(draws);count('bootstrap_draws',draws)
 for coefficient,mask,column in terms:
  f=frame[mask & np.isfinite(frame[column])];den=np.array([f.loc[f.test_year.eq(k),'weight'].sum() for k in names]);num=np.array([(f.loc[f.test_year.eq(k),column]*f.loc[f.test_year.eq(k),'weight']).sum() for k in names])
  observed+=coefficient*(num.sum()/den.sum() if den.sum() else np.nan)
  b+=coefficient*np.divide(W@num,W@den,out=np.full(draws,np.nan),where=W@den>0)
 return describe(observed,b,len(names),len(frame))
def holm(p):
 p=np.asarray(p,float);order=np.argsort(p,kind='stable');out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
 return out

def select_screen(frame,screen,fallback_years=None):
 name={'RC':'ADF_ct','RL':'ADF_c','RP':'theta_21'}[screen];f=frame[frame.statistic_name.eq(name)].copy()
 f['BH_pass']=f.empirical_q.le(.10);f['pass']=f.BH_pass;f['screen']=screen;f['fallback']=False
 mode=spec()['decisions']['fallback']
 if mode=='none':return f
 for (world,rep,window),g in f.groupby(['world','replicate','window']):
  if window!='T':continue
  # Fallback is an explicitly frozen decision, never based on outcomes.
  sparse=g.groupby('test_year').BH_pass.mean().lt(.01)
  active=(sparse.sum()>len(sparse)/2) if mode=='global_posthoc' else None
  for year,v in g.groupby('test_year'):
   use=((world,int(rep),int(year)) in fallback_years) if fallback_years is not None else (active if active is not None else bool(sparse.loc[year]))
   if not use:continue
   z=v.sort_values(['statistic','pair'],ascending=[screen!='RP',True],kind='stable');take=int(np.ceil(.2*len(z)))
   f.loc[v.index,'pass']=False;f.loc[z.index[:take],'pass']=True;f.loc[v.index,'fallback']=True
 return f
