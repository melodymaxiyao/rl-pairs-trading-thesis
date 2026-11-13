"""Causal rolling residual estimates. All returns and cumulative residuals are in bps."""
import numpy as np
from scipy.linalg import eigh
from scipy.stats import t as student_t
from numpy.lib.stride_tricks import sliding_window_view

def ou_fit(residual):
 e=np.asarray(residual,float);scalar=e.ndim==1
 if scalar:e=e[None,:]
 n=e.shape[-1];x=np.concatenate([np.zeros((*e.shape[:-1],1)),np.cumsum(e,axis=-1)],axis=-1);u=x[...,:-1];v=x[...,1:];du=v-u
 uc=u-u.mean(axis=-1,keepdims=True);vc=v-v.mean(axis=-1,keepdims=True);den=np.sum(uc*uc,axis=-1)
 valid=np.isfinite(e).all(axis=-1)&(den>0)&(n>2)
 with np.errstate(divide='ignore',invalid='ignore'):
  b=np.sum(uc*vc,axis=-1)/den;a=v.mean(axis=-1)-b*u.mean(axis=-1);error=v-a[...,None]-b[...,None]*u;variance=np.sum(error**2,axis=-1)/(n-2);se=np.sqrt(variance/den)
  # Increment regression uses slope b-1, not the much larger level predictive R2.
  delta_total=np.sum((du-du.mean(axis=-1,keepdims=True))**2,axis=-1);r2=1-np.sum(error**2,axis=-1)/delta_total
  reverting=valid&(b>0)&(b<1)&(variance>0);logb=np.log(np.where(reverting,b,np.nan));half=np.log(2)/(-logb);kappa=-252*logb;m=a/(1-b);sigma_eq=np.sqrt(variance/(1-b*b));score=(x[...,-1]-m)/sigma_eq
  p=student_t.cdf((b-1)/se,df=n-2)
 out=dict(b=b,b_se=se,kappa=np.where(reverting,kappa,np.nan),half_life=np.where(reverting,half,np.nan),m=np.where(reverting,m,np.nan),sigma_epsilon=np.sqrt(variance),sigma_eq=np.where(reverting,sigma_eq,np.nan),s=np.where(reverting,score,np.nan),increment_predictive_r2=np.where(valid&(delta_total>0),np.clip(r2,0,1),np.nan),valid=valid,reverting=reverting,no_reversion=valid&(b>=1),nonpositive_b=valid&(b<=0),fast=reverting&(half<30),significantly_below_one=valid&(p<.05),x_last=x[...,-1])
 for key,value in out.items():
  if key not in ['valid','reverting','no_reversion','nonpositive_b','fast','significantly_below_one']:out[key]=np.where(valid,value,np.nan)
 return {k:v[0] if scalar else v for k,v in out.items()}

def regress(y,factors):
 y=np.asarray(y,float);f=np.asarray(factors,float)
 if f.ndim==1:f=f[:,None]
 if y.ndim==1:y=y[:,None]
 X=np.column_stack([np.ones(len(f)),f]);assert np.isfinite(X).all() and np.isfinite(y).all()
 if np.linalg.matrix_rank(X)<X.shape[1]:raise ValueError('Rank-deficient residual regression')
 coef=np.linalg.lstsq(X,y,rcond=None)[0]
 return y-X@coef,coef[0],coef[1:].T

def pair_rolling(a,b,window=60):
 a=np.asarray(a,float);b=np.asarray(b,float);A=sliding_window_view(a,window);B=sliding_window_view(b,window);am=A.mean(axis=1);bm=B.mean(axis=1);ac=A-am[:,None];bc=B-bm[:,None];den=np.sum(bc*bc,axis=1)
 with np.errstate(divide='ignore',invalid='ignore'):beta=np.sum(ac*bc,axis=1)/den;alpha=am-beta*bm
 residual=A-alpha[:,None]-beta[:,None]*B;ok=np.isfinite(residual).all(axis=1)&(den>0);residual[~ok]=np.nan
 return residual,alpha,beta

def eigenportfolio(returns,k=15):
 r=np.asarray(returns,float);assert r.ndim==2 and np.isfinite(r).all();n,p=r.shape;assert 1<=k<min(n,p)
 sd=r.std(axis=0,ddof=1)
 if np.any(sd<=0):raise ValueError('Constant PCA member')
 z=(r-r.mean(axis=0))/sd
 # Dual eigensystem gives the same top correlation eigenvectors when p>n.
 if p>n:
  values,u=eigh(z@z.T/(n-1),subset_by_index=[n-k,n-1],driver='evr');order=np.argsort(values)[::-1];values=values[order];v=z.T@u[:,order]/np.sqrt((n-1)*values)
 else:
  values,v=eigh(z.T@z/(n-1),subset_by_index=[p-k,p-1],driver='evr');order=np.argsort(values)[::-1];values=values[order];v=v[:,order]
 if np.any(values<=np.finfo(float).eps):raise ValueError('Insufficient PCA rank')
 signs=np.sign(v[np.argmax(abs(v),axis=0),np.arange(k)]);v=v*np.where(signs==0,1,signs)
 weights=v/sd[:,None]
 return weights,values,v,sd

def factor_window(history,k=15):
 history=np.asarray(history,float);W,values,V,sd=eigenportfolio(history,k);f=history[-60:]@W;residual,alpha,beta=regress(history[-60:],f)
 hedge=beta@W.T;gross=1+abs(hedge).sum(axis=1)
 return dict(residual=residual,alpha=alpha,beta=beta,weights=W,hedge=hedge,gross=gross,eigenvalues=values)

def sign_series(seed,identity,replicate,dates):
 import hashlib
 key=hashlib.sha256(f'{seed}|{identity}|{replicate}'.encode()).digest();offset=np.uint64(int.from_bytes(key[:8],'little'))
 # Counter-based keyed mixing is invariant to chunk size, workers and resume.
 z=np.asarray(dates,dtype='datetime64[D]').astype('int64').astype('uint64')
 with np.errstate(over='ignore'):
  z=z+offset+np.uint64(0x9e3779b97f4a7c15);z=(z^(z>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);z=(z^(z>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);z=z^(z>>np.uint64(31))
 return np.where(z&np.uint64(1),1.,-1.)

def update_positions(previous,score,half_life,eligible):
 p=np.asarray(previous,np.int8).copy();s=np.asarray(score,float);h=np.asarray(half_life,float);ok=np.asarray(eligible,bool)&np.isfinite(s)&np.isfinite(h)&(h<30)&(h>0)
 p[~ok| (abs(s)<.5)]=0
 flat=(p==0)&ok;p[flat&(s>1.25)]=-1;p[flat&(s< -1.25)]=1
 return p
