"""Exact diffuse/profile likelihood, deterministic multi-start; no data access."""
import ctypes as C,math,pathlib
import numpy as np
from scipy.optimize import minimize
ROOT=pathlib.Path(__file__).resolve().parents[1];DP=C.POINTER(C.c_double);_LIB=None
STARTS=[(.1,1),(.5,.01),(.9,0),(.98,1),(.5,99),(.999,100),(0,1)]
def pointer(x):return x.ctypes.data_as(DP)
def filter_values(x,rho,q,states=False,steady=False,burn=1,chunk_size=75):
 x=np.atleast_2d(x)
 if chunk_size and len(x)>chunk_size:
  parts=[filter_values(x[i:i+chunk_size],rho,q,states,steady,burn,0) for i in range(0,len(x),chunk_size)]
  return np.concatenate([v[0] for v in parts]),np.concatenate([v[1] for v in parts]) if states else None
 return _filter_values_unchunked(x,rho,q,states,steady,burn)
def _filter_values_unchunked(x,rho,q,states=False,steady=False,burn=1):
 global _LIB
 x=np.ascontiguousarray(np.atleast_2d(x),dtype=float);assert 0<=rho<=.999 and q>=0
 if _LIB is None:
  _LIB=C.CDLL(str(ROOT/'runtime/kalman.so'));_LIB.pci_filter.argtypes=[C.c_int,C.c_int,DP,C.c_double,C.c_double,DP,DP,C.c_int,C.c_int,C.c_int];_LIB.pci_filter.restype=None
 out=np.empty((len(x),3));v=np.empty((*x.shape,2)) if states else np.empty(1)
 time_major=np.ascontiguousarray(x.T);_LIB.pci_filter(len(x),x.shape[1],pointer(time_major),rho,q,pointer(out),pointer(v),int(states),int(steady),int(burn));return out,v if states else None
def profile(x,rho,q,states=False,steady=False,burn=1):
 out,v=filter_values(x,rho,q,states,steady=steady,burn=burn);ss,ld,n=out.T
 with np.errstate(divide='ignore',invalid='ignore'):scale=ss/n;ll=-n/2*(np.log(2*np.pi)+1+np.log(scale))-ld/2
 return dict(ll=ll,scale=scale,n=n,ss=ss,logdet=ld,states=v)
def full_loglike(x,rho,q,sigma2,burn=1):
 o,_=filter_values(x,rho,q,burn=burn);ss,ld,n=o.T;sigma2=np.asarray(sigma2)
 return -.5*(n*np.log(2*np.pi*sigma2)+ld+ss/sigma2)
def rw_profile(x):
 x=np.asarray(x,float);v=np.flatnonzero(np.isfinite(x));n=len(v)-1
 if n<1:return dict(ll=-np.inf,scale=np.nan,n=n)
 gaps=np.diff(v);ss=np.sum(np.diff(x[v])**2/gaps);scale=ss/n
 ll=-n/2*(np.log(2*np.pi)+1+np.log(scale))-.5*np.log(gaps).sum() if scale>0 else -np.inf
 return dict(ll=float(ll),scale=float(scale),n=n)
def ratio(rho,q):return (2/(1+rho))/(2/(1+rho)+q)
def half(rho):return 0. if rho==0 else float(np.log(.5)/np.log(rho))
def encode(rho,q):
 p=np.clip(rho/.999,1e-8,1-1e-8);return np.array([np.log(p/(1-p)),np.log1p(q)])
def decode(z):return float(.999/(1+np.exp(-z[0]))),float(np.expm1(z[1]))
def fit(x,pooled=True,checkpoint=lambda:None,warm_start=None,burn=1,warm_starts=()):
 x=np.ascontiguousarray(np.atleast_2d(x),float);counts=np.isfinite(x).sum(axis=1);assert (counts>=3).all()
 qmax=np.log1p(100) if pooled else 700.;bounds=[(-25,25),(0,qmax)];runs=[];candidates=[]
 def objective(z,face=None):
  checkpoint()
  rho,q=decode(z)
  if face:
   if face[0]=='rho':rho=face[1]
   else:q=face[1]
  value=profile(x,rho,q,steady=True,burn=burn);ok=np.isfinite(value['ll'])&(value['scale']>0)
  return -float(value['ll'].sum())/float(value['n'].sum()) if ok.all() else 1e100
 def solve(start,face=None):
  checkpoint();result=minimize(lambda z:objective(z,face),encode(*start),method='L-BFGS-B',bounds=bounds,options=dict(maxiter=200,ftol=1e-11,gtol=1e-7,maxls=30))
  rho,q=decode(result.x)
  if face:
   if face[0]=='rho':rho=face[1]
   else:q=face[1]
  value=profile(x,rho,q,burn=burn);finite=np.isfinite(value['ll']).all() and (value['scale']>0).all();runs.append(dict(start=list(start),face=face,success=bool(result.success),status=int(result.status),message=str(result.message),iterations=int(result.nit),rho=rho,q=q,ll=float(value['ll'].sum()) if finite else None))
  if result.success and finite:candidates.append((float(value['ll'].sum()),rho,q,value))
 for start in STARTS:solve(start)
 if warm_start is not None and warm_start.get('converged',True):solve((float(warm_start['rho']),float(warm_start['q'])))
 for warm in warm_starts:
  if warm and warm.get('converged',True):solve((float(warm['rho']),float(warm['q'])))
 # Exact closed faces are required: logit alone cannot attain a boundary.
 best=max(candidates,key=lambda v:v[0]) if candidates else None
 reference=(best[1],best[2]) if best else (.9,1.)
 for face in [('rho',0.),('rho',.999),('q',0.)]+([('q',100.)] if pooled else []):solve(reference,face)
 if not candidates:return dict(converged=False,runs=runs,reason='No finite converged multi-start optimum')
 ll,rho,q,value=max(candidates,key=lambda v:v[0]);fast=profile(x,rho,q,steady=True,burn=burn);error=float(np.max(np.abs(fast['ll']-value['ll'])));assert error<=1e-6,('Steady/full likelihood disagreement',error)
 result=dict(converged=True,rho=rho,q=q,R2_MR=ratio(rho,q),half_life=half(rho),ll=ll,scale=value['scale'],n=value['n'],runs=runs,steady_full_max_LL_error=error,likelihood_burn=burn,boundary=bool(rho<1e-6 or rho>.999-1e-6 or q<1e-6 or pooled and q>100-1e-5),RW_boundary=False)
 if not pooled:
  assert len(x)==1;null=rw_profile(x[0]);result.update(null_ll=null['ll'],LR=max(0.,2*(ll-null['ll'])))
  if null['ll']>=ll:result.update(RW_boundary=True,boundary=True,LR=0.,R2_MR=0.)
 return result
