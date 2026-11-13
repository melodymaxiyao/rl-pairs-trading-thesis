"""Descriptive LLT+AR1 exact-diffuse fits and matched-burn model comparisons."""
import ctypes as C,time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from io_utils import *
import pci
DP=C.POINTER(C.c_double);_LIB=None
def profile(x,rho,qe,qz,states=False,steady=False,chunk_size=75):
 x=np.atleast_2d(x)
 if chunk_size and len(x)>chunk_size:
  parts=[profile(x[i:i+chunk_size],rho,qe,qz,states,steady,0) for i in range(0,len(x),chunk_size)]
  return {k:np.concatenate([v[k] for v in parts]) if k!='states' or states else None for k in parts[0]}
 return _profile_unchunked(x,rho,qe,qz,states,steady)
def _profile_unchunked(x,rho,qe,qz,states=False,steady=False):
 global _LIB
 x=np.ascontiguousarray(np.atleast_2d(x),float);n,T=x.shape
 if _LIB is None:
  _LIB=C.CDLL(str(ROOT/'runtime/kalman.so'));_LIB.llt_filter.argtypes=[C.c_int,C.c_int,DP,C.c_double,C.c_double,C.c_double,DP,DP,C.c_int,C.c_int];_LIB.llt_filter.restype=None
 out=np.empty((n,3));v=np.empty((n,T,6)) if states else np.empty(1);xx=np.ascontiguousarray(x.T);p=lambda a:a.ctypes.data_as(DP);_LIB.llt_filter(n,T,p(xx),rho,qe,qz,p(out),p(v),int(states),int(steady));ss,ld,num=out.T
 with np.errstate(invalid='ignore',divide='ignore'):scale=ss/num;ll=-num/2*(np.log(2*np.pi)+1+np.log(scale))-ld/2
 return dict(ll=ll,scale=scale,n=num,ss=ss,logdet=ld,states=v if states else None)
def full_loglike(x,rho,qe,qz,sigma2):
 v=profile(x,rho,qe,qz);return -.5*(v['n']*np.log(2*np.pi*np.asarray(sigma2))+v['logdet']+v['ss']/sigma2)
def fit(x,restricted=False,warm=None,checkpoint=lambda:None):
 x=np.ascontiguousarray(np.atleast_2d(x),float);starts=[(.1,1,1e-5),(.5,.01,.001),(.9,0,0),(.98,1,.01),(.5,99,1),(.999,100,0),(0,1,.1)];runs=[];candidates=[]
 if warm and warm.get('converged'):starts.append((warm['rho'],warm['q_eta'],warm.get('q_zeta',0)))
 def unpack(z):
  rho,qe=pci.decode(z[:2]);return rho,qe,0. if restricted else float(np.expm1(z[2]))
 def objective(z,face=None):
  checkpoint()
  rho,qe,qz=unpack(z)
  if face:
   if face[0]=='rho':rho=face[1]
   elif face[0]=='q_eta':qe=face[1]
   else:qz=face[1]
  v=profile(x,rho,qe,qz,steady=True);return -float(v['ll'].sum())/float(v['n'].sum()) if np.isfinite(v['ll']).all() and (v['scale']>0).all() else 1e100
 def solve(start,face=None):
  checkpoint();z=np.r_[pci.encode(start[0],start[1]),np.log1p(start[2])];bounds=[(-25,25),(0,np.log1p(100)),(0,0 if restricted else 700)]
  f=minimize(lambda z:objective(z,face),z,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=250,ftol=1e-11,gtol=1e-7,maxls=30));rho,qe,qz=unpack(f.x)
  if face:
   if face[0]=='rho':rho=face[1]
   elif face[0]=='q_eta':qe=face[1]
   else:qz=face[1]
  v=profile(x,rho,qe,qz);finite=np.isfinite(v['ll']).all() and (v['scale']>0).all();runs.append(dict(start=start,face=face,success=bool(f.success),iterations=int(f.nit),message=str(f.message),ll=float(v['ll'].sum()) if finite else None))
  if finite and f.success:candidates.append((float(v['ll'].sum()),rho,qe,qz,v))
 for s in starts:solve(s)
 reference=max(candidates,key=lambda v:v[0])[1:4] if candidates else (.9,1,0)
 for face in [('rho',0),('rho',.999),('q_eta',0),('q_eta',100)]+([] if restricted else [('q_zeta',0)]):solve(reference,face)
 if not candidates:return dict(converged=False,runs=runs)
 ll,rho,qe,qz,v=max(candidates,key=lambda v:v[0]);fast=profile(x,rho,qe,qz,steady=True);error=float(np.max(abs(fast['ll']-v['ll'])));assert error<=1e-6,('LLT steady-state/full disagreement',error)
 return dict(converged=True,rho=rho,q_eta=qe,q_zeta=qz,half_life=pci.half(rho),scale=v['scale'],n=v['n'],ll=ll,restricted=restricted,boundary=bool(rho<1e-6 or rho>.999-1e-6 or qe<1e-6 or qe>100-1e-6 or qz<1e-6),steady_full_max_LL_error=error,runs=runs)
def compare(x,primary,warm=None,checkpoint=lambda:None):
 start=time.monotonic();x=np.atleast_2d(x);pooled_comparison=pci.fit(x,pooled=True,checkpoint=checkpoint,warm_start=primary,burn=2,warm_starts=[] if warm is None else [warm.get('primary_comparison')]);restricted=fit(x,True,None if warm is None else warm.get('restricted'),checkpoint);full=fit(x,False,None if warm is None else warm.get('full'),checkpoint)
 if restricted.get('converged') and full.get('converged') and restricted['ll']>full['ll']:
  # Include the fitted restricted boundary as a full-model candidate: nestedness.
  full=dict(restricted,restricted=False,runs=full['runs'],nested_boundary_candidate=True)
 models=[('primary_no_drift',pooled_comparison,2),('restricted_constant_drift',restricted,2),('full_LLT',full,3)];rows=[];N=len(x);baseline=None
 for name,v,k in models:
  row=dict(model=name,converged=v['converged'],series=N,shared_parameters=k,total_parameters=N+k,common_startup_excluded=2)
  if v['converged']:
   nn=int(np.sum(v['n']));qe=v.get('q_eta',v.get('q'));qz=v.get('q_zeta',0);row.update(loglikelihood=v['ll'],observations=nn,AIC=-2*v['ll']+2*(N+k),BIC=-2*v['ll']+np.log(nn)*(N+k),rho=v['rho'],q_eta=qe,q_zeta=qz,R_daily_change_to_level_shock=2/((1+v['rho'])*qe) if qe>0 else None,slope_innov_to_level_innov=qz/qe if qe>0 else None,R_stationary_variance_to_slope_shock=1/((1-v['rho']**2)*qz) if qz>0 else None,zero_level_shock=qe==0,zero_slope_shock=qz==0,SNR_note='Innovation/conditional ratios only; LLT integrated trend has no finite stationary long-run variance share')
   if name=='primary_no_drift':baseline=row
  rows.append(row)
 for row in rows:
  if row['converged'] and baseline:row.update(delta_AIC=row['AIC']-baseline['AIC'],delta_BIC=row['BIC']-baseline['BIC'])
 valid=restricted['converged'] and full['converged'];lr=max(0.,2*(full['ll']-restricted['ll'])) if valid else None;p=1. if lr==0 else float(.5*chi2.sf(lr,1)) if lr is not None else None
 test=dict(LR_slope=lr,p_slope_boundary_mixture=p,slope_needed_nominal5percent=bool(p is not None and p<.05),constant_drift_2LL_difference=2*(restricted['ll']-pooled_comparison['ll']) if restricted['converged'] and pooled_comparison['converged'] else None,constant_drift_p=None,mixture_reference='user-requested nominal0.5chi0+0.5chi1; LR0 tail includes point mass',primary_fit_for_signals_unchanged=True,measured_LLT_seconds=time.monotonic()-start)
 return dict(primary_comparison=pooled_comparison,restricted=restricted,full=full,models=rows,test=test)
