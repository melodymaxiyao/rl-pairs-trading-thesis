"""Independent stationary pair-years; shared streams never contain a learner key."""
from common import *
from dataclasses import dataclass
@dataclass(frozen=True)
class Model:
 half_life:float=20.;sigma_e:float=150.;mean:float=0.;family:str='OU';w:float=0.
 @property
 def phi(self):return float(np.exp(-np.log(2)/self.half_life))
 @property
 def sd(self):return self.sigma_e/np.sqrt(1-self.phi**2)
def simulate(model,n,key,sessions=252,start=0):
 # Per-pair streams make batching/worker count irrelevant.
 paths=np.empty((n,sessions+1))
 for j in range(n):
  rng=np.random.default_rng(stable_seed('paths',key,start+j));phi=model.phi
  if model.family=='T4':
   x=0.
   for z in rng.standard_t(4,int(np.ceil(100*model.half_life)))/np.sqrt(2):x=phi*x+model.sigma_e*z
  else:x=model.mean+model.sd*rng.normal()
  if model.family=='W':
   x=(x-model.mean)*np.sqrt(1-model.w);rw=0.
   paths[j,0]=x
   z=rng.standard_normal((sessions,2))
   for t in range(sessions):x=phi*x+model.sigma_e*np.sqrt(1-model.w)*z[t,0];rw+=model.sigma_e*np.sqrt(model.w)*z[t,1];paths[j,t+1]=x+rw
  else:
   paths[j,0]=x;z=rng.standard_t(4,sessions)/np.sqrt(2) if model.family=='T4' else rng.standard_normal(sessions)
   for t in range(sessions):x=model.mean+phi*(x-model.mean)+model.sigma_e*z[t];paths[j,t+1]=x
 count('synthetic_pair_years',n);return paths
def training(model,pairs,world,sessions=252):return simulate(model,3*pairs,['train',model.family,model.half_life,model.w,pairs,world],sessions)
def estimate_ar(paths):
 x=paths[:,:-1].ravel();y=paths[:,1:].ravel();design=np.column_stack([np.ones(len(x)),x]);beta=np.linalg.lstsq(design,y,rcond=None)[0];res=y-design@beta;n=len(x);var=float(res@res/(n-2));cov=var*np.linalg.inv(design.T@design);intercept,phi=map(float,beta);valid=0<phi<1 and var>0
 out=dict(intercept=intercept,phi=phi,sigma_e=np.sqrt(var),phi_SE=np.sqrt(cov[1,1]),sigma_e_SE=np.sqrt(var/(2*(n-2))),n_transitions=n,valid=valid,Q_scale=float(paths.std(ddof=1)),SE_scope='Gaussian working-model OLS; misspecification SEs not robust coverage claims')
 if valid:
  mean=intercept/(1-phi);grad=np.array([1/(1-phi),intercept/(1-phi)**2]);h=-np.log(2)/np.log(phi)
  out.update(mean=mean,mean_SE=float(np.sqrt(grad@cov@grad)),half_life=h,half_life_SE=float(np.log(2)/(phi*np.log(phi)**2)*out['phi_SE']),stationary_sd=float(np.sqrt(var/(1-phi**2))))
 else:out.update(mean=np.nan,mean_SE=np.nan,half_life=np.nan,half_life_SE=np.nan,stationary_sd=np.nan)
 return out
def model_from_estimate(estimate):
 assert estimate['valid'];return Model(estimate['half_life'],estimate['sigma_e'],estimate['mean'])
