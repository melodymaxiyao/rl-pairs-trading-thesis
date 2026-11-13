"""Exact-diffuse Kalman maximum likelihood for a fixed-beta spread level."""
from mio import count,record
import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.initialization import Initialization

STARTS=((-0.5,0.25),(0.5,0.5),(0.95,0.8))

class Likelihood:
    def __init__(self,level):
        self.level=np.asarray(level,float)
        self.normalizer=float(np.std(np.diff(self.level),ddof=1))
        assert len(self.level)>=20 and np.isfinite(self.level).all() and self.normalizer>0
        y=np.ascontiguousarray((self.level-self.level[0])/self.normalizer)
        self.model=KalmanFilter(k_endog=1,k_states=2,k_posdef=2)
        self.model.bind(y)
        self.model.design=np.array([[1.,1.]])
        self.model.selection=np.eye(2);self.model.obs_cov=np.zeros((1,1))
        initial=Initialization(2);initial.set(0,'stationary');initial.set(1,'diffuse')
        self.model.initialize(initial);self.model.filter_concentrated=True
        self.evaluations=0

    def update(self,parameters):
        rho,fraction=parameters
        self.model.transition=np.diag([rho,1.])
        self.model.state_cov=np.diag([fraction,1-fraction])

    def objective(self,parameters):
        count('pci_objectives_started');self.evaluations+=1
        try:
            self.update(parameters);value=-float(self.model.loglike())
            return value if np.isfinite(value) else 1e100
        finally:count('pci_objectives_completed')

def fit(level):
    count('pci_fits_started');attempts=[]
    try:
        likelihood=Likelihood(level)
        for start in STARTS:
            count('pci_optimizers_started')
            try:
                result=minimize(likelihood.objective,start,method='L-BFGS-B',bounds=[(-.99999999,.99999999),(0.,1.)],
                    options=dict(maxiter=300,ftol=1e-10,gtol=1e-7))
                attempts.append(dict(start=list(start),success=bool(result.success),status=int(result.status),
                    message=str(result.message),nfev=int(result.nfev),nit=int(result.nit),objective=float(result.fun),parameters=result.x.tolist()))
            except Exception as error:
                attempts.append(dict(start=list(start),success=False,error=repr(error),objective=None))
            finally:count('pci_optimizers_completed')
        finite=[a for a in attempts if a.get('objective') is not None and np.isfinite(a['objective']) and a['objective']<1e90]
        if not finite:return dict(converged=False,reason='All starts failed',starts=attempts,n_obs=len(level))
        best=min(finite,key=lambda a:a['objective']);rho,fraction=best['parameters']
        likelihood.update([rho,fraction]);count('pci_final_filters_started')
        try:filtered=likelihood.model.filter()
        finally:count('pci_final_filters_completed')
        scale=float(filtered.scale)*likelihood.normalizer**2
        sm=float(np.sqrt(max(0.,scale*fraction)));sr=float(np.sqrt(max(0.,scale*(1-fraction))))
        total=2*sm*sm+(1+rho)*sr*sr
        hl=float(np.log(.5)/np.log(rho)) if 0<rho<1 else np.nan
        envelope=float(np.log(.5)/np.log(abs(rho))) if 0<abs(rho)<1 else 0. if rho==0 else np.nan
        boundary_rho=abs(rho)>=.999
        boundary_M=sm<=1e-6*likelihood.normalizer;boundary_R=sr<=1e-6*likelihood.normalizer
        result=dict(converged=best['success'],n_obs=len(level),rho_M=float(rho),sigma_M=sm,sigma_R=sr,
            R2_MR=2*sm*sm/total if total>0 else np.nan,half_life=hl,
            half_life_order=-np.inf if rho<=0 else hl,envelope_half_life=envelope,oscillatory=bool(rho<=0),
            sd_M=sm/np.sqrt(1-rho*rho),swing_bps=2*sm/np.sqrt(1-rho*rho),
            boundary_rho=bool(boundary_rho),boundary_sigma_M=bool(boundary_M),boundary_sigma_R=bool(boundary_R),
            boundary=bool(boundary_rho or boundary_M or boundary_R),standardized_negative_log_likelihood=best['objective'],
            scale_normalizer=likelihood.normalizer,objective_evaluations=likelihood.evaluations,starts=attempts)
        result['swing_ge_20']=bool(result['swing_bps']>=20)
        return result
    except Exception as error:
        return dict(converged=False,reason=repr(error),n_obs=len(level),starts=attempts)
    finally:count('pci_fits_completed')
