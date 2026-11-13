"""Vectorised time-axis OLS/HAC statistics; no strategy or selection code."""
from common import count
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.adfvalues import mackinnonp,mackinnoncrit

def ols(y,X,hac=None):
 y=np.asarray(y,float);X=np.asarray(X,float)
 assert np.isfinite(y).all() and np.isfinite(X).all() and len(y)>X.shape[1]
 pinv=np.linalg.pinv(X);b=pinv@y;u=y-X@b;n,k=X.shape
 if hac is None:cov=(pinv@pinv.T)*(u@u)/(n-np.linalg.matrix_rank(X))
 else:
  xu=X*u[:,None];meat=xu.T@xu
  for j in range(1,min(hac,n-1)+1):
   cross=xu[j:].T@xu[:-j];meat+=(1-j/(hac+1))*(cross+cross.T)
  bread=pinv@pinv.T;cov=bread@meat@bread
 se=np.sqrt(np.maximum(0,np.diag(cov)));t=np.divide(b,se,out=np.full(k,np.nan),where=se>0)
 tss=np.sum((y-y.mean())**2)
 return dict(b=b,se=se,t=t,cov=cov,n=n,r2=1-(u@u)/tss if tss else np.nan,ssr=float(u@u))

def df_fit(S,trend='c',lag=5):
 S=np.asarray(S,float);ds=np.diff(S);n=len(ds)-lag
 X=np.column_stack([S[lag:-1],*[ds[lag-j:len(ds)-j] for j in range(1,lag+1)],np.ones(n)])
 if trend=='ct':X=np.column_stack([X,np.arange(1,n+1)])
 f=ols(ds[lag:],X);return f

def adf(S,trend='c',lag=5):
 f=df_fit(S,trend,lag);v=float(f['t'][0]);count('statistic_evaluations')
 return dict(statistic=v,nominal_p=float(mackinnonp(v,regression=trend,N=1)),n=f['n'],lag=lag,
  critical_5=float(mackinnoncrit(N=1,regression=trend,nobs=f['n'])[1]))

def kpss(S,trend='c'):
 x=np.asarray(S,float);n=len(x);X=np.ones((n,1)) if trend=='c' else np.column_stack([np.ones(n),np.arange(1,n+1)])
 e=x-X@(np.linalg.pinv(X)@x);lag=min(int(np.ceil(12*(n/100)**.25)),n-1)
 lrv=e@e
 for j in range(1,lag+1):lrv+=2*(1-j/(lag+1))*(e[j:]@e[:-j])
 v=float(np.cumsum(e)@np.cumsum(e)/(n*lrv)) if lrv>0 else np.nan
 critical=[.347,.463,.574,.739] if trend=='c' else [.119,.146,.176,.216]
 p=float(np.interp(v,critical,[.10,.05,.025,.01]));count('statistic_evaluations')
 return dict(statistic=v,nominal_p=p,n=n,lag=lag,critical_5=critical[1],nominal_p_boundary=p in [.01,.10])

def halflife(alpha,se):
 if alpha>=0:return dict(half_life=np.inf,half_life_se=np.nan,half_life_low=np.nan,half_life_high=np.inf,half_life_status='non_mean_reverting')
 if alpha<=-1:return dict(half_life=np.nan,half_life_se=np.nan,half_life_low=np.nan,half_life_high=np.nan,half_life_status='oscillatory_or_invalid_log')
 log=np.log1p(alpha);h=-np.log(2)/log;hs=abs(np.log(2)/((1+alpha)*log**2))*se
 return dict(half_life=float(h),half_life_se=float(hs),half_life_low=float(h-1.959963984540054*hs),half_life_high=float(h+1.959963984540054*hs),half_life_status='finite_delta_ci_unclipped')

def ecm(S):
 f=df_fit(S,'c',5);a=float(f['b'][0]);se=float(f['se'][0]);count('statistic_evaluations')
 return dict(statistic=a,alpha=a,alpha_se=se,t=float(f['t'][0]),nominal_p=float(norm.cdf(f['t'][0])),n=f['n'],lag=5,**halflife(a,se))

def predict(S,d,h):
 S=np.asarray(S,float);d=np.asarray(d,float)
 assert len(S)==len(d) and h>0
 y=-np.sign(d[:-h])*(S[h:]-S[:-h]);x=abs(d[:-h]);ok=np.isfinite(y)&np.isfinite(x)
 f=ols(y[ok],np.column_stack([np.ones(ok.sum()),x[ok]]),hac=h);count('statistic_evaluations')
 return dict(statistic=float(f['t'][1]),theta=float(f['b'][1]),intercept=float(f['b'][0]),theta_se=float(f['se'][1]),t=float(f['t'][1]),nominal_p=float(norm.sf(f['t'][1])),r2=float(f['r2']),n=f['n'],lag=h,horizon=h,window_end_exclusions=h,invalid_state_exclusions=int((~ok).sum()))

def vr(S,q):
 r=np.diff(np.asarray(S,float));mu=r.mean();n=len(r);base=np.sum((r-mu)**2)/(n-1)
 summed=np.convolve(r-mu,np.ones(q),'valid')
 # Lo-MacKinlay finite-sample overlap correction, matching governing diagnostic.
 denom=q*(n-q+1)*(1-q/n);v=float((summed@summed)/denom/base)
 count('statistic_evaluations');return dict(statistic=v,n=n,horizon=q)

def all_stats(S,d,window='T'):
 out={}
 for trend in ['c','ct']:
  out['ADF_'+trend]=adf(S,trend);out['KPSS_'+trend]=kpss(S,trend)
 out['ECM']=ecm(S)
 for h in [5,21,63]:out['theta_'+str(h)]=predict(S,d,h)
 for q in [21,63]:out['VR_'+str(q)]=vr(S,q)
 return out

def secondary(a,b):
 from statsmodels.tsa.stattools import coint,adfuller
 from statsmodels.tsa.vector_ar.vecm import coint_johansen
 out={}
 for trend in ['c','ct']:
  r=coint(a,b,trend=trend,maxlag=5,autolag=None);count('statistic_evaluations')
  out['EG_'+trend]=dict(statistic=float(r[0]),nominal_p=float(r[1]),n=len(a),lag=5)
 j=coint_johansen(np.column_stack([a,b]),det_order=0,k_ar_diff=5);count('statistic_evaluations')
 out['Johansen']=dict(statistic=float(j.lr1[0]),critical_5=float(j.cvt[0,1]),nominal_reject_5=bool(j.lr1[0]>j.cvt[0,1]),n=len(a),lag=5)
 return out

def adf_aic(S):
 from statsmodels.tsa.stattools import adfuller
 out={}
 for trend in ['c','ct']:
  a=adfuller(S,maxlag=10,regression=trend,autolag='AIC');count('statistic_evaluations')
  out['ADF_'+trend+'_AIC']=dict(statistic=float(a[0]),nominal_p=float(a[1]),n=a[3],lag=a[2])
 return out

def window_statistics(S,d,calendar,year,window):
 years=np.asarray(calendar.year)
 mask=(years>=year-3)&(years<year) if window=='T' else (years>=year-3)&(years<=year-2) if window=='A' else years==year-1
 assert window in ['T','A','B'] and mask.any()
 return all_stats(np.asarray(S)[mask],np.asarray(d)[mask],window)
