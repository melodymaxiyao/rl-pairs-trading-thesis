from common import *
import numpy as np
from scipy.stats import norm,t

def nw_mean(x,lag=63):
 x=np.asarray(x,float);assert np.isfinite(x).all() and len(x)>1;n=len(x);u=x-x.mean();meat=float(u@u)
 for j in range(1,min(lag,n-1)+1):meat+=2*(1-j/(lag+1))*float(u[j:]@u[:-j])
 se=np.sqrt(max(0.,meat))/n;mu=float(x.mean());z=mu/se if se else np.inf if mu>0 else -np.inf if mu<0 else 0.
 return dict(estimate=252*mu,SE=252*se,z=z,p=float(norm.sf(z)),n_days=n,lag=lag)

def holm(p,alpha=.05):
 p=np.asarray(p,float);order=np.argsort(p,kind='stable');reject=np.zeros(len(p),bool);adjusted=np.zeros(len(p));last=0.;alive=True
 for j,i in enumerate(order):
  last=max(last,(len(p)-j)*p[i]);adjusted[i]=min(1.,last)
  if alive and p[i]<=alpha/(len(p)-j):reject[i]=True
  else:alive=False
 return reject,adjusted

def interval(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)];n=len(x);m=float(x.mean()) if n else np.nan;sd=float(x.std(ddof=1)) if n>1 else np.nan;half=float(t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else np.nan
 return dict(estimate=m,ci_low=m-half,ci_high=m+half,sd=sd,n=n)
