"""Unit-root/stationarity gates: stdlib and NumPy only, fixed constants/lag1."""
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
ADF5=np.array([-2.86154,-2.8903,-4.234,-40.04])
DFGLS5=-1.94;KPSS5=.463;VR5=-1.6448536269514722
def adf_critical(n):return float(ADF5@np.array([1,1/(n-2),1/(n-2)**2,1/(n-2)**3]))
def regression_t(levels,constant=True):
 x=np.asarray(levels,float);shape=x.shape[:-1];n=x.shape[-1];flat=x.reshape(-1,n);out=np.full(len(flat),np.nan);good=np.isfinite(flat).all(axis=1)
 if not good.any():return out.reshape(shape)
 a=flat[good];d=np.diff(a,axis=1);y=d[:,1:];cols=[a[:,1:-1],d[:,:-1]]
 if constant:cols.append(np.ones_like(y))
 X=np.stack(cols,axis=-1);P=np.linalg.pinv(X,rcond=1e-15);b=np.einsum('nij,nj->ni',P,y);err=y-np.einsum('nij,nj->ni',X,b);rank=np.linalg.matrix_rank(X);dof=(n-2)-rank
 with np.errstate(divide='ignore',invalid='ignore'):t=b[:,0]/np.sqrt(np.sum(err*err,axis=1)/dof*np.sum(P[:,0]**2,axis=1))
 t[~np.isfinite(t)]=np.nan;out[good]=t;return out.reshape(shape)
def adf(e):return regression_t(np.cumsum(e,axis=-1),True)
def normal_cdf(x):
 a=np.asarray(x,float);return np.array([.5*math.erfc(-v/math.sqrt(2)) if np.isfinite(v) else np.nan for v in a.ravel()]).reshape(a.shape)
def adf_pvalue(t):
 a=np.asarray(t,float);small=np.array([2.1659,1.4412,.038269]);large=np.array([1.7339,.93202,-.12745,-.010368]);z=np.where(a<=-1.61,np.polynomial.polynomial.polyval(a,small),np.polynomial.polynomial.polyval(a,large));p=normal_cdf(z);return np.where(a>2.74,1.,np.where(a< -18.83,0.,p))
def dfgls(e):
 x=np.cumsum(e,axis=-1);n=x.shape[-1];a=1-7/n;delta=x.copy();delta[...,1:]=x[...,1:]-a*x[...,:-1];z=np.full(n,1-a);z[0]=1.;beta=np.sum(delta*z,axis=-1)/np.sum(z*z);return regression_t(x-beta[...,None],False)
def kpss(e):
 x=np.cumsum(e,axis=-1);shape=x.shape[:-1];n=x.shape[-1];a=x.reshape(-1,n);out=np.full(len(a),np.nan);good=np.isfinite(a).all(axis=1)
 if not good.any():return out.reshape(shape)
 r=a[good]-a[good].mean(axis=1,keepdims=True);s0=np.sum(r*r,axis=1)/n;s1=np.zeros(len(r));covlags=int(n**(2/9))
 for lag in range(1,covlags+1):prod=2*np.sum(r[:,lag:]*r[:,:-lag],axis=1)/n;s0+=prod;s1+=lag*prod
 with np.errstate(divide='ignore',invalid='ignore'):ratio=s1/s0;auto=1.1447*np.power(ratio*ratio,1/3)*n**(1/3)
 lags=np.clip(np.nan_to_num(auto,nan=0,posinf=n-1,neginf=0).astype(int),0,n-1);s=np.sum(r*r,axis=1)
 for lag in range(1,int(lags.max())+1):s+=np.where(lags>=lag,2*(1-lag/(lags+1))*np.sum(r[:,lag:]*r[:,:-lag],axis=1),0)
 with np.errstate(divide='ignore',invalid='ignore'):statistic=np.sum(np.cumsum(r,axis=1)**2,axis=1)/n**2/(s/n)
 statistic[~np.isfinite(statistic)]=np.nan;out[good]=statistic;return out.reshape(shape)
def vr(e,q):
 e=np.asarray(e,float);n=e.shape[-1];r=e-e.mean(axis=-1,keepdims=True);s=np.sum(r*r,axis=-1);one=s/(n-1);roll=sliding_window_view(r,q,axis=-1).sum(axis=-1);m=q*(n-q+1)*(1-q/n)
 with np.errstate(divide='ignore',invalid='ignore'):
  ratio=np.sum(roll*roll,axis=-1)/m/one;theta=np.zeros_like(ratio)
  for j in range(1,q):theta+=(2*(q-j)/q)**2*np.sum(r[...,j:]**2*r[...,:-j]**2,axis=-1)/s**2
  z=(ratio-1)/np.sqrt(theta)
 z=np.where(np.isfinite(z),z,np.nan);return ratio,z
def statistics(e):
 a=adf(e);d=dfgls(e);k=kpss(e);r5,z5=vr(e,5);r10,z10=vr(e,10)
 return dict(ADF=a,DFGLS=d,KPSS=k,VR5=z5,VR10=z10)
def rejection(name,values,window):
 threshold=adf_critical(window) if name=='ADF' else DFGLS5 if name=='DFGLS' else KPSS5 if name=='KPSS' else VR5
 return np.isfinite(values)&(values>threshold if name=='KPSS' else values<threshold)
def gate(name,values,window):
 reject=rejection(name,values,window);return np.isfinite(values)&(~reject if name=='KPSS' else reject)
