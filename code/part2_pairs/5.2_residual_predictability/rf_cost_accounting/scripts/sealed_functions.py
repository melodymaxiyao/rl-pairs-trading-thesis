import numpy as np
from collections import Counter
COUNTS=Counter()

def update_positions(previous,score,half_life,eligible):
 p=np.asarray(previous,np.int8).copy();s=np.asarray(score,float);h=np.asarray(half_life,float);ok=np.asarray(eligible,bool)&np.isfinite(s)&np.isfinite(h)&(h<30)&(h>0)
 p[~ok| (abs(s)<.5)]=0
 flat=(p==0)&ok;p[flat&(s>1.25)]=-1;p[flat&(s< -1.25)]=1
 return p

def holdings_update(old,new,durations,closed):
 durations[old!=0]+=1
 for i in np.flatnonzero((old!=0)&(new==0)):closed.append(int(durations[i]));durations[i]=0
 durations[(old==0)&(new!=0)]=0

def ols_hac(y,x,lag,annualisation):
 y=np.asarray(y,float);x=np.asarray(x,float);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok];COUNTS['rows_aggregated']+=len(y)
 X=np.column_stack([np.ones(len(x)),x])
 if len(y)<=2 or np.linalg.matrix_rank(X)<2:return dict(status='rank deficient',n=len(y))
 b=np.linalg.lstsq(X,y,rcond=None)[0];u=y-X@b;A=np.linalg.inv(X.T@X);z=X*u[:,None];meat=z.T@z
 for k in range(1,min(lag,len(y)-1)+1):
  C=z[k:].T@z[:-k];meat+=(1-k/(lag+1))*(C+C.T)
 cov=A@meat@A;se=np.sqrt(max(0.,cov[0,0]));mean=y.mean()*annualisation
 return dict(status='ok',n=len(y),beta=float(b[1]),alpha_annual_bps=float(b[0]*annualisation),alpha_annual_se_bps=float(se*annualisation),t_alpha=float(b[0]/se) if se>0 else np.nan,correlation=float(np.corrcoef(x,y)[0,1]) if y.std()>0 else np.nan,mean_annual_bps=float(mean),market_share=float(1-b[0]*annualisation/mean) if mean!=0 else np.nan,appraisal_ratio=np.nan,lag=lag,annualisation=annualisation)
