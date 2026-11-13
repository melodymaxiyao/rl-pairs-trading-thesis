"""Byte-identical imported numerical functions; IO supplied by this run."""
from common import *
from scipy.stats import rankdata,norm,t
WORLD=[("real",0),("N2",0),("N2",1)]

def weighted_quantile(x,w,q):
 x=np.asarray(x,float);w=np.asarray(w,float);ok=np.isfinite(x)&np.isfinite(w)&(w>0);x=x[ok];w=w[ok]
 if not len(x):return np.nan
 order=np.argsort(x,kind='stable');x=x[order];w=w[order];return float(x[min(np.searchsorted(w.cumsum(),q*w.sum(),side='left'),len(x)-1)])

def weighted_ranks(x,w):
 x=np.asarray(x,float);w=np.asarray(w,float);order=np.argsort(x,kind='stable');a=x[order];v=w[order];start=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1];stop=np.r_[start[1:],len(a)];group=np.add.reduceat(v,start);mid=group.cumsum()-group/2;out=np.empty(len(x))
 for lo,hi,value in zip(start,stop,mid):out[order[lo:hi]]=value/w.sum()
 return out

def spearman(x,y,w=None):
 x=np.asarray(x,float);y=np.asarray(y,float);w=np.ones(len(x)) if w is None else np.asarray(w,float);ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0);x=x[ok];y=y[ok];w=w[ok]
 if len(x)<3:return np.nan
 rx=weighted_ranks(x,w);ry=weighted_ranks(y,w);rx-=np.average(rx,weights=w);ry-=np.average(ry,weights=w);den=np.sqrt(np.sum(w*rx**2)*np.sum(w*ry**2));return float(np.sum(w*rx*ry)/den) if den>0 else np.nan

def ic_records(kind,date,score,half,previous,future,weight):
 rows=[]
 for wi,(world,rep) in enumerate(WORLD):
  top=np.zeros(len(weight),bool);ok=np.isfinite(previous[wi])&np.isfinite(weight)&(weight>0)
  if ok.any():top[ok]=weighted_ranks(previous[wi,ok],weight[ok])>2/3
  for hi,h in enumerate(spec()['horizons']):
   x=-score[wi];y=future[wi,:,hi];valid=np.isfinite(x)&np.isfinite(y)&(weight>0);w=weight[valid];xx=x[valid];yy=y[valid]
   rows.append(dict(definition=kind,date=str(date),world=world,replicate=rep,h=h,ic=spearman(x,y,weight),ic_previous_fast_tercile=spearman(x[top],y[top],weight[top]),n=int(valid.sum()),sum_w=float(w.sum()),sum_x=float(w@xx),sum_y=float(w@yy),sum_xx=float(w@(xx*xx)),sum_xy=float(w@(xx*yy)),sum_yy=float(w@(yy*yy))))
 return rows

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
