"""Predefined policies; costs enter the unchanged POW trainer at2*c."""
from common import *
from native import fit_arrays
from observed import execute,policy,al,band,spi,summary
from source_stats import nw_mean,holm

def q_pow(data,rank,world,cost):
 stem=f'runtime/models/{rank}_world{world:02d}_c{cost}';old=cached(stem)
 if old is not None:return old
 fits=[];returns=np.column_stack([data['returns'],np.zeros(len(data['returns']))]);z=np.where(data['valid'],data['s'],np.inf)
 for j in range(10):
  key=stem+f'_seed{j:02d}';f=cached(key)
  if f is None:
   os.environ['OU_SCOPE']='TD_production';f=fit_arrays(z,returns,td_seed(j),'POW',2*cost,np.inf,spec()['training']['epochs'],pairids=np.arange(len(z),dtype=np.int32)//3);save(key,f)
  fits.append(f);print('Q_POW seed',rank,world,cost,j,'epochs',f['epochs'],flush=True);checkpoint()
 q=np.mean([f['q'] for f in fits],axis=0);bits=sum(int(q[s,1]>q[s,0])<<s for s in range(8));result=dict(q=q,bits=bits,seeds=[td_seed(j) for j in range(10)],fit_epochs=[f['epochs'] for f in fits],step='n^-0.7',gamma=1,ensemble='mean of10 Q tables, strict greater switch, tie action0');return save(stem,result)

def band_train(data,cost):
 policies=[band(a,b) for a in spec()['band_enter'] for b in spec()['band_exit']];values=[summary(execute(data,p,cost))['net'] for p in policies];chosen=int(np.argmax(values));return dict(policy=policies[chosen],values=values,criterion='training mean net at this cost; fixed grid tie order')

def spi_td(data,qbits,cost):
 base,days=execute(data,al(),cost,daily=True);assert len(days)%3==0;P=len(days)//3;baseline=days.reshape(P,3,-1).mean(axis=0).ravel();tests=[]
 for state in range(8):
  _,daily=execute(data,spi(qbits,1<<state),cost,daily=True);delta=daily.reshape(P,3,-1).mean(axis=0).ravel()-baseline
  if not np.any(delta):continue
  tests.append(dict(state=state,bit=state,proposed=True,**nw_mean(delta,lag=21)))
 reject,padj=holm([t['p'] for t in tests],alpha=.05);mask=0
 for row,yes,p in zip(tests,reject,padj):row.update(accepted=bool(yes),p_Holm=float(p));mask|=1<<row['state'] if yes else 0
 return dict(policy=spi(qbits,mask),tests=tests,lag=21,training_days=len(baseline),test_rows_used=0)
