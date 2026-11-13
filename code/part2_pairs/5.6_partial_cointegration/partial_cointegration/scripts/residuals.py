"""Pre-registered one-step-ahead residuals; old core numerical code is read-only."""
import ast,collections,io,types
import numpy as np,pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from io_utils import *
from groups import *
def imported_stats():
 import bootstrap
 c,d,a=bootstrap.core_modules();p=CORE/'vendor/residual_ou_real_v1/scripts/stats.py';reg=J(CORE/'code_registration.json');assert H(p)==reg['files'][str(p.relative_to(CORE))]
 names=['weighted_ranks','spearman','nw_mean','pooled_regression'];tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];ns=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 p=CORE/'vendor/residual_ou_real_v1/scripts/report.py';assert H(p)==reg['files'][str(p.relative_to(CORE))];node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='pooled_hac');exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
 p=CORE/'vendor/TAV/inference.py';assert H(p)==reg['files'][str(p.relative_to(CORE))];nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['blocks','estimate','holm']];ns.update(pd=pd,count=count);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return types.SimpleNamespace(**ns)
def pair_out_of_sample(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);A=sliding_window_view(a[:-1],60);B=sliding_window_view(b[:-1],60);am=A.mean(axis=1);bm=B.mean(axis=1);ac=A-am[:,None];bc=B-bm[:,None];den=np.sum(bc*bc,axis=1)
 with np.errstate(invalid='ignore',divide='ignore'):beta=np.sum(ac*bc,axis=1)/den;alpha=am-beta*bm
 e=a[60:]-alpha-beta*b[60:];e[(den<=0)|~np.isfinite(A).all(axis=1)|~np.isfinite(B).all(axis=1)]=np.nan
 return e,alpha,beta
def factor_out_of_sample(returns,indices,W,alpha,beta):
 returns=np.asarray(returns,float);needed=np.flatnonzero(np.any(W!=0,axis=1));out=np.full(len(returns),np.nan)
 if np.isfinite(returns[needed]).all():out[indices]=returns[indices]-alpha-(returns[needed]@W[needed])@beta.T
 return out
def rf_year(year):
 root=ROOT/f'runtime/RF_increments/{year}'
 if done(root):return dict(root=str(root),cached=True)
 import bootstrap
 c,d,a=bootstrap.core_modules();panel=d.panel();R=panel['returns'];years=panel['dates'].astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(years==year);N=R.shape[1];e=np.full((N,len(ts)),np.nan);old=e.copy();months={}
 def day(t):
  month=str(panel['dates'][t].astype('datetime64[M]'));yy,mm=map(int,month.split('-'));stem=f'runtime/jobs/RF_{yy}_{mm:02d}'
  if month not in months:
   if stem+'.pkl.gz' in c.source().expected:months[month]={v['index']:v for v in c.source().cache(stem)['days']}
   else:months[month]={}
   # Only the two most recently visited months are required.
   if len(months)>3:months.pop(next(iter(months)))
  if t in months[month]:
   v=months[month][t]
   return None if v['status']!='ok' else (v['indices'],v['weights'],v['alpha'],v['beta'])
  history=R[max(0,t-251):t+1];members=np.flatnonzero(panel['membership'][t]);valid=members[np.isfinite(history[:,members]).all(axis=0)] if len(history)==252 else np.array([],int);valid=valid[np.std(history[:,valid],axis=0)>0] if len(valid) else valid
  if len(valid)<16:return None
  try:model=d.M.factor_window(history[:,valid])
  except (ValueError,np.linalg.LinAlgError):return None
  W=np.zeros((N,15));W[valid]=model['weights'];return valid,W,model['alpha'],model['beta']
 for j,t in enumerate(ts):
  checkpoint()
  if t>0:
   previous=day(int(t)-1)
   if previous is not None:
    ids,W,alpha,beta=previous;e[:,j]=factor_out_of_sample(R[t],ids,W,alpha,beta);e[~panel['membership'][t],j]=np.nan
  current=day(int(t))
  if current is not None:
   ids,W,alpha,beta=current;old[:,j]=factor_out_of_sample(R[t],ids,W,alpha,beta)
  if j%30==0:print('Causal RF residuals',year,j,'/',len(ts),flush=True)
 return finish(root,{'increments.npz':array_bytes(e=e,core_inwindow=old,indices=ts)},phase='RF_causal_increments',year=year)
def load_group(kind,year):
 import bootstrap
 c,d,a=bootstrap.core_modules();p=d.panel();Y=p['dates'].astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero((Y>=year-3)&(Y<=year));formation=int(np.sum(Y[ts]<year));T=len(ts)
 if kind=='RP':
  rows=p['meta']['pairs'][str(year)];ids=[r['pair'] for r in rows];e=np.full((len(rows),T),np.nan);old=e.copy();prefix=[]
  for i,row in enumerate(rows):
   ia,ib=d.index(row['leg_A'],p),d.index(row['leg_B'],p)
   if min(ia,ib)>=0 and ts[0]>=60:
    aa=p['returns'][:ts[-1]+1,ia];bb=p['returns'][:ts[-1]+1,ib];whole,_,_=pair_out_of_sample(aa,bb);e[i]=whole[ts-60];inwindow,_,_=d.M.pair_rolling(aa[ts[0]-59:],bb[ts[0]-59:]);old[i]=inwindow[:,-1]
    prefix.append(whole[:ts[0]-60])
   else:prefix.append(np.empty(0))
   if i%100==0:checkpoint()
 else:
  ids=list(p['meta']['symbols']);rows=None;ee=[];oo=[];prefix=[]
  firstyear=int(Y.min())
  for yy in range(firstyear,year-3):
   root=ROOT/f'runtime/RF_increments/{yy}';assert done(root);q=np.load(root/'increments.npz',allow_pickle=False);prefix.append((q['e'],q['indices']))
  for y in range(year-3,year+1):
   root=ROOT/f'runtime/RF_increments/{y}';assert done(root);q=np.load(root/'increments.npz',allow_pickle=False);ee.append(q['e']);oo.append(q['core_inwindow'])
  e=np.concatenate(ee,axis=1);old=np.concatenate(oo,axis=1)
 assert e.shape==(len(ids),T);return dict(kind=kind,year=year,ids=ids,rows=rows,e=e,core_inwindow=old,indices=ts,dates=p['dates'][ts],formation=formation,prefix=prefix)
def signed(group,wi):
 import bootstrap
 c,d,a=bootstrap.core_modules();e=group['e'];dates=group['dates'];kind=group['kind'];year=group['year']
 signs=np.ones_like(e) if wi==0 else np.stack([d.M.sign_series(c.source().prereg['null_seed'],f'RP|{year}:{name}' if kind=='RP' else 'RF|'+name,wi-1,dates) for name in group['ids']])
 offset=np.zeros(len(e));panel=d.panel()
 if kind=='RP':
  for i,prior in enumerate(group['prefix']):
   if len(prior)==0:continue
   dates_before=panel['dates'][60:group['indices'][0]];sign=np.ones(len(prior)) if wi==0 else d.M.sign_series(c.source().prereg['null_seed'],f'RP|{year}:{group["ids"][i]}',wi-1,dates_before);offset[i]=np.nansum(prior*sign)
 else:
  for prior,indices in group['prefix']:
   for i,name in enumerate(group['ids']):
    sign=np.ones(len(indices)) if wi==0 else d.M.sign_series(c.source().prereg['null_seed'],'RF|'+name,wi-1,panel['dates'][indices]);offset[i]+=np.nansum(prior[i]*sign)
 values=e*signs;levels=offset[:,None]+np.cumsum(np.where(np.isfinite(values),values,0.),axis=1);levels[~np.isfinite(values)]=np.nan
 # Zero in the internal cumulative accumulator is bookkeeping for an undefined
 # increment, never an observation, OU input, IC target or trade signal.
 s,half=rolling_chunked(values,d.M.ou_fit)
 return values,levels,s,half
def core_targets(kind,year):
 import bootstrap
 c,d,a=bootstrap.core_modules();p=d.panel();Y=p['dates'].astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(Y==year);N=len(p['meta']['pairs'][str(year)]) if kind=='RP' else len(p['meta']['symbols']);target=np.full((3,N,len(ts),2),np.nan);score=np.full((3,N,len(ts)),np.nan);half=score.copy()
 if kind=='RP':
  refs={}
  for part in range(12):refs.update({r['pair']:r for r in c.source().cache(f'runtime/jobs/RP_{year}_{part:02d}')['series']})
  for i,row in enumerate(p['meta']['pairs'][str(year)]):
   v=refs[row['pair']];score[:,i]=v['score'];half[:,i]=v['half'];target[:,i]=v['future'][:,:,:2]
  w=np.array([r['ipw'] for r in p['meta']['pairs'][str(year)]],float);w/=w.sum()
 else:
  lookup={int(t):i for i,t in enumerate(ts)}
  for month in range(1,13):
   for v in c.source().cache(f'runtime/jobs/RF_{year}_{month:02d}')['days']:
    if v['status']!='ok':continue
    j=lookup[v['index']];ids=v['indices'];score[:,ids,j]=v['score'];half[:,ids,j]=v['half'];target[:,ids,j]=v['future'][:,:,:2]
  w=a.capital(dict(kind='RF',ids=p['meta']['symbols']))
 score[~(np.isfinite(half)&(half>0)&(half<30))]=np.nan;return target,score,w

def rolling_chunked(values,fit_function,chunk_size=75):
 s=np.full_like(values,np.nan);half=s.copy()
 for i in range(0,len(values),chunk_size):
  checkpoint();fit=fit_function(sliding_window_view(values[i:i+chunk_size],60,axis=1));s[i:i+chunk_size,59:]=fit['s'];half[i:i+chunk_size,59:]=fit['half_life']
 valid=np.isfinite(s)&np.isfinite(half)&(half>0)&(half<30);s[~valid]=np.nan
 return s,half
