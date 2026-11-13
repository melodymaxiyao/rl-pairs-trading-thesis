"""F4 frozen-policy execution robustness. No learning, OU fit or PCA is invoked."""
from pathlib import Path
import os,sys,ast,json,gzip,io,hashlib
R4=Path(__file__).resolve().parents[1];B=R4.parent
sys.path.insert(0,str(B/'scripts'))
import common as C
C.P=R4;C.guard()
import sources as S
import numpy as np,pandas as pd
from types import SimpleNamespace

CORE=S.CORE;YEARS=list(range(2004,2026));LEARNERS=['AL','Band','Q_best'];NS={'Path':Path,'np':np,'pd':pd,'RUN':CORE,'INPUT':C.FYP,'read':C.js,'sha':C.sha,'_STATE':None}
assert C.sha(CORE/'preregistration.json')==S.SEAL['preregistration_sha256']
core_prereg=C.js(CORE/'preregistration.json')
assert C.sha(CORE/'code_registration.json')==S.SEAL['code_registration_sha256']
CODE=C.js(CORE/'code_registration.json')['files']
NS['spec']=lambda:core_prereg
class Source:
 prereg=json.loads(S.resraw('preregistration.json'))
 def verify(self,rel):S.resraw(rel);return S.RES/rel
NS['source']=lambda:Source()
PANEL=S.panel();NS['panel']=lambda:PANEL

def extract(rel,names):
 b=C.raw(CORE/rel);assert hashlib.sha256(b).hexdigest()==CODE[rel];tree=ast.parse(b.decode());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CORE/rel),'exec'),NS)
 return tree
extract('scripts/data.py',['entry_vector','scores'])
extract('scripts/missing_prices.py',['marked_prices','causal_target'])
tree=extract('scripts/accounting.py',['state','capital','execution'])
NS['LOG_COLUMNS']=[] # only a return/report label; original execution does not access this global.
ORIGINAL_EXECUTION=NS['execution']
# Read-only CSV adapter: preserve literal null while retaining the original numerical conversion rules.
class PandasView:
 def __getattr__(self,k):return getattr(pd,k)
 def read_csv(self,*a,**kw):kw.setdefault('keep_default_na',False);return pd.read_csv(*a,**kw)
NS['pd']=PandasView()

def frame(raw):return pd.read_csv(io.BytesIO(raw),compression='gzip',keep_default_na=False,float_precision='round_trip')

def context(kind,year):
 p=PANEL;Y=p['dates'].astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(Y==year);names=p['meta']['symbols'];lookup={n:i for i,n in enumerate(names)}
 if kind=='RP':
  rows=p['meta']['pairs'][str(year)];ids=[r['pair'] for r in rows];N=len(ids);refs={}
  for part in range(12):refs.update({v['pair']:v for v in S.cache(f'runtime/jobs/RP_{year}_{part:02d}')['series']})
  alpha=np.stack([refs[n]['alpha'] for n in ids]);beta=np.stack([refs[n]['beta'] for n in ids])[:,:,None];score=np.stack([refs[n]['score'] for n in ids],axis=1);half=np.stack([refs[n]['half'] for n in ids],axis=1)
  for n in ids:np.testing.assert_array_equal(refs[n]['indices'],ts)
  def index(n):n=p['meta']['aliases'].get(str(n).upper(),str(n).upper());return lookup.get(n,-1)
  return dict(kind=kind,year=year,ids=ids,rows=rows,N=N,T=len(ts),ts=ts,dates=p['dates'][ts],score=score,half=half,alpha=alpha,beta=beta,gross=1+abs(beta[:,:,0]),ia=np.array([index(r['leg_A']) for r in rows]),ib=np.array([index(r['leg_B']) for r in rows]))
 N=len(names);T=len(ts);c=dict(kind=kind,year=year,ids=list(names),rows=None,N=N,T=T,ts=ts,dates=p['dates'][ts],score=np.full((3,N,T),np.nan),half=np.full((3,N,T),np.nan),alpha=np.full((N,T),np.nan),beta=np.full((N,T,15),np.nan),gross=np.full((N,T),np.nan),W=np.zeros((T,N,15)))
 local={int(t):j for j,t in enumerate(ts)}
 for month in range(1,13):
  for v in S.cache(f'runtime/jobs/RF_{year}_{month:02d}')['days']:
   if v['status']!='ok':continue
   j=local[int(v['index'])];ii=v['indices'];c['score'][:,ii,j]=v['score'];c['half'][:,ii,j]=v['half'];c['alpha'][ii,j]=v['alpha'];c['beta'][ii,j]=v['beta'];c['gross'][ii,j]=v['gross'];c['W'][j]=v['weights']
 return c

def cell_source(kind,year,learner):
 stem=f'outputs/cells/{kind}_{year}_G0_c5_{learner}/';complete=json.loads(S.core(stem+'complete.json'));pol=complete['policy'];assert complete['status']=='complete'
 daily=frame(S.core(stem+'portfolio_daily.csv.gz'));daily=daily[daily.world=='real'].reset_index(drop=True)
 series=frame(S.core(stem+'per_series_year.csv.gz'));series=series[series.world=='real'].reset_index(drop=True)
 return pol,daily,series

def hedge_basis(c,result):
 p=result['positions'];N,T=p.shape;basis=np.full((N,T),-1,np.int32);lookup={str(n):i for i,n in enumerate(c['ids'])};dates={str(d):j for j,d in enumerate(c['dates'])};special={}
 for row in result['missing_price_log']:
  if row['executed_action']=='entry':special[lookup[str(row['series'])],dates[row['date']]]=dates[row['requested_date']]
 current=np.full(N,-1,np.int32);old=np.zeros(N,np.int32)
 for t in range(T):
  ii=np.flatnonzero((old==0)&(p[:,t]!=0))
  for i in ii:current[i]=special.get((int(i),t),t)
  current[p[:,t]==0]=-1;basis[:,t]=current;old=p[:,t]
 return basis

def replay(c,targets,bases,lag,market=None):
 market=NS['marked_prices'](PANEL) if market is None else market;returns=market['returns'];quotes=market['available'];N,T=c['N'],c['T'];nstock=returns.shape[1];canonical=np.argsort(np.asarray(c['ids'],str),kind='stable');rank=np.empty(N,int);rank[canonical]=np.arange(N);w=NS['capital'](c)
 book=np.zeros((N,nstock));stock=np.zeros(N);old=np.zeros(N,np.int32);pending_exit=np.zeros(N,bool);raw=np.zeros((N,4));daily=[];positions=np.zeros((N,T),np.int32);deferred=[]
 for ti,t in enumerate(c['ts']):
  active=old!=0;gross=np.zeros(N)
  if active.any():
   used=np.any(book[active]!=0,axis=0);assert np.isfinite(returns[t,used]).all(),('unknown held return',c['kind'],c['year'],ti)
   gross[active]=book[active][:,used]@returns[t,used]
  q=ti-lag;desired=targets[:,q].copy() if q>=0 else np.zeros(N,np.int32)
  if ti==T-1:desired[:]=0
  desired[pending_exit&(old!=0)]=0
  new=desired.copy();vectors={};basis_dates={}
  for i in canonical:
   if desired[i]==old[i]:continue
   if old[i]:
    # Preserve source no same-close side reversal and service a deferred exit first.
    new[i]=0;needed=book[i]!=0
   else:
    assert q>=0 and bases[i,q]>=0,('missing original hedge basis',i,ti,q)
    b=int(bases[i,q]);v=NS['entry_vector'](c,i,b)*desired[i]*w[i];vectors[i]=v;basis_dates[i]=b;needed=v!=0
   blocked=np.flatnonzero(needed&~quotes[t])
   if len(blocked):
    new[i]=old[i]
    if old[i]:pending_exit[i]=True
    deferred.append({'series':str(c['ids'][i]),'date':str(c['dates'][ti]),'action':'exit' if old[i] else 'entry','source_target_index':int(q),'unquoted_legs':';'.join(PANEL['meta']['symbols'][j] for j in blocked)})
  positions[:,ti]=new;changed=np.flatnonzero(new!=old);changed=changed[np.argsort(rank[changed],kind='stable')];delta=np.zeros((len(changed),nstock));own_delta=np.zeros(N)
  for j,i in enumerate(changed):
   if old[i]:delta[j]=-book[i];own_delta[i]=-stock[i];book[i]=0;stock[i]=0
   if new[i]:
    v=vectors[i];b=basis_dates[i];delta[j]+=v;book[i]=v;stock[i]=new[i]*w[i]/c['gross'][i,b];own_delta[i]+=stock[i]
  if len(changed):
   absd=abs(delta);den=absd.sum(axis=0);net=delta.sum(axis=0);netturn=abs(net);share=np.divide(netturn,den,out=np.zeros(nstock),where=den>0);allocation=absd@share;raw[changed,1]+=allocation
  else:netturn=np.zeros(nstock)
  raw[:,0]+=gross;raw[:,2]+=w*abs(new-old);raw[:,3]+=abs(own_delta)
  daily.append(dict(date=str(c['dates'][ti]),gross_bps=float(gross[canonical].sum()),K2=float(netturn.sum()),PP_FULL=float(np.sum((w*abs(new-old))[canonical])),K1=float(abs(own_delta[canonical]).sum()),gross_exposure=float(np.sum((w*(new!=0))[canonical])),net_stock_gross=float(abs(book[canonical].sum(axis=0)).sum()),positions=int(np.count_nonzero(new)),entries=int(np.sum((old==0)&(new!=0))),exits=int(np.sum((old!=0)&(new==0)))))
  pending_exit[changed]=False;old=new.copy()
 assert not old.any(),('unresolved terminal position; no invented liquidation',c['kind'],c['year'],int(np.sum(old!=0)))
 rows=[]
 for i,name in enumerate(c['ids']):
  if w[i]==0:continue
  rows.append(dict(series=str(name),weight=float(w[i]),gross_bps=float(raw[i,0]/w[i]),K2=float(raw[i,1]/w[i]),net_bps=float(raw[i,0]/w[i]-5*(raw[i,1]/w[i]))))
 f=pd.DataFrame(daily);f['net_bps']=f.gross_bps-5*f.K2
 return {'daily':f,'series':pd.DataFrame(rows),'positions':positions,'deferred':deferred}

def exact_frame(actual,old,cols,label):
 assert len(actual)==len(old),(label,'row count')
 for k in cols:np.testing.assert_array_equal(actual[k].to_numpy(),old[k].to_numpy(),err_msg=label+'/'+k)

def validate_all():
 assert C.js(R4/'audit/unit_tests.json')['passes'];proof=[];annual=[]
 for kind in ['RP','RF']:
  for year in YEARS:
   C.resources();c=context(kind,year)
   for learner in LEARNERS:
    pol,olddaily,oldseries=cell_source(kind,year,learner);v=ORIGINAL_EXECUTION(c,pol,'G0',0,details=False,cost=5.)
    d=pd.DataFrame(v['daily']);d['net_bps']=d.gross_bps-5*d.K2
    cols=['date','gross_bps','K2','PP_FULL','K1','gross_exposure','net_stock_gross','positions','entries','exits','net_bps'];exact_frame(d,olddaily,cols,f'original engine {kind}/{year}/{learner}')
    vs=pd.DataFrame(v['series']);vs['net_bps']=vs.gross_bps-5*vs.K2;exact_frame(vs,oldseries,['series','weight','gross_bps','K2','net_bps'],f'original per-series {kind}/{year}/{learner}')
    bases=hedge_basis(c,v);r=replay(c,v['positions'],bases,0);exact_frame(r['daily'],olddaily,cols,f'zero-delay frozen replay {kind}/{year}/{learner}');exact_frame(r['series'],oldseries,['series','weight','gross_bps','K2','net_bps'],f'zero-delay series {kind}/{year}/{learner}');np.testing.assert_array_equal(r['positions'],v['positions'])
    file=R4/f'frozen_decisions/{kind}_{year}_{learner}.npz';np.savez_compressed(file,positions=v['positions'],basis=bases,policy_json=np.asarray(json.dumps(pol,sort_keys=True)),ids=np.asarray(c['ids'],str))
    proof.append({'family':kind,'year':year,'learner':learner,'original_engine_exact':True,'frozen_replay_lag0_exact':True,'positions_exact':True,'frozen_decisions_sha256':C.sha(file),'source_complete_sha256':C.sha(CORE/f'outputs/cells/{kind}_{year}_G0_c5_{learner}/complete.json')})
    annual.append({'family':kind,'year':year,'learner':learner,'net_bps':float(np.sum(oldseries.net_bps.to_numpy()*oldseries.weight.to_numpy())),'turnover':float(np.sum(oldseries.K2.to_numpy()*oldseries.weight.to_numpy())),'robustness_check':True})
    C.dump(R4/'audit/zero_delay_progress.json',proof);print('F4 exact delay0 PASS',kind,year,learner,flush=True)
 C.csvout(R4/'same_close_annual.csv',annual);C.dump(R4/'audit/zero_delay_validation.json',{'passes':True,'fold_policy_checks':len(proof),'proof':proof,'exact_float_comparison':True,'new_real_trials':0,'cumulative_real_trials':580});assert len(proof)==132

def start_trial(kind,learner):
 p=R4/'trial_register.json';v=C.js(p);key=kind+'|'+learner+'|G0|5bps|next-close'
 if key not in v['started_cells']:
  v['started_cells'].append(key);v['new_real_trials_executed']=len(v['started_cells']);v['cumulative_real_trials']=580+len(v['started_cells']);C.dump(p,v)
 return v

def delayed_all():
 assert C.js(R4/'audit/zero_delay_validation.json')['passes'];assert len(C.js(R4/'audit/zero_delay_validation.json')['proof'])==132
 annual=[]
 for kind in ['RP','RF']:
  for learner in LEARNERS:
   for year in YEARS:
    C.resources();c=context(kind,year);z=np.load(R4/f'frozen_decisions/{kind}_{year}_{learner}.npz',allow_pickle=False);np.testing.assert_array_equal(z['ids'],c['ids']);start_trial(kind,learner);r=replay(c,z['positions'],z['basis'],1)
    stem=R4/f'outputs/{kind}_{year}_{learner}';stem.mkdir()
    r['daily'].to_csv(stem/'daily.csv.gz',index=False,compression='gzip',float_format='%.17g');r['series'].to_csv(stem/'series.csv.gz',index=False,compression='gzip',float_format='%.17g');C.csvout(stem/'deferred_quotes.csv',r['deferred'],['series','date','action','source_target_index','unquoted_legs'])
    f=r['series'];annual.append({'family':kind,'year':year,'learner':learner,'net_bps':float(np.sum(f.net_bps.to_numpy()*f.weight.to_numpy())),'turnover':float(np.sum(f.K2.to_numpy()*f.weight.to_numpy())),'robustness_check':True});C.csvout(R4/'next_close_annual_progress.csv',annual);print('F4 next-close',kind,year,learner,flush=True)
 C.csvout(R4/'next_close_annual.csv',annual)
 old=pd.DataFrame(C.rows(R4/'same_close_annual.csv'));new=pd.DataFrame(annual);ix=np.random.default_rng(20261026).integers(22,size=(5000,22));rows=[]
 def stat(x):
  x=np.asarray(x,float);v=x[ix].mean(axis=1);lo,hi=np.percentile(v,[2.5,97.5]);return float(x.mean()),float(lo),float(hi)
 for kind in ['RP','RF']:
  for learner in LEARNERS:
   a=new[(new.family==kind)&(new.learner==learner)].sort_values('year');b=old[(old.family==kind)&(old.learner==learner)].sort_values('year');row={'family':kind,'learner':learner,'cost_bps':5,'years':22,'robustness_check':True}
   for metric in ['net_bps','turnover']:
    av=a[metric].to_numpy(float);bv=b[metric].to_numpy(float)
    for name,vals in [('same_close',bv),('next_close',av),('change',av-bv)]:
     estimate,lo,hi=stat(vals);row[f'{name}_{metric}']=estimate;row[f'{name}_{metric}_ci_low']=lo;row[f'{name}_{metric}_ci_high']=hi
   rows.append(row)
 C.csvout(R4/'F4_next_close.csv',rows)
 h2=[]
 for label,f in [('same_close',old),('next_close',new)]:
  q=f[(f.family=='RF')&(f.learner=='Q_best')].sort_values('year').net_bps.to_numpy(float);a=f[(f.family=='RF')&(f.learner=='AL')].sort_values('year').net_bps.to_numpy(float);x=q-a;estimate,lo,hi=stat(x);h2.append({'execution':label,'estimate':estimate,'ci_low':lo,'ci_high':hi,'robustness_check':True})
 q=new[(new.family=='RF')&(new.learner=='Q_best')].sort_values('year').net_bps.to_numpy(float)-new[(new.family=='RF')&(new.learner=='AL')].sort_values('year').net_bps.to_numpy(float);a=old[(old.family=='RF')&(old.learner=='Q_best')].sort_values('year').net_bps.to_numpy(float)-old[(old.family=='RF')&(old.learner=='AL')].sort_values('year').net_bps.to_numpy(float);estimate,lo,hi=stat(q-a);h2.append({'execution':'change_next_minus_same','estimate':estimate,'ci_low':lo,'ci_high':hi,'robustness_check':True});C.csvout(R4/'F4_H2.csv',h2)
 assert C.js(R4/'trial_register.json')['cumulative_real_trials']==586
 C.dump(R4/'scientific_complete.json',{'status':'complete','new_real_trials':6,'cumulative_real_trials':586,'robustness_check':True,'H2':h2});print('F4 COMPLETE:6new trials;586 cumulative',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validate','delayed'],required=True);stage=p.parse_args().stage
 try:validate_all() if stage=='validate' else delayed_all()
 except Exception as e:
  C.dump(R4/'audit/STOP.json',{'status':'STOPPED','stage':stage,'error':str(e),'trial_register':C.js(R4/'trial_register.json')});raise
 finally:C.dump(R4/f'audit/{stage}_inputs.json',C.READS)
