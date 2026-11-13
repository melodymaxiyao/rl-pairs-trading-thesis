"""Fold tasks. No real task is invoked by preparation/tests."""
from common import *
import pandas as pd,gc
from data import panel,pool,rp_context,rf_context,reward_cube,training_scores,fitted_ar1
from accounting import execution
from kernel import cube_values
from learning import train_ensemble,band,spi
import plugin
LEARNERS=['AL','Band','PlugIn','Q_best','Q_POW','SPI']

def wanted(tier):
 if tier==1:return [(5,'G0',LEARNERS),(5,'G1',['AL','Q_best'])]
 if tier==2:return [(5,'G1',['Band','PlugIn','Q_POW','SPI'])]
 return [(c,g,LEARNERS) for c in [2,10] for g in ['G0','G1']]

def cell_root(kind,year,gate,cost,learner):return RUN/f'outputs/cells/{kind}_{year}_{gate}_c{cost}_{learner}'
def done_cell(kind,year,gate,cost,learner):
 p=cell_root(kind,year,gate,cost,learner)/'complete.json'
 if not p.exists():return False
 c=read(p);assert c['preregistration_sha256']==sha(RUN/'preregistration.json')
 actual={str(q.relative_to(RUN)) for q in p.parent.rglob('*') if q.is_file()};assert actual==set(c['files'])|{str(p.relative_to(RUN))},('Sealed cell namespace changed',str(p))
 for rel,h in c['files'].items():assert sha(RUN/rel)==h
 return True

def model_path(kind,year,gate,cost,name):return RUN/f'outputs/models/{kind}_{year}_{gate}_c{cost}_{name}.json.gz'
def load_model(path):
 receipt=path.with_suffix(path.suffix+'.ready.json')
 if not path.exists():return None
 assert sha(path)==read(receipt)['sha256'];return json.loads(gzip.decompress(path.read_bytes()))
def store_model(path,v):
 publish(path,gzip.compress(json.dumps(clean(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode(),mtime=0));write(path.with_suffix(path.suffix+'.ready.json'),dict(sha256=sha(path),preregistration_sha256=sha(RUN/'preregistration.json')))

def compact_bins(rows):
 f=pd.DataFrame(rows);return f.groupby(['side','bin'],as_index=False)[['opportunities','transitions']].sum().assign(rate=lambda x:x.transitions/np.where(x.opportunities>0,x.opportunities,np.nan))

def evaluate_cell(c,pol,gate,cost,learner):
 root=cell_root(c['kind'],c['year'],gate,cost,learner);series=[];daily=[];bins=[];comparisons=[];missing=[]
 for wi in range(3):
  checkpoint();v=execution(c,pol,gate,wi,details=False,cost=cost);meta=dict(definition=c['kind'],gate=gate,learner=learner,cost=cost,world='real' if wi==0 else 'N2',replicate=0 if wi==0 else wi-1)
  missing.extend(dict(row,**meta) for row in v['missing_price_log'])
  f=pd.DataFrame(v['series']).assign(**meta);f['net_bps']=f.gross_bps-cost*f.K2;f['per_position_net_bps']=f.gross_bps-cost*f.PP_FULL;series.append(f)
  g=pd.DataFrame(v['daily']).assign(**meta);g['net_bps']=g.gross_bps-cost*g.K2;g['per_position_net_bps']=g.gross_bps-cost*g.PP_FULL;g['K1_net_bps']=g.gross_bps-cost*g.K1;daily.append(g)
  bins.append(compact_bins(v['bins']).assign(**meta))
  if learner=='AL' and wi==0:
   legacy=execution(c,dict(kind=2,enter=1.25,exit=.5),gate,wi,details=False)
   from data import scores
   z=scores(c,gate,wi);prev=np.column_stack([np.zeros(c['N'],int),v['positions'][:,:-1]]);qual=np.isfinite(z)&(prev!=0);overshoot=qual&(prev*z>0)&(abs(z)>=.5)
   comparisons.append(dict(**meta,sealed_abs_exit_comparator_same_new_frozen_capital=True,days_with_exit_rule_difference=int(overshoot.sum()),eligible_holding_decisions=int(qual.sum()),main_K2_net_bps=float(g.net_bps.sum()),abs_exit_K2_net_bps=float(sum(r['gross_bps']-cost*r['K2'] for r in legacy['daily'])),position_days_different=int(np.sum(v['positions']!=legacy['positions']))))
 files={}
 for name,frame in [('per_series_year.csv.gz',pd.concat(series,ignore_index=True)),('portfolio_daily.csv.gz',pd.concat(daily,ignore_index=True)),('entries_exits.csv.gz',pd.concat(bins,ignore_index=True))]:
  p=root/name;publish(p,csv_bytes(frame));files[str(p.relative_to(RUN))]=sha(p)
 from missing_prices import LOG_COLUMNS
 p=root/'missing_price_log.csv';publish(p,pd.DataFrame(missing).reindex(columns=LOG_COLUMNS).to_csv(index=False));files[str(p.relative_to(RUN))]=sha(p)
 p=root/'exit_comparison.json';write(p,comparisons);files[str(p.relative_to(RUN))]=sha(p)
 write(root/'complete.json',dict(status='complete',files=files,preregistration_sha256=sha(RUN/'preregistration.json'),policy=pol,source_completion_sha256=seal('residual_ou_real_v1').anchor['complete_sha256']))

def work(task):
 kind,year,tier=task;os.environ['OU_SCOPE']='production';guard();checkpoint();start=time.monotonic()
 todo=[(c,g,[l for l in ls if not done_cell(kind,year,g,c,l)]) for c,g,ls in wanted(tier)];todo=[r for r in todo if r[2]]
 if not todo:return dict(task=task,cached=True)
 print('Building in-memory contexts',kind,year,'Tier',tier,flush=True)
 if kind=='RP':
  p=pool(year);mapping={r['pair']:r for r in p['P_RAND']};assert set(mapping)=={r['pair'] for r in panel()['meta']['pairs'][str(year)]};trainrows=[mapping[name] for name in p['training_subset']];testrows=panel()['meta']['pairs'][str(year)]
  contexts=[rp_context(y,trainrows) for y in range(year-3,year)];test=rp_context(year,testrows,verify_stored=True)
 else:
  contexts=[rf_context(y) for y in range(year-3,year)];test=rf_context(year)
 model=fitted_ar1([c['stats'] for c in contexts]);cohort=np.concatenate([np.isfinite(c['score'][0]).any(axis=1) if kind=='RF' else np.ones(c['N'],bool) for c in contexts]);cube=reward_cube(contexts)[cohort];plugin_fit=None
 for cost,gate,learners in todo:
  print('Training/evaluation',kind,year,gate,'cost',cost,','.join(learners),flush=True)
  z=training_scores(contexts,gate)[cohort];lengths=[(int(np.isfinite(c['score'][0]).any(axis=1).sum()) if kind=='RF' else c['N'],c['T']) for c in contexts];models={}
  def get(name,compute):
   path=model_path(kind,year,gate,cost,name);v=load_model(path)
   if v is None:v=compute();store_model(path,v)
   return v
  models['AL']=dict(kind=1,enter=1.25,exit=.5)
  if any(l in learners for l in ['Q_best','SPI']):models['Q_best']=get('Q_best',lambda:train_ensemble(z,cube,kind,year,gate,cost,True))['policy']
  if 'Q_POW' in learners:models['Q_POW']=get('Q_POW',lambda:train_ensemble(z,cube,kind,year,gate,cost,False))['policy']
  if 'Band' in learners:models['Band']=get('Band',lambda:band(z,cube,cost))
  if 'PlugIn' in learners:
   if plugin_fit is None:
    # Retain a compact permanent export so later tiers never need deleted bodies.
    path=RUN/f'outputs/models/{kind}_{year}_Plugin_MC.json.gz';plugin_fit=load_model(path)
    if plugin_fit is None:plugin_fit=plugin.choose(model,kind,year);store_model(path,plugin_fit)
   models['PlugIn']=get('PlugIn',lambda:plugin_fit[gate][str(cost)])
  if 'SPI' in learners:models['SPI']=get('SPI',lambda:spi(z,cube,cost,models['Q_best'],lengths))
  for learner in learners:
   checkpoint();evaluate_cell(test,models[learner],gate,cost,learner);print('Committed',kind,year,gate,'c='+str(cost),learner,flush=True);count('real_strategy_fold_evaluations')
 del cube,contexts,test;gc.collect();return dict(task=task,cached=False,seconds=time.monotonic()-start,disk=budget_guard(True))
