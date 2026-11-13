"""One seed stream per pool/fold, one real fit set, all executor worlds."""
from common import *
import numpy as np
from sources import setup,frozen,training,members,risk,test_episode,MEM
def fitted(pool,y):
 stem=f'outputs/models/{pool}_{y}_TD8_tuned';old=cached(stem)
 if old is not None:return old
 es,pairs=training(pool,y);s=setup();r=risk(pool,y);tables=[];meta=[]
 for j in range(10):
  fitstem=f'runtime/fits/{pool}_{y}_{j}';out=cached(fitstem)
  if out is None:
   out=s.tuned.fit(es,[],0.,1.,seed(y,j),r,True);save(fitstem,out)
  tables.append(out['q']);meta.append(out['meta'])
 return save(stem,dict(models=np.asarray(tables),ensemble=np.asarray(tables).mean(axis=0),feature=0,training_pairs=pairs,training_signature=s.io.arrays_hash(es),selected=dict(alpha='1/n(s,a)^0.7',gamma=1.,inner_selection=False,seeds=[seed(y,j) for j in range(10)]),convergence=meta,test_rows_used=0))
def run_fold(task):
 pool,y=task;started=time.monotonic();scope=os.environ.get('TUNED_SCOPE','production')
 if scope=='production':guard()
 marker=f'runtime/folds/{pool}_{y}';old=cached(marker)
 if old is not None:return dict(pool=pool,year=y,seconds=old['seconds'],resumed=True)
 s=setup();model=fitted(pool,y);rows=members(pool,y);proofs=[]
 for world,rep in WORLDS:
  r=risk(pool,y,world,rep);p=frozen(f'runtime/plans/{pool}_{y}_{world}_{rep}_primary')
  for offset in range(0,len(rows),100):
   stem=f'runtime/results/{pool}_{y}_{world}_{rep}_{offset:05d}';proofs.append(stem)
   if cached(stem) is not None:continue
   scores=[];port={}
   for row in rows[offset:offset+100]:
    e,reason=test_episode(row,y,world,rep)
    base=dict(experiment=HERE.name,status='development',period='2004-2025',pool=pool,test_year=y,fold=y-2003,source_fold=source_fold(y),pair=row['pair'],leg_A=row['leg_A'],leg_B=row['leg_B'],relation=row['sector_relation'],weight=float(row['weight']),variant='primary',scenario='primary',world=world,replicate=rep,anchor='A_TR',alignment='aligned',available=e is not None,era=era(y),development_only=True,construction='fixed_beta',train_years=','.join(str(v) for v in range(y-3,y)),source_file=str((HERE/(stem+'.pkl.gz')).relative_to(ROOT)))
    if e is None:
     for cell in CELLS:
      for cost in COSTS:scores.append(dict(base,cell=cell,method=cell,cost=cost,cost_case=str(cost),net_bps=np.nan,gross_bps=np.nan,n_trades=np.nan,unavailable_reason=reason))
     MEM.clear();continue
    paths=[s.native.policy(e,q=q,feature=0,aligned=True,risk=r) for q in [model['ensemble'],*model['models']]]
    if world!='real':count('null_executor_calls',len(paths))
    for cell,selected in [('TD8_tuned',paths[:1]),('TD8_tuned_z',paths[1:])]:
     for cost in COSTS:
      result,d=s.execution.stats(e,[v[0] for v in selected],[v[1] for v in selected],cost,p['signs'][row['pair']])
      scores.append(dict(base,cell=cell,method=cell,method_family='TD',state='TD8',cost=cost,cost_case=str(cost),cost_bps_total=result['cost_bps'],cost_bps=cost,trades_per_year=result['n_trades'],hit_rate=result.get('trade_hit_rate'),days_in_market=result.get('fraction_in_market'),**{k:v for k,v in result.items() if k!='cost_bps'}))
      key=(cell,cost);a=port.setdefault(key,dict(dates=d['dates'],weight=0.,net=np.zeros(len(d['net'])),gross=np.zeros(len(d['net'])),fee=np.zeros(len(d['net'])),turnover=np.zeros(len(d['net']))))
      assert np.array_equal(a['dates'],d['dates']);a['weight']+=base['weight']
      for k in ['net','gross','fee','turnover']:a[k]+=base['weight']*d[k]
    MEM.clear()
   save(stem,dict(scores=scores,portfolios=port))
  print('Completed executor',pool,y,world,rep,flush=True)
 seconds=time.monotonic()-started;save(marker,dict(pool=pool,year=y,chunks=proofs,seconds=seconds))
 print('Completed TD8_tuned',pool,y,'seconds',round(seconds,1),flush=True)
 return dict(pool=pool,year=y,seconds=seconds,resumed=False)
def fixture_worker(seed_value):
 from fixtures import tiny
 s=setup();out=s.tuned.fit(tiny(),[],0.,1.,seed_value,dict(H=12,stop=5.,lock=True),True,maxepochs=20,early=False)
 return hashlib.sha256(out['q'].tobytes()+out['visits'].tobytes()).hexdigest()
