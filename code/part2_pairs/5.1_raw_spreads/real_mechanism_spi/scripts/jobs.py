"""One deterministic pool/fold job, durable per-cell/seed/world checkpoints."""
from common import *
from real_sources import setup,training,members,risk,plan,tuned_model,test_episode,reset_contexts
from switches import gate,execute
import numpy as np
CELLS=['TD8_C01g1','SPI_all','SPI_TD']
def fit_model(pool,y,cell='TD8_C01g1',es=None):
 stem=f'outputs/models/{pool}_{y}_{cell}';old=cached(stem)
 if old is not None:return old
 if es is None:es,_,_=training(pool,y)
 assert all(e['window']=='train' for e in es)
 step='C01' if cell=='TD8_C01g1' else 'CAP';fits=[]
 for j in range(10):
  key=f'runtime/fits/{pool}_{y}_{cell}_{j}';f=cached(key)
  if f is None:
   f=setup().kernel.fit(es,[],step,1.,seed(y,j),risk(pool,y));save(key,f)
  fits.append(f);checkpoint_guard()
 models=np.array([f['q'] for f in fits]);value=dict(ensemble=models.mean(axis=0),models=models,visits=np.array([f['visits'] for f in fits]),
  diagnostics=np.array([f['diagnostics'] for f in fits]),epochs=[f['meta']['epochs'] for f in fits],
  trajectories=np.array([f['trajectory'] for f in fits],dtype=object) if y==2004 else None,
  selected=dict(step=step,gamma=1.,inner_selection=False,seeds=[seed(y,j) for j in range(10)]),training_world='real',training_cost=10,test_rows_used=0)
 return save(stem,value)
def gates(pool,y,world,rep):
 all_ready={c:cached(f'runtime/gates/{pool}_{y}_{c}_{world}_{rep}') for c in ['SPI_all','SPI_TD']}
 if all(v is not None for v in all_ready.values()):return all_ready
 es,weights,missing=training(pool,y,world,rep);r=risk(pool,y,world,rep);proposal=setup().policies.propose(tuned_model(pool,y)['ensemble'])
 for cell,mask in [('SPI_all',1023),('SPI_TD',proposal)]:
  if all_ready[cell] is not None:continue
  tests,deployed=gate(es,weights,r,mask,world)
  value=dict(tests=tests,deployed=deployed,unavailable_training_members=missing,train_years=list(range(y-3,y)),test_rows_used=0)
  all_ready[cell]=save(f'runtime/gates/{pool}_{y}_{cell}_{world}_{rep}',value);checkpoint_guard()
 return all_ready
def base_row(pool,y,row,world,rep,cell,stem):
 return dict(experiment=HERE.name,status='development',development_only=True,period='2004-2025',pool=pool,test_year=y,fold=y-2003,source_fold=source_fold(y),
  pair=row['pair'],leg_A=row['leg_A'],leg_B=row['leg_B'],relation=row['sector_relation'],weight=float(row['weight']),variant='primary',scenario='primary',world=world,replicate=rep,
  cell=cell,method=cell,anchor='A_TR',alignment='aligned',construction='fixed_beta',train_years=','.join(map(str,range(y-3,y))),
  era=era(y),available=True,category=row.get('category','all'),origin=row.get('origin','new'),method_family='TD' if cell.startswith('TD8') else 'SPI',state='TD8',source_file=str((HERE/(stem+'.pkl.gz')).relative_to(ROOT)))
def run_fold(task):
 pool,y,tier=task;assert tier==1,'Only Tier 1 is authorized by addendum 001'
 mark=f'runtime/folds/tier{tier}_{pool}_{y}';old=cached(mark)
 if old is not None:return dict(pool=pool,year=y,seconds=old['seconds'],resumed=True,timing=old['timing'])
 t0=time.monotonic();s=setup();model=fit_model(pool,y);fit_seconds=time.monotonic()-t0;t0g=time.monotonic()
 # Freeze every gate before generating any full/test context.
 gs={(w,rep):gates(pool,y,w,rep) for w,rep in WORLDS};gate_seconds=time.monotonic()-t0g
 tuned=tuned_model(pool,y);q=tuned['ensemble'];ungated=s.policies.propose(q);rows=members(pool,y);chunks=[];te=time.monotonic()
 for world,rep in WORLDS:
  p=plan(pool,y,world,rep);r=p['risks']['A_TR']['final']
  bits={c:{v['cost']:v['deployed_bits'] for v in gs[world,rep][c]['deployed'] if v['primary_lag']} for c in ['SPI_all','SPI_TD']}
  for offset in range(0,len(rows),100):
   names={c:f'runtime/results/{pool}_{y}_{c}_{world}_{rep}_{offset:05d}' for c in CELLS};names['L0_plus_TD']=f'runtime/checks/{pool}_{y}_L0_plus_TD_{world}_{rep}_{offset:05d}'
   chunks.extend(names.values())
   if all(cached(v) is not None for v in names.values()):continue
   scores={c:[] for c in names};port={c:{} for c in names}
   for row in rows[offset:offset+100]:
    e,reason=test_episode(row,y,world,rep)
    if e is None:
     for cell in names:
      for cost in COSTS:scores[cell].append(dict(base_row(pool,y,row,world,rep,cell,names[cell]),cost=cost,cost_case=str(cost),available=False,unavailable_reason=reason,net_bps=np.nan,gross_bps=np.nan,n_trades=np.nan))
     reset_contexts();continue
    direct=s.native.policy(e,q=q,risk=r);count('executor_calls');count('mapping_checks')
    if world!='real':count('null_executions')
    mapped=execute(e,ungated,r,world);np.testing.assert_array_equal(direct[0],mapped[0])
    td=s.native.policy(e,q=model['ensemble'],risk=r);count('executor_calls')
    if world!='real':count('null_executions')
    for cost in COSTS:
     paths={'TD8_C01g1':td,'L0_plus_TD':mapped}
     for cell in ['SPI_all','SPI_TD']:paths[cell]=execute(e,bits[cell][cost],r,world)
     for cell,(path,details) in paths.items():
      result,d=s.execution.stats(e,[path],[details],cost,p['signs'][row['pair']])
      scores[cell].append(dict(base_row(pool,y,row,world,rep,cell,names[cell]),cost=cost,cost_case=str(cost),cost_bps=cost,applied_cost_bps=cost,cost_bps_total=result['cost_bps'],trades_per_year=result['n_trades'],**{k:v for k,v in result.items() if k!='cost_bps'}))
      a=port[cell].setdefault(cost,dict(dates=d['dates'],weight=0.,net=np.zeros(len(path)),gross=np.zeros(len(path)),fee=np.zeros(len(path)),turnover=np.zeros(len(path))))
      assert np.array_equal(a['dates'],d['dates']);a['weight']+=row['weight']
      for key in ['net','gross','fee','turnover']:a[key]+=row['weight']*d[key]
    reset_contexts()
   for cell,stem in names.items():
    if cached(stem) is None:save(stem,dict(scores=scores[cell],portfolios=port[cell]))
   checkpoint_guard()
  for cell in CELLS:save(f'runtime/cells/{pool}_{y}_{cell}_{world}_{rep}',dict(chunks=[v for v in chunks if f'_{cell}_{world}_{rep}_' in v],complete=True))
 seconds=time.monotonic()-t0;timing=dict(fit=fit_seconds,gates=gate_seconds,test=time.monotonic()-te)
 save(mark,dict(pool=pool,year=y,tier=tier,chunks=chunks,seconds=seconds,timing=timing))
 print('Completed pool/fold',pool,y,'Tier',tier,'seconds',round(seconds,1),flush=True)
 return dict(pool=pool,year=y,seconds=seconds,timing=timing,resumed=False)
