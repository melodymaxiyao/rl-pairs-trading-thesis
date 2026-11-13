"""Small synthetic preparation tests; full real acceptance is main-only."""
from common import *
import numpy as np,pandas as pd
def tiny():
 z=np.array([0.,2.1,2.2,1.,0.,-2.1,-2.2,-.7,0.,2.2,1.,0.])
 return [dict(pair='A_B',year=2001,window='train',z=z.copy(),drift=-np.sign(z),strong=np.ones(len(z),np.int32),forward_bps=np.r_[np.diff(z)*10,0.],bins=np.searchsorted([-2,-1.5,-1,-.5,.5,1,1.5,2],z).astype(np.int32),dates=pd.bdate_range('2001-01-02',periods=len(z)),fold=1,test_year=2004,construction='F',category='all',origin='new',beta=np.ones(len(z)))]
def reproduction():
 stem='runtime/acceptance/real_reproduction';old=cached(stem)
 if old is not None:return old
 from sources import setup,training,frozen,risk,members,test_episode,MEM
 s=setup();rows=[]
 storage=load(SOURCE/'audit/repairs/bounded_storage_v4/storage_adapter.py','_tuned_storage_reader')
 for pool,y in [('P_EXP',2004),('P_QUAL',2005)]:
  es,pairs=training(pool,y);model=frozen(f'outputs/models/{pool}_{y}_TD8');r=risk(pool,y);selection=model['selected'];tables=[]
  for j in range(10):
   args=(es,[],selection['alpha'],selection['gamma'],seed(y,j),r)
   out=s.tuned.fit(*args,False);np.testing.assert_array_equal(out['q'],model['models'][j]);tables.append(out['q'])
   if j==0:
    native=s.native.fit(*args)
    for key in ['q','curves','snapshots','visits']:np.testing.assert_array_equal(out[key],native[key])
  np.testing.assert_array_equal(np.asarray(tables).mean(axis=0),model['ensemble'])
  for rep in [0,1]:
   cell=frozen(f'runtime/cells/{pool}_{y}_TD8_N2_{rep}_primary');chunk=storage.unpack_chunk(frozen(cell['chunks'][0]));daily={(d['pair'],d['cost']):d for d in chunk['daily']}
   expected={(d['pair'],d['cost']):d for d in chunk['scores']};by_pair={r['pair']:r for r in members(pool,y)}
   chosen=sorted({p for p,c in daily})[:3];assert len(chosen)==3
   for pair in chosen:
    e,reason=test_episode(by_pair[pair],y,'N2',rep);assert reason is None
    p,d=s.native.policy(e,q=model['ensemble'],risk=risk(pool,y,'N2',rep));np.testing.assert_array_equal(p,daily[pair,10]['paths'][0]);np.testing.assert_array_equal(e['forward_bps'],daily[pair,10]['forward'])
    for cost in COSTS:
     sign=frozen(f'runtime/plans/{pool}_{y}_N2_{rep}_primary')['signs'][pair]
     actual,_=s.execution.stats(e,[p],[d],cost,sign)
     for key in ['net_bps','gross_bps','n_trades','drift_bps','timing_bps','other_bps']:assert abs(actual[key]-expected[pair,cost][key])<=1e-9,(pool,y,pair,key)
    MEM.clear()
  rows.append(dict(pool=pool,test_year=y,final_tables_exact=10,ensemble_exact=True,null_replicates=2,null_pairs_per_replicate=3))
 return save(stem,dict(status='PASS',fixtures=rows))
def workers():
 stem='runtime/acceptance/workers';old=cached(stem)
 if old is not None:return old
 from jobs import fixture_worker
 from concurrent.futures import ProcessPoolExecutor
 import multiprocessing as mp
 jobs=[111,222];single=[fixture_worker(j) for j in jobs]
 with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as ex:parallel=list(ex.map(fixture_worker,jobs))
 assert single==parallel
 # Re-reading a fully committed checkpoint is independent of execution order.
 save('runtime/acceptance/resume_fixture',dict(values=single));assert cached('runtime/acceptance/resume_fixture')['values']==parallel
 return save(stem,dict(status='PASS',workers_1_equals_2=True,resume_identical=True))
def accounting():
 stem='runtime/acceptance/accounting';old=cached(stem)
 if old is not None:return old
 from sources import setup,frozen
 source=setup().sources.original_inference();f=pd.read_csv(note(SOURCE/'outputs/tables/dsr_pbo.csv'))
 for r in f[f.kind.eq('DSR')].to_dict('records'):
  if not np.isfinite(r.get('DSR',np.nan)):continue
  out=source.dsr(r['SR'],int(r['T']),r['skewness'],r['kurtosis'],r['cross_trial_SR_variance'],int(r['N']));assert abs(out['DSR']-r['DSR'])<=1e-12
 c=frozen('runtime/accounting/CSCV_GLOBAL');out,_=source.pbo(c['annual_matrix']);assert abs(out['PBO']-c['summary']['PBO'])<=1e-12;count('CSCV_splits',out['n_splits'])
 return save(stem,dict(status='PASS',DSR_exact=True,PBO_exact=True,splits=out['n_splits']))
