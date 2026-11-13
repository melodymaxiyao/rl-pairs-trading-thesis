"""Main-only exact reproduction before any new production strategy fit."""
from common import *
from real_sources import setup,training,tuned_model,members,test_episode,risk,plan,reset_contexts
from switches import execute
import numpy as np,pandas as pd
def real_reproduction():
 checkpoint_guard()
 old=cached('runtime/acceptance/real_reproduction')
 if old is not None:return old
 from references import pairs
 refs=pairs();s=setup();proofs=[]
 # Three fixed 2004 folds meet the >=2 pool/fold test and supply the requested
 # diagnostic Q trajectories. Production always reuses sealed tuned tables.
 for pool in POOLS:
  y=2004;es,weights,missing=training(pool,y);assert not missing;tuned=tuned_model(pool,y);fits=[]
  assert sorted(weights)==tuned['training_pairs']
  for j in range(10):
   stem=f'runtime/acceptance/pow_reproduction/{pool}_{y}_{j}';f=cached(stem)
   if f is None:f=save(stem,s.kernel.fit(es,[],'POW',1.,seed(y,j),risk(pool,y)))
   np.testing.assert_array_equal(f['q'],tuned['models'][j]);fits.append(f);checkpoint_guard()
   original=source_pickle_at(TUNED/f'runtime/fits/{pool}_{y}_{j}.pkl.gz')
   for key in ['q','curves','snapshots','visits']:np.testing.assert_array_equal(f[key],original[key])
   assert f['meta']['epochs']==original['meta']['epochs']
  np.testing.assert_array_equal(np.array([f['q'] for f in fits]).mean(axis=0),tuned['ensemble'])
  save(f'runtime/acceptance/tuned_trajectories/{pool}_{y}',dict(fits=fits,source='exact instrumented acceptance reproduction; not a production refit',production_uses='sealed TD8_tuned tables'))
  by_pair={r['pair']:r for r in members(pool,y)};checks=0
  for world,rep in WORLDS:
   base=refs.loc[refs.pool.eq(pool)&refs.test_year.eq(y)&refs.world.eq(world)&refs.replicate.eq(rep)&refs.cell.eq('R_TR_aln')&refs.cost.eq(10)&refs.available.eq(True)]
   chosen=base.sort_values('pair').head(3);assert len(chosen)==3
   for row in chosen.to_dict('records'):
    e,reason=test_episode(by_pair[row['pair']],y,world,rep);assert reason is None;r=risk(pool,y,world,rep);p=plan(pool,y,world,rep)
    l0=execute(e,0,r,world);direct=s.native.policy(e,entry=2.,risk=r);np.testing.assert_array_equal(l0[0],direct[0])
    mapped=execute(e,s.policies.propose(tuned['ensemble']),r,world);direct_td=s.native.policy(e,q=tuned['ensemble'],risk=r);np.testing.assert_array_equal(mapped[0],direct_td[0])
    for cost in COSTS:
     for cell,path in [('R_TR_aln',l0),('TD8_tuned',mapped)]:
      expected=refs.loc[refs.pool.eq(pool)&refs.test_year.eq(y)&refs.world.eq(world)&refs.replicate.eq(rep)&refs.cell.eq(cell)&refs.cost.eq(cost)&refs.pair.eq(row['pair'])]
      assert len(expected)==1;actual,_=s.execution.stats(e,[path[0]],[path[1]],cost,p['signs'][row['pair']])
      for key in ['net_bps','gross_bps','n_trades','drift_bps','timing_bps','other_bps']:
       assert abs(float(actual[key])-float(expected.iloc[0][key]))<=1e-9,(pool,y,world,rep,row['pair'],cell,cost,key)
      checks+=1
    reset_contexts();checkpoint_guard()
  proofs.append(dict(pool=pool,test_year=y,seed_tables_exact=10,L0_and_tuned_real_null_cost_checks=checks,training_weights_unchanged=True,null_refits=0))
 result=dict(status='PASS',fixtures=proofs,source_runtime_context_reads=0,source_runtime_cell_reads=0,new_production_fits=0)
 return save('runtime/acceptance/real_reproduction',result)
def statistics_reproduction():
 checkpoint_guard()
 old=cached('runtime/acceptance/statistics_reproduction')
 if old is not None:return old
 from references import pairs,baseline_accounting
 s=setup();source=s.sources.original_inference();f=csv_read(TUNED/'outputs/tables/dsr_pbo.csv');n=0
 for row in f.loc[f.kind.eq('DSR')&f.N.eq(544)].to_dict('records'):
  actual=source.dsr(row['SR'],int(row['T']),row['skewness'],row['kurtosis'],row['cross_trial_SR_variance'],544)
  assert abs(actual['DSR']-row['DSR'])<=1e-12;n+=1;count('DSR_evaluations')
 assert n==4
 account=baseline_accounting();out,_=source.pbo(account['annual_matrix']);assert out['PBO']==account['summary']['PBO'];count('CSCV_splits',out['n_splits']);checkpoint_guard()
 refs=pairs();base=refs.loc[refs.pool.eq('P_RAND')&refs.cost.eq(10)&refs.world.eq('real')]
 joined=s.inference.paired(base[base.cell.eq('TD8_tuned')],base[base.cell.eq('TD8')]);actual=s.inference.estimate(joined)
 contrast=csv_read(TUNED/'outputs/tables/contrasts.csv');expected=contrast.loc[contrast.pool.eq('P_RAND')&contrast.cost.eq(10)&contrast.era.eq('pooled')&contrast.id.eq('U1')&contrast.scheme.eq('year')];assert len(expected)==1
 for key in ['estimate','ci95_low','ci95_high','p_one_sided']:assert abs(actual[key]-expected.iloc[0][key])<=1e-9,(key,actual[key],expected.iloc[0][key])
 return save('runtime/acceptance/statistics_reproduction',dict(status='PASS',DSR_rows=n,PBO_exact=True,CSCV_splits=out['n_splits'],bootstrap_draws=5000,bootstrap_sealed_U1_exact=True))
