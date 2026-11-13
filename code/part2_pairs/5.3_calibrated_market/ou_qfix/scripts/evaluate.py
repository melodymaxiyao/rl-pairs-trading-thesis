"""Same independent truth sample; additional common-history MC Q diagnostics."""
from common import *
from reproduction import point,source_world,equal
from benchmarks import observations,model,statistics
from observed import execute,policy,al,summary
from qkernel import mc_q,entries
import gc
def get_truth(data,bits,c):
 stem=f'runtime/checkpoints/truthQ_c{c}_bits{bits}';old=cached(stem)
 if old is not None:return old
 out=mc_q(data,bits,c);save(stem,out);checkpoint();return out
def evaluate(results):
 completed=cached('runtime/checkpoints/evaluation')
 if completed is not None:return completed
 source=synthetic();select=source.cache('runtime/benchmarks/selection_best');oldrows=source.cache('runtime/point_results/best')['rows'];oldmap={(r['world'],r['cost'],r['learner']):r for r in oldrows}
 truth=observations(model(point()),50000,['truth']);refs={c:execute(truth,policy(select['bits'][str(c)]),c)[:,0] for c in [5,10]};bo=execute(truth,al(),0);baselines={c:bo[:,1]-c*bo[:,2] for c in [5,10]};cache={};entry_cache={};rows=[];bias=[];entryrows=[]
 for result in results:
  w,c=result['world'],result['cost'];sourcecost=source_world(w,c);pols={k:v for k,v in sourcecost['policies'].items() if k in ['AL rule','Band-train','Oracle-8']}
  pols.update({name:policy(m['bits']) for name,m in result['models'].items()})
  for name,pol in pols.items():
   key=(c,json.dumps(pol,sort_keys=True))
   if key not in cache:
    cache[key]=statistics(execute(truth,pol,0),pol,c,refs[c],baselines[c]);entry_cache[key]=entries(truth,pol)
   stat=cache[key];ent=entry_cache[key];flat=ent['all_flat_days'];eligible=ent['eligible_flat_days'];entrybins=[s for s in range(4) if ent['entries'][s]>0]
   if name in ['AL rule','Band-train','Oracle-8','Q_POW']:
    frozen=oldmap[w,c,name]
    for a,b in [('net','true_value'),('turnover','turnover'),('holding_period','holding_period')]:equal(stat[a],frozen[b],f'evaluation.{w}.{c}.{name}.{a}')
   row=dict(world=w,cost=c,learner=name,true_value=stat['net'],true_value_MC_SE=stat['net_SE'],turnover=stat['turnover'],holding_period=stat['holding_period'],gross_value=stat['gross'],breakeven_cost=stat['breakeven'],gross_exposure=stat['gross_exposure'],regret_oracle=stat['regret_oracle'],gain_over_AL=stat['gain_over_AL'],entry_bins=json.dumps(entrybins),number_entry_bins=len(entrybins),entries_per_series_year=ent['all_entries']/50000,entry_rate_given_flat=ent['all_entries']/flat if flat else np.nan,entry_rate_given_eligible_flat=ent['all_entries']/eligible if eligible else np.nan,entry_rate_per_session=ent['all_entries']/ent['all_days'],truth_series_years=50000,policy=pol)
   rows.append(row)
   for s in range(4):entryrows.append(dict(world=w,cost=c,learner=name,state=s,abs_s_bin=['[0,.5)','[.5,1.5)','[1.5,2.5)','[2.5,infinity)'][s],eligible_flat_opportunities=int(ent['opportunities'][s]),realised_entries=int(ent['entries'][s]),entry_share_given_bin=float(ent['entries'][s]/ent['opportunities'][s]) if ent['opportunities'][s] else np.nan,entry_share_of_all_entries=float(ent['entries'][s]/ent['all_entries']) if ent['all_entries'] else np.nan,entry_rate_given_flat=row['entry_rate_given_flat'],entry_rate_given_eligible_flat=row['entry_rate_given_eligible_flat'],entry_rate_per_session=row['entry_rate_per_session']))
   if name not in result['models']:continue
   m=result['models'][name];variants={'deployed':m['bits']}
   if m['raw_greedy_bits']!=m['bits']:variants['unrestricted_greedy']=m['raw_greedy_bits']
   for continuation,bits in variants.items():
    actual=get_truth(truth,bits,c);trueq=actual['q'];learned=np.asarray(m['q']);se=np.asarray(m['se'])
    for s in range(8):
     for a in range(2):bias.append(dict(world=w,cost=c,learner=name,continuation=continuation,state=s,action=a,state_kind='flat' if s<4 else 'holding',learned_Q=float(learned[s,a]),true_Q=float(trueq[s,a]),bias=float(learned[s,a]-trueq[s,a]),true_Q_MC_SE=float(actual['MC_SE'][s,a]),reference_observations=int(actual['observations'][s,a]),nominal_TD_target_SE=float(se[s,a]),learned_advantage=float(learned[s,1]-learned[s,0]),true_advantage=float(trueq[s,1]-trueq[s,0]),advantage_bias=float((learned[s,1]-learned[s,0])-(trueq[s,1]-trueq[s,0])),LCB_threshold=float(1.645*np.hypot(se[s,0],se[s,1])) if s<4 and m['entry_LCB'] else 0.,policy_bits=bits,SE_label=m['SE_label']))
  print('Truth/entry/Q diagnostics committed in memory',w,c,flush=True);checkpoint()
 del truth;gc.collect()
 # Fresh tests reproduce the source streams; no episode arrays are persisted.
 for w in range(20):
  test=observations(model(point()),5000,['test',w])
  for row in rows:
   if row['world']!=w:continue
   st=summary(execute(test,row['policy'],row['cost']));row.update(test_net=st['net'],test_net_SE=st['net_SE'])
   if row['learner'] in ['AL rule','Band-train','Oracle-8','Q_POW']:equal(st['net'],oldmap[w,row['cost'],row['learner']]['test_net'],f'final-test.{w}.{row["cost"]}.{row["learner"]}')
  del test;gc.collect();checkpoint()
 return save('runtime/checkpoints/evaluation',dict(rows=rows,bias=bias,entries=entryrows,reference='original50000-series-year truth and5000-series-year/world test; same streams; no new selection sample'))
