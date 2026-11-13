"""Descriptive equal-year estimates and preregistered shared year-block bootstrap."""
import engine as e
import numpy as np,json,hashlib,datetime,shutil
P=e.P
allrows=[];checks=[];signals=[];folds=[]
for year in range(2004,2026):
 for family in ['RP','RF']:
  p=P/f'folds/{family}_{year}.json';assert p.exists(),('Missing completed fold',family,year);r=e.read(p);assert r['preregistration_sha256']==e.sha(P/'preregistration.md');allrows+=r['rows'];checks+=r['baseline_checks'];signals+=r['signal_checks'];folds.append(r)
assert len(checks)==88 and len(signals)==88 and all(r['passes']for r in signals)
e.U.csvout(P/'yearly.csv',allrows)
rng=np.random.default_rng(20261026);draws=rng.integers(0,22,size=(5000,22));result=[]
def stat(x):
 x=np.asarray(x,float);assert x.shape==(22,)and np.isfinite(x).all();b=x[draws].mean(axis=1);return float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975)),float(np.mean(b<=0))
for family in ['RP','RF']:
 for unit in ['AL_K_main','Band_K_main','AL_K_intersection']:
  rows=sorted([r for r in allrows if r['family']==family and r['unit']==unit and r['world']=='real'],key=lambda r:r['year']);assert len(rows)==22
  out=dict(family=family,unit=unit,cost_bps_per_side=5,n_years=22,classification='exploratory robustness; descriptive',annualization='equal mean of calendar-year sums')
  for key,col in [('net_bps','net_bps'),('gross_bps','gross_bps'),('annual_K2','annual_K2'),('net_difference_bps','net_difference_bps'),('gross_difference_bps','gross_difference_bps')]:
   value,lo,hi,p=stat([r[col]for r in rows]);out.update({key:value,key+'_ci_lo':lo,key+'_ci_hi':hi,key+'_p_one_sided':p})
  out.update(baseline_learner=rows[0]['baseline_learner'],baseline_net_bps=np.mean([r['baseline_net_bps']for r in rows]),baseline_gross_bps=np.mean([r['baseline_gross_bps']for r in rows]),valid_series_days=sum(r['valid_series_days']for r in rows),original_AL_valid_series_days=sum(r['original_AL_valid_series_days']for r in rows),common_valid_series_days=sum(r['common_valid_series_days']for r in rows),original_only_valid_series_days=sum(r['original_only_valid_series_days']for r in rows),baseline_band_0bps_reselected='',baseline_band_0bps_status='NOT_RUN: no 0bps re-selection; gross uses the same5bps trades',bootstrap_seed=20261026,bootstrap_draws=5000,Holm='not applied')
  if unit=='AL_K_main':
   null=sorted([r for r in allrows if r['family']==family and r['unit']==unit and r['world']=='CLEAN_null_r0_original_N2_marks'],key=lambda r:r['year']);assert len(null)==22
   val,lo,hi,p=stat([r['net_bps']-n['net_bps']for r,n in zip(rows,null)]);out.update(real_minus_null_net_bps=val,real_minus_null_ci_lo=lo,real_minus_null_ci_hi=hi,real_minus_null_p_one_sided=p,null_definition='saved CLEAN_null r0 signal with original G0 N2 execution mark convention (carry alpha unflipped)')
  result.append(out)
e.U.csvout(P/'results.csv',result);e.U.csvout(P/'audit/baseline_exact_checks.csv',checks);e.U.csvout(P/'audit/K_regeneration_checks.csv',signals)
# All originally read input digests, including preflight, current test and scientific runs.
H=dict(e.H)
if (P/'audit/trading_inputs.json').exists():
 for k,v in e.U.js(P/'audit/trading_inputs.json').items():
  if k in H:assert H[k]==v
  H[k]=v
for k,v in e.U.INPUTS.items():
 if not str(k).startswith(str(P)):
  if k in H:assert H[k]==v
  H[k]=v
# Do not include mutable own outputs as historical scientific inputs.
H={k:v for k,v in H.items()if not str(k).startswith(str(P))or k==str(P/'preregistration.md')}
e.U.dump(P/'input_hashes_before.json',H);after={}
for k,v in H.items():e.raw(k);after[k]=e.H[k];assert after[k]==v,('Changed source after execution',k)
e.U.dump(P/'input_hashes_after.json',after)
e.U.dump(P/'audit/final_verification.json',dict(folds=44,baseline_exact_checks=88,K_regeneration_checks=88,max_K_error=max(r['maximum_absolute_error']for r in signals),inputs_verified=len(H),unchanged_inputs=True,no_new_Q_fits=True,no_new_MLE=True,new_real_trials=6,cumulative_real_trials=592,DSR_changed=False,mask='main CLEAN; sensitivity CLEAN intersect OU',all_statistics_descriptive=True))
print('AGGREGATION_COMPLETE',len(result),'rows',len(allrows),'yearly rows',len(H),'inputs',flush=True)
for r in result:print(r['family'],r['unit'],'net',r['net_bps'],'gross',r['gross_bps'],'netdiff',r['net_difference_bps'],flush=True)
