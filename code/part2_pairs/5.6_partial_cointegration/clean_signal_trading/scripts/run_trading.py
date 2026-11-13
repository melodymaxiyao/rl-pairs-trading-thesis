"""Registered six-cell rule backtest; no new pooled fits or Q-learning."""
import engine as e
import argparse,time,gc,hashlib,os,json
import numpy as np
P=e.P
for rel in ['folds','logs','tests','runtime']:(P/rel).mkdir(exist_ok=True)
def trial(family,unit):
 p=P/'trial_register.csv';rows=e.U.rows(p)if p.exists()else[]
 if not any(r['family']==family and r['unit']==unit for r in rows):
  rows.append(dict(family=family,unit=unit,cost_bps=5,started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),real_trial=True,cumulative_after=586+len(rows)+1,classification='exploratory robustness',DSR_changed=False));e.U.csvout(p,rows)
def one(family,year):
 p=P/f'folds/{family}_{year}.json'
 if p.exists():
  cached=e.U.js(p);assert cached['preregistration_sha256']==e.sha(P/'preregistration.md');return cached
 start=time.monotonic();space_before=__import__('shutil').disk_usage(P).free;e.checkpoint();print('START',family,year,flush=True)
 data,signals,sigchecks=e.signal_history(family,year);print('K_REGEN_PASS',family,year,sigchecks,flush=True)
 ctx,contexts=e.original_contexts(family,year);baseline,proofs=e.baseline_check(ctx,year,family);print('ORIGINAL_AL_BAND_EXACT_PASS',family,year,flush=True)
 z,cohort=e.clean_training(data,signals[0],contexts);cube=e.d.reward_cube(contexts)[cohort];bp=e.band(z,cube,5);del cube,z;gc.collect();print('BAND_THRESHOLD',family,year,bp['enter'],bp['exit'],flush=True)
 k=signals[0][:,data['form']:];null=signals[1][:,data['form']:];old=np.isfinite(e.d.scores(ctx,'G0',0));effective_old=old.copy();effective_old[:,-1]=False;rows=[];daily={}
 for unit,pol,s in [('AL_K_main',dict(kind=1,enter=1.25,exit=.5),k),('Band_K_main',bp,k),('AL_K_intersection',dict(kind=1,enter=1.25,exit=.5),np.where(old,k,np.nan))]:
  trial(family,unit);v=e.execute(ctx,pol,s);valid=np.isfinite(s);valid[:,-1]=False;b=baseline['Band'if unit=='Band_K_main'else'AL'];rows.append(e.summarize_run(ctx,v,unit,year,family,b,valid,effective_old));daily[unit]=v['daily'];print('DONE',family,year,unit,'validdays',rows[-1]['valid_series_days'],flush=True)
 v=e.execute(ctx,dict(kind=1,enter=1.25,exit=.5),null,wi=1);valid=np.isfinite(null);valid[:,-1]=False;rows.append(e.summarize_run(ctx,v,'AL_K_main',year,family,baseline['AL'],valid,effective_old,world='CLEAN_null_r0_original_N2_marks'));daily['AL_K_null_r0']=v['daily']
 root=P/f'runtime/{family}_{year}';root.mkdir(exist_ok=True)
 for unit,rr in daily.items():e.U.csvout(root/f'{unit}_daily.csv',rr)
 result=dict(family=family,year=year,rows=rows,baseline_checks=proofs,signal_checks=sigchecks,band_policy=bp,seconds=time.monotonic()-start,disk_delta_bytes=space_before-__import__('shutil').disk_usage(P).free,preregistration_sha256=e.sha(P/'preregistration.md'))
 e.U.dump(p,result);e.persist_inputs();print('FOLD_COMPLETE',family,year,'seconds',result['seconds'],flush=True);return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--family',choices=['RP','RF']);a.add_argument('--year',type=int,default=2004);a.add_argument('--all',action='store_true');arg=a.parse_args()
 try:
  if arg.all:
   for year in range(2004,2026):
    for family in ['RP','RF']:one(family,year)
  else:one(arg.family or'RP',arg.year)
 except BaseException as exc:
  e.persist_inputs();e.U.dump(P/'audit/STOPPED.json',dict(reason=type(exc).__name__,message=str(exc),stage='no publication on failed verification',time=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()));print('STOPPED',type(exc).__name__,str(exc),flush=True);raise
