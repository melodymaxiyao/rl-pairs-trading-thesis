import common7 as S
S.guard()
import ast,sys,hashlib,time
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0,str(S.R))
import fyp_rl.data as FD

def blocked(*a,**k):raise RuntimeError('Downloads blocked; restored cache only')
FD.yf.download=blocked
from fyp_rl.metrics import sharpe_ratio
T=['SPY','XLU','KO','IWM','NVDA','TSLA','GOOGL'];H=[5,10,20];KEY={5:7,10:14,20:27}
ROOT=S.R/'experiments/turning_point_detection/new_stage1'
def original_functions(path,names,namespace):
 tree=ast.parse(S.raw(path).decode());selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
 assert {n.name for n in selected}==set(names)
 code=compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec',flags=__import__('__future__').annotations.compiler_flag)
 exec(code,namespace);return namespace
NS={'np':np,'pd':pd,'hashlib':hashlib}
original_functions(ROOT/'kernel_smoothing.py',['local_linear_slope_weights','_fixed_weight_slope','bilateral_local_linear_slope'],NS)
original_functions(ROOT/'slope_accuracy.py',['reference_events'],NS)
original_functions(ROOT/'slope_accuracy_diagnostics.py',['null_seed','sign_flip_returns','empirical_null_path'],NS)
def imports():
 for m in list(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and Path(p).is_relative_to(S.R) and Path(p).suffix=='.py':S.raw(p)
def load():
 cfg=S.js(ROOT/'configs/slope_accuracy.json');supp=S.js(ROOT/'configs/slope_accuracy_diagnostics.json')
 assert cfg['seed']==supp['seed']==12345 and supp['null_repetitions']==100 and cfg['reference_radius_multiplier']==3
 manifests=[S.js(S.R/'MANIFEST.json'),S.js(S.R/'MANIFEST_ADDENDUM_001.json')]
 expected={v['relative_path']:v['sha256'] for m in manifests for v in m.get('files',[])}
 def auth(p):
  h=S.sha(p);rel=str(p.relative_to(S.R));assert rel in expected and h==expected[rel],('restoration digest',p)
 for rel in ['tables/fold_calendar.csv','tables/reference_turns.csv','tables/null_samples_diagnostics.csv']:
  auth(ROOT/rel)
 cal=S.frame(ROOT/'tables/fold_calendar.csv');ref=S.frame(ROOT/'tables/reference_turns.csv');ref['date']=pd.to_datetime(ref.date)
 saved=S.frame(ROOT/'tables/null_samples_diagnostics.csv');saved=saved[(saved.source=='B1')&saved.ticker.isin(T)]
 assert len(saved)==6300 and set(saved['repeat'])==set(range(100))
 expected_counts={(int(z['repeat']),z['ticker'],int(z['h_ref']),z['type']):int(z['n_ref']) for z in saved.to_dict('records')}
 assert len(expected_counts)==6300
 price={};folds={}
 for t in T:
  p=S.R/f'data/raw/{t.lower()}_daily_adjusted_ohlcv.csv';auth(p)
  dates=pd.DatetimeIndex(S.frame(p).iloc[:,0]);start=max(dates.min(),pd.Timestamp(cfg['tsla_min_train_start'])) if t=='TSLA' else dates.min();end=min(dates.max(),pd.Timestamp(cfg['as_of']))
  px=FD.load_ohlcv(t,str(start.date()),str(end.date()),raw_dir=S.R/'data/raw');price[t]=pd.Series(np.log(px['Close']),name='log_adjusted_close')
  folds[t]=cal[cal.ticker==t].sort_values('test_start').to_dict('records')
  assert len(folds[t])==(11 if t=='TSLA' else 16)
 imports()
 # Every source with a restoration hash must retain it; sources lacking old sealed code hashes are explicitly only restoration provenance.
 for p,v in list(S.INPUTS.items()):
  p=Path(p)
  if p.is_relative_to(S.R) and str(p.relative_to(S.R)) in expected:assert v['sha256']==expected[str(p.relative_to(S.R))]
 part2key=S.frame(S.L/'lagged_oracle_key.csv');S.raw(S.L/'complete_final.json');S.raw(S.L/'manifest_final.json')
 for t in T:S.raw(S.L/f'BH_daily_{t}.csv.gz')
 return price,folds,ref,expected_counts,part2key

def make_refs(path):return {h:NS['reference_events'](NS['bilateral_local_linear_slope'](path,bandwidth_h=h,radius_multiplier=3).to_numpy()) for h in H}
def bound(index,f,h,segment,v2=False):
 lo=int(index.searchsorted(pd.Timestamp(f[segment+'_start'])));end=int(index.searchsorted(pd.Timestamp(f[segment+'_end']),side='right')-1)
 hi=min(end,len(index)-1-3*h) if v2 and segment=='test' else end-3*h
 return lo,hi

def subset(events,a,b):return events[(events[:,0]>=a)&(events[:,0]<=b)]
def fold_events(events,index,f,h):
 return np.concatenate([subset(events,*bound(index,f,h,s)) for s in ['train','test']],axis=0)
def real_reference_check(price,folds,ref,allrefs):
 checked=0
 for t in T:
  index=price[t].index
  for h in H:
   for f in folds[t]:
    for segment in ['train','test']:
     e=subset(allrefs[t][h],*bound(index,f,h,segment))
     old=ref[(ref.ticker==t)&(ref.h_ref==h)&(ref.fold_id==f['fold_id'])&(ref.segment==segment)].sort_values('date')
     desired=np.column_stack([index.get_indexer(old.date),np.where(old.type=='trough',1,-1)]).astype(int)
     np.testing.assert_array_equal(e,desired,err_msg=f'real reference {t}/{h}/{f["fold_id"]}/{segment}')
     checked+=len(e)
 return checked

def validate_counts(rep,t,index,folds,refs,expected):
 rows=[]
 for h in H:
  stitched=np.concatenate([subset(refs[h],*bound(index,f,h,'test',v2=True)) for f in folds])
  for kind,sign in [('peak',-1),('trough',1),('combined',0)]:
   n=len(stitched) if sign==0 else int(np.sum(stitched[:,1]==sign));old=expected[(rep,t,h,kind)]
   assert n==old,('B1 reference count mismatch',rep,t,h,kind,n,old)
   rows.append({'repeat':rep,'ticker':t,'h_ref':h,'type':kind,'regenerated_n_ref':n,'sealed_n_ref':old,'match':True,'INFEASIBLE':True})
 return rows

def signal_targets(index,folds,refs,h,lags=np.arange(41)):
 begin=int(index.searchsorted(pd.Timestamp(folds[0]['test_start'])))-1;end=int(index.searchsorted(pd.Timestamp(folds[-1]['test_end']),side='right'))
 assert begin>=0
 w=np.full((len(lags),end-begin),np.nan)
 for f in folds:
  e=fold_events(refs,index,f,h);assert len(e)>0
  a=int(index.searchsorted(pd.Timestamp(f['test_start'])));b=int(index.searchsorted(pd.Timestamp(f['test_end']),side='right'))
  dates=np.arange(a,b)
  for j,lag in enumerate(lags):
   loc=e[:,0]+lag;valid=loc<len(index);loc=loc[valid];sgn=e[valid,1]
   p=np.searchsorted(loc,dates,side='right')-1
   w[j,a-begin:b-begin]=np.where(p>=0,sgn[np.maximum(p,0)]==1,False).astype(float)
   prev=np.searchsorted(loc,a-1,side='right')-1
   w[j,a-1-begin]=float(sgn[prev]==1) if prev>=0 else 0.
 assert np.isfinite(w).all()
 return w,begin,end

def close_returns(logp,w,cost=.001):
 # Targets at d earn interval d->d+1; fee belongs to that same next interval.
 held=w[:,:-1];previous=np.concatenate([np.zeros((len(w),1)),held[:,:-1]],axis=1)
 gross=1+held*np.expm1(np.diff(np.asarray(logp)))[None,:]
 return (1-cost*np.abs(held-previous))*gross-1

def sharpes(a):
 a=np.asarray(a);sd=a.std(axis=-1,ddof=0)
 return np.divide(np.sqrt(252)*a.mean(axis=-1),sd,out=np.zeros_like(sd),where=sd>0)
def curve_for(path,folds,refs):
 points=[]
 for h in H:
  w,a,b=signal_targets(path.index,folds,refs[h],h);x=path.iloc[a:b].to_numpy()
  r=close_returns(x,w);bh=close_returns(x,np.ones((1,len(x))))
  points.append(sharpes(r)-sharpes(bh)[0])
 return np.array(points)

def prepare():
 assert S.sha(S.P/'preregistration_addendum.md')=='9f074cf488c99b8484c76f5f2f01d83095051e7e5071e0f862095e270ff97125'
 S.resources();price,folds,ref,expected,part2key=load();allreal={t:make_refs(price[t]) for t in T}
 real_n=real_reference_check(price,folds,ref,allreal)
 S.dump(S.P/'audit/input_hashes_before.json',dict(S.INPUTS))
 counts=[];hashes=[]
 cache=S.P/'audit/reference_cache';cache.mkdir(exist_ok=True)
 for rep in range(100):
  S.resources();arrays={}
  for t in T:
   seed=NS['null_seed'](12345,'B1',rep,t);path,audit=NS['empirical_null_path'](price[t],'B1',seed);refs=make_refs(path)
   counts+=validate_counts(rep,t,path.index,folds[t],refs,expected)
   for h in H:arrays[t+'_'+str(h)]=refs[h]
   hashes.append({'repeat':rep,'ticker':t,'seed':seed,'path_float64_sha256':hashlib.sha256(path.to_numpy().tobytes()).hexdigest(),'n_prices':len(path),'start':str(path.index[0].date()),'end':str(path.index[-1].date()),'INFEASIBLE':True})
  np.savez_compressed(cache/f'replica_{rep:03d}_INFEASIBLE.npz',**arrays)
  if rep%10==9:print('INFEASIBLE B1 validation',rep+1,'/100 replicas; counts all exact so far',flush=True)
 pd.DataFrame(counts).to_csv(S.P/'audit/B1_reference_count_validation.csv.gz',index=False,compression='gzip')
 pd.DataFrame(hashes).to_csv(S.P/'audit/B1_path_hashes.csv',index=False)
 S.verify(S.js(S.P/'audit/input_hashes_before.json'))
 S.dump(S.P/'audit/validation.json',{'status':'PASS','INFEASIBLE':True,'B1_count_checks':len(counts),'B1_mismatches':0,'real_reference_events_exact':real_n,'outcomes_computed':False,'v2_count_mask':'test_end=min(year_end,last_price-3h)','v1_trading_mask':'same saved train/test reference bounds as Part2'})
 return price,folds,ref,expected,part2key

def md(records):
 f=pd.DataFrame(records);header='| '+' | '.join(f.columns)+' |';rule='| '+' | '.join(['---']*len(f.columns))+' |'
 return '\n'.join([header,rule]+['| '+' | '.join(str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)])

def label_cross(x):
 k=np.flatnonzero(np.asarray(x)<=0);return str(int(k[0])) if len(k) else '>40'
def quantile_discrete(values,q):
 v=sorted([float('inf') if z=='>40' else int(z) for z in values]);z=v[max(0,int(np.ceil(q*len(v)))-1)]
 return '>40' if not np.isfinite(z) else str(int(z))

def main():
 assert S.js(S.P/'audit/validation.json')['status']=='PASS'
 assert S.js(S.P/'audit/test_results.json')['passes']
 price,folds,ref,expected,part2key=load();before=S.js(S.P/'audit/input_hashes_before.json');S.verify(before)
 allreal={t:make_refs(price[t]) for t in T};real_reference_check(price,folds,ref,allreal)
 real=np.mean([curve_for(price[t],folds[t],allreal[t]) for t in T],axis=0)
 null=np.zeros((100,3,41));hashes=S.frame(S.P/'audit/B1_path_hashes.csv');hashmap={(z['repeat'],z['ticker']):z['path_float64_sha256'] for z in hashes.to_dict('records')}
 for rep in range(100):
  S.resources()
  with np.load(S.P/f'audit/reference_cache/replica_{rep:03d}_INFEASIBLE.npz') as cached:
   for t in T:
    path,_=NS['empirical_null_path'](price[t],'B1',NS['null_seed'](12345,'B1',rep,t))
    assert hashlib.sha256(path.to_numpy().tobytes()).hexdigest()==hashmap[(rep,t)]
    refs={h:cached[t+'_'+str(h)] for h in H};null[rep]+=curve_for(path,folds[t],refs)/7
  if rep%10==9:print('INFEASIBLE outcomes',rep+1,'/100 replicas',flush=True)
 quant=np.percentile(null,[5,50,95],axis=0)
 keys=[];curves=[];repvals=[];breakeven=[];dist=[];comparison=[]
 for i,h in enumerate(H):
  for lag in range(41):
   curves.append({'h_ref':h,'L':lag,'real':real[i,lag],'B1_q05':quant[0,i,lag],'B1_median':quant[1,i,lag],'B1_q95':quant[2,i,lag],'INFEASIBLE':True})
   repvals.extend({'h_ref':h,'L':lag,'repeat':rep,'delta_sharpe':null[rep,i,lag],'INFEASIBLE':True} for rep in range(100))
  lag=KEY[h];values=null[:,i,lag]
  keys.append({'h_ref':h,'L':lag,'F':3*h,'L_vs_F':'L < F; future prices still used','real':real[i,lag],'B1_median':quant[1,i,lag],'B1_q05':quant[0,i,lag],'B1_q95':quant[2,i,lag],'midrank_percentile':100*(np.sum(values<real[i,lag])+.5*np.sum(values==real[i,lag]))/100,'B1_positive_fraction':np.mean(values>0),'primary':h==10,'INFEASIBLE':True})
  br=[label_cross(null[rep,i]) for rep in range(100)]
  breakeven.append({'source':'real','repeat':-1,'h_ref':h,'L_star':label_cross(real[i]),'INFEASIBLE':True})
  breakeven.extend({'source':'B1','repeat':rep,'h_ref':h,'L_star':br[rep],'INFEASIBLE':True} for rep in range(100))
  dist.append({'h_ref':h,'real_L_star':label_cross(real[i]),'B1_q05':quantile_discrete(br,.05),'B1_median':quantile_discrete(br,.5),'B1_q95':quantile_discrete(br,.95),'B1_right_censored_fraction':br.count('>40')/100,'INFEASIBLE':True})
  old=part2key[part2key.h_ref==h].iloc[0]
  comparison.append({'h_ref':h,'L':lag,'real_close_to_close':real[i,lag],'Part2_next_open':float(old.delta_sharpe),'Part2_next_open_CI_low':float(old.ci_lo),'Part2_next_open_CI_high':float(old.ci_hi),'difference_close_minus_open':real[i,lag]-float(old.delta_sharpe),'reference_events':'identical Part2 v1 saved real events','INFEASIBLE':True})
 for name,records in [('part7_key.csv',keys),('part7_curve.csv',curves),('part7_replicas.csv.gz',repvals),('part7_breakeven.csv',breakeven),('part7_breakeven_distribution.csv',dist),('part7_real_execution_comparison.csv',comparison)]:
  pd.DataFrame(records).to_csv(S.P/name,index=False,float_format='%.17g',compression='gzip' if name.endswith('.gz') else None)
 row=keys[1];verdict='(a)' if row['real']>row['B1_q95'] else '(b)' if row['real']>=row['B1_q05'] else 'below 5%; no interpretation'
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True);checks=[]
 for i,(h,ax) in enumerate(zip(H,axes)):
  x=np.arange(41);line,=ax.plot(x,real[i],color='#2166ac',lw=1.6,label='Real');median,=ax.plot(x,quant[1,i],color='#555555',ls='--',lw=1.3,label='B1 median')
  ax.fill_between(x,quant[0,i],quant[2,i],color='#999999',alpha=.25,label='B1 5%–95%')
  ax.axvline(KEY[h],color='#b2182b',ls='--',lw=1,label=f'Lag = {KEY[h]}');ax.axvline(3*h,color='#888888',ls=':',lw=1,label=f'F = {3*h}')
  ax.axhline(0,color='#aaaaaa',lw=.7);ax.set_xlim(0,max(40,3*h)+1);ax.set_xlabel('L (trading sessions)');ax.legend(frameon=False,fontsize=7)
  np.testing.assert_array_equal(line.get_ydata(),real[i]);np.testing.assert_array_equal(median.get_ydata(),quant[1,i])
  from_table=pd.DataFrame(curves).query('h_ref == @h')
  np.testing.assert_array_equal(line.get_ydata(),from_table.real.to_numpy());np.testing.assert_array_equal(quant[0,i],from_table.B1_q05.to_numpy());np.testing.assert_array_equal(quant[2,i],from_table.B1_q95.to_numpy());assert ax.get_title()==''
  checks.append({'h_ref':h,'real_B1_median_band_equal_table':True,'no_internal_title':True})
 axes[0].set_ylabel('Equal-weight ΔSharpe');fig.tight_layout();fig.savefig(S.P/'part7_INFEASIBLE.png',dpi=300);plt.close(fig)
 S.dump(S.P/'audit/plot_table_tests.json',{'status':'PASS','checks':checks,'INFEASIBLE':True,'source_table_sha256':S.sha(S.P/'part7_curve.csv')})
 (S.P/'figures.md').write_text('# INFEASIBLE — Part7 figure\n\npart7_INFEASIBLE.png, 300dpi, three panels in h_ref order 5,10,20; solid real, dashed B1 median, shaded 5–95%. Key lags red dashed; future extent F dotted. No internal title or second y-axis. h_ref20 axis reaches60 to show F; curves only evaluated through40. Exact plot/table tests and source table hash: audit/plot_table_tests.json.\n')
 text='# Part 7 — INFEASIBLE（使用未来信息）\n\nPrimary registered verdict: '+verdict+'. No new real trading trials; cumulative 580. Part3 NOT_RUN; R1–R4 NOT_EVALUATED.\n\n'+md(keys)+'\n\n## Real close-to-close vs Part2 next-open\n\n'+md(comparison)+'\n\n## Breakeven distribution\n\n'+md(dist)+'\n\nReference future extent F=3*h_ref=15,30,60. Source: restored new_stage1/kernel_smoothing.py:82–102,130–137. All key lags are less than F; even L=40 remains inside F for h_ref20.\n\nB1 uses the exact sealed mean-restored sign-flip generator and 100 registered replicas0..99. Each B1 path has its own BH. Count validation:6300/6300 v2 peak/trough/combined cells exact; see audit/B1_reference_count_validation.csv.gz. Only historical counts, not exact B1 event arrays, were retained. Newly regenerated arrays and path hashes are retained. No detector selection or fitting.\n\nAs registered, v2 masks validate B1 generation against the v2 accuracy tables. Trading retains Part2 v1 reference masks for both real and B1, including each segment tail exclusion; this keeps the real comparison execution-only. Real saved references reproduced exactly. Initial state, year-boundary overrides and costs use the registered close-to-close formula. Daily returns before the first test day are excluded; initial entry cost is included. EW is the mean of seven per-ticker differences, not a rebalanced portfolio Sharpe.\n\nB1 5/50/95 quantiles are empirical replica-distribution summaries, not confidence intervals. The registered (b) label denotes compatibility and cannot establish that the null explanation is uniquely true. B1 preserves the full-series mean and volatility magnitudes; sign randomization is not a zero-drift martingale claim. Part2 next-open results and decisions remain unchanged.\n'
 (S.P/'README.md').write_text(text)
 (S.P/'open_questions.md').write_text('# INFEASIBLE — open questions / evidence limits\n\n- No unresolved numerical validation mismatch: all 6300 B1 counts and real reference events agree.\n- Historical B1 arrays were not retained; the available historical validation is exact peak/trough/combined counts, as allowed in the prompt. This does not prove bitwise historical path identity beyond the original generator/seed reconstruction.\n- v2 validation and Part2 v1 trading masks differ in annual tail trimming. Both are fixed in preregistration_addendum.md before outcomes, applied consistently within each comparison, and never mixed between real and B1 trading.\n- L* >40 is right-censored; no curve or crossing is extrapolated.\n- Part3 remains NOT_RUN by user decision, R1–R4 NOT_EVALUATED; the synthetic model lacks hedged-return/intercept layers.\n')
 imports();S.verify(before)
 S.dump(S.P/'audit/input_hashes_after.json',before)
 S.dump(S.P/'scientific_complete.json',{'status':'scientific_complete_pending_final_integrity','INFEASIBLE':True,'verdict':verdict,'new_real_trials':0,'cumulative_real_trials':580,'keys':keys,'breakeven':dist,'comparison':comparison})
 print('INFEASIBLE NUMERICS COMPLETE',verdict,flush=True);print(pd.DataFrame(keys).to_csv(index=False),flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--validate-only',action='store_true');a=p.parse_args()
 prepare() if a.validate_only else main()
