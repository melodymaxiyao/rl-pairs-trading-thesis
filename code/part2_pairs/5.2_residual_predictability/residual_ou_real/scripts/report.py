"""Frozen inference, calibration transfer, readable results and 300dpi figures."""
from common import *
import pandas as pd
from stats import nw_mean,block_estimate,bootstrap_module,market_hac,weighted_quantile
from inputs import panel

def table(name,frame):
 text=frame.to_csv(index=False,float_format='%.17g',lineterminator='\n')
 publish(RUN/'outputs/tables'/name,gzip.compress(text.encode(),mtime=0) if name.endswith('.gz') else text)
def sharpe(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)];return float(np.sqrt(252)*a.mean()/a.std(ddof=1)) if len(a)>1 and a.std(ddof=1)>0 else np.nan

def pooled_hac(group,h):
 a=group.sort_values('date');sw=a.sum_w.to_numpy();sx=a.sum_x.to_numpy();sy=a.sum_y.to_numpy();sxx=a.sum_xx.to_numpy();sxy=a.sum_xy.to_numpy();A=np.array([[sw.sum(),sx.sum()],[sx.sum(),sxx.sum()]])
 if np.linalg.matrix_rank(A)<2:return dict(pooled_slope=np.nan,pooled_slope_se=np.nan,pooled_intercept=np.nan)
 inv=np.linalg.inv(A);b=inv@np.array([sy.sum(),sxy.sum()]);z=np.column_stack([sy-b[0]*sw-b[1]*sx,sxy-b[0]*sx-b[1]*sxx]);meat=z.T@z
 for k in range(1,min(h,len(z)-1)+1):C=z[k:].T@z[:-k];meat+=(1-k/(h+1))*(C+C.T)
 cov=inv@meat@inv
 return dict(pooled_slope=float(b[1]),pooled_slope_se=float(np.sqrt(max(0,cov[1,1]))),pooled_intercept=float(b[0]),pooled_n=int(a.n.sum()),pooled_se='cross-section aggregated day scores, Newey-West lag h')

def calibration_summary(frame):
 result={}
 for kind,g in frame.groupby('definition',sort=True):
  result[kind]={}
  for label,x in [('real',g[g.world=='real']),('N2',g[g.world=='N2'])]+[(f'N2_rep{rep}',g[(g.world=='N2')&(g.replicate==rep)]) for rep in [0,1]]:
   summary=dict(series=int(len(x)),valid_reverting_days=int(x.valid_reverting_days.sum()),total_series_days=int(x.total_days.sum()),weighting='temporal median per series, then unweighted cross-series quantiles',units=dict(half_life='sessions',sigma_epsilon='bps per one-session innovation',sigma_eq='bps',increment_predictive_r2='unitless'),increment_predictive_r2_label='AR(1) increment predictive R²; not an OU/random-walk variance decomposition')
   for key in ['half_life','sigma_epsilon','sigma_eq','increment_predictive_r2']:
    values=x[key].to_numpy(float);values=values[np.isfinite(values)];q=np.quantile(values,[.25,.5,.75]) if len(values) else np.full(3,np.nan);summary[key]=dict(median=q[1],q25=q[0],q75=q[2],IQR=q[2]-q[0])
    if kind=='RP':summary[key]['IPW_companion']={str(q):weighted_quantile(x[key],x.weight,q) for q in [.25,.5,.75]}
   result[kind][label]=summary
 return result

def era_of_year(y):
 return 'pre2004' if y<2004 else next((f'{a}-{b}' for a,b in [(2004,2008),(2009,2012),(2013,2016),(2017,2020),(2021,2025)] if a<=y<=b),'post2025')
def pooled_and_eras(frame):
 yield 'pooled',frame
 for era,g in frame.groupby('era',sort=True):yield era,g

def render(manifests):
 sections=[cached(s) for s in manifests];ic=pd.DataFrame([r for s in sections for r in s['ic']]);daily=pd.DataFrame([r for s in sections for r in s['daily']]);structure=pd.DataFrame([r for s in sections for r in s['structure']]);cal=pd.DataFrame([r for s in sections for r in s['calibration']]);persistence=pd.DataFrame([r for s in sections for r in s['persistence']]);coverage=pd.DataFrame([r for s in sections for r in s.get('coverage',[])]);unavailable=pd.DataFrame([r for s in sections for r in s.get('unavailable',[])])
 assert not ic.duplicated(['definition','world','replicate','date','h']).any() and not daily.duplicated(['definition','world','replicate','date']).any()
 # RP persistence retains per-series train and test endpoints; RF/RE already daily cross-sectional correlations.
 if 'current' in persistence:
  detail=persistence[persistence.definition=='RP'];summaries=[]
  from stats import spearman
  for key,g in detail.groupby(['definition','world','replicate','date','previous_date','fold','window'],sort=True):
   summaries.append(dict(zip(['definition','world','replicate','date','previous_date','fold','window'],key),rank_correlation=spearman(g.current,g.previous,g.weight),n=int((np.isfinite(g.current)&np.isfinite(g.previous)).sum())))
  persistence=pd.concat([pd.DataFrame(summaries),persistence[persistence.definition!='RP'].drop(columns=['current','previous','series','weight'],errors='ignore')],ignore_index=True)
 ic['era']=[era_of_year(int(str(d)[:4])) for d in ic.date]
 persistence['era']=[era_of_year(int(str(d)[:4])) for d in persistence.date]
 structure['era']=[era_of_year(int(y)) for y in structure.test_year]
 summary=[]
 for (kind,world,rep,h),all_g in ic.groupby(['definition','world','replicate','h'],sort=True):
  for era,g in pooled_and_eras(all_g):
   g=g.sort_values('date');r=nw_mean(g.ic,h);b=block_estimate(g.ic,g.date);conditional=nw_mean(g.ic_previous_fast_tercile,h)
   summary.append(dict(definition=kind,world=world,replicate=rep,h=h,era=era,ic_mean=r['mean'],nw_se=r['se'],days=r['n'],conditional_previous_top_tercile_ic=conditional['mean'],**b,**pooled_hac(g,h)))
 ic_summary=pd.DataFrame(summary);rule_rows=[];rule_daily=[];data=panel();spy=pd.Series(data['benchmark'],index=pd.Index(data['dates'].astype(str)))
 for (kind,world,rep),g in daily.groupby(['definition','world','replicate'],sort=True):
  for cost in spec()['cost_bps_per_side']:
   x=g.sort_values('date').copy();x['cost_bps']=cost*x.turnover;x['net_bps']=x.gross_bps-x.cost_bps;x['cost_per_side_bps']=cost;rule_daily.append(x)
   for era,z in [('pooled',x)]+[(f'{a}-{b}',x[x.date.between(str(a)+'-01-01',str(b)+'-12-31')]) for a,b in [(2004,2008),(2009,2012),(2013,2016),(2017,2020),(2021,2025)]]:
    market=spy.reindex(z.date).to_numpy();hac=market_hac(z.net_bps,market);hac['SPY_benchmark_missing_days']=int((~np.isfinite(market)).sum());hac['SPY_benchmark_alignment']='sealed forward_SPY shifted to realized close; first annual session unavailable';rule_rows.append(dict(definition=kind,world=world,replicate=rep,cost_per_side_bps=cost,era=era,gross_Sharpe=sharpe(z.gross_bps),net_Sharpe=sharpe(z.net_bps),gross_annual_bps=float(z.gross_bps.mean()*252),net_annual_bps=float(z.net_bps.mean()*252),annual_turnover=float(z.turnover.mean()*252),days=int(z.net_bps.notna().sum()),missing_days=int(z.net_bps.isna().sum()),mean_holding_period=float(z.closed_duration_today.sum()/z.closed_trades_today.sum()) if z.closed_trades_today.sum()>0 else np.nan,**hac))
 rule_summary=pd.DataFrame(rule_rows);all_rule_daily=pd.concat(rule_daily,ignore_index=True)
 tests=[];paired={}
 for kind,hyp in [('RP','H1'),('RF','H2')]:
  g=ic[(ic.definition==kind)&(ic.h==5)].pivot(index='date',columns=['world','replicate'],values='ic').sort_index();valid=g[[('real',0),('N2',0),('N2',1)]].dropna();v=valid[('real',0)]-(valid[('N2',0)]+valid[('N2',1)])/2;paired[hyp]=dict(estimate=block_estimate(v,v.index),difference=v);tests.append(dict(id=hyp,**paired[hyp]['estimate']))
 rp=ic[(ic.definition=='RP')&(ic.world=='real')&(ic.h==5)].set_index('date').ic;rf=ic[(ic.definition=='RF')&(ic.world=='real')&(ic.h==5)].set_index('date').ic;z=pd.concat([rf.rename('RF'),rp.rename('RP')],axis=1).dropna();tests.append(dict(id='H3',**block_estimate(z.RF-z.RP,z.index)))
 z=daily[daily.definition=='RF'].pivot(index='date',columns=['world','replicate'],values='gross_bps').sort_index().dropna();difference=z[('real',0)]-(z[('N2',0)]+z[('N2',1)])/2;tests.append(dict(id='H4',**block_estimate(difference,difference.index)))
 adjusted=bootstrap_module().holm([t['p_one_sided'] for t in tests])
 for row,p in zip(tests,adjusted):row.update(holm_p=float(p),supported=bool(p<spec()['holm_alpha'] and row['estimate']>0))
 predictions=pd.DataFrame(tests);supported=[('RP',tests[0]),('RF',tests[1])];supported=[(k,v) for k,v in supported if v['supported']];chosen=max(supported,key=lambda p:(p[1]['estimate'],p[0]=='RP'))[0] if supported else None;calibration=calibration_summary(cal)
 copied_ic={}
 for kind in ['RP','RF']+(['RE'] if spec()['available_sector_etfs'] else []):
  copied_ic[kind]={}
  for h in [1,5]:
   g=ic_summary[(ic_summary.definition==kind)&(ic_summary.h==h)&(ic_summary.era=='pooled')];real=g[g.world=='real'];null=g[g.world=='N2'];paired_ic=ic[(ic.definition==kind)&(ic.h==h)].pivot(index='date',columns=['world','replicate'],values='ic').dropna();paired_real=paired_ic[('real',0)];paired_null=(paired_ic[('N2',0)]+paired_ic[('N2',1)])/2;copied_ic[kind][str(h)]=dict(paired_common_days=len(paired_ic),paired_real_mean=float(paired_real.mean()),paired_N2_mean=float(paired_null.mean()),paired_real_minus_N2=float((paired_real-paired_null).mean()),real=real.to_dict('records'),N2=null.to_dict('records'),N2_equal_replica_mean=float(null.ic_mean.mean()),real_minus_N2=float(real.ic_mean.iloc[0]-null.ic_mean.mean()) if len(real) else np.nan)
 transfer=dict(chosen_residual=chosen,recommendation=chosen if chosen else 'none: neither H1 nor H2 supported after Holm',selection_rule=spec()['transfer'],hypotheses=tests,chosen_calibration=calibration.get(chosen) if chosen else None,candidate_calibration_descriptive=calibration,IC_h1_h5=copied_ic,variance_metric_label='AR(1) increment predictive R²; not an OU/random-walk variance decomposition',variance_decomposition_fitted=False,calibration_rule=spec()['calibration'],real_strategy_trials_before=spec()['cumulative_trials_before'],new_real_strategy_trials=spec()['new_primary_strategies']+spec()['new_optional_RE_strategies'],cumulative_real_strategy_trials=spec()['cumulative_trials_after'],development_evidence=True)
 if chosen:
  transfer['chosen_residual_real_medians']={key:calibration[chosen]['real'][key]['median'] for key in ['half_life','sigma_epsilon','sigma_eq','increment_predictive_r2']}
 else:transfer['chosen_residual_real_medians']=None
 table('ic_daily.csv.gz',ic);table('ic_summary.csv',ic_summary);table('persistence.csv',persistence);table('rule_daily.csv.gz',all_rule_daily);table('rule_summary.csv',rule_summary);table('predictions.csv',predictions);table('calibration_by_series.csv.gz',cal);table('coverage.csv',coverage);table('unavailable_series.csv',unavailable)
 # Earlier fixed-beta residual is retained as a sealed descriptive comparison,
 # with exact pair-fold membership and no claim that its different horizons match.
 from baseline import comparison
 baseline=comparison(cal);table('fixed_beta_comparison.csv',baseline)
 write(RUN/'residual_transfer.json',transfer);write(RUN/'outputs/tables/residual_transfer.json',transfer)
 groups=[];hist_rows=[]
 from jobs import HALF_BINS
 for key,all_g in structure.groupby(['definition','world','replicate'],sort=True):
  for era,g in pooled_and_eras(all_g):
   counts=np.stack(g.half_histogram.to_list());hist=(counts*g.weight.to_numpy()[:,None]).sum(axis=0)
   for j,value in enumerate(hist):hist_rows.append(dict(definition=key[0],world=key[1],replicate=key[2],era=era,bin_low=HALF_BINS[j],bin_high=HALF_BINS[j+1],weighted_windows=float(value)))
   weight=g.weight.to_numpy();den=float(weight@g.valid);groups.append(dict(zip(['definition','world','replicate'],key),era=era,fast_share=float(weight@g.fast/den) if den>0 else np.nan,significant_share=float(weight@g.significant/den) if den>0 else np.nan,observed_windows=float(weight@g.days),valid_windows=den,no_reversion=float(weight@g.no_reversion),nonpositive_b=float(weight@g.nonpositive_b)))
 structure_summary=pd.DataFrame(groups);table('ou_structure_summary.csv',structure_summary)
 histograms=pd.DataFrame(hist_rows);table('half_life_histogram.csv',histograms)
 make_plots(histograms,ic,all_rule_daily,spy)
 lines=[f'Half-life<30 shares, real/N2: '+json.dumps(clean(structure_summary[(structure_summary.era=='pooled')&structure_summary.definition.isin(['RP','RF'])].to_dict('records'))),f'H1/H2: '+json.dumps(clean(tests[:2])),f'H3: '+json.dumps(clean(tests[2]))+'; persistence '+json.dumps(clean(persistence.groupby(['definition','world']).rank_correlation.mean().to_dict()),default=str),f'H4: '+json.dumps(clean(tests[3]))+'; RF gross/net10 Sharpe '+json.dumps(clean(rule_summary[(rule_summary.definition=='RF')&(rule_summary.cost_per_side_bps==10)&(rule_summary.era=='pooled')][['world','replicate','gross_Sharpe','net_Sharpe']].to_dict('records'))),f'Transfer: {transfer["recommendation"]}']
 answers=[]
 accepted=[name for name,t in [('RP',tests[0]),('RF',tests[1])] if t['supported']]
 answers.append('Predictability beyond N2: '+(', '.join(accepted)+' has positive primary h5 IC beyond N2 after Holm.' if accepted else 'Neither RP nor RF has supported positive h5 IC beyond N2 after Holm.'))
 persist_means=persistence.groupby(['definition','world']).rank_correlation.mean().to_dict();answers.append('Persistence remains descriptive. Mean consecutive non-overlapping-window kappa rank correlations: '+json.dumps(clean(persist_means))+'.')
 economic=rule_summary[(rule_summary.world=='real')&(rule_summary.cost_per_side_bps==10)&(rule_summary.era=='pooled')]
 answers.append('After10bps-per-side costs: '+'; '.join(f'{r.definition}: annual mean net {r.net_annual_bps:.3f} bps, net Sharpe {r.net_Sharpe:.3f}' for r in economic.itertuples())+'. Net profitability has no confirmatory hypothesis in this task.')
 answers.append('Next Q-learning residual: '+(chosen if chosen else 'none; no supported H1/H2 candidate')+'.')
 text='# Real residual OU results\n\n'+'\n\n'.join(answers)+'\n\n'+ '\n\n'.join(lines)+'\n\nLimitations: '+ '; '.join(spec()['limitations'])+'. No RL was run. RE covers only sectors with sealed ETFs. All hypotheses use the frozen Holm family; calibration candidates do not override the transfer rule.\n'
 publish(RUN/'RESULTS.md',text)
 prefix=(RUN/'README_PREFIX.md').read_bytes();readme=RUN/'README.md';expected=prefix+b'\n\n'+text.encode()
 if readme.read_bytes()==prefix:
  with readme.open('ab') as f:f.write(b'\n\n'+text.encode());f.flush();os.fsync(f.fileno())
 else:assert readme.read_bytes()==expected,'Existing README suffix differs; not overwritten'
 return dict(summary=lines,chosen_residual=chosen,hypotheses=tests,IC_rows=len(ic),rule_rows=len(all_rule_daily))

def make_plots(histograms,ic,daily,spy):
 import matplotlib.pyplot as plt,io
 def finish(fig,name):
  b=io.BytesIO();fig.savefig(b,format='png',dpi=300,bbox_inches='tight',metadata={'Software':'residual_ou_real_v1'});plt.close(fig);publish(RUN/'outputs/plots'/name,b.getvalue())
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,kind in zip(axes,['RP','RF']):
  for world in ['real','N2']:
   g=histograms[(histograms.definition==kind)&(histograms.world==world)&(histograms.era=='pooled')].groupby('bin_low').weighted_windows.sum().sort_index();total=g.sum();v=g[g.index<120];ax.stairs(v.to_numpy()/total/2 if total else v.to_numpy()*np.nan,np.arange(0,122,2),label=world)
  ax.axvline(30,color='black',ls=':');ax.set(title=kind,xlabel='Window half-life (sessions; ≥120 tail in CSV)',ylabel='Density');ax.legend()
 finish(fig,'half_life_distributions.png')
 fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
 for ax,kind in zip(axes,['RP','RF']):
  for world in ['real','N2']:
   g=ic[(ic.definition==kind)&(ic.world==world)&(ic.h==5)].pivot(index='date',columns='replicate',values='ic').sort_index().mean(axis=1,skipna=False);values=g.to_numpy();m=[];se=[]
   for t in range(len(values)):
    r=nw_mean(values[max(0,t-251):t+1],5);m.append(r['mean']);se.append(r['se'])
   x=pd.to_datetime(g.index);m=np.array(m);se=np.array(se);ax.plot(x,m,label=world);ax.fill_between(x,m-1.96*se,m+1.96*se,alpha=.15)
  ax.set(title=kind,ylabel='Rolling252-session IC');ax.legend()
 finish(fig,'rolling_IC.png')
 fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
 for ax,kind in zip(axes[:2],['RP','RF']):
  g=daily[(daily.definition==kind)&(daily.world=='real')&(daily.cost_per_side_bps==10)].sort_values('date');x=pd.to_datetime(g.date);ax.plot(x,g.gross_bps.cumsum(),label='gross');ax.plot(x,g.net_bps.cumsum(),label='net10bps per side');ax.set(title=kind,ylabel='Cumulative bps');ax.legend()
 x=pd.to_datetime(spy.index);ok=(x.year>=2004)&(x.year<=2025);axes[2].plot(x[ok],spy.iloc[np.flatnonzero(ok)].cumsum());axes[2].set(title='SPY separate panel',ylabel='Cumulative bps');finish(fig,'cumulative_PnL.png')
 fig,ax=plt.subplots(figsize=(7,4))
 for kind in ['RP','RF']:
  g=ic[(ic.definition==kind)&(ic.world=='real')].groupby('h').ic.mean();ax.plot(g.index,g,marker='o',label=kind)
 ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Forward horizon (sessions)',ylabel='Mean daily IC');ax.legend();finish(fig,'IC_by_horizon.png')
