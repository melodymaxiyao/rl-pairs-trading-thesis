"""Seal the Part B decision; Part C remains unexecuted irrespective of GO."""
import gzip,io
import numpy as np,pandas as pd
from scipy.special import ndtr
from io_utils import *
from groups import done
import residuals
def frame(p):return pd.read_csv(p,float_precision='round_trip')
def emit(name,value):
 raw=value.to_csv(index=False,float_format='%.17g').encode() if isinstance(value,pd.DataFrame) else encode(value)
 if name.endswith('.gz'):raw=gzip.compress(raw,compresslevel=6,mtime=0)
 publish(ROOT/name,raw)
def num(x):return 'NA' if x is None or not np.isfinite(x) else f'{x:.6f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def contrast(stats,ic,kind,h):
 g=ic[(ic.definition==kind)&(ic.comparison=='roll_new')&(ic.h==h)];v=g.pivot(index='date',columns=['world','replicate','signal'],values='ic').sort_index()
 keys=[(w,r,s) for w,r in [('real',0),('N2',0),('N2',1)] for s in ['Kalman','roll_new']];matched=v.reindex(columns=pd.MultiIndex.from_tuples(keys)).dropna();delta=[matched[(w,r,'Kalman')]-matched[(w,r,'roll_new')] for w,r in [('real',0),('N2',0),('N2',1)]];d=delta[0]-(delta[1]+delta[2])/2
 years=np.asarray(d.index,dtype='datetime64[D]').astype('datetime64[Y]').astype(int)+1970;f=pd.DataFrame(dict(value=d.to_numpy(),test_year=years,weight=np.ones(len(d))));result=stats.estimate(f,draws=5000);nw=stats.nw_mean(d.to_numpy(),h);p_nw=float(ndtr(-nw['mean']/nw['se'])) if np.isfinite(nw['se']) and nw['se']>0 else 1.
 return dict(definition=kind,h=h,**result,NW_mean=nw['mean'],NW_SE=nw['se'],NW_one_sided_p=p_nw,NW_lag=h,matched_days=len(d))
def report():
 if (ROOT/'runtime/report_B_complete.json').exists():
  old=J(ROOT/'runtime/report_B_complete.json')
  for rel,digest in old['files'].items():assert H(ROOT/rel)==digest
  return J(ROOT/'summary.json')['lines']
 stats=residuals.imported_stats();ic=[];params=[];pooled=[];example=None;construction=[];LLT_models=[];LLT_tests=[];LLT_slope=[]
 for kind in ['RP','RF']:
  for year in range(2004,2026):
   for wi in range(3):
    p=ROOT/f'runtime/part_B/{kind}_{year}_{wi}';assert done(p);ic.append(frame(p/'ic.csv.gz'));overlay=ROOT/f'runtime/B2_bucket_overlays/{kind}_{year}_{wi}';params.append(frame((overlay if (overlay/'complete.json').exists() and done(overlay) else p)/'params.csv.gz'));pooled.append(frame(p/'pooled.csv.gz'));construction.append(frame(p/'construction_correlation.csv.gz'));LLT_models.append(frame(p/'LLT_models.csv.gz'));LLT_tests.append(frame(p/'LLT_tests.csv.gz'));LLT_slope.append(frame(p/'LLT_slope_IC.csv.gz'))
    if example is None and (p/'example.csv.gz').stat().st_size>100:
     e=frame(p/'example.csv.gz')
     if not e.empty:example=e
 ic=pd.concat(ic,ignore_index=True);params=pd.concat(params,ignore_index=True);pooled=pd.concat(pooled,ignore_index=True);construction=pd.concat(construction,ignore_index=True)
 primary=[dict(id=name,**contrast(stats,ic,kind,5)) for name,kind in [('K1','RF'),('K2','RP')]];adj=stats.holm([r['p_one_sided'] for r in primary])
 for row,q in zip(primary,adj):row['holm_p']=float(q)
 descriptive=[dict(id='descriptive_h1_'+kind,**contrast(stats,ic,kind,1)) for kind in ['RP','RF']];go=any(np.isfinite(r['estimate']) and r['estimate']>0 and r['p_one_sided']<.1 for r in primary)
 if (ROOT/'runtime/B1_report/complete.json').exists() and done(ROOT/'runtime/B1_report'):
  b1=J(ROOT/'runtime/B1_report/decision.json');assert b1['decision']==('GO' if go else 'NO_GO')
  for before,after in zip(b1['tests'],primary):
   for key in ['id','estimate','ci95_low','ci95_high','p_one_sided','holm_p']:assert before[key]==clean(after[key]),('B1/B2 primary changed',key)
 decision=dict(decision='GO' if go else 'NO_GO',rule='At least one RP/RF h5 paired year-block difference>0 with one-sided unadjusted p<.10; Holm acrossK1/K2 reported separately',tests=primary,part_B_complete=True,part_C_started=False,requires_user_confirmation_before_C=True,planned_total_trials=594,new_real_trading_cells_executed=0)
 emit('go_nogo.json',decision);emit('contrasts.csv',pd.DataFrame(primary+descriptive));emit('ic.csv',ic);emit('params.csv.gz',params);emit('pooled_params.csv.gz',pooled);emit('construction_correlation.csv',construction)
 LLT_models=pd.concat(LLT_models,ignore_index=True);LLT_tests=pd.concat(LLT_tests,ignore_index=True);LLT_slope=pd.concat(LLT_slope,ignore_index=True);emit('LLT_models.csv',LLT_models);emit('LLT_tests.csv',LLT_tests);emit('LLT_slope_IC.csv',LLT_slope)
 LLT_summary=[]
 for (kind,world),g in LLT_tests.groupby(['definition','world']):
  valid=g[g.p_slope_boundary_mixture.notna()];LLT_summary.append(dict(definition=kind,world=world,groups=len(g),valid_slope_tests=len(valid),invalid_slope_tests=len(g)-len(valid),slope_needed_share=float(valid.slope_needed_nominal5percent.mean()) if len(valid) else None,sum_group_LLT_seconds=float(g.measured_LLT_total_seconds.sum()),timing='sum of group wall-seconds; groups may run in parallel,not total elapsed wall time'))
 emit('LLT_summary.csv',pd.DataFrame(LLT_summary));summaries=[]
 LLT_model_summary=[]
 for (kind,world,model),g in LLT_models.groupby(['definition','world','model']):
  good=g[g.converged==True];LLT_model_summary.append(dict(definition=kind,world=world,model=model,groups=len(g),converged_groups=len(good),median_LL=float(good.loglikelihood.median()) if len(good) else None,median_delta_AIC=float(good.delta_AIC.median()) if len(good) else None,median_delta_BIC=float(good.delta_BIC.median()) if len(good) else None))
 emit('LLT_model_summary.csv',pd.DataFrame(LLT_model_summary))
 for (kind,world),g in params.groupby(['definition','world']):
  valid=g[g.converged==True];summaries.append(dict(definition=kind,world=world,series_years=len(g),converged=len(valid),secondary_rho_median=float(valid.rho.median()) if len(valid) else None,secondary_R2_MR_median=float(valid.R2_MR.median()) if len(valid) else None,pooled_rho_median=float(g.pooled_rho.median()),pooled_R2_MR_median=float(g.pooled_R2_MR.median()),boundary_share=float(valid.boundary.mean()) if len(valid) else None,PCI_pass_share_all=float(g.PCI_pass.mean()),PCI_pass_share_valid=float(valid.PCI_pass.mean()) if len(valid) else None))
 emit('parameter_summary.csv',pd.DataFrame(summaries));inference=[]
 for (kind,world,rep,h,comparison,signal),g in ic.groupby(['definition','world','replicate','h','comparison','signal']):
  g=g.sort_values('date');nw=stats.nw_mean(g.ic,h);fmb=stats.nw_mean(g.FMB_slope,h);inference.append(dict(definition=kind,world=world,replicate=int(rep),h=int(h),comparison=comparison,signal=signal,IC_mean=nw['mean'],IC_NW_SE=nw['se'],FMB_mean=fmb['mean'],FMB_NW_SE=fmb['se'],mean_matched_fraction=float(g.matched_fraction.mean()),**stats.pooled_hac(g,h)))
 emit('IC_inference.csv',pd.DataFrame(inference));recovery=[];power=[];synthetic_ic=[];oracle_rows=[];predictions=[]
 for world in ['OU20','calibrated','second','RW']:
  pp=[];ss=[];ii=[];al={};oracle={};LLT_Amodels=[];LLT_Atests=[];LLT_Aic=[];LLT_Aslope=[]
  for index in range(200):
   checkpoint();root=ROOT/f'runtime/part_A/{world}_{index:03d}';assert done(root);pp.append(J(root/'primary.json'));ss.append(frame(root/'secondary.csv.gz'));ii.append(frame(root/'ic.csv.gz'))
   for signal,row in J(root/'AL.json').items():
    if signal not in al:al[signal]=dict(net_sum=0.,net_sumsq=0.,turnover_sum=0.,series=0)
    for k,v in row.items():al[signal][k]+=v
   z=np.load(root/'oracle.npz',allow_pickle=False)
   for key in z.files:oracle[key]=oracle.get(key,0)+z[key]
   comp=json.loads(gzip.decompress((root/'LLT_comparison.json.gz').read_bytes()));LLT_Amodels.extend(dict(world=world,panel=index,**r) for r in comp['models']);LLT_Atests.append(dict(world=world,panel=index,**comp['test']));LLT_Aic.append(frame(root/'LLT_ic.csv.gz'));LLT_Aslope.append(frame(root/'LLT_slope_IC.csv.gz'))
  ss=pd.concat(ss,ignore_index=True);ii=pd.concat(ii,ignore_index=True);assert len(ss)==20000
  for estimator in ['primary','secondary']:
   for parameter,key in [('rho','rho'),('R2_MR','R2_MR')]:
    truth=pp[0]['truth_'+key]
    if estimator=='primary':v=np.array([p['fit'].get(key,np.nan) for p in pp],float)
    else:v=ss[key].to_numpy(float)
    finite=np.isfinite(v);bias=float(np.mean(v[finite]-truth)) if truth is not None and finite.any() else None;rmse=float(np.sqrt(np.mean((v[finite]-truth)**2))) if truth is not None and finite.any() else None
    recovery.append(dict(world=world,estimator=estimator,parameter=parameter,truth=truth,estimate_mean=float(v[finite].mean()) if finite.any() else None,bias=bias,RMSE=rmse,estimates=int(finite.sum()),undefined=int((~finite).sum()),identification='rho undefined under pure RW' if world=='RW' and parameter=='rho' else 'defined'))
  valid=ss[ss.converged==True];pci_rate=float(valid.PCI_reject.mean()) if len(valid) else None;adf_rate=float(ss.ADF60_rejection.mean());power += [dict(world=world,test='PCI',null_status='H0 true' if world=='RW' else 'H0 false',interpretation='size' if world=='RW' else 'power',rejection_rate=pci_rate,series_years=len(valid),invalid_series_years=20000-len(valid),information='one756-observation formation LR'),dict(world=world,test='ADF60',null_status='H0 false' if world=='OU20' else 'H0 true',interpretation='power' if world=='OU20' else 'size',rejection_rate=adf_rate,series_years=20000,information='mean252test-window rejection rates per series-year')]
  means=ii.groupby(['h','signal'],dropna=False).ic.mean();net={}
  for signal in ['Kalman','roll_new']:
   row=al[signal];net[signal]=row['net_sum']/row['series'];variance=max(0,(row['net_sumsq']-row['net_sum']**2/row['series'])/(row['series']-1));synthetic_ic += [dict(world=world,signal=signal,h=h,IC=float(means.get((h,signal),np.nan)),AL_net=net[signal],AL_net_SE=float(np.sqrt(variance/row['series'])),AL_turnover=row['turnover_sum']/row['series']) for h in [1,5]]
   key='K' if signal=='Kalman' else 'roll_new';selection=oracle['selection_'+key];truth=oracle['truth_'+key];bits=int(np.argmax(selection[:,0]));oracle_rows.append(dict(world=world,signal=signal,bits=bits,selection_series=20000,truth_series=20000,selection_net=float(selection[bits,0]/20000),truth_net=float(truth[bits,0]/20000),truth_net_SE=float(np.sqrt(max(0,(truth[bits,1]-truth[bits,0]**2/20000)/19999)/20000)),truth_turnover=float(truth[bits,2]/20000)))
  if world=='calibrated':
   ka1=float(means.get((5,'Kalman'),np.nan))>float(means.get((5,'roll_new'),np.nan));r={x['signal']:x for x in oracle_rows if x['world']==world};predictions += [dict(id='KA1',supported=bool(ka1)),dict(id='KA2',supported=bool(r['Kalman']['truth_net']>=r['roll_new']['truth_net'])),dict(id='KA3',supported=bool(pci_rate is not None and pci_rate>adf_rate))]
  emit('outputs/LLT_A_models_'+world+'.csv.gz',pd.DataFrame(LLT_Amodels));emit('outputs/LLT_A_tests_'+world+'.csv.gz',pd.DataFrame(LLT_Atests));emit('outputs/LLT_A_IC_'+world+'.csv.gz',pd.concat(LLT_Aic,ignore_index=True));emit('outputs/LLT_A_slope_IC_'+world+'.csv.gz',pd.concat(LLT_Aslope,ignore_index=True));print('LLT Part A',world,'summed group seconds',sum(r['measured_LLT_total_seconds'] for r in LLT_Atests),flush=True)
 emit('recovery.csv',pd.DataFrame(recovery));emit('power.csv',pd.DataFrame(power));emit('synthetic_IC_and_AL.csv',pd.DataFrame(synthetic_ic));emit('Oracle8.csv',pd.DataFrame(oracle_rows));emit('predictions.csv',pd.DataFrame(predictions));emit('cells.csv',pd.DataFrame(columns=['definition','learner','cost','net','turnover','status']))
 plot(example,params,inference)
 summary_tests='; '.join(f"{x['id']}={num(x['estimate'])} CI[{num(x['ci95_low'])},{num(x['ci95_high'])}],Holm p={num(x['holm_p'])},unadjusted p={num(x['p_one_sided'])}" for x in primary)
 LLT_IC=[r for r in inference if r['signal']=='LLT'];LLT_slopes=[]
 for (kind,world,rep,h),g in LLT_slope.groupby(['definition','world','replicate','h']):LLT_slopes.append(dict(definition=kind,world=world,replicate=int(rep),h=int(h),**stats.nw_mean(g.sort_values('date').ic,h)))
 emit('LLT_IC_summary.csv',pd.DataFrame(LLT_IC));emit('LLT_slope_IC_summary.csv',pd.DataFrame(LLT_slopes))
 lines=[f'Part A: primary pooled and secondary recovery in recovery.csv; KA1-KA3={predictions}.',f'{summary_tests}; decision={decision["decision"]}; stopped before Part C.',f'R2_MR/rho medians, boundary shares and PCI pass rates real/N2: {clean(summaries)}.','C1-C4: not run; Part C requires a later user confirmation.','Q_best_K net/turnover/consensus: not run; sealed core baseline unchanged.','The paired same-construction signal test '+('passes the frozen exploratory go gate; this permits consideration of trading tests and does not establish profitability or unique decomposition.' if go else 'does not pass the frozen go gate; partial cointegration has not demonstrated enough signal improvement to launch trading tests under this protocol.'),f'LLT model comparison: {clean(LLT_model_summary)}; slope-needed shares/measured runtime: {clean(LLT_summary)}.',f'sLLT matched IC: {clean(LLT_IC)}; P(slope>0) future-X IC: {clean(LLT_slopes)}.']
 body=['# partial_coint_kalman_v1 — sealed through Part B',*lines,'## Decision',table(['Test','Estimate','95%year-block CI','Unadjusted p','Holm p'],[(x['id'],num(x['estimate']),f"[{num(x['ci95_low'])},{num(x['ci95_high'])}]",num(x['p_one_sided']),num(x['holm_p'])) for x in primary]),'The go rule uses the preregistered one-sided unadjusted year-block p<.10 and positive h5point in either family. GO is a checkpoint, not authorization to bypass the explicit user stop before Part C. No new real trading cells, real Q fits, or consensus replays have been executed. Four extra descriptive Part Ccells are registered, so the conditional full plan is580+14=594trials.','## Plain answers','Residual construction: each day uses only the prior-day core regression/PCA/loading coefficients. RP uses the frozen annual pair cohort; RF uses point-in-time current membership and prior-day admissible PCA. No missing raw price is invented for signals. Undefined increments stay missing observations and prediction-only filter steps; the known-increment cumulative accumulator is masked on all such dates. The comparator rolling60OU is computed from the same new increments with the unchanged core fit and validity filter. Sealed core scores remain secondary comparators. Correlations with the original in-window last residual are descriptive in construction_correlation.csv.','Primary MLE pools only within (calendar year,family,world), with per-series scale concentrated out. The exact diffuse M and stationary R likelihoods are used. Test-year data never enter estimation. Secondary per-series PCI likelihoods can be boundary fits; these are reported, including exact RW-null closure. PCI rejection and ADF rejection answer different nulls, and synthetic information lengths differ (formation756 versus rolling60). Correctly specified synthetic worlds do not identify a unique real-world decomposition.','Does the model give a better signal? See matched-data K1/K2, h1companions, coverage and IC_inference.csv. Does Q-learning trade it better? Not tested: Part C has not run. Is PCI a better trading pre-test? Its synthetic detection and real pass rates are available, but cost-adjusted trading superiority awaits authorized Part C. Implication for cointegration: test a transitory component and its net value separately from the presence of a unit root.','## Plots','![Example X,M,R](outputs/plots/01_example.png)','![R2 MR real vsN2](outputs/plots/02_R2_distribution.png)','![IC horizons](outputs/plots/03_IC.png)','![Core net/turnover](outputs/plots/04_net_turnover.png)','## Inference and integrity','IC, weighted ranks, Fama-MacBeth slopes, pooled HAC and Newey-West functions are imported literally from authenticated residual_ou_real code. NW lag equals horizon. Primary inference uses unchanged TAV year-block seed20261026,5000draws and Holm across K1/K2. Each compared signal uses identical series-days and future targets within each world, followed by common-date alignment across real/N2. No missing p-value is imputed. Exact original AL/Band/Q_best daily values were reproduced before new computation. B1 primary decision was published before B2 under the user-authorized pre-result amendment. B2 PCI gates use three effective-length bucket criticals; original completed gates remain immutable and amended gate overlays are separately retained. Pattern critical sensitivities are in runtime/bucket_pattern_sensitivity. See preregistration/amendments/speedup_B1_B2_v1. All task writes are outside iCloud except authorized RUN_LOCATIONS bookkeeping and prompt revision bookkeeping. Frozen inputs, code and source environment are verified before and after work.']
 body += ['## LLT descriptive comparison',table(['Family','World','Model','Converged groups','Median LL','Median ΔAIC','Median ΔBIC'],[(r['definition'],r['world'],r['model'],r['converged_groups'],num(r['median_LL']),num(r['median_delta_AIC']),num(r['median_delta_BIC'])) for r in LLT_model_summary]),'All three comparison models exclude the same first two finite observations per series from likelihood, retaining those observations for filtering. The comparison-only no-drift fit is separate from the original primary signal fit, so sK and the go gate remain unchanged. Parameter count follows the user convention: shared pooled parameters plus one scale per series. Slope LR compares full LLT to restricted q_zeta0 LLT with identical diffuse L/S; the user-requested boundary-mixture p is nominal. The constant-drift versus primary signed2LL difference has no chi-square p. Innovation SNR ratios are reported; an LLT integrated trend has no stationary long-run variance fraction. Group wall-seconds are summed separately for LLT; these are not elapsed parallel wall time.']
 publish(ROOT/'README.md','\n\n'.join(body)+'\n')
 if (ROOT/'summary.json').exists():assert J(ROOT/'summary.json')['lines']==lines
 else:write(ROOT/'summary.json',dict(lines=lines,go=go,part_C_started=False,new_real_trading_trials=0,planned_trials=594,df=df()))
 files={str(p.relative_to(ROOT)):H(p) for p in [ROOT/'README.md',ROOT/'summary.json',ROOT/'go_nogo.json',*ROOT.glob('*.csv'),*ROOT.glob('*.csv.gz'),*[p for p in (ROOT/'outputs').rglob('*') if p.is_file()]]};write(ROOT/'runtime/report_B_complete.json',dict(files=files,preregistration_sha256=H(ROOT/'preregistration.json')));return lines
def plot(example,params,inference):
 import matplotlib.pyplot as plt
 def save(fig,name):
  checkpoint();fig.tight_layout();s=io.BytesIO();fig.savefig(s,format='png',dpi=180);publish(ROOT/'outputs/plots'/name,s.getvalue());plt.close(fig)
 fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
 if example is not None:
  e=example.tail(400)
  for ax,key in zip(axes,['X','M','R']):ax.plot(pd.to_datetime(e.date),e[key]);ax.set_ylabel(key+' (bps)')
 axes[0].set_title('Example causal level and filtered components; one axis per component');save(fig,'01_example.png')
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,kind in zip(axes,['RP','RF']):
  for world in ['real','N2']:
   v=params[(params.definition==kind)&(params.world==world)&(params.converged==True)].R2_MR.dropna();ax.hist(v,bins=np.linspace(0,1,26),density=True,histtype='step',label=world)
  ax.set(title=kind,xlabel='Secondary per-series R²_MR',ylabel='Density');ax.legend()
 save(fig,'02_R2_distribution.png');fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,kind in zip(axes,['RP','RF']):
  for signal in ['Kalman','roll_new']:
   rows=sorted([r for r in inference if r['definition']==kind and r['world']=='real' and r['comparison']=='roll_new' and r['signal']==signal],key=lambda r:r['h']);ax.errorbar([r['h'] for r in rows],[r['IC_mean'] for r in rows],yerr=[1.96*r['IC_NW_SE'] for r in rows],marker='o',label=signal)
  ax.set(title=kind,xlabel='Horizon (sessions)',ylabel='Matched real IC');ax.legend()
 save(fig,'03_IC.png');mf=J(CORE/'tier_3/manifest.json')['files'];rel='tier_3/performance.csv.gz';assert H(CORE/rel)==mf[rel];f=frame(CORE/rel);f=f[(f.world=='real')&(f.gate=='G0')&(f.cost==5)&(f.era=='pooled')&(f.accounting=='K2')];fig,ax=plt.subplots(figsize=(8,5))
 for row in f.itertuples():ax.scatter(row.annual_K2_turnover,row.net_annual_bps);ax.annotate(row.definition+'/'+row.learner,(row.annual_K2_turnover,row.net_annual_bps),fontsize=7)
 ax.set(title='Sealed core5bps benchmarks; Part C new cells not run',xlabel='Annual K2 turnover',ylabel='Net bps/year');save(fig,'04_net_turnover.png')
