"""Pooled formation MLE and matched-data signal inference; no new trade execution."""
import gzip,io,time
import numpy as np,pandas as pd
from io_utils import *
from groups import *
import pci,residuals,llt
from scipy.special import ndtr
def secondary(levels,names,root):
 all_rows=[]
 for offset in range(0,len(levels),25):
  chunk=root/f'secondary_{offset:05d}'
  if not done(chunk):
   rows=[]
   for j in range(offset,min(len(levels),offset+25)):
    checkpoint();x=levels[j];n=int(np.isfinite(x).sum());row=dict(index=j,series=names[j],valid_formation_sessions=n,converged=False,rho=None,q=None,R2_MR=None,LR=None,boundary=None,RW_boundary=None)
    if n>=500:
     fit=pci.fit(x,pooled=False,checkpoint=checkpoint);row.update({k:v for k,v in fit.items() if k not in ['scale','n']});row['sigma2_R']=float(fit['scale'][0]) if fit.get('converged') else None
    rows.append(row)
   finish(chunk,{'fits.json.gz':gzip.compress(encode(rows),compresslevel=6,mtime=0)},phase='secondary_PCI')
  all_rows+=json.loads(gzip.decompress((chunk/'fits.json.gz').read_bytes()))
  if offset%100==0:print('Secondary PCI',root.name,offset+len(rows) if not done(chunk) else min(len(levels),offset+25),'/',len(levels),flush=True)
 return all_rows
def job(kind,year,wi_only):
 root=ROOT/f'runtime/part_B/{kind}_{year}_{wi_only}'
 if done(root):
  import null_buckets
  return null_buckets.overlay_completed(kind,year,wi_only)
 import bootstrap
 c,d,a=bootstrap.core_modules();group=residuals.load_group(kind,year);form=group['formation'];target,core_score,w=residuals.core_targets(kind,year);stats=residuals.imported_stats();params=[];ic=[];arrays={};pooled_rows=[];example=None;correlation=[];LLT_models=[];LLT_tests=[];slope_ic=[];LLT_example=pd.DataFrame(columns=['date','L','S','R','S_variance','P_S_positive'])
 for wi in [wi_only]:
  checkpoint();world='real' if wi==0 else 'N2';rep=0 if wi==0 else wi-1;e,levels,roll,half_roll=residuals.signed(group,wi);eligible=np.isfinite(levels[:,:form]).sum(axis=1)>=500;ids=np.flatnonzero(eligible);primary_root=ROOT/f'runtime/B_fits/{kind}_{year}_{wi}';primary_root.mkdir(parents=True,exist_ok=True)
  pooled_root=primary_root/'pooled';pooled_path=pooled_root/'pooled.json'
  previous=ROOT/f'runtime/B_fits/{kind}_{year-1}_{wi}';warm=J(previous/'pooled/pooled.json') if year>2004 and done(previous/'pooled') else None
  if done(pooled_root):pooled=J(pooled_path)
  elif len(ids):
   pooled=pci.fit(levels[ids,:form],pooled=True,checkpoint=checkpoint,warm_start=warm);finish(pooled_root,{'pooled.json':pooled},phase='primary_pooled_MLE')
  else:pooled=dict(converged=False,reason='No series with >=500valid formation sessions');finish(pooled_root,{'pooled.json':pooled},phase='primary_pooled_MLE')
  pooled_rows.append(dict(definition=kind,test_year=year,world=world,replicate=rep,**{k:v for k,v in pooled.items() if k not in ['scale','n','runs']},formation_series=len(ids)))
  print('Pooled',kind,year,world,rep,'rho',pooled.get('rho'),'q',pooled.get('q'),'converged',pooled['converged'],flush=True)
  LLT_start=time.monotonic();LLT_root=primary_root/'LLT';meta=dict(definition=kind,test_year=year,world=world,replicate=rep)
  LLT_was_cached=done(LLT_root)
  if LLT_was_cached:comparison=J(LLT_root/'comparison.json')
  elif len(ids):
   warmLLT=J(previous/'LLT/comparison.json') if year>2004 and done(previous/'LLT') else None;comparison=llt.compare(levels[ids,:form],pooled,warm=warmLLT,checkpoint=checkpoint);finish(LLT_root,{'comparison.json':comparison},phase='LLT_descriptive_comparison')
  else:comparison=dict(models=[dict(model=name,converged=False,series=0,shared_parameters=k,total_parameters=k,common_startup_excluded=2,loglikelihood=None,AIC=None,BIC=None,delta_AIC=None,delta_BIC=None) for name,k in [('primary_no_drift',2),('restricted_constant_drift',2),('full_LLT',3)]],test=dict(measured_LLT_seconds=0.,LR_slope=None,p_slope_boundary_mixture=None,slope_needed_nominal5percent=False,constant_drift_2LL_difference=None,constant_drift_p=None),full=dict(converged=False))
  LLT_models.extend(dict(**meta,**r) for r in comparison['models']);LLT_test=dict(**meta,**comparison['test']);full=comparison['full'];LLT_score=np.full_like(levels,np.nan);slope_probability=np.full_like(levels,np.nan)
  if full['converged']:
   lv=llt.profile(levels[ids],full['rho'],full['q_eta'],full['q_zeta'],states=True)['states'];scale=np.asarray(full['scale']);LLT_score[ids]=lv[:,:,2]/np.sqrt(scale[:,None]/(1-full['rho']**2));variance=lv[:,:,4]*scale[:,None];sd=np.sqrt(variance);prob=np.divide(lv[:,:,1],sd,out=np.full_like(sd,np.nan),where=sd>0);prob=ndtr(prob);zero=(sd==0)&np.isfinite(lv[:,:,1]);prob[zero]=(lv[:,:,1][zero]>0).astype(float);slope_probability[ids]=prob
   if not full['half_life']<30:LLT_score[:]=np.nan
   LLT_score[~np.isfinite(levels)]=np.nan;slope_probability[~np.isfinite(levels)]=np.nan
   LLT_example=pd.DataFrame(dict(date=group['dates'].astype(str),series=group['ids'][ids[0]],L=lv[0,:,0],S=lv[0,:,1],R=lv[0,:,2],S_variance=variance[0],P_S_positive=prob[0]))
   vv=variance[:,form:];finite_variance=vv[np.isfinite(vv)];LLT_test['filtered_slope_variance_median']=float(np.median(finite_variance)) if len(finite_variance) else None;LLT_test['filtered_slope_variance_mean']=float(np.mean(finite_variance)) if len(finite_variance) else None
  LLT_posterior_seconds=time.monotonic()-LLT_start;LLT_IC_seconds=0.
  import part_b1
  b1=part_b1.root_for(kind,year,wi);assert done(b1)
  primary_scale=np.full(len(levels),np.nan)
  if pooled['converged']:primary_scale[ids]=pooled['scale']
  with np.load(b1/'signals.npz',allow_pickle=False) as z:
   score=np.full_like(levels,np.nan);score[:,form:]=z[f'sK_{wi}']
  example=pd.read_csv(b1/'example.csv.gz',float_precision='round_trip')
  ic=pd.read_csv(b1/'ic.csv.gz',float_precision='round_trip').to_dict('records')
  fits=secondary(levels[:,:form],group['ids'],primary_root);assert len(fits)==len(levels);pci_pass=np.zeros(len(levels),bool)
  import null_buckets
  for i,row in enumerate(fits):
   if row['converged']:
    crit=null_buckets.for_mask(np.isfinite(levels[i,:form]));row['critical_value']=crit['critical_5percent'];row['critical_bucket']=crit['bucket'];pci_pass[i]=bool(row['LR']>crit['critical_5percent'] and row['R2_MR']>=.5)
   params.append(dict(definition=kind,test_year=year,world=world,replicate=rep,**{k:v for k,v in row.items() if k!='runs'},pooled_sigma2_R=float(primary_scale[i]),pooled_rho=pooled.get('rho'),pooled_q=pooled.get('q'),pooled_R2_MR=pooled.get('R2_MR'),PCI_pass=bool(pci_pass[i])))
  arrays[f'sK_{wi}']=score[:,form:];arrays[f'roll_new_{wi}']=roll[:,form:];arrays[f'PCI_{wi}']=pci_pass
  valid=np.isfinite(e[:,form:])&np.isfinite(group['core_inwindow'][:,form:]);aa=e[:,form:][valid];bb=(group['core_inwindow'][:,form:]*(np.ones_like(e[:,form:]) if wi==0 else np.divide(e[:,form:],group['e'][:,form:],out=np.full_like(e[:,form:],np.nan),where=group['e'][:,form:]!=0)))[valid];good=np.isfinite(aa)&np.isfinite(bb);correlation.append(dict(definition=kind,test_year=year,world=world,replicate=rep,n=int(good.sum()),correlation=float(np.corrcoef(aa[good],bb[good])[0,1]) if good.sum()>2 else None))
  for t,date in enumerate(group['dates'][form:]):
   if t%30==0:checkpoint()
   for hi,h in enumerate([1,5]):
    future=target[wi,:,t,hi]
    LLT_IC_start=time.monotonic()
    for comparison,baseline,label in [('LLT_vs_K',score[:,form+t],'Kalman'),('LLT_vs_roll_new',roll[:,form+t],'roll_new')]:
     common=np.isfinite(LLT_score[:,form+t])&np.isfinite(baseline)&np.isfinite(future)&(w>0);weight=w[common];y=future[common]
     for signal,z in [('LLT',LLT_score[:,form+t]),(label,baseline)]:
      x=-z[common];reg=stats.pooled_regression(x,y,weight);ic.append(dict(**meta,date=str(date),h=h,comparison=comparison,signal=signal,ic=stats.spearman(x,y,weight),matched_series=int(common.sum()),eligible_series=int(np.sum(w>0)),matched_fraction=float(common.sum()/max(1,np.sum(w>0))),sum_w=float(weight.sum()),sum_x=float(weight@x),sum_y=float(weight@y),sum_xx=float(weight@(x*x)),sum_xy=float(weight@(x*y)),n=int(common.sum()),FMB_slope=reg['slope']))
    LLT_IC_seconds+=time.monotonic()-LLT_IC_start
   LLT_IC_start=time.monotonic()
   for h in [5,20]:
    if form+t+h>=levels.shape[1]:continue
    target_change=levels[:,form+t+h]-levels[:,form+t];valid_future=np.isfinite(e[:,form+t+1:form+t+h+1]).all(axis=1);common=valid_future&np.isfinite(target_change)&np.isfinite(slope_probability[:,form+t])&(w>0);slope_ic.append(dict(**meta,date=str(date),h=h,ic=stats.spearman(slope_probability[common,form+t],target_change[common],w[common]),matched_series=int(common.sum())))
   LLT_IC_seconds+=time.monotonic()-LLT_IC_start
  timing=primary_root/'LLT_timing'
  if done(timing):timing_record=J(timing/'time.json')
  else:
   timing_record=dict(measured_LLT_this_resume_fit_posterior_seconds=LLT_posterior_seconds,measured_LLT_IC_seconds=LLT_IC_seconds,measured_LLT_total_seconds=float(LLT_test.get('measured_LLT_seconds',0))+max(0,LLT_posterior_seconds-(0 if LLT_was_cached else float(LLT_test.get('measured_LLT_seconds',0))))+LLT_IC_seconds);finish(timing,{'time.json':timing_record},phase='descriptive_LLT_timing')
  LLT_test.update(timing_record);LLT_tests.append(LLT_test);print('LLT measured',kind,year,wi,LLT_test['measured_LLT_seconds'],'fit seconds;',LLT_test['measured_LLT_IC_seconds'],'descriptive IC seconds',flush=True)
 return finish(root,{'params.csv.gz':pd.DataFrame(params),'pooled.csv.gz':pd.DataFrame(pooled_rows),'ic.csv.gz':pd.DataFrame(ic),'construction_correlation.csv.gz':pd.DataFrame(correlation),'signals.npz':array_bytes(**arrays),'LLT_models.csv.gz':pd.DataFrame(LLT_models),'LLT_tests.csv.gz':pd.DataFrame(LLT_tests),'LLT_slope_IC.csv.gz':pd.DataFrame(slope_ic),'LLT_example.csv.gz':LLT_example,'example.csv.gz':example if example is not None else pd.DataFrame(columns=['date','X','M','R','series','test_year','definition'])},phase='B',kind=kind,year=year,wi=wi_only)
