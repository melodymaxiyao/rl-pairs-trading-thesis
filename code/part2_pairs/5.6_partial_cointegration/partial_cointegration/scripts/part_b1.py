"""Primary-only stage. Reuse all previously committed primary fits and groups."""
import gzip
import numpy as np,pandas as pd
from io_utils import *
from groups import *
import residuals,pci

def root_for(kind,year,wi):return ROOT/f'runtime/part_B1/{kind}_{year}_{wi}'
def job(kind,year,wi):
 root=root_for(kind,year,wi)
 if done(root):return dict(root=str(root),cached=True)
 group=residuals.load_group(kind,year);form=group['formation'];counts=np.isfinite(group['e'][:,:form]).sum(axis=1)
 bucket_rows=[dict(index=i,n=int(n),missing_share=float(1-n/form),bucket=500 if n<600 else 600 if n<700 else 700) for i,n in enumerate(counts) if n>=500]
 old=ROOT/f'runtime/part_B/{kind}_{year}_{wi}'
 if done(old):
  # Read primary columns only; preserve old secondary/LLT bytes and receipts.
  ic=pd.read_csv(old/'ic.csv.gz',float_precision='round_trip');ic=ic[ic.comparison.isin(['roll_new','sealed_core'])]
  with np.load(old/'signals.npz',allow_pickle=False) as z:arrays={k:z[k] for k in [f'sK_{wi}',f'roll_new_{wi}']}
  outputs={'ic.csv.gz':ic,'signals.npz':array_bytes(**arrays),'pooled.csv.gz':(old/'pooled.csv.gz').read_bytes(),'construction_correlation.csv.gz':(old/'construction_correlation.csv.gz').read_bytes(),'example.csv.gz':(old/'example.csv.gz').read_bytes(),'formation_masks.json':bucket_rows,'reuse.json':dict(original_receipt=str((old/'complete.json').relative_to(ROOT)),sha256=H(old/'complete.json'),no_fits=True)}
  return finish(root,outputs,phase='B1',reused_completed_group=True)
 target,core_score,w=residuals.core_targets(kind,year);stats=residuals.imported_stats();e,levels,roll,half_roll=residuals.signed(group,wi);ids=np.flatnonzero(counts>=500);world='real' if wi==0 else 'N2';rep=0 if wi==0 else wi-1
 primary_root=ROOT/f'runtime/B_fits/{kind}_{year}_{wi}';pooled_root=primary_root/'pooled';previous=ROOT/f'runtime/B_fits/{kind}_{year-1}_{wi}'
 warm=J(previous/'pooled/pooled.json') if year>2004 and done(previous/'pooled') else None
 if done(pooled_root):pooled=J(pooled_root/'pooled.json')
 else:
  pooled=pci.fit(levels[ids,:form],pooled=True,checkpoint=checkpoint,warm_start=warm) if len(ids) else dict(converged=False,reason='No series with >=500valid formation sessions')
  finish(pooled_root,{'pooled.json':pooled},phase='primary_pooled_MLE')
 pooled_rows=[dict(definition=kind,test_year=year,world=world,replicate=rep,**{k:v for k,v in pooled.items() if k not in ['scale','n','runs']},formation_series=len(ids))]
 score=np.full_like(levels,np.nan);example=pd.DataFrame(columns=['date','X','M','R','series','test_year','definition'])
 if pooled['converged']:
  scale=np.asarray(pooled['scale'])
  for offset in range(0,len(ids),75):
   checkpoint();ii=ids[offset:offset+75];_,states=pci.filter_values(levels[ii],pooled['rho'],pooled['q'],states=True);score[ii]=states[:,:,1]/np.sqrt(scale[offset:offset+75,None]/(1-pooled['rho']**2))
   if offset==0 and wi==0:example=pd.DataFrame(dict(date=group['dates'].astype(str),X=levels[ii[0]],M=states[0,:,0],R=states[0,:,1],series=group['ids'][ii[0]],test_year=year,definition=kind))
  if not pooled['half_life']<30:score[:]=np.nan
  score[~np.isfinite(levels)]=np.nan
 ic=[]
 for t,date in enumerate(group['dates'][form:]):
  if t%30==0:checkpoint()
  for hi,h in enumerate([1,5]):
   future=target[wi,:,t,hi]
   for comparison,baseline in [('roll_new',roll[:,form+t]),('sealed_core',core_score[wi,:,t])]:
    common=np.isfinite(score[:,form+t])&np.isfinite(baseline)&np.isfinite(future)&(w>0);weight=w[common];y=future[common]
    for signal,z in [('Kalman',score[:,form+t]),(comparison,baseline)]:
     x=-z[common];ic_value=stats.spearman(x,y,weight);reg=stats.pooled_regression(x,y,weight);ic.append(dict(definition=kind,test_year=year,date=str(date),world=world,replicate=rep,h=h,comparison=comparison,signal=signal,ic=ic_value,matched_series=int(common.sum()),eligible_series=int(np.sum(w>0)),matched_fraction=float(common.sum()/max(1,np.sum(w>0))),sum_w=float(weight.sum()),sum_x=float(weight@x),sum_y=float(weight@y),sum_xx=float(weight@(x*x)),sum_xy=float(weight@(x*y)),n=int(common.sum()),FMB_slope=reg['slope']))
 valid=np.isfinite(e[:,form:])&np.isfinite(group['core_inwindow'][:,form:]);aa=e[:,form:][valid];bb=(group['core_inwindow'][:,form:]*(np.ones_like(e[:,form:]) if wi==0 else np.divide(e[:,form:],group['e'][:,form:],out=np.full_like(e[:,form:],np.nan),where=group['e'][:,form:]!=0)))[valid];good=np.isfinite(aa)&np.isfinite(bb)
 correlation=[dict(definition=kind,test_year=year,world=world,replicate=rep,n=int(good.sum()),correlation=float(np.corrcoef(aa[good],bb[good])[0,1]) if good.sum()>2 else None)]
 return finish(root,{'ic.csv.gz':pd.DataFrame(ic),'signals.npz':array_bytes(**{f'sK_{wi}':score[:,form:],f'roll_new_{wi}':roll[:,form:]}),'pooled.csv.gz':pd.DataFrame(pooled_rows),'construction_correlation.csv.gz':pd.DataFrame(correlation),'example.csv.gz':example,'formation_masks.json':bucket_rows},phase='B1')

def report():
 import report_partial as rp
 root=ROOT/'runtime/B1_report'
 if done(root):return J(root/'decision.json')
 stats=residuals.imported_stats();ic=[];pooled=[]
 for kind in ['RP','RF']:
  for year in range(2004,2026):
   for wi in range(3):
    p=root_for(kind,year,wi);assert done(p);ic.append(rp.frame(p/'ic.csv.gz'));pooled.append(rp.frame(p/'pooled.csv.gz'))
 ic=pd.concat(ic,ignore_index=True);primary=[dict(id=name,**rp.contrast(stats,ic,kind,5)) for name,kind in [('K1','RF'),('K2','RP')]]
 for row,q in zip(primary,stats.holm([r['p_one_sided'] for r in primary])):row['holm_p']=float(q)
 go=any(np.isfinite(r['estimate']) and r['estimate']>0 and r['p_one_sided']<.1 for r in primary)
 decision=dict(decision='GO' if go else 'NO_GO',tests=primary,part_B1_complete=True,part_B2_complete=False,part_C_started=False,rule='Unchanged positive h5point and one-sided unadjusted p<.10 in either family; Holm2 reported',B2_required_before_G_PCI_cells=True)
 inference=[]
 for key,g in ic.groupby(['definition','world','replicate','h','comparison','signal']):
  kind,world,rep,h,comparison,signal=key;g=g.sort_values('date');nw=stats.nw_mean(g.ic,h);fmb=stats.nw_mean(g.FMB_slope,h);inference.append(dict(definition=kind,world=world,replicate=int(rep),h=int(h),comparison=comparison,signal=signal,IC_mean=nw['mean'],IC_NW_SE=nw['se'],FMB_mean=fmb['mean'],FMB_NW_SE=fmb['se'],**stats.pooled_hac(g,h)))
 text='# Part B1 primary decision\n\n'+rp.table(['Test','Estimate','95% CI','Unadjusted p','Holm p'],[(r['id'],rp.num(r['estimate']),f"[{rp.num(r['ci95_low'])}, {rp.num(r['ci95_high'])}]",rp.num(r['p_one_sided']),rp.num(r['holm_p'])) for r in primary])+f'\n\nDecision: {decision["decision"]}. B2 remains secondary and must finish before G_PCI cells. Part C has not started.\n'
 finish(root,{'decision.json':decision,'contrasts.csv.gz':pd.DataFrame(primary),'IC_inference.csv.gz':pd.DataFrame(inference),'ic.csv.gz':ic,'pooled.csv.gz':pd.concat(pooled,ignore_index=True),'B1_report.md':text.encode()},phase='B1_primary_report')
 publish(ROOT/'B1_report.md',text);journal('B1_go_nogo',decision=decision);return decision
