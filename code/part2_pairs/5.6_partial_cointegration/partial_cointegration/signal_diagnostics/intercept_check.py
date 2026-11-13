"""Intercept-channel diagnostic; no new pooled/OU fits or go decision."""
import pathlib,sys,importlib.util,signal,ast
P=pathlib.Path('/Users/maxiyao/FYP_runs/partial_coint_kalman_v1');HERE=P/'audit/A1_sanity_v2';OUT=P/'outputs/A1_sanity_v2'
sys.path[:0]=[str(P/'audit/disk_budget_10GiB_v1'),str(P/'scripts')]
import entry as resource_policy
import numpy as np,pandas as pd
from io_utils import *
from groups import done,finish,array_bytes
import bootstrap,residuals,pci,report_partial as rp
spec=importlib.util.spec_from_file_location('_sanity_v1_readonly',P/'audit/A1_sanity_v1/audit.py');v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)

def no_fit(*a,**kw):raise AssertionError('No new pooled MLE, per-series MLE, rollingOU or policy fits')
def authenticate():
 resource_policy.authenticate();bootstrap.validate_sources();v1.authenticate();reg=J(HERE/'registration.json')
 for rel,h in reg['files'].items():assert H(P/rel)==h,rel
 policy=J(HERE/'hedge_policy.json');assert policy['policy'] in ['strict_cached_only','replay_original_hedge'], 'Hedge-history decision pending'
 return policy

def add_intercept(original,alpha,signs,h):
 """Remove alpha from every signed future increment, not from a signed sum."""
 N,T=alpha.shape;out=np.full_like(original,np.nan)
 for t in range(T-h):out[:,t]=original[:,t]+alpha[:,t]*signs[:,t+1:t+h+1].sum(axis=1)
 return out

def residualize(score,alpha,w):
 """Requested descriptive same-day IPW projection; no outcome enters it."""
 w=np.asarray(w,float);w=w/w.sum();am=float(w@alpha);sm=float(w@score);ac=alpha-am;sc=score-sm;den=float(w@(ac*ac));b=float(w@(ac*sc))/den if den>0 else 0.;return sc-b*ac

def year_data(kind,year):
 c,d,a=bootstrap.core_modules();s=c.source();panel=d.panel();Y=panel['dates'].astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(Y==year);ids=[r['pair'] for r in panel['meta']['pairs'][str(year)]] if kind=='RP' else list(panel['meta']['symbols']);lookup={name:i for i,name in enumerate(ids)};time_lookup={int(t):i for i,t in enumerate(ts)};alpha=np.full((len(ids),len(ts)),np.nan);original_cache=s.cache
 def cache(stem):
  value=original_cache(stem)
  if stem.startswith(f'runtime/jobs/RP_{year}_') and kind=='RP':
   for row in value['series']:
    if row['pair'] in lookup:np.testing.assert_array_equal(row['indices'],ts);alpha[lookup[row['pair']]]=row['alpha']
  if stem.startswith(f'runtime/jobs/RF_{year}_') and kind=='RF':
   for day in value['days']:
    if day['status']=='ok':alpha[day['indices'],time_lookup[day['index']]]=day['alpha']
  return value
 s.cache=cache
 try:data=v1.data_for_year(kind,year)
 finally:s.cache=original_cache
 np.testing.assert_array_equal(data['ids'],ids);data['alpha']=alpha;data['Ytr']=np.stack([np.stack([add_intercept(data['target'][wi,:,:,hi],alpha,data['signs'][wi],h) for hi,h in enumerate([1,5])],axis=-1) for wi in range(3)])
 return data

def check_parameter(stats,name,actual,sealed):
 actual=np.asarray(actual,dtype=float);sealed=np.asarray(sealed,dtype=float);assert actual.shape==sealed.shape
 np.testing.assert_allclose(actual,sealed,rtol=1e-10,atol=1e-8,equal_nan=True,err_msg='Hedge replay versus sealed '+name)
 good=np.isfinite(actual)&np.isfinite(sealed);error=float(np.max(abs(actual[good]-sealed[good]))) if good.any() else 0.
 row=stats.setdefault(name,dict(arrays_checked=0,elements_checked=0,bitwise_equal_arrays=0,max_abs_error=0.));row['arrays_checked']+=1;row['elements_checked']+=actual.size;row['bitwise_equal_arrays']+=int(actual.tobytes()==sealed.tobytes());row['max_abs_error']=max(row['max_abs_error'],error)

def rp_references(year):
 root=OUT/f'RP_hedge_reference/{year}'
 if done(root):return
 c,d,a=bootstrap.core_modules();panel=d.panel();Y=panel['dates'].astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(Y==year);refs={};source=c.source();digests={}
 for part in range(12):
  stem=f'runtime/jobs/RP_{year}_{part:02d}';refs.update({v['pair']:v for v in source.cache(stem)['series']});digests[stem+'.pkl.gz']=source.expected[stem+'.pkl.gz']
 names=[v['pair'] for v in panel['meta']['pairs'][str(year)]];alpha=[];beta=[]
 for name in names:
  v=refs[name];np.testing.assert_array_equal(v['indices'],ts);alpha.append(v['alpha']);beta.append(v['beta'])
 finish(root,{'parameters.npz':array_bytes(ids=np.asarray(names,str),indices=ts,alpha=np.asarray(alpha),beta=np.asarray(beta)),'sources.json':digests},phase='sealed_RP_parameter_reference_for_item3_only');print('Item3 sealed RP reference copied',year,flush=True)

def rp_reference_subset(year,data):
 if '_RP_reference_subset' in data:return data['_RP_reference_subset']
 lookup={v:i for i,v in enumerate(data['ids'])};refs={}
 for yy in range(2004,year+1):
  root=OUT/f'RP_hedge_reference/{yy}';assert done(root)
  with np.load(root/'parameters.npz',allow_pickle=False) as z:
   names=z['ids'];dates=z['indices'];alpha=z['alpha'];beta=z['beta']
   for j,name in enumerate(names):
    if str(name) in lookup:refs.setdefault(lookup[str(name)],[]).append((dates.copy(),alpha[j].copy(),beta[j].copy()))
 data['_RP_reference_subset']=refs;return refs

def rf_alpha_free(year):
 root=OUT/f'RF_alpha_free/{year}'
 if done(root):return
 import copy
 c,d,a=bootstrap.core_modules();p=d.panel();R=p['returns'];Y=p['dates'].astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(Y==year);N=R.shape[1];months={}
 # Use the literal original fallback body (history/member/PCA/OLS definitions),
 # replaying it independently even where parameters are cached, for validation.
 path=P/'scripts/residuals.py';outer=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='rf_year');original=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='day');node=copy.deepcopy(original);start=next(i for i,v in enumerate(node.body) if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='history' for t in v.targets));node.body=node.body[start:];node.name='replay_day';ns=dict(c=c,d=d,R=R,N=N,months=months,np=np);ast.fix_missing_locations(node);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
 oldroot=P/f'runtime/RF_increments/{year}';assert done(oldroot)
 with np.load(oldroot/'increments.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['indices'],ts);old=z['e']
 ep=np.full_like(old,np.nan);error=0.;checks={};covered=0;uncached=0
 for j,t in enumerate(ts):
  checkpoint()
  if t==0:continue
  if j%30==0:print('Item3 RF replay',year,j,'/',len(ts),'covered checked',covered,flush=True)
  prior=int(t)-1;month=str(p['dates'][prior].astype('datetime64[M]'));yy,mm=map(int,month.split('-'));stem=f'runtime/jobs/RF_{yy}_{mm:02d}'
  if month not in months:
   months[month]={v['index']:v for v in c.source().cache(stem)['days']} if stem+'.pkl.gz' in c.source().expected else {}
   if len(months)>3:months.pop(next(iter(months)))
  sealed=months[month].get(prior);model=ns['replay_day'](prior)
  if sealed is not None:
   if sealed['status']!='ok':assert model is None,('Previously unavailable hedge became available',prior);continue
   assert model is not None;ids,W,alpha,beta=model;np.testing.assert_array_equal(ids,sealed['indices']);check_parameter(checks,'weights',W,sealed['weights']);check_parameter(checks,'alpha',alpha,sealed['alpha']);check_parameter(checks,'beta',beta,sealed['beta']);covered+=1
   # Retain bit-identical sealed values on covered dates after independent check.
   model=(sealed['indices'],sealed['weights'],sealed['alpha'],sealed['beta'])
  elif model is not None:uncached+=1
  if model is None:continue
  ids,W,alpha,beta=model;replay=residuals.factor_out_of_sample(R[t],ids,W,alpha,beta);replay[~p['membership'][t]]=np.nan;np.testing.assert_allclose(replay,old[:,j],rtol=1e-10,atol=1e-8,equal_nan=True);good=np.isfinite(replay);error=max(error,float(np.max(abs(replay[good]-old[good,j]))) if good.any() else 0.)
  ep[:,j]=residuals.factor_out_of_sample(R[t],ids,W,np.zeros_like(alpha),beta);ep[~p['membership'][t]]=np.nan
 np.testing.assert_array_equal(np.isfinite(ep),np.isfinite(old));finish(root,{'increments.npz':array_bytes(eprime=ep,indices=ts),'proof.json':dict(original_receipt_sha256=H(oldroot/'complete.json'),original_replay_max_abs_error=error,pooled_parameters_refit=False,audit_registration_sha256=H(HERE/'registration.json'),covered_days_independently_replayed=covered,uncached_days_replayed=uncached,parameter_checks=checks,covered_days_use_exact_sealed_values=True,only_item3=True)},phase='diagnostic_original_hedge_replay_alpha_removed')

def alpha_free_signal(kind,year,wi,data,policy):
 if policy['policy']=='strict_cached_only':return None
 root=OUT/f'alpha_free_signals/{kind}_{year}_{wi}'
 if done(root):
  with np.load(root/'signal.npz',allow_pickle=False) as z:return z['sK_alpha_free']
 c,d,a=bootstrap.core_modules();panel=d.panel();Y=panel['dates'].astype('datetime64[Y]').astype(int)+1970;window=np.flatnonzero((Y>=year-3)&(Y<=year));form=int(np.sum(Y[window]<year));fitroot=P/f'runtime/B_fits/{kind}_{year}_{wi}/pooled';assert done(fitroot);fit=J(fitroot/'pooled.json');eligible=J(P/f'runtime/part_B1/{kind}_{year}_{wi}/formation_masks.json');ids=np.array([r['index'] for r in eligible],int);score=np.full((len(data['ids']),len(data['dates'])),np.nan);seed=c.source().prereg['null_seed']
 if kind=='RF':
  ep=[]
  for yy in range(year-3,year+1):
   r=OUT/f'RF_alpha_free/{yy}';assert done(r)
   with np.load(r/'increments.npz',allow_pickle=False) as z:ep.append(z['eprime'])
  ep=np.concatenate(ep,axis=1);prefix=np.zeros(len(data['ids']))
  first=int(Y.min())
  for yy in range(first,year-3):
   r=OUT/f'RF_alpha_free/{yy}';assert done(r)
   with np.load(r/'increments.npz',allow_pickle=False) as z:prior=z['eprime'];dates=panel['dates'][z['indices']]
   for i in ids:
    sign=np.ones(len(dates)) if wi==0 else d.M.sign_series(seed,'RF|'+data['ids'][i],wi-1,dates);prefix[i]+=np.nansum(prior[i]*sign)
 else:ep=None
 for offset in range(0,len(ids),75):
  checkpoint();ii=ids[offset:offset+75];levels=[]
  if offset%300==0:print('Item3 frozen-parameter signal',kind,year,wi,offset,'/',len(ids),flush=True)
  for i in ii:
   if kind=='RP':
    cache=data.setdefault('_RP_unsigned_alpha_free',{})
    if i not in cache:
     row=panel['meta']['pairs'][str(year)][i];ia=d.index(row['leg_A'],panel);ib=d.index(row['leg_B'],panel);assert min(ia,ib)>=0;aa=panel['returns'][:window[-1]+1,ia];bb=panel['returns'][:window[-1]+1,ib];old,alpha,beta=residuals.pair_out_of_sample(aa,bb);values=aa[60:]-beta*bb[60:];values[~np.isfinite(old)]=np.nan;np.testing.assert_allclose(old[window[form+1:]-60],data['e'][i,1:],rtol=1e-10,atol=1e-8,equal_nan=True)
     checks=data.setdefault('_RP_parameter_checks',{})
     for refdates,refalpha,refbeta in rp_reference_subset(year,data).get(int(i),[]):
      used=(refdates>=59)&(refdates<window[-1]);ix=refdates[used]-59;check_parameter(checks,'alpha',alpha[ix],refalpha[used]);check_parameter(checks,'beta',beta[ix],refbeta[used]);alpha[ix]=refalpha[used];beta[ix]=refbeta[used]
     values=aa[60:]-beta*bb[60:];values[~np.isfinite(old)]=np.nan;cache[i]=values
    values=cache[i];dates=panel['dates'][60:window[-1]+1];sign=np.ones(len(values)) if wi==0 else d.M.sign_series(seed,f'RP|{year}:{data["ids"][i]}',wi-1,dates);values=values*sign;lev=np.cumsum(np.where(np.isfinite(values),values,0.));lev[~np.isfinite(values)]=np.nan;levels.append(lev[window-60])
   else:
    values=ep[i];sign=np.ones(len(window)) if wi==0 else d.M.sign_series(seed,'RF|'+data['ids'][i],wi-1,panel['dates'][window]);values=values*sign;lev=prefix[i]+np.cumsum(np.where(np.isfinite(values),values,0.));lev[~np.isfinite(values)]=np.nan;levels.append(lev)
  levels=np.asarray(levels);_,states=pci.filter_values(levels,fit['rho'],fit['q'],states=True);scale=np.asarray(fit['scale'])[offset:offset+75];part=states[:,form:,1]/np.sqrt(scale[:,None]/(1-fit['rho']**2));part[~np.isfinite(levels[:,form:])]=np.nan;score[ii]=part
 # Same global cumulative origin and original formation-window initialization;
 # no changed warm-up interval, covariance or scale.
 original=P/f'outputs/amended_A1/groups/{kind}_{year}_{wi}';assert done(original)
 with np.load(original/'signals.npz',allow_pickle=False) as z:np.testing.assert_array_equal(np.isfinite(score),np.isfinite(z[f'sK_{wi}']))
 finish(root,{'signal.npz':array_bytes(sK_alpha_free=score),'proof.json':dict(pooled_receipt_sha256=H(fitroot/'complete.json'),same_rho_q_scale=True,no_MLE_or_OU_refit=True,audit_registration_sha256=H(HERE/'registration.json'),hedge_policy_sha256=H(HERE/'hedge_policy.json'),diagnostic_only=True,parameter_checks=data.get('_RP_parameter_checks',{}),covered_days_use_exact_sealed_values=True,only_item3=True)},phase='alpha_free_increment_frozen_parameter_signal')
 return score

def job(kind,year,policy,part3=False):
 root=OUT/(f'groups_part3/{kind}_{year}' if part3 else f'groups/{kind}_{year}')
 if done(root):return
 data=year_data(kind,year);stats,cache=v1.cached_stats();records=[];projections=[]
 for wi in range(3):
  a=P/f'outputs/amended_A1/groups/{kind}_{year}_{wi}';assert done(a)
  with np.load(a/'signals.npz',allow_pickle=False) as z:K=z[f'sK_{wi}'];roll=z[f'roll_new_{wi}']
  e=data['e']*data['signs'][wi];signals=dict(K=K,roll_new=roll,core_roll=data['core'][wi],neg_e=-e);prime=alpha_free_signal(kind,year,wi,data,policy) if part3 else None;old=v1.rp.frame(P/f'outputs/A1_sanity_v1/groups/{kind}_{year}/daily_IC.csv.gz');old=old[(old.world==wi)&(old.predictor=='K')&old.case.isin(['original_h1','original_h5'])]
  old=old.set_index(['date','h'])
  for t,date in enumerate(data['dates']):
   if t%20==0:checkpoint()
   cache.clear()
   for hi,h in enumerate([1,5]):
    original=data['target'][wi,:,t,hi];tradable=data['Ytr'][wi,:,t,hi];common=(data['w']>0)&np.isfinite(original)&np.isfinite(tradable)
    for s in signals.values():common &= np.isfinite(s[:,t])
    idx=np.flatnonzero(common);w=data['w'][idx];alpha=data['alpha'][idx,t];assert np.isfinite(alpha).all();baseline=stats.spearman(-K[idx,t],original[idx],w) if not part3 else None;ref=old.loc[(str(date),h)];assert len(idx)==int(ref.n),(kind,year,wi,str(date),h,'mask differs');
    if not part3:np.testing.assert_allclose(baseline,ref.ic,rtol=0,atol=1e-12,equal_nan=True)
    if part3:
     assert prime is not None and np.isfinite(prime[idx,t]).all();records.append(dict(family=kind,year=year,world=wi,date=str(date),h=h,scope='v1_fixed_common_mask',signal='K_alpha_free_increment',target='tradable',ic=stats.spearman(-prime[idx,t],tradable[idx],w),n=len(idx)));continue
    if len(idx):residual=residualize(K[idx,t],alpha,w)
    else:residual=np.empty(0)
    for target,y in [('original',original[idx]),('tradable',tradable[idx])]:
     for name,s in signals.items():
      predictor=s[idx,t] if name=='neg_e' else -s[idx,t];records.append(dict(family=kind,year=year,world=wi,date=str(date),h=h,scope='v1_fixed_common_mask',signal=name,target=target,ic=stats.spearman(predictor,y,w),n=len(idx)))
     if target=='original':
      for name,x in [('neg_alpha',-alpha),('K_residualized_on_alpha',-residual)]:records.append(dict(family=kind,year=year,world=wi,date=str(date),h=h,scope='v1_fixed_common_mask',signal=name,target=target,ic=stats.spearman(x,y,w),n=len(idx)))
    if prime is not None:
     assert np.isfinite(prime[idx,t]).all();records.append(dict(family=kind,year=year,world=wi,date=str(date),h=h,scope='v1_fixed_common_mask',signal='K_alpha_free_increment',target='tradable',ic=stats.spearman(-prime[idx,t],tradable[idx],w),n=len(idx)))
    # Exact original A1 K/roll pair mask, separate from the stricter v1 panel.
    mask=(data['w']>0)&np.isfinite(K[:,t])&np.isfinite(roll[:,t])&np.isfinite(original)&np.isfinite(tradable);ww=data['w'][mask]
    for target,y in [('original',original[mask]),('tradable',tradable[mask])]:
     delta=stats.spearman(-K[mask,t],y,ww)-stats.spearman(-roll[mask,t],y,ww);records.append(dict(family=kind,year=year,world=wi,date=str(date),h=h,scope='A1_original_pair_mask',signal='K_minus_roll_new',target=target,ic=delta,n=int(mask.sum())))
    # Earlier chapter's complete raw-core universe and original weighting.
    weights=np.ones(len(K)) if kind=='RF' else data['w'];mask=np.isfinite(data['core'][wi,:,t])&np.isfinite(original)&np.isfinite(tradable)&(weights>0)
    for target,y in [('original',original[mask]),('tradable',tradable[mask])]:records.append(dict(family=kind,year=year,world=wi,date=str(date),h=h,scope='earlier_chapter_original_scope',signal='core_headline',target=target,ic=stats.spearman(-data['core'][wi,mask,t],y,weights[mask]),n=int(mask.sum())))
 finish(root,{'daily_IC.csv.gz':pd.DataFrame(records),'proof.json':dict(v1_masks_and_original_IC_reproduced=True,A1_receipts_unchanged=True,audit_registration_sha256=H(HERE/'registration.json'),hedge_policy_sha256=H(HERE/'hedge_policy.json'),Part_C_started=False,part3=part3)},phase='A1_sanity_v2_intercept_diagnostic')
 print('Intercept audit sealed',kind,year,flush=True)

def paired_summary(frame,stats):
 v=frame.pivot(index='date',columns='world',values='ic').reindex(columns=[0,1,2]).dropna();real=v[0];null=(v[1]+v[2])/2;delta=real-null;years=np.asarray(v.index,dtype='datetime64[D]').astype('datetime64[Y]').astype(int)+1970;result=stats.estimate(pd.DataFrame(dict(value=delta.to_numpy(),test_year=years,weight=np.ones(len(v)))),draws=5000)
 return dict(real=float(real.mean()),N2_mean=float(null.mean()),real_minus_N2=result['estimate'],ci95_low=result['ci95_low'],ci95_high=result['ci95_high'],years=result['clusters'],common_days=len(v),bootstrap_seed=20261026,draws=5000)

def report(policy,base_only=False):
 root=OUT/('base_report' if base_only else 'report');report_name='A1_sanity_v2_items_1_2_4.md' if base_only else 'A1_sanity_v2.md'
 if done(root):publish(P/report_name,(root/'A1_sanity_v2.md').read_bytes());return
 frames=[]
 for kind in ['RP','RF']:
  for year in range(2004,2026):
   r=OUT/f'groups/{kind}_{year}';assert done(r);frames.append(rp.frame(r/'daily_IC.csv.gz'))
 if not base_only and policy['policy']=='replay_original_hedge':
  for kind in ['RP','RF']:
   for year in range(2004,2026):
    r=OUT/f'groups_part3/{kind}_{year}';assert done(r);frames.append(rp.frame(r/'daily_IC.csv.gz'))
 f=pd.concat(frames,ignore_index=True);stats=residuals.imported_stats();rows=[]
 for key,g in f.groupby(['scope','signal','target','family','h']):rows.append(dict(zip(['scope','signal','target','family','h'],key),**paired_summary(g,stats)))
 summary=pd.DataFrame(rows)
 # Reproduce the earlier chapter on its original dates/masks/weights before
 # interpreting alpha removal. Do not force it onto the A1/v1 filtered universe.
 c,d,a=bootstrap.core_modules();source=c.source();old=rp.frame(source.verify('outputs/tables/ic_daily.csv.gz'));old=old[old.definition.isin(['RP','RF'])&old.h.isin([1,5])].copy();old['world_id']=np.where(old.world=='real',0,old.replicate+1)
 own=f[(f.scope=='earlier_chapter_original_scope')&(f.target=='original')];ref=own.merge(old[['definition','world_id','date','h','ic']],left_on=['family','world','date','h'],right_on=['definition','world_id','date','h'],validate='one_to_one',suffixes=('_audit','_sealed'));assert len(ref)==len(own);np.testing.assert_allclose(ref.ic_audit,ref.ic_sealed,rtol=0,atol=1e-12,equal_nan=True)
 old_tests=rp.frame(source.verify('outputs/tables/predictions.csv'));reproduction=[]
 for test,kind in [('H1','RP'),('H2','RF')]:
  sealed=old_tests[old_tests.id==test].iloc[0];row=summary[(summary.scope=='earlier_chapter_original_scope')&(summary.family==kind)&(summary.h==5)&(summary.target=='original')].iloc[0];np.testing.assert_allclose(row.real_minus_N2,sealed.estimate,rtol=0,atol=1e-12);reproduction.append(dict(family=kind,sealed_headline_excess=float(sealed.estimate),reproduced_original_excess=float(row.real_minus_N2),original_scope_reproduced=True))
 # Exact A1 original pair-mask contrast must also reproduce, before changing Y.
 a1=J(P/'outputs/amended_A1/report/summary.json')
 for test in a1['tests']:
  r=summary[(summary.scope=='A1_original_pair_mask')&(summary.family==test['definition'])&(summary.h==5)&(summary.target=='original')].iloc[0];np.testing.assert_allclose(r.real_minus_N2,test['estimate'],rtol=0,atol=1e-12)
 def table(frame):return rp.table(list(frame.columns),[[rp.num(v) if isinstance(v,(float,np.floating)) else str(v) for v in row] for row in frame.itertuples(index=False,name=None)])
 display=summary.copy();display['CI_excess']=display.apply(lambda r:f"[{rp.num(r.ci95_low)}, {rp.num(r.ci95_high)}]",axis=1);display=display[['signal','target','family','h','real','N2_mean','real_minus_N2','CI_excess','scope','years','common_days']]
 target_proof='Y_tr,real=Y_original+h alpha_t. Y_tr,N2=Y_original,N2+alpha_t Σ[u=t+1..t+h]xi_i,u. Adding h alpha_t to an N2 horizon is incorrect: each future residual increment has its own sign. No future sign is used in any signal. Alpha-only means the already sealed real-hedge alpha_t for every world; the core never re-estimated hedge alpha on N2 data.'
 item3='Item3 has not started. Items1,2,4 are reported first and are independent of hedge replay.' if base_only else 'Alpha-free increment signal uses the SAME rho,q and original per-series formation scale, without reconcentrating scale or refitting. eprime is signed in its entirety before N2 cumulation. The original missing masks and formation eligibility IDs are retained, with diffuse M/stationary R initialization. The global cumulative origin, formation-window initialization and covariance are retained. Replayed hedge functions/definitions are unchanged. Every covered replayed parameter date is checked at rtol1e−10/atol1e−8; bitwise-equal array counts and maximum errors are retained separately. Covered dates use the exact sealed coefficient values after independent checks; only uncovered dates use the unchanged original hedge calculation. Original residuals are checked against retained RF increments before removing alpha. The original result files are never modified. This diagnostic applies parameters estimated on alpha-subtracted data to a different input process and is not a newly fitted or validated model.' if policy['policy']=='replay_original_hedge' else 'Item3 is unavailable under strict no-hedge-recomputation: current-fold formation beta and early RF PCA histories are not completely retained. No initialization, missing-value substitution, selective surviving-history analysis or fit is invented. The other requested calculations remain available; this limitation is explicit.'
 if base_only:item3='Item3 has not started; this report covers items1,2,4 before any hedge replay.'
 body='\n\n'.join(['# A1_sanity_v2 — intercept subtraction audit','Read-only diagnostic analysis of sealed inputs. No change to A1 GO, no trading/policy fitting, no Part C execution. CIs are descriptive, without a new go/no-go or Holm decision.','## Compact results',table(display),'All ICs use minus-score; neg_e and neg_alpha already carry their minus sign. CI is the real-minus-equal-mean-N2 paired year-block95% interval, using the exact frozen TAV seed20261026 and5000 draws. Point estimates use common dates across all3 worlds. h1 and h5 are reported separately.','## Target and null definitions',target_proof,'Y_tr follows the user\'s specified unnormalised frozen-hedge payoff, before costs or gross-capital normalisation. It is not a portfolio backtest. The original outcomes, dates and missing-price masks remain frozen; removing alpha does not create observations that were previously unavailable.','## Masks and intercept-only controls','Scope v1_fixed_common_mask exactly reconstructs A1_sanity_v1 original_h1/original_h5 masks for all four original predictors and asserts the original K ICs agree to1e−12 on every date. K,roll_new,core_roll,−e and −alpha share this scope. K_residualized_on_alpha is the same-day IPW least-squares residual K−c0−c1 alpha, with an intercept and the original weights; no outcome enters the projection. This is the specifically requested descriptive cross-sectional projection, not a new time-series/forecast fit. Constant alpha reduces to demeaning K. The extra-day/first-session restrictions from v1 stay in place.','## A1 K1/K2 contrasts on tradable targets','Scope A1_original_pair_mask uses exactly the original A1 K/roll pair mask, rather than v1\'s stricter4-signal cross-section. K_minus_roll_new is the within-world IC difference; real-minus-N2 of that row is K1 for RF/K2 for RP. Original-target contrasts are independently reproduced before interpreting the tradable-target changes. No A2 trading restriction is applied to any IC.','## Alpha-free signal diagnostic',item3,'## Earlier chapter robustness','Scope earlier_chapter_original_scope uses the complete raw core-score universe, original RP IPW and RF equal weights, with no K/rolling_new/−e or half-life selection imposed. Its original per-day ICs are checked against sealed residual_ou_real_v1/ic_daily and headline h5 estimates against sealed predictions.csv before reporting Y_tr results.',table(pd.DataFrame(reproduction)),'## Integrity and stop','Original A1, B1, v1 audit and B2 files/receipts are unchanged. No pooled or per-series MLE, rollingOU, new policy fit, cell selection or trading trial is performed. The sole hedge-replay policy is documented in audit/A1_sanity_v2/hedge_policy.json; it cannot silently change after registration. Original pooled rho/q/scales remain immutable. All new outputs are separate under outputs/A1_sanity_v2. Part C remains on hold.'])+'\n'
 finish(root,{'summary.csv.gz':summary,'compact_table.csv.gz':display,'chapter_reproduction.csv.gz':pd.DataFrame(reproduction),'A1_sanity_v2.md':body.encode(),'status.json':dict(status='complete_items_1_2_4' if base_only else 'complete',diagnostic_only=True,A1_go_unchanged=True,Part_C_started=False,hedge_policy=policy['policy'])},phase='A1_sanity_v2_report');publish(P/report_name,body)

def main(base_only=False):
 policy=authenticate();install_guard();c,d,a=bootstrap.core_modules();pci.fit=no_fit
 import llt
 llt.fit=no_fit;d.M.ou_fit=no_fit
 os.environ['GATE_POWER_STOP']=str(P/f'runtime/STOP_A1_sanity_v2_{os.getpid()}');m=memory_snapshot();os.environ['PARTIAL_RAM_GUARD']='1';os.environ['PARTIAL_INITIAL_SWAPOUTS']=str(m['swapouts']);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 with (P/'runtime/A1_sanity_v2.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);checkpoint()
  # Items1,2,4 categorically cannot invoke a hedge replay.
  names=['factor_window','regress','pair_rolling','eigenportfolio'];saved={name:getattr(d.M,name) for name in names};saved_pair=residuals.pair_out_of_sample
  try:
   for name in names:setattr(d.M,name,no_fit)
   residuals.pair_out_of_sample=no_fit
   for kind in ['RP','RF']:
    for year in range(2004,2026):job(kind,year,policy,False)
   report(policy,True)
   print('ITEMS1,2,4 COMPLETE — BEFORE ITEM3 HEDGE REPLAY',flush=True);print(rp.frame(OUT/'base_report/compact_table.csv.gz').to_string(index=False),flush=True);print('Report:',P/'A1_sanity_v2_items_1_2_4.md',flush=True)
   journal('A1_sanity_v2_items_1_2_4_complete',receipt_sha256=H(OUT/'base_report/complete.json'),Part_C_started=False)
  finally:
   for name,value in saved.items():setattr(d.M,name,value)
   residuals.pair_out_of_sample=saved_pair
  if base_only:print('STOPPED after items1,2,4; item3 not started; PartC stays on hold.',flush=True);return
  if policy['policy']=='replay_original_hedge':
   print('STARTING ITEM3 ONLY: original hedge replay with sealed-parameter checks; pooled/OU fitting remain blocked.',flush=True)
   first=int(d.panel()['dates'].astype('datetime64[Y]').astype(int).min()+1970)
   for year in range(2004,2026):rp_references(year)
   for year in range(first,2026):rf_alpha_free(year)
   for kind in ['RP','RF']:
    for year in range(2004,2026):job(kind,year,policy,True)
  report(policy);authenticate();print(rp.frame(OUT/'report/compact_table.csv.gz').to_string(index=False),flush=True);journal('A1_sanity_v2_complete',receipt_sha256=H(OUT/'report/complete.json'),Part_C_started=False);print('STOPPED after A1_sanity_v2. A1 go/no-go unchanged. Part C stays on hold. Report:',P/'A1_sanity_v2.md',flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--base-only',action='store_true');args=parser.parse_args();main(args.base_only)
