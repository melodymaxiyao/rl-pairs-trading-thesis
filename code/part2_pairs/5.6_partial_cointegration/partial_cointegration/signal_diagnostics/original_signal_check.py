"""Diagnostic only: sealed signals/hedges, no fits, no revised go gate or Part C."""
import pathlib,sys,hashlib,collections,signal
P=pathlib.Path('/Users/maxiyao/FYP_runs/partial_coint_kalman_v1');HERE=P/'audit/A1_sanity_v1';OUT=P/'outputs/A1_sanity_v1'
sys.path[:0]=[str(P/'audit/disk_budget_10GiB_v1'),str(P/'scripts')]
import entry as resource_policy
import numpy as np,pandas as pd
from io_utils import *
from groups import done,finish
import bootstrap,residuals,report_partial as rp

def no_fit(*a,**kw):raise AssertionError('Sanity audit forbids all new MLE/OU/hedge/PCA fits')
def authenticate():
 resource_policy.authenticate();bootstrap.validate_sources();reg=J(HERE/'registration.json')
 for rel,h in reg['files'].items():assert H(P/rel)==h,rel
 return reg

def future_sum(e,offset,h):
 """X[t+offset+h-1]-X[t+offset-1], strict finite endpoints/increments."""
 e=np.asarray(e,float);N,T=e.shape;out=np.full((N,T),np.nan)
 for t in range(max(0,T-offset-h+1)):
  v=e[:,t+offset:t+offset+h];ok=np.isfinite(v).all(axis=1)&np.isfinite(e[:,t+offset-1]);out[ok,t]=v[ok].sum(axis=1)
 return out

def lag(x):
 out=np.full_like(x,np.nan);out[:,1:]=x[:,:-1];return out

def data_for_year(kind,year):
 c,d,a=bootstrap.core_modules();source=c.source();assert source.prereg['horizons'][:2]==[1,5];p=d.panel();dates=np.asarray(p['dates']);years=dates.astype('datetime64[Y]').astype(int)+1970;ts=np.flatnonzero(years==year);T=len(ts);R=p['returns'];meta=p['meta'];digests={}
 def cache(stem):
  value=source.cache(stem);digests[str((source.root/(stem+'.pkl.gz')).relative_to(P.parent))]=source.expected[stem+'.pkl.gz'];return value
 if kind=='RP':
  rows=meta['pairs'][str(year)];ids=[r['pair'] for r in rows];N=len(rows);alpha=np.full((N,T),np.nan);beta=alpha.copy();core=np.full((3,N,T),np.nan);targets=np.full((3,N,T,2),np.nan);A=alpha.copy();B=alpha.copy();refs={}
  for part in range(12):refs.update({r['pair']:r for r in cache(f'runtime/jobs/RP_{year}_{part:02d}')['series']})
  for i,row in enumerate(rows):
   checkpoint();ref=refs[row['pair']];np.testing.assert_array_equal(ref['indices'],ts);alpha[i]=ref['alpha'];beta[i]=ref['beta'];core[:,i]=ref['score'];targets[:,i]=ref['future'][:,:,:2];ia=d.index(row['leg_A'],p);ib=d.index(row['leg_B'],p)
   if min(ia,ib)>=0:A[i]=R[ts,ia];B[i]=R[ts,ib]
  # Identical causal OOS increments from already sealed prior-day coefficients.
  # No regression is executed. First session omitted: prior-fold pair hedge need
  # not be cached; do not invent/re-estimate that missing coefficient.
  e=np.full((N,T),np.nan);e[:,1:]=A[:,1:]-alpha[:,:-1]-beta[:,:-1]*B[:,1:]
  w=np.array([r['ipw'] for r in rows],float);w/=w.sum();seed=source.prereg['null_seed'];signs=np.stack([np.ones((N,T))]+[np.stack([d.M.sign_series(seed,f'RP|{year}:{name}',rep,dates[ts]) for name in ids]) for rep in [0,1]])
  frozen_extra={key:np.full((3,N,T),np.nan) for key in ['frozen_skip1_h5','frozen_tplus2_h1']};maxerr=0.
  for t in range(T):
   for offset,h,key in [(2,5,'frozen_skip1_h5'),(2,1,'frozen_tplus2_h1'),(1,1,None),(1,5,None)]:
    if t+offset+h>T:continue
    v=A[:,t+offset:t+offset+h]-alpha[:,t,None]-beta[:,t,None]*B[:,t+offset:t+offset+h]
    for wi in range(3):
     vv=v*signs[wi,:,t+offset:t+offset+h];ok=np.isfinite(vv).all(axis=1);result=np.where(ok,vv.sum(axis=1),np.nan)
     if key:frozen_extra[key][wi,:,t]=result
     else:
      old=targets[wi,:,t,0 if h==1 else 1];np.testing.assert_allclose(result,old,rtol=1e-11,atol=1e-9,equal_nan=True);good=np.isfinite(old);maxerr=max(maxerr,float(np.max(abs(result[good]-old[good]))) if good.any() else 0.)
 else:
  ids=list(meta['symbols']);N=len(ids);core=np.full((3,N,T),np.nan);targets=np.full((3,N,T,2),np.nan);seed=source.prereg['null_seed'];signs=np.stack([np.ones((N,T))]+[np.stack([d.M.sign_series(seed,'RF|'+name,rep,dates[ts]) for name in ids]) for rep in [0,1]])
  r=P/f'runtime/RF_increments/{year}';assert done(r)
  with np.load(r/'increments.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['indices'],ts);e=z['e']
  digests[str((r/'complete.json').relative_to(P.parent))]=H(r/'complete.json');w=a.capital(dict(kind='RF',ids=ids));lookup={int(t):i for i,t in enumerate(ts)};frozen_extra={key:np.full((3,N,T),np.nan) for key in ['frozen_skip1_h5','frozen_tplus2_h1']};maxerr=0.
  for month in range(1,13):
   for day in cache(f'runtime/jobs/RF_{year}_{month:02d}')['days']:
    checkpoint()
    if day['status']!='ok':continue
    idx=day['indices'];t=lookup[day['index']];core[:,idx,t]=day['score'];targets[:,idx,t]=day['future'][:,:,:2];W=day['weights'];needed=np.flatnonzero(np.any(W!=0,axis=1));alpha=day['alpha'];beta=day['beta'];global_t=day['index']
    for offset,h,key in [(2,5,'frozen_skip1_h5'),(2,1,'frozen_tplus2_h1'),(1,1,None),(1,5,None)]:
     if t+offset+h>T:continue
     rf=R[global_t+offset:global_t+offset+h];v=rf[:,idx]-alpha-(rf[:,needed]@W[needed])@beta.T
     for wi in range(3):
      vv=v*signs[wi,idx,t+offset:t+offset+h].T;ok=np.isfinite(vv).all(axis=0);result=np.where(ok,vv.sum(axis=0),np.nan)
      if key:frozen_extra[key][wi,idx,t]=result
      else:
       old=targets[wi,idx,t,0 if h==1 else 1];np.testing.assert_allclose(result,old,rtol=1e-11,atol=1e-9,equal_nan=True);good=np.isfinite(old);maxerr=max(maxerr,float(np.max(abs(result[good]-old[good]))) if good.any() else 0.)
 # Never infer N2 targets from real values by merely flipping a summed horizon.
 # Every future increment is signed using the same sealed date/identity stream.
 return dict(ids=ids,dates=dates[ts],e=e,signs=signs,target=targets,core=core,w=w,extra=frozen_extra,source_digests=digests,target_reproduction_max_abs_error=maxerr)

def cached_stats():
 stats=residuals.imported_stats();original=stats.spearman.__globals__['weighted_ranks'];cache={}
 def ranks(x,w):
  key=(np.asarray(x).tobytes(),np.asarray(w).tobytes())
  if key not in cache:cache[key]=original(x,w)
  # Original spearman centers ranks in place: never expose the cached array.
  return cache[key].copy()
 stats.spearman.__globals__['weighted_ranks']=ranks
 return stats,cache

def job(kind,year):
 root=OUT/f'groups/{kind}_{year}'
 if done(root):return
 data=data_for_year(kind,year);stats,rank_cache=cached_stats();rows=[];exact=[];placebo=[];comparisons=[];proofs=[];dependency_flags=[]
 for wi in range(3):
  a=P/f'outputs/amended_A1/groups/{kind}_{year}_{wi}';assert done(a);proof=J(a/'provenance.json');fitroot=P/f'runtime/B_fits/{kind}_{year}_{wi}/pooled';assert done(fitroot);assert proof['pooled_receipt_sha256']==H(fitroot/'complete.json');proofs.append(dict(world=wi,A1_receipt_sha256=H(a/'complete.json'),pooled_receipt_sha256=H(fitroot/'complete.json')))
  with np.load(a/'signals.npz',allow_pickle=False) as z:K=z[f'sK_{wi}'];roll=z[f'roll_new_{wi}']
  old=rp.frame(a/'ic.csv.gz');pair=old[old.comparison=='roll_new'];core_old=old[(old.comparison=='sealed_core')&(old.signal=='sealed_core')]
  for h in [1,5]:
   kg=pair[(pair.h==h)&(pair.signal=='Kalman')].set_index('date');rg=pair[(pair.h==h)&(pair.signal=='roll_new')].set_index('date');cg=core_old[core_old.h==h].set_index('date');dates=kg.index[kg.ic.notna()&rg.ic.reindex(kg.index).notna()];cg=cg.reindex(dates)
   for date in dates:
    t=int(np.searchsorted(data['dates'],np.datetime64(date)));dependency_flags.append(dict(family=kind,year=year,world=wi,h=h,date=str(date),signal_last_raw_date=str(date),target_first_payoff_date=str(data['dates'][t+1]),target_last_payoff_date=str(data['dates'][t+h]),target_hedge_estimation_last_date=str(date),direct_payoff_overlap_at_t=False,shared_current_day_in_hedge_estimation=True))
   exact.append(dict(family=kind,year=year,world=wi,h=h,K_sum=float(kg.loc[dates].ic.sum()),roll_sum=float(rg.loc[dates].ic.sum()),days=len(dates),core_reference_sum=float(cg.ic.sum()),core_reference_days=int(cg.ic.notna().sum())))
  e=data['e']*data['signs'][wi];signals=dict(K=K,roll_new=roll,core_roll=data['core'][wi],neg_e=-e);lagged={k:lag(v) for k,v in signals.items()};tgt=data['target'][wi];cases=[('original_h1',1,tgt[:,:,0],False),('original_h5',5,tgt[:,:,1],False),('lag1_original_h1',1,tgt[:,:,0],True),('lag1_original_h5',5,tgt[:,:,1],True),('causal_X_h1',1,future_sum(e,1,1),False),('causal_X_h5',5,future_sum(e,1,5),False),('skip1_X_h5',5,future_sum(e,2,5),False),('tplus2_X_h1',1,future_sum(e,2,1),False),('skip1_frozen_h5',5,data['extra']['frozen_skip1_h5'][wi],False),('tplus2_frozen_h1',1,data['extra']['frozen_tplus2_h1'][wi],False)]
  for t,date in enumerate(data['dates']):
   if t%20==0:checkpoint()
   rank_cache.clear()
   for case,h,y,shifted in cases:
    chosen=lagged if shifted else signals;common=(data['w']>0)&np.isfinite(y[:,t])
    for name in signals:common &= np.isfinite(signals[name][:,t])&np.isfinite(chosen[name][:,t])
    w=data['w'][common];yy=y[common,t];baseY=tgt[common,t,0 if h==1 else 1]
    # For each timing/target variant, compare against current-day original target
    # on exactly the same observations. The finite frozen target is a guard.
    common2=np.isfinite(baseY)
    idx=np.flatnonzero(common)[common2];w=data['w'][idx];yy=y[idx,t];baseY=tgt[idx,t,0 if h==1 else 1]
    for name in signals:
     x=chosen[name][idx,t];xx=x if name=='neg_e' else -x;base=signals[name][idx,t];base=base if name=='neg_e' else -base
     value=stats.spearman(xx,yy,w);reference=stats.spearman(base,baseY,w);rows.append(dict(family=kind,year=year,world=wi,date=str(date),case=case,h=h,predictor=name,ic=value,reference_current_original_ic=reference,n=len(idx)))
    if case in ['causal_X_h1','causal_X_h5']:
     original=tgt[idx,t,0 if h==1 else 1];comparisons.append(dict(world=wi,h=h,n=len(idx),max_abs_difference=float(np.max(abs(original-yy))) if len(idx) else None,absolute_difference_sum=float(np.sum(abs(original-yy))),original_target_abs_sum=float(np.sum(abs(original)))))
   # Fixed-year daily within-cross-section permutation; no year/seed selection.
   if year==2004:
    for hi,h in enumerate([1,5]):
     yy=tgt[:,t,hi];common=(data['w']>0)&np.isfinite(yy)
     for arr in signals.values():common &= np.isfinite(arr[:,t])
     idx=np.flatnonzero(common);w=data['w'][idx];seed=int.from_bytes(hashlib.sha256(f'A1_sanity_v1|20261105|{kind}|{year}|{wi}|{date}|{h}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed);x=-K[idx,t];permuted=x[rng.permutation(len(x))];placebo.append(dict(family=kind,year=year,world=wi,h=h,date=str(date),ic=stats.spearman(permuted,yy[idx],w),unpermuted_ic=stats.spearman(x,yy[idx],w),n=len(idx)))
 finish(root,{'daily_IC.csv.gz':pd.DataFrame(rows),'sealed_exact_IC_sums.csv.gz':pd.DataFrame(exact),'placebo.csv.gz':pd.DataFrame(placebo,columns=['family','year','world','h','date','ic','unpermuted_ic','n']),'dependency_flags.csv.gz':pd.DataFrame(dependency_flags),'target_comparison.json':comparisons,'provenance.json':dict(source_digests=data['source_digests'],A1_groups=proofs,original_target_reproduction_max_abs_error=data['target_reproduction_max_abs_error'],new_fits=0,first_RP_session_increment='unavailable in this audit; no prior-fold hedge refit',Part_C_started=False)},phase='A1_read_only_sanity_audit')
 print('Sanity audit sealed',kind,year,flush=True)

def steady_gain(rho,q,tol=1e-13):
 c=1/(1-rho*rho);a=c;b=-c
 for step in range(100000):
  old=np.array([a,b,c]);a+=q;b*=rho;c=rho*rho*c+1;F=a+2*b+c;km=(a+b)/F;kr=(b+c)/F;a-=km*km*F;b-=km*kr*F;c-=kr*kr*F
  if np.max(abs(np.array([a,b,c])-old))<tol:return dict(K_M=km,K_R=kr,iterations=step+1)
 raise AssertionError('Gain did not reach steady state')

def report():
 root=OUT/'report'
 if done(root):
  publish(P/'A1_sanity_audit.md',(root/'A1_sanity_audit.md').read_bytes());return
 daily=[];exact=[];placebo=[];comparison=[];input_proofs=[];dependencies=[]
 for kind in ['RP','RF']:
  for year in range(2004,2026):
   r=OUT/f'groups/{kind}_{year}';assert done(r);daily.append(rp.frame(r/'daily_IC.csv.gz'));exact.append(rp.frame(r/'sealed_exact_IC_sums.csv.gz'));placebo.append(rp.frame(r/'placebo.csv.gz'));comparison.extend(dict(family=kind,year=year,**v) for v in J(r/'target_comparison.json'));input_proofs.append(J(r/'provenance.json'));dependencies.append(rp.frame(r/'dependency_flags.csv.gz'))
 daily=pd.concat(daily,ignore_index=True);exact=pd.concat(exact,ignore_index=True);placebo=pd.concat(placebo,ignore_index=True);stats=residuals.imported_stats();raw=[];timing=[];permutations=[]
 for (kind,wi,h),g in exact.groupby(['family','world','h']):raw.append(dict(family=kind,world=int(wi),h=int(h),K=float(g.K_sum.sum()/g.days.sum()),roll_new=float(g.roll_sum.sum()/g.days.sum()),core_reference=float(g.core_reference_sum.sum()/g.core_reference_days.sum()),days=int(g.days.sum()),core_reference_days=int(g.core_reference_days.sum())))
 for (kind,wi,case,h,predictor),g in daily.groupby(['family','world','case','h','predictor']):
  g=g.sort_values('date');value=stats.nw_mean(g.ic,int(h)+1);reference=stats.nw_mean(g.reference_current_original_ic,int(h)+1);timing.append(dict(family=kind,world=int(wi),case=case,h=int(h),predictor=predictor,IC=value['mean'],nominal_NW_SE=value['se'],reference_current_original_IC=reference['mean'],difference=value['mean']-reference['mean'],valid_days=int(g.ic.notna().sum()),years=int(g[g.ic.notna()].year.nunique()),mean_series=float(g.n.mean())))
 for (kind,wi,h),g in placebo.groupby(['family','world','h']):
  v=stats.nw_mean(g.sort_values('date').ic,int(h));old=stats.nw_mean(g.sort_values('date').unpermuted_ic,int(h));permutations.append(dict(family=kind,world=int(wi),year=2004,h=int(h),permuted_IC=v['mean'],nominal_NW_SE=v['se'],nominal_95_low=v['mean']-1.96*v['se'],nominal_95_high=v['mean']+1.96*v['se'],unpermuted_IC_same_mask=old['mean'],days=int(g.ic.notna().sum())))
 gains=[];parameters=rp.frame(P/'outputs/amended_A1/report/pooled_parameters.csv.gz')
 for kind in ['RP','RF']:
  g=parameters[(parameters.family==kind)&(parameters.world=='real')].copy();median=float(g.half_life.median());g['distance']=abs(g.half_life-median);chosen=g.sort_values(['distance','year']).iloc[0];year=int(chosen.year);fit=J(P/f'runtime/B_fits/{kind}_{year}_0/pooled/pooled.json');gain=steady_gain(fit['rho'],fit['q']);scales=np.sqrt(np.asarray(fit['scale'])/(1-fit['rho']**2));gains.append(dict(family=kind,selected_actual_group_year=year,selection='nearest to median real pooled half-life; earlier year breaks ties',median_real_half_life=median,group_half_life=fit['half_life'],rho=fit['rho'],q=fit['q'],**gain,median_sigma_equilibrium=float(np.median(scales)),median_derivative_sK_wrt_e=float(np.median(gain['K_R']/scales))))
 timing_frame=pd.DataFrame(timing);wide=timing_frame.pivot(index=['family','world','case','h'],columns='predictor',values='IC').reset_index();coverage=timing_frame[timing_frame.predictor=='K'][['family','world','case','h','valid_days','years','reference_current_original_IC','difference']];wide=wide.merge(coverage,on=['family','world','case','h']);rawf=pd.DataFrame(raw);placebof=pd.DataFrame(permutations)
 def tab(frame):
  return rp.table(list(frame.columns),[[rp.num(v) if isinstance(v,(float,np.floating)) else str(v) for v in row] for row in frame.itertuples(index=False,name=None)])
 c,d,a=bootstrap.core_modules();source=c.source();jobs_path=P.parent/'coint_residual_qlearning_v1/vendor/residual_ou_real_v1/scripts/jobs.py';models_path=jobs_path.with_name('models.py')
 import ast
 def line(path,name):return next(n.lineno for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 refs=dict(target_RP=f'{jobs_path}:{line(jobs_path,"rp_rows")}',target_RF=f'{jobs_path}:{line(jobs_path,"factor_rows")}',A1_IC=f'{P}/audit/A1_A2/analyze.py',level_construction=f'{P}/scripts/residuals.py',Kalman_filter=f'{P}/scripts/kalman.c',rolling_OU=f'{models_path}:{line(models_path,"ou_fit")}',null_stream=f'{models_path}:{line(models_path,"sign_series")}')
 difference=pd.DataFrame(comparison);target_summary=[]
 for (kind,wi,h),g in difference.groupby(['family','world','h']):target_summary.append(dict(family=kind,world=int(wi),h=int(h),compared_series_days=int(g.n.sum()),max_abs_original_minus_X=float(g.max_abs_difference.max()),mean_abs_original_minus_X=float(g.absolute_difference_sum.sum()/g.n.sum()),relative_absolute_difference=float(g.absolute_difference_sum.sum()/g.original_target_abs_sum.sum())))
 original_label=J(P/'preregistration/amendments/A1_A2/original_B1_seal.json')['status']
 text=['# A1 leakage and sanity audit','Diagnostic only; no new fits, scientific re-selection, new go/no-go decision or Part C execution. A1 remains a post-coverage-failure amended analysis. All original A1/B1/B2 result files and receipts are unchanged. World0=real; world1=N2 replica0; world2=N2 replica1. IC means weighted cross-sectional Spearman of **minus** s-score versus the target, then equal-day averaging.',
 '## 1. Raw IC from sealed A1 outputs',tab(rawf),'K and roll_new use the exact A1 pair-matched series and dates. The stored core reference is on those same dates but uses its own Kalman/core pair-matched cross-section and the original core half-life<30 mask from residuals.core_targets. This panel must not be mistaken for identical series sets. The controlled audit panel below instead uses the original sealed core score without that extra half-life mask, and matches K, rolling_new, core and −e on identical observations within each case.',
 'The N2 sK h5 ICs are near zero while real sK IC is large: '+ '; '.join(f"{kind}: real={rp.num(next(r['K'] for r in raw if r['family']==kind and r['world']==0 and r['h']==5))}, N2r0={rp.num(next(r['K'] for r in raw if r['family']==kind and r['world']==1 and r['h']==5))}, N2r1={rp.num(next(r['K'] for r in raw if r['family']==kind and r['world']==2 and r['h']==5))}" for kind in ['RP','RF'])+'. This is an observation, not by itself proof of predictive value or leakage.',
 '## 2. Exact definitions and last data indices',rp.table(['Definition','Authenticated code path'],refs.items()),
 '**Original A1 h5 target, RP:** Y_i,t^(5)=Σ[u=t+1..t+5] ξ_i,u·(r_A,u−α_i,t−β_i,t r_B,u). The core alpha/beta are fitted on t−59..t, frozen for the horizon. Real ξ=1; N2 ξ is the keyed per-day sign stream. Last payoff data index=t+5; last coefficient-estimation data index=t. The target does not contain the day-t return as a payoff, but its alpha/beta depend on day-t data.',
 '**Original A1 h5 target, RF:** Y_i,t^(5)=Σ[u=t+1..t+5] ξ_i,u·(r_i,u−α_i,t−β_i,tᵀW_tᵀr_u). Core PCA weights use t−251..t; core regression/loadings use its last60 sessions through t. W_t,alpha_t,beta_t stay fixed over the target horizon. Last payoff index=t+5, last hedge/PCA estimation index=t. No new alpha/beta/PCA is fitted by this audit: all coefficients are read from sealed caches.',
 '**New causal residual levels:** e_RP,t=r_A,t−α_t−1−β_t−1 r_B,t. RF e_t uses W_t−1,alpha_t−1,beta_t−1. X_t=Σ known e_u, with missing increments masked as unavailable. RP hedge estimation ends t−1 with60 sessions; RF PCA ends t−1 with252 sessions and loadings with60. Last observation used for e_t is r_t. In this audit RP e_t is reconstructed directly from already sealed prior-day alpha/beta within the same fold; its first annual session is unavailable and omitted, rather than fitting an uncached prior-year hedge. RF e_t is read from immutable causal increment files.',
 '**sK_t:** Rhat_t|t/[sigmahat_R,i/sqrt(1−rho²)]. rho,q and each scale were estimated only on years y−3..y−1 (last index strictly before test year y). The forward Kalman recursion consumes X through t only; there is no smoothing/backward recursion. N2 uses signed new e, N2 X and its own world-specific pooled fit. The feature −sK_t is used in IC. A1 has no half-life cutoff; A2 does not enter this audit.',
 '**s_roll_new,t:** unchanged core ou_fit on [e_t−59,…,e_t], summing within the60-session window to x0=0,…,x60; fit x_j=a+b x_j−1+epsilon_j; s=(x60−a/(1−b))/sqrt(sigma_epsilon²/(1−b²)). It retains the original per-series finite/stationary/0<half-life<30 validity. Last data index=t; its OU window includes the current increment. The sealed map is reused without any OU fit here. Core s_roll uses the60 **in-window residuals**, re-hedged with current coefficients/PCA through t, rather than the sequence of causal prior-hedge increments.',
 '**Overlap flags:** no direct day-t payoff is shared by signal and original future-return target (payoffs start t+1). However every valid original target uses hedge/PCA estimates trained through t, sharing historical raw observations—including day t—with the signal. This is an explicit shared-estimation-data/mechanical-dependence flag, not a demonstrated use of future prices in sK. The target and new X use different residual constructions. A fixed-hedge target must not be silently described as X_t+5−X_t.',
 'Every original A1 date with a finite matched IC pair is explicitly flagged in outputs/A1_sanity_v1/report/dependency_flags.csv.gz, including dates omitted by stricter audit masks. It records the actual last signal-data date, first/last payoff dates and last hedge-estimation date, distinguishing shared current-day estimation input from direct payoff overlap.',
 '## 3. Timing controls, same IC code',tab(wide),
 'All controlled rows within each case use a common positive-weight, finite-target intersection of current K/roll_new/core/−e and the shifted signals when applicable. A variant\'s reference_current_original_IC uses the current signal against the original frozen-hedge target on exactly that variant\'s observations; difference compares K to that reference. original_h1/h5 use the sealed original target. lag1_original uses s_t−1 against that **same original** target; annual first session is unavailable. causal_X_h5=Σ e_t+1..t+5. skip1_X_h5=X_t+6−X_t+1=Σ e_t+2..t+6. tplus2_X_h1=e_t+2. To separate timing from target-construction changes, skip1_frozen_h5 and tplus2_frozen_h1 also keep the original time-t hedge coefficients while shifting only the payoff window. Strict finite endpoints/increments and no year crossing apply. RP first annual e is omitted, not imputed. For the new-X skip-one target, e_t+2 still uses hedge parameters dated t+1, so excluding the t+1 payoff does not exclude that raw day from future hedge-estimation inputs. The frozen-hedge skip controls avoid this extra change in hedge estimation. A sharp decline is a diagnostic observation; short-lived genuine signals can also decay, and comparing different target constructions is not a pure lag test.',
 'The literal authenticated Spearman implementation is used. Only memoization of its unchanged weighted_ranks outputs is added within each day; cached ranks are copied because the original code centers them in place. Every returned rank value is from the original function. No rank/weight/inference formula is replaced.',
 '### Original frozen-hedge target versus new-X target',tab(pd.DataFrame(target_summary)),'These errors compare the two target constructions on audit-matched observations. They are not numerical reproduction errors: the sealed original targets were independently reconstructed from their original sealed hedge parameters and checked within rtol1e−11/atol1e−9 for every audited family/year/world.',
 '## 4. Fixed-year permutation placebo',tab(placebof),'The year2004 is fixed in advance for both families, with deterministic seed namespace A1_sanity_v1|20261105|family|year|world|date|h. Exactly one permutation per day/horizon is drawn across the matched finite series, preserving the target, weights, missingness and signal distribution. No year or seed was selected by outcome. The mean should be near zero in expectation, not exactly zero for one realization; nominal NW intervals are descriptive and do not authorize any decision change.',
 '## 5. Null construction check','Source/provenance checks confirm separate sealed real/N2 pooled parameter receipts and matching world-indexed A1 signal arrays. The frozen residuals.signed implementation constructs N2 levels from per-date signed new causal increments with the exact source seed and identity RP|year:pair or RF|stock. No real pooled fit is substituted for an N2 fit. This is a code/provenance check; the audit does not refilter all signals or estimate any parameter.',
 '**The literal statement “the N2 target uses the N2 new levels” is false for the sealed A1 target.** It is built from per-date signed **core frozen-hedge forward residuals**. The sign stream is the same keyed stream, and targets have been numerically reproduced from those coefficients/signs, but they are not differences of the N2 causal X levels. The new-X controls in section3 explicitly use N2 causal increments when world1/2, so these diagnostic targets are internally construction-matched. This difference is recorded without modifying the original A1 target or result.',
 'A methodological implication (inference from this code, not a new test): N2 signs are applied after constructing residuals with real-price hedge coefficients. Large real-minus-N2 IC therefore does not by itself separate genuine forecasting from dependence induced by rolling centering/hedge estimation. The controls above are diagnostic, and do not change the accepted null recipe.',
 '## 6. Latest-increment weight and one-day reversal',tab(pd.DataFrame(gains)),
 'For normalized covariance P and H=[1,1], F=[1,0;0,rho], Q=diag(q,1), steady prediction Pminus=F P Fᵀ+Q, K=Pminus Hᵀ/(H Pminus Hᵀ), and P=Pminus−K(H Pminus). The reported K_R is ∂Rhat_t|t/∂e_t holding past observations fixed, since e_t changes X_t one for one. Thus ∂sK/∂e_t=K_R/sigma_equilibrium,i; the table reports its median over eligible formation scales. This is the effective weight on the latest normalized increment, not a fraction of forecast performance. An actual group nearest the median real half-life is used (earlier year breaks ties); no synthetic median parameter model is fitted.',
 'The −e_t-only IC is the neg_e column of section3, on exactly the same target and observations as K. It is the one-day reversal comparator. It must be assessed jointly with target construction, timing controls and placebo; a large level-score IC alone does not demonstrate value beyond latest-increment reversal.',
 '## Integrity and stop','No A1/B1/B2 result or receipt was changed. Models/OU/OLS/PCA fitting functions are blocked in the audit process. Only sealed prices, signals, alpha/beta/weights, causal RF increments, pooled metadata and original IC records were read. No downloads, hydration, new trials, policy selection or go/no-go update occurred. The original B1 label remains '+original_label+'. Part C has not started. B2 was not signalled or modified. This audit stops after publication.']
 body='\n\n'.join(text)+'\n'
 finish(root,{'raw_sealed_IC.csv.gz':rawf,'dependency_flags.csv.gz':pd.concat(dependencies,ignore_index=True),'controlled_summary.csv.gz':wide,'controlled_long_summary.csv.gz':timing_frame,'placebo_summary.csv.gz':placebof,'latest_increment_gain.csv.gz':pd.DataFrame(gains),'target_construction_comparison.csv.gz':pd.DataFrame(target_summary),'A1_sanity_audit.md':body.encode(),'summary.json':dict(status='complete',new_fits=0,Part_C_started=False,unmodified_A1=True,raw_N2_near_zero_observed=True,original_target_is_new_X_difference=False)},phase='A1_sanity_audit_report')
 publish(P/'A1_sanity_audit.md',body)

def main():
 authenticate();install_guard();c,d,a=bootstrap.core_modules()
 import pci,llt
 pci.fit=no_fit;llt.fit=no_fit
 for name in ['ou_fit','pair_rolling','factor_window','regress','eigenportfolio']:setattr(d.M,name,no_fit)
 os.environ['GATE_POWER_STOP']=str(P/f'runtime/STOP_A1_sanity_{os.getpid()}');m=memory_snapshot();os.environ['PARTIAL_RAM_GUARD']='1';os.environ['PARTIAL_INITIAL_SWAPOUTS']=str(m['swapouts']);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 with (P/'runtime/A1_sanity.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);checkpoint()
  for kind in ['RP','RF']:
   for year in range(2004,2026):job(kind,year)
  report();authenticate();journal('A1_sanity_audit_complete',receipt_sha256=H(OUT/'report/complete.json'),new_fits=0,Part_C_started=False)
  print('World0=real; world1=N2r0; world2=N2r1. K/roll/core IC uses minus-score; neg_e uses minus-increment.',flush=True)
  for name in ['raw_sealed_IC.csv.gz','controlled_summary.csv.gz','placebo_summary.csv.gz','latest_increment_gain.csv.gz']:
   print(name,flush=True);print(rp.frame(OUT/'report'/name).to_string(index=False),flush=True)
  print('STOPPED after read-only A1 sanity audit. Part C not started. Report:',P/'A1_sanity_audit.md',flush=True)
if __name__=='__main__':main()
