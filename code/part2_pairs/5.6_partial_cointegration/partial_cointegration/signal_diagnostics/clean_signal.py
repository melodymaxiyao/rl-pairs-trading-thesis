"""Append-only CLEAN pooled/IC diagnostic; no PCI or trading execution."""
import argparse, fcntl, importlib.util, os, pathlib, resource, subprocess, sys, time
P=pathlib.Path('/Users/maxiyao/FYP_runs/partial_coint_kalman_v1')
HERE=P/'audit/N3_intercept_null/CLEAN_addendum';OUT=P/'outputs/N3_intercept_null/CLEAN'
sys.path[:0]=[str(P/'audit/disk_budget_10GiB_v1'),str(P/'scripts')]
import entry as resource_policy
import numpy as np
import pandas as pd
import io_utils as u
from groups import done,finish,array_bytes
import bootstrap,pci,residuals
spec=importlib.util.spec_from_file_location('_registered_N3_clean',P/'audit/N3_intercept_null/run.py');n3=importlib.util.module_from_spec(spec);spec.loader.exec_module(n3)

def authenticate(code=True):
    n3.authenticate();resource_policy.authenticate()
    for name in ['registration.json']+(['code_registration.json'] if code else []):
        for rel,digest in u.J(HERE/name)['files'].items():assert u.H(P/rel)==digest,rel
    assert done(P/'outputs/A1_sanity_v2/report')

def clean_inputs(kind,year):
    """Replay no hedge changes; retain original masks, cumulative origin and dates."""
    c,d,a=bootstrap.core_modules();p=d.panel();dates=p['dates'];Y=dates.astype('datetime64[Y]').astype(int)+1970
    window=np.flatnonzero((Y>=year-3)&(Y<=year));form=int(np.sum(Y[window]<year));checks={}
    module=n3.item3_module()
    if kind=='RF':
        names=list(p['meta']['symbols']);values=[];offset=np.zeros((3,len(names)))
        for yy in range(int(Y.min()),year+1):
            root=P/f'outputs/A1_sanity_v2/RF_alpha_free/{yy}';assert done(root)
            with np.load(root/'increments.npz',allow_pickle=False) as z:h=z['eprime'];ix=z['indices']
            signed=np.stack([h]+[h*n3.sign_rows(kind,year,names,dates[ix],rep) for rep in range(2)])
            if yy<year-3:offset+=np.nansum(signed,axis=-1)
            else:values.append(signed)
        increments=np.concatenate(values,axis=-1);levels=offset[:,:,None]+n3.levels(increments)
    else:
        rows=p['meta']['pairs'][str(year)];names=[r['pair'] for r in rows];data={'ids':names};refs=module.rp_reference_subset(year,data)
        increments=np.full((3,len(names),len(window)),np.nan);levels=increments.copy();end=int(window[-1])
        for i,row in enumerate(rows):
            if i%75==0:u.checkpoint()
            aa=p['returns'][:end+1,d.index(row['leg_A'],p)];bb=p['returns'][:end+1,d.index(row['leg_B'],p)]
            real,alpha,beta=residuals.pair_out_of_sample(aa,bb)
            for refdates,refalpha,refbeta in refs.get(i,[]):
                used=(refdates>=59)&(refdates<end);ix=refdates[used]-59
                module.check_parameter(checks,'alpha',alpha[ix],refalpha[used]);module.check_parameter(checks,'beta',beta[ix],refbeta[used])
                alpha[ix]=refalpha[used];beta[ix]=refbeta[used]
            h=aa[60:]-beta*bb[60:];h[~np.isfinite(real)]=np.nan
            for wi in range(3):
                signs=np.ones(len(h)) if wi==0 else n3.sign_rows(kind,year,[names[i]],dates[60:end+1],wi-1)[0]
                signed=h*signs;increments[wi,i]=signed[window-60];levels[wi,i]=n3.levels(signed)[window-60]
            if i%600==0:print('CLEAN causal inputs',kind,year,i,'/',len(names),flush=True)
    eligible=np.flatnonzero(np.isfinite(levels[0,:,:form]).sum(axis=1)>=500)
    for wi in range(3):
        np.testing.assert_array_equal(np.isfinite(increments[wi]),np.isfinite(increments[0]))
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(levels[wi,:,:form]).sum(axis=1)>=500),eligible)
        original=u.J(P/f'runtime/part_B1/{kind}_{year}_{wi}/formation_masks.json')
        np.testing.assert_array_equal(eligible,[int(r['index']) for r in original])
    return dict(increments=increments,levels=levels,eligible=eligible,form=form,ids=np.asarray(names,str),indices=window,parameter_checks=checks)

def score_world(data,wi,fit,rolling_fit):
    x=data['levels'][wi];inc=data['increments'][wi];form=data['form'];ids=data['eligible'];score=np.full_like(x,np.nan)
    if fit['converged']:
        _,states=pci.filter_values(x[ids],fit['rho'],fit['q'],states=True)
        score[ids]=states[:,:,1]/np.sqrt(np.asarray(fit['scale'])[:,None]/(1-fit['rho']**2))
        score[~np.isfinite(x)]=np.nan
    roll,_=residuals.rolling_chunked(inc,rolling_fit)
    return score[:,form:],roll[:,form:]

def one_year(kind,year):
    yearroot=OUT/f'IC/{kind}_{year}'
    if done(yearroot):return
    c,d,a=bootstrap.core_modules();module=n3.item3_module();inputs=None
    for wi in range(3):
        root=OUT/f'signals/{kind}_{year}_{wi}'
        if done(root):continue
        if inputs is None:inputs=clean_inputs(kind,year)
        fitroot=OUT/f'pooled/{kind}_{year}_{wi}'
        if not done(fitroot):
            prev=OUT/f'pooled/{kind}_{year-1}_{wi}';warm=u.J(prev/'fit.json') if done(prev) else None
            fit=pci.fit(inputs['levels'][wi,inputs['eligible'],:inputs['form']],pooled=True,warm_start=warm,checkpoint=u.checkpoint)
            finish(fitroot,{'fit.json':fit,'eligible.json':inputs['eligible'].tolist(),'proof.json':dict(clean_addendum_sha256=u.H(HERE/'registration.json'),same_original_formation_eligibility=True,only_pooled_MLE=True,hedge_parameter_checks=inputs['parameter_checks'])},phase='CLEAN_primary_pooled_diagnostic',kind=kind,year=year,wi=wi)
        fit=u.J(fitroot/'fit.json');print('CLEAN pooled',kind,year,wi,'converged',fit['converged'],'rho',fit.get('rho'),'q',fit.get('q'),flush=True)
        score,roll=score_world(inputs,wi,fit,d.M.ou_fit)
        finish(root,{'signals.npz':array_bytes(K_clean=score,roll_clean=roll),'proof.json':dict(pooled_receipt_sha256=u.H(fitroot/'complete.json'),target_independent_signal=True,Kalman_no_half_life_cutoff=True,rolling_original_validity_unchanged=True,diagnostic_only=True)},phase='CLEAN_filtered_signals')
        del score,roll
    del inputs
    # Reuse the independently reproduced, accepted frozen tradable target.
    target=module.year_data(kind,year);stats,cache=module.v1.cached_stats();records=[]
    for wi in range(3):
        root=OUT/f'signals/{kind}_{year}_{wi}';assert done(root)
        with np.load(root/'signals.npz',allow_pickle=False) as z:K=z['K_clean'];roll=z['roll_clean']
        for t,date in enumerate(target['dates']):
            if t%30==0:u.checkpoint()
            for hi,h in enumerate([1,5]):
                y=target['Ytr'][wi,:,t,hi];w=target['w'];valid=(w>0)&np.isfinite(K[:,t])&np.isfinite(roll[:,t])&np.isfinite(y)
                ww=w[valid];yy=y[valid];ik=stats.spearman(-K[valid,t],yy,ww);ir=stats.spearman(-roll[valid,t],yy,ww)
                if not (np.isfinite(ik) and np.isfinite(ir)):ik=ir=np.nan
                for label,value in [('K_clean',ik),('roll_clean',ir),('K_minus_roll_clean',ik-ir)]:
                    records.append(dict(family=kind,year=year,world=wi,date=str(date),h=h,signal=label,ic=value,n=int(valid.sum()),eligible_series=int(np.sum(w>0))))
    finish(yearroot,{'daily_IC.csv.gz':pd.DataFrame(records),'proof.json':dict(target='accepted A1_sanity_v2 frozen time-t hedge, no intercept; unchanged N2 sign streams',weights='original RP IPW / RF weights',common_set='K_clean and roll_clean finite within world, identical y and weights',clean_addendum_sha256=u.H(HERE/'registration.json'))},phase='CLEAN_tradable_IC')
    print('CLEAN IC sealed',kind,year,flush=True)

def params(fit,family,year,world,replica):
    return dict(family=family,year=year,world=world,replica=replica,converged=fit['converged'],rho=fit.get('rho'),half_life=fit.get('half_life'),q=fit.get('q'),R2_MR=fit.get('R2_MR'),rho_boundary=bool(fit['converged'] and fit['rho']==.999),any_estimator_boundary=fit.get('boundary'))

def parameter_rows(combined=False):
    rows=[]
    for family in ['RP','RF']:
        for year in range(2004,2026):
            for wi in range(3):
                root=OUT/f'pooled/{family}_{year}_{wi}';assert done(root)
                rows.append(params(u.J(root/'fit.json'),family,year,'CLEAN_real' if wi==0 else 'CLEAN_null',-1 if wi==0 else wi-1))
            original=P/f'runtime/B_fits/{family}_{year}_0/pooled';assert done(original)
            rows.append(params(u.J(original/'pooled.json'),family,year,'REAL',-1))
            if combined:
                for rep in range(2):
                    root=P/f'outputs/N3_intercept_null/market_pooled/{family}_{year}_{rep}';assert done(root)
                    rows.append(params(u.J(root/'fit.json'),family,year,'N3',rep))
                    old=P/f'runtime/B_fits/{family}_{year}_{rep+1}/pooled';assert done(old)
                    rows.append(params(u.J(old/'pooled.json'),family,year,'N2_reference',rep))
    return pd.DataFrame(rows)

def medians(frame):
    rows=[]
    for (family,world),f in frame.groupby(['family','world']):
        ok=f[f.converged.astype(bool)]
        row=dict(family=family,world=world,groups=len(f),converged_groups=len(ok),years=f.year.nunique(),rho_boundary_share_all=float(f.rho_boundary.mean()),rho_boundary_share_converged=float(ok.rho_boundary.mean()) if len(ok) else None,any_estimator_boundary_share_converged=float(ok.any_estimator_boundary.astype(bool).mean()) if len(ok) else None)
        row.update({f'median_{k}':float(ok[k].median()) if len(ok) else None for k in ['rho','half_life','q','R2_MR']});rows.append(row)
    return pd.DataFrame(rows)

def ic_report():
    root=OUT/'report'
    if done(root):return
    module=n3.item3_module();stats=residuals.imported_stats();frames=[]
    for family in ['RP','RF']:
        for year in range(2004,2026):
            r=OUT/f'IC/{family}_{year}';assert done(r);frames.append(pd.read_csv(r/'daily_IC.csv.gz',float_precision='round_trip'))
    daily=pd.concat(frames,ignore_index=True);rows=[]
    for (family,h,label),f in daily.groupby(['family','h','signal']):
        rows.append(dict(family=family,h=h,signal=label,**module.paired_summary(f,stats)))
    summary=pd.DataFrame(rows);coverage=daily.groupby(['family','year','world','h','signal']).agg(valid_IC_days=('ic','count'),mean_matched_series=('n','mean')).reset_index();table=parameter_rows();med=medians(table)
    text='# CLEAN addendum to N3 — diagnostic only\n\nPart C is not run; cumulative real trading trials remain 580. No per-series MLE or PCI is run for CLEAN.\n\n## Clean refit signal IC on the tradable target\n\n'+n3.markdown(summary)+'\n\n## Pooled parameter medians\n\n'+n3.markdown(med)
    text+='\n\nICs use minus scores, original weights, exactly the accepted frozen-hedge alpha-free target, and within-world matched K/rolling series. The excess is real minus the equal mean of null r0/r1, on common valid dates. The K-minus-rolling row is the paired difference of excess ICs; its CI comes from the same year-block bootstrap, not subtraction of CI endpoints. Seed 20261026, 5000 draws; descriptive only, no go/no-go or Holm decision.\n\n'
    text+='Kalman is defined for every converged formation fit, including rho=0.999, with >=500 valid formation sessions and a finite current input. The rolling score keeps the original per-series 0<half-life<30 validity and original 60-increment OU code (61 cumulative levels); it is computed on clean increments, not the old intercept-subtracted path. Coverage is reported explicitly. Pooled rho_boundary means the exact rho=0.999 face; other estimator boundaries are separate.\n\n'
    text+='The complete REAL/N3/CLEAN table is published separately after the original N3 diagnostic seals its final report; this interim report does not treat incomplete N3 groups as a complete comparison. No original N3, A1, Item3 or B2 file is changed.\n'
    finish(root,{'IC_summary.csv':summary,'coverage.csv':coverage,'parameters_by_group.csv':table,'parameter_medians.csv':med,'CLEAN_report.md':text.encode(),'status.json':dict(status='CLEAN_complete',Part_C_run=False,cumulative_trials=580)},phase='CLEAN_descriptive_report')
    u.publish(P/'N3_CLEAN_IC_report.md',text);print(text,flush=True)

def combined_report():
    root=OUT/'combined_report'
    if done(root):return
    assert done(P/'outputs/N3_intercept_null/report');assert done(OUT/'report')
    table=parameter_rows(True);med=medians(table);summary=pd.read_csv(OUT/'report/IC_summary.csv',float_precision='round_trip')
    wide=table.pivot(index=['family','year','replica'],columns='world',values=['rho','half_life','q','R2_MR','rho_boundary']);wide.columns=[f'{w}_{metric}' for metric,w in wide.columns];wide=wide.reset_index()
    # REAL/CLEAN_real are single fits (replica=-1), not duplicated independent
    # estimates; propagate these reference columns into replica comparison rows.
    for family in ['RP','RF']:
        for year in range(2004,2026):
            mask=(wide.family==family)&(wide.year==year)
            for world in ['REAL','CLEAN_real']:
                for metric in ['rho','half_life','q','R2_MR','rho_boundary']:
                    col=f'{world}_{metric}';values=wide.loc[mask,col].dropna()
                    if len(values):wide.loc[mask,col]=values.iloc[0]
    wide=wide[wide.replica>=0].reset_index(drop=True)
    text='# N3 with CLEAN arms — final diagnostic comparison\n\nPart C remains closed; cumulative real trading trial count stays 580. This appendix extends the same N3 registration without modifying its original report or any decision.\n\n## Pooled parameter medians and boundary shares\n\n'+n3.markdown(med)+'\n\n## CLEAN excess IC on the tradable target\n\n'+n3.markdown(summary)
    text+='\n\nThe per-family/year/replica table is supplied alongside this report. REAL and CLEAN_real have 22 groups per family; N3, CLEAN_null and N2_reference have 44. Boundary shares are calculated across pooled groups, separately for the rho=0.999 face and any estimator boundary. The original N3 report retains its per-series PCI and simulation diagnostics; CLEAN adds no per-series MLE or PCI. All results are descriptive, with no go/no-go change. See N3_CLEAN_IC_report.md for signal, weighting, target and bootstrap definitions and coverage.\n'
    finish(root,{'parameters_by_group.csv':table,'parameters_side_by_side.csv':wide,'parameter_medians.csv':med,'IC_summary.csv':summary,'N3_CLEAN_report.md':text.encode(),'status.json':dict(status='complete',Part_C_run=False,cumulative_real_trading_trials=580),'source_receipts.json':dict(N3_report=u.H(P/'outputs/N3_intercept_null/report/complete.json'),CLEAN_report=u.H(OUT/'report/complete.json'),CLEAN_registration=u.H(HERE/'registration.json'))},phase='N3_CLEAN_combined_diagnostic_report')
    u.publish(P/'N3_CLEAN_report.md',text);print(text,flush=True)

def admission():
    # Same conservative policy: half of memory above 2GiB reserve, 1.25x peaks.
    # Use a conservative 2GiB CLEAN peak until this process measures a larger one.
    while True:
        u.checkpoint();m=u.memory_snapshot();other=0
        for line in subprocess.check_output(['ps','-axo','pid=,rss=,command='],text=True).splitlines():
            bits=line.strip().split(None,2)
            if len(bits)==3 and int(bits[0])!=os.getpid() and str(P/'audit/N3_intercept_null/run.py') in bits[2] and 'python' in bits[2]:other+=int(bits[1])*1024
        peak=max(2*u.GIB,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        need=1.25*(peak+other);allowed=.5*(m['available']-2*u.GIB)
        if allowed>=need:
            print('CLEAN RAM admission',dict(available=m['available'],other_N3_RSS=other,CLEAN_peak_budget=peak,allowed=allowed,required=need),flush=True);return
        print('CLEAN waiting for conservative parallel RAM budget',flush=True);time.sleep(30)

def main(wait_n3_pid=None):
    authenticate();u.install_guard();os.environ['PARTIAL_RAM_GUARD']='1';os.environ['PARTIAL_INITIAL_SWAPOUTS']=str(u.memory_snapshot()['swapouts'])
    with (P/'runtime/N3_CLEAN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        u.journal('N3_CLEAN_started',registration_sha256=u.H(HERE/'registration.json'),Part_C_run=False,real_trials=580)
        for year in range(2004,2026):
            for family in ['RP','RF']:
                if not done(OUT/f'IC/{family}_{year}'):admission();one_year(family,year)
        ic_report()
        while not (P/'outputs/N3_intercept_null/report/complete.json').exists():
            if wait_n3_pid is None:
                print('CLEAN complete. Original N3 still running; resume this command after N3 for the combined report.',flush=True);return
            try:os.kill(wait_n3_pid,0)
            except ProcessLookupError:raise RuntimeError('Original N3 stopped without final report; CLEAN complete and retained. Resume after N3 is complete.')
            print('CLEAN complete; waiting for original N3 final report, no repeated fits.',flush=True);time.sleep(30)
        combined_report();authenticate();u.journal('N3_CLEAN_complete',Part_C_run=False,real_trials=580)
        print('STOPPED after combined diagnostic report. Part C not run. Trials remain580.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-n3-pid',type=int);args=parser.parse_args()
    try:main(args.wait_n3_pid)
    except BaseException as exc:print('CLEAN PAUSED/FAILED; completed groups retained:',type(exc).__name__,str(exc),flush=True);raise
