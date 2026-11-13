"""Registered N3 construction diagnostic. Never invokes trading or policy code."""
import argparse, ast, copy, gzip, hashlib, importlib.util, json, os, pathlib, signal, sys, time
P = pathlib.Path('/Users/maxiyao/FYP_runs/partial_coint_kalman_v1')
HERE = P / 'audit/N3_intercept_null'
OUT = P / 'outputs/N3_intercept_null'
sys.path[:0] = [str(P/'audit/disk_budget_10GiB_v1'), str(P/'scripts')]
import entry as resource_policy
import numpy as np
import pandas as pd
import io_utils as u
from groups import done, finish, array_bytes
import bootstrap, pci, residuals, nulls, null_buckets

def authenticate(code=True):
    resource_policy.authenticate(); bootstrap.validate_sources()
    for name in ['registration.json'] + (['code_registration.json'] if code else []):
        for rel, digest in u.J(HERE/name)['files'].items():
            assert u.H(P/rel) == digest, rel
    assert u.J(P/'audit/partC_decision_seal_v1.json')['cumulative_real_trading_trials'] == 580

def item3_module():
    spec=importlib.util.spec_from_file_location('_N3_hedge_replay',P/'audit/A1_sanity_v2_mask_repair_v1/entry.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.load()

def require_item3():
    assert done(P/'outputs/A1_sanity_v2/report'), 'Finish Item3 before running N3'

def levels(e):
    x=np.cumsum(np.where(np.isfinite(e),e,0.),axis=-1)
    x[~np.isfinite(e)]=np.nan
    return x

def flipped_intercept(historical, signs):
    # Original regressions require all 60 observations finite. Never introduce
    # a new missing-day policy by silently replacing this with nanmean.
    assert historical.shape == signs.shape and historical.shape[-1] == 60
    assert np.isfinite(historical).all()
    return (historical*signs).mean(axis=-1)

def sign_rows(kind, year, ids, dates, replica):
    c,d,a=bootstrap.core_modules()
    return np.stack([d.M.sign_series(c.source().prereg['null_seed'],
        f'RP|{year}:{name}' if kind=='RP' else 'RF|'+name, replica, dates) for name in ids])

def check_n2(causal, signed, signs):
    np.testing.assert_allclose(signed,causal*signs,rtol=1e-10,atol=1e-8,equal_nan=True)

def validate_primary(kind,year,arrays):
    proof=[]
    for wi,x in enumerate(arrays):
        root=P/f'runtime/B_fits/{kind}_{year}_{wi}/pooled';assert done(root)
        fit=u.J(root/'pooled.json');eligible=np.flatnonzero(np.isfinite(x).sum(axis=1)>=500)
        masks=u.J(P/f'runtime/part_B1/{kind}_{year}_{wi}/formation_masks.json')
        np.testing.assert_array_equal(eligible,[int(v['index']) for v in masks])
        assert fit['converged']
        value=pci.profile(x[eligible],fit['rho'],fit['q'])
        np.testing.assert_allclose(value['ll'].sum(),fit['ll'],rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(value['scale'],fit['scale'],rtol=1e-8,atol=1e-8)
        proof.append(dict(world=wi,sealed_pooled_receipt_sha256=u.H(root/'complete.json'),LL=float(value['ll'].sum()),sealed_LL=fit['ll'],scales_match=True))
    return proof

def rf_replay(panel):
    # The identical literal original fallback used and checked by Item3.
    source=ast.parse((P/'scripts/residuals.py').read_text())
    outer=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='rf_year')
    node=copy.deepcopy(next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='day'))
    start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='history' for t in n.targets))
    node.body=node.body[start:];node.name='replay_day';ast.fix_missing_locations(node)
    c,d,a=bootstrap.core_modules();ns=dict(panel=panel,R=panel['returns'],N=panel['returns'].shape[1],d=d,np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(P/'scripts/residuals.py'),'exec'),ns)
    return ns['replay_day']

def rf_increments(year):
    root=OUT/f'increments_RF/{year}'
    if done(root):return
    require_item3();c,d,a=bootstrap.core_modules();p=d.panel();R=p['returns'];dates=p['dates'];Y=dates.astype('datetime64[Y]').astype(int)+1970
    ts=np.flatnonzero(Y==year);names=list(p['meta']['symbols']);N=len(names)
    with np.load(P/f'runtime/RF_increments/{year}/increments.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(ts,z['indices']);real=z['e']
    with np.load(P/f'outputs/A1_sanity_v2/RF_alpha_free/{year}/increments.npz',allow_pickle=False) as z:prime=z['eprime']
    out=np.full((2,N,len(ts)),np.nan);months={};replay=rf_replay(p);checks={};module=item3_module();max_real_error=0.;covered=0
    # Sign streams start before the first construction window, not at year start.
    start=max(0,int(ts[0])-60);sign_dates=dates[start:int(ts[-1])+1]
    signs=np.stack([sign_rows('RF',year,names,sign_dates,r) for r in range(2)])
    for j,t in enumerate(ts):
        u.checkpoint()
        if t==0:continue
        prior=int(t)-1;month=str(dates[prior].astype('datetime64[M]'));yy,mm=map(int,month.split('-'));stem=f'runtime/jobs/RF_{yy}_{mm:02d}'
        if month not in months:
            months[month]={v['index']:v for v in c.source().cache(stem)['days']} if stem+'.pkl.gz' in c.source().expected else {}
            if len(months)>3:months.pop(next(iter(months)))
        sealed=months[month].get(prior)
        if sealed is not None:
            model=None if sealed['status']!='ok' else (sealed['indices'],sealed['weights'],sealed['alpha'],sealed['beta'])
            # Already independently replayed and checked in the completed Item3.
            if model is not None:covered+=1
        else:model=replay(prior)
        if model is None:
            assert not np.isfinite(real[:,j]).any();continue
        ids,W,alpha,beta=model;needed=np.flatnonzero(np.any(W!=0,axis=1))
        hist=R[t-60:t,ids]-(R[t-60:t][:,needed]@W[needed])@beta.T
        assert hist.shape==(60,len(ids)) and np.isfinite(hist).all()
        module.check_parameter(checks,'alpha',hist.mean(axis=0),alpha)
        reconstructed=prime[ids,j]-hist.mean(axis=0)
        np.testing.assert_allclose(reconstructed,real[ids,j],rtol=1e-10,atol=1e-8,equal_nan=True)
        good=np.isfinite(reconstructed);max_real_error=max(max_real_error,float(np.max(abs(reconstructed[good]-real[ids,j][good]))) if good.any() else 0.)
        for rep in range(2):
            prior_sign=signs[rep,ids,t-start-60:t-start]
            alpha_star=flipped_intercept(hist.T,prior_sign)
            out[rep,ids,j]=signs[rep,ids,t-start]*prime[ids,j]-alpha_star
            out[rep,~np.isfinite(real[:,j]),j]=np.nan
        if j%60==0:print('N3 RF increments',year,j,'/',len(ts),flush=True)
    for rep in range(2):
        np.testing.assert_array_equal(np.isfinite(out[rep]),np.isfinite(real))
        check_n2(real,real*signs[rep][:,ts-start],signs[rep][:,ts-start])
    finish(root,{'increments.npz':array_bytes(e=out,indices=ts),'validation.json':dict(xi_one_matches_real=True,N2_post_intercept_flip_checked=True,max_real_error=max_real_error,parameter_checks=checks,covered_days_reused_from_Item3=covered,Item3_receipt_sha256=u.H(P/f'outputs/A1_sanity_v2/RF_alpha_free/{year}/complete.json'))},phase='N3_RF_construction_validated')

def rp_data(year):
    root=OUT/f'RP_formation/{year}'
    if done(root):return
    require_item3();c,d,a=bootstrap.core_modules();p=d.panel();dates=p['dates'];Y=dates.astype('datetime64[Y]').astype(int)+1970
    window=np.flatnonzero((Y>=year-3)&(Y<year));end=int(window[-1]);rows=p['meta']['pairs'][str(year)];ids=[r['pair'] for r in rows]
    x=np.full((3,len(ids),len(window)),np.nan);module=item3_module();refdata={'ids':ids};refs=module.rp_reference_subset(year,refdata);checks={};max_error=0.
    # Construct complete histories, including pre-formation cumulative origin,
    # matching the primary levels' global origin and predict-only mask behavior.
    signed_levels=np.full((2,len(ids),len(window)),np.nan)
    with np.load(P/f'outputs/amended_A1/groups/RP_{year}_0/signals.npz',allow_pickle=False):pass
    for i,row in enumerate(rows):
        if i%75==0:u.checkpoint()
        aa=p['returns'][:end+1,d.index(row['leg_A'],p)];bb=p['returns'][:end+1,d.index(row['leg_B'],p)]
        real,alpha,beta=residuals.pair_out_of_sample(aa,bb)
        for ix,refalpha,refbeta in refs.get(i,[]):
            used=(ix>=59)&(ix<end);loc=ix[used]-59
            module.check_parameter(checks,'alpha',alpha[loc],refalpha[used]);module.check_parameter(checks,'beta',beta[loc],refbeta[used])
            alpha[loc]=refalpha[used];beta[loc]=refbeta[used]
        A=np.lib.stride_tricks.sliding_window_view(aa[:-1],60);B=np.lib.stride_tricks.sliding_window_view(bb[:-1],60)
        historical=A-beta[:,None]*B;mask=np.isfinite(real);prime=aa[60:]-beta*bb[60:]
        reconstructed=prime-historical.mean(axis=1)
        np.testing.assert_allclose(reconstructed[mask],real[mask],rtol=1e-10,atol=1e-8)
        max_error=max(max_error,float(np.max(abs(reconstructed[mask]-real[mask]))) if mask.any() else 0.)
        x[0,i]=levels(real)[window-60]
        for rep in range(2):
            sign=sign_rows('RP',year,[ids[i]],dates[:end+1],rep)[0]
            sign_history=np.lib.stride_tricks.sliding_window_view(sign[:-1],60)
            n3=sign[60:]*prime-(historical*sign_history).mean(axis=1);n3[~mask]=np.nan
            signed=real*sign[60:];check_n2(real,signed,sign[60:])
            x[rep+1,i]=levels(signed)[window-60];signed_levels[rep,i]=levels(n3)[window-60]
        if i%300==0:print('N3 RP construction',year,i,'/',len(ids),flush=True)
    eligible=np.isfinite(x[0]).sum(axis=1)>=500
    for rep in range(2):np.testing.assert_array_equal(np.isfinite(signed_levels[rep]),np.isfinite(x[0]))
    # Validate N2 against the sealed B2 formation masks and Item3 reference
    # construction without invoking rolling OU or any new fit.
    for wi in range(3):
        original=u.J(P/f'runtime/part_B1/RP_{year}_{wi}/formation_masks.json')
        np.testing.assert_array_equal(np.flatnonzero(eligible),[int(v['index']) for v in original])
    sealed_checks=validate_primary('RP',year,x)
    finish(root,{'formation.npz':array_bytes(real=x[0],N2=x[1:],N3=signed_levels,ids=np.asarray(ids,str),indices=window),'validation.json':dict(xi_one_matches_real=True,N2_post_intercept_flip_checked=True,max_real_error=max_error,parameter_checks=checks,formation_sessions=len(window),sealed_real_N2_primary_likelihood_checks=sealed_checks)},phase='N3_RP_construction_validated')

def rf_data(year):
    root=OUT/f'RF_formation/{year}'
    if done(root):return
    c,d,a=bootstrap.core_modules();p=d.panel();dates=p['dates'];Y=dates.astype('datetime64[Y]').astype(int)+1970;names=list(p['meta']['symbols']);N=len(names)
    x=[];n3=[];window=[];offset=np.zeros((3,N));offset_n3=np.zeros((2,N))
    for yy in range(int(Y.min()),year):
        assert done(OUT/f'increments_RF/{yy}')
        with np.load(P/f'runtime/RF_increments/{yy}/increments.npz',allow_pickle=False) as z:real=z['e'];ix=z['indices']
        with np.load(OUT/f'increments_RF/{yy}/increments.npz',allow_pickle=False) as z:inc=z['e']
        old=np.stack([real]+[real*sign_rows('RF',year,names,dates[ix],rep) for rep in range(2)])
        for rep in range(2):check_n2(real,old[rep+1],sign_rows('RF',year,names,dates[ix],rep))
        if yy<year-3:
            offset+=np.nansum(old,axis=-1);offset_n3+=np.nansum(inc,axis=-1)
        else:x.append(old);n3.append(inc);window.extend(ix)
    original=np.concatenate(x,axis=-1);new=np.concatenate(n3,axis=-1)
    lv=offset[:,:,None]+levels(original);newlv=offset_n3[:,:,None]+levels(new)
    for rep in range(2):np.testing.assert_array_equal(np.isfinite(newlv[rep]),np.isfinite(lv[0]))
    eligible=np.isfinite(lv[0]).sum(axis=1)>=500
    for wi in range(3):
        rows=u.J(P/f'runtime/part_B1/RF_{year}_{wi}/formation_masks.json')
        np.testing.assert_array_equal(np.flatnonzero(eligible),[int(v['index']) for v in rows])
    sealed_checks=validate_primary('RF',year,lv)
    finish(root,{'formation.npz':array_bytes(real=lv[0],N2=lv[1:],N3=newlv,ids=np.asarray(names,str),indices=np.asarray(window)),'validation.json':dict(original_masks=True,all_year_increment_validations_passed=True,sealed_real_N2_primary_likelihood_checks=sealed_checks)},phase='N3_RF_formation_validated')

def critical(n):
    bucket=null_buckets.bucket(n);r=P/f'preregistration/amendments/speedup_B1_B2_v1/critical_values/{bucket}'
    assert done(r),'Never recalibrate a sealed critical value'
    return u.J(r/'critical.json')['critical_5percent']

def read_B2(kind,year,wi):
    root=P/f'runtime/B2_bucket_overlays/{kind}_{year}_{wi}'
    if not done(root):root=P/f'runtime/part_B/{kind}_{year}_{wi}'
    assert done(root)
    return pd.read_csv(root/'params.csv.gz',float_precision='round_trip').set_index('index')

def sample_indices(ids,key):
    return np.sort(np.random.default_rng(nulls.seed(key)).choice(ids,size=min(500,len(ids)),replace=False))

def pci_rows(x,indices,root,fixed_critical=None):
    allrows=[]
    for offset in range(0,len(indices),25):
        chunk=root/f'{offset:05d}'
        if not done(chunk):
            rows=[]
            for i in indices[offset:offset+25]:
                u.checkpoint();fit=pci.fit(x[i],pooled=False,checkpoint=u.checkpoint)
                n=int(np.isfinite(x[i]).sum());cv=critical(n) if fixed_critical is None else fixed_critical
                rows.append(dict(index=int(i),converged=fit['converged'],boundary=fit.get('boundary'),rho=fit.get('rho'),q=fit.get('q'),R2_MR=fit.get('R2_MR'),LR=fit.get('LR'),critical_value=cv,
                    LR_reject=bool(fit['converged'] and fit['LR']>cv),PCI_pass=bool(fit['converged'] and fit['LR']>cv and fit['R2_MR']>=.5)))
            finish(chunk,{'fits.json':rows},phase='N3_diagnostic_PCI',registration_sha256=u.H(HERE/'registration.json'))
            print('N3 PCI',root.parent.name,offset+len(rows),'/',len(indices),flush=True)
        allrows.extend(u.J(chunk/'fits.json'))
    return allrows

def metrics(fit,rows,**meta):
    f=pd.DataFrame(rows);valid=f.converged.astype(bool);n=len(f)
    return dict(**meta,pooled_converged=fit['converged'],pooled_rho=fit.get('rho'),pooled_half_life=fit.get('half_life'),pooled_q=fit.get('q'),pooled_R2_MR=fit.get('R2_MR'),pooled_rho_boundary=bool(fit['converged'] and fit['rho']==.999),
        sampled_series=n,converged_series=int(valid.sum()),boundary_series=int(f.loc[valid,'boundary'].astype(bool).sum()),PCI_pass_count=int(f.PCI_pass.astype(bool).sum()),LR_reject_count=int(f.LR_reject.astype(bool).sum()),
        boundary_share_all=float((valid & f.boundary.fillna(False).astype(bool)).mean()),boundary_share_valid=float(f.loc[valid,'boundary'].astype(bool).mean()) if valid.any() else None,
        PCI_pass_rate_all=float(f.PCI_pass.astype(bool).mean()),PCI_pass_rate_valid=float(f.loc[valid,'PCI_pass'].astype(bool).mean()) if valid.any() else None,LR_reject_rate_all=float(f.LR_reject.astype(bool).mean()))

def market(kind,year,rep):
    root=OUT/f'market/{kind}_{year}_{rep}'
    if done(root):return
    data=OUT/f'{kind}_formation/{year}';assert done(data)
    with np.load(data/'formation.npz',allow_pickle=False) as z:x=z['N3'][rep];ids=z['ids']
    eligible=np.flatnonzero(np.isfinite(x).sum(axis=1)>=500);key=f'N3_PCI|{kind}|{year}|{rep}';sample=sample_indices(eligible,key)
    fitroot=OUT/f'market_pooled/{kind}_{year}_{rep}'
    if not done(fitroot):
        previous=OUT/f'market_pooled/{kind}_{year-1}_{rep}';warm=u.J(previous/'fit.json') if done(previous) else None
        fit=pci.fit(x[eligible],pooled=True,checkpoint=u.checkpoint,warm_start=warm)
        finish(fitroot,{'fit.json':fit,'eligible.json':eligible.tolist()},phase='N3_primary_estimator_diagnostic')
    fit=u.J(fitroot/'fit.json');rows=pci_rows(x,sample,OUT/f'market_PCI/{kind}_{year}_{rep}')
    result=[metrics(fit,rows,family=kind,year=year,replica=rep,world='N3',sample_seed_key=key)]
    for world,wi in [('real',0),('N2',rep+1)]:
        oldfitroot=P/f'runtime/B_fits/{kind}_{year}_{wi}/pooled';assert done(oldfitroot);oldfit=u.J(oldfitroot/'pooled.json');oldrows=read_B2(kind,year,wi).loc[sample].reset_index().to_dict('records')
        for row in oldrows:
            cv=critical(int(row['valid_formation_sessions']));row['LR_reject']=bool(row['converged'] and row['LR']>cv);row['PCI_pass']=bool(row['LR_reject'] and row['R2_MR']>=.5)
        result.append(metrics(oldfit,oldrows,family=kind,year=year,replica=rep,world=world,sample_seed_key=key))
    finish(root,{'summary.json':result,'sample.json':dict(key=key,indices=sample.tolist(),series=ids[sample].tolist()),'provenance.json':dict(formation_receipt_sha256=u.H(data/'complete.json'),pooled_receipt_sha256=u.H(fitroot/'complete.json'),real_N2_reused_no_refits=True,registration_sha256=u.H(HERE/'registration.json'))},phase='N3_market_diagnostic')

def simulation_data(size,rep):
    rng=np.random.default_rng(nulls.seed(f'N3_sim|{size}|{rep}'));increments=rng.normal(size=(size,816));history=np.lib.stride_tricks.sliding_window_view(increments[:,:-1],60,axis=1)
    adjusted=increments[:,60:]-history.mean(axis=-1)
    return dict(rolling_intercept=levels(adjusted),RW_control=levels(increments[:,60:]))

def simulation(size,rep,arm):
    root=OUT/f'simulation/{size}_{rep}_{arm}'
    if done(root):return
    x=simulation_data(size,rep)[arm];sample=sample_indices(np.arange(size),f'N3_sim|{size}|{rep}|PCI');fitroot=OUT/f'simulation_pooled/{size}_{rep}_{arm}'
    if not done(fitroot):finish(fitroot,{'fit.json':pci.fit(x,pooled=True,checkpoint=u.checkpoint)},phase='N3_iid_simulation_primary_MLE')
    fit=u.J(fitroot/'fit.json');rows=pci_rows(x,sample,OUT/f'simulation_PCI/{size}_{rep}_{arm}',critical(756))
    summary=metrics(fit,rows,family='RP_like' if size==3000 else 'RF_like',size=size,replica=rep,world=arm,seed_key=f'N3_sim|{size}|{rep}')
    finish(root,{'summary.json':summary,'sample.json':sample.tolist()},phase='N3_iid_simulation_diagnostic')

def markdown(frame):
    def value(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x)
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def report():
    root=OUT/'report'
    if done(root):return
    rows=[];sim=[]
    for kind in ['RP','RF']:
        for year in range(2004,2026):
            for rep in range(2):
                r=OUT/f'market/{kind}_{year}_{rep}';assert done(r);rows+=u.J(r/'summary.json')
    for size in [3000,400]:
        for rep in range(10):
            for arm in ['rolling_intercept','RW_control']:
                r=OUT/f'simulation/{size}_{rep}_{arm}';assert done(r);sim.append(u.J(r/'summary.json'))
    market=pd.DataFrame(rows);simulation=pd.DataFrame(sim)
    columns=['pooled_rho','pooled_half_life','pooled_q','pooled_R2_MR','pooled_rho_boundary','boundary_share_all','boundary_share_valid','PCI_pass_rate_all','PCI_pass_rate_valid','LR_reject_rate_all']
    med=market.groupby(['family','world'])[columns].median().reset_index();sm=simulation.groupby(['family','world'])[columns].median().reset_index()
    # Keep replicas, sample keys and denominators in the long tables; side-by-side
    # rows are paired by family/year/replica, with identical series samples.
    wide=market.pivot(index=['family','year','replica'],columns='world',values=columns);wide.columns=[f'{world}_{metric}' for metric,world in wide.columns];wide=wide.reset_index()
    text='# N3_intercept_null — diagnostic only\n\nNo Part C execution; cumulative real trading trials stay at 580. This diagnostic does not change any sealed result or the decision not to run Part C.\n\n## Market medians across year/replica groups\n\n'+markdown(med)+'\n\n## Pure i.i.d. simulation medians (10 groups per size and arm)\n\n'+markdown(sm)
    text+='\n\n## Descriptive interpretation\n\nIf N3 and the simulated rolling-intercept arm reproduce half-lives of about 25–35 days, R2_MR far above N2 and high PCI pass rates, the real transitory component is mainly produced by the rolling-intercept construction. If N3 resembles N2 (boundary rho, low R2_MR, approximately 5% LR rejections), the construction does not explain the real component. Read the observed tables against both patterns; intermediate patterns remain inconclusive.\n\n'
    for family in ['RP','RF']:
        block=med[med.family==family].set_index('world')
        text+=f"Observed {family}: median half-life real={block.loc['real','pooled_half_life']:.4f}, N2={block.loc['N2','pooled_half_life']:.4f}, N3={block.loc['N3','pooled_half_life']:.4f}; median R2_MR real={block.loc['real','pooled_R2_MR']:.4f}, N2={block.loc['N2','pooled_R2_MR']:.4f}, N3={block.loc['N3','pooled_R2_MR']:.4f}; median PCI pass rates real={block.loc['real','PCI_pass_rate_all']:.4f}, N2={block.loc['N2','PCI_pass_rate_all']:.4f}, N3={block.loc['N3','PCI_pass_rate_all']:.4f}.\n\n"
    text+='PCI_pass is the original B2 joint gate (LR above sealed bucket critical and R2_MR >= 0.5); LR_reject is the test alone. Boundary shares refer to sampled per-series fits; pooled_rho_boundary is reported separately. All/valid denominators are explicit. RF samples below 500 use every eligible series without replacement; RF-like simulation uses all 400. Real entries are repeated for the two matched replica-specific subsamples, not independent real fits.\n\nFull per-year side-by-side results and simulation replications are in the accompanying CSV files. New fits use unchanged pci.fit, chunk size 75, bounds, starts and concentration. Simulation burn-in is excluded from the likelihood. The 700+ critical value is reused without recalibration.\n'
    finish(root,{'market_by_group.csv':market,'market_side_by_side.csv':wide,'market_medians.csv':med,'simulation_by_group.csv':simulation,'simulation_medians.csv':sm,'N3_intercept_null.md':text.encode(),'status.json':dict(status='complete',diagnostic_only=True,part_C_executed=False,cumulative_real_trading_trials=580)},phase='N3_diagnostic_report')
    u.publish(P/'N3_intercept_null.md',text);print(text,flush=True)

def main():
    authenticate();require_item3();u.install_guard();os.environ['PARTIAL_RAM_GUARD']='1';os.environ['PARTIAL_INITIAL_SWAPOUTS']=str(u.memory_snapshot()['swapouts'])
    import fcntl
    with (P/'runtime/N3_intercept_null.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        u.journal('N3_diagnostic_started',registration_sha256=u.H(HERE/'registration.json'),part_C_started=False,real_trials=580)
        # All construction validations precede every N3 market fit.
        c,d,a=bootstrap.core_modules();first=int(d.panel()['dates'].astype('datetime64[Y]').astype(int).min()+1970)
        for year in range(first,2025):rf_increments(year)
        for year in range(2004,2026):rp_data(year);rf_data(year)
        for year in range(2004,2026):
            for kind in ['RP','RF']:
                for rep in range(2):market(kind,year,rep)
        for size in [3000,400]:
            for rep in range(10):
                for arm in ['rolling_intercept','RW_control']:simulation(size,rep,arm)
        report();authenticate();u.journal('N3_diagnostic_complete',part_C_started=False,real_trials=580)
        print('STOPPED after N3 report; Part C not run; cumulative real trading trials=580',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('N3 PAUSED/FAILED; completed groups retained:',type(exc).__name__,str(exc),flush=True);raise
