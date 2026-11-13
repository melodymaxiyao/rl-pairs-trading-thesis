"""Required no-predictability sanity and saved-ledger replay fixtures."""
from rw_io import *
from simulation import rng_for,draw_logs
from sources import helpers,episodes,original,bins,decomposition
from execution import statistics
import numpy as np
import pandas as pd

def sanity(count=20000,force=False):
    marker=HERE/'audit/martingale_sanity.json'
    if marker.exists() and not force:
        result=read(marker);assert result['n_pairs']==20000 and result['PASS'];return result
    core=original().numerical().core;h=helpers();cal=h.full_cal[h.full_cal.year<=2004]
    rows=[]
    for index in range(count):
        rng,_=rng_for('SANITY',2004,str(index),0)
        la,lb=draw_logs(len(cal),.02,.015,.4,rng)
        es=episodes(cal,la,lb,2004,f'SANITY_{index}');e=es[-1]
        p,d=core.threshold_targets(e,1.5,0.,core.OFF)
        ledger=core.ledger(e,p,0.,'rw_sanity',details=d)
        rows.append(statistics(e,p,0.,1,ledger.gross_bps.to_numpy(),ledger.cost_bps.to_numpy()))
        record('sanity_policy',sanity_policy_executions=1,policy_executions=1)
    gross=np.array([r['gross_bps'] for r in rows]);mean=float(gross.mean());se=float(gross.std(ddof=1)/np.sqrt(count))
    drift=float(np.mean([r['drift_bps'] for r in rows]));timing=float(np.mean([r['timing_bps'] for r in rows]))
    passed=abs(mean)<=3*se and drift<0 and timing>0
    result=dict(n_pairs=count,gross_mean_bps=mean,gross_MC_SE_bps=se,drift_bps=drift,timing_bps=timing,PASS=passed,
        primary_log_mean=0.,fixed_sigma_A=.02,fixed_sigma_B=.015,rho=.4,forced_stop_cap_lock=False,annual_terminal_bookkeeping=True)
    if count==20000:write(marker,result)
    assert passed,('Martingale sanity failed; stop',result)
    return result

def real_fixture():
    pool='ALL'
    for year in range(2004,2021):
        with np.load(note(NOISE/f'outputs/ledgers/{pool}_{year}.npz'),allow_pickle=False) as archive:
            a={k:archive[k] for k in archive.files}
        descriptors=[json.loads(str(v)) for v in a['descriptors']]
        slot=next(i for i,v in enumerate(descriptors) if v[0]=='TD_ens')
        ci=list(a['cases']).index('10')
        active=np.flatnonzero((a['positions'][:,ci,slot]!=0).any(axis=1))
        if len(active):break
    assert len(active),'A non-flat sealed TD_ens fixture is required'
    pair_index=int(active[0]);pair=str(a['pairs'][pair_index])
    e=bins(dict(pair=pair,fold=1,year=year,window='test',dates=pd.to_datetime(a['dates']),z=a['z'][pair_index],forward_bps=a['forward'][pair_index],beta=np.zeros(len(a['dates']))))
    with np.load(note(NOISE/f'outputs/models/{pool}_{year}.npz'),allow_pickle=False) as z:q=z['TD'].mean(axis=0)
    risk=read(note(NOISE/f'outputs/models/{pool}_{year}.json'))['final_risk']
    p,_=original().numerical().core.q_targets(e,q,risk)
    np.testing.assert_array_equal(p,a['positions'][pair_index,ci,slot])
    real=next(r for r in read(note(TREND/'outputs/tables/per_pair_fold.json')) if r['pool']==pool and r['pair']==pair and r['test_year']==year and r['cell']=='z9|TD_ens' and str(r['cost_case'])=='10')
    from rw_io import STATE
    with np.load(note(STATE/f'outputs/ledgers/z9_{pool}_{year}.npz'),allow_pickle=False) as sealed:
        index=list(sealed['pairs']).index(pair)
        np.testing.assert_array_equal(p,sealed['positions'][index,ci,10])
        actual=statistics(e,p,10.,real['train_drift_sign'],sealed['gross_bps'][index,ci,10],sealed['cost_bps'][index,ci,10])
    for key in ['gross_bps','net_bps','n_trades','drift_bps','timing_bps','other_bps']:assert abs(actual[key]-real[key])<=1e-9,(key,actual[key],real[key])
    record('sealed_fixture_only',real_fixture_policy_executions=1,policy_executions=1,real_test_numeric_rows=0)
    result=dict(status='PASS',pair=pair,test_year=year,TD_path_exact=True,active_row_decomposition_exact=True,real_prices_read=False)
    write(HERE/'audit/sealed_fixture_active.json',result)
    return result
