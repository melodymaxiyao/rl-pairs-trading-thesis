"""Frozen source memberships, copied results and all-or-stop continuity joins."""
from rw_io import *
import numpy as np

CELLS={'F|B0':'B0','F|B1':'B1','F|B2':'B2','F|B3':'B3','F|H1s':'H1s',
       'z9|TD_z':'z9/TD_z','z9|TD_ens':'z9/TD_ens','z9|MB':'z9/MB',
       'z9_R|TD_ens':'z9_R/TD_ens','z9_R|MB':'z9_R/MB'}

def sample_stratified(rows,size=3000,seed=20261022):
    assert len(rows)>=size
    years=sorted({r['test_year'] for r in rows});counts=np.array([sum(r['test_year']==y for r in rows) for y in years])
    quota=counts*size/len(rows);allocation=np.floor(quota).astype(int)
    for i in np.argsort(-(quota-allocation),kind='stable')[:size-int(allocation.sum())]:allocation[i]+=1
    assert (allocation>0).all() and allocation.sum()==size
    rng=np.random.default_rng(seed);result=[]
    for year,n in zip(years,allocation):
        part=sorted([r for r in rows if r['test_year']==year],key=lambda r:r['pair'])
        result.extend(part[i] for i in sorted(rng.choice(len(part),int(n),replace=False)))
    return result

def source_rows(pool,year,state):
    path=BASELINE/f'runtime/production_{pool}_{year}_{state}_pair_fold.json'
    return read(note(path))

def row_key(r):return r['pair'],int(r['test_year']),r['cell'],str(r['cost_case'])

def continuity_real():
    trend=read(note(TREND/'outputs/tables/per_pair_fold.json'))
    needed=[r for r in trend if r['test_year']<=2020 and r['cell'] in CELLS and str(r['cost_case']) in ['0','10','20']]
    output=[];raw_cache={};keys=set()
    # Validate the entire join in memory before returning or writing any copied rows.
    for r in needed:
        pool=r['pool'];year=r['test_year'];state=r['state'];unit=('bench_'+state if state in ['F','R'] else state)+'_'+pool+'_'+str(year)
        if unit not in raw_cache:
            raw_cache[unit]=read(note(STATE/f'runtime/{unit}_raw.json'))
        source=[v for v in raw_cache[unit] if v['pair']==r['pair'] and v['cell']==r['cell'] and str(v['cost_case'])==str(r['cost_case']) and v.get('feasible',True)]
        assert source,('Missing sealed continuity source',pool,row_key(r))
        for target,original in [('gross_bps','gross_return_bps'),('net_bps','net_return_bps'),('n_trades','n_trades')]:
            value=float(np.mean([v[original] for v in source]))
            assert abs(value-r[target])<=1e-9,('Continuity mismatch: no rows copied',pool,row_key(r),target,value,r[target])
        assert abs(r['gross_bps']-r['drift_bps']-r['timing_bps']-r['other_bps'])<=1e-9
        key=pool,row_key(r);assert key not in keys;keys.add(key)
        output.append(dict(r,pool='OLD_'+pool,cell=CELLS[r['cell']],cost=int(r['cost_case']),view='OLD',
                           available=True,decomposition_available=True,real_source='trend matched to state_ablation'))
    # Costs/cells lacking a sealed split remain NA; no real decomposition is recomputed.
    members={(r['pool'],r['pair'],r['test_year']) for r in needed if r['cell']=='F|B2' and str(r['cost_case'])=='10'}
    for pool,pair,year in sorted(members):
        for state in ['bench_F','z9','z9_R']:
            unit=f'{state}_{pool}_{year}'
            if unit not in raw_cache:raw_cache[unit]=read(note(STATE/f'runtime/{unit}_raw.json'))
            rows=raw_cache[unit]
            for cell,newcell in CELLS.items():
                if not cell.startswith(('F|' if state=='bench_F' else state+'|')):continue
                for cost in [0,10,20]:
                    if (pool,(pair,year,cell,str(cost))) in keys:continue
                    values=[v for v in rows if v['pair']==pair and v['cell']==cell and str(v['cost_case'])==str(cost) and v.get('feasible',True)]
                    if not values:continue
                    output.append(dict(pool='OLD_'+pool,pair=pair,test_year=year,era=era(year),cell=newcell,cost=cost,view='OLD',
                        available=True,decomposition_available=False,reason='No sealed decomposition; copied source metrics only',
                        gross_bps=float(np.mean([v['gross_return_bps'] for v in values])),net_bps=float(np.mean([v['net_return_bps'] for v in values])),
                        n_trades=float(np.mean([v['n_trades'] for v in values])),drift_bps=None,timing_bps=None,other_bps=None))
    record('continuity_join_validated',copied_continuity_rows=len(output),matched_decomposition_rows=len(needed),real_policy_executions=0)
    return output,members

def prepare_cohorts():
    target=HERE/'runtime/cohorts.pkl.gz'
    ready=HERE/'audit/cohorts_ready.json'
    if checkpoint(ready):return load_gzip(target)
    assert checkpoint(HERE/'audit/preflight_ready.json')
    inventory=[r for r in read(note(SECTOR/'outputs/tables/pair_fold_inventory.json')) if r['EXP_ALL']]
    real=[];members={};configs={}
    for pool,k,kind in [('EXP_G_cap',5,'G'),('EXP_G',2,'G'),('EXP_ALL',2,'ALL')]:
        rows=[r for r in inventory if r[pool] and not r['existing']]
        if pool=='EXP_ALL':rows=sample_stratified(rows)
        members[pool]=rows;configs[pool]=dict(K=k,filter=kind,TD_pool=pool)
        for year in range(2004,2026):
            for state in ['bench_F','z9','z9_R']:
                for r in source_rows(pool,year,state):
                    if r['cell'] not in CELLS or r['cost_case'] not in ['0','10','20']:continue
                    real.append(dict(r,cell=CELLS[r['cell']],cost=int(r['cost_case']),view=r['pair_view'],decomposition_available=r['available']))
    old,membership=continuity_real();real.extend(old)
    lineage={(r['pair'],r['test_year']):r for r in read(note(NOISE/'outputs/tables/selected_sources.json'))}
    for pool in ['ALL','G']:
        name='OLD_'+pool;members[name]=[]
        for p,pair,year in sorted(membership):
            if p!=pool:continue
            original=lineage[pair,year]
            a,b=pair.split('_')
            members[name].append(dict(original,anchor=a,candidate=b,existing=True))
        configs[name]=dict(K=5,filter='ALL' if pool=='ALL' else 'G',TD_pool=pool)
    for pool,rows in members.items():
        assert rows and len({(r['pair'],r['test_year']) for r in rows})==len(rows)
    real_path=HERE/'runtime/copied_real.pkl.gz';dump_gzip(real_path,real)
    result=dict(members=members,configs=configs,real_path=str(real_path.relative_to(HERE)))
    dump_gzip(target,result)
    summary={pool:dict(pair_folds=len(rows),K=configs[pool]['K'],requested_null_replicates=len(rows)*configs[pool]['K']) for pool,rows in members.items()}
    write(HERE/'audit/cohort_totals.json',dict(pools=summary,FAST_cap=dict(excluded=True,reason='No sealed real trading results; outside M family; next pool differs'),
          price_martingale_sensitivity=len(members['EXP_G_cap']),continuity_null=44000))
    receipt(ready,[target,real_path,HERE/'audit/cohort_totals.json'],no_real_test_prices=True)
    return result

def tables(pool,year):
    if pool.startswith('OLD_'):path=NOISE/f'outputs/models/{pool[4:]}_{year}.npz'
    else:path=BASELINE/f'outputs/models/production_{pool}_{year}_z9_TD.npz'
    with np.load(note(path),allow_pickle=False) as z:q=z['TD'].copy()
    assert q.shape==(10,9,3,3) and np.isfinite(q).all()
    record('frozen_TD_tables_loaded',TD_tables_loaded=10,rl_fits=0)
    return q
