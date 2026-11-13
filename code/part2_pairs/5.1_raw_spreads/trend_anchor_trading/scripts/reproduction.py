"""Mandatory deferred real/source acceptance checks; failures stop the main run."""
from aio import *
from store import pickle_read,commit,cached
from sources import baseline_sources,baseline_fold,core,anchor_sample,original_inference
from originals import fixed_spread,read_prices
from market import context,surfaces
import native
import numpy as np

def legacy_arrays_hash(es):
    # Exact sealed baseline signature, which predates the new feature arrays.
    h=hashlib.sha256()
    for e in es:
        h.update(json.dumps([e['pair'],e['year'],e['window']]).encode())
        for key in ['bins','z','forward_bps']:
            a=np.ascontiguousarray(e[key]);h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def exact_arrays(expected,actual,label):
    for k in ['q','curves','snapshots','visits']:
        try:np.testing.assert_array_equal(expected[k],actual[k])
        except AssertionError:
            a=np.asarray(expected[k]);b=np.asarray(actual[k]);ix=np.argwhere(~((a==b)|(np.isnan(a)&np.isnan(b))))
            first=ix[0].tolist() if len(ix) else None
            write(HERE/f'audit/kernel_first_difference_{time.time_ns()}.json',dict(label=label,array=k,first_index=first,expected=float(a[tuple(first)]) if first else None,actual=float(b[tuple(first)]) if first else None))
            raise RuntimeError('Exact legacy kernel equivalence failed; first differing tensor index preserved; user clarification required')

def saved_fit():
    stem='runtime/reproduction/saved_fit';old=cached(stem)
    if old is not None:return old
    source=BASELINE/'runtime/reproduction_input_G_2004.pkl';b=pickle_read(note(source));P=b['pools']['G'];es=b['episodes'];year=2004
    final=[e for e in es if e['window']=='train' and e['pair'] in set(P['final'])]
    folder=BASELINE/'runtime/fit_cache/reproduction_G_2004_z9';assert folder.is_dir(),('Saved v3 fit fixture missing',folder)
    candidates=[]
    for path in sorted(folder.glob('*_ready.json')):
        marker=read(note(path));sig=marker['signature']
        if sig['train']==legacy_arrays_hash(final) and sig['validation']==legacy_arrays_hash([]):candidates.append((path,marker))
    assert candidates,'No matching sealed final-fit signature; stop without substituting fixture'
    path,marker=candidates[0];sig=marker['signature'];expected=pickle_read(note(BASELINE/marker['pickle']))
    c=core();old_kernel=c.kernel;original=old_kernel()
    kwargs=dict(alpha=sig['alpha'],gamma=sig['gamma'],seed=sig['seed'],tag='sealed_fixture',risk=sig['risk'],**sig['kwargs'])
    first=c.fit(final,[],**kwargs);exact_arrays(expected,first,'original producer versus saved v3')
    adapter=native.kernel()
    for name in ['train','policy','ledger','allowed_mask','max_allowed','reward_bps']:
        getattr(adapter,name).argtypes=getattr(original,name).argtypes;getattr(adapter,name).restype=getattr(original,name).restype
    c.kernel=lambda:adapter
    try:
        second=c.fit(final,[],**kwargs);exact_arrays(expected,second,'adapter original configuration versus saved v3')
        for e in final:
            actual=c.q_targets(e,second['q'],sig['risk'])[0]
            c.kernel=old_kernel
            wanted=c.q_targets(e,first['q'],sig['risk'])[0];c.kernel=lambda:adapter
            np.testing.assert_array_equal(wanted,actual)
    finally:c.kernel=old_kernel
    return commit(stem,dict(status='PASS',fixture=str(path.relative_to(ROOT)),Q_tables_exact=True,policies_exact=True,first_difference=None))

def train_anchor_fixture():
    stem='runtime/reproduction/anchor_events';old=cached(stem)
    if old is not None:return old
    from engine import anchors,events
    from originals import original_episodes
    checked=0
    for year in [2004,2012]:
        rows=anchor_sample(year)
        for relation in ['cross_sector','same_sector','same_subindustry']:
            row=next(r for r in rows if r['sector_relation']==relation);c=context(row,year)
            cal=c['cal'];a,b=[np.log(read_prices(FRESH/f'data/raw/prices/{row[k].lower()}_yahoo.csv',year)['Adj Close'].reindex(cal).to_numpy(float)) for k in ['leg_A','leg_B']]
            spread=fixed_spread(cal,a,b,year);assert c['beta']==spread['beta']
            np.testing.assert_array_equal(c['r'],spread['r']);np.testing.assert_array_equal(c['level'],spread['level'])
            es=original_episodes(cal,a,b,year,row['pair']);old_anchors=anchors(cal,spread['level'],year,es)
            for name in ['A_TR','A_R63','A_F']:
                for key in ['m','s','d']:np.testing.assert_array_equal(c['anchors'][name][key],old_anchors[name][key])
                left,_=events(cal,c['level'],c['r'],c['anchors'][name],name,'within-year',year)
                right,_=events(cal,spread['level'],spread['r'],old_anchors[name],name,'within-year',year)
                assert equivalent(left,right)
                checked+=1
    return commit(stem,dict(status='PASS',pair_folds=6,anchor_pair_folds=checked,events_exact=True))

def continuity():
    stem='runtime/reproduction/B2';old=cached(stem)
    if old is not None:return old
    from execution import job
    checked=0;maximum=0.
    for year in YEARS:
        actual=job('P_EXP',year,'R_F_B2')['scores']
        expected=read(note(BASELINE/f'runtime/production_EXP_G_cap_{year}_bench_F_pair_fold.json'))
        expected={(r['pair'],str(r['cost_case'])):r for r in expected if r['cell']=='F|B2' and not r['existing'] and str(r['cost_case']) in ['5','10','20']}
        assert len(actual)==len(expected)
        archive=note(BASELINE/f'outputs/ledgers/production_EXP_G_cap_{year}_bench_F.npz')
        with np.load(archive,allow_pickle=False) as z:
            pi={str(p):i for i,p in enumerate(z['pairs'])};ci={str(c):i for i,c in enumerate(z['cases'])};slot=list(z['methods']).index('B2')
            from store import daily_rows
            ours=daily_rows(job('P_EXP',year,'R_F_B2'))
            for d in ours:np.testing.assert_array_equal(d['paths'][0],z['positions'][pi[d['pair']],ci[str(d['cost'])],slot])
        for r in actual:
            oldrow=expected[r['pair'],str(r['cost'])];assert bool(r['available'])==bool(oldrow['available'])
            if not r['available']:continue
            error=max(abs(r[k]-oldrow[k]) for k in ['net_bps','gross_bps','n_trades','drift_bps','timing_bps','other_bps'])
            assert error<=1e-9,('B2 continuity mismatch',year,r['pair'],r['cost'],error)
            checked+=1;maximum=max(maximum,error)
    return commit(stem,dict(status='PASS',rows=checked,maximum_error=maximum,positions_exact=True))

def accounting_fixture():
    stem='runtime/reproduction/accounting';old=cached(stem)
    if old is not None:return old
    f=original_inference();d=read(note(BASELINE/'outputs/tables/DSR_EXP_G_cap.json'))
    for row in d:
        if row.get('DSR') is None:continue
        actual=f.dsr(row['SR'],row['T'],row['skewness'],row['kurtosis'],row['cross_trial_SR_variance'],row['N'])
        assert abs(actual['DSR']-row['DSR'])<=1e-12 and abs(actual['SR_star']-row['SR_star'])<=1e-12
    rows=read(note(BASELINE/'outputs/tables/PBO_matrix_EXP_G_cap.json'));cells=list(dict.fromkeys(r['cell'] for r in rows));lookup={(r['cell'],r['test_year']):r['mean_net_bps'] for r in rows}
    matrix=np.array([[lookup[cell,y] for y in YEARS] for cell in cells]);actual,split=f.pbo(matrix);expected=read(note(BASELINE/'outputs/tables/PBO_EXP_G_cap.json'))[0]
    assert actual['n_splits']==expected['n_splits']==705432 and abs(actual['PBO']-expected['PBO'])<=1e-12
    return commit(stem,dict(status='PASS',DSR_cases=len(d),PBO_splits=actual['n_splits'],PBO=actual['PBO']))
