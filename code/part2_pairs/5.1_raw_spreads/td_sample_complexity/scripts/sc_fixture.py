"""Main-only sealed real fixture; cached contexts only, no raw price reader."""
from sc_common import *
from sc_sources import anchor,native,core,load
import numpy as np

def source_pickle(path):
    with gzip.open(path,'rb') as f:return pickle.load(f)

def run():
    ready=HERE/'audit/real_fixture.json'
    if ready.exists():return read(ready)
    year=2004;pool='P_EXP'
    plan=source_pickle(SOURCE/f'runtime/plans/{pool}_{year}_real_0_primary.pkl.gz');risk=plan['risks']['A_TR']['final']
    choices=source_pickle(SOURCE/f'runtime/choices/{pool}_{year}_real_0_primary.pkl.gz')['choices']['10']
    model=source_pickle(SOURCE/f'outputs/models/{pool}_{year}_TD8.pkl.gz')
    expected={cell:source_pickle(SOURCE/f'runtime/cells/{pool}_{year}_{cell}_real_0_primary.pkl.gz')['scores'] for cell in ['R_TR_aln','B3_TR_aln','TD8']}
    rows=[r for r in expected['R_TR_aln'] if r['cost']==10 and r['available']][:3];assert len(rows)==3
    checked=[]
    storage=load(SOURCE/'audit/repairs/bounded_storage_v4/storage_adapter.py','_sc_fixture_storage')
    archives={}
    for row in rows:
        path=SOURCE/f'runtime/contexts/{year}/{row["pair"]}/{row["leg_A"]}--{row["leg_B"]}/real_0_primary_full.pkl.gz'
        assert path.is_file(),('Required sealed real fixture context missing',path)
        c=source_pickle(path);m,s=anchor().rolling(c['level'],252,True)
        np.testing.assert_array_equal(m,c['anchors']['A_TR']['m']);np.testing.assert_array_equal(s,c['anchors']['A_TR']['s'])
        e=next(e for e in c['episodes']['A_TR'] if e['year']==year)
        for name in expected:
            if name=='R_TR_aln':p,d=native().policy(e,entry=2.,aligned=True,risk=risk)
            elif name=='B3_TR_aln':p,d=native().policy(e,entry=choices['entry'],exit=choices['exit'],aligned=True,risk=risk) if not choices['no_trade'] else native().policy(e,q=np.zeros((8,2)),risk=risk)
            else:p,d=native().policy(e,q=model['ensemble'],risk=risk)
            actual=core().metrics(core().ledger(e,p,10,'synthetic_task_fixture',details=d),10)[0]
            target=next(r for r in expected[name] if r['pair']==row['pair'] and r['cost']==10)
            assert abs(actual['net_return_bps']-target['net_bps'])<=1e-9,(name,row['pair'])
            for key in set(actual)&set(target):
                a,b=actual[key],target[key]
                if isinstance(a,(int,float,np.number)) and isinstance(b,(int,float,np.number)):
                    assert (np.isnan(a) and np.isnan(b)) or abs(a-b)<=1e-9,(name,row['pair'],key,a,b)
            if name not in archives:
                source=source_pickle(SOURCE/f'runtime/cells/{pool}_{year}_{name}_real_0_primary.pkl.gz')
                chunk=storage.unpack_chunk(source_pickle(SOURCE/(source['chunks'][0]+'.pkl.gz')))
                archives[name]={(r['pair'],r['cost']):r for r in chunk['daily']}
            daily=archives[name][row['pair'],10]
            np.testing.assert_array_equal(p,daily['paths'][0])
            assert actual['n_trades']==target['n_trades'];checked.append(dict(pair=row['pair'],cell=name,max_net_error=abs(actual['net_return_bps']-target['net_bps'])))
    value=dict(status='PASS',pair_folds=3,cells=3,rows=checked,raw_real_price_reads=0,real_fixture_context_reads=3,source='sealed real full contexts and unchanged executor/ledger')
    write(ready,value);return value
