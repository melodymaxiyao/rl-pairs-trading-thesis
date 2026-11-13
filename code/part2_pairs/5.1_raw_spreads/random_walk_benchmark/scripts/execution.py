"""Original policies, full-cost ledger and active-row decomposition; never learns."""
from rw_io import *
from sources import original,decomposition
import numpy as np

RULES=['B0','B1','B2','B3','H1s']
METRICS=['gross_bps','net_bps','n_trades','drift_bps','timing_bps','other_bps']

def statistics(e,paths,cost,sign,gross_arrays=None,cost_arrays=None):
    p=np.atleast_2d(paths);r=e['forward_bps']
    g=p*r if gross_arrays is None else np.atleast_2d(gross_arrays)
    c=np.abs(np.diff(p,axis=1,prepend=0))*cost/2 if cost_arrays is None else np.atleast_2d(cost_arrays)
    result=decomposition()(p,r,g,c,sign or 1)
    if not sign:result['train_drift_sign']=0
    assert abs(result['net_bps']-result['gross_bps']+result['n_trades']*cost)<=1e-9
    assert abs(result['gross_bps']-result['drift_bps']-result['timing_bps']-result['other_bps'])<=1e-9
    return result

def select_B3(es,risk,cost):
    assert es and all(e['window']=='train' for e in es)
    choice,grid=original().numerical().core.select_threshold(es,cost,risk,'rw_null_B3_train')
    record('B3_selected',B3_selections=1,train_policy_executions=8*len(es),real_test_numeric_rows=0)
    return dict(choice=choice,grid=grid)

def execute(e,risk,choices,sign,tables=None,costs=(0,10,20),cells=None,scope='null_test'):
    core=original().numerical().core;output=[]
    cells=list(cells or RULES+(['z9/TD_z','z9/TD_ens'] if tables is not None else []))
    for cell in cells:
        paths=[]
        if cell=='z9/TD_z':
            assert tables is not None and tables.shape==(10,9,3,3)
            for table in tables:paths.append(core.q_targets(e,table,risk))
        elif cell=='z9/TD_ens':paths=[core.q_targets(e,tables.mean(axis=0),risk)]
        elif cell in ['B1','B2']:paths=[core.threshold_targets(e,2. if cell=='B1' else 1.5,0.,risk)]
        elif cell in ['B0','H1s']:paths=[core.targets(e,constant=0 if cell=='B0' else sign,risk=core.OFF)]
        elif cell!='B3':raise ValueError(cell)
        if paths:record('policy_executed',policy_executions=len(paths),cell=cell,execution_scope=scope)
        for cost in costs:
            used=paths
            if cell=='B3':
                choice=choices[str(cost)]['choice']
                used=[core.targets(e,constant=0,risk=core.OFF) if choice['no_trade'] else core.threshold_targets(e,choice['entry'],choice['exit'],risk)]
                record('policy_executed',policy_executions=1,cell=cell,execution_scope=scope)
            ledgers=[core.ledger(e,p,cost,'rw_'+scope,details=d) for p,d in used]
            result=statistics(e,[p for p,d in used],cost,sign,
                [ledger.gross_bps.to_numpy() for ledger in ledgers],[ledger.cost_bps.to_numpy() for ledger in ledgers])
            record('decomposed_null',decomposition_calls=1,identities_checked=2)
            output.append(dict(cell=cell,cost=int(cost),n_seeds=len(used),**result))
    return output
