"""Measured operations from all attempts, with validation and production scopes."""
from rw_io import *
from collections import Counter,defaultdict

def operations():
    scopes=defaultdict(Counter);imports={};events=0
    for path in sorted((HERE/'audit').glob('operations_*.jsonl')):
        lines=path.read_text().splitlines()
        for index,line in enumerate(lines):
            try:r=json.loads(line)
            except json.JSONDecodeError:
                assert index==len(lines)-1;scopes['interrupted']['partial_log_lines']+=1;continue
            assert not any(r.get(k,0) for k in ['downloads','rl_fits','model_based_plans','real_test_numeric_rows'])
            events+=1
            for key,value in r.items():
                if key in ['year','fold','test_year','score_year','max_numeric_year','seed','cost_bps','exit_code','workers']:continue
                if isinstance(value,int) and not isinstance(value,bool):scopes[r['scope']][key]+=value
            if r['operation']=='unchanged_source_import':
                name=r['path'];value=r['fingerprint'];assert name not in imports or imports[name]==value;imports[name]=value
            if 'max_numeric_year' in r:assert r['max_numeric_year']<r['test_year']
    after={name:fp(ROOT/name) for name in imports};assert imports==after
    return dict(downloads=0,package_installs=0,rl_fits=0,model_based_plans=0,real_test_numeric_rows=0,
        actual_operation_records=events,scopes={k:dict(v) for k,v in scopes.items()},imported_modules_before=imports,imported_modules_after=after,
        total_policy_executions=sum(v['policy_executions']+v['train_policy_executions'] for v in scopes.values()),
        FAST_cap=dict(excluded=True,null_replicates=0,real_executions=0,reason='No sealed real trading results; not in M2-M5; next planned pool differs'),
        counting='policy_executions counts actual target-path calls; B3 grid calls are train_policy_executions. policy_ledger trading_evaluations counts every physical cost ledger, including seeds. Synthetic validation and sealed-fixture replays are separate scopes. Original input verifier arithmetic stays in original logs.')
