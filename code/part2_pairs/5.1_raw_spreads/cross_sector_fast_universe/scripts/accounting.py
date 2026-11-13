"""Aggregate actual operation records, including repeated interrupted attempts."""
from universe_io import *
from collections import Counter,defaultdict

def operations():
    scopes=defaultdict(Counter);imports={};readers=[];events=0
    for path in sorted((HERE/'audit').glob('operations_*.jsonl')):
        lines=path.read_text().splitlines()
        for index,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                assert index==len(lines)-1,('Malformed operation ledger',path,index)
                scopes['interrupted']['partial_log_lines']+=1;continue
            events+=1;scope=row['scope'];scopes[scope]['operation_records']+=1
            for key,value in row.items():
                if key in ['operation','scope','utc','test_year','max_numeric_year','test_rows_used','chunk','year','inner',
                           'score_year','exit_code','workers','chunk_size','BLAS_threads','maximum_sessions_per_pair_window']:continue
                if isinstance(value,int) and not isinstance(value,bool):scopes[scope][key]+=value
            assert not any(row.get(k,0) for k in ['downloads','rl_fits','trading_evaluations','test_numeric_rows'])
            if row['operation']=='unchanged_source_import':
                name=row['path'];value=row['fingerprint']
                assert name not in imports or imports[name]==value
                imports[name]=value
            if row['operation'] in ['date_filtered_input','fold_cache_load']:
                assert row.get('max_numeric_year') is None or row['max_numeric_year']<row['test_year']
                readers.append({k:row[k] for k in ['scope','test_year','max_numeric_year','test_numeric_rows'] if k in row})
    after={name:fp(ROOT/name) for name in imports};assert imports==after
    estimates=[read(p) for p in sorted((HERE/'audit').glob('runtime_estimate*.json'))]
    preflight={}
    for path in list((HERE/'audit').glob('prerequisites_*.json'))+list((HERE/'audit').glob('remaining_prerequisites_*.json'))+list((HERE/'audit').glob('*_scoped_verifier.json')):
        value=read(path)
        values=value if isinstance(value,list) else value.get('results',[value])
        for row in values:
            if 'log' in row and 'exit_code' in row:preflight[row['log']]=row
    removals=defaultdict(Counter)
    for path in (HERE/'runtime').glob('final_fold_*.json'):
        for label,counts in read(path).get('unique_event_removals',{}).items():removals[label].update(counts)
    return dict(downloads=0,package_installs=0,rl_fits=0,trading_evaluations=0,test_numeric_rows=0,
                actual_operation_records=events,scopes={k:dict(v) for k,v in scopes.items()},
                imported_modules_before=imports,imported_modules_after=after,date_reader_audit=readers,
                runtime_estimates=estimates,
                preparation_input_verifier_invocations=len(preflight),preparation_input_verifier_passes=sum(r['exit_code']==0 for r in preflight.values()),
                preparation_original_failures_preserved=[r['log'] for r in preflight.values() if r['exit_code']!=0],
                unique_completed_fold_event_removals={k:dict(v) for k,v in removals.items()},
                counting_rule='Actual recorded calls, including repeated attempts, are separated by production, synthetic, and reproduction-test scope. Historical source-verifier arithmetic is retained only in its original logs, not counted as new trading.',
                overlap_removal_rule='At anchor 1.5, missing retained entry dates are overlaps or first-session exclusions; retained entries whose legacy endpoint crossed a year boundary count separately as year-end-restricted events. Categories are disjoint.')
