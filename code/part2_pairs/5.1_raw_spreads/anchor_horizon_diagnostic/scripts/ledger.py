from aio import *
from collections import Counter

def ledger():
    measured={};records=Counter();readers={};imports={};verifiers=[];F_calls=Counter()
    for path in (HERE/'audit').glob('count_*__*__*.bin'):
        scope,operation,pid=path.name[len('count_'):-4].rsplit('__',2)
        measured.setdefault(scope,Counter())[operation]+=path.stat().st_size
    for path in (HERE/'audit').glob('operations_*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line);scope=row['scope'];records[(scope,row['operation'])]+=1
            F_calls[scope]+=int(row.get('F_construction_calls',0))
            if row['operation']=='train_reader':
                r=readers.setdefault(scope,dict(price_files_read=0,train_numeric_rows=0,test_rows_parsed=0,max_numeric_year_by_fold={}))
                for key in ['price_files_read','train_numeric_rows','test_rows_parsed']:r[key]+=row[key]
                assert row['test_rows_parsed']==0 and (row['max_numeric_year'] is None or row['max_numeric_year']<row['test_year'])
                if row['max_numeric_year'] is not None:
                    y=str(row['test_year']);r['max_numeric_year_by_fold'][y]=max(r['max_numeric_year_by_fold'].get(y,0),row['max_numeric_year'])
            if row['operation']=='unchanged_source_import':
                assert fp(ROOT/row['path'])==row['fingerprint'];imports[row['path']]=row['fingerprint']
            if row['operation']=='input_verifier':verifiers.append(row)
    return dict(created_at_utc=utc(),measurement='Append-only one-byte counters for each actual call/evaluated event; append-only operation records for train admissions, original F calls and source imports. Interrupted/repeated work is counted.',
        downloads=0,package_installs=0,rl_fits=0,trading_evaluations=0,
        event_pnl_evaluations=sum(v.get('event_pnl_evaluations',0) for v in measured.values()),
        scopes={k:dict(v) for k,v in measured.items()},reader_by_scope=readers,original_F_construction_calls_by_scope=dict(F_calls),
        operation_records=[dict(scope=k[0],operation=k[1],count=v) for k,v in sorted(records.items())],
        imported_source_fingerprints=imports,input_verifier_attempts=verifiers,
        enforcement='Network and writes outside this task denied; original learner/trading entrypoints disabled by original frozen helpers. Event accounting is separate from trading evaluation.')
