"""Measured calls, admissions and preserved attempts; no inferred fit counts."""
from mio import *
from collections import Counter

def ledger():
    counters={};operations=Counter();reader=dict(files_read=0,train_numeric_rows=0,test_rows_parsed=0,max_numeric_year_by_fold={})
    imported={};verifiers=[];F_construction_calls=0;formation={}
    for p in sorted((HERE/'audit').glob('count_*__*__*.bin')):
        scope,operation,pid=p.name[len('count_'):-len('.bin')].rsplit('__',2)
        counters.setdefault(scope,Counter())[operation]+=p.stat().st_size
    for path in sorted((HERE/'audit').glob('operations_*.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line);scope=row['scope'];operations[(scope,row['operation'])]+=1
            F_construction_calls+=int(row.get('F_construction_calls',0))
            if row['operation']=='factor_formation_window':
                key=str(row['test_year']);value=formation.setdefault(scope,{}).setdefault(key,dict(sessions_used=set(),sessions_lost=set(),records=0))
                value['sessions_used'].add(row['sessions_used']);value['sessions_lost'].add(row['sessions_lost']);value['records']+=1
            if row['operation']=='unchanged_source_import':
                assert fp(ROOT/row['path'])==row['fingerprint'];imported[row['path']]=row['fingerprint']
            if row['operation']=='train_reader':
                reader['files_read']+=row['price_files_read'];reader['train_numeric_rows']+=row['train_numeric_rows'];reader['test_rows_parsed']+=row['test_rows_parsed']
                assert row['max_numeric_year'] is None or row['max_numeric_year']<row['test_year']
                if row['max_numeric_year'] is not None:
                    key=str(row['test_year']);reader['max_numeric_year_by_fold'][key]=max(reader['max_numeric_year_by_fold'].get(key,0),row['max_numeric_year'])
            if row['operation']=='input_verifier':verifiers.append(row)
    assert reader['test_rows_parsed']==0
    return dict(created_at_utc=utc(),measurement='One append-only byte per actual counted call; operation JSON logs for admissions/imports/verifiers.',
                downloads=0,package_installs=0,rl_fits=0,trading_evaluations=0,reader=reader,
                scopes={k:dict(v) for k,v in counters.items()},operation_records=[dict(scope=k[0],operation=k[1],count=v) for k,v in sorted(operations.items())],
                imported_source_fingerprints=imported,input_verifier_attempts=verifiers,F_construction_calls=F_construction_calls,
                factor_formation_windows_by_scope={scope:{k:dict(sessions_used=sorted(v['sessions_used']),sessions_lost=sorted(v['sessions_lost']),records=v['records']) for k,v in folds.items()} for scope,folds in formation.items()},
                first_fold_factor_membership_rule='For Y=2004 only, first session after 2000-01-31 through end-2000; no January membership back-fill; other sections retain full formation year.',
                forbidden_enforcement='Python audit hook denies networking and external mutations; no learner/trading entrypoint is imported or invoked.')
