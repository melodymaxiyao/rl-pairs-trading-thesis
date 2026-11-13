"""One-command, resumable train-only universe builder. No trading entry point."""
from universe_io import *
sys.path.insert(0,str(HERE))
import argparse,subprocess,time,traceback,pickle,gzip,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from checkpoints import ready,receipt,save_context,load_context,load_rows
from check_prerequisites import protected
from verification_scope import check_sealed_inputs
from accounting import operations
from candidates import stocks_for_year,candidate_pairs
from numeric import FoldData
from worker import real_chunk,null_chunk
from export import csv_new,finalize_fold,export_all
from report import report_all,summary_lines

def command(arguments,label,extra=None):
    token=str(time.time_ns());env=dict(os.environ,**(extra or {}))
    path=HERE/'audit'/(label+'_'+token+'.log')
    with path.open('x') as stream:
        with subprocess.Popen(arguments,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1) as process:
            for line in process.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            code=process.wait()
    record('subprocess_completed',subprocess_calls=1,label=label,exit_code=code)
    if code:raise RuntimeError('Check failed; preserved log: '+str(path))
    return path

def initialize():
    for name in ['audit','runtime/tmp','runtime/cache','runtime/mpl_cache','runtime/checkpoints','runtime/parts','outputs/tables','outputs/plots','pools']:
        (HERE/name).mkdir(parents=True,exist_ok=True)
    from verify_cross_sector_universe import registered
    registered()

def preflight():
    marker=HERE/'audit/production_preflight_ready.json'
    inventories=check_sealed_inputs()
    if ready(marker):return read(marker)
    before=HERE/'audit/production_protected_before.json'
    if not before.exists():write(before,protected())
    finder=[]
    for parent,dirs,names in os.walk(ROOT,followlinks=False):
        parent=Path(parent);dirs[:]=[n for n in dirs if n not in ('.git','.venv','node_modules') and parent/n not in (HERE,ROOT/'experiments/turning_point_detection')]
        if '.DS_Store' in names:finder.append(str((parent/'.DS_Store').relative_to(ROOT)))
    write(HERE/'audit/production_finder_paths_before.json',finder)
    logs=[]
    inputs=[('fresh_holdout_data_v1','verify_fresh_data.py'),('universe_sector_expansion_v1','scripts/verify_sector_universe.py'),
            ('learner_noise_robustness_v1','scripts/verify_learner_noise.py'),('expanded_convergence_gate_v1','scripts/verify_gate.py')]
    for name,entry in inputs:
        scoped=name in ['universe_sector_expansion_v1','learner_noise_robustness_v1']
        wrapper=HERE/'scripts'/('scoped_readonly_entry.py' if scoped else 'readonly_entry.py')
        print('Read-only input verifier:',name,flush=True)
        logs.append(command([sys.executable,'-B',str(wrapper),str(PT/name/entry)],'production_input_'+name))
    after=check_sealed_inputs();assert inventories==after
    write(HERE/'audit/production_input_seals.json',after)
    receipt(marker,logs+[HERE/'audit/production_input_seals.json'],input_verifiers=4,input_files_exact=True)
    return read(marker)

def tests(phase):
    marker=HERE/f'audit/tests_{phase}_ready.json'
    if ready(marker):return read(marker)
    token=str(time.time_ns());xml=HERE/f'audit/tests_{phase}_{token}.xml'
    arguments=[sys.executable,'-B',str(HERE/'scripts/test_entry.py'),'--junitxml',str(xml),
               '--basetemp',str(HERE/f'runtime/pytest_{phase}_{token}'),str(HERE/'tests'),'-q','-s']
    log=command(arguments,'tests_'+phase,dict(FAST_SCOPE='tests_'+phase,FAST_REAL_REPRODUCTION='1',FAST_TEST_PHASE=phase))
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert not any(int(s.get(k,0)) for s in suites for k in ['failures','errors'])
    receipt(marker,[xml,log],tests=sum(int(s.get('tests',0)) for s in suites),phase=phase)
    return read(marker)

def execute_fold(executor,year,chunk_size):
    existing=ready(HERE/f'runtime/final_fold_{year}.json')
    if existing:return existing
    stocks,missing=stocks_for_year(year)
    cache=HERE/f'runtime/fold_data_{year}.json'
    if not ready(cache):save_context(FoldData(year,stocks))
    context=load_context(year)
    candidates=list(candidate_pairs(year));parts=HERE/'runtime/parts'
    csv_new(parts/f'candidates_{year}.csv.gz',candidates)
    csv_new(parts/f'missing_members_by_year_{year}.csv.gz',missing,
            ['ticker','symbol','sector','subindustry','security','classification_group','stored_price','membership_month','test_year','era','exclusion_reason'])
    members=stocks+missing
    gap=[dict(test_year=year,era=era(year),classification_group=kind,members=sum(r['classification_group']==kind for r in members),
              stored_prices=sum(r['classification_group']==kind and r['stored_price'] for r in members),enters_pool=kind=='explicit_GICS')
         for kind in ['explicit_GICS','Yahoo_only','unclassified']]
    # Included stocks already satisfy stored-price admission.
    for stock in stocks:stock.setdefault('stored_price',True)
    csv_new(parts/f'yahoo_only_gap_{year}.csv.gz',gap)
    markers=[];futures=[]
    for chunk,start in enumerate(range(0,len(candidates),chunk_size)):
        marker=HERE/f'runtime/checkpoints/real_{year}_{chunk:05d}.json';markers.append(str(marker))
        if not ready(marker):futures.append(executor.submit(real_chunk,year,chunk,candidates[start:start+chunk_size]))
    for future in as_completed(futures):future.result()
    eligible_tickers=set()
    for marker in markers:
        for row in load_rows(Path(marker)):
            if row['eligible']:eligible_tickers.update((row['leg_A'],row['leg_B']))
    import numpy as np
    train=(context.cal.year>=year-3)&(context.cal.year<year)
    vols=[float(np.std(np.diff(context.logs[t][train]),ddof=1)) for t in sorted(eligible_tickers)]
    assert vols and all(np.isfinite(v) and v>0 for v in vols),('No eligible ticker volatilities for null',year)
    write(HERE/f'runtime/null_volatilities_{year}.json',dict(test_year=year,eligible_tickers=sorted(eligible_tickers),volatilities=vols,ddof=1))
    nullmarkers=[];futures=[]
    for chunk,start in enumerate(range(0,2000,100)):
        marker=HERE/f'runtime/checkpoints/null_{year}_{chunk:05d}.json';nullmarkers.append(str(marker))
        if not ready(marker):futures.append(executor.submit(null_chunk,year,chunk,list(range(start,min(start+100,2000))),vols))
    for future in as_completed(futures):future.result()
    return finalize_fold(year,markers,nullmarkers)

def seal():
    verified=HERE/'audit/unsealed_verification_ready.json'
    if ready(verified):log=HERE/read(verified)['log']
    else:
        log=command([sys.executable,'-B',str(HERE/'verify_cross_sector_universe.py'),'--unsealed'],'verify_unsealed')
        receipt(verified,[log],log=str(log.relative_to(HERE)))
    if (HERE/'operation_ledger.json').exists():ledger=read(HERE/'operation_ledger.json')
    else:
        ledger=operations();write(HERE/'operation_ledger.json',ledger)
    audit=dict(status='PASS',development_only=True,held_out_part=False,trading=False,
               preregistration_sha256=sha(HERE/'preregistration.json'),governing_addendum=read(HERE/'preregistration.json')['governing_addendum'],
               input_seals=read(HERE/'audit/production_input_seals.json'),historical_scope_addendum_sha256=sha(HERE/'addendum_003_verification_scope.json'),
               protected_before_sha256=sha(HERE/'audit/production_protected_before.json'),protected_after_sha256=sha(HERE/'audit/production_protected_after.json'),
               protected_unchanged=True,finder_metadata_excluded_and_listed=True,
               tests_before=read(HERE/'audit/tests_before_ready.json'),tests_after=read(HERE/'audit/tests_after_ready.json'),
               cross_sector_reproduction=read(HERE/'audit/cross_sector_reproduction.json'),
               operation_ledger_sha256=sha(HERE/'operation_ledger.json'),independent_verifier_log=str(log.relative_to(HERE)),
               downloads=0,rl_fits=0,trading_evaluations=0,test_numeric_rows_read=0,PNGs=6)
    write(HERE/'audit.json',audit)
    lock=HERE/'pipeline.lock'
    if not lock.read_text().endswith('sealed\n'):
        with lock.open('a') as stream:stream.write('sealed\n')
    artifacts={}
    for parent,dirs,names in os.walk(HERE,followlinks=False):
        for name in names:
            p=Path(parent)/name
            if p.parent==HERE and name in ['manifest.json','complete.json']:continue
            artifacts[str(p.relative_to(HERE))]=fp(p)
    if not (HERE/'manifest.json').exists():write(HERE/'manifest.json',dict(files=artifacts,created_at_utc=utc()))
    else:
        for name,value in read(HERE/'manifest.json')['files'].items():assert fp(HERE/name)==value,('Seal-resume artifact changed',name)
    listed=read(HERE/'manifest.json')['files']
    late={name:value for name,value in artifacts.items() if name not in listed}
    write(HERE/'complete.json',dict(status='complete',development_only=True,trading=False,
          **{name+'_sha256':sha(HERE/(name+'.json')) for name in ['manifest','audit','operation_ledger','preregistration']},
          candidate_pair_folds=sum(read(HERE/f'runtime/final_fold_{y}.json')['candidate_pair_folds'] for y in range(2004,2026)),
          null_pairs=44000,downloads=0,rl_fits=0,trading_evaluations=0,test_numeric_rows_read=0,late_seal_audit_artifacts=late))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=min(12,os.cpu_count() or 1));parser.add_argument('--chunk-size',type=int,default=256)
    args=parser.parse_args();assert args.workers>=1 and args.chunk_size>=1
    workers=min(args.workers,12,os.cpu_count() or 1)
    guard();initialize()
    with (HERE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'complete.json').exists():
            from verify_cross_sector_universe import verify
            print(verify());print('README:',HERE/'README.md');print('\n'.join(summary_lines(read(HERE/'summary.json'))));return
        if (HERE/'operation_ledger.json').exists():
            check_sealed_inputs();assert protected()==read(HERE/'audit/production_protected_before.json')
            seal()
            print('README:',HERE/'README.md');print('\n'.join(summary_lines(read(HERE/'summary.json'))));return
        try:
            preflight();tests('before')
            write(HERE/'audit/chunk_configuration.json',dict(pair_chunk_size=args.chunk_size,null_chunk_size=100,seed=20261020))
            totals={y:len(stocks_for_year(y)[0])*(len(stocks_for_year(y)[0])-1)//2 for y in range(2004,2026)}
            record('execution_configuration',workers=workers,chunk_size=args.chunk_size,BLAS_threads=1,
                   maximum_sessions_per_pair_window=3*254,z_strategy='One pair at a time; no all-pair z matrix')
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
                for year in range(2004,2026):
                    started=time.perf_counter();already=bool(ready(HERE/f'runtime/final_fold_{year}.json'))
                    result=execute_fold(executor,year,args.chunk_size)
                    if year==2004 and not already:
                        elapsed=time.perf_counter()-started
                        estimate=elapsed/(totals[year]+2000)*sum(totals[y]+2000 for y in range(year+1,2026))
                        obj=dict(observed_first_fold_seconds=elapsed,observed_candidates=totals[year],observed_null_pairs=2000,workers=workers,
                                 remaining_candidate_pair_folds=sum(totals[y] for y in range(year+1,2026)),estimated_remaining_seconds=estimate,
                                 method='Measured first-fold wall time per real+null pair; excludes earlier verifier/reproduction time and later reporting.')
                        write(HERE/'audit/runtime_estimate_first_fold.json',obj)
                        print('Measured remaining runtime estimate:',round(estimate/3600,2),'hours; continuing.',flush=True)
            export_all();report_all();check_sealed_inputs()
            after=protected();write(HERE/'audit/production_protected_after.json',after)
            assert after==read(HERE/'audit/production_protected_before.json'),'Protected roots changed'
            write(HERE/'audit/counter_checkpoint.json',operations())
            tests('after')
            final_after=protected();assert final_after==after,'Protected roots changed during final tests'
            write(HERE/'audit/protected_after_tests_final.json',final_after)
            seal()
        except BaseException as error:
            write(HERE/f'audit/failure_{time.time_ns()}.json',dict(status='stopped',error=repr(error),traceback=traceback.format_exc(),utc=utc()))
            raise
    from verify_cross_sector_universe import verify
    print(verify());print('README:',HERE/'README.md');print('\n'.join(summary_lines(read(HERE/'summary.json'))))

if __name__=='__main__':main()
