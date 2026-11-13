"""One-command resumable benchmark; policies are executed, never trained/planned."""
from rw_io import *
from preflight import preflight
from protection import protected
from verification import check_inputs,finder_inventory
from accounting import operations
from cohorts import prepare_cohorts
from jobs import run_pool_fold
from report import report_all,headline
import argparse,subprocess,traceback,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed

def tests(phase):
    marker=HERE/f'audit/tests_{phase}_ready.json'
    if checkpoint(marker):return
    token=str(time.time_ns());xml=HERE/f'audit/tests_{phase}_{token}.xml';log=xml.with_suffix('.log')
    env=dict(os.environ,RW_SCOPE='tests_'+phase,RW_FULL_TESTS='1',RW_TEST_PHASE=phase)
    args=[sys.executable,'-B',str(HERE/'scripts/test_entry.py'),str(HERE/'tests'),'-q',
          '--basetemp',str(HERE/f'runtime/pytest_{phase}_{token}'),'--junitxml',str(xml)]
    with log.open('x') as output:code=subprocess.run(args,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT).returncode
    print(log.read_text(),flush=True);assert code==0,('Tests failed; original log preserved',log)
    receipt(marker,[xml,log],phase=phase)

def seal():
    def independent(unsealed):
        path=HERE/f'audit/verify_{"unsealed" if unsealed else "sealed"}_{time.time_ns()}.log'
        args=[sys.executable,'-B',str(HERE/'verify_rw_benchmark.py'),'--no-current-protected']
        if unsealed:args.append('--unsealed')
        with path.open('x') as output:code=subprocess.run(args,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT).returncode
        print(path.read_text(),flush=True);assert code==0,('Independent verification failed; log preserved',path)
        return path
    independent(True)
    if not (HERE/'operation_ledger.json').exists():write(HERE/'operation_ledger.json',operations())
    audit=dict(status='PASS',input_inventories=read(HERE/'audit/preflight_ready.json')['inputs'],
        protected_unchanged=True,protected_before_sha256=sha(HERE/'audit/protected_before.json'),protected_after_sha256=sha(HERE/'audit/protected_after.json'),
        tests_before=read(HERE/'audit/tests_before_ready.json'),tests_after=read(HERE/'audit/tests_after_ready.json'),
        real_fixture=read(HERE/'audit/sealed_fixture_active.json'),sanity=read(HERE/'audit/martingale_sanity.json'),
        primary_null_log_mean=0.,FAST_cap_excluded=True,downloads=0,rl_fits=0,model_based_plans=0,real_test_numeric_rows=0)
    write(HERE/'audit.json',audit)
    if 'Sealed complete.' not in (HERE/'pipeline.lock').read_text():
        with (HERE/'pipeline.lock').open('a') as lock:lock.write('Sealed complete.\n')
    excluded={'manifest.json','complete.json'}
    files={str(p.relative_to(HERE)):fp(p) for p in HERE.rglob('*') if (p.is_file() or p.is_symlink()) and p not in [HERE/name for name in excluded]}
    if (HERE/'manifest.json').exists():
        for name,value in read(HERE/'manifest.json')['files'].items():assert fp(HERE/name)==value,('Seal resume changed artifact',name)
    else:write(HERE/'manifest.json',dict(files=files))
    listed=read(HERE/'manifest.json')['files']
    late={name:value for name,value in files.items() if name not in listed}
    write(HERE/'complete.json',dict(status='complete',development_only=True,**{k+'_sha256':sha(HERE/(k+'.json')) for k in ['manifest','audit','preregistration','operation_ledger']},
         downloads=0,rl_fits=0,model_based_plans=0,real_test_numeric_rows=0,FAST_cap_excluded=True,late_seal_artifacts=late))
    # A sealed read-only check does not create a post-seal file.
    subprocess.run([sys.executable,'-B',str(HERE/'verify_rw_benchmark.py'),'--no-current-protected'],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=min(12,os.cpu_count() or 1));args=parser.parse_args()
    assert 1<=args.workers<=12
    guard();registered();sys.path.insert(0,str(HERE))
    with (HERE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'complete.json').exists():
            from verify_rw_benchmark import verify
            print(verify());return
        try:
            if (HERE/'operation_ledger.json').exists():
                check_inputs();assert protected()==read(HERE/'audit/protected_before.json');seal()
                print('README:',HERE/'README.md');print('\n'.join(headline(read(HERE/'summary.json')['family'])));return
            preflight();check_inputs();tests('before')
            launch=protected();assert launch==read(HERE/'audit/protected_before.json'),'Protected roots changed before benchmark launch'
            write(HERE/'audit/launch_protected_check.json',dict(status='PASS',protected_files=len(launch['files'])))
            cohort=prepare_cohorts();jobs=[]
            for pool,rows in cohort['members'].items():
                for year in sorted({r['test_year'] for r in rows}):
                    jobs.append((pool,year,[r for r in rows if r['test_year']==year],cohort['configs'][pool],'primary'))
            jobs.extend(('EXP_G_cap',y,[r for r in cohort['members']['EXP_G_cap'] if r['test_year']==y],cohort['configs']['EXP_G_cap'],'price_martingale') for y in range(2004,2026))
            jobs.extend(('UNIVERSE_CONTINUITY',y,[],{},'continuity') for y in range(2004,2026))
            start=time.monotonic()
            # Finish one pool-fold for an empirical runtime estimate, then parallelise.
            first=jobs.pop(0);observed=run_pool_fold(*first)
            write(HERE/'audit/runtime_estimate.json',dict(first_pool_fold_seconds=observed['elapsed_seconds'],
                remaining_pool_folds=len(jobs),workers=args.workers,rough_remaining_seconds=observed['elapsed_seconds']*len(jobs)/args.workers,
                caveat='Pool-fold sizes, gate acceptance and continuity workloads differ. Estimate is approximate.'))
            print('Rough remaining hours:',round(observed['elapsed_seconds']*len(jobs)/args.workers/3600,2),flush=True)
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
                futures=[executor.submit(run_pool_fold,*job) for job in jobs]
                for future in as_completed(futures):future.result()
            report_all();tests('after');check_inputs()
            after=protected();assert after==read(HERE/'audit/protected_before.json'),'Protected roots changed'
            write(HERE/'audit/protected_after.json',after)
            finder=finder_inventory();before=read(HERE/'audit/finder_before.json')
            write(HERE/'audit/finder_after.json',dict(files=finder,changed=[k for k in sorted(set(before)|set(finder)) if before.get(k)!=finder.get(k)]))
            seal()
        except BaseException:
            write(HERE/f'audit/failure_{time.time_ns()}.json',dict(status='stopped',traceback=traceback.format_exc(),utc=utc()));raise
    print('README:',HERE/'README.md');print('\n'.join(headline(read(HERE/'summary.json')['family'])))

if __name__=='__main__':main()
