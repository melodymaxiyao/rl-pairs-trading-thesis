"""User-launched synthetic experiment; Tier 1 is sealed before Tier 3 starts."""
from sc_common import *
import argparse,subprocess,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp

def initialize():
    guard()

def tracked(function,job):
    COUNTS.clear()
    try:return function(job)
    finally:
        path=HERE/f'audit/operations_{os.getpid()}_{time.time_ns()}.json'
        write(path,dict(job=job,scope=os.environ.get('SC_SCOPE','production'),counts=dict(COUNTS),includes_completed_operations_before_interruption=True))

def parallel(jobs,function,workers):
    if workers==1:return [tracked(function,j) for j in jobs]
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize)
    futures=[]
    try:
        futures={executor.submit(tracked,function,j):i for i,j in enumerate(jobs)};result=[None]*len(jobs)
        for future in as_completed(futures):result[futures[future]]=future.result()
        return result
    except BaseException:
        for future in futures:future.cancel()
        # Stop only this executor's children; successful fit/world checkpoints
        # are immutable and survive. No cleanup deletes checkpoint data.
        import signal
        for process in list(executor._processes.values()):
            if process.is_alive():os.kill(process.pid,signal.SIGINT)
        raise
    finally:executor.shutdown(wait=True,cancel_futures=True)

def tests(phase,full=True):
    path=HERE/f'audit/tests_{phase}_{time.time_ns()}.log'
    args=[sys.executable,'-B',str(HERE/'scripts/test_entry.py'),'--workers','1']
    if full:args+=['--full']
    with path.open('x') as f:p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    assert p.returncode==0,('Tests failed; original log preserved',path)
    print('PASS tests_'+phase,str(path),flush=True)

def estimate(pilot,workers):
    from schedule import runtime_estimates
    t=pilot['timings'];total=sum(t.values())
    # Conservative component projection: fixed-gamma v3 inner fits can in fact
    # reuse prior candidate fits, and tuned-50 reuses its first ten seeds.
    v3fit=t['TD_v3']/50;tunedfit=t['TD_tuned']/10
    overhead=t['train_and_rules']+t['test_execution']
    tier1=30*total
    tier3=5*(3*overhead+2*t['oracle']+160*v3fit+120*tunedfit)
    rows=runtime_estimates({1:tier1,3:tier3},workers)
    path=HERE/f'audit/runtime_estimate_{time.time_ns()}.json'
    write(path,dict(pilot=pilot['job'],measured_seconds=total,component_seconds=t,estimates=rows,Tier2='deferred, not run',includes_calibration=False,warning='Ideal parallel scaling, excludes final verification and shared-machine contention; Tier3 extrapolated without executing it. Cached fits can reduce runtime.'))
    for r in rows:print('Estimated Tier',r['tier'],'workers',r['workers'],r['worker_setting'],'hours',round(r['estimated_parallel_seconds']/3600,2),'(pilot-based ideal scaling; final verification extra)',flush=True)

def final_seal():
    from sc_protection import snapshot,registered
    from sc_report import verify_tier
    registered();verify_tier(1);verify_tier(3)
    current,finder=snapshot();assert current==read(HERE/'audit/protected_before.json'),'Protected non-Finder workspace changed'
    write(HERE/'audit/protected_after.json',current);write(HERE/'audit/finder_after.json',finder)
    old=read(HERE/'audit/finder_before.json');write(HERE/'audit/finder_changes.json',dict(changed_paths=sorted(k for k in set(old)|set(finder) if old.get(k)!=finder.get(k))))
    # Count saved successful fits once, even if reused across cells/worlds.
    fits=[]
    for p in sorted((HERE/'runtime/fits').glob('*.pkl.ready.json')):
        info=read(p);v=cached(str(Path(info['file']).with_suffix('').with_suffix('')));fits.append(v['meta'])
    cal=read(HERE/'outputs/tables/calibration.json');worlds=[]
    from sc_report import tier_results
    for tier in [1,3]:worlds.extend(tier_results(tier))
    counts={}
    for v in cal['stage1']+cal['stage2']+worlds:
        for k,n in v.get('counts',{}).items():counts[k]=counts.get(k,0)+n
    actual={}
    for path in (HERE/'audit').glob('operations_*.json'):
        operation=read(path)
        if operation.get('scope')!='production':continue
        for k,n in operation['counts'].items():actual[k]=actual.get(k,0)+n
    ledger=dict(actual_completed_operations=actual,completed_checkpoint_operations=counts,TD_fits=len(fits),learning_steps=sum(f['learning_steps'] for f in fits),oracle_policy_evaluations=actual.get('oracle_policy_evaluations',0),synthetic_paths=actual.get('synthetic_paths',0),real_price_reads=0,real_fixture_context_reads=3,downloads=0,package_installs=0,real_data_trials=0,Tier2='deferred, not run',note='Actual completed operations include interrupted jobs via finally journals. Saved successful fits counted once separately. Partial C fits without a returned result are not claimed as completed.')
    write(HERE/'operation_ledger.json',ledger)
    write(HERE/'audit.json',dict(status='PASS',protected_match=True,input_seals=spec()['input_seals'],Tier2='deferred, not run',tier_order=[1,3],synthetic_only=True,real_price_reads=0))
    files={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json','pipeline.lock','.DS_Store'] and '.pending_' not in p.name}
    write(HERE/'manifest.json',dict(files=files));write(HERE/'complete.json',dict(status='complete',synthetic_only=True,Tier2='deferred, not run',manifest_sha256=sha(HERE/'manifest.json'),audit_sha256=sha(HERE/'audit.json'),operation_ledger_sha256=sha(HERE/'operation_ledger.json'),preregistration_sha256=sha(HERE/'preregistration.json')))

def main():
    from schedule import default_workers,world_jobs
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=default_workers());args=parser.parse_args();assert args.workers>=1
    guard()
    for directory in PT.iterdir():
        p=directory/'pipeline.lock'
        if directory!=HERE and directory.name!='turning_point_detection' and p.is_file():
            with p.open('rb') as f:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:raise RuntimeError('Another workspace pipeline is active: '+str(p))
                finally:fcntl.flock(f,fcntl.LOCK_UN)
    with (HERE/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'complete.json').exists():
            from verify_td_sample_complexity import verify
            verify();return
        stage='preflight'
        try:
            from sc_protection import preflight
            preflight();tests('before')
            from sc_kernel import kernel
            kernel()
            from sc_fixture import run as fixture
            fixture()
            from sc_calibration import run as calibration
            stage='calibration';cal=calibration(args.workers,parallel)
            if cal['status']!='calibrated':
                from sc_protection import snapshot
                current,finder=snapshot();assert current==read(HERE/'audit/protected_before.json')
                write(HERE/'audit/protected_after_calibration_stop.json',current)
                write(HERE/'audit/calibration_stop.json',dict(status='stopped_no_edge',reason='Coarse calibration chose w1=0',input_seals=spec()['input_seals'],protected_match=True,Tier1='not run',Tier2='deferred, not run',Tier3='not run'))
                print('Stopped: coarse calibration chose w1=0; source hashes unchanged; no learner tiers run. See calibration.json and audit/calibration_stop.json.',flush=True);return
            from sc_worlds import execute
            from sc_report import seal_tier,verify_tier,report
            stage='Tier1';pilot=tracked(execute,dict(tier=1,k=1,D=1,world=0,case='gaussian'));estimate(pilot,args.workers)
            parallel(world_jobs(1),execute,args.workers);seal_tier(1);verify_tier(1);print('Tier 1 sealed; Tier 2 deferred, not run. Starting Tier 3.',flush=True)
            stage='Tier3';parallel(world_jobs(3,tier1_verified=True),execute,args.workers);seal_tier(3);print('Tier 3 sealed.',flush=True)
            stage='report';report();tests('after');stage='seal';final_seal()
            from verify_td_sample_complexity import verify
            verify(check_current=False)
            print(str(HERE/'README.md'),flush=True)
            for line in read(HERE/'summary.json')['lines']:print(line,flush=True)
        except BaseException:
            write(HERE/f'audit/failure_{time.time_ns()}.json',dict(stage=stage,traceback=traceback.format_exc(),counts=dict(COUNTS)));raise

if __name__=='__main__':main()
