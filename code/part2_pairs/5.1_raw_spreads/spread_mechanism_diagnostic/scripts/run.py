"""User entry point: sealed preparation -> resumable folds -> report -> seal."""
from mio import *
from preflight import preflight
from verification import check_inputs,finder_inventory
from protection import protected
from jobs import run_fold
from ledger import ledger
import argparse,subprocess
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing

def tests(phase):
    log=HERE/f'audit/tests_{phase}_{time.time_ns()}.log'
    env=dict(os.environ,MECH_SCOPE='tests_'+phase)
    args=[sys.executable,'-B',str(HERE/'scripts/test_entry.py'),'-q',str(HERE/'tests')]
    with log.open('x') as f:code=subprocess.run(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,('Tests failed; preserved log',log)
    return log

def compare_protected(name):
    expected=read(HERE/'audit/protected_before.json');current=protected()
    write(HERE/f'audit/protected_{name}.json',current)
    assert current==expected,'Protected workspace changed; stop and inspect before/after hashes'
    return current

def main():
    guard();parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1));args=parser.parse_args()
    with (HERE/'pipeline.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registered();assert checkpoint(HERE/'audit/preparation_ready_final.json'),'Code preparation has not passed'
        if (HERE/'complete.json').exists():
            subprocess.run([sys.executable,'-B',str(HERE/'verify_mechanism.py')],check=True);return
        preflight();check_inputs();compare_protected('user_start_'+str(time.time_ns()));before=tests('before')
        with ProcessPoolExecutor(max_workers=max(1,args.workers),mp_context=multiprocessing.get_context('spawn')) as ex:
            futures=[ex.submit(run_fold,y) for y in YEARS]
            for future in as_completed(futures):future.result()
        from report import report_all
        summary=report_all();after=tests('after');check_inputs();registered()
        compare_protected('after');write(HERE/'audit/finder_after.json',finder_inventory())
        write(HERE/'operation_ledger.json',ledger())
        write(HERE/'audit.json',dict(status='PASS',diagnostic_only=True,test_year_numeric_rows=0,
            input_info=check_inputs(),tests_before=str(before.relative_to(HERE)),tests_after=str(after.relative_to(HERE)),
            protected_match=True,source_hashes_match=True,summary=summary))
        # Full immutable inventory is created last, before complete.json.
        files={str(p.relative_to(HERE)):fp(p) for p in HERE.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json','.DS_Store']}
        write(HERE/'manifest.json',dict(files=files))
        subprocess.run([sys.executable,'-B',str(HERE/'verify_mechanism.py'),'--unsealed'],check=True)
        write(HERE/'complete.json',dict(status='complete',sealed_at_utc=utc(),diagnostic_only=True,manifest_sha256=sha(HERE/'manifest.json'),
            audit_sha256=sha(HERE/'audit.json'),ledger_sha256=sha(HERE/'operation_ledger.json')))
        subprocess.run([sys.executable,'-B',str(HERE/'verify_mechanism.py')],check=True)
        print(HERE/'README.md',flush=True)
        for line in summary:print(line,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(HERE/f'audit/run_failure_{time.time_ns()}.json',dict(status='stopped',traceback=traceback.format_exc()));raise
