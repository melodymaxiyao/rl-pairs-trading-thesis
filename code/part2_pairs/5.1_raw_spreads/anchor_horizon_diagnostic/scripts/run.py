from aio import *
from preflight import preflight
from verification import check_inputs,finder_inventory
from protection import protected
from jobs import fold
from ledger import ledger
import subprocess,argparse,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed

def tests(phase):
    path=HERE/f'audit/tests_{phase}_{time.time_ns()}.log';env=dict(os.environ,AH_SCOPE='tests_'+phase)
    with path.open('x') as f:
        code=subprocess.run([sys.executable,'-B',str(HERE/'scripts/test_entry.py'),'-q',str(HERE/'tests')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        f.write('\nTest exit code: '+str(code)+'\n')
    assert code==0,('Tests failed; preserved',path,code);return path

def protected_check(label):
    after=protected();write(HERE/f'audit/protected_{label}.json',after)
    assert after==read(HERE/'audit/protected_before.json'),'Protected workspace changed'

def main():
    guard();parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1));args=parser.parse_args()
    with (HERE/'pipeline.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);registered();assert checkpoint(HERE/'audit/preparation_ready.json')
        if (HERE/'complete.json').exists():
            subprocess.run([sys.executable,'-B',str(HERE/'verify_anchor_horizon.py')],check=True);return
        preflight();check_inputs();protected_check('run_start_'+str(time.time_ns()));before=tests('before')
        with ProcessPoolExecutor(max_workers=max(1,args.workers),mp_context=multiprocessing.get_context('spawn')) as executor:
            futures=[executor.submit(fold,y) for y in YEARS]
            for future in as_completed(futures):future.result()
        from report import report_all
        summary=report_all();after=tests('after');inputs=check_inputs();registered();protected_check('after')
        write(HERE/'audit/finder_after.json',finder_inventory());write(HERE/'operation_ledger.json',ledger())
        write(HERE/'audit.json',dict(status='PASS',diagnostic_only=True,train_only=True,test_year_numeric_rows=0,protected_match=True,source_hashes_match=True,
            input_info=inputs,tests_before=str(before.relative_to(HERE)),tests_after=str(after.relative_to(HERE)),summary=summary))
        files={str(p.relative_to(HERE)):fp(p) for p in HERE.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json','.DS_Store']}
        write(HERE/'manifest.json',dict(files=files))
        subprocess.run([sys.executable,'-B',str(HERE/'verify_anchor_horizon.py'),'--unsealed'],check=True)
        write(HERE/'complete.json',dict(status='complete',sealed_at_utc=utc(),diagnostic_only=True,manifest_sha256=sha(HERE/'manifest.json'),audit_sha256=sha(HERE/'audit.json'),ledger_sha256=sha(HERE/'operation_ledger.json')))
        subprocess.run([sys.executable,'-B',str(HERE/'verify_anchor_horizon.py')],check=True)
        print(HERE/'README.md',flush=True)
        for line in summary:print(line,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(HERE/f'audit/run_failure_{time.time_ns()}.json',dict(status='stopped',traceback=traceback.format_exc()));raise
