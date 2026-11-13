"""Main only: preflight, acceptance, pilot, resumable production, report and seal."""
from common import *
import argparse,subprocess,traceback,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
def tests(phase,full=False):
 log=HERE/f'audit/tests_{phase}_{time.time_ns()}.log'
 env=dict(os.environ,TUNED_FULL_TESTS='1' if full else '0',TUNED_SCOPE='tests_'+phase)
 with log.open('x') as f:p=subprocess.run([sys.executable,'-B',str(HERE/'scripts/test_entry.py'),str(HERE/'tests'),'-q','-p','no:cacheprovider','--basetemp',str(HERE/f'runtime/pytest_{time.time_ns()}')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
 assert p.returncode==0,('Tests failed; original log retained',log);print('PASS tests',phase,log,flush=True)
def lock():
 for d in PT.iterdir():
  p=d/'pipeline.lock'
  if d==HERE or not p.exists():continue
  with p.open('rb') as f:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise RuntimeError('Another workspace pipeline is active: '+str(p))
   finally:fcntl.flock(f,fcntl.LOCK_UN)
 h=(HERE/'pipeline.lock').open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def parallel(jobs,workers):
 from jobs import run_fold
 if workers==1:
  for j in jobs:run_fold(j)
  return
 ex=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))
 try:
  fs=[ex.submit(run_fold,j) for j in jobs]
  for f in as_completed(fs):f.result()
 except BaseException:
  for f in fs:f.cancel()
  for p in list(ex._processes.values()):p.terminate()
  ex.shutdown(wait=True,cancel_futures=True);raise
 else:ex.shutdown(wait=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 1)//2));parser.add_argument('mode',nargs='?',choices=['run','prepare','verify'],default='run');args=parser.parse_args();assert args.workers>=1
 if args.mode=='verify' or (HERE/'complete.json').exists():
  sys.path.insert(0,str(HERE));from verify_td_tuned import verify;return verify()
 h=lock()
 try:
  from protection import registered,preflight,finish
  registered();guard()
  if args.mode=='prepare':
   assert args.workers==1;tests('preparation');write(HERE/'audit/preparation_ready.json',dict(status='prepared_not_run',production_started=False,full_checks_deferred=spec()['deferred_to_main']));return
  preflight();tests('before',True)
  os.environ['TUNED_SCOPE']='acceptance'
  from fixtures import reproduction,workers,accounting
  reproduction();workers();accounting()
  os.environ['TUNED_SCOPE']='production'
  from jobs import run_fold
  first=('P_EXP',2004);alljobs=[first]+[(pool,y) for pool in POOLS for y in YEARS if (pool,y)!=first];pilot=run_fold(first)
  # No unmeasured speedup is asserted: first-fold extrapolation is labelled rough.
  remaining=[j for j in alljobs[1:] if cached(f'runtime/folds/{j[0]}_{j[1]}') is None]
  pilot_size=len(source_members(*alljobs[0]));serial=pilot['seconds']*sum(len(source_members(*j)) for j in remaining)/max(pilot_size,1)
  print(f'Measured pilot {pilot["seconds"]:.1f}s for {pilot_size} pairs. Rough remaining executor/fit ETA at workers={args.workers}: {serial/max(1,min(args.workers,len(remaining)))/3600:.2f}h; pair-count extrapolation, excludes fixed checks/report/accounting and contention.',flush=True)
  parallel(remaining,args.workers)
  from finish_report import run as report
  if not (HERE/'audit/report_ready.json').exists():report()
  tests('after',True);finish()
  sys.path.insert(0,str(HERE));from verify_td_tuned import verify;verify(check_current=False)
  print(str(HERE/'README.md'),flush=True)
  for line in read(HERE/'summary.json')['lines']:print(line,flush=True)
 except BaseException:
  write(HERE/f'audit/failure_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),utc=utc(),ledger=ledger()));raise
 finally:h.close()
def source_members(pool,y):
 from sources import members
 return members(pool,y)
if __name__=='__main__':main()
