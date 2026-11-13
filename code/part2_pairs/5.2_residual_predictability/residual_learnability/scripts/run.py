"""Explicit main entry; importing this module never starts a run."""
from common import *
import argparse,subprocess
from protection import preflight,locks,registered,seal

def tests(phase,full=False):
 args=[sys.executable,'-B',str(HERE/'scripts/test_entry.py'),'--workers','1']+(['--full'] if full else [])
 path=HERE/f'audit/tests_{phase}_{time.time_ns()}.log';env=dict(os.environ,RESIDUAL_SCOPE='tests_'+phase)
 with path.open('x') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
 assert r.returncode==0,('Tests failed; original log preserved',path)
 print('PASS tests',phase,path,flush=True)

def checkpoint_phase(name,paths):
 p=HERE/f'audit/{name}_sealed.json';files={str(v.relative_to(HERE)):sha(v) for v in paths};write(p,dict(status='complete',phase=name,files=files,preregistration_sha256=sha(HERE/'preregistration.json')))

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 1)//2));args=p.parse_args();assert args.workers>=1
 if (HERE/'complete.json').exists():
  from verify_residual_learnability import verify
  verify(check_current=False);return
 stage='preflight';configure();os.environ['RESIDUAL_WORKERS']=str(args.workers)
 with locks():
  guard()
  try:
   preflight();tests('before',True)
   from reproduction import fixture
   stage='single_L0_N2_reproduction_fixture';fixture()
   from jobs import train_fold,hindsight_fold,sample_plan
   from parallel import execute
   stage='train_statistics';sample_plan()
   pending=[y for y in YEARS if any(cached(f'runtime/pool_folds/{pool}_{y}') is None for pool in POOLS)]
   if pending:
    pilot=train_fold(pending[0]);remaining=pending[1:];seconds=pilot['seconds'];eta=seconds*len(remaining)/args.workers
    print(f'Measured first fold: {seconds:.1f}s across 3 pools ({seconds/3:.1f}s/pool-fold equivalent). Train-statistics remaining estimate at --workers {args.workers}: {eta/3600:.2f}h; full hashes, secondary/report/power/hindsight add time. Continues.',flush=True)
    write(HERE/f'audit/runtime_estimate_{time.time_ns()}.json',dict(first_fold=pilot,workers=args.workers,remaining_fold_count=len(remaining),remaining_train_seconds_estimate=eta,estimate_includes='train statistics only; price overlap, report, power and hashes not extrapolated'))
    for _ in execute(train_fold,remaining,args.workers):pass
   checkpoint_phase('train_statistics',[HERE/f'runtime/pool_folds/{pool}_{y}.ready.json' for pool in POOLS for y in YEARS])
   # Freeze the train-only screen decisions before any hindsight test prices.
   stage='freeze_screens'
   from report import compact,long_screens
   import pandas as pd
   if not (HERE/'audit/screens_frozen.json').exists():
    frames=[compact(cached(f'runtime/pool_folds/{pool}_{y}')['frame']) for pool in POOLS for y in YEARS];screens=long_screens(pd.concat(frames,ignore_index=True));csv_new(TABLES/'screens.csv',screens);write(HERE/'audit/screens_frozen.json',dict(status='frozen_before_hindsight',file='outputs/tables/screens.csv',sha256=sha(TABLES/'screens.csv')));del frames,screens
   else:assert sha(TABLES/'screens.csv')==read(HERE/'audit/screens_frozen.json')['sha256']
   stage='synthetic_power'
   import power
   jobs=[(k,w,o) for k,w in power.grid() for o in range(0,20000,100) if cached(f'runtime/power_chunks/{k}/{o:05d}') is None]
   for _ in execute(power.chunk,jobs,args.workers):pass
   checkpoint_phase('power',[HERE/f'runtime/power_chunks/{k}/{o:05d}.ready.json' for k,w in power.grid() for o in range(0,20000,100)])
   stage='hindsight'
   for _ in execute(hindsight_fold,[y for y in YEARS if cached(f'runtime/hindsight/{y}') is None],args.workers):pass
   checkpoint_phase('hindsight',[HERE/f'runtime/hindsight/{y}.ready.json' for y in YEARS])
   stage='report'
   from report import final_report
   result=final_report();stage='tests_after';tests('after',True);stage='seal';seal()
   print('PASS sealed residual learnability diagnostic; no RL fits; one L0 reproduction fixture only.',flush=True);print(HERE/'README.md',flush=True)
   for line in result['summary']:print(line,flush=True)
  except BaseException:
   import traceback
   write(HERE/f'audit/failure_{stage}_{time.time_ns()}.json',dict(stage=stage,status='stopped',traceback=traceback.format_exc(),utc=utc()));raise
if __name__=='__main__':
 sys.path.insert(0,str(HERE));main()
