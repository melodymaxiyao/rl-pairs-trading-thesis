"""Low-priority resumable tier runner. User launch is the real-fit authorization."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import argparse,signal,traceback,contextlib
from experiment import work,wanted,done_cell
from integrity import code,input_gate,historical_gate

@contextlib.contextmanager
def single_run():
 path=RUN/'runtime/runner.lock'
 with path.open('a+') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('This task is already running')
  yield

def stop(signum,frame):
 Path(os.environ['COINT_STOP']).touch(exist_ok=True);print('Stop requested; saving the current seed/cell before pausing.',flush=True)

def tier_done(tier):
 return all(done_cell(kind,y,g,c,l) for kind in ['RP','RF'] for y in range(2004,2026) for c,g,ls in wanted(tier) for l in ls)

def pilot(workers):
 p=RUN/'audit/pilot.json'
 if p.exists():return read(p)
 start=time.monotonic();before=budget_guard(True);rows=[work((kind,2004,1)) for kind in ['RP','RF']];after=budget_guard(True);seconds=time.monotonic()-start;delta=max(0,after['logical']-before['logical']);snapshot=max([q.stat().st_size for q in (RUN/'audit').glob('protected_*.jsonl.gz')]+[64*2**20]);projected=before['logical']+delta*22*4.5+snapshot*6+256*2**20
 record=dict(status='PASS' if projected<=6*2**30 else 'BLOCKED',fold=2004,retained_ordinary_fold=True,measured_seconds=seconds,measured_bytes=delta,fold_tasks=rows,projected_Tier1_seconds=seconds*22/max(1,workers),projected_all_tiers_seconds_upper=seconds*22*4.5/max(1,workers),projected_peak_bytes=projected,projection='conservative scaling of two residual tasks and all planned stages; includes compact final outputs, checkpoint peak, duplicate/current reports, six gzip snapshots; actual guard remains binding')
 write(p,record);print('Pilot: measured',round(seconds/60,1),'minutes,',round(delta/2**20,1),'MiB; projected Tier1',round(record['projected_Tier1_seconds']/3600,2),'hours; all-tier peak',round(projected/2**30,3),'GiB',flush=True)
 if record['status']!='PASS':raise Pause('Pilot projection exceeds6GiB; no further folds launched')
 return record

def discard_checkpoints(tier,delete=True):
 receipt=RUN/f'tier_{tier}/checkpoint_discard.json'
 if receipt.exists():plan=read(receipt)
 else:
  files={}
  for p in sorted((RUN/'runtime/checkpoints').glob('*.pkl.gz')):
   ready=RUN/(str(p.relative_to(RUN))[:-7]+'.ready.json');assert sha(p)==read(ready)['sha256'];files[str(p.relative_to(RUN))]=dict(sha256=sha(p),size=p.stat().st_size,ready=str(ready.relative_to(RUN)),ready_sha256=sha(ready))
  plan=dict(authorization='User task: delete own compressed checkpoint bodies after tier sealing; final scientific models and receipts retained',files=files,tier=tier);write(receipt,plan)
 if not delete:return plan
 assert (RUN/f'tier_{tier}/complete.json').exists(),'Checkpoint bodies retained until tier sealing'
 manifest=read(RUN/f'tier_{tier}/manifest.json')['files'];assert manifest[str(receipt.relative_to(RUN))]==sha(receipt)
 for rel,r in plan['files'].items():
  assert sha(RUN/r['ready'])==r['ready_sha256']
  p=RUN/rel;assert p.is_relative_to(RUN/'runtime/checkpoints')
  if p.exists():assert sha(p)==r['sha256'] and p.stat().st_size==r['size'];p.unlink()
 return plan

def finalize(tier):
 root=RUN/f'tier_{tier}'
 if (root/'complete.json').exists():
  from verify import verify_tier
  verify_tier(tier);discard_checkpoints(tier);return read(root/'summary.json')['lines']
 from report import report
 lines=report(tier);write_tier_ledger(tier);discard_checkpoints(tier,False);code();input_gate(f'after_tier{tier}');historical_gate('after_tier'+str(tier)+'_'+str(time.time_ns()));checkpoint();files={}
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.name not in ['manifest.json','complete.json']:files[str(p.relative_to(RUN))]=sha(p)
 for folder in ['outputs/cells','outputs/models']:
  for p in sorted((RUN/folder).rglob('*')):
   if p.is_file():files[str(p.relative_to(RUN))]=sha(p)
 for previous in range(1,tier):
  for p in sorted((RUN/f'tier_{previous}').rglob('*')):
   if p.is_file():files[str(p.relative_to(RUN))]=sha(p)
 for p in sorted((RUN/'runtime/checkpoints').glob('*.ready.json')):files[str(p.relative_to(RUN))]=sha(p)
 for p in sorted((RUN/'audit').rglob('*')):
  if p.is_file() and not p.name.startswith('counts_') and '.pending_' not in p.name:files[str(p.relative_to(RUN))]=sha(p)
 write(root/'manifest.json',dict(files=files));write(root/'complete.json',dict(status='complete',tier=tier,protocol_complete=tier==3,real_trials=572,new_real_trials=20,manifest_sha256=sha(root/'manifest.json'),preregistration_sha256=sha(RUN/'preregistration.json'),code_registration_sha256=sha(RUN/'code_registration.json')))
 discard_checkpoints(tier)
 # Current view is replaceable; immutable tier records remain the scientific seal.
 for p in sorted(root.iterdir()):
  if p.name in ['complete.json','manifest.json','checkpoint_discard.json'] or not p.is_file():continue
  dst=RUN/p.name if p.name in ['README.md','summary.json','operation_ledger.json','missing_price_log.csv'] else RUN/'outputs/tables'/p.name
  dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.pending_view');tmp.write_bytes(p.read_bytes());os.replace(tmp,dst)
 replace(RUN/'complete.json',read(root/'complete.json'));replace(RUN/'status.json',dict(status='SEALED_TIER',tier=tier,protocol_complete=tier==3));return lines

def write_tier_ledger(tier):
 from collections import Counter
 counts=Counter()
 for p in (RUN/'audit').glob('counts_production_*.jsonl'):
  for line in p.read_text().splitlines():
   key,n=json.loads(line);counts[key]+=n
 ledger=dict(phase='sealed_tier',tier=tier,counts=dict(counts),real_trials_before=552,new_registered_strategy_cells=20,cumulative_real_trials=572,downloads=0,package_changes=0,scientific_source_writes=0,trial_convention='Full20-cell family registered conservatively; seeds/folds/nulls/replays do not add trials',fit_count_fields='real_kernel_seed_fits counts attempts; real_TD_seed_fits counts successfully checkpointed new seeds; do not sum these two fields')
 write(RUN/f'tier_{tier}/operation_ledger.json',ledger)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,choices=[1,2],default=1);parser.add_argument('--stop-after-tier',type=int,choices=[1,2,3],default=1);args=parser.parse_args();guard();os.environ['OU_SCOPE']='production';os.environ['COINT_STOP']=str(RUN/'runtime'/('STOP_'+str(os.getpid())));signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 with single_run():
  space(True);no_other_task();code();print('Tier1 started: verifying environment and sealed upstream inputs.',flush=True);input_gate('launch');print('Upstream inputs passed; starting full historical integrity check.',flush=True);historical_gate('launch_'+str(time.time_ns()));write(RUN/'audit/real_protocol_started.json',dict(authorization='User explicitly instructed assistant to startTier1 after implementing the missing-price rule; see audit/Tier1_launch_authorization.json',preregistration_sha256=sha(RUN/'preregistration.json'),real_trials_before=552,registered_new_family=20,cumulative_trials=572));replace(RUN/'status.json',dict(status='running',requested_tier=args.stop_after_tier,real_trials=572));replace(RUN/'operation_ledger.json',dict(phase='running',real_trials=572,new_registered_strategy_cells=20,scientific_source_writes=0,downloads=0,package_changes=0));pilot(args.workers)
  from concurrent.futures import ProcessPoolExecutor,as_completed
  import multiprocessing
  for tier in range(1,args.stop_after_tier+1):
   if not (RUN/f'tier_{tier}/complete.json').exists():
    tasks=[(kind,y,tier) for y in range(2004,2026) for kind in ['RP','RF'] if not all(done_cell(kind,y,g,c,l) for c,g,ls in wanted(tier) for l in ls)]
    if args.workers==1:
     for task in tasks:checkpoint();work(task)
    else:
     with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
      futures={pool.submit(work,t):t for t in tasks}
      try:
       for f in as_completed(futures):print('Fold finished',f.result(),flush=True);checkpoint()
      except BaseException:
       Path(os.environ['COINT_STOP']).touch(exist_ok=True)
       for f in futures:f.cancel()
       raise
   assert tier_done(tier);lines=finalize(tier)
   from verify import verify_tier
   verify_tier(tier);print('README:',RUN/'README.md',flush=True)
   for line in lines:print(line,flush=True)
if __name__=='__main__':
 try:main()
 except Pause as exc:
  try:write(event('paused'),dict(reason=str(exc),traceback=traceback.format_exc()))
  except BaseException:pass
  print('Paused cleanly; completed seeds/cells retained. Resume the same command.',str(exc),flush=True);sys.exit(130)
 except Exception as exc:
  path=event('stopped')
  try:write(path,dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc()))
  except BaseException:path='Audit write could not be completed; console reason retained'
  print('STOPPED:',type(exc).__name__,str(exc),'Details:',path,flush=True);sys.exit(1)
