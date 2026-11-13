"""User-launched real statistics; low priority, bounded checkpointed jobs, no RL."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import argparse,signal,multiprocessing as mp,subprocess
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from protection import main_gate
from disk_policy import require_space
from inputs import build_panel,DATA_STEM
from jobs import design,identifier,work
from streams import ParamStream

def worker_init():signal.signal(signal.SIGINT,signal.SIG_IGN)
def verify_complete():subprocess.run([sys.executable,'-I','-B',str(RUN/'verify_residual_ou.py')],check=True)
def seal(before,after,result):
 measured=ledger();measured.update(new_real_strategy_trials=spec()['new_primary_strategies']+spec()['new_optional_RE_strategies'],cumulative_real_strategy_trials=spec()['cumulative_trials_after'],RL_fits=0);write(RUN/'operation_ledger.json',measured)
 paths=[RUN/'README.md',RUN/'RESULTS.md',RUN/'residual_transfer.json',RUN/'operation_ledger.json']
 for folder in ['outputs','runtime/jobs','runtime/aggregates','runtime/data','runtime/streams']:
  paths.extend(p for p in (RUN/folder).rglob('*') if p.is_file())
 for receipt in [before,after]:
  paths.append(RUN/receipt['file']);g=json.loads((RUN/receipt['file']).read_text())
  paths.extend(RUN/v['log'] for v in g['original_verifiers']);proof=g['historical']['registered_conditionals'];paths.extend([RUN/proof['file'],RUN/proof['metadata_log']]);paths.append(RUN/g['historical']['ignored_conflict_copies']['file'])
 audit=dict(status='PASS_WITH_REGISTERED_CONDITIONALS',code_registration_sha256=sha(RUN/'code_registration.json'),preregistration_sha256=sha(RUN/'preregistration.json'),artifacts={str(p.relative_to(RUN)):sha(p) for p in sorted(set(paths))},gate_receipts=[before,after],summary=result,limitation='Exact registered unavailable/dataless provenance hashes conditionally excluded; available computation inputs verified by sealed SHA256 and size; no fresh full original-verifier pass claimed for unavailable provenance',RL_fits=0,downloads=0,package_installs=0)
 write(RUN/'audit.json',audit);write(RUN/'complete.json',dict(status='complete',audit_sha256=sha(RUN/'audit.json'),operation_ledger_sha256=sha(RUN/'operation_ledger.json'),cumulative_trials=spec()['cumulative_trials_after'],jobs=len(design()),RL_fits=0));verify_complete()
 print('README:',RUN/'README.md');print('RESULTS:',RUN/'RESULTS.md')
 for line in result['summary']:print(line)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 2)//2));parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--finalize-only',action='store_true');args=parser.parse_args();assert args.workers>=1 and not(args.preflight_only and args.finalize_only)
 from verify_code import verify_code
 verify_code();guard();os.environ['ROU_SCOPE']='production';token=str(time.time_ns());stop=RUN/f'runtime/STOP_{token}';os.environ['ROU_STOP_FILE']=str(stop)
 def pause(signum,frame):
  if not stop.exists():publish(stop,'pause after saved job\n')
  print('Stop requested: completing and saving in-flight jobs.',flush=True)
 signal.signal(signal.SIGINT,pause)
 with lock():
  if (RUN/'complete.json').exists():verify_complete();print('Already complete and verified; no computation.');return
  jobs=design()
  if args.finalize_only:assert all((RUN/f'runtime/jobs/{identifier(j)}.ready.json').is_file() for j in jobs),'Finalization requires every completed residual-fitting job'
  if args.preflight_only:os.environ['ROU_PREFLIGHT_ONLY']='1'
  before=main_gate('before_'+token,phase='checkpoint' if args.finalize_only else 'preflight')
  if args.preflight_only:print('Full preflight PASS_WITH_REGISTERED_CONDITIONALS; no real computation.');return
  if args.finalize_only:assert cached(DATA_STEM) is not None
  else:build_panel();checkpoint_guard()
  stream=ParamStream();existing={identifier(j) for j in jobs if (RUN/f'runtime/jobs/{identifier(j)}.pkl.gz').exists()};pending=[j for j in jobs if identifier(j) not in existing];completed=set(existing);next_index=0
  if args.finalize_only:assert not pending
  pool=None;futures={}
  try:
   if pending:pool=ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init)
   while next_index<len(jobs):
    while next_index<len(jobs) and identifier(jobs[next_index]) in completed:
     j=jobs[next_index];r=cached('runtime/jobs/'+identifier(j));assert r['job']==j;stream.append(identifier(j),r['parameter_shard']);next_index+=1;print('Committed',identifier(j),next_index,'/',len(jobs),flush=True);checkpoint_guard()
    while pending and len(futures)<args.workers:
     checkpoint_guard();j=pending.pop(0);futures[pool.submit(work,j)]=j
    if futures:
     done,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
     for future in done:
      j=futures.pop(future);future.result();completed.add(identifier(j))
    elif next_index<len(jobs):raise RuntimeError('Checkpoint schedule incomplete')
  except BaseException:
   if not stop.exists():publish(stop,'pause after saved job\n')
   for f in futures:f.cancel()
   raise
  finally:
   if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
  assert set(stream.done)=={identifier(j) for j in jobs};stream.publish();checkpoint_guard()
  from aggregate import aggregate_all
  from report import render
  manifests=aggregate_all(jobs);result=render(manifests);checkpoint_guard();after=main_gate('after_'+token,phase='checkpoint');seal(before,after,result)
if __name__=='__main__':
 try:main()
 except (Pause,KeyboardInterrupt) as exc:print('Paused cleanly; completed jobs retained. Resume the same command.',str(exc),flush=True);sys.exit(130)
 except Exception as exc:
  import traceback
  p=event_path('stopped');write(p,dict(status='STOPPED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),downloads=0,package_installs=0,source_repairs=0));print('STOPPED:',type(exc).__name__,str(exc) or '(see traceback in audit record)','Details:',p,flush=True);sys.exit(2)
