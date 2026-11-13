"""User-launched production; every expensive path follows strict input gates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import argparse,signal,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from protection import main_gate
from jobs import design,identifier,world,worker
def stop_worker_init():signal.signal(signal.SIGINT,signal.SIG_IGN)
def seal_tier(tier,results,gates):
 from report import report
 result=report(results,tier,base=RUN/f'tier_{tier}');write(RUN/f'tier_{tier}/operation_ledger.json',ledger())
 files={}
 for p in sorted((RUN/f'tier_{tier}').rglob('*')):
  if p.is_file() and p.name not in ['audit.json','complete.json']:files[str(p.relative_to(RUN))]=sha(p)
 for job in design(tier):
  stem=f'runtime/worlds/{identifier(job)}'
  for suffix in ['.pkl.gz','.ready.json']:files[stem+suffix]=sha(RUN/(stem+suffix))
 for p in sorted((RUN/'runtime/seeds').rglob('*')):
  if p.is_file() and p.suffix in ['.gz','.json']:files[str(p.relative_to(RUN))]=sha(p)
 for gate in gates:
  files[gate['file']]=gate['sha256']
  receipt=json.loads((RUN/gate['file']).read_text())
  for v in receipt['original_verifiers']:files[v['log']]=v['sha256']
  proof=receipt['historical']['registered_conditionals'];files[proof['file']]=proof['sha256'];files[proof['metadata_log']]=proof['metadata_log_sha256']
  proof=receipt['historical']['ignored_conflict_copies'];files[proof['file']]=proof['sha256']
 if tier==3:
  from report_artifacts import share_final_report
  share_final_report(RUN/f'tier_{tier}')
  for p in sorted((RUN/'outputs').rglob('*')):
   if p.is_file():files[str(p.relative_to(RUN))]=sha(p)
  files['RESULTS.md']=sha(RUN/'RESULTS.md')
 gate_states=[json.loads((RUN/g['file']).read_text()) for g in gates];conditional=any(g.get('provenance_limitation') for g in gate_states);state='PASS_WITH_REGISTERED_CONDITIONALS' if conditional else 'PASS'
 audit=dict(status=state,tier=tier,code_registration_sha256=sha(RUN/'code_registration.json'),preregistration_sha256=sha(RUN/'preregistration.json'),artifacts=files,summary=result,all_inputs_before_and_after=state,gate_receipts=gates,synthetic_only=True,provenance_limitation='Exact registered paths use the standing conditional rule: unavailable/dataless hashes excluded, visible metadata changes logged, all materialized files require sealed SHA256 and size. No full fresh original-verifier pass is claimed where provenance remains unavailable.' if conditional else None)
 path=RUN/f'tier_{tier}/audit.json';write(path,audit);write(RUN/f'tier_{tier}/complete.json',dict(status='complete',tier=tier,audit_sha256=sha(path),worlds=len(results),real_data_trials=0))
 if tier==3:
  # Full task seal is created once. Partial tiers retain their own immutable seals.
  write(RUN/'audit.json',audit);write(RUN/'operation_ledger.json',ledger());write(RUN/'complete.json',dict(status='complete',tier=3,audit_sha256=sha(RUN/'audit.json'),operation_ledger_sha256=sha(RUN/'operation_ledger.json'),worlds=len(results),real_data_trials=0))
 print('README:',RUN/'README.md');print('RESULTS:',RUN/f'tier_{tier}/RESULTS.md')
 for line in result['summary']:print(line)
def completed_results(tier):
 # Receipts must already exist: cached() must not recreate one in this mode.
 results=[]
 for job in design(tier):
  stem=f'runtime/worlds/{identifier(job)}'
  assert all((RUN/(stem+suffix)).is_file() for suffix in ('.pkl.gz','.ready.json')),('Finalization requires every completed world and receipt',stem)
  value=cached(stem)
  assert value is not None and value['job']==job and value['test_mode'] is False,('Invalid completed world',stem)
  results.append(value)
 return results
def verify_existing_tier(tier):
 import subprocess
 subprocess.run([sys.executable,'-I','-B',str(RUN/'verify_ou_dp.py'),'--tier',str(tier)],check=True)
def finalize_only(tier,token):
 # Strict recovery path: this function never calls world(), worker(), or a pool.
 from disk_policy import require_space
 require_space('checkpoint');results=completed_results(tier)
 ledger_before=ledger();previous=None
 for current in range(1,tier+1):
  if (RUN/f'tier_{current}/complete.json').exists():
   verify_existing_tier(current);previous=current
  else:
   assert current==tier and previous==tier-1,('Finalization requires sealed preceding tiers',current)
 if previous==tier:print('Requested tier already sealed and verified; no computation.');return
 # The preceding tier's authenticated after-gate preceded these completed fits.
 # Retain it as the before receipt; obtain a fresh full historical after-gate.
 previous_audit=RUN/f'tier_{previous}/audit.json'
 before=json.loads(previous_audit.read_text())['gate_receipts'][-1]
 assert sha(RUN/before['file'])==before['sha256']
 write(RUN/f'audit/finalization_resume_{token}.json',dict(mode='finalize_only',worlds=len(results),new_fits=0,prior_tier_audit=str(previous_audit.relative_to(RUN)),prior_tier_audit_sha256=sha(previous_audit),authenticated_before_gate=before,production_ledger_before=ledger_before))
 after=main_gate('after_tier_'+str(tier)+'_finalize_'+token,phase='checkpoint')
 assert ledger()==ledger_before,('Operation ledger changed during finalization',ledger())
 seal_tier(tier,results,[before,after]);checkpoint_guard();verify_existing_tier(tier)
 print('Finalization complete; no new fits, downloads or package changes.',flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 2)//2));parser.add_argument('--stop-after-tier',type=int,choices=[1,2,3],default=3);parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--finalize-only',action='store_true',help='Require all requested checkpoints; verify and seal without starting computation');args=parser.parse_args();assert args.workers>=1 and not (args.finalize_only and args.preflight_only)
 from verify_code import verify_code
 verify_code();guard();os.environ['OU_SCOPE']='production';token=str(time.time_ns());stop=RUN/f'runtime/STOP_{token}';os.environ['OU_STOP_FILE']=str(stop)
 if args.preflight_only:os.environ['OU_PREFLIGHT_ONLY']='1'
 def request_stop(signum,frame):
  publish(stop,'checkpoint stop\n');print('Stop requested; waiting for the next saved checkpoint.',flush=True)
 signal.signal(signal.SIGINT,request_stop)
 with lock():
  if args.finalize_only:
   finalize_only(args.stop_after_tier,token);return
  before=main_gate('before_'+token)
  if args.preflight_only:
   state=json.loads((RUN/before['file']).read_text())['status'];print('Full main preflight',state,'; no production computation');return
  jobs=design(args.stop_after_tier);pilot_job=jobs[0];begin=time.monotonic();pilot=world(pilot_job);elapsed=time.monotonic()-begin
  remaining=sum(cached(f'runtime/worlds/{identifier(j)}') is None for j in jobs)
  estimate=dict(first_world_seconds=pilot['seconds'],this_invocation_pilot_seconds=elapsed,remaining_worlds=remaining,workers=args.workers,sequential_hours=pilot['seconds']*remaining/3600,ideal_parallel_hours=pilot['seconds']*remaining/args.workers/3600,limitation='first-cell timing only: larger P, half-life, convergence and Monte Carlo arms change cost; ideal scaling is not a promise',pilot_reused=elapsed<pilot['seconds']/2)
  write(RUN/f'audit/runtime_estimate_{token}.json',estimate);print('Measured pilot estimate:',json.dumps(estimate),flush=True)
  # One tier at a time; save seed fits before checkpoint guards and world results before reporting.
  for tier in range(1,args.stop_after_tier+1):
   if (RUN/f'tier_{tier}/complete.json').exists():
    import subprocess
    subprocess.run([sys.executable,'-I','-B',str(RUN/'verify_ou_dp.py'),'--tier',str(tier)],check=True);print('Already sealed; verified Tier',tier,flush=True);continue
   pending=[j for j in jobs if j['tier']==tier and cached(f'runtime/worlds/{identifier(j)}') is None]
   pool=ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=stop_worker_init);futures={}
   try:
    while pending or futures:
     while pending and len(futures)<args.workers:
      checkpoint_guard();j=pending.pop(0);futures[pool.submit(worker,j)]=j
     done,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED) if futures else (set(),set())
     for f in done:
      j=futures.pop(f);r=f.result();print('Completed',identifier(j),'seconds',round(r['seconds'],1),flush=True);checkpoint_guard()
   except BaseException:
    publish(stop,'checkpoint stop\n')
    for f in futures:f.cancel()
    raise
   finally:pool.shutdown(wait=True,cancel_futures=True)
   results=[cached(f'runtime/worlds/{identifier(j)}') for j in design(tier)];assert all(r is not None and not r['test_mode'] for r in results)
   after=main_gate('after_tier_'+str(tier)+'_'+token,phase='checkpoint');seal_tier(tier,results,[before,after]);checkpoint_guard()
if __name__=='__main__':
 try:main()
 except (KeyboardInterrupt,Pause) as e:print('Paused cleanly; saved checkpoints retained. Resume with the same command.',str(e),flush=True);sys.exit(130)
 except (AssertionError,RuntimeError,OSError) as e:
  import traceback
  from registered_scope import REGISTRY,REGISTRY_SHA256
  path=RUN/f'audit/stopped_{time.time_ns()}.json';write(path,dict(status='STOPPED',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),source_repairs=0,authorized_conditional_registry=str(REGISTRY),authorized_conditional_registry_sha256=REGISTRY_SHA256,unapproved_hash_exclusions=0,package_installs=0))
  print('STOPPED:',str(e),flush=True);print('Details:',path,flush=True);sys.exit(2)
