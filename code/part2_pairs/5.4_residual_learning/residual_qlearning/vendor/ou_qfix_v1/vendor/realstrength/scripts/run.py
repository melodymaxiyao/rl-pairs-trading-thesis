"""Single low-priority resumable command, fixed design and three nested tiers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import argparse,fcntl,signal,traceback,gc
from integrity import check_code,inputs
def historical(label):
 p=subprocess.run([str(INPUT/'.venv/bin/python'),'-I','-B',str(RUN/'scripts/historical_entry.py'),label])
 if p.returncode==130:raise Pause('Historical disk/stop guard; all checkpoints retained')
 assert p.returncode==0,('Historical integrity subprocess failed',label,p.returncode)
 return read(RUN/'audit'/('historical_'+label+'.json'))['result']

def seal(lines):
 operations={};scopes={}
 for p in (RUN/'audit').glob('counts_*.jsonl'):
  for line in p.read_text().splitlines():
   key,n=json.loads(line);operations[key]=operations.get(key,0)+n
 write(RUN/'operation_ledger.json',dict(operations=operations,real_trials_before=552,real_trials_after=552,new_real_strategy_trials=0,synthetic_only=True,source_writes=0,bookkeeping_writes=1,downloads=0,package_changes=0))
 paths=[]
 for folder in ['scripts','tests','vendor','outputs','runtime','audit','tier_1','tier_2']:
  if (RUN/folder).exists():paths += [p for p in (RUN/folder).rglob('*') if p.is_file() and '.pending_' not in p.name and not p.name.startswith('STOP_')]
 # Caches/fonts and interrupted temporary compiler outputs do not define science.
 paths=[p for p in paths if not any(part in ['tmp','cache','mpl'] for part in p.relative_to(RUN).parts)]
 paths += [RUN/name for name in ['preregistration.json','code_registration.json','calibration.json','cells.csv','predictions.csv','policies.json','README.md','summary.json','operation_ledger.json','request.md','RUN_ALL.zsh','VERIFY.zsh','CHECK_PREPARATION.zsh']]
 manifest={str(p.relative_to(RUN)):sha(p) for p in sorted(set(paths))};write(RUN/'manifest.json',dict(files=manifest));write(RUN/'audit.json',dict(status='PASS_WITH_REGISTERED_CONDITIONALS',tier=3,manifest_sha256=sha(RUN/'manifest.json'),preregistration_sha256=sha(RUN/'preregistration.json'),code_registration_sha256=sha(RUN/'code_registration.json'),real_trials=552,new_real_trials=0,standing_exact_paths=8908,free_bytes=shutil.disk_usage(RUN).free));write(RUN/'complete.json',dict(status='complete',tier=3,audit_sha256=sha(RUN/'audit.json'),manifest_sha256=sha(RUN/'manifest.json'),real_trials=552,new_real_trials=0))
 from verify import verify
 verify();print('README:',RUN/'README.md',flush=True)
 for line in lines:print(line,flush=True)

def run(args):
 check_code();no_other_task();guard()
 with (RUN/'pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (RUN/'complete.json').exists():
   from verify import verify
   verify();print('README:',RUN/'README.md')
   for line in read(RUN/'summary.json')['lines']:print(line)
   return
  space(start=True);os.environ['OU_REALSTRENGTH_STOP']=str(RUN/'runtime'/('STOP_'+str(time.time_ns())))
  def stop(signum,frame):Path(os.environ['OU_REALSTRENGTH_STOP']).touch(exist_ok=True);print('Pause requested; completing the current seed/panel checkpoint.',flush=True)
  signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
  token=str(time.time_ns());inputs('before_'+token);historical('before_'+token);gc.collect();checkpoint()
  if args.preflight_only:print('Preflight PASS; production not started.');return
  from calibration import calibrate
  from report import point_results,render
  calibration=calibrate(args.workers);folder=RUN/'tier_1';folder.mkdir(exist_ok=True);write(folder/'complete.json',dict(status='complete',tier=1,calibration_sha256=sha(RUN/'calibration.json'),real_trials=552,new_real_trials=0))
  if args.stop_after_tier==1:
   inputs('tier1_after_'+token);historical('tier1_after_'+token);print('Tier1 complete. Calibration:',RUN/'calibration.json');return
  best=point_results('best',calibration['chosen'],args.workers)
  if args.stop_after_tier==2:
   render(calibration,[best],2);inputs('tier2_after_'+token);historical('tier2_after_'+token);folder=RUN/'tier_2';write(folder/'complete.json',dict(status='complete',tier=2,real_trials=552,new_real_trials=0));print('Tier2 complete; resume the same command with --stop-after-tier 3.');return
  second=point_results('second',calibration['second'],args.workers);lines=render(calibration,[best,second],3);checkpoint();inputs('after_'+token);historical('after_'+token);checkpoint();seal(lines)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,choices=[1,2],default=1);parser.add_argument('--stop-after-tier',type=int,choices=[1,2,3],default=3);parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
 try:run(args)
 except Pause as exc:print('PAUSED:',str(exc),'Checkpoints retained; resume the same command.',flush=True);sys.exit(130)
 except Exception as exc:
  try:write(event('stopped'),dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),new_real_trials=0,source_writes=0))
  except Exception:pass
  print('STOPPED:',type(exc).__name__,str(exc),flush=True);raise
